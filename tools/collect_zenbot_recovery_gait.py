"""Collect native MuJoCo recovery-to-gait teacher actions for offline single-actor distillation."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from instantamp import zenbot
from instantamp.evaluate import fallen_starts
from instantamp.sim import build_scene
from tools.evaluate_zenbot_walk_recovery import score
from tools.score_getup_in_place import pre_rise_attempts
from tools.train_zenbot_skill_actor import DenseActor, native_frames

ROOT = Path(__file__).resolve().parents[1]
FLOOR = ROOT / "data/zenbot23_walk_run_recovery/recovery"
RISE = ROOT / "runs/zenbot23_getup_standing_amp_pd/ckpt/002575.npz"


def motor_step(model, data, action, previous, older, delays, soles, floor, body_weight):
    load = np.zeros(2)
    flight = np.ones(4, dtype=bool)
    wrench = np.empty(6)
    for j in range(4):
        delayed = np.where(j >= delays, action, np.where(j + 4 >= delays, previous, older))
        target = zenbot.DEFAULT_Q + zenbot.ACTION_SCALE * delayed
        vel = data.qvel[6:]
        torque = np.clip(zenbot.KP * (target - data.qpos[7:]) - zenbot.KD * vel, -zenbot.EFFORT, zenbot.EFFORT)
        data.ctrl[:] = np.where((np.abs(vel) >= zenbot.SPEED) & (torque * vel > 0.), 0., torque)
        mj.mj_step(model, data)
        for i, contact in enumerate(data.contact):
            if floor not in contact.geom:
                continue
            mj.mj_contactForce(model, data, i, wrench)
            if wrench[0] > .1:
                flight[j] = False
            other = contact.geom1 if contact.geom2 == floor else contact.geom2
            if other in soles:
                load[soles.index(other)] += wrench[0] / (4 * body_weight)
    return load, flight


def first_stand(q):
    up = 1. - 2. * np.square(q[:, 4:6]).sum(axis=1)
    upright = (q[:, 2] > .58) & (q[:, 2] <= .78) & (up > .8660254)
    hits = np.flatnonzero(np.convolve(upright, np.ones(10), "valid") == 10)
    return int(hits[0]) if len(hits) else None


def reference_rise(path, offset):
    with np.load(path) as motion:
        start = round(offset * float(motion["fps"]))
        pose = np.c_[motion["root_pos"], motion["root_quat"][:, [3, 0, 1, 2]]][start:]
    frame = first_stand(pose)
    return None if frame is None else frame * .02


def rollout(a, scene, experts, trial, name, pose, velocity, delays):
    model, data, bodies, soles, floor, weight = scene
    rise, gait = experts
    mj.mj_resetData(model, data)
    data.qpos[:], data.qvel[:] = pose, velocity
    if a.seed != 42:
        rng = np.random.default_rng([a.seed, trial, 7143])
        data.qpos[7:] += np.deg2rad(rng.uniform(-2, 2, 23))
        rotation = Rotation.from_quat(data.qpos[3:7], scalar_first=True)
        noise = Rotation.from_euler("xy", rng.uniform(-2, 2, 2), degrees=True)
        data.qpos[3:7] = (rotation * noise).as_quat(scalar_first=True)
        data.qvel[:2] += rng.uniform(-.07, .07, 2)
    previous = older = np.zeros(23, np.float32)
    history = np.zeros((5, 99), np.float32)
    gait_history = np.zeros((5, 78), np.float32)
    poses, velocities, loads, flights, foot_xy = [data.qpos.copy()], [], [], [], []
    inputs, actions, modes = [], [], []
    teacher_speeds, gait_blends = [], []
    count, switch, speed, fraction = 0, None, 0., 0.
    for i in range(a.steps):
        mj.mj_kinematics(model, data)
        height = data.qpos[2]
        up = data.xmat[model.jnt_bodyid[0], 8]
        supported = bool(loads and loads[-1].sum() > .55 and loads[-1].min() > a.weak_load)
        ready = (height > a.min_height and up > a.up_cos and np.linalg.norm(data.qvel[:3]) < a.max_speed and supported)
        count = count + 1 if ready else 0
        if switch is None and count >= a.delay:
            switch = i
        if switch is not None:
            fraction = min((i - switch + 1) / a.blend, 1.)
            if i >= switch + a.blend + a.warm and i % 4 == 0:
                speed = min(speed + .1, a.command)
        if fraction >= 1. and (height < .35 or up < .45):
            break
        intent = a.command if a.hold_command else speed
        frames = native_frames(data, model.jnt_bodyid[0], bodies, previous, intent, speed, zenbot.DEFAULT_Q)
        for h, frame in zip((history, gait_history), frames):
            if i == 0:
                h[:] = frame
            else:
                h[:-1] = h[1:]
                h[-1] = frame
        action = (1. - fraction) * rise(history[:, :96]) if fraction < 1. else 0.
        if fraction:
            action = action + fraction * gait(gait_history)
        inputs.append(history.copy().ravel())
        actions.append(action)
        modes.append(0 if fraction < .05 else 1 if fraction < 1. else 2 if speed < .3 else 3 if speed < 1.7 else 4)
        teacher_speeds.append(gait_history[:, 6].copy())
        gait_blends.append(fraction)
        load, flight = motor_step(model, data, action, previous, older, delays, soles, floor, weight)
        older, previous = previous, action
        poses.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        loads.append(load)
        flights.append(flight)
        foot_xy.append(data.xpos[bodies[:2], :2].copy())
    q, v, f, air, feet = map(np.asarray, (poses, velocities, loads, flights, foot_xy))
    upright = first_stand(q)
    up = 1. - 2. * np.square(q[:, 4:6]).sum(axis=1)
    refell = bool(upright is not None and ((q[upright + 10:, 2] < .42) | (up[upright + 10:] < .55)).any())
    row = score("run" if a.command > 1.7 else "walk", a.command, q[1:], v, f, air, None)
    if a.command == 0.:
        row.update(tail_speed_mps=float(np.linalg.norm(v[-100:, :2], axis=-1).mean()),
                   tail_joint_radps=float(np.abs(v[-100:, 6:]).mean()),
                   tail_weak_foot_bw=float(f[-100:].min(axis=1).mean()))
        held = ((q[-100:, 2] > .58) & (q[-100:, 2] <= .78) & (up[-100:] > .8660254)).all()
        row["passed"] = bool(held and (f[-100:].sum(axis=1) >= .5).mean() >= .9
                             and np.linalg.norm(v[-100:, :2], axis=1).max() < .15
                             and row["tail_speed_mps"] < .08 and row["tail_joint_radps"] < .15
                             and row["tail_weak_foot_bw"] >= .20)
    rise_s = None if upright is None else upright * .02
    attempts = pre_rise_attempts(q[:, 2], .02, rise_s)
    row.update(trial=trial, seed=a.seed, clip=name[0], offset=float(name[1]), switch=switch, frames=len(v),
               rise_s=rise_s, refell=refell, pre_rise_attempts=attempts, command=a.command,
               reference_rise_s=reference_rise(a.floor / name[0], name[1]))
    if upright is not None:
        foot_start = max(0, upright - 1)
        row.update(pelvis_excursion_m=float(np.linalg.norm(q[upright:, :2] - q[upright, :2], axis=-1).max()),
                   foot_excursion_m=float(np.linalg.norm(feet[foot_start:] - feet[foot_start], axis=-1).max()))
    row["usable"] = bool(upright is not None and not refell and row["passed"] and switch is not None
                         and (a.allow_pre_rise_retries or attempts["retries"] == 0)
                         and fraction >= 1. and len(v) == a.steps and abs(speed - a.command) < .01)
    return row, (inputs, actions, modes, teacher_speeds, gait_blends)


def collect(a):
    a.out.mkdir(parents=True, exist_ok=True)
    if min(a.delay, a.blend) < 1 or a.steps < 100 or a.warm < 0:
        raise ValueError("Invalid duration for teacher collection")
    model = build_scene(robot_name="zenbot23_getup")
    bodies = [model.body("robot_" + name).id for name in zenbot.KEY_BODIES]
    soles = [model.geom("robot_" + name + "_collision").id for name in zenbot.FEET]
    scene = model, mj.MjData(model), bodies, soles, model.geom("floor").id, model.body_mass.sum() * abs(model.opt.gravity[2])
    names, poses, velocities = fallen_starts(a.floor, zenbot.RESET_LIFT)
    experts = DenseActor(a.rise), DenseActor(a.gait)
    hashes = {key + "_sha256": hashlib.sha256(getattr(a, key).read_bytes()).hexdigest() for key in ("rise", "gait")}
    if a.replay_stands:
        recipe = json.loads(a.replay_stands.read_text())
        if any(recipe[key] != digest for key, digest in hashes.items()):
            raise ValueError("The recovery and gait teachers must match the recorded selection")
        cases = []
        for trial, seed, option in recipe["trial_seed_setting"]:
            params = vars(a) | recipe["settings"][option] | dict(seed=seed, steps=recipe["steps"], command=recipe["command"])
            cases.append((trial, argparse.Namespace(**params)))
    else:
        cases = [(i, a) for i in (a.trials if a.trials is not None else range(len(poses)))]
    rows, accepted = [], []
    for i, config in cases:
        delays = np.random.default_rng(config.seed).integers(3, 8, (len(poses), 5))
        row, samples = rollout(config, scene, experts, i, names[i], poses[i], velocities[i], delays[i][zenbot.GROUP])
        if a.replay_stands and not row["usable"]:
            raise ValueError(f"Cannot reproduce the selected standing rollout at seed {config.seed}, trial {i}")
        if row["usable"]:
            row["sample_episode"] = len(accepted)
            accepted.append(samples)
        rows.append(row)
    (a.out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    if accepted:
        x, u, mode, teacher_speed, gait_blend = zip(*accepted)
        np.savez(a.out / "expert.npz", x=np.concatenate(x).astype("f"), u=np.concatenate(u).astype("f"),
                 mode=np.concatenate(mode).astype("b"), teacher_speed=np.concatenate(teacher_speed).astype("f"),
                 gait_blend=np.concatenate(gait_blend).astype("f"))
    else:
        np.savez(a.out / "expert.npz", x=np.empty((0, 495), "f"), u=np.empty((0, 23), "f"), mode=np.empty(0, "b"),
                 teacher_speed=np.empty((0, 5), "f"), gait_blend=np.empty(0, "f"))
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    settings.update(hashes)
    (a.out / "collection.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"{len(accepted)} / {len(rows)} single-rise-to-command episodes passed in {a.out}")


def select_stands(directories, output, per_clip):
    """Select high-quality zero-command floor episodes from independent seeds for each source clip."""
    if per_clip < 1:
        raise ValueError("--per-clip must be positive")
    groups = {}
    for path in directories:
        for row in json.loads((path / "summary.json").read_text()):
            if (not row["usable"] or row["command"] != 0. or row["offset"] != 0.
                    or row.get("pre_rise_attempts", {}).get("retries") != 0):
                continue
            # Compare maxima from the *first* supported upright frame, across the entire stand.
            # One second of timing error counts as 10 cm of additional excursion.
            error = abs(row["rise_s"] - row["reference_rise_s"]) if row["reference_rise_s"] is not None else 0.
            rank = max(row["pelvis_excursion_m"], row["foot_excursion_m"]) + .1 * error
            groups.setdefault(row["clip"], []).append((rank, path, row))
    selected = []
    for items in groups.values():
        seeds = set()
        for rank, path, row in sorted(items, key=lambda item: item[0]):
            if row["seed"] in seeds:
                continue
            seeds.add(row["seed"])
            selected.append(dict(row, collection=str(path), selection_cost=rank))
            if len(seeds) == per_clip:
                break
    if not selected:
        raise ValueError("No passing zero-command source-clip starts in the supplied collections")
    samples = {k: [] for k in ("x", "u", "mode", "teacher_speed", "gait_blend")}
    for row in selected:
        start = row["sample_episode"] * row["frames"]
        with np.load(Path(row["collection"]) / "expert.npz") as data:
            for name in samples:
                if name in data:
                    samples[name].append(data[name][start:start + row["frames"]])
                elif name in ("teacher_speed", "gait_blend"):
                    # Legacy episodes do not record the teacher command or its blend.
                    shape = (row["frames"], 5) if name == "teacher_speed" else (row["frames"],)
                    samples[name].append(np.full(shape, np.nan, np.float32))
                else:
                    raise ValueError(f"Missing {name} samples in {row['collection']}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "expert.npz", **{k: np.concatenate(values) for k, values in samples.items()})
    (output / "summary.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(f"Selected {len(selected)} episodes across {len(groups)} source clips in {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gait", type=Path, help="converted 390-input gait teacher")
    p.add_argument("--rise", type=Path, default=RISE)
    p.add_argument("--floor", type=Path, default=FLOOR)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--replay-stands", type=Path, help="reproduce the ordered floor starts and handoffs in a selection recipe")
    p.add_argument("--select-stands", nargs="+", type=Path, help="rank previously collected zero-command clips")
    p.add_argument("--per-clip", type=int, default=2, help="independent seeds selected for each clip")
    p.add_argument("--command", type=float, choices=(0., 1., 2.5), default=1.)
    p.add_argument("--hold-command", action="store_true", help="expose final command to the student from the floor")
    p.add_argument("--allow-pre-rise-retries", action="store_true", help="reproduce older training collections")
    p.add_argument("--delay", type=int, default=24, help="number of consecutive supported frames before transition")
    p.add_argument("--blend", type=int, default=70)
    p.add_argument("--warm", type=int, default=35)
    p.add_argument("--max-speed", type=float, default=.65)
    p.add_argument("--min-height", type=float, default=.58)
    p.add_argument("--up-cos", type=float, default=.90)
    p.add_argument("--weak-load", type=float, default=.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=750)
    p.add_argument("--trials", type=int, nargs="+", help="fallen start indices, ordered by clip and then offset")
    args = p.parse_args()
    if args.select_stands:
        select_stands(args.select_stands, args.out, args.per_clip)
    elif args.gait is None:
        p.error("Collecting demonstrations requires a --gait checkpoint")
    else:
        collect(args)
