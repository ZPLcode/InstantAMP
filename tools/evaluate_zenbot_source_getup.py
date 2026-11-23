"""Evaluate zero-command floor recovery in the original Zenbot contact model with immediate PD."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from instantamp import zenbot
from instantamp.evaluate import fallen_starts
from instantamp.zenbot_motion_train_obs import MOTION_TRAIN_INDEX
from tools.distill_zenbot_directions import FLOOR, ROOT, original_model
from tools.score_getup_in_place import pre_rise_attempts
from tools.score_zenbot_sequence import first_run, ground_timing
from tools.train_zenbot_skill_actor import DenseActor, native_frames

DT = 0.02


def measure(q, v, foot_load):
    up = 1.0 - 2.0 * np.square(q[:, 4:6]).sum(axis=1)
    standing = (q[:, 2] > 0.58) & (q[:, 2] <= 0.78) & (up > 0.8660254)
    first = first_run(standing, 10)
    rise = None if first is None else first * DT
    refell = bool(
        first is not None and ((q[first + 10 :, 2] < 0.42) | (up[first + 10 :] < 0.55)).any()
    )
    held = bool(
        standing[-100:].all()
        and (foot_load[-100:] >= 0.50).mean() >= 0.90
        and np.linalg.norm(v[-100:, :2], axis=1).max() < 0.15
    )
    retries = pre_rise_attempts(q[:, 2], DT, rise)["retries"]
    return {
        "rise": rise,
        "held": held,
        "first": held and rise is not None and not refell and not retries,
        "refell": refell,
        "retries": retries,
        "ground_timing": ground_timing(q, v, first),
    }


def rollout(model, actor, pose, seed, trial, steps, body_ids, foot_geoms):
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    if seed != 42:
        rng = np.random.default_rng([seed, trial, 45])
        data.qpos[7:] += np.deg2rad(rng.uniform(-2.0, 2.0, 23))
        rotation = Rotation.from_quat(data.qpos[3:7], scalar_first=True)
        tilt = Rotation.from_euler("xy", rng.uniform(-2.0, 2.0, 2), degrees=True)
        data.qpos[3:7] = (rotation * tilt).as_quat(scalar_first=True)
        data.qvel[:2] = rng.uniform(-0.07, 0.07, 2)
    mujoco.mj_forward(model, data)
    floor = model.geom("ground").id
    weight = model.body_mass.sum() * abs(model.opt.gravity[2])
    wrench = np.empty(6)
    history = np.zeros((5, 78), np.float32)
    action = np.zeros(model.nu)
    q = np.empty((steps + 1, model.nq))
    v = np.empty((steps + 1, model.nv))
    foot_load = np.zeros(steps)
    q[0], v[0] = data.qpos, data.qvel
    for tick in range(steps):
        # The source IMU is sampled before the last integrator update.
        _, frame = native_frames(
            data, model.jnt_bodyid[0], body_ids, action, 0.0, 0.0, zenbot.DEFAULT_Q
        )
        frame[:3] = data.sensor("angular-velocity").data
        if tick:
            history[:-1] = history[1:]
            history[-1] = frame
        else:
            history[:] = frame
        action = actor(history.ravel()[MOTION_TRAIN_INDEX])
        target = zenbot.DEFAULT_Q + zenbot.ACTION_SCALE * action
        for _ in range(4):
            torque = zenbot.KP * (target - data.qpos[7:]) - zenbot.KD * data.qvel[6:]
            data.ctrl[:] = np.clip(torque, -zenbot.EFFORT, zenbot.EFFORT)
            mujoco.mj_step(model, data)
            for ci, contact in enumerate(data.contact):
                if floor in contact.geom and foot_geoms.intersection(contact.geom):
                    mujoco.mj_contactForce(model, data, ci, wrench)
                    foot_load[tick] += wrench[0]
        q[tick + 1], v[tick + 1] = data.qpos, data.qvel
    return measure(q, v, foot_load / (4.0 * weight))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 337, 819])
    parser.add_argument(
        "--trials", type=int, nargs="+", help="default: the first frame of each source clip"
    )
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--floor", type=Path, default=FLOOR)
    parser.add_argument(
        "--model-xml",
        type=Path,
        help="optional path to the original MJCF with its visual mesh directory",
    )
    args = parser.parse_args()
    if args.seconds < 2.0 or min(args.seeds) < 0:
        parser.error("A minimum of two seconds and nonnegative seeds are required")
    model = (
        original_model()
        if args.model_xml is None
        else mujoco.MjModel.from_xml_path(str(args.model_xml))
    )
    model.opt.timestep = DT / 4
    actor = DenseActor(args.checkpoint, allow_layer_norm=True)
    if len(actor.mean) != len(MOTION_TRAIN_INDEX):
        parser.error("Expected a Motion Train 234-input actor")
    bodies = [model.body(name).id for name in zenbot.KEY_BODIES]
    feet = [model.body(name).id for name in zenbot.FEET]
    foot_geoms = set(np.flatnonzero(np.isin(model.geom_bodyid, feet)))
    names, starts, _ = fallen_starts(args.floor, zenbot.RESET_LIFT)
    trials = args.trials or [i for i, (_, offset) in enumerate(names) if offset == 0.0]
    if min(trials) < 0 or max(trials) >= len(names):
        parser.error("Trial number is outside the floor reference bank")
    rows = []
    for seed in args.seeds:
        for trial in trials:
            row = rollout(
                model,
                actor,
                np.asarray(starts[trial]),
                seed,
                trial,
                round(args.seconds / DT),
                bodies,
                foot_geoms,
            )
            rows.append(dict(seed=seed, trial=trial, clip=names[trial][0], **row))
    durations = [r["rise"] for r in rows if r["rise"] is not None]
    summary = {
        "cases": len(rows),
        "rose": len(durations),
        "held": sum(r["held"] for r in rows),
        "strict_first_rise": sum(r["first"] for r in rows),
        "median_rise_s": float(np.median(durations)) if durations else None,
    }
    xml = args.model_xml or ROOT / "assets/zenbot_23dof_43kg.xml"
    report = {
        "checkpoint": args.checkpoint.name,
        "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
        "seeds": args.seeds,
        "seconds": args.seconds,
        "observation_width": 234,
        "policy_hz": 50,
        "physics_hz": 200,
        "motor_delay_steps": 0,
        "summary": summary,
        "cases": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
