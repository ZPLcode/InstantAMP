"""Fixed-command native MuJoCo rollouts for the joint Zenbot walk/run/recovery actor."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from instantamp import env, zenbot
from instantamp.native_zenbot import ZenbotNative
from tools.score_zenbot_gait import straight_walk_clips


def starts(walk_dir, floor_dir):
    walk = straight_walk_clips(walk_dir)[0]
    for task, paths, speed in (
        ("walk", [walk], 1.),
        ("run", [walk_dir / "run__C3_-_run_stageii.npz"], 2.5),
        ("recover_to_walk", sorted(floor_dir.glob("*.npz")), 1.),
        ("walk_fall_resume", [walk], 1.),
        ("run_fall_resume", [walk_dir / "run__C3_-_run_stageii.npz"], 2.5),
    ):
        for path in paths:
            with np.load(path) as clip:
                frames = [0] if task == "recover_to_walk" else [int(p * len(clip["dof_pos"])) for p in (.2, .5, .8)]
                if task.endswith("_fall_resume"):
                    frames = frames[1:2]
                for i in frames:
                    quat = clip["root_quat"][i]
                    pos = [0., 0., clip["root_pos"][i, 2] + (zenbot.RESET_LIFT if task == "recover_to_walk" else 0.)]
                    pose = np.r_[pos, quat[[3, 0, 1, 2]], clip["dof_pos"][i]]
                    gyro = Rotation.from_quat(quat).inv().apply(clip["root_ang_vel"][i])
                    vel = np.r_[clip["root_lin_vel"][i], gyro, clip["dof_vel"][i]]
                    yield task, path.stem, i, speed, pose, vel


def score(task, speed, poses, velocities, loads, flight, push_step):
    quat = poses[:, [4, 5, 6, 3]]
    up = (poses[:, 2] > zenbot.UP_HEIGHT) & (1 - 2 * np.sum(quat[:, :2] ** 2, axis=1) > .8)
    # Exclude the initial mocap velocity. Score the final two seconds of physical motion.
    tail = slice(-round(2 / env.DT), None)
    yaw = np.arctan2(2 * (quat[:, 3]*quat[:, 2] + quat[:, 0]*quat[:, 1]), 1 - 2 * (quat[:, 1]**2 + quat[:, 2]**2))
    forward = np.cos(yaw) * velocities[:, 0] + np.sin(yaw) * velocities[:, 1]
    tail_forward = float(np.mean(forward[tail]))
    left, right = (loads[tail] > .08).T
    single = (left ^ right)
    leading = left[single]
    switches = int(np.count_nonzero(leading[1:] != leading[:-1]))
    flight = (flight[tail] & up[tail, None]).ravel()
    edges = np.flatnonzero(np.diff(flight.astype(int), prepend=0, append=0))
    flights = int(np.count_nonzero(edges[1::2] - edges[::2] >= 2))
    upright = bool(np.all(up[tail]))
    on_speed = abs(tail_forward - speed) <= max(.35, .2 * speed)
    gait = bool(upright and switches >= 2 and on_speed and (not task.startswith("run") or flights >= 2))
    recovered = bool(task == "recover_to_walk" and np.all(up[-50:]) and np.mean(loads[-50:].sum(-1)) > .5)
    row = dict(final_upright=upright, final_forward_mps=tail_forward, single_support_switches=switches,
               flight_intervals=flights, recovered=recovered, passed=gait and (task != "recover_to_walk" or recovered))
    if task.endswith("_fall_resume"):
        approach = slice(push_step - 50, push_step)
        moving = bool(up[approach].all() and abs(forward[approach].mean() - speed) < max(.4, .3 * speed))
        impact = slice(push_step, push_step + 100)
        down = (poses[impact, 2] < .48) | (1 - 2 * np.sum(quat[impact, :2] ** 2, axis=1) < .5)
        row.update(moving_before_push=moving, fell_after_push=bool(down.any()), passed=bool(moving and down.any() and gait))
        if down.any():
            fallen = push_step + int(np.flatnonzero(down)[0])
            # Ignore a transient upright crossing during impact. Require 0.2 s of balance.
            stable = np.flatnonzero(np.convolve(up[fallen:].astype(int), np.ones(10), "valid") == 10)
            row["fall_to_stable_upright_s"] = float(stable[0] * env.DT) if len(stable) else None
    return row


def rise_style(policy_qpos, reference):
    """Compare the first rise to its source clip, with both trajectories starting at frame zero."""
    fps = float(reference["fps"])
    expert = np.column_stack((reference["root_pos"], reference["root_quat"][:, [3, 0, 1, 2]], reference["dof_pos"]))

    def first_upright(poses):
        tilt = 2 * np.square(poses[:, 4:6]).sum(axis=1)
        frames = np.flatnonzero((poses[:, 2] > zenbot.UP_HEIGHT) & (tilt < env.UP_TILT))
        return int(frames[0]) if len(frames) else None

    i, j = first_upright(policy_qpos), first_upright(expert)
    result = dict(policy_upright_s=None if i is None else i * env.DT,
                  reference_upright_s=None if j is None else j / fps,
                  phase_aligned_joint_mae_deg=None)
    # Score even a failed rise and compare at the source speed. Holding still
    # or delaying a rapid stand should not pass as following the source motion.
    if np.isclose(fps * env.DT, 1.) and len(policy_qpos) >= len(expert):
        actual = policy_qpos[:len(expert)]
        result["source_time_joint_mae_deg"] = float(np.degrees(np.abs(actual[:, 7:] - expert[:, 7:]).mean()))
        result["source_time_pelvis_rmse_m"] = float(np.sqrt(np.square(actual[:, 2] - expert[:, 2]).mean()))
    if i is not None and j is not None and min(i, j) > 0:
        # Remove a uniform change of tempo before comparing the 23 joint angles.
        progress = np.linspace(0., 1., 51)
        def sample(poses, end):
            at = progress * end
            low = at.astype(int)
            frac = (at - low)[:, None]
            return poses[low, 7:] * (1. - frac) + poses[np.minimum(low + 1, end), 7:] * frac

        delta = sample(policy_qpos, i) - sample(expert, j)
        result["phase_aligned_joint_mae_deg"] = float(np.degrees(np.abs(delta).mean()))
    return result


def limb_clearances(model, poses):
    """Signed height of each foot, knee and hand collision surface over the floor."""
    geoms = [model.geom("robot_" + name + "_collision").id for name in zenbot.KEY_BODIES]
    kinds = model.geom_type[geoms]
    box = kinds == mujoco.mjtGeom.mjGEOM_BOX
    if not np.all(box | (kinds == mujoco.mjtGeom.mjGEOM_CAPSULE)):
        raise ValueError("Expected Zenbot sole boxes and limb capsules")
    sizes = model.geom_size[geoms]
    data = mujoco.MjData(model)
    out = np.empty((len(poses), len(geoms)))
    for i, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        axes_z = np.abs(data.geom_xmat[geoms].reshape(-1, 3, 3)[:, 2])
        extent = np.where(box, (axes_z * sizes).sum(1), sizes[:, 0] + sizes[:, 1] * axes_z[:, 2])
        out[i] = data.geom_xpos[geoms, 2] - extent
    return out


def support_sequence(model, policy_qpos, reference, near_floor_m=.025):
    """Offline comparison of limb geometry up to standing, with separate source-time overlap."""
    dt = 1. / float(reference["fps"])
    if not np.isclose(dt, env.DT):
        raise ValueError("The support comparison requires the 50 Hz evaluation motion bank")
    expert = np.column_stack((reference["root_pos"], reference["root_quat"][:, [3, 0, 1, 2]], reference["dof_pos"]))

    def rise_end(q):
        upright = (q[:, 2] > zenbot.UP_HEIGHT) & (2 * np.square(q[:, 4:6]).sum(1) < .2)
        n = round(.2 / dt)
        stable = np.flatnonzero(np.convolve(upright.astype(int), np.ones(n), "valid") == n) if len(q) >= n else []
        return int(stable[0]) if len(stable) else len(q)

    def masks(q):
        near = limb_clearances(model, q) < near_floor_m
        feet, knees, hands = near[:, :2], near[:, 2:4], near[:, 4:]
        # The pelvis forward axis points toward the floor during the demonstrated
        # hand-supported turn. Backward palm contact while supine is also possible.
        w, x, y, z = q[:, 3:7].T
        forward_down = 2 * (x * z - w * y) < -.3
        return {"either_hand": hands.any(1), "both_hands": hands.all(1),
                "both_hands_facing_floor": hands.all(1) & forward_down,
                "hand_with_foot": hands.any(1) & feet.any(1), "hand_with_knee": hands.any(1) & knees.any(1),
                "two_feet_without_hands": feet.all(1) & ~hands.any(1)}

    ref_end, policy_end = rise_end(expert), rise_end(policy_qpos)
    ref = masks(expert[:ref_end])
    actual = masks(policy_qpos[:max(ref_end, policy_end)])
    result = {"near_floor_m": near_floor_m, "reference_duration_s": {}, "policy_duration_s": {},
              "source_time_iou": {}, "knee_flexion_mae_deg_during_source_hand_support": None}
    for name, demo_mask in ref.items():
        policy_mask = actual[name]
        result["reference_duration_s"][name] = float(demo_mask.sum() * dt)
        result["policy_duration_s"][name] = float(policy_mask[:policy_end].sum() * dt)
        aligned = policy_mask[:ref_end]
        overlap = None
        if len(aligned) == ref_end:
            union = np.count_nonzero(aligned | demo_mask)
            if union:
                overlap = float(np.count_nonzero(aligned & demo_mask) / union)
        result["source_time_iou"][name] = overlap
    source_hand = ref["either_hand"]
    if len(policy_qpos) >= ref_end and source_hand.any():
        knees = [7 + zenbot.DOF.index(j) for j in ("left_knee", "right_knee")]
        difference = policy_qpos[:ref_end, knees] - expert[:ref_end, knees]
        result["knee_flexion_mae_deg_during_source_hand_support"] = float(np.degrees(np.abs(difference[source_hand]).mean()))
    return result


def ground_support(poses, loads, start):
    """Physical ground loads from the fall until the first 0.2 s of upright balance."""
    pose = np.asarray(poses)[start:]
    load = np.asarray(loads)[start:]
    up = (pose[:, 2] > zenbot.UP_HEIGHT) & (2 * np.square(pose[:, 4:6]).sum(1) < .2)
    stable = np.flatnonzero(np.convolve(up.astype(int), np.ones(10), "valid") == 10) if len(up) >= 10 else []
    end = max(1, int(stable[0])) if len(stable) else len(load)
    result = {"rose_to_balance": bool(len(stable)), "time_to_balance_s": float(end * env.DT) if len(stable) else None}
    for name, part in (("hand", load[:end, 2:4]), ("knee", load[:end, 6:8])):
        force = part.sum(1)
        result[name + "_load_bw_s"] = float(force.sum() * env.DT)
        result[name + "_support_s"] = float(np.count_nonzero(force >= .08) * env.DT)
        result[name + "_sustained"] = bool(len(force) >= 4 and np.any(np.convolve((force >= .08).astype(int), np.ones(4), "valid") == 4))
    result["both_hands_loaded_s"] = float((load[:end, 2:4] >= .04).all(1).sum() * env.DT)
    return result


def summarize_support(trials):
    """Summarize floor starts across clips and seeds, including failed recoveries."""
    rows = [r for r in trials if "support_sequence" in r]
    if not rows:
        return {}

    def median(values):
        values = [v for v in values if v is not None]
        return float(np.median(values)) if values else None

    sequence = [r["support_sequence"] for r in rows]
    stats = {"floor_starts": len(rows), "walked_after_rise": sum(r["passed"] for r in rows)}
    for key in ("reference_duration_s", "policy_duration_s", "source_time_iou"):
        stats["median_" + key] = {stage: median(s[key][stage] for s in sequence) for stage in sequence[0][key]}
    stats["median_knee_flexion_mae_deg_during_source_hand_support"] = median(
        s["knee_flexion_mae_deg_during_source_hand_support"] for s in sequence)
    loads = [r["ground_support"]["both_hands_loaded_s"] for r in rows if "ground_support" in r]
    stats["complete_force_traces"] = len(loads)
    stats["median_both_hands_loaded_s"] = median(loads)
    return stats


def evaluate(run, checkpoint, seconds, seed, traces=None, push_at=2., push_angle=None):
    cfg = json.loads((run / "config.json").read_text())
    if cfg["robot"] != zenbot.MIXED_TASK or not 1. <= push_at <= seconds - 4. or not np.isfinite(seconds):
        raise ValueError("Allow at least one second before the push and four seconds after it")
    push_step = round(push_at / env.DT)
    if push_angle is not None and not np.isfinite(push_angle):
        raise ValueError("The horizontal push angle must be finite")
    angle = np.deg2rad(90. if push_angle is None else push_angle)
    push_xy = 400. * np.round([np.cos(angle), np.sin(angle)], 12)
    walk, recovery = Path(cfg["walk_dir"]), Path(cfg["floor_dir"])
    native = ZenbotNative(recovery, checkpoint, walk_dir=walk)
    if traces is not None:
        traces.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for task, clip, frame, speed, pose, velocity in starts(walk, recovery):
        delays = rng.integers(3, 8, 5)
        if push_angle is not None and not task.endswith("_fall_resume"):
            continue
        native.reset(pose, velocity, delays, command=(speed, 0., 0.))
        poses, velocities, loads, flight = [], [], [], []
        for step in range(round(seconds / env.DT)):
            if task.endswith("_fall_resume"):
                native.data.xfrc_applied[native.sim.torso, :2] = push_xy if push_step <= step < push_step + 10 else 0.
            native.step()
            data = native.data
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or np.max(np.abs(data.qvel)) > 1000:
                break
            poses.append(data.qpos.copy())
            velocities.append(data.qvel.copy())
            loads.append(native.support_load.copy())
            flight.append(native.flight.copy())
        complete = len(poses) == round(seconds / env.DT)
        result = score(task, speed, np.array(poses), np.array(velocities), np.array(loads)[:, :2], np.array(flight), push_step) if complete else dict(passed=False, recovered=False)
        if task == "recover_to_walk":
            with np.load(recovery / f"{clip}.npz") as reference:
                actual = np.vstack([pose, *poses])
                result["rise_style"] = rise_style(actual, reference)
                result["support_sequence"] = support_sequence(native.model, actual, reference)
            if complete:
                result["ground_support"] = ground_support(poses, loads, 0)
        elif task.endswith("_fall_resume") and result.get("fell_after_push"):
            impact = np.asarray(poses)[push_step:push_step + 100]
            down = (impact[:, 2] < .48) | (2 * np.square(impact[:, 4:6]).sum(1) > .5)
            result["ground_support"] = ground_support(poses, loads, push_step + int(np.flatnonzero(down)[0]))
        rows.append(dict(task=task, clip=clip, frame=frame, command_mps=speed, complete=complete, **result))
        if traces is not None:
            np.savez_compressed(traces / f"{task}_{clip}_{frame}.npz", initial_qpos=pose, qpos=poses, qvel=velocities,
                                foot_load_fraction=np.asarray(loads).reshape(-1, len(native.support_load))[:, :2], support_load_fraction=loads,
                                airborne_substeps=flight, delays=delays, command_mps=speed, fps=50)
    summary = {}
    for task in dict.fromkeys(row["task"] for row in rows):
        group = [row for row in rows if row["task"] == task]
        summary[task] = dict(passed=sum(row["passed"] for row in group), total=len(group))
    if "recover_to_walk" in summary:
        summary["recover_to_walk"]["stood_up"] = sum(row["recovered"] for row in rows)
    result = dict(checkpoint=checkpoint.name, iteration=json.loads(checkpoint.with_suffix(".json").read_text())["it"],
                  seconds=seconds, push_at_s=push_at, push_angle_deg=90. if push_angle is None else push_angle,
                  seed=seed, assist_newtons=0, summary=summary, trials=rows)
    if "recover_to_walk" in summary:
        result["support_summary"] = summarize_support(rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="export path or filename within ckpt, default latest")
    parser.add_argument("--seconds", type=float, default=8.)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--push-at", type=float, default=2., help="time in seconds of the lateral push in the two perturbation trials")
    parser.add_argument("--push-angle", type=float, help="run only the two push trials at this angle in degrees from world +X")
    parser.add_argument("--traces", type=Path)
    args = parser.parse_args()
    files = sorted((args.run / "ckpt").glob("*.npz"))
    if not files:
        parser.error("The training run has no exported policies yet")
    checkpoint = files[-1]
    if args.checkpoint:
        checkpoint = args.checkpoint if args.checkpoint.is_file() else args.run / "ckpt" / args.checkpoint
    result = evaluate(args.run, checkpoint, args.seconds, args.seed, args.traces, args.push_at, args.push_angle)
    suffix = "" if args.push_at == 2. and args.seconds == 8. else f"_push{args.push_at:g}s_total{args.seconds:g}s"
    if args.push_angle is not None:
        suffix += f"_angle{args.push_angle:g}"
    path = args.run / f"eval_{checkpoint.stem}_seed{args.seed}{suffix}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]), path, flush=True)
