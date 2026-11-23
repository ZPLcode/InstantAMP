"""Compare steady simulated gait with unpaired, speed-selected motion_train poses offline."""
import argparse
import json
from pathlib import Path

import numpy as np

from instantamp import zenbot


def straight_walk_clips(walk_dir):
    """Use the source's four straight walk transitions or the earlier sustained-walk bank."""
    active = [walk_dir / f"Male{subject}_{clip}.npz" for subject in (1, 2)
              for clip in ("B1_StandToWalk", "B2_WalkToStand")]
    if all(path.is_file() for path in active):
        return active
    return [walk_dir / (name + ".npz") for name in ("143_02_stageii", "143_03_stageii")]


def heading_speed(quat_xyzw, velocity):
    x, y, z, w = quat_xyzw.T
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return np.cos(yaw) * velocity[:, 0] + np.sin(yaw) * velocity[:, 1]


def gait_metrics(angles):
    degrees = np.rad2deg(angles)
    quantiles = np.percentile(degrees, np.arange(10, 100, 10), axis=0)
    span = quantiles[-1] - quantiles[0]

    def pair(joint):
        return [zenbot.DOF.index(side + "_" + joint) for side in ("left", "right")]

    shoulder = degrees[:, pair("shoulder_pitch")]
    hip = degrees[:, pair("hip_pitch")]
    arm_diff = shoulder[:, 0] - shoulder[:, 1]
    hip_diff = hip[:, 0] - hip[:, 1]
    correlation = (float(np.corrcoef(arm_diff, hip_diff)[0, 1])
                   if min(np.std(arm_diff), np.std(hip_diff)) > 1e-5 else None)
    return dict(joint_quantiles_deg=quantiles.tolist(), shoulder_swing_deg=float(span[pair("shoulder_pitch")].mean()),
                knee_excursion_deg=float(span[pair("knee")].mean()), hip_excursion_deg=float(span[pair("hip_pitch")].mean()),
                elbow_angle_deg=float(degrees[:, pair("elbow")].mean()), arm_hip_correlation=correlation)


def reference_frames(walk_dir, task, speed, reference_clips=None):
    paths = reference_clips or {"walk": straight_walk_clips(walk_dir), "run": [walk_dir / "run__C3_-_run_stageii.npz"]}[task]
    angles, speeds = [], []
    for path in paths:
        with np.load(path) as clip:
            forward = heading_speed(clip["root_quat"], clip["root_lin_vel"])
            # Average over 0.4 s to reduce within-stride variation in pelvis velocity.
            radius = round(.2 * float(clip["fps"]))
            smoothed = np.convolve(np.pad(forward, radius, mode="edge"),
                                   np.ones(2 * radius + 1) / (2 * radius + 1), mode="valid")
            angles.extend(clip["dof_pos"])
            speeds.extend(smoothed)
    speeds = np.asarray(speeds)
    distance = np.abs(speeds - speed)
    # Give the percentile estimates at least one second's worth of samples. Expose
    # the actual speed range since starts/stops and fast runs cover different speeds.
    kth = min(50, len(distance)) - 1
    width = max(.35, np.partition(distance, kth)[kth])
    select = distance <= width
    return gait_metrics(np.asarray(angles)[select]), dict(clips=[p.stem for p in paths], frames=int(select.sum()),
            mean_mps=float(speeds[select].mean()), min_mps=float(speeds[select].min()), max_mps=float(speeds[select].max()))


def compare(reference, actual):
    delta = np.abs(np.asarray(actual["joint_quantiles_deg"]) - reference["joint_quantiles_deg"])
    return {"joint_distribution_mae_deg": float(delta.mean()),
            "joint_mae_deg": dict(zip(zenbot.DOF, delta.mean(axis=0).tolist()))}


def score_trace(path, walk_dir, task, reference_clips=None):
    with np.load(path) as trial:
        speed = float(trial["command_mps"])
        frames = round(4 * float(trial["fps"]))
        if len(trial["qpos"]) < 2 * frames:
            raise ValueError("Expected at least 8 s to allow settling and measure the last 4 s")
        q, v = trial["qpos"][-frames:], trial["qvel"][-frames:]
    ref, info = reference_frames(walk_dir, task, speed, reference_clips)
    measured = gait_metrics(q[:, 7:])
    return dict(command_mps=speed, actual_mps=float(heading_speed(q[:, [4, 5, 6, 3]], v).mean()),
                reference_speed=info, reference=ref, simulated=measured, **compare(ref, measured))


def drive(checkpoint, walk_dir, floor_dir, out, seed, walk_reference=None):
    """Exercise the viewer's 0.1 m/s speed changes continuously from standing."""
    from instantamp.native_zenbot import ZenbotNative
    from tools.evaluate_zenbot_walk_recovery import score

    native = ZenbotNative(floor_dir, checkpoint, walk_dir=walk_dir)
    delays = np.random.default_rng(seed).integers(3, 8, 5)
    pose = np.r_[0., 0., zenbot.DEFAULT_HEIGHT, 1., 0., 0., 0., zenbot.DEFAULT_Q]
    native.reset(pose, np.zeros(native.model.nv), delays)
    out.mkdir(exist_ok=True, parents=True)
    command, rows = 0., {}
    for stage, target in (("stand", 0.), ("walk", 1.), ("run", 2.5), ("fast_run", 3.8), ("stop", 0.)):
        poses, velocities, feet, flight = [], [], [], []
        for i in range(500):
            if i % 4 == 0:
                command = round(command + np.clip(target - command, -.1, .1), 1)
                native.command = native.command.at[0, 0].set(command)
            native.step()
            poses.append(native.data.qpos.copy())
            velocities.append(native.data.qvel.copy())
            feet.append(native.support_load[:2].copy())
            flight.append(native.flight.copy())
        q, v, feet, flight = map(np.asarray, (poses, velocities, feet, flight))
        path = out / (stage + ".npz")
        np.savez_compressed(path, qpos=q, qvel=v, fps=50, command_mps=target, delays=delays,
                            foot_load_fraction=feet, airborne_substeps=flight)
        task = "run" if stage in ("run", "fast_run") else stage
        row = score(task, target, q, v, feet, flight, None)
        row["fell"] = bool(np.any((q[:, 2] < .48) | (1 - 2 * np.square(q[:, 4:6]).sum(1) < .5)))
        if task in ("walk", "run"):
            row["style"] = score_trace(path, walk_dir, task, walk_reference if task == "walk" else None)
        else:
            row["passed"] = row["final_upright"] and abs(row["final_forward_mps"]) < .15 and not row["fell"]
        rows[stage] = row
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    trace = commands.add_parser("trace", help="score an existing physics trace")
    trace.add_argument("path", type=Path)
    trace.add_argument("--task", choices=("walk", "run"), required=True)
    trace.add_argument("--reference", type=Path, nargs="+", help="50 Hz reference clips for this gait")
    sweep = commands.add_parser("drive", help="record a 50 s command sweep without resetting the policy")
    sweep.add_argument("checkpoint", type=Path)
    sweep.add_argument("out", type=Path)
    sweep.add_argument("--walk-reference", type=Path, nargs="+", help="steady walking clips for the 1 m/s stage")
    sweep.add_argument("--seed", type=int, default=241)
    sweep.add_argument("--floor-dir", type=Path, default=Path("data/zenbot23_walk_run_recovery/recovery"))
    parser.add_argument("--walk-dir", type=Path, default=Path("data/zenbot23_walk_run_recovery/walk_run"))
    args = parser.parse_args()
    if args.operation == "drive":
        rows = drive(args.checkpoint, args.walk_dir, args.floor_dir, args.out, args.seed, args.walk_reference)
        (args.out / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    else:
        print(json.dumps(score_trace(args.path, args.walk_dir, args.task, args.reference), indent=2, allow_nan=False))
