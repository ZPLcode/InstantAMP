"""Replay independent velocity commands from the same post-getup native MuJoCo state."""

import argparse
import hashlib
import json
import os
from copy import copy
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from instantamp import zenbot
from tools.evaluate_zenbot_getup import DT, evaluate
from tools.play_zenbot_getup import ZenbotPlayer
from tools.score_zenbot_gait import joint_jitter

COMMANDS = {
    "forward": (1.0, 0.0, 0.0),
    "back": (-0.5, 0.0, 0.0),
    "left": (0.0, 0.4, 0.0),
    "right": (0.0, -0.4, 0.0),
    "turn_left": (0.0, 0.0, 0.6),
    "turn_right": (0.0, 0.0, -0.6),
    "still": (0.0, 0.0, 0.0),
}
WINDOW = round(1.0 / DT)
TURN_RADIUS_M = 0.35


def heading_kinematics(poses):
    """Each position delta is rotated into the pelvis's mid-step horizontal heading."""
    w, x, y, z = poses[:, 3:7].T
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    mid = 0.5 * (yaw[:-1] + yaw[1:])
    dx, dy = np.diff(poses[:, :2], axis=0).T / DT
    c, s = np.cos(mid), np.sin(mid)
    speeds = np.column_stack((c * dx + s * dy, -s * dx + c * dy, np.diff(yaw) / DT))
    return speeds, yaw


def startup_metrics(poses, command):
    """Pelvis motion in the first second relative to the heading when the command arrived."""
    first = poses[:WINDOW + 1]
    _, yaw = heading_kinematics(first)
    c, s = np.cos(yaw[0]), np.sin(yaw[0])
    xy = (first[:, :2] - first[0, :2]) @ np.array([[c, -s], [s, c]])
    target = np.asarray(command)[:2]
    speed = np.linalg.norm(target)
    if speed:
        along = target / speed
        departure = np.abs(xy @ [-along[1], along[0]])
    else:
        departure = np.linalg.norm(xy, axis=-1)
    w, x, y, z = first[:, 3:7].T
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    return {"startup_off_axis_m": float(departure.max()),
            "startup_peak_abs_roll_deg": float(np.rad2deg(np.max(np.abs(roll)))),
            "startup_along_m": float(xy[-1] @ along) if speed else 0.}


def foot_contact_metrics(substep_forces):
    """Count control intervals with a measured force between the two feet."""
    forces = np.asarray(substep_forces)
    return {"foot_contact_steps": int(np.count_nonzero((forces > 0.).any(axis=1))),
            "max_foot_pair_force_n": float(forces.max())}


def direction_metrics(poses, loads, command, *, relative_active_speed=False):
    speeds, yaw = heading_kinematics(poses)
    windows = speeds[-3 * WINDOW :].reshape(3, WINDOW, 3).mean(axis=1)
    actual = windows.mean(axis=0)
    tolerance = np.array([0.15, 0.15, 0.20])
    target = np.asarray(command)
    if np.any(target):
        axis = int(np.argmax(np.abs(target)))
        relative = 0.25 * abs(target[axis])
        tolerance[axis] = relative if relative_active_speed else max(relative, 0.18 if axis == 2 else 0.15)
    window_matches = (np.abs(windows - target) <= tolerance).all(axis=1)
    tracks = bool((np.abs(actual - target) <= tolerance).all() and window_matches.sum() >= 2)
    up = 1 - 2 * np.square(poses[:, 4:6]).sum(axis=1)
    fell = bool(np.any((poses[:, 2] < 0.42) | (up < 0.55)))
    end_upright = bool(np.all((poses[-3 * WINDOW :, 2] > zenbot.UP_HEIGHT) & (up[-3 * WINDOW :] > 0.8)))
    departure = float(np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1).max())
    pure_turn = target[2] != 0 and not np.any(target[:2])
    left, right = (loads[-3 * WINDOW :] > 0.08).T
    leading = left[left ^ right]
    return {
        "command": target.tolist(),
        "mean_last_3s": actual.tolist(),
        "one_second_means": windows.tolist(),
        "matching_windows": int(window_matches.sum()),
        "single_support_switches": int(np.count_nonzero(leading[1:] != leading[:-1])),
        "fell": fell,
        "upright_last_3s": end_upright,
        "max_departure_m": departure,
        "total_yaw_rad": float(yaw[-1] - yaw[0]),
        "passed": bool(tracks and not fell and end_upright and (not pure_turn or departure <= TURN_RADIUS_M)),
    }


def evaluate_checkpoint(checkpoint, cases, names, steps, *, cold_start=False, targets=COMMANDS):
    player = ZenbotPlayer(checkpoint, visuals=False)
    native = player.native
    if not native.command_gait or native.native_tracking is not None:
        raise ValueError(f"{checkpoint} must be a standalone command-conditioned Zenbot actor")
    rows = []
    for seed, trial in cases:
        floor, _ = evaluate(player, trial, seed, round(12 / DT), -1, cold_start=cold_start)
        snapshot = copy(native.data), native.history, native.memory, native.previous
        for name in names:
            # MjData includes contact/solver state; history and previous motor commands are JAX arrays.
            native.data = copy(snapshot[0])
            native.history, native.memory, native.previous = snapshot[1:]
            native.command = jnp.asarray(targets[name], dtype=jnp.float32)[None]
            poses = np.empty((steps + 1, native.model.nq))
            loads = np.empty((steps, 2))
            foot_force = np.empty((steps, len(native.foot_pair_force)))
            poses[0] = native.data.qpos
            for i in range(steps):
                native.step()
                poses[i + 1] = native.data.qpos
                loads[i] = native.support_load[:2]
                foot_force[i] = native.foot_pair_force
            row = direction_metrics(poses, loads, targets[name])
            row.update(startup_metrics(poses, targets[name]), **joint_jitter(poses[-3 * WINDOW:, 7:]),
                       **foot_contact_metrics(foot_force))
            rows.append(
                dict(seed=seed, trial=trial, clip=floor["clip"], name=name, from_held=floor["held"], **row)
            )
        print(f"{checkpoint.name} seed {seed} trial {trial} completed", flush=True)

    summary = {}
    for name in names:
        group = [r for r in rows if r["name"] == name]
        held = [r for r in group if r["from_held"]]
        summary[name] = {
            "cases": len(group),
            "from_held": len(held),
            "passed_from_held": sum(r["passed"] for r in held),
            "falls": sum(r["fell"] for r in group),
            "trials_with_foot_contact_from_held": sum(r["foot_contact_steps"] > 0 for r in held),
            "median_measured_from_held": np.median([r["mean_last_3s"] for r in held], axis=0).tolist()
            if held
            else None,
            "median_max_departure_m_from_held": float(np.median([r["max_departure_m"] for r in held]))
            if held
            else None,
        }
    return {
        "checkpoint": str(checkpoint),
        "cold_start": cold_start,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "summary": summary,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Path to the JSON results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7927, 41183, 68011, 92399, 133337])
    parser.add_argument(
        "--trials",
        nargs="+",
        type=int,
        default=[0, 6, 8, 14, 20],
        help="Floor openings cycled across seeds, or fully crossed with --all-pairs",
    )
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--cold-start", action="store_true", help="Start each floor trial at rest before adding the test perturbation")
    parser.add_argument("--commands", nargs="+", choices=COMMANDS)
    parser.add_argument("--setpoint", type=float, nargs=3, action="append", metavar=("VX", "VY", "WZ"),
                        help="Test an additional body-frame velocity setpoint; may be repeated")
    parser.add_argument("--stage-seconds", type=float, default=8.0)
    args = parser.parse_args()
    if not np.isfinite(args.stage_seconds) or args.stage_seconds < 3:
        parser.error("--stage-seconds must be finite and at least 3")
    if any(s < 0 for s in args.seeds) or any(t < 0 for t in args.trials):
        parser.error("Seeds and trial indices must be non-negative")
    if args.all_pairs:
        cases = [(s, t) for s in args.seeds for t in args.trials]
    else:
        cases = [(s, args.trials[i % len(args.trials)]) for i, s in enumerate(args.seeds)]
    steps = round(args.stage_seconds / DT)
    names = args.commands if args.commands is not None else ([] if args.setpoint else list(COMMANDS))
    targets = {name: COMMANDS[name] for name in names}
    for i, triple in enumerate(args.setpoint or [], 1):
        if not np.isfinite(triple).all():
            parser.error("--setpoint needs three finite velocities")
        targets[f"setpoint_{i}"] = triple
    reports = [evaluate_checkpoint(c, cases, targets.keys(), steps, cold_start=args.cold_start, targets=targets)
               for c in args.checkpoints]
    result = {
        "cases": cases,
        "commands": targets,
        "coordinate_order": ["forward_mps", "left_mps", "yaw_ccw_radps"],
        "floor_seconds": 12.0,
        "stage_seconds": steps * DT,
        "tail_seconds": 3.0,
        "yaw_max_radius_m": TURN_RADIUS_M,
        "checkpoints": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for report in reports:
        print(
            report["checkpoint"],
            {k: f"{v['passed_from_held']}/{v['from_held']}" for k, v in report["summary"].items()},
        )


if __name__ == "__main__":
    main()
