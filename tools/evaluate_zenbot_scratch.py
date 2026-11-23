"""Evaluate one 234-input Zenbot policy from independent standing and fallen starts in native MuJoCo."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.spatial.transform import Rotation

from instantamp.zenbot_motion_train_obs import MOTION_TRAIN_WIDTH
from tools.evaluate_zenbot_directions import COMMANDS, TURN_RADIUS_M, direction_metrics
from tools.evaluate_zenbot_four_skills import standing_metrics
from tools.evaluate_zenbot_getup import DT, evaluate
from tools.evaluate_zenbot_rise_latency import summarize
from tools.evaluate_zenbot_walk_recovery import rise_style, score
from tools.play_zenbot_getup import ZenbotPlayer
from tools.score_zenbot_gait import compare, gait_metrics, reference_frames

ALL_COMMANDS = {**COMMANDS, "run": (2.5, 0.0, 0.0)}
ZERO = ALL_COMMANDS["still"]


def standing_start(bank, seed):
    paths = sorted((bank / "standing").glob("*.npz"))
    if not paths:
        raise ValueError(f"No standing reference clips in {bank / 'standing'}")
    rng = np.random.default_rng([seed, 2917])
    path = paths[rng.integers(len(paths))]
    with np.load(path, allow_pickle=False) as clip:
        i = rng.integers(len(clip["dof_pos"]))
        quat = clip["root_quat"][i]
        gyro = Rotation.from_quat(quat).inv().apply(clip["root_ang_vel"][i])
        pose = np.r_[clip["root_pos"][i], quat[[3, 0, 1, 2]], clip["dof_pos"][i]]
        velocity = np.r_[clip["root_lin_vel"][i], gyro, clip["dof_vel"][i]]
    return path.name, int(i), pose, velocity, rng.integers(3, 8, 5)


def simulate(native, command, steps):
    native.command = native.command.at[0].set(np.asarray(command, np.float32))
    poses = np.empty((steps + 1, native.model.nq))
    velocities = np.empty((steps, native.model.nv))
    loads = np.empty((steps, 2))
    flight = np.empty((steps, len(native.flight)), bool)
    poses[0] = native.data.qpos
    for i in range(steps):
        native.step()
        poses[i + 1] = native.data.qpos
        velocities[i] = native.data.qvel
        loads[i] = native.support_load[:2]
        flight[i] = native.flight
    return poses, velocities, loads, flight


def stance(native, steps, reference_joints):
    q, v, loads, _ = simulate(native, ZERO, steps)
    result = standing_metrics(q[1:], v, loads, reference_joints, frames=round(1.0 / DT))
    result["held"] = bool(result["upright_fraction"] == 1.0 and result["bilateral_still_fraction"] >= 0.9)
    return result


def from_standing(player, bank, seed, names, stage_steps, *, immediate_motors=False):
    native = player.native
    clip, i, q, v, delays = standing_start(bank, seed)
    if immediate_motors:
        delays = np.zeros_like(delays)
    rows = {}
    for name in names:
        native.reset(q, v, delays, command=ZERO)
        start = stance(native, round(2.0 / DT), q[7:])
        command = ALL_COMMANDS[name]
        poses, velocities, loads, flight = simulate(native, command, stage_steps)
        row = direction_metrics(poses, loads, command)
        if name in ("forward", "run"):
            task = "run" if name == "run" else "walk"
            gait = score(task, command[0], poses[1:], velocities, loads, flight, None)
            row["flight_intervals_last_2s"] = gait["flight_intervals"]
            row["gait_passed"] = gait["passed"]
            reference, _ = reference_frames(bank / "walk_run", task, command[0])
            style = compare(reference, gait_metrics(poses[-round(3.0 / DT):, 7:]))
            row["joint_distribution_mae_deg"] = style["joint_distribution_mae_deg"]
        if name.startswith("turn"):
            # This is the distance from the turn origin, matching direction_metrics' in-place test.
            row["max_turn_radius_from_start_m"] = row["max_departure_m"]
        if name == "still":
            row["stance"] = standing_metrics(poses[1:], velocities, loads, q[7:])
        stop = stance(native, round(3.0 / DT), q[7:])
        row["start_stance"], row["stop_stance"] = start, stop
        row["sequence_passed"] = bool(start["held"] and row["passed"] and row.get("gait_passed", True) and stop["held"]
                                      and (name != "still" or row["stance"]["bilateral_still_fraction"] >= 0.9))
        rows[name] = row
    return dict(seed=seed, clip=clip, frame=i, motor_delays=np.asarray(native.delays).tolist(),
                initial_state_sha256=hashlib.sha256(np.r_[q,v].astype('<f8').tobytes()).hexdigest(), commands=rows)


def report(checkpoint, bank, seeds, names, floor_steps, stage_steps, *, immediate_motors=False):
    player = ZenbotPlayer(checkpoint, bank / "recovery", visuals=False)
    if player.native.actor_width != MOTION_TRAIN_WIDTH or not player.native.command_gait or player.native.native_tracking:
        raise ValueError("Expected one 234-input command-conditioned Zenbot actor")
    ground, standing = [], []
    openings = [i for i, (_, offset) in enumerate(player.names) if offset == 0]
    for seed in seeds:
        standing.append(from_standing(player, bank, seed, names, stage_steps, immediate_motors=immediate_motors))
        for trial in openings:
            row, trace = evaluate(player, trial, seed, floor_steps, -1, immediate_motors=immediate_motors)
            initial = np.r_[trace["qpos"][0], trace["qvel"][0]].astype("<f8")
            row["initial_state_sha256"] = hashlib.sha256(initial.tobytes()).hexdigest()
            with np.load(player.reference / row["clip"], allow_pickle=False) as clip:
                row["style"] = rise_style(trace["qpos"], clip)
            ground.append(row)
    command_summary = {}
    for name in names:
        rows = [r["commands"][name] for r in standing]
        summary = dict(
            cases=len(rows), started_held=sum(r["start_stance"]["held"] for r in rows),
            tracks_and_upright=sum(r["passed"] for r in rows), falls=sum(r["fell"] for r in rows),
            stopped_held=sum(r["stop_stance"]["held"] for r in rows), full_sequence=sum(r["sequence_passed"] for r in rows),
            median_velocity_last_3s=np.median([r["mean_last_3s"] for r in rows], axis=0).tolist(),
        )
        if name.startswith("turn"):
            summary["median_max_turn_radius_from_start_m"] = float(np.median([r["max_turn_radius_from_start_m"] for r in rows]))
        if name in ("forward", "run"):
            summary["median_joint_distribution_mae_deg"] = float(np.median([r["joint_distribution_mae_deg"] for r in rows]))
        command_summary[name] = summary
    ground_summary = summarize(ground, floor_steps * DT)
    drift = [r["drift_cm"] for r in ground if r["rise"] is not None]
    style = [r["style"]["source_time_joint_mae_deg"] for r in ground if "source_time_joint_mae_deg" in r["style"]]
    ground_summary.update(
        held=sum(r["held"] for r in ground),
        refalls=sum(r["refall"] is True for r in ground),
        median_max_drift_cm_after_rise=float(np.median(drift)) if drift else None,
        median_joint_mae_at_source_speed_deg=float(np.median(style)) if style else None,
    )
    return dict(
        checkpoint=str(player.checkpoint), sha256=hashlib.sha256(player.checkpoint.read_bytes()).hexdigest(),
        seeds=seeds, immediate_motors=immediate_motors, command_targets={name: ALL_COMMANDS[name] for name in names},
        floor_seconds=floor_steps * DT, command_seconds=stage_steps * DT, pre_stand_seconds=2, stop_seconds=3,
        velocity_order=["forward_mps", "left_mps", "yaw_ccw_radps"], max_in_place_turn_radius_m=TURN_RADIUS_M,
        ground_summary=ground_summary, command_summary=command_summary,
        standing_trials=standing, ground_trials=ground,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bank", type=Path, default=Path("data/zenbot23_walk_run_recovery"))
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--commands", choices=ALL_COMMANDS, nargs="+", default=list(ALL_COMMANDS))
    p.add_argument("--immediate-motors", action="store_true", help="Apply all five motor groups without delay in every standing and floor episode")
    p.add_argument("--floor-seconds", type=float, default=12)
    p.add_argument("--stage-seconds", type=float, default=8)
    args = p.parse_args()
    if not all(np.isfinite([args.floor_seconds, args.stage_seconds])) or args.floor_seconds < 2 or args.stage_seconds < 3:
        p.error("Use at least two seconds per floor case and three seconds per command")
    if min(args.seeds) < 0 or len(args.seeds) != len(set(args.seeds)) or len(args.commands) != len(set(args.commands)):
        p.error("Use distinct nonnegative seeds and distinct commands")
    result = report(args.checkpoint, args.bank, args.seeds, args.commands, round(args.floor_seconds / DT),
                    round(args.stage_seconds / DT), immediate_motors=args.immediate_motors)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("ground_summary", "command_summary")}), flush=True)


if __name__ == "__main__":
    main()
