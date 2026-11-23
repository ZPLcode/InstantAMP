"""Evaluate one Zenbot actor through floor recovery, standing, walking, running and stopping."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from instantamp import env, zenbot
from tools.evaluate_zenbot_getup import DT, evaluate
from tools.evaluate_zenbot_walk_recovery import ground_support, rise_style, score
from tools.play_zenbot_getup import ZenbotPlayer
from tools.score_zenbot_gait import (
    compare,
    gait_metrics,
    heading_speed,
    reference_frames,
)
from tools.score_zenbot_sequence import first_run

BANK = Path("data/zenbot23_walk_run_recovery")


def standing_metrics(poses, speeds, loads, expert, frames=100):
    tail = poses[-frames:]
    joints = np.abs(tail[:, 7:] - expert)
    angle = np.rad2deg(np.arccos(np.clip(1.0 - 2.0 * np.square(tail[:, 4:6]).sum(-1), -1.0, 1.0)))
    feet = loads[-frames:, :2]
    bilateral = np.all(feet >= 0.1, axis=1)
    planar_speed = np.linalg.norm(speeds[-frames:, :2], axis=-1)
    slow = planar_speed < 0.08
    # Pitch and roll relative to the pelvis heading distinguish forward crouch from a lateral lean.
    w, x, y, z = tail[:, 3:7].T
    pitch = np.rad2deg(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    roll = np.rad2deg(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    return {
        "joints_deg": float(np.rad2deg(joints.mean())),
        "tilt_deg": float(angle.mean()),
        "height_m": float(tail[:, 2].mean()),
        "legs_deg": float(np.rad2deg(joints[:, :12].mean())),
        "arms_deg": float(np.rad2deg(joints[:, 13:].mean())),
        "bilateral_still_fraction": float((bilateral & slow).mean()),
        "pitch_deg": float(pitch.mean()),
        "roll_deg": float(roll.mean()),
        "abs_roll_deg": float(np.abs(roll).mean()),
        "foot_load_imbalance_bw": float(np.abs(feet[:, 0] - feet[:, 1]).mean()),
        "sway_radius_cm": float(100 * np.linalg.norm(tail[:, :2] - tail[:, :2].mean(0), axis=1).max()),
        "max_displacement_cm": float(100 * np.linalg.norm(tail[:, :2] - tail[0, :2], axis=1).max()),
        "peak_planar_speed_mps": float(planar_speed.max()),
        "upright_fraction": float(((tail[:, 2] > 0.58) & (tail[:, 2] <= 0.78) & (angle < 30.0)).mean()),
    }


def transition_metrics(initial_pose, poses, velocities, target):
    """Measure sustained response and path after each command switch in the same world frame."""

    def yaw(q):
        w, x, y, z = q[..., 3:7].T
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    forward = heading_speed(poses[:, [4, 5, 6, 3]], velocities)
    heading = yaw(initial_pose)
    xy = poses[:, :2] - initial_pose[:2]
    along = xy @ [np.cos(heading), np.sin(heading)]
    lateral = xy @ [-np.sin(heading), np.cos(heading)]
    heading_change = np.unwrap(np.r_[heading, yaw(poses)])[1:] - heading
    if target:
        on_target = np.abs(forward - target) <= max(0.35, 0.2 * target)
    else:
        on_target = np.linalg.norm(velocities[:, :2], axis=1) <= 0.15
    up = (poses[:, 2] > zenbot.UP_HEIGHT) & (1 - 2 * np.square(poses[:, 4:6]).sum(1) > 0.8)
    settled = first_run(on_target & up, round(0.5 / DT))
    return {
        "settled_speed_s": None if settled is None else float((settled + 1) * DT),
        "forward_distance_m": float(along[-1]),
        "max_lateral_departure_m": float(np.abs(lateral).max()),
        "max_heading_departure_deg": float(np.rad2deg(np.abs(heading_change)).max()),
        "max_distance_before_settled_m": None
        if settled is None
        else float(np.linalg.norm(xy[: settled + 1], axis=1).max()),
    }


def continuous(player, expert, references, out=None, stage_seconds=8):
    rows = {}
    for task, speed in (("walk", 1.0), ("run", 2.5), ("stop", 0.0)):
        initial_pose = player.native.data.qpos.copy()
        player.native.command = player.native.command.at[0, 0].set(speed)
        q, v, feet, flight = [], [], [], []
        for _ in range(round(stage_seconds / env.DT)):
            player.native.step()
            q.append(player.native.data.qpos.copy())
            v.append(player.native.data.qvel.copy())
            feet.append(player.native.support_load[:2].copy())
            flight.append(player.native.flight.copy())
        q, v, feet, flight = map(np.asarray, (q, v, feet, flight))
        row = score(task, speed, q, v, feet, flight, None)
        row["transition"] = transition_metrics(initial_pose, q, v, speed)
        upright_projection = 1.0 - 2.0 * np.square(q[:, 4:6]).sum(axis=1)
        row["fell"] = bool(((q[:, 2] < 0.42) | (upright_projection < 0.55)).any())
        row["command_mps"] = speed
        if task == "stop":
            row["stance"] = standing_metrics(q, v, feet, expert)
            row["passed"] = bool(
                row["final_upright"]
                and abs(row["final_forward_mps"]) < 0.15
                and row["stance"]["bilateral_still_fraction"] >= 0.9
            )
        else:
            ref_metrics = references[task]["metrics"]
            actual = gait_metrics(q[-200:, 7:])
            row["style"] = {
                **compare(ref_metrics, actual),
                **{
                    k: actual[k]
                    for k in (
                        "shoulder_swing_deg",
                        "knee_excursion_deg",
                        "hip_excursion_deg",
                        "elbow_angle_deg",
                        "arm_hip_correlation",
                    )
                },
            }
            row["actual_mps_last4s"] = float(heading_speed(q[-200:, [4, 5, 6, 3]], v[-200:]).mean())
        row["passed"] = bool(row["passed"] and not row["fell"])
        rows[task] = row
        if out is not None:
            np.savez_compressed(
                out / f"{task}.npz",
                fps=50,
                qpos=q,
                qvel=v,
                foot_load_fraction=feet,
                airborne_substeps=flight,
                command_mps=speed,
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101])
    parser.add_argument("--trials", type=int, nargs="+", help="By default test all 11 clip openings")
    parser.add_argument("--transition-trials", type=int, nargs="+", default=[0, 8])
    parser.add_argument(
        "--floor-seconds", type=float, default=12.0, help="Duration under zero command before walking"
    )
    parser.add_argument(
        "--stage-seconds", type=float, default=8.0, help="Duration of each walk, run, and stop command"
    )
    parser.add_argument("--save-transitions", action="store_true", help="Save the four stage physics traces")
    parser.add_argument("--cold-start", action="store_true", help="Start each floor trial at rest before adding the test perturbation")
    parser.add_argument("--bank", type=Path, default=BANK)
    parser.add_argument("--stand-reference", type=Path, help="Score standing against this AMP clip")
    parser.add_argument(
        "--gait-reference", type=Path, help="Use saved walking and running reference statistics"
    )
    args = parser.parse_args()
    if not np.isfinite(args.floor_seconds) or args.floor_seconds < 2.0:
        parser.error("Standing is scored over at least two seconds")
    if not np.isfinite(args.stage_seconds) or args.stage_seconds < 4.0:
        parser.error("Gait style is scored over at least four seconds")
    floor_steps, stage_steps = round(args.floor_seconds / DT), round(args.stage_seconds / DT)
    args.out.mkdir(exist_ok=True, parents=True)
    player = ZenbotPlayer(args.checkpoint, args.bank / "recovery", visuals=False)
    if not player.native.command_gait:
        parser.error("Expected a command-conditioned single Zenbot actor")
    trials = args.trials or [i for i, item in enumerate(player.names) if item[1] == 0.0]
    if not trials or any(i < 0 or i >= len(player.names) for i in trials):
        parser.error("Choose at least one valid recovery trial")
    references = {}
    if args.gait_reference:
        reference = json.loads(args.gait_reference.read_text())
        if reference["command_mps"] != {"walk": 1.0, "run": 2.5}:
            parser.error("Gait reference speeds must match the 1.0 m/s walk and 2.5 m/s run")
        references = reference["gaits"]
    else:
        for task, speed in (("walk", 1.0), ("run", 2.5)):
            stats, samples = reference_frames(args.bank / "walk_run", task, speed)
            references[task] = {"metrics": stats, "samples": samples}
    stand, stand_heights = [], []
    stand_files = [args.stand_reference] if args.stand_reference else sorted((args.bank / "standing").glob("*.npz"))
    for p in stand_files:
        with np.load(p) as clip:
            stand.append(clip["dof_pos"])
            stand_heights.append(clip["root_pos"][:, 2])
    expert = np.median(np.vstack(stand), axis=0)
    references["stand"] = {
        "files": [str(p) for p in stand_files],
        "clips": len(stand),
        "frames": sum(map(len, stand)),
        "median_height_m": float(np.median(np.concatenate(stand_heights))),
        "median_joint_rad": expert.tolist(),
    }
    floor, sequences = [], []
    for seed in args.seeds:
        for trial in trials:
            player.speed = 0.0
            row, arrays = evaluate(player, trial, seed, floor_steps, -1, cold_start=args.cold_start)
            q, v, loads = arrays["qpos"], arrays["qvel"], arrays["support_load_bw"]
            row["ground_support"] = ground_support(q[1:], loads, 0)
            source = args.bank / "recovery" / row["clip"]
            with np.load(source) as z:
                start = round(row["offset"] * float(z["fps"]))
                ref = {k: z[k][start:] for k in ("root_pos", "root_quat", "dof_pos")}
                ref["fps"] = float(z["fps"])
            row["rise_style_deg"] = rise_style(q, ref)["phase_aligned_joint_mae_deg"]
            row["stance"] = standing_metrics(q, v, loads, expert)
            # A 72-second floor trial adds a complete one-minute hold after the normal 12-second check.
            if floor_steps > round(12 / DT):
                hold_frames = floor_steps - round(12 / DT)
                row["extended_stand"] = standing_metrics(q, v, loads, expert, frames=hold_frames)
                row["extended_stand"]["seconds"] = hold_frames * DT
            row["first_rise_held"] = bool(
                row["rise"] is not None
                and not row["refall"]
                and row["held"]
                and row["pre_rise_attempts"]["retries"] == 0
            )
            floor.append(row)
            if trial in args.transition_trials:
                dst = args.out / f"{seed}_{trial:02}"
                if args.save_transitions:
                    dst.mkdir(exist_ok=True)
                    np.savez_compressed(dst / "getup_stand.npz", **arrays)
                stages = continuous(
                    player,
                    expert,
                    references,
                    dst if args.save_transitions else None,
                    stage_seconds=stage_steps * DT,
                )
                sequences.append(
                    {
                        "seed": seed,
                        "trial": trial,
                        "from_held": bool(row["held"]),
                        "stages": stages,
                        "passed": bool(row["first_rise_held"] and all(s["passed"] for s in stages.values())),
                    }
                )
    held = [r for r in floor if r["held"]]

    def median(values):
        return float(np.median(values)) if values else None

    summary = {
        "floor": len(floor),
        "rose": sum(r["rise"] is not None for r in floor),
        "held": len(held),
        "strict_first_rise": sum(r["first_rise_held"] for r in floor),
        "refalls": sum(r["refall"] is True for r in floor),
        "median_standing_joint_deg": median([r["stance"]["joints_deg"] for r in held]),
        "median_standing_tilt_deg": median([r["stance"]["tilt_deg"] for r in held]),
        "median_standing_abs_roll_deg": median([r["stance"]["abs_roll_deg"] for r in held]),
        "median_standing_sway_radius_cm": median([r["stance"]["sway_radius_cm"] for r in held]),
        "median_standing_load_imbalance_bw": median([r["stance"]["foot_load_imbalance_bw"] for r in held]),
        "hands_loaded_then_transferred": sum(
            r["real_hand_load_s"] is not None and r["transfer_s"] is not None for r in floor
        ),
        "sustained_knee_support": sum(r["ground_support"]["knee_sustained"] for r in floor),
        "median_knee_support_s": median([r["ground_support"]["knee_support_s"] for r in floor]),
        "median_rise_s": median([r["rise"] for r in floor if r["rise"] is not None]),
        "median_reference_rise_s": median(
            [r["reference_rise_s"] for r in floor if r["reference_rise_s"] is not None]
        ),
        "median_rise_style_deg": median(
            [r["rise_style_deg"] for r in floor if r["rise_style_deg"] is not None]
        ),
        "median_postrise_pelvis_cm": median([r["drift_cm"] for r in held]),
        "median_postrise_foot_cm": median([r["feet_cm"] for r in held]),
        "sequences": len(sequences),
        "whole_sequence": sum(r["passed"] for r in sequences),
    }
    for task in ("walk", "run", "stop"):
        rows = [r["stages"][task] for r in sequences]
        summary[task] = {
            "passed": sum(r["passed"] for r in rows),
            "median_speed": median([r["final_forward_mps"] for r in rows]),
            "median_flights": median([r["flight_intervals"] for r in rows]),
            "median_switches": median([r["single_support_switches"] for r in rows]),
            "settled_count": sum(r["transition"]["settled_speed_s"] is not None for r in rows),
            "median_settled_speed_s": median(
                [
                    r["transition"]["settled_speed_s"]
                    for r in rows
                    if r["transition"]["settled_speed_s"] is not None
                ]
            ),
            "median_lateral_departure_m": median([r["transition"]["max_lateral_departure_m"] for r in rows]),
        }
        if task != "stop":
            summary[task]["median_style_deg"] = median(
                [r["style"]["joint_distribution_mae_deg"] for r in rows if r["passed"]]
            )
    report = {
        "checkpoint": str(args.checkpoint),
        "cold_start": args.cold_start,
        "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": floor_steps * DT,
        "stage_seconds": stage_steps * DT,
        "seeds": args.seeds,
        "trials": trials,
        "references": references,
        "summary": summary,
        "floor": floor,
        "transitions": sequences,
    }
    tmp = args.out / "summary.tmp"
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tmp.replace(args.out / "summary.json")
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
