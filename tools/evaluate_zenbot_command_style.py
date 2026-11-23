"""Compare each commanded Zenbot gait with a contiguous, independent Motion Train AMP excerpt."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

from instantamp import env, zenbot
from instantamp.native_zenbot import ZenbotNative
from instantamp.zenbot_gait_demos import command_motion_windows, source_command_speeds
from tools.evaluate_zenbot_directions import direction_metrics, heading_kinematics
from tools.evaluate_zenbot_fall_recovery import standing_pose
from tools.evaluate_zenbot_four_skills import standing_metrics
from tools.evaluate_zenbot_walk_recovery import score
from tools.score_zenbot_gait import compare, gait_metrics

NAMES = ("stand", "walk", "run", "back", "left", "right", "turn_left", "turn_right")
FRAMES = 101  # Two full seconds between the first and final pose at 50 Hz.


def reference_excerpts(directory, commands=zenbot.DIRECTIONAL_COMMANDS):
    """Select examples by source motion and command alone, before loading any policy."""
    span = 28
    best = {}
    bank_angles = [[] for _ in NAMES]
    clip_counts = [{} for _ in NAMES]
    matched_angles = [[] for _ in NAMES]
    matched_counts = [{} for _ in NAMES]
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as clip:
            q = np.column_stack((clip["root_pos"], clip["root_quat"][:, [3, 0, 1, 2]], clip["dof_pos"]))
            for group, starts in command_motion_windows(clip, path.stem, span, match_speed=True, targets=commands):
                if len(starts):
                    matched_angles[group].append(q[starts[:, None] + np.arange(0, span, 3), 7:].reshape(-1, len(zenbot.DOF)))
                    matched_counts[group][path.name] = len(starts)
            if len(q) >= FRAMES:
                speeds = heading_kinematics(q)[0]
                sums = np.vstack((np.zeros(3), speeds.cumsum(axis=0)))
                means = (sums[FRAMES-1:] - sums[:len(q)-FRAMES+1]) / (FRAMES - 1)
            for group, starts in command_motion_windows(clip, path.stem, span):
                if not len(starts):
                    continue
                bank_angles[group].append(q[starts[:, None] + np.arange(0, span, 3), 7:].reshape(-1, len(zenbot.DOF)))
                clip_counts[group][path.name] = len(starts)
                if len(q) < FRAMES:
                    continue
                valid = np.zeros(len(q) - span + 1)
                valid[starts] = 1.
                coverage = np.convolve(valid, np.ones(FRAMES-span+1) / (FRAMES-span+1), "valid")
                target = commands[group]
                error = np.linalg.norm(means - target, axis=1)
                # Prefer excerpts with at least 80% eligible AMP starts, then closest measured speed.
                i = int(np.lexsort((error, -np.minimum(coverage, .8)))[0])
                key = (-min(coverage[i], .8), error[i], path.name, i)
                if group not in best or key < best[group][0]:
                    best[group] = (key, q[i:i+FRAMES].copy(), dict(clip=path.name, first_frame=i,
                                   last_frame=i+FRAMES-1, compatible_window_fraction=float(coverage[i])))
    if len(best) != len(NAMES):
        raise ValueError(f"Missing two-second reference excerpts for {sorted(set(range(len(NAMES))) - best.keys())}")
    refs = {}
    for group, name in enumerate(NAMES):
        _, q, info = best[group]
        bank = gait_metrics(np.concatenate(bank_angles[group]))
        matched = None
        if matched_angles[group]:
            matched = {**gait_metrics(np.concatenate(matched_angles[group])), "windows_per_clip": matched_counts[group]}
        refs[name] = (q, {**info, "bank": {**bank, "windows_per_clip": clip_counts[group]},
                          "speed_matched_bank": matched})
    return refs


def pose_style(model, data, poses):
    feet = [model.body("robot_" + name).id for name in zenbot.FEET]
    torso = model.body("robot_" + zenbot.TORSO).id
    locations, rotations = [], []
    speeds, yaw = heading_kinematics(poses)
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        locations.append(data.xpos[feet].copy())
        rotations.append(data.xmat[torso].reshape(3, 3).copy())
    feet_xyz, rot = np.asarray(locations), np.asarray(rotations)
    dz = feet_xyz[:, 0, 2] - feet_xyz[:, 1, 2]
    # Count alternating left/right swing peaks in both motion capture and simulation with the same geometry.
    events = sorted((int(frame), side) for side in (-1, 1)
                    for frame in find_peaks(side * dz, distance=7, prominence=.025)[0])
    durations = [(a[0]-b[0]) * env.DT for a, b in zip(events[1:], events[:-1]) if a[1] != b[1]]
    foot_delta = feet_xyz[:, 0] - feet_xyz[:, 1]
    width = -np.sin(yaw) * foot_delta[:, 0] + np.cos(yaw) * foot_delta[:, 1]
    step = np.cos(yaw) * foot_delta[:, 0] + np.sin(yaw) * foot_delta[:, 1]
    angles = poses[:, 7:]
    stats = gait_metrics(angles)
    qtiles = stats.pop("joint_quantiles_deg")
    for prefix, suffix in (("upper_arm", "shoulder_pitch"), ("elbow", "elbow"), ("knee", "knee")):
        for side in ("left", "right"):
            degrees = np.rad2deg(angles[:, zenbot.DOF.index(side + "_" + suffix)])
            stats[f"{side}_{prefix}_mean_deg"] = float(degrees.mean())
    stats.update(
        measured_velocity=speeds.mean(axis=0).tolist(),
        torso_tilt_deg=float(np.rad2deg(np.arccos(np.clip(rot[:, 2, 2], -1., 1.))).mean()),
        torso_pitch_deg=float(np.rad2deg(np.arcsin(np.clip(-rot[:, 2, 0], -1., 1.))).mean()),
        torso_roll_deg=float(np.rad2deg(np.arctan2(rot[:, 2, 1], rot[:, 2, 2])).mean()),
        foot_width_cm=float(100 * np.mean(np.abs(width))),
        foot_lateral_range_cm=float(100 * np.ptp(np.percentile(width, [10, 90]))),
        foot_fore_aft_range_cm=float(100 * np.ptp(np.percentile(step, [10, 90]))),
        foot_height_difference_range_cm=float(100 * np.ptp(np.percentile(dz, [10, 90]))),
        slower_ankle_planar_speed_mps=float(np.linalg.norm(np.diff(feet_xyz[:, :, :2], axis=0), axis=-1).min(axis=1).mean() / env.DT),
        alternating_swings=len(durations),
        steps_per_minute=float(60. / np.median(durations)) if len(durations) >= 2 else None,
        max_displacement_m=float(np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1).max()),
    )
    return stats, {"joint_quantiles_deg": qtiles}


def centered(poses):
    """Set only the initial horizontal origin and initial yaw; keep all relative steps and turns."""
    result = poses.copy()
    _, yaw = heading_kinematics(poses)
    rotation = Rotation.from_euler("z", -yaw[0])
    result[:, :2] = rotation.apply(np.c_[poses[:, :2] - poses[0, :2], np.zeros(len(poses))])[:, :2]
    quats = rotation * Rotation.from_quat(poses[:, [4, 5, 6, 3]])
    result[:, 3:7] = quats.as_quat()[:, [3, 0, 1, 2]]
    return result


def video(path, tracks, refs, first_seed):
    from PIL import Image, ImageDraw
    from tools.render import font, visual_model

    width, height, header, fps = 480, 320, 66, 25
    model = visual_model("zenbot23_getup", Path(__file__).resolve().parents[1] / "data/zenbot23_getup/meshes")
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.60, 125., -12.
    main_font, sub_font = font(19), font(15)
    with mujoco.Renderer(model, height, width) as renderer, TemporaryDirectory(dir=path.parent) as tmp:
        movie = Path(tmp) / "paired.mp4"
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
               f"{2*width}x{height+header}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
               "-threads", "2", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(movie)]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
            for name in NAMES:
                pair = [centered(refs[name][0]), centered(tracks[name])]
                means = [heading_kinematics(q)[0].mean(axis=0) for q in pair]
                for t in range(0, FRAMES-1, 2):
                    frame = Image.new("RGB", (2*width, height+header), (28, 35, 43))
                    draw = ImageDraw.Draw(frame)
                    for j, label in enumerate(("Motion Train AMP", f"MuJoCo policy   seed {first_seed}")):
                        pose = pair[j][t]
                        data.qpos[:] = pose
                        mujoco.mj_forward(model, data)
                        camera.lookat[:] = [*pose[:2], .67]
                        renderer.update_scene(data, camera=camera)
                        x = j * width
                        frame.paste(Image.fromarray(renderer.render()), (x, header))
                        draw.text((x+10, 3), f"{name.replace('_', ' ').upper()}  |  {label}", fill="white", font=main_font)
                        vx, vy, wz = means[j]
                        draw.text((x+10, 28), f"Forward {vx:+.2f}   Side {vy:+.2f} m/s   Turn {wz:+.2f} rad/s", fill="white", font=sub_font)
                        draw.text((x+10, 47), f"Original speed   {t*env.DT:.2f} s", fill="lightgray", font=sub_font)
                    proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError("Could not encode the native AMP comparison")
        movie.replace(path)


def evaluate(checkpoint, out, bank, seeds, render, recorded_speeds=False):
    commands = source_command_speeds(bank / "walk_run")[len(NAMES):] if recorded_speeds else zenbot.DIRECTIONAL_COMMANDS
    refs = reference_excerpts(bank / "walk_run", commands)
    player = ZenbotNative(bank / "recovery", checkpoint)
    if player.actor_width != 234 or player.native_tracking is not None:
        raise ValueError("Expected a standalone 234-input Motion Train actor")
    fk = mujoco.MjData(player.model)
    start = standing_pose(bank / "walk_run/stand_4s_120f.npz")
    summary, tracks = {}, {}
    for name, command in zip(NAMES, commands):
        reference, source = refs[name]
        expert, reference_q = pose_style(player.model, fk, reference)
        rows = []
        for seed in seeds:
            delays = np.random.default_rng(seed).integers(3, 8, 5)
            player.reset(start, np.zeros(player.model.nv), delays, command=command)
            poses, speeds, loads, flight = [player.data.qpos.copy()], [], [], []
            for _ in range(round(6/env.DT)):
                player.step()
                poses.append(player.data.qpos.copy())
                speeds.append(player.data.qvel.copy())
                loads.append(player.support_load[:2].copy())
                flight.append(player.flight.copy())
            poses, speeds, loads, flight = map(np.asarray, (poses, speeds, loads, flight))
            motion = direction_metrics(poses, loads, command, relative_active_speed=recorded_speeds)
            if name in ("walk", "run"):
                actual_gait = score(name, command[0], poses[1:], speeds, loads, flight, None)
                motion["passed"] &= actual_gait["passed"]
                motion["gait"] = actual_gait
            if name == "stand":
                motion["standing"] = standing_metrics(poses[1:], speeds, loads, start[7:])
                motion["passed"] &= motion["standing"]["bilateral_still_fraction"] >= .9
            tail = poses[-FRAMES:]
            style, actual_q = pose_style(player.model, fk, tail)
            matched = source["speed_matched_bank"]
            matched_error = compare(matched, actual_q)["joint_distribution_mae_deg"] if matched is not None else None
            rows.append(dict(seed=seed, motor_delay_substeps=delays.tolist(), motion=motion,
                             style=style, **compare(reference_q, actual_q),
                             joint_distribution_mae_to_bank_deg=compare(source["bank"], actual_q)["joint_distribution_mae_deg"],
                             joint_distribution_mae_to_speed_matched_bank_deg=matched_error))
            if seed == seeds[0]:
                tracks[name] = tail
        upright = [r for r in rows if r["motion"]["upright_last_3s"] and not r["motion"]["fell"]]
        medians = {}
        for key in expert:
            values = [r["style"][key] for r in upright if r["style"][key] is not None]
            medians[key] = np.median(values, axis=0).tolist() if values else None
        summary[name] = dict(command=command.tolist(), reference={**source, **expert},
                             trials=len(rows), upright_without_fall=len(upright), command_passes=sum(int(r["motion"]["passed"]) for r in rows),
                             policy_medians=medians, joint_distribution_mae_deg=float(np.median([r["joint_distribution_mae_deg"] for r in upright])) if upright else None,
                             joint_distribution_mae_to_bank_deg=float(np.median([r["joint_distribution_mae_to_bank_deg"] for r in upright])) if upright else None,
                             joint_distribution_mae_to_speed_matched_bank_deg=float(np.median([r["joint_distribution_mae_to_speed_matched_bank_deg"] for r in upright])) if upright and source["speed_matched_bank"] is not None else None,
                             trials_detail=rows)
    report = dict(checkpoint=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  fps=50, rollout_seconds=6, analysis_seconds=2, video_seed=seeds[0], velocity_axes=["forward_mps","left_mps","yaw_radps"],
                  reference_selection="Contiguous independent 2 s of original motion at 50 Hz, selected by command before loading the actor.",
                  bank_reference="Common direction benchmark across checkpoints, counting ten poses per eligible 540 ms window.",
                  speed_matched_bank_reference="Original gait windows compatible with the tested command and mean-speed filters. Standing uses the neutral recording.",
                  command_speeds="recorded" if recorded_speeds else "nominal",
                  active_axis_tolerance="25% of the requested speed" if recorded_speeds else "25% with the nominal absolute tolerance floor",
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  style_population="Last 2 s of upright no-fall trials, including upright trials that missed the command; all cases listed separately.",
                  summary=summary)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if render:
        video(out / "reference_vs_policy.mp4", tracks, refs, seeds[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--bank", type=Path, default=Path("data/zenbot23_walk_run_recovery"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[305711, 934837, 8217539, 1654679, 5969183])
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--recorded-speeds", action="store_true", help="Request the original sustained speed for reverse, sidesteps and pivots.")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.out, args.bank, args.seeds, args.video, args.recorded_speeds)
