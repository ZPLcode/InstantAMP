"""Compare a named G1 get-up evaluation with its 50 Hz source motion, at the recorded time.

Example: python tools/compare_recovery_rollouts.py runs/g1_getup/rollouts/0002_avg_recovery.npz \
    --out runs/g1_getup/style_reference.json --video runs/g1_getup/style_reference.mp4
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from instantamp import env, g1_getup
from instantamp.sim import build_scene
from tools.render import font, visual_model
from tools.render_g1_comparison import center_camera
from tools.score_g1_tracking import _body_local, _poses, _yaw_local, load_native_reference


def matrices(qpos):
    result = np.empty((len(qpos), 9))
    for q, mat in zip(qpos, result):
        mujoco.mju_quat2Mat(mat, q[3:7])
    return result.reshape(-1, 3, 3)


def landmark_fk(model, poses, ids):
    points = np.empty((len(poses), len(ids), 3))
    data = mujoco.MjData(model)
    for frame, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        points[frame] = data.xpos[ids]
    return points


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def first_upright(qpos, hold_frames=10):
    """First frame in a sustained 0.20 s posture run. This uses no load information."""
    if len(qpos) < hold_frames:
        return None
    upright = (
        (qpos[:, 2] > g1_getup.UP_HEIGHT)
        & (qpos[:, 2] <= 2 * g1_getup.RISE_HEIGHT - g1_getup.UP_HEIGHT)
        & (matrices(qpos)[:, 2, 2] > np.cos(np.deg2rad(30.0)))
    )
    starts = np.flatnonzero(np.convolve(upright.astype(int), np.ones(hold_frames, dtype=int), "valid") == hold_frames)
    return int(starts[0]) if len(starts) else None


def source_audit(model, pose, velocity, stored, native):
    rotations = matrices(pose)
    derived = np.empty((len(pose) - 1, model.nv))
    for i in range(len(derived)):
        mujoco.mj_differentiatePos(model, derived[i], env.DT, pose[i], pose[i + 1])
    midpoint = (derived[1:] + derived[:-1]) / 2
    free_vel = velocity[1:-1].copy()
    com = model.body_ipos[model.body("robot_pelvis").id]
    free_vel[:, :3] -= np.einsum("tij,tj->ti", rotations[1:-1], np.cross(free_vel[:, 3:6], com))
    gaps = 100 * np.linalg.norm(native - stored, axis=-1)
    diff = midpoint - free_vel
    return {
        "frames": len(pose),
        "fk_to_stored_landmark_mean_cm": dict(zip(g1_getup.KEY_BODIES, gaps.mean(0).tolist())),
        "fk_to_stored_landmark_max_cm": float(gaps.max()),
        "central_difference_joint_velocity_rmse_rad_s": rms(diff[:, 6:]),
        "central_difference_root_ang_velocity_vector_rmse_rad_s": rms(np.linalg.norm(diff[:, 3:6], axis=1)),
        "central_difference_root_lin_velocity_vector_rmse_m_s": rms(np.linalg.norm(diff[:, :3], axis=1)),
        "root_lin_velocity_uncorrected_vector_rmse_m_s": rms(
            np.linalg.norm(midpoint[:, :3] - velocity[1:-1, :3], axis=1)
        ),
    }


def errors(reference, actual, stored, ref_fk, act_fk):
    rr, ar = matrices(reference), matrices(actual)
    gravity_dot = np.einsum("ti,ti->t", rr[:, 2], ar[:, 2])
    relative_trace = (rr * ar).sum((1, 2))
    # xyz distances are Euclidean per individual landmark. Six native reference
    # landmarks come directly from the exported clips; knees are reconstructed.
    rp = np.concatenate((stored, ref_fk[:, 6:]), axis=1)
    out = {
        "pelvis_height_mae_cm": 100 * abs(reference[:, 2] - actual[:, 2]),
        "pelvis_height_rmse_cm": 100 * (reference[:, 2] - actual[:, 2]),
        "pelvis_xy_rmse_cm": 100 * np.linalg.norm(reference[:, :2] - actual[:, :2], axis=1),
        "pelvis_tilt_rmse_deg": np.degrees(np.arccos(np.clip(gravity_dot, -1, 1))),
        "pelvis_attitude_rmse_deg": np.degrees(np.arccos(np.clip((relative_trace - 1) / 2, -1, 1))),
        "joint_rmse_rad": actual[:, 7:] - reference[:, 7:],
    }
    for label, refp, actp in (
        ("world", rp, act_fk),
        ("pelvis", _body_local(rp, reference), _body_local(act_fk, actual)),
        ("heading", _yaw_local(rp, reference), _yaw_local(act_fk, actual)),
    ):
        distances = 100 * np.linalg.norm(refp - actp, axis=2)
        out[f"six_landmark_{label}_rmse_cm"] = distances[:, :6]
        for j, part in enumerate(("ankle", "wrist", "shoulder", "knee")):
            out[f"{part}_{label}_rmse_cm"] = distances[:, 2 * j : 2 * j + 2]
    return out


def summarize(measures):
    return {name: (float(values.mean()) if "mae" in name else rms(values)) for name, values in measures.items()}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(archive, motion_dir, physics_path=None, timeline_writer=None, *, show_hold=False):
    with np.load(archive, allow_pickle=False) as recording:
        if physics_path is None and "physics_file" in recording:
            physics_path = archive.parent / str(recording["physics_file"])
        expected_hash = str(recording["physics_sha256"]) if "physics_sha256" in recording else None
        world = str(recording["world"]) if "world" in recording else "nominal"
        noise = bool(recording["actor_sensor_noise"]) if "actor_sensor_noise" in recording else False
        seed = int(recording["seed"]) if "seed" in recording else 42
        jitter_seed = int(recording.get("reset_jitter_seed",-1))
        push_s = float(recording.get("push_s", -1.))
        push_angle = float(recording.get("push_angle_deg", 0.))
        policy_hash = str(recording["checkpoint_sha256"]) if "checkpoint_sha256" in recording else None
        protocol = str(recording["starts_protocol"]) if "starts_protocol" in recording else "fallen"
    if physics_path is not None:
        model_hash = file_hash(physics_path)
        if expected_hash is not None and expected_hash != model_hash:
            raise ValueError("Saved evaluation geometry and requested FK model disagree")
        model = mujoco.MjModel.from_binary_path(str(physics_path))
    else:
        model = build_scene(robot_name="g1_getup")
        binary = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, None, binary)
        model_hash = hashlib.sha256(binary).hexdigest()
    dofs = tuple(model.joint(i).name.removeprefix("robot_").removesuffix("_joint") for i in range(1, model.njnt))
    if dofs != tuple(env.DOF):
        raise ValueError("The MuJoCo joint ordering differs from the G1 reference convention")
    bodies = (*g1_getup.KEY_BODIES, "left_knee_link", "right_knee_link")
    ids = [model.body("robot_" + name).id for name in bodies]
    sources, audits = {}, {}
    for path in sorted(motion_dir.glob("*.npz")):
        pose, vel = load_native_reference(path)
        with np.load(path, allow_pickle=False) as clip:
            points = clip["key_body_pos"].astype(float)
        fk = landmark_fk(model, pose, ids)
        sources[path.name] = (pose, points, fk)
        audits[path.name] = {"sha256": file_hash(path), **source_audit(model, pose, vel, points, fk[:, :6])}

    with np.load(archive, allow_pickle=False) as recording:
        if str(recording["robot"]) != "g1_getup" or not np.isclose(float(recording["dt"]), env.DT):
            raise ValueError("Expected a named G1 get-up archive at 50 Hz")
        names, offsets, qpos = recording["clips"], recording["offsets_s"], recording["qpos"]
        reset_hash = hashlib.sha256(qpos[0].astype("<f8").tobytes()).hexdigest()
        ended, success = recording["terminated"], recording["success"]
        if qpos.shape != (ended.shape[0] + 1, len(names), model.nq):
            raise ValueError("qpos and transition arrays disagree")
        policy, which = str(recording["checkpoint"]), str(recording["which"])
    pooled, rise_pool, cases, paths = {}, {}, [], []
    for trial, (name, offset) in enumerate(zip(names, offsets)):
        src, points, src_fk = sources[name]
        start = round(offset / env.DT)
        term = np.flatnonzero(ended[:, trial])
        eligible = min(len(src) - start, len(qpos)) - 1
        steps = min(eligible, int(term[0]) if len(term) else eligible)
        sl = slice(start, start + steps + 1)
        ref, act = src[sl].copy(), _poses(model, qpos[: steps + 1, trial])
        translation = np.r_[act[0, :2] - ref[0, :2], 0.0]
        ref[:, :3] += translation
        fk = landmark_fk(model, act, ids)
        source_tilt = float(np.degrees(np.arccos(np.clip(matrices(ref[:1])[0, 2, 2], -1, 1))))
        case = {
            "source_start_height_m": float(ref[0, 2]),
            "physical_start_height_m": float(act[0, 2]),
            "source_start_tilt_deg": source_tilt,
            "qualifies_fallen_height": bool(act[0, 2] <= 0.5),
            "source_upright_posture": bool(
                g1_getup.UP_HEIGHT < ref[0, 2] <= 2 * g1_getup.RISE_HEIGHT - g1_getup.UP_HEIGHT and source_tilt < 30.0
            ),
            "trial": trial,
            "clip": str(name),
            "offset_s": float(offset),
            "hold_passed": bool(success[trial]),
            "available_reference_transitions": eligible,
            "scored_transitions": steps,
            "first_termination_s": float(term[0] + 1) * env.DT if len(term) else None,
        }
        source_rise = first_upright(src[start:])
        # The stand may start after the reference clip ends; use the entire valid physical episode here.
        physical_steps = int(term[0]) if len(term) else len(qpos) - 1
        actual_rise = first_upright(_poses(model, qpos[: physical_steps + 1, trial]))
        case["source_rise_s"] = None if source_rise is None else source_rise * env.DT
        case["actual_rise_s"] = None if actual_rise is None else actual_rise * env.DT
        case["complete_nonempty_rise_window"] = source_rise is not None and 0 < source_rise <= steps
        if steps:
            values = errors(ref[1:], act[1:], points[sl][1:] + translation, src_fk[sl][1:] + translation, fk[1:])
            if case["complete_nonempty_rise_window"]:
                rise_values = {field: value[:source_rise] for field, value in values.items()}
                case["rise_errors"] = summarize(rise_values)
                for field, value in rise_values.items():
                    rise_pool.setdefault(field, []).append(value)
            if timeline_writer is not None:
                reference_tilt = np.degrees(np.arccos(np.clip(matrices(ref[1:])[:, 2, 2], -1, 1)))
                policy_tilt = np.degrees(np.arccos(np.clip(matrices(act[1:])[:, 2, 2], -1, 1)))
                for i in range(steps):
                    elapsed = (i + 1) * env.DT
                    row = {
                        "trial": trial,
                        "clip": str(name),
                        "elapsed_s": elapsed,
                        "source_time_s": offset + elapsed,
                        "reference_x_m": ref[i + 1, 0],
                        "reference_y_m": ref[i + 1, 1],
                        "reference_height_m": ref[i + 1, 2],
                        "policy_x_m": act[i + 1, 0],
                        "policy_y_m": act[i + 1, 1],
                        "policy_height_m": act[i + 1, 2],
                        "reference_tilt_deg": reference_tilt[i],
                        "policy_tilt_deg": policy_tilt[i],
                        **summarize({field: value[i : i + 1] for field, value in values.items()}),
                    }
                    if not timeline_writer.fieldnames:
                        timeline_writer.fieldnames = list(row)
                        timeline_writer.writeheader()
                    timeline_writer.writerow(row)
            for field, value in values.items():
                pooled.setdefault(field, []).append(value)
            case["errors"] = summarize(values)
            case["by_second"] = [
                {
                    "from_s_exclusive": k * env.DT,
                    "to_s_inclusive": min(k + 50, steps) * env.DT,
                    **summarize(
                        {
                            key: arr[k : k + 50]
                            for key, arr in values.items()
                            if key
                            in (
                                "joint_rmse_rad",
                                "pelvis_height_mae_cm",
                                "pelvis_tilt_rmse_deg",
                                "six_landmark_pelvis_rmse_cm",
                            )
                        }
                    ),
                }
                for k in range(0, steps, 50)
            ]
        cases.append(case)
        if show_hold and physical_steps > steps:
            act = _poses(model, qpos[: physical_steps + 1, trial])
        paths.append((ref, act))
    all_values = {name: np.concatenate(arrays) for name, arrays in pooled.items()}
    joint_errors = all_values.get("joint_rmse_rad")
    report = {
        "archive": str(archive),
        "archive_sha256": file_hash(archive),
        "checkpoint": policy,
        "which": which,
        "physics_mjb": str(physics_path) if physics_path is not None else None,
        "physics_sha256": model_hash,
        "starts_protocol": protocol,
        "world": world,
        "actor_sensor_noise": noise,
        "seed": seed,
        "reset_jitter_seed": jitter_seed,
        "push_s": push_s,
        "push_angle_deg": push_angle,
        "reset_pose_sha256": reset_hash,
        "dt_s": env.DT,
        "alignment": "Reference XY is translated once to match the initial simulation XY. Z and orientation are unmodified. Both trajectories use the native 50 Hz clock. Frame 0 is drawn; scoring starts after the first step and ends at the source clip boundary or before the first termination.",
        "landmark_frames": "World is the initial-XY-aligned world. Pelvis removes full pelvis position and orientation; heading removes pelvis position and yaw. Source ankles, wrists and shoulders are the six stored world-space landmarks; knees use native MuJoCo forward kinematics.",
        "aggregation": "Errors pool all eligible post-step frames across trials. Joint RMSE averages over 29 joint coordinates. Limb RMSE averages squared Euclidean distances over the named left/right pair. Pelvis tilt compares the world vertical expressed in each pelvis frame.",
        "successes": int(success.sum()),
        "total_trials": len(cases),
        "source_clips_absent_from_this_protocol": sorted(set(sources) - set(names)),
        "scored_transitions": sum(item["scored_transitions"] for item in cases),
        "available_reference_transitions": sum(item["available_reference_transitions"] for item in cases),
        "pooled": summarize(all_values),
        "rise_phase": {
            "definition": "Reference rise ends at the first frame starting ten consecutive 50 Hz frames with pelvis height in (0.65,0.85] m and tilt under 30 degrees. Pool only complete nonempty reference windows, starting at the first post-step. Simulated rise uses the same posture test, independent from final load-bearing hold success.",
            "complete_nonempty_trials": sum(c["complete_nonempty_rise_window"] for c in cases),
            "physical_trials_reaching_posture": sum(c["actual_rise_s"] is not None for c in cases),
            "pooled": summarize({field: np.concatenate(values) for field, values in rise_pool.items()}),
        },
        "joint_rmse_by_name_rad": {}
        if joint_errors is None
        else dict(zip(dofs, np.sqrt(np.mean(joint_errors**2, axis=0)).tolist())),
        "source_audit": audits,
        "trials": cases,
    }
    checkpoint = archive.parent.parent / "ckpt" / policy
    if checkpoint.is_file():
        current_hash = file_hash(checkpoint)
        if policy_hash is not None and current_hash != policy_hash:
            raise ValueError("Recorded checkpoint SHA differs from the local checkpoint with the same name")
        policy_hash = current_hash
    if policy_hash is not None:
        report["checkpoint_sha256"] = policy_hash
    return report, paths


def render_video(output, report, paths, trial, baseline=None, policy_label=None, baseline_label=None, palm_loads=None, seconds=None):
    if trial is not None and not 0 <= trial < len(paths):
        raise ValueError("Requested trial index is outside the recording")
    if baseline is not None:
        baseline_report, baseline_paths = baseline

        def signature(r):
            return (
                r["physics_sha256"],
                r["starts_protocol"],
                r["world"],
                r["seed"],
                r["reset_jitter_seed"], r["reset_pose_sha256"],
                r["push_s"], r["push_angle_deg"],
                [(c["clip"], c["offset_s"]) for c in r["trials"]],
            )

        if signature(report) != signature(baseline_report):
            raise ValueError("Three-column rendering requires identical starts, world and nominal model")
    model = visual_model("g1_getup")
    data = mujoco.MjData(model)
    width, height, header = 352, 256, (80 if palm_loads else 62)
    shape = ((3 if baseline is not None else 2) * width, height + header)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{shape[0]}x{shape[1]}",
        "-r",
        "25",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(output),
    ]
    fonts = font(16), font(12)
    with mujoco.Renderer(model, height, width) as renderer, subprocess.Popen(command, stdin=subprocess.PIPE) as writer:
        for case, pair in zip(report["trials"], paths):
            if trial is not None and case["trial"] != trial:
                continue
            if baseline is not None:
                base_case = baseline_report["trials"][case["trial"]]
                base_ref, base_act = baseline_paths[case["trial"]]
                reference = pair[0] if len(pair[0]) >= len(base_ref) else base_ref
                trajectories = (reference, base_act, pair[1])
                policies = ((baseline_label or f"Base {baseline_report['checkpoint']}", base_case),
                            (policy_label or f"Policy {report['checkpoint']}", case))
            else:
                trajectories = pair
                policies = ((policy_label or f"Policy {report['checkpoint']}", case),)
            if seconds is not None:
                trajectories = tuple(path[:round(seconds / env.DT)+1] for path in trajectories)
            camera = center_camera(model, data, trajectories)
            for frame in range(0, max(len(path) for path in trajectories[1:]), 2):
                im = Image.new("RGB", shape, "#18242f")
                draw = ImageDraw.Draw(im)
                source_frame = min(frame, len(trajectories[0]) - 1)
                reference_label = "Reference motion" if frame == source_frame else "Reference ended   |   final pose"
                labels = [reference_label]
                for role, policy_case in policies:
                    hold = "passed" if policy_case["hold_passed"] else "failed"
                    labels.append(f"{role}  |  final hold {hold}")
                for column, path in enumerate(trajectories):
                    x = column * width
                    pose = path[min(frame, len(path) - 1)]
                    draw.text((x + 8, 2), labels[column], font=fonts[0], fill="white")
                    draw.text(
                        (x + 8, 23),
                        f"Trial {case['trial']:02d}   {Path(case['clip']).stem}",
                        font=fonts[1],
                        fill="#cddbe9",
                    )
                    elapsed = frame * report["dt_s"]
                    source_time = source_frame * report["dt_s"] + case["offset_s"]
                    status = f"Elapsed {elapsed:.2f}s   source {source_time:.2f}s   pelvis z {pose[2]:.2f}m"
                    if column:
                        first_rise = policies[column-1][1]["actual_rise_s"]
                        if first_rise is not None and elapsed >= first_rise and frame < len(path):
                            rise_xy = path[round(first_rise / report['dt_s']), :2]
                            distance = 100 * np.linalg.norm(pose[:2] - rise_xy)
                            status = f"Elapsed {elapsed:.2f}s   XY from first upright {distance:.1f} cm"
                    draw.text(
                        (x + 8, 40),
                        status,
                        font=fonts[1],
                        fill="#cddbe9",
                    )
                    if palm_loads:
                        if column == 0:
                            label = "Motion-capture reference"
                        else:
                            loads = palm_loads[column-1]
                            left, right = loads[min(max(0, frame-1), len(loads)-1), case["trial"]] if frame else (0.,0.)
                            label = f"Palm load (body weight)    L {left:.0%}    R {right:.0%}"
                        draw.text((x+8,59),label,font=fonts[1],fill="#cddbe9")
                    if column and frame >= len(path):
                        draw.text((x + 8, header + 20), "Recorded episode ended", font=fonts[0], fill="white")
                        continue
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    im.paste(Image.fromarray(renderer.render()), (x, header))
                writer.stdin.write(im.tobytes())
        writer.stdin.close()
        if writer.wait():
            raise RuntimeError("ffmpeg failed to encode the comparison video")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--motion-dir", type=Path, default=Path("data/g1_getup/recovery"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--timeline",
        type=Path,
        help="write paired base trajectories and limb/joint errors for every scored 50 Hz frame as CSV",
    )
    parser.add_argument(
        "--physics-model", type=Path, help="Frozen .mjb for an existing archive lacking a physics_file field"
    )
    parser.add_argument(
        "--video", type=Path, help="Write a synchronized real-time MP4 for all source-supported trial frames"
    )
    parser.add_argument("--trial", type=int, help="Render only this zero-based trial (metrics always cover all trials)")
    parser.add_argument("--policy-label", help="Name the evaluated policy column in the video")
    parser.add_argument("--baseline-label", help="Name the baseline column in a three-column video")
    parser.add_argument(
        "--show-hold",
        action="store_true",
        help="Continue the physical rollout after the source ends, keeping the last reference frame visible",
    )
    parser.add_argument(
        "--baseline-archive",
        type=Path,
        help="add a third column showing a baseline on the same starts, scale and clock",
    )
    parser.add_argument("--video-seconds",type=float,help="Maximum actual simulated seconds shown for each trial")
    parser.add_argument("--palm-load",action="store_true",help="Overlay the measured normal force at each physical palm")
    args = parser.parse_args()
    if args.video_seconds is not None and args.video_seconds <= 0:
        parser.error("--video-seconds must be positive")
    if args.baseline_archive and not args.video:
        parser.error("--baseline-archive requires --video")
    if args.timeline:
        args.timeline.parent.mkdir(parents=True, exist_ok=True)
        with args.timeline.open("w", newline="") as stream:
            report, paths = compare(
                args.archive,
                args.motion_dir,
                args.physics_model,
                csv.DictWriter(stream, fieldnames=[]),
                show_hold=args.show_hold,
            )
    else:
        report, paths = compare(args.archive, args.motion_dir, args.physics_model, show_hold=args.show_hold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"scored_transitions": report["scored_transitions"], **report["pooled"]}, indent=2), flush=True)
    if args.video:
        baseline = (
            compare(args.baseline_archive, args.motion_dir, show_hold=args.show_hold) if args.baseline_archive else None
        )
        palms=[]
        if args.palm_load:
            for path in ([args.baseline_archive,args.archive] if args.baseline_archive else [args.archive]):
                with np.load(path,allow_pickle=False) as archive:
                    if "hand_load_bw" in archive:
                        palms.append(archive["hand_load_bw"][:,:,:,0])
                    else:
                        names=archive["support_body_names"].tolist()
                        palms.append(archive["support_load_bw"][:,:,[names.index(f"{side}_wrist_yaw_link") for side in ("left","right")]])
        render_video(args.video, report, paths, args.trial, baseline, args.policy_label, args.baseline_label, palms, args.video_seconds)


if __name__ == "__main__":
    main()
