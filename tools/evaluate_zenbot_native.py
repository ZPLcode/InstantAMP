"""Validate a Zenbot reference bank, or run its exported 480-input policy in native CPU MuJoCo."""

import argparse
import hashlib
import json
import os
from pathlib import Path

if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.spatial.transform import Rotation

from instantamp import env, zenbot
from instantamp.evaluate import fallen_starts
from instantamp.native_zenbot import ZenbotNative
from tools.convert_motions import velocities
from tools.retarget_g1_zenbot import Skeleton


def validate(directory):
    skeleton = Skeleton(False)
    m, d = skeleton.model, skeleton.data
    summary = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as f:
            pos, rot, q = f["root_pos"], f["root_quat"], f["dof_pos"]
            n = len(q)
            if (float(f["fps"]) != 50 or n < 10 or q.shape != (n, 23) or pos.shape != (n, 3)
                    or rot.shape != (n, 4) or f["key_body_pos"].shape != (n, 6, 3)
                    or not all(np.isfinite(f[k]).all() for k in f.files)):
                raise ValueError(f"Invalid Zenbot clip layout in {path}")
            if np.max(np.abs(np.linalg.norm(rot, axis=1) - 1)) > 2e-5:
                raise ValueError(f"Nonunit XYZW orientation in {path}")
            expected = velocities(pos, rot, q, 50.)
            error = max(np.max(np.abs(f[k] - v)) for v, k in zip(expected, ("root_lin_vel", "root_ang_vel", "dof_vel")))
            floor, separation, points = [], [], []
            for xyz, quat, joints in zip(pos, rot, q):
                skeleton.set_pose(xyz, quat, joints)
                points.append(d.xpos[skeleton.support].copy())
                floor.append(skeleton.lowest())
                separation.extend(c.dist for c in d.contact if c.geom1 and c.geom2)
            limits = m.jnt_range[1:]
            excess = max(0., (q - limits[:, 1]).max(), (limits[:, 0] - q).max())
            landmark_error = np.max(np.abs(np.asarray(points) - f["key_body_pos"]))
            row = dict(clip=path.name, frames=n, start_height_m=float(pos[0, 2]), end_height_m=float(pos[-1, 2]),
                       min_floor_m=float(min(floor)), worst_self_overlap_m=float(-min(separation, default=0.)),
                       joint_limit_error_rad=float(excess), max_speed_ratio=float(np.max(abs(f["dof_vel"]) / zenbot.SPEED)),
                       max_landmark_error_m=float(landmark_error), max_velocity_error=float(error))
            if min(floor) < -.001 or excess > 1e-5 or landmark_error > 1e-5 or error > 5e-4:
                raise ValueError(f"Reference validation failed: {row}")
            summary.append(row)
    if not summary:
        raise ValueError(f"No Zenbot recovery recordings in {directory}")
    return dict(mass_kg=float(m.body_mass.sum()), joints=m.nu, frames=sum(r["frames"] for r in summary),
                clips=summary)


def standing_starts(directory):
    """Diagnostic restarts at the first upright pose and near the end of each reference."""
    names, poses, velocities = [], [], []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as clip:
            pos, quat = clip["root_pos"], clip["root_quat"]
            vertical = 1. - 2. * (quat[:, 0] ** 2 + quat[:, 1] ** 2)
            eligible = np.flatnonzero((pos[:, 2] > zenbot.UP_HEIGHT) & (vertical > np.cos(np.pi / 6)))
            if not len(eligible):
                continue
            for i in sorted({int(eligible[0]), max(0, len(pos) - 12)}):
                names.append((path.name, i / float(clip["fps"])))
                poses.append(np.r_[0., 0., pos[i, 2] + zenbot.RESET_LIFT, quat[i, [3, 0, 1, 2]], clip["dof_pos"][i]])
                gyro = Rotation.from_quat(quat[i]).inv().apply(clip["root_ang_vel"][i])
                velocities.append(np.r_[clip["root_lin_vel"][i], gyro, clip["dof_vel"][i]])
    return names, np.asarray(poses, np.float32), np.asarray(velocities, np.float32)


def evaluate(directory, checkpoint, seconds, seed, starts="fallen", sensor_noise=0., trajectories=None):
    names, poses, velocities = (fallen_starts(directory, zenbot.RESET_LIFT) if starts == "fallen"
                                else standing_starts(directory))
    poses, velocities = np.asarray(poses), np.asarray(velocities)
    native = ZenbotNative(directory, checkpoint, seed, sensor_noise)
    rng = np.random.default_rng(seed)
    result = []
    if trajectories is not None:
        trajectories = Path(trajectories)
        trajectories.mkdir(parents=True, exist_ok=True)
        checkpoint_hash = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    steps = round(seconds / env.DT)
    for index, (name, start) in enumerate(names):
        delays = rng.integers(3, 8, size=5)
        native.reset(poses[index], velocities[index], delays)
        d = native.data
        if trajectories is not None:
            q_trace, v_trace = [d.qpos.copy()], [d.qvel.copy()]
            foot_trace, action_trace, stance_trace = [], [], []
        with np.load(directory / name, allow_pickle=False) as reference:
            reference_q = reference["dof_pos"]
        rows, joint_mse = [], []
        first_frame = round(start / env.DT)
        for tick in range(steps):
            feet_weight = native.step()
            if trajectories is not None:
                q_trace.append(d.qpos.copy())
                v_trace.append(d.qvel.copy())
                foot_trace.append(feet_weight)
                action_trace.append(np.asarray(native.previous).copy())
                stance_trace.append(0. if native.memory is None else float(native.memory[0, 0]))
            upright = 1. - 2. * np.sum(d.qpos[4:6]**2)
            rows.append((d.qpos[2], upright, feet_weight, np.linalg.norm(d.qvel[:2])))
            target_index = first_frame + tick + 1
            if target_index < len(reference_q):
                joint_mse.append(np.mean(np.square(d.qpos[7:] - reference_q[target_index])))
        if trajectories is not None:
            np.savez_compressed(trajectories / f"{index:02d}.npz", qpos=q_trace, qvel=v_trace, action=action_trace,
                                feet_weight=foot_trace, stance_weight=stance_trace, motor_group_delay=delays,
                                clip=name, offset_s=start, dt=env.DT, assist_newtons=0., seed=seed,
                                checkpoint_sha256=checkpoint_hash, source="native_closed_loop_policy")
        x = np.asarray(rows)
        up = (zenbot.UP_HEIGHT < x[:, 0]) & (x[:, 0] <= 2 * zenbot.RISE_HEIGHT - zenbot.UP_HEIGHT) & (x[:, 1] > np.cos(np.pi / 6))
        holds = np.flatnonzero(np.convolve(up.astype(int), np.ones(10), "valid") >= 10)
        final_stand = bool(up[-50:].all() and np.mean(x[-50:, 2] >= .50) >= .90)
        result.append(dict(clip=name, start_s=start, delay_physics_steps=delays.tolist(),
                           first_upright_s=float(holds[0] * env.DT) if len(holds) else None,
                           final_stand=final_stand, final_quiet=final_stand and bool((x[-50:, 3] < .15).all()), final_root_m=float(x[-1, 0]),
                           max_pelvis_m=float(x[:, 0].max()), upright_s=float(up.sum() * env.DT),
                           reference_joint_rms_deg=float(np.degrees(np.sqrt(np.mean(joint_mse)))) if joint_mse else None))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("data/zenbot23_getup/recovery"))
    parser.add_argument("--checkpoint", type=Path, help="Optional trained Zenbot actor for actual closed-loop evaluation")
    parser.add_argument("--starts", choices=("fallen", "standing"), default="fallen",
                        help="Full recoveries from the floor, or a separate standing-balance diagnostic")
    parser.add_argument("--seconds", type=float, default=12.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sensor-noise", type=float, default=0.,
                        help="Uniform proprioceptive noise as a fraction of the training amplitudes")
    parser.add_argument("--out", type=Path, help="Write the validation and optional policy metrics as JSON")
    parser.add_argument("--trajectories", type=Path, help="Save actual closed-loop actor motion/contact arrays per evaluation start")
    args = parser.parse_args()
    if args.trajectories is not None and args.checkpoint is None:
        parser.error("--trajectories requires --checkpoint")
    if args.seconds < 1. or not np.isclose(round(args.seconds / env.DT) * env.DT, args.seconds):
        parser.error("--seconds must be at least 1 and a multiple of 0.02")
    if not np.isfinite(args.sensor_noise) or args.sensor_noise < 0.:
        parser.error("--sensor-noise must be finite and nonnegative")
    report = validate(args.reference)
    if args.checkpoint:
        report["starts"] = args.starts
        report["seed"], report["sensor_noise"] = args.seed, args.sensor_noise
        report["policy"] = evaluate(args.reference, args.checkpoint, args.seconds, args.seed, args.starts,
                                    args.sensor_noise, args.trajectories)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
