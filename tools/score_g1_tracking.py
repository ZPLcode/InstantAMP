"""Score physical G1 get-ups against the native 50 Hz reference on its original clock.

For one trial call score_tracking(model, reference_qpos, simulated_qpos). Both paths
include the initial pose and use the same world frame and MuJoCo qpos order. A
reference-only run audits collision compatibility and stored velocity conventions.

    python tools/score_g1_tracking.py --out runs/g1_tracking/reference_audit.json
    python tools/score_g1_tracking.py --start-30hz 458 --rollout trial.npz

A rollout NPZ contains qpos (T, 36), optionally ended (T-1). To score an entire
segmented native clip, pass --reference path/to/clip.npz --whole-clip.
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instantamp import amp, env
from instantamp.sim import build_scene
from tools.evaluate_g1_style import STARTS_30HZ, TAKE


def load_native_reference(path):
    """Read world positions and MuJoCo local-axis free-joint angular velocities."""
    with np.load(path, allow_pickle=False) as data:
        if not np.isclose(float(data["fps"]) * env.DT, 1) or data["dof_pos"].shape[1] != env.J:
            raise ValueError(f"Expected 50 Hz G1 motion with {env.J} joint columns in {path}")
        quat = data["root_quat"][:, [3, 0, 1, 2]].astype(np.float64)
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        vector = data["root_ang_vel"]
        # Rotate world angular velocity by the conjugate unit quaternion.
        axis = -quat[:, 1:]
        cross = 2 * np.cross(axis, vector)
        local_omega = vector + quat[:, :1] * cross + np.cross(axis, cross)
        pose = np.column_stack((data["root_pos"], quat, data["dof_pos"]))
        velocity = np.column_stack((data["root_lin_vel"], local_omega, data["dof_vel"]))
    return pose, velocity


def _poses(model, trajectory):
    pose = np.asarray(trajectory, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != model.nq or not len(pose) or not np.isfinite(pose).all():
        raise ValueError(f"Expected a nonempty finite (frames, {model.nq}) qpos array")
    norm = np.linalg.norm(pose[:, 3:7], axis=1)
    if np.any(np.abs(norm - 1) > 0.005):
        raise ValueError("qpos contains a non-unit pelvis quaternion")
    pose = pose.copy()
    pose[:, 3:7] /= norm[:, None]
    return pose


def _up(pose):
    rzz = 1 - 2 * np.sum(pose[:, 4:6] ** 2, axis=-1)
    return (pose[:, 2] > env.UP_HEIGHT) & ((1 - rzz) < env.UP_TILT)


def _first(mask):
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def _window(reference, dt, settle_s):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(settle_s) or settle_s < 0:
        raise ValueError("dt must be positive and settle_s nonnegative")
    step = _first(_up(reference))
    if step is None:
        raise ValueError("The supplied get-up reference never reaches the G1 upright threshold")
    return step, min(len(reference), step + round(settle_s / dt) + 1)


def _collider_dimensions(model):
    floor = model.geom("floor").id
    geoms = np.flatnonzero((model.geom_contype != 0) & (np.arange(model.ngeom) != floor))
    types = model.geom_type[geoms]
    is_capsule = types == mujoco.mjtGeom.mjGEOM_CAPSULE
    if not np.all(is_capsule | (types == mujoco.mjtGeom.mjGEOM_SPHERE)):
        raise ValueError("This ground-clearance calculation supports the G1 capsule and sphere colliders")
    return geoms, model.geom_size[geoms, 0], model.geom_size[geoms, 1] * is_capsule


def _landmarks_and_clearance(model, poses):
    data = mujoco.MjData(model)
    names = (*amp.KEY_BODIES, "left_knee_link", "right_knee_link")
    ids = [model.body("robot_" + name).id for name in names]
    geoms, radii, half_lengths = _collider_dimensions(model)
    parts = ["left_foot", "right_foot", "left_hand", "right_hand"]
    group = [[j for j,g in enumerate(geoms) if model.geom(int(g)).name.startswith("robot_" + part)] for part in parts]
    positions = np.empty((len(poses), len(ids), 3))
    gaps = np.empty((len(poses), len(parts)))
    for i, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        positions[i] = data.xpos[ids]
        height = data.geom_xpos[geoms, 2] - radii - half_lengths * np.abs(data.geom_xmat[geoms, 8])
        gaps[i] = [height[indices].min() for indices in group]
    return positions, gaps


def _yaw_local(points, qpos):
    w, x, y, z = qpos[:, 3:7].T
    heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    offset = points - qpos[:, None, :3]
    c, s = np.cos(heading)[:, None], np.sin(heading)[:, None]
    return np.stack((c * offset[..., 0] + s * offset[..., 1], -s * offset[..., 0] + c * offset[..., 1],
                     offset[..., 2]), axis=-1)


def _pose_metrics(ref, sim, ref_points, sim_points):
    relative_ref, relative_sim = _yaw_local(ref_points, ref), _yaw_local(sim_points, sim)
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    length_rmse = lambda x: rms(np.linalg.norm(x, axis=-1))
    sign = np.where(np.sum(sim[:, 3:7] * ref[:, 3:7], axis=-1) < 0, -1., 1.)
    same_sign = sign[:, None] * ref[:, 3:7]
    attitude = np.rad2deg(4 * np.arctan2(np.linalg.norm(sim[:, 3:7] - same_sign, axis=1),
                                         np.linalg.norm(sim[:, 3:7] + same_sign, axis=1)))
    result = {
        "pelvis_height_rmse_cm": 100 * rms(sim[:, 2] - ref[:, 2]),
        "pelvis_xy_rmse_cm": 100 * length_rmse(sim[:, :2] - ref[:, :2]),
        "pelvis_attitude_rmse_deg": rms(attitude),
        "joint_rmse_rad": rms(sim[:, 7:] - ref[:, 7:]),
    }
    for name, sl in (("ankle", slice(0, 2)), ("wrist", slice(2, 4)), ("knee", slice(4, 6))):
        result[name + "_world_rmse_cm"] = 100 * length_rmse(sim_points[:, sl] - ref_points[:, sl])
        result[name + "_heading_rmse_cm"] = 100 * length_rmse(relative_sim[:, sl] - relative_ref[:, sl])
    return result


def score_tracking(model, reference_qpos, simulated_qpos, dt=env.DT, ended=None, settle_s=0.5):
    """Score one unassisted trial without phase warping or heading/translation realignment.

    Errors include the initial pose. If ending early, the last valid pose is held
    for the rest of the fixed score window and completion is marked false.
    """
    ref = _poses(model, reference_qpos)
    sim = _poses(model, simulated_qpos)
    rise, stop = _window(ref, dt, settle_s)
    terminated = False
    if ended is not None:
        ended = np.asarray(ended, dtype=bool)
        if ended.shape != (len(sim) - 1,):
            raise ValueError("ended must have one boolean for each transition after the initial pose")
        boundary = _first(ended)
        if boundary is not None:
            sim = sim[:boundary + 1]
            terminated = True
    complete = len(sim) >= stop
    up = _up(sim)
    sim_rise = _first(up[:round(env.GETUP_S / dt) + 1])
    actual = sim[np.minimum(np.arange(stop), len(sim) - 1)]
    ref = ref[:stop]
    rp, ref_clear = _landmarks_and_clearance(model, ref)
    ap, sim_clear = _landmarks_and_clearance(model, actual)
    proximity = {}
    for j, part in enumerate(("left_foot", "right_foot", "left_hand", "right_hand")):
        ref_low = ref_clear[:, j] <= 0.02
        sim_low = sim_clear[:, j] <= 0.02
        either = np.count_nonzero(ref_low | sim_low)
        proximity[part] = {
            "reference_frames": int(ref_low.sum()),
            "simulated_frames": int(sim_low.sum()),
            "intersection_over_union": float(np.count_nonzero(ref_low & sim_low) / either) if either else None,
        }
    hold_steps = round(0.5 / dt)
    return {
        "reference_rise_s": rise * dt,
        "simulated_rise_s": None if sim_rise is None else sim_rise * dt,
        "rise_time_error_s": None if sim_rise is None else (sim_rise - rise) * dt,
        "rose_by_deadline": sim_rise is not None,
        "completed_reference_window": complete,
        "reference_window_s": (stop - 1) * dt,
        "final_half_second_upright": bool(not terminated and len(up) >= hold_steps and up[-max(1, hold_steps):].all()),
        "upright_reference_window_end": bool(complete and _up(actual[-1:])[0]),
        "rise_only": _pose_metrics(ref[:rise + 1], actual[:rise + 1], rp[:rise + 1], ap[:rise + 1]),
        "rise_and_settle": _pose_metrics(ref, actual, rp, ap),
        "geometric_floor_proximity_2cm": proximity,
    }


def reference_diagnostics(model, qpos, qvel=None, dt=env.DT, settle_s=0.5):
    """Audit actual collision geometry against a flat z=0 floor and joint limits."""
    poses = _poses(model, qpos)
    _, stop = _window(poses, dt, settle_s)
    poses = poses[:stop]
    floor = model.geom("floor").id
    geoms, radii, half_lengths = _collider_dimensions(model)
    clearance = np.empty((len(poses), len(geoms)))
    self_distance = np.zeros(len(poses))
    data = mujoco.MjData(model)
    for i, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_fwdPosition(model, data)
        clearance[i] = data.geom_xpos[geoms, 2] - radii - half_lengths * np.abs(data.geom_xmat[geoms, 8])
        self_distance[i] = min((c.dist for c in data.contact if floor not in c.geom and c.dist < 0), default=0.)
    frame, site = np.unravel_index(clearance.argmin(), clearance.shape)
    nearest = clearance.min(axis=1)
    lower, upper = model.jnt_range[1:].T
    excess = np.maximum(np.maximum(lower - poses[:, 7:], poses[:, 7:] - upper), 0.)
    result = {
        "frames": len(poses),
        "initial_floor_clearance_cm": 100 * nearest[0],
        "max_floor_penetration_cm": 100 * max(0., -nearest[frame]),
        "worst_floor_geom": model.geom(int(geoms[site])).name,
        "worst_floor_time_s": int(frame) * dt,
        "floor_penetration_over_1cm_fraction": float(np.mean(nearest < -.01)),
        "all_geoms_above_2cm_fraction": float(np.mean(nearest > .02)),
        "max_self_collision_overlap_cm": float(-100 * self_distance.min()),
        "max_joint_limit_excess_rad": float(excess.max()),
    }
    if qvel is not None:
        velocity = np.asarray(qvel, dtype=np.float64)
        if velocity.ndim != 2 or velocity.shape[1] != model.nv or len(velocity) < stop or not np.isfinite(velocity[:stop]).all():
            raise ValueError(f"Expected finite (frames, {model.nv}) qvel covering the reference")
        if len(poses) >= 3:
            step_velocity = np.empty((len(poses) - 1, model.nv))
            for i in range(len(poses) - 1):
                mujoco.mj_differentiatePos(model, step_velocity[i], dt, poses[i], poses[i + 1])
            error = (step_velocity[1:] + step_velocity[:-1]) / 2 - velocity[1:stop - 1]
            result["root_linear_velocity_rmse_m_s"] = float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1))))
            result["root_angular_velocity_rmse_rad_s"] = float(np.sqrt(np.mean(np.sum(error[:, 3:6] ** 2, axis=1))))
            result["joint_velocity_rmse_rad_s"] = float(np.sqrt(np.mean(error[:, 6:] ** 2)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("data/recovery/lafan1") / f"{TAKE}.npz")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--start-30hz", type=int, help="Source start frame in the original 30 Hz take")
    choice.add_argument("--whole-clip", action="store_true", help="Start at frame zero in a segmented get-up")
    parser.add_argument("--rollout", type=Path, help="One simulation NPZ, with qpos in the centered reference world")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    starts = [0] if args.whole_clip else ([args.start_30hz] if args.start_30hz is not None else STARTS_30HZ)
    if args.rollout and len(starts) != 1:
        parser.error("A single simulation requires --start-30hz or --whole-clip")
    model = build_scene()
    pose, vel = load_native_reference(args.reference)
    results = []
    for frame in starts:
        start = int(np.rint(frame / (30 * env.DT)))
        if frame < 0 or start >= len(pose):
            parser.error("Start lies outside the reference")
        stop = start + env.GETUP_STEPS + round(0.5 / env.DT) + 1
        ref = pose[start:stop].copy()
        ref[:, :2] -= ref[0, :2].copy()
        item = {"source_frame_30hz": int(frame), "reference": reference_diagnostics(model, ref, vel[start:stop])}
        if args.rollout:
            with np.load(args.rollout, allow_pickle=False) as archive:
                item["tracking"] = score_tracking(model, ref, archive["qpos"], ended=archive.get("ended"))
        results.append(item)
    report = json.dumps({"reference": str(args.reference), "dt_s": env.DT, "trials": results}, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
