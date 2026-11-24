"""Retarget Zenbot gait references in the training robot's joint and sole frames."""

import mujoco as mj
import numpy as np
from scipy.linalg import solveh_banded
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, uniform_filter1d
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from instantamp import zenbot
from instantamp.sim import build_scene

TARGET_WIDTH = 0.18
MAX_FOOT_SHIFT = 0.025


def smoothstep(x):
    t = np.clip(x, 0., 1.)
    return t * t * (3. - 2. * t)


def settled_support(heights, pitch_gap, speed, fps):
    """Estimate foot-flat from source FK. A low toe or heel alone is insufficient."""
    low = heights.min(axis=1, keepdims=True)
    # The source clips have different floor offsets. Also reject airborne pairs.
    near = (1. - smoothstep((heights - low - .006) / .016)) * (
        1. - smoothstep((heights - np.percentile(low, 10) - .050) / .025))
    quiet = 1. - smoothstep((speed - .18) / .40)
    rolling = smoothstep((pitch_gap - .046) / .027)
    ready = near * quiet * (1. - rolling)
    # Require a short period of settled contact. The original rise in speed or heel
    # height releases the foot immediately, regardless of the preceding stance.
    lookback = max(1, round(.06 * fps) + 1)
    held = minimum_filter1d(ready, size=lookback, axis=0, origin=(lookback - 1) // 2, mode="nearest")
    ramp = max(1, round(.06 * fps))
    loaded = uniform_filter1d(held, size=ramp, axis=0, origin=(ramp - 1) // 2, mode="nearest")
    return np.clip(loaded, 0., ready)


def limit_step_change(values, limit, bounds=None, max_step=None):
    """Project local violations onto speed, acceleration and optional position bounds."""
    out = values.astype(np.float64, copy=True)
    if bounds is not None:
        np.clip(out, *bounds, out=out)
    for _ in range(1200):
        largest = 0.
        if max_step is not None:
            for phase in range(2):
                left, right = out[phase:-1:2], out[phase + 1::2]
                if len(left):
                    step = right - left
                    cap = max_step[phase::2]
                    excess = step - np.clip(step, -cap, cap)
                    largest = max(largest, np.max(np.abs(excess)))
                    left += excess / 2.
                    right -= excess / 2.
        # Disjoint triples can be projected together. Spread a correction over all three frames.
        for phase in range(3):
            left = out[phase:-2:3]
            if not len(left):
                continue
            mid = out[phase + 1:-1:3]
            right = out[phase + 2::3]
            accel = left - 2. * mid + right
            cap = limit[phase::3]
            excess = accel - np.clip(accel, -cap, cap)
            largest = max(largest, np.max(np.abs(excess)))
            left -= excess / 6.
            mid += excess / 3.
            right -= excess / 6.
        if bounds is not None:
            np.clip(out, *bounds, out=out)
        if largest < 1e-7:
            break
    return out


def planted_track(track, support, fps, contact_steps=None):
    """Fit a stationary track through support, with soft transitions to the recorded swing."""
    weight = ((.45 * fps) * np.minimum(support[1:], support[:-1])) ** 2
    edges = weight + (.04 * fps) ** 2
    bands = np.zeros((2, len(track)))
    bands[0] = 1.
    bands[0, :-1] += edges
    bands[0, 1:] += edges
    bands[1, :-1] = -edges
    steps = np.diff(track, axis=0) if contact_steps is None else contact_steps
    travel = weight[:, None] * steps
    rhs = np.zeros_like(track)
    rhs[:-1] += travel
    rhs[1:] -= travel
    return track + solveh_banded(bands, rhs, lower=True)


class StepWidth:
    def __init__(self):
        self.model = build_scene(robot_name=zenbot.MIXED_TASK)
        self.data = mj.MjData(self.model)
        self.feet = [self.model.body("robot_" + name).id for name in zenbot.FEET]
        self.boxes = [self.model.geom("robot_" + name + "_collision").id for name in zenbot.FEET]
        self.legs = []
        for side in ("left", "right"):
            joints = [self.model.joint(f"robot_{side}_{name}_joint").id
                      for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
            self.legs.append((self.model.jnt_qposadr[joints], self.model.jnt_dofadr[joints],
                              self.model.jnt_range[joints].T))

    def sole_poses(self, poses):
        centers = np.empty((len(poses), 2, 3))
        rotations = np.empty((len(poses), 2, 3, 3))
        for frame, pose in enumerate(poses):
            self.data.qpos[:] = pose
            mj.mj_kinematics(self.model, self.data)
            centers[frame] = self.data.geom_xpos[self.boxes]
            rotations[frame] = self.data.geom_xmat[self.boxes].reshape(2, 3, 3)
        return centers, rotations

    def adjust(self, pos, wxyz, joints, fps):
        """Return MJCF-ordered angles and the requested per-foot horizontal shift in metres."""
        moving = np.linalg.norm(np.gradient(pos[:, :2], axis=0) * fps, axis=1) > 0.25
        if not moving.any():
            return joints, 0.
        rotations = Rotation.from_quat(wxyz, scalar_first=True).as_matrix()
        lateral = np.column_stack((-rotations[:, 1, 0], rotations[:, 0, 0]))
        lateral /= np.maximum(np.linalg.norm(lateral, axis=1, keepdims=True), 1e-8)
        axis = lateral[moving].mean(axis=0)
        if np.linalg.norm(axis) < 0.8:  # No common transverse direction in multi-revolution turns.
            return joints, 0.
        axis /= np.linalg.norm(axis)
        poses = np.column_stack((pos, wxyz, joints))
        centers, orientations = self.sole_poses(poses)
        width = ((centers[:, 0, :2] - centers[:, 1, :2]) * lateral).sum(axis=1)
        shift = np.clip((TARGET_WIDTH - np.median(width[moving])) / 2, 0., MAX_FOOT_SHIFT)
        if not shift:
            return joints, 0.
        # A fixed world-space offset has zero velocity, including through each foot strike.
        targets = centers + np.array([1., -1.])[:, None] * np.r_[axis * shift, 0.]
        self.track_soles(poses, targets, orientations)
        return poses[:, 7:].astype(joints.dtype), float(shift)

    def track_soles(self, poses, targets, orientations):
        model, data = self.model, self.data
        jac = np.empty((6, model.nv))
        # The nearly straight knee has two IK branches. A posture cost keeps the
        # correction continuous as the recorded knee passes through full extension.
        posture = np.diag([0.10, 0.015, 0.035, 0.15, 0.06, 0.015])
        posture_cost = posture.T @ posture
        task_weights = np.array([3., 3., 3., 0.8, 0.8, 0.8])
        for frame, pose in enumerate(poses):
            data.qpos[:] = pose
            for _ in range(2):
                mj.mj_kinematics(model, data)
                mj.mj_comPos(model, data)
                for side, (qpos, dof, limits) in enumerate(self.legs):
                    box = self.boxes[side]
                    center = data.geom_xpos[box]
                    current = data.geom_xmat[box].reshape(3, 3)
                    rotation = Rotation.from_matrix(orientations[frame, side] @ current.T).as_rotvec()
                    error = task_weights * np.r_[targets[frame, side] - center, rotation]
                    mj.mj_jac(model, data, jac[:3], jac[3:], center, self.feet[side])
                    leg = task_weights[:, None] * jac[:, dof]
                    rest = pose[qpos] - data.qpos[qpos]
                    delta = np.linalg.solve(leg.T @ leg + posture_cost, leg.T @ error + posture_cost @ rest)
                    lo, hi = limits - data.qpos[qpos]
                    if np.any((delta < lo) | (delta > hi)):
                        delta = lsq_linear(np.vstack((leg, posture)), np.r_[error, posture @ rest],
                                           bounds=(lo, hi), method="bvls").x
                    data.qpos[qpos] = np.clip(data.qpos[qpos] + delta, *limits)
            for qpos, _, limits in self.legs:
                pose[qpos] = np.clip(np.round(data.qpos[qpos], 6), *limits)

    def correct_footfalls(self, pos, wxyz, joints, fps):
        if len(pos) < 3 or (np.all(pos == pos[0]) and np.all(joints == joints[0]) and np.all(wxyz == wxyz[0])):
            return pos, joints
        speed = np.linalg.norm(np.gradient(pos[:, :2], axis=0) * fps, axis=1)
        walk = 1. - smoothstep((gaussian_filter1d(speed, .12 * fps, mode="nearest") - 1.3) / .7)
        poses = np.column_stack((pos, wxyz, joints))
        centers, rotations = self.sole_poses(poses)
        sizes = self.model.geom_size[self.boxes]
        heights = centers[:, :, 2] - (abs(rotations[:, :, 2, :]) * sizes).sum(axis=2)
        foot_speed = np.linalg.norm(np.gradient(centers[:, :, :2], axis=0) * fps, axis=-1)
        pitch_gap = 2. * sizes[:, 0] * abs(rotations[:, :, 2, 0])  # heel-to-toe height difference
        support = walk[:, None] * settled_support(heights, pitch_gap, foot_speed, fps)
        adjusted = joints.astype(float, copy=True)
        for side, (qpos, _, limits) in enumerate(self.legs):
            # The ankle is a Y-pitch / X-roll pair. Express world up in the shank
            # frame to solve both angles for a level sole without constraining yaw.
            columns = qpos[-2:] - 7
            ankle = joints[:, columns]
            up = Rotation.from_euler("YX", ankle).apply(rotations[:, side, 2, :])
            flat = np.column_stack((np.arctan2(up[:, 0], up[:, 2]), -np.arcsin(np.clip(up[:, 1], -1., 1.))))
            correction = np.clip(flat, *limits[:, -2:]) - ankle
            adjusted[:, columns] += support[:, side, None] * correction

        # Levelling a heel/toe changes its contact point. Anchor the planar sole position
        # during slow, low support before differentiating the resulting joint trajectories.
        ranges = self.model.jnt_range[1:].T
        poses[:, 7:] = np.clip(adjusted, *ranges)
        if np.any(support):
            centers, rotations = self.sole_poses(poses)
            # Preserve the swing-foot angle while giving its lowest corner room
            # to pass the supporting foot. Taper the clearance at toe-off and landing.
            swing = walk[:, None] * smoothstep((foot_speed - .55) / 1.25) * smoothstep(
                (foot_speed - foot_speed[:, ::-1] - .20) / .80)
            clearance = heights - heights[:, ::-1]
            lift = swing * np.clip(.032 - clearance, 0., .035)
            centers[:, :, 2] += gaussian_filter1d(lift, .040 * fps, axis=0, mode="nearest")
            for side in range(2):
                # Roll around the heel or toe that is touching down. Anchoring the
                # sole center during this interval would interfere with that rotation.
                rotation = rotations[:, side]
                up = (rotation[:-1, 2, :] + rotation[1:, 2, :]) / 2.
                corner = -sizes[side] * np.tanh(up / .08)
                xy = centers[:, side, :2]
                steps = np.diff(xy, axis=0) + np.einsum('nij,nj->ni', np.diff(rotation[:, :2], axis=0), corner)
                rate = np.linalg.norm(steps, axis=1) * fps
                rate = np.r_[rate[0], (rate[1:] + rate[:-1]) / 2., rate[-1]]
                near = 1. - smoothstep((clearance[:, side] - .005) / .018)
                rolling = walk * near * (1. - smoothstep((rate - .16) / .52)) * (1. - swing[:, side])
                anchor = np.maximum(support[:, side], rolling)
                xy += anchor[:, None] * (planted_track(xy, anchor, fps, steps) - xy)
                plant = support[:, side]
                yaw = np.unwrap(np.arctan2(rotations[:, side, 1, 0], rotations[:, side, 0, 0]))
                pivot = smoothstep((np.abs(np.gradient(yaw) * fps) - .6) / 1.3)
                heading = planted_track(yaw[:, None], plant * (1. - pivot), fps)[:, 0]
                rotations[:, side] = Rotation.from_euler("z", (plant * (heading - yaw))[:, None]).as_matrix() @ rotations[:, side]
            self.track_soles(poses, centers, rotations)
        # A fast swing can cover many degrees in one tick. Limit changes in that
        # displacement, with separate acceleration budgets for walking and running.
        dt_scale = (50. / fps) ** 2
        # Permit a fast recorded swing, but do not introduce a new peak in its place.
        angular_limit = np.deg2rad(3.8 - 2.2 * walk[1:-1, None]) * dt_scale
        angular_limit = np.maximum(angular_limit, np.minimum(abs(np.diff(joints, n=2, axis=0)), np.deg2rad(3.8) * dt_scale))
        step_limit = np.deg2rad(12. - 5.2 * walk[:-1, None]) * (50. / fps)
        step_limit = np.maximum(step_limit, np.minimum(abs(np.diff(joints, axis=0)), np.deg2rad(12.) * (50. / fps)))
        adjusted = limit_step_change(poses[:, 7:], angular_limit, ranges, step_limit)
        poses[:, 7:] = adjusted
        centers, rotations = self.sole_poses(poses)
        bottom = (centers[:, :, 2] - (abs(rotations[:, :, 2, :]) * sizes).sum(axis=2)).min(axis=1)
        # Use the levelled soles to set walking height. Blend through exchanges of
        # the lowest foot; preserve the recorded flight trajectory at running speeds.
        support_z = pos[:, 2] + .002 - bottom
        z = pos[:, 2] + walk * (gaussian_filter1d(support_z, .042 * fps, mode="nearest") - pos[:, 2])
        z = limit_step_change(z, (.007 - .0054 * walk[1:-1]) * dt_scale, (support_z - .0015, np.inf))
        result = pos.copy()
        result[:, 2] = z.round(6)
        return result, adjusted.round(6).astype(joints.dtype)
