"""23-DoF Zenbot AMP controls in MJCF hinge order (motion_train, 43 kg variant)."""

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .rot import mv3, quat_to_mat, tr3

# Each row gives the PD gains, action offset, and delayed-actuator group.
# Group IDs correspond to hips/waist yaw, knees, ankles,
# shoulders/wrists and elbows in the source articulation.
_LEG = {"hip_pitch": (180, 4, -0.174, 0), "hip_roll": (180, 4, 0, 0),
        "hip_yaw": (100, 2.6, 0, 0), "knee": (180, 4, 0.348, 1),
        "ankle_pitch": (70, 2.3, -0.174, 2), "ankle_roll": (60, 2, 0, 2)}
_ARM = {"shoulder_pitch": (25, 2.3, 0.348, 3), "shoulder_roll": (25, 2.17, 0.279, 3),
        "shoulder_yaw": (25, 0.94, 0, 3), "elbow": (25, 1.15, 0.785, 4), "wrist_roll": (25, 0.61, 0, 3)}


def _paired(rows):
    return {f"{side}_{joint}": (kp, kd, -q if joint == "shoulder_roll" and side == "right" else q, group)
            for side in ("left", "right") for joint, (kp, kd, q, group) in rows.items()}


_JOINTS = {**_paired(_LEG), "waist_yaw": (150, 4, 0, 0), **_paired(_ARM)}
DOF = tuple(_JOINTS)
KP, KD, DEFAULT_Q, _groups = np.array(list(_JOINTS.values()), dtype=np.float32).T
GROUP = _groups.astype(np.int32)
EFFORT = np.array([130, 130, 120, 45, 45], np.float32)[GROUP]
SPEED = np.array([31.4, 21.4, 18.3, 34.9, 30], np.float32)[GROUP]
ACTION_SCALE = 0.25
DEFAULT_HEIGHT = 0.69
TORSO = "torso_Link"
FEET = ("left_ankle_roll_Link", "right_ankle_roll_Link")
HANDS = ("left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand")
TERMINATION_BODIES = ("torso_Link", "head_Link", "left_hip_yaw_Link", "right_hip_yaw_Link",
                      "left_knee_Link", "right_knee_Link", "left_shoulder_yaw_Link", "right_shoulder_yaw_Link", *HANDS)
WRIST_IDX = tuple(i for i, name in enumerate(DOF) if "wrist" in name)
KNEES = ("left_knee_Link", "right_knee_Link")
KEY_BODIES = FEET + KNEES + HANDS
UP_HEIGHT, RISE_HEIGHT = 0.58, 0.68
MASS = 43.0678308
MIXED_TASK = "zenbot23_walk_run_recovery"
# Forward walking/running and the lateral steps present in the motion_train AMP bank.
GAIT_COMMANDS = np.asarray([[0., 3.8], [-.35, .35], [-.5, .5]], np.float32)
INITIAL_ASSIST = 200 * MASS / 33.341142
ASSIST_DECREMENT = INITIAL_ASSIST / 20
FULL_FORCE_HEIGHT, RESET_LIFT = 0.48, 0.02
SUPPORT_BODIES = KEY_BODIES
HIST = 5
# URDF cylinders imported as capsules by the reference get-up task (radius and half-length).
CAPSULES = {**dict.fromkeys(("left_hip_yaw_Link", "right_hip_yaw_Link"), (0.07, 0.055)),
            **dict.fromkeys(KNEES, (0.06, 0.07)), "torso_Link": (0.125, 0.005),
            **dict.fromkeys(("left_shoulder_yaw_Link", "right_shoulder_yaw_Link"), (0.045, 0.005)),
            **dict.fromkeys(HANDS, (0.04, 0.135))}


def actuator_ctrl(q, qd, target):
    """Torque motors with PD tracking, effort saturation and a motoring speed limit."""
    tau = jnp.clip(KP * (target - q) - KD * qd, -EFFORT, EFFORT)
    return jnp.where((jnp.abs(qd) >= SPEED) & (tau * qd > 0), 0.0, tau)

# legged_lab ZenbotGetUpEnvCfg, in reward units per second.
WEIGHTS = {"joint_acc_l2": -1e-7, "action_rate_l2": -0.005, "joint_torques_l2": -2e-6,
           "joint_pos_limits": -10., "ang_vel_xy": 2., "lin_vel_xy": 2., "target_orientation": 2.,
           "target_base_height": 5., "target_joint_deviation_l2": -0.1}


def standing_posture(height, up_cos):
    return (height > UP_HEIGHT) & (height <= 2 * RISE_HEIGHT - UP_HEIGHT) & (up_cos > np.cos(np.pi / 6))


class SupportedStand(NamedTuple):
    """LeggedLab's one-second upright streak and foot-load history, independently for each world."""

    upright_steps: jax.Array
    loaded: jax.Array

    @classmethod
    def empty(cls, num_envs, dt):
        window = max(1, round(1. / dt))
        return cls(jnp.zeros(num_envs, dtype=jnp.int32), jnp.zeros((num_envs, window), dtype=bool))

    def reset(self, done):
        return SupportedStand(jnp.where(done, 0, self.upright_steps), self.loaded & ~done[:, None])

    def advance(self, posture, loaded):
        window = self.loaded.shape[1]
        upright_steps = jnp.where(posture, jnp.minimum(self.upright_steps + 1, window), 0)
        history = jnp.concatenate((self.loaded[:, 1:], loaded[:, None]), axis=1)
        support_fraction = history.sum(axis=1) / int(np.ceil(.90 * window))
        progress = jnp.minimum(upright_steps / window, support_fraction)
        score = progress * (loaded | (progress >= 1.))
        return SupportedStand(upright_steps, history), score


def _standing_velocity_costs(vel, ang, qd):
    return 5. * (vel**2).sum(-1), 2. * (ang**2).sum(-1), jnp.mean(qd**2, axis=-1)


def quiet_stand(state, foot_force, weight):
    """Reward the default pose and low velocity only while the feet carry the upright robot."""
    pos, rot, vel, ang, q, qd = state
    standing = standing_posture(pos[..., 2], rot[..., 2, 2]) & (foot_force >= .5 * weight)
    linear, angular, joint_speed = _standing_velocity_costs(vel, ang, qd)
    error = 4. * jnp.mean((q - DEFAULT_Q)**2, axis=-1) + linear + angular + joint_speed + 2. * (1. - rot[..., 2, 2])
    return standing, jnp.where(standing, jnp.exp(-jnp.maximum(error, 0.)), 0.)


def default_stand_terms(state, hold):
    """Score the measured default joint pose and upright pelvis after the feet support the robot."""
    _, rot, vel, ang, q, qd = state
    tolerance = np.deg2rad(10.)
    joints = jnp.mean(jnp.square((q - DEFAULT_Q) / tolerance), axis=-1)
    tilt = (1. - jnp.clip(rot[..., 2, 2], -1., 1.)) / (1. - np.cos(tolerance))
    pose = 1. / (1. + joints + tilt)
    quiet = jnp.exp(-sum(_standing_velocity_costs(vel, ang, qd)))
    return (jnp.where(hold > 0., hold * pose * quiet, 0.),
            jnp.where(hold > 0., hold * (.25 + .75 * pose), 0.))


class RecoveryReference:
    """Phase-aligned poses for the native Zenbot clips used to initialize episodes."""

    def __init__(self, directory, key_positions, joints=len(DOF), *, coupled_pose=False):
        self.joints = joints
        self.coupled_pose = coupled_pose
        self.files = sorted(Path(directory).glob("*.npz"))
        poses, lengths = [], []
        for path in self.files:
            with np.load(path) as clip:
                pos, q = clip["root_pos"], clip["dof_pos"]
                if float(clip["fps"]) != 50 or not len(pos) or q.shape != (len(pos), self.joints):
                    raise ValueError(f"Expected a 50 Hz recovery reference in {path}")
                rot = quat_to_mat(clip["root_quat"][:, [3, 0, 1, 2]])
                points = clip["key_body_pos"] if "key_body_pos" in clip else key_positions(pos, rot, q)
                poses.append(self.pose(pos, rot, q, points))
                lengths.append(len(pos))
        if not lengths:
            raise ValueError(f"No recovery references in {directory}")
        self.poses = jnp.concatenate(poses)
        self.offsets = jnp.asarray(np.cumsum([0, *lengths[:-1]]), dtype=jnp.int32)
        self.lengths = jnp.asarray(lengths, dtype=jnp.int32)

    @staticmethod
    def pose(pos, rot, q, points):
        # Landmark coordinates in the pelvis frame and world vertical in that frame.
        local = mv3(tr3(rot)[..., None, :, :], points - pos[..., None, :])
        return jnp.concatenate((pos[..., 2:3], rot[..., 2, :], q, local.reshape(*q.shape[:-1], -1)), axis=-1)

    def errors(self, pose, clip, step):
        index = self.offsets[clip] + jnp.clip(step, 0, self.lengths[clip] - 1)
        delta = pose - self.poses[index]
        height = delta[..., 0] ** 2
        tilt = (delta[..., 1:4] ** 2).sum(-1)
        joints = (delta[..., 4:4 + self.joints] ** 2).mean(-1)
        landmarks = delta[..., 4 + self.joints:].reshape(*height.shape, len(KEY_BODIES), 3)
        points = (landmarks ** 2).sum(-1).mean(-1)
        return (height, tilt, joints, points), (step >= 0) & (step < self.lengths[clip])

    def score(self, pose, clip, step):
        errors, active = self.errors(pose, clip, step)
        # Normalize height, tilt, joints and landmarks in their respective units.
        scales = (0.15, 0.5, 0.5, 0.15)
        squared_errors = [error / scale**2 for error, scale in zip(errors, scales)]
        if self.coupled_pose:
            # A pelvis on the floor cannot match a standing pose through its joint angles alone.
            score = 1 / (1 + sum(squared_errors) / len(scales))
        else:
            score = sum(1 / (1 + error) for error in squared_errors) / len(scales)
        return jnp.where(active, score, 0.)


def reward_terms(state, next_state, action, last_action, torque, limits, dt, *,
                 up_height=UP_HEIGHT, target_height=RISE_HEIGHT, default_q=DEFAULT_Q, joint_acc=None, com_offset=None):
    from .rot import mv3, tr3

    pos, rot, vel, ang, q, qd = next_state
    if com_offset is not None:
        vel = vel + jnp.cross(ang, mv3(rot, com_offset))
    z = pos[..., 2]
    standing = z > up_height
    square = lambda x: jnp.square(x).sum(-1)
    return {"joint_acc_l2": square((qd - state[5]) / dt if joint_acc is None else joint_acc), "action_rate_l2": square(action - last_action),
            "joint_torques_l2": square(torque), "joint_pos_limits": (jnp.maximum(limits[:, 0] - q, 0) + jnp.maximum(q - limits[:, 1], 0)).sum(-1),
            "ang_vel_xy": standing * jnp.exp(-2 * square(mv3(tr3(rot), ang)[..., :2])),
            "lin_vel_xy": standing * jnp.exp(-5 * square(mv3(tr3(rot), vel)[..., :2])),
            "target_orientation": standing * jnp.exp(-5 * square(rot[..., 2, :2])),
            "target_base_height": standing * jnp.exp(-20 * jnp.abs(z - target_height)),
            "target_joint_deviation_l2": standing * square(q - default_q)}
