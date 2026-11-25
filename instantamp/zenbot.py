"""Shared gait/recovery task logic and the 23-DoF articulation profile."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .rot import mv3, tr3
from .zenbot_models import ZENBOT23 as MODEL

# Public names also used by archived controllers and the generic simulator.
DOF = MODEL.dof
KP, KD = MODEL.kp, MODEL.kd
DEFAULT_Q, GROUP = MODEL.default_q, MODEL.group
EFFORT, SPEED = MODEL.effort, MODEL.speed
ACTION_SCALE = 0.25
DEFAULT_HEIGHT = 0.69
TORSO = "torso_Link"
FEET = ("left_ankle_roll_Link", "right_ankle_roll_Link")
HANDS = ("left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand")
TERMINATION_BODIES = ("torso_Link", "head_Link", "left_hip_yaw_Link", "right_hip_yaw_Link",
                      "left_knee_Link", "right_knee_Link", "left_shoulder_yaw_Link", "right_shoulder_yaw_Link", *HANDS)
WRIST_IDX = MODEL.wrist_indices
KNEES = ("left_knee_Link", "right_knee_Link")
KEY_BODIES = FEET + KNEES + HANDS
UP_HEIGHT, RISE_HEIGHT = 0.58, 0.68
MASS = MODEL.mass
MIXED_TASK = MODEL.mixed_task
# Forward walking/running and the lateral steps present in the motion_train AMP bank.
GAIT_COMMANDS = np.asarray([[0., 3.8], [-.35, .35], [-.5, .5]], np.float32)
DIRECTIONAL_COMMANDS = np.asarray([[0., 0., 0.], [1., 0., 0.], [2.5, 0., 0.], [-.5, 0., 0.],
                           [0., .4, 0.], [0., -.4, 0.], [0., 0., .6], [0., 0., -.6]], np.float32)
RESET_LIFT = 0.02
SUPPORT_BODIES = KEY_BODIES
HIST = 5
# URDF cylinders imported as capsules by the reference get-up task (radius and half-length).
CAPSULES = {**dict.fromkeys(("left_hip_yaw_Link", "right_hip_yaw_Link"), (0.07, 0.055)),
            **dict.fromkeys(KNEES, (0.06, 0.07)), "torso_Link": (0.125, 0.005),
            **dict.fromkeys(("left_shoulder_yaw_Link", "right_shoulder_yaw_Link"), (0.045, 0.005)),
            **dict.fromkeys(HANDS, (0.04, 0.135))}


def actuator_ctrl(q, qd, target, *, kp=KP, kd=KD, effort=EFFORT, speed=SPEED):
    """Torque motors with PD tracking, effort saturation and a motoring speed limit."""
    tau = jnp.clip(kp * (target - q) - kd * qd, -effort, effort)
    return jnp.where((jnp.abs(qd) >= speed) & (tau * qd > 0), 0.0, tau)

# legged_lab ZenbotGetUpEnvCfg, in reward units per second.
WEIGHTS = {"joint_acc_l2": -1e-7, "action_rate_l2": -0.005, "joint_torques_l2": -2e-6,
           "joint_pos_limits": -10., "ang_vel_xy": 2., "lin_vel_xy": 2., "target_orientation": 2.,
           "target_base_height": 5., "target_joint_deviation_l2": -0.1}


def standing_posture(height, up_cos):
    return (height > UP_HEIGHT) & (height <= 2 * RISE_HEIGHT - UP_HEIGHT) & (up_cos > np.cos(np.pi / 6))


def gait_weight(height, up_cos):
    """Current-pose suitability for locomotion, independent of history and foot contact."""
    def smooth_fraction(value, low, high):
        fraction = jnp.clip((value - low) / (high - low), 0., 1.)
        return fraction**2 * (3. - 2. * fraction)

    return (smooth_fraction(height, .44, UP_HEIGHT)
            * smooth_fraction(up_cos, .5, np.cos(np.pi / 6)))


def quiet_command_weight(command):
    """Fade the quiet-stance objective out as a linear or angular command grows."""
    return jnp.exp(-jnp.sum(jnp.square(command / .1), axis=-1))


class SupportedStand(NamedTuple):
    """One-second posture and foot-load success criterion; not a training reward gate."""

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


def floor_stand_score(state, foot_contact):
    """Quality of a supported standing pose. Lifting the pelvis below standing height earns zero."""
    pos, rot, vel, angular, _, _ = state
    height = jnp.exp(-20. * jnp.abs(pos[..., 2] - RISE_HEIGHT))
    still = jnp.exp(-4. * jnp.square(vel).sum(-1) - .5 * jnp.square(angular).sum(-1))
    upright = standing_posture(pos[..., 2], rot[..., 2, 2])
    return upright * height * still * jnp.mean(foot_contact, axis=-1)


def quiet_stand_score(state, foot_contact):
    """Continuous stance quality; the caller supplies pose and command suitability."""
    pos, _, vel, angular, _, _ = state
    height = jnp.exp(-20. * jnp.abs(pos[..., 2] - RISE_HEIGHT))
    still = jnp.exp(-4. * jnp.square(vel).sum(-1) - .5 * jnp.square(angular).sum(-1))
    return height * still * jnp.mean(foot_contact, axis=-1)


def lift_progress(height, onset):
    """A bounded rise from the hand-support height to the upright threshold."""
    fraction = jnp.clip((height - onset) / (UP_HEIGHT - onset), 0., 1.)
    return fraction**2 * (3. - 2. * fraction)


def reward_terms(state, next_state, action, last_action, torque, limits, dt, *,
                 up_height=UP_HEIGHT, target_height=RISE_HEIGHT, default_q=DEFAULT_Q, joint_acc=None, com_offset=None):
    pos, rot, vel, ang, q, qd = next_state
    if com_offset is not None:
        vel = vel + jnp.cross(ang, mv3(rot, com_offset))
    z = pos[..., 2]
    standing = z > up_height

    def square(x):
        return jnp.square(x).sum(-1)

    return {"joint_acc_l2": square((qd - state[5]) / dt if joint_acc is None else joint_acc), "action_rate_l2": square(action - last_action),
            "joint_torques_l2": square(torque), "joint_pos_limits": (jnp.maximum(limits[:, 0] - q, 0) + jnp.maximum(q - limits[:, 1], 0)).sum(-1),
            "ang_vel_xy": standing * jnp.exp(-2 * square(mv3(tr3(rot), ang)[..., :2])),
            "lin_vel_xy": standing * jnp.exp(-5 * square(mv3(tr3(rot), vel)[..., :2])),
            "target_orientation": standing * jnp.exp(-5 * square(rot[..., 2, :2])),
            "target_base_height": standing * jnp.exp(-20 * jnp.abs(z - target_height)),
            "target_joint_deviation_l2": standing * square(q - default_q)}
