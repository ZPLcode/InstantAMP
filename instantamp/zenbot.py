"""23-DoF Zenbot AMP controls in MJCF hinge order (motion_train, 43 kg variant)."""

import jax.numpy as jnp
import numpy as np

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
INITIAL_ASSIST = 200 * MASS / 33.341142
ASSIST_DECREMENT = INITIAL_ASSIST / 20
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


def reward_terms(state, next_state, action, last_action, torque, limits, dt):
    from .rot import mv3, tr3

    pos, rot, vel, ang, q, qd = next_state
    z = pos[..., 2]
    standing = z > UP_HEIGHT
    square = lambda x: jnp.square(x).sum(-1)
    return {"joint_acc_l2": square((qd - state[5]) / dt), "action_rate_l2": square(action - last_action),
            "joint_torques_l2": square(torque), "joint_pos_limits": (jnp.maximum(limits[:, 0] - q, 0) + jnp.maximum(q - limits[:, 1], 0)).sum(-1),
            "ang_vel_xy": standing * jnp.exp(-2 * square(mv3(tr3(rot), ang)[..., :2])),
            "lin_vel_xy": standing * jnp.exp(-5 * square(mv3(tr3(rot), vel)[..., :2])),
            "target_orientation": standing * jnp.exp(-5 * square(rot[..., 2, :2])),
            "target_base_height": standing * jnp.exp(-20 * jnp.abs(z - RISE_HEIGHT)),
            "target_joint_deviation_l2": standing * square(q - DEFAULT_Q)}
