"""LeggedLab G1 get-up task, with controls and motion columns in MJCF joint order."""
import jax.numpy as jnp
import numpy as np

from .env import DOF, TERMINATION_BODIES, WRIST_IDX
from .getup_amp import LEGGED_BODIES as KEY_BODIES

TASK_ID = "LeggedLab-Isaac-AMP-G1-GET-UP-v0"
TORSO = "torso_link"
SUPPORT_BODIES = (*KEY_BODIES[:4], "left_knee_link", "right_knee_link")
UP_HEIGHT, RISE_HEIGHT = 0.65, 0.75
FULL_FORCE_HEIGHT, INITIAL_ASSIST, ASSIST_DECREMENT = 0.55, 200., 10.
RESET_LIFT = 0.1
ACTION_SCALE = 0.25
HIST = 5

# Joint gains, effort and velocity limits of UNITREE_G1_29DOF_CFG.
_DRIVES = {
    "hip_pitch": (100, 2, 88, 32), "hip_yaw": (100, 2, 88, 32),
    "waist_yaw": (200, 5, 88, 32), "hip_roll": (100, 2, 139, 20),
    "knee": (150, 4, 139, 20),
    **dict.fromkeys(("ankle_pitch", "ankle_roll"), (40, 2, 25, 37)),
    **dict.fromkeys(("waist_pitch", "waist_roll"), (40, 5, 25, 37)),
    **dict.fromkeys(("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"), (40, 1, 25, 37)),
    **dict.fromkeys(("wrist_pitch", "wrist_yaw"), (40, 1, 5, 22)),
}
_NAMES = [name.removeprefix("left_").removeprefix("right_") for name in DOF]
KP, KD, EFFORT, SPEED = np.array([_DRIVES[name] for name in _NAMES], np.float32).T
ARMATURE = np.full(len(DOF), 0.01, np.float32)
_POSE = {"hip_pitch": -.1, "knee": .3, "ankle_pitch": -.2, "shoulder_pitch": .3, "elbow": .97,
         "left_shoulder_roll": .25, "right_shoulder_roll": -.25, "left_wrist_roll": .15, "right_wrist_roll": -.15}
DEFAULT_Q = np.array([_POSE.get(full, _POSE.get(name, 0.)) for full, name in zip(DOF, _NAMES)], np.float32)


def actuator_torque(q, qd, target):
    tau = jnp.clip(KP * (target - q) - KD * qd, -EFFORT, EFFORT)
    return jnp.where((jnp.abs(qd) >= SPEED) & (tau * qd > 0), 0., tau)


def actuator_ctrl(q, qd, target):
    # Retain MuJoCo's implicit damping while enforcing the drive limits.
    return q + (actuator_torque(q, qd, target) + KD * qd) / KP

# IsaacLab flattens the five samples of each observation term before concatenation.
ACTOR_TERMS = (3, 6, 29, 29, 29, 18)
CRITIC_TERMS = (3, 3, 6, 1, 29, 29, 29, 18)


def push_history(history, frame, widths, fresh=None):
    """Append one sample to each term's history, repeating the first frame on reset."""
    terms = jnp.split(frame, np.cumsum(widths)[:-1], axis=-1)
    initial = jnp.concatenate([jnp.tile(term, (1, HIST)) for term in terms], axis=-1)
    if history is None:
        return initial
    prior = jnp.split(history, np.cumsum(widths)[:-1] * HIST, axis=-1)
    rolled = jnp.concatenate([jnp.concatenate((old[:, width:], term), axis=-1)
                              for width, old, term in zip(widths, prior, terms)], axis=-1)
    return rolled if fresh is None else jnp.where(fresh[:, None], initial, rolled)
