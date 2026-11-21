"""LeggedLab G1 get-up task, with controls and motion columns in MJCF joint order."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .env import DOF, WRIST_IDX
from .getup_amp import LEGGED_BODIES as KEY_BODIES
from . import env

TASK_ID = "LeggedLab-Isaac-AMP-G1-GET-UP-v0"
TORSO = "torso_link"
SUPPORT_BODIES = (*KEY_BODIES[:4], "left_knee_link", "right_knee_link")
# Non-support links with source collision shapes, used for clearance diagnostics.
# The named task itself terminates on timeout.
TERMINATION_BODIES = ("pelvis", "torso_link", "left_hip_pitch_link", "right_hip_pitch_link",
                      "left_hip_yaw_link", "right_hip_yaw_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link",
                      "left_shoulder_yaw_link", "right_shoulder_yaw_link", "left_wrist_roll_link", "right_wrist_roll_link")
UP_HEIGHT, RISE_HEIGHT = 0.65, 0.75
FULL_FORCE_HEIGHT, INITIAL_ASSIST, ASSIST_DECREMENT = 0.55, 200., 10.
RESET_LIFT = 0.1
ACTION_SCALE = 0.25
# Three times the source get-up penalty on the squared change between successive control targets.
ACTION_RATE_L2_WEIGHT = -0.015
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


def rise_anchor_weight(reference, clip, frame, pose_score):
    """Strength of the frozen actor's target on a real post-step source pose."""
    last = reference.lengths[clip] - 1
    index = reference.offsets[clip] + jnp.clip(frame, 0, last)
    rising = (frame >= 0) & (frame <= last) & (reference.poses[index, 0] < UP_HEIGHT)
    return jnp.where(rising, pose_score**2, 0.)


def rise_anchor_loss(mu, target, weights):
    error = jnp.mean((mu - target)**2, axis=-1)
    return jnp.sum(weights * error) / jnp.maximum(weights.sum(), 1.)


def load_reference(directory):
    """Load native interpolation knots, with compatibility for older 50 Hz banks."""
    directory = Path(directory)
    files = sorted(directory.glob("*.npz"))
    if (directory / "native").exists():
        native = sorted((directory / "native").glob("*.npz"))
        if [p.name for p in files] != [p.name for p in native]:
            raise ValueError("The native G1 recovery bank must cover the same clips as its 50 Hz bank")
        files = native
    if not files:
        raise ValueError(f"No G1 recovery clips in {directory}")
    clips, fps, lengths = [], [], []
    for path in files:
        with np.load(path) as d:
            rate = float(d["fps"])
            if rate <= 0 or (len(d["root_pos"]) - 1) / rate < 10 * env.DT or d["dof_pos"].shape != (len(d["root_pos"]), len(DOF)):
                raise ValueError(f"Expected a 29-joint G1 clip with at least 0.2 s of motion in {path}")
            clip = dict(pos=d["root_pos"], quat=d["root_quat"][:, [3, 0, 1, 2]], q=d["dof_pos"], qd=d["dof_vel"],
                        v=d["root_lin_vel"], w=d["root_ang_vel"])
            if "key_body_pos" in d:
                clip["points"] = d["key_body_pos"]
            if clips and clip.keys() != clips[0].keys():
                raise ValueError(f"Inconsistent G1 reference landmarks in {path}")
            clips.append(clip)
            fps.append(rate)
            lengths.append(len(clip["pos"]))
    return dict(knots={k: jnp.asarray(np.concatenate([c[k] for c in clips]), jnp.float32) for k in clips[0]},
                fps=jnp.array(fps), lengths=jnp.array(lengths), offsets=jnp.array(np.cumsum([0, *lengths[:-1]]), jnp.int32))


def sample_reference(bank, key, count, window=10, *, fetch_steps=None):
    """Source fetch: uniform clip and continuous time; successive queries are 20 ms apart."""
    kc, kt = jax.random.split(key)
    clip = jax.random.randint(kc, (count,), 0, len(bank["fps"]))
    rate, lengths = bank["fps"][clip], bank["lengths"][clip]
    fetch_steps = window if fetch_steps is None else fetch_steps
    latest = jnp.maximum(lengths - 1 - fetch_steps * env.DT * rate, 0.)
    start = jax.random.uniform(kt, (count,)) * latest
    source = jnp.minimum(start[:, None] + jnp.arange(window) * (env.DT * rate[:, None]), lengths[:, None] - 1)
    lo = jnp.floor(source).astype(jnp.int32)
    w = source - lo
    i = lo + bank["offsets"][clip, None]
    j = jnp.minimum(lo + 1, lengths[:, None] - 1) + bank["offsets"][clip, None]
    out = {}
    for name, arr in bank["knots"].items():
        left, right = arr[i], arr[j]
        weight = w[..., None, None] if name == "points" else w[..., None]
        if name == "quat":
            dot = (left * right).sum(-1, keepdims=True)
            right = jnp.where(dot < 0, -right, right)
            angle = jnp.arccos(jnp.abs(dot).clip(0., 1.))
            denom = jnp.maximum(jnp.sin(angle), 1e-8)
            spherical = (jnp.sin((1-weight)*angle)*left + jnp.sin(weight*angle)*right) / denom
            q = jnp.where(angle < 1e-3, left + weight*(right-left), spherical)
            out[name] = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        else:
            out[name] = left + weight * (right - left)
    return out


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
