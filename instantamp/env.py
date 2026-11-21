"""The Unitree G1 AMP-Flat environment as pure JAX functions of the robot's state, shared by the simulator (sim.py) and by imagination
in the world model (train.py).

The robot, actuators, observations, commands and most rewards follow the G1 AMP-Flat task of supermimic_AMP (mjlab), ported formula by
formula; terminations, get-up recovery, pushes and shoves follow XSoma's AMP task.

  robot      mjlab's G1 (KNEES_BENT keyframe, full collision, per-joint action scale 0.25 effort / stiffness) and the Unitree actuator
             model (torque-speed curve and friction)
  commands   mjlab's UniformVelocityCommand (heading envs, standing envs, resampling every 3 to 8 s)
  rewards    velocity tracking, regularizers, termination penalty, XSoma's rise penalty while down, a wrist pose penalty

A state is S = (p, R, v, w, q, qd): pelvis position and rotation (body to world), the pelvis origin's world linear velocity, the world
angular velocity, joint positions and velocities in DOF order.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .rot import mv3, tr3

# ------------------------------------------------------------------ robot
DOF = ("left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
       "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
       "waist_yaw", "waist_roll", "waist_pitch",
       "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
       "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch",
       "right_wrist_yaw")  # fmt: skip
J = len(DOF)
SIM_DT, DECIMATION = 0.005, 4
DT = SIM_DT * DECIMATION  # 50 Hz
EPISODE_S = 20.0
MAX_EPISODE = math.ceil(EPISODE_S / DT)  # 1000 steps; time-out once the counter reaches it


def _reflected(rotor, gear):
    """Reflected inertia of a two-stage planetary gearbox (mjlab.utils.actuator.reflected_inertia_from_two_stage_planetary)."""
    return rotor[0] * (gear[1] * gear[2]) ** 2 + rotor[1] * gear[2] ** 2 + rotor[2]


ARM_5020 = _reflected((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16))
ARM_7520_14 = _reflected((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22))
ARM_7520_22 = _reflected((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
ARM_5010_16 = 0.0021812
OMEGA, ZETA = 10 * 2.0 * 3.1415926535, 2.0  # 10 Hz natural frequency, damping ratio 2

# motor classes: armature, effort limit, the torque-speed curve (X1, X2 rad/s; Y1, Y2 N m) and friction (Fs, Fd)
_M = {
    "5020": (ARM_5020, 25.0, 30.86, 40.13, 24.8, 31.9, 0.6, 0.06),  # shoulders, elbows, wrist roll
    "7520_14": (ARM_7520_14, 88.0, 22.63, 35.52, 71.0, 83.3, 1.6, 0.16),  # hip yaw, waist yaw
    "7520_22": (ARM_7520_22, 139.0, 14.5, 22.7, 111.0, 131.0, 2.4, 0.24),  # hip pitch and roll, knee
    "5010": (ARM_5010_16, 10.0, 27.0, 41.5, 9.5, 17.0, 0.0, 0.0),  # wrist pitch and yaw
    "5020x2": (2 * ARM_5020, 50.0, 30.86, 40.13, 24.8, 31.9, 0.6, 0.06),  # waist pitch and roll, ankles: two 5020s
}


def _motor(name):
    if any(k in name for k in ("hip_pitch", "hip_roll", "knee")):
        return "7520_22"
    if "hip_yaw" in name or name == "waist_yaw":
        return "7520_14"
    if "ankle" in name or name in ("waist_roll", "waist_pitch"):
        return "5020x2"
    if "wrist_pitch" in name or "wrist_yaw" in name:
        return "5010"
    return "5020"


_P = np.asarray([_M[_motor(n)] for n in DOF], np.float64)
ARMATURE = _P[:, 0].astype(np.float32)
EFFORT = _P[:, 1].astype(np.float32)
X1, X2, Y1, Y2, FS, FD = (_P[:, i].astype(np.float32) for i in range(2, 8))
VA = 0.01  # rad/s at which the static friction is fully active
KP = (_P[:, 0] * OMEGA**2).astype(np.float32)
KD = (2.0 * ZETA * _P[:, 0] * OMEGA).astype(np.float32)
ACTION_SCALE = (0.25 * _P[:, 1] / (_P[:, 0] * OMEGA**2)).astype(np.float32)  # 0.25 effort / stiffness

_KNEES_BENT = {"hip_pitch": -0.312, "knee": 0.669, "ankle_pitch": -0.363, "elbow": 0.6, "left_shoulder_roll": 0.2, "left_shoulder_pitch": 0.2,
               "right_shoulder_roll": -0.2, "right_shoulder_pitch": 0.2}  # fmt: skip


def _default(name):
    for k, v in _KNEES_BENT.items():
        if (k == name) or (k in ("hip_pitch", "knee", "ankle_pitch", "elbow") and name.endswith(k)):
            return v
    return 0.0


DEFAULT_Q = np.asarray([_default(n) for n in DOF], np.float32)
DEFAULT_HEIGHT = 0.78  # pelvis height of the default pose
SOFT_LIMIT = 0.9  # soft joint limits: this share of each joint's range, around its middle


def joint_target(a, bias):
    """The PD law's joint target of raw actions a: q0 + scale a, minus the encoder bias (the joint settles off the commanded target by
    the bias)."""
    return DEFAULT_Q + ACTION_SCALE * a - bias


def actuator_torque(q, qd, target, *, xp=jnp):
    """Unitree actuator: PD torque, clipped by the torque-speed curve (Y1 when torque and velocity agree, Y2 when they oppose, both capped
    by the effort limit, falling linearly to 0 between X1 and X2), minus the friction Fs tanh(qd / Va) + Fd qd."""
    tau = KP * (target - q) - KD * qd
    lim = xp.minimum(xp.where(qd * tau > 0, Y1, Y2), EFFORT)
    slope = xp.clip(-lim / xp.maximum(X2 - X1, 1e-6) * (xp.abs(qd) - X1) + lim, 0.0, None)
    lim = xp.where(xp.abs(qd) < X1, lim, slope)
    tau = xp.clip(tau, -lim, lim)
    return tau - (FS * xp.tanh(qd / VA) + FD * qd)


def actuator_ctrl(q, qd, target, *, xp=jnp):
    """The control of MuJoCo's position actuator (gain kp, bias -kp q - kd qd) that yields `actuator_torque`: q + (tau + kd qd) / kp.
    The actuator's force range (+- effort) clamps the result, and the implicitfast integrator treats its -kd qd term implicitly."""
    return q + (actuator_torque(q, qd, target, xp=xp) + KD * qd) / KP


# ------------------------------------------------------------------ commands (mjlab UniformVelocityCommand)
CMD_RANGE = np.asarray([[-1.5, 3.0], [-1.0, 1.0], [-math.pi / 2, math.pi / 2]], np.float32)  # vx, vy, wz
HEADING_RANGE = (-math.pi / 2, math.pi / 2)  # world heading targets
REL_STANDING, REL_HEADING = 0.05, 0.25
HEADING_K = 0.5  # heading control stiffness
RESAMPLE_S = (3.0, 8.0)


class Cmd(NamedTuple):
    vel: jax.Array  # (N, 3) drawn vx, vy, wz (body frame)
    heading: jax.Array  # (N,) world heading target of a heading env
    is_heading: jax.Array  # (N,) bool: wz follows the heading target
    is_standing: jax.Array  # (N,) bool: the command is zero
    time_left: jax.Array  # (N,) seconds until the next draw


def sample_cmd(key, n, ranges=CMD_RANGE) -> Cmd:
    k = jax.random.split(key, 5)
    vel = jax.random.uniform(k[0], (n, 3), minval=ranges[:, 0], maxval=ranges[:, 1])
    heading = jax.random.uniform(k[1], (n,), minval=HEADING_RANGE[0], maxval=HEADING_RANGE[1])
    return Cmd(vel, heading, jax.random.uniform(k[2], (n,)) <= REL_HEADING, jax.random.uniform(k[3], (n,)) <= REL_STANDING,
               jax.random.uniform(k[4], (n,), minval=RESAMPLE_S[0], maxval=RESAMPLE_S[1]))  # fmt: skip


def where_cmd(mask, a: Cmd, b: Cmd) -> Cmd:
    return Cmd(*(jnp.where(mask[:, None] if x.ndim == 2 else mask, x, y) for x, y in zip(a, b)))


def heading(R):
    """The world heading of a body of rotation R (the yaw of its x axis)."""
    return jnp.arctan2(R[..., 1, 0], R[..., 0, 0])


def command(c: Cmd, R):
    """The command the policy observes (N, 3): the drawn one, with wz = clip(0.5 wrap(heading target - heading)) in heading envs, and zero
    in standing envs."""
    err = c.heading - heading(R)
    err = jnp.arctan2(jnp.sin(err), jnp.cos(err))
    wz_hd = jnp.clip(HEADING_K * err, CMD_RANGE[2, 0], CMD_RANGE[2, 1])
    wz = jnp.where(c.is_heading, wz_hd, c.vel[:, 2])
    out = jnp.concatenate((c.vel[:, :2], wz[:, None]), -1)
    return jnp.where(c.is_standing[:, None], 0.0, out)


def tick(c: Cmd, key, ranges=CMD_RANGE) -> Cmd:
    """One policy step of the command term: time_left -= dt, then a fresh draw where it ran out."""
    c = c._replace(time_left=c.time_left - DT)
    return where_cmd(c.time_left <= 0.0, sample_cmd(key, c.vel.shape[0], ranges), c)


# ------------------------------------------------------------------ observations (4 frames, time-major)
HIST = 4
FRAME_DIM = 3 + 3 + 3 + 3 * J  # base angular velocity, projected gravity, command, q - q0, qd, last action
OBS_DIM = HIST * FRAME_DIM  # 384
def observation_noise(joints):
    return np.concatenate([np.full(3, 0.2), np.full(3, 0.05), np.zeros(3), np.full(joints, 0.01),
                           np.full(joints, 0.5), np.zeros(joints)]).astype(np.float32)


_NOISE = observation_noise(J)


def obs_frame(S, a_last, cmd, key=None, default_q=DEFAULT_Q, noise=_NOISE):
    """One frame of the actor observation, unscaled: the pelvis IMU's angular velocity (pelvis frame), gravity in the pelvis frame, the
    command, q - q0, qd, the last raw action. key: add uniform noise (+-0.2, +-0.05, 0, +-0.01, +-0.5, 0)."""
    _, R, _, w, q, qd = S
    x = jnp.concatenate((mv3(tr3(R), w), -R[..., 2, :], cmd, q - default_q, qd, a_last), -1)
    if key is not None:
        x = x + noise * jax.random.uniform(key, x.shape, minval=-1.0, maxval=1.0)
    return x


def push_frame(hist, frame, fresh=None):
    """The history (N, HIST * FRAME_DIM) with `frame` appended; where `fresh`, every slot is the frame (backfilled after a reset)."""
    rolled = jnp.concatenate((hist[:, frame.shape[-1]:], frame), -1)
    if fresh is None:
        return rolled
    return jnp.where(fresh[:, None], jnp.tile(frame, (1, hist.shape[-1] // frame.shape[-1])), rolled)


# ------------------------------------------------------------------ terminations and get-up recovery
# A robot that is up (or has been up in this episode) is ended by a net contact force above 1 N on a termination body. A robot that
# starts down (a get-up reset) is not ended by contact, is given GETUP_S to rise, and is penalized for being low and tilted until it has
# been up.
UP_HEIGHT, UP_TILT = 0.6, 0.3  # "up": pelvis above 0.6 m and 1 - cos(tilt) below 0.3 (about 45 degrees)
GETUP_S = 5.0
GETUP_STEPS = int(GETUP_S / DT)
CONTACT_FORCE = 1.0  # N
# pelvis, hips, shoulders, torso and hands (the links that carry a collision geom); without the torso and the hands a policy learned to
# walk on all fours without ever ending its episode
TERMINATION_BODIES = ("pelvis", "left_hip_roll_link", "left_hip_yaw_link", "right_hip_roll_link", "right_hip_yaw_link",
                      "left_shoulder_yaw_link", "right_shoulder_yaw_link", "torso_link",
                      "left_wrist_pitch_link", "left_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_yaw_link")  # fmt: skip


def is_up(height, R, up_height=UP_HEIGHT):
    return (height > up_height) & (1 - R[..., 2, 2] < UP_TILT)


def soft_up(height, R, up_height=UP_HEIGHT):
    """Smooth version of is_up (the gradient of the get-up deadline in imagination)."""
    return jax.nn.sigmoid((height - up_height) / 0.02) * jax.nn.sigmoid((UP_TILT - (1 - R[..., 2, 2])) / 0.04)


# ------------------------------------------------------------------ pushes and shoves
PUSH_INTERVAL = (5.0, 10.0)  # s between pushes, per env; a push overwrites the root's world (vx, vy) with U[-1, 1] each
PUSH_VEL = 1.0
SHOVE_ROOT = (1.0, 1.0, 0.0, 1.0, 1.0, 1.5)  # a shove adds U[-1, 1] times these to the root's (vx, vy, vz, wx, wy, wz) ...
SHOVE_JOINT = 1.0  # ... and U[-1, 1] rad/s to every joint velocity


# ------------------------------------------------------------------ rewards (every weight times dt)
WEIGHTS = {
    "track_lin": 1.0,  # exp(-|v_pelvis (yaw frame) - (cmd_xy, 0)|^2 / 1^2), off while down
    "track_ang": 1.0,  # exp(-((w_pelvis,z - cmd_wz)^2 + |w_pelvis,body xy|^2) / 3.14^2), off while down
    "rise": -1.0,  # 2 (1 - clip(z / 0.75)) + 2 clip(1 - R_zz), only while down
    "body_ang_vel_xy": 0.5,  # exp(-|w_pelvis,body xy|^2 / 3.14^2), off while down
    "termination": -200.0,
    "joint_acc": -2.5e-7,
    "joint_limits": -10.0,
    "action_rate": -0.01,
    "foot_slip": -0.25,
    "self_collision": -0.1,
    "wrist_pose": -0.2,  # sum over the 6 wrist joints of (q - default)^2
}
WRIST_IDX = tuple(i for i, n in enumerate(DOF) if "wrist" in n)
TERMS = tuple(WEIGHTS)
TRACK_LIN_STD = 1.0  # std (m/s) of the linear tracking kernel
TRACK_ANG_STD = 3.14  # std (rad/s) of the yaw-rate part of the angular tracking kernel
RISE_HEIGHT = 0.75  # the pelvis height at which the rise penalty's "low" part vanishes


class Kin(NamedTuple):
    """The frames the rewards read (the simulator and imagination fill it from forward kinematics)."""

    torso_R: jax.Array  # (N, 3, 3)
    torso_v: jax.Array  # (N, 3) world linear velocity of the torso_link origin
    torso_w: jax.Array  # (N, 3) world angular velocity
    foot_v: jax.Array  # (N, 2, 3) world velocities of the left_foot / right_foot sites


def reward_terms(S0, S1, kin: Kin, a, a_prev, cmd, recovering, terminated, contact, collisions, q_soft,
                 wrist_idx=WRIST_IDX, default_q=DEFAULT_Q, rise_height=RISE_HEIGHT):
    """Unweighted terms of one step, read on the state S1 after it. cmd: the command the policy saw; recovering (N,): the robot has not
    been up in this episode; terminated (N,) in [0, 1]; contact (N, 2) foot-ground contact in [0, 1]; collisions (N,) the number of
    substeps with a self-collision above 10 N; q_soft (J, 2) the soft joint limits."""
    _, R1, v1, w1, q1, qd1 = S1
    rec = recovering.astype(jnp.float32)
    # tracking on the pelvis: read on the torso, a policy turned its waist by ~1 rad and walked sideways with the torso facing the command
    yaw_c, yaw_s = jnp.cos(heading(R1)), jnp.sin(heading(R1))
    local = jnp.stack((yaw_c * v1[:, 0] + yaw_s * v1[:, 1], -yaw_s * v1[:, 0] + yaw_c * v1[:, 1], v1[:, 2]), -1)  # in its yaw frame
    target = jnp.concatenate((cmd[:, :2], jnp.zeros_like(cmd[:, :1])), -1)
    lin = jnp.exp(-jnp.sum(jnp.square(local - target), -1) / TRACK_LIN_STD**2)
    body_w = mv3(tr3(R1), w1)
    ang = jnp.exp(-(jnp.square(w1[:, 2] - cmd[:, 2]) * (3.14**2 / TRACK_ANG_STD**2) + jnp.sum(jnp.square(body_w[:, :2]), -1)) / 3.14**2)
    root_rot = jnp.exp(-jnp.sum(jnp.square(body_w[:, :2]), -1) / 3.14**2)
    low = 1 - jnp.clip(S1[0][:, 2] / rise_height, 0.0, 1.0)
    tilt = jnp.clip(1 - R1[:, 2, 2], 0.0, 1.0)
    acc = (qd1 - S0[5]) / DT
    lim = jnp.sum(jnp.clip(q_soft[:, 0] - q1, 0.0, None) + jnp.clip(q1 - q_soft[:, 1], 0.0, None), -1)
    active = (jnp.linalg.norm(cmd[:, :2], axis=-1) + jnp.abs(cmd[:, 2]) > 0.1).astype(jnp.float32)
    slip = jnp.sum(jnp.sum(jnp.square(kin.foot_v[..., :2]), -1) * contact, -1) * active
    return {"track_lin": (1 - rec) * lin, "track_ang": (1 - rec) * ang, "rise": rec * (2.0 * low + 2.0 * tilt), "body_ang_vel_xy": (1 - rec) * root_rot,
            "termination": terminated, "joint_acc": jnp.sum(jnp.square(acc), -1), "joint_limits": lim,
            "action_rate": jnp.sum(jnp.square(a - a_prev), -1), "foot_slip": slip, "self_collision": collisions,
            "wrist_pose": jnp.sum(jnp.square(q1[:, wrist_idx] - default_q[list(wrist_idx)]), -1)}  # fmt: skip


def weighted(terms):
    return {k: DT * WEIGHTS[k] * terms[k] for k in TERMS}


# ------------------------------------------------------------------ domain randomization (at start-up, once per env)
FOOT_FRICTION = (0.3, 1.2)  # one value shared by the foot geoms of an env
N_FOOT_GEOMS = 14  # 7 sole capsules a foot
ENCODER_BIAS = 0.015  # +- rad per joint
TORSO_COM = (0.025, 0.025, 0.03)  # +- m added to torso_link's CoM
FOOT_CONTACT_H, FOOT_CONTACT_SOFT = 0.001, 0.001  # imagination: a foot whose lowest sole point is within 1 mm of the floor touches it
SELF_FORCE = 10.0  # N: a self-contact above this counts
