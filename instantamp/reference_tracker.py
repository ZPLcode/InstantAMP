"""Run a reference-conditioned, 29-joint SuperMimic G1 policy in InstantAMP's MuJoCo model."""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.special import expit
from scipy.spatial.transform import Rotation

from . import env
from .sim import build_scene

HISTORY = 10
# The source actor's append/flatten convention groups ten consecutive frames of
# each sensor before moving to the next sensor.
OBS_WIDTHS = (env.J, env.J, 3, 3, env.J)
LANDMARKS = ("left_ankle_roll", "right_ankle_roll", "left_wrist_yaw", "right_wrist_yaw")


def _rotation(wxyz):
    return Rotation.from_quat(wxyz[..., [1, 2, 3, 0]]).as_matrix()


@dataclass
class Motion:
    """MuJoCo qpos and qvel at 50 Hz. Free-joint angular velocity is local."""
    qpos: np.ndarray
    qvel: np.ndarray

    @classmethod
    def load(cls, path, start=0, end=None):
        with np.load(path, allow_pickle=False) as clip:
            if not np.isclose(float(clip["fps"]), 1 / env.DT):
                raise ValueError(f"{path} requires the native 50 Hz retarget")
            length = len(clip["root_pos"])
            end = length if end is None else end
            if not 0 <= start < end <= length:
                raise ValueError(f"Invalid frame range [{start}, {end}) for {length} frames")
            sl = slice(start, end)
            xyz = clip["root_pos"][sl].copy()
            xyz[:, :2] -= xyz[0, :2].copy()
            quat = clip["root_quat"][sl]
            joints = clip["dof_pos"][sl]
            if joints.shape != (end - start, env.J):
                raise ValueError(f"{path} requires the 29 G1 joints in native order")
            local_angular = Rotation.from_quat(quat).apply(clip["root_ang_vel"][sl], inverse=True)
            qpos = np.column_stack((xyz, quat[:, [3, 0, 1, 2]], joints))
            qvel = np.column_stack((clip["root_lin_vel"][sl], local_angular, clip["dof_vel"][sl]))
            if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                raise ValueError(f"Nonfinite motion frames in {path}")
        return cls(qpos, qvel)

    def frames(self, count):
        indices = np.minimum(np.arange(count), len(self.qpos) - 1)
        velocity = self.qvel[indices].copy()
        velocity[len(self.qvel):] = 0
        return self.qpos[indices], velocity


class ReferencePolicy:
    """The source policy's motion encoder and proprioceptive decoder, evaluated on CPU."""
    def __init__(self, archive):
        with np.load(archive, allow_pickle=False) as weights:
            self.encoder_scale = weights["encoders.motion.scale"]
            self.decoder_scale = weights["decoder.scale"]
            if self.encoder_scale.shape != (689,) or self.decoder_scale.shape != (930,):
                raise ValueError("Expected the G1 actor with a 10-frame torso-anchor motion goal")
            self.encoder = self._layers(weights, "encoders.motion", (689, 2048, 1024, 512, 256, 32))
            self.decoder = self._layers(weights, "decoder", (962, 2048, 1024, 512, 256, 29))

    @staticmethod
    def _layers(weights, name, sizes):
        layers = []
        for i, (inputs, outputs) in enumerate(zip(sizes, sizes[1:])):
            key = f"{name}.mlp.net.{2 * i}"
            weight, bias = weights[key + ".weight"], weights[key + ".bias"]
            if weight.shape != (outputs, inputs) or bias.shape != (outputs,):
                raise ValueError(f"Unexpected layer dimensions in {key}")
            layers.append((weight.T, bias))
        return layers

    @staticmethod
    def _apply(layers, x):
        for i, (weight, bias) in enumerate(layers):
            x = x @ weight + bias
            if i + 1 < len(layers):
                x *= expit(x)
        return x

    def __call__(self, history, goal):
        n = len(history)
        terms = np.split(history, np.cumsum(OBS_WIDTHS)[:-1], axis=-1)
        obs = np.concatenate([term.reshape(n, -1) for term in terms], axis=-1)
        latent = np.tanh(self._apply(self.encoder, goal.astype(np.float32) * self.encoder_scale))
        return self._apply(self.decoder, np.concatenate((latent, obs * self.decoder_scale), axis=-1))


def _reference_signals(model, motion, frames, body_ids):
    qpos, qvel = motion.frames(frames)
    orientation = _rotation(qpos[:, 3:7])
    heading = np.arctan2(orientation[:, 1, 0], orientation[:, 0, 0])
    c, s = np.cos(heading), np.sin(heading)
    vx, vy, vz = qvel[:, :3].T
    velocity_heading = np.column_stack((c * vx + s * vy, -s * vx + c * vy, vz))
    features = np.column_stack((qpos[:, 7:] - env.DEFAULT_Q, qvel[:, 6:], -orientation[:, 2], qvel[:, 3:6],
                                velocity_heading, qpos[:, 2]))
    # This data instance is exclusively for reference forward kinematics.
    ref_data = mujoco.MjData(model)
    anchor_pos, anchor_rot, key_pos = [], [], []
    for pose in qpos:
        ref_data.qpos[:] = pose
        mujoco.mj_kinematics(model, ref_data)
        anchor_pos.append(ref_data.xpos[body_ids[0]].copy())
        anchor_rot.append(ref_data.xmat[body_ids[0]].reshape(3, 3).copy())
        key_pos.append(ref_data.xpos[body_ids[1:]].copy())
    return qpos, features, np.stack(anchor_pos), np.stack(anchor_rot), np.stack(key_pos)


def _source_servos():
    """The source G1 maps hip pitch to a 14:1 motor and wrist pitch/yaw to the 4010."""
    kp, kd, limit = env.KP.copy(), env.KD.copy(), env.EFFORT.copy()
    for i, name in enumerate(env.DOF):
        if "hip_pitch" in name:
            inertia, limit[i] = env.ARM_7520_14, 88.
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            inertia, limit[i] = 0.068e-4 * 25**2, 5.
        else:
            continue
        kp[i], kd[i] = inertia * env.OMEGA**2, 2 * env.ZETA * inertia * env.OMEGA
    return kp, kd, 0.25 * limit / kp


_SOURCE_KP, _SOURCE_KD, _SOURCE_SCALE = _source_servos()


def motor_control(q, qd, action):
    """Map the source impedance command onto the native torque-speed-limited motors.

    The source servo is frictionless. Request compensation through the native motors
    before applying their speed and torque limits. Return both the ordinary G1 PD
    position target and the corresponding raw control for a MuJoCo substep.
    """
    source_target = env.DEFAULT_Q + _SOURCE_SCALE * action
    source_torque = _SOURCE_KP * (source_target - q) - _SOURCE_KD * qd
    friction = env.FS * np.tanh(qd / env.VA) + env.FD * qd
    requested = source_torque + friction
    native_target = q + (requested + env.KD * qd) / env.KP
    limit = np.minimum(np.where(qd * requested > 0, env.Y1, env.Y2), env.EFFORT)
    speed_scale = np.clip(1 - np.maximum(np.abs(qd) - env.X1, 0.) / (env.X2 - env.X1), 0., 1.)
    limit = limit * speed_scale
    torque = np.clip(requested, -limit, limit) - friction
    return native_target, q + (torque + env.KD * qd) / env.KP


def track(motions, policy, steps, *, support_friction=True, lift=0.02):
    """Run one independent unconstrained robot per clip. Targets hold the last pose after clip end.

    qpos, qvel and reference include the initial frame. native_target and ctrl hold
    the motor setpoints and actuator inputs for each of the four 5 ms substeps.
    """
    if not motions or steps < 1 or not np.isfinite(lift) or lift < 0:
        raise ValueError("Expected at least one motion, positive steps and a finite nonnegative lift")
    model = build_scene(g1_support_friction=support_friction)
    ids = [model.body("robot_torso_link").id] + [model.body(f"robot_{name}_link").id for name in LANDMARKS]
    frames = steps + HISTORY
    ref, features, target_pos, target_rot, ref_points = [
        np.stack(field) for field in zip(*[_reference_signals(model, motion, frames, ids) for motion in motions])]
    robots = [mujoco.MjData(model) for _ in motions]
    for robot, motion in zip(robots, motions):
        robot.qpos[:] = motion.qpos[0]
        robot.qpos[2] += lift
        robot.qvel[:] = motion.qvel[0]
        mujoco.mj_forward(model, robot)
    n = len(robots)
    actions = np.zeros((n, env.J), dtype=np.float32)
    history = np.zeros((n, HISTORY, sum(OBS_WIDTHS)), dtype=np.float32)
    poses, velocities, landmarks, command_log, control_log, action_log = [], [], [], [], [], []
    for t in range(steps + 1):
        qpos = np.stack([robot.qpos for robot in robots])
        qvel = np.stack([robot.qvel for robot in robots])
        for robot in robots:
            mujoco.mj_kinematics(model, robot)
        poses.append(qpos)
        velocities.append(qvel)
        landmarks.append(np.stack([robot.xpos[ids[1:]] for robot in robots]))
        if t == steps:
            break
        orientation = _rotation(qpos[:, 3:7])
        obs = np.column_stack((qpos[:, 7:] - env.DEFAULT_Q, qvel[:, 6:], qvel[:, 3:6], -orientation[:, 2], actions))
        if t == 0:
            history[:] = obs[:, None]
        else:
            history[:, :-1] = history[:, 1:]
            history[:, -1] = obs
        position = np.stack([d.xpos[ids[0]] for d in robots])
        world_to_anchor = np.stack([d.xmat[ids[0]].reshape(3, 3).T for d in robots])
        relative_position = (world_to_anchor @ (target_pos[:, t] - position)[..., None])[..., 0]
        rot6d = (world_to_anchor @ target_rot[:, t])[..., :2].reshape(n, 6)
        goal = np.concatenate((features[:, t:t + HISTORY].reshape(n, -1), relative_position, rot6d), axis=-1)
        actions = policy(history, goal)
        action_log.append(actions)
        for _ in range(env.DECIMATION):
            q = np.stack([robot.qpos[7:] for robot in robots])
            qd = np.stack([robot.qvel[6:] for robot in robots])
            target, ctrl = motor_control(q, qd, actions)
            command_log.append(target)
            control_log.append(ctrl)
            for robot, command in zip(robots, ctrl):
                robot.ctrl[:] = command
                mujoco.mj_step(model, robot)
    return dict(qpos=np.stack(poses, axis=1), qvel=np.stack(velocities, axis=1), landmarks=np.stack(landmarks, axis=1),
                reference=ref[:, :steps + 1], reference_landmarks=ref_points[:, :steps + 1],
                native_target=np.stack(command_log, axis=1).reshape(n, steps, env.DECIMATION, env.J),
                ctrl=np.stack(control_log, axis=1).reshape(n, steps, env.DECIMATION, env.J), raw_actor=np.stack(action_log, axis=1))
