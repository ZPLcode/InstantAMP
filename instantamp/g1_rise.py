"""A local rise policy fitted to proprioception/action pairs from physical get-up demonstrations."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


def features(obs, action_weight):
    """Yaw-independent sensor features. The input is the ordinary five-frame G1 actor history."""
    gyro, rot, q, qd, action, points = (
        field.reshape(*obs.shape[:-1], 5, width)
        for field, width in zip(jnp.split(obs, [15, 45, 190, 335, 480], axis=-1), (3, 6, 29, 29, 29, 18))
    )
    key = jnp.concatenate(
        (
            1.5 * rot[..., -1, :],
            q[..., -1, :],
            0.30 * q[..., 0, :],
            0.025 * qd[..., -1, :],
            0.035 * gyro[..., -1, :],
            action_weight * action[..., -1, :],
            0.2 * action_weight * action[..., 2, :],
            0.70 * points[..., -1, :],
        ),
        axis=-1,
    )
    pose = jnp.concatenate((rot[..., -1, :], q[..., -1, :], gyro[..., -1, :], qd[..., -1, :]), axis=-1)
    return key, pose


def _rotation(six):
    x, z = six[..., :3], six[..., 3:]
    return jnp.stack((x, jnp.cross(z, x), z), axis=-1)


class LocalRise(NamedTuple):
    keys: object
    poses: object
    actions: object
    gain: object
    settings: object

    handoff_keys: object = None
    handoff_settings: object = None
    residual: object = None
    aligned_handoff: object = None

    @classmethod
    def read(cls, archive, prefix=""):
        keys, poses, actions, gain, settings = (archive[prefix + name] for name in cls._fields[:5])
        n = len(keys)
        if (
            n < 3
            or keys.shape != (n, 172)
            or poses.shape != (n, 67)
            or actions.shape != (n, 29)
            or gain.shape != (n, 29, 64)
            or settings.shape != (3,)
            or not all(np.isfinite(x).all() for x in (keys, poses, actions, gain, settings))
            or np.any(settings < 0)
            or settings[1] <= 0
        ):
            raise ValueError("Invalid G1 local rise policy arrays or scale")
        handoff = ()
        if prefix + "handoff_keys" in archive or prefix + "handoff_settings" in archive:
            samples, thresholds = (archive[prefix + name] for name in ("handoff_keys", "handoff_settings"))
            if (
                samples.ndim != 2
                or samples.shape[0] < 1
                or samples.shape[1] != keys.shape[1]
                or not np.isfinite(samples).all()
                or thresholds.shape != (4,)
                or not np.isfinite(thresholds).all()
                or thresholds[0] <= 0
                or thresholds[1] <= 0
                or not -1 <= thresholds[2] <= 1
                or not 0 < thresholds[3] <= 1
            ):
                raise ValueError("Invalid G1 local rise hand-off neighborhood")
            handoff = (samples, thresholds)
        residual = None
        if prefix + "residual/scale" in archive:
            fields = ("w0", "b0", "w1", "b1", "scale")
            residual = tuple(jnp.asarray(archive[prefix + "residual/" + name]) for name in fields)
            w0, b0, w1, b1, scale = residual
            if (
                w0.ndim != 2
                or w0.shape[0] != 172
                or b0.shape != (w0.shape[-1],)
                or w1.shape != (len(b0), 87)
                or b1.shape != (87,)
                or scale.shape != ()
                or float(scale) <= 0
                or not all(np.isfinite(x).all() for x in residual)
            ):
                raise ValueError("Invalid AMP rise residual parameters")
        base = tuple(map(jnp.asarray, (keys, poses, actions, gain, settings)))
        handoff = tuple(map(jnp.asarray, handoff)) if handoff else (None, None)
        aligned = None
        if prefix + "aligned_handoff" in archive:
            aligned = np.asarray(archive[prefix + "aligned_handoff"])
            if (
                aligned.shape != (4,)
                or not np.isfinite(aligned).all()
                or not -1 <= aligned[0] <= 1
                or aligned[1] < 0
                or aligned[2] <= 0
                or not 0 < aligned[3] <= 1
            ):
                raise ValueError("Invalid aligned-ankle G1 balance hand-off")
            aligned = jnp.asarray(aligned)
        return cls(*base, *handoff, residual, aligned)

    def handoff_thresholds(self, obs, defaults):
        if self.handoff_keys is None:
            return defaults[0], defaults[1], defaults[2]
        key, _ = features(obs, self.settings[0])
        distance_sq = jnp.square(key[..., None, :] - self.handoff_keys).sum(axis=-1)
        radius, depth, upright, increment = self.handoff_settings
        inside = distance_sq.min(axis=-1) <= radius * radius
        return tuple(
            jnp.where(inside, local, standard) for local, standard in zip((depth, upright, increment), defaults)
        )

    def apply(self, obs, fallback):
        action_weight, radius, gain_scale = self.settings
        key, current = features(obs, action_weight)
        distance_sq = jnp.square(key[..., None, :] - self.keys).sum(axis=-1)
        neg_sq, indices = jax.lax.top_k(-distance_sq, 3)
        weight = 1.0 / jnp.square(jnp.maximum(-neg_sq, 1e-8))
        weight /= weight.sum(axis=-1, keepdims=True)
        nominal = (weight[..., :, None] * self.poses[indices]).sum(axis=-2)
        command = (weight[..., :, None] * self.actions[indices]).sum(axis=-2)
        rotation = jnp.swapaxes(_rotation(nominal[..., :6]), -1, -2) @ _rotation(current[..., :6])
        # Local rotation error and encoder/gyro residuals are expressed in the teacher's joint order.
        vector = 0.5 * jnp.stack(
            (
                rotation[..., 2, 1] - rotation[..., 1, 2],
                rotation[..., 0, 2] - rotation[..., 2, 0],
                rotation[..., 1, 0] - rotation[..., 0, 1],
            ),
            axis=-1,
        )
        error = jnp.concatenate((vector, current[..., 6:] - nominal[..., 6:]), axis=-1)
        feedback = jnp.einsum("...ij,...j->...i", self.gain[indices[..., 0]].astype(obs.dtype), error)
        command += gain_scale * jnp.clip(feedback, -0.75, 0.75)
        distance = jnp.sqrt(jnp.maximum(-neg_sq[..., 0], 0.0))
        confidence = jnp.clip(3 - 2 * distance / radius, 0.0, 1.0)
        # Preserve exact source-network output for states outside the training neighborhood.
        action = jnp.where(
            confidence[..., None] >= 1.0, command, fallback + confidence[..., None] * (command - fallback)
        )
        if self.residual is not None:
            action += rise_residual(self.residual, obs)
        return action


def rise_residual(parameters, obs):
    """Bounded action offset with an encoder-velocity and a previous-command-rate basis."""
    w0, b0, w1, b1, scale = parameters
    key, _ = features(obs, 0.65)
    h = jnp.tanh(jnp.clip(key, -5.0, 5.0) @ w0 + b0)
    direct, damping, slew = jnp.split(jnp.tanh(h @ w1 + b1), 3, axis=-1)
    velocity = jnp.clip(obs[..., 306:335] / 5.0, -1.0, 1.0)
    change = jnp.clip(obs[..., 451:480] - obs[..., 422:451], -1.0, 1.0)
    return scale * (direct - damping * velocity - slew * change)
