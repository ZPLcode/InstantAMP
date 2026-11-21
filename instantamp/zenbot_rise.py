"""Infer a Zenbot motor command from nearby examples in its own sensor history."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


SENSOR_SCALES = np.repeat([.15, 1.4, 1., .055, .09, 1.4], [3, 6, 23, 23, 23, 18]).astype(np.float32)


def features(obs):
    frames = obs.reshape(*obs.shape[:-1], 5, 96)
    current = frames[..., -1, :]
    difference = (current - frames[..., -3, :]) * .5
    return jnp.concatenate((current * SENSOR_SCALES, difference * SENSOR_SCALES), axis=-1)


class ZenbotLibrary(NamedTuple):
    keys: object
    key_norms: object
    commands: object
    joints: object
    velocities: object
    gain: object

    def action_with_distance(self, obs):
        x = features(obs)
        # Shortlist by squared Euclidean distance, then compare the actual vectors
        # to retain precision at or near a demonstration pose.
        dot = jnp.matmul(x, self.keys.T, precision=jax.lax.Precision.HIGHEST)
        _, shortlist = jax.lax.top_k(2 * dot - self.key_norms, min(12, len(self.keys)))
        distances = jnp.linalg.norm(x[..., None, :] - self.keys[shortlist], axis=-1)
        _, order = jax.lax.top_k(-distances, 4)
        ids = jnp.take_along_axis(shortlist, order, axis=-1)
        distances = jnp.take_along_axis(distances, order, axis=-1)
        best = distances[..., :1]
        weights = jax.nn.softmax(-(distances - best) / jnp.maximum(.002, .09 * best), axis=-1)
        weights = jnp.where(best < .003, jnp.array([1., 0., 0., 0.]), weights)[..., None]
        joint_error = (weights * self.joints[ids]).sum(-2) - obs[..., -87:-64]
        speed_error = (weights * self.velocities[ids]).sum(-2) - obs[..., -64:-41]
        correction = self.gain * 4 * jnp.clip(joint_error + .035 * speed_error, -.20, .20)
        return (weights * self.commands[ids]).sum(-2) + correction, best[..., 0]

    def action(self, obs):
        return self.action_with_distance(obs)[0]

    @classmethod
    def from_archive(cls, archive):
        def array(name):
            a = np.asarray(archive['zenbot_library/' + name], dtype=np.float32)
            if not np.isfinite(a).all():
                raise ValueError(f'Invalid Zenbot library {name}')
            return a
        keys, commands, joints, velocities = [array(name) for name in ('keys','commands','joints','velocities')]
        gain = array('gain')
        if (keys.ndim != 2 or keys.shape[1] != 192 or len(keys) < 4 or
                any(a.shape != (len(keys), 23) for a in (commands,joints,velocities)) or
                gain.shape or not 0 <= gain <= 2):
            raise ValueError('Invalid Zenbot motor library dimensions or feedback gain')
        return cls(jnp.asarray(keys), jnp.asarray((keys*keys).sum(-1)), jnp.asarray(commands),
                   jnp.asarray(joints), jnp.asarray(velocities), jnp.asarray(gain))


def residual_weight(distance, gate):
    if gate is None:
        return jnp.ones_like(distance)
    t = jnp.clip((distance - gate[0]) / gate[1], 0., 1.)
    return t*t*(3.-2.*t)


class ZenbotRisePolicy(NamedTuple):
    library: ZenbotLibrary
    layers: list
    scale: object
    gate: object = None
    stance: object = None
