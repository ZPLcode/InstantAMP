"""Estimate whether quiet stance can arrest the robot's motion from its onboard sensors."""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from . import env


def capture_features(obs):
    gyro, rot, q, qd, _, points = (
        term.reshape(*obs.shape[:-1], 5, width)
        for term, width in zip(jnp.split(obs, [15, 45, 190, 335, 480], axis=-1), (3, 6, 29, 29, 29, 18))
    )
    x, z = rot[..., -1, :3], rot[..., -1, 3:]
    orientation = jnp.stack((x, jnp.cross(z, x), z), axis=-1)
    feet = points.reshape(*obs.shape[:-1], 5, 6, 3)[..., :2, :]
    current = feet[..., -1, :, :]
    ankles = jnp.einsum('...ij,...fj->...fi', orientation, current)
    depths = -ankles[..., 2]
    # Fit a line through the five previous ankle samples in the rotating pelvis frame.
    slope = jnp.einsum('t,...tfj->...fj', jnp.arange(-2, 3) / (10 * env.DT), feet)
    omega = gyro[..., -1, :]
    velocities = -jnp.einsum('...ij,...fj->...fi', orientation, slope + jnp.cross(omega[..., None, :], current))
    stance_foot = jnp.argmax(depths, axis=-1)
    root_velocity = jnp.take_along_axis(velocities, stance_foot[..., None, None], axis=-2)[..., 0, :]
    midpoint = ankles[..., :2].mean(axis=-2)
    # Inverted pendulum time constant at the get-up task's nominal standing height.
    capture_point = .275 * root_velocity[..., :2] - midpoint
    speeds = jnp.linalg.norm(velocities[..., :2], axis=-1)
    knees = q[..., -1, [3, 9]]
    return jnp.stack((
        depths.min(-1), z[..., 2], jnp.linalg.norm(omega, axis=-1), depths.max(-1) - depths.min(-1),
        jnp.abs(capture_point[..., 0]), jnp.abs(capture_point[..., 1]), speeds.min(-1), speeds.max(-1),
        jnp.abs(root_velocity[..., 2]), jnp.abs(knees[..., 0] - knees[..., 1]),
        jnp.linalg.norm(ankles[..., 0, :2] - ankles[..., 1, :2], axis=-1),
        jnp.abs(midpoint[..., 0]), jnp.abs(midpoint[..., 1]), jnp.abs(qd[..., -1, [3, 9]]).max(-1),
    ), axis=-1)


class CaptureGate(NamedTuple):
    mean: object
    scale: object
    weight: object
    bias: object
    settings: object
    calm_limits: object = None

    @classmethod
    def read(cls, archive):
        fields = [archive['capture_gate/' + key] for key in ('mean', 'scale', 'weight', 'bias', 'settings')]
        mean, scale, weight, bias, settings = fields
        if (any(v.shape != (14,) for v in (mean, scale, weight)) or bias.shape != () or settings.shape not in ((5,), (9,))
                or not all(np.isfinite(v).all() for v in fields) or (scale <= 0).any()):
            raise ValueError("Invalid G1 quiet stance capture model")
        threshold, frames, blend, release_cosine, release_gyro = settings[:5]
        if not (0 < threshold < 1 and frames >= 1 and frames == int(frames) and blend >= 1
                and -1 <= release_cosine <= 1 and release_gyro > 0):
            raise ValueError("Invalid G1 quiet stance switch parameters")
        if len(settings) == 9 and (np.any(settings[5:7] <= 0) or np.any(settings[7:9] <= settings[5:7])):
            raise ValueError("Invalid G1 quiet stance velocity limits")
        calm_limits = archive.get("capture_gate/calm_limits")
        if calm_limits is not None:
            if calm_limits.shape != (6,) or not np.isfinite(calm_limits).all() or (calm_limits <= 0).any():
                raise ValueError("Invalid calm double-support limits")
            calm_limits = jnp.asarray(calm_limits)
        return cls(*map(jnp.asarray, fields), calm_limits)

    def transition(self, obs, previous, share):
        threshold, frames, blend, release_cosine, release_gyro = self.settings[:5]
        features = capture_features(obs)
        logit = jnp.clip((features - self.mean) / self.scale, -5., 5.) @ self.weight + self.bias
        ready = (logit >= jnp.log(threshold / (1-threshold))) & (share > .5)
        stable = (features[..., 1] >= release_cosine) & (features[..., 2] <= release_gyro)
        if self.settings.shape[0] == 9:
            entry_xy, entry_z, release_xy, release_z = self.settings[5:]
            # Require both feet to have settled before takeover. A moving swing foot alone
            # must not cancel an established stance while the support foot can arrest motion.
            ready &= (features[..., 7] <= entry_xy) & (features[..., 8] <= entry_z)
            stable &= (features[..., 6] <= release_xy) & (features[..., 8] <= release_z)
        if self.calm_limits is not None:
            depth, tilt_cos, gyro, spread, xy_speed, knee_speed = self.calm_limits
            calm = ((features[..., 0] >= depth) & (features[..., 1] >= tilt_cos)
                    & (features[..., 2] <= gyro) & (features[..., 3] <= spread)
                    & (features[..., 7] <= xy_speed) & (features[..., 8] <= xy_speed)
                    & (features[..., 13] <= knee_speed))
            ready |= calm & (share > .5)
        count = jnp.where(ready, jnp.minimum(previous[..., 1] + 1., frames), 0.)
        active = stable & ((count >= frames) | (previous[..., 0] > 0))
        fraction = jnp.where(active, jnp.minimum(previous[..., 0] + 1. / blend, 1.), 0.)
        # Start from the last action the robot actually received and keep that target throughout the blend.
        anchor = jnp.where((previous[..., 0] > 0)[..., None], previous[..., 2:], obs[..., 451:480])
        return jnp.concatenate((fraction[..., None], count[..., None], anchor), axis=-1)
