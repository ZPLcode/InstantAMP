"""Sensor-based late-rise corrections and the training-only fixed stance objective."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from . import env
from .g1_rise import _rotation, features


def stance_inputs(obs):
    """Estimate pelvis velocity from foot kinematics and the lower foot's support."""
    ankles = obs[..., -90:].reshape(*obs.shape[:-1], 5, 6, 3)[..., :2, :]
    x, z = obs[..., 39:42], obs[..., 42:45]
    rotation = jnp.stack((x, jnp.cross(z, x), z), axis=-1)
    feet = jnp.einsum("...ij,...fj->...fi", rotation, ankles[..., -1, :, :])
    slope = jnp.einsum("t,...tfj->...fj", jnp.arange(-2, 3) / (10 * env.DT), ankles)
    gyro = obs[..., 12:15]
    rates = -(slope + jnp.cross(gyro[..., None, :], ankles[..., -1, :, :]))
    rates = jnp.einsum("...ij,...fj->...fi", rotation, rates)
    depth = -feet[..., 2]
    weights = jnp.exp((depth - depth.max(-1, keepdims=True)) / 0.02)
    pelvis_speed = (weights[..., None] * rates).sum(-2) / weights.sum(-1, keepdims=True)
    evidence = jnp.concatenate(
        (
            feet[..., :2].reshape(*obs.shape[:-1], 4),
            pelvis_speed[..., :2],
            rates[..., 0, :2] - rates[..., 1, :2],
        ),
        axis=-1,
    )
    gate = jnp.clip((depth.max(-1) - 0.30) / 0.24, 0.0, 1.0) * jnp.clip(
        (z[..., 2] - 0.30) / 0.52, 0.0, 1.0
    )
    return evidence, gate


def residual(params, obs):
    w0, b0, w1, b1, scale = params
    observable, _ = features(obs, 0.65)
    kin, _ = stance_inputs(obs)
    key = jnp.clip(jnp.concatenate((observable, kin), axis=-1), -5.0, 5.0)
    h = jnp.tanh(key @ w0 + b0)
    direct, damping, slew = jnp.split(jnp.tanh(h @ w1 + b1), 3, axis=-1)
    qd = jnp.clip(obs[..., 306:335] / 5.0, -1.0, 1.0)
    da = jnp.clip(obs[..., 451:480] - obs[..., 422:451], -1.0, 1.0)
    return scale * (direct - damping * qd - slew * da)


def read(archive):
    prefix = "in_place_residual/"
    if prefix + "scale" not in archive:
        return None
    w, b, v, c, scale = [archive[prefix + k] for k in ("w0", "b0", "w1", "b1", "scale")]
    if (
        w.ndim != 2
        or w.shape[0] != 180
        or b.shape != (w.shape[1],)
        or v.shape != (len(b), 87)
        or c.shape != (87,)
        or scale.shape != ()
        or float(scale) <= 0
        or not all(np.isfinite(a).all() for a in (w, b, v, c, scale))
    ):
        raise ValueError("Invalid G1 in-place feedback weights")
    return tuple(map(jnp.asarray, (w, b, v, c, scale)))


class Anchor(NamedTuple):
    root: object
    feet: object
    consecutive: object
    rose: object
    fell: object
    peak: object


def initial(n):
    z = jnp.zeros(n)
    return Anchor(
        jnp.zeros((n, 2)),
        jnp.zeros((n, 2, 2)),
        z.astype(jnp.int32),
        z.astype(bool),
        z.astype(bool),
        z,
    )


def score(anchor, new, feet, contact, ended):
    """Remember the first frame of the first continuous 0.2 s rise, including across rollouts."""
    pos, rot, vel = new[:3]
    xy = pos[:, :2]
    upright = (
        (pos[:, 2] > 0.65) & (pos[:, 2] <= 0.85) & (rot[:, 2, 2] > np.cos(np.pi / 6))
    )
    begin = upright & (anchor.consecutive == 0) & ~anchor.rose
    root = jnp.where(begin[:, None], xy, anchor.root)
    soles = jnp.where(begin[:, None, None], feet[..., :2], anchor.feet)
    count = jnp.where(upright, jnp.minimum(10, anchor.consecutive + 1), 0)
    rose = anchor.rose | (count == 10)
    fell = anchor.fell | (rose & ((pos[:, 2] < 0.50) | (rot[:, 2, 2] < 0.50)))
    drift = jnp.linalg.norm(xy - root, axis=-1)
    peak = jnp.where(
        rose | (count > 0), jnp.maximum(jnp.where(begin, 0.0, anchor.peak), drift), 0.0
    )
    foot_motion = jnp.linalg.norm(feet[..., :2] - soles, axis=-1).max(-1)
    speed = jnp.linalg.norm(vel[:, :2], axis=-1)
    # Encourage preparation while approaching upright, including before the first upright frame.
    near = jnp.clip((pos[:, 2] - 0.46) / 0.19, 0.0, 1.0) * jnp.clip(
        (rot[:, 2, 2] - 0.45) / 0.42, 0.0, 1.0
    )
    supported = contact.any(axis=-1)
    quiet = 1.0 / (1.0 + (speed / 0.23) ** 2)
    close = 1.0 / (1.0 + (peak / 0.17) ** 2 + (foot_motion / 0.16) ** 2)
    success = rose & upright & supported & ~fell
    reward = env.DT * (
        6.0 * near * quiet + success * (12.0 + 30.0 * close * quiet) - 7.0 * fell
    )
    next_anchor = Anchor(root, soles, count, rose, fell, peak)
    clear = initial(len(rose))
    next_anchor = Anchor(
        *(
            jnp.where(ended.reshape((len(rose),) + (1,) * (a.ndim - 1)), b, a)
            for a, b in zip(next_anchor, clear)
        )
    )
    return next_anchor, reward


def imitation_keys(obs):
    """Weight encoder histories, angular rates and foot motion in body heading coordinates."""
    key, _ = features(obs, 0.18)
    kinematics, _ = stance_inputs(obs)
    return jnp.concatenate((key, jnp.clip(kinematics, -5.0, 5.0) * 0.20), axis=-1)


class LandingLibrary(NamedTuple):
    keys: object
    norms: object
    actions: object
    radius: object
    poses: object = None
    feedback: object = None
    feedback_gain: object = None

    @classmethod
    def read(cls, archive):
        if "in_place_demo/keys" not in archive:
            return None
        k, a, r = [
            archive["in_place_demo/" + name] for name in ("keys", "actions", "radius")
        ]
        if (
            k.ndim != 2
            or len(k) < 5
            or k.shape[1] != 180
            or a.shape != (len(k), 29)
            or r.shape
            or float(r) <= 0
            or not all(np.isfinite(x).all() for x in (k, a, r))
        ):
            raise ValueError("Invalid in-place G1 demonstrations")
        optional = (None, None, None)
        if "in_place_demo/feedback" in archive:
            poses, feedback, gain = [
                archive["in_place_demo/" + n]
                for n in ("poses", "feedback", "feedback_gain")
            ]
            if (
                poses.shape != (len(k), 67)
                or feedback.shape != (len(k), 29, 64)
                or gain.shape
                or float(gain) < 0
                or not all(np.isfinite(x).all() for x in (poses, feedback, gain))
            ):
                raise ValueError("Invalid observable G1 landing feedback")
            optional = tuple(map(jnp.asarray, (poses, feedback, gain)))
        return cls(
            jnp.asarray(k),
            jnp.asarray((k * k).sum(-1)),
            jnp.asarray(a),
            jnp.asarray(r),
            *optional,
        )

    def target(self, obs):
        import jax

        key = imitation_keys(obs)
        dot = jnp.matmul(key, self.keys.T, precision=jax.lax.Precision.HIGHEST)
        _, candidates = jax.lax.top_k(2.0 * dot - self.norms, min(12, len(self.keys)))
        diff = key[..., None, :] - self.keys[candidates]
        dist = jnp.linalg.norm(diff, axis=-1)
        _, order = jax.lax.top_k(-dist, 4)
        ids = jnp.take_along_axis(candidates, order, axis=-1)
        d = jnp.take_along_axis(dist, order, axis=-1)
        best = d[..., :1]
        weight = jax.nn.softmax(-(d - best) / jnp.maximum(0.0005, 0.14 * best), -1)
        weight = jnp.where(best < 0.002, jnp.array([1.0, 0.0, 0.0, 0.0]), weight)
        command = (weight[..., None] * self.actions[ids]).sum(-2)
        if self.feedback is not None:
            nominal = (weight[..., None] * self.poses[ids]).sum(-2)
            _, current = features(obs, 0.18)
            dr = jnp.swapaxes(_rotation(nominal[..., :6]), -1, -2) @ _rotation(
                current[..., :6]
            )
            angular = 0.5 * jnp.stack(
                (
                    dr[..., 2, 1] - dr[..., 1, 2],
                    dr[..., 0, 2] - dr[..., 2, 0],
                    dr[..., 1, 0] - dr[..., 0, 1],
                ),
                -1,
            )
            error = jnp.concatenate(
                (angular, current[..., 6:] - nominal[..., 6:]), axis=-1
            )
            correction = jnp.einsum(
                "...ij,...j->...i", self.feedback[ids[..., 0]].astype(obs.dtype), error
            )
            command += jnp.where(
                best >= 0.002, self.feedback_gain * jnp.clip(correction, -2.0, 2.0), 0.0
            )
        # The first sensor sample has no velocity history. Handle exact starts and nearby states identically.
        certainty = jnp.clip(2.0 - best[..., 0] / self.radius, 0.0, 1.0)
        return command, certainty, best[..., 0]

    def residual_gate(self, obs):
        _, _, d = self.target(obs)
        weight = jnp.clip((d - 0.015) / 0.35, 0.0, 1.0)
        return weight * weight * (3.0 - 2.0 * weight)
