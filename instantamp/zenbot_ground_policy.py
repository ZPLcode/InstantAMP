"""Blend two Zenbot AMP action increments using the pelvis IMU."""
from typing import NamedTuple

import jax.numpy as jnp


class ZenbotGroundPolicy(NamedTuple):
    base: list
    rise: list
    max_share: object
    supine_only: bool = False
    supine_limits: tuple = (0., .6)


def ground_share(raw_obs, supine_only=False, supine_limits=(0., .6)):
    """Smoothly enable the recovery branch between 0.10 and 0.45 observed tilt."""
    if raw_obs.shape[-1] != 390:
        raise ValueError("Expected five 78-value mixed Zenbot sensor frames")
    history = raw_obs.reshape(*raw_obs.shape[:-1], 5, 78)
    gravity = history[..., -3:, 3:6]
    # Treat an empty history sample as upright. Norms can vary with IMU noise.
    norm_sq = jnp.sum(jnp.square(gravity), axis=-1)
    norm = jnp.sqrt(jnp.maximum(norm_sq, 1e-6))
    gravity_z = jnp.where(norm_sq > 1e-6, gravity[..., 2] / norm, -1.)
    weights = jnp.array([.2, .3, .5], dtype=gravity_z.dtype)
    tilt = 1. + jnp.sum(gravity_z * weights, axis=-1)
    t = jnp.clip((tilt - .10) / .35, 0., 1.)
    supine = -jnp.sum(gravity[..., 0] / norm * weights, axis=-1)
    lo, hi = supine_limits
    s = jnp.clip((supine - lo) / (hi - lo), 0., 1.)
    return t * t * (3. - 2. * t) * jnp.where(supine_only, s * s * (3. - 2. * s), 1.)


def combine(raw_base, raw_rise, raw_obs, share, supine_only=False, supine_limits=(0., .6)):
    """Combine neural ODE increments before applying the common integrator."""
    weight = (ground_share(raw_obs, supine_only, supine_limits) * share)[..., None]
    return jnp.where(weight >= 1., raw_rise, raw_base + weight * (raw_rise - raw_base))
