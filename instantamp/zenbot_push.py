"""Motion Train's per-episode shoulder/hip force-pulse scheduler.

Parameters and scheduling follow ``TimedRandomShoulderHipPush`` in Motion Train
ad923704. Advance once per policy step, then recompute the world-frame wrench
from each physics substep's link orientation. These are physical forces, not
root-velocity jumps. Resetting an episode clears its pulse and resamples its
clean/normal/strong membership.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

CLEAN, NORMAL, STRONG = 0, 1, 2
SHOULDER, HIP = 0, 1
_EPS = 1e-6
_BODIES = (
    "torso_Link",
    "left_hip_pitch_Link", "left_hip_roll_Link", "left_hip_yaw_Link",
    "right_hip_pitch_link", "right_hip_roll_Link", "right_hip_yaw_Link",
)


class PushState(NamedTuple):
    profile: jax.Array  # CLEAN/NORMAL/STRONG, sampled once per episode
    remaining: jax.Array  # seconds left in the current pulse
    wait: jax.Array  # seconds to next pulse start, infinity for clean episodes
    kind: jax.Array  # SHOULDER/HIP, -1 outside a pulse
    body: jax.Array  # MuJoCo body id, -1 outside a pulse
    point: jax.Array  # selected shoulder point
    force_w: jax.Array  # (N, 3), world direction is fixed during a pulse
    count: jax.Array


@dataclass(frozen=True)
class TimedPush:
    nbody: int
    bodies: tuple[int, ...]  # torso followed by six hip links
    shoulder_points: tuple[tuple[float, float, float], ...] = (
        (.008, .13787, .29817), (.008, -.13787, .29817),
    )

    @classmethod
    def from_model(cls, model):
        bodies = tuple(model.body("robot_" + name).id for name in _BODIES)
        if len(set(bodies)) != 7 or min(bodies) <= 0:
            raise ValueError("Shoulder/hip pushes require seven distinct robot bodies")
        # Both source tasks apply the force at the shoulder pitch origins in the torso frame.
        shoulders = [model.body(f"robot_{side}_shoulder_pitch_Link").id for side in ("left", "right")]
        if any(model.body_parentid[body] != bodies[0] for body in shoulders):
            raise ValueError("Shoulder pitch origins must be expressed in the torso frame")
        return cls(model.nbody, bodies, tuple(tuple(model.body_pos[body]) for body in shoulders))

    def init(self, key, num_envs):
        draws = jax.random.uniform(key, (num_envs, 2))
        profile = jnp.where(draws[:, 0] < .70, CLEAN,
                            jnp.where(draws[:, 0] >= .95, STRONG, NORMAL))
        zero = jnp.zeros(num_envs, jnp.float32)
        inactive = jnp.full((num_envs,), -1, jnp.int32)
        return PushState(
            profile, zero, jnp.where(profile == CLEAN, jnp.inf, 5. + 5. * draws[:, 1]),
            inactive, inactive, jnp.zeros(num_envs, jnp.int32),
            jnp.zeros((num_envs, 3), jnp.float32), zero,
        )

    def reset(self, state, fresh, key):
        chosen = self.init(key, fresh.shape[0])
        return jax.tree.map(
            lambda new, old: jnp.where(fresh.reshape((-1,) + (1,) * (old.ndim - 1)), new, old),
            chosen, state,
        )

    def advance(self, state, key, dt=.02):
        """Expire old pulses, count down, and start due pulses (start-to-start timer)."""
        remaining = jnp.maximum(state.remaining - dt, 0.)
        wait = state.wait - dt
        start = (state.profile != CLEAN) & (wait <= _EPS) & (remaining <= _EPS)
        # Independent variates for shoulder, hip, shoulder side, magnitude,
        # azimuth, elevation, duration and next start. Sampling all rows keeps
        # reset masks and JIT shapes independent of how many pulses are due.
        draws = jax.random.uniform(key, (state.profile.shape[0], 8))
        strong = state.profile == STRONG
        shoulder = strong | (draws[:, 0] < .5)
        slot = jnp.where(shoulder, 0, 1 + (draws[:, 1] * 6).astype(jnp.int32))
        point = (draws[:, 2] * 2).astype(jnp.int32)
        low = jnp.where(strong, 150., jnp.where(shoulder, 50., 30.))
        width = jnp.where(strong, 250., jnp.where(shoulder, 100., 70.))
        magnitude = low + width * draws[:, 3]
        azimuth = (2. * draws[:, 4] - 1.) * jnp.pi
        elevation = (2. * draws[:, 5] - 1.) * (jnp.pi / 18.)
        horizontal = jnp.cos(elevation)
        force = magnitude[:, None] * jnp.stack(
            (horizontal * jnp.cos(azimuth), horizontal * jnp.sin(azimuth), jnp.sin(elevation)), -1,
        )
        duration = jnp.where(strong, .08 + .04 * draws[:, 6], .10 + .10 * draws[:, 6])
        remaining = jnp.where(start, duration, remaining)
        active = remaining > _EPS
        return PushState(
            state.profile, remaining, jnp.where(start, 5. + 5. * draws[:, 7], wait),
            jnp.where(active, jnp.where(start, jnp.where(shoulder, SHOULDER, HIP), state.kind), -1),
            jnp.where(active, jnp.where(start, jnp.asarray(self.bodies)[slot], state.body), -1),
            jnp.where(start, point, state.point),
            jnp.where(active[:, None], jnp.where(start[:, None], force, state.force_w), 0.),
            state.count + start,
        )

    def wrench(self, state, xmat):
        """Return MuJoCo world-frame ``xfrc_applied`` (force, torque).

        Match the source's submitted wrench: hip pushes have no added torque;
        shoulder pushes add ``cross(R @ shoulder_point, force_w)``. The source
        supplies this torque explicitly to its wrench composer without a
        position argument, so no additional COM-offset correction is applied.
        """
        n = state.profile.shape[0]
        rotations = xmat.reshape(n, self.nbody, 3, 3)
        rows = jnp.arange(n)
        body = jnp.maximum(state.body, 0)
        active = (state.remaining > _EPS) & (state.profile != CLEAN)
        force = jnp.where(active[:, None], state.force_w, 0.)
        lever = jnp.where((state.kind == SHOULDER)[:, None],
                          jnp.asarray(self.shoulder_points, dtype=jnp.float32)[state.point], 0.)
        lever_w = jnp.einsum("nij,nj->ni", rotations[rows, body], lever)
        wrench = jnp.concatenate((force, jnp.cross(lever_w, force)), -1)
        return jnp.zeros((n, self.nbody, 6), jnp.float32).at[rows, body].set(wrench)
