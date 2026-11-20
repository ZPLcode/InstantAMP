"""Receding-horizon joint control for a time-indexed, native 29-DoF G1 reference.

Optimize spline offsets to a reference PD servo in batched MuJoCo Warp predictions.
Only the selected offsets are returned. The caller supplies the measured state
at every replan and advances its own physical system.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from . import env
from .rot import quat_to_mat


@dataclass(frozen=True)
class PlanningConfig:
    population: int = 200
    horizon: int = 24
    stride: int = 3
    knot_steps: int = 6
    iterations: int = 4
    position_gain: float = 4.5
    velocity_gain: float = 1.7
    joint_weight: float = 24.
    fine_sampling: bool = True

    def __post_init__(self):
        if self.population < 4 or self.iterations < 1 or not 1 <= self.stride <= self.horizon or self.knot_steps < 1:
            raise ValueError("Invalid population, lookahead, knot interval or control stride")
        if any(not np.isfinite(x) or x <= 0 for x in (self.position_gain, self.velocity_gain, self.joint_weight)):
            raise ValueError("Feedback gains and joint cost must be positive and finite")

    @property
    def knots(self):
        return 1 + (self.horizon + self.knot_steps - 1) // self.knot_steps


def servo_target(q, qd, reference_q, reference_qd, offset, cfg):
    """Command for the existing torque-limited position motors, in radians."""
    return (q + cfg.position_gain * (reference_q + offset - q)
            + (env.KD / env.KP) * (cfg.velocity_gain * reference_qd - (cfg.velocity_gain - 1) * qd))


def build_planner(sim, reference, velocity, cfg):
    """Return a jitted solver and initial spline. Inputs use meters, radians and seconds.

    reference and velocity contain a separate synchronized trajectory per world,
    padded with the final standing pose and zero velocity through the last lookahead.
    """
    reference = jnp.asarray(reference, dtype=jnp.float32)
    velocity = jnp.asarray(velocity, dtype=jnp.float32)
    cases, frames, nq = reference.shape
    pop = cfg.population
    if nq != 7 + env.J or velocity.shape != (cases, frames, 6 + env.J) or sim.N != cases * pop:
        raise ValueError("Reference poses, velocities and planning worlds have incompatible shapes")
    ref_points = sim.key_positions(reference[..., :3], quat_to_mat(reference[..., 3:7]), reference[..., 7:])
    ids = jnp.arange(cases)
    sigma = jnp.array([.42 if "ankle" in j else .25 if "wrist" in j else .47 if "hip" in j or "knee" in j else .37 for j in env.DOF])
    clone = lambda x: jnp.repeat(x, pop, axis=0)
    fraction = lambda j: (j % cfg.knot_steps) / cfg.knot_steps

    def offsets_at(spline, j):
        k = j // cfg.knot_steps
        return (1. - fraction(j)) * spline[..., k, :] + fraction(j) * spline[..., k + 1, :]

    def predict_cost(q0, v0, warm, tick, spline):
        d = jax.vmap(lambda q: sim.dx.replace(qpos=q))(clone(q0))
        d = d.replace(qvel=clone(v0), qacc_warmstart=clone(warm))

        def step(d, j):
            t = tick + j + 1
            offset = offsets_at(spline, j)
            target = servo_target(d.qpos[:, 7:], d.qvel[:, 6:], clone(reference[:, t, 7:]),
                                  clone(velocity[:, t, 6:]), offset.reshape(sim.N, env.J), cfg)

            def substep(_, d):
                ctrl = env.actuator_ctrl(d.qpos[:, 7:], d.qvel[:, 6:], target)
                return jax.vmap(mjx.step, in_axes=(None, 0))(sim.mx, d.replace(ctrl=ctrl))

            d = jax.lax.fori_loop(0, env.DECIMATION, substep, d)
            q = d.qpos.reshape(cases, pop, nq)
            v = d.qvel.reshape(cases, pop, -1)
            e = q - reference[:, None, t]
            dv = v - velocity[:, None, t]
            orientation = 1 - jnp.sum(q[..., 3:7] * reference[:, None, t, 3:7], axis=-1) ** 2
            points = sim.key_positions(q[..., :3], quat_to_mat(q[..., 3:7]), q[..., 7:])
            key_error = jnp.mean(jnp.sum((points - ref_points[:, None, t]) ** 2, axis=-1), axis=-1)
            cost = (37 * jnp.sum(e[..., :2] ** 2, -1) + 85 * e[..., 2] ** 2 + 18 * orientation
                    + 55 * key_error + cfg.joint_weight * jnp.mean(e[..., 7:] ** 2, -1)
                    + .19 * jnp.sum(dv[..., :3] ** 2, -1) + .04 * jnp.sum(dv[..., 3:6] ** 2, -1)
                    + .004 * jnp.mean(dv[..., 6:] ** 2, -1) + .15 * jnp.mean(offset ** 2, -1))
            return d, jnp.minimum(jnp.nan_to_num(cost, nan=1e4), 1e4)

        _, cost = jax.lax.scan(step, d, jnp.arange(cfg.horizon))
        return cost.mean(0) + .12 * cost[-1]

    def sample(center, key, scale):
        noise = jax.random.normal(key, (cases, pop, cfg.knots, env.J))
        noise = (noise + .9 * noise[:, :, :1]) / 1.6
        if cfg.fine_sampling:
            scale = scale * jnp.asarray([.18, .40, .75, 1.3])[jnp.arange(pop) % 4][None, :, None, None]
        candidates = jnp.clip(center[:, None] + noise * sigma * scale, -1.35, 1.35)
        return candidates.at[:, 0].set(center).at[:, 1].set(jnp.zeros_like(center))

    def choose(q, v, warm, tick, center, key):
        def improve(i, carry):
            center, key = carry
            key, draw = jax.random.split(key)
            samples = sample(center, draw, jnp.maximum(.4, 1 - .3 * i))
            costs = predict_cost(q, v, warm, tick, samples)
            top = jnp.argsort(costs, axis=-1)[:, :max(2, pop // 12)]
            elite = samples[ids[:, None], top]
            return .7 * elite.mean(1) + .3 * elite[:, 0], key

        center, key = jax.lax.fori_loop(0, cfg.iterations - 1, improve, (center, key))
        samples = sample(center, key, .56 if cfg.iterations > 1 else 1.)
        best = jnp.argmin(predict_cost(q, v, warm, tick, samples), axis=1)
        chosen = samples[ids, best]
        offsets = jax.vmap(lambda j: offsets_at(chosen, j))(jnp.arange(cfg.stride))
        shifted = jnp.minimum(jnp.arange(cfg.knots) + cfg.stride // cfg.knot_steps, cfg.knots - 1)
        center = ((1 - fraction(cfg.stride)) * chosen[:, shifted]
                  + fraction(cfg.stride) * chosen[:, jnp.minimum(shifted + 1, cfg.knots - 1)])
        return offsets, center

    return jax.jit(choose), jnp.zeros((cases, cfg.knots, env.J), dtype=jnp.float32)
