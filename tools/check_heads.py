"""Imagination's reward terms read on real transitions (the true next state) against the simulator's terms of the same steps, and the
agreement of imagination's geometric termination and soft foot contact with the simulator's (needs a GPU and the clips).

  python tools/check_heads.py [--n 1024 --steps 100]
"""

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from instantamp import env  # noqa: E402
from instantamp.sim import Sim, unflatten  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", choices=("g1", "zenbot23_getup"), default="g1")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--steps", type=int, default=100)
    a = p.parse_args()
    N = a.n
    sim = Sim(num_envs=N, prob_shove=0.2, robot=a.robot)
    s = sim.init(jax.random.PRNGKey(1))
    step = jax.jit(sim.step)

    @jax.jit
    def heads(s, tr, act):
        S0, S1 = unflatten(tr.qpos, tr.qvel), unflatten(tr.qpos1, tr.qvel1)
        v1 = sim.view_state(S1)
        up1 = s.was_up | sim.is_up(v1.height, S1[1])
        geom = v1.clearance.min(-1) < 0  # imagination's termination of a robot that has been up
        late = (s.t + 1 >= env.GETUP_STEPS) & ~sim.is_up(v1.height, S1[1])
        term = jnp.where(s.was_up, geom, late)
        if sim.recovery:
            term = jnp.zeros_like(term)
        c = sim.soft_contact(v1.soles)
        terms = sim.reward_terms(S0, S1, v1.kin, act, s.a_last, s.c_obs, ~up1, term.astype(jnp.float32), c, jnp.zeros(N), sim.q_soft,
                                 torque=sim.robot.actuator_ctrl(S0[4], S0[5], tr.target[:, -sim.J:]) if sim.recovery else None)
        w = sim.weighted(terms)
        return jnp.stack([w[k] for k in sim.terms], -1), c, term

    key = jax.random.PRNGKey(2)
    im, re_, ci, cr, ti, tr_ = [], [], [], [], [], []
    for _ in range(a.steps):
        key, k = jax.random.split(key)
        act = 0.3 * jax.random.normal(k, (N, sim.J))
        s0 = s
        s, tr = step(s, act)
        parts, c, t_im = heads(s0, tr, act)
        ok = ~np.asarray(tr.diverged)
        ti.append(np.asarray(t_im)[ok])
        tr_.append(np.asarray(tr.terminated)[ok])
        im.append(np.asarray(parts)[ok])
        re_.append(np.asarray(tr.parts)[ok])
        ci.append(np.asarray(c)[ok])
        cr.append(np.asarray(tr.contact)[ok])
    im, re_, ci, cr, ti, tr_ = map(np.concatenate, (im, re_, ci, cr, ti, tr_))
    print(f"{'term':16s} {'sim mean':>10s} {'imag mean':>10s} {'max |diff|':>10s}")
    for j, k in enumerate(sim.terms):
        print(f"{k:16s} {re_[:, j].mean():10.5f} {im[:, j].mean():10.5f} {np.abs(re_[:, j] - im[:, j]).max():10.2e}")
    print(f"termination: sim {tr_.mean():.4f} imag {ti.mean():.4f} agreement {(ti == tr_).mean():.4f}")
    print(f"foot contact: sim {cr.mean():.3f} imag (soft) {ci.mean():.3f} agreement {((ci > 0.5) == (cr > 0.5)).mean():.3f}")
    print("(foot_slip uses the soft contact and self_collision is 0 in imagination: those two differ by design)")


if __name__ == "__main__":
    main()
