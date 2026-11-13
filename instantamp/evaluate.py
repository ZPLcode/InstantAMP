"""Evaluate InstantAMP checkpoints (ckpt/NNNN.npz of train.py) in the simulator: the averaged policy without noise, `n` envs, `seconds`
of simulation (20-s episodes; resets, pushes and shoves as in training). Per checkpoint: weighted reward terms per env-second, the share of episodes that
start up and end in a fall, the share of episodes that start down in which the robot rises, and the mean |v_xy - cmd_xy| while up.

  python -m instantamp.evaluate runs/instantamp [--n 1024 --seconds 20 --which avg --last 3]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import env, nets
from .sim import Sim, unflatten


def load(path, which="avg"):
    z = np.load(path)
    layers, i = [], 0
    while f"{which}/{i}/w" in z.files:
        layers.append({k.split("/")[-1]: jnp.asarray(z[k]) for k in z.files if k.startswith(f"{which}/{i}/")})
        i += 1
    norm = {"mean": jnp.asarray(z["obs_norm/mean"]), "var": jnp.asarray(z["obs_norm/var"])}
    ode = (float(z["ode/dmax"]) or None, float(z["ode/amax"]) or None) if "ode/dmax" in z.files else None
    return layers, norm, ode


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--which", default="avg")
    p.add_argument("--last", type=int, default=0, help="only the last k checkpoints (0: all)")
    p.add_argument("--obs_clip", type=float, default=10.0)
    a = p.parse_args(argv)
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    sim = Sim(a.n, cfg.get("nconmax", 64), cfg.get("njmax", 640), cfg["walk_dir"], cfg["floor_dir"], True, 123, cfg["reset_floor"], cfg["prob_shove"],
              bool(cfg["push"]))  # fmt: skip
    ckpts = sorted(glob.glob(os.path.join(a.run, "ckpt", "*.npz")))
    if a.last:
        ckpts = ckpts[-a.last :]

    def run(pi, norm, ode, key):
        def body(carry, k):
            s, acc = carry
            o = jnp.clip(nets.rn_apply(norm, s.obs), -a.obs_clip, a.obs_clip)
            out = nets.pi_apply(pi, o)
            act = nets.ode_step(out, s.a_ode, *ode) if ode is not None else out
            s1, tr = sim.step(s, act)
            S1 = unflatten(tr.qpos1, tr.qvel1)
            yaw = env.heading(S1[1])
            v = S1[2]
            vb = jnp.stack((jnp.cos(yaw) * v[:, 0] + jnp.sin(yaw) * v[:, 1], -jnp.sin(yaw) * v[:, 0] + jnp.cos(yaw) * v[:, 1]), -1)
            upright = s.was_up & env.is_up(S1[0][:, 2], S1[1])
            err = jnp.linalg.norm(vb - s.c_obs[:, :2], axis=-1)
            fin = (tr.terminated | tr.truncated).astype(jnp.float32)
            up_ep, down_ep = fin * ~s.down, fin * s.down
            rose = s.was_up | env.is_up(S1[0][:, 2], S1[1])
            acc = {"parts": acc["parts"] + tr.parts.sum(0), "steps": acc["steps"] + a.n,
                   "ep_up": acc["ep_up"] + up_ep.sum(), "fall_up": acc["fall_up"] + (up_ep * tr.terminated).sum(),
                   "ep_down": acc["ep_down"] + down_ep.sum(), "rose": acc["rose"] + (down_ep * rose).sum(),
                   "err": acc["err"] + (err * upright).sum(), "n_err": acc["n_err"] + upright.sum()}  # fmt: skip
            return (s1, acc), None

        s = sim.init(key)
        z = jnp.zeros(())
        acc = {"parts": jnp.zeros((len(env.TERMS),)), "steps": z, "ep_up": z, "fall_up": z, "ep_down": z, "rose": z, "err": z, "n_err": z}
        (_, acc), _ = jax.lax.scan(body, (s, acc), jax.random.split(key, int(a.seconds / env.DT)))
        return acc

    run_j = jax.jit(run, static_argnums=(2,))
    out = open(os.path.join(a.run, "eval.jsonl"), "w")
    for path in ckpts:
        pi, norm, ode = load(path, a.which)
        acc = jax.device_get(run_j(pi, norm, ode, jax.random.PRNGKey(0)))
        per_s = acc["parts"] / (acc["steps"] * env.DT)  # weighted terms per env-second
        rec = {"ckpt": os.path.basename(path), **json.load(open(path[:-4] + ".json")),
               **{f"rew_{k}": float(v) for k, v in zip(env.TERMS, per_s)},
               "fall_rate": float(acc["fall_up"] / max(acc["ep_up"], 1)), "getup_rate": float(acc["rose"] / max(acc["ep_down"], 1)),
               "down_episodes": float(acc["ep_down"]), "vel_err": float(acc["err"] / max(acc["n_err"], 1))}  # fmt: skip
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"{rec['ckpt']} it {rec['it']} | lin {rec['rew_track_lin']:.3f} ang {rec['rew_track_ang']:.3f} | |v-cmd| {rec['vel_err']:.3f} m/s | "
              f"falls {rec['fall_rate']:.3f} of upright starts | get-up {rec['getup_rate']:.3f} of {rec['down_episodes']:.0f} down starts", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
