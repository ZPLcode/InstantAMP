"""Measure tracking, falls and successful recoveries in the combined training task."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import env, zenbot_models
from .inference import load, policy_action
from .sim import Sim, unflatten
from .training_sim import simulation_options


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument(
        "--seconds",
        type=float,
        default=20.0,
        help="physical rollout duration in seconds",
    )
    p.add_argument("--which", choices=("avg", "live"), default="avg")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument(
        "--last",
        type=int,
        default=0,
        help="only the last k checkpoints (0: all)",
    )
    selection.add_argument(
        "--checkpoint",
        help="evaluate a single exported filename in this run's ckpt directory",
    )
    p.add_argument("--obs_clip", type=float, default=10.0)
    p.add_argument(
        "--results",
        type=Path,
        help="write evaluation JSONL here instead of the protocol default in the run",
    )
    a = p.parse_args(argv)
    if a.seconds is not None and (not np.isfinite(a.seconds) or a.seconds <= 0):
        p.error("--seconds must be finite and positive")
    if a.n < 1 or a.last < 0 or a.seconds < env.DT:
        p.error(
            "--n must be positive, --last nonnegative, and --seconds at least one control step"
        )
    run_dir = Path(a.run)
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpts = sorted((run_dir / "ckpt").glob("*.npz"))
    if a.checkpoint is not None:
        if Path(a.checkpoint).name != a.checkpoint:
            p.error(
                "--checkpoint expects a filename in this run's ckpt directory"
            )
        ckpts = [p for p in ckpts if Path(p).name == a.checkpoint]
    if a.last:
        ckpts = ckpts[-a.last :]

    if not ckpts:
        raise ValueError(f"No policy checkpoints in {a.run}")
    if cfg.get("robot", "g1") not in ("g1", *zenbot_models.MIXED_TASKS):
        p.error("Expected a combined walking, running and recovery run")
    options = simulation_options(cfg)
    options.update(num_envs=a.n, seed=123, random_start=True)
    sim = Sim(**options)

    def run(pi, norm, ode, key):
        def body(carry, k):
            s, acc = carry
            act = policy_action(sim, s, pi, norm, ode, a.obs_clip)
            s1, tr = sim.step(s, act)
            S1 = unflatten(tr.qpos1, tr.qvel1)
            yaw = env.heading(S1[1])
            v = S1[2]
            vb = jnp.stack(
                (
                    jnp.cos(yaw) * v[:, 0] + jnp.sin(yaw) * v[:, 1],
                    -jnp.sin(yaw) * v[:, 0] + jnp.cos(yaw) * v[:, 1],
                ),
                -1,
            )
            upright = s.was_up & sim.is_up(S1[0][:, 2], S1[1])
            err = jnp.linalg.norm(vb - s.c_obs[:, :2], axis=-1)
            fin = (tr.terminated | tr.truncated).astype(jnp.float32)
            up_ep, down_ep = fin * ~s.down, fin * s.down
            rose = sim.next_upright(s.was_up, S1[0][:, 2], S1[1])
            fell = acc["fell"] | (s.was_up & ~rose) | tr.terminated
            acc = {
                "parts": acc["parts"] + tr.parts.sum(0),
                "steps": acc["steps"] + a.n,
                "ep_up": acc["ep_up"] + up_ep.sum(),
                "fall_up": acc["fall_up"] + (up_ep * fell).sum(),
                "fell": fell & ~(tr.terminated | tr.truncated),
                "ep_down": acc["ep_down"] + down_ep.sum(),
                "rose": acc["rose"] + (down_ep * rose).sum(),
                "err": acc["err"] + (err * upright).sum(),
                "n_err": acc["n_err"] + upright.sum(),
            }
            return (s1, acc), None

        s = sim.init(key)
        z = jnp.zeros(())
        acc = {
            "parts": jnp.zeros((len(sim.terms),)),
            "steps": z,
            "ep_up": z,
            "fall_up": z,
            "fell": jnp.zeros(a.n, bool),
            "ep_down": z,
            "rose": z,
            "err": z,
            "n_err": z,
        }
        (_, acc), _ = jax.lax.scan(
            body, (s, acc), jax.random.split(key, int(a.seconds / env.DT))
        )
        return acc

    run_j = jax.jit(run, static_argnums=(2,))
    results = a.results or Path(a.run) / "eval.jsonl"
    results.parent.mkdir(parents=True, exist_ok=True)
    with results.open("w") as out:
        for path in ckpts:
            pi, norm, ode = load(path, a.which)
            acc = jax.device_get(run_j(pi, norm, ode, jax.random.PRNGKey(0)))
            per_s = acc["parts"] / (acc["steps"] * env.DT)
            rec = {
                "ckpt": path.name,
                **json.loads(path.with_suffix(".json").read_text()),
                **{f"rew_{k}": float(v) for k, v in zip(sim.terms, per_s)},
                "fall_rate": float(acc["fall_up"] / max(acc["ep_up"], 1)),
                "getup_rate": float(acc["rose"] / max(acc["ep_down"], 1)),
                "down_episodes": float(acc["ep_down"]),
                "vel_err": float(acc["err"] / max(acc["n_err"], 1)),
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(
                f"{rec['ckpt']} it {rec['it']} | |v-cmd| {rec['vel_err']:.3f} m/s | "
                f"falls {rec['fall_rate']:.3f} of upright starts | get-up {rec['getup_rate']:.3f} of {rec['down_episodes']:.0f} down starts",
                flush=True,
            )


if __name__ == "__main__":
    main()
