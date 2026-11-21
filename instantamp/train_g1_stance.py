"""Train a stationary G1 balance actor from physical states encountered late in get-up.

Pelvis translation, root linear velocity and contact loads are privileged training signals.
The exported actor uses the ordinary five-frame LeggedLab observation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import env, nets, optim
from .inference import load, policy_output
from .rot import quat_to_mat
from .sim import Sim
from .train import save_policy, set_jax
from .train_getup_ppo import gae


class StanceSim(Sim):
    def __init__(self, num_envs, bank):
        super().__init__(num_envs, robot="g1_getup", floor_dir="data/g1_getup/recovery",
                         random_start=True, push=False, assist_newtons=0.)
        poses, velocities = [], []
        for path in sorted(bank.glob("*/rollout.npz")):
            with np.load(path) as f:
                q, v = f["qpos"], f["qvel"]
            upright = 1 - 2 * np.sum(q[:, 4:6]**2, axis=1)
            keep = np.flatnonzero((q[:, 2] > .54) & (q[:, 2] < .84) & (upright > .52))[::5]
            if len(keep):
                poses.append(q[keep])
                velocities.append(v[keep])
        if not poses:
            raise ValueError(f"No physical stance transitions in {bank}")
        self.stance_q = jnp.asarray(np.concatenate(poses), jnp.float32)
        self.stance_v = jnp.asarray(np.concatenate(velocities), jnp.float32)
        self.episode_steps = 250

    def _fresh(self, key, world, clip=None, return_source=False):
        select, kick = jax.random.split(key)
        index = jax.random.randint(select, (self.N,), 0, len(self.stance_q))
        q = self.stance_q[index].at[:, :2].set(0.)
        # Small velocity disturbances include both forward and sideways arrivals.
        v = self.stance_v[index].at[:, :2].add(.12 * jax.random.normal(kick, (self.N, 2)))
        return q, v, self.is_up(q[:, 2], quat_to_mat(q[:, 3:7]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-policy", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--lr", type=float, default=0.000035)
    args = parser.parse_args()
    if min(args.envs, args.iterations, args.save_every) <= 0 or not (np.isfinite(args.lr) and args.lr > 0):
        parser.error("Environment count, iterations, save interval and learning rate must be positive")
    out = args.log_dir
    (out / "ckpt").mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "task": "LeggedLab-Isaac-AMP-G1-GET-UP-v0", "robot": "g1_getup",
              "floor_dir": "data/g1_getup/recovery", "walk_dir": None, "reset_floor": 1.,
              "prob_shove": 0., "push": False, "nconmax": 64, "njmax": 640,
              "initial_sha256": hashlib.sha256(args.init_policy.read_bytes()).hexdigest()}
    (out / "config.json").write_text(json.dumps(config, default=str, indent=2) + "\n")
    set_jax("~/.cache/instantamp/jax")
    sim = StanceSim(args.envs, args.bank)
    anchor, norm, ode = load(args.init_policy)
    if not isinstance(anchor, list) or ode is not None or anchor[0]["w"].shape[0] != sim.obs_dim:
        raise ValueError("Expected a direct-action G1 get-up policy")
    key, reset, initialize = jax.random.split(jax.random.key(args.seed), 3)
    live = sim.init(reset)
    params = dict(pi=anchor, v=nets.mlp_init(initialize, [sim.obs_dim+5, 256, 128, 1], layer_norm=True))
    mask = jax.tree.map(lambda _: False, params)
    opt = optim.init(params, mask)
    rollout, epochs, splits, std = 24, 3, 4, .22

    def critic_input(s):
        return jnp.concatenate((s.obs, s.data.qpos[:, :3], s.data.qvel[:, :2]), axis=-1)

    def value(params, obs):
        return nets.mlp_apply(params["v"], obs, jax.nn.elu)[:, 0]

    @jax.jit
    def collect(state, params, key):
        def step(state, key):
            x, cx = state.obs, critic_input(state)
            mu = policy_output(params["pi"], norm, x)
            act = mu + std * jax.random.normal(key, mu.shape)
            nxt, tr = sim.step(state, act)
            q, v = tr.qpos1, tr.qvel1
            torso = 1 - 2 * (q[:, 4:6]**2).sum(-1)
            load = tr.stats["foot_load"] / (state.world.body_mass.sum(-1) * 9.81)
            up = (q[:, 2] > .65) & (q[:, 2] < .85) & (torso > .866)
            quiet = jnp.exp(-14 * (v[:, :2]**2).sum(-1) - .4 * (v[:, 3:6]**2).sum(-1))
            position = jnp.exp(-14 * (q[:, :2]**2).sum(-1))
            support = jnp.clip(load, 0., 1.) * jnp.mean(tr.contact, axis=-1)
            height = jnp.exp(-70 * (q[:, 2] - .74)**2)
            reward = env.DT * (2 * height + torso + up * (1 + support * (2 + 4 * quiet + position)))
            reward = jnp.where(tr.terminated, -1., reward)
            metrics = jnp.array([up.mean(), (up * support).mean(), (up * quiet).mean(), reward.mean()])
            return nxt, ((x, cx, mu, act, value(params, cx), reward, tr.terminated | tr.truncated, tr.truncated), metrics)

        state, (rows, metrics) = jax.lax.scan(step, state, jax.random.split(key, rollout))
        x, cx, mu, act, old_value, reward, end, timeout = rows
        adv, returns = gae(reward, old_value, value(params, critic_input(state)), end, timeout, .99, .95)
        adv = (adv - adv.mean()) / (1e-6 + adv.std())
        batch = tuple(y.reshape(-1, *y.shape[2:]) for y in (x, cx, mu, act, adv, returns))
        return state, batch, metrics.mean(0)

    @jax.jit
    def learn(params, opt, batch, key):
        def update(carry, ix):
            params, opt = carry
            x, cx, old, act, adv, ret = (field[ix] for field in batch)
            teacher = policy_output(anchor, norm, x)

            def losses(p):
                mu = policy_output(p["pi"], norm, x)
                delta = (((act-old)**2 - (act-mu)**2) / (2*std**2)).sum(-1)
                ratio = jnp.exp(jnp.clip(delta, -12., 12.))
                pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, .8, 1.2) * adv).mean()
                # Constrain mean-action drift on the actual on-policy transition states.
                reg = ((mu - teacher)**2).mean() / (2*std**2)
                return pg + .5 * ((value(p, cx) - ret)**2).mean() + .5 * reg

            grad = jax.grad(losses)(params)
            grad, _ = optim.clip_by_global_norm(grad, 1.)
            return optim.step(params, grad, opt, mask, args.lr), None

        def epoch(carry, key):
            group = jax.random.permutation(key, len(batch[0])).reshape(splits, -1)
            return jax.lax.scan(update, carry, group)[0], None
        return jax.lax.scan(epoch, (params, opt), jax.random.split(key, epochs))[0]

    for it in range(1, args.iterations + 1):
        key, kc, ku = jax.random.split(key, 3)
        live, batch, metrics = collect(live, params, kc)
        params, opt = learn(params, opt, batch, ku)
        actor = params["pi"]
        labels = ("upright", "support", "quiet", "reward")
        if it == 1 or it % 10 == 0:
            print(it, dict(zip(labels, map(float, metrics))), flush=True)
        if it % args.save_every == 0 or it == args.iterations:
            if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(actor)):
                raise FloatingPointError(f"Non-finite weights at iteration {it}")
            meta = {"it": it, "env_steps": it * args.envs * rollout, "assist_n": 0.}
            save_policy(str(out / "ckpt" / f"{it:06d}"), {"pi": actor, "avg": actor, "obs_norm": norm}, meta)


if __name__ == "__main__":
    main()
