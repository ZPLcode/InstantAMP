"""Train the LeggedLab G1 get-up task with on-policy PPO and AMP on the InstantAMP simulator.

The actor sees only the five-frame robot observation. Ten-frame demonstration windows go
to the discriminator. The policy is exported using the same NPZ format as instantamp.train.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import amp, env, g1_getup, nets, optim
from .inference import policy_output
from .sim import Sim, unflatten
from .train import initialize_actor, save_policy, set_jax
from .train_getup_ppo import gae


@dataclasses.dataclass
class Settings:
    log_dir: str = "runs/g1_getup_amp"
    floor_dir: str = "data/g1_getup/recovery"
    num_envs: int = 1024
    iterations: int = 500000
    save_every: int = 50
    log_every: int = 10
    rollout: int = 24
    epochs: int = 5
    minibatches: int = 4
    disc_batch: int = 0  # zero uses the same minibatch size as PPO
    assist_newtons: float = 200.0
    reference_weight: float = 0.0
    reference_hold_scale: float = 1.0
    init_policy: str = ""
    init_std: float = 1.0
    init_lr: float = 0.001
    critic_warmup: int = 0  # iterations to fit a fresh critic and discriminator before updating the actor
    rise_anchor: float = 0.0  # preserve the initial actor's actions where the physical rise matches the reference
    seed: int = 42


class GetUpAMP:
    def __init__(self, cfg):
        self.c = c = cfg
        if min(c.num_envs, c.rollout, c.epochs, c.minibatches, c.save_every, c.log_every) <= 0:
            raise ValueError("Environment, rollout, optimizer and checkpoint counts must be positive")
        if c.num_envs * c.rollout % c.minibatches or c.rollout > 244:
            raise ValueError("Rollout must fit in the source replay and split into equal PPO minibatches")
        if not (0 < c.init_std and 0 < c.init_lr and 0 <= c.critic_warmup < c.iterations):
            raise ValueError("Expected positive initial std and learning rate and a warmup shorter than training")
        if not np.isfinite(c.rise_anchor) or c.rise_anchor < 0 or (c.rise_anchor and not (c.init_policy and c.reference_weight > 0)):
            raise ValueError("A positive rise anchor requires an initial actor and an active reference reward")
        self.sim = Sim(
            c.num_envs,
            robot="g1_getup",
            floor_dir=c.floor_dir,
            random_start=True,
            reset_floor=1.0,
            prob_shove=0.0,
            push=False,
            assist_newtons=c.assist_newtons,
            reference_weight=c.reference_weight,
            reference_hold_scale=c.reference_hold_scale,
        )
        self.demos = amp.Demos(c.floor_dir, self.sim, 10)
        self.disc_batch = c.disc_batch or c.num_envs * c.rollout // c.minibatches

    def init(self):
        ks = jax.random.split(jax.random.PRNGKey(self.c.seed), 5)
        s = self.sim.init(ks[0])
        frame = self.sim.amp_frame(unflatten(s.data.qpos, s.data.qvel))
        history = jnp.repeat(frame[:, None, :], 10, axis=1)
        ac = {
            "pi": nets.mlp_init(ks[2], [self.sim.obs_dim, 512, 256, 128, self.sim.J]),
            "v": nets.mlp_init(ks[3], [s.critic_obs.shape[-1], 512, 256, 128, 1]),
            "std": jnp.full(self.sim.J, self.c.init_std),
        }
        self.obs_norm = {"raw": True, **nets.rn_init(self.sim.obs_dim)}
        if self.c.init_policy:
            template = {"pi": nets.pi_init(ks[2], self.sim.obs_dim, self.sim.J), "obs_norm": nets.rn_init(self.sim.obs_dim)}
            resumed = initialize_actor(template, self.c.init_policy, expected_ode=None)
            ac["pi"], self.obs_norm = resumed["pi"], resumed["obs_norm"]
        if self.c.rise_anchor:
            self.anchor_pi = jax.tree.map(jnp.copy, ac["pi"])
            self.pose_term = self.sim.terms.index("motion_reference")
        self.mask = jax.tree.map(lambda _: False, ac)
        disc = amp.init(ks[4], 10, frame_dim=self.sim.amp_dim)
        replay = jnp.zeros((244, *history.shape), dtype=frame.dtype)
        return (s, history, replay, jnp.int32(0)), (ac, optim.init(ac, self.mask), jnp.float32(self.c.init_lr)), disc

    def collect(self, live, ac, disc, key):
        sim, c = self.sim, self.c

        def step(live, key):
            s, hist, replay, written = live
            mu = policy_output(ac["pi"], self.obs_norm, s.obs)
            std = jnp.maximum(ac["std"], 0.01)
            action = mu + std * jax.random.normal(key, mu.shape)
            value = nets.mlp_apply(ac["v"], s.critic_obs, jax.nn.elu)[:, 0]
            nxt, tr = sim.step(s, action)
            end = tr.terminated | tr.truncated
            frame = sim.amp_frame(unflatten(tr.qpos1, tr.qvel1))
            window = jnp.concatenate([hist[:, 1:], frame[:, None]], axis=1)
            first = sim.amp_frame(unflatten(nxt.data.qpos, nxt.data.qvel))
            hist = jnp.where(end[:, None, None], first[:, None], window)
            # IsaacLab computes its observation groups after resetting the ended worlds.
            style = env.DT * amp.recovery_style(disc, hist, 50.0, leggedlab=True)
            reward = 0.5 * (tr.reward + style)
            replay = replay.at[written % len(replay)].set(hist)
            row = (s.obs, s.critic_obs, mu, action, value, reward, end, tr.truncated)
            if c.rise_anchor:
                step = s.reference_start + s.t + 1
                pose_score = tr.parts[:, self.pose_term] / (env.DT * c.reference_weight)
                row += (g1_getup.rise_anchor_weight(sim.reference, s.reference_clip, step, pose_score),)
            stats = jnp.array(
                [
                    tr.reward.mean(),
                    style.mean(),
                    nxt.assist.mean(),
                    tr.qpos1[:, 2].mean(),
                    tr.stats["down_done"],
                    tr.stats["down_up"],
                    tr.stats["diverged"],
                ]
            )
            return (nxt, hist, replay, written + 1), (row, stats)

        live, (rows, stats) = jax.lax.scan(step, live, jax.random.split(key, c.rollout))
        xs, cs, mu, actions, value, r, end, timeouts, *anchor = rows
        bootstrap = nets.mlp_apply(ac["v"], live[0].critic_obs, jax.nn.elu)[:, 0]
        adv, returns = gae(r, value, bootstrap, end, timeouts, 0.99, 0.95)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch = tuple(x.reshape(-1, *x.shape[2:]) for x in (xs, cs, mu, actions, value, adv, returns, *anchor))
        return live, batch, stats.mean(0)

    def learn(self, optimizer, disc, batch, replay, written, key, train_actor=True):
        c = self.c
        old_std = jnp.maximum(optimizer[0]["std"], 0.01)

        def minibatch(carry, inputs):
            ac, opt, lr, disc = carry
            idx, key = inputs
            obs, critic_obs, old_mu, action, old_v, adv, returns, *anchor = (x[idx] for x in batch)
            if c.rise_anchor:
                target = policy_output(self.anchor_pi, self.obs_norm, obs)

            def objective(ac):
                mu = policy_output(ac["pi"], self.obs_norm, obs)
                std = jnp.maximum(ac["std"], 0.01)
                delta = jnp.log(old_std / std) + 0.5 * (
                    ((action - old_mu) / old_std) ** 2 - ((action - mu) / std) ** 2
                )
                ratio = jnp.exp(jnp.clip(delta.sum(-1), -20.0, 20.0))
                surrogate = -jnp.minimum(adv * ratio, adv * jnp.clip(ratio, 0.8, 1.2)).mean()
                v = nets.mlp_apply(ac["v"], critic_obs, jax.nn.elu)[:, 0]
                clipped = old_v + jnp.clip(v - old_v, -0.2, 0.2)
                value_loss = jnp.maximum((v - returns) ** 2, (clipped - returns) ** 2).mean()
                entropy = (jnp.log(std) + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum()
                kl = (
                    (jnp.log(std / old_std) + (old_std**2 + (old_mu - mu) ** 2) / (2 * std**2) - 0.5)
                    .sum(-1)
                    .mean()
                )
                loss = surrogate + value_loss - 0.01 * entropy
                if c.rise_anchor:
                    loss += c.rise_anchor * g1_getup.rise_anchor_loss(mu, target, anchor[0])
                return loss, (kl, surrogate, value_loss)

            (_, (kl, loss_pi, loss_v)), grad = jax.value_and_grad(objective, has_aux=True)(ac)
            if c.critic_warmup:
                grad = {**grad, "pi": jax.tree.map(lambda g: jnp.where(train_actor, g, 0.), grad["pi"]),
                        "std": jnp.where(train_actor, grad["std"], 0.)}
            lr = jnp.where(
                kl > 0.02,
                jnp.maximum(1e-5, lr / 1.5),
                jnp.where((kl > 0.0) & (kl < 0.005), jnp.minimum(0.01, lr * 1.5), lr),
            )
            grad, _ = optim.clip_by_global_norm(grad, 1.0)
            ac, opt = optim.step(ac, grad, opt, self.mask, lr)
            kr, ke, kd = jax.random.split(key, 3)
            sample = jax.random.randint(
                kr, (self.disc_batch,), 0, jnp.minimum(written, len(replay)) * c.num_envs
            )
            policy = replay[sample // c.num_envs, sample % c.num_envs]
            disc, ds = amp.train_step(
                disc,
                self.demos.sample(ke, self.disc_batch),
                policy,
                jnp.ones(self.disc_batch),
                kd,
                0.001,
                recovery=True,
                leggedlab=True,
            )
            return (ac, opt, lr, disc), jnp.array([kl, lr, loss_pi, loss_v, ds["d_demo"], ds["d_robot"]])

        def epoch(carry, key):
            kp, kd = jax.random.split(key)
            groups = jax.random.permutation(kp, len(batch[0])).reshape(c.minibatches, -1)
            return jax.lax.scan(minibatch, carry, (groups, jax.random.split(kd, c.minibatches)))

        (ac, opt, lr, disc), stats = jax.lax.scan(epoch, (*optimizer, disc), jax.random.split(key, c.epochs))
        return (ac, opt, lr), disc, stats.mean((0, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in dataclasses.fields(Settings):
        parser.add_argument("--" + field.name, type=type(field.default), default=field.default)
    cfg = Settings(**vars(parser.parse_args()))
    run = Path(cfg.log_dir)
    (run / "ckpt").mkdir(parents=True, exist_ok=False)
    config = {
        **dataclasses.asdict(cfg),
        "task": g1_getup.TASK_ID,
        "robot": "g1_getup",
        "nconmax": 64,
        "njmax": 640,
        "learner": "on_policy_ppo_amp",
        "action_rate_l2_weight": g1_getup.ACTION_RATE_L2_WEIGHT,
    }
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    set_jax("~/.cache/instantamp/jax")
    trainer = GetUpAMP(cfg)
    live, opt, disc = trainer.init()
    collect = jax.jit(trainer.collect, donate_argnums=(0,))
    learn = jax.jit(trainer.learn, donate_argnums=(0, 1))
    key = jax.random.PRNGKey(cfg.seed + 1)
    start = time.monotonic()
    with (run / "log.jsonl").open("x", buffering=1) as log:
        for it in range(1, cfg.iterations + 1):
            key, kc, kl = jax.random.split(key, 3)
            live, batch, rollout = collect(live, opt[0], disc, kc)
            opt, disc, stats = learn(opt, disc, batch, live[2], live[3], kl, it > cfg.critic_warmup)
            if it == 1 or it % cfg.log_every == 0 or it % cfg.save_every == 0 or it == cfg.iterations:
                keys = (
                    "task_reward",
                    "style_reward",
                    "assist_n",
                    "height",
                    "down_done",
                    "down_up",
                    "diverged",
                    "kl",
                    "lr",
                    "policy_loss",
                    "value_loss",
                    "disc_expert",
                    "disc_robot",
                )
                metrics = dict(zip(keys, map(float, jax.device_get(jnp.r_[rollout, stats]))))
                if not np.isfinite(list(metrics.values())).all():
                    raise FloatingPointError(metrics)
                meta = {
                    "it": it,
                    "env_steps": it * cfg.num_envs * cfg.rollout,
                    "train_s": time.monotonic() - start,
                }
                line = json.dumps({**meta, **metrics})
                print(line, flush=True)
                log.write(line + "\n")
            if it % cfg.save_every == 0 or it == cfg.iterations:
                policy = jax.device_get(opt[0]["pi"])
                save_policy(
                    str(run / "ckpt" / f"{it:06d}"), {"pi": policy, "avg": policy, "obs_norm": trainer.obs_norm}, meta
                )


if __name__ == "__main__":
    main()
