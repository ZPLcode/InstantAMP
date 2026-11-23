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


def initial_live(trainer, state, history, replay):
    live = (state, history, replay, jnp.int32(0))
    if getattr(trainer, "virtual_ground_queue", False):
        live += (jnp.zeros((trainer.sim.N, 2, trainer.sim.J), dtype=state.a_last.dtype),)
    return live


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


def normalized_input(norm, obs):
    return jnp.clip(nets.rn_apply(norm, obs), -10., 10.)


class GetUpAMP:
    amp_frames = 10
    amp_stride = 1

    score_terminal_window = False
    max_lr = .01
    disc_lr = .001
    extra_rollout_stats = ()
    motion_metric_names = ()
    condition_dim = 0
    gamma = .99
    gae_lambda = .95

    @property
    def amp_span(self):
        """Number of 50 Hz simulator frames spanned by one discriminator input."""
        return 1 + (self.amp_frames - 1) * self.amp_stride

    @staticmethod
    def validate(c):
        if min(c.num_envs, c.rollout, c.epochs, c.minibatches, c.save_every, c.log_every) <= 0:
            raise ValueError("Environment, rollout, optimizer and checkpoint counts must be positive")
        if c.num_envs * c.rollout % c.minibatches or c.rollout > 244:
            raise ValueError("Rollout must fit in the source replay and split into equal PPO minibatches")
        if not (0 < c.init_std and 0 < c.init_lr and 0 <= c.critic_warmup < c.iterations):
            raise ValueError("Expected positive initial std and learning rate and a warmup shorter than training")
        if not np.isfinite(c.rise_anchor) or c.rise_anchor < 0 or (c.rise_anchor and not (c.init_policy and c.reference_weight > 0)):
            raise ValueError("A positive rise anchor requires an initial actor and an active reference reward")

    def __init__(self, cfg):
        self.c = c = cfg
        self.validate(c)
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
        self.demos = amp.Demos(c.floor_dir, self.sim, self.amp_span)
        self.disc_batch = c.disc_batch or c.num_envs * c.rollout // c.minibatches

    def critic_input(self, state):
        return state.critic_obs

    def actor_input(self, state):
        return state.obs if state.motion_train_sensors is None else state.motion_train_sensors.observation()

    def actor_mean(self, actor, obs):
        return policy_output(actor, self.obs_norm, obs)

    def action_std(self, ac):
        if getattr(self.c, "fixed_std", False):
            return jnp.full_like(ac["std"], self.c.init_std)
        return jnp.maximum(ac["std"], 0.01)

    def rollout_guidance(self, state, kind, iteration):
        from .zenbot_online_ground import rollout_control
        return rollout_control(kind, state.c_obs, iteration)

    def sample_reference_rows(self, key, count):
        offset_key, row_key = jax.random.split(key)
        # Half the examples are rise; the remaining eight skills split the rest.
        # Rotate the order so partial groups in small batches have equal probability.
        n_rise = count // 2
        offset = jax.random.randint(offset_key, (), 0, 8)
        other = 1 + (jnp.arange(count - n_rise) + offset) % 8
        skill = jnp.concatenate((jnp.zeros(n_rise, dtype=jnp.int32), other))
        row = jax.random.randint(row_key, (count,), 0, self.guide_counts[skill])
        return self.guide_rows[skill, row]

    def motion_features(self, window):
        return window

    def motion_reward(self, disc, motion, transition):
        return env.DT * amp.recovery_style(disc, motion, 50.0, leggedlab=True)

    @property
    def action_anchor_strength(self):
        return self.c.rise_anchor

    def action_anchor_weights(self, state, transition):
        step = state.reference_start + state.t + 1
        pose_score = transition.parts[:, self.pose_term] / (env.DT * self.c.reference_weight)
        return g1_getup.rise_anchor_weight(self.sim.reference, state.reference_clip, step, pose_score)

    def action_anchor_examples(self, obs, weights, actor_obs=None):
        return self.actor_mean(self.anchor_pi, obs if actor_obs is None else actor_obs), weights

    def blend_reward(self, transition, motion_reward):
        return .5 * (transition.reward + motion_reward)

    def rollout_motion(self, disc, motion, transition):
        return self.motion_reward(disc, motion, transition), motion, ()

    def discriminator_examples(self, key, policy):
        expert = self.demos.sample(key, len(policy))
        return self.motion_features(expert[:, ::self.amp_stride]), policy

    def fix_condition_norm(self, disc):
        if not self.condition_dim:
            return disc
        # A binary condition has a fixed scale even during rollouts that see only one skill.
        norm = disc["norm"]
        return {**disc, "norm": {**norm, "mean": norm["mean"].at[-self.condition_dim:].set(.5),
                                 "var": norm["var"].at[-self.condition_dim:].set(.25)}}

    def initialize_policy(self, template, path):
        return initialize_actor(template, path, expected_ode=None)

    def init(self):
        ks = jax.random.split(jax.random.PRNGKey(self.c.seed), 5)
        s = self.sim.init(ks[0])
        frame = self.sim.amp_frame(unflatten(s.data.qpos, s.data.qvel))
        history = jnp.repeat(frame[:, None, :], self.amp_span, axis=1)
        actor_dim = self.actor_input(s).shape[-1]
        ac = {
            "pi": nets.mlp_init(ks[2], [actor_dim, 512, 256, 128, self.sim.J]),
            "v": nets.mlp_init(ks[3], [self.critic_input(s).shape[-1], 512, 256, 128, 1]),
            "std": jnp.full(self.sim.J, self.c.init_std),
        }
        self.obs_norm = {"raw": True, **nets.rn_init(actor_dim)}
        scratch = getattr(self.c, "scratch_ppo", False)
        if scratch and (self.c.init_policy or self.action_anchor_strength):
            raise ValueError("Scratch PPO requires a random actor and zero actor anchors")
        if self.c.init_policy:
            template = {"pi": nets.pi_init(ks[2], actor_dim, self.sim.J), "obs_norm": nets.rn_init(actor_dim)}
            resumed = self.initialize_policy(template, self.c.init_policy)
            ac["pi"], self.obs_norm = resumed["pi"], resumed["obs_norm"]
        if self.action_anchor_strength:
            self.anchor_pi = jax.tree.map(jnp.copy, ac["pi"])
        if self.c.rise_anchor:
            self.pose_term = self.sim.terms.index("motion_reference")
        self.mask = jax.tree.map(lambda _: False, ac)
        disc = self.fix_condition_norm(amp.init(ks[4], self.amp_frames, frame_dim=self.sim.amp_dim + self.condition_dim))
        replay = jnp.zeros((244, frame.shape[0], self.amp_frames, frame.shape[-1] + self.condition_dim), dtype=frame.dtype)
        optimizer = (ac, optim.init(ac, self.mask), jnp.float32(self.c.init_lr))
        if scratch:
            # These estimates are rollout state and receive no Adam gradients.
            norms = {"actor": nets.rn_init(actor_dim), "critic": nets.rn_init(self.critic_input(s).shape[-1])}
            optimizer += (norms,)
        live = initial_live(self, s, history, replay)
        return live, optimizer, disc

    def collect(self, live, ac, disc, key, norms=None, *, ppo_iteration=None):
        sim, c = self.sim, self.c
        online = bool(getattr(c, "online_ground_bank", ""))

        virtual_ground_queue = getattr(self, "virtual_ground_queue", False)

        def step(live, key):
            s, hist, replay, written = live[:4]
            actor_obs = self.actor_input(s)
            actor_input = actor_obs if norms is None else normalized_input(norms["actor"], actor_obs)
            mu = self.actor_mean(ac["pi"], actor_input)
            std = self.action_std(ac)
            if online:
                from .zenbot_online_ground import behavior_gaussian
                if virtual_ground_queue:
                    label, kind, queue = self.online_ground_with_queue(s, live[4])
                else:
                    label, kind = self.online_ground_targets(s)
                guide_fraction, noise_scale = self.rollout_guidance(s, kind, ppo_iteration)
                mu, std = behavior_gaussian(mu, std, label, guide_fraction, noise_scale)
            action = mu + std * jax.random.normal(key, mu.shape)
            critic_obs = self.critic_input(s)
            value = critic(critic_obs)
            nxt, tr = sim.step(s, action)
            end = tr.terminated | tr.truncated
            frame = sim.amp_frame(unflatten(tr.qpos1, tr.qvel1))
            window = jnp.concatenate([hist[:, 1:], frame[:, None]], axis=1)
            first = sim.amp_frame(unflatten(nxt.data.qpos, nxt.data.qvel))
            hist = jnp.where(end[:, None, None], first[:, None], window)
            # The G1 port follows IsaacLab's post-reset observation convention.
            raw = window if self.score_terminal_window else hist
            motion = self.motion_features(raw[:, ::self.amp_stride])
            style, replay_window, motion_metrics = self.rollout_motion(disc, motion, tr)
            reward = self.blend_reward(tr, style)
            replay = replay.at[written % len(replay)].set(replay_window)
            row = (actor_obs, critic_obs, mu, action, value, reward, end, tr.truncated)
            if online:
                row += (label, kind, guide_fraction, noise_scale)
                motion_metrics += ((kind == 1).mean(), (kind == 2).mean(), guide_fraction.mean())
            if self.action_anchor_strength:
                row += (self.action_anchor_weights(s, tr),)
                if s.motion_train_sensors is not None:
                    row += (s.obs,)  # privileged history for teachers only
            if getattr(c, "source_gait", False):
                row += (tr.stats["source_gait_active"],)
            stats = jnp.array(
                [
                    tr.reward.mean(),
                    style.mean(),
                    nxt.assist.mean(),
                    tr.qpos1[:, 2].mean(),
                    tr.stats["down_done"],
                    tr.stats["down_up"],
                    tr.stats["diverged"],
                ] + [tr.stats[name] for name in self.extra_rollout_stats] + list(motion_metrics)
            )
            next_live = (nxt, hist, replay, written + 1)
            if virtual_ground_queue:
                next_live += (jnp.where(end[:, None, None], 0., queue),)
            return next_live, (row, stats)

        def critic(obs):
            if norms is not None:
                obs = normalized_input(norms["critic"], obs)
            return nets.mlp_apply(ac["v"], obs, jax.nn.elu)[:, 0]

        live, (rows, stats) = jax.lax.scan(step, live, jax.random.split(key, c.rollout))
        xs, cs, mu, actions, value, r, end, timeouts, *anchor = rows
        bootstrap = critic(self.critic_input(live[0]))
        adv, returns = gae(r, value, bootstrap, end, timeouts, self.gamma, self.gae_lambda)
        if getattr(c, "source_gait", False):
            active_gait = anchor.pop()
            def within_task(mask):
                count = jnp.maximum(mask.sum(), 1)
                mean = jnp.where(mask, adv, 0.).sum() / count
                deviation = adv - mean
                std = jnp.sqrt(jnp.where(mask, deviation ** 2, 0.).sum() / count)
                return deviation / jnp.maximum(std, 1e-6), std
            gait_adv, gait_std = within_task(active_gait)
            floor_adv, floor_std = within_task(~active_gait)
            adv = jnp.where(active_gait, gait_adv, floor_adv)
            rollout_stats = jnp.r_[stats.mean(0), floor_std, gait_std]
        else:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            rollout_stats = stats.mean(0)
        batch = tuple(x.reshape(-1, *x.shape[2:]) for x in (xs, cs, mu, actions, value, adv, returns, *anchor))
        return live, batch, rollout_stats

    def learn(self, optimizer, disc, batch, replay, written, key, train_actor=True, *, ppo_iteration=None):
        c = self.c
        scratch = getattr(c, "scratch_ppo", False)
        guide = bool(getattr(c, "reference_guide_path", ""))
        online = bool(getattr(c, "online_ground_bank", ""))
        if online and not guide:
            raise ValueError("Online ground labels require the scratch reference Huber learner")
        if guide:
            if not scratch or ppo_iteration is None:
                raise ValueError("Reference guidance requires scratch PPO and the current iteration")
            fraction = jnp.clip((c.guide_end_iteration - ppo_iteration) / (c.guide_end_iteration - 1), 0., 1.)
            weight = c.guide_final_weight + (c.guide_weight - c.guide_final_weight) * fraction
        if scratch:
            ac, opt, lr, norms = optimizer
            optimizer = ac, opt, lr
            actor_obs, critic_obs = batch[:2]
            next_norms = {"actor": nets.rn_update(norms["actor"], actor_obs),
                          "critic": nets.rn_update(norms["critic"], critic_obs)}
            # Every PPO epoch must reproduce the preprocessing used to sample old_mu.
            batch = (normalized_input(norms["actor"], actor_obs),
                     normalized_input(norms["critic"], critic_obs), *batch[2:])
        old_std = self.action_std(optimizer[0])

        def minibatch(carry, inputs):
            ac, opt, lr, disc = carry
            idx, key = inputs
            actor_obs, critic_obs, old_mu, action, old_v, adv, returns, *anchor = (x[idx] for x in batch)
            if self.action_anchor_strength:
                teacher_obs = anchor[1] if len(anchor) > 1 else actor_obs
                target, anchor_weights = self.action_anchor_examples(teacher_obs, anchor[0], actor_obs)
                target = jax.lax.stop_gradient(target)

            def objective(ac):
                mu = self.actor_mean(ac["pi"], actor_obs)
                std = self.action_std(ac)
                policy_mu = mu
                previous_std = old_std
                if online:
                    from .zenbot_online_ground import behavior_gaussian
                    label, _, guide_fraction, noise_scale = anchor
                    mu, std = behavior_gaussian(policy_mu, std, label, guide_fraction, noise_scale)
                    previous_std = old_std * noise_scale[:, None]
                delta = jnp.log(previous_std / std) + 0.5 * (
                    ((action - old_mu) / previous_std) ** 2 - ((action - mu) / std) ** 2
                )
                ratio = jnp.exp(jnp.clip(delta.sum(-1), -20.0, 20.0))
                surrogate = -jnp.minimum(adv * ratio, adv * jnp.clip(ratio, 0.8, 1.2)).mean()
                v = nets.mlp_apply(ac["v"], critic_obs, jax.nn.elu)[:, 0]
                clipped = old_v + jnp.clip(v - old_v, -0.2, 0.2)
                value_loss = jnp.maximum((v - returns) ** 2, (clipped - returns) ** 2).mean()
                entropy = (jnp.log(std) + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum(-1).mean()
                kl = (
                    (jnp.log(std / previous_std) + (previous_std**2 + (old_mu - mu) ** 2) / (2 * std**2) - 0.5)
                    .sum(-1)
                    .mean()
                )
                loss = surrogate + value_loss - 0.01 * entropy
                if self.action_anchor_strength:
                    loss += self.action_anchor_strength * g1_getup.rise_anchor_loss(mu, target, anchor_weights)
                guide_loss = jnp.float32(0.)
                if guide:
                    def reference_loss(pi):
                        # An independent label sample leaves the on-policy batch and replay RNG unchanged.
                        idx_ref = self.sample_reference_rows(jax.random.fold_in(key, 4319), c.guide_batch)
                        obs_ref = normalized_input(norms["actor"], self.guide_obs[idx_ref])
                        error = jnp.abs(self.actor_mean(pi, obs_ref) - self.guide_targets[idx_ref])
                        reference_huber = jnp.mean(jnp.where(error < 1., 0.5 * error**2, error - 0.5))
                        if online:
                            target, kind, _, _ = anchor
                            # mu came from the same frozen normalization as the sampled old_mu.
                            eligible = kind > 0
                            error = jnp.abs(policy_mu - target)
                            row_loss = jnp.where(error < 1., .5 * error**2, error - .5).mean(axis=-1)
                            count = eligible.sum()
                            current_huber = jnp.where(eligible, row_loss, 0.).sum() / jnp.maximum(count, 1)
                            return jnp.where(count > 0, .5 * (reference_huber + current_huber), reference_huber)
                        return reference_huber

                    guide_loss = jax.lax.cond(weight > 0., reference_loss, lambda _: jnp.float32(0.), ac["pi"])
                    loss += weight * guide_loss
                return loss, (kl, surrogate, value_loss, guide_loss)

            (_, (kl, loss_pi, loss_v, guide_loss)), grad = jax.value_and_grad(objective, has_aux=True)(ac)
            if c.critic_warmup:
                grad = {**grad, "pi": jax.tree.map(lambda g: jnp.where(train_actor, g, 0.), grad["pi"]),
                        "std": jnp.where(train_actor, grad["std"], 0.)}
            lr = jnp.where(
                kl > 0.02,
                jnp.maximum(1e-5, lr / 1.5),
                jnp.where((kl > 0.0) & (kl < 0.005), jnp.minimum(self.max_lr, lr * 1.5), lr),
            )
            grad, _ = optim.clip_by_global_norm(grad, 1.0)
            ac, opt = optim.step(ac, grad, opt, self.mask, lr)
            if scratch:
                ac = {**ac, "std": jnp.maximum(ac["std"], 0.01)}
            kr, ke, kd = jax.random.split(key, 3)
            sample = jax.random.randint(kr, (self.disc_batch,), 0, jnp.minimum(written, len(replay)) * c.num_envs)
            policy = replay[sample // c.num_envs, sample % c.num_envs]
            expert, policy = self.discriminator_examples(ke, policy)
            disc, ds = amp.train_step(
                disc,
                expert,
                policy,
                jnp.ones(self.disc_batch),
                kd,
                self.disc_lr,
                recovery=True,
                leggedlab=True,
                include_demo_norm=scratch,
            )
            disc = self.fix_condition_norm(disc)
            metrics = [kl, lr, loss_pi, loss_v, ds["d_demo"], ds["d_robot"]]
            if guide:
                metrics.extend((guide_loss, weight))
            return (ac, opt, lr, disc), jnp.array(metrics)

        def epoch(carry, key):
            kp, kd = jax.random.split(key)
            groups = jax.random.permutation(kp, len(batch[0])).reshape(c.minibatches, -1)
            return jax.lax.scan(minibatch, carry, (groups, jax.random.split(kd, c.minibatches)))

        (ac, opt, lr, disc), stats = jax.lax.scan(epoch, (*optimizer, disc), jax.random.split(key, c.epochs))
        optimizer = (ac, opt, lr, next_norms) if scratch else (ac, opt, lr)
        return optimizer, disc, stats.mean((0, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in dataclasses.fields(Settings):
        parser.add_argument("--" + field.name, type=type(field.default), default=field.default)
    cfg = Settings(**vars(parser.parse_args()))
    train(cfg, GetUpAMP, {
        "task": g1_getup.TASK_ID,
        "robot": "g1_getup",
        "nconmax": 64,
        "njmax": 640,
        "learner": "on_policy_ppo_amp",
        "action_rate_l2_weight": g1_getup.ACTION_RATE_L2_WEIGHT,
    })


def train(cfg, trainer_type, metadata):
    run = Path(cfg.log_dir)
    (run / "ckpt").mkdir(parents=True, exist_ok=False)
    config = {**dataclasses.asdict(cfg), **metadata}
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    set_jax("~/.cache/instantamp/jax")
    trainer = trainer_type(cfg)
    if hasattr(trainer, "gait_reset_summary"):
        config["gait_reset_bank"] = trainer.gait_reset_summary
        (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    live, opt, disc = trainer.init()
    scratch = getattr(cfg, "scratch_ppo", False)

    def save_actor(it, norm, meta):
        policy = jax.device_get(opt[0]["pi"])
        save_policy(str(run / "ckpt" / f"{it:06d}"), {"pi": policy, "avg": policy, "obs_norm": norm}, meta)

    if scratch:
        save_actor(0, opt[3]["actor"], {"it": 0, "env_steps": 0, "train_s": 0.0})
    # Upright resets can give command-gait state fields the same device buffer.
    # XLA cannot donate a buffer twice when it flattens that state.
    collect = jax.jit(trainer.collect, donate_argnums=() if trainer.sim.zenbot_command_gait else (0,))
    learn = jax.jit(trainer.learn, donate_argnums=(0, 1))
    fit = None
    if getattr(cfg, "reference_fit_steps", 0):
        fit = jax.jit(trainer.fit_reference)
        reference_moments = optim.init(opt[0]["pi"], trainer.mask["pi"])
    key = jax.random.PRNGKey(cfg.seed + 1)
    start = time.monotonic()
    with (run / "log.jsonl").open("x", buffering=1) as log:
        for it in range(1, cfg.iterations + 1):
            key, kc, kl = jax.random.split(key, 3)
            if fit is not None and 1 < it <= 2500:
                pi, reference_moments = fit(opt[0]["pi"], reference_moments, opt[3]["actor"], jax.random.fold_in(kc, 419))
                opt = ({**opt[0], "pi": pi}, *opt[1:])
            iteration_arg = {"ppo_iteration": jnp.int32(it)} if getattr(cfg, "assist_curriculum", False) else {}
            live, batch, rollout = collect(live, opt[0], disc, kc, *opt[3:], **iteration_arg)
            save = it % cfg.save_every == 0 or it == cfg.iterations
            if save:
                # The learner donates opt and replaces its norm for the next rollout.
                norm = jax.tree.map(lambda x: np.array(x, copy=True), opt[3]["actor"]) if scratch else trainer.obs_norm
            guide_iteration = {"ppo_iteration": jnp.int32(it)} if getattr(cfg, "reference_guide_path", "") else {}
            opt, disc, stats = learn(opt, disc, batch, live[2], live[3], kl, it > cfg.critic_warmup, **guide_iteration)
            if it == 1 or it % cfg.log_every == 0 or it % cfg.save_every == 0 or it == cfg.iterations:
                keys = (
                    "task_reward",
                    "style_reward",
                    "assist_n",
                    "height",
                    "down_done",
                    "down_up",
                    "diverged",
                    *trainer.extra_rollout_stats,
                    *trainer.motion_metric_names,
                    "kl",
                    "lr",
                    "policy_loss",
                    "value_loss",
                    "disc_expert",
                    "disc_robot",
                )
                if getattr(cfg, "reference_guide_path", ""):
                    keys += ("reference_huber", "reference_guide_weight")
                values = jnp.r_[rollout, stats]
                if scratch:
                    keys += ("action_std_min", "action_std_mean", "action_std_max")
                    std = opt[0]["std"]
                    values = jnp.r_[values, std.min(), std.mean(), std.max()]
                metrics = dict(zip(keys, map(float, jax.device_get(values))))
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
            if save:
                save_actor(it, norm, meta)


if __name__ == "__main__":
    main()
