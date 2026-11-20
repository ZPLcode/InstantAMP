"""On-policy AMP training of the unconditioned 29-DoF G1 actor from physical rollouts."""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import amp, env, nets, optim
from .getup_amp import GetupDemos, getup_frame
from .inference import G1LimbBlend, load
from .sim import Sim, unflatten
from .train import G1StyleConfig, save_policy, set_jax


@dataclasses.dataclass
class Settings:
    init_policy: str = "runs/g1_4096_20261120_01/ckpt/0011.npz"
    log_dir: str = "runs/g1_autonomous_amp_ppo"
    floor_dir: str = "data/recovery/g1_rises/train"
    num_envs: int = 4096
    iterations: int = 600
    rollout: int = 24
    minibatch: int = 4096
    epochs: int = 3
    pi_lr: float = 0.000045
    value_lr: float = 0.0004
    sigma: float = 0.55
    gamma: float = 0.985
    gae_lambda: float = 0.95
    clip: float = 0.2
    reference_weight: float = 40.
    style_scale: float = 25.
    aligned_mix: float = .5
    hold_scale: float = .32
    tracking_anchor: float = .1
    floor_share: float = .82
    reset_frames: int = 32
    disc_batch: int = 512
    disc_lr: float = .0002
    disc_updates: int = 5
    save_every: int = 20
    episode_steps: int = 300
    seed: int = 7


def gae(reward, values, final_value, end, timeout, gamma, lam):
    """Use local terminal bootstraps for time limits, and stop traces at each reset."""
    after = jnp.concatenate((values[1:], final_value[None]), axis=0)
    bootstrap = jnp.where(timeout, values, jnp.where(end, 0., after))
    delta = reward + gamma * bootstrap - values
    def back(next_advantage, data):
        td, boundary = data
        result = td + gamma * lam * (1. - boundary) * next_advantage
        return result, result
    _, advantage = jax.lax.scan(back, jnp.zeros_like(final_value), (delta, end.astype(jnp.float32)), reverse=True)
    return advantage, advantage + values


class PhysicalAMP:
    def __init__(self, cfg):
        self.c = c = cfg
        if (c.num_envs * c.rollout) % c.minibatch or not 0 < c.aligned_mix < 1 or c.style_scale <= 0:
            raise ValueError("The rollout must split into complete minibatches and AMP must contribute to training")
        self.pi, self.norm, self.ode = load(c.init_policy)
        if isinstance(self.pi, G1LimbBlend) or self.ode is None or self.pi[0]["w"].shape[0] != env.OBS_DIM:
            raise ValueError("Initialize from one standard autonomous G1 ODE actor")
        self.sim = Sim(c.num_envs, floor_dir=c.floor_dir, reset_floor=c.floor_share, prob_shove=.04,
                       push=False, random_start=False, g1_style=True, reference_weight=c.reference_weight,
                       getup_rise_weight=-.20, getup_reset_frames=c.reset_frames, g1_support_friction=True)
        self.sim.episode_steps = c.episode_steps
        self.demonstrations = GetupDemos(c.floor_dir, self.sim, 10, segmented=True)
        self.pi_mask = jax.tree.map(lambda _: False, self.pi)
        self.reference_term = self.sim.terms.index("motion_reference")
        self.value_dim = env.OBS_DIM + 5
        self.disc_samples = min(c.disc_batch, c.num_envs)

    def value_input(self, s, obs):
        # Only the critic can see the current time, reference progress and pelvis height.
        valid_clip = jnp.maximum(s.reference_clip, 0)
        progress = jnp.clip((s.reference_start + s.t) / self.sim.reference.lengths[valid_clip], 0., 1.)
        priv = jnp.stack((s.data.qpos[:, 2], s.t / self.c.episode_steps,
                          progress * s.down, s.down, s.was_up), axis=-1)
        return jnp.column_stack((obs, priv))

    def initialize(self):
        c = self.c
        key = jax.random.PRNGKey(c.seed)
        s = self.sim.init(key)
        feat = getup_frame(self.sim, unflatten(s.data.qpos, s.data.qvel))
        hist = jnp.repeat(feat[:, None], 10, axis=1)
        d = amp.init(jax.random.fold_in(key, 1), 10, (512, 256), feat.shape[-1])
        d["norm"] = nets.rn_update(d["norm"], self.demonstrations.frames)
        v = nets.mlp_init(jax.random.fold_in(key, 2), [self.value_dim, 256, 192, 1], layer_norm=True)
        v[-1]["w"] = jnp.zeros_like(v[-1]["w"])
        self.v_mask = jax.tree.map(lambda _: False, v)
        params = (self.pi, optim.init(self.pi, self.pi_mask), v, optim.init(v, self.v_mask))
        return (s, hist, jax.random.fold_in(key, 3)), params, d

    def collect(self, live, pi, value_net, disc):
        c, sim = self.c, self.sim
        def step(live, _):
            s, hist, key = live
            key, sample_key, disc_key = jax.random.split(key, 3)
            obs = jnp.clip(nets.rn_apply(self.norm, s.obs), -10., 10.)
            mean = nets.pi_apply(pi, obs)
            latent = mean + c.sigma * jax.random.normal(sample_key, mean.shape)
            action = nets.ode_step(latent, s.a_ode, *self.ode)
            value_input = self.value_input(s, obs)
            value = nets.mlp_apply(value_net, value_input, jax.nn.silu)[:, 0]
            # The sampled ODE increment is integrated in the physical plant exactly as
            # the deterministic increment is integrated at deployment.
            next_s, tr = sim.step(s, action)
            end = tr.terminated | tr.truncated
            actual = unflatten(tr.qpos1, tr.qvel1)
            feat = getup_frame(sim, actual)
            window = jnp.concatenate((hist[:, 1:], feat[:, None]), axis=1)
            active = sim.reference.active(s.reference_clip, s.reference_start, s.t + 1)
            style = amp.recovery_style(disc, window, c.style_scale)
            styled = env.DT * (1 - c.aligned_mix) * jnp.where(active & ~tr.diverged, style, 0.)
            # Continue to value a stable stance after the demonstration ends.
            upright = jnp.clip(actual[1][:, 2, 2], 0., 1.)**3
            hold = (s.down & ~active & ~tr.diverged) * c.hold_scale * upright * jnp.exp(-((tr.qpos1[:, 2] - .76) / .13)**2)
            reward = c.aligned_mix * tr.reward + styled + hold - 3. * tr.terminated
            f0 = getup_frame(sim, unflatten(next_s.data.qpos, next_s.data.qvel))
            hist = jnp.where(end[:, None, None], f0[:, None], window)
            # Small, randomly chosen robot windows are sufficient for the discriminator.
            ids = jax.random.randint(disc_key, (self.disc_samples,), 0, c.num_envs)
            rows = (value_input, mean, latent, value, reward, end, tr.truncated, ~s.down)
            reservoir = (window[ids], (active & (s.t >= 9) & ~tr.diverged)[ids])
            stats = jnp.stack((reward.mean(), styled.mean(), tr.parts[:, self.reference_term].mean(),
                               active.mean(), s.down.mean(), tr.stats["down_done"], tr.stats["down_up"]))
            return (next_s, hist, key), (rows, reservoir, stats)
        live, (rows, reservoir, stats) = jax.lax.scan(step, live, None, length=c.rollout)
        s = live[0]
        x = self.value_input(s, jnp.clip(nets.rn_apply(self.norm, s.obs), -10., 10.))
        v1 = nets.mlp_apply(value_net, x, jax.nn.silu)[:, 0]
        xs, mean, latent, values, reward, ends, timeouts, walk = rows
        advantages, targets = gae(reward, values, v1, ends, timeouts, c.gamma, c.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        return live, tuple(map(flat, (xs, mean, latent, advantages, targets, walk))), tuple(map(flat, reservoir)), stats.mean(0)

    def update_disc(self, disc, pool, key):
        c = self.c
        windows, valid = pool
        def update(d, k):
            kd, kr, kg = jax.random.split(k, 3)
            idx = jax.random.randint(kr, (c.disc_batch,), 0, len(valid))
            return amp.train_step(d, self.demonstrations.sample(kd, c.disc_batch), windows[idx],
                                  valid[idx].astype(jnp.float32), kg, c.disc_lr, recovery=True)
        disc, stats = jax.lax.scan(update, disc, jax.random.split(key, c.disc_updates))
        return disc, jax.tree.map(jnp.mean, stats)

    def optimize(self, params, data, key):
        c = self.c
        def minibatch(params, indices):
            pi, pi_opt, val, val_opt = params
            x, old_mean, latent, adv, targets, walk = (a[indices] for a in data)
            obs = x[:, :env.OBS_DIM]
            def policy_loss(pi):
                mean = nets.pi_apply(pi, obs)
                log_ratio = (((latent - old_mean)**2 - (latent - mean)**2).sum(-1)) / (2 * c.sigma**2)
                ratio = jnp.exp(jnp.clip(log_ratio, -10., 10.))
                surr = jnp.minimum(adv * ratio, adv * jnp.clip(ratio, 1-c.clip, 1+c.clip))
                original = nets.pi_apply(self.pi, obs)
                walk_anchor = (((mean-original)**2).mean(-1) * walk).sum() / jnp.maximum(walk.sum(), 1.)
                loss = -surr.mean() + c.tracking_anchor * walk_anchor
                return loss, jnp.stack(((jnp.abs(ratio-1)>c.clip).mean(),
                                         ((mean-old_mean)**2).sum(-1).mean()/(2*c.sigma**2),walk_anchor))
            (_, pstats), grad = jax.value_and_grad(policy_loss, has_aux=True)(pi)
            grad, _ = optim.clip_by_global_norm(grad, 1.)
            pi, pi_opt = optim.step(pi, grad, pi_opt, self.pi_mask, c.pi_lr)
            def value_loss(v):
                return .5*((nets.mlp_apply(v,x,jax.nn.silu)[:, 0] - targets)**2).mean()
            loss_v, g_v = jax.value_and_grad(value_loss)(val)
            g_v, _ = optim.clip_by_global_norm(g_v, 2.)
            val, val_opt = optim.step(val, g_v, val_opt, self.v_mask, c.value_lr)
            return (pi, pi_opt, val, val_opt), jnp.r_[pstats, loss_v]
        def epoch(params, key):
            idx = jax.random.permutation(key, len(data[0])).reshape(-1, c.minibatch)
            return jax.lax.scan(minibatch, params, idx)
        params, stats = jax.lax.scan(epoch, params, jax.random.split(key, c.epochs))
        return params, stats.mean((0,1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for f in dataclasses.fields(Settings):
        p.add_argument("--"+f.name, type=type(f.default), default=f.default)
    c = Settings(**vars(p.parse_args()))
    path = Path(c.log_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path/"ckpt").mkdir(exist_ok=False)
    cfg = G1StyleConfig(num_envs=c.num_envs, reference_weight=c.reference_weight, g1_support_friction=1,
                        getup_reset_frames=c.reset_frames, reset_floor=c.floor_share, floor_dir=c.floor_dir,
                        log_dir=str(path), getup_rise_weight=-.2)
    (path/"config.json").write_text(json.dumps({**dataclasses.asdict(cfg),"physical_ppo":dataclasses.asdict(c)},indent=2)+"\n")
    set_jax(".venv/cache/jax")
    trainer = PhysicalAMP(c)
    live, params, disc = trainer.initialize()
    collect = jax.jit(trainer.collect, donate_argnums=(0,))
    fit_disc = jax.jit(trainer.update_disc, donate_argnums=(0,))
    optimize = jax.jit(trainer.optimize)
    key = jax.random.PRNGKey(c.seed+1)
    names = ("reward","amp_reward","weighted_reference","reference_active","floor_episode","down_done","down_up",
             "clipfrac","kl","walk_anchor","value_loss")
    t0 = time.monotonic()
    with (path/"log.jsonl").open("x",buffering=1) as log:
        for it in range(1, c.iterations+1):
            key, kd, kp = jax.random.split(key, 3)
            live, batch, pool, stats = collect(live, params[0], params[2], disc)
            disc, discr = fit_disc(disc, pool, kd)
            params, learning = optimize(params, batch, kp)
            if it == 1 or it%5 == 0 or it == c.iterations:
                row = dict(zip(names,map(float,jax.device_get(jnp.r_[stats,learning]))))
                row.update(it=it,env_steps=it*c.rollout*c.num_envs,seconds=time.monotonic()-t0,
                           **{k:float(v) for k,v in discr.items()})
                if not np.isfinite(list(row.values())).all():
                    raise FloatingPointError(row)
                print(json.dumps(row),flush=True); log.write(json.dumps(row)+"\n")
            if it%c.save_every == 0 or it==c.iterations:
                export={"pi":params[0],"avg":params[0],"obs_norm":trainer.norm}
                save_policy(str(path/"ckpt"/f"{it:04d}"),jax.device_get(export),
                            {"it":it,"train_s":time.monotonic()-t0,"env_steps":it*c.rollout*c.num_envs},ode=trainer.ode)

if __name__ == "__main__":
    main()
