"""Train one Zenbot policy with a shared AMP discriminator for rising and standing."""

import argparse
import dataclasses

import jax
import jax.numpy as jnp

from . import amp, env
from .sim import Sim
from .train_g1_getup_amp import GetUpAMP, train


@dataclasses.dataclass
class Settings:
    log_dir: str = "runs/zenbot23_getup_standing_amp"
    floor_dir: str = "data/zenbot23_walk_run_recovery/recovery"
    stand_dir: str = "data/zenbot23_walk_run_recovery/standing"
    standing_amp: int = 1
    init_policy: str = ""
    num_envs: int = 384
    iterations: int = 10000
    rollout: int = 96
    epochs: int = 3
    minibatches: int = 4
    disc_batch: int = 256
    support_weight: float = 8.
    default_stand: int = 1
    clip_pd_targets: int = 1
    init_std: float = .24
    init_lr: float = .00003
    critic_warmup: int = 10
    save_every: int = 25
    log_every: int = 5
    seed: int = 42

    rise_anchor = 0.
    reference_weight = 0.


class RiseStandAMP(GetUpAMP):
    amp_stride = 3
    score_terminal_window = True
    extra_rollout_stats = ("supported_stand",)
    disc_lr = .0002
    max_lr = .00005
    gamma = .997
    gae_lambda = .985

    def __init__(self, cfg):
        self.validate(cfg)
        if cfg.standing_amp not in (0, 1) or (cfg.standing_amp and not cfg.support_weight > 0.):
            raise ValueError("standing_amp must be 0 or 1 and requires physical support measurements when enabled")
        if cfg.default_stand not in (0, 1):
            raise ValueError("default_stand must be 0 or 1")
        if cfg.clip_pd_targets not in (0, 1):
            raise ValueError("clip_pd_targets must be 0 or 1")
        self.c = cfg
        self.condition_dim = int(cfg.standing_amp)
        self.disc_batch = cfg.disc_batch or cfg.num_envs * cfg.rollout // cfg.minibatches
        self.sim = Sim(cfg.num_envs, robot="zenbot23_getup", floor_dir=cfg.floor_dir,
                       zenbot_getup_stand=True, zenbot_support_weight=cfg.support_weight,
                       zenbot_default_stand=bool(not cfg.standing_amp and cfg.default_stand and cfg.support_weight),
                       zenbot_standing_amp=bool(cfg.standing_amp),
                       clip_pd_targets=bool(cfg.clip_pd_targets),
                       reset_floor=1., prob_shove=0., push=False,
                       assist_newtons=0., random_start=True)
        if cfg.clip_pd_targets:
            self.extra_rollout_stats += ("pd_request_outside_fraction",)
        if cfg.support_weight:
            self.extra_rollout_stats += ("support_hold_score",)
        self.demos = amp.Demos(cfg.floor_dir, self.sim, 1 + 9 * self.amp_stride, per_clip=True,
                               frame_fn=self.sim.amp_frame)
        if cfg.standing_amp:
            self.stand_demos = amp.Demos(cfg.stand_dir, self.sim, 1 + 9 * self.amp_stride,
                                         per_clip=True, frame_fn=self.sim.amp_frame)
            self.motion_metric_names = ("rise_amp_contribution", "stand_amp_contribution")

    def critic_input(self, state):
        return jnp.concatenate((state.obs, state.data.qpos[:, 2:3], state.data.qvel[:, :3]), axis=-1)

    def motion_features(self, window):
        # Estimate velocity at the same sampling rate in both motion sources.
        n = self.sim.J
        dq = jnp.diff(window[..., 9:9+n], axis=-2) / (env.DT * self.amp_stride)
        return window.at[..., 9+n:9+2*n].set(jnp.concatenate((dq[..., :1, :], dq), axis=-2))

    @staticmethod
    def conditioned(motion, label):
        channel = jnp.broadcast_to(label.astype(motion.dtype)[:, None, None], (*motion.shape[:-1], 1))
        return jnp.concatenate((motion, channel), axis=-1)

    def rollout_motion(self, disc, motion, transition):
        if not self.condition_dim:
            style = env.DT * amp.recovery_style(disc, motion, scale=10., leggedlab=True)
            blend = transition.stats["stand_blend"] if "stand_blend" in transition.stats else transition.stats["stand_quality"]
            return (1. - blend) * style, motion, ()

        hold = transition.stats["stand_blend"]
        # One forward pass scores the observed motion under each skill condition.
        windows = jnp.concatenate((self.conditioned(motion, jnp.zeros_like(hold)),
                                   self.conditioned(motion, jnp.ones_like(hold))), axis=0)
        rise, stand = jnp.split(env.DT * amp.recovery_style(disc, windows, scale=10., leggedlab=True), 2)
        rise_part, stand_part = (1. - hold) * rise, hold * stand
        return rise_part + stand_part, self.conditioned(motion, hold), (rise_part.mean(), stand_part.mean())

    def motion_reward(self, disc, motion, transition):
        return self.rollout_motion(disc, motion, transition)[0]

    def discriminator_examples(self, key, policy):
        if not self.condition_dim:
            return super().discriminator_examples(key, policy)
        k_group, k_rise, k_stand = jax.random.split(key, 3)
        # Draw the same label for both sides of each independent expert-policy pair.
        is_stand = jax.random.bernoulli(k_group, policy[:, -1, -1])
        rise = self.demos.sample(k_rise, len(policy))
        stand = self.stand_demos.sample(k_stand, len(policy))
        source = jnp.where(is_stand[:, None, None], stand[:, ::self.amp_stride], rise[:, ::self.amp_stride])
        return (self.conditioned(self.motion_features(source), is_stand),
                self.conditioned(policy[..., :-1], is_stand))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in dataclasses.fields(Settings):
        parser.add_argument("--" + field.name, type=type(field.default), default=field.default)
    cfg = Settings(**vars(parser.parse_args()))
    metadata = dict(task="zenbot23_getup_stand", robot="zenbot23_getup", learner="on_policy_ppo_amp",
                    assist_newtons=0., reference_weight=0., ode=0)
    if cfg.standing_amp:
        metadata.update(default_stand=0, policies=1, discriminators=1,
                        amp_weight="(1 - stand_blend) * rise_amp + stand_blend * standing_amp",
                        expert_directories={"rise": cfg.floor_dir, "standing": cfg.stand_dir})
    else:
        metadata["amp_weight"] = "1 - stand_blend" if cfg.default_stand and cfg.support_weight else "1 - stand_quality"
    train(cfg, RiseStandAMP, metadata)


if __name__ == "__main__":
    main()
