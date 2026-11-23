"""Train one Zenbot actor on recovery and standing, optionally with commanded walk/run AMP."""

import argparse
import dataclasses
import math

import jax
import jax.numpy as jnp

from . import amp, env, zenbot
from .sim import Sim
from .train_g1_getup_amp import GetUpAMP, train


@dataclasses.dataclass
class Settings:
    log_dir: str = "runs/zenbot23_getup_standing_amp"
    floor_dir: str = "data/zenbot23_walk_run_recovery/recovery"
    stand_dir: str = "data/zenbot23_walk_run_recovery/standing"
    gait_dir: str = ""  # 50 Hz Zenbot walk_run bank; enables the command-conditioned actor
    directional_gait: int = 0  # add retreat, strafe and in-place turns to the command and AMP banks
    motion_train_obs: int = -1  # select the Motion Train actor interface by default for directional locomotion
    floor_reset_fraction: float = .35  # four-skill episodes that start on the floor instead of in gait
    floor_reset_weights: str = ""  # optional integer sample counts per opening, in sorted clip filename order
    standing_amp: int = 1
    init_policy: str = ""
    ground_actor_anchor: float = 0.  # match the initial actor on low-pelvis get-up states
    policy_anchor: float = 0.  # match a pretrained multi-skill actor on its on-policy inputs during fine-tuning
    rise_teacher: str = ""  # optional original 480-input teacher for low-pelvis samples of the same imitation loss
    rise_teacher_share: float = 1.  # fraction of the low-pelvis label supplied by the rise teacher
    rise_teacher_after_s: float = 0.  # optional time before blending in the floor teacher over one second
    gait_teacher: str = ""  # optional Motion Train actor for supported, recovered states of the same loss
    walk_teacher_pose_gain: float = 0.  # correct out-of-reference poses in supervised walking labels
    num_envs: int = 384
    iterations: int = 10000
    rollout: int = 96
    epochs: int = 3
    minibatches: int = 4
    disc_batch: int = 256
    support_weight: float = 8.
    stand_reference_weight: float = 8.  # AMP mode only, reward per second at the supported expert pose
    gait_speed_weight: float = 4.  # reward per second for matching the commanded linear and yaw velocities
    default_stand: int = 1
    init_std: float = .24
    init_lr: float = .00003
    critic_warmup: int = 10
    save_every: int = 25
    log_every: int = 5
    seed: int = 42

    rise_anchor = 0.
    reference_weight = 0.

    def __post_init__(self):
        if self.motion_train_obs == -1:
            self.motion_train_obs = int(bool(self.directional_gait))


def standing_pose_match(frame, target):
    """Score the physical pose channels of two heading-aligned Zenbot AMP frames."""
    joint_start = 9  # two rotation columns followed by root angular velocity
    joint_count = len(zenbot.DOF)
    height_index = joint_start + 2 * joint_count + 3 * len(zenbot.KEY_BODIES)
    angles = slice(joint_start, joint_start + joint_count)
    error = jnp.abs(frame[..., angles] - target[angles])
    waist = zenbot.DOF.index("waist_yaw")
    legs = error[..., :waist].mean(axis=-1) / math.radians(25.)
    upper_body = error[..., waist:].mean(axis=-1) / math.radians(25.)
    # The second stored rotation column is the local vertical in heading-aligned coordinates.
    # The chest is attached to the pelvis by a yaw hinge, so both have the same up axis.
    tilt = jnp.linalg.norm(frame[..., 3:6] - target[3:6], axis=-1) / (2. * math.sin(math.radians(10.) / 2.))
    height = jnp.abs(frame[..., height_index] - target[height_index]) / .06
    distances = jnp.stack((legs, upper_body, tilt, height), axis=-1)
    return (1. / (1. + distances)).mean(axis=-1)


def gait_skill_weights(hold, command, ready):
    """Return rise, stand, walk and run weights per world, using the command that generated this transition."""
    travel = ready.astype(hold.dtype) * jnp.any(jnp.abs(command) > .05, axis=-1)
    run = jnp.clip((command[..., 0] - 1.) / (2.5 - 1.), 0., 1.)
    return jnp.stack(((1. - travel) * (1. - hold), (1. - travel) * hold,
                      travel * (1. - run), travel * run), axis=-1)


class RiseStandAMP(GetUpAMP):
    standing_target = None
    amp_stride = 3
    score_terminal_window = True
    extra_rollout_stats = ("supported_stand", "pd_request_outside_fraction")
    disc_lr = .0002
    max_lr = .00005
    gamma = .997
    gae_lambda = .985

    def simulation_options(self, cfg):
        return {}

    @staticmethod
    def validate(cfg):
        GetUpAMP.validate(cfg)
        if cfg.directional_gait not in (0, 1) or cfg.motion_train_obs not in (0, 1):
            raise ValueError("directional_gait and motion_train_obs must be 0 or 1")
        if (cfg.directional_gait or cfg.motion_train_obs) and not cfg.gait_dir:
            raise ValueError("Directional commands and Motion Train observations require a gait directory")
        if cfg.floor_reset_weights and not (cfg.gait_dir and cfg.floor_reset_fraction > 0):
            raise ValueError("floor_reset_weights requires floor resets in the command-conditioned task")
        if not math.isfinite(cfg.stand_reference_weight) or cfg.stand_reference_weight < 0.:
            raise ValueError("stand_reference_weight must be finite and nonnegative")
        if not math.isfinite(cfg.ground_actor_anchor) or cfg.ground_actor_anchor < 0:
            raise ValueError("ground_actor_anchor must be finite and nonnegative")
        if cfg.ground_actor_anchor and not cfg.init_policy:
            raise ValueError("An initial policy is required to anchor ground actions")
        if not math.isfinite(cfg.policy_anchor) or cfg.policy_anchor < 0.:
            raise ValueError("policy_anchor must be finite and nonnegative")
        if cfg.policy_anchor and (not cfg.init_policy or cfg.ground_actor_anchor):
            raise ValueError("policy_anchor requires an initial policy and a zero ground_actor_anchor")
        if cfg.rise_teacher and not (cfg.gait_dir and cfg.policy_anchor):
            raise ValueError("rise_teacher requires the four-skill task and a positive policy_anchor")
        if not 0. <= cfg.rise_teacher_share <= 1. or (cfg.rise_teacher_share != 1. and not cfg.rise_teacher):
            raise ValueError("rise_teacher_share must be in [0, 1] and requires a rise teacher when changed")
        if not math.isfinite(cfg.rise_teacher_after_s) or cfg.rise_teacher_after_s < 0. or (cfg.rise_teacher_after_s and not cfg.rise_teacher):
            raise ValueError("rise_teacher_after_s must be finite and nonnegative and requires a rise teacher")
        if cfg.gait_teacher and not cfg.rise_teacher:
            raise ValueError("gait_teacher requires rise_teacher for floor samples")
        if not 0. <= cfg.walk_teacher_pose_gain <= 1. or (cfg.walk_teacher_pose_gain and not cfg.gait_teacher):
            raise ValueError("walk_teacher_pose_gain must be in [0, 1] and needs a gait teacher when enabled")
        if cfg.gait_dir and not cfg.standing_amp:
            raise ValueError("Commanded gait requires standing AMP for the zero-speed command")
        if not 0. <= cfg.floor_reset_fraction <= 1.:
            raise ValueError("floor_reset_fraction must be in [0, 1]")
        if not math.isfinite(cfg.gait_speed_weight) or cfg.gait_speed_weight < 0.:
            raise ValueError("gait_speed_weight must be finite and nonnegative")

    @property
    def action_anchor_strength(self):
        return self.c.policy_anchor or self.c.ground_actor_anchor

    def action_anchor_weights(self, state, transition):
        # Use the state observed when the action was chosen. A landing or reset
        # after that action must not change which teacher target PPO receives.
        height = state.data.qpos[:, 2]
        if self.c.policy_anchor and not self.c.rise_teacher:
            return jnp.ones_like(height)
        weight = jnp.clip((zenbot.UP_HEIGHT - height) / (zenbot.UP_HEIGHT - zenbot.FULL_FORCE_HEIGHT), 0., 1.)
        if self.c.rise_teacher_after_s:
            weight *= jnp.clip(state.t * env.DT - self.c.rise_teacher_after_s, 0., 1.)
        if self.condition_dim == 4:
            weight = jnp.where(state.gait_ready, 0., weight)
            if self.c.gait_teacher:
                return jnp.stack((weight, state.gait_ready.astype(weight.dtype)), axis=-1)
        return weight

    def action_anchor_examples(self, obs, weights, actor_obs=None):
        initial, weights = super().action_anchor_examples(obs, weights, actor_obs)
        if not self.c.rise_teacher:
            return initial, weights
        from .inference import policy_output
        frames = obs.reshape((len(obs), zenbot.HIST, -1))
        if self.c.gait_teacher:
            from .zenbot_command_actor import gait_teacher_observation
            floor_share, ready = weights[:, 0], weights[:, 1]
            gait_frames = frames
            if self.c.directional_gait:
                from .zenbot_motion_train_obs import teacher_yaw
                gait_frames = frames.at[..., -1].set(teacher_yaw(frames[..., -1], frames[..., 2]))
            gait = policy_output(*self.locomotion_teacher, gait_teacher_observation(gait_frames))
            if self.c.walk_teacher_pose_gain:
                # Correct the teacher's label from the measured pose. All rollout actions still
                # come directly from the trainable network and use the regular free PD targets.
                q = frames[:, -1, 9:9+len(zenbot.DOF)]
                error = jnp.clip(q, *self.walk_angle_range) - q
                walk_label = initial + self.c.walk_teacher_pose_gain * error / zenbot.ACTION_SCALE
                command = frames[:, -1, -3:]
                forward_walk = (command[:, 0] > .05) & (command[:, 0] < 1.75) & jnp.all(jnp.abs(command[:, 1:]) <= .05, axis=-1)
                gait = jnp.where(forward_walk[:, None], walk_label, gait)
            initial = jnp.where(ready[:, None] > 0., gait, initial)
            weights = floor_share
        floor = policy_output(*self.floor_teacher, frames[..., :-3].reshape((len(obs), -1)))
        return initial + self.c.rise_teacher_share * weights[:, None] * (floor - initial), jnp.ones_like(weights)

    def __init__(self, cfg):
        self.validate(cfg)
        if cfg.standing_amp not in (0, 1) or (cfg.standing_amp and not cfg.support_weight > 0.):
            raise ValueError("standing_amp must be 0 or 1 and requires physical support measurements when enabled")
        if cfg.default_stand not in (0, 1):
            raise ValueError("default_stand must be 0 or 1")
        self.c = cfg
        if cfg.rise_teacher:
            from .inference import load
            def teacher(path, width):
                actor, norm, ode = load(path)
                if not isinstance(actor, list) or ode is not None or actor[0]["w"].shape[0] != width or actor[-1]["b"].shape != (23,):
                    raise ValueError(f"Expected a dense {width}-input, 23-output Zenbot teacher in {path}")
                return actor, norm
            self.floor_teacher = teacher(cfg.rise_teacher, 480)
            if cfg.gait_teacher:
                self.locomotion_teacher = teacher(cfg.gait_teacher, 390)
        self.condition_dim = 4 if cfg.gait_dir else int(cfg.standing_amp)
        self.disc_batch = cfg.disc_batch or cfg.num_envs * cfg.rollout // cfg.minibatches
        self.sim = Sim(cfg.num_envs, robot="zenbot23_getup", floor_dir=cfg.floor_dir, walk_dir=cfg.gait_dir or None,
                       zenbot_getup_stand=True, zenbot_support_weight=cfg.support_weight,
                       zenbot_default_stand=bool(not cfg.standing_amp and cfg.default_stand and cfg.support_weight),
                       zenbot_standing_amp=bool(cfg.standing_amp),
                       zenbot_command_gait=bool(cfg.gait_dir), zenbot_directional_gait=cfg.directional_gait,
                       zenbot_motion_train_obs=cfg.motion_train_obs,
                       reset_floor=cfg.floor_reset_fraction if cfg.gait_dir else 1., prob_shove=0., push=False,
                       assist_newtons=0., random_start=True, **self.simulation_options(cfg))
        if cfg.floor_reset_weights:
            counts = [int(value) for value in cfg.floor_reset_weights.split(",")]
            if len(counts) != len(self.sim.floor["z"]) or min(counts) < 1:
                raise ValueError("Supply one positive integer in floor_reset_weights per floor opening")
            indices = jnp.repeat(jnp.arange(len(counts)), jnp.asarray(counts), total_repeat_length=sum(counts))
            self.sim.floor = jax.tree.map(lambda frames: frames[indices], self.sim.floor)
        if cfg.support_weight:
            self.extra_rollout_stats += ("support_hold_score",)
        self.demos = amp.Demos(cfg.floor_dir, self.sim, self.amp_span, per_clip=True,
                               frame_fn=self.sim.amp_frame)
        if cfg.standing_amp:
            self.stand_demos = amp.Demos(cfg.stand_dir, self.sim, self.amp_span,
                                         per_clip=True, frame_fn=self.sim.amp_frame)
            self.motion_metric_names = ("rise_amp_contribution", "stand_amp_contribution")
            if cfg.gait_dir and not getattr(cfg, "source_gait", False):
                from .zenbot_gait_demos import StraightGaitDemos
                self.walk_demos = StraightGaitDemos(cfg.gait_dir, self.sim, self.amp_span, gait="walk",
                                                   directional=bool(cfg.directional_gait))
                self.run_demos = StraightGaitDemos(cfg.gait_dir, self.sim, self.amp_span, gait="run")
                if cfg.walk_teacher_pose_gain:
                    self.walk_angle_range = self.walk_demos.joint_angle_range(self.sim.J)
                self.motion_metric_names += ("walk_amp_contribution", "run_amp_contribution", "gait_speed_contribution")
            if cfg.stand_reference_weight:
                self.standing_target = jnp.median(self.stand_demos.frames, axis=0)
                self.motion_metric_names += ("stand_reference_contribution",)

    def initialize_policy(self, template, path):
        if self.condition_dim == 4 and not self.c.motion_train_obs:
            from .zenbot_command_actor import initialize_command_actor
            return initialize_command_actor(template, path)
        return super().initialize_policy(template, path)

    def critic_input(self, state):
        velocity = state.data.qvel[:, :3]
        if getattr(self.c, "source_gait", False):
            from .rot import mv3, quat_to_mat, tr3
            # Commands and rewards use the robot frame even when a reset randomizes global heading.
            velocity = mv3(tr3(quat_to_mat(state.data.qpos[:, 3:7])), velocity)
        return jnp.concatenate((state.obs, state.data.qpos[:, 2:3], velocity), axis=-1)

    def actor_mean(self, actor, obs):
        if self.c.motion_train_obs:
            from .zenbot_motion_train_obs import motion_train_actor_mean
            return motion_train_actor_mean(actor, self.obs_norm, obs)
        return super().actor_mean(actor, obs)

    def motion_features(self, window):
        # Estimate velocity at the same sampling rate in both motion sources.
        n = self.sim.J
        dq = jnp.diff(window[..., 9:9+n], axis=-2) / (env.DT * self.amp_stride)
        return window.at[..., 9+n:9+2*n].set(jnp.concatenate((dq[..., :1, :], dq), axis=-2))

    @staticmethod
    def conditioned(motion, label):
        if label.ndim == 1:
            label = label[:, None]
        channel = jnp.broadcast_to(label.astype(motion.dtype)[:, None, :], (*motion.shape[:-1], label.shape[-1]))
        return jnp.concatenate((motion, channel), axis=-1)

    def rollout_motion(self, disc, motion, transition):
        if not self.condition_dim:
            style = env.DT * amp.recovery_style(disc, motion, scale=10., leggedlab=True)
            blend = transition.stats["stand_blend"] if "stand_blend" in transition.stats else transition.stats["stand_quality"]
            return (1. - blend) * style, motion, ()

        hold = transition.stats["stand_blend"]
        if self.condition_dim == 4:
            command = transition.stats["gait_command"]
            weights = gait_skill_weights(hold, command, transition.stats["gait_ready"])
            labels = jnp.broadcast_to(jnp.eye(4, dtype=motion.dtype)[:, None, :], (4, motion.shape[0], 4))
            replay = self.conditioned(motion, weights)
        else:
            weights = jnp.stack((1. - hold, hold), axis=-1)
            labels = jnp.stack((jnp.zeros_like(hold), jnp.ones_like(hold)))
            replay = self.conditioned(motion, hold)
        windows = jax.vmap(self.conditioned, in_axes=(None, 0))(motion, labels)
        windows = windows.reshape((-1, *windows.shape[2:]))
        scores = env.DT * amp.recovery_style(disc, windows, scale=10., leggedlab=True)
        parts = weights * scores.reshape(len(labels), len(motion)).T
        reward = parts.sum(axis=-1)
        metrics = tuple(parts.mean(axis=0))
        if self.condition_dim == 4:
            # One velocity target. A circle or a diagonal step can have the correct forward speed.
            velocity_error = transition.stats["gait_velocity_error"] / jnp.array([.5, .5, .6])
            speed_reward = (env.DT * self.c.gait_speed_weight * weights[:, 2:].sum(axis=-1)
                            / (1. + jnp.square(velocity_error).sum(axis=-1)))
            reward += speed_reward
            metrics += (speed_reward.mean(),)
        if self.standing_target is not None:
            match = standing_pose_match(motion[:, -1], self.standing_target)
            bilateral = jnp.clip(2. * transition.stats["min_foot_load_bw"], 0., 1.)
            reference = env.DT * self.c.stand_reference_weight * weights[:, 1] * bilateral * match
            reward += reference
            metrics += (reference.mean(),)
        return reward, replay, metrics

    def motion_reward(self, disc, motion, transition):
        return self.rollout_motion(disc, motion, transition)[0]

    def discriminator_examples(self, key, policy):
        if not self.condition_dim:
            return super().discriminator_examples(key, policy)
        if self.condition_dim == 4:
            # Replay stores the four blend weights. Give the sampled expert and robot
            # the same discrete condition so the label alone cannot distinguish them.
            k_label, *keys = jax.random.split(key, 5)
            weights = policy[:, -1, -4:]
            skill = jax.random.categorical(k_label, jnp.log(weights), axis=-1)
            banks = (self.demos, self.stand_demos, self.walk_demos, self.run_demos)
            experts = jnp.stack([bank.sample(k, len(policy))[:, ::self.amp_stride] for bank, k in zip(banks, keys)], axis=1)
            source = experts[jnp.arange(len(policy)), skill]
            label = jax.nn.one_hot(skill, 4)
            return self.conditioned(self.motion_features(source), label), self.conditioned(policy[..., :-4], label)
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
        flags = ["--" + field.name]
        if field.name in ("directional_gait", "motion_train_obs"):
            flags.append("--" + field.name.replace("_", "-"))
        parser.add_argument(*flags, type=type(field.default), default=field.default)
    cfg = Settings(**vars(parser.parse_args()))
    metadata = dict(task="zenbot23_getup_stand", robot="zenbot23_getup", learner="on_policy_ppo_amp",
                    assist_newtons=0., reference_weight=0., ode=0)
    if cfg.standing_amp:
        metadata.update(default_stand=0, policies=1, discriminators=1,
                        amp_weight="(1 - stand_blend) * rise_amp + stand_blend * standing_amp",
                        standing_reference="Median channels of the standing AMP bank; scaled by stand_blend and measured bilateral loading",
                        expert_directories={"rise": cfg.floor_dir, "standing": cfg.stand_dir})
        if cfg.gait_dir:
            metadata.update(task="zenbot23_command_getup_stand_walk_run", skills=["rise", "stand", "walk", "run"],
                            amp_weight="four skill-conditioned AMP scores weighted by support, gait readiness and commanded velocity")
            metadata["expert_directories"]["gait"] = cfg.gait_dir
            if cfg.directional_gait:
                metadata["command_axes"] = ["forward_mps", "left_mps", "yaw_radps"]
                metadata["skills"] += ["backward", "strafe", "turn_in_place"]
            if cfg.motion_train_obs:
                metadata["actor_observation"] = dict(source="Motion Train ZENBOT-AMP iteration 14300",
                                                     width=234, history=3, history_order="term",
                                                     actor_compute_dtype="float32",
                                                     imu_delay_ms=[[.9, 2, 10], [.1, 10, 35]],
                                                     uniform_noise=dict(gyro=.28, gravity=.042, joint_pos=.03, joint_vel=1.5))
    else:
        metadata["amp_weight"] = "1 - stand_blend" if cfg.default_stand and cfg.support_weight else "1 - stand_quality"
    train(cfg, RiseStandAMP, metadata)


if __name__ == "__main__":
    main()
