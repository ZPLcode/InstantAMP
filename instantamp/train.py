"""Train a policy to walk, run, and recover from falls on G1 or Zenbot.

Each iteration collects physics rollouts, fits locomotion and recovery motion
priors, and trains a world model. The policy learns from returns in imagined
rollouts and can also use PPO gradients from real simulator trajectories.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# JAX reserves 30 % of the GPU (its default is 75 %); the rest is left to MuJoCo Warp, which allocates its own buffers at run time. An
# XLA_PYTHON_CLIENT_MEM_FRACTION set by the caller wins.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import jax  # noqa: E402
import jax.numpy as jnp
import numpy as np

from . import (
    amp,
    buffer as rb,
    critic,
    env,
    nets,
    optim,
    zenbot,
    zenbot_gait_rewards as gait_rewards,
    zenbot_models,
    zenbot_source_gait as source_gait,
)
from .sim import Sim, motion_dirs, unflatten
from .training_sim import simulation_options
from .zenbot_gait_demos import reference_command_index
from .zenbot_motion_train_obs import (
    joint_layout,
    mirror_action,
    mirror_observation,
    motion_train_actor_mean,
)

# error floors of the world-model loss for the state's position, rotation, velocity and angular velocity (then q: 3e-3, qd: 5e-2)
ERR_FLOOR = ((3, 3e-3), (3, 5e-3), (3, 2e-2), (3, 5e-2))


def widths(s: str):
    """Layer widths from a comma-separated string: "512,512,512" gives (512, 512, 512)."""
    return tuple(int(x) for x in s.split(","))


def action_log_prob(action, mean, std):
    """Joint log probability under a diagonal Gaussian in policy action units."""
    return -0.5 * (jnp.square((action - mean) / std) + 2 * jnp.log(std) + np.log(2 * np.pi)).sum(-1)


def set_jax(cache_dir: str = ""):
    """Global JAX settings: bf16 matrix multiplications, and (if cache_dir is set) a compilation cache on disk that keeps every program
    and XLA's autotuning results, so that later runs start faster."""
    jax.config.update("jax_default_matmul_precision", "bfloat16")
    if cache_dir:
        jax.config.update("jax_compilation_cache_dir", os.path.expanduser(cache_dir))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


@dataclasses.dataclass
class Config:
    """G1 training defaults. Each field also supplies a command-line option."""

    # environment (sim.py)
    num_envs: int = 4096
    nconmax: int = 64  # contacts per world (self-collisions included)
    njmax: int = 640  # constraint rows per world
    walk_dir: str = ""  # walking / running clips: reset frames of the upright starts
    floor_dir: str = ""  # get-up clips: their not-up frames are the floor starts
    reset_floor: float = 0.2  # share of resets that start down, in a not-up frame of a get-up clip
    prob_shove: float = 0.2  # share of the upright resets that get a kick of the root and joint velocities
    push: int = 1  # the root's world xy velocity overwritten with U[-1, 1] m/s every 5 to 10 s
    random_start: int = 1  # initial episode counters uniform in [0, 1000), so that episodes end at different times
    term_tau: float = 0.02  # width (m) of the sigmoid that gives imagined contact terminations their gradient
    # motion priors (amp.py, getup_amp.py)
    amp_w: float = 2.0  # > 0: AMP on (and the style weight when amp_lerp = 0); 0: AMP off
    amp_lerp: float = 0.5  # > 0: r = (1 - amp_lerp) r_task + amp_lerp dt mix; 0: r = r_task + amp_w dt mix
    #                        (mix = gate_loco style_loco + gate_getup style_getup)
    amp_gate: str = "track_shift"  # gate of the locomotion style while up: track_shift (moving x (amp_track_lo + track_lin), in
    #                                [0.25, 1.25]) | moving (a moving command) | track (moving x track_lin) | none
    amp_track_lo: float = 0.25  # the floor of the track_shift gate
    amp_dir: str = ""  # demos of the locomotion prior
    amp_getup: int = 1  # a second discriminator for the get-up style while the robot has not been up
    amp_getup_dir: str = ""  # its demos: windows of not-up frames of the fallAndGetUp takes
    amp_getup_full: int = 0  # 1: the get-up style at full scale from the first iteration; 0: it shares amp_delay / amp_ramp
    amp_frames: int = 10  # frames per discriminator window (0.2 s)
    amp_wrists: int = 1  # the wrist angles and velocities in the discriminator's frame
    amp_style_scale: float = 5.0  # style = (1 + tanh((d - mean robot score) / scale)) / 2; 0: midpoint / half-gap of the two mean scores
    amp_disc_steps: int = 2  # discriminator steps per iteration (each prior)
    amp_batch: int = 4096
    amp_recent: int = 100  # robot windows from the newest this many steps of the buffer
    amp_lr: float = 1e-4
    amp_delay: int = 500  # iterations before the style reward starts
    amp_ramp: int = 4000  # iterations over which it ramps up to full scale
    # critic (critic.py): bootstraps the imagined return at the horizon; 0: pure BPTT
    critic: int = 1
    critic_lr: float = 3e-4
    critic_epochs: int = 5
    critic_mb: int = 4
    lam: float = 0.95
    critic_hidden: str = "512,512,256"
    # schedule
    max_iterations: int = 0  # zero follows the wall-clock budget
    minutes: float = 30.0  # training time (compilation not included)
    env_steps: int = 1  # simulator steps of all envs per iteration
    buffer_steps: int = 200  # replay buffer length (steps of all envs)
    warmup_steps: int = 32
    wm_only_iters: int = 200  # iterations that fit only the world model
    # world model
    wm_hidden: str = "512,512,512"
    wm_window: int = 8  # open-loop window of the world-model loss
    wm_batches: int = 3
    wm_batch: int = 2048
    wm_lr: float = 4e-4
    wm_muon: int = 1  # Muon on the world model's hidden matrices
    wm_gate: float = 2.0  # skip the policy update while the world-model loss is above this times its running mean
    wm_eval: int = 1  # log open-loop height errors and termination rates of the world model
    # policy
    horizon: int = 48  # imagined steps per rollout (~1 s)
    pi_batch: int = 2048  # imagined rollouts per update
    pi_updates: int = 2  # policy updates per iteration
    pi_lr: float = 3e-4
    pi_clip: float = 1.0  # gradient-norm clip
    pi_gn_gate: float = 10.0  # skip an update whose gradient norm is above this times its running mean
    pi_ema: float = 0.995  # the averaged policy that checkpoints hold
    ode: int = 1  # ODE actor: the network outputs the change of the action (ODE Policy)
    ode_dmax: float = 3.24  # bound of the action change per step (0: none)
    ode_amax: float = 20.0  # bound of the action (0: none)
    gamma: float = 0.99
    action_noise: float = 0.3  # exploration noise of the collector's noisy envs
    det_share: float = 0.5  # share of collector envs without action noise
    cmd_fresh: float = 0.5  # share of imagined rollouts that start with a freshly drawn command
    imag_down: float = 0.0  # share of imagined rollouts that start in a not-up state (0: uniform starts)
    obs_clip: float = 10.0  # clip of the normalized observation
    # hybrid PPO: a share of the policy gradient from real collector steps
    ppo_frac: float = 0.1  # share p: p * num_envs noisy collector envs give PPO data, (1 - p) * pi_batch imagined rollouts per update
    ppo_len: int = 8  # the newest steps of those envs that make a PPO batch (re-read at every update)
    ppo_clip: float = 0.2
    ppo_w: float = 1.0  # PPO gradient scaled to ppo_w * |BPTT gradient|; < 0: p / (1 - p)
    # runtime
    imag_unroll: int = 4
    imag_remat: str = "none"  # none | dots | full: rematerialization of the imagined steps (less memory, slower)
    mm_bf16: str = "imag"  # bf16 matmul outputs: imag | all | none
    log_every: int = 50  # iterations per log line
    ckpt_minutes: float = 2.0
    ckpt_iterations: int = 0  # positive: checkpoint on this iteration interval instead of elapsed minutes
    full_ckpt_keep: int = 0  # positive: retain this many complete learner checkpoints; keep all policy exports
    seed: int = 0
    log_dir: str = ""
    jax_cache: str = "~/.cache/instantamp/jax"
    xla_flags: str = "--xla_gpu_enable_triton_gemm=false"

    robot: str = "g1"  # select the articulation and motion banks

    def __post_init__(self):
        if self.ckpt_iterations < 0:
            raise ValueError("ckpt_iterations must be nonnegative")
        if self.full_ckpt_keep < 0:
            raise ValueError("full_ckpt_keep must be nonnegative")
        walk, floor = motion_dirs(self.robot)
        self.walk_dir = self.walk_dir or walk
        self.amp_dir = self.amp_dir or walk
        self.floor_dir = self.floor_dir or floor
        use_floor_bank = self.robot in zenbot_models.MIXED_TASKS
        self.amp_getup_dir = self.amp_getup_dir or (self.floor_dir if use_floor_bank else floor)
        self.log_dir = self.log_dir or ("runs/instantamp" if self.robot == "g1" else f"runs/{self.robot}")
        full_path = getattr(self, "init_full", "")
        if full_path and Path(full_path).resolve().parent == (Path(self.log_dir) / "ckpt/full").resolve():
            raise ValueError("Full-checkpoint continuation requires a new log directory")


@dataclasses.dataclass
class ZenbotMotionTrainWMConfig(Config):
    """Shared world-model configuration for Zenbot locomotion and recovery."""
    robot: str = zenbot.MIXED_TASK
    motion_train_wm: int = 1
    motion_train_randomization: int = 0  # 1: articulation-specific Motion Train dynamics and terrain
    external_pushes: int = 0  # enable shoulder/hip force pulses or the legacy velocity perturbations
    reset_floor: float = .35
    zenbot_episode_steps: int = 400
    amp_delay: int = 100
    amp_ramp: int = 600
    pi_lr: float = .00012
    init_full: str = ""
    init_policy: str = ""  # optional trainable single actor with this observation and action layout
    recovery_amp_seconds: float = 4.0  # include the transition to balance after any fall
    recovery_amp_stride: int = 3  # ten unpaired poses spanning 0.54 s
    recovery_amp_scale: float = 6.  # legacy recovery scale; joint LS priors share gait_ls_scale
    getup_amp_scale: float = 0.  # positive: override only the recovery prior; zero keeps the existing shared scale
    gait_amp_stride: int = 1
    gait_ls_scale: float = 0.  # positive: least-squares gait AMP on matched finite-difference features
    recovery_amp_loss: str = "least_squares"
    support_rsi: float = 0.  # fraction of floor starts drawn from intermediate non-upright AMP states
    fall_push_speed: float = 0.  # optional gait perturbation magnitude in m/s
    wm_floor_amp: int = 0  # cold floor openings, with state-based blending of get-up and gait AMP
    wm_fixed_floor_envs: int = 0  # reserve reset_floor of the collectors for fallen starts at every reset
    wm_tracking_scale: float = 1.  # multiply the planar and yaw tracking weights in the joint task
    wm_range_commands: int = 0  # continuous Motion Train velocity ranges for the joint task
    wm_global_gait: int = 0  # Motion Train's global gait AMP and standing locomotion resets
    wm_gait_frames: int = source_gait.AMP_STEPS  # consecutive 50 Hz frames for global gait AMP
    wm_initial_gait_stage: int = 0  # global gait speed caps: 1, 2, 3 m/s for stages 0, 1, 2
    wm_side_share: float = 0.  # probability per direction of a pure lateral command
    wm_posture_and_tracking: int = 0  # pelvis posture rewards and separate planar/yaw tracking diagnostics
    wm_gait_rewards: int = 0  # Motion Train foot and joint terms for the joint floor/gait task
    floor_stop_command_share: float = 0.  # fraction of fallen starts requesting a quiet stance for the episode
    gait_source_commands: int = 0  # practice the recorded and nominal directional speeds
    gait_mirror_weight: float = 0.  # mirror consistency weighted by the current pose's gait suitability
    imag_gait_mirror_weight: float = 0.  # action consistency on imagined locomotion frames
    wm_lift_gate: float = 0.  # pelvis height where positive lifting progress begins, in metres
    wm_lift_weight: float = .5  # weight of bounded lifting progress above wm_lift_gate
    wm_tempo_negatives: float = 0.  # fraction of recovery discriminator negatives made by mistiming moving reference segments
    wm_reset_imag_share: float = 0.  # portion of imagination starts in the first real WM window after reset
    ode: int = 0
    action_rate_l2_weight: float = -.015
    amp_gate: str = "none"
    max_iterations: int = 10000
    minutes: float = 0.
    seed: int = 42

    @property
    def command_gait_amp(self):
        return bool(self.wm_floor_amp and self.gait_ls_scale and not self.wm_global_gait)

    def __post_init__(self):
        super().__post_init__()
        if self.motion_train_randomization not in (0, 1):
            raise ValueError("motion_train_randomization must be 0 or 1")
        if self.external_pushes not in (0, 1):
            raise ValueError("external_pushes must be 0 or 1")
        if self.motion_train_randomization and self.robot not in zenbot_models.MIXED_TASKS:
            raise ValueError("Motion Train randomization requires a Zenbot joint task")
        if self.init_full:
            source_config = Path(self.init_full).parent.parent.parent / "config.json"
            if source_config.is_file():
                saved_randomization = json.loads(source_config.read_text()).get("motion_train_randomization", 0)
                if bool(saved_randomization) != bool(self.motion_train_randomization):
                    raise ValueError(
                        "Full learner checkpoints require the same motion_train_randomization context. "
                        "Use --init_policy with the companion policy to train with the new dynamics, "
                        "or keep the saved randomization setting for an exact continuation."
                    )
        if self.amp_w <= 0 or not self.amp_getup:
            raise ValueError("The mixed recovery AMP recipe requires both motion discriminators")
        if self.amp_frames < 2:
            raise ValueError("Recovery AMP needs at least two frames for joint velocities")
        if self.recovery_amp_stride < 1 or not np.isfinite(self.recovery_amp_scale) or self.recovery_amp_scale <= 0:
            raise ValueError("Use a positive recovery discriminator stride and scale")
        if not np.isfinite(self.getup_amp_scale) or self.getup_amp_scale < 0:
            raise ValueError("getup_amp_scale must be finite and nonnegative")
        if not np.isfinite(self.wm_lift_weight) or self.wm_lift_weight < 0:
            raise ValueError("wm_lift_weight must be finite and nonnegative")
        if self.recovery_amp_loss not in ("least_squares", "wasserstein"):
            raise ValueError("Unknown recovery AMP discriminator loss")
        if (not np.isfinite(self.gait_ls_scale) or self.gait_ls_scale < 0
                or not 1 <= self.gait_amp_stride <= self.recovery_amp_stride
                or self.gait_amp_stride != 1 and not self.gait_ls_scale
                or self.gait_ls_scale and not (
                    self.command_gait_amp or getattr(self, "wm_global_gait", False))):
            raise ValueError("Gait AMP needs a compatible stride and gait discriminator")
        span = (self.amp_frames - 1) * self.recovery_amp_stride + 1
        if not np.isfinite(self.recovery_amp_seconds) or self.recovery_amp_seconds < span * env.DT:
            raise ValueError("The recovery AMP interval must fit at least one complete window")
        if self.buffer_steps < span + self.ppo_len + 1:
            raise ValueError("The buffer must hold the full recovery AMP window and PPO rollout")
        if self.ode or not self.motion_train_wm:
            raise ValueError("The Motion Train WM profile needs direct actions and a learned world model")
        joint_task = bool(self.wm_floor_amp)
        for name in ("wm_range_commands", "wm_posture_and_tracking", "wm_gait_rewards", "wm_fixed_floor_envs", "wm_global_gait"):
            enabled = getattr(self, name)
            if enabled not in (0, 1) or (enabled and not joint_task):
                raise ValueError(f"{name} must be 0 or 1 and requires joint floor/locomotion training when enabled")
        if not np.isfinite(self.wm_tracking_scale) or self.wm_tracking_scale < 0 or (self.wm_tracking_scale != 1. and not joint_task):
            raise ValueError("A custom wm_tracking_scale must be finite, nonnegative and used with joint floor/locomotion training")
        if self.gait_source_commands and not self.command_gait_amp:
            raise ValueError("Source-speed commands need command-conditioned locomotion AMP")
        if self.wm_global_gait:
            if not (self.wm_fixed_floor_envs and self.wm_range_commands and self.wm_gait_rewards
                    and self.gait_ls_scale > 0 and self.gait_amp_stride == 1 and 0 < self.amp_lerp <= 1):
                raise ValueError("Global gait AMP requires fixed gait worlds, ranged commands, gait rewards, LS style and unit stride")
            if not 1 <= self.wm_gait_frames <= span:
                raise ValueError("wm_gait_frames must fit in the available rollout history")
        if not 0 <= self.wm_side_share <= .375 or (self.wm_side_share and not self.wm_global_gait):
            raise ValueError("Pure lateral probability per side must fit within the global gait translation share")
        if not 0 <= self.ppo_frac < 1:
            raise ValueError("ppo_frac must be in [0, 1) so imagination contributes policy updates")
        if self.ppo_frac and (not self.critic or not self.action_noise > 0 or
                              round(self.ppo_frac * self.num_envs) < 1 or round((1-self.ppo_frac)*self.pi_batch) < 1):
            raise ValueError("Hybrid PPO requires a critic, exploration noise, and real and imagined samples")
        if self.wm_floor_amp and not (self.ppo_frac and self.amp_w and self.amp_getup):
            raise ValueError("Floor-opening AMP needs hybrid training and two motion priors")
        if not 0. <= self.wm_lift_gate < zenbot.UP_HEIGHT or (self.wm_lift_gate and not self.wm_floor_amp):
            raise ValueError("A late lift gate must be below upright pelvis height and requires floor AMP")
        if not 0. <= self.wm_tempo_negatives <= .5 or (self.wm_tempo_negatives and not self.wm_floor_amp):
            raise ValueError("Tempo negatives need floor AMP and a share between zero and one half")
        if not 0. <= self.wm_reset_imag_share <= 1. or (self.wm_reset_imag_share and
                (not self.wm_floor_amp or self.imag_down)):
            raise ValueError("Reset-priority imagination requires floor AMP and imag_down=0")
        if (not np.isfinite(self.gait_mirror_weight) or self.gait_mirror_weight < 0 or
                self.gait_mirror_weight and not ((self.command_gait_amp or self.wm_global_gait) and self.ppo_frac > 0)):
            raise ValueError("Upright mirror consistency requires joint floor and gait AMP with real PPO samples")
        if (not np.isfinite(self.imag_gait_mirror_weight) or self.imag_gait_mirror_weight < 0 or
                self.imag_gait_mirror_weight and not self.wm_floor_amp):
            raise ValueError("imag_gait_mirror_weight must be finite and nonnegative and requires joint floor/gait training when enabled")
        if self.obs_clip != 10.:
            raise ValueError("The Motion Train deployment normalizes observations with a clip of 10")


@dataclasses.dataclass
class ZenbotJointConfig(ZenbotMotionTrainWMConfig):
    """One command-conditioned actor for locomotion and floor recovery, with two AMP priors."""
    motion_train_randomization: int = 1
    num_envs: int = 2048
    max_iterations: int = 20_000
    ckpt_iterations: int = 500
    reset_floor: float = .4
    push: int = 0
    prob_shove: float = 0.
    env_steps: int = 16
    ppo_len: int = 16
    warmup_steps: int = 64
    buffer_steps: int = 100
    wm_only_iters: int = 150
    wm_hidden: str = "1024,1024,1024"
    wm_batch: int = 512
    amp_batch: int = 512
    pi_batch: int = 384
    pi_updates: int = 3
    horizon: int = 28
    pi_lr: float = .00015
    action_noise: float = 1.  # initial per-joint std, or fixed std when learning is disabled
    learn_action_std: int = 1
    action_entropy_coef: float = .01
    amp_delay: int = 0
    amp_ramp: int = 0
    amp_lerp: float = .7
    wm_floor_amp: int = 1
    wm_fixed_floor_envs: int = 1
    wm_range_commands: int = 1
    wm_global_gait: int = 1
    wm_gait_rewards: int = 1
    wm_posture_and_tracking: int = 1
    wm_tracking_scale: float = 1.
    wm_side_share: float = .1
    gait_ls_scale: float = 1.5
    gait_mirror_weight: float = .2
    action_rate_l2_weight: float = -.05
    recovery_amp_scale: float = 10.
    support_rsi: float = .35
    floor_stop_command_share: float = .5
    wm_lift_gate: float = .44
    wm_reset_imag_share: float = .2
    wm_tempo_negatives: float = .25
    recovery_teacher: str = ""  # optional train-time expert for physical fallen states

    def __post_init__(self):
        if not 0 < self.reset_floor < 1:
            raise ValueError("Joint training requires both floor and locomotion starts")
        if self.learn_action_std and (not np.isfinite(self.action_noise) or self.action_noise <= 0
                                      or not np.isfinite(self.action_entropy_coef) or self.action_entropy_coef < 0):
            raise ValueError("Learned exploration requires a positive finite initial std and nonnegative finite entropy weight")
        available_history = 1 + (self.amp_frames - 1) * self.recovery_amp_stride
        if self.recovery_teacher and available_history < zenbot.HIST:
            raise ValueError("A recovery teacher needs at least five AMP history frames")
        if self.recovery_teacher and not (self.wm_floor_amp and self.ppo_frac):
            raise ValueError("A recovery teacher requires joint floor AMP and real PPO samples")
        super().__post_init__()


@dataclasses.dataclass
class Zenbot23JointConfig(ZenbotJointConfig):
    """Shared joint training recipe with the 23-DoF articulation selected."""

    robot: str = zenbot_models.ZENBOT23.mixed_task
    imag_gait_mirror_weight: float = .2


class InstantAMP:
    @property
    def actor_dim(self):
        return len(joint_layout(self.sim.J).history_index) if getattr(self.cfg, "motion_train_wm", False) else self.sim.obs_dim

    @property
    def mixed_amp(self):
        return bool(getattr(self.cfg, "recovery_amp_seconds", 0.))

    @property
    def recovery_stride(self):
        return self.cfg.recovery_amp_stride if self.mixed_amp else 1

    @property
    def amp_span(self):
        return 1 + (self.cfg.amp_frames - 1) * self.recovery_stride

    @property
    def gait_span(self):
        if getattr(self.cfg, "wm_global_gait", False):
            return self.cfg.wm_gait_frames
        return 1 + (self.cfg.amp_frames - 1) * getattr(self.cfg, "gait_amp_stride", 1)

    def __init__(self, cfg: Config):
        self.cfg = c = cfg
        set_jax(c.jax_cache)
        stand_clip = Path(c.amp_dir) / "stand_4s_120f.npz" if c.amp_w > 0 and getattr(c, "motion_train_wm", False) else None
        self.sim = Sim(**simulation_options(vars(c)))
        self.recovery_teacher = self.gait_teacher = None
        if getattr(c, "recovery_teacher", ""):
            from .inference import load
            teacher, norm, ode = load(c.recovery_teacher)
            joints = self.sim.J
            frame_size = 9 + 3 * joints + 3 * len(zenbot.KEY_BODIES)
            width = self.sim.history * frame_size
            if (ode is not None or not isinstance(teacher, list)
                    or teacher[0]["w"].shape[0] != width
                    or teacher[-1]["b"].shape != (joints,)):
                raise ValueError(
                    f"Expected a {width}-input, {joints}-output direct get-up teacher"
                )
            self.recovery_teacher = teacher, norm
            if c.init_policy:
                gait_actor, gait_norm, _ = load(c.init_policy)
                self.gait_teacher = gait_actor, gait_norm
        self.J = self.JS = self.sim.J  # joints (actions, targets) and the world model's q, qd
        self.ode = bool(c.ode)
        self.ode_dmax = c.ode_dmax if c.ode_dmax > 0 else None
        self.ode_amax = c.ode_amax if c.ode_amax > 0 else None
        self.amp = c.amp_w > 0
        if self.amp:
            if getattr(c, "wm_global_gait", False):
                from .zenbot_gait_demos import GlobalGaitDemos
                self.demos = GlobalGaitDemos(c.amp_dir, self.sim, self.gait_span)
            elif getattr(c, "command_gait_amp", False):
                from .zenbot_gait_demos import CommandGaitDemos
                self.demos = CommandGaitDemos(c.amp_dir, self.sim, self.gait_span)
            else:
                self.demos = amp.Demos(c.amp_dir, self.sim, self.gait_span,
                                       per_clip=self.sim.mixed)
            assert c.amp_gate in ("track_shift", "moving", "track", "none"), c.amp_gate
            self.getup = bool(c.amp_getup)
            if self.getup:
                from .getup_amp import GetupDemos

                self.getup_demos = (amp.Demos(c.amp_getup_dir, self.sim, self.amp_span, per_clip=True,
                                             extra_clip=None if getattr(c, "wm_floor_amp", False) else stand_clip) if self.sim.mixed else
                                    GetupDemos(c.amp_getup_dir, self.sim, c.amp_frames))
                if self.mixed_amp and c.recovery_amp_loss == "wasserstein":
                    self.getup_demos.prepare_ground_heights()
            self.amp_envs = np.nonzero(~((np.arange(c.num_envs) * c.det_share) % 1.0 >= c.det_share))[0].astype(np.int32)
        self.use_critic = bool(c.critic)
        self.ppo = c.ppo_frac > 0
        self.B_im = int(round((1 - c.ppo_frac) * c.pi_batch)) if self.ppo else c.pi_batch  # imagined rollouts per update
        if self.ppo:
            assert self.use_critic, "the PPO advantages need the critic (--critic 1)"
            noisy = np.nonzero((np.arange(c.num_envs) * c.det_share) % 1.0 >= c.det_share)[0]
            n_ppo = int(round(c.ppo_frac * c.num_envs))
            assert 0 < n_ppo <= len(noisy), "the PPO envs are taken among the noisy collector envs"
            self.ppo_envs = noisy[:n_ppo].astype(np.int32)
            self.ppo_w = c.ppo_w if c.ppo_w >= 0 else c.ppo_frac / (1 - c.ppo_frac)
        self.n_soles = len(self.sim.sole_radius)
        self.err_floor = np.concatenate([np.full((n,), v, np.float32) for n, v in ERR_FLOOR] + [np.full((self.JS,), 3e-3, np.float32), np.full((self.JS,), 5e-2, np.float32)])

    # ------------------------------------------------------------------ state
    def init(self, key):
        c, J = self.cfg, self.J
        ks = jax.random.split(key, 4)
        e = self.sim.init(ks[0])
        action_std = jnp.full((J,), c.action_noise, jnp.float32) if self.ppo and getattr(c, "learn_action_std", False) else None
        zero_action = jnp.zeros((c.num_envs, J))
        rec_t = jax.eval_shape(lambda s: self.record(s, *self.sim.step(s, zero_action), zero_action, zero_action, action_std)[0], e)
        rec_t = rb.flatten({**rec_t, "ok": jax.ShapeDtypeStruct((c.num_envs,), jnp.float32)})
        self.xi_dim = rec_t["xi"].shape[-1]
        self.wm_in = (13 + 2 * self.JS) + self.n_soles + rec_t["target"].shape[-1] + self.xi_dim
        pi = nets.pi_init(ks[1], self.actor_dim, J)
        wm, wm_b = nets.wm_init(ks[2], self.wm_in, 12 + 2 * self.JS, widths(c.wm_hidden))
        self.pi_mask = jax.tree.map(lambda _: False, pi)
        self.wm_mask = {"net": nets.muon_mask(wm["net"]) if c.wm_muon else jax.tree.map(lambda _: False, wm["net"])}
        record_template = {k: jnp.zeros(v.shape, v.dtype) for k, v in rec_t.items()}
        buf = rb.init(c.buffer_steps, record_template, full_precision_obs=bool(getattr(c, "motion_train_wm", False)))
        carry = {"env": e, "buf": buf,
                 "pi": pi, "avg": jax.tree.map(jnp.copy, pi), "wm": wm, "wm_b": wm_b, "obs_norm": nets.rn_init(self.actor_dim),
                 "opt_pi": optim.init(pi, self.pi_mask), "opt_wm": optim.init(wm, self.wm_mask),
                 "key": ks[3], "it": jnp.zeros((), jnp.int32), "env_steps": jnp.zeros((), jnp.float32), "nonfinite": jnp.zeros((), jnp.int32),
                 "wm_ema": jnp.zeros((), jnp.float32), "gn_ema": jnp.zeros((), jnp.float32), "wm_ok": jnp.ones((), bool)}  # fmt: skip
        if action_std is not None:
            carry.update(action_std=action_std, opt_std=optim.init(action_std, False))
        if self.amp:
            gait_dim = self.sim.amp_dim + (3 if getattr(c, "command_gait_amp", False) else 0)
            global_gait = getattr(c, "wm_global_gait", False)
            carry["amp"] = amp.init(jax.random.fold_in(ks[3], 7), self.gait_span if global_gait else c.amp_frames,
                                    frame_dim=source_gait.amp_frame_width(self.J) if global_gait else gait_dim)
            if self.getup:
                key_up = jax.random.fold_in(ks[3], 13)
                carry["amp_up"] = amp.init(key_up, c.amp_frames, frame_dim=self.getup_demos.frames.shape[-1])
        if self.use_critic:
            self.critic_dim = 13 + 2 * self.JS + 3 + J + self.xi_dim + 2
            carry["critic"] = critic.init(jax.random.fold_in(ks[3], 11), self.critic_dim, widths(c.critic_hidden))
        strong = lambda x: jnp.asarray(x, dtype=x.dtype) if isinstance(x, jax.Array) and x.weak_type else x  # noqa: E731
        return jax.tree.map(strong, carry)

    # ------------------------------------------------------------------ actor
    def norm_obs(self, obs_norm, obs):
        if "raw" in obs_norm:
            return obs
        return jnp.clip(nets.rn_apply(obs_norm, obs), -self.cfg.obs_clip, self.cfg.obs_clip)

    def export(self, carry):
        """The live and averaged policies and the normalizer as a checkpoint holds them."""
        policy = {"pi": carry["pi"], "avg": carry["avg"], "obs_norm": carry["obs_norm"]}
        if getattr(self.cfg, "wm_floor_amp", False):
            policy["delay_uses_first_command"] = True
            policy["floor_reward_version"] = 1
        return policy

    def act(self, pi, obs_norm, obs, a_prev, out_bf16=False):
        if getattr(self.cfg, "motion_train_wm", False):
            return motion_train_actor_mean(pi, obs_norm, obs)
        out = nets.pi_apply(pi, self.norm_obs(obs_norm, obs), out_bf16=out_bf16)
        return nets.ode_step(out, a_prev, self.ode_dmax, self.ode_amax) if self.ode else out

    # ------------------------------------------------------------------ collection
    def record(self, s, s1, tr, a, mu, action_std=None):
        """One step's record: the state the action was taken in with its task context (a start for imagination), the joint target and
        the state after the physics (a world-model transition), the context, and the flags and reward."""
        cmd = {"vel": s.cmd.vel, "heading": s.cmd.heading, "is_heading": s.cmd.is_heading, "is_standing": s.cmd.is_standing, "time_left": s.cmd.time_left}
        rec = {"s_qpos": tr.qpos, "s_qvel": tr.qvel, "s1_qpos": tr.qpos1, "s1_qvel": tr.qvel1, "target": tr.target, "xi": self.sim.context(s),
               "bias": s.world.bias, "was_up": s.was_up, "obs": s.obs, "a_last": s.a_last, "a_ode": s.a_ode, "cmd": cmd, "c_obs": s.c_obs,
               "t": s.t, "done": (tr.terminated | tr.truncated).astype(jnp.float32), "term": tr.terminated.astype(jnp.float32),
               "r": tr.reward}  # fmt: skip
        if getattr(self.cfg, "motion_train_wm", False):
            sensors = s.motion_train_sensors
            rec.update(obs=sensors.observation(), imu_delay=sensors.delay_s,
                       current_imu=sensors.current_imu, previous_imu=sensors.previous_imu)
        if getattr(s, "step_history", None) is not None:
            rec["step_history"] = s.step_history._asdict()
        if self.mixed_amp and not getattr(self.cfg, "wm_floor_amp", False):
            rec["recovery_left"] = s.recovery_left
        if getattr(self.cfg, "wm_floor_amp", False):
            rec["gait_weight"] = self.sim.gait_weight(unflatten(tr.qpos, tr.qvel))
            if self.sim.wm_floor_locomotion:
                rec["gait_world"] = self.sim.wm_gait_worlds
                if self.sim.wm_global_gait:
                    rec["gait_stage"] = jnp.full_like(s.t, s.gait_curriculum.stage)
            rec["episode_limit"] = s.episode_limit
        elif self.ppo:
            i = env.TERMS.index("track_lin")
            rec["rlin"] = tr.parts[:, i] / (env.DT * env.WEIGHTS["track_lin"])
        if getattr(self.cfg, "wm_floor_amp", False) or self.ppo:
            rec["a"] = a
            if action_std is None:
                rec["mu"] = mu
            else:
                rec["behavior_log_prob"] = action_log_prob(a, mu, action_std)
        return rec, tr.stats

    def ingest(self, carry, flat):
        rec = rb.nest(flat)
        ok = rec["ok"] > 0
        S0 = unflatten(rec["s_qpos"], rec["s_qvel"])
        S1 = unflatten(rec["s1_qpos"], rec["s1_qvel"])
        v0 = self.sim.view_state(S0)
        wb = carry["wm_b"]
        upd = lambda st, x: nets.rn_update(st, x, weights=ok)  # noqa: E731
        wm_b = {"in_norm": upd(wb["in_norm"], nets.wm_inputs(S0, v0.height, v0.soles, rec["target"], rec["xi"])),
                "out_norm": upd(wb["out_norm"], nets.wm_targets(S0, S1, env.DT)), "inc_norm": upd(wb["inc_norm"], nets.increments(S0, S1))}  # fmt: skip
        freeze_norm = bool(getattr(self.cfg, "init_policy", ""))
        obs_norm = carry["obs_norm"] if freeze_norm else nets.rn_update(carry["obs_norm"], rec["obs"], weights=ok)
        return {**carry, "buf": rb.add(carry["buf"], flat), "wm_b": wm_b, "obs_norm": obs_norm,
                "env_steps": carry["env_steps"] + self.cfg.num_envs}  # fmt: skip

    def collect_step(self, carry, key):
        c = self.cfg
        s = carry["env"]
        noisy = (jnp.arange(c.num_envs) * c.det_share) % 1.0 >= c.det_share
        obs = s.motion_train_sensors.observation() if getattr(c, "motion_train_wm", False) else s.obs
        clean = self.act(carry["pi"], carry["obs_norm"], obs, s.a_ode)
        std = carry.get("action_std", c.action_noise)
        a = clean + jnp.where(noisy[:, None], std * jax.random.normal(key, clean.shape), 0.)
        s1, tr = self.sim.step(s, a, clean)
        rec, stats = self.record(s, s1, tr, a, clean, carry.get("action_std"))
        flat = rb.sanitize(rb.flatten(rec))
        flat["ok"] = flat["ok"] * (1 - tr.diverged.astype(jnp.float32))
        stats = {**stats, "nonfinite_env": 1 - flat["ok"].mean()}
        return self.ingest({**carry, "env": s1}, flat), stats

    # ------------------------------------------------------------------ world model (as XSoma's)
    def wm_rollout(self, wm, wm_b, S, targets, xis, unroll=True):
        def step(S, x):
            tgt, xi = x
            v = self.sim.view_state(S)
            S1 = nets.wm_step(wm, wm_b, S, v.height, v.soles, tgt, xi, env.DT, out_bf16=self.cfg.mm_bf16 == "all")
            return S1, S1

        _, Ss = jax.lax.scan(step, S, (jnp.swapaxes(targets, 0, 1), jnp.swapaxes(xis, 0, 1)), unroll=unroll)
        return tuple(jnp.swapaxes(P, 0, 1) for P in Ss)

    def wm_loss(self, wm, wm_b, w):
        S_hat = self.wm_rollout(wm, wm_b, unflatten(w["s_qpos"][:, 0], w["s_qvel"][:, 0]), w["target"], w["xi"])
        S_true = unflatten(w["s1_qpos"], w["s1_qvel"])
        std = jnp.maximum(nets.rn_std(wm_b["inc_norm"], 1e-4), self.err_floor)
        e = ((nets.state_error(S_hat, S_true) / std) ** 2).mean(-1)
        return (e * w["mask"]).sum() / w["mask"].shape[0], e[:, -1]

    def wm_update(self, carry, key):
        c = self.cfg
        w = rb.sample_windows(carry["buf"], key, c.wm_batch, c.wm_window)
        (loss, e_last), grads = jax.value_and_grad(self.wm_loss, has_aux=True)(carry["wm"], carry["wm_b"], w)
        grads, _ = optim.clip_by_global_norm(grads, 1.0)
        wm, opt = optim.step(carry["wm"], grads, carry["opt_wm"], self.wm_mask, c.wm_lr, betas=(0.9, 0.999))
        m = w["mask"][:, -1]
        return {**carry, "wm": wm, "opt_wm": opt}, {"wm": loss / c.wm_window, "wm8": (e_last * m).sum() / jnp.maximum(m.sum(), 1)}

    def wm_eval(self, carry, key, B: int = 512, K: int = 32):
        """Open-loop check on real windows: pelvis height error after 8 and K steps, and the termination rate of the simulator against the
        rate of the model's states with a termination collider below the floor."""
        w = rb.sample_windows(carry["buf"], key, B, K)
        S_hat = self.wm_rollout(carry["wm"], carry["wm_b"], unflatten(w["s_qpos"][:, 0], w["s_qvel"][:, 0]), w["target"], w["xi"], unroll=2)
        S_true = unflatten(w["s1_qpos"], w["s1_qvel"])
        herr = jnp.abs(S_hat[0][..., 2] - S_true[0][..., 2])
        m = w["mask"]
        mean = lambda x, j: (x[:, j] * m[:, j]).sum() / jnp.maximum(m[:, j].sum(), 1)  # noqa: E731
        f_env = w["term"]
        f_env = f_env * w["was_up"]  # contact terminations only (a robot that has been up), as the model's geometric event
        f_wm = (self.sim.view_state(S_hat).clearance.min(-1) < 0).astype(jnp.float32) * w["was_up"]
        if self.sim.mixed:
            f_wm = jnp.zeros_like(f_env)
        return {"h_err8": mean(herr, 7), "h_errK": mean(herr, K - 1), "fall_env": (f_env * m).sum() / jnp.maximum(m.sum(), 1),
                "fall_wm": (f_wm * m).sum() / jnp.maximum(m.sum(), 1)}  # fmt: skip

    # ------------------------------------------------------------------ imagination
    def sample_starts(self, buf, key):
        c, B = self.cfg, self.B_im
        k_s, k_u, k_c = jax.random.split(key, 3)
        start = rb.nest(rb.sample_starts(buf, k_s, B, self.amp_span - 1 if self.amp else 0, down_frac=c.imag_down,
                                         reset_frac=getattr(c, "wm_reset_imag_share", 0.), reset_window=getattr(c, "wm_window", 8)))
        fresh = jax.random.uniform(k_u, (B,)) < c.cmd_fresh
        motion_train = getattr(c, "motion_train_wm", False)
        cmd = self.sim.sample_wm_command(k_c, B, stage=start.get("gait_stage", 0)) if motion_train else env.sample_cmd(k_c, B, self.sim.cmd_ranges)
        old = env.Cmd(start["cmd"]["vel"], start["cmd"]["heading"], start["cmd"]["is_heading"] > 0, start["cmd"]["is_standing"] > 0, start["cmd"]["time_left"])
        start["cmd"] = env.where_cmd(fresh, cmd, old)
        S0 = unflatten(start["s_qpos"], start["s_qvel"])
        if motion_train:
            current = self.sim.wm_velocity_command(start["cmd"], S0[1], start["was_up"] > 0, start.get("recovery_left"))
            # Gyro and gravity histories precede the three xyz commands, oldest first.
            start["obs"] = start["obs"].at[..., 24:27].set(current)
            start["c_obs"] = current
            return start
        c_new = env.command(start["cmd"], S0[1])
        # a fresh command replaces the command channels of the newest observation frame (the policy sees it from the start)
        hist = start["obs"]
        lo = (self.sim.history - 1) * self.sim.frame_dim + 6
        start["obs"] = jnp.where(fresh[:, None], hist.at[:, lo : lo + 3].set(c_new), hist)
        start["c_obs"] = jnp.where(fresh[:, None], c_new, start["c_obs"])
        return start

    def critic_in(self, S, h, a_last, c_obs, xi, up, t, recovery_left=None):
        """State, command, previous action and context, followed by two task-phase features.

        Joint floor/gait training uses current-pose suitability and its complement. Other tasks
        retain their upright flag and recovery clock; the input width stays unchanged.
        """
        upf = up.astype(jnp.float32)
        left = jnp.clip(1 - t.astype(jnp.float32) / env.GETUP_STEPS, 0.0, 1.0) * (1 - upf)
        if getattr(self.cfg, "wm_floor_amp", False):
            upf = self.sim.gait_weight(S)
            left = 1. - upf
        elif recovery_left is not None:
            left = recovery_left / self.sim.recovery_amp_steps
        return jnp.concatenate((nets.invariant_features(S, h), c_obs, a_last, xi, upf[:, None], left[:, None]), -1)

    def termination(self, clearance):
        """XSoma's imagined termination of a robot that has been up: the exact event (a termination collider below the floor), with the
        gradient of a sigmoid of the smallest clearance."""
        cmin = clearance.min(-1)
        soft = jax.nn.sigmoid(-cmin / self.cfg.term_tau)
        return (cmin < 0).astype(jnp.float32) + soft - jax.lax.stop_gradient(soft)

    def termination_down(self, t, v, R):
        """XSoma's imagined termination of a robot that has not been up: out of time to rise (t after the step) and not up; the value is
        the exact event, the gradient that of a sigmoid of the margins to being up."""
        late = (t >= env.GETUP_STEPS).astype(jnp.float32)
        hard = late * (1 - self.sim.is_up(v.height, R).astype(jnp.float32))
        soft = late * (1 - env.soft_up(v.height, R, self.sim.robot.UP_HEIGHT))
        return hard + soft - jax.lax.stop_gradient(soft)

    def disc_frame(self, state):
        return self.sim.amp_frame(state)

    def rollout_frame(self, state):
        if getattr(self.cfg, "wm_global_gait", False):
            pos, rotation, _, angular, joints, rates = state
            points = self.sim.key_positions(pos, rotation, joints)
            gait = source_gait.amp_frame(rotation, angular, joints, rates, points - pos[..., None, :])
            return jnp.concatenate((self.sim._gait_amp_frame(state, points), gait), -1)
        return self.disc_frame(state)

    def rollout_gait_style(self, st, windows, command):
        dim = self.sim.amp_dim
        if getattr(self.cfg, "wm_global_gait", False):
            gait = windows[..., -self.gait_span:, dim:]
            return self.style(st, gait), windows[..., :dim]
        return self.style(st, windows[..., -self.gait_span:, :dim], command), windows

    def mix_amp_reward(self, task, styled):
        """Keep task costs and style on the same fixed scale throughout recovery and gait."""
        lerp = self.cfg.amp_lerp
        style_reward = env.DT * lerp * styled
        return (1. - lerp) * task + style_reward, style_reward

    def getup_style(self, st, windows):
        if self.mixed_amp:
            features = self.ground_features(windows)
            scale = (self.cfg.gait_ls_scale if getattr(self.cfg, "wm_floor_amp", False) and self.cfg.gait_ls_scale > 0
                     else self.cfg.recovery_amp_scale)
            scale = getattr(self.cfg, "getup_amp_scale", 0.) or scale
            if self.cfg.recovery_amp_loss == "wasserstein":
                value = amp.score(st["net"], st["norm"], features)
                spread = jnp.maximum(4., .25 * jnp.abs(st["mu"][0] - st["mu"][1]))
                return scale * jnp.tanh((value - st["mu"][1]) / spread)
            return amp.recovery_style(st, features, scale)
        return amp.style(st, amp.score(st["net"], st["norm"], windows), self.cfg.amp_style_scale)

    def ground_features(self, frames):
        # In the mixed AMP layout, vertical landmark coordinates are pelvis-relative.
        # Give the recovery discriminator absolute foot, knee and hand clearances.
        joints = len(self.sim.amp_joints)
        first_z = 2*joints + 8
        frames = frames.at[..., first_z:first_z+3*len(zenbot.KEY_BODIES):3].add(frames[..., -4:-3])
        # Use the same velocity estimator for noisy physics and kinematic demonstrations.
        q = frames[..., :joints]
        dq = jnp.diff(q, axis=-2) / (self.recovery_stride * env.DT)
        return frames.at[..., joints:2*joints].set(jnp.concatenate((dq[..., :1, :], dq), axis=-2))

    def imagine(self, pi, wm, wm_b, obs_norm, start, keys, amp_st=None, critic_st=None, amp_up_st=None):
        c, B, sim = self.cfg, self.B_im, self.sim
        motion_train = getattr(c, "motion_train_wm", False)
        floor_amp = getattr(c, "wm_floor_amp", False)
        mirror_weight = getattr(c, "imag_gait_mirror_weight", 0.)
        if motion_train:
            from .zenbot_wm_sensors import (
                pd_targets,
                predicted_sensors,
                restore_sensors,
            )

        bf16 = c.mm_bf16 in ("imag", "all")
        S0 = unflatten(start["s_qpos"], start["s_qvel"])
        xi, bias = start["xi"], start["bias"]
        use_amp, use_critic = amp_st is not None, critic_st is not None
        if use_amp:
            hist_s = unflatten(start["h_qpos"], start["h_qvel"])
            hf = self.rollout_frame(hist_s)
            f0 = self.rollout_frame(S0)
            win0 = jnp.concatenate((hf, f0[:, None]), 1)
            ramp = amp_st["ramp"]
        zeros = jnp.zeros((B,))

        def step(carry, k):
            S, a_last, a_ode, hist, cmd, c_obs, t, up, remaining, step_history = carry
            k_o, k_c = jax.random.split(k)
            obs = hist.observation() if motion_train else hist
            a = self.act(pi, obs_norm, obs, a_ode, out_bf16=bf16)
            v = sim.view_state(S)
            if motion_train:
                target = pd_targets(a, hist.frames[..., -1, -self.J:], hist.frames[..., -2, -self.J:], bias, xi,
                                    profile=sim.robot,
                                    elapsed=t if getattr(c, "wm_floor_amp", False) else None)
            else:
                target = sim.control_targets(a, bias, hist, xi)
            S1 = nets.wm_step(wm, wm_b, S, v.height, v.soles, target, xi, env.DT, out_bf16=bf16)
            v1 = sim.view_state(S1)
            t1 = t + 1
            up1 = sim.next_upright(up, v1.height, S1[1])
            moving = jnp.linalg.norm(c_obs[:, :2], axis=-1) + jnp.abs(c_obs[:, 2]) > .05
            mirror_sample = None
            if mirror_weight:
                # Weight the observation that produced a using the same current-pose suitability.
                eligible = sim.gait_weight(S) * moving if floor_amp else up & up1 & (remaining == 0) & moving
                # Treat the sampled trajectory as fixed supervision for the reflected actor.
                mirror_sample = (jax.lax.stop_gradient(obs), jax.lax.stop_gradient(a), jax.lax.stop_gradient(eligible))
            d = jnp.where(up, self.termination(v1.clearance), self.termination_down(t1, v1, S1[1]))
            if sim.mixed:
                d = (t1 >= start.get("episode_limit", sim.episode_steps)).astype(jnp.float32)
            if sim.terrain is not None:
                d = jnp.maximum(d, ~sim.terrain.contains(S1[0][..., :2]))
            if sim.wm_global_gait:
                illegal = (v1.feet.bottom[..., 2:] < 0.) & sim.is_term[sim.term_geoms]
                d = jnp.maximum(d, start["gait_world"] * source_gait.failed(v1.height, S1[1][..., 2, 2], illegal.any(-1), False))
            elif sim.wm_floor_locomotion:
                d = jnp.maximum(d, (start["gait_world"] & up & ~up1).astype(d.dtype))
            recovering = ~up1
            active, deadline = None, False
            if self.mixed_amp and not floor_amp:
                deadline = remaining == 1
                remaining = sim.next_recovery_interval(remaining, up, up1)
                active = remaining > 0
            fall = (up & ~up1).astype(jnp.float32) if sim.mixed else d
            if motion_train:
                fall = sim.wm_fall(up, up1, v1.height, S1[1])
            foot_contact = sim.soft_contact(v1.soles)
            terms = sim.reward_terms(S, S1, v1.kin, a, a_last, c_obs, recovering, fall, foot_contact, zeros, sim.q_soft,
                                     recovery_active=active, recovery_deadline=deadline,
                                     root_com=sim.context_root_com(xi))
            if step_history is not None:
                bottom = v1.feet.bottom
                # Predicted collision heights substitute for force sensors in the world model.
                feet_down = bottom[..., :2] <= env.FOOT_CONTACT_H
                undesired = jax.nn.sigmoid((env.FOOT_CONTACT_H - bottom[..., 2:]) / .003).sum(-1)
                foot_acc_z = (v1.feet.com_velocity_z - v.feet.com_velocity_z) / env.DT
                step_history, added = gait_rewards.reward_terms(
                    S1, v1.feet, c_obs, sim.estimated_torque(S, target, xi), (S1[5] - S[5]) / env.DT,
                    sim.q_soft, feet_down, undesired, v1.feet.predicted_slip, foot_acc_z, step_history,
                    sim.gait_weight(S1) if floor_amp else ~active,
                    min_ankle_spacing=sim.min_ankle_spacing)
                terms.update(added)
            r = sum(sim.weighted(terms).values())
            if use_amp:  # the gates of the two styles at this step: the locomotion one while up, the get-up one while not yet up
                frame1 = self.rollout_frame(S1)
                if floor_amp:
                    g_loco = sim.gait_weight(S1)
                    g_up = 1. - g_loco
                else:
                    upf = up1.astype(jnp.float32)
                    g_loco = {"track_shift": moving * upf * (c.amp_track_lo + terms.get("track_lin", 0.)), "moving": moving * upf,
                              "track": moving * terms.get("track_lin", 0.), "none": upf}[c.amp_gate]  # fmt: skip
                    g_up = 1. - upf
                    if self.mixed_amp:
                        g_up = active.astype(jnp.float32)
                        g_loco *= 1. - g_up
                gate = jnp.stack((g_loco, g_up), -1)
            else:
                frame1, gate = jnp.zeros((B, 1)), jnp.zeros((B, 2))
            if motion_train:
                cmd1 = sim.tick_wm_command(cmd, k_c, S1, up1, remaining, start.get("gait_stage", 0))
                c1 = sim.wm_velocity_command(cmd1, S1[1], up1, remaining)
                hist1 = predicted_sensors(hist, S1, c1, a, k_o, sim.noisy_obs)
            else:
                cmd1 = env.tick(cmd, k_c, sim.cmd_ranges)
                c1 = sim.command(cmd1, S1[1])
                hist1 = sim.push_actor(hist, sim.obs_frame(S1, a, c1, k_o))
            cin1 = (self.critic_in(S1, v1.height, a, c1, xi, up1, t1, recovery_left=remaining)
                    if use_critic else zeros[:, None])
            gait_cmd = c_obs if getattr(c, "command_gait_amp", False) else None
            return (S1, a, a, hist1, cmd1, c1, t1, up1, remaining, step_history), (
                r, d, frame1, gate, cin1, recovering.astype(jnp.float32), gait_cmd, mirror_sample)

        if c.imag_remat == "dots":
            step = jax.checkpoint(step, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        elif c.imag_remat == "full":
            step = jax.checkpoint(step)
        history = (restore_sensors(start["obs"], start["imu_delay"], start["current_imu"],
                                   start["previous_imu"]) if motion_train else start["obs"])
        init = (S0, start["a_last"], start["a_ode"], history, start["cmd"], start["c_obs"], start["t"], start["was_up"] > 0,
                None if floor_amp else start.get("recovery_left"),
                gait_rewards.StepHistory(**start["step_history"]) if "step_history" in start else None)
        _, (r, d, frames, gate, cins, rec, gait_cmd, mirror_samples) = jax.lax.scan(step, init, keys, unroll=c.imag_unroll)
        if use_amp:
            F = self.amp_span
            seq = jnp.concatenate((win0, jnp.swapaxes(frames, 0, 1)), 1)
            idx = jnp.arange(1, c.horizon + 1)[:, None] + jnp.arange(F)[None, :]
            wins = seq[:, idx]
            cmd = None if gait_cmd is None else jnp.swapaxes(gait_cmd, 0, 1)
            style, wins = self.rollout_gait_style(amp_st, wins, cmd)
            style = jnp.swapaxes(style, 0, 1)
            mix = gate[..., 0] * style
            up_full = 0.0  # amp_getup_full: the get-up style at full scale, outside the ramp
            if amp_up_st is not None:
                score = self.getup_style(amp_up_st, wins[..., ::self.recovery_stride, :])
                style_up = jnp.swapaxes(score, 0, 1)
                if c.amp_getup_full:
                    up_full = gate[..., 1] * style_up
                else:
                    mix = mix + gate[..., 1] * style_up
            if c.amp_lerp > 0:  # r = (1 - lerp) r_task + lerp dt mix: mix in [0, 1], dt mix the scale of one task term of weight 1
                r, r_amp = self.mix_amp_reward(r, ramp * mix + up_full)
            else:  # XSoma's: r = r_task + amp_w dt ramp mix
                r_amp = c.amp_w * env.DT * (ramp * mix + up_full) if c.amp_getup_full else c.amp_w * env.DT * ramp * mix
                r = r + r_amp
        alive = jnp.cumprod(1 - d, 0)
        alive_before = jnp.concatenate((jnp.ones_like(alive[:1]), alive[:-1]), 0)
        disc = c.gamma ** jnp.arange(c.horizon, dtype=jnp.float32)[:, None]
        G = (disc * alive_before * r).sum(0)
        aux = {"G": G.mean(), "r": (alive_before * r).sum(0).mean() / c.horizon, "surv": alive[-1].mean(), "rec_im": rec.mean()}
        if use_critic:
            v0 = sim.view_state(S0)
            cin0 = self.critic_in(S0, v0.height, start["a_last"], start["c_obs"], xi, start["was_up"] > 0, start["t"], start.get("recovery_left"))
            v_end = critic.value(critic_st, cins[-1])
            G = G + c.gamma**c.horizon * alive[-1] * v_end
            aux["v_end"] = v_end.mean()
            aux["traj"] = jax.lax.stop_gradient((jnp.concatenate((cin0[None], cins), 0), r, d))
        if use_amp:
            aux["r_amp"] = (alive_before * r_amp).sum(0).mean() / c.horizon
        loss = -G.mean()
        if mirror_weight:
            observations, actions, gait = mirror_samples
            eligible = gait * (alive > 0)
            observations = jnp.where(eligible[..., None] > 0, observations, 0.)
            actions = jnp.where(eligible[..., None] > 0, actions, 0.)
            reflected = motion_train_actor_mean(pi, obs_norm, mirror_observation(observations))
            error = jnp.square(reflected - mirror_action(actions)).mean(-1)
            mse = (eligible * error).sum() / jnp.maximum(eligible.sum(), 1)
            loss += mirror_weight * mse
            aux.update(imag_gait_mirror_mse=mse, imag_gait_mirror_fraction=eligible.mean())
        return loss, aux

    # ------------------------------------------------------------------ hybrid PPO
    def gait_features(self, windows, command):
        features = amp.gait_window(windows, len(self.sim.amp_joints), self.cfg.gait_amp_stride, env.DT)
        if getattr(self.cfg, "command_gait_amp", False):
            if getattr(self.cfg, "wm_range_commands", False):
                # Condition the style critic on the reference motion it can actually observe.
                speeds = jnp.asarray(self.sim.wm_command_speeds)
                command = speeds[reference_command_index(command, speeds)]
            labels = jnp.broadcast_to(command[..., None, :], (*features.shape[:-1], 3))
            features = jnp.concatenate((features, labels), axis=-1)
        return features

    def style(self, st, windows, command=None):
        if getattr(self.cfg, "wm_global_gait", False):
            return amp.recovery_style(st, windows, self.cfg.gait_ls_scale, motion_train=True)
        if getattr(self.cfg, "gait_ls_scale", 0.):
            features = self.gait_features(windows, command)
            return amp.recovery_style(st, features, self.cfg.gait_ls_scale,
                                      soft_floor=getattr(self.cfg, "command_gait_amp", False))
        return amp.style(st, amp.score(st["net"], st["norm"], windows), self.cfg.amp_style_scale)

    def ppo_batch(self, carry):
        """The newest `ppo_len` real transitions of the PPO envs (from the replay buffer: the collector's own steps, no extra simulation),
        with the imagination's reward (the environment's reward, the two styles with their gates, amp_lerp) and GAE(lambda) advantages
        of the critic. Flat (ppo_len * n_ppo, ...) arrays: obs, a_ode, the action and its behavior mean, the advantage and a mask."""
        c, sim, L, F = self.cfg, self.sim, self.cfg.ppo_len, self.amp_span
        floor_amp = getattr(c, "wm_floor_amp", False)
        buf, n = carry["buf"], len(self.ppo_envs)
        T = (F - 1) + L + 1  # style history, the transitions, the state after the last one (bootstrap)
        ages = buf["size"] - T + jnp.arange(T)
        rows = rb._rows(buf, ages)
        w = rb.gather(buf, rows[:, None], jnp.asarray(self.ppo_envs)[None, :])  # (T, n, ...)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])  # noqa: E731
        tr = slice(F - 1, F - 1 + L)
        S1 = unflatten(flat(w["s1_qpos"][: T - 1]), flat(w["s1_qvel"][: T - 1]))
        r = w["r"][tr]
        if self.amp:
            feats = self.rollout_frame(S1).reshape(T - 1, n, -1)
            idx = jnp.arange(L)[:, None] + jnp.arange(F)[None, :]  # the window of transition k: post-step frames k .. k + F - 1 of the rows
            if self.mixed_amp:
                # Match imagination: fill missing pre-episode history with the reset pose.
                first = idx[:, -1:] - w["t"][tr]
                rows0 = jnp.maximum(first, 0)
                reset_states = unflatten(w["s_qpos"][rows0, jnp.arange(n)], w["s_qvel"][rows0, jnp.arange(n)])
                sequence = jnp.where((idx[:, :, None] < first[:, None, :])[..., None], self.rollout_frame(reset_states)[:, None], feats[idx])
            else:
                sequence = feats[idx]
            wins = jnp.moveaxis(sequence, 2, 0).reshape(n * L, F, -1)  # (n, L) order
            h1 = sim.ground_state(S1)[0][..., 2].reshape(T - 1, n)[tr]
            R1 = S1[1].reshape(T - 1, n, 3, 3)[tr]
            upf = sim.next_upright(w["was_up"][tr] > 0, h1, R1).astype(jnp.float32)
            cob = w["c_obs"][tr]
            if floor_amp:
                g_loco = sim.gait_weight(S1).reshape(T - 1, n)[tr]
                g_up = 1. - g_loco
            else:
                moving = (jnp.linalg.norm(cob[..., :2], axis=-1) + jnp.abs(cob[..., 2]) > 0.05).astype(jnp.float32)
                g_loco = {"track_shift": moving * upf * (c.amp_track_lo + w["rlin"][tr]), "moving": moving * upf,
                          "track": moving * w["rlin"][tr], "none": upf}[c.amp_gate]  # fmt: skip
                g_up = 1. - upf
                if self.mixed_amp:
                    remaining = sim.next_recovery_interval(w["recovery_left"][tr], w["was_up"][tr] > 0, upf > 0)
                    g_up = (remaining > 0).astype(jnp.float32)
                    g_loco *= 1. - g_up
            gait_cmd = None
            if getattr(c, "command_gait_amp", False):
                gait_cmd = cob.swapaxes(0, 1).reshape(n * L, 3)
            gait_style, wins = self.rollout_gait_style(carry["amp"], wins, gait_cmd)
            gait_style = gait_style.reshape(n, L).T
            mix = g_loco * gait_style
            ramp = carry["amp"]["ramp"]
            up_full = 0.0
            if self.getup:  # (amp_getup_full: the get-up style at full scale, outside the ramp, as in imagination)
                score = self.getup_style(carry["amp_up"], wins[..., ::self.recovery_stride, :])
                recovery_reward = g_up * score.reshape(n, L).T
                if c.amp_getup_full:
                    up_full = recovery_reward
                else:
                    mix = mix + recovery_reward
            styled = ramp * mix + up_full if c.amp_getup_full else ramp * mix
            r = self.mix_amp_reward(r, styled)[0] if c.amp_lerp > 0 else r + c.amp_w * env.DT * styled
        st = slice(F - 1, T)  # the L + 1 states s_0 .. s_L
        S = unflatten(flat(w["s_qpos"][st]), flat(w["s_qvel"][st]))
        cin = self.critic_in(S, sim.view_state(S).height, flat(w["a_last"][st]), flat(w["c_obs"][st]), flat(w["xi"][st]),
                             flat(w["was_up"][st]) > 0, flat(w["t"][st]),
                             flat(w["recovery_left"][st]) if self.mixed_amp and not floor_amp else None)  # fmt: skip
        V = critic.value(carry["critic"], cin).reshape(L + 1, n)
        done, term = w["done"][tr], w["term"][tr]
        trunc = done * (1 - term)
        if getattr(c, "wm_floor_amp", False):
            # Use the post-action pose before auto-reset for time limits.
            def timeout_bootstrap():
                last = unflatten(flat(w["s1_qpos"][tr]), flat(w["s1_qvel"][tr]))
                inputs = self.critic_in(last, sim.ground_state(last)[0][:, 2], flat(w["a"][tr]), flat(w["c_obs"][tr]),
                                        flat(w["xi"][tr]), flat(upf) > 0, flat(w["t"][tr]) + 1)
                return critic.value(carry["critic"], inputs).reshape(L, n)
            timeout_value = jax.lax.cond(trunc.any(), timeout_bootstrap, lambda: V[:-1])
            delta = r + c.gamma * ((1 - done) * V[1:] + trunc * timeout_value) - V[:-1]
        else:
            delta = r + c.gamma * (1 - done) * V[1:] + c.gamma * trunc * V[:-1] - V[:-1]
        mask = w["ok"][tr]

        def back(A, x):
            d_k, dn = x
            A = d_k + c.gamma * c.lam * (1 - dn) * A
            return A, A

        _, adv = jax.lax.scan(back, jnp.zeros((n,)), (delta, done), reverse=True)
        mean = (adv * mask).sum() / jnp.maximum(mask.sum(), 1.0)
        std = jnp.sqrt(((adv - mean) ** 2 * mask).sum() / jnp.maximum(mask.sum(), 1.0)) + 1e-6
        behavior = "behavior_log_prob" if "behavior_log_prob" in w else "mu"
        out = {"obs": w["obs"][tr], "a_ode": w["a_ode"][tr], "a": w["a"][tr], behavior: w[behavior][tr], "adv": (adv - mean) / std, "mask": mask}
        if getattr(c, "gait_mirror_weight", 0.) or getattr(c, "wm_global_gait", False):
            out["gait_mirror"] = w["gait_weight"][tr]
        if getattr(c, "wm_floor_amp", False):
            out.update(value_input=cin[:-n].reshape(L, n, -1), old_value=V[:-1], value_target=adv + V[:-1])
        out = jax.lax.stop_gradient({k: flat(v) for k, v in out.items()})
        if getattr(self, "recovery_teacher", None) is not None:
            # Rebuild the expert's five-frame sensor history on the learner's own physical rollouts.
            # At a reset, pad the history with the first pose from the current episode.
            states = unflatten(w["s_qpos"], w["s_qvel"])
            rot, gyro, points = sim.recovery_features(states)
            frames = jnp.concatenate((gyro, rot, states[4], states[5], w["a_last"], points), axis=-1)
            end = jnp.arange(F - 1, F - 1 + L)[:, None, None]
            ages = jnp.arange(sim.history - 1, -1, -1)[None, :, None]
            ages = jnp.minimum(ages, w["t"][tr, None, :])
            history = frames[end - ages, jnp.arange(n)].transpose(0, 2, 1, 3).reshape(L * n, -1)
            teacher, norm = self.recovery_teacher
            out["getup_target"] = jax.lax.stop_gradient(motion_train_actor_mean(teacher, norm, history))
            out["getup_active"] = 1. - flat(w["gait_weight"][tr])
            if self.gait_teacher is not None:
                gait_actor, gait_norm = self.gait_teacher
                out["gait_target"] = jax.lax.stop_gradient(motion_train_actor_mean(gait_actor, gait_norm, out["obs"]))
        stats = {"ppo_r": (r * mask).sum() / jnp.maximum(mask.sum(), 1.0)}
        if self.mixed_amp:
            scored = g_up * mask * (w["t"][tr]+1 >= F-1)
            stats["real_getup_amp_per_active_step"] = (recovery_reward * scored).sum() / jnp.maximum(scored.sum(), 1.) * env.DT * c.amp_lerp
            if getattr(c, "wm_lift_gate", 0.):
                before = w["s_qpos"][tr, :, 2] - sim.ground_height(w["s_qpos"][tr, :, :2])
                floor = mask * (before <= c.wm_lift_gate) * (h1 <= c.wm_lift_gate)
                style_weight = c.amp_lerp if c.amp_lerp > 0 else c.amp_w
                stats["real_below_lift_amp_per_step"] = (styled * floor).sum() / jnp.maximum(floor.sum(), 1.) * env.DT * style_weight
        return out, stats

    def ppo_loss(self, pi, obs_norm, b, action_std=None):
        """Clipped PPO objective against the distribution that generated each action."""
        c = self.cfg
        mu = self.act(pi, obs_norm, b["obs"], b["a_ode"])
        if action_std is None:
            old_error = jnp.square(b["a"] - b["mu"]).sum(-1)

        def surrogate(action, mean):
            if action_std is None:
                logr = (old_error - jnp.square(action - mean).sum(-1)) / (2 * c.action_noise**2)
            else:
                logr = action_log_prob(action, mean, action_std) - b["behavior_log_prob"]
            ratio = jnp.exp(jnp.clip(logr, -20., 20.))
            gain = jnp.minimum(ratio * b["adv"], jnp.clip(ratio, 1 - c.ppo_clip, 1 + c.ppo_clip) * b["adv"])
            return gain, logr, ratio
        surr, logr, ratio = surrogate(b["a"], mu)
        m = b["mask"]
        nm = jnp.maximum(m.sum(), 1.0)
        aux = {"ppo_clipfrac": ((jnp.abs(ratio - 1) > c.ppo_clip) * m).sum() / nm, "ppo_logr": (jnp.abs(logr) * m).sum() / nm}
        loss = -(surr * m).sum() / nm
        if action_std is not None:
            # The Gaussian entropy's additive constant has zero gradient.
            loss -= c.action_entropy_coef * jnp.log(action_std).sum() * (m.sum() / nm)
        if "gait_mirror" in b:
            reflected = self.act(pi, obs_norm, mirror_observation(b["obs"]), mirror_action(b["a_ode"]))
            target = jax.lax.stop_gradient(mirror_action(mu))
            eligible = m * b["gait_mirror"]
            if getattr(c, "wm_global_gait", False):
                # Reflecting the behavior distribution and its action preserves their log probability.
                reflected_gain = surrogate(mirror_action(b["a"]), reflected)[0]
                loss -= .5 * ((reflected_gain - surr) * eligible).sum() / nm
            mse = (eligible * jnp.square(reflected - target).mean(-1)).sum() / jnp.maximum(eligible.sum(), 1.)
            loss += c.gait_mirror_weight * mse
            aux["gait_mirror_mse"] = mse
        if "getup_target" in b:
            active = m * b["getup_active"]
            error = jnp.abs(mu - b["getup_target"])
            huber = jnp.where(error < 1., .5 * error**2, error - .5).mean(axis=-1)
            imitation = (active * huber).sum() / jnp.maximum(active.sum(), 1.)
            loss += imitation
            aux["getup_teacher_huber"] = imitation
            if "gait_target" in b:
                gait = m * (1. - b["getup_active"])
                deviation = jnp.abs(mu - b["gait_target"])
                gait_huber = jnp.where(deviation < 1., .5 * deviation**2, deviation - .5).mean(axis=-1)
                gait_loss = (gait * gait_huber).sum() / jnp.maximum(gait.sum(), 1.)
                loss += .5 * gait_loss
                aux["gait_teacher_huber"] = gait_loss
        return loss, aux

    def policy_update(self, carry, start, key, ppo_b=None):
        c = self.cfg
        (_, aux), grads = jax.value_and_grad(self.imagine, has_aux=True)(carry["pi"], carry["wm"], carry["wm_b"], carry["obs_norm"], start,
                                                                         jax.random.split(key, c.horizon), carry.get("amp"), carry.get("critic"),
                                                                         carry.get("amp_up"))  # fmt: skip
        action_std = carry.get("action_std")
        grads = (grads, jax.tree.map(jnp.zeros_like, action_std))
        if ppo_b is not None:  # g = g_bptt + ppo_w |g_bptt| / |g_ppo| g_ppo
            # the PPO gradient of each half of the PPO envs (even / odd), their mean the batch's: the cosine of the halves measures the
            # PPO gradient's signal against its noise, its cosine with the BPTT gradient whether the two point the same way
            n = len(self.ppo_envs)
            half = (jnp.arange(ppo_b["mask"].shape[0]) % n) % 2
            g_half = [jax.value_and_grad(self.ppo_loss, argnums=(0, 3), has_aux=True)(
                          carry["pi"], carry["obs_norm"], {**ppo_b, "mask": ppo_b["mask"] * (half == h)}, action_std)
                      for h in (0, 1)]  # fmt: skip
            (_, paux), g0 = g_half[0]
            (_, paux1), g1 = g_half[1]
            gp = jax.tree.map(lambda a, b: 0.5 * (a + b), g0, g1)
            paux = jax.tree.map(lambda a, b: 0.5 * (a + b), paux, paux1)
            dot = lambda a, b: sum(jnp.sum(x * y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))  # noqa: E731
            norm = lambda g: jnp.sqrt(dot(g, g))  # noqa: E731
            nb, npp = norm(grads), norm(gp)
            cos_bp = dot(grads, gp) / (nb * npp + 1e-12)
            cos_half = dot(g0, g1) / (norm(g0) * norm(g1) + 1e-12)
            scale = self.ppo_w * nb / (npp + 1e-8)
            grads = jax.tree.map(lambda a, b: a + scale * b, grads, gp)
            aux = {**aux, **paux, "bptt_gn": nb, "ppo_gn": npp, "ppo_cos_bp": cos_bp, "ppo_cos_half": cos_half}
        grads, gn = optim.clip_by_global_norm(grads, c.pi_clip)
        finite = jnp.isfinite(gn)
        first = carry["gn_ema"] <= 0
        ok = finite & (first | (c.pi_gn_gate <= 0) | (gn < c.pi_gn_gate * carry["gn_ema"]))
        gn_ema = jnp.where(ok, jnp.where(first, gn, 0.98 * carry["gn_ema"] + 0.02 * gn), 1.01 * carry["gn_ema"])
        grads = jax.tree.map(lambda g: jnp.where(ok, g, 0.0), grads)
        mean_grads, std_grads = grads
        pi, opt_pi = optim.step(carry["pi"], mean_grads, carry["opt_pi"], self.pi_mask, c.pi_lr, betas=(0.9, 0.95), ok=ok)
        avg = optim.ema(carry["avg"], pi, c.pi_ema, carry["it"] == c.wm_only_iters)
        out = {"pi": pi, "opt_pi": opt_pi, "avg": avg, "gn_ema": gn_ema, "nonfinite": carry["nonfinite"] + (~finite).astype(jnp.int32)}
        if action_std is not None:
            new_std, opt_std = optim.step(action_std, std_grads, carry["opt_std"], False, c.pi_lr, betas=(0.9, 0.95), ok=ok)
            out.update(action_std=jnp.maximum(new_std, 1.e-6), opt_std=opt_std)
        if self.use_critic:
            x, r, d = aux.pop("traj")
            real_values = ((ppo_b["value_input"], ppo_b["value_target"], ppo_b["mask"])
                           if getattr(c, "wm_floor_amp", False) else None)
            out["critic"], aux["vloss"] = critic.fit(carry["critic"], x, r, d, jax.random.fold_in(key, 5), c.gamma, c.lam,
                                                     c.critic_lr, c.critic_epochs, c.critic_mb, real_values=real_values)
        return out, {**aux, "gn": gn, "gn_gated": (finite & ~ok).astype(jnp.float32)}

    # ------------------------------------------------------------------ iteration programs
    SUMMED = ("velocity_errors", "done", "terminated", "len_sum", "rew_sums", "diverged", "down_done", "down_up", "pushes")

    def collect_iter(self, carry):
        key, k_c = jax.random.split(carry["key"])
        carry, st = jax.lax.scan(self.collect_step, {**carry, "key": key}, jax.random.split(k_c, self.cfg.env_steps))
        return carry, {k: (v.sum(0) if k in self.SUMMED else v.mean(0)) for k, v in st.items()}

    def gait_amp_eligible(self, window, state):
        """Use equal weights for valid global-gait windows and task masks for other priors."""
        valid = window["mask"][:, -1] > 0
        if getattr(self.cfg, "wm_global_gait", False):
            return valid
        if getattr(self.cfg, "wm_floor_amp", False):
            return valid * self.sim.gait_weight(state)[:, -1]
        up = self.sim.is_up(self.sim.ground_state(state)[0][..., 2], state[1]).all(-1) & (window["was_up"] > 0).all(-1)
        valid &= up
        if not getattr(self.cfg, "motion_train_wm", False):
            cmd = window["c_obs"][:, -1]
            valid &= jnp.linalg.norm(cmd[:, :2], axis=-1) + jnp.abs(cmd[:, 2]) > .05
        if self.mixed_amp and not getattr(self.cfg, "wm_floor_amp", False):
            valid &= (window["recovery_left"] == 0).all(-1)
        return valid

    def amp_iter(self, carry):
        """Fit each discriminator on eligible real windows and its own reference clips, then update the style ramp."""
        c, sim = self.cfg, self.sim
        disc_step = amp.train_step_with_robot if sim.wm_floor_amp else amp.train_step

        def one(st, k):
            k_w, k_d, k_g = jax.random.split(k, 3)
            w = rb.sample_windows(carry["buf"], k_w, c.amp_batch, self.gait_span, recent=c.amp_recent, envs=self.amp_envs)
            S = unflatten(w["s_qpos"], w["s_qvel"])
            global_gait = getattr(c, "wm_global_gait", False)
            if global_gait:
                p, rot, _, ang, q, qd = S
                feats = source_gait.amp_frame(rot, ang, q, qd, sim.key_positions(p, rot, q) - p[..., None, :])
            else:
                feats = sim.amp_frame(S)
            ok = self.gait_amp_eligible(w, S)
            usable = ok.mean()
            label = None
            if getattr(c, "command_gait_amp", False):
                # Match expert and robot command frequencies exactly, even when many windows ended in a fall.
                k_select, k_d = jax.random.split(k_d)
                valid = jnp.flatnonzero(ok > 0, size=c.amp_batch)
                source = valid[jax.random.randint(k_select, (c.amp_batch,), 0, jnp.maximum(1, (ok > 0).sum()))]
                label = w["c_obs"][source, -1]
                feats = feats[source]
                ok = ok[source]
                # A real expert motion can also be the wrong response to this command.
                quarter = c.amp_batch // 4
                if quarter:
                    k_wrong, k_short, k_d = jax.random.split(k_d, 3)
                    wrong = self.demos.sample_commands(k_wrong, label[:quarter], mismatch=True)
                    feats = feats.at[:quarter].set(wrong)
                    # A stationary reference is the positive example for standing.
                    part = slice(quarter, 2 * quarter)
                    short = self.demos.sample_commands(k_short, label[part], reduced=True)
                    moving = jnp.any(label[part] != 0., axis=-1)
                    feats = feats.at[part].set(jnp.where(moving[:, None, None], short, feats[part]))
                demo = self.demos.sample_commands(k_d, label)
            elif global_gait and self.demos.command_logits is not None:
                # Match the expert command distribution to the valid policy windows.
                k_cmd, k_d = jax.random.split(k_d)
                weights = jnp.where(ok.sum() > 0., ok, jnp.ones_like(ok))
                source = jax.random.choice(k_cmd, len(ok), (c.amp_batch,), p=weights / weights.sum())
                demo = self.demos.sample_for_commands(k_d, w["c_obs"][source, -1])
            else:
                demo = self.demos.sample(k_d, c.amp_batch)
            gait_ls = bool(getattr(c, "gait_ls_scale", 0.))
            if gait_ls and not global_gait:
                demo = self.gait_features(demo, label)
                feats = self.gait_features(feats, label)
            st, aux = disc_step(st, demo, feats, ok.astype(jnp.float32), k_g, c.amp_lr,
                                     recovery=gait_ls, motion_train=global_gait)
            return st, {**aux, "usable": usable}

        def one_up(st, k):
            k_w, k_d, k_g = jax.random.split(k, 3)
            padded_opening = bool(getattr(c, "wm_lift_gate", 0.))
            w = rb.sample_windows(carry["buf"], k_w, c.amp_batch, self.amp_span, recent=c.amp_recent, envs=self.amp_envs,
                                  recovery_only=self.mixed_amp, recovery_steps=sim.recovery_amp_steps if padded_opening else 0)
            S = unflatten(w["s_qpos"][:, ::self.recovery_stride], w["s_qvel"][:, ::self.recovery_stride])
            feats = self.disc_frame(S)
            phase = 1. - w["was_up"].astype(jnp.float32)
            wasserstein = self.mixed_amp and c.recovery_amp_loss == "wasserstein"
            if wasserstein:
                demo = self.getup_demos.sample_ground_heights(k_d, self.sim.ground_state(S)[0][:, -1, 2])
            elif padded_opening:
                demo = self.getup_demos.sample_opening_windows(k_d, c.amp_batch)
            else:
                demo = self.getup_demos.sample(k_d, c.amp_batch)
            demo = demo[:, ::self.recovery_stride]
            if self.mixed_amp:
                phase = (1. - sim.gait_weight(S)[:, -1:]) if getattr(c, "wm_floor_amp", False) else w["recovery_left"] > 0
                feats, demo = self.ground_features(feats), self.ground_features(demo)
            ok = ((w["mask"][:, -1] > 0) * phase[:, -1] if getattr(c, "wm_floor_amp", False)
                  else (w["mask"][:, -1] > 0) & (phase > 0).all(-1))
            n_tempo = round(c.amp_batch * getattr(c, "wm_tempo_negatives", 0.))
            if n_tempo:
                wrong, moving = self.getup_demos.sample_wrong_tempo(
                    jax.random.fold_in(k_d, 1), n_tempo, c.amp_frames, self.recovery_stride, len(sim.amp_joints),
                    freeze_start=sim.wm_floor_locomotion)
                feats = feats.at[:n_tempo].set(jnp.where(moving[:, None, None], self.ground_features(wrong), feats[:n_tempo]))
            st, aux = disc_step(st, demo, feats, ok.astype(jnp.float32), k_g,
                                     c.amp_lr,
                                     recovery=self.mixed_amp and not wasserstein, include_demo_norm=wasserstein)
            if n_tempo:
                aux["tempo_negative_share"] = (moving * ok[:n_tempo]).sum() / jnp.maximum(ok.sum(), 1)
            return st, {**aux, "usable": ok.mean()}

        key, k, k2 = jax.random.split(carry["key"], 3)
        st, aux = jax.lax.scan(one, carry["amp"], jax.random.split(k, c.amp_disc_steps))
        ramp = jnp.clip((carry["it"] - c.amp_delay) / float(c.amp_ramp), 0., 1.) if c.amp_ramp else (carry["it"] >= c.amp_delay).astype(jnp.float32)
        aux = jax.tree.map(lambda x: x[-1], aux)
        out = {**carry, "amp": {**st, "ramp": ramp}, "key": key}
        stats = {"amp_d_demo": aux["d_demo"], "amp_d_robot": aux["d_robot"], "amp_gp": aux["gp"], "amp_usable": aux["usable"], "amp_ramp": ramp}
        if self.getup:
            disc_keys = jax.random.split(k2, c.amp_disc_steps)
            st2, aux2 = jax.lax.scan(one_up, carry["amp_up"], disc_keys)
            aux2 = jax.tree.map(lambda x: x[-1], aux2)
            out["amp_up"] = st2
            stats.update({"ampg_d_demo": aux2["d_demo"], "ampg_d_robot": aux2["d_robot"], "ampg_usable": aux2["usable"]})
            if "tempo_negative_share" in aux2:
                stats["ampg_tempo_negative_share"] = aux2["tempo_negative_share"]
        return out, stats

    def wm_iter(self, carry):
        c = self.cfg
        key, k_w = jax.random.split(carry["key"])
        carry, ws = jax.lax.scan(self.wm_update, {**carry, "key": key}, jax.random.split(k_w, c.wm_batches))
        wm_now = ws["wm"].mean()
        first = carry["wm_ema"] <= 0
        healthy = first | (c.wm_gate <= 0) | (wm_now < c.wm_gate * carry["wm_ema"])
        wm_ema = jnp.where(healthy, jnp.where(first, wm_now, 0.98 * carry["wm_ema"] + 0.02 * wm_now), 1.005 * carry["wm_ema"])
        return {**carry, "wm_ema": wm_ema, "wm_ok": healthy}, {"wm": wm_now, "wm8": ws["wm8"][-1]}

    def pi_iter(self, carry):
        c = self.cfg
        key, k_p = jax.random.split(carry["key"])
        names = ("pi", "opt_pi", "avg", "gn_ema", "nonfinite") + (("critic",) if self.use_critic else ())
        if "action_std" in carry:
            names += ("action_std", "opt_std")

        def do(sub):
            ppo_b, pst = self.ppo_batch(carry) if self.ppo else (None, {})

            def one(sub, k):
                k_s, k_u = jax.random.split(k)
                return self.policy_update({**carry, **sub}, self.sample_starts(carry["buf"], k_s), k_u, ppo_b)

            sub, st = jax.lax.scan(one, sub, jax.random.split(k_p, c.pi_updates))
            return sub, {**jax.tree.map(jnp.mean, st), **pst}

        stat_keys = (("G", "r", "surv", "rec_im", "gn", "gn_gated") + (("r_amp",) if self.amp else ()) + (("v_end", "vloss") if self.use_critic else ())
                     + (("bptt_gn", "ppo_clipfrac", "ppo_logr", "ppo_gn", "ppo_r", "ppo_cos_bp", "ppo_cos_half") if self.ppo else ()))  # fmt: skip
        if getattr(c, "gait_mirror_weight", 0.) or getattr(c, "wm_global_gait", False):
            stat_keys += ("gait_mirror_mse",)
        if getattr(c, "imag_gait_mirror_weight", 0.):
            stat_keys += ("imag_gait_mirror_mse", "imag_gait_mirror_fraction")
        if getattr(self, "recovery_teacher", None) is not None:
            stat_keys += ("getup_teacher_huber",)
            if self.gait_teacher is not None:
                stat_keys += ("gait_teacher_huber",)

        if self.mixed_amp and self.ppo:
            stat_keys += ("real_getup_amp_per_active_step",)
            if getattr(c, "wm_lift_gate", 0.):
                stat_keys += ("real_below_lift_amp_per_step",)

        def skip(sub):
            return sub, {k: jnp.zeros((), jnp.float32) for k in stat_keys}

        active = carry["it"] >= c.wm_only_iters
        sub, ps = jax.lax.cond(active & carry["wm_ok"], do, skip, {k: carry[k] for k in names})
        return {**carry, **sub, "key": key, "it": carry["it"] + 1}, {**ps, "wm_gated": (active & ~carry["wm_ok"]).astype(jnp.float32)}

    def compile_programs(self, carry):
        programs = {"collect": jax.jit(self.collect_iter, donate_argnums=(0,)),
                    "pi": jax.jit(self.pi_iter, donate_argnums=(0,))}
        programs["wm"] = jax.jit(self.wm_iter, donate_argnums=(0,))
        if self.amp:
            programs["amp"] = jax.jit(self.amp_iter, donate_argnums=(0,))
        if self.cfg.wm_eval:
            programs["wm_eval"] = jax.jit(lambda cr: self.wm_eval(cr, jax.random.fold_in(cr["key"], 3)))
        if self.sim.wm_floor_locomotion:
            # Release the motion-loader kernels before compiling the long joint policy rollout.
            jax.clear_caches()
            gc.collect()
            compiled = {}
            for name, program in programs.items():
                print(f"[instantamp] compiling {name}", flush=True)
                compiled[name] = program.lower(carry).compile()
            return compiled
        lowered = [program.lower(carry) for program in programs.values()]
        with ThreadPoolExecutor(len(programs)) as ex:
            return dict(zip(programs, ex.map(lambda low: low.compile(), lowered)))


FULL_KEYS = ("pi", "avg", "wm", "wm_b", "obs_norm", "opt_pi", "opt_wm", "amp", "amp_up", "critic", "it", "env_steps", "gn_ema", "wm_ema", "nonfinite")


def exploration_key(path):
    return "exploration/" + "/".join(entry.key for entry in path)


def save_full(path, carry):
    leaves = jax.tree.leaves(jax.device_get({k: carry[k] for k in FULL_KEYS if k in carry}))
    fields = {f"{i:05d}": np.asarray(x) for i, x in enumerate(leaves)}
    if "action_std" in carry:
        state = {k: carry[k] for k in ("action_std", "opt_std")}
        fields.update({exploration_key(p): np.asarray(v) for p, v in jax.tree_util.tree_leaves_with_path(state)})
    curriculum = getattr(carry.get("env"), "gait_curriculum", None)
    if curriculum is not None:
        # Resumed collectors start new episodes, so only completed-episode statistics carry over.
        fields.update({f"gait_{k}": np.asarray(v) for k, v in curriculum._asdict().items() if k != "partial"})
    np.savez(path + ".tmp.npz", **fields)
    os.replace(path + ".tmp.npz", path + ".npz")


def prune_full_checkpoints(directory, keep):
    """Prune numbered full checkpoints only after their companion exports exist."""
    if keep <= 0:
        return
    directory = Path(directory)
    complete = sorted((p for p in directory.glob("*.npz") if p.stem.isdecimal()
                       and (directory.parent / p.name).is_file()
                       and (directory.parent / p.with_suffix(".json").name).is_file()),
                      key=lambda p: int(p.stem))
    for path in complete[:-keep]:
        path.unlink()


def load_full(carry, path):
    part = {k: carry[k] for k in FULL_KEYS if k in carry}
    leaves, tdef = jax.tree.flatten(part)
    with np.load(path, allow_pickle=False) as z:
        if "action_std" in carry and any(name.startswith("exploration/") for name in z.files):
            def read_exploration(p, template):
                name = exploration_key(p)
                value = z[name]
                if value.shape != template.shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {name} in {path}")
                if name == "exploration/action_std" and np.any(value <= 0):
                    raise ValueError(f"Nonpositive action standard deviation in {path}")
                return jnp.asarray(value, dtype=template.dtype)

            carry = {**carry, **jax.tree_util.tree_map_with_path(
                read_exploration, {k: carry[k] for k in ("action_std", "opt_std")})}
        weights = [name for name in z.files if name.isdigit()]
        if len(weights) != len(leaves):
            raise ValueError(f"{path} has {len(weights)} learner arrays, expected {len(leaves)}")
        values = []
        for i, leaf in enumerate(leaves):
            arr = z[f"{i:05d}"]
            if arr.shape != leaf.shape:
                raise ValueError(f"{path} array {i} has shape {arr.shape}, expected {leaf.shape}")
            values.append(jnp.asarray(arr, dtype=leaf.dtype))
        curriculum = getattr(carry.get("env"), "gait_curriculum", None)
        if curriculum is not None and "gait_stage" in z:
            saved = {k: jnp.asarray(z[f"gait_{k}"], dtype=v.dtype) for k, v in curriculum._asdict().items() if k != "partial"}
            carry = {**carry, "env": carry["env"]._replace(gait_curriculum=curriculum._replace(**saved))}
    return {**carry, **jax.tree.unflatten(tdef, values)}


def initialize_actor(carry, path, expected_ode):
    """Load the averaged actor and its normalizer into a fresh training run."""
    from .inference import load

    pi, norm, ode = load(path)
    if not isinstance(pi, list):
        raise ValueError("The training initializer requires a single dense policy")
    if ode != expected_ode and (ode is None or expected_ode is None or not np.allclose(
            [v or 0. for v in ode], [v or 0. for v in expected_ode])):
        raise ValueError("The checkpoint and training ODE action limits differ")
    template = carry["pi"]
    if "ln_g" not in pi[0]:
        # Native Motion Train and LeggedLab actors can use plain ELU layers.
        # Its affine dimensions must still match the named task's policy.
        template = [{k: layer[k] for k in ("w", "b")} for layer in template]
    expected = template, {k: carry["obs_norm"][k] for k in ("mean", "var")}
    loaded = pi, {k: norm[k] for k in ("mean", "var")}
    if jax.tree.structure(loaded) != jax.tree.structure(expected) or any(
            a.shape != b.shape for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(expected))):
        raise ValueError("The checkpoint does not match this task's policy or observations")
    if "opt_pi" in carry:
        carry = {**carry, "opt_pi": optim.init(pi, jax.tree.map(lambda _: False, pi))}
    obs_norm = {k: v for k, v in carry["obs_norm"].items() if k == "count"}
    return {**carry, "pi": pi, "avg": jax.tree.map(jnp.copy, pi), "obs_norm": {**obs_norm, **norm}}


def initialize_native_learner(fresh, full_path, actor_path, expected_ode, *, floor_reward_version=0):
    """Restore the learner, then optionally replace its actor and reset the actor optimizer."""
    carry = fresh
    if full_path:
        companion = Path(full_path).parent.parent / Path(full_path).name
        source_template = initialize_actor(fresh, companion, expected_ode)
        carry = load_full(source_template, full_path)
        if floor_reward_version:
            with np.load(companion, allow_pickle=False) as archive:
                saved_version = int(archive.get("zenbot/floor_reward_version", 0))
            if saved_version != floor_reward_version and "critic" in fresh:
                # Its input width is unchanged, but both the phase features and return have changed.
                carry["critic"] = fresh["critic"]
                print("[instantamp] Joint reward objective changed: initialized a fresh critic and critic optimizer; "
                      "restored actor, world model and motion discriminators.", flush=True)
    if actor_path:
        # The fresh template includes the native LayerNorm parameters even when
        # the restored companion used plain layers and raw observations.
        actor_keys = ("pi", "obs_norm", "opt_pi")
        selected = initialize_actor({k: fresh[k] for k in actor_keys}, actor_path, expected_ode)
        carry = {**carry, **selected}
    return carry


def save_policy(path, carry, meta, ode=None):
    """The averaged and live policy and the observation normalizer (<path>.npz), and the metadata (<path>.json)."""
    flat = {"obs_norm/mean": np.asarray(carry["obs_norm"]["mean"]), "obs_norm/var": np.asarray(carry["obs_norm"]["var"])}
    if "raw" in carry["obs_norm"]:
        flat["obs_norm/raw"] = np.bool_(True)
    if carry.get("delay_uses_first_command", False):
        flat["zenbot/delay_uses_first_command"] = np.bool_(True)
    if "floor_reward_version" in carry:
        flat["zenbot/floor_reward_version"] = np.int32(carry["floor_reward_version"])
    if ode is not None:
        flat["ode/dmax"] = np.float32(0.0 if ode[0] is None else ode[0])
        flat["ode/amax"] = np.float32(0.0 if ode[1] is None else ode[1])
    for name in ("avg", "pi"):
        for i, layer in enumerate(carry[name]):
            for k, v in layer.items():
                flat[f"{'live' if name == 'pi' else 'avg'}/{i}/{k}"] = np.asarray(v)
    np.savez(path + ".tmp.npz", **flat)
    os.replace(path + ".tmp.npz", path + ".npz")
    with open(path + ".json", "w") as f:
        json.dump(meta, f)


def parse(argv=None) -> Config:
    parser = argparse.ArgumentParser(
        description="Train one velocity-commanded policy for walking, running and floor recovery."
    )
    tasks = ("g1", *zenbot_models.MIXED_TASKS)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--robot", choices=tasks, default=Config.robot)
    chosen = selector.parse_known_args(argv)[0]
    recipe = {
        "g1": Config,
        zenbot_models.ZENBOT23.mixed_task: Zenbot23JointConfig,
        zenbot_models.ZENBOT26.mixed_task: ZenbotJointConfig,
    }[chosen.robot]

    for field in dataclasses.fields(recipe):
        options = {}
        if field.name == "robot":
            options["choices"] = tasks
        elif issubclass(recipe, ZenbotJointConfig):
            if field.name == "motion_train_wm":
                options.update(choices=(1,), help=argparse.SUPPRESS)
            elif field.name == "recovery_amp_loss":
                options["choices"] = ("least_squares", "wasserstein")
        parser.add_argument(
            f"--{field.name}", type=type(field.default), default=field.default, **options
        )
    cfg = recipe(**vars(parser.parse_args(argv)))
    if not 0 < cfg.reset_floor < 1 or not cfg.amp_getup or cfg.amp_w <= 0:
        parser.error("Joint training requires walking, running and floor recovery")
    return cfg


def main(argv=None):
    t_launch = time.time()
    cfg = parse(argv)
    flags = [f for f in cfg.xla_flags.split(",") if f.startswith("--")]
    if flags:
        os.environ["XLA_FLAGS"] = " ".join([os.environ.get("XLA_FLAGS", "")] + flags).strip()
    record_initialization = getattr(cfg, "motion_train_wm", False)
    if record_initialization and (Path(cfg.log_dir) / "config.json").exists():
        raise FileExistsError(f"Choose a fresh log directory for the WM run: {cfg.log_dir}")
    os.makedirs(os.path.join(cfg.log_dir, "ckpt", "full"), exist_ok=True)
    with open(os.path.join(cfg.log_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=1)
    xs = InstantAMP(cfg)
    carry = xs.init(jax.random.PRNGKey(cfg.seed))
    actor_path = getattr(cfg, "init_policy", "")
    full_path = getattr(cfg, "init_full", "")
    if full_path or actor_path:
        new_head = None
        new_gait = None
        if full_path and xs.sim.mixed and "amp_up" in carry:
            # The full-checkpoint directory can live on another volume via a symlink.
            previous_cfg = Path(full_path).absolute().parents[2] / "config.json"
            old = json.loads(previous_cfg.read_text()) if previous_cfg.is_file() else {}
            old_global = bool(old.get("wm_global_gait"))
            # Global gait checkpoints predating wm_gait_frames always used three frames.
            old_frames = old.get("wm_gait_frames", 3) if old_global else old.get("amp_frames", cfg.amp_frames)
            old_conditioned = bool(old.get("wm_floor_amp") and not old.get("wm_getup_only") and old.get("gait_ls_scale") and not old_global)
            if (bool(old.get("gait_ls_scale")) != bool(getattr(cfg, "gait_ls_scale", 0.)) or
                    old.get("gait_amp_stride", 1) != getattr(cfg, "gait_amp_stride", 1) or
                    old_global != bool(getattr(cfg, "wm_global_gait", False)) or
                    (old_global and old_frames != xs.gait_span) or
                    old_conditioned != getattr(cfg, "command_gait_amp", False)):
                new_gait = carry["amp"]
                old_dim = source_gait.amp_frame_width(xs.J) if old_global else xs.sim.amp_dim + (3 if old_conditioned else 0)
                old_gait = jax.eval_shape(lambda k: amp.init(k, old_frames, frame_dim=old_dim), jax.random.key(0))
                carry = {**carry, "amp": old_gait}
            old_loss = old.get("recovery_amp_loss", "least_squares") if old.get("recovery_amp_seconds") else "legacy"
            if old_loss == "legged_support":
                raise ValueError("This checkpoint uses the retired support discriminator")
            if old_loss != cfg.recovery_amp_loss or old.get("recovery_amp_stride", 1) != cfg.recovery_amp_stride:
                new_head = carry["amp_up"]
        carry = initialize_native_learner(carry, full_path, actor_path, (xs.ode_dmax, xs.ode_amax) if xs.ode else None,
                                          floor_reward_version=1 if getattr(cfg, "wm_floor_amp", False) else 0)
        if new_head is not None:
            carry["amp_up"] = new_head
            print("[instantamp] Initialized the new recovery AMP discriminator. Restored the actor and other learner state.", flush=True)
        if new_gait is not None:
            carry["amp"] = new_gait
        xs.pi_mask = jax.tree.map(lambda _: False, carry["pi"])
    if full_path:
        if cfg.max_iterations and int(carry["it"]) >= cfg.max_iterations:
            raise ValueError("max_iterations must exceed the checkpoint's absolute iteration")
        if xs.sim.mixed:
            carry["key"] = jax.random.fold_in(carry["key"], carry["it"])
    if record_initialization and not full_path and not actor_path:
        run = Path(cfg.log_dir)
        initial = run / "initial_policy"
        save_policy(str(initial), jax.device_get(xs.export(carry)), {"it": 0, "env_steps": 0, "seed": cfg.seed})
        wm_hash = hashlib.sha256()
        for leaf in jax.tree.leaves(jax.device_get(carry["wm"])):
            wm_hash.update(np.asarray(leaf).tobytes())
        with initial.with_suffix(".npz").open("rb") as stream:
            policy_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        (run / "initialization.json").write_text(json.dumps(dict(seed=cfg.seed, iteration=0,
            initialization="seeded_random", actor_sha256=policy_hash, world_model_sha256=wm_hash.hexdigest(),
            actor_inputs=xs.actor_dim, action_outputs=xs.J, policy_gradient=(
                "BPTT through the learned world model and PPO on real rollouts" if xs.ppo else "BPTT through the learned world model"),
            ppo_envs=len(xs.ppo_envs) if xs.ppo else 0, imagined_rollouts=xs.B_im),
            indent=2) + "\n")
    print(f"[instantamp] {cfg.num_envs} envs, world-model input {xs.wm_in}, actor input {xs.actor_dim}, "
          f"buffer {sum(v.nbytes for v in jax.tree.leaves(carry['buf'])) / 2**30:.2f} GiB, device {jax.devices()[0]}", flush=True)  # fmt: skip
    t_c = time.time()
    progs = xs.compile_programs(carry)
    print(f"[instantamp] compile {time.time() - t_c:.0f} s, start-up {time.time() - t_launch:.0f} s", flush=True)
    gc.collect()
    gc.freeze()
    logf = open(os.path.join(cfg.log_dir, "log.jsonl"), "a", buffering=1)
    t0 = time.time()
    warmup = max(cfg.warmup_steps, xs.amp_span + cfg.ppo_len + 1) if xs.mixed_amp and xs.ppo else cfg.warmup_steps
    for _ in range(-(-warmup // cfg.env_steps)):
        carry, _ = progs["collect"](carry)
    last_ckpt, n_ckpt, prev_s = 0.0, 0, 0.0
    while True:
        cs, ls = [], []
        remaining = cfg.max_iterations - int(carry["it"]) if cfg.max_iterations else cfg.log_every
        block = min(remaining, cfg.log_every)
        if cfg.ckpt_iterations:
            block = min(block, cfg.ckpt_iterations - int(carry["it"]) % cfg.ckpt_iterations)
        for _ in range(block):
            carry, cst = progs["collect"](carry)
            ast = {}
            if "amp" in progs:
                carry, ast = progs["amp"](carry)
            wst = {}
            if "wm" in progs:
                carry, wst = progs["wm"](carry)
            carry, pst = progs["pi"](carry)
            cs.append(cst)
            ls.append({**wst, **pst, **ast})
        cs, ls, ev = jax.device_get((cs, ls, progs["wm_eval"](carry) if "wm_eval" in progs else {}))
        n_done = max(sum(float(x["done"]) for x in cs), 1.0)
        rew = np.sum([x["rew_sums"] for x in cs], 0) / n_done
        st = {"ep_len": sum(float(x["len_sum"]) for x in cs) / n_done, "done": n_done, "fell": sum(float(x["terminated"]) for x in cs) / n_done,
              **{f"rew_{k}": float(v) for k, v in zip(xs.sim.terms, rew)},
              **{k: float(np.mean([x[k] for x in cs])) for k in ("reward", "nacon", "nonfinite_env", "up_frac", "term_geom", "self_hits", "foot_contact")},
              **{k: sum(float(x[k]) for x in cs) for k in ("diverged", "down_done", "down_up", "pushes")},
              **{k: float(np.mean([x[k] for x in ls])) for k in ls[0]}, **{k: float(v) for k, v in ev.items()}}  # fmt: skip
        if xs.sim.wm_posture_and_tracking:
            planar, yaw, samples = np.sum([x["velocity_errors"] for x in cs], axis=0)
            if samples > 0:
                st["planar_velocity_error_m_s"] = float(planar / samples)
                st["yaw_rate_error_rad_s"] = float(yaw / samples)
        if xs.sim.wm_global_gait:
            curriculum = carry["env"].gait_curriculum
            st["gait_max_forward_m_s"] = int(curriculum.stage) + 1
            st["gait_last_tracking_error"], st["gait_last_forward_error"], st["gait_last_backward_error"], st["gait_last_fall_rate"] = map(float, curriculum.last_metrics)
        if xs.sim.mixed:
            for name in ("gait_falls", "rises"):
                st[name] = sum(int(x[name]) for x in cs)
        if getattr(cfg, "wm_floor_amp", False):
            st["gait_weight"] = float(np.mean([x["gait_weight"] for x in cs]))
        if record_initialization:
            st["actor_updates"] = int(carry["opt_pi"]["t"])
            st["wm_updates"] = int(carry["opt_wm"]["t"])
        if "action_std" in carry:
            std = np.asarray(carry["action_std"])
            st["action_std_by_joint"] = dict(zip(xs.sim.robot.DOF, std.tolist(), strict=True))
            st["action_std_mean"] = float(std.mean())
        train_s = time.time() - t0
        it, env_steps = int(carry["it"]), float(carry["env_steps"])
        logf.write(json.dumps({"it": it, "train_s": train_s, "wall_s": time.time() - t_launch, "env_steps": env_steps, "nonfinite": int(carry["nonfinite"]), **st}) + "\n")
        # len: mean length of finished episodes, fell: their share ended by a fall, lin / ang: tracking terms per episode second
        metrics = (f"stand {st['rew_stand']:.3f} gait {st['gait_weight']:.2f}" if getattr(cfg, "wm_floor_amp", False) else
                   f"lin {st['rew_track_lin']:.3f} ang {st['rew_track_ang']:.3f}")
        falls = f"falls {st['gait_falls']} rises {st['rises']}" if xs.sim.mixed else f"fell {st['fell']:.2f}"
        learner = (f"wm {st['wm']:.4f} e8 {st['wm8']:.4f} | G {st['G']:.3f} r_im {st['r']:.4f}")
        print(f"it {it} | {train_s / 60:.2f} min | {env_steps / 1e6:.1f}M | len {st['ep_len']:.0f} {falls} {metrics} r {st['reward']:.4f} | up {st['up_frac']:.2f} get-ups {st['down_up']:.0f}/{st['down_done']:.0f} | {learner} "
              f"surv {st['surv']:.3f} gn {st['gn']:.2f} | {block / max(train_s - prev_s, 1e-9):.1f} it/s"
              + (f" | h_err {st['h_err8'] * 100:.1f}/{st['h_errK'] * 100:.1f} cm term env {st['fall_env']:.3f} wm {st['fall_wm']:.3f}" if "wm_eval" in progs else "")
              + (f" | amp d {st['amp_d_demo']:.2f}/{st['amp_d_robot']:.2f} r {st['r_amp']:.4f} use {st['amp_usable']:.2f} ramp {st['amp_ramp']:.2f}" if xs.amp else "")
              + (f" | getup d {st['ampg_d_demo']:.2f}/{st['ampg_d_robot']:.2f} use {st['ampg_usable']:.2f}" if xs.amp and xs.getup else "")
              + (f" | V end {st['v_end']:.2f} loss {st['vloss']:.4f}" if xs.use_critic else "")
              + (f" | ppo clip {st['ppo_clipfrac']:.2f} |logr| {st['ppo_logr']:.2f} gn {st['ppo_gn']:.3f} r {st['ppo_r']:.4f}" if xs.ppo else "")
              + (f" std {st['action_std_mean']:.3f}" if "action_std_mean" in st else "")
              + (f" cos bptt {st['ppo_cos_bp']:.3f} halves {st['ppo_cos_half']:.3f}" if "ppo_cos_bp" in st else "")
              + (f" | gated wm {st['wm_gated']:.2f} gn {st['gn_gated']:.2f}" if st["wm_gated"] or st["gn_gated"] else "")
              + (f" | diverged {st['diverged']:.0f}" if st["diverged"] else ""), flush=True)  # fmt: skip
        prev_s = train_s
        finished = (cfg.minutes > 0 and train_s >= 60 * cfg.minutes) or (cfg.max_iterations > 0 and it >= cfg.max_iterations)
        checkpoint_due = (it % cfg.ckpt_iterations == 0 if cfg.ckpt_iterations else
                          train_s - last_ckpt >= 60 * cfg.ckpt_minutes)
        if checkpoint_due or finished:
            save_full(os.path.join(cfg.log_dir, "ckpt", "full", f"{n_ckpt:04d}"), carry)
            save_policy(os.path.join(cfg.log_dir, "ckpt", f"{n_ckpt:04d}"), jax.device_get(xs.export(carry)),
                        {"it": it, "train_s": train_s, "env_steps": env_steps}, ode=(xs.ode_dmax, xs.ode_amax) if xs.ode else None)  # fmt: skip
            prune_full_checkpoints(Path(cfg.log_dir) / "ckpt/full", cfg.full_ckpt_keep)
            last_ckpt, n_ckpt = train_s, n_ckpt + 1
        if finished:
            break
    print(f"done: {it} iterations, {env_steps / 1e6:.1f}M env steps, {train_s / 60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
