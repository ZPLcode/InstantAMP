"""InstantAMP training: a world-model learner (SuperMimic's method in XSoma's JAX implementation) with adversarial motion priors on the
Unitree G1 AMP-Flat environment (env.py, sim.py).

One iteration: collect `env_steps` steps of all envs; `amp_disc_steps` steps of each motion prior's critic; `wm_batches` world-model fits;
`pi_updates` policy updates, each the gradient of the discounted return of `horizon` imagined steps (BPTT through the world model), plus
optionally a PPO gradient on a share of real collector steps (`ppo_frac`). In imagination the environment's own functions read every
imagined state: the observation frame (with its noise), the command (heading control and resampling), the rewards, and the termination
and get-up logic (contact termination by collider clearance once up, the 5-s get-up deadline while down; exact value, sigmoid gradient).
Two quantities have no geometry in the imagined state and are replaced: the foot contact of the slip penalty (a sigmoid of the soles'
clearance) and the self-collision count (0).

The reward the policy maximizes in imagination is r = (1 - amp_lerp) r_task + amp_lerp dt (ramp gate_loco style_loco + gate_getup
style_getup): a locomotion style from the walking / running clips while the robot is up, and a get-up style from the LAFAN1 get-up clips
while it has not been up yet.

The actor observations are unscaled (joint velocities of tens of rad/s), so they are normalized with running statistics (`obs_norm`,
updated from collected observations, fixed inside an update).
"""

from __future__ import annotations

from pathlib import Path

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

# JAX reserves 30 % of the GPU (its default is 75 %); the rest is left to MuJoCo Warp, which allocates its own buffers at run time. An
# XLA_PYTHON_CLIENT_MEM_FRACTION set by the caller wins.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import jax  # noqa: E402
import jax.numpy as jnp
import numpy as np

from . import amp, critic, env, g1_getup, nets, optim, support_discriminator, zenbot
from . import buffer as rb
from . import zenbot_gait_rewards as gait_rewards
from .zenbot_gait_demos import reference_command_index
from .sim import Sim, motion_dirs, unflatten
from .zenbot_motion_train_obs import MOTION_TRAIN_WIDTH, mirror_action, mirror_observation, motion_train_actor_mean

# error floors of the world-model loss for the state's position, rotation, velocity and angular velocity (then q: 3e-3, qd: 5e-2)
ERR_FLOOR = ((3, 3e-3), (3, 5e-3), (3, 2e-2), (3, 5e-2))


def widths(s: str):
    """Layer widths from a comma-separated string: "512,512,512" gives (512, 512, 512)."""
    return tuple(int(x) for x in s.split(","))


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
    """Every option; each field is also a command-line flag of the same name (`--name value`). The defaults are the recommended
    recipe for 30 minutes of training on one GPU."""

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
    seed: int = 0
    log_dir: str = ""
    jax_cache: str = "~/.cache/instantamp/jax"
    xla_flags: str = "--xla_gpu_enable_triton_gemm=false"

    robot: str = "g1"  # select the articulation and motion banks
    g1_style: int = 0  # train the G1 get-up prior on segmented rises, including the transition to standing
    init_g1_full: str = ""  # initialize this recipe from a legacy G1 ckpt/full/NNNN.npz

    def __post_init__(self):
        if self.ckpt_iterations < 0:
            raise ValueError("ckpt_iterations must be nonnegative")
        if (self.g1_style and self.robot != "g1") or (self.init_g1_full and not self.g1_style):
            raise ValueError("The G1 recovery recipe and its initializer require --robot g1 --g1_style 1")
        walk, floor = motion_dirs(self.robot)
        self.walk_dir = self.walk_dir or walk
        self.amp_dir = self.amp_dir or walk
        self.floor_dir = self.floor_dir or floor
        self.amp_getup_dir = self.amp_getup_dir or (self.floor_dir if self.g1_style or self.robot == zenbot.MIXED_TASK else floor)
        self.log_dir = self.log_dir or ("runs/instantamp" if self.robot == "g1" else f"runs/{self.robot}")
        full_path = getattr(self, "init_full", "")
        if full_path and Path(full_path).resolve().parent == (Path(self.log_dir) / "ckpt/full").resolve():
            raise ValueError("Full-checkpoint continuation requires a new log directory")


@dataclasses.dataclass
class G1StyleConfig(Config):
    """G1 locomotion baseline with a stronger prior on complete recovery transitions."""
    g1_style: int = 1
    floor_dir: str = "data/recovery/g1_rises/train"
    log_dir: str = "runs/g1_amp_style"
    reset_floor: float = 0.5
    imag_down: float = 0.35
    amp_getup_full: int = 1
    amp_delay: int = 0
    amp_ramp: int = 0
    pi_lr: float = 0.00008
    getup_reward_scale: float = 20.
    getup_disc_lr: float = 0.0005
    g1_getup_prior: str = "native"  # native | legged | legged_time (same 85-D LSGAN plus one native clock input)
    g1_phase_share: float = 0.  # share of recovery style from a discriminator paired to the native clip clock
    g1_phase_heading: int = 0  # include torso orientation relative to the reset heading in the phase classifier
    g1_feature_share: float = 0.  # optional matched-window similarity in the phase discriminator's first hidden layer
    g1_time_negatives: float = 0.  # probability of a distinct, wrong-time expert negative in the phase discriminator
    getup_duration_s: float = 4.
    reference_weight: float = 0.  # auxiliary pose reward on the reset clip clock, training only
    getup_reset_frames: int = 20  # number of eligible opening frames in each recovery clip
    g1_support_friction: int = 0  # lateral support for hand and knee contacts with the floor
    getup_rise_weight: float = -0.25
    getup_style_ramp: int = 0  # ramp only the new recovery head over these iterations
    getup_hold_score: float = 0.  # expert score fraction after a window of settled standing, or zero to keep scoring every window
    getup_anchor: float = 0.  # frozen baseline action KL coefficient on real PPO observations

    def __post_init__(self):
        super().__post_init__()
        if self.g1_getup_prior not in ("native", "legged", "legged_time"):
            raise ValueError("g1_getup_prior must be native, legged or legged_time")
        if self.g1_getup_prior != "native" and (self.amp_frames != 10 or self.amp_w <= 0 or not self.amp_getup
                or self.reset_floor <= 0 or self.reference_weight or self.g1_phase_share or self.getup_hold_score
                or Path(self.amp_getup_dir).resolve() != Path(self.floor_dir).resolve()):
            raise ValueError("Legged G1 AMP needs matching segmented resets and demos, one 10-frame recovery discriminator, and zero reference_weight")
        if not 0 <= self.g1_phase_share <= 1 or (self.g1_phase_share and
                (not self.amp_getup or self.amp_w <= 0 or self.reset_floor <= 0 or self.getup_hold_score)):
            raise ValueError("Phase AMP requires active recovery AMP, floor resets, and a learned standing score")
        if (not 0 <= self.g1_feature_share < 1 or not 0 <= self.g1_time_negatives < 1
                or self.g1_phase_heading not in (0, 1)
                or (self.g1_phase_heading or self.g1_feature_share or self.g1_time_negatives) and not self.g1_phase_share):
            raise ValueError("G1 feature matching and time-shift negatives require an active phase discriminator")
        if self.g1_phase_share and Path(self.amp_getup_dir).resolve() != Path(self.floor_dir).resolve():
            raise ValueError("Phase AMP requires the same segmented native bank for resets and expert windows")
        if self.getup_reset_frames < 1 or self.reference_weight < 0 or not np.isfinite(self.reference_weight):
            raise ValueError("Expected a positive getup_reset_frames and a finite nonnegative reference_weight")
        if self.reference_weight and not self.reset_floor > 0:
            raise ValueError("Aligned recovery rewards require floor resets")
        if not np.isfinite(self.getup_duration_s) or self.getup_duration_s <= 0:
            raise ValueError("getup_duration_s must be finite and positive")
        if self.init_g1_full and (self.amp_w <= 0 or not self.critic):
            raise ValueError("The full G1 initializer requires the walking AMP head and critic")
        if self.getup_anchor and (not self.init_g1_full or self.ppo_frac <= 0):
            raise ValueError("getup_anchor requires a baseline --init_g1_full and real PPO samples")
        if not np.isfinite(self.getup_anchor) or self.getup_anchor < 0 or self.getup_style_ramp < 0 or not 0 <= self.getup_hold_score <= 1:
            raise ValueError("Invalid G1 recovery anchor, ramp or standing score")


@dataclasses.dataclass
class RecoveryConfig(Config):
    """LeggedLab get-up tasks trained with the InstantAMP world model and hybrid policy optimizer."""
    reset_floor: float = 1.
    prob_shove: float = 0.
    push: int = 0
    random_start: int = 0
    cmd_fresh: float = 0.
    amp_getup: int = 0  # the primary discriminator already covers the full recovery
    amp_gate: str = "none"
    amp_delay: int = 0
    amp_ramp: int = 0
    amp_lr: float = 1e-3
    amp_style_scale: float = 50.
    critic_hidden: str = "512,256,128"
    ppo_len: int = 24
    warmup_steps: int = 40
    max_iterations: int = 100_000
    minutes: float = 0.
    init_policy: str = ""  # load the exported actor and its normalization
    init_full: str = ""  # restore the learner into a new run with a fresh replay buffer
    assist_newtons: float = 0.

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.assist_newtons) or self.assist_newtons < 0.:
            raise ValueError("assist_newtons must be finite and nonnegative")


@dataclasses.dataclass
class ZenbotWalkRunRecoveryConfig(Config):
    """Continuous command tracking, falls and unassisted rises with two native motion priors."""
    robot: str = zenbot.MIXED_TASK
    motion_train_wm: int = 0
    reset_floor: float = .35
    zenbot_episode_steps: int = 400
    amp_delay: int = 100
    amp_ramp: int = 600
    pi_lr: float = .00012
    init_full: str = ""
    init_policy: str = ""  # optional trainable single actor with this observation and action layout
    recovery_amp_seconds: float = 4.0  # include the transition to balance after any fall
    recovery_amp_stride: int = 3  # ten unpaired poses spanning 0.54 s
    recovery_amp_scale: float = 6.
    gait_amp_stride: int = 1
    gait_ls_scale: float = 0.  # positive: least-squares gait AMP on matched finite-difference features
    gait_command_weight: float = 0.  # physical speed progress, per second, for the single-actor PPO learner
    recovery_amp_loss: str = "least_squares"
    legged_prior_file: str = ""  # converted, frozen native 10-frame LeggedLab discriminator
    legged_real_ppo: int = 0  # train actor and value only on fresh physical AMP rollouts
    recovery_base_actor: str = ""  # fixed AMP gait policy for bounded recovery specialization
    recovery_actor_share: float = .65
    support_rsi: float = 0.  # fraction of floor starts drawn from intermediate non-upright AMP states
    fall_push_speed: float = 0.  # optional gait perturbation magnitude in m/s

    @property
    def command_gait_amp(self):
        return bool(getattr(self, "wm_floor_amp", False) and not self.wm_getup_only and self.gait_ls_scale)

    def __post_init__(self):
        super().__post_init__()
        if self.recovery_amp_seconds and (self.amp_w <= 0 or not self.amp_getup):
            raise ValueError("The mixed recovery AMP recipe requires both motion discriminators")
        if self.recovery_amp_stride < 1 or not np.isfinite(self.recovery_amp_scale) or self.recovery_amp_scale <= 0:
            raise ValueError("Use a positive recovery discriminator stride and scale")
        if self.recovery_amp_loss not in ("least_squares", "wasserstein", "legged_support"):
            raise ValueError("Unknown recovery AMP discriminator loss")
        if (not np.isfinite(self.gait_ls_scale) or self.gait_ls_scale < 0 or
                not 1 <= self.gait_amp_stride <= self.recovery_amp_stride or
                (self.gait_amp_stride != 1 and not self.gait_ls_scale) or
                (self.gait_ls_scale and not (self.legged_real_ppo or self.command_gait_amp)) or
                not np.isfinite(self.gait_command_weight) or self.gait_command_weight < 0 or
                (self.gait_command_weight and not self.legged_real_ppo)):
            raise ValueError("Least-squares gait AMP requires a supported gait task and a stride within the recovery history")
        if self.recovery_base_actor and (not self.ode or self.recovery_amp_loss != "legged_support" or not 0. < self.recovery_actor_share <= 1.):
            raise ValueError("A recovery base actor requires ODE controls, LeggedLab support AMP and a share in (0, 1]")
        if self.recovery_amp_loss == "legged_support" and (not self.recovery_amp_seconds or self.amp_frames < 2):
            raise ValueError("LeggedLab recovery AMP requires an active recovery interval and at least two poses")
        if self.legged_real_ppo:
            if self.recovery_amp_loss != "legged_support" or self.recovery_base_actor or not self.critic:
                raise ValueError("Real PPO requires a single mixed legged_support actor and a critic")
            if self.env_steps != self.ppo_len or self.ppo_len < 2 or self.pi_updates < 1:
                raise ValueError("Real PPO needs env_steps equal to ppo_len and at least one policy epoch")
            if self.action_noise <= 0 or not 0 <= self.det_share < 1:
                raise ValueError("Real PPO needs stochastic physical rollouts")
        if self.legged_prior_file:
            if self.recovery_amp_loss != "legged_support" or self.amp_frames != 10 or not self.amp_wrists:
                raise ValueError("The frozen LeggedLab prior requires legged_support with ten full-joint frames")
            if not Path(self.legged_prior_file).is_file():
                raise ValueError(f"Missing converted LeggedLab discriminator: {self.legged_prior_file}")
        span = (self.amp_frames - 1) * self.recovery_amp_stride + 1
        if not np.isfinite(self.recovery_amp_seconds) or (self.recovery_amp_seconds and self.recovery_amp_seconds < span * env.DT):
            raise ValueError("The recovery AMP interval must fit at least one complete window")
        if self.recovery_amp_seconds and self.buffer_steps < span + self.ppo_len + 1:
            raise ValueError("The buffer must hold the full recovery AMP window and PPO rollout")


@dataclasses.dataclass
class ZenbotMotionTrainWMConfig(ZenbotWalkRunRecoveryConfig):
    """Train the directly deployable Zenbot policy through the InstantAMP world model."""
    motion_train_wm: int = 1
    wm_reference_training: int = 0  # matched physical recovery poses are a training-only signal
    wm_floor_amp: int = 0  # cold floor openings, get-up AMP until sustained foot support
    wm_range_commands: int = 0  # continuous Motion Train velocity ranges for the joint task
    wm_posture_and_tracking: int = 0  # pelvis posture rewards and separate planar/yaw tracking diagnostics
    wm_gait_rewards: int = 0  # Motion Train foot and joint terms for the joint floor/gait task
    floor_stop_command_share: float = 0.  # fraction of fallen starts requesting a quiet stance for the episode
    gait_source_commands: int = 0  # practice the recorded and nominal directional speeds
    gait_mirror_weight: float = 0.  # mirror consistency on physical observations after supported standing
    wm_lift_gate: float = 0.  # pelvis height where positive lifting progress begins, in metres
    reference_weight: float = 0.  # score the native opening clip at its recorded speed during training
    wm_rate_ramp: int = 0  # first learn a supported rise, then ramp its speed tracking over the same number of updates
    wm_tempo_negatives: float = 0.  # fraction of recovery discriminator negatives made by mistiming moving reference segments
    wm_getup_only: int = 0  # a zero command throughout recovery and standing
    wm_stance_start_share: float = 0.  # portion of all resets drawn from supported physical stands
    wm_reset_imag_share: float = 0.  # portion of imagination starts in the first real WM window after reset
    wm_min_rise_score: float = 0.  # positive: end attempts that diverge from the matched reference during recovery
    wm_motion_replay: int = 0  # fit half of each dynamics batch on measured native recovery transitions
    ode: int = 0
    action_rate_l2_weight: float = -.015
    amp_gate: str = "none"
    max_iterations: int = 10000
    minutes: float = 0.
    seed: int = 42

    def __post_init__(self):
        super().__post_init__()
        if self.ode or self.legged_real_ppo or self.recovery_base_actor:
            raise ValueError("The Motion Train WM profile needs direct actions and a learned world model")
        joint_task = self.wm_floor_amp and not self.wm_getup_only
        for name in ("wm_range_commands", "wm_posture_and_tracking", "wm_gait_rewards"):
            enabled = getattr(self, name)
            if enabled not in (0, 1) or (enabled and not joint_task):
                raise ValueError(f"{name} must be 0 or 1 and requires joint floor/locomotion training when enabled")
        if self.gait_source_commands and not self.command_gait_amp:
            raise ValueError("Source-speed commands need command-conditioned locomotion AMP")
        if not 0 <= self.ppo_frac < 1:
            raise ValueError("ppo_frac must be in [0, 1) so imagination contributes policy updates")
        if self.ppo_frac and (not self.critic or not self.action_noise > 0 or
                              round(self.ppo_frac * self.num_envs) < 1 or round((1-self.ppo_frac)*self.pi_batch) < 1):
            raise ValueError("Hybrid PPO requires a critic, exploration noise, and real and imagined samples")
        if self.wm_reference_training not in (0, 1) or (self.wm_reference_training and not self.motion_train_wm):
            raise ValueError("wm_reference_training must be 0 or 1 and requires the Motion Train WM profile")
        if self.wm_getup_only not in (0, 1) or self.wm_motion_replay not in (0, 1):
            raise ValueError("wm_getup_only and wm_motion_replay must be 0 or 1")
        if self.wm_floor_amp and (self.wm_reference_training or self.wm_motion_replay
                                  or self.wm_stance_start_share or not self.ppo_frac or not self.amp_w or not self.amp_getup):
            raise ValueError("Floor-opening AMP needs hybrid training and online dynamics")
        if self.wm_floor_amp and not self.wm_getup_only and (self.reference_weight or not 0. < self.reset_floor < 1.):
            raise ValueError("Joint floor AMP needs unpaired style and both floor and gait starts")
        if not np.isfinite(self.reference_weight) or self.reference_weight < 0 or (self.reference_weight and not self.wm_floor_amp):
            raise ValueError("Native pose tracking requires floor AMP and a nonnegative finite reference_weight")
        if not 0. <= self.wm_lift_gate < zenbot.UP_HEIGHT or (self.wm_lift_gate and not self.wm_floor_amp):
            raise ValueError("A late lift gate must be below upright pelvis height and requires floor AMP")
        if self.wm_rate_ramp < 0 or (self.wm_rate_ramp and not (self.wm_floor_amp and self.reference_weight)):
            raise ValueError("A motion-rate ramp needs floor AMP with a reference tracking reward")
        if not 0. <= self.wm_tempo_negatives <= .5 or (self.wm_tempo_negatives and not self.wm_floor_amp):
            raise ValueError("Tempo negatives need floor AMP and a share between zero and one half")
        if self.wm_getup_only:
            self.reset_floor, self.prob_shove, self.push = 1., 0., 0
        if not 0. <= self.wm_stance_start_share <= .5 or (self.wm_stance_start_share and not self.wm_getup_only):
            raise ValueError("wm_stance_start_share must be in [0, .5] and requires the get-up stage")
        if not 0. <= self.wm_reset_imag_share <= 1. or (self.wm_reset_imag_share and
                (not (self.wm_getup_only or self.wm_floor_amp) or self.imag_down)):
            raise ValueError("Reset-priority imagination requires a recovery task, a share in [0, 1], and imag_down=0")
        if not 0. <= self.wm_min_rise_score < 1. or (self.wm_min_rise_score and not (self.wm_getup_only and self.wm_reference_training)):
            raise ValueError("A minimum rise score requires paired get-up training and a threshold in [0, 1)")
        if self.wm_motion_replay and self.wm_batch < 2:
            raise ValueError("Mixed physical dynamics replay needs at least two transitions per batch")
        if (not np.isfinite(self.gait_mirror_weight) or self.gait_mirror_weight < 0 or
                self.gait_mirror_weight and not (self.command_gait_amp and self.ppo_frac > 0)):
            raise ValueError("Upright mirror consistency requires joint floor and gait AMP with real PPO samples")
        if self.obs_clip != 10.:
            raise ValueError("The Motion Train deployment normalizes observations with a clip of 10")


@dataclasses.dataclass
class ZenbotGetUpConfig(RecoveryConfig):
    assist_newtons: float = zenbot.INITIAL_ASSIST
    zenbot_phase_resets: float = 0.  # probability of starting at an intermediate reference pose
    zenbot_episode_steps: int = 500
    action_rate_l2_weight: float = -0.005
    robot: str = "zenbot23_getup"
    reference_weight: float = 5.0  # per second while the selected native recovery clip is playing


@dataclasses.dataclass
class G1GetUpConfig(RecoveryConfig):
    task: str = g1_getup.TASK_ID
    action_rate_l2_weight: float = g1_getup.ACTION_RATE_L2_WEIGHT
    robot: str = "g1_getup"
    random_start: int = 1  # Source train.py staggers the first episode's timeout.
    ode: int = 0  # LeggedLab's policy directly specifies an offset from the default joint pose each step.
    amp_recent: int = 244  # source discriminator history buffer in policy steps
    buffer_steps: int = 244
    max_iterations: int = 500_000
    ckpt_iterations: int = 50
    assist_newtons: float = g1_getup.INITIAL_ASSIST
    reference_weight: float = 0.  # optional training reward for the reset clip and its final standing pose
    reference_hold_scale: float = 1.  # multiplier on that reward after the last source frame
    rise_anchor: float = 0.  # preserve the specified starting actor on source-matched, low-pelvis PPO steps

    def __post_init__(self):
        if not 0. <= self.reference_hold_scale <= 1.:
            raise ValueError("reference_hold_scale must be between 0 and 1")
        if not np.isfinite(self.rise_anchor) or self.rise_anchor < 0 or (self.rise_anchor and
                (not self.init_policy or self.ppo_frac <= 0 or self.reference_weight <= 0)):
            raise ValueError("A positive rise_anchor requires init_policy, an active reference reward and real PPO samples")
        self.amp_dir = self.amp_dir or self.floor_dir
        super().__post_init__()


class InstantAMP:
    @property
    def actor_dim(self):
        return MOTION_TRAIN_WIDTH if getattr(self.cfg, "motion_train_wm", False) else self.sim.obs_dim

    @property
    def legged_support(self):
        return getattr(self.cfg, "recovery_amp_loss", "") == "legged_support"

    @property
    def real_ppo(self):
        return bool(getattr(self.cfg, "legged_real_ppo", 0))

    @property
    def wm_tracks_reference(self):
        return bool(getattr(self.cfg, "wm_reference_training", False) or
                    (getattr(self.cfg, "wm_floor_amp", False) and getattr(self.cfg, "reference_weight", 0.)))

    def reference_rate_weight(self, iteration):
        length = self.cfg.wm_rate_ramp
        return jnp.clip((iteration - length) / length, 0., 1.) if length else 1.

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
        return 1 + (self.cfg.amp_frames - 1) * getattr(self.cfg, "gait_amp_stride", 1)

    def __init__(self, cfg: Config):
        self.cfg = c = cfg
        self.paired = bool(getattr(c, "g1_phase_share", 0.))
        self.legged = getattr(c, "g1_getup_prior", "native") != "native"
        self.recovery_clock = self.paired or self.legged
        set_jax(c.jax_cache)
        self.frozen_native_prior = (support_discriminator.load_legged_prior(c.legged_prior_file)
                                    if getattr(c, "legged_prior_file", "") else None)
        stand_clip = Path(c.amp_dir) / "stand_4s_120f.npz" if c.amp_w > 0 and getattr(c, "motion_train_wm", False) else None
        self.sim = Sim(c.num_envs, c.nconmax, c.njmax, c.walk_dir, c.floor_dir, bool(c.random_start), c.seed, c.reset_floor, c.prob_shove, bool(c.push),
                       robot=c.robot, amp_wrists=bool(c.amp_wrists), assist_newtons=getattr(c, "assist_newtons", None), reference_weight=getattr(c, "reference_weight", 0.),
                       g1_style=bool(c.g1_style), getup_rise_weight=getattr(c, "getup_rise_weight", -0.25),
                       g1_support_friction=bool(getattr(c, "g1_support_friction", 0)), getup_reset_frames=getattr(c, "getup_reset_frames", 20),
                       g1_phase_amp=self.recovery_clock, reference_hold_scale=getattr(c, "reference_hold_scale", 1.),
                       action_rate_l2_weight=getattr(c, "action_rate_l2_weight", None),
                       zenbot_phase_resets=getattr(c, "zenbot_phase_resets", 0.),
                       zenbot_episode_steps=getattr(c, "zenbot_episode_steps", 500),
                       recovery_amp_seconds=getattr(c, "recovery_amp_seconds", 0.), fall_push_speed=getattr(c, "fall_push_speed", 0.),
                       legged_support=self.legged_support, support_rsi=getattr(c, "support_rsi", 0.),
                       zenbot_motion_train_wm=bool(getattr(c, "motion_train_wm", False)),
                       zenbot_wm_reference_training=bool(getattr(c, "wm_reference_training", False)),
                       zenbot_wm_getup_only=bool(getattr(c, "wm_getup_only", False)),
                       wm_stance_start_share=getattr(c, "wm_stance_start_share", 0.),
                       wm_min_rise_score=getattr(c, "wm_min_rise_score", 0.), wm_floor_amp=getattr(c, "wm_floor_amp", 0),
                       wm_stand_reference=stand_clip, wm_lift_gate=getattr(c, "wm_lift_gate", 0.),
                       floor_stop_command_share=getattr(c, "floor_stop_command_share", 0.),
                       gait_source_commands=getattr(c, "gait_source_commands", False),
                       wm_gait_rewards=bool(getattr(c, "wm_gait_rewards", 0)),
                       wm_posture_and_tracking=bool(getattr(c, "wm_posture_and_tracking", 0)),
                       wm_range_commands=bool(getattr(c, "wm_range_commands", 0)))
        self.wm_motion_replay = None
        if getattr(c, "wm_motion_replay", False):
            from .zenbot_wm_motion_replay import PhysicalMotionReplay
            self.wm_motion_replay = PhysicalMotionReplay()
        self.J = self.JS = self.sim.J  # joints (actions, targets) and the world model's q, qd
        self.ode = bool(c.ode)
        self.ode_dmax = c.ode_dmax if c.ode_dmax > 0 else None
        self.ode_amax = c.ode_amax if c.ode_amax > 0 else None
        self.ground_base = None
        if getattr(c, "recovery_base_actor", ""):
            from .inference import load
            layers, norm, ode = load(c.recovery_base_actor)
            if (not isinstance(layers, list) or layers[0]["w"].shape[0] != self.sim.obs_dim or
                    len(layers[-1]["b"]) != self.J or ode is None or not np.allclose(
                        [v or 0. for v in ode], [self.ode_dmax or 0., self.ode_amax or 0.])):
                raise ValueError("Expected a single mixed Zenbot AMP actor with matching ODE limits")
            self.ground_base = (layers, norm)
        self.amp = c.amp_w > 0
        if self.amp:
            upright_source = stand_clip if getattr(c, "wm_getup_only", False) else c.amp_dir
            if getattr(c, "command_gait_amp", False):
                from .zenbot_gait_demos import CommandGaitDemos
                self.demos = CommandGaitDemos(upright_source, self.sim, self.gait_span)
            else:
                self.demos = amp.Demos(upright_source, self.sim, self.gait_span, per_clip=self.sim.mixed)
            assert c.amp_gate in ("track_shift", "moving", "track", "none"), c.amp_gate
            self.getup = bool(c.amp_getup)
            if self.getup:
                from .getup_amp import GetupDemos

                recovery_frame = (lambda state: self.sim.amp_support_frame(state, with_native=True)) if self.legged_support else None
                self.getup_demos = (amp.Demos(c.amp_getup_dir, self.sim, self.amp_span, per_clip=True,
                                             frame_fn=recovery_frame, extra_clip=None if getattr(c, "wm_floor_amp", False) else stand_clip) if self.sim.mixed else
                                    GetupDemos(c.amp_getup_dir, self.sim, c.amp_frames, segmented=bool(c.g1_style), legged=self.legged))
                if self.mixed_amp and c.recovery_amp_loss == "wasserstein":
                    self.getup_demos.prepare_ground_heights()
                if self.legged:
                    from .getup_amp import LeggedRecoveryPrior
                    self.legged_prior = LeggedRecoveryPrior(self.sim, self.getup_demos, c.g1_getup_prior == "legged_time")
                if self.paired:
                    from .g1_phase_amp import G1PhaseDemos
                    self.phase_demos = G1PhaseDemos(self.sim, self.getup_demos, heading_features=bool(c.g1_phase_heading))
            if self.sim.robot is g1_getup or getattr(c, "legged_real_ppo", 0):
                # The source discriminator sees windows from the entire behavior policy,
                # including the noisy rollouts that contribute real PPO gradients.
                self.amp_envs = np.arange(c.num_envs, dtype=np.int32)
            else:
                self.amp_envs = np.nonzero(~((np.arange(c.num_envs) * c.det_share) % 1.0 >= c.det_share))[0].astype(np.int32)
        self.use_critic = bool(c.critic)
        self.ppo = self.real_ppo or c.ppo_frac > 0
        self.B_im = int(round((1 - c.ppo_frac) * c.pi_batch)) if self.ppo else c.pi_batch  # imagined rollouts per update
        if self.ppo:
            assert self.use_critic, "the PPO advantages need the critic (--critic 1)"
            noisy = np.nonzero((np.arange(c.num_envs) * c.det_share) % 1.0 >= c.det_share)[0]
            n_ppo = len(noisy) if self.real_ppo else int(round(c.ppo_frac * c.num_envs))
            assert 0 < n_ppo <= len(noisy), "the PPO envs are taken among the noisy collector envs"
            self.ppo_envs = noisy[:n_ppo].astype(np.int32)
            if self.real_ppo and n_ppo * c.ppo_len < 4:
                raise ValueError("Real PPO needs at least four stochastic transitions per batch")
            self.ppo_w = 1. if self.real_ppo else (c.ppo_w if c.ppo_w >= 0 else c.ppo_frac / (1 - c.ppo_frac))
        self.n_soles = len(self.sim.sole_radius)
        self.err_floor = np.concatenate([np.full((n,), v, np.float32) for n, v in ERR_FLOOR] + [np.full((self.JS,), 3e-3, np.float32), np.full((self.JS,), 5e-2, np.float32)])

    # ------------------------------------------------------------------ state
    def init(self, key):
        c, J = self.cfg, self.J
        ks = jax.random.split(key, 4)
        e = self.sim.init(ks[0])
        rec_t = jax.eval_shape(lambda s: self.record(s, *self.sim.step(s, jnp.zeros((c.num_envs, J))), jnp.zeros((c.num_envs, J)), jnp.zeros((c.num_envs, J)))[0], e)
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
        if self.amp:
            gait_dim = self.sim.amp_dim + (3 if getattr(c, "command_gait_amp", False) else 0)
            carry["amp"] = amp.init(jax.random.fold_in(ks[3], 7), c.amp_frames, frame_dim=gait_dim)
            if self.getup:
                key_up = jax.random.fold_in(ks[3], 13)
                carry["amp_up"] = (self.legged_prior.init(key_up) if self.legged else
                                   amp.init(key_up, c.amp_frames, frame_dim=self.getup_demos.frames.shape[-1]))
                if self.legged_support:
                    native_dim = self.getup_demos.frames.shape[-1] - amp.SUPPORT_FRAME_DIM
                    carry["amp_up"] = {
                        "native": amp.init(key_up, c.amp_frames, frame_dim=native_dim),
                        "support": support_discriminator.init(jax.random.fold_in(key_up, 1), c.amp_frames,
                                                              amp.ORIENTED_SUPPORT_DIM if self.real_ppo else amp.SUPPORT_FEATURE_DIM),
                    }
                    if self.frozen_native_prior is not None:
                        carry["amp_up"]["native"].update(self.frozen_native_prior)
        if self.paired:
            carry["amp_phase"] = self.phase_demos.init(jax.random.fold_in(ks[3], 21))
            carry["amp_up"]["norm"] = nets.rn_update(carry["amp_up"]["norm"], self.getup_demos.frames)
        if self.use_critic:
            self.critic_dim = 13 + 2 * self.JS + 3 + J + self.xi_dim + 2
            if self.sim.robot is g1_getup:
                self.critic_dim = e.critic_obs.shape[-1]
            carry["critic"] = critic.init(jax.random.fold_in(ks[3], 11), self.critic_dim, widths(c.critic_hidden))
        if c.g1_style and c.getup_anchor:
            carry["anchor"] = {"pi": jax.tree.map(jnp.copy, pi), "obs_norm": nets.rn_init(self.sim.obs_dim)}
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
        if self.ground_base is not None:
            policy.update(ground_base=self.ground_base, ground_max_share=self.cfg.recovery_actor_share)
        return policy

    def act(self, pi, obs_norm, obs, a_prev, out_bf16=False):
        if getattr(self.cfg, "motion_train_wm", False):
            return motion_train_actor_mean(pi, obs_norm, obs)
        out = nets.pi_apply(pi, self.norm_obs(obs_norm, obs), out_bf16=out_bf16)
        if self.ground_base is not None:
            from .zenbot_ground_policy import combine
            base, norm = self.ground_base
            base_out = nets.pi_apply(base, self.norm_obs(norm, obs), out_bf16=out_bf16)
            out = combine(base_out, out, obs, self.cfg.recovery_actor_share)
        return nets.ode_step(out, a_prev, self.ode_dmax, self.ode_amax) if self.ode else out

    # ------------------------------------------------------------------ collection
    def record(self, s, s1, tr, a, mu):
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
        if self.sim.reference is not None or getattr(self.cfg, "wm_reference_training", False):
            rec["reference_clip"] = s.reference_clip
            if s.reference_start is not None:
                rec["reference_start"] = s.reference_start
            if s.reference_yaw is not None:
                rec["reference_yaw"] = s.reference_yaw
            if s.episode_limit is not None:
                rec["episode_limit"] = s.episode_limit
        if self.cfg.g1_style:
            rec["down_start"] = s.down
        if self.mixed_amp:
            rec["recovery_left"] = s.recovery_left
        if self.sim.robot is g1_getup:
            rec["obs_critic"] = s.critic_obs
            if getattr(self.cfg, "rise_anchor", 0.):
                frame = s.reference_start + s.t + 1
                pose_score = tr.parts[:, self.sim.terms.index("motion_reference")] / (env.DT * self.cfg.reference_weight)
                rec["rise_anchor_weight"] = g1_getup.rise_anchor_weight(self.sim.reference, s.reference_clip, frame, pose_score)
        if getattr(self.cfg, "wm_floor_amp", False):
            rec["stand_blend"] = tr.stats["stand_blend"]
            if self.sim.wm_floor_locomotion:
                rec["gait_world"] = self.sim.wm_gait_worlds
            rec["upright_steps"] = s.supported_hold.upright_steps
            rec["loaded_feet"] = s.supported_hold.loaded
            rec["episode_limit"] = s.episode_limit
            rec.update(a=a, mu=mu, rlin=jnp.zeros_like(tr.reward))
        elif self.ppo and self.sim.recovery:
            rec.update(a=a, mu=mu)
        elif self.ppo:  # the action taken, the mean it was drawn around, and the unweighted tracking term (the locomotion style's gate)
            i = env.TERMS.index("track_lin")
            rec.update({"a": a, "mu": mu, "rlin": tr.parts[:, i] / (env.DT * env.WEIGHTS["track_lin"])})
        return rec, tr.stats

    def ingest(self, carry, flat):
        if self.real_ppo:
            # Preserve the behavior policy's observation normalization across its rollout and PPO epochs.
            return {**carry, "buf": rb.add(carry["buf"], flat), "env_steps": carry["env_steps"] + self.cfg.num_envs}
        rec = rb.nest(flat)
        ok = rec["ok"] > 0
        dynamics, wm_ok = rec, ok
        if self.wm_motion_replay is not None:
            sample_key = jax.random.fold_in(carry["key"], carry["env_steps"].astype(jnp.uint32))
            measured = self.wm_motion_replay.sample_windows(sample_key, self.cfg.num_envs, 1)
            dynamics = {k: jnp.concatenate((rec[k], v[:, 0])) for k, v in measured.items() if k != "mask"}
            wm_ok = jnp.concatenate((ok, measured["mask"][:, 0] > 0))
        S0 = unflatten(dynamics["s_qpos"], dynamics["s_qvel"])
        S1 = unflatten(dynamics["s1_qpos"], dynamics["s1_qvel"])
        v0 = self.sim.view_state(S0)
        wb = carry["wm_b"]
        upd = lambda st, x: nets.rn_update(st, x, weights=wm_ok)  # noqa: E731
        wm_b = {"in_norm": upd(wb["in_norm"], nets.wm_inputs(S0, v0.height, v0.soles, dynamics["target"], dynamics["xi"])),
                "out_norm": upd(wb["out_norm"], nets.wm_targets(S0, S1, env.DT)), "inc_norm": upd(wb["inc_norm"], nets.increments(S0, S1))}  # fmt: skip
        freeze_norm = self.sim.recovery and (getattr(self.cfg, "init_policy", "") or getattr(self.cfg, "init_full", ""))
        obs_norm = carry["obs_norm"] if freeze_norm else nets.rn_update(carry["obs_norm"], rec["obs"], weights=ok)
        return {**carry, "buf": rb.add(carry["buf"], flat), "wm_b": wm_b, "obs_norm": obs_norm,
                "env_steps": carry["env_steps"] + self.cfg.num_envs}  # fmt: skip

    def collect_step(self, carry, key):
        c = self.cfg
        s = carry["env"]
        noisy = (jnp.arange(c.num_envs) * c.det_share) % 1.0 >= c.det_share
        obs = s.motion_train_sensors.observation() if getattr(c, "motion_train_wm", False) else s.obs
        clean = self.act(carry["pi"], carry["obs_norm"], obs, s.a_ode)
        a = clean + jnp.where(noisy, c.action_noise, 0.0)[:, None] * jax.random.normal(key, (c.num_envs, self.J))
        if getattr(c, "wm_rate_ramp", 0):
            s1, tr = self.sim.step(s, a, clean, reference_rate_weight=self.reference_rate_weight(carry["it"]))
        else:
            s1, tr = self.sim.step(s, a, clean)
        rec, stats = self.record(s, s1, tr, a, clean)
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
        if self.wm_motion_replay is None:
            w = rb.sample_windows(carry["buf"], key, c.wm_batch, c.wm_window)
        else:
            online_key, motion_key = jax.random.split(key)
            n_motion = c.wm_batch // 2
            online = rb.sample_windows(carry["buf"], online_key, c.wm_batch - n_motion, c.wm_window)
            measured = self.wm_motion_replay.sample_windows(motion_key, n_motion, c.wm_window)
            w = {k: jnp.concatenate((online[k], v)) for k, v in measured.items()}
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
        if self.sim.recovery or self.sim.mixed:
            f_wm = jnp.zeros_like(f_env)
        return {"h_err8": mean(herr, 7), "h_errK": mean(herr, K - 1), "fall_env": (f_env * m).sum() / jnp.maximum(m.sum(), 1),
                "fall_wm": (f_wm * m).sum() / jnp.maximum(m.sum(), 1)}  # fmt: skip

    # ------------------------------------------------------------------ imagination
    def sample_starts(self, buf, key):
        c, B = self.cfg, self.B_im
        k_s, k_u, k_c = jax.random.split(key, 3)
        start = rb.nest(rb.sample_starts(buf, k_s, B, self.amp_span - 1 if self.amp else 0, down_frac=c.imag_down,
                                         reset_frac=getattr(c, "wm_reset_imag_share", 0.), reset_window=getattr(c, "wm_window", 8)))
        if self.sim.recovery:
            start["cmd"] = env.Cmd(**start["cmd"])
            return start
        fresh = jax.random.uniform(k_u, (B,)) < c.cmd_fresh
        if self.recovery_clock:
            fresh &= ~start["down_start"]
        motion_train = getattr(c, "motion_train_wm", False)
        cmd = self.sim.sample_wm_command(k_c, B) if motion_train else env.sample_cmd(k_c, B, self.sim.cmd_ranges)
        old = env.Cmd(start["cmd"]["vel"], start["cmd"]["heading"], start["cmd"]["is_heading"] > 0, start["cmd"]["is_standing"] > 0, start["cmd"]["time_left"])
        start["cmd"] = env.where_cmd(fresh, cmd, old)
        if self.recovery_clock:
            start["cmd"] = self.sim.recovery_command(start["cmd"], start["down_start"])
        S0 = unflatten(start["s_qpos"], start["s_qvel"])
        if motion_train:
            current = self.sim.wm_velocity_command(start["cmd"], S0[1], start["was_up"] > 0, start.get("recovery_left"))
            # Term-major 234: gyro[0:9], gravity[9:18], then three xyz commands oldest first.
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

    def critic_in(self, S, h, a_last, c_obs, xi, up, t, down_start=None, recovery_left=None):
        """The critic's input: the invariant state, command, last action, context, whether the robot has been up and the share of its time
        to rise that is left (0 once it has been up)."""
        upf = up.astype(jnp.float32)
        left = jnp.clip(1 - t.astype(jnp.float32) / env.GETUP_STEPS, 0.0, 1.0) * (1 - upf)
        if recovery_left is not None:
            left = recovery_left / self.sim.recovery_amp_steps
        if self.cfg.g1_style and self.cfg.amp_w > 0 and self.cfg.amp_getup:
            style_left = jnp.clip(1 - t * env.DT / self.cfg.getup_duration_s, 0., 1.)
            left += upf * down_start * style_left
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
        if self.legged_support:
            return self.sim.amp_support_frame(state, with_gait=True, with_native=True)
        if getattr(self, "legged", False):
            return self.getup_demos.policy_frames(state)
        if getattr(self, "paired", False):
            return self.phase_demos.frame(state)
        if self.cfg.g1_style:
            from .getup_amp import getup_frame
            return getup_frame(self.sim, state)
        return self.sim.amp_frame(state)

    def getup_gate(self, up, down_start, elapsed):
        if self.cfg.g1_style:
            if not self.cfg.amp_getup:
                return jnp.zeros_like(up, dtype=jnp.float32)
            # A quick first stand still gets judged on its transition to balance.
            return (down_start & (elapsed <= round(self.cfg.getup_duration_s / env.DT))).astype(jnp.float32)
        return 1. - up.astype(jnp.float32)

    def getup_style(self, st, windows):
        if self.mixed_amp:
            features = self.ground_features(windows)
            if self.cfg.recovery_amp_loss == "wasserstein":
                value = amp.score(st["net"], st["norm"], features)
                spread = jnp.maximum(4., .25 * jnp.abs(st["mu"][0] - st["mu"][1]))
                return self.cfg.recovery_amp_scale * jnp.tanh((value - st["mu"][1]) / spread)
            return amp.recovery_style(st, features, self.cfg.recovery_amp_scale)
        if self.cfg.g1_style:
            c = self.cfg
            score = amp.recovery_style(st, windows, c.getup_reward_scale)
            if c.getup_hold_score:
                # Upright holds beyond the short tails in the expert bank should still earn style credit.
                gravity_z = windows[..., 2 * len(self.sim.amp_joints) + 5]
                settled = ((windows[..., -2] >= 0.72) & (gravity_z < -0.90) & (jnp.abs(windows[..., -1]) < 0.22)).all(-1)
                score = jnp.where(settled, c.getup_hold_score * c.getup_reward_scale, score)
            return score * st["ramp"] if c.getup_style_ramp else score
        return amp.style(st, amp.score(st["net"], st["norm"], windows), self.cfg.amp_style_scale)

    def legged_support_inputs(self, raw):
        native = raw[..., -self.cfg.amp_frames:, :-amp.SUPPORT_FRAME_DIM]
        support = raw[..., ::self.recovery_stride, -amp.SUPPORT_FRAME_DIM:]
        return native, amp.support_window(support, self.recovery_stride * env.DT, oriented=self.real_ppo)

    def legged_support_reward(self, st, windows, remaining):
        # Repeat the first physical pose for history before the fall or floor reset.
        elapsed = self.sim.recovery_amp_steps - remaining
        first = jnp.maximum(0, self.amp_span - 1 - elapsed)
        idx = jnp.maximum(jnp.arange(self.amp_span), first[..., None])
        raw = windows[..., self.sim.amp_dim:]
        raw = jnp.take_along_axis(raw, idx[..., None], axis=-2)
        native, support = self.legged_support_inputs(raw)
        scale = self.cfg.recovery_amp_scale
        if self.cfg.legged_prior_file:
            score = support_discriminator.legged_logit(st["native"], native)
            style = support_discriminator.legged_generator_reward(score, scale)
        else:
            style = amp.recovery_style(st["native"], native, scale, leggedlab=True)
        return style + support_discriminator.reward(st["support"], support, scale * .45)

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

    def paired_style(self, native, paired, windows, clip, source, yaw, elapsed):
        c = self.cfg
        if self.legged:
            score = self.legged_prior.style(native, windows[..., self.sim.amp_dim:], clip, source, elapsed, c.getup_reward_scale)
            return score * native["ramp"] if c.getup_style_ramp else score
        ordinary = self.getup_style(native, windows[..., :self.phase_demos.base_dim])
        conditioned, valid = self.phase_demos.condition_robot(windows, clip, source, yaw, elapsed)
        matched = amp.recovery_style(paired, conditioned, c.getup_reward_scale)
        if c.g1_feature_share:
            feature_score, _ = self.phase_demos.matched_feature_reward(paired, windows, clip, source, yaw, elapsed,
                                                                      scale=c.getup_reward_scale)
            matched = (1-c.g1_feature_share)*matched + c.g1_feature_share*feature_score
        if c.getup_style_ramp:
            matched *= native["ramp"]
        return jnp.where(valid, (1-c.g1_phase_share)*ordinary + c.g1_phase_share*matched, 0.)

    def imagine(self, pi, wm, wm_b, obs_norm, start, keys, amp_st=None, critic_st=None, amp_up_st=None, phase_st=None):
        c, B, sim = self.cfg, self.B_im, self.sim
        motion_train = getattr(c, "motion_train_wm", False)
        paired_floor = self.wm_tracks_reference
        if motion_train:
            from .zenbot_wm_sensors import pd_targets, predicted_sensors, restore_sensors

        bf16 = c.mm_bf16 in ("imag", "all")
        S0 = unflatten(start["s_qpos"], start["s_qvel"])
        xi, bias = start["xi"], start["bias"]
        use_amp, use_critic = amp_st is not None, critic_st is not None
        if use_amp:
            hist_s = unflatten(start["h_qpos"], start["h_qvel"])
            hf = self.disc_frame(hist_s)
            f0 = self.disc_frame(S0)
            win0 = jnp.concatenate((hf, f0[:, None]), 1)
            ramp = amp_st["ramp"]
        zeros = jnp.zeros((B,))

        def step(carry, k):
            S, a_last, a_ode, hist, cmd, c_obs, t, up, critic_hist, remaining, paired_clip, hold, step_history = carry
            k_o, k_c = jax.random.split(k)
            obs = hist.observation() if motion_train else hist
            a = self.act(pi, obs_norm, obs, a_ode, out_bf16=bf16)
            v = sim.view_state(S)
            if motion_train:
                target = pd_targets(a, hist.frames[..., -1, 55:], hist.frames[..., -2, 55:], bias, xi,
                                    elapsed=t if getattr(c, "wm_floor_amp", False) else None)
            else:
                target = sim.control_targets(a, bias, hist, xi)
            S1 = nets.wm_step(wm, wm_b, S, v.height, v.soles, target, xi, env.DT, out_bf16=bf16)
            v1 = sim.view_state(S1)
            t1 = t + 1
            up1 = sim.next_upright(up, v1.height, S1[1])
            d = jnp.where(up, self.termination(v1.clearance), self.termination_down(t1, v1, S1[1]))
            if self.recovery_clock:
                d = jnp.maximum(d, sim.recovery_timeout(start["reference_clip"], start["reference_start"], t1))
            if sim.recovery or sim.mixed:
                d = (t1 >= start.get("episode_limit", sim.episode_steps)).astype(jnp.float32)
            if sim.wm_floor_locomotion:
                d = jnp.maximum(d, (start["gait_world"] & up & ~up1).astype(d.dtype))
            recovering = ~up1
            gait_was_ready = remaining == 0 if step_history is not None else None
            active, deadline = None, False
            if self.mixed_amp and not sim.wm_floor_locomotion:
                deadline = remaining == 1
                remaining = sim.next_recovery_interval(remaining, up, up1)
                active = remaining > 0
            fall = (up & ~up1).astype(jnp.float32) if sim.mixed else d
            if motion_train:
                fall = sim.wm_fall(up, up1, v1.height, S1[1])
            foot_contact = sim.soft_contact(v1.soles)
            blend = None
            if hold is not None:
                # In imagination the sole clearances approximate the measured load gate.
                posture = zenbot.standing_posture(v1.height, S1[1][..., 2, 2])
                hold, blend = sim.advance_floor_stand(hold, posture, foot_contact.max(-1) > .5, paired_clip, t1)
                if sim.wm_floor_locomotion:
                    remaining = sim.floor_gait_interval(remaining, up1, blend)
                    active = remaining > 0
            terms = sim.reward_terms(S, S1, v1.kin, a, a_last, c_obs, recovering, fall, foot_contact, zeros, sim.q_soft,
                                     torque=sim.estimated_torque(S, target) if sim.recovery or self.legged_support else None,
                                     clip=paired_clip if paired_floor else start.get("reference_clip"), step=t1, reference_start=start.get("reference_start"), reference_yaw=start.get("reference_yaw"),
                                     recovery_active=active, recovery_deadline=deadline, stand_blend=blend,
                                     reference_rate_weight=start.get("reference_rate_weight", 1.))
            if step_history is not None:
                bottom = v1.feet.bottom
                # Predicted collision heights substitute for force sensors in the world model.
                feet_down = bottom[..., :2] <= env.FOOT_CONTACT_H
                undesired = jax.nn.sigmoid((env.FOOT_CONTACT_H - bottom[..., 2:]) / .003).sum(-1)
                foot_acc_z = (v1.feet.com_velocity_z - v.feet.com_velocity_z) / env.DT
                step_history, added = gait_rewards.reward_terms(
                    S1, v1.feet, c_obs, sim.estimated_torque(S, target), (S1[5] - S[5]) / env.DT,
                    sim.q_soft, feet_down, undesired, v1.feet.predicted_slip, foot_acc_z, step_history, gait_was_ready | ~active)
                terms.update(added)
            r = sum(sim.weighted(terms).values())
            reference_score = None
            if paired_floor:
                reference_score = terms["motion_reference"]
                if getattr(c, "wm_min_rise_score", 0.):
                    failed = sim.wm_reference.failed_rise(reference_score, paired_clip, start["reference_start"],
                                                          t1, active, up & ~up1, c.wm_min_rise_score)
                    d = jnp.maximum(d, failed.astype(d.dtype))
                if getattr(c, "wm_reference_training", False):
                    paired_clip = jnp.where(up & ~up1, -1, paired_clip)
            if use_amp:  # the gates of the two styles at this step: the locomotion one while up, the get-up one while not yet up
                frame1 = self.disc_frame(S1)
                moving = (jnp.linalg.norm(c_obs[:, :2], axis=-1) + jnp.abs(c_obs[:, 2]) > 0.05).astype(jnp.float32)
                upf = up1.astype(jnp.float32)
                g_loco = jnp.ones_like(upf) if sim.recovery else {"track_shift": moving * upf * (c.amp_track_lo + terms.get("track_lin", 0.)), "moving": moving * upf,
                          "track": moving * terms.get("track_lin", 0.), "none": upf}[c.amp_gate]  # fmt: skip
                g_up = self.getup_gate(up1, start.get("down_start"), t1)
                if self.recovery_clock:
                    g_up = sim.reference.active(start["reference_clip"], start["reference_start"], t1).astype(jnp.float32)
                if self.mixed_amp:
                    g_up = (1. - blend) if hold is not None else active.astype(jnp.float32)
                    if sim.wm_floor_locomotion:
                        g_up = jnp.where(active, g_up, 0.)
                if c.g1_style or self.mixed_amp:
                    g_loco *= 1. - g_up
                gate = jnp.stack((g_loco, g_up), -1)
            else:
                frame1, gate = jnp.zeros((B, 1)), jnp.zeros((B, 2))
            if motion_train:
                cmd1 = sim.tick_wm_command(cmd, k_c, S1, up1, remaining)
                c1 = sim.wm_velocity_command(cmd1, S1[1], up1, remaining)
                hist1 = predicted_sensors(hist, S1, c1, a, k_o, sim.noisy_obs)
            else:
                cmd1 = env.tick(cmd, k_c, sim.cmd_ranges)
                if self.recovery_clock:
                    cmd1 = sim.recovery_command(cmd1, start["down_start"])
                c1 = sim.command(cmd1, S1[1])
                hist1 = sim.push_actor(hist, sim.obs_frame(S1, a, c1, k_o))
            if critic_hist is not None:
                critic_hist = sim.push_critic(critic_hist, S1, a)
                cin1 = critic_hist
            else:
                cin1 = self.critic_in(S1, v1.height, a, c1, xi, up1, t1, start.get("down_start"), remaining) if use_critic else zeros[:, None]
            gait_cmd = jnp.where(remaining[..., None] > 0, 0., c_obs) if getattr(c, "command_gait_amp", False) else None
            return (S1, a, a, hist1, cmd1, c1, t1, up1, critic_hist, remaining, paired_clip, hold, step_history), (r, d, frame1, gate, cin1, recovering.astype(jnp.float32), remaining if self.legged_support else None, reference_score, gait_cmd)

        if c.imag_remat == "dots":
            step = jax.checkpoint(step, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        elif c.imag_remat == "full":
            step = jax.checkpoint(step)
        history = restore_sensors(start["obs"], start["imu_delay"], start["current_imu"], start["previous_imu"]) if motion_train else start["obs"]
        hold = zenbot.SupportedStand(start["upright_steps"], start["loaded_feet"]) if "upright_steps" in start else None
        init = (S0, start["a_last"], start["a_ode"], history, start["cmd"], start["c_obs"], start["t"], start["was_up"] > 0,
                start.get("obs_critic") if use_critic else None, start.get("recovery_left"), start.get("reference_clip") if paired_floor else None, hold,
                gait_rewards.StepHistory(**start["step_history"]) if "step_history" in start else None)
        _, (r, d, frames, gate, cins, rec, support_left, reference_score, gait_cmd) = jax.lax.scan(step, init, keys, unroll=c.imag_unroll)
        if use_amp:
            F = self.amp_span
            seq = jnp.concatenate((win0, jnp.swapaxes(frames, 0, 1)), 1)
            idx = jnp.arange(1, c.horizon + 1)[:, None] + jnp.arange(F)[None, :]
            wins = seq[:, idx]
            cmd = None if gait_cmd is None else jnp.swapaxes(gait_cmd, 0, 1)
            style = jnp.swapaxes(self.style(amp_st, wins[..., -self.gait_span:, :sim.amp_dim], cmd), 0, 1)  # (H, B)
            mix = gate[..., 0] * style
            up_full = 0.0  # amp_getup_full: the get-up style at full scale, outside the ramp
            if amp_up_st is not None:
                if self.legged_support:
                    score = self.legged_support_reward(amp_up_st, wins, support_left.T)
                elif self.recovery_clock:
                    elapsed = start["t"][:, None] + jnp.arange(1, c.horizon+1)
                    score = self.paired_style(amp_up_st, phase_st, wins, start["reference_clip"][:, None],
                                              start["reference_start"][:, None], start["reference_yaw"][:, None], elapsed)
                else:
                    score = self.getup_style(amp_up_st, wins[..., ::self.recovery_stride, :])
                style_up = jnp.swapaxes(score, 0, 1)
                if c.amp_getup_full:
                    up_full = gate[..., 1] * style_up
                else:
                    mix = mix + gate[..., 1] * style_up
            if c.amp_lerp > 0:  # r = (1 - lerp) r_task + lerp dt mix: mix in [0, 1], dt mix the scale of one task term of weight 1
                r_amp = c.amp_lerp * env.DT * (ramp * mix + up_full) if c.amp_getup_full else c.amp_lerp * env.DT * ramp * mix
                r = (1 - c.amp_lerp) * r + r_amp
            else:  # XSoma's: r = r_task + amp_w dt ramp mix
                r_amp = c.amp_w * env.DT * (ramp * mix + up_full) if c.amp_getup_full else c.amp_w * env.DT * ramp * mix
                r = r + r_amp
        alive = jnp.cumprod(1 - d, 0)
        alive_before = jnp.concatenate((jnp.ones_like(alive[:1]), alive[:-1]), 0)
        disc = c.gamma ** jnp.arange(c.horizon, dtype=jnp.float32)[:, None]
        G = (disc * alive_before * r).sum(0)
        aux = {"G": G.mean(), "r": (alive_before * r).sum(0).mean() / c.horizon, "surv": alive[-1].mean(), "rec_im": rec.mean()}
        if paired_floor:
            paired = alive_before * (reference_score != 0.)
            aux["reference_im_coverage"] = paired.mean()
            aux["reference_im_score"] = (alive_before * reference_score).sum() / jnp.maximum(paired.sum(), 1.)
        if use_critic:
            v0 = sim.view_state(S0)
            cin0 = start["obs_critic"] if "obs_critic" in start else self.critic_in(S0, v0.height, start["a_last"], start["c_obs"], xi, start["was_up"] > 0, start["t"], start.get("down_start"), start.get("recovery_left"))
            v_end = critic.value(critic_st, cins[-1])
            G = G + c.gamma**c.horizon * alive[-1] * v_end
            aux["v_end"] = v_end.mean()
            aux["traj"] = jax.lax.stop_gradient((jnp.concatenate((cin0[None], cins), 0), r, d))
        if use_amp:
            aux["r_amp"] = (alive_before * r_amp).sum(0).mean() / c.horizon
        return -G.mean(), aux

    # ------------------------------------------------------------------ hybrid PPO
    def gait_features(self, windows, command):
        features = amp.gait_window(windows, self.J, self.cfg.gait_amp_stride, env.DT)
        if getattr(self.cfg, "command_gait_amp", False):
            if getattr(self.cfg, "wm_range_commands", False):
                # Condition the style critic on the reference motion it can actually observe.
                speeds = jnp.asarray(self.sim.wm_command_speeds)
                command = speeds[reference_command_index(command, speeds)]
            labels = jnp.broadcast_to(command[..., None, :], (*features.shape[:-1], 3))
            features = jnp.concatenate((features, labels), axis=-1)
        return features

    def style(self, st, windows, command=None):
        if getattr(self.cfg, "gait_ls_scale", 0.):
            features = self.gait_features(windows, command)
            return amp.recovery_style(st, features, self.cfg.gait_ls_scale,
                                      soft_floor=getattr(self.cfg, "command_gait_amp", False))
        if self.sim.recovery:
            return amp.recovery_style(st, windows, self.cfg.amp_style_scale, leggedlab=self.sim.robot is g1_getup)
        return amp.style(st, amp.score(st["net"], st["norm"], windows), self.cfg.amp_style_scale)

    def ppo_batch(self, carry):
        """The newest `ppo_len` real transitions of the PPO envs (from the replay buffer: the collector's own steps, no extra simulation),
        with the imagination's reward (the environment's reward, the two styles with their gates, amp_lerp) and GAE(lambda) advantages
        of the critic. Flat (ppo_len * n_ppo, ...) arrays: obs, a_ode, the action and its behavior mean, the advantage and a mask."""
        c, sim, L, F = self.cfg, self.sim, self.cfg.ppo_len, self.amp_span
        buf, n = carry["buf"], len(self.ppo_envs)
        T = (F - 1) + L + 1  # style history, the transitions, the state after the last one (bootstrap)
        if self.real_ppo:
            # Include the newest transition. The extra row is a placeholder for
            # the current physical state's bootstrap, filled below.
            ages = buf["size"] - (T - 1) + jnp.minimum(jnp.arange(T), T - 2)
        else:
            ages = buf["size"] - T + jnp.arange(T)
        rows = rb._rows(buf, ages)
        w = rb.gather(buf, rows[:, None], jnp.asarray(self.ppo_envs)[None, :])  # (T, n, ...)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])  # noqa: E731
        tr = slice(F - 1, F - 1 + L)
        S1 = unflatten(flat(w["s1_qpos"][: T - 1]), flat(w["s1_qvel"][: T - 1]))
        r = w["r"][tr]
        if self.amp:
            feats = self.disc_frame(S1).reshape(T - 1, n, -1)
            idx = jnp.arange(L)[:, None] + jnp.arange(F)[None, :]  # the window of transition k: post-step frames k .. k + F - 1 of the rows
            if c.g1_style or sim.robot is g1_getup or self.mixed_amp:
                # Match imagination: fill missing pre-episode history with the reset pose.
                first = idx[:, -1:] - w["t"][tr]
                rows0 = jnp.maximum(first, 0)
                reset_states = unflatten(w["s_qpos"][rows0, jnp.arange(n)], w["s_qvel"][rows0, jnp.arange(n)])
                sequence = jnp.where((idx[:, :, None] < first[:, None, :])[..., None], self.disc_frame(reset_states)[:, None], feats[idx])
            else:
                sequence = feats[idx]
            wins = jnp.moveaxis(sequence, 2, 0).reshape(n * L, F, -1)  # (n, L) order
            h1 = S1[0][..., 2].reshape(T - 1, n)[tr]
            R1 = S1[1].reshape(T - 1, n, 3, 3)[tr]
            upf = sim.next_upright(w["was_up"][tr] > 0, h1, R1).astype(jnp.float32)
            cob = w["c_obs"][tr]
            moving = (jnp.linalg.norm(cob[..., :2], axis=-1) + jnp.abs(cob[..., 2]) > 0.05).astype(jnp.float32)
            g_loco = jnp.ones_like(upf) if sim.recovery else {"track_shift": moving * upf * (c.amp_track_lo + w["rlin"][tr]), "moving": moving * upf,
                      "track": moving * w["rlin"][tr], "none": upf}[c.amp_gate]  # fmt: skip
            g_up = self.getup_gate(upf, w["down_start"][tr] if c.g1_style else None, w["t"][tr] + 1)
            if self.recovery_clock:
                g_up = sim.reference.active(w["reference_clip"][tr], w["reference_start"][tr], w["t"][tr]+1).astype(jnp.float32)
            if self.mixed_amp:
                remaining = sim.next_recovery_interval(w["recovery_left"][tr], w["was_up"][tr] > 0, upf > 0)
                g_up = (1. - w["stand_blend"][tr]) if getattr(c, "wm_floor_amp", False) else (remaining > 0).astype(jnp.float32)
            if c.g1_style or self.mixed_amp:
                g_loco *= 1. - g_up
            if getattr(c, "gait_command_weight", 0.):
                # A stationary robot at a high command has almost no slope under
                # the Gaussian tracking reward. Give physical progress toward the
                # commanded planar velocity a linear slope across the command range.
                heading = jnp.arctan2(R1[..., 1, 0], R1[..., 0, 0])
                v = S1[2].reshape(T - 1, n, 3)[tr]
                planar = jnp.stack((jnp.cos(heading)*v[..., 0] + jnp.sin(heading)*v[..., 1],
                                    -jnp.sin(heading)*v[..., 0] + jnp.cos(heading)*v[..., 1]), -1)
                progress = jnp.linalg.norm(cob[..., :2], axis=-1) - jnp.linalg.norm(planar - cob[..., :2], axis=-1)
                r += c.gait_command_weight * env.DT * upf * (1. - g_up) * jnp.clip(progress, -1., 4.)
            gait_cmd = None
            if getattr(c, "command_gait_amp", False):
                gait_cmd = jnp.where(w["stand_blend"][tr, :, None] < 1., 0., cob).swapaxes(0, 1).reshape(n * L, 3)
            mix = g_loco * self.style(carry["amp"], wins[..., -self.gait_span:, :sim.amp_dim], gait_cmd).reshape(n, L).T
            ramp = carry["amp"]["ramp"]
            up_full = 0.0
            if self.getup:  # (amp_getup_full: the get-up style at full scale, outside the ramp, as in imagination)
                score_up = self.style if sim.recovery else self.getup_style  # legacy Zenbot PPO uses its main reward map for both heads
                if self.legged_support:
                    score = self.legged_support_reward(carry["amp_up"], wins, remaining.T.reshape(-1))
                elif self.recovery_clock:
                    timing = [w[k][tr].T.reshape(-1) for k in ("reference_clip", "reference_start", "reference_yaw", "t")]
                    score = self.paired_style(carry["amp_up"], carry.get("amp_phase"), wins,
                                              *timing[:3], timing[-1]+1)
                else:
                    score = score_up(carry["amp_up"], wins[..., ::self.recovery_stride, :])
                recovery_reward = g_up * score.reshape(n, L).T
                if c.amp_getup_full:
                    up_full = recovery_reward
                else:
                    mix = mix + recovery_reward
            styled = ramp * mix + up_full if c.amp_getup_full else ramp * mix
            r = (1 - c.amp_lerp) * r + c.amp_lerp * env.DT * styled if c.amp_lerp > 0 else r + c.amp_w * env.DT * styled
        st = slice(F - 1, T)  # the L + 1 states s_0 .. s_L
        S = unflatten(flat(w["s_qpos"][st]), flat(w["s_qvel"][st]))
        cin = flat(w["obs_critic"][st]) if "obs_critic" in w else self.critic_in(S, sim.view_state(S).height, flat(w["a_last"][st]), flat(w["c_obs"][st]), flat(w["xi"][st]),
                             flat(w["was_up"][st]) > 0, flat(w["t"][st]), flat(w["down_start"][st]) if c.g1_style else None,
                             flat(w["recovery_left"][st]) if self.mixed_amp else None)  # fmt: skip
        if self.real_ppo:
            ids, current = self.ppo_envs, carry["env"]
            last = unflatten(current.data.qpos[ids], current.data.qvel[ids])
            end = self.critic_in(last, last[0][:, 2], current.a_last[ids], current.c_obs[ids],
                                 sim.context(current)[ids], current.was_up[ids], current.t[ids],
                                 recovery_left=current.recovery_left[ids])
            cin = cin.at[-n:].set(end)
        V = critic.value(carry["critic"], cin).reshape(L + 1, n)
        done, term = w["done"][tr], w["term"][tr]
        trunc = done * (1 - term)
        if self.real_ppo or getattr(c, "wm_floor_amp", False):
            # Use the post-action pose before auto-reset for time limits.
            def timeout_bootstrap():
                last = unflatten(flat(w["s1_qpos"][tr]), flat(w["s1_qvel"][tr]))
                inputs = self.critic_in(last, last[0][:, 2], flat(w["a"][tr]), flat(w["c_obs"][tr]),
                                        flat(w["xi"][tr]), flat(upf) > 0, flat(w["t"][tr]) + 1,
                                        recovery_left=flat(remaining))
                return critic.value(carry["critic"], inputs).reshape(L, n)
            timeout_value = jax.lax.cond(trunc.any(), timeout_bootstrap, lambda: V[:-1])
            delta = r + c.gamma * ((1 - done) * V[1:] + trunc * timeout_value) - V[:-1]
        else:
            delta = r + c.gamma * (1 - done) * V[1:] + c.gamma * trunc * V[:-1] - V[:-1]
        mask = w["ok"][tr]
        if self.real_ppo:
            delta = jnp.where(mask > 0, delta, 0.)
            done = jnp.maximum(done, 1. - mask)

        def back(A, x):
            d_k, dn = x
            A = d_k + c.gamma * c.lam * (1 - dn) * A
            return A, A

        _, adv = jax.lax.scan(back, jnp.zeros((n,)), (delta, done), reverse=True)
        mean = (adv * mask).sum() / jnp.maximum(mask.sum(), 1.0)
        std = jnp.sqrt(((adv - mean) ** 2 * mask).sum() / jnp.maximum(mask.sum(), 1.0)) + 1e-6
        out = {"obs": w["obs"][tr], "a_ode": w["a_ode"][tr], "a": w["a"][tr], "mu": w["mu"][tr], "adv": (adv - mean) / std, "mask": mask}
        if getattr(c, "gait_mirror_weight", 0.):
            out["gait_mirror"] = w["stand_blend"][tr] >= 1.
        if self.real_ppo or getattr(c, "wm_floor_amp", False):
            out.update(value_input=cin[:-n].reshape(L, n, -1), old_value=V[:-1], value_target=adv + V[:-1])
        out = jax.lax.stop_gradient({k: flat(v) for k, v in out.items()})
        if "anchor" in carry or "rise_teacher" in carry:
            teacher = carry["anchor"] if "anchor" in carry else carry["rise_teacher"]
            out["anchor_mu"] = self.act(teacher["pi"], teacher["obs_norm"], out["obs"], out["a_ode"])
        if "rise_teacher" in carry:
            out["rise_anchor_weight"] = flat(w["rise_anchor_weight"][tr])
        elif "anchor" in carry:
            out["down"] = flat(w["was_up"][tr]) == 0
            out["early"] = flat(w["t"][tr]) * env.DT < 2.
            if self.recovery_clock:
                active = flat(sim.reference.active(w["reference_clip"][tr], w["reference_start"][tr], w["t"][tr]+1))
                out["preserve_baseline"] = ~active
        stats = {"ppo_r": (r * mask).sum() / jnp.maximum(mask.sum(), 1.0)}
        if self.real_ppo:
            w_style = c.amp_lerp if c.amp_lerp > 0 else c.amp_w
            stats.update(G=(out["value_target"] * flat(mask)).sum() / jnp.maximum(mask.sum(), 1.),
                         r_amp=(w_style * env.DT * styled * mask).sum() / jnp.maximum(mask.sum(), 1.),
                         surv=1. - done.mean(), v_end=V[-1].mean())
        if self.recovery_clock or self.mixed_amp:
            scored = g_up * mask * (w["t"][tr]+1 >= F-1)
            stats["real_getup_amp_per_active_step"] = (recovery_reward * scored).sum() / jnp.maximum(scored.sum(), 1.) * env.DT * c.amp_lerp
            if getattr(c, "wm_lift_gate", 0.):
                floor = mask * (w["s_qpos"][tr, :, 2] <= c.wm_lift_gate) * (h1 <= c.wm_lift_gate)
                style_weight = c.amp_lerp if c.amp_lerp > 0 else c.amp_w
                stats["real_below_lift_amp_per_step"] = (styled * floor).sum() / jnp.maximum(floor.sum(), 1.) * env.DT * style_weight
        return out, stats

    def ppo_loss(self, pi, obs_norm, b):
        """PPO's clipped surrogate of the live policy on the real batch: Gaussian actions of fixed std action_noise around the mean."""
        c = self.cfg
        mu = self.act(pi, obs_norm, b["obs"], b["a_ode"])
        logr = (jnp.sum(jnp.square(b["a"] - b["mu"]), -1) - jnp.sum(jnp.square(b["a"] - mu), -1)) / (2 * c.action_noise**2)
        ratio = jnp.exp(jnp.clip(logr, -20.0, 20.0))
        surr = jnp.minimum(ratio * b["adv"], jnp.clip(ratio, 1 - c.ppo_clip, 1 + c.ppo_clip) * b["adv"])
        m = b["mask"]
        nm = jnp.maximum(m.sum(), 1.0)
        aux = {"ppo_clipfrac": ((jnp.abs(ratio - 1) > c.ppo_clip) * m).sum() / nm, "ppo_logr": (jnp.abs(logr) * m).sum() / nm}
        loss = -(surr * m).sum() / nm
        if "gait_mirror" in b:
            reflected = self.act(pi, obs_norm, mirror_observation(b["obs"]), mirror_action(b["a_ode"]))
            target = jax.lax.stop_gradient(mirror_action(mu))
            eligible = m * b["gait_mirror"]
            mse = (eligible * jnp.square(reflected - target).mean(-1)).sum() / jnp.maximum(eligible.sum(), 1.)
            loss += c.gait_mirror_weight * mse
            aux["gait_mirror_mse"] = mse
        if "rise_anchor_weight" in b:
            anchor_loss = g1_getup.rise_anchor_loss(mu, b["anchor_mu"], m * b["rise_anchor_weight"])
            loss = loss + c.rise_anchor * anchor_loss
            aux["rise_anchor_mse"] = anchor_loss
        elif "anchor_mu" in b:
            # Balance rare recovery states against the long upright episodes. Permit more exploration during the opening.
            kl = jnp.sum(jnp.square(mu - b["anchor_mu"]), axis=-1) / (2 * c.action_noise**2)
            if "preserve_baseline" in b:
                m = m * b["preserve_baseline"]
            down = m * b["down"]
            upright = m * ~b["down"]
            opening = jnp.where(b["early"], 0.35, 1.)
            down_kl = (down * opening * kl).sum() / jnp.maximum(down.sum(), 1.)
            up_kl = (upright * kl).sum() / jnp.maximum(upright.sum(), 1.)
            loss = loss + c.getup_anchor * 0.5 * (down_kl + up_kl)
            aux["anchor_kl"] = (kl * m).sum() / nm
        return loss, aux

    def policy_update(self, carry, start, key, ppo_b=None):
        c = self.cfg
        if getattr(c, "wm_rate_ramp", 0):
            start = {**start, "reference_rate_weight": self.reference_rate_weight(carry["it"])}
        (_, aux), grads = jax.value_and_grad(self.imagine, has_aux=True)(carry["pi"], carry["wm"], carry["wm_b"], carry["obs_norm"], start,
                                                                         jax.random.split(key, c.horizon), carry.get("amp"), carry.get("critic"),
                                                                         carry.get("amp_up"), carry.get("amp_phase"))  # fmt: skip
        if ppo_b is not None:  # g = g_bptt + ppo_w |g_bptt| / |g_ppo| g_ppo
            # the PPO gradient of each half of the PPO envs (even / odd), their mean the batch's: the cosine of the halves measures the
            # PPO gradient's signal against its noise, its cosine with the BPTT gradient whether the two point the same way
            n = len(self.ppo_envs)
            half = (jnp.arange(ppo_b["mask"].shape[0]) % n) % 2
            g_half = [jax.value_and_grad(self.ppo_loss, has_aux=True)(carry["pi"], carry["obs_norm"], {**ppo_b, "mask": ppo_b["mask"] * (half == h)})
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
        pi, opt_pi = optim.step(carry["pi"], grads, carry["opt_pi"], self.pi_mask, c.pi_lr, betas=(0.9, 0.95), ok=ok)
        avg = optim.ema(carry["avg"], pi, c.pi_ema, carry["it"] == c.wm_only_iters)
        out = {"pi": pi, "opt_pi": opt_pi, "avg": avg, "gn_ema": gn_ema, "nonfinite": carry["nonfinite"] + (~finite).astype(jnp.int32)}
        if self.use_critic:
            x, r, d = aux.pop("traj")
            real_values = ((ppo_b["value_input"], ppo_b["value_target"], ppo_b["mask"])
                           if getattr(c, "wm_floor_amp", False) else None)
            out["critic"], aux["vloss"] = critic.fit(carry["critic"], x, r, d, jax.random.fold_in(key, 5), c.gamma, c.lam,
                                                     c.critic_lr, c.critic_epochs, c.critic_mb, real_values=real_values)
        return out, {**aux, "gn": gn, "gn_gated": (finite & ~ok).astype(jnp.float32)}

    # ------------------------------------------------------------------ iteration programs
    SUMMED = ("velocity_errors", "done", "terminated", "len_sum", "rew_sums", "diverged", "down_done", "down_up", "pushes",
              "reference_score_sum", "reference_frames", "reference_moving_score_sum", "reference_moving_frames")

    def collect_iter(self, carry):
        key, k_c = jax.random.split(carry["key"])
        carry, st = jax.lax.scan(self.collect_step, {**carry, "key": key}, jax.random.split(k_c, self.cfg.env_steps))
        return carry, {k: (v.sum(0) if k in self.SUMMED else v.mean(0)) for k, v in st.items()}

    def gait_amp_eligible(self, window, state):
        valid = window["mask"][:, -1] > 0
        if self.sim.recovery:
            return valid
        if "stand_blend" in window:
            return valid & (window["stand_blend"][:, -1] > 0.)
        up = self.sim.is_up(state[0][..., 2], state[1]).all(-1) & (window["was_up"] > 0).all(-1)
        valid &= up
        if not getattr(self.cfg, "motion_train_wm", False):
            cmd = window["c_obs"][:, -1]
            valid &= jnp.linalg.norm(cmd[:, :2], axis=-1) + jnp.abs(cmd[:, 2]) > .05
        if self.mixed_amp and not getattr(self.cfg, "wm_floor_amp", False):
            valid &= (window["recovery_left"] == 0).all(-1)
        return valid

    def amp_iter(self, carry):
        """Fit each discriminator on eligible real windows and its own reference clips, then update the style ramp."""
        c, sim, F = self.cfg, self.sim, self.cfg.amp_frames
        disc_step = amp.train_step_with_robot if getattr(c, "wm_getup_only", False) or sim.wm_floor_amp else amp.train_step

        def one(st, k):
            k_w, k_d, k_g = jax.random.split(k, 3)
            w = rb.sample_windows(carry["buf"], k_w, c.amp_batch, self.gait_span, recent=c.amp_recent, envs=self.amp_envs)
            S = unflatten(w["s_qpos"], w["s_qvel"])
            feats = sim.amp_frame(S)
            ok = self.gait_amp_eligible(w, S)
            usable = ok.mean()
            label = None
            if getattr(c, "command_gait_amp", False):
                # Match expert and robot command frequencies exactly, even when many windows ended in a fall.
                k_select, k_d = jax.random.split(k_d)
                valid = jnp.flatnonzero(ok, size=c.amp_batch)
                source = valid[jax.random.randint(k_select, (c.amp_batch,), 0, jnp.maximum(1, ok.sum()))]
                label = jnp.where(w["stand_blend"][source, -1, None] < 1., 0., w["c_obs"][source, -1])
                feats = feats[source]
                ok = jnp.full_like(ok, ok.any())
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
            else:
                demo = self.demos.sample(k_d, c.amp_batch)
            gait_ls = bool(getattr(c, "gait_ls_scale", 0.))
            if gait_ls:
                demo = self.gait_features(demo, label)
                feats = self.gait_features(feats, label)
            st, aux = disc_step(st, demo, feats, ok.astype(jnp.float32), k_g, c.amp_lr,
                                     recovery=sim.recovery or gait_ls, leggedlab=sim.robot is g1_getup)
            return st, {**aux, "usable": usable}

        def paired_update(states, key):
            from .g1_phase_amp import update_paired
            native, paired = states
            kw, kd, kg, kp = jax.random.split(key, 4)
            w = rb.sample_windows(carry["buf"], kw, c.amp_batch, F, recent=c.amp_recent, envs=self.amp_envs,
                                  recovery_lengths=sim.reference.lengths)
            raw = self.disc_frame(unflatten(w["s_qpos"], w["s_qvel"]))
            timing = [w[name][:, -1] for name in ("reference_clip", "reference_start", "reference_yaw", "t")]
            expert, robot, valid = self.phase_demos.paired_windows(raw, *timing)
            valid &= w["mask"][:, -1] > 0
            def update_native(st):
                return amp.train_step(st, self.getup_demos.sample(kd, c.amp_batch), raw[..., :self.phase_demos.base_dim],
                                      valid.astype(jnp.float32), kg, c.getup_disc_lr, recovery=True)
            def skip(st):
                return st, {k: jnp.zeros(()) for k in ("d_demo", "d_robot", "gp")}
            native, stats = jax.lax.cond(valid.any(), update_native, skip, native)
            if c.g1_time_negatives:
                kt, km = jax.random.split(jax.random.fold_in(kp, 1))
                offsets = jnp.array([-25, -12, 12, 25])
                drift = jax.random.choice(kt, offsets, (len(valid),))
                shifted, distinct = self.phase_demos.time_shifted_expert(timing[0], timing[1], timing[-1], drift)
                use_shifted = valid & distinct & jax.random.bernoulli(km, c.g1_time_negatives, (len(valid),))
                robot = jnp.where(use_shifted[:, None, None], shifted, robot)
                stats["shifted_fraction"] = use_shifted.sum() / jnp.maximum(valid.sum(), 1)
            paired, aux = update_paired(paired, expert, robot, valid, kp, c.getup_disc_lr)
            return (native, paired), {**stats, "usable": valid.mean(), "phase_demo": aux["d_demo"], "phase_robot": aux["d_robot"]}

        def one_up(st, k):
            k_w, k_d, k_g = jax.random.split(k, 3)
            if self.legged_support:
                w = rb.sample_windows(carry["buf"], k_w, c.amp_batch, self.amp_span,
                                      recent=c.amp_recent, recovery_only=True, recovery_steps=sim.recovery_amp_steps)
                S = unflatten(w["s_qpos"], w["s_qvel"])
                native, support = self.legged_support_inputs(sim.amp_support_frame(S, with_native=True))
                demo_native, demo_support = self.legged_support_inputs(self.getup_demos.sample_opening_windows(k_d, c.amp_batch))
                ok = (w["mask"][:, -1] > 0) & (w["recovery_left"][:, -1] > 0)
                if c.legged_prior_file:
                    nst = st["native"]
                    pos = support_discriminator.legged_logit(nst, demo_native)
                    neg = support_discriminator.legged_logit(nst, native)
                    naux = {"d_demo": pos.mean(), "d_robot": jnp.sum(neg * ok) / jnp.maximum(ok.sum(), 1)}
                else:
                    nst, naux = amp.train_step(st["native"], demo_native, native, ok, k_g, .001,
                                               recovery=True, leggedlab=True)
                sst, saux = support_discriminator.train_step(st["support"], demo_support, support, ok,
                                                             jax.random.fold_in(k_g, 1), c.amp_lr)
                return {"native": nst, "support": sst}, {**naux, "usable": ok.mean(), "support_demo": saux["d_demo"], "support_robot": saux["d_robot"]}
            padded_opening = bool(getattr(c, "wm_lift_gate", 0.))
            w = rb.sample_windows(carry["buf"], k_w, c.amp_batch, self.amp_span, recent=c.amp_recent, envs=self.amp_envs,
                                  recovery_only=self.mixed_amp, recovery_steps=sim.recovery_amp_steps if padded_opening else 0)
            S = unflatten(w["s_qpos"][:, ::self.recovery_stride], w["s_qvel"][:, ::self.recovery_stride])
            feats = self.disc_frame(S)
            phase = self.getup_gate(w["was_up"], w.get("down_start"), w["t"])
            wasserstein = self.mixed_amp and c.recovery_amp_loss == "wasserstein"
            if wasserstein:
                demo = self.getup_demos.sample_ground_heights(k_d, S[0][:, -1, 2])
            elif padded_opening:
                demo = self.getup_demos.sample_opening_windows(k_d, c.amp_batch)
            else:
                demo = self.getup_demos.sample(k_d, c.amp_batch)
            demo = demo[:, ::self.recovery_stride]
            if self.mixed_amp:
                phase = (w["stand_blend"][:, -1:] < 1.) if getattr(c, "wm_floor_amp", False) else w["recovery_left"] > 0
                feats, demo = self.ground_features(feats), self.ground_features(demo)
            ok = (w["mask"][:, -1] > 0) & (phase > 0).all(-1)
            n_tempo = round(c.amp_batch * getattr(c, "wm_tempo_negatives", 0.))
            if n_tempo:
                wrong, moving = self.getup_demos.sample_wrong_tempo(
                    jax.random.fold_in(k_d, 1), n_tempo, c.amp_frames, self.recovery_stride, len(sim.amp_joints),
                    freeze_start=sim.wm_floor_locomotion)
                feats = feats.at[:n_tempo].set(jnp.where(moving[:, None, None], self.ground_features(wrong), feats[:n_tempo]))
            st, aux = disc_step(st, demo, feats, ok.astype(jnp.float32), k_g,
                                     c.getup_disc_lr if c.g1_style else c.amp_lr,
                                     recovery=bool(c.g1_style or (self.mixed_amp and not wasserstein)), include_demo_norm=wasserstein)
            if n_tempo:
                aux["tempo_negative_share"] = (moving & ok[:n_tempo]).sum() / jnp.maximum(ok.sum(), 1)
            return st, {**aux, "usable": ok.mean()}

        def legged_update(st, key):
            from .g1_phase_amp import update_paired
            kw, kd = jax.random.split(key)
            w = rb.sample_windows(carry["buf"], kw, c.amp_batch, F, recent=c.amp_recent, envs=self.amp_envs,
                                  recovery_lengths=sim.reference.lengths)
            state = unflatten(w["s_qpos"], w["s_qvel"])
            raw = self.getup_demos.policy_frames(state)[..., sim.amp_dim:]
            # These are *pre-step* physical states. Their last tick is t, not t+1.
            timing = [w[k][:, -1] for k in ("reference_clip", "reference_start", "t")]
            demo, robot, valid = self.legged_prior.pair(raw, *timing)
            valid &= w["mask"][:, -1] > 0
            st, aux = update_paired(st, demo, robot, valid, kd, c.getup_disc_lr)
            return st, {**aux, "usable": valid.mean()}

        key, k, k2 = jax.random.split(carry["key"], 3)
        st, aux = jax.lax.scan(one, carry["amp"], jax.random.split(k, c.amp_disc_steps))
        ramp = jnp.clip((carry["it"] - c.amp_delay) / float(c.amp_ramp), 0., 1.) if c.amp_ramp else (carry["it"] >= c.amp_delay).astype(jnp.float32)
        aux = jax.tree.map(lambda x: x[-1], aux)
        out = {**carry, "amp": {**st, "ramp": ramp}, "key": key}
        stats = {"amp_d_demo": aux["d_demo"], "amp_d_robot": aux["d_robot"], "amp_gp": aux["gp"], "amp_usable": aux["usable"], "amp_ramp": ramp}
        if self.getup:
            disc_keys = jax.random.split(k2, c.amp_disc_steps)
            if self.paired:
                (st2, out["amp_phase"]), aux2 = jax.lax.scan(paired_update, (carry["amp_up"], carry["amp_phase"]), disc_keys)
                stats["phase_demo"] = aux2["phase_demo"].mean()
                stats["phase_robot"] = aux2["phase_robot"].mean()
                if c.g1_time_negatives:
                    stats["phase_shifted_fraction"] = aux2["shifted_fraction"].mean()
            elif self.legged:
                st2, aux2 = jax.lax.scan(legged_update, carry["amp_up"], disc_keys)
            else:
                st2, aux2 = jax.lax.scan(one_up, carry["amp_up"], disc_keys)
            aux2 = jax.tree.map(lambda x: x[-1], aux2)
            if c.g1_style and c.getup_style_ramp:
                st2 = {**st2, "ramp": jnp.clip((carry["it"] - c.wm_only_iters) / c.getup_style_ramp, 0., 1.)}
                stats["getup_ramp"] = st2["ramp"]
            out["amp_up"] = st2
            stats.update({"ampg_d_demo": aux2["d_demo"], "ampg_d_robot": aux2["d_robot"], "ampg_usable": aux2["usable"]})
            if "tempo_negative_share" in aux2:
                stats["ampg_tempo_negative_share"] = aux2["tempo_negative_share"]
            if self.legged_support:
                stats.update(support_d_demo=aux2["support_demo"], support_d_robot=aux2["support_robot"])
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

    def real_ppo_iter(self, carry):
        c = self.cfg
        batch, stats = self.ppo_batch(carry)
        key, epochs_key = jax.random.split(carry["key"])
        value_mask = jax.tree.map(lambda _: False, carry["critic"]["net"])
        value_norm = carry["critic"]["norm"]
        size = len(batch["mask"])
        pad = (-size) % 4
        valid_indices = (jnp.arange(size + pad) < size).reshape(4, -1)

        def value_loss(net, b):
            v = critic.value({"net": net, "norm": value_norm}, b["value_input"])
            old, target = b["old_value"], b["value_target"]
            clipped = old + jnp.clip(v - old, -c.ppo_clip, c.ppo_clip)
            error = jnp.maximum(jnp.square(v - target), jnp.square(clipped - target))
            return (error * b["mask"]).sum() / jnp.maximum(b["mask"].sum(), 1.)

        def minibatch(state, selection):
            pi, opt_pi, value, opt_value, avg = state
            indices, valid = selection
            b = jax.tree.map(lambda a: a[indices], batch)
            b["mask"] = b["mask"] * valid
            (_, ps), pg = jax.value_and_grad(self.ppo_loss, has_aux=True)(pi, carry["obs_norm"], b)
            vl, vg = jax.value_and_grad(value_loss)(value, b)
            pg, gn = optim.clip_by_global_norm(pg, c.pi_clip)
            vg, vn = optim.clip_by_global_norm(vg, 1.)
            finite = jnp.isfinite(gn) & jnp.isfinite(vn)
            ok = finite & (b["mask"].sum() > 0)
            pi, opt_pi = optim.step(pi, pg, opt_pi, self.pi_mask, c.pi_lr, ok=ok)
            value, opt_value = optim.step(value, vg, opt_value, value_mask, c.critic_lr, ok=ok)
            avg = jax.lax.cond(ok, lambda: optim.ema(avg, pi, c.pi_ema, False), lambda: avg)
            return (pi, opt_pi, value, opt_value, avg), {**ps, "vloss": vl, "gn": gn, "bad": ~finite}

        def epoch(state, epoch_key):
            order = jnp.pad(jax.random.permutation(epoch_key, size), (0, pad)).reshape(4, -1)
            return jax.lax.scan(minibatch, state, (order, valid_indices))

        value = carry["critic"]
        initial = carry["pi"], carry["opt_pi"], value["net"], value["opt"], carry["avg"]
        final, hist = jax.lax.scan(epoch, initial, jax.random.split(epochs_key, c.pi_updates))
        pi, opt_pi, vnet, opt_value, avg = final
        value_norm = nets.rn_update(value_norm, batch["value_input"], weights=batch["mask"])
        obs_norm = carry["obs_norm"]
        if "raw" not in obs_norm:
            obs_norm = nets.rn_update(obs_norm, batch["obs"], weights=batch["mask"])
        stats.update(ppo_clipfrac=hist["ppo_clipfrac"].mean(), ppo_logr=hist["ppo_logr"].mean(),
                     vloss=hist["vloss"].mean(), gn=hist["gn"].mean(), ppo_gn=hist["gn"].mean(),
                     r=stats["ppo_r"], gn_gated=jnp.zeros(()), wm_gated=jnp.zeros(()))
        return {**carry, "pi": pi, "opt_pi": opt_pi, "avg": avg, "obs_norm": obs_norm,
                "critic": {"net": vnet, "opt": opt_value, "norm": value_norm},
                "nonfinite": carry["nonfinite"] + hist["bad"].sum(dtype=jnp.int32),
                "key": key, "it": carry["it"] + 1}, stats

    def pi_iter(self, carry):
        c = self.cfg
        key, k_p = jax.random.split(carry["key"])
        names = ("pi", "opt_pi", "avg", "gn_ema", "nonfinite") + (("critic",) if self.use_critic else ())

        def do(sub):
            ppo_b, pst = self.ppo_batch(carry) if self.ppo else (None, {})

            def one(sub, k):
                k_s, k_u = jax.random.split(k)
                return self.policy_update({**carry, **sub}, self.sample_starts(carry["buf"], k_s), k_u, ppo_b)

            sub, st = jax.lax.scan(one, sub, jax.random.split(k_p, c.pi_updates))
            return sub, {**jax.tree.map(jnp.mean, st), **pst}

        stat_keys = (("G", "r", "surv", "rec_im", "gn", "gn_gated") + (("r_amp",) if self.amp else ()) + (("v_end", "vloss") if self.use_critic else ())
                     + (("bptt_gn", "ppo_clipfrac", "ppo_logr", "ppo_gn", "ppo_r", "ppo_cos_bp", "ppo_cos_half") if self.ppo else ()))  # fmt: skip
        if getattr(c, "gait_mirror_weight", 0.):
            stat_keys += ("gait_mirror_mse",)
        if self.wm_tracks_reference:
            stat_keys += ("reference_im_coverage", "reference_im_score")
        if "anchor" in carry:
            stat_keys += ("anchor_kl",)
        if "rise_teacher" in carry:
            stat_keys += ("rise_anchor_mse",)

        if (self.recovery_clock or self.mixed_amp) and self.ppo:
            stat_keys += ("real_getup_amp_per_active_step",)
            if getattr(c, "wm_lift_gate", 0.):
                stat_keys += ("real_below_lift_amp_per_step",)

        def skip(sub):
            return sub, {k: jnp.zeros((), jnp.float32) for k in stat_keys}

        active = carry["it"] >= c.wm_only_iters
        sub, ps = jax.lax.cond(active & carry["wm_ok"], do, skip, {k: carry[k] for k in names})
        return {**carry, **sub, "key": key, "it": carry["it"] + 1}, {**ps, "wm_gated": (active & ~carry["wm_ok"]).astype(jnp.float32)}

    def compile_programs(self, carry):
        policy_update = self.real_ppo_iter if self.real_ppo else self.pi_iter
        programs = {"collect": jax.jit(self.collect_iter, donate_argnums=(0,)),
                    "pi": jax.jit(policy_update, donate_argnums=(0,))}
        if not self.real_ppo:
            programs["wm"] = jax.jit(self.wm_iter, donate_argnums=(0,))
        if self.amp:
            programs["amp"] = jax.jit(self.amp_iter, donate_argnums=(0,))
        if self.cfg.wm_eval and not self.real_ppo:
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


FULL_KEYS = ("pi", "avg", "wm", "wm_b", "obs_norm", "opt_pi", "opt_wm", "amp", "amp_up", "critic", "it", "env_steps", "gn_ema", "wm_ema", "nonfinite", "anchor", "amp_phase")


def save_full(path, carry):
    leaves = jax.tree.leaves(jax.device_get({k: carry[k] for k in FULL_KEYS if k in carry}))
    np.savez(path + ".tmp.npz", **{f"{i:05d}": np.asarray(x) for i, x in enumerate(leaves)})
    os.replace(path + ".tmp.npz", path + ".npz")


def load_full(carry, path):
    part = {k: carry[k] for k in FULL_KEYS if k in carry}
    leaves, tdef = jax.tree.flatten(part)
    with np.load(path, allow_pickle=False) as z:
        if len(z.files) != len(leaves):
            raise ValueError(f"{path} has {len(z.files)} arrays, expected {len(leaves)} for this learner")
        values = []
        for i, leaf in enumerate(leaves):
            arr = z[f"{i:05d}"]
            if arr.shape != leaf.shape:
                raise ValueError(f"{path} array {i} has shape {arr.shape}, expected {leaf.shape}")
            values.append(jnp.asarray(arr, dtype=leaf.dtype))
    return {**carry, **jax.tree.unflatten(tdef, values)}


def initialize_g1_style(carry, path):
    """Start from a trained G1's averaged actor, dynamics, critic and walking prior."""
    # Legacy G1 checkpoints used the walking feature width for both discriminators.
    legacy = {k: v for k, v in carry.items() if k not in ("anchor", "amp_phase")}
    old = load_full({**legacy, "amp_up": carry["amp"]}, path)
    source = {k: old[k] for k in ("avg", "obs_norm", "wm", "wm_b", "amp", "critic")}
    if "anchor" in carry:
        source["anchor"] = jax.tree.map(jnp.copy, {"pi": old["avg"], "obs_norm": old["obs_norm"]})
    return {**carry, **source, "pi": jax.tree.map(jnp.copy, old["avg"])}


def initialize_actor(carry, path, expected_ode):
    """Load the averaged actor and its normalizer into a fresh training run."""
    from .inference import load

    pi, norm, ode = load(path)
    from .zenbot_ground_policy import ZenbotGroundPolicy
    if isinstance(pi, ZenbotGroundPolicy):
        pi, norm = pi.rise, norm[1]
    if ode != expected_ode and (ode is None or expected_ode is None or not np.allclose(
            [v or 0. for v in ode], [v or 0. for v in expected_ode])):
        raise ValueError("The checkpoint and training ODE action limits differ")
    template = carry["pi"]
    if "raw" in norm:
        # The LeggedLab PPO actor uses plain ELU layers and raw observations.
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


def initialize_native_learner(fresh, full_path, actor_path, expected_ode, rise_anchor=0.):
    """Restore the learner, then optionally replace its actor and reset the actor optimizer."""
    carry = fresh
    if full_path:
        companion = Path(full_path).parent.parent / Path(full_path).name
        source_template = initialize_actor(fresh, companion, expected_ode)
        carry = load_full(source_template, full_path)
    if actor_path:
        # The fresh template includes the native LayerNorm parameters even when
        # the restored companion used plain layers and raw observations.
        actor_keys = ("pi", "obs_norm", "opt_pi")
        selected = initialize_actor({k: fresh[k] for k in actor_keys}, actor_path, expected_ode)
        carry = {**carry, **selected}
    if rise_anchor:
        # Reconstructed from --init_policy for this training stage, excluded from exports.
        carry = {**carry, "rise_teacher": jax.tree.map(jnp.copy, {"pi": carry["pi"], "obs_norm": carry["obs_norm"]})}
    return carry


def save_policy(path, carry, meta, ode=None):
    """The averaged and live policy and the observation normalizer (<path>.npz), and the metadata (<path>.json)."""
    flat = {"obs_norm/mean": np.asarray(carry["obs_norm"]["mean"]), "obs_norm/var": np.asarray(carry["obs_norm"]["var"])}
    if "raw" in carry["obs_norm"]:
        flat["obs_norm/raw"] = np.bool_(True)
    if carry.get("delay_uses_first_command", False):
        flat["zenbot/delay_uses_first_command"] = np.bool_(True)
    if ode is not None:
        flat["ode/dmax"] = np.float32(0.0 if ode[0] is None else ode[0])
        flat["ode/amax"] = np.float32(0.0 if ode[1] is None else ode[1])
    for name in ("avg", "pi"):
        for i, layer in enumerate(carry[name]):
            for k, v in layer.items():
                flat[f"{'live' if name == 'pi' else 'avg'}/{i}/{k}"] = np.asarray(v)
    if "ground_base" in carry:
        layers, norm = carry["ground_base"]
        flat["ground_max_share"] = np.float32(carry["ground_max_share"])
        for i, layer in enumerate(layers):
            flat.update({f"ground_base/avg/{i}/{k}": np.asarray(v) for k, v in layer.items()})
        flat.update({f"ground_base/obs_norm/{k}": np.asarray(v) for k, v in norm.items()})
    np.savez(path + ".tmp.npz", **flat)
    os.replace(path + ".tmp.npz", path + ".npz")
    with open(path + ".json", "w") as f:
        json.dump(meta, f)


def parse(argv=None) -> Config:
    p = argparse.ArgumentParser(description="InstantAMP: train a humanoid velocity-tracking policy with AMP through a learned world model.")
    select = argparse.ArgumentParser(add_help=False)
    select.add_argument("--robot", choices=("g1", "g1_getup", "zenbot23_getup", zenbot.MIXED_TASK))
    select.add_argument("--task", choices=(g1_getup.TASK_ID,))
    select.add_argument("--g1_style", type=int, choices=(0, 1), default=0)
    select.add_argument("--motion_train_wm", type=int, choices=(0, 1), default=0)
    chosen = select.parse_known_args(argv)[0]
    if chosen.task and chosen.robot not in (None, "g1_getup"):
        p.error(f"{chosen.task} requires the g1_getup profile")
    if chosen.motion_train_wm and (chosen.robot != zenbot.MIXED_TASK or chosen.task):
        p.error("--motion_train_wm 1 requires the mixed Zenbot task")
    if chosen.motion_train_wm:
        recipe = ZenbotMotionTrainWMConfig
    elif chosen.task or chosen.robot == "g1_getup":
        recipe = G1GetUpConfig
    elif chosen.robot == zenbot.MIXED_TASK:
        recipe = ZenbotWalkRunRecoveryConfig
    else:
        recipe = G1StyleConfig if chosen.g1_style else (ZenbotGetUpConfig if chosen.robot == "zenbot23_getup" else Config)
    if recipe is not G1GetUpConfig:
        p.add_argument("--task", choices=(g1_getup.TASK_ID,), default=argparse.SUPPRESS,
                       help="select the LeggedLab G1 get-up task")
    for f in dataclasses.fields(recipe):
        p.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return recipe(**vars(p.parse_args(argv)))


def mixed_recovery_restore_template(carry, cfg, old, sim):
    """Return the source tree required to decode a flat checkpoint, and any new recovery head."""
    old_mode = old.get("recovery_amp_loss", "least_squares") if old.get("recovery_amp_seconds") else "legacy"
    new_mode = cfg.recovery_amp_loss if cfg.recovery_amp_seconds else "legacy"
    same_stride = old_mode == "legacy" or old.get("recovery_amp_stride", 1) == cfg.recovery_amp_stride
    oriented_change = old_mode == new_mode == "legged_support" and bool(old.get("legged_real_ppo")) != bool(getattr(cfg, "legged_real_ppo", 0))
    if old_mode == new_mode and same_stride and not oriented_change:
        return carry, None

    new_head = carry["amp_up"]
    if old_mode == "legged_support" and (new_mode != "legged_support" or oriented_change):
        from . import support_discriminator
        frames = old.get("amp_frames", cfg.amp_frames)
        native_dim = 9 + 2 * len(sim.amp_joints) + 3 * len(zenbot.KEY_BODIES)
        key = jax.random.PRNGKey(0)
        source_head = jax.eval_shape(lambda k: {
            "native": amp.init(k, frames, frame_dim=native_dim),
            "support": support_discriminator.init(k, frames, amp.ORIENTED_SUPPORT_DIM if old.get("legged_real_ppo") else amp.SUPPORT_FEATURE_DIM),
        }, key)
        carry = {**carry, "amp_up": source_head}
    elif new_mode == "legged_support" and old_mode != "legged_support":
        # Every historical mixed-task recovery discriminator used the gait
        # discriminator's frame width, architecture and optimizer layout.
        carry = {**carry, "amp_up": carry["amp"]}
    return carry, new_head


def main(argv=None):
    t_launch = time.time()
    cfg = parse(argv)
    flags = [f for f in cfg.xla_flags.split(",") if f.startswith("--")]
    if flags:
        os.environ["XLA_FLAGS"] = " ".join([os.environ.get("XLA_FLAGS", "")] + flags).strip()
    if getattr(cfg, "motion_train_wm", False) and (Path(cfg.log_dir) / "config.json").exists():
        raise FileExistsError(f"Choose a fresh log directory for the WM run: {cfg.log_dir}")
    os.makedirs(os.path.join(cfg.log_dir, "ckpt", "full"), exist_ok=True)
    with open(os.path.join(cfg.log_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=1)
    xs = InstantAMP(cfg)
    carry = xs.init(jax.random.PRNGKey(cfg.seed))
    if cfg.init_g1_full:
        carry = initialize_g1_style(carry, cfg.init_g1_full)
    actor_path = getattr(cfg, "init_policy", "")
    full_path = getattr(cfg, "init_full", "")
    if full_path or actor_path:
        new_head = None
        new_gait = None
        if full_path and xs.sim.mixed and "amp_up" in carry:
            # The full-checkpoint directory can live on another volume via a symlink.
            previous_cfg = Path(full_path).absolute().parents[2] / "config.json"
            old = json.loads(previous_cfg.read_text()) if previous_cfg.is_file() else {}
            old_conditioned = bool(old.get("wm_floor_amp") and not old.get("wm_getup_only") and old.get("gait_ls_scale"))
            if (bool(old.get("gait_ls_scale")) != bool(getattr(cfg, "gait_ls_scale", 0.)) or
                    old.get("gait_amp_stride", 1) != getattr(cfg, "gait_amp_stride", 1) or
                    old_conditioned != getattr(cfg, "command_gait_amp", False)):
                new_gait = carry["amp"]
                old_dim = xs.sim.amp_dim + (3 if old_conditioned else 0)
                old_gait = jax.eval_shape(lambda k: amp.init(k, old.get("amp_frames", cfg.amp_frames), frame_dim=old_dim), jax.random.key(0))
                carry = {**carry, "amp": old_gait}
            carry, new_head = mixed_recovery_restore_template(carry, cfg, old, xs.sim)
        carry = initialize_native_learner(carry, full_path, actor_path, (xs.ode_dmax, xs.ode_amax) if xs.ode else None,
                                          getattr(cfg, "rise_anchor", 0.))
        if new_head is not None:
            carry["amp_up"] = new_head
            print("[instantamp] Initialized the new recovery AMP discriminator. Restored the actor and other learner state.", flush=True)
        if new_gait is not None:
            carry["amp"] = new_gait
        if xs.frozen_native_prior is not None:
            # A full online checkpoint may carry a different native discriminator.
            # Restore its support critic, then honor the requested fixed source prior.
            carry["amp_up"]["native"].update(xs.frozen_native_prior)
        xs.pi_mask = jax.tree.map(lambda _: False, carry["pi"])
    if full_path:
        if cfg.max_iterations and int(carry["it"]) >= cfg.max_iterations:
            raise ValueError("max_iterations must exceed the checkpoint's absolute iteration")
        if xs.sim.mixed:
            carry["key"] = jax.random.fold_in(carry["key"], carry["it"])
    if getattr(cfg, "motion_train_wm", False) and not full_path and not actor_path:
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
    last_ckpt, n_ckpt, prev_s, prev_env = 0.0, 0, 0.0, 0.0
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
        if xs.sim.recovery:
            st["assist_n"] = float(np.mean([x["assist_n"] for x in cs]))
        if xs.sim.mixed:
            for name in ("gait_falls", "rises"):
                st[name] = sum(int(x[name]) for x in cs)
        if getattr(cfg, "motion_train_wm", False):
            if getattr(cfg, "wm_floor_amp", False):
                st["supported_stand_blend"] = float(np.mean([x["stand_blend"] for x in cs]))
            st["actor_updates"] = int(carry["opt_pi"]["t"])
            st["wm_updates"] = int(carry["opt_wm"]["t"])
        if xs.sim.reference_reward:
            frames = sum(int(x["reference_frames"]) for x in cs)
            if frames:
                st["reference_pose_score"] = sum(float(x["reference_score_sum"]) for x in cs) / frames
                if xs.sim.track_getup:
                    moving = sum(int(x["reference_moving_frames"]) for x in cs)
                    moving_score = sum(float(x["reference_moving_score_sum"]) for x in cs)
                    st["reference_moving_fraction"] = moving / frames
                    if moving:
                        st["reference_moving_pose_score"] = moving_score / moving
                    if moving < frames:
                        total_score = sum(float(x["reference_score_sum"]) for x in cs)
                        st["reference_hold_pose_score"] = (total_score - moving_score) / (frames - moving)
        train_s = time.time() - t0
        it, env_steps = int(carry["it"]), float(carry["env_steps"])
        logf.write(json.dumps({"it": it, "train_s": train_s, "wall_s": time.time() - t_launch, "env_steps": env_steps, "nonfinite": int(carry["nonfinite"]), **st}) + "\n")
        # len: mean length of finished episodes, fell: their share ended by a fall, lin / ang: tracking terms per episode second
        metrics = (f"height {st['rew_target_base_height']:.3f} assist {st['assist_n']:.1f} N" if xs.sim.recovery else
                   f"stand {st['rew_stand']:.3f} blend {st['supported_stand_blend']:.2f}" if getattr(cfg, "wm_floor_amp", False) else
                   f"lin {st['rew_track_lin']:.3f} ang {st['rew_track_ang']:.3f}")
        falls = f"falls {st['gait_falls']} rises {st['rises']}" if xs.sim.mixed else f"fell {st['fell']:.2f}"
        learner = (f"real PPO | G {st['G']:.3f} r {st['r']:.4f}" if xs.real_ppo else
                   f"wm {st['wm']:.4f} e8 {st['wm8']:.4f} | G {st['G']:.3f} r_im {st['r']:.4f}")
        print(f"it {it} | {train_s / 60:.2f} min | {env_steps / 1e6:.1f}M | len {st['ep_len']:.0f} {falls} {metrics} r {st['reward']:.4f} | up {st['up_frac']:.2f} get-ups {st['down_up']:.0f}/{st['down_done']:.0f} | {learner} "
              f"surv {st['surv']:.3f} gn {st['gn']:.2f} | {block / max(train_s - prev_s, 1e-9):.1f} it/s"
              + (f" | h_err {st['h_err8'] * 100:.1f}/{st['h_errK'] * 100:.1f} cm term env {st['fall_env']:.3f} wm {st['fall_wm']:.3f}" if "wm_eval" in progs else "")
              + (f" | amp d {st['amp_d_demo']:.2f}/{st['amp_d_robot']:.2f} r {st['r_amp']:.4f} use {st['amp_usable']:.2f} ramp {st['amp_ramp']:.2f}" if xs.amp else "")
              + (f" | getup d {st['ampg_d_demo']:.2f}/{st['ampg_d_robot']:.2f} use {st['ampg_usable']:.2f}" if xs.amp and xs.getup else "")
              + (f" | V end {st['v_end']:.2f} loss {st['vloss']:.4f}" if xs.use_critic else "")
              + (f" | ppo clip {st['ppo_clipfrac']:.2f} |logr| {st['ppo_logr']:.2f} gn {st['ppo_gn']:.3f} r {st['ppo_r']:.4f}" if xs.ppo else "")
              + (f" cos bptt {st['ppo_cos_bp']:.3f} halves {st['ppo_cos_half']:.3f}" if "ppo_cos_bp" in st else "")
              + (f" | gated wm {st['wm_gated']:.2f} gn {st['gn_gated']:.2f}" if st["wm_gated"] or st["gn_gated"] else "")
              + (f" | diverged {st['diverged']:.0f}" if st["diverged"] else ""), flush=True)  # fmt: skip
        prev_s, prev_env = train_s, env_steps
        finished = (cfg.minutes > 0 and train_s >= 60 * cfg.minutes) or (cfg.max_iterations > 0 and it >= cfg.max_iterations)
        checkpoint_due = (it % cfg.ckpt_iterations == 0 if cfg.ckpt_iterations else
                          train_s - last_ckpt >= 60 * cfg.ckpt_minutes)
        if checkpoint_due or finished:
            save_full(os.path.join(cfg.log_dir, "ckpt", "full", f"{n_ckpt:04d}"), carry)
            save_policy(os.path.join(cfg.log_dir, "ckpt", f"{n_ckpt:04d}"), jax.device_get(xs.export(carry)),
                        {"it": it, "train_s": train_s, "env_steps": env_steps}, ode=(xs.ode_dmax, xs.ode_amax) if xs.ode else None)  # fmt: skip
            last_ckpt, n_ckpt = train_s, n_ckpt + 1
        if finished:
            break
    print(f"done: {it} iterations, {env_steps / 1e6:.1f}M env steps, {train_s / 60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
