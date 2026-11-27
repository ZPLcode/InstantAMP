"""Training recipes for G1 and Zenbot, with run settings for the command line."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from . import env, zenbot, zenbot_models, zenbot_source_gait as source_gait
from .sim import motion_dirs


@dataclasses.dataclass
class Config:
    """G1 training defaults."""

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
    wm_stand_pose_weight: float = 0.  # bounded joint-angle cost around the robot's default double-support stance
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
        if (not np.isfinite(self.wm_stand_pose_weight) or self.wm_stand_pose_weight > 0.
                or self.wm_stand_pose_weight and not self.wm_floor_amp):
            raise ValueError("Stand pose weight must be finite, nonpositive and requires joint floor AMP")
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


@dataclasses.dataclass
class Zenbot26JointConfig(ZenbotJointConfig):
    """26-DoF walk, run and recovery training from scratch."""

    robot: str = zenbot_models.ZENBOT26.mixed_task
    num_envs: int = 4096
    max_iterations: int = 100_000
    wm_only_iters: int = 500
    ckpt_iterations: int = 1000
    full_ckpt_keep: int = 2
    external_pushes: int = 1
    wm_gait_frames: int = 10
    gait_ls_scale: float = 1.5
    getup_amp_scale: float = 10.
    imag_gait_mirror_weight: float = .2
    wm_lift_weight: float = .2
    wm_stand_pose_weight: float = -2.


_RECIPES = {
    "g1": Config,
    zenbot_models.ZENBOT23.mixed_task: Zenbot23JointConfig,
    zenbot_models.ZENBOT26.mixed_task: Zenbot26JointConfig,
}

_RUN_OPTIONS = {
    "log_dir": "Training output directory",
    "num_envs": "Number of parallel environments",
    "max_iterations": "Final training iteration",
    "seed": "Random seed",
    "ckpt_iterations": "Iterations between checkpoints",
    "init_policy": "Initialize the actor from an exported policy",
}


def parse(argv=None) -> Config:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--robot", choices=_RECIPES, default=Config.robot, help="Robot and training recipe")
    selected, remaining = selector.parse_known_args(argv)
    recipe = _RECIPES[selected.robot]
    parser = argparse.ArgumentParser(
        parents=[selector],
        description="Train one policy for walking, running and floor recovery.",
        epilog="Edit training hyperparameters in instantamp/train_config.py.",
    )
    # Register old hyperparameter flags only when a script actually supplies them.
    # Keep argparse's existing support for --name=value and unambiguous prefixes.
    explicit = tuple(arg.partition("=")[0] for arg in remaining if arg.startswith("--") and arg != "--")
    for field in dataclasses.fields(recipe):
        if field.name == "robot":
            continue
        help_text = _RUN_OPTIONS.get(field.name, argparse.SUPPRESS)
        if recipe is Config and field.name == "minutes":
            help_text = "Training time budget in minutes"
        flag = f"--{field.name}"
        if help_text == argparse.SUPPRESS and not flag.startswith(explicit):
            continue
        choices = None
        if field.name == "motion_train_wm":
            choices = (1,)
        elif field.name == "recovery_amp_loss":
            choices = ("least_squares", "wasserstein")
        parser.add_argument(flag, type=type(field.default), choices=choices,
                            default=argparse.SUPPRESS, help=help_text)
    cfg = recipe(**vars(parser.parse_args(argv)))
    if not 0 < cfg.reset_floor < 1 or not cfg.amp_getup or cfg.amp_w <= 0:
        parser.error("Joint training requires walking, running and floor recovery")
    return cfg
