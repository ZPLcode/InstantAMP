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

import argparse
import dataclasses
import gc
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

from . import amp, critic, env, nets, optim
from . import buffer as rb
from .sim import Sim, motion_dirs, unflatten

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
    seed: int = 0
    log_dir: str = ""
    jax_cache: str = "~/.cache/instantamp/jax"
    xla_flags: str = "--xla_gpu_enable_triton_gemm=false"

    robot: str = "g1"  # select the articulation and motion banks

    def __post_init__(self):
        walk, floor = motion_dirs(self.robot)
        self.walk_dir = self.walk_dir or walk
        self.amp_dir = self.amp_dir or walk
        self.floor_dir = self.floor_dir or floor
        self.amp_getup_dir = self.amp_getup_dir or floor
        self.log_dir = self.log_dir or ("runs/instantamp" if self.robot == "g1" else "runs/zenbot23_getup")


@dataclasses.dataclass
class ZenbotGetUpConfig(Config):
    """legged_lab ZenbotGetUpEnvCfg task with the InstantAMP world model and hybrid policy optimizer."""
    robot: str = "zenbot23_getup"
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
    reference_weight: float = 5.0  # per second while the selected native recovery clip is playing
    critic_hidden: str = "512,256,128"
    ppo_len: int = 24
    warmup_steps: int = 40
    max_iterations: int = 100_000
    minutes: float = 0.


class InstantAMP:
    def __init__(self, cfg: Config):
        self.cfg = c = cfg
        set_jax(c.jax_cache)
        self.sim = Sim(c.num_envs, c.nconmax, c.njmax, c.walk_dir, c.floor_dir, bool(c.random_start), c.seed, c.reset_floor, c.prob_shove, bool(c.push),
                       robot=c.robot, amp_wrists=bool(c.amp_wrists), reference_weight=getattr(c, "reference_weight", 0.))
        self.J = self.JS = self.sim.J  # joints (actions, targets) and the world model's q, qd
        self.ode = bool(c.ode)
        self.ode_dmax = c.ode_dmax if c.ode_dmax > 0 else None
        self.ode_amax = c.ode_amax if c.ode_amax > 0 else None
        self.amp = c.amp_w > 0
        if self.amp:
            self.demos = amp.Demos(c.amp_dir, self.sim, c.amp_frames)
            assert c.amp_gate in ("track_shift", "moving", "track", "none"), c.amp_gate
            self.getup = bool(c.amp_getup)
            if self.getup:
                from .getup_amp import GetupDemos

                self.getup_demos = GetupDemos(c.amp_getup_dir, self.sim, c.amp_frames)
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
        rec_t = jax.eval_shape(lambda s: self.record(s, *self.sim.step(s, jnp.zeros((c.num_envs, J))), jnp.zeros((c.num_envs, J)), jnp.zeros((c.num_envs, J)))[0], e)
        rec_t = rb.flatten({**rec_t, "ok": jax.ShapeDtypeStruct((c.num_envs,), jnp.float32)})
        self.xi_dim = rec_t["xi"].shape[-1]
        self.wm_in = (13 + 2 * self.JS) + self.n_soles + rec_t["target"].shape[-1] + self.xi_dim
        pi = nets.pi_init(ks[1], self.sim.obs_dim, J)
        wm, wm_b = nets.wm_init(ks[2], self.wm_in, 12 + 2 * self.JS, widths(c.wm_hidden))
        self.pi_mask = jax.tree.map(lambda _: False, pi)
        self.wm_mask = {"net": nets.muon_mask(wm["net"]) if c.wm_muon else jax.tree.map(lambda _: False, wm["net"])}
        carry = {"env": e, "buf": rb.init(c.buffer_steps, {k: jnp.zeros(v.shape, v.dtype) for k, v in rec_t.items()}),
                 "pi": pi, "avg": jax.tree.map(jnp.copy, pi), "wm": wm, "wm_b": wm_b, "obs_norm": nets.rn_init(self.sim.obs_dim),
                 "opt_pi": optim.init(pi, self.pi_mask), "opt_wm": optim.init(wm, self.wm_mask),
                 "key": ks[3], "it": jnp.zeros((), jnp.int32), "env_steps": jnp.zeros((), jnp.float32), "nonfinite": jnp.zeros((), jnp.int32),
                 "wm_ema": jnp.zeros((), jnp.float32), "gn_ema": jnp.zeros((), jnp.float32), "wm_ok": jnp.ones((), bool)}  # fmt: skip
        if self.amp:
            carry["amp"] = amp.init(jax.random.fold_in(ks[3], 7), c.amp_frames, frame_dim=self.sim.amp_dim)
            if self.getup:
                carry["amp_up"] = amp.init(jax.random.fold_in(ks[3], 13), c.amp_frames, frame_dim=self.sim.amp_dim)
        if self.use_critic:
            self.critic_dim = 13 + 2 * self.JS + 3 + J + self.xi_dim + 2
            carry["critic"] = critic.init(jax.random.fold_in(ks[3], 11), self.critic_dim, widths(c.critic_hidden))
        strong = lambda x: jnp.asarray(x, dtype=x.dtype) if isinstance(x, jax.Array) and x.weak_type else x  # noqa: E731
        return jax.tree.map(strong, carry)

    # ------------------------------------------------------------------ actor
    def norm_obs(self, obs_norm, obs):
        return jnp.clip(nets.rn_apply(obs_norm, obs), -self.cfg.obs_clip, self.cfg.obs_clip)

    def export(self, carry):
        """The live and averaged policies and the normalizer as a checkpoint holds them."""
        return {"pi": carry["pi"], "avg": carry["avg"], "obs_norm": carry["obs_norm"]}

    def act(self, pi, obs_norm, obs, a_prev, out_bf16=False):
        out = nets.pi_apply(pi, self.norm_obs(obs_norm, obs), out_bf16=out_bf16)
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
        if self.sim.reference is not None:
            rec["reference_clip"] = s.reference_clip
        if self.ppo and self.sim.recovery:
            rec.update(a=a, mu=mu)
        elif self.ppo:  # the action taken, the mean it was drawn around, and the unweighted tracking term (the locomotion style's gate)
            i = env.TERMS.index("track_lin")
            rec.update({"a": a, "mu": mu, "rlin": tr.parts[:, i] / (env.DT * env.WEIGHTS["track_lin"])})
        return rec, tr.stats

    def ingest(self, carry, flat):
        rec = rb.nest(flat)
        S0, S1 = unflatten(rec["s_qpos"], rec["s_qvel"]), unflatten(rec["s1_qpos"], rec["s1_qvel"])
        v0 = self.sim.view_state(S0)
        wb, ok = carry["wm_b"], rec["ok"] > 0
        upd = lambda st, x: nets.rn_update(st, x, weights=ok)  # noqa: E731
        wm_b = {"in_norm": upd(wb["in_norm"], nets.wm_inputs(S0, v0.height, v0.soles, rec["target"], rec["xi"])),
                "out_norm": upd(wb["out_norm"], nets.wm_targets(S0, S1, env.DT)), "inc_norm": upd(wb["inc_norm"], nets.increments(S0, S1))}  # fmt: skip
        return {**carry, "buf": rb.add(carry["buf"], flat), "wm_b": wm_b, "obs_norm": upd(carry["obs_norm"], rec["obs"]),
                "env_steps": carry["env_steps"] + self.cfg.num_envs}  # fmt: skip

    def collect_step(self, carry, key):
        c = self.cfg
        s = carry["env"]
        noisy = (jnp.arange(c.num_envs) * c.det_share) % 1.0 >= c.det_share
        clean = self.act(carry["pi"], carry["obs_norm"], s.obs, s.a_ode)
        a = clean + jnp.where(noisy, c.action_noise, 0.0)[:, None] * jax.random.normal(key, (c.num_envs, self.J))
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
        if self.sim.recovery:
            f_wm = jnp.zeros_like(f_env)
        return {"h_err8": mean(herr, 7), "h_errK": mean(herr, K - 1), "fall_env": (f_env * m).sum() / jnp.maximum(m.sum(), 1),
                "fall_wm": (f_wm * m).sum() / jnp.maximum(m.sum(), 1)}  # fmt: skip

    # ------------------------------------------------------------------ imagination
    def sample_starts(self, buf, key):
        c, B = self.cfg, self.B_im
        k_s, k_u, k_c = jax.random.split(key, 3)
        start = rb.nest(rb.sample_starts(buf, k_s, B, c.amp_frames - 1 if self.amp else 0, down_frac=c.imag_down))
        if self.sim.recovery:
            start["cmd"] = env.Cmd(**start["cmd"])
            return start
        fresh = jax.random.uniform(k_u, (B,)) < c.cmd_fresh
        cmd = env.sample_cmd(k_c, B)
        old = env.Cmd(start["cmd"]["vel"], start["cmd"]["heading"], start["cmd"]["is_heading"] > 0, start["cmd"]["is_standing"] > 0, start["cmd"]["time_left"])
        start["cmd"] = env.where_cmd(fresh, cmd, old)
        S0 = unflatten(start["s_qpos"], start["s_qvel"])
        c_new = env.command(start["cmd"], S0[1])
        # a fresh command replaces the command channels of the newest observation frame (the policy sees it from the start)
        hist = start["obs"]
        lo = (env.HIST - 1) * self.sim.frame_dim + 6
        start["obs"] = jnp.where(fresh[:, None], hist.at[:, lo : lo + 3].set(c_new), hist)
        start["c_obs"] = jnp.where(fresh[:, None], c_new, start["c_obs"])
        return start

    def critic_in(self, S, h, a_last, c_obs, xi, up, t):
        """The critic's input: the invariant state, command, last action, context, whether the robot has been up and the share of its time
        to rise that is left (0 once it has been up)."""
        upf = up.astype(jnp.float32)
        left = jnp.clip(1 - t.astype(jnp.float32) / env.GETUP_STEPS, 0.0, 1.0) * (1 - upf)
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

    def imagine(self, pi, wm, wm_b, obs_norm, start, keys, amp_st=None, critic_st=None, amp_up_st=None):
        c, B, sim = self.cfg, self.B_im, self.sim
        bf16 = c.mm_bf16 in ("imag", "all")
        S0 = unflatten(start["s_qpos"], start["s_qvel"])
        xi, bias = start["xi"], start["bias"]
        use_amp, use_critic = amp_st is not None, critic_st is not None
        if use_amp:
            hist_s = unflatten(start["h_qpos"], start["h_qvel"])
            hf = sim.amp_frame(hist_s)
            f0 = sim.amp_frame(S0)
            win0 = jnp.concatenate((hf, f0[:, None]), 1)
            ramp = amp_st["ramp"]
        zeros = jnp.zeros((B,))

        def step(carry, k):
            S, a_last, a_ode, hist, cmd, c_obs, t, up = carry
            k_o, k_c = jax.random.split(k)
            a = self.act(pi, obs_norm, hist, a_ode, out_bf16=bf16)
            v = sim.view_state(S)
            target = sim.control_targets(a, bias, hist, xi)
            S1 = nets.wm_step(wm, wm_b, S, v.height, v.soles, target, xi, env.DT, out_bf16=bf16)
            v1 = sim.view_state(S1)
            t1 = t + 1
            up1 = up | self.sim.is_up(v1.height, S1[1])  # (up: whether the robot had been up before this step, as in the simulator)
            d = jnp.where(up, self.termination(v1.clearance), self.termination_down(t1, v1, S1[1]))
            if sim.recovery:
                d = (t1 >= sim.episode_steps).astype(jnp.float32)
            recovering = ~up1
            terms = sim.reward_terms(S, S1, v1.kin, a, a_last, c_obs, recovering, d, sim.soft_contact(v1.soles), zeros, sim.q_soft,
                                     torque=sim.robot.actuator_ctrl(S[4], S[5], target[..., -sim.J:]) if sim.recovery else None,
                                     clip=start.get("reference_clip"), step=t1)
            r = sum(sim.weighted(terms).values())
            if use_amp:  # the gates of the two styles at this step: the locomotion one while up, the get-up one while not yet up
                frame1 = sim.amp_frame(S1)
                moving = (jnp.linalg.norm(c_obs[:, :2], axis=-1) + jnp.abs(c_obs[:, 2]) > 0.05).astype(jnp.float32)
                upf = up1.astype(jnp.float32)
                g_loco = jnp.ones_like(upf) if sim.recovery else {"track_shift": moving * upf * (c.amp_track_lo + terms["track_lin"]), "moving": moving * upf,
                          "track": moving * terms["track_lin"], "none": upf}[c.amp_gate]  # fmt: skip
                gate = jnp.stack((g_loco, 1 - upf), -1)
            else:
                frame1, gate = jnp.zeros((B, 1)), jnp.zeros((B, 2))
            cmd1 = env.tick(cmd, k_c)
            c1 = sim.command(cmd1, S1[1])
            hist1 = env.push_frame(hist, sim.obs_frame(S1, a, c1, k_o))
            cin1 = self.critic_in(S1, v1.height, a, c1, xi, up1, t1) if use_critic else zeros[:, None]
            return (S1, a, a, hist1, cmd1, c1, t1, up1), (r, d, frame1, gate, cin1, recovering.astype(jnp.float32))

        if c.imag_remat == "dots":
            step = jax.checkpoint(step, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        elif c.imag_remat == "full":
            step = jax.checkpoint(step)
        init = (S0, start["a_last"], start["a_ode"], start["obs"], start["cmd"], start["c_obs"], start["t"], start["was_up"] > 0)
        _, (r, d, frames, gate, cins, rec) = jax.lax.scan(step, init, keys, unroll=c.imag_unroll)
        if use_amp:
            F = c.amp_frames
            seq = jnp.concatenate((win0, jnp.swapaxes(frames, 0, 1)), 1)
            idx = jnp.arange(1, c.horizon + 1)[:, None] + jnp.arange(F)[None, :]
            wins = seq[:, idx]
            style = jnp.swapaxes(self.style(amp_st, wins), 0, 1)  # (H, B)
            mix = gate[..., 0] * style
            up_full = 0.0  # amp_getup_full: the get-up style at full scale, outside the ramp
            if amp_up_st is not None:
                style_up = jnp.swapaxes(amp.style(amp_up_st, amp.score(amp_up_st["net"], amp_up_st["norm"], wins), c.amp_style_scale), 0, 1)
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
        if use_critic:
            v0 = sim.view_state(S0)
            cin0 = self.critic_in(S0, v0.height, start["a_last"], start["c_obs"], xi, start["was_up"] > 0, start["t"])
            v_end = critic.value(critic_st, cins[-1])
            G = G + c.gamma**c.horizon * alive[-1] * v_end
            aux["v_end"] = v_end.mean()
            aux["traj"] = jax.lax.stop_gradient((jnp.concatenate((cin0[None], cins), 0), r, d))
        if use_amp:
            aux["r_amp"] = (alive_before * r_amp).sum(0).mean() / c.horizon
        return -G.mean(), aux

    # ------------------------------------------------------------------ hybrid PPO
    def style(self, st, windows):
        if self.sim.recovery:
            return amp.recovery_style(st, windows, self.cfg.amp_style_scale)
        return amp.style(st, amp.score(st["net"], st["norm"], windows), self.cfg.amp_style_scale)

    def ppo_batch(self, carry):
        """The newest `ppo_len` real transitions of the PPO envs (from the replay buffer: the collector's own steps, no extra simulation),
        with the imagination's reward (the environment's reward, the two styles with their gates, amp_lerp) and GAE(lambda) advantages
        of the critic. Flat (ppo_len * n_ppo, ...) arrays: obs, a_ode, the action and its behavior mean, the advantage and a mask."""
        c, sim, L, F = self.cfg, self.sim, self.cfg.ppo_len, self.cfg.amp_frames
        buf, n = carry["buf"], len(self.ppo_envs)
        T = (F - 1) + L + 1  # style history, the transitions, the state after the last one (bootstrap)
        rows = rb._rows(buf, buf["size"] - T + jnp.arange(T))
        w = rb.gather(buf, rows[:, None], jnp.asarray(self.ppo_envs)[None, :])  # (T, n, ...)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])  # noqa: E731
        tr = slice(F - 1, F - 1 + L)
        S1 = unflatten(flat(w["s1_qpos"][: T - 1]), flat(w["s1_qvel"][: T - 1]))
        r = w["r"][tr]
        if self.amp:
            feats = sim.amp_frame(S1).reshape(T - 1, n, -1)
            idx = jnp.arange(L)[:, None] + jnp.arange(F)[None, :]  # the window of transition k: post-step frames k .. k + F - 1 of the rows
            wins = jnp.moveaxis(feats[idx], 2, 0).reshape(n * L, F, -1)  # (n, L) order
            score = lambda st: self.style(st, wins).reshape(n, L).T  # noqa: E731
            h1 = S1[0][..., 2].reshape(T - 1, n)[tr]
            R1 = S1[1].reshape(T - 1, n, 3, 3)[tr]
            upf = ((w["was_up"][tr] > 0) | self.sim.is_up(h1, R1)).astype(jnp.float32)
            cob = w["c_obs"][tr]
            moving = (jnp.linalg.norm(cob[..., :2], axis=-1) + jnp.abs(cob[..., 2]) > 0.05).astype(jnp.float32)
            g_loco = jnp.ones_like(upf) if sim.recovery else {"track_shift": moving * upf * (c.amp_track_lo + w["rlin"][tr]), "moving": moving * upf,
                      "track": moving * w["rlin"][tr], "none": upf}[c.amp_gate]  # fmt: skip
            mix = g_loco * score(carry["amp"])
            ramp = carry["amp"]["ramp"]
            up_full = 0.0
            if self.getup:  # (amp_getup_full: the get-up style at full scale, outside the ramp, as in imagination)
                if c.amp_getup_full:
                    up_full = (1 - upf) * score(carry["amp_up"])
                else:
                    mix = mix + (1 - upf) * score(carry["amp_up"])
            styled = ramp * mix + up_full if c.amp_getup_full else ramp * mix
            r = (1 - c.amp_lerp) * r + c.amp_lerp * env.DT * styled if c.amp_lerp > 0 else r + c.amp_w * env.DT * styled
        st = slice(F - 1, T)  # the L + 1 states s_0 .. s_L
        S = unflatten(flat(w["s_qpos"][st]), flat(w["s_qvel"][st]))
        cin = self.critic_in(S, sim.view_state(S).height, flat(w["a_last"][st]), flat(w["c_obs"][st]), flat(w["xi"][st]),
                             flat(w["was_up"][st]) > 0, flat(w["t"][st]))  # fmt: skip
        V = critic.value(carry["critic"], cin).reshape(L + 1, n)
        done, term = w["done"][tr], w["term"][tr]
        trunc = done * (1 - term)  # a timeout: bootstrap with the value of the last state, s_k's (the state after it is not stored)
        delta = r + c.gamma * (1 - done) * V[1:] + c.gamma * trunc * V[:-1] - V[:-1]

        def back(A, x):
            d_k, dn = x
            A = d_k + c.gamma * c.lam * (1 - dn) * A
            return A, A

        _, adv = jax.lax.scan(back, jnp.zeros((n,)), (delta, done), reverse=True)
        mask = w["ok"][tr]
        mean = (adv * mask).sum() / jnp.maximum(mask.sum(), 1.0)
        std = jnp.sqrt(((adv - mean) ** 2 * mask).sum() / jnp.maximum(mask.sum(), 1.0)) + 1e-6
        out = {"obs": w["obs"][tr], "a_ode": w["a_ode"][tr], "a": w["a"][tr], "mu": w["mu"][tr], "adv": (adv - mean) / std, "mask": mask}
        out = jax.lax.stop_gradient({k: flat(v) for k, v in out.items()})
        return out, {"ppo_r": (r * mask).sum() / jnp.maximum(mask.sum(), 1.0)}

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
        return -(surr * m).sum() / nm, aux

    def policy_update(self, carry, start, key, ppo_b=None):
        c = self.cfg
        (_, aux), grads = jax.value_and_grad(self.imagine, has_aux=True)(carry["pi"], carry["wm"], carry["wm_b"], carry["obs_norm"], start,
                                                                         jax.random.split(key, c.horizon), carry.get("amp"), carry.get("critic"),
                                                                         carry.get("amp_up"))  # fmt: skip
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
            aux = {**aux, **paux, "ppo_gn": npp, "ppo_cos_bp": cos_bp, "ppo_cos_half": cos_half}
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
            out["critic"], aux["vloss"] = critic.fit(carry["critic"], x, r, d, jax.random.fold_in(key, 5), c.gamma, c.lam, c.critic_lr, c.critic_epochs, c.critic_mb)
        return out, {**aux, "gn": gn, "gn_gated": (finite & ~ok).astype(jnp.float32)}

    # ------------------------------------------------------------------ iteration programs
    SUMMED = ("done", "terminated", "len_sum", "rew_sums", "diverged", "down_done", "down_up", "pushes",
              "reference_score_sum", "reference_frames")

    def collect_iter(self, carry):
        key, k_c = jax.random.split(carry["key"])
        carry, st = jax.lax.scan(self.collect_step, {**carry, "key": key}, jax.random.split(k_c, self.cfg.env_steps))
        return carry, {k: (v.sum(0) if k in self.SUMMED else v.mean(0)) for k, v in st.items()}

    def amp_iter(self, carry):
        """The motion prior's critic on windows of the newest motion of the noise-free envs (up in every frame and up since, a
        moving command at the newest frame) against windows of the clips; then the ramp of the style reward."""
        c, sim, F = self.cfg, self.sim, self.cfg.amp_frames

        def one(st, k):
            k_w, k_d, k_g = jax.random.split(k, 3)
            w = rb.sample_windows(carry["buf"], k_w, c.amp_batch, F, recent=c.amp_recent, envs=self.amp_envs)
            S = unflatten(w["s_qpos"], w["s_qvel"])
            feats = sim.amp_frame(S)
            up = self.sim.is_up(S[0][..., 2], S[1]).all(-1) & (w["was_up"] > 0).all(-1)
            cmd = w["c_obs"][:, -1]
            moving = jnp.linalg.norm(cmd[:, :2], axis=-1) + jnp.abs(cmd[:, 2]) > 0.05
            ok = (w["mask"][:, -1] > 0) & up & moving
            if sim.recovery:
                ok = w["mask"][:, -1] > 0
            st, aux = amp.train_step(st, self.demos.sample(k_d, c.amp_batch), feats, ok.astype(jnp.float32), k_g, c.amp_lr, recovery=sim.recovery)
            return st, {**aux, "usable": ok.mean()}

        def one_up(st, k):  # the get-up prior: windows in which the robot has not been up, against not-up windows of the get-up clips
            k_w, k_d, k_g = jax.random.split(k, 3)
            w = rb.sample_windows(carry["buf"], k_w, c.amp_batch, F, recent=c.amp_recent, envs=self.amp_envs)
            S = unflatten(w["s_qpos"], w["s_qvel"])
            feats = sim.amp_frame(S)
            ok = (w["mask"][:, -1] > 0) & (w["was_up"] == 0).all(-1)
            st, aux = amp.train_step(st, self.getup_demos.sample(k_d, c.amp_batch), feats, ok.astype(jnp.float32), k_g, c.amp_lr)
            return st, {**aux, "usable": ok.mean()}

        key, k, k2 = jax.random.split(carry["key"], 3)
        st, aux = jax.lax.scan(one, carry["amp"], jax.random.split(k, c.amp_disc_steps))
        ramp = jnp.clip((carry["it"] - c.amp_delay) / float(c.amp_ramp), 0., 1.) if c.amp_ramp else (carry["it"] >= c.amp_delay).astype(jnp.float32)
        aux = jax.tree.map(lambda x: x[-1], aux)
        out = {**carry, "amp": {**st, "ramp": ramp}, "key": key}
        stats = {"amp_d_demo": aux["d_demo"], "amp_d_robot": aux["d_robot"], "amp_gp": aux["gp"], "amp_usable": aux["usable"], "amp_ramp": ramp}
        if self.getup:
            st2, aux2 = jax.lax.scan(one_up, carry["amp_up"], jax.random.split(k2, c.amp_disc_steps))
            aux2 = jax.tree.map(lambda x: x[-1], aux2)
            out["amp_up"] = st2
            stats.update({"ampg_d_demo": aux2["d_demo"], "ampg_d_robot": aux2["d_robot"], "ampg_usable": aux2["usable"]})
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

        def do(sub):
            ppo_b, pst = self.ppo_batch(carry) if self.ppo else (None, {})

            def one(sub, k):
                k_s, k_u = jax.random.split(k)
                return self.policy_update({**carry, **sub}, self.sample_starts(carry["buf"], k_s), k_u, ppo_b)

            sub, st = jax.lax.scan(one, sub, jax.random.split(k_p, c.pi_updates))
            return sub, {**jax.tree.map(jnp.mean, st), **pst}

        stat_keys = (("G", "r", "surv", "rec_im", "gn", "gn_gated") + (("r_amp",) if self.amp else ()) + (("v_end", "vloss") if self.use_critic else ())
                     + (("ppo_clipfrac", "ppo_logr", "ppo_gn", "ppo_r", "ppo_cos_bp", "ppo_cos_half") if self.ppo else ()))  # fmt: skip

        def skip(sub):
            return sub, {k: jnp.zeros((), jnp.float32) for k in stat_keys}

        active = carry["it"] >= c.wm_only_iters
        sub, ps = jax.lax.cond(active & carry["wm_ok"], do, skip, {k: carry[k] for k in names})
        return {**carry, **sub, "key": key, "it": carry["it"] + 1}, {**ps, "wm_gated": (active & ~carry["wm_ok"]).astype(jnp.float32)}

    def compile_programs(self, carry):
        lowered = {"collect": jax.jit(self.collect_iter, donate_argnums=(0,)).lower(carry),
                   "wm": jax.jit(self.wm_iter, donate_argnums=(0,)).lower(carry),
                   "pi": jax.jit(self.pi_iter, donate_argnums=(0,)).lower(carry)}  # fmt: skip
        if self.amp:
            lowered["amp"] = jax.jit(self.amp_iter, donate_argnums=(0,)).lower(carry)
        if self.cfg.wm_eval:
            lowered["wm_eval"] = jax.jit(lambda cr: self.wm_eval(cr, jax.random.fold_in(cr["key"], 3))).lower(carry)
        with ThreadPoolExecutor(len(lowered)) as ex:
            return dict(zip(lowered, ex.map(lambda low: low.compile(), lowered.values())))


FULL_KEYS = ("pi", "avg", "wm", "wm_b", "obs_norm", "opt_pi", "opt_wm", "amp", "amp_up", "critic", "it", "env_steps", "gn_ema", "wm_ema", "nonfinite")


def save_full(path, carry):
    leaves = jax.tree.leaves(jax.device_get({k: carry[k] for k in FULL_KEYS if k in carry}))
    np.savez(path + ".tmp.npz", **{f"{i:05d}": np.asarray(x) for i, x in enumerate(leaves)})
    os.replace(path + ".tmp.npz", path + ".npz")


def load_full(carry, path):
    part = {k: carry[k] for k in FULL_KEYS if k in carry}
    leaves, tdef = jax.tree.flatten(part)
    z = np.load(path)
    assert len(z.files) == len(leaves), f"{path} holds {len(z.files)} arrays, this run's state has {len(leaves)}: different options?"
    return {**carry, **jax.tree.unflatten(tdef, [jnp.asarray(z[f"{i:05d}"], dtype=x.dtype).reshape(x.shape) for i, x in enumerate(leaves)])}


def save_policy(path, carry, meta, ode=None):
    """The averaged and live policy and the observation normalizer (<path>.npz), and the metadata (<path>.json)."""
    flat = {"obs_norm/mean": np.asarray(carry["obs_norm"]["mean"]), "obs_norm/var": np.asarray(carry["obs_norm"]["var"])}
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
    p = argparse.ArgumentParser(description="InstantAMP: train a humanoid velocity-tracking policy with AMP through a learned world model.")
    select = argparse.ArgumentParser(add_help=False)
    select.add_argument("--robot", choices=("g1", "zenbot23_getup"), default="g1")
    recipe = ZenbotGetUpConfig if select.parse_known_args(argv)[0].robot == "zenbot23_getup" else Config
    for f in dataclasses.fields(recipe):
        p.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return recipe(**vars(p.parse_args(argv)))


def main(argv=None):
    t_launch = time.time()
    cfg = parse(argv)
    flags = [f for f in cfg.xla_flags.split(",") if f.startswith("--")]
    if flags:
        os.environ["XLA_FLAGS"] = " ".join([os.environ.get("XLA_FLAGS", "")] + flags).strip()
    os.makedirs(os.path.join(cfg.log_dir, "ckpt", "full"), exist_ok=True)
    with open(os.path.join(cfg.log_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=1)
    xs = InstantAMP(cfg)
    carry = xs.init(jax.random.PRNGKey(cfg.seed))
    print(f"[instantamp] {cfg.num_envs} envs, world-model input {xs.wm_in}, actor input {xs.sim.obs_dim}, "
          f"buffer {sum(v.nbytes for v in jax.tree.leaves(carry['buf'])) / 2**30:.2f} GiB, device {jax.devices()[0]}", flush=True)  # fmt: skip
    t_c = time.time()
    progs = xs.compile_programs(carry)
    print(f"[instantamp] compile {time.time() - t_c:.0f} s, start-up {time.time() - t_launch:.0f} s", flush=True)
    gc.collect()
    gc.freeze()
    logf = open(os.path.join(cfg.log_dir, "log.jsonl"), "a", buffering=1)
    t0 = time.time()
    for _ in range(-(-cfg.warmup_steps // cfg.env_steps)):
        carry, _ = progs["collect"](carry)
    last_ckpt, n_ckpt, prev_s, prev_env = 0.0, 0, 0.0, 0.0
    while True:
        cs, ls = [], []
        remaining = cfg.max_iterations - int(carry["it"]) if cfg.max_iterations else cfg.log_every
        for _ in range(min(remaining, cfg.log_every)):
            carry, cst = progs["collect"](carry)
            ast = {}
            if "amp" in progs:
                carry, ast = progs["amp"](carry)
            carry, wst = progs["wm"](carry)
            carry, pst = progs["pi"](carry)
            cs.append(cst)
            ls.append({**wst, **pst, **ast})
        cs, ls, ev = jax.device_get((cs, ls, progs["wm_eval"](carry) if cfg.wm_eval else {}))
        n_done = max(sum(float(x["done"]) for x in cs), 1.0)
        rew = np.sum([x["rew_sums"] for x in cs], 0) / n_done
        st = {"ep_len": sum(float(x["len_sum"]) for x in cs) / n_done, "done": n_done, "fell": sum(float(x["terminated"]) for x in cs) / n_done,
              **{f"rew_{k}": float(v) for k, v in zip(xs.sim.terms, rew)},
              **{k: float(np.mean([x[k] for x in cs])) for k in ("reward", "nacon", "nonfinite_env", "up_frac", "term_geom", "self_hits", "foot_contact")},
              **{k: sum(float(x[k]) for x in cs) for k in ("diverged", "down_done", "down_up", "pushes")},
              **{k: float(np.mean([x[k] for x in ls])) for k in ls[0]}, **{k: float(v) for k, v in ev.items()}}  # fmt: skip
        if xs.sim.recovery:
            st["assist_n"] = float(np.mean([x["assist_n"] for x in cs]))
        if xs.sim.reference is not None:
            frames = sum(int(x["reference_frames"]) for x in cs)
            if frames:
                st["reference_pose_score"] = sum(float(x["reference_score_sum"]) for x in cs) / frames
        train_s = time.time() - t0
        it, env_steps = int(carry["it"]), float(carry["env_steps"])
        logf.write(json.dumps({"it": it, "train_s": train_s, "wall_s": time.time() - t_launch, "env_steps": env_steps, "nonfinite": int(carry["nonfinite"]), **st}) + "\n")
        # len: mean length of finished episodes, fell: their share ended by a fall, lin / ang: tracking terms per episode second
        metrics = (f"height {st['rew_target_base_height']:.3f} assist {st['assist_n']:.1f} N" if xs.sim.recovery else
                   f"lin {st['rew_track_lin']:.3f} ang {st['rew_track_ang']:.3f}")
        print(f"it {it} | {train_s / 60:.2f} min | {env_steps / 1e6:.1f}M | len {st['ep_len']:.0f} fell {st['fell']:.2f} {metrics} r {st['reward']:.4f} | up {st['up_frac']:.2f} get-ups {st['down_up']:.0f}/{st['down_done']:.0f} | wm {st['wm']:.4f} e8 {st['wm8']:.4f} | "
              f"G {st['G']:.3f} r_im {st['r']:.4f} surv {st['surv']:.3f} gn {st['gn']:.2f} | {cfg.log_every / max(train_s - prev_s, 1e-9):.1f} it/s"
              + (f" | h_err {st['h_err8'] * 100:.1f}/{st['h_errK'] * 100:.1f} cm term env {st['fall_env']:.3f} wm {st['fall_wm']:.3f}" if cfg.wm_eval else "")
              + (f" | amp d {st['amp_d_demo']:.2f}/{st['amp_d_robot']:.2f} r {st['r_amp']:.4f} use {st['amp_usable']:.2f} ramp {st['amp_ramp']:.2f}" if xs.amp else "")
              + (f" | getup d {st['ampg_d_demo']:.2f}/{st['ampg_d_robot']:.2f} use {st['ampg_usable']:.2f}" if xs.amp and xs.getup else "")
              + (f" | V end {st['v_end']:.2f} loss {st['vloss']:.4f}" if xs.use_critic else "")
              + (f" | ppo clip {st['ppo_clipfrac']:.2f} |logr| {st['ppo_logr']:.2f} gn {st['ppo_gn']:.3f} r {st['ppo_r']:.4f} cos bptt {st['ppo_cos_bp']:.3f} halves {st['ppo_cos_half']:.3f}" if xs.ppo else "")
              + (f" | gated wm {st['wm_gated']:.2f} gn {st['gn_gated']:.2f}" if st["wm_gated"] or st["gn_gated"] else "")
              + (f" | diverged {st['diverged']:.0f}" if st["diverged"] else ""), flush=True)  # fmt: skip
        prev_s, prev_env = train_s, env_steps
        finished = (cfg.minutes > 0 and train_s >= 60 * cfg.minutes) or (cfg.max_iterations > 0 and it >= cfg.max_iterations)
        if train_s - last_ckpt >= 60 * cfg.ckpt_minutes or finished:
            save_full(os.path.join(cfg.log_dir, "ckpt", "full", f"{n_ckpt:04d}"), carry)
            save_policy(os.path.join(cfg.log_dir, "ckpt", f"{n_ckpt:04d}"), jax.device_get(xs.export(carry)),
                        {"it": it, "train_s": train_s, "env_steps": env_steps}, ode=(xs.ode_dmax, xs.ode_amax) if xs.ode else None)  # fmt: skip
            last_ckpt, n_ckpt = train_s, n_ckpt + 1
        if finished:
            break
    print(f"done: {it} iterations, {env_steps / 1e6:.1f}M env steps, {train_s / 60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
