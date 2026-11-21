"""Learn Zenbot get-up feedback with real MuJoCo rollouts and the LeggedLab PPO/AMP updates.

The 480-input actor uses its encoder and IMU history. Training can start partway
through motions, guide a subset of drives with the retargeted joint references and
distill those targets on nearby states. The guidance fades to zero. The critic
uses the reference and floating-base state during training.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import amp, env, nets, optim, zenbot
from .inference import load, policy_output
from .sim import Sim, unflatten
from .train import save_policy, set_jax
from .train_g1_getup_amp import GetUpAMP, Settings as PPOSettings
from .train_getup_ppo import gae
from .zenbot_rise import residual_weight


@dataclasses.dataclass
class Settings(PPOSettings):
    log_dir: str = "runs/zenbot23_getup_ppo_amp"
    floor_dir: str = "data/zenbot23_getup/recovery"
    num_envs: int = 256
    iterations: int = 2000
    save_every: int = 200
    log_every: int = 20
    rollout: int = 16
    epochs: int = 3
    minibatches: int = 4
    disc_batch: int = 256
    assist_newtons: float = 0.
    reference_weight: float = 12.
    init_std: float = .65
    init_lr: float = .0003
    phase_resets: float = .67
    episode_steps: int = 190
    guide_steps: int = 900
    guidance: float = .85
    action_rate_l2_weight: float = -.015
    sensor_noise_min: float = 1.
    sensor_noise_max: float = 1.
    library: str = ""  # state-only physical teacher with a bounded PPO correction
    teacher_rollouts: str = ""  # closed-loop, successful physical demonstrations


class PhaseSim(Sim):
    def with_critic(self, state):
        pose = self.reference.poses[self.index(state, 2)]
        critic_obs = jnp.concatenate((state.obs, state.data.qvel[:, :3], state.data.qpos[:, 2:3], pose), axis=-1)
        return state._replace(critic_obs=critic_obs)

    def index(self, state, lookahead):
        clip = state.reference_clip
        frame = jnp.minimum(state.reference_start + state.t + lookahead, self.reference.lengths[clip] - 1)
        return self.reference.offsets[clip] + frame

    def init(self, key):
        return self.with_critic(super().init(key))

    def step(self, s, a, *args, **kwargs):
        nxt, transition = super().step(s, a, *args, **kwargs)
        return self.with_critic(nxt), transition


class ZenbotAMP(GetUpAMP):
    def __init__(self, cfg):
        self.c = cfg
        if cfg.reference_weight <= 0 or not 0 < cfg.phase_resets <= 1 or cfg.guide_steps < 0:
            raise ValueError("Use a positive tracking weight and a nonzero share of reference starts")
        if not (0 <= cfg.guidance <= 1) or cfg.num_envs * cfg.rollout % cfg.minibatches:
            raise ValueError("Invalid guidance or PPO minibatch size")
        if cfg.rise_anchor or cfg.critic_warmup:
            raise ValueError("This recipe uses the decaying reference guide for initialization")
        self.sim = PhaseSim(cfg.num_envs, robot="zenbot23_getup", floor_dir=cfg.floor_dir, reset_floor=1.,
                            push=False, random_start=True, assist_newtons=cfg.assist_newtons,
                            reference_weight=cfg.reference_weight, reference_hold_scale=cfg.reference_hold_scale,
                            action_rate_l2_weight=cfg.action_rate_l2_weight,
                            zenbot_phase_resets=cfg.phase_resets, zenbot_episode_steps=cfg.episode_steps)
        if not 0. <= cfg.sensor_noise_min <= cfg.sensor_noise_max:
            raise ValueError("Expected ordered nonnegative sensor noise amplitudes")
        if (cfg.sensor_noise_min, cfg.sensor_noise_max) != (1., 1.):
            scale = np.linspace(cfg.sensor_noise_min, cfg.sensor_noise_max, cfg.num_envs, dtype=np.float32)
            self.sim.obs_noise = scale[:, None] * self.sim.obs_noise
        self.disc_batch = cfg.disc_batch or cfg.num_envs * cfg.rollout // cfg.minibatches
        self.demos = amp.Demos(cfg.floor_dir, self.sim, 10)
        self.library = None
        if cfg.library:
            base, self.base_norm, ode = load(cfg.library)
            if not hasattr(base, "library") or ode is not None or cfg.guidance or cfg.init_policy:
                raise ValueError("A library residual needs a direct Zenbot library and zero drive guidance")
            self.library, self.residual_scale, self.base_layers = base.library, base.scale, base.layers
            self.residual_gate = base.gate
            if base.stance is not None:
                raise ValueError("Train this rise residual before attaching the stateful stance controller")
        self.examples = self.physical_examples(Path(cfg.teacher_rollouts)) if cfg.teacher_rollouts else None
        self.drive_share = cfg.guidance * jnp.clip(jnp.arange(cfg.num_envs) / (.65 * cfg.num_envs) - .20, 0., 1.)

    def init(self):
        live, optimizer, disc = super().init()
        if self.library is not None:
            ac, _, lr = optimizer
            ac = {**ac, "pi": self.base_layers}
            self.obs_norm = self.base_norm
            self.mask = jax.tree.map(lambda _: False, ac)
            optimizer = ac, optim.init(ac, self.mask), lr
        return live, optimizer, disc

    def motor_action(self, raw, observations):
        if self.library is not None:
            base, distance = self.library.action_with_distance(observations)
            strength = self.residual_scale * residual_weight(distance, self.residual_gate)
            return base + strength[..., None] * jnp.tanh(raw)
        return raw

    def export(self, destination, params, meta):
        """Preserve the validated state library and attach the standalone trained residual."""
        pairs = {"pi": params, "avg": params, "obs_norm": self.obs_norm}
        if not self.c.library:
            save_policy(str(destination), pairs, meta)
            return
        staging = destination.parent / ".writing"
        staging.mkdir(exist_ok=True)
        neural = staging / destination.name
        save_policy(str(neural), pairs, meta)
        with np.load(neural.with_suffix(".npz"), allow_pickle=False) as actor, np.load(self.c.library, allow_pickle=False) as base:
            weights = {k: actor[k] for k in actor.files}
            weights.update({k: base[k] for k in base.files if k.startswith("zenbot_library/")})
        archive = staging / "composite.npz"
        np.savez(archive, **weights)
        archive.replace(destination.with_suffix(".npz"))
        neural.with_suffix(".json").replace(destination.with_suffix(".json"))
        neural.with_suffix(".npz").unlink()

    def physical_examples(self, directory):
        """Rebuild actor histories before each successful teacher motor command."""
        states, actions = [], []
        sim = self.sim
        for path in sorted(directory.glob("*/rollout.npz")):
            with np.load(path, allow_pickle=False) as f:
                qpos, qvel, action = f["qpos"], f["qvel"], f["action"]
                if len(qpos) != len(action) + 1 or qvel.shape != (len(qpos), 29) or action.shape[1] != sim.J:
                    raise ValueError(f"Invalid physical teacher trajectory {path}")
                if "terminal_feet_weight" not in f or len(f["terminal_feet_weight"]) < 50:
                    continue
                force = f["terminal_feet_weight"]
                upright = 1. - 2. * np.sum(qpos[-50:, 4:6] ** 2, axis=-1)
                if np.min(upright) <= np.cos(np.pi / 6) or np.mean(force[-50:] > .5) < .9:
                    continue
                old = np.concatenate((np.zeros_like(action[:1]), action[:-1]))
                frame = np.asarray(sim.obs_frame(unflatten(jnp.asarray(qpos[:-1], dtype=jnp.float32),
                                                          jnp.asarray(qvel[:-1], dtype=jnp.float32)),
                                                 jnp.asarray(old, dtype=jnp.float32), None, None))
                times = np.arange(len(action))[:, None] + np.arange(-4, 1)
                history = frame[np.maximum(times, 0)].reshape(len(action), -1)
                # Most captured states in each long rollout are quiet. Preserve enough moving examples.
                rising = max(1, len(action) - len(force))
                pick = np.r_[np.arange(rising), np.arange(rising, len(action), 5)]
                states.append(history[pick])
                actions.append(action[pick])
        if not actions:
            raise ValueError(f"No supported physical get-up demonstrations in {directory}")
        return jnp.asarray(np.concatenate(states), dtype=jnp.float32), jnp.asarray(np.concatenate(actions), dtype=jnp.float32)

    def collect(self, live, ac, disc, key, guide):
        sim, c = self.sim, self.c

        def step(live, key):
            s, hist, replay, written = live
            mu = policy_output(ac["pi"], self.obs_norm, s.obs)
            action = mu + jnp.maximum(ac["std"], .01) * jax.random.normal(key, mu.shape)
            i = sim.index(s, 3)
            q = s.world.walk["q"][i]
            demand = q + np.asarray(zenbot.KD / zenbot.KP) * s.world.walk["qd"][i]
            label = jnp.clip((demand - zenbot.DEFAULT_Q) / zenbot.ACTION_SCALE, -15., 15.)
            motor = self.motor_action(action, s.obs)
            applied = motor + guide * self.drive_share[:, None] * (label - motor)
            value = nets.mlp_apply(ac["v"], s.critic_obs, jax.nn.elu)[:, 0]
            nxt, tr = sim.step(s, applied)
            end = tr.terminated | tr.truncated
            frame = sim.amp_frame(unflatten(tr.qpos1, tr.qvel1))
            window = jnp.concatenate((hist[:, 1:], frame[:, None]), axis=1)
            first = sim.amp_frame(unflatten(nxt.data.qpos, nxt.data.qvel))
            hist = jnp.where(end[:, None, None], first[:, None], window)
            style = env.DT * amp.recovery_style(disc, hist, 50., leggedlab=True)
            reward = .5 * (tr.reward + style)
            replay = replay.at[written % len(replay)].set(hist)
            pose = sim.reference_pose(unflatten(s.data.qpos, s.data.qvel))
            score = sim.reference.score(pose, s.reference_clip, s.reference_start + s.t)
            # Teacher drives are most informative near the demonstrated physical state.
            weight = jnp.clip((score - .30) / .50, 0., 1.) ** 2
            row = (s.obs, s.critic_obs, mu, action, value, reward, end, tr.truncated, label, weight)
            stats = jnp.array((tr.reward.mean(), style.mean(), tr.qpos1[:, 2].mean(),
                               tr.stats["down_done"], tr.stats["down_up"], tr.stats["diverged"], weight.mean()))
            return (nxt, hist, replay, written + 1), (row, stats)

        live, (rows, stats) = jax.lax.scan(step, live, jax.random.split(key, c.rollout))
        obs, critic, mu, action, value, reward, end, timeouts, label, weight = rows
        bootstrap = nets.mlp_apply(ac["v"], live[0].critic_obs, jax.nn.elu)[:, 0]
        adv, returns = gae(reward, value, bootstrap, end, timeouts, .99, .95)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat = lambda arr: arr.reshape(-1, *arr.shape[2:])
        batch = tuple(map(flat, (obs, critic, mu, action, value, adv, returns)))
        return live, batch, (flat(label), flat(weight)), stats.mean(0)

    def imitate(self, ac, opt, obs, labels, key, guide):
        index = jax.random.randint(key, (512,), 0, len(obs))
        if self.examples is None:
            target, weight = (x[index] for x in labels)
            inputs = obs[index]
        else:
            histories, targets = self.examples
            index = jax.random.randint(key, (512,), 0, len(histories))
            inputs, target = histories[index], targets[index]
            weight = jnp.ones((len(index),), dtype=obs.dtype)

        def objective(pi):
            prediction = self.motor_action(policy_output(pi, self.obs_norm, inputs), inputs)
            error = jnp.mean((prediction - target) ** 2, axis=-1)
            return jnp.sum(weight * error) / jnp.maximum(1., jnp.sum(weight))

        loss, grad = jax.value_and_grad(objective)(ac["pi"])
        grad, _ = optim.clip_by_global_norm(grad, 1.)
        pi, opt = optim.step(ac["pi"], grad, opt, self.mask["pi"], .00035 * guide)
        return {**ac, "pi": pi}, opt, loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in dataclasses.fields(Settings):
        parser.add_argument("--" + field.name, type=type(field.default), default=field.default)
    c = Settings(**vars(parser.parse_args()))
    p = Path(c.log_dir)
    (p / "ckpt").mkdir(parents=True, exist_ok=False)
    (p / "config.json").write_text(json.dumps({**dataclasses.asdict(c), "robot": "zenbot23_getup",
                                               "learner": "on_policy_ppo_amp"}, indent=2))
    set_jax("~/.cache/instantamp/jax")
    trainer = ZenbotAMP(c)
    samples = 0 if trainer.examples is None else len(trainer.examples[0])
    print(f"Zenbot on {jax.devices()[0]}, {c.num_envs} worlds, {samples} physical teacher states, "
          f"{c.assist_newtons:.1f} N initial assistance", flush=True)
    live, optimizer, disc = trainer.init()
    if c.library:
        trainer.export(p / "ckpt/000000", jax.device_get(optimizer[0]["pi"]),
                       dict(it=0, env_steps=0, train_s=0., residual_scale=float(trainer.residual_scale)))
    auxopt = optim.init(optimizer[0]["pi"], trainer.mask["pi"])
    key = jax.random.PRNGKey(c.seed + 1)
    collect = jax.jit(trainer.collect, donate_argnums=(0,))
    learn = jax.jit(trainer.learn, donate_argnums=(0, 1))
    imitate = jax.jit(trainer.imitate, donate_argnums=(0, 1))
    start = time.monotonic()
    with (p / "log.jsonl").open("x", buffering=1) as log:
        for it in range(1, c.iterations + 1):
            key, kc, kl, ki = jax.random.split(key, 4)
            guide = max(0., 1 - (it - 1) / max(c.guide_steps, 1)) if c.guide_steps else 0.
            live, batch, labels, stats = collect(live, optimizer[0], disc, kc, jnp.float32(guide))
            optimizer, disc, learned = learn(optimizer, disc, batch, live[2], live[3], kl)
            loss = 0.
            if guide > 0:
                ac, opt, lr = optimizer
                ac, auxopt, loss = imitate(ac, auxopt, batch[0], labels, ki, jnp.float32(guide))
                optimizer = ac, opt, lr
            if it == 1 or it % c.log_every == 0 or it % c.save_every == 0 or it == c.iterations:
                names = ("task", "amp", "height", "down_done", "down_up", "diverged", "imitation_weight",
                         "kl", "lr", "policy_loss", "value_loss", "disc_expert", "disc_robot", "imitation_mse")
                st = dict(zip(names, map(float, jax.device_get(jnp.r_[stats, learned, loss]))))
                if not np.isfinite(list(st.values())).all():
                    raise FloatingPointError(st)
                meta = dict(it=it, env_steps=it * c.num_envs * c.rollout, train_s=time.monotonic() - start,
                            guide=c.guidance * guide, assist_n=c.assist_newtons)
                line = json.dumps({**meta, **st})
                print(line, flush=True)
                log.write(line + "\n")
            if it % c.save_every == 0 or it == c.iterations:
                pi = jax.device_get(optimizer[0]["pi"])
                trainer.export(p / "ckpt" / f"{it:06d}", pi, meta)
                flat = jax.tree.leaves(jax.device_get((optimizer, auxopt, disc, key)))
                np.savez(p / "ckpt" / f"{it:06d}.learner.npz", **{f"{i:04d}": x for i, x in enumerate(flat)})


if __name__ == "__main__":
    main()
