"""Fine-tune an exported G1 get-up controller using real MuJoCo Warp rollouts, PPO and AMP.

Only the bounded proprioceptive rise correction learns. Reference phases, poses, velocities and
contact forces are restricted to reset sampling, training rewards and the AMP discriminator.
"""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import amp, env, g1_getup, g1_inplace, nets, optim
from .evaluate import fallen_starts
from .g1_rise import rise_residual
from .inference import G1StandController, load, policy_memory, policy_step, reset_policy_memory
from .rot import quat_to_mat
from .sim import Sim, unflatten
from .train import set_jax
from .train_getup_ppo import gae


ACTION_RATE_BASE_COEFFICIENT = 2.0


def action_rate_reward(slew, multiplier):
    """Preserve the zero-slew reward and scale the saturating penalty at every action change."""
    unit_penalty = 1.0 / (1.0 + slew) - 1.0
    penalty = multiplier * ACTION_RATE_BASE_COEFFICIENT * unit_penalty
    return ACTION_RATE_BASE_COEFFICIENT + penalty, penalty


class RiseSim(Sim):
    """Initialize the training episode at a source start or a low source pose half a second later."""

    def __init__(self, n, directory, reset_angle=0.):
        self.reset_angle = np.deg2rad(reset_angle)
        super().__init__(
            n,
            robot="g1_getup",
            floor_dir=str(directory),
            reference_weight=10.0,
            random_start=True,
            reset_floor=1.0,
            assist_newtons=0.0,
            push=False,
            obs_noise=False,
            randomize_world=False,
        )
        self.episode_steps = 280
        starts = []
        for protocol in ("source-play", "fallen"):
            names, q, v = fallen_starts(directory, g1_getup.RESET_LIFT, protocol)
            starts.extend(
                (name, sec, x, y) for (name, sec), x, y in zip(names, q, v) if protocol == "source-play" or sec > 0.0
            )
        by_name = {p.name: i for i, p in enumerate(self.reference.files)}
        self.start_q = jnp.stack([row[2] for row in starts])
        self.start_v = jnp.stack([row[3] for row in starts])
        self.start_clip = jnp.array([by_name[row[0]] for row in starts])
        self.start_frame = jnp.array([round(row[1] / env.DT) for row in starts])
        velocity, acceleration, arm_limit = [], [], []
        for p in self.reference.files:
            with np.load(p) as clip:
                v = clip["dof_vel"]
            velocity.append(v)
            acceleration.append(np.diff(v, axis=0, prepend=v[:1]) / env.DT)
            # Local reference envelope, including the bilateral arms' fast valid contact motions.
            rows = np.clip(np.arange(len(v))[:, None] + np.arange(-4, 5), 0, len(v) - 1)
            arm_limit.append(np.percentile(np.abs(v[rows, 15:]), 95, axis=(1, 2)))
        self.ref_velocity = jnp.asarray(np.concatenate(velocity), dtype=jnp.float32)
        self.ref_acceleration = jnp.asarray(np.concatenate(acceleration), dtype=jnp.float32)
        self.arm_envelope = jnp.asarray(np.concatenate(arm_limit), dtype=jnp.float32)

    def _fresh(self, key, world, clip=None, return_source=False):
        choose, noise = jax.random.split(key)
        i = jax.random.randint(choose, (self.N,), 0, len(self.start_q))
        # Interleave exact resets and small physical velocity perturbations.
        v = (
            self.start_v[i]
            .at[:, :2]
            .add(0.055 * jax.random.normal(noise, (self.N, 2)) * (jnp.arange(self.N) % 2)[:, None])
        )
        q = self.start_q[i]
        if self.reset_angle:
            angles, joints = jax.random.split(jax.random.fold_in(noise, 1))
            jittered = (jnp.arange(self.N) % 4 != 0)[:, None]
            perturbation = self.reset_angle * jittered
            q = q.at[:, 7:].add(perturbation * jax.random.uniform(joints, (self.N, 29), minval=-1., maxval=1.))
            half = .5 * perturbation * jax.random.uniform(angles, (self.N, 2), minval=-1., maxval=1.)
            sr, sp = jnp.sin(half).T
            cr, cp = jnp.cos(half).T
            local = jnp.stack((cr*cp, sr*cp, cr*sp, -sr*sp), axis=-1)
            a, b = q[:, 3:7], local
            rotated = jnp.concatenate((a[:, :1]*b[:, :1] - (a[:, 1:]*b[:, 1:]).sum(axis=-1, keepdims=True),
                                       a[:, :1]*b[:, 1:] + b[:, :1]*a[:, 1:] + jnp.cross(a[:, 1:], b[:, 1:])), axis=-1)
            q = q.at[:, 3:7].set(rotated)
        state = (q, self.reference_velocity(q, v), self.is_up(q[:, 2], quat_to_mat(q[:, 3:7])))
        return (state, (self.start_clip[i], self.start_frame[i])) if return_source else state


def save_export(base, run, i, residual, disc, in_place=False):
    stem = f"{i:06d}"
    with np.load(base, allow_pickle=False) as z:
        fields = dict(z)
    for name, array in zip(("w0", "b0", "w1", "b1", "scale"), residual):
        fields[("in_place_residual/" if in_place else "local_rise/residual/") + name] = np.asarray(array)
    prefix = "in_place_residual/" if in_place else "local_rise/"
    library = {k.removeprefix(prefix): v for k, v in fields.items() if k.startswith(prefix)}
    discriminator = {f"net/{j}/{k}": np.asarray(v) for j, layer in enumerate(disc["net"]) for k, v in layer.items()}
    discriminator.update({"norm/" + k: np.asarray(v) for k, v in disc["norm"].items()})
    # Publish the player checkpoint last, after every companion file has closed successfully.
    with TemporaryDirectory(dir=run) as scratch:
        temporary = Path(scratch)
        np.savez_compressed(temporary / "library.npz", **library)
        np.savez(temporary / "discriminator.npz", **discriminator)
        np.savez(temporary / "ckpt.npz", **fields)
        for name in ("library", "discriminator", "ckpt"):
            (temporary / f"{name}.npz").replace(run / name / f"{stem}.npz")


def save_training_state(path, iteration, seconds, state):
    """Save every dynamic rollout, optimizer, discriminator and RNG leaf atomically."""
    arrays = {}
    for i, leaf in enumerate(jax.tree.leaves(state)):
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        arrays[str(i)] = np.asarray(leaf)
    with TemporaryDirectory(dir=path.parent) as directory:
        temp = Path(directory) / "training.npz"
        np.savez(temp, iteration=np.int64(iteration), seconds=np.float64(seconds), **arrays)
        temp.replace(path)


def load_training_state(path, template):
    leaves, structure = jax.tree.flatten(template)
    with np.load(path, allow_pickle=False) as data:
        if len(data.files) != len(leaves) + 2:
            raise ValueError("Training snapshot does not match the initialized simulator structure")
        restored = []
        for i, leaf in enumerate(leaves):
            array = data[str(i)]
            if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
                value = jax.random.wrap_key_data(jnp.asarray(array), impl=jax.random.key_impl(leaf))
            else:
                value = jnp.asarray(array)
            if value.shape != leaf.shape or value.dtype != leaf.dtype:
                raise ValueError(f"Training snapshot leaf {i} has incompatible shape or dtype")
            restored.append(value)
        return int(data["iteration"]), float(data["seconds"]), structure.unflatten(restored)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--motions", type=Path, default=Path("data/g1_getup/recovery"))
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--rollout", type=int, default=24)
    parser.add_argument("--std", type=float, default=0.065)
    parser.add_argument("--scale", type=float, default=0.45)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument(
        "--action-rate-multiplier",
        type=float,
        default=3.0,
        help="Multiplier of the original saturating action-rate penalty; preserves the zero-change reward",
    )
    parser.add_argument(
        "--acceleration-weight",
        type=float,
        default=0.0,
        help="Reward for matching source joint acceleration and avoiding high-frequency velocity changes",
    )
    parser.add_argument(
        "--velocity-supervision",
        type=float,
        default=0.0,
        help="Auxiliary loss toward a bounded corrective motor target from privileged reference velocities",
    )
    parser.add_argument("--in-place-weight",type=float,default=0.,
                        help="Train the late-rise feedback through the balance transition and reward the first standing location")
    parser.add_argument("--reset-angle", type=float, default=0., help="Maximum initial joint, local roll and pitch perturbation in degrees")
    parser.add_argument("--seed", type=int, default=631)
    parser.add_argument(
        "--warm-discriminator",
        type=Path,
        help="Initialize AMP weights and normalization from an exported discriminator",
    )
    parser.add_argument(
        "--resumable", action="store_true", help="Save a full atomic training snapshot at every export"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the existing run with its optimizer, simulator and exact RNG state",
    )
    args = parser.parse_args()
    if min(args.envs, args.iterations, args.rollout, args.save_every, args.lr, args.std, args.scale) <= 0:
        parser.error("Counts and scales must be positive")
    if args.envs * args.rollout % 4:
        parser.error("The rollout must split into four minibatches")
    if min(args.velocity_supervision, args.acceleration_weight, args.action_rate_multiplier, args.in_place_weight, args.reset_angle) < 0:
        parser.error("Velocity supervision weight must be nonnegative")
    if args.resume and not (args.out / "training.npz").is_file():
        parser.error("No training snapshot exists in the requested run directory")
    if args.resume:
        args.resumable = True
    for sub in ("ckpt", "library", "discriminator"):
        (args.out / sub).mkdir(parents=True, exist_ok=args.resume)
    config = {
        **vars(args),
        "robot": "g1_getup",
        "floor_dir": str(args.motions),
        "walk_dir": None,
        "nconmax": 64,
        "njmax": 640,
        "push": False,
        "prob_shove": 0.0,
        "reset_floor": 1.0,
        "learner": "on_policy_rise_residual_ppo_amp",
        "initial_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
        "assist_newtons": 0.0,
        "action_rate_base_coefficient": ACTION_RATE_BASE_COEFFICIENT,
        "action_rate_effective_coefficient": ACTION_RATE_BASE_COEFFICIENT * args.action_rate_multiplier,
    }
    config.pop("resume")
    config["warm_discriminator_sha256"] = (
        hashlib.sha256(args.warm_discriminator.read_bytes()).hexdigest() if args.warm_discriminator else None
    )
    config_file = args.out / "config.json"
    config_data = json.dumps(config, default=str, indent=2)
    if args.resume:
        saved_config = json.loads(config_file.read_text())
        saved_config.setdefault("in_place_weight",0.)
        saved_config.setdefault("reset_angle",0.)
        if saved_config != json.loads(config_data):
            raise ValueError("Resume parameters differ from the original training configuration")
    else:
        config_file.write_text(config_data)
    set_jax("~/.cache/instantamp/jax")
    prior, norm, _ = load(args.base)
    if not isinstance(prior, G1StandController) or prior.local_rise is None:
        raise ValueError("Provide a frozen exported controller with a local rise policy")
    in_place = args.in_place_weight > 0
    compute_residual = g1_inplace.residual if in_place else rise_residual
    if in_place and len(prior.actors) != 3:
        raise ValueError("In-place training needs the three-stage G1 stand controller")
    warm_actor = prior.in_place_residual if in_place else prior.local_rise.residual
    if warm_actor is not None:
        if not np.isclose(float(warm_actor[-1]), args.scale):
            raise ValueError("A resumed rise residual must use the source checkpoint scale")
        # Train the exported actor itself. The frozen base supplies only its original, uncorrected action.
        prior = (prior._replace(in_place_residual=None) if in_place else
                 prior._replace(local_rise=prior.local_rise._replace(residual=None)))
    sim = RiseSim(args.envs, args.motions, args.reset_angle)
    if in_place:
        sim.episode_steps = 500
    demos = amp.Demos(str(args.motions), sim, 10)
    key, init, who, init_disc, init_sim = jax.random.split(jax.random.key(args.seed), 5)
    state = sim.init(init_sim)
    layers = nets.mlp_init(init, [180 if in_place else 172, 64, 87])
    actor = (layers[0]["w"], layers[0]["b"], jnp.zeros_like(layers[1]["w"]), jnp.zeros(87))
    if warm_actor is not None:
        actor = warm_actor[:-1]
    params = {"pi": actor, "v": nets.mlp_init(who, [state.critic_obs.shape[-1], 128, 64, 1])}
    mask = jax.tree.map(lambda _: False, params)
    opt = optim.init(params, mask)
    disc = amp.init(init_disc, 10, hidden=(256, 128), frame_dim=sim.amp_dim)
    if args.warm_discriminator:
        with np.load(args.warm_discriminator, allow_pickle=False) as saved:
            layers = [{k: jnp.asarray(saved[f"net/{i}/{k}"]) for k in layer} for i, layer in enumerate(disc["net"])]
            normalizer = {k: jnp.asarray(saved["norm/" + k]) for k in disc["norm"]}
        if [x.shape for x in jax.tree.leaves(layers)] != [x.shape for x in jax.tree.leaves(disc["net"])]:
            raise ValueError("Discriminator architecture differs between training runs")
        disc = {**disc, "net": layers, "norm": normalizer}
    frame = sim.amp_frame(unflatten(state.data.qpos, state.data.qvel))
    live = state, policy_memory(prior, state.obs), jnp.repeat(frame[:, None], 10, axis=1)

    if in_place:
        live = (*live, g1_inplace.initial(args.envs))

    def value(params, x):
        return nets.mlp_apply(params["v"], x, jax.nn.elu)[:, 0]

    @jax.jit
    def collect(live, params, disc, key):
        def step(live, sample):
            s, memory, hist = live[:3]
            base, mem = policy_step(prior, norm, s.obs, memory)
            if in_place:
                _, active = g1_inplace.stance_inputs(s.obs)
                if prior.capture_gate is not None:
                    active *= 1. - mem[:, 3]
                if prior.in_place_demo is not None:
                    active *= prior.in_place_demo.residual_gate(s.obs)
            else:
                active = 1.0 - mem[:, 0]
            mu = compute_residual((*params["pi"], args.scale), s.obs)
            delta = mu + args.std * jax.random.normal(sample, mu.shape)
            action = base + active[:, None] * delta
            nxt, tr = sim.step(s, action)
            end = tr.terminated | tr.truncated
            old = unflatten(tr.qpos, tr.qvel)
            new = unflatten(tr.qpos1, tr.qvel1)
            world_points = sim.key_positions(new[0], new[1], new[4])
            frame = sim.amp_frame(new, world_points)
            window = jnp.concatenate((hist[:, 1:], frame[:, None]), axis=1)
            ff = sim.amp_frame(unflatten(nxt.data.qpos, nxt.data.qvel))
            hist = jnp.where(end[:, None, None], ff[:, None], window)
            valid = s.t >= 9
            style = env.DT * amp.recovery_style(disc, window, 20.0, leggedlab=True) * valid
            cursor = s.reference_start + s.t + 1
            lengths = sim.reference.lengths[s.reference_clip]
            index = sim.reference.offsets[s.reference_clip] + jnp.minimum(cursor, lengths - 1)
            v = new[5]
            ref = jnp.where((cursor < lengths)[:, None], sim.ref_velocity[index], 0.0)
            vel_error = jnp.mean(jnp.square(v - ref), axis=1)
            acceleration = (v - old[5]) / env.DT
            ref_accel = jnp.where((cursor < lengths)[:, None], sim.ref_acceleration[index], 0.0)
            accel_mse = jnp.mean((acceleration - ref_accel) ** 2, axis=-1)
            # The second finite difference of velocities emphasizes alternating-sample oscillations.
            prior_velocity = s.obs[:, 277:306]
            alternating = v - 2.0 * old[5] + prior_velocity
            hf_mse = jnp.mean(alternating**2, axis=-1)
            arm_limit = jnp.where(cursor < lengths, sim.arm_envelope[index], 0.0)
            arm_p95 = jnp.quantile(jnp.abs(v[:, 15:]), 0.95, axis=1)
            overspeed = jnp.maximum(0.0, arm_p95 - arm_limit)
            slew = jnp.mean(jnp.square(action - s.a_last), axis=1)
            rate_reward, rate_penalty = action_rate_reward(slew, args.action_rate_multiplier)
            up = sim.is_up(new[0][:, 2], new[1])
            old_points = sim.key_positions(old[0], old[1], old[4])
            feet_speed = jnp.sqrt(
                jnp.mean(jnp.square((world_points[:, :2] - old_points[:, :2]) / env.DT), axis=(1, 2))
            )
            qd_rms = jnp.sqrt(jnp.mean(v**2, axis=1))
            support = jnp.mean(tr.contact.astype(jnp.float32), axis=1)
            quiet = up * support * jnp.exp(-6 * feet_speed - 0.50 * qd_rms - 0.5 * jnp.linalg.norm(new[3], axis=1))
            pose = tr.parts[:, sim.terms.index("motion_reference")]
            reward = (
                style
                + pose
                + env.DT
                * (8.0 / (1.0 + vel_error / 4.0) + 5.0 / (1.0 + overspeed**2) + rate_reward + 8.0 * quiet + 2.0 * up)
            )
            reward += args.acceleration_weight * env.DT * (8.0 / (1.0 + accel_mse / 2500.0) + 5.0 / (1.0 + hf_mse))
            next_live = (nxt, reset_policy_memory(mem, end), hist)
            if in_place:
                anchor, stationary_reward = g1_inplace.score(live[3], new, world_points[:,:2], tr.contact, end)
                reward += args.in_place_weight * stationary_reward
                next_live = (*next_live, anchor)
            reward = jnp.where(tr.diverged, -1.0, reward)
            # Match measured post-step rates to the reference. The target is in action units and
            # uses a smaller correction on the load-bearing legs. Only PPO training sees the reference.
            velocity_gain = jnp.concatenate((jnp.full(15, 0.012), jnp.full(14, 0.080)))
            slew_gain = jnp.concatenate((jnp.full(15, 0.008), jnp.full(14, 0.060)))
            target = jnp.clip(velocity_gain * (ref - v) + slew_gain * (s.a_last - base), -0.65, 0.65)
            target = jnp.clip(
                target
                - jnp.minimum(args.acceleration_weight, 1.0)
                * velocity_gain
                * jnp.clip((acceleration - ref_accel) * env.DT, -6.0, 6.0),
                -0.65,
                0.65,
            )
            rows = (
                s.obs,
                s.critic_obs,
                mu,
                delta,
                value(params, s.critic_obs),
                reward,
                end,
                tr.truncated,
                active,
                target,
            )
            metrics = jnp.array(
                [
                    style.mean(),
                    pose.mean(),
                    vel_error.mean(),
                    overspeed.mean(),
                    slew.mean(),
                    quiet.mean(),
                    up.mean(),
                    feet_speed.mean(),
                    qd_rms.mean(),
                    active.mean(),
                    jnp.abs(active[:, None] * mu).mean(),
                    jnp.sqrt(jnp.mean(acceleration**2)),
                    jnp.sqrt(hf_mse.mean()),
                    rate_penalty.mean(),
                    jnp.asarray(ACTION_RATE_BASE_COEFFICIENT),
                    jnp.asarray(args.action_rate_multiplier),
                    jnp.asarray(ACTION_RATE_BASE_COEFFICIENT * args.action_rate_multiplier),
                ]
            )
            return next_live, (rows, window, valid, metrics)

        live, (rows, robot, valid, metrics) = jax.lax.scan(step, live, jax.random.split(key, args.rollout))
        x, c, old, action, val, r, end, to, active, target = rows
        adv, ret = gae(r, val, value(params, live[0].critic_obs), end, to, 0.99, 0.95)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        def flat(samples):
            return samples.reshape(-1, *samples.shape[2:])

        return (
            live,
            tuple(map(flat, (x, c, old, action, adv, ret, active, target))),
            flat(robot),
            flat(valid),
            metrics.mean(0),
        )

    @jax.jit
    def learn(params, opt, disc, batch, robot, valid, key):
        def update(carry, ix_key):
            p, o, d = carry
            ix, ks = ix_key
            x, c, old, action, adv, ret, active, target = (item[ix] for item in batch)

            def objective(p):
                mu = compute_residual((*p["pi"], args.scale), x)
                log_ratio = jnp.sum(((action - old) ** 2 - (action - mu) ** 2) / (2 * args.std**2), axis=1)
                ratio = jnp.exp(jnp.clip(log_ratio, -12.0, 12.0))
                w = (active > 0.0).astype(jnp.float32)
                pg = -(w * jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv)).sum() / jnp.maximum(
                    1.0, w.sum()
                )
                anchor = (w * jnp.mean(mu**2, axis=1)).sum() / jnp.maximum(1.0, w.sum()) / (2 * args.std**2)
                v_loss = ((value(p, c) - ret) ** 2).mean()
                guided = (
                    (w * jnp.mean((mu - target) ** 2, axis=1)).sum() / jnp.maximum(1.0, w.sum()) / (2 * args.std**2)
                )
                return (
                    pg + 0.35 * v_loss + 0.05 * anchor + args.velocity_supervision * guided,
                    jnp.array([pg, v_loss, anchor, jnp.abs(log_ratio).mean(), guided]),
                )

            (_, stats), grad = jax.value_and_grad(objective, has_aux=True)(p)
            grad, _ = optim.clip_by_global_norm(grad, 1.0)
            p, o = optim.step(p, grad, o, mask, args.lr)
            k1, k2, k3 = jax.random.split(ks, 3)
            db = min(128, len(ix))
            ids = jax.random.randint(k1, (db,), 0, len(robot))
            d, ds = amp.train_step(
                d,
                demos.sample(k2, db),
                robot[ids],
                valid[ids].astype(jnp.float32),
                k3,
                0.0005,
                recovery=True,
                leggedlab=True,
            )
            return (p, o, d), jnp.r_[stats, ds["d_demo"], ds["d_robot"]]

        def epoch(carry, ks):
            k1, k2 = jax.random.split(ks)
            chunks = jax.random.permutation(k1, len(batch[0])).reshape(4, -1)
            return jax.lax.scan(update, carry, (chunks, jax.random.split(k2, 4)))

        out, metrics = jax.lax.scan(epoch, (params, opt, disc), jax.random.split(key, 2))
        return *out, metrics.mean((0, 1))

    labels = (
        "amp_reward",
        "reference_reward",
        "joint_velocity_mse",
        "arm_p95_excess",
        "action_slew_sq",
        "quiet_supported",
        "up",
        "foot_speed",
        "joint_rms",
        "rise_share",
        "residual_abs",
        "measured_acceleration_rms",
        "velocity_highpass_rms",
        "action_rate_penalty",
        "action_rate_base_coefficient",
        "action_rate_multiplier",
        "action_rate_effective_coefficient",
        "ppo_policy_loss",
        "value_loss",
        "anchor",
        "abs_log_ratio",
        "velocity_supervision_loss",
        "discriminator_expert",
        "discriminator_robot",
    )
    trained = 0
    elapsed = 0.0
    log = args.out / "log.jsonl"
    if args.resume:
        trained, elapsed, (live, params, opt, disc, key) = load_training_state(
            args.out / "training.npz", (live, params, opt, disc, key)
        )
        if log.exists():
            # Drop any tail written between the last durable snapshot and interruption.
            retained = []
            for line in log.read_text().splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    break
                if item["iteration"] <= trained:
                    retained.append(line)
            with TemporaryDirectory(dir=args.out) as temp:
                cleaned = Path(temp) / "log.jsonl"
                cleaned.write_text("".join(x + "\n" for x in retained))
                cleaned.replace(log)
    elif args.resumable:
        save_export(args.base, args.out, 0, (*params["pi"], args.scale), disc, in_place)
        save_training_state(args.out / "training.npz", 0, 0.0, (live, params, opt, disc, key))
    start = time.monotonic() - elapsed
    with log.open("a" if args.resume else "x", buffering=1) as stream:
        for i in range(trained + 1, args.iterations + 1):
            key, k1, k2 = jax.random.split(key, 3)
            live, batch, robot, valid, ro = collect(live, params, disc, k1)
            params, opt, disc, lo = learn(params, opt, disc, batch, robot, valid, k2)
            if i == 1 or i % 5 == 0 or i % args.save_every == 0 or i == args.iterations:
                metrics = np.asarray(jnp.r_[ro, lo])
                if not np.isfinite(metrics).all():
                    raise FloatingPointError(metrics)
                row = {
                    "iteration": i,
                    "env_steps": i * args.rollout * args.envs,
                    "seconds": round(time.monotonic() - start, 2),
                    **dict(zip(labels, map(float, metrics))),
                }
                stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if i % args.save_every == 0 or i == args.iterations:
                save_export(args.base, args.out, i, (*params["pi"], args.scale), disc, in_place)
                if args.resumable:
                    save_training_state(
                        args.out / "training.npz", i, time.monotonic() - start, (live, params, opt, disc, key)
                    )


if __name__ == "__main__":
    main()
