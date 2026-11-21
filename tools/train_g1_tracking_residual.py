"""Train the 29-joint G1 landing residual on full native MuJoCo rollouts using PPO and AMP."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from instantamp import amp, env, g1_getup, nets, optim
from instantamp.evaluate import fallen_starts
from instantamp.inference import load, policy_memory, policy_step
from instantamp.native_getup import advance, observation_terms
from instantamp.sim import build_scene


def reference_windows(directory, duration_scale=1.0):
    demos = []
    for f in sorted(directory.glob("*.npz")):
        with np.load(f) as d:
            rotation = Rotation.from_quat(d["root_quat"]).as_matrix()
            q, v = d["dof_pos"], d["dof_vel"]
            pos, points, omega = d["root_pos"], d["key_body_pos"], d["root_ang_vel"]
            if duration_scale != 1.0:
                source = np.arange(len(q))
                phase = np.linspace(
                    0.0, source[-1], round(source[-1] * duration_scale) + 1
                )

                def interp(x):
                    flat = x.reshape(len(x), -1)
                    return np.column_stack(
                        [np.interp(phase, source, col) for col in flat.T]
                    ).reshape((len(phase), *x.shape[1:]))

                rotation = Slerp(source, Rotation.from_matrix(rotation))(
                    phase
                ).as_matrix()
                q, v, pos, points, omega = (
                    interp(q),
                    interp(v) / duration_scale,
                    interp(pos),
                    interp(points),
                    interp(omega) / duration_scale,
                )
            yaw = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
            local = Rotation.from_euler("z", -yaw[:, None]).as_matrix() @ rotation
            gyro = np.einsum("nij,ni->nj", rotation, omega)
            points = ((points - pos[:, None]) @ rotation).reshape(len(yaw), 18)
            frame = np.c_[local[:, :, 0], local[:, :, 2], gyro, q, v, points].astype(
                "f"
            )
            demos.append(frame[np.arange(10) + np.arange(len(yaw) - 9)[:, None]])
    return demos


def onboard(d, links, previous):
    gyro, axes, q, v, action, feet = observation_terms(d, links, previous)
    return np.concatenate((gyro, axes, q, v, action, feet)), np.concatenate(
        (axes, gyro, q, v, feet)
    )


class RiseReference:
    """Training-only joint reference, sampled near the current physical phase."""

    def __init__(self, file):
        with np.load(file) as d:
            self.q = d["dof_pos"]
            self.v = d["dof_vel"]
            self.a = np.diff(self.v, axis=0, prepend=self.v[:1]) / env.DT

    def score(self, frame, q, v, before, duration_scale=1.0):
        # A small window accounts for contact timing without allowing an arbitrary phase jump.
        ids = np.arange(
            max(0, min(frame - 6, len(self.q) - 1)), min(frame + 7, len(self.q))
        )
        index = ids[
            np.argmin(
                np.mean((self.q[ids] - q) ** 2, axis=1) + 0.00025 * (ids - frame) ** 2
            )
        ]
        pose = np.mean((self.q[index] - q) ** 2)
        speed = np.mean((v[:12] - self.v[index, :12] / duration_scale) ** 2)
        accel = np.mean(
            ((v[:12] - before[:12]) / env.DT - self.a[index, :12] / duration_scale**2)
            ** 2
        )
        return (
            2.0 / (1.0 + pose / 0.14)
            + 2.0 / (1.0 + speed / 4.0)
            + 1.0 / (1.0 + accel / 1600.0)
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--discriminator", type=Path, required=True)
    p.add_argument("--motions", type=Path, default=Path("data/g1_getup/recovery"))
    p.add_argument("--envs", type=int, default=6)
    p.add_argument("--steps", type=int, default=275)
    p.add_argument("--iterations", type=int, default=32)
    p.add_argument("--seed", type=int, default=77231)
    p.add_argument("--scale", type=float, default=0.06)
    p.add_argument("--std", type=float, default=0.27)
    p.add_argument("--lr", type=float, default=0.00022)
    p.add_argument(
        "--reference-motion-weight",
        type=float,
        default=0.0,
        help="Weight of source joint pose, leg speed and acceleration matching before first stance",
    )
    p.add_argument(
        "--rise-duration-scale",
        type=float,
        default=1.0,
        help="Stretch reference motion in physical time for AMP and tracking",
    )
    p.add_argument(
        "--ground-support-weight",
        type=float,
        default=0.0,
        help="Reward sustained hand loading during the ground movement and penalize early extension",
    )
    args = p.parse_args()
    if (
        min(args.envs, args.steps, args.iterations, args.scale, args.std, args.lr) <= 0
        or args.steps < 10
        or args.reference_motion_weight < 0
        or args.ground_support_weight < 0
        or args.rise_duration_scale <= 0
    ):
        p.error("Invalid rollout length or optimizer configuration")
    from instantamp.g1_tracking_capture import residual_features

    # The baseline already contains the trained AMP rise and stance actors and tracking feedback.
    prior, norm, _ = load(args.base)
    head = prior.tracking_capture
    if head is None:
        p.error("Expected tracking_capture parameters in the baseline")
    with np.load(args.base) as z:
        fields = dict(z)
    for name in ("residual_w", "residual_b"):
        if "floating_base/sensor" not in fields and np.any(
            fields.get("tracking_capture/" + name, 0.0)
        ):
            p.error("Start the correction from an unmodified tracking head")
    dest = args.out
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "base_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
                "action_rate_multiplier": 3.0,
                "assist_n": 0.0,
                "learner": "native_ppo_amp",
            },
            default=str,
            indent=2,
        )
    )
    starts = {}
    for protocol in ("source-play", "fallen"):
        names, q, v = fallen_starts(args.motions, g1_getup.RESET_LIFT, protocol)
        starts.update(
            {name: (np.array(x), np.array(y)) for name, x, y in zip(names, q, v)}
        )
    references = (
        {name: RiseReference(args.motions / name) for name, offset in starts}
        if args.reference_motion_weight or args.ground_support_weight
        else {}
    )
    starts = [
        (q, v, references.get(name), round(offset / env.DT))
        for (name, offset), (q, v) in starts.items()
    ]
    demo = reference_windows(args.motions, args.rise_duration_scale)
    model = build_scene(robot_name="g1_getup", getup_visuals=True)
    from instantamp.g1_ground_phase import HandContacts

    hands = HandContacts(model)
    if args.ground_support_weight:
        from tools.score_g1_tracking import load_native_reference

        for name, ref in references.items():
            poses, _ = load_native_reference(args.motions / name)
            ref.root_z = poses[:, 2]
            ref.ground = hands.reference_clearances(poses)[:, :, 0] < 0.065
    data = [mujoco.MjData(model) for _ in range(args.envs)]
    measured = None
    if "floating_base/sensor" in fields:
        from instantamp.g1_floating_base import load_native_tracker

        measured = [load_native_tracker(model, prior, args.base) for _ in data]
        prior = prior._replace(tracking_capture=None)
    ids = [model.body("robot_" + n).id for n in g1_getup.KEY_BODIES]
    rng = np.random.default_rng(args.seed)
    pars = (head.residual_w, head.residual_b, jnp.zeros(97))
    if any(np.any(x) for x in pars[:2]) and not np.isclose(
        float(head.residual_scale), args.scale
    ):
        p.error("A warm-started motor residual must keep its physical action scale")
    trained_action = jax.jit(
        lambda h, w, b: args.scale * jnp.tanh(residual_features(h) @ w + b)
    )
    mask = jax.tree.map(lambda _: False, pars)
    opt = optim.init(pars, mask)
    with np.load(args.discriminator) as d:
        weights = []
        while f"net/{len(weights)}/w" in d:
            i = len(weights)
            weights.append({k: jnp.asarray(d[f"net/{i}/{k}"]) for k in ("w", "b")})
        disc = amp.init(
            jax.random.key(args.seed),
            10,
            hidden=tuple(x["w"].shape[1] for x in weights[:-1]),
            frame_dim=85,
        )
        disc = {
            **disc,
            "net": weights,
            "norm": {n[5:]: jnp.asarray(d[n]) for n in d if n.startswith("norm/")},
        }
    head_size = head.memory(jnp.zeros((1, 570))).shape[-1]

    @jax.jit
    def infer(hist, frame, memory):
        hist = g1_getup.push_history(hist, frame, g1_getup.ACTOR_TERMS)
        act, mem = policy_step(prior, norm, hist, memory)
        return (
            hist,
            mem,
            act,
            residual_features(hist),
            (
                mem[:, -head_size] != 0
                if measured is None
                else jnp.zeros(len(hist), bool)
            ),
        )

    @jax.jit
    def style(d, windows):
        return amp.recovery_style(d, windows, 1.0, leggedlab=True)

    disc_update = jax.jit(
        lambda d, real, fake, key: amp.train_step(
            d,
            real,
            fake,
            jnp.ones(len(real)),
            key,
            0.00010,
            recovery=True,
            leggedlab=True,
        )
    )

    @jax.jit
    def update(params, optimizer, x, raw, oldlogp, target, advantage, active):
        def loss(theta):
            w, b, v = theta
            mu = x @ w + b
            logp = -0.5 * jnp.square((raw - mu) / args.std).sum(-1)
            r = jnp.exp(jnp.clip(logp - oldlogp, -5.0, 5.0))
            pg = -jnp.sum(
                active * jnp.minimum(r * advantage, jnp.clip(r, 0.8, 1.2) * advantage)
            ) / jnp.maximum(active.sum(), 1.0)
            critic = 0.5 * jnp.square(x @ v[:-1] + v[-1] - target).mean()
            return pg + 0.22 * critic + 0.006 * jnp.mean(jnp.square(mu))

        return optim.step(params, jax.grad(loss)(params), optimizer, mask, args.lr)

    for it in range(1, args.iterations + 1):
        previous = np.zeros((args.envs, 29), np.float32)
        if measured:
            for controller in measured:
                controller.reset()
                if controller.planning:
                    controller.predict_residual = lambda h: trained_action(h, *pars[:2])
        palm_streak = np.zeros((args.envs, 2), int)
        ref_starts = []
        for d in data:
            q, v, ref, offset = starts[rng.integers(len(starts))]
            ref_starts.append((ref, offset))
            mujoco.mj_resetData(model, d)
            d.qpos[:], d.qvel[:] = q.copy(), v.copy()
            d.qpos[7:] += rng.uniform(-0.04, 0.04, 29)
            r = Rotation.from_quat(q[3:7], scalar_first=True) * Rotation.from_euler(
                "xy", rng.uniform(-0.04, 0.04, 2)
            )
            d.qpos[3:7] = r.as_quat(scalar_first=True)
            d.qvel[:2] += rng.uniform(-0.075, 0.075, 2)
            mujoco.mj_kinematics(model, d)
            d.qvel[:3] -= d.xmat[1].reshape(3, 3) @ np.cross(
                d.qvel[3:6], model.body_ipos[1]
            )
            mujoco.mj_forward(model, d)
        first = np.array(
            [onboard(d, ids, u)[0] for d, u in zip(data, previous)], dtype="f"
        )
        hist = g1_getup.push_history(None, jnp.asarray(first), g1_getup.ACTOR_TERMS)
        memory = policy_memory(prior, hist)
        count = np.zeros(args.envs, int)
        rose = np.zeros(args.envs, bool)
        fell = rose.copy()
        anchor = np.zeros((args.envs, 2))
        sole = np.zeros((args.envs, 2, 2))
        peak = np.zeros(args.envs)
        samples = []
        motion = []
        w, b, value = map(np.asarray, pars)
        for t in range(args.steps):
            observed = [onboard(d, ids, u) for d, u in zip(data, previous)]
            frame = np.asarray([f for f, _ in observed], dtype="f")
            motion.append([f for _, f in observed])
            hist, memory, base, x, active = infer(hist, frame, memory)
            x, active = np.asarray(x), np.asarray(active)
            if measured:
                actors = np.asarray(base)
                base = np.stack(
                    [
                        controller.action(d, u, prior_action)
                        for controller, d, u, prior_action in zip(measured, data, actors, previous)
                    ]
                )
                active = np.array([controller.engaged for controller in measured])
            noise = rng.normal(size=(len(x), 29)).astype("f")
            raw = x @ w + b + args.std * noise
            u = (np.asarray(base) + active[:, None] * args.scale * np.tanh(raw)).astype(
                "f"
            )
            if measured:
                for i, controller in enumerate(measured):
                    u[i] = controller.balance_action(u[i], actors[i])
            old_z = np.array([d.qpos[2] for d in data])
            for d, a in zip(data, u):
                advance(model, d, a)
            pos = np.array([d.qpos[:3] for d in data])
            feet = np.array([d.xpos[ids[:2], :2] for d in data])
            cos = np.array([d.xmat[1, 8] for d in data])
            xy_speed = np.array([np.linalg.norm(d.qvel[:2]) for d in data])
            up = (pos[:, 2] > 0.65) & (pos[:, 2] <= 0.85) & (cos > 0.8660254)
            begin = (count == 0) & up & ~rose
            anchor[begin], sole[begin], peak[begin] = pos[begin, :2], feet[begin], 0.0
            count = np.where(up, count + 1, 0)
            rose |= count >= 10
            fell |= rose & ((pos[:, 2] < 0.50) | (cos < 0.5))
            drift = np.linalg.norm(pos[:, :2] - anchor, axis=-1)
            peak = np.where(rose | (count > 0), np.maximum(peak, drift), 0.0)
            slip = np.linalg.norm(feet - sole, axis=-1).max(-1)
            slew = np.square(u - previous).mean(-1)
            # reward per 20ms, using the same tripled action-rate term as the rise trainer.
            penalty = -6.0 * slew / (1.0 + slew)
            reward = 1.8 * (pos[:, 2] - old_z) + env.DT * (
                2.0 * up
                + 8.0 * rose * up / (1.0 + (peak / 0.13) ** 2 + (slip / 0.13) ** 2)
                + 3.0 * up / (1.0 + (xy_speed / 0.16) ** 2)
                - 6.0 * fell
                + penalty
            )
            if args.reference_motion_weight:
                for i, (ref, offset) in enumerate(ref_starts):
                    if not rose[i]:
                        reward[i] += (
                            env.DT
                            * args.reference_motion_weight
                            * ref.score(
                                round((t + 1) / args.rise_duration_scale) + offset,
                                data[i].qpos[7:],
                                data[i].qvel[6:],
                                frame[i, 38:67],
                                args.rise_duration_scale,
                            )
                        )
            if args.ground_support_weight:
                for i, (ref, offset) in enumerate(ref_starts):
                    k = min(
                        len(ref.q) - 1,
                        offset + round((t + 1) / args.rise_duration_scale),
                    )
                    expected = ref.ground[k]
                    palm_load = hands.forces(data[i])[:, 0]
                    palm_streak[i] = np.where(palm_load > 0.07, palm_streak[i] + 1, 0)
                    if expected.any() and ref.root_z[k] < 0.60:
                        # Only reward the hand indicated by mocap. Saturate after a sustained load.
                        bearing = np.clip(
                            palm_streak[i, expected].max() / 8.0, 0.0, 1.0
                        )
                        high = max(0.0, data[i].qpos[2] - ref.root_z[k] - 0.045)
                        knees = data[i].qpos[[10, 16]].mean()
                        early_extension = max(
                            0.0, ref.q[k, [3, 9]].mean() - knees - 0.15
                        )
                        reward[i] += (
                            env.DT
                            * args.ground_support_weight
                            * (bearing - 4.0 * high - early_extension)
                        )
            samples.append(
                (
                    x,
                    raw,
                    -0.5 * (noise**2).sum(-1),
                    reward,
                    active,
                    x @ value[:-1] + value[-1],
                )
            )
            previous = u
        frames = np.asarray(motion, dtype="f")
        indices = np.maximum(np.arange(args.steps)[:, None] - np.arange(9, -1, -1), 0)
        windows = frames[indices].transpose(0, 2, 1, 3).reshape(-1, 10, 85)
        style_reward = np.asarray(style(disc, windows)).reshape(args.steps, args.envs)
        x, raw, logp, reward, active, values = map(np.asarray, zip(*samples))
        reward += env.DT * 3.0 * style_reward
        adv = np.zeros_like(values)
        carry = np.zeros(args.envs)
        for t in reversed(range(args.steps)):
            bootstrap = values[t + 1] if t + 1 < args.steps else 0.0
            carry = reward[t] + 0.995 * bootstrap - values[t] + 0.995 * 0.95 * carry
            adv[t] = carry
        target = adv + values
        good = adv[active]
        if not len(good):
            raise ValueError(
                "The tracking head never issued an action in the current rollout"
            )
        adv = (adv - good.mean()) / (good.std() + 1.0e-5)
        flat = lambda a: a.reshape((-1,) + a.shape[2:])
        arrays = list(map(flat, (x, raw, logp, target, adv, active)))
        for _ in range(3):
            order = rng.permutation(len(windows))
            for indices in np.array_split(order, 4):
                pars, opt = update(pars, opt, *(a[indices] for a in arrays))
        idx = rng.integers(len(windows), size=192)
        real = np.stack(
            [
                demo[n][rng.integers(len(demo[n]))]
                for n in rng.integers(len(demo), size=192)
            ]
        )
        disc, metrics = disc_update(
            disc, real, windows[idx], jax.random.key(args.seed + it)
        )
        print(
            json.dumps(
                dict(
                    iteration=it,
                    transitions=it * args.steps * args.envs,
                    active=float(active.mean()),
                    stand=int(rose.sum()),
                    refall=int(fell.sum()),
                    mean_peak_cm=float(100 * peak.mean()),
                    style=float(style_reward.mean()),
                    d_demo=float(metrics["d_demo"]),
                    d_robot=float(metrics["d_robot"]),
                    rms_action_correction=float(
                        args.scale
                        * np.sqrt(
                            np.mean(
                                np.tanh(x @ np.array(pars[0]) + np.array(pars[1])) ** 2
                            )
                        )
                    ),
                )
            ),
            flush=True,
        )
        if it == 1 or it % 5 == 0 or it == args.iterations:
            path = dest / f"{it:06}"
            (path / "ckpt").mkdir(parents=True)
            fields.update(
                {
                    "tracking_capture/residual_w": np.array(pars[0]),
                    "tracking_capture/residual_b": np.array(pars[1]),
                    "tracking_capture/residual_scale": np.float32(args.scale),
                }
            )
            np.savez(path / "ckpt/0000.npz", **fields)
            (path / "config.json").write_text(
                json.dumps(
                    dict(
                        robot="g1_getup",
                        floor_dir=str(args.motions),
                        assist_newtons=0.0,
                        learner="native_ppo_amp",
                    ),
                    indent=2,
                )
            )
            np.savez(
                path / "ppo_residual.npz",
                weight=pars[0],
                bias=pars[1],
                value=pars[2],
                discriminator_mean=disc["norm"]["mean"],
                discriminator_variance=disc["norm"]["var"],
                **{
                    f"disc_{i}_{k}": v
                    for i, layer in enumerate(disc["net"])
                    for k, v in layer.items()
                },
            )


if __name__ == "__main__":
    main()
