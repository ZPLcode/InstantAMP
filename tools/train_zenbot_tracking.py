"""Fit a Zenbot state-dependent motor residual using delayed native MuJoCo rollouts and PPO+AMP."""

import argparse
import ctypes as ct
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation
from instantamp import amp, optim, zenbot as z
from instantamp.sim import build_scene
from instantamp.zenbot_native_tracking import backend, ptr


def references(directory):
    rows = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path) as d:
            r = Rotation.from_quat(d["root_quat"]).as_matrix()
            yaw = np.arctan2(r[:, 1, 0], r[:, 0, 0])
            a = Rotation.from_euler("z", -yaw[:, None]).as_matrix() @ r
            gyro = np.einsum("nij,ni->nj", r, d["root_ang_vel"])
            pts = ((d["key_body_pos"] - d["root_pos"][:, None]) @ r).reshape(len(r), 18)
            f = np.c_[
                a[:, :, 0], a[:, :, 2], gyro, d["dof_pos"], d["dof_vel"], pts
            ].astype("f")
            rows.append(f[np.arange(10) + np.arange(len(f) - 9)[:, None]])
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--reference", type=Path, default=Path("data/zenbot23_getup/recovery")
    )
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--envs", type=int, default=48)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--seed", type=int, default=93119)
    p.add_argument("--scale", type=float, default=None)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with np.load(args.base) as d:
        fields = dict(d)
    key = "zenbot_tracking/"
    off = fields[key + "offset"]
    q = fields[key + "q"]
    v = fields[key + "v"]
    goals = np.ascontiguousarray(z.DEFAULT_Q + 0.25 * fields[key + "u"], np.float64)
    gains = np.ascontiguousarray(fields[key + "K"], np.float32)
    model = build_scene(robot_name="zenbot23_getup")
    mot = np.ascontiguousarray([z.KP, z.KD, z.EFFORT, z.SPEED], np.float64)
    neutral = z.DEFAULT_Q.astype("d")
    links = np.array([model.body("robot_" + n).id for n in z.KEY_BODIES], "i")
    rng = np.random.default_rng(args.seed)
    net = jnp.asarray(fields[key + "actor"])
    value = jnp.zeros(76)
    params = (net, value)
    mask = (False, False)
    opt = optim.init(params, mask)
    n, h = args.envs, args.horizon
    sigma = 0.32
    scale = float(fields[key + "scale"])
    obs = np.empty((n, h, 76), "f")
    latent = np.empty((n, h, 23), "f")
    reward = np.empty((n, h), "f")
    motion = np.empty((n, h, 73), "f")
    window = np.maximum(np.arange(h)[:, None] - np.arange(9, -1, -1), 0)
    refs = references(args.reference)
    disc = amp.init(jax.random.key(args.seed), 10, hidden=(128, 64), frame_dim=73)
    from instantamp import nets

    disc["norm"] = nets.rn_update(
        disc["norm"], jnp.asarray(np.concatenate(refs)[::3].reshape(-1, 73))
    )
    disc_update = jax.jit(
        lambda d, r, f, k: amp.train_step(
            d, r, f, jnp.ones(len(f)), k, 0.00027, recovery=True, leggedlab=True
        )
    )
    style = jax.jit(lambda d, f: amp.recovery_style(d, f, 1.0, leggedlab=True))

    @jax.jit
    def update(par, o, x, a, log0, adv, ret):
        def loss(para):
            w, c = para
            mu = x @ w
            ratio = jnp.exp(
                jnp.clip(-0.5 * jnp.square((a - mu) / sigma).sum(-1) - log0, -6, 6)
            )
            return (
                -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv))
                + 0.5 * jnp.square(x @ c - ret).mean()
                + 0.02 * jnp.square(mu).mean()
            )

        return optim.step(par, jax.grad(loss)(par), o, mask, 0.00036)

    support = np.ascontiguousarray(
        fields.get(key + "desired_palm_load", np.zeros((len(q), 2))), dtype="d"
    )
    if args.scale is not None:
        scale = args.scale
        fields[key + "scale"] = np.float32(scale)
    func = backend().training
    func.restype = None
    (args.out / "config.json").write_text(
        json.dumps(
            dict(
                robot="zenbot23_getup",
                algorithm="native_PPO_AMP",
                assist_n=0.0,
                action_rate_l2=-0.015,
                **vars(args),
            ),
            default=str,
            indent=2,
        )
    )
    for it in range(1, args.iterations + 1):
        row = rng.integers(len(off) - 1, size=n)
        start = np.array(
            [rng.integers(off[r] + 2, off[r + 1] - h - 1) for r in row], "i"
        )
        lags = rng.integers(3, 8, (n, 5))
        nom = fields[key + "delay"][row]
        lags[::3] = nom[::3]
        lags = np.ascontiguousarray(lags[:, z.GROUP], dtype="i")
        weights = np.ascontiguousarray(params[0], dtype="f")
        noise = np.ascontiguousarray(rng.normal(size=(n, h, 23)), dtype="f")
        func(
            ct.c_void_p(model._address),
            *map(ct.c_int, (n, h, 2)),
            *[
                ptr(a)
                for a in (mot, neutral, start, lags, q, v, goals, gains, links, support)
            ],
            ct.c_double(model.body_mass.sum() * 9.81),
            ptr(weights),
            ptr(noise),
            ct.c_float(sigma),
            ct.c_float(scale),
            ct.c_int(args.seed + it),
            *[ptr(a) for a in (obs, latent, reward, motion)],
        )
        windows = motion[:, window].reshape(-1, 10, 73)
        idx = rng.integers(len(windows), size=192)
        real = np.stack(
            [
                refs[k][rng.integers(len(refs[k]))]
                for k in rng.integers(len(refs), size=192)
            ]
        )
        disc, stats = disc_update(
            disc, real, windows[idx], jax.random.key(args.seed + it)
        )
        amp_value = np.asarray(style(disc, windows)).reshape(n, h)
        total = reward + 0.02 * 2.7 * amp_value
        val = obs @ np.asarray(params[1])
        adv = np.zeros_like(val)
        carry = np.zeros(n)
        for t in reversed(range(h)):
            next_value = val[:, t + 1] if t + 1 < h else 0.0
            carry = total[:, t] + 0.99 * next_value - val[:, t] + 0.99 * 0.93 * carry
            adv[:, t] = carry
        target = (adv + val).ravel()
        adv = ((adv - adv.mean()) / (adv.std() + 1e-5)).ravel()
        x = obs.reshape(-1, 76)
        a = latent.reshape(-1, 23)
        log0 = -0.5 * (noise.reshape(-1, 23) ** 2).sum(-1)
        for _ in range(2):
            for idx in np.array_split(rng.permutation(len(x)), 4):
                params, opt = update(
                    params, opt, x[idx], a[idx], log0[idx], adv[idx], target[idx]
                )
        if it == 1 or it % 10 == 0:
            print(
                json.dumps(
                    dict(
                        iteration=it,
                        steps=it * n * h,
                        amp=float(amp_value.mean()),
                        reward=float(reward.mean()),
                        demo=float(stats["d_demo"]),
                        robot=float(stats["d_robot"]),
                        correction_rms=float(
                            scale
                            * np.sqrt(
                                np.square(np.tanh(x @ np.array(params[0]))).mean()
                            )
                        ),
                    )
                ),
                flush=True,
            )
        if it in (1, 10, 50, 100, 200, args.iterations):
            dest = args.out / f"{it:06}" / "ckpt"
            dest.mkdir(parents=True)
            fields[key + "actor"] = np.asarray(params[0])
            np.savez_compressed(dest / "0000.npz", **fields)
            np.savez_compressed(
                dest.parent / "training.npz",
                actor=params[0],
                critic=params[1],
                norm=disc["norm"]["mean"],
                **{
                    f"{i}_{k}": x
                    for i, layer in enumerate(disc["net"])
                    for k, x in layer.items()
                },
            )


if __name__ == "__main__":
    main()
