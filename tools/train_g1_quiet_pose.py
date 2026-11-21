"""Train the standing posture residual with physical native MuJoCo episodes and LeggedLab AMP.

The low-level standing controller stabilizes the feet and center of mass. PPO chooses a
bounded arm/trunk pose when double support begins. Its 13 inputs are joint targets observed
at that instant. Training uses 10-frame real AMP windows from the upright portion of each
recovery, native policy windows, and the repository's LeggedLab discriminator and style reward.
"""

import argparse
import multiprocessing as mp
from pathlib import Path
import tempfile

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from instantamp import g1_getup as robot
from instantamp.g1_sensor_stand import _StandingDynamics, pose_features
from instantamp.native_getup import advance, observation_terms
from instantamp.sim import build_scene


def reference_data(directory):
    windows, poses = [], []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as f:
            if float(f["fps"]) != 50:
                raise ValueError(f"Expected a 50 Hz recovery recording in {path}")
            pos, q = f["root_pos"], f["dof_pos"]
            rotation = Rotation.from_quat(f["root_quat"]).as_matrix()
            yaw = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
            local = Rotation.from_euler("z", -yaw[:, None]).as_matrix() @ rotation
            omega = np.einsum("nij,ni->nj", rotation, f["root_ang_vel"])
            points = ((f["key_body_pos"] - pos[:, None, :]) @ rotation).reshape(len(pos), 18)
            frames = np.c_[local[:, :, 0], local[:, :, 2], omega, q, f["dof_vel"], points]
            upright = (pos[:, 2] > .64) & (rotation[:, 2, 2] > .80)
            starts = np.flatnonzero(np.convolve(upright, np.ones(10), "valid") == 10)
            windows.extend(frames[start:start + 10] for start in starts)
            poses.append(q[-6:, -14:].mean(axis=0))
    if not windows:
        raise ValueError(f"No upright ten-frame reference windows in {directory}")
    return np.stack(windows).astype(np.float32), np.stack(poses)


def capture_states(checkpoints, directory):
    """Record seeds from the actual fall-to-stand controllers before training the pose residual."""
    from tools.play_getup import GetupPlayer

    paths = []
    for number, checkpoint in enumerate(checkpoints):
        player = GetupPlayer(checkpoint)
        for trial in range(len(player.names)):
            player.reset(trial)
            q, v, actions = [], [], []
            for tick in range(1251):
                q.append(player.data.qpos.copy())
                v.append(player.data.qvel.copy())
                actions.append(player.previous.copy())
                if tick < 1250:
                    player.step()
            path = directory / f"{number}_{trial}.npz"
            np.savez(path, qpos=q, qvel=v, action=actions)
            paths.append(path)
    return paths


def initialize_worker(paths):
    global model, regulator, data, links, starts
    model = build_scene(robot_name="g1_getup")
    regulator = _StandingDynamics(model)
    data = mujoco.MjData(model)
    links = [model.body("robot_" + name).id for name in robot.KEY_BODIES]
    starts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as f:
            starts.append({key: f[key] for key in ("qpos", "qvel", "action")})


def simulate(args):
    trial, index, seed, pose, steps = args
    sample = starts[trial]
    q, anchor = sample["qpos"][index], sample["action"][index]
    rng = np.random.default_rng(seed)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q
    data.qvel[:] = sample["qvel"][index]
    data.qvel[:2] += rng.normal(0., .023, 2)
    mujoco.mj_forward(model, data)
    previous = anchor.copy()
    frames, windows, metrics, arm_speed = [], [], [], []
    for tick in range(steps):
        if tick == 55 and seed % 3:
            data.qvel[0] += .12 * (-1 if seed % 2 else 1)
        gyro, axes, joints, speeds, applied, points = observation_terms(data, links, previous)
        frames.append(np.concatenate((axes, gyro, joints, speeds, points)))
        if (tick + 1) % 20 == 0:
            windows.append(frames[-10:])
        # The low-level feedback reads the latest gyro, axes, encoders and joint velocities.
        obs = np.concatenate([np.tile(term, 5) for term in (gyro, axes, joints, speeds, applied, points)])
        action = regulator(obs, pose)
        mix = min(1., (tick + 1) / 20)
        previous = anchor + mix * (action - anchor)
        advance(model, data, previous)
        upright = 1. - 2. * np.sum(data.qpos[4:6]**2)
        metrics.append((data.qpos[2], upright, np.linalg.norm(data.qvel[:2]), np.linalg.norm(data.qpos[:2] - q[:2])))
        arm_speed.append(data.qvel[21:].copy())
        if data.qpos[2] < .49:
            break
    if not windows:
        windows.append((frames + [frames[-1]] * 10)[-10:])
    metrics = np.asarray(metrics)
    tail = metrics[-25:]
    stable = len(metrics) == steps and (tail[:, 0] > .63).all() and (tail[:, 1] > .82).all()
    return np.asarray(windows), np.array([stable, tail[:, 2].mean(), np.sqrt(np.mean(np.square(arm_speed[-25:]))),
                                           metrics[:, 3].max()]), data.qpos[22:].copy()


def reset_choices(paths):
    eligible, actions = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as f:
            q, v = f["qpos"], f["qvel"]
            good = ((q[:, 2] > .67) & (1 - 2 * np.square(q[:, 4:6]).sum(-1) > .90)
                    & (np.linalg.norm(v[:, :2], axis=-1) < .30) & (np.linalg.norm(v[:, 3:6], axis=-1) < 1.3))
            ii = np.flatnonzero(good)
            if not len(ii):
                raise ValueError(f"No standing states in {path}")
            early = ii[ii < min(ii[0] + 220, 700)]
            eligible.append((ii, early if len(early) else ii))
            actions.append(f["action"])
    return eligible, actions


def reference_initializer(pool, eligible, actions, reference):
    # Estimate how each high-level action changes the actual settled joint posture.
    poses = [np.zeros(7)] + [sign * .4 * axis for sign in (-1, 1) for axis in np.eye(7)]
    trials = list(range(min(3, len(eligible))))
    pairs = [(trial, eligible[trial][0][-1], 816, pose, 195) for pose in poses for trial in trials]
    results = pool.map(simulate, pairs)
    joints = np.stack([r[2] for r in results]).reshape(15, len(trials), -1).mean(1)
    jacobian = ((joints[8:] - joints[1:8]) / .8).T
    matrix = np.vstack((jacobian, .18 * np.eye(7), [0., 0., 0., 0., 0., 0., .8]))
    targets = [np.arctanh(lsq_linear(matrix, np.r_[pose - joints[0], np.zeros(7), .20],
                                     bounds=(-.88, .88)).x) for pose in reference]
    x, y = [], []
    for trial, ((all_states, early), actor) in enumerate(zip(eligible, actions)):
        ids = np.r_[early[:10], all_states[::max(1, len(all_states) // 12)][:12]]
        x.extend(pose_features(actor[t]) for t in ids)
        y.extend([targets[trial % len(reference)]] * len(ids))
    x = np.asarray(x)
    return np.linalg.solve(x.T @ x + 5. * np.eye(x.shape[1]), x.T @ y).astype(np.float32)


def train(args, paths):
    import jax
    import jax.numpy as jnp
    from instantamp import amp, nets, optim

    windows, references = reference_data(args.reference)
    eligible, actions = reset_choices(paths)
    args.out.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(args.workers, initializer=initialize_worker, initargs=(paths,)) as pool:
        if args.init:
            with np.load(args.init, allow_pickle=False) as initial:
                prior = jnp.asarray(initial["weight"], jnp.float32)
        else:
            prior = jnp.asarray(reference_initializer(pool, eligible, actions, references))
        if prior.shape != (14, 7):
            raise ValueError("Expected an affine 13-sensor, 7-action posture policy")
        weight, optimizer = prior, optim.init(prior, False)
        discriminator = amp.init(jax.random.key(392), 10, hidden=(96, 64), frame_dim=85)
        discriminator["norm"] = nets.rn_update(discriminator["norm"], jnp.asarray(windows.reshape(-1, 85)))
        disc_update = jax.jit(lambda d, key, real, fake: amp.train_step(
            d, real, fake, jnp.ones(len(real)), key, .00022, recovery=True, leggedlab=True))
        score = jax.jit(lambda d, x: amp.recovery_style(d, x, 1., leggedlab=True))
        sample_rng, policy_rng = np.random.default_rng(9484), np.random.default_rng(639)
        sigma = .31

        @jax.jit
        def ppo(w, opt, x, action, previous_logp, advantage):
            def loss(params):
                mean = x @ params
                logp = -.5 * jnp.sum(((action - mean) / sigma)**2, axis=1)
                ratio = jnp.exp(jnp.clip(logp - previous_logp, -6., 6.))
                surrogate = jnp.minimum(ratio * advantage, jnp.clip(ratio, .8, 1.2) * advantage)
                return -surrogate.mean() + .18 * jnp.mean((mean - x @ prior)**2)
            return optim.step(w, jax.grad(loss)(w), opt, False, .0058)

        for iteration in range(1, args.iterations + 1):
            trials = sample_rng.integers(0, len(paths), args.batch)
            indices = [int(sample_rng.choice(eligible[trial][sample % 2])) for sample, trial in enumerate(trials)]
            x = np.stack([pose_features(actions[trial][i]) for trial, i in zip(trials, indices)]).astype(np.float32)
            noise = policy_rng.normal(size=(args.batch, 7)).astype(np.float32)
            sampled = x @ np.asarray(weight) + sigma * noise
            batch = [(trial, index, 48000 * iteration + i, np.tanh(u), args.steps)
                     for i, (trial, index, u) in enumerate(zip(trials, indices, sampled))]
            result = pool.map(simulate, batch)
            lengths = [len(r[0]) for r in result]
            fake = np.concatenate([r[0] for r in result]).astype(np.float32)
            for k in range(2):
                real = windows[policy_rng.integers(len(windows), size=96)]
                sampled_fake = fake[policy_rng.integers(len(fake), size=96)]
                discriminator, stats = disc_update(discriminator, jax.random.key(iteration * 13 + k), real, sampled_fake)
            windows_by_episode = np.split(np.asarray(score(discriminator, fake)), np.cumsum(lengths)[:-1])
            style_reward = np.array([scores[-2:].mean() for scores in windows_by_episode])
            z = np.stack([r[1] for r in result])
            errors = np.array([np.sqrt(np.mean((r[2] - references[t % len(references)])**2))
                               for r, t in zip(result, trials)])
            task = (1.3 * z[:, 0] + .3 * np.exp(-7 * z[:, 1]**2) + .28 * np.exp(-.5 * z[:, 2]**2)
                    + 1.5 * np.exp(-5 * errors**2) - .2 * np.minimum(z[:, 3], 1.))
            reward = task + .8 * style_reward
            context = np.column_stack((np.ones(len(z)), trials % len(references), indices, x[:, :-1]))
            baseline = context @ np.linalg.solve(context.T @ context + .2 * np.eye(context.shape[1]), context.T @ reward)
            advantage = (reward - baseline).astype(np.float32)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
            for _ in range(4):
                weight, optimizer = ppo(weight, optimizer, x, sampled, -.5 * np.square(noise).sum(1), advantage)
            print(f"{iteration:04d}  standing {z[:,0].mean():.2f}  arm RMS {np.degrees(errors).mean():.2f} deg  "
                  f"AMP {style_reward.mean():.3f}  D(demo) {float(stats['d_demo']):.2f}  "
                  f"D(policy) {float(stats['d_robot']):.2f}", flush=True)
            if iteration == 1 or iteration % 5 == 0 or iteration == args.iterations:
                np.savez(args.out / f"{iteration:03d}.npz", weight=np.asarray(weight), demo_windows=len(windows),
                         iterations=iteration, episodes=iteration * args.batch, AMP_weight=np.float32(.8))
                layers = {f"layer_{i}_{k}": v for i, layer in enumerate(discriminator["net"]) for k, v in layer.items()}
                np.savez(args.out / f"{iteration:03d}_disc.npz", **layers, mean=discriminator["norm"]["mean"],
                         var=discriminator["norm"]["var"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--controllers", nargs="+", type=Path, help="Get-up checkpoints to roll out from each floor reset")
    source.add_argument("--rollouts", nargs="+", type=Path,
                        help="Existing native qpos/qvel/action files in checkpoint-major, sorted reference-clip order")
    parser.add_argument("--reference", type=Path, default=Path("data/g1_getup/recovery"))
    parser.add_argument("--init", type=Path, help="Optional initial pose residual with a 14-by-7 weight")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=54)
    args = parser.parse_args()
    if min(args.iterations, args.workers, args.batch) < 1 or args.steps < 20:
        parser.error("Use positive iteration, worker and batch counts and at least twenty physics control steps")
    with tempfile.TemporaryDirectory() as temp:
        paths = args.rollouts or capture_states(args.controllers, Path(temp))
        train(args, paths)


if __name__ == "__main__":
    main()
