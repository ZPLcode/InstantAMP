"""Collect native Zenbot teacher examples and distill one command-conditioned 495→23 actor.

The optional CPU fitting step requires PyTorch. The exported weights use InstantAMP's
ordinary dense actor format and can initialize the AMP/PPO trainer or the native viewer.
"""
import argparse
from pathlib import Path

import numpy as np


class DenseActor:
    """Float32 NumPy inference for the dense ELU teacher or student in native data collection."""

    def __init__(self, checkpoint, *, allow_layer_norm=False):
        with np.load(checkpoint, allow_pickle=False) as data:
            self.layers = [(data[f"avg/{i}/w"], data[f"avg/{i}/b"]) for i in range(4)]
            self.mean = data["obs_norm/mean"]
            self.std = np.sqrt(data["obs_norm/var"] + .01**2)
            self.raw = "obs_norm/raw" in data
            self.layer_norm = {i: (data[f"avg/{i}/ln_g"], data[f"avg/{i}/ln_b"])
                               for i in range(len(self.layers) - 1) if f"avg/{i}/ln_g" in data}
            if (self.layer_norm and not allow_layer_norm) or any(k.startswith("ode/") for k in data.files):
                raise ValueError("This collector requires dense ELU actors that output absolute actions")

    def __call__(self, history):
        x = history.reshape(-1).astype(np.float32)
        if not self.raw:
            x = np.clip((x - self.mean) / self.std, -10., 10.)
        for i, (weight, bias) in enumerate(self.layers):
            x = x @ weight + bias
            if i < len(self.layers) - 1:
                if i in self.layer_norm:
                    gain, offset = self.layer_norm[i]
                    centered = x - x.mean()
                    x = centered / np.sqrt(np.mean(centered**2) + np.float32(1e-6)) * gain + offset
                x = np.maximum(0., x) + np.expm1(np.minimum(0., x))
        return x


def native_frames(data, root, bodies, previous, command, teacher_speed, default_q):
    """Physical, heading-independent student frame and original gait-teacher frame."""
    pos, vel = data.qpos, data.qvel
    rot = data.xmat[root].reshape(3, 3)
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    local_rot = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]]) @ rot
    points = (data.xpos[bodies] - pos[:3]) @ rot
    student = np.r_[vel[3:6], local_rot[:, 0], local_rot[:, 2], pos[7:], vel[6:],
                    previous, points.ravel(), command, 0., 0.]
    gait = np.r_[vel[3:6], -rot[2], teacher_speed, 0., 0., pos[7:] - default_q, vel[6:], previous]
    return student, gait


def collect(args):
    import mujoco
    from instantamp import env, zenbot
    from instantamp.evaluate import fallen_starts
    from instantamp.sim import build_scene

    rng = np.random.default_rng(args.seed)
    rise, gait = DenseActor(args.rise), DenseActor(args.gait)
    student = DenseActor(args.student) if args.student else None
    model = build_scene(robot_name="zenbot23_getup")
    _, poses, velocities = fallen_starts(args.floor, zenbot.RESET_LIFT)
    poses, velocities = np.asarray(poses), np.asarray(velocities)
    bodies = [model.body("robot_" + n).id for n in zenbot.KEY_BODIES]
    data = mujoco.MjData(model)
    inputs, actions, labels, speeds, blends = [], [], [], [], []
    for episode in range(args.episodes):
        on_floor = args.floor_only or episode % 3 != 2
        mujoco.mj_resetData(model, data)
        if on_floor:
            k = (episode if args.floor_only else episode // 3 * 2 + episode % 3) % len(poses)
            data.qpos[:] = poses[k]
            data.qvel[:] = velocities[k]
            data.qpos[7:] += rng.normal(0., .015, model.nu)
        else:
            data.qpos[:] = np.r_[0., 0., zenbot.DEFAULT_HEIGHT, 1., 0., 0., 0., zenbot.DEFAULT_Q]
            data.qpos[7:] += rng.normal(0., .02, model.nu)
        lags = rng.integers(3, 8, 5)[zenbot.GROUP]
        previous = np.zeros(model.nu, np.float32)
        older = previous.copy()
        history = np.zeros((5, 99), np.float32)
        gait_history = np.zeros((5, 78), np.float32)
        speed = 0.
        blend = 0. if on_floor else 1.
        stable = student_steps = 0
        intent = float(rng.choice(args.speeds))
        command = intent if on_floor else 0.
        if not on_floor:
            intent = float(rng.choice(args.speeds))
        for step in range(args.steps):
            mujoco.mj_kinematics(model, data)
            pos, vel = data.qpos, data.qvel
            upright = data.xmat[model.jnt_bodyid[0], 8]
            if step and step % 100 == 0:
                intent = float(rng.choice(args.speeds))
            if step % 4 == 0:
                command += np.clip(intent-command, -.1, .1)
            if on_floor:
                stable = stable + 1 if pos[2] > .58 and upright > .88 and np.linalg.norm(vel[:3]) < 1. else 0
                if stable >= 18 or blend > 0.:
                    blend = min(blend + 1. / 65, 1.)
                if pos[2] < .42 or upright < .5:
                    stable, blend = 0, 0.
            if step % 4 == 0:
                speed += np.clip((command if blend > .9 else 0.) - speed, -.1, .1)
            student_frame, gait_frame = native_frames(data, model.jnt_bodyid[0], bodies, previous,
                                                      command, speed, zenbot.DEFAULT_Q)
            for bank, frame in ((history, student_frame), (gait_history, gait_frame)):
                if step == 0:
                    bank[:] = frame
                else:
                    bank[:-1] = bank[1:]
                    bank[-1] = frame
            target = rise(history[:, :96]) * (1. - blend) if blend < 1. else 0.
            if blend > 0.:
                target = target + blend * gait(gait_history)
            if on_floor or (pos[2] > .45 and upright > .58):
                phase = 0 if blend < .3 else 1 if blend < 1. else 2 if speed < .2 else 3 if speed < 1.6 else 4
                inputs.append(history.flatten())
                actions.append(target)
                labels.append(phase)
                speeds.append(gait_history[:, 6].copy())
                blends.append(blend)
            if student_steps <= 0:
                student_steps = int(rng.integers(4, 42))
                student_weight = rng.uniform(args.min_share, args.share)
            student_steps -= 1
            action = (1. - student_weight) * target + student_weight * student(history) if student else target
            if args.noise:
                action = action + rng.normal(0., args.noise, model.nu)
                if step and step % 55 == 0:
                    vel[:3] += rng.normal(0., (.08, .08, .04))
                    vel[3:6] += rng.normal(0., .10, 3)
            for substep in range(env.DECIMATION):
                delayed = np.where(substep >= lags, action,
                                   np.where(substep + env.DECIMATION >= lags, previous, older))
                joint_target = zenbot.DEFAULT_Q + zenbot.ACTION_SCALE * delayed
                torque = np.clip(zenbot.KP * (joint_target - pos[7:]) - zenbot.KD * vel[6:],
                                 -zenbot.EFFORT, zenbot.EFFORT)
                data.ctrl[:] = np.where((np.abs(vel[6:]) >= zenbot.SPEED) & (torque * vel[6:] > 0.), 0., torque)
                mujoco.mj_step(model, data)
            older, previous = previous, action
            if not on_floor and (pos[2] < .45 or upright < .6):
                break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, x=np.asarray(inputs, np.float32), u=np.asarray(actions, np.float32),
             mode=np.asarray(labels, np.int8), default_q=zenbot.DEFAULT_Q,
             teacher_speed=np.asarray(speeds, np.float32), gait_blend=np.asarray(blends, np.float32))
    print(f"Saved {len(inputs)} physics transitions to {args.out}")


def initial_weights(source, default_q):
    """Map the input affine layer into the five-frame command-conditioned observation."""
    weight, bias = source.layers[0]
    if weight.shape[0] == 480:
        weight = np.pad(weight.reshape(5, 96, 512), ((0, 0), (0, 3), (0, 0))).reshape(495, 512)
    elif weight.shape[0] == 390:
        # Initialize at upright pitch, where -R_zy equals the y component of the third
        # heading-independent rotation column. Teacher labels below use exact gravity.
        mapping = np.zeros((99, 78), np.float32)
        mapping[:3, :3] = np.eye(3)
        mapping[5, 3], mapping[7, 4], mapping[8, 5] = -1., 1., -1.
        mapping[96:, 6:9] = np.eye(3)
        mapping[9:78, 9:] = np.eye(69)
        offset = np.tile(np.r_[np.zeros(9), -default_q, np.zeros(46)].astype(np.float32), 5)
        if source.raw:
            bias = bias + offset @ weight
        else:
            weight = weight / source.std[:, None]
            bias = bias + (offset - source.mean) @ weight
        weight = np.einsum("ik,fko->fio", mapping, weight.reshape(5, 78, -1)).reshape(495, -1)
    elif weight.shape[0] != 495:
        raise ValueError(f"Expected a 390-, 480- or 495-input teacher/student, got {weight.shape[0]}")
    if len(source.mean) != 390 and not source.raw:
        raise ValueError("The 480- and 495-input initializers must use raw observations")
    return [(weight, bias), *source.layers[1:]]


def torch_actor(actor, torch):
    weights = [(torch.from_numpy(w), torch.from_numpy(b)) for w, b in actor.layers]
    mean, std = torch.from_numpy(actor.mean), torch.from_numpy(actor.std)

    def apply(frames):
        x = frames.flatten(1)
        if not actor.raw:
            x = ((x - mean) / std).clamp(-10, 10)
        for i, (weight, bias) in enumerate(weights):
            x = x @ weight + bias
            if i < len(weights) - 1:
                x = torch.nn.functional.elu(x)
        return x

    return apply


def gait_features(frames, default_q, torch):
    # The middle gravity coordinate uses the cross product of the supplied third and first columns.
    gravity = torch.stack((-frames[:, :, 5], frames[:, :, 7] * frames[:, :, 3] - frames[:, :, 6] * frames[:, :, 4],
                           -frames[:, :, 8]), -1)
    return torch.cat((frames[:, :, :3], gravity, frames[:, :, 96:], frames[:, :, 9:32] - default_q,
                      frames[:, :, 32:78]), -1)


def export_actor(layers, mean, scale, destination):
    # Fold fitting normalization into the first affine layer for normal raw 495-input deployment.
    fields = {"obs_norm/mean": np.zeros(495, np.float32), "obs_norm/var": np.ones(495, np.float32),
              "obs_norm/raw": True}
    for i in range(4):
        weight = layers[2*i].weight.detach().numpy().T
        bias = layers[2*i].bias.detach().numpy()
        if i == 0:
            weight = weight / scale.numpy()[:, None]
            bias = bias - mean.numpy() @ weight
        for key in ("avg", "live"):
            fields[f"{key}/{i}/w"] = weight
            fields[f"{key}/{i}/b"] = bias
    np.savez_compressed(destination, **fields)


def teacher_perturbation(frames, target, phase, augmentable, speeds, blends, noise,
                         rise, gait, default, perturb_handoff, torch):
    """Use measured teacher inputs and leave mismatched legacy examples exactly as recorded."""
    known = torch.isfinite(blends) & torch.isfinite(speeds).all(1)
    share = blends.nan_to_num()
    mixed = torch.where(known, (share > 0.) & (share < 1.), phase == 1)
    floor = torch.where(known, share == 0., phase == 0)
    moving = ~(mixed | floor)
    permit = augmentable.clone()
    if not perturb_handoff:
        permit &= ~mixed
    altered = frames + noise
    delta = torch.zeros_like(target)

    def matches(original, recorded):
        # Legacy demonstrations sometimes expose the final operator command
        # to the student while the teacher still sees zero or a speed ramp.
        return (original - recorded).square().mean(1) < .015**2

    for mask, use_rise, use_gait in ((floor, True, False), (moving, False, True),
                                     (mixed, True, True)):
        mask = mask & permit
        if not mask.any():
            continue
        pair = torch.cat((altered[mask], frames[mask]))
        if use_rise:
            rise_new, rise_old = rise(pair[:, :, :96]).chunk(2)
        if use_gait:
            gv = gait_features(pair, default, torch)
            actual_speed = speeds[mask]
            requested = frames[mask, :, 96]
            # Recorded teacher speeds take precedence over the exposed command.
            # For an undocumented handoff try the zero-speed standing teacher;
            # below, an unmatched output falls back to the original recorded pair.
            fallback = torch.zeros_like(requested) if use_rise else requested
            actual_speed = torch.where(known[mask, None], actual_speed, fallback)
            response = torch.where(torch.isclose(requested, actual_speed, atol=.01),
                                   noise[mask, :, 96], 0.)
            gv[:, :, 6] = torch.cat((actual_speed + response, actual_speed))
            gait_new, gait_old = gait(gv).chunk(2)
        if use_rise and use_gait:
            direction = gait_old - rise_old
            estimate = (((target[mask] - rise_old) * direction).sum(1, keepdim=True)
                        / direction.square().sum(1, keepdim=True).clamp_min(1e-8)).clamp(0., 1.)
            mix = torch.where(known[mask, None], share[mask, None], estimate)
            base = (1. - mix) * rise_old + mix * gait_old
            response = (1. - mix) * (rise_new - rise_old) + mix * (gait_new - gait_old)
        elif use_rise:
            base, response = rise_old, rise_new - rise_old
        else:
            base, response = gait_old, gait_new - gait_old
        valid = matches(base, target[mask])
        delta[mask] = torch.where(valid[:, None], response, 0.)
        permit[mask] = valid
    noise[~permit] = 0.
    return noise, delta


def fit(args):
    import torch
    from torch import nn

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    inputs, actions, modes, augmentable, speeds, blends = [], [], [], [], [], []
    default_q = np.load(args.default_q) if args.default_q else None
    for path in [*args.data, *args.recorded_only]:
        with np.load(path) as batch:
            inputs.append(batch["x"])
            actions.append(batch["u"])
            modes.append(batch["mode"])
            n = len(modes[-1])
            augmentable.append(np.full(n, path not in args.recorded_only))
            speeds.append(batch["teacher_speed"] if "teacher_speed" in batch else np.full((n,5), np.nan, np.float32))
            blends.append(batch["gait_blend"] if "gait_blend" in batch else np.full(n, np.nan, np.float32))
            if default_q is None and "default_q" in batch:
                default_q = batch["default_q"]
    if default_q is None:
        from instantamp.zenbot import DEFAULT_Q
        default_q = DEFAULT_Q
    x = torch.from_numpy(np.concatenate(inputs))
    y = torch.from_numpy(np.concatenate(actions))
    mode = torch.from_numpy(np.concatenate(modes))
    augmentable = torch.from_numpy(np.concatenate(augmentable))
    speeds = torch.from_numpy(np.concatenate(speeds))
    blends = torch.from_numpy(np.concatenate(blends))
    if x.shape != (len(y), 495) or y.shape[1:] != (23,) or mode.shape != (len(y),):
        raise ValueError("Expected arrays x[N,495], u[N,23], mode[N] in the training bank")
    groups = [torch.nonzero(mode == m).flatten() for m in range(5)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("The training data must cover rise, handoff, stand, walk and run")
    layers = nn.Sequential(nn.Linear(495, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                           nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 23))
    mean, scale = x.mean(0), x.std(0).clamp_min(.15)
    for i, (weight, bias) in enumerate(initial_weights(DenseActor(args.init), default_q)):
        weight, bias = torch.as_tensor(weight), torch.as_tensor(bias)
        if i == 0:
            bias = bias + mean @ weight
            weight = scale[:, None] * weight
        layers[2*i].weight.data.copy_(weight.T)
        layers[2*i].bias.data.copy_(bias)
    x = (x - mean) / scale
    default = torch.from_numpy(default_q)
    rise, gait = (torch_actor(DenseActor(path), torch) for path in (args.rise, args.gait))
    sigma = torch.tensor([.23]*3 + [.045]*6 + [.065]*23 + [.60]*23 + [.28]*23 + [.025]*18 + [.06, 0., 0.])
    optimizer = torch.optim.Adam(layers.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    for iteration in range(1, args.updates + 1):
        idx = torch.cat([group[torch.randint(len(group), (count,))] for group, count in
                         zip(groups, (112, 36, 36, 36, 36))])
        features, target = x[idx], y[idx]
        if args.perturb:
            # Learn the stabilizing local response around demonstration states. Handoff
            # labels already blend the two teachers and retain their measured targets.
            with torch.no_grad():
                frames = (features * scale + mean).reshape(-1, 5, 99)
                noise = (.5 * torch.randn_like(frames) + torch.randn_like(frames[:, :1])) * sigma
                noise, delta = teacher_perturbation(frames, target, mode[idx], augmentable[idx],
                                                   speeds[idx], blends[idx], noise, rise, gait, default,
                                                   args.perturb_handoff, torch)
                features = features + noise.flatten(1) / scale
                target = target + delta
        loss = (layers(features) - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(layers.parameters(), 5.)
        optimizer.step()
        if iteration % args.save_every == 0 or iteration == args.updates:
            export_actor(layers, mean, scale, args.out / f"{iteration:06}.npz")
            print(f"{iteration} mean squared action error {loss.detach().item():.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="command", required=True)
    collect_cmd = cmd.add_parser("collect", help="simulate frozen teachers or a teacher/student mixture in MuJoCo")
    fit_cmd = cmd.add_parser("fit", help="fit one dense actor to one or more demonstration banks on CPU")
    for sub in (collect_cmd, fit_cmd):
        sub.add_argument("--rise", type=Path, required=True, help="480-input Zenbot get-up checkpoint")
        sub.add_argument("--gait", type=Path, required=True, help="390-input command-conditioned gait checkpoint")
        sub.add_argument("--out", type=Path, required=True)
    collect_cmd.add_argument("--floor", type=Path, required=True, help="directory containing Zenbot 50 Hz motion clips")
    collect_cmd.add_argument("--episodes", type=int, default=90)
    collect_cmd.add_argument("--steps", type=int, default=500)
    collect_cmd.add_argument("--student", type=Path, help="optional student used during physical data collection")
    collect_cmd.add_argument("--share", type=float, default=0., help="maximum student action weight per rollout segment")
    collect_cmd.add_argument("--min-share", type=float, default=0., help="minimum student weight per segment")
    collect_cmd.add_argument("--floor-only", action="store_true", help="collect exclusively from fallen starts")
    collect_cmd.add_argument("--speeds", type=float, nargs="+", default=[0., 1., 2.5], help="forward commands in m/s")
    collect_cmd.add_argument("--noise", type=float, default=0., help="Gaussian raw-action noise standard deviation")
    collect_cmd.add_argument("--seed", type=int, default=491)
    fit_cmd.add_argument("data", nargs="+", type=Path)
    fit_cmd.add_argument("--recorded-only", type=Path, nargs="+", default=[],
                         help="additional banks from alternative teachers, trained only on their recorded sensor/action pairs")
    fit_cmd.add_argument("--init", type=Path, required=True, help="gait, get-up or previously distilled dense checkpoint")
    fit_cmd.add_argument("--default-q", type=Path, help="optional robot neutral pose .npy for legacy external data banks")
    fit_cmd.add_argument("--perturb", action=argparse.BooleanOptionalAction, default=True,
                         help="imitate the teachers' responses to local sensor and action-history changes")
    fit_cmd.add_argument("--perturb-handoff", action="store_true",
                         help="also imitate the local responses of both teachers during a graded handoff")
    fit_cmd.add_argument("--updates", type=int, default=6000)
    fit_cmd.add_argument("--threads", type=int, default=2)
    fit_cmd.add_argument("--seed", type=int, default=444)
    fit_cmd.add_argument("--lr", type=float, default=.00016)
    fit_cmd.add_argument("--save-every", type=int, default=250)
    args = parser.parse_args()
    if args.command == "collect":
        if args.episodes <= 0 or args.steps <= 0 or not 0. <= args.min_share <= args.share <= 1. or args.noise < 0:
            parser.error("Invalid collection length, action noise or student mixing fraction")
        collect(args)
    else:
        if min(args.updates, args.threads, args.save_every) <= 0 or args.lr <= 0:
            parser.error("Fitting counts and learning rate must be positive")
        fit(args)


if __name__ == "__main__":
    main()
