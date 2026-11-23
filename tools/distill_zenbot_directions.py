"""Transfer the four-skill and multidirectional gait teachers into one Motion Train actor."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from instantamp.zenbot_motion_train_obs import SENSOR_NOISE, teacher_yaw
from tools.train_zenbot_skill_actor import DenseActor

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "releases/zenbot_four_skills"
FLOOR = ROOT / "data/zenbot23_walk_run_recovery/recovery"
COMMANDS = np.asarray([[-0.5, 0, 0], [0, 0.4, 0], [0, -0.4, 0], [0, 0, 0.6], [0, 0, -0.6]])


def original_model():
    """Compile the shipped original MJCF without its zero-mass visual mesh assets."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(ROOT / "assets/zenbot_23dof_43kg.xml"))
    for geom in tuple(spec.geoms):
        if (
            geom.type == mujoco.mjtGeom.mjGEOM_MESH
            and not geom.contype
            and not geom.conaffinity
            and geom.density == 0.0
        ):
            spec.delete(geom)
    for mesh in tuple(spec.meshes):
        spec.delete(mesh)
    spec.option.timestep = 0.005
    return spec.compile()


def collect(args):
    import mujoco as mj

    from instantamp import zenbot
    from instantamp.evaluate import fallen_starts
    from instantamp.sim import build_scene
    from instantamp.zenbot_motion_train_obs import MOTION_TRAIN_INDEX
    from tools.collect_zenbot_recovery_gait import motor_step
    from tools.train_zenbot_skill_actor import native_frames

    rng = np.random.default_rng(args.seed)
    old, gait = DenseActor(args.old), DenseActor(args.gait)
    ground = DenseActor(args.ground_teacher) if args.ground_teacher else None
    student = DenseActor(args.student) if args.student else None
    if student:
        import jax

        from instantamp.zenbot_motion_train_obs import motion_train_observation

        project = jax.jit(motion_train_observation)
        if args.corrupt_student:
            from instantamp.zenbot_motion_train_obs import advance_sensors

            advance = jax.jit(advance_sensors)
    model = original_model() if args.original_physics else build_scene(robot_name="zenbot23_getup")
    prefix = "" if args.original_physics else "robot_"
    data = mj.MjData(model)
    base = model.jnt_bodyid[0]
    bodies = [model.body(prefix + n).id for n in zenbot.KEY_BODIES]
    soles = (
        []
        if args.original_physics
        else [model.geom("robot_" + n + "_collision").id for n in zenbot.FEET]
    )
    floor = model.geom("ground" if args.original_physics else "floor").id
    weight = model.body_mass.sum() * 9.81
    _, starts, velocities = fallen_starts(FLOOR, zenbot.RESET_LIFT)
    states, modes, shares, side_labels, on_ground = [], [], [], [], []
    palms = {model.body(prefix + name).id for name in zenbot.HANDS}
    palm_geoms = set(np.flatnonzero(np.isin(model.geom_bodyid, list(palms))))
    wrench = np.empty(6)
    for episode in range(args.episodes):
        mj.mj_resetData(model, data)
        on_floor = args.after_rise or args.floor_only or episode % 3 == 0
        released = on_floor or episode % 3 == 1
        if on_floor:
            sample = episode if args.floor_only or args.after_rise else episode // 3
            k = (
                (sample + args.seed) % len(starts)
                if not args.floor_trials
                else args.floor_trials[(sample + args.seed) % len(args.floor_trials)]
            )
            data.qpos[:], data.qvel[:] = starts[k], velocities[k]
            if args.reference_jitter:
                from scipy.spatial.transform import Rotation

                data.qpos[7:] += np.deg2rad(rng.uniform(-2.0, 2.0, 23))
                rotation = Rotation.from_quat(data.qpos[3:7], scalar_first=True)
                tilt = Rotation.from_euler("xy", rng.uniform(-2.0, 2.0, 2), degrees=True)
                data.qpos[3:7] = (rotation * tilt).as_quat(scalar_first=True)
                data.qvel[:2] += rng.uniform(-0.07, 0.07, 2)
            elif args.source_jitter:
                from scipy.spatial.transform import Rotation

                data.qvel[:] = 0.0
                rot = Rotation.from_quat(data.qpos[3:7], scalar_first=True)
                tilt = Rotation.from_euler("xy", rng.uniform(-2.0, 2.0, 2), degrees=True)
                data.qpos[3:7] = (rot * tilt).as_quat(scalar_first=True)
                data.qvel[:2] = rng.uniform(-0.07, 0.07, 2)
            elif args.original_physics and episode % 2 == 0:
                data.qvel[:] = 0.0
            else:
                data.qvel[:2] += rng.normal(0.0, 0.025, 2)
        else:
            data.qpos[:] = np.r_[
                0.0, 0.0, zenbot.DEFAULT_HEIGHT, 1.0, 0.0, 0.0, 0.0, zenbot.DEFAULT_Q
            ]
        if not args.reference_jitter:
            data.qpos[7:] += rng.normal(0.0, 0.025, 23)
        if args.original_physics:
            mj.mj_forward(model, data)
        delays = rng.integers(3, 8, 5)[zenbot.GROUP]
        previous = older = np.zeros(23)
        history = np.zeros((5, 99), np.float32)
        gh = np.zeros((5, 78), np.float32)
        begin = len(states)
        up_ticks = palm_ticks = 0
        peak_height = side_velocity = 0.0
        intent = np.zeros(3, np.float32)
        if args.floor_intent:
            requests = np.r_[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]], COMMANDS]
            intent = requests[episode % len(requests)]
        sensors = None
        if args.corrupt_student:
            sensor_seed = jax.random.fold_in(jax.random.key(args.seed), episode)
        for tick in range(args.steps):
            if not args.original_physics:
                mj.mj_kinematics(model, data)
            if args.after_rise:
                commands = COMMANDS[1:3] if args.side_only else COMMANDS
                command = (0.0, 0.0, 0.0) if tick < 600 else commands[episode % len(commands)]
                if tick == 600 and (data.qpos[2] < 0.58 or data.xmat[base, 8] < 0.866):
                    break
            elif released:
                command = (
                    (0.0, 0.0, 0.0)
                    if on_floor or tick < 60
                    else (1.0 if tick < 225 else 2.5 if tick < 420 else 0.0, 0.0, 0.0)
                )
            else:
                if tick == 0 or tick == 50 or tick % 190 == 0:
                    command = (
                        (0.0, 0.0, 0.0)
                        if tick == 0
                        else COMMANDS[rng.integers(5)] * rng.uniform(0.9, 1.1)
                    )
            frame, gf = native_frames(
                data, base, bodies, previous, command[0], command[0], zenbot.DEFAULT_Q
            )
            if args.original_physics:
                frame[:3] = gf[:3] = data.sensor("angular-velocity").data
            frame[96:] = intent if args.floor_intent else command
            gf[6:9] = command
            gf[7] *= args.side_gain
            if args.side_feedback:
                rot = data.xmat[base].reshape(3, 3)
                yaw = np.arctan2(rot[1, 0], rot[0, 0])
                vy = -np.sin(yaw) * data.qvel[0] + np.cos(yaw) * data.qvel[1]
                side_velocity = 0.92 * side_velocity + 0.08 * vy
                if command[1]:
                    gf[7] += args.side_feedback * (command[1] - side_velocity)
            gf[8] = teacher_yaw(command[2], gf[2])
            for h, f in ((history, frame), (gh, gf)):
                if tick == 0:
                    h[:] = f
                else:
                    h[:-1] = h[1:]
                    h[-1] = f
            using_gait = (not released and tick >= 50) or (args.after_rise and tick >= 600)
            mode = (
                0
                if not using_gait and on_floor and (tick < 170 or data.qpos[2] < 0.57)
                else 1
                if not using_gait
                else 2
                if command[0]
                else 3
                if command[1]
                else 4
            )
            peak_height = max(peak_height, data.qpos[2])
            ground_share = (
                np.clip((0.57 - peak_height) / 0.16, 0.0, 1.0) if ground is not None else 0.0
            )
            ground_share *= args.ground_weight
            if ground_share > 0:
                mode = 5
            if tick % 2 == 0 and (not args.after_rise or tick >= 585):
                states.append(history.copy())
                modes.append(mode)
                shares.append(ground_share)
                side_labels.append(gh[:, 7].copy())
                on_ground.append(mode in (0, 5) and data.qpos[2] < 0.45)
            if using_gait:
                act = gait(gh)
            elif len(old.mean) == len(MOTION_TRAIN_INDEX):
                act = old(gh.ravel()[MOTION_TRAIN_INDEX])
            else:
                act = old(history)
            if ground_share:
                act = act * (1.0 - ground_share) + ground(history[:, :96]) * ground_share
            if args.corrupt_student:
                sensors = advance(
                    sensors,
                    gf[:6],
                    frame[96:],
                    data.qpos[7:],
                    data.qvel[6:],
                    previous,
                    np.asarray(tick == 0),
                    jax.random.fold_in(sensor_seed, tick),
                )
            if student and tick > 0 and (not args.after_rise or tick >= 600):
                observed = sensors.observation() if args.corrupt_student else project(history)
                act = args.share * student(np.asarray(observed)) + (1.0 - args.share) * act
            if not on_floor and (data.qpos[2] < 0.37 or data.xmat[base, 8] < 0.4):
                break
            act += rng.normal(0.0, 0.025, 23)
            if args.original_physics:
                target = zenbot.DEFAULT_Q + zenbot.ACTION_SCALE * act
                for _ in range(4):
                    torque = zenbot.KP * (target - data.qpos[7:]) - zenbot.KD * data.qvel[6:]
                    data.ctrl[:] = np.clip(torque, -zenbot.EFFORT, zenbot.EFFORT)
                    mj.mj_step(model, data)
            else:
                load, _ = motor_step(
                    model, data, act, previous, older, delays, soles, floor, weight
                )
            older, previous = previous, act
            if args.require_palm and data.qpos[2] < 0.55:
                for contact_index, contact in enumerate(data.contact):
                    if floor in contact.geom and palm_geoms.intersection(contact.geom):
                        mj.mj_contactForce(model, data, contact_index, wrench)
                        if wrench[0] > 0.05 * weight:
                            palm_ticks += 1
                            break
            if args.floor_only:
                upright = data.qpos[2] > 0.58 and data.xmat[base, 8] > 0.866
                if args.quiet:
                    upright = (
                        upright
                        and load.min() > 0.08
                        and load.sum() > 0.5
                        and np.linalg.norm(data.qvel[:2]) < 0.12
                        and np.abs(data.qvel[6:]).mean() < 0.36
                    )
                up_ticks = up_ticks + 1 if upright else 0
                if up_ticks >= 45:
                    break
        rise_seconds = (tick + 1 - up_ticks) * 0.02
        if (args.success_only and (up_ticks < 45 or rise_seconds < args.min_rise_seconds)) or (
            args.require_palm and palm_ticks < 2
        ):
            del states[begin:]
            del modes[begin:]
            del shares[begin:]
            del side_labels[begin:]
            del on_ground[begin:]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    side = {"teacher_side": np.asarray(side_labels, np.float16)} if args.side_feedback else {}
    floor_information = {"on_ground": on_ground} if args.record_ground else {}
    np.savez(
        args.out,
        **side,
        **floor_information,
        x=np.asarray(states, np.float16),
        mode=np.asarray(modes, np.int8),
        default_q=zenbot.DEFAULT_Q,
        motion_train_index=MOTION_TRAIN_INDEX,
        ground_share=np.asarray(shares, np.float16),
    )
    print(args.out, np.bincount(modes), flush=True)


def motion_train_noisy_history(clean, torch, *, latency_samples=None, noise_samples=None):
    """Perturb the last three sensor frames as one Motion Train actor history."""
    device = clean.device
    count = len(clean)
    if latency_samples is None:
        latency_samples = torch.rand((count, 2), device=device)
    u = latency_samples[:, 1]
    delay = torch.where(latency_samples[:, 0] < 0.9, 0.002 + 0.008 * u, 0.010 + 0.025 * u)
    at = torch.arange(2.0, 5.0, device=device)[None] - delay[:, None] / 0.020
    left = at.floor().long()
    rows = torch.arange(count, device=device)[:, None]
    delayed = torch.lerp(
        clean[rows, left, :6], clean[rows, left + 1, :6], (at - left).unsqueeze(-1)
    )
    sensed = clean.clone()
    delayed[:, :, 3:6] = torch.nn.functional.normalize(delayed[:, :, 3:6], dim=-1)
    sensed[:, -3:, :6] = delayed
    if noise_samples is None:
        noise_samples = torch.empty_like(sensed[:, -3:]).uniform_(-1.0, 1.0)
    sensed[:, -3:] += noise_samples * torch.as_tensor(SENSOR_NOISE, device=device)
    return sensed


def with_ground_intents(sensed, on_ground, intents, probability, torch):
    """Give a subset of grounded zero-command students new instructions, preserving teacher inputs."""
    output = sensed.clone()
    eligible = on_ground & (sensed[:, -1, 6:9].abs().sum(-1) == 0)
    change = eligible & (torch.rand(len(sensed), device=sensed.device) < probability)
    sampled = intents[torch.randint(len(intents), (len(sensed),), device=sensed.device)]
    output[:, -3:, 6:9] = torch.where(
        change[:, None, None], sampled[:, None, :], sensed[:, -3:, 6:9]
    )
    return output


def fit(args):
    import torch
    from torch import nn

    from tools.train_zenbot_skill_actor import gait_features

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    samples, modes, shares, side_labels, corrections, on_ground = [], [], [], [], [], []
    for path in args.banks:
        with np.load(path) as bank:
            corrections.append(np.full(len(bank["mode"]), path in args.correction_banks))
            idx = bank["motion_train_index"]
            default_q = bank["default_q"]
            frames = bank["x"].reshape(-1, 5, 99)
            samples.append(frames)
            if "on_ground" in bank:
                on_ground.append(bank["on_ground"])
            else:
                # Older banks did not save height. A grounded label and a pelvis
                # more than 53 degrees from upright select lying/reaching states.
                on_ground.append(np.isin(bank["mode"], [0, 5]) & (frames[:, -1, 8] < 0.6))
            side_labels.append(
                bank["teacher_side"]
                if "teacher_side" in bank
                else np.full((len(bank["mode"]), 5), np.nan, np.float16)
            )
            modes.append(bank["mode"])
            shares.append(
                bank["ground_share"]
                if "ground_share" in bank
                else np.zeros(len(bank["mode"]), np.float16)
            )
    for path in args.floor_replay:
        with np.load(path) as bank:
            retained = bank["mode"] <= 1
            corrections.append(np.zeros(retained.sum(), dtype=bool))
            samples.append(bank["x"][retained].reshape(-1, 5, 99).astype(np.float16))
            modes.append(np.zeros(retained.sum(), dtype=np.int8))
            on_ground.append(samples[-1][:, -1, 8] < 0.6)
            shares.append(np.zeros(retained.sum(), np.float16))
            side_labels.append(np.full((retained.sum(), 5), np.nan, np.float16))
    samples = torch.as_tensor(np.concatenate(samples), device=device)
    modes = np.concatenate(modes)
    corrections = torch.as_tensor(np.concatenate(corrections), device=device)
    on_ground = torch.as_tensor(np.concatenate(on_ground), device=device)
    intents = torch.as_tensor(
        np.r_[[[1.0, 0.0, 0.0], [2.5, 0.0, 0.0]], COMMANDS], dtype=torch.float32, device=device
    )
    shares = torch.as_tensor(np.concatenate(shares), device=device)
    side_labels = torch.as_tensor(np.concatenate(side_labels), device=device)
    groups = [
        torch.as_tensor(np.flatnonzero(modes == group), device=device)
        for group in (0, 1, 5, 2, 3, 4)
    ]
    idx = torch.as_tensor(idx, device=device)
    default_q = torch.as_tensor(default_q, device=device)

    def make_actor(checkpoint):
        dense = DenseActor(checkpoint)
        dims = [len(dense.mean), *(len(b) for _, b in dense.layers)]
        net = nn.Sequential(
            *[
                component
                for i, (left, right) in enumerate(itertools.pairwise(dims))
                for component in (
                    [nn.Linear(left, right), nn.ELU()] if i < 3 else [nn.Linear(left, right)]
                )
            ]
        ).to(device)
        for layer, (weight, bias) in zip(net[::2], dense.layers):
            layer.weight.data.copy_(torch.from_numpy(weight.T))
            layer.bias.data.copy_(torch.from_numpy(bias))
        mean, std = (torch.as_tensor(a, device=device) for a in (dense.mean, dense.std))

        def apply(values):
            if not dense.raw:
                values = ((values - mean) / std).clamp(-10.0, 10.0)
            return net(values)

        return dense, net, apply

    _, _, old = make_actor(args.old)
    gait_actor, _, gait = make_actor(args.gait)
    ground = make_actor(args.ground_teacher)[2] if args.ground_teacher else None
    if args.rehearse_init or args.correction_actor:
        keep_policy = make_actor(args.init)[2]
    if args.correction_actor:
        correct_policy = make_actor(args.correction_actor)[2]
    with torch.no_grad():
        init = DenseActor(args.init) if args.init else gait_actor
        seed_x = gait_features(samples[::9].float(), default_q, torch).flatten(1)[:, idx]
        mean, scale = seed_x.mean(0), seed_x.std(0).clamp_min(0.15)
        if len(init.mean) == 495:
            # Regress its first hidden preactivation onto the reduced physical sensors;
            # joint angles and gravity explain much of the omitted body-position input.
            sample = samples[::3].float()
            projected = gait_features(sample, default_q, torch).flatten(1)[:, idx]
            design = ((projected - mean) / scale).double()
            targets = (
                sample.flatten(1).double() @ torch.tensor(init.layers[0][0], device=device).double()
            )
            targets += torch.as_tensor(init.layers[0][1], device=device)
            xbar, ybar = design.mean(0), targets.mean(0)
            xcenter = design - xbar
            ridge = torch.eye(design.shape[-1], device=device) * 8.0
            reduced_weight = torch.linalg.solve(
                xcenter.T @ xcenter + ridge, xcenter.T @ (targets - ybar)
            )
            reduced_bias = ybar - xbar @ reduced_weight
        student = nn.Sequential(
            nn.Linear(234, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 23),
        ).to(device)
        for i, (weight, bias) in enumerate(init.layers):
            weight, bias = (torch.as_tensor(a, device=device) for a in (weight, bias))
            if i == 0:
                if not init.raw:
                    weight = weight / torch.as_tensor(init.std[:, None], device=device)
                    bias = bias - torch.as_tensor(init.mean, device=device) @ weight
                if weight.shape[0] == 390:
                    weight = weight[idx]
                if len(init.mean) != 495:
                    bias = bias + mean @ weight
                    weight = scale[:, None] * weight
            if i == 0 and len(init.mean) == 495:
                weight, bias = reduced_weight.float(), reduced_bias.float()
            student[2 * i].weight.copy_(weight.T)
            student[2 * i].bias.copy_(bias)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    moving = torch.optim.swa_utils.AveragedModel(
        student, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.998)
    )
    counts = (
        args.floor_batch,
        96,
        args.ground_batch if ground or args.correction_actor or args.rehearse_init else 0,
        48,
        56,
        56,
    )
    old_count = sum(counts[:3])
    sigma = torch.as_tensor(
        [0.09] * 3
        + [0.011] * 6
        + [0.015] * 23
        + [0.27] * 23
        + [0.09] * 23
        + [0.008] * 18
        + [0.0] * 3,
        device=device,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    files = [
        args.init,
        args.old,
        args.gait,
        args.ground_teacher,
        args.correction_actor,
        *args.banks,
        *args.floor_replay,
    ]
    artifacts = {}
    for path in filter(None, files):
        with path.open("rb") as handle:
            name = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            artifacts[str(name)] = hashlib.file_digest(handle, "sha256").hexdigest()
    recipe = {
        "inputs_sha256": artifacts,
        "updates": args.updates,
        "learning_rate": args.lr,
        "seed": args.seed,
        "motion_train_noise": args.motion_train_noise,
        "sensor_teacher": args.sensor_teacher,
        "side_gain": args.side_gain,
        "groups": ["floor", "old_locomotion", "floor_assisted", "back", "side", "turn"],
        "samples_per_batch": counts,
        "floor_command_augmentation": args.ground_command_share,
        "rehearse_init": args.rehearse_init,
        "source_correction_banks": [str(p) for p in args.correction_banks],
        "torch": str(torch.__version__),
        "export_ema": 0.998,
        "input_width": 234,
        "output_order": "MJCF",
    }
    (args.out / "training.json").write_text(json.dumps(recipe, indent=2) + "\n")
    for iteration in range(1, args.updates + 1):
        with torch.no_grad():
            batch = torch.cat(
                [g[torch.randint(len(g), (n,), device=device)] for g, n in zip(groups, counts) if n]
            )
            frames = samples[batch].float()
            if not args.motion_train_noise:
                frames += (torch.randn_like(frames[:, :1]) + 0.4 * torch.randn_like(frames)) * sigma
            teacher_frames = gait_features(frames, default_q, torch)
            sensed = (
                motion_train_noisy_history(teacher_frames, torch)
                if args.motion_train_noise
                else teacher_frames
            )
            student_sensors = sensed
            if args.ground_command_share:
                # Change only the student's command. The teacher still supervises
                # the same measured hand-supported floor action under zero intent.
                student_sensors = with_ground_intents(
                    sensed, on_ground[batch], intents, args.ground_command_share, torch
                )
            inputs = (student_sensors.flatten(1)[:, idx] - mean) / scale
            if args.rehearse_init:
                labels = keep_policy(sensed.flatten(1)[:, idx])
            elif args.correction_actor:
                actor_obs = sensed.flatten(1)[:, idx]
                labels = torch.where(
                    corrections[batch, None], correct_policy(actor_obs), keep_policy(actor_obs)
                )
            else:
                privileged = frames
                if args.sensor_teacher:
                    # Teach the same local stabilizing feedback as the deployed teachers.
                    # Their older history and privileged body points remain available.
                    privileged = frames.clone()
                    privileged[:, -3:, :3] = sensed[:, -3:, :3]
                    privileged[:, -3:, 9:32] = sensed[:, -3:, 9:32] + default_q
                    privileged[:, -3:, 32:55] = sensed[:, -3:, 32:55]
                    teacher_frames = sensed.clone()
                turn_frames = teacher_frames[old_count:]
                recorded_side = side_labels[batch[old_count:]]
                turn_frames[:, :, 7] = torch.where(
                    torch.isfinite(recorded_side),
                    recorded_side,
                    turn_frames[:, :, 7] * args.side_gain,
                )
                turn_frames[:, :, 8] = teacher_yaw(turn_frames[:, :, 8], turn_frames[:, :, 2])
                old_labels = old(privileged[:old_count].flatten(1))
                if ground:
                    selection = slice(sum(counts[:2]), old_count)
                    alt = ground(privileged[selection, :, :96].flatten(1))
                    old_labels[selection] = torch.lerp(
                        old_labels[selection], alt, shares[batch[selection], None].float()
                    )
                labels = torch.cat((old_labels, gait(teacher_frames[old_count:].flatten(1))))
        loss = (student(inputs) - labels).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        optimizer.step()
        moving.update_parameters(student)
        if iteration % args.save_every == 0 or iteration == args.updates:
            fields = {
                "obs_norm/mean": np.zeros(234, np.float32),
                "obs_norm/var": np.ones(234, np.float32),
                "obs_norm/raw": True,
            }
            for i, layer in enumerate(moving.module[::2]):
                weight, bias = layer.weight.detach().T, layer.bias.detach()
                if i == 0:
                    weight = weight / scale[:, None]
                    bias = bias - mean @ weight
                for prefix in ("pi", "avg"):
                    fields[f"{prefix}/{i}/w"] = weight.cpu().numpy()
                    fields[f"{prefix}/{i}/b"] = bias.cpu().numpy()
            target = args.out / f"{iteration:06d}.npz"
            tmp = target.with_suffix(".tmp.npz")
            np.savez(tmp, **fields)
            tmp.replace(target)
            print(f"{target} MSE {loss.item():.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="operation", required=True)
    collect_parser = subs.add_parser("collect", help="record closed-loop demonstrations in MuJoCo")
    fit_parser = subs.add_parser(
        "fit", help="fit a 234-input actor to the physical demonstration banks"
    )
    for child in (collect_parser, fit_parser):
        child.add_argument(
            "--ground-teacher",
            type=Path,
            help="optional 480-input teacher for the initial floor phase",
        )
        child.add_argument(
            "--old",
            type=Path,
            default=RELEASE / "policy.npz",
            help="baseline recovery and commanded locomotion actor",
        )
        child.add_argument(
            "--gait",
            type=Path,
            default=RELEASE / "teachers/motion_train_14300.npz",
            help="converted 390-input Motion Train demonstrator",
        )
        child.add_argument(
            "--out",
            type=Path,
            required=True,
            help="output NPZ for collection or checkpoint directory for fitting",
        )
        child.add_argument(
            "--side-gain",
            type=float,
            default=1.0,
            help="lateral command multiplier for gait demonstration labels",
        )
        child.add_argument("--seed", type=int, default=41)
    collect_parser.add_argument(
        "--side-feedback",
        type=float,
        default=0.0,
        help="teacher gain on the measured lateral speed error",
    )
    collect_parser.add_argument(
        "--side-only",
        action="store_true",
        help="use just lateral commands following the quiet stance",
    )
    collect_parser.add_argument(
        "--after-rise", action="store_true", help="give one direction after 12s of quiet recovery"
    )
    collect_parser.add_argument(
        "--record-ground",
        action="store_true",
        help="save measured low-pelvis labels for floor-command augmentation",
    )
    collect_parser.add_argument(
        "--ground-weight",
        type=float,
        default=1.0,
        help="maximum ground demonstrator fraction before standing",
    )
    collect_parser.add_argument("--min-rise-seconds", type=float, default=0.0)
    collect_parser.add_argument("--require-palm", action="store_true")
    collect_parser.add_argument("--floor-only", action="store_true")
    collect_parser.add_argument("--success-only", action="store_true")
    collect_parser.add_argument(
        "--reference-jitter",
        action="store_true",
        help="perturb heading-relative tilt and joints while preserving clip velocities",
    )
    collect_parser.add_argument(
        "--quiet",
        action="store_true",
        help="select stable bilateral support after rising on the native model",
    )
    collect_parser.add_argument(
        "--source-jitter",
        action="store_true",
        help="perturb original stationary floor starts in tilt and horizontal velocity",
    )
    collect_parser.add_argument(
        "--floor-intent",
        action="store_true",
        help="expose the physical student to a sustained command from the first step",
    )
    collect_parser.add_argument(
        "--original-physics",
        action="store_true",
        help="use the original MJCF contact solver and immediate PD",
    )
    collect_parser.add_argument(
        "--floor-trials",
        type=int,
        nargs="+",
        help="indices from the floor reference bank, sampled in order",
    )
    collect_parser.add_argument("--episodes", type=int, default=60)
    collect_parser.add_argument(
        "--steps", type=int, default=525, help="maximum number of 50 Hz control steps per episode"
    )
    collect_parser.add_argument(
        "--student", type=Path, help="optional 234-input actor used to visit student states"
    )
    collect_parser.add_argument(
        "--corrupt-student",
        action="store_true",
        help="drive the student using shared Motion Train training sensors",
    )
    collect_parser.add_argument(
        "--share",
        type=float,
        default=0.6,
        help="fraction of the sampled student action used for physics",
    )
    fit_parser.add_argument(
        "banks", type=Path, nargs="+", help="demonstration NPZ files from the collect command"
    )
    fit_parser.add_argument(
        "--init", type=Path, help="495-input baseline or earlier 234-input student"
    )
    fit_parser.add_argument(
        "--rehearse-init",
        action="store_true",
        help="preserve the initial actor response under the original teacher command",
    )
    fit_parser.add_argument(
        "--correction-actor",
        type=Path,
        help="source-domain 234-input teacher used only for the selected banks",
    )
    fit_parser.add_argument(
        "--correction-banks",
        type=Path,
        nargs="+",
        default=[],
        help="banks for which the correction actor supplies the target actions",
    )
    fit_parser.add_argument(
        "--floor-replay",
        type=Path,
        nargs="*",
        default=[],
        help="optional recovery demonstrations in the legacy five-frame format",
    )
    fit_parser.add_argument(
        "--sensor-teacher",
        action="store_true",
        help="provide the student and labels the same noisy proprioception",
    )
    fit_parser.add_argument(
        "--motion-train-noise",
        action="store_true",
        help="use the original IMU-delay distribution and per-channel sensor noise",
    )
    fit_parser.add_argument(
        "--ground-batch",
        type=int,
        default=200,
        help="number of alternate ground-teacher examples per training update",
    )
    fit_parser.add_argument(
        "--ground-command-share",
        type=float,
        default=0.0,
        help="randomize student velocity commands in physically grounded zero-command examples",
    )
    fit_parser.add_argument(
        "--floor-batch",
        type=int,
        default=128,
        help="number of baseline floor examples per training update",
    )
    fit_parser.add_argument("--updates", type=int, default=6000)
    fit_parser.add_argument("--save-every", type=int, default=500)
    fit_parser.add_argument("--lr", type=float, default=0.0002)
    fit_parser.add_argument(
        "--device", default="cuda", help="PyTorch fitting device, such as cuda or cpu"
    )
    args = parser.parse_args()
    if args.operation == "collect":
        if args.episodes < 1 or args.steps < 1 or not 0.0 <= args.share <= 1.0:
            parser.error(
                "Provide a positive rollout duration and a student fraction from zero to one"
            )
        if (args.success_only or args.require_palm) and not args.floor_only:
            parser.error("Floor demonstration filtering requires --floor-only")
        if (
            not 0.0 <= args.ground_weight <= 1.0
            or args.min_rise_seconds < 0.0
            or args.side_feedback < 0.0
        ):
            parser.error(
                "Ground fraction, rise duration and side feedback must be valid nonnegative values"
            )
        if args.quiet and (not args.floor_only or args.original_physics):
            parser.error("Quiet bilateral filtering requires the native floor-only collection")
        if args.corrupt_student and not args.student:
            parser.error("--corrupt-student requires a 234-input student")
        if args.after_rise and args.steps <= 600:
            parser.error("Post-rise data collection requires more than 600 policy steps")
        collect(args)
    else:
        if not 0.0 <= args.ground_command_share <= 1.0:
            parser.error("Ground command probability must be between zero and one")
        if args.rehearse_init and (not args.init or args.correction_actor):
            parser.error("Initial policy rehearsal requires a single initial teacher")
        if args.correction_actor and (
            not args.init
            or not args.correction_banks
            or not set(args.correction_banks) <= set(args.banks)
        ):
            parser.error(
                "The paired teacher requires an initial actor and a nonempty subset of the provided banks"
            )
        if args.sensor_teacher and not args.motion_train_noise:
            parser.error("--sensor-teacher requires --motion-train-noise")
        if (
            min(args.updates, args.save_every, args.floor_batch, args.ground_batch) < 1
            or args.lr <= 0.0
        ):
            parser.error("Fit counts and learning rate must be positive")
        fit(args)


if __name__ == "__main__":
    main()
