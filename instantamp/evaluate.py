"""Evaluate an exported InstantAMP policy in its physical task.

Fixed-start recovery runs once from each selected pose for the requested duration.
G1 results include falls after first rise, continuous standing, foot support and pelvis drift.

  python -m instantamp.evaluate runs/g1_getup --last 1 --starts source-play --seconds 30 --trajectories runs/g1_getup/rollouts

Locomotion runs include training-style resets, pushes and shoves, and report velocity tracking and fall rates.

  python -m instantamp.evaluate runs/instantamp --n 1024 --seconds 20 --which avg --last 3
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from . import env, nets, zenbot
from .stability import stability
from .inference import load, policy_memory, policy_output, policy_step
from .rot import mv3, quat_to_mat, tr3
from .sim import Sim, robot_profile, unflatten


def fallen_starts(directory, lift=zenbot.RESET_LIFT, protocol="fallen"):
    """Fixed fallen poses or the frame-zero start of every source PLAY demonstration."""
    if protocol not in ("fallen", "source-play", "source-play-world"):
        raise ValueError(f"Unknown starts protocol {protocol!r}")
    names, poses, velocities = [], [], []
    for path in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        with np.load(path) as clip:
            for sec in ((0., 0.5) if protocol == "fallen" else (0.,)):
                i = round(sec * float(clip["fps"]))
                if i >= len(clip["root_pos"]) or (protocol == "fallen" and clip["root_pos"][i, 2] + lift > 0.5):
                    continue
                quat = clip["root_quat"][i, [3, 0, 1, 2]]
                ang = mv3(tr3(quat_to_mat(quat)), clip["root_ang_vel"][i])
                poses.append(np.r_[0., 0., clip["root_pos"][i, 2] + lift, quat, clip["dof_pos"][i]])
                velocities.append(np.r_[clip["root_lin_vel"][i], ang, clip["dof_vel"][i]])
                names.append((os.path.basename(path), sec))
    if not names:
        raise ValueError(f"No {protocol} evaluation starts in {directory}")
    return names, jnp.asarray(np.stack(poses), jnp.float32), jnp.asarray(np.stack(velocities), jnp.float32)


def evaluate_recovery(cfg, paths, args):
    robot = robot_profile(cfg["robot"])
    protocol = args.starts
    randomized = protocol == "source-play-world"
    seed = args.world_seed if randomized else 42
    if protocol != "fallen" and cfg["robot"] != "g1_getup":
        raise ValueError("Source PLAY starts apply to the named G1 get-up task")
    names, poses, velocities = fallen_starts(cfg["floor_dir"], robot.RESET_LIFT, protocol)
    sim = Sim(len(names), cfg["nconmax"], cfg["njmax"], floor_dir=cfg["floor_dir"], robot=cfg["robot"],
              random_start=False, assist_newtons=0., obs_noise=randomized, randomize_world=randomized)
    velocities = sim.reference_velocity(poses, velocities)
    reference = zenbot.RecoveryReference(cfg["floor_dir"], sim.key_positions, sim.J)
    clips_by_name = {path.name: i for i, path in enumerate(reference.files)}
    clips = jnp.asarray([clips_by_name[name] for name, _ in names])
    offsets = jnp.asarray([round(sec / env.DT) for _, sec in names])
    duration = 10.0 if args.seconds is None else args.seconds
    steps = round(duration / env.DT)
    if steps < round(1. / env.DT) or not np.isclose(steps * env.DT, duration):
        raise ValueError("Recovery --seconds must cover at least 1 second and be a multiple of 0.02 s")
    sim.episode_steps = steps + 1
    hold = round(1. / env.DT)
    pose_array = np.asarray(poses)
    initial_height = pose_array[:, 2] - robot.RESET_LIFT
    initial_tilt = np.degrees(np.arccos(np.clip(1. - 2. * (pose_array[:, 4:6]**2).sum(-1), -1., 1.)))
    binary = np.zeros(mujoco.mj_sizeModel(sim.m), dtype=np.uint8)
    mujoco.mj_saveModel(sim.m, None, binary)
    model_hash = hashlib.sha256(binary).hexdigest()
    capture = args.trajectories is not None
    trace_trial = args.trace_trial
    if trace_trial is not None and not 0 <= trace_trial < len(names):
        raise ValueError(f"--trace-trial must be between 0 and {len(names)-1}")
    if capture:
        args.trajectories.mkdir(parents=True, exist_ok=True)
        model_path = args.trajectories / f"physics_{model_hash[:16]}.mjb"
        model_path.write_bytes(binary.tobytes())

    key = jax.random.PRNGKey(seed)
    def run(pi, norm, ode):
        s = sim.init(key)
        data = s.data.replace(qpos=poses, qvel=velocities)
        initial = unflatten(poses, velocities)
        # Reuse the initial sensor-noise draw after replacing Sim.init's sampled pose.
        noise_key = jax.random.split(key, 5)[3] if randomized else None
        frame = sim.obs_frame(initial, s.a_last, s.c_obs, noise_key)
        weight_n = s.world.body_mass.sum(axis=-1) * 9.81 if randomized else sim.m.body_mass.sum() * 9.81
        up = sim.is_up(poses[:, 2], initial[1])
        critic_obs = sim.push_critic(None, initial, s.a_last) if s.critic_obs is not None else None
        s = s._replace(data=data, obs=sim.push_actor(None, frame), critic_obs=critic_obs, was_up=up, down=~up)
        def step(carry, frame_index):
            state, memory = carry
            out, memory = policy_step(pi, norm, state.obs, memory, args.obs_clip)
            act = nets.ode_step(out, state.a_ode, *ode) if ode is not None else out
            trace = None
            if trace_trial is not None:
                target = sim.control_targets(act, state.world.bias, state.obs, sim.context(state))
                trace = (state.obs[trace_trial], act[trace_trial], state.data.qvel[trace_trial], target[trace_trial])
            state, tr = sim.step(state, act, auto_reset=False)
            height = tr.qpos1[:, 2]
            upright = quat_to_mat(tr.qpos1[:, 3:7])[:, 2, 2] > np.cos(np.pi / 6)
            posture = (height > robot.UP_HEIGHT) & (height <= 2 * robot.RISE_HEIGHT - robot.UP_HEIGHT) & upright
            loaded = tr.stats["foot_load"] >= 0.5 * weight_n
            pose = sim.reference_pose(unflatten(tr.qpos1, tr.qvel1))
            errors, active = reference.errors(pose, clips, offsets + frame_index + 1)
            record = (tr.qpos1, tr.stats["foot_load"], tr.stats["support_load"], state.t, trace, memory if capture else None)
            return (state, memory), (posture, loaded, tr.terminated, errors, active, record)
        _, (posture, loaded, ended, errors, active, record) = jax.lax.scan(
            step, (s, policy_memory(pi, s.obs)), jnp.arange(steps))
        passed = posture[-hold:].all(0) & (loaded[-hold:].sum(0) >= np.ceil(0.9 * hold)) & ~ended.any(0)
        valid = active & (jnp.cumsum(ended, axis=0) == 0)
        count = valid.sum()
        def mean(x):
            return jnp.where(valid, x, 0.).sum() / jnp.maximum(count, 1)
        height, _, joints, points = errors
        metrics = passed, count, 100 * mean(jnp.sqrt(height)), jnp.sqrt(mean(joints)), 100 * jnp.sqrt(mean(points))
        world = (s.world.body_mass, s.world.geom_friction, s.obs, weight_n) if capture and randomized else None
        return metrics, (record, ended, world, weight_n)

    run_j = jax.jit(run, static_argnums=(2,))
    suffix = "" if protocol == "fallen" else "_source_play"
    if randomized:
        suffix += f"_world_seed{seed}"
    if not np.isclose(duration, 10.):
        suffix += f"_{duration:g}s"
    report_file = args.results or Path(args.run) / f"eval{suffix}.jsonl"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("w") as stream:
        for path in paths:
            with open(path, "rb") as policy_file:
                checkpoint_hash = hashlib.file_digest(policy_file, "sha256").hexdigest()
            metrics, recording = run_j(*load(path, args.which))
            passed, count, height_cm, joint_rad, points_cm = map(np.asarray, metrics)
            result = {"ckpt": os.path.basename(path), "checkpoint_sha256": checkpoint_hash, "assist_n": 0., "starts_protocol": protocol,
                      "world": "randomized" if randomized else "nominal", "actor_sensor_noise": randomized, "seed": seed,
                      "physics_sha256": model_hash, "duration_s": steps * env.DT,
                      "successes": int(passed.sum()), "total": len(passed), "getup_rate": float(passed.mean()),
                      "reference_frames": int(count), "reference_height_mae_cm": float(height_cm) if count else None,
                      "reference_joint_rmse_rad": float(joint_rad) if count else None,
                      "reference_landmark_rmse_cm": float(points_cm) if count else None,
                      "starts": [{"clip": name, "offset_s": sec, "success": bool(passed[i]),
                                  "source_height_m": float(initial_height[i]), "initial_height_m": float(pose_array[i, 2]),
                                  "initial_tilt_deg": float(initial_tilt[i]),
                                  "qualifies_fallen_height": bool(pose_array[i, 2] <= .5),
                                  "source_upright_posture": bool(robot.UP_HEIGHT < initial_height[i] <= 2 * robot.RISE_HEIGHT - robot.UP_HEIGHT
                                                                 and initial_tilt[i] < 30.)}
                                 for i, (name, sec) in enumerate(names)]}
            (qpos, foot_load, support_load, episode_step, trace, gate_weights), ended, world, weight_n = recording
            qpos = np.concatenate([pose_array[None], np.asarray(qpos)])
            ended, episode_step, weight_n = map(np.asarray, (ended, episode_step, weight_n))
            if not np.all(episode_step == np.arange(1, steps+1)[:,None]):
                raise RuntimeError("Recovery evaluation episode counter restarted")
            foot_bw = np.asarray(foot_load) / weight_n
            support_bw = np.asarray(support_load) / weight_n[...,None]
            result["automatic_resets"] = False
            if cfg["robot"] == "g1_getup":
                per_case = [stability(qpos[:,i], foot_bw[:,i], ended[:,i], env.DT,
                                      support_bw=support_bw[:,i], support_names=robot.SUPPORT_BODIES) for i in range(len(names))]
                for row, value in zip(result["starts"], per_case):
                    row["standing_stability"] = value
                result["final_3s_supported_upright"] = sum(v["final_3s_supported_upright"] for v in per_case)
                result["trials_falling_after_first_rise"] = sum(bool(v["post_rise_falls"]) for v in per_case)
            if capture:
                extra = {}
                if world is not None:
                    masses, friction, first_obs, weight_n = map(np.asarray, world)
                    extra = dict(body_mass_kg=masses, geom_friction=friction, initial_actor_obs=first_obs,
                                 body_weight_n=weight_n, foot_load_n=np.asarray(foot_load))
                if gate_weights is not None:
                    history = np.asarray(gate_weights)
                    extra["controller_gate_weights"] = history[..., :2]
                    if history.shape[-1] == 3:
                        extra["stance_confirmed_steps"] = history[..., 2].astype(np.int32)
                if trace is not None:
                    actor_obs, action, qvel_before, joint_target = map(np.asarray, trace)
                    extra.update(trace_trial=trace_trial, actor_obs=actor_obs, action=action,
                                 pre_step_qvel=qvel_before, joint_target=joint_target)
                archive = args.trajectories / f"{Path(path).stem}_{args.which}{suffix}_recovery.npz"
                np.savez_compressed(archive, qpos=qpos, foot_load_bw=foot_bw, support_load_bw=support_bw,
                                    support_body_names=np.array(robot.SUPPORT_BODIES), episode_step=episode_step,
                                    duration_s=duration, automatic_resets=False,
                                    terminated=np.asarray(ended), success=passed,
                                    clips=np.array([name for name, _ in names]), offsets_s=np.array([sec for _, sec in names]),
                                    dt=env.DT, assist_n=0., robot=cfg["robot"], checkpoint=Path(path).name, which=args.which,
                                    starts_protocol=protocol, physics_sha256=model_hash, physics_file=model_path.name,
                                    checkpoint_sha256=checkpoint_hash, world="randomized" if randomized else "nominal",
                                    actor_sensor_noise=randomized, seed=seed, **extra)
                result["trajectories"] = str(archive)
            stream.write(json.dumps(result) + "\n")
            print(f"{result['ckpt']} | unassisted get-up {passed.sum()}/{len(passed)}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--seconds", type=float, help="physical rollout duration (default 10 s for recovery, 20 s otherwise)")
    p.add_argument("--which", default="avg")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--last", type=int, default=0, help="only the last k checkpoints (0: all)")
    selection.add_argument("--checkpoint", help="evaluate a single exported filename in this run's ckpt directory")
    p.add_argument("--obs_clip", type=float, default=10.0)
    p.add_argument("--trajectories", type=Path, metavar="DIR",
                   help="save each fixed recovery trial's initial pose, post-step poses, contact loads and episode-step counters")
    p.add_argument("--trace-trial", type=int, help="also save this trial's pre-step actor observations, applied actions, velocities and joint targets")
    p.add_argument("--starts", choices=("fallen", "source-play", "source-play-world"), default="fallen",
                   help="fallen and source-play use nominal physics and clean sensors; source-play-world adds "
                        "randomized materials, torso mass and actor sensor noise to the same frame-zero PLAY starts")
    p.add_argument("--world-seed", type=int, default=42, help="seed for --starts source-play-world")
    p.add_argument("--results", type=Path, help="write evaluation JSONL here instead of the protocol default in the run")
    a = p.parse_args(argv)
    if a.seconds is not None and (not np.isfinite(a.seconds) or a.seconds <= 0):
        p.error("--seconds must be finite and positive")
    if a.trace_trial is not None and a.trajectories is None:
        p.error("--trace-trial requires --trajectories")
    if a.world_seed != 42 and a.starts != "source-play-world":
        p.error("--world-seed requires --starts source-play-world")
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    ckpts = sorted(glob.glob(os.path.join(a.run, "ckpt", "*.npz")))
    if a.checkpoint is not None:
        if Path(a.checkpoint).name != a.checkpoint:
            p.error("--checkpoint expects a filename in this run's ckpt directory")
        ckpts = [p for p in ckpts if Path(p).name == a.checkpoint]
    if a.last:
        ckpts = ckpts[-a.last :]

    if not ckpts:
        raise ValueError(f"No policy checkpoints in {a.run}")
    if cfg.get("robot") in ("zenbot23_getup", "g1_getup"):
        return evaluate_recovery(cfg, ckpts, a)
    if a.seconds is None:
        a.seconds = 20.0
    if a.starts != "fallen":
        p.error("--starts source-play requires the named G1 get-up task")
    if a.trajectories is not None:
        p.error("--trajectories is available for fixed-start recovery evaluations")
    sim = Sim(a.n, cfg.get("nconmax", 64), cfg.get("njmax", 640), cfg["walk_dir"], cfg["floor_dir"], True, 123, cfg["reset_floor"], cfg["prob_shove"],
              bool(cfg["push"]), robot=cfg.get("robot", "g1"), g1_style=bool(cfg.get("g1_style", 0)),
              getup_rise_weight=cfg.get("getup_rise_weight", -0.25),
              g1_support_friction=bool(cfg.get("g1_support_friction", 0)), getup_reset_frames=cfg.get("getup_reset_frames", 20),
              g1_phase_amp=bool(cfg.get("g1_phase_share", 0.)))  # fmt: skip

    def run(pi, norm, ode, key):
        def body(carry, k):
            s, acc = carry
            out = policy_output(pi, norm, s.obs, a.obs_clip)
            act = nets.ode_step(out, s.a_ode, *ode) if ode is not None else out
            s1, tr = sim.step(s, act)
            S1 = unflatten(tr.qpos1, tr.qvel1)
            yaw = env.heading(S1[1])
            v = S1[2]
            vb = jnp.stack((jnp.cos(yaw) * v[:, 0] + jnp.sin(yaw) * v[:, 1], -jnp.sin(yaw) * v[:, 0] + jnp.cos(yaw) * v[:, 1]), -1)
            upright = s.was_up & sim.is_up(S1[0][:, 2], S1[1])
            err = jnp.linalg.norm(vb - s.c_obs[:, :2], axis=-1)
            fin = (tr.terminated | tr.truncated).astype(jnp.float32)
            up_ep, down_ep = fin * ~s.down, fin * s.down
            rose = s.was_up | sim.is_up(S1[0][:, 2], S1[1])
            acc = {"parts": acc["parts"] + tr.parts.sum(0), "steps": acc["steps"] + a.n,
                   "ep_up": acc["ep_up"] + up_ep.sum(), "fall_up": acc["fall_up"] + (up_ep * tr.terminated).sum(),
                   "ep_down": acc["ep_down"] + down_ep.sum(), "rose": acc["rose"] + (down_ep * rose).sum(),
                   "err": acc["err"] + (err * upright).sum(), "n_err": acc["n_err"] + upright.sum()}  # fmt: skip
            return (s1, acc), None

        s = sim.init(key)
        z = jnp.zeros(())
        acc = {"parts": jnp.zeros((len(env.TERMS),)), "steps": z, "ep_up": z, "fall_up": z, "ep_down": z, "rose": z, "err": z, "n_err": z}
        (_, acc), _ = jax.lax.scan(body, (s, acc), jax.random.split(key, int(a.seconds / env.DT)))
        return acc

    run_j = jax.jit(run, static_argnums=(2,))
    results = a.results or Path(a.run) / "eval.jsonl"
    results.parent.mkdir(parents=True, exist_ok=True)
    out = results.open("w")
    for path in ckpts:
        pi, norm, ode = load(path, a.which)
        acc = jax.device_get(run_j(pi, norm, ode, jax.random.PRNGKey(0)))
        per_s = acc["parts"] / (acc["steps"] * env.DT)  # weighted terms per env-second
        rec = {"ckpt": os.path.basename(path), **json.load(open(path[:-4] + ".json")),
               **{f"rew_{k}": float(v) for k, v in zip(env.TERMS, per_s)},
               "fall_rate": float(acc["fall_up"] / max(acc["ep_up"], 1)), "getup_rate": float(acc["rose"] / max(acc["ep_down"], 1)),
               "down_episodes": float(acc["ep_down"]), "vel_err": float(acc["err"] / max(acc["n_err"], 1))}  # fmt: skip
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"{rec['ckpt']} it {rec['it']} | lin {rec['rew_track_lin']:.3f} ang {rec['rew_track_ang']:.3f} | |v-cmd| {rec['vel_err']:.3f} m/s | "
              f"falls {rec['fall_rate']:.3f} of upright starts | get-up {rec['getup_rate']:.3f} of {rec['down_episodes']:.0f} down starts", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
