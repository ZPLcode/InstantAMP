"""Evaluate InstantAMP checkpoints (ckpt/NNNN.npz of train.py) in the simulator: the averaged policy without noise, `n` envs, `seconds`
of simulation (20-s episodes; resets, pushes and shoves as in training). Per checkpoint: weighted reward terms per env-second, the share of episodes that
start up and end in a fall, the share of episodes that start down in which the robot rises, and the mean |v_xy - cmd_xy| while up.

  python -m instantamp.evaluate runs/instantamp [--n 1024 --seconds 20 --which avg --last 3]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import env, nets, zenbot
from .rot import mv3, quat_to_mat, tr3
from .sim import Sim, unflatten


def load(path, which="avg"):
    z = np.load(path)
    layers, i = [], 0
    while f"{which}/{i}/w" in z.files:
        layers.append({k.split("/")[-1]: jnp.asarray(z[k]) for k in z.files if k.startswith(f"{which}/{i}/")})
        i += 1
    norm = {"mean": jnp.asarray(z["obs_norm/mean"]), "var": jnp.asarray(z["obs_norm/var"])}
    ode = (float(z["ode/dmax"]) or None, float(z["ode/amax"]) or None) if "ode/dmax" in z.files else None
    return layers, norm, ode


def fallen_starts(directory):
    """Repeatable legged_lab evaluation poses at 0 and 0.5 s, below 0.5 m including the lift."""
    names, poses, velocities = [], [], []
    for path in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        with np.load(path) as clip:
            for sec in (0., 0.5):
                i = round(sec * float(clip["fps"]))
                if i >= len(clip["root_pos"]) or clip["root_pos"][i, 2] + 0.02 > 0.5:
                    continue
                quat = clip["root_quat"][i, [3, 0, 1, 2]]
                ang = mv3(tr3(quat_to_mat(quat)), clip["root_ang_vel"][i])
                poses.append(np.r_[0., 0., clip["root_pos"][i, 2] + 0.02, quat, clip["dof_pos"][i]])
                velocities.append(np.r_[clip["root_lin_vel"][i], ang, clip["dof_vel"][i]])
                names.append((os.path.basename(path), sec))
    if not names:
        raise ValueError(f"No fallen evaluation starts in {directory}")
    return names, jnp.asarray(np.stack(poses), jnp.float32), jnp.asarray(np.stack(velocities), jnp.float32)


def evaluate_recovery(cfg, paths, args):
    names, poses, velocities = fallen_starts(cfg["floor_dir"])
    sim = Sim(len(names), cfg["nconmax"], cfg["njmax"], floor_dir=cfg["floor_dir"], robot=cfg["robot"],
              random_start=False, assist_newtons=0., obs_noise=False, randomize_world=False)
    reference = zenbot.RecoveryReference(cfg["floor_dir"], sim.key_positions)
    clips_by_name = {path.name: i for i, path in enumerate(reference.files)}
    clips = jnp.asarray([clips_by_name[name] for name, _ in names])
    offsets = jnp.asarray([round(sec / env.DT) for _, sec in names])
    steps = sim.episode_steps
    hold = round(1. / env.DT)

    def run(pi, norm, ode):
        s = sim.init(jax.random.PRNGKey(42))
        data = s.data.replace(qpos=poses, qvel=velocities)
        frame = sim.obs_frame(unflatten(poses, velocities), s.a_last, s.c_obs, None)
        s = s._replace(data=data, obs=jnp.tile(frame, (1, sim.history)))
        def step(state, frame_index):
            out = nets.pi_apply(pi, jnp.clip(nets.rn_apply(norm, state.obs), -args.obs_clip, args.obs_clip))
            act = nets.ode_step(out, state.a_ode, *ode) if ode is not None else out
            state, tr = sim.step(state, act)
            height = tr.qpos1[:, 2]
            upright = quat_to_mat(tr.qpos1[:, 3:7])[:, 2, 2] > np.cos(np.pi / 6)
            posture = (height > zenbot.UP_HEIGHT) & (height <= 2 * zenbot.RISE_HEIGHT - zenbot.UP_HEIGHT) & upright
            loaded = tr.stats["foot_load"] >= 0.5 * zenbot.MASS * 9.81
            pose = sim.reference_pose(unflatten(tr.qpos1, tr.qvel1))
            errors, active = reference.errors(pose, clips, offsets + frame_index + 1)
            return state, (posture, loaded, tr.terminated, errors, active)
        _, (posture, loaded, ended, errors, active) = jax.lax.scan(step, s, jnp.arange(steps))
        passed = posture[-hold:].all(0) & (loaded[-hold:].sum(0) >= np.ceil(0.9 * hold)) & ~ended.any(0)
        valid = active & (jnp.cumsum(ended, axis=0) == 0)
        count = valid.sum()
        mean = lambda x: jnp.where(valid, x, 0.).sum() / jnp.maximum(count, 1)
        height, _, joints, points = errors
        return passed, count, 100 * mean(jnp.sqrt(height)), jnp.sqrt(mean(joints)), 100 * jnp.sqrt(mean(points))

    run_j = jax.jit(run, static_argnums=(2,))
    with open(os.path.join(args.run, "eval.jsonl"), "w") as stream:
        for path in paths:
            passed, count, height_cm, joint_rad, points_cm = map(np.asarray, run_j(*load(path, args.which)))
            result = {"ckpt": os.path.basename(path), "assist_n": 0., "duration_s": steps * env.DT,
                      "successes": int(passed.sum()), "total": len(passed), "getup_rate": float(passed.mean()),
                      "reference_frames": int(count), "reference_height_mae_cm": float(height_cm) if count else None,
                      "reference_joint_rmse_rad": float(joint_rad) if count else None,
                      "reference_landmark_rmse_cm": float(points_cm) if count else None,
                      "starts": [{"clip": name, "offset_s": sec, "success": bool(ok)} for (name, sec), ok in zip(names, passed)]}
            stream.write(json.dumps(result) + "\n")
            print(f"{result['ckpt']} | unassisted get-up {passed.sum()}/{len(passed)}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--which", default="avg")
    p.add_argument("--last", type=int, default=0, help="only the last k checkpoints (0: all)")
    p.add_argument("--obs_clip", type=float, default=10.0)
    a = p.parse_args(argv)
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    ckpts = sorted(glob.glob(os.path.join(a.run, "ckpt", "*.npz")))
    if a.last:
        ckpts = ckpts[-a.last :]

    if not ckpts:
        raise ValueError(f"No policy checkpoints in {a.run}")
    if cfg.get("robot") == "zenbot23_getup":
        return evaluate_recovery(cfg, ckpts, a)
    sim = Sim(a.n, cfg.get("nconmax", 64), cfg.get("njmax", 640), cfg["walk_dir"], cfg["floor_dir"], True, 123, cfg["reset_floor"], cfg["prob_shove"],
              bool(cfg["push"]), robot=cfg.get("robot", "g1"))  # fmt: skip

    def run(pi, norm, ode, key):
        def body(carry, k):
            s, acc = carry
            o = jnp.clip(nets.rn_apply(norm, s.obs), -a.obs_clip, a.obs_clip)
            out = nets.pi_apply(pi, o)
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
    out = open(os.path.join(a.run, "eval.jsonl"), "w")
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
