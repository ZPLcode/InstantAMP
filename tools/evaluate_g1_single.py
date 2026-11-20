"""Evaluate one autonomous G1 checkpoint from early, physically fallen starts in a single native clip.

All methods share deterministic resets, nominal native MuJoCo dynamics and the unchanged 50 Hz
reference clock. The actor gets only the standard four-frame proprioceptive/command history.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from instantamp import env, nets
from instantamp.inference import load, policy_output
from instantamp.sim import DIVERGED, build_scene, unflatten
from tools.evaluate_g1_style import actor_spec
from tools.score_g1_tracking import (_body_local, _landmarks_and_clearance, _pose_metrics, _up,
                                     load_native_reference)

STEPS = env.GETUP_STEPS + round(1 / env.DT)
FINAL_HOLD = round(1 / env.DT)
OFFSETS = (0, 3, 6)
SPEED_OFFSETS = (0., -.05, .05)


def fallen_cases(poses, exclusive_end=None):
    """Retain only the initial low, tilted posture before it starts to rise."""
    rzz = 1 - 2 * np.sum(poses[:, 4:6]**2, axis=1)
    fallen = (poses[:, 2] <= min(.45, poses[0, 2] + .03)) & (rzz <= np.cos(np.deg2rad(45.)))
    stop = int(np.flatnonzero(~fallen)[0]) if np.any(~fallen) else len(fallen)
    if exclusive_end is not None:
        if exclusive_end <= 0:
            raise ValueError("fallen-prefix-end must be positive")
        stop = min(stop, exclusive_end)
    offsets = [t for t in OFFSETS if t < stop]
    if not offsets:
        raise ValueError("The first frame is not an eligible low, tilted fallen pose")
    return [(t, vx) for t in offsets for vx in SPEED_OFFSETS]


def release_time(low):
    """Final release of a hand that was near for >=3 frames, with >=5 clear frames after it."""
    groups = np.flatnonzero(np.diff(np.r_[False, low, False].astype(int))).reshape(-1, 2)
    sustained = [(a, b) for a, b in groups if b-a >= 3]
    if not sustained:
        return None, "no_sustained_approach"
    cleared = sustained[-1][1]
    if len(low)-cleared < 5 or np.any(low[cleared:cleared+5]):
        return None, "no_confirmed_release"
    return float(cleared*env.DT), "released"


def contacts(model, data, foot_geoms, term_bodies, floor):
    foot = 0.
    touch = False
    force = np.empty(6)
    for i, item in enumerate(data.contact):
        if item.geom1 == floor:
            geom = item.geom2
        elif item.geom2 == floor:
            geom = item.geom1
        else:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        if geom in foot_geoms:
            foot += force[0]
        if model.geom_bodyid[geom] in term_bodies and force[0] > env.CONTACT_FORCE:
            touch = True
    return foot / (model.body_mass.sum() * 9.81), touch


def rollout(model, checkpoint, starts, velocities):
    pi, norm, ode = load(checkpoint)
    if actor_spec(pi, norm)["type"] != "single_actor":
        raise ValueError("A single G1 384→29 actor export is required")
    zero_action = jnp.zeros((len(starts), env.J))
    zero_command = jnp.zeros((len(starts), 3))
    obs_frame = jax.jit(lambda q, v, a: env.obs_frame(unflatten(q, v), a, zero_command))
    @jax.jit
    def infer(h, old):
        a = policy_output(pi, norm, h, 10.)
        return a if ode is None else nets.ode_step(a, old, *ode)

    data = [mujoco.MjData(model) for _ in starts]
    for d, q, v in zip(data, starts, velocities):
        d.qpos[:] = q
        d.qvel[:] = v
        mujoco.mj_forward(model, d)
    foot_geoms = {g for g in range(model.ngeom) if model.geom(g).name.startswith(("robot_left_foot", "robot_right_foot"))
                  and model.geom_contype[g]}
    if len(foot_geoms) != env.N_FOOT_GEOMS:
        raise ValueError("Unexpected set of physical G1 sole collision geometries")
    term = {model.body("robot_"+name).id for name in env.TERMINATION_BODIES}
    floor = model.geom("floor").id
    q, v = starts.copy(), velocities.copy()
    history = jnp.tile(obs_frame(q, v, zero_action), (1, env.HIST))
    action = zero_action
    qlog = np.broadcast_to(q[:, None], (len(q), STEPS+1, model.nq)).copy()
    forces = np.zeros((len(q), STEPS))
    done = np.zeros_like(forces, dtype=bool)
    elapsed = np.full(len(q), STEPS, int)
    seen_up = np.zeros(len(q), bool)
    reasons = ["time_limit"]*len(q)
    for t in range(STEPS):
        action = infer(history, action)
        targets = env.joint_target(np.asarray(action), 0.)
        for i, d in enumerate(data):
            if t >= elapsed[i]:
                qlog[i, t+1] = q[i]
                continue
            for _ in range(env.DECIMATION):
                d.ctrl[:] = env.actuator_ctrl(d.qpos[7:], d.qvel[6:], targets[i], xp=np)
                mujoco.mj_step(model, d)
            f, touched = contacts(model, d, foot_geoms, term, floor)
            diverged = not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()) or np.max(np.abs(d.qvel)) > DIVERGED
            if diverged:
                reason = "diverged"
            else:
                q[i], v[i] = d.qpos, d.qvel
                up = bool(_up(q[i:i+1])[0])
                reason = ("post_rise_contact" if seen_up[i] and touched else
                          "missed_5s_deadline" if not seen_up[i] and t+1 >= env.GETUP_STEPS and not up else None)
                seen_up[i] |= up
            qlog[i, t+1], forces[i, t] = q[i], f
            if reason:
                done[i, t], elapsed[i], reasons[i] = True, t+1, reason
        history = env.push_frame(history, obs_frame(q, v, action))
    return qlog, forces, done, reasons


def event_pair(reference, actual, release):
    if not release:
        a = np.flatnonzero(reference)
        b = np.flatnonzero(actual)
        rt, st = (float(a[0]*env.DT) if len(a) else None), (float(b[0]*env.DT) if len(b) else None)
        rr, sr = ("rose" if len(a) else "absent"), ("rose" if len(b) else "absent")
    else:
        rt, rr = release_time(reference)
        st, sr = release_time(actual)
    return {"reference_s": rt, "policy_s": st, "reference_status": rr, "policy_status": sr,
            "abs_error_s": None if rt is None or st is None else abs(st-rt)}


def evaluate(args):
    raw, velocity = load_native_reference(args.clip)
    starts = fallen_cases(raw, args.fallen_prefix_end)
    model = build_scene(g1_support_friction=args.support_friction)
    initial_q, initial_v, refs = [], [], []
    for t, dx in starts:
        ref = raw[t:].copy()
        ref[:, :2] -= ref[0, :2].copy()
        q, v = ref[0].copy(), velocity[t].copy()
        q[2] += .02
        v[0] += dx
        initial_q.append(q)
        initial_v.append(v)
        refs.append(ref)
    qlog, forces, ended, reasons = rollout(model, args.checkpoint, np.stack(initial_q), np.stack(initial_v))
    archive = {"qpos": qlog, "ended": ended, "foot_load_bw": forces,
               "reference": np.stack([ref[np.minimum(np.arange(STEPS+1), len(ref)-1)] for ref in refs])}
    fields = ("pelvis_height_abs_cm", "pelvis_xy_cm", "joint_rms_rad", "wrist_world_cm", "wrist_body_cm")
    per_frame = np.full((len(starts), STEPS+1, len(fields)), np.nan)
    rows = []
    for i, ((offset, dx), ref, q, f, terminal) in enumerate(zip(starts, refs, qlog, forces, ended)):
        stop = np.flatnonzero(terminal)
        # Scoring freezes the last valid pre-terminal pose and excludes subsequent reset trajectories.
        valid = min(int(stop[0])+1, len(q)) if len(stop) else len(q)
        measured = q[:valid]
        rr, sr = _up(ref), _up(measured)
        rise = event_pair(rr, sr[:env.GETUP_STEPS+1], False)
        if rise["reference_s"] is None:
            raise ValueError("The reference did not reach the standard G1 up threshold")
        n = min(len(ref), round((rise["reference_s"]+.5)/env.DT)+1, STEPS+1)
        if n < 2:
            raise ValueError("The reference must cover at least two frames")
        actual = measured[np.minimum(np.arange(n), len(measured)-1)]
        rp, rh = _landmarks_and_clearance(model, ref[:n])
        ap, _ = _landmarks_and_clearance(model, actual)
        actual_stop = min(valid, round((rise["policy_s"]+.5)/env.DT)+1) if rise["policy_s"] is not None else valid
        _, ah = _landmarks_and_clearance(model, measured[:actual_stop])
        release = {}
        for col, side in ((2, "left"), (3, "right")):
            release[side] = event_pair(rh[:, col] <= .02, ah[:, col] <= .02, True)
        metrics = _pose_metrics(ref[:n], actual, rp, ap)
        delta = actual-ref[:n]
        world_wrist = ap[:, 2:4]-rp[:, 2:4]
        local_wrist = _body_local(ap, actual)[:, 2:4]-_body_local(rp, ref[:n])[:, 2:4]
        rmse_points = lambda x: 100*np.sqrt(np.mean(np.sum(x**2, axis=-1), axis=-1))
        per_frame[i, :n] = np.column_stack((100*abs(delta[:, 2]), 100*np.linalg.norm(delta[:, :2], axis=1),
                                             np.sqrt(np.mean(delta[:, 7:]**2, axis=1)),
                                             rmse_points(world_wrist), rmse_points(local_wrist)))
        p95 = {k: float(v) for k,v in zip(fields, np.percentile(per_frame[i,:n],95,axis=0))}
        loaded = (not terminal.any() and sr[-FINAL_HOLD:].all() and (f[-FINAL_HOLD:] >= .5).mean() >= .9)
        rows.append({"source_frame": offset, "source_seconds": offset*env.DT, "initial_world_vx_perturbation_m_s": dx,
                     "reference_frames": len(ref), "score_frames": n, "valid_policy_frames": valid,
                     "end_reason": reasons[i], "loaded_hold": bool(loaded),
                     "final_mean_foot_load_bw": float(f[-FINAL_HOLD:].mean()) if not terminal.any() else None,
                     "rise": rise, "hand_release": release, "rise_and_settle": metrics, "frame_error_p95": p95})
    summary = {"trials": len(rows), "rose": sum(r["rise"]["policy_s"] is not None for r in rows),
               "loaded_hold": sum(r["loaded_hold"] for r in rows), "canonical_trial": 0}
    for event in ("rise", "left", "right"):
        vals = [r["rise"] if event == "rise" else r["hand_release"][event] for r in rows]
        errors = [v["abs_error_s"] for v in vals if v["abs_error_s"] is not None]
        summary[event] = {"matched": len(errors), "missing_reference": sum(v["reference_s"] is None for v in vals),
                          "missing_policy": sum(v["policy_s"] is None for v in vals),
                          "median_abs_error_s": float(np.median(errors)) if errors else None}
    for key in ("rise_and_settle", "frame_error_p95"):
        summary[key] = {k: float(np.median([r[key][k] for r in rows])) for k in rows[0][key]}
    protocol = {"clip": str(args.clip.resolve()), "clip_sha256": hashlib.sha256(args.clip.read_bytes()).hexdigest(),
                "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "actor": "single 384→29", "backend": "native CPU MuJoCo", "support_friction": args.support_friction,
                "assist_n": 0, "policy_hz": 50, "physics_hz": 200, "reset_z_lift_m": .02, "duration_s": STEPS*env.DT,
                "fallen_prefix_end": args.fallen_prefix_end,
                "fallen_starts": "First 0/3/6 frames within contiguous prefix, z<=min(0.45m,z0+0.03m), tilt>=45deg",
                "hand_release": "Hand collision geometry within 2cm for >=3 frames, then final clear streak >=5 frames in own rise+0.5s window",
                "style_window": "reference onset to first rise + 0.5s, clipped at source end, no retiming",
                "loaded_hold": "upright throughout final 1s, at least 0.5 body weight on soles in >=90% of that second"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    archive.update(frame_errors=per_frame, frame_error_fields=np.array(fields))
    np.savez_compressed(args.out.with_suffix(".npz"), **archive)
    args.out.write_text(json.dumps({"protocol": protocol, "summary": summary, "trials": rows}, indent=2, allow_nan=False)+"\n")
    print(f"{len(starts)} trials | rose {summary['rose']} | loaded hold {summary['loaded_hold']} | {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fallen-prefix-end", type=int, help="optional exclusive source frame marking start of lift")
    parser.add_argument("--support-friction", action="store_true")
    evaluate(parser.parse_args())
