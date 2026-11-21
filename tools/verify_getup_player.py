"""Record and score the native G1 interactive player's physical rollout.

    python -m tools.verify_getup_player record --out runs/native_check
    python -m tools.verify_getup_player score runs/native_check/source-play.npz

Each archive includes measured contacts and velocities and can also be rendered with
tools.compare_recovery_rollouts. All times are seconds of simulated physics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import mujoco
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

from instantamp import env, g1_getup
from instantamp.evaluate import fallen_starts
from instantamp.stability import first_sustained, stability
from tools.compare_recovery_rollouts import file_hash, first_upright, landmark_fk, rms
from tools.play_getup import GetupPlayer, RELEASE, ROOT
from tools.score_g1_tracking import _body_local, load_native_reference

MOTIONS = ROOT / "data/g1_getup/recovery"


def contact_loads(model, data, floor, support, weight):
    """Normal forces between each named support body and the floor, in body weights."""
    loads = np.zeros(len(g1_getup.SUPPORT_BODIES))
    wrench = np.zeros(6)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        a, b = contact.geom
        other = b if a == floor else a if b == floor else -1
        if other in support and contact.efc_address >= 0:
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            loads[support[other]] += wrench[0]
    return loads / weight


def jitter_start(q, v, seed, clip, offset):
    tag = int.from_bytes(hashlib.sha256(f"{clip}|{offset}".encode()).digest()[:4], "little")
    rng = np.random.default_rng([seed, tag])
    q[7:] += np.deg2rad(rng.uniform(-2, 2, 29))
    local = Rotation.from_euler("xy", rng.uniform(-2, 2, 2), degrees=True)
    q[3:7] = (Rotation.from_quat(q[3:7], scalar_first=True) * local).as_quat(scalar_first=True)
    v[:2] += rng.uniform(-.07, .07, 2)


def record(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    player = GetupPlayer(args.checkpoint)
    model, data = player.model, player.data
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    model_hash = hashlib.sha256(binary).hexdigest()
    model_file = out / f"physics_{model_hash[:16]}.mjb"
    if not model_file.exists():
        model_file.write_bytes(binary.tobytes())
    floor = model.geom("floor").id
    torso = model.body("robot_torso_link").id
    support_bodies = [model.body("robot_" + body).id for body in g1_getup.SUPPORT_BODIES]
    support = {i: support_bodies.index(body) for i, body in enumerate(model.geom_bodyid) if body in support_bodies}
    weight = float(model.body_mass.sum() * 9.81)
    from instantamp.g1_ground_phase import HandContacts
    hands = HandContacts(model)
    frames = round(args.seconds / env.DT)
    checkpoint_hash = file_hash(args.checkpoint)
    known = {}
    for protocol in args.protocol:
        names, q, v = fallen_starts(MOTIONS, g1_getup.RESET_LIFT, protocol)
        player.names, player.poses, player.velocities = names, np.array(q, float), np.array(v, float)
        trials = args.trials if args.trials is not None else range(len(names))
        clips, offsets, recordings = [], [], []
        for trial in trials:
            if trial >= len(names):
                raise ValueError(f"{protocol} trial {trial} is outside the range 0..{len(names)-1}")
            clip, offset = names[trial]
            key = (clip, offset)
            if key in known:
                clips.append(clip)
                offsets.append(offset)
                recordings.append(known[key])
                continue
            if args.jitter_seed is not None:
                jitter_start(player.poses[trial], player.velocities[trial], args.jitter_seed, clip, offset)
            player.reset(trial)
            trajectory = dict(qpos=[data.qpos.copy()], qvel=[data.qvel.copy()], support_load_bw=[],
                              feet_xyz=[], torso_tilt_deg=[], normalized_action=[], controller_gate_weights=[], hand_load_bw=[])
            for step in range(frames):
                if args.push_at is not None and step == round(args.push_at / env.DT):
                    player.push(args.push_angle)
                player.step()
                trajectory["qpos"].append(data.qpos.copy())
                trajectory["qvel"].append(data.qvel.copy())
                trajectory["support_load_bw"].append(contact_loads(model, data, floor, support, weight))
                trajectory["hand_load_bw"].append(hands.forces(data))
                trajectory["feet_xyz"].append(data.xpos[player.links[:2]].copy())
                trajectory["torso_tilt_deg"].append(np.degrees(np.arccos(np.clip(data.xmat[torso, 8], -1, 1))))
                trajectory["normalized_action"].append(player.previous.copy())
                if player.memory is not None:
                    trajectory["controller_gate_weights"].append(np.asarray(player.memory[0]))
            if not np.isfinite(trajectory["qpos"]).all() or np.any(data.warning.number[4:]):
                raise RuntimeError(f"MuJoCo instability in {protocol} trial {trial}")
            clips.append(clip)
            offsets.append(offset)
            recordings.append(trajectory)
            known[key] = trajectory
            print(f"{protocol} trial {trial} complete", flush=True)
        arrays = {name: np.stack([r[name] for r in recordings], axis=1)
                  for name in recordings[0] if len(recordings[0][name])}
        foot_load = arrays["support_load_bw"][:, :, :2].sum(-1)
        terminated = np.zeros(foot_load.shape, bool)
        success = [stability(arrays["qpos"][:, j], foot_load[:, j], terminated[:, j], env.DT)
                   ["final_3s_supported_upright"] for j in range(len(clips))]
        archive = out / f"{protocol}.npz"
        np.savez_compressed(archive, **arrays, qpos_includes_reset=True, dt=env.DT, success=success,
                            terminated=terminated, foot_load_bw=foot_load, support_body_names=g1_getup.SUPPORT_BODIES,
                            clips=clips, source_sha256=[file_hash(MOTIONS/clip) for clip in clips],
                            offsets_s=offsets, trials=list(trials), duration_s=frames*env.DT,
                            checkpoint=args.checkpoint.name, checkpoint_path=str(args.checkpoint),
                            checkpoint_sha256=checkpoint_hash, which="avg",
                            physics_file=model_file.name, physics_sha256=model_hash, starts_protocol=protocol,
                            episode_step=np.broadcast_to(np.arange(1, frames+1)[:, None], foot_load.shape),
                            assist_n=0., automatic_resets=False, robot="g1_getup", actor_sensor_noise=False,
                            world="nominal", seed=42, reset_jitter_seed=-1 if args.jitter_seed is None else args.jitter_seed,
                            push_s=-1. if args.push_at is None else args.push_at,
                            push_angle_deg=args.push_angle if args.push_at is not None else 0.)
        score(archive)


def speed_metrics(q, v, points, dt):
    knees = q[:, [10, 16]]
    reversals = sum(len(find_peaks(sign * joint, prominence=.1)[0])
                    for joint in knees.T for sign in (1., -1.))
    joint_speed = v[:, 6:]
    metrics = dict(leg_speed_p95_rad_s=float(np.percentile(abs(v[:, 6:18]), 95)),
                   knee_speed_rms_rad_s=rms(v[:, [9, 15]]), ankle_speed_rms_rad_s=rms(v[:, [10, 11, 16, 17]]),
                   knee_reversals=reversals,
                   feet_path_m=float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=2).sum()))
    if len(q) >= 3:
        # One-sided, Hann-windowed spectral energy at 8 Hz and above, in physical units.
        window = np.hanning(len(q))
        spectrum = np.fft.rfft((joint_speed - joint_speed.mean(axis=0)) * window[:, None], axis=0)
        power = np.abs(spectrum)**2
        power[1:None if len(q) % 2 else -1] *= 2
        high = np.fft.rfftfreq(len(q), dt) >= 8.
        metrics["joint_speed_8hz_to_nyquist_rms_rad_s"] = float(np.sqrt(
            np.mean(power[high].sum(axis=0)) / (len(q) * np.sum(window**2))))
        metrics["joint_jerk_rms_rad_s3"] = rms(np.diff(joint_speed, n=2, axis=0) / dt**2)
    return metrics


def score(archive):
    with np.load(archive, allow_pickle=False) as z:
        model_file = archive.parent / str(z["physics_file"])
        if file_hash(model_file) != str(z["physics_sha256"]):
            raise ValueError(f"Physics hash mismatch for {model_file}")
        model = mujoco.MjModel.from_binary_path(str(model_file))
        ids = [model.body("robot_" + body).id for body in g1_getup.KEY_BODIES]
        dt = float(z["dt"])
        q, v, feet, tilt = z["qpos"], z["qvel"], z["feet_xyz"], z["torso_tilt_deg"]
        support, ended = z["support_load_bw"], z["terminated"]
        controls = z["normalized_action"] if "normalized_action" in z else None
        report = {k: z[k].item() for k in ("checkpoint", "checkpoint_path", "checkpoint_sha256", "physics_sha256", "starts_protocol",
                                           "duration_s", "reset_jitter_seed", "push_s", "push_angle_deg") if k in z}
        clips, offsets = z["clips"], z["offsets_s"]
        expected_sources = z["source_sha256"] if "source_sha256" in z else None
        trials = z["trials"].tolist() if "trials" in z else list(range(len(clips)))
    report["cases"] = []
    for j, (clip, offset) in enumerate(zip(clips, offsets)):
        if ended[:, j].any():
            raise ValueError("Expected an uninterrupted native-player recording")
        ref_file = MOTIONS / clip
        source_hash = file_hash(ref_file)
        if expected_sources is not None and str(expected_sources[j]) != source_hash:
            raise ValueError(f"Reference clip differs from the recording for {ref_file}")
        pose, velocity = load_native_reference(ref_file)
        start = round(float(offset) / dt)
        pose, velocity = pose[start:], velocity[start:]
        up = first_upright(pose)
        count = min(len(q), len(pose), up + 1 if up is not None and up >= 2 else len(pose))
        physical, reference = q[:count, j], pose[:count]
        actual_points = _body_local(landmark_fk(model, physical, ids), physical)
        ref_points = _body_local(landmark_fk(model, reference, ids), reference)
        actual = speed_metrics(physical, v[:count, j], actual_points, dt)
        ref = speed_metrics(reference, velocity[:count], ref_points, dt)
        difference = v[:count, j] - velocity[:count]
        forces = support[:, j, :2]
        standing = stability(q[:, j], forces.sum(axis=1), ended[:, j], dt)
        foot_speeds = np.linalg.norm(np.diff(feet[:, j, :, :2], axis=0, prepend=feet[:1, j, :, :2]), axis=2) / dt
        quiet = ((q[1:, j, 2] > .65) & (q[1:, j, 2] <= .85) & (tilt[:, j] < 15)
                 & (np.linalg.norm(v[1:, j, :2], axis=1) < .08)
                 & (np.sqrt(np.mean(v[1:, j, 21:]**2, axis=1)) < .4) & (foot_speeds.max(axis=1) < .05)
                 & (forces.sum(axis=1) >= .5) & (forces.min(axis=1) >= .1))
        if standing["rise_s"] is None:
            quiet[:] = False
        else:
            quiet[:round(standing["rise_s"] / dt) - 1] = False
        first = first_sustained(quiet, round(1/dt))
        initial_state = np.concatenate((q[0,j], v[0,j])).astype("<f8").tobytes()
        row = dict(trial=trials[j], clip=str(clip), source_sha256=source_hash,
                   reset_state_sha256=hashlib.sha256(initial_state).hexdigest(), offset_s=float(offset),
                   measured_until_s=(count-1)*dt, reference_rise_s=None if up is None else up*dt,
                   first_quiet_s=None if first is None else (first+1)*dt, actual=actual, reference=ref, **standing,
                   joint_rmse_deg=float(np.degrees(rms(physical[:, 7:] - reference[:, 7:]))),
                   landmark_rmse_cm=100*rms(np.linalg.norm(actual_points-ref_points, axis=2)),
                   knee_velocity_rmse_rad_s=rms(difference[:, [9, 15]]),
                   ankle_velocity_rmse_rad_s=rms(difference[:, [10, 11, 16, 17]]),
                   feet_path_ratio=actual["feet_path_m"]/ref["feet_path_m"] if ref["feet_path_m"] > 0 else None)
        tail = min(len(forces), round(10/dt))
        xy = feet[-tail:, j, :, :2]
        row["tail"] = dict(duration_s=tail*dt,
                           both_feet_loaded_fraction=float(np.mean((forces[-tail:] >= .1).all(axis=1))),
                           torso_tilt_mean_deg=float(tilt[-tail:, j].mean()),
                           pelvis_xy_speed_rms_m_s=rms(np.linalg.norm(v[-tail:, j, :2], axis=1)),
                           foot_excursion_cm=float(100*np.linalg.norm(xy-xy[:1], axis=2).max()))
        push_s = report.get("push_s", -1.)
        if push_s >= 0:
            at = round(push_s/dt)
            before = q[at,j]
            next_quiet = first_sustained(quiet[at:], round(1/dt))
            row["push"] = dict(time_s=push_s,
                               started_upright=bool(.65 < before[2] <= .85 and
                                                    1-2*np.sum(before[4:6]**2) > np.cos(np.deg2rad(30))),
                               fall_within_3s=any(push_s <= x["fall_s"] <= push_s+3 for x in row["post_rise_falls"]),
                               next_quiet_after_s=None if next_quiet is None else (next_quiet+1)*dt)
        if controls is not None and count > 1:
            commands = controls[:count-1, j]
            delta = np.diff(commands, axis=0, prepend=np.zeros_like(commands[:1]))
            row["normalized_action_rate_mse"] = rms(delta)**2
        if standing["rise_s"] is not None and up is not None:
            row["rise_lag_s"] = standing["rise_s"] - up*dt
        report["cases"].append(row)
    rows = report["cases"]
    summary = dict(trials=len(rows), final_supported=sum(r["final_3s_supported_upright"] for r in rows),
                   no_refall_and_final_supported=sum(r["final_3s_supported_upright"] and not r["post_rise_falls"] for r in rows))
    for key in ("first_quiet_s", "rise_lag_s", "feet_path_ratio", "knee_velocity_rmse_rad_s", "ankle_velocity_rmse_rad_s"):
        values = [r[key] for r in rows if r.get(key) is not None]
        summary["median_"+key] = float(np.median(values)) if values else None
    for key in ("joint_rmse_deg", "landmark_rmse_cm"):
        summary["mean_"+key] = float(np.mean([r[key] for r in rows]))
    for key in ("actual", "reference"):
        summary[key] = {name: float(np.mean([r[key][name] for r in rows if name in r[key]]))
                        for name in sorted(set().union(*(r[key] for r in rows)))}
    report["summary"] = summary
    output = archive.with_name(archive.stem+"-metrics.json")
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(output, json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("record", help="Run GetupPlayer.step and score each capture")
    run.add_argument("--checkpoint", type=Path, default=RELEASE)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--seconds", type=float, default=30.)
    run.add_argument("--protocol", choices=("source-play", "fallen"), nargs="+", default=["source-play", "fallen"])
    run.add_argument("--trials", type=int, nargs="+", help="Zero-based indices within each chosen protocol")
    run.add_argument("--jitter-seed", type=int, help="Perturb reset joints and roll/pitch up to 2 deg, XY velocity up to 0.07 m/s")
    run.add_argument("--push-at", type=float, help="Apply the viewer's +0.20 m/s impulse at this time in seconds")
    run.add_argument("--push-angle", type=float, default=0., help="Horizontal impulse direction in degrees from world +X")
    replay = commands.add_parser("score", help="Rescore one or more existing native-player archives")
    replay.add_argument("archives", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "score":
        for path in args.archives:
            score(path)
    else:
        if not np.isfinite(args.seconds) or args.seconds < 1:
            parser.error("--seconds must be at least one")
        if args.push_at is not None and not 0 <= args.push_at < args.seconds:
            parser.error("--push-at must be inside the recorded interval")
        if not np.isfinite(args.push_angle) or (args.push_at is None and args.push_angle != 0.):
            parser.error("--push-angle requires a finite direction and --push-at")
        if args.trials is not None and (min(args.trials) < 0 or len(set(args.trials)) != len(args.trials)):
            parser.error("--trials expects distinct nonnegative indices")
        record(args)


if __name__ == "__main__":
    main()
