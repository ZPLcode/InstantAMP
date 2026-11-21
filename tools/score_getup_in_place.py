"""Measure G1 stance from its first upright frame with no settling grace.

    python -m tools.score_getup_in_place SOURCE_PLAY.npz FALLEN.npz --out in_place.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


MOTIONS = Path(__file__).resolve().parents[1] / "data/g1_getup/recovery"
LIMITS_CM = (10, 20, 30)


def displacement(xy):
    distances = np.linalg.norm(xy - xy[0], axis=-1)
    return dict(net_cm=float(100*distances[-1]), max_cm=float(100*distances.max()),
                path_cm=float(100*np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum()))


def planted(ankle_xy, load, dt):
    from instantamp.stability import contiguous
    # Three consecutive loaded samples reject isolated impact spikes. During each
    # such bout score every sample, including slip while the foot stays on the floor.
    bouts = [(a,b) for a,b in contiguous(load >= .10) if b-a >= round(.06/dt)]
    if not bouts:
        return None
    positions = np.concatenate([ankle_xy[a:b] for a,b in bouts])
    return dict(first_loaded_after_rise_s=float(bouts[0][0]*dt), bouts=len(bouts), **displacement(positions))


def pre_rise_attempts(height, dt, first_rise):
    """Identify an elevation, a substantial return to the floor, and another elevation."""
    frames = max(1,round(.10/dt))
    stop = len(height) if first_rise is None else round(first_rise/dt)+1
    trough = peak = float(height[0])
    elevated = False
    above = below = 0
    events = []
    for frame,h in enumerate(height[:stop]):
        if not elevated:
            trough = min(trough,h)
            above = above+1 if h >= .50 and h-trough >= .10 else 0
            if above == frames:
                start = frame-frames+1
                if events:
                    events[-1]["new_elevation_s"] = float(start*dt)
                elevated = True
                peak_at = start+int(np.argmax(height[start:frame+1]))
                peak = height[peak_at]
                below = 0
        else:
            if h > peak:
                peak,peak_at = h,frame
            below = below+1 if h <= .45 and peak-h >= .12 else 0
            if below == frames:
                low = frame-frames+1
                events.append(dict(elevation_start_s=float(start*dt), peak_s=float(peak_at*dt),
                                   peak_height_m=float(peak), low_s=float(low*dt),
                                   low_height_m=float(height[low]), new_elevation_s=None))
                trough = float(np.min(height[low:frame+1]))
                elevated = False
                above = 0
    if events and first_rise is not None and events[-1]["new_elevation_s"] is None:
        events[-1]["new_elevation_s"] = float(first_rise)
    return dict(failed_elevations=len(events), retries=sum(e["new_elevation_s"] is not None for e in events),
                events=events)


def case(q, feet, support, torso_deg, dt, ref, ref_feet, drift_info, trial):
    from instantamp.stability import stability
    from tools.compare_recovery_rollouts import first_upright
    loads = support.sum(axis=1)
    regular = stability(q, loads, np.zeros(len(loads),bool), dt)
    rise = regular["rise_s"]
    tail_count = min(len(loads), round(10/dt))
    tail = feet[-tail_count:]
    tail_foot_cm = float(100*np.linalg.norm(tail-tail[:1], axis=2).max())
    tail_both = float(np.mean((support[-tail_count:] >= .10).all(axis=1)))
    tail_tilt = float(torso_deg[-tail_count:].mean())
    tail_quiet = bool(len(loads) >= round(10/dt) and regular["final_3s_supported_upright"] and tail_both >= .90 and
                      tail_foot_cm <= 3 and drift_info["tail_pelvis_xy_speed_rms_m_s"] < .08 and tail_tilt < 15)
    ref_up = first_upright(ref)
    available = min(len(q), len(ref))
    rel = (q[:available,:2]-q[0,:2]) - (ref[:available,:2]-ref[0,:2])
    reference = dict(available_s=float((available-1)*dt),
                     attempts=pre_rise_attempts(ref[:,2],dt,None if ref_up is None else ref_up*dt),
                     first_upright_s=None if ref_up is None else float(ref_up*dt),
                     reset_to_clip_end=displacement(ref[:,:2]),
                     robot_reset_to_common_end=displacement(q[:available,:2]),
                     aligned_pelvis_xy_rmse_cm=float(100*np.sqrt(np.mean(np.sum(rel**2,axis=1)))))
    if ref_up is not None:
        reference["remaining_after_ref_rise_s"] = float((len(ref)-1-ref_up)*dt)
        reference["ref_rise_to_clip_end"] = displacement(ref[ref_up:,:2])
        reference["ankle_origin_max_cm_after_ref_rise"] = [
            displacement(ref_feet[ref_up:,i,:2])["max_cm"] for i in (0,1)]
    row = dict(trial=int(trial), first_upright_s=rise, horizon_s=float((len(q)-1)*dt),
               attempts=pre_rise_attempts(q[:,2],dt,rise),
               reset_to_final=displacement(q[:,:2]), reference=reference, final_supported=regular["final_3s_supported_upright"],
               final_bilateral_quiet_10s=tail_quiet, tail_both_loaded_fraction=tail_both,
               tail_foot_max_cm=tail_foot_cm, post_rise_falls=regular["post_rise_falls"], **drift_info)
    row["throughout_stable"] = False
    row["pelvis_within_cm"] = {str(k):False for k in LIMITS_CM}
    row["pelvis_and_planted_feet_within_cm"] = row["pelvis_within_cm"].copy()
    if rise is None:
        return row
    at = round(rise/dt)
    xy = q[at:,:2]
    foot_xy = feet[at-1:]
    row["full_after_rise"] = displacement(xy)
    row["followup_s"] = float((len(xy)-1)*dt)
    row["reset_to_first_rise"] = displacement(q[:at+1,:2])
    for seconds in (1,2):
        n = round(seconds/dt)
        row[f"first_{seconds}s_after_rise"] = displacement(xy[:n+1]) if len(xy)>n else None
    row["ankles_after_rise"] = [displacement(foot_xy[:,i]) for i in (0,1)]
    row["loaded_ankles_after_rise"] = [planted(foot_xy[:,i],support[at-1:,i],dt) for i in (0,1)]
    if at < available:
        reference["common_after_robot_rise_s"] = float((available-1-at)*dt)
        reference["reference_after_robot_rise"] = displacement(ref[at:available,:2])
        reference["robot_after_robot_rise"] = displacement(q[at:available,:2])
    orientation = q[at:,3:7]
    upright = ((q[at:,2] > .65) & (q[at:,2] <= .85) &
               (1-2*np.sum(orientation[:,1:3]**2,axis=1) > np.cos(np.deg2rad(30))))
    loaded = loads[at-1:] >= .50
    w = round(1/dt)
    worst = np.convolve(loaded.astype(int),np.ones(w,dtype=int),"valid").min()/w if len(loaded)>=w else 0.
    row["every_frame_upright_after_rise"] = bool(upright.all())
    row["worst_1s_loaded_fraction_after_rise"] = float(worst)
    row["throughout_stable"] = bool(row["followup_s"] >= 3 and upright.all() and worst >= .90
                                    and not row["post_rise_falls"] and tail_quiet)
    for limit in LIMITS_CM:
        ok = row["throughout_stable"] and row["full_after_rise"]["max_cm"] <= limit
        row["pelvis_within_cm"][str(limit)] = ok
        row["pelvis_and_planted_feet_within_cm"][str(limit)] = ok and all(
            foot is not None and foot["max_cm"] <= limit for foot in row["loaded_ankles_after_rise"])
    return row


def summarize(rows):
    risen = [r for r in rows if r["first_upright_s"] is not None]
    def median(values):
        return statistics.median(values) if values else None
    def mean(values):
        return statistics.mean(values) if values else None
    peaks = [r["full_after_rise"]["max_cm"] for r in risen]
    diagnosed = [r for r in rows if "attempts" in r]
    clean = [r for r in diagnosed if r["first_upright_s"] is not None and not r["attempts"]["retries"]
             and not r["post_rise_falls"]]
    return dict(trials=len(rows), ever_rose=len(risen), final_supported=sum(r["final_supported"] for r in rows),
                attempt_diagnostics_trials=len(diagnosed),
                trials_with_pre_rise_retries=sum(r["attempts"]["retries"]>0 for r in diagnosed),
                total_pre_rise_retries=sum(r["attempts"]["retries"] for r in diagnosed),
                reference_starts_with_pre_rise_retries=sum(r["reference"]["attempts"]["retries"]>0 for r in diagnosed),
                rose_without_retry_or_refall=len(clean),
                first_attempt_pelvis_within_cm={str(k):sum(r["pelvis_within_cm"][str(k)] for r in clean) for k in LIMITS_CM},
                first_attempt_pelvis_and_feet_within_cm={str(k):sum(r["pelvis_and_planted_feet_within_cm"][str(k)] for r in clean) for k in LIMITS_CM},
                rose_no_refall=sum(not r["post_rise_falls"] for r in risen),
                final_bilateral_quiet_10s=sum(r["final_bilateral_quiet_10s"] for r in rows),
                throughout_stable=sum(r["throughout_stable"] for r in rows),
                median_first_rise_s=median([r["first_upright_s"] for r in risen]),
                median_time_from_rise_to_quiet_s=median([r["rise_to_quiet_s"] for r in risen if r["rise_to_quiet_s"] is not None]),
                median_first_1s_net_cm=median([r["first_1s_after_rise"]["net_cm"] for r in risen if r["first_1s_after_rise"]]),
                median_first_1s_max_cm=median([r["first_1s_after_rise"]["max_cm"] for r in risen if r["first_1s_after_rise"]]),
                median_first_2s_max_cm=median([r["first_2s_after_rise"]["max_cm"] for r in risen if r["first_2s_after_rise"]]),
                mean_full_max_cm=mean(peaks), median_full_max_cm=median(peaks),
                p90_full_max_cm=float(np.percentile(peaks,90)) if peaks else None,
                mean_first_1s_max_cm=mean([r["first_1s_after_rise"]["max_cm"] for r in risen if r["first_1s_after_rise"]]),
                mean_first_2s_max_cm=mean([r["first_2s_after_rise"]["max_cm"] for r in risen if r["first_2s_after_rise"]]),
                median_reset_to_end_net_cm=median([r["reset_to_final"]["net_cm"] for r in rows]),
                median_max_loaded_ankle_cm=median([max(x["max_cm"] for x in r["loaded_ankles_after_rise"])
                                                   for r in risen if all(r["loaded_ankles_after_rise"])]),
                **{key:{str(k):sum(r[key][str(k)] for r in rows) for k in LIMITS_CM}
                   for key in ("pelvis_within_cm","pelvis_and_planted_feet_within_cm")})


def score(archives, out):
    import mujoco
    from instantamp import g1_getup
    from instantamp.stability import first_sustained
    from tools.compare_recovery_rollouts import file_hash, landmark_fk
    from tools.score_g1_tracking import load_native_reference
    unique, groups, inputs = {}, {}, []
    checkpoint_hash = None
    settings = None
    for archive in archives:
        with np.load(archive,allow_pickle=False) as z:
            if (z["assist_n"].item() != 0 or z["automatic_resets"].item() or z["terminated"].any()
                    or list(z["support_body_names"][:2]) != list(g1_getup.SUPPORT_BODIES[:2])):
                raise ValueError(f"Expected uninterrupted, unassisted native player trajectories and two named ankle loads in {archive}")
            digest = str(z["checkpoint_sha256"])
            if checkpoint_hash is not None and checkpoint_hash != digest:
                raise ValueError("Supply only archives for the same checkpoint")
            checkpoint_hash = digest
            signature = tuple(z[k].item() if k in z else default for k,default in
                              (("physics_sha256",None),("dt",None),("duration_s",None),
                               ("reset_jitter_seed",-1),("push_s",-1.),("actor_sensor_noise",False),("push_angle_deg",0.)))
            if settings is not None and signature != settings:
                raise ValueError("Physical model, time horizon or perturbation differs between archives")
            settings = signature
            physics = archive.parent / str(z["physics_file"])
            if file_hash(physics) != str(z["physics_sha256"]):
                raise ValueError("MuJoCo model hash mismatch")
            model = mujoco.MjModel.from_binary_path(str(physics))
            ids = [model.body("robot_"+body).id for body in g1_getup.KEY_BODIES[:2]]
            protocol,dt = str(z["starts_protocol"]),float(z["dt"])
            q,v,feet,support,tilt = z["qpos"],z["qvel"],z["feet_xyz"][...,:2],z["support_load_bw"][...,:2],z["torso_tilt_deg"]
            trials = z["trials"] if "trials" in z else np.arange(q.shape[1])
            for j,(clip,offset) in enumerate(zip(z["clips"],z["offsets_s"])):
                clip = str(clip)
                initial = np.concatenate((q[0,j],v[0,j])).astype("<f8").tobytes()
                key = (clip,float(offset),hashlib.sha256(initial).hexdigest())
                trajectory = hashlib.sha256(q[:,j].astype("<f8").tobytes())
                trajectory.update(support[:,j].astype("<f8").tobytes())
                trajectory_hash = trajectory.hexdigest()
                if key in unique and unique[key]["trajectory_sha256"] != trajectory_hash:
                    raise ValueError(f"Two recordings with the same reset diverged for {clip} at {offset}s")
                if key not in unique:
                    ref,_ = load_native_reference(MOTIONS/clip)
                    ref = ref[round(offset/dt):]
                    ref_feet = landmark_fk(model,ref,ids)
                    speed = np.linalg.norm(v[1:,j,:2],axis=1)
                    foot_speed = np.linalg.norm(np.diff(feet[:,j],axis=0,prepend=feet[:1,j]),axis=2)/dt
                    quiet = ((q[1:,j,2]>.65) & (q[1:,j,2]<= .85) & (tilt[:,j]<15) & (speed<.08) &
                             (np.sqrt(np.mean(v[1:,j,21:]**2,axis=1))<.4) & (foot_speed.max(axis=1)<.05) &
                             (support[:,j].min(axis=1)>=.1) & (support[:,j].sum(axis=1)>=.5))
                    extra = dict(tail_pelvis_xy_speed_rms_m_s=float(np.sqrt(np.mean(speed[-round(10/dt):]**2))))
                    row = case(q[:,j],feet[:,j],support[:,j],tilt[:,j],dt,ref,ref_feet,extra,trials[j])
                    first = row["first_upright_s"]
                    quiet_index = None if first is None else first_sustained(quiet[round(first/dt)-1:],round(1/dt))
                    row["rise_to_quiet_s"] = None if quiet_index is None else float(quiet_index*dt)
                    row.update(clip=clip, offset_s=float(offset), reset_state_sha256=key[2],
                               trajectory_sha256=trajectory_hash, protocols={})
                    unique[key] = row
                row = unique[key]
                row["protocols"][protocol] = int(trials[j])
                groups.setdefault(protocol,[]).append(row)
            inputs.append(str(archive))
    rows = list(unique.values())
    result = dict(checkpoint_sha256=checkpoint_hash, archives=inputs, physics_sha256=settings[0],
                  dt_s=settings[1],duration_s=settings[2],reset_jitter_seed=settings[3],push_s=settings[4],push_angle_deg=settings[6],
                  definition="First 0.20s sustained upright frame, no grace. All following upright frames, at least 90% foot-supported in every 1s, final 10s bilateral quiet, 3s minimum followup. Distances in world-plane XY. Count whole episodes including never-risen starts in each denominator. Planted feet use >=0.1 BW for >=3 samples. Reference comparison stops at source end.",
                  retry_definition="Before first upright: gain 10 cm from a prior low and reach >=0.50 m for 0.10 s, then drop >=0.12 m from the peak and reach <=0.45 m for 0.10 s. Count a retry if another qualifying elevation or the first sustained upright follows. Never-risen trials are retained and incomplete attempts are recorded separately.",
                  by_protocol={k:summarize(v) for k,v in groups.items()},unique=summarize(rows),cases=rows)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(out,result["unique"],flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("archives",type=Path,nargs="+")
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    score(args.archives,args.out)
