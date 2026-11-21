"""Compare measured palm support, knee extension and stationary standing across native Zenbot rollouts."""

import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from tools.score_zenbot_sequence import first_run
from tools.score_getup_in_place import pre_rise_attempts


DT = 0.02
KNEES = [3, 9]


@lru_cache(maxsize=1)
def palm_geometry():
    import mujoco
    from instantamp import zenbot
    from instantamp.sim import build_scene

    model = build_scene(robot_name="zenbot23_getup")
    ground = model.geom("floor").id
    colliders = np.flatnonzero((model.geom_contype != 0) & (np.arange(model.ngeom) != ground))
    hand_bodies = [model.body("robot_" + body).id for body in zenbot.HANDS]
    hands = np.isin(model.geom_bodyid[colliders], hand_bodies)
    return model, mujoco.MjData(model), ground, colliders, hands


@lru_cache(maxsize=32)
def reference_clearance(clip):
    import mujoco

    model, data, ground, colliders, hands = palm_geometry()
    with np.load(clip) as z:
        poses = np.column_stack((z["root_pos"], z["root_quat"][:, [3, 0, 1, 2]], z["dof_pos"]))
    result = np.empty(len(poses))
    for i, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        distances = np.array([mujoco.mj_geomDistance(model, data, ground, int(g), 2.0, None) for g in colliders])
        result[i] = distances[hands].min() - min(0.0, distances.min())
    return result


def phase(joints, palm, stop):
    """Find sustained 20 degree knee extension following maximum bilateral knee flexion."""
    end = max(1, len(joints) if stop is None else stop + 1)
    knee = joints[:end, KNEES].mean(axis=1)
    peak = int(np.argmax(knee))
    extended = first_run(knee[peak:] < knee[peak] - np.deg2rad(20), 4)
    extension = None if extended is None else peak + extended
    mask = np.asarray(palm[:end if extension is None else extension], bool)
    spans = np.diff(np.r_[False, mask, False].astype(int))
    lengths = np.flatnonzero(spans < 0) - np.flatnonzero(spans > 0)
    hand = first_run(mask, 6)
    return dict(knee_tick=extension, hand_tick=hand,
                knee_to_stand_s=None if extension is None or stop is None else DT * (stop - extension),
                hand_s=float(DT * lengths[lengths >= 6].sum()))


def describe(path, reference):
    result = json.loads(path.read_text())
    with np.load(path.with_suffix('.npz')) as z:
        q, v, feet, load = (z[key] for key in ('qpos', 'qvel', 'feet_xy', 'support_load_bw'))
    if len(q) != len(load) + 1 or len(load) < 500 or load.shape[1] != 6:
        raise ValueError(f'Expected aligned 50 Hz robot states and at least ten seconds of contact loads in {path}')
    start = round(result['offset'] / DT)
    with np.load(reference / result['clip']) as z:
        ref_joints = z['dof_pos'][start:]
        ref_velocity = z['dof_vel'][start:]
        ref_xy = z['root_pos'][start:, :2]
        marker_height = z['key_body_pos'][start:, -2:, 2].min(axis=1)
    up = round(result['rise'] / DT) if result['rise'] is not None else None
    ref_up = round(result['reference_rise_s'] / DT) if result['reference_rise_s'] is not None else None
    # Each contact measurement belongs to the transition into the aligned qpos frame.
    actual = phase(q[:, 7:], np.r_[False, load[:, 2:4].sum(-1) >= 0.07], up)
    surface_height = reference_clearance(reference / result['clip'])[start:]
    source = phase(ref_joints, surface_height < 0.065, ref_up)
    marker_phase = phase(ref_joints, marker_height < 0.10, ref_up)
    ground_end = source["knee_tick"] or len(marker_height)
    ext, ref_ext = actual['knee_tick'], source['knee_tick']
    floor_error = None
    if ext and ref_ext:
        sample = np.rint(np.linspace(0, ext, ref_ext + 1)).astype(int)
        floor_error = float(np.rad2deg(np.sqrt(np.square(q[sample, 7:] - ref_joints[:ref_ext+1]).mean())))
    bilateral = (load[-500:, :2] >= 0.10).all(axis=1)
    foot_speed = np.linalg.norm(np.diff(feet, axis=0, prepend=feet[:1])[-500:], axis=-1) / DT
    quiet = (bilateral & (np.linalg.norm(v[-500:, :2], axis=1) < 0.08)
             & (foot_speed.max(axis=1) < 0.05))
    if 'pre_rise_attempts' not in result:
        result['pre_rise_attempts'] = pre_rise_attempts(q[:, 2], DT, result['rise'])
    # The native Zenbot model supplies rubber-palm collision loads. The elbow
    # links have no separate forearm contact surface on this articulation.
    result.pop("real_forearm_peak_bw", None)
    once = bool(up is not None and result['held'] and not result['refall']
                and result['pre_rise_attempts']['retries'] == 0)
    actual_end = len(v) if up is None else max(1, up)
    source_end = len(ref_velocity) if ref_up is None else max(1, ref_up)
    return dict(**result, actual_phase=actual, reference_phase=source, reference_marker_phase=marker_phase,
                minimum_hand_landmark_m=float(marker_height[:ground_end].min()),
                minimum_palm_surface_m=float(surface_height[:ground_end].min()), ground_joint_rmse_deg=floor_error,
                ground_pelvis_shift_cm=None if up is None else float(100 * np.linalg.norm(q[up, :2] - q[0, :2])),
                reference_ground_pelvis_shift_cm=None if ref_up is None else float(100 * np.linalg.norm(ref_xy[ref_up] - ref_xy[0])),
                last_10s_pelvis_excursion_cm=float(100 * np.linalg.norm(q[-501:, :2] - q[-501, :2], axis=1).max()),
                last_10s_foot_excursion_cm=float(100 * np.linalg.norm(feet[-500:] - feet[-500], axis=2).max()),
                knee_speed_p90_rad_s=float(np.percentile(np.maximum(0, -v[:actual_end, [9, 15]]), 90)),
                reference_knee_speed_p90_rad_s=float(np.percentile(np.maximum(0, -ref_velocity[:source_end, KNEES]), 90)),
                first_attempt_held=once, quiet_fraction_last_10s=float(quiet.mean()),
                bilateral_quiet_10s=bool(quiet.mean() >= 0.90),
                both_feet_and_pelvis_within_10cm=bool(once and max(result['drift_cm'],result['feet_cm']) <= 10))


def summarize(rows):
    def med(key, subset=rows):
        vals = [r[key] for r in subset if r[key] is not None]
        return float(np.median(vals)) if vals else None
    need = [r for r in rows if r['reference_phase']['hand_tick'] is not None
            and r['reference_phase']['knee_tick'] is not None]
    palm_first = sum(r['actual_phase']['hand_tick'] is not None and r['actual_phase']['knee_tick'] is not None
                     for r in need)
    return dict(trials=len(rows), first_attempt_held=sum(r['first_attempt_held'] for r in rows),
                throughout_upright=sum(r['throughout'] for r in rows),
                bilateral_quiet_10s=sum(r['bilateral_quiet_10s'] for r in rows),
                both_feet_and_pelvis_within_10cm=sum(r['both_feet_and_pelvis_within_10cm'] for r in rows),
                reference_palm_required=len(need), palm_before_knee_extension=palm_first,
                median_rise_s=med('rise'), median_reference_rise_s=med('reference_rise_s'),
                median_extension_s=med('extension_s'), median_reference_extension_s=med('reference_extension_s'),
                median_postrise_pelvis_cm=med('drift_cm'), median_postrise_feet_cm=med('feet_cm'),
                median_last_10s_pelvis_excursion_cm=med('last_10s_pelvis_excursion_cm'),
                median_last_10s_foot_excursion_cm=med('last_10s_foot_excursion_cm'),
                median_ground_pelvis_shift_cm=med('ground_pelvis_shift_cm'),
                median_reference_ground_pelvis_shift_cm=med('reference_ground_pelvis_shift_cm'),
                max_postrise_pelvis_cm=max((r['drift_cm'] for r in rows if r['drift_cm'] is not None),default=None),
                median_hip_speed_p90_rad_s=med('hip_velocity_p90_rad_s'),
                median_ref_hip_speed_p90_rad_s=med('reference_hip_velocity_p90_rad_s'),
                median_knee_extension_speed_p90_rad_s=med('knee_speed_p90_rad_s'),
                median_reference_knee_extension_speed_p90_rad_s=med('reference_knee_speed_p90_rad_s'),
                median_vertical_speed_p90_m_s=med('vertical_velocity_p90_m_s'),
                median_reference_vertical_speed_p90_m_s=med('reference_vertical_velocity_p90_m_s'),
                median_ground_joint_rmse_deg=med('ground_joint_rmse_deg',need))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories', nargs='+', type=Path)
    p.add_argument('--reference', type=Path, default=Path('data/zenbot23_getup/recovery'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    groups = {}
    for directory in args.directories:
        reports = sorted(directory.glob('[0-9]*_[0-9]*.json'))
        if any(not f.with_suffix('.npz').is_file() for f in reports):
            p.error(f'Missing trajectories in {directory}. Record all trials with --trajectories before summarizing.')
        cases = [describe(f, args.reference) for f in reports]
        if not cases:
            p.error(f'No paired trial reports and recordings in {directory}')
        groups[str(directory)] = dict(summary=summarize(cases), cases=cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(groups, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v['summary'] for k,v in groups.items()}, indent=2))


if __name__ == '__main__':
    main()
