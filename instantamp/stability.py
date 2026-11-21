"""Physical stability metrics for named G1 recovery evaluations."""

import hashlib

import mujoco
import numpy as np


def contiguous(mask):
    """Half-open runs of true samples."""
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(boundaries[::2], boundaries[1::2]))


def first_sustained(mask, n):
    return next((int(start) for start, end in contiguous(mask) if end - start >= n), None)


def longest_supported(posture, loaded):
    longest = 0
    for start, end in contiguous(posture):
        if end - start <= longest:
            continue
        values = np.r_[0, np.cumsum(loaded[start:end])]
        for length in range(end - start, longest, -1):
            if np.any(10 * (values[length:] - values[:-length]) >= 9 * length):
                longest = length
                break
    return longest


def drift(xy):
    if not len(xy):
        return None
    distance = np.linalg.norm(xy - xy[0], axis=1)
    return {"max_cm": float(distance.max() * 100), "net_cm": float(distance[-1] * 100)}


def support_geoms(model):
    groups = {}
    for part, link in (("hand", "wrist_yaw_link"), ("knee", "knee_link")):
        for side in ("left", "right"):
            body = model.body(f"robot_{side}_{link}").id
            indices = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body and model.geom_contype[i]]
            # Score the knee collider itself; the distal shank can touch independently.
            if part == "knee":
                knee = [i for i in indices if "knee_collision" in model.geom(i).name]
                if not knee:
                    raise ValueError("This geometry has no separate proximal knee collider")
                indices = knee
            if not indices:
                raise ValueError(f"No contact geometry found for {side} {part}")
            groups[f"{side}_{part}"] = indices
    return groups


def floor_gaps(model, poses, groups):
    data = mujoco.MjData(model)
    floor = model.geom("floor").id
    gaps = np.empty((len(poses), len(groups)))
    for t, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        for j, indices in enumerate(groups.values()):
            gaps[t, j] = min(mujoco.mj_geomDistance(model, data, floor, g, 2.0, None) for g in indices)
    return gaps


def stability(poses, loads, ended, dt, model=None, groups=None, support_bw=None, support_names=()):
    """Poses include the reset pose; load and termination rows describe post-step poses."""
    end = np.flatnonzero(ended)
    valid = int(end[0]) if len(end) else len(loads)
    q = poses[1 : valid + 1]
    qw = q[:, 3:7].astype(float)
    qw /= np.linalg.norm(qw, axis=1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * np.square(qw[:, 1:3]).sum(axis=1), -1, 1)))
    upright = (q[:, 2] > 0.65) & (q[:, 2] <= 0.85) & (tilt < 30)
    loaded = loads[:valid] >= 0.5
    hold_frames = round(0.2 / dt)
    tail = round(3 / dt)
    rise = first_sustained(upright, hold_frames)
    falls = []
    losses = []
    if rise is not None:
        bad = (q[:, 2] < 0.50) | (tilt > 60)
        bad[: rise + hold_frames] = False
        # Once fallen, require a later sustained upright posture before counting another fall.
        cursor = rise + hold_frames
        while cursor < valid:
            idx = first_sustained(bad[cursor:], hold_frames)
            if idx is None:
                break
            fell = cursor + idx
            recovered = first_sustained(upright[fell + hold_frames :], hold_frames)
            recovered = None if recovered is None else fell + hold_frames + recovered
            falls.append(
                {
                    "fall_s": float((fell + 1) * dt),
                    "recovered_s": None if recovered is None else float((recovered + 1) * dt),
                }
            )
            if recovered is None:
                break
            cursor = recovered + hold_frames
        losses = [(a + rise, b + rise) for a, b in contiguous(~upright[rise:]) if b - a >= hold_frames]
    final_available = not len(end) and valid >= tail
    tail_posture = bool(final_available and upright[-tail:].all())
    tail_load_fraction = float(loaded[-tail:].mean()) if final_available else None
    passed = bool(tail_posture and 10 * loaded[-tail:].sum() >= 9 * tail)
    tail_drift = drift(q[-tail:, :2]) if final_available else None
    # A final hold can hide an earlier walk across the floor. Also score the
    # entire interval after allowing one second to settle into the first rise.
    settled = rise + round(1 / dt) if rise is not None else valid
    result = {
        "rise_s": None if rise is None else float((rise + 1) * dt),
        "post_rise_falls": falls,
        "post_rise_posture_loss_s": [[float((a + 1) * dt), float(b * dt)] for a, b in losses],
        "longest_upright_s": float(max((b - a for a, b in contiguous(upright)), default=0) * dt),
        "longest_supported_upright_s": float(longest_supported(upright, loaded) * dt),
        "final_3s_upright": tail_posture,
        "final_3s_loaded_fraction": tail_load_fraction,
        "final_3s_supported_upright": passed,
        "final_3s_xy_drift": tail_drift,
        "post_rise_xy_drift": None if rise is None else drift(q[rise:, :2]),
        "after_settling_xy_drift": drift(q[settled:, :2]),
        "final_3s_tilt_max_deg": float(tilt[-tail:].max()) if final_available else None,
        "final_3s_min_height_m": float(q[-tail:, 2].min()) if final_available else None,
        "final_3s_max_height_m": float(q[-tail:, 2].max()) if final_available else None,
        "termination_s": float((valid + 1) * dt) if len(end) else None,
    }
    if support_bw is not None and rise is not None:
        start = rise + hold_frames
        result["post_rise_limb_load"] = {}
        for i, name in enumerate(support_names):
            if "knee" not in name.lower() and "wrist" not in name.lower() and "hand" not in name.lower():
                continue
            load = support_bw[start:valid, i]
            touch = load >= 0.05
            first = first_sustained(touch, hold_frames)
            result["post_rise_limb_load"][name] = {
                "peak_bw": float(load.max()) if len(load) else None,
                "first_sustained_s": None if first is None else float((start + first + 1) * dt),
                "seconds_above_5pct_bw": float(touch.sum() * dt),
                "final_3s_seconds_above_5pct_bw": float((support_bw[-tail:, i] >= 0.05).sum() * dt)
                if final_available
                else None,
            }
    if model is not None and rise is not None:
        start = rise + hold_frames
        gaps = floor_gaps(model, q[start:], groups)
        result["limb_near_floor_after_rise"] = {}
        for i, name in enumerate(groups):
            mask = gaps[:, i] <= 0.02
            bouts = [(a, b) for a, b in contiguous(mask) if b - a >= hold_frames]
            result["limb_near_floor_after_rise"][name] = {
                "min_clearance_cm": float(100 * gaps[:, i].min()) if len(gaps) else None,
                "first_sustained_near_s": float((start + bouts[0][0] + 1) * dt) if bouts else None,
                "seconds_within_2cm": float(mask.sum() * dt),
                "longest_near_s": float(max((b - a for a, b in bouts), default=0) * dt),
            }
    return result


def score(path, geometry=True):
    with np.load(path, allow_pickle=False) as z:
        q, force, ended = z["qpos"], z["foot_load_bw"], z["terminated"]
        if q.ndim != 3 or q.shape[:2] != (len(force) + 1, force.shape[1]) or ended.shape != force.shape:
            raise ValueError("qpos, post-step load and termination axes do not match")
        dt = float(z["dt"])
        if not np.isfinite(dt) or dt <= 0 or not np.isclose(round(0.2 / dt) * dt, 0.2):
            raise ValueError("Expected a sampling interval compatible with a 0.2 s posture window")
        if str(z["robot"]) != "g1_getup" or float(z["assist_n"]) != 0:
            raise ValueError("Expected an unassisted named G1 get-up rollout")
        model, groups = None, None
        if geometry:
            filename = path.parent / str(z["physics_file"])
            if hashlib.sha256(filename.read_bytes()).hexdigest() != str(z["physics_sha256"]):
                raise ValueError("Physics model fingerprint mismatch")
            model = mujoco.MjModel.from_binary_path(str(filename))
            groups = support_geoms(model)
        support = z["support_load_bw"] if "support_load_bw" in z else None
        support_names = z["support_body_names"].tolist() if support is not None else ()
        trials = []
        for i, name in enumerate(z["clips"]):
            trial = stability(
                q[:, i],
                force[:, i],
                ended[:, i],
                dt,
                model,
                groups,
                support_bw=support[:, i] if support is not None else None,
                support_names=support_names,
            )
            trials.append(dict(trial=i, clip=str(name), offset_s=float(z["offsets_s"][i]), **trial))
        meta = {
            k: str(z[k]) for k in ("checkpoint", "starts_protocol", "physics_sha256", "checkpoint_sha256") if k in z
        }

    def count(predicate):
        return sum(bool(predicate(t)) for t in trials)

    def median(values):
        return float(np.median(values)) if values else None

    final = [t for t in trials if t["final_3s_supported_upright"]]
    no_falls = [t for t in final if not t["post_rise_falls"]]
    settled = [t["after_settling_xy_drift"]["max_cm"] for t in no_falls if t["after_settling_xy_drift"]]
    summary = {
        "trials": len(trials),
        "rose": count(lambda t: t["rise_s"] is not None),
        "fell_after_rise": count(lambda t: t["post_rise_falls"]),
        "recovered_after_fall": count(lambda t: any(f["recovered_s"] is not None for f in t["post_rise_falls"])),
        "posture_loss_after_rise": count(lambda t: t["post_rise_posture_loss_s"]),
        "final_3s_upright": count(lambda t: t["final_3s_upright"]),
        "final_3s_supported_upright": len(final),
        "no_fall_and_final_3s_supported": sum(not t["post_rise_falls"] for t in final),
        "no_posture_loss_and_final_3s_supported": sum(not t["post_rise_posture_loss_s"] for t in final),
        "final_3s_supported_and_drift_le_10cm": sum(t["final_3s_xy_drift"]["max_cm"] <= 10 for t in final),
        "final_3s_supported_and_drift_le_20cm": sum(t["final_3s_xy_drift"]["max_cm"] <= 20 for t in final),
        "median_longest_supported_s": median([t["longest_supported_upright_s"] for t in trials]),
        "median_max_drift_cm_among_final_holds": median([t["final_3s_xy_drift"]["max_cm"] for t in final]),
        "median_after_settling_drift_cm_among_no_fall_final_holds": median(settled),
        "no_fall_final_3s_supported_and_settled_drift_le_20cm": sum(distance <= 20 for distance in settled),
    }
    if support is not None:
        for part, names in (("hand", ("wrist", "hand")), ("knee", ("knee",))):
            summary[f"{part}_load_after_rise"] = count(
                lambda t: any(
                    any(word in name.lower() for word in names) and v["first_sustained_s"] is not None
                    for name, v in t.get("post_rise_limb_load", {}).items()
                )
            )
    if geometry:
        for part in ("hand", "knee"):
            summary[f"{part}_near_floor_after_rise"] = count(
                lambda t: any(
                    name.endswith(part) and gap["first_sustained_near_s"] is not None
                    for name, gap in t.get("limb_near_floor_after_rise", {}).items()
                )
            )
    return dict(
        archive=str(path),
        **meta,
        dt_s=dt,
        duration_s=len(force) * dt,
        definitions={
            "posture": "Pelvis height (0.65,0.85] m and inclination under 30 degrees.",
            "rise": "First 0.2 s continuously in posture; times refer to post-step poses.",
            "fall": "After rise, pelvis below 0.50 m or incline above 60 degrees for 0.2 s. Recovery requires 0.2 s continuously in posture.",
            "longest_supported": "Longest consecutive posture interval with combined foot load at least 0.5 BW in at least 90% of its samples.",
            "final": "Entire final 3 s continuously in posture; combined foot load at least 0.5 BW for at least 90% of those samples; no physical termination.",
            "drift": "Maximum and net horizontal pelvis displacement from the first sample in the indicated interval. Distances are retained in cm.",
            "after_settling": "Horizontal pelvis excursion from one second after onset of the first sustained rise through the final valid frame.",
            "limb": "Optional enabled hand (wrist-yaw) and proximal knee collision geometry within 2 cm of the floor for 0.2 s after confirmed rise. Geometric proximity is not a force measurement.",
            "limb_load": "Measured normal force on each wrist or knee-link collision body. Sustained means at least 0.05 body weights for 0.2 s. Knee-link loads include its shank geometry.",
        },
        summary=summary,
        trials=trials,
    )
