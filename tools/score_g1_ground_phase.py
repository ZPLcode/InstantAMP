"""Compare palm loading before knee extension and real-time get-up speed to each source motion."""

import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np

from instantamp import env
from instantamp.g1_ground_phase import (
    HandContacts,
    knee_extension_frame,
    sustained_palm,
)
from instantamp.stability import contiguous
from tools.compare_recovery_rollouts import first_upright
from tools.score_g1_tracking import load_native_reference


def profile(q, palm, dt, velocity):
    up = first_upright(q)
    end = len(q) if up is None else up + 1
    knee = knee_extension_frame(q, up)
    limit = end if knee is None else min(knee, end)
    onset = sustained_palm(palm, limit)
    mask = palm[:limit]
    spans = [(a, b) for a, b in contiguous(mask) if b - a >= round(0.12 / dt)]
    n = max(2, end)
    knee_speed = np.rad2deg(np.maximum(0.0, -velocity[:n, [9, 15]].mean(axis=1)))
    return dict(
        stand_s=None if up is None else up * dt,
        extension_onset_s=None if knee is None else knee * dt,
        extension_to_stand_s=None
        if up is None or knee is None or up <= knee
        else (up - knee) * dt,
        sustained_palm_before_extension=onset is not None,
        palm_onset_s=None if onset is None else onset * dt,
        palm_before_extension_s=sum(b - a for a, b in spans) * dt,
        pelvis_vertical_speed_p95_m_s=float(
            np.percentile(np.maximum(0.0, velocity[:n, 2]), 95)
        ),
        knee_extension_speed_p95_deg_s=float(np.percentile(knee_speed, 95)),
    )


def score(archives, motions, scale):
    cases = {}
    fingerprint = None
    for file in archives:
        with np.load(file) as z:
            if (
                float(z["assist_n"]) != 0.0
                or bool(z["automatic_resets"])
                or z["terminated"].any()
            ):
                raise ValueError("Expected continuous unassisted physical rollouts")
            digest = str(z["checkpoint_sha256"])
            if fingerprint not in (None, digest):
                raise ValueError("Mixed checkpoints")
            fingerprint = digest
            model = mujoco.MjModel.from_binary_path(
                str(file.parent / str(z["physics_file"]))
            )
            hands = HandContacts(model)
            q, v = z["qpos"], z["qvel"]
            if "hand_load_bw" in z:
                force = z["hand_load_bw"][:, :, :, 0]
            else:
                names = z["support_body_names"].tolist()
                force = z["support_load_bw"][
                    :,
                    :,
                    [
                        names.index(f"{side}_wrist_yaw_link")
                        for side in ("left", "right")
                    ],
                ]
            for j, (clip, offset) in enumerate(zip(z["clips"], z["offsets_s"])):
                key = (
                    str(clip),
                    float(offset),
                    hashlib.sha256(
                        np.r_[q[0, j], v[0, j]].astype("<f8").tobytes()
                    ).hexdigest(),
                )
                if key in cases:
                    continue
                dt = float(z["dt"])
                start = round(offset / dt)
                ref, rv = load_native_reference(motions / str(clip))
                ref, rv = ref[start:], rv[start:]
                near = (hands.reference_clearances(ref)[:, :, 0] < 0.065).any(axis=1)
                source = profile(ref, near, dt, rv)
                loaded = np.r_[False, force[:, j].sum(axis=1) >= 0.07]
                actual = profile(q[:, j], loaded, dt, v[:, j])
                coverage = pose_error = None
                if (
                    actual["extension_onset_s"] is not None
                    and source["extension_onset_s"] is not None
                ):
                    ref_end = round(source["extension_onset_s"] / dt)
                    real_end = round(actual["extension_onset_s"] / dt)
                    if ref_end and real_end:
                        observed = np.rint(
                            np.linspace(0, real_end, ref_end + 1)
                        ).astype(int)
                        # Allow 40 ms for contacts to transfer between the two palms.
                        nearby = np.clip(
                            observed[:, None] + np.arange(-2, 3), 0, real_end
                        )
                        expected = near[: ref_end + 1]
                        if expected.any():
                            coverage = float(
                                loaded[nearby].any(axis=1)[expected].mean()
                            )
                        pose_error = float(
                            np.degrees(
                                np.sqrt(
                                    np.square(
                                        q[observed, j, 7:] - ref[: ref_end + 1, 7:]
                                    ).mean()
                                )
                            )
                        )

                def ratio(key):
                    return (
                        actual[key] / source[key]
                        if actual[key] is not None and source[key] and source[key] > 0
                        else None
                    )

                cases[key] = dict(
                    clip=str(clip),
                    offset_s=float(offset),
                    reset_sha256=key[-1],
                    source=source,
                    actual=actual,
                    rise_duration_ratio=ratio("stand_s"),
                    extension_duration_ratio=ratio("extension_to_stand_s"),
                    source_requires_palm=source["sustained_palm_before_extension"],
                    reference_ground_palm_coverage=coverage,
                    ground_joint_rmse_deg=pose_error,
                    mean_palm_load_before_extension_bw=None
                    if actual["extension_onset_s"] is None
                    else float(
                        force[: max(1, round(actual["extension_onset_s"] / dt)), j]
                        .sum(axis=1)
                        .mean()
                    ),
                )
    rows = list(cases.values())
    needed = [r for r in rows if r["source_requires_palm"]]

    def median(rows, name, domain=None):
        values = [r[domain][name] if domain else r[name] for r in rows]
        finite = [v for v in values if v is not None]
        return float(np.median(finite)) if finite else None

    return dict(
        checkpoint_sha256=fingerprint,
        definition="Reference palm distance below 6.5 cm after ground alignment. Physical palms together >=0.07 BW for >=0.12 s before sustained bilateral 20-degree knee extension. Time in simulation seconds.",
        duration_target=scale,
        unique_starts=len(rows),
        rose=sum(r["actual"]["stand_s"] is not None for r in rows),
        reference_palm_required=len(needed),
        required_with_physical_palm_before_extension=sum(
            r["actual"]["sustained_palm_before_extension"] for r in needed
        ),
        median_reference_ground_palm_coverage=median(
            needed, "reference_ground_palm_coverage"
        ),
        median_ground_joint_rmse_deg=median(needed, "ground_joint_rmse_deg"),
        median_palm_duration_s=median(needed, "palm_before_extension_s", "actual"),
        median_reference_palm_duration_s=median(
            needed, "palm_before_extension_s", "source"
        ),
        median_rise_duration_ratio=median(rows, "rise_duration_ratio"),
        median_extension_duration_ratio=median(rows, "extension_duration_ratio"),
        at_or_above_target_duration=sum(
            r["rise_duration_ratio"] is not None and r["rise_duration_ratio"] >= scale
            for r in rows
        ),
        median_pelvis_vertical_speed_p95_m_s=median(
            rows, "pelvis_vertical_speed_p95_m_s", "actual"
        ),
        median_knee_extension_speed_p95_deg_s=median(
            rows, "knee_extension_speed_p95_deg_s", "actual"
        ),
        cases=rows,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archives", nargs="+", type=Path)
    p.add_argument("--motions", type=Path, default=Path("data/g1_getup/recovery"))
    p.add_argument("--duration-target", type=float, default=1.3)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = score(args.archives, args.motions, args.duration_target)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print({k: v for k, v in result.items() if k != "cases"})


if __name__ == "__main__":
    main()
