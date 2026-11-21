"""Measure when both feet take the load after the last sustained palm contact."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from instantamp.stability import contiguous, first_sustained


def transfer_time(palms, feet, dt, stand_s):
    """Inputs are post-step force samples in body weights; the first sample is at dt."""
    if stand_s is None:
        return None
    stand = round(stand_s / dt)
    duration = max(1, round(0.12 / dt))
    end = min(len(palms), stand + round(0.5 / dt))
    loaded = palms.sum(axis=1)
    spans = [(a, b) for a, b in contiguous(loaded[:end] >= 0.07)
             if a < stand and b - a >= duration]
    if not spans:
        return None
    last_release = spans[-1][1]
    transferred = (feet.min(axis=1) >= 0.05) & (feet.sum(axis=1) >= 0.5) & (loaded < 0.05)
    start = first_sustained(transferred[last_release:end], duration)
    return None if start is None else (last_release + start + 1) * dt


def score(archives, phases):
    rows = {(r["clip"], r["offset_s"], r["reset_sha256"]): r for r in phases["cases"]}
    cases = {}
    for path in archives:
        with np.load(path) as data:
            if str(data["checkpoint_sha256"]) != phases["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint differs from the ground-phase report in {path}")
            names = data["support_body_names"].tolist()
            loads = data["support_load_bw"]
            feet = loads[:, :, [names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]]
            palms = (data["hand_load_bw"][:, :, :, 0] if "hand_load_bw" in data else
                     loads[:, :, [names.index(f"{side}_wrist_yaw_link") for side in ("left", "right")]])
            q, v = data["qpos"][0], data["qvel"][0]
            for j, (clip, offset) in enumerate(zip(data["clips"], data["offsets_s"])):
                reset = hashlib.sha256(np.r_[q[j], v[j]].astype("<f8").tobytes()).hexdigest()
                key = (str(clip), float(offset), reset)
                if key in cases:
                    continue
                row = rows[key]
                actual = row["actual"]
                transfer = transfer_time(palms[:, j], feet[:, j], float(data["dt"]), actual["stand_s"])
                cases[key] = dict(clip=key[0], offset_s=key[1], reset_sha256=reset,
                                  reference_requires_palm=row["source_requires_palm"],
                                  palm_onset_s=actual["palm_onset_s"], leg_extension_s=actual["extension_onset_s"],
                                  foot_transfer_s=transfer, first_stand_s=actual["stand_s"])
    if cases.keys() != rows.keys():
        raise ValueError("The archives must cover every reset in the ground-phase report")
    required = [r for r in cases.values() if r["reference_requires_palm"]]
    return dict(checkpoint_sha256=phases["checkpoint_sha256"],
                definition="After the last sustained palm contact starting before standing, each foot bears at least 5% body weight, feet together at least 50%, and palms less than 5% for 120 ms. Observation ends 0.5 s after first upright.",
                total=len(cases), required=len(required),
                palm_before_extension_and_weight_transferred=sum(
                    r["palm_onset_s"] is not None and r["foot_transfer_s"] is not None for r in required),
                cases=list(cases.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--ground-phase", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.archives, json.loads(args.ground_phase.read_text()))
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print({k: v for k, v in report.items() if k != "cases"})
