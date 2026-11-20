"""Slice complete G1 get-ups from the six 50 Hz LAFAN1 takes in tools/download_data.sh.

Run from the repository root with python tools/prepare_g1_rises.py. Each output NPZ
retains the source joint order, velocities and coordinate conventions. Frame ranges
in the manifest are zero-based at 50 Hz, with exclusive end indices.
"""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np


FPS = 50
HELDOUT_TAKE = "fallAndGetUp1_subject1"


def rise_intervals(height, quat_xyzw, vertical_speed, angular_velocity):
    """Return (start, established upright stance, end) frame indices."""
    tilt_cosine = 1 - 2 * (quat_xyzw[:, 0] ** 2 + quat_xyzw[:, 1] ** 2)
    upright = (height >= 0.70) & (tilt_cosine >= np.cos(np.deg2rad(30)))
    boundaries = np.flatnonzero(np.diff(np.r_[False, upright, False])).reshape(-1, 2)
    previous_end = 0
    for stand, end in boundaries:
        # Ignore brief passages through the stance threshold during rolls or falls.
        if end - stand < 20:
            continue
        lower = previous_end
        previous_end = end
        approach = height[lower:stand]
        if len(approach) < 20 or approach.min() > 0.35:
            continue
        # The final low, slow pose precedes the lift. Earlier low poses can belong
        # to the fall, an extended pause, or a separate attempt in the same take.
        near_floor = approach <= approach.min() + 0.09
        slow = (np.abs(vertical_speed[lower:stand]) < 0.18) & (np.linalg.norm(angular_velocity[lower:stand], axis=1) < 2)
        settled = np.flatnonzero(near_floor & slow)
        if not len(settled):
            continue
        start = max(lower, lower + settled[-1] - 8)  # retain the lead-in to limb movement
        if height[start] < 0.42 and 0.35 * FPS <= stand - start <= 5 * FPS:
            yield int(start), int(stand), int(min(end, stand + 25))


def prepare(source, destination, heldout_take=HELDOUT_TAKE):
    paths = sorted(source.glob("fallAndGetUp*.npz"))
    if not any(path.stem == heldout_take for path in paths):
        raise ValueError(f"Held-out take {heldout_take} is missing from {source}")
    schema = {"root_pos": 3, "root_quat": 4, "root_lin_vel": 3, "root_ang_vel": 3, "dof_pos": 29, "dof_vel": 29}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        bank = Path(temp) / "bank"
        splits = {"train": [], "heldout": []}
        for split in splits:
            (bank / split).mkdir(parents=True)
        for path in paths:
            with np.load(path) as data:
                if float(data["fps"]) != FPS:
                    raise ValueError(f"{path} must be sampled at {FPS} Hz")
                motion = {key: data[key] for key in schema}
            n = len(motion["root_pos"])
            if any(arr.shape != (n, schema[key]) or not np.isfinite(arr).all() for key, arr in motion.items()):
                raise ValueError(f"Expected finite 29-DoF G1 motion in {path}")
            ranges = rise_intervals(motion["root_pos"][:, 2], motion["root_quat"], motion["root_lin_vel"][:, 2], motion["root_ang_vel"])
            split = "heldout" if path.stem == heldout_take else "train"
            for start, stand, end in ranges:
                name = f"{path.stem}_{start:05d}_{end:05d}.npz"
                np.savez_compressed(bank / split / name, fps=np.int64(FPS), **{k: v[start:end] for k, v in motion.items()})
                splits[split].append({"file": name, "source_take": path.stem, "start_frame": start, "stand_frame": stand, "end_frame": end})
        for split, clips in splits.items():
            if not clips:
                raise ValueError(f"No complete {split} get-ups found in {source}")
            manifest = {"kind": "g1_rise_clips", "fps": FPS, "heldout_take": heldout_take, "clips": clips}
            (bank / split / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if destination.exists():
            files = sorted(path.relative_to(bank) for path in bank.rglob("*") if path.is_file())
            existing = sorted(path.relative_to(destination) for path in destination.rglob("*") if path.is_file())
            if existing != files or any((bank / file).read_bytes() != (destination / file).read_bytes() for file in files):
                raise FileExistsError(f"Different output already exists at {destination}; choose a new --dst")
        else:
            bank.rename(destination)
    return {split: len(clips) for split, clips in splits.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/recovery/lafan1"))
    parser.add_argument("--dst", type=Path, default=Path("data/recovery/g1_rises"))
    parser.add_argument("--heldout-take", default=HELDOUT_TAKE)
    args = parser.parse_args()
    print(prepare(args.src, args.dst, args.heldout_take))
