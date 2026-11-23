"""Find each get-up plan's first sustained foot-supported state for phase resets."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


def first_support_frame(physical):
    """Return the state index at the beginning of ten consecutive 50 Hz supported steps."""
    if physical.ndim != 2 or physical.shape[1] != 5 or len(physical) < 10 or not np.isfinite(physical).all():
        raise ValueError("Expected at least ten finite native height, tilt and contact-force records")
    height, up_cos, feet_bw, _, nonfoot_bw = physical.T
    supported = (height > 0.58) & (height <= 0.78) & (up_cos > 0.8660254) & (feet_bw >= 0.5) & (nonfoot_bw < 0.1)
    first = np.flatnonzero(np.convolve(supported, np.ones(10, dtype=int), mode="valid") == 10)
    if not len(first):
        raise ValueError("The physical plan never reaches ten consecutive foot-supported frames")
    # Contact record t is measured during action t and describes the resulting state q[t + 1].
    return int(first[0] + 1)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def support_frames(ground_bank, plans):
    with np.load(ground_bank, allow_pickle=False) as bank:
        names = bank["clip_names"].tolist()
        digests = bank["plan_sha256"].tolist()
        lengths = bank["lengths"].tolist()
    manifest = json.loads((plans / "manifest.json").read_text())
    paths = {entry["sha256"]: plans / entry["filename"] for entry in manifest["trajectories"].values()}
    frames = []
    for name, digest, length in zip(names, digests, lengths, strict=True):
        path = paths.get(digest)
        if path is None or sha256(path) != digest:
            raise ValueError(f"Missing or changed physical plan for {name}")
        with np.load(path, allow_pickle=False) as plan:
            if str(plan["source_clip"]) != name or not np.isclose(float(plan["dt"]), 0.02):
                raise ValueError(f"Physical plan source or time step differs for {name}")
            frame = first_support_frame(plan["physical"][:length])
        if not 0 <= frame < length:
            raise ValueError(f"Support frame falls outside the ground bank for {name}")
        frames.append(frame)
    return {"ground_bank_sha256": sha256(ground_bank), "clip_names": names, "first_support": frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-bank", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plans", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/zenbot23_physical_recovery")
    args = parser.parse_args()
    result = support_frames(args.ground_bank, args.plans)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=args.out.parent, delete=False) as temp:
        json.dump(result, temp, indent=2)
        temp.write("\n")
    Path(temp.name).replace(args.out)
    print(f"Wrote {len(result['first_support'])} 50 Hz support frames to {args.out}")


if __name__ == "__main__":
    main()
