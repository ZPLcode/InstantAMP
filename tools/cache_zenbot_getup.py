"""Cache deterministic native motor plans from the same controller for identical standard reset states."""

import argparse
import json
from pathlib import Path

import numpy as np


def pack(checkpoint, plans, out):
    with np.load(checkpoint) as z:
        fields = dict(z)
    with_paths = sorted(plans.glob("*.npz"))
    if not with_paths:
        raise ValueError(f"No physical plans in {plans}")
    keys = ("state", "lags", "source", "score", "parameters", "q", "v", "action")
    collected = []
    initial = set()
    for path in with_paths:
        with np.load(path) as z:
            run = {k: z[k] for k in keys}
        n = len(run["action"])
        if n < 1 or run["q"].shape != (n, 30) or run["v"].shape != (n, 29) or run["action"].shape != (n, 23):
            raise ValueError(f"Inconsistent measured motor-plan length in {path}")
        if run["state"].shape != (59,) or run["lags"].shape != (23,) or not all(np.isfinite(a).all() for a in run.values()):
            raise ValueError(f"Invalid initial state or motor group delay in {path}")
        state = (run["state"].astype("d").tobytes(), run["lags"].astype("i").tobytes())
        if state in initial:
            raise ValueError(f"Repeated physical starting state and delays in {path}")
        initial.add(state)
        collected.append(run)
    for key in ("q", "v", "action"):
        fields["zenbot_tracking/cache_" + key] = np.concatenate([r[key] for r in collected])
    for key in ("state", "lags", "source", "score", "parameters"):
        fields["zenbot_tracking/cache_" + key] = np.stack([r[key] for r in collected])
    fields["zenbot_tracking/cache_offsets"] = np.r_[0, np.cumsum([len(r["q"]) for r in collected])]
    (out / "ckpt").mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "ckpt/0000.npz", **fields)
    config = json.loads((checkpoint.parents[1] / "config.json").read_text())
    config.update(cache_source=str(checkpoint), matched_nominal_plans=len(collected))
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "plans.json").write_text(json.dumps(dict(checkpoint=str(checkpoint), plans=[str(p) for p in with_paths]), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pack(args.checkpoint, args.plans, args.out)
