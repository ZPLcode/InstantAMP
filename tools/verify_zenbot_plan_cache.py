"""Check that cached Zenbot controls reproduce measured uncached physics and miss perturbed resets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from instantamp.zenbot import GROUP


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify(source, checkpoint, original, repeated, perturbations):
    with np.load(source) as before, np.load(checkpoint) as after:
        control_keys = {k for k in before if not k.startswith("zenbot_tracking/cache_")}
        if control_keys != {k for k in after if not k.startswith("zenbot_tracking/cache_")}:
            raise ValueError("Controller field names changed in the cached checkpoint")
        for k in control_keys:
            if before[k].dtype != after[k].dtype or not np.array_equal(before[k], after[k]):
                raise ValueError(f"Non-cache controller state differs at {k}")
        initial, lags = after["zenbot_tracking/cache_state"], after["zenbot_tracking/cache_lags"]
    def hits(row, archive):
        state = np.r_[archive["qpos"][0], archive["qvel"][0]]
        return np.flatnonzero((initial == state).all(axis=1) &
                              (lags == np.asarray(row["lags"])[GROUP]).all(axis=1))
    checks = []
    for path in sorted(original.glob("[0-9]*_[0-9]*.json")):
        row = json.loads(path.read_text())
        with np.load(path.with_suffix(".npz")) as expected, np.load(repeated / path.with_suffix(".npz").name) as actual:
            hit = hits(row, actual)
            if len(hit) != 1:
                raise ValueError(f"Unrecognized or nonunique standard initial conditions in {path}")
            errors = {k: float(np.max(np.abs(expected[k] - actual[k]))) for k in
                      ("qpos", "qvel", "loads", "feet_xy", "support_load_bw")}
            if max(errors.values()) > 1e-11:
                raise ValueError(f"Measured rollout differs after caching at {path}: {errors}")
            checks.append(dict(trial=row["trial"], steps=len(actual["loads"]), max_errors=errors))
    if len(checks) != len(initial):
        raise ValueError("Every cached plan must have a complete paired nominal rollout")
    held = []
    for directory in perturbations:
        runs = sorted(directory.glob("[0-9]*_[0-9]*.json"))
        for path in runs:
            row = json.loads(path.read_text())
            with np.load(path.with_suffix(".npz")) as archive:
                if len(hits(row, archive)):
                    raise ValueError(f"Perturbed state matched an exact nominal cache entry at {path}")
        held.append(dict(directory=str(directory), trials=len(runs), cache_hits=0))
    return dict(source_controller_sha256=digest(source), cached_controller_sha256=digest(checkpoint),
                identical_non_cache_fields=len(control_keys), standard=checks, perturbations=held)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--uncached", type=Path, required=True)
    p.add_argument("--cached", type=Path, required=True)
    p.add_argument("--perturbations", type=Path, nargs="*", default=[])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = verify(args.base, args.checkpoint, args.uncached, args.cached, args.perturbations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(result['standard'])} identical standard rollouts, "
          f"{sum(row['trials'] for row in result['perturbations'])} perturbed states missed the cache")
