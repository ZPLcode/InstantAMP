"""Add successful physical get-up examples to the native Zenbot planner's fallback search."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from instantamp.zenbot_native_tracking import features


def augment(checkpoint, directories, out, on_exhaustion=False):
    with np.load(checkpoint) as archive:
        fields = dict(archive)
    prefix = "zenbot_tracking/"
    db = {k[len(prefix):]: v for k, v in fields.items() if k.startswith(prefix)}
    per_tick = ("q", "v", "u", "K", "desired_palm_load")
    per_motion = ("delay", "look_gain", "quality", "support_style", "reference_style")
    tables = {k: [db[k]] for k in (*per_tick, *per_motion)}
    offsets = list(db["offset"])
    primary_count = int(db.get("primary_count", len(offsets) - 1))
    seen, sources = set(), []
    for folder in directories:
        for path in sorted(folder.glob("*.json")):
            if not path.with_suffix(".npz").exists():
                continue
            row = json.loads(path.read_text())
            track = row.get("tracking", {})
            if not row.get("held") or row.get("refall") or not track.get("planned_ticks"):
                continue
            if row.get("pre_rise_attempts", {}).get("retries"):
                continue
            with np.load(path.with_suffix(".npz")) as z:
                if "action" not in z:
                    continue
                n = min(track["planned_ticks"] + 26, len(z["action"]))
                q = z["qpos"][:n]
                v = z["qvel"][:n]
                u = z["action"][:n].astype(float)
            delays = np.random.default_rng(row["seed"]).integers(3, 8, (17, 5))[row["trial"]]
            state = np.r_[q[0], v[0], delays].astype("d").tobytes()
            if state in seen:
                continue
            seen.add(state)
            previous = np.vstack([np.zeros(23), u[:-1]])
            state_features = np.stack([features(*frame) for frame in zip(q, v, previous)])
            # The beam applies the trained residual online. Remove it from recorded
            # motor commands to retain the same nominal command at the measured state.
            nominal = u - db["scale"] * np.tanh(state_features @ db["actor"])
            source = track["matched"]
            lo, hi = db["offset"][source:source + 2]
            idx = np.minimum(lo + np.arange(n), hi - 1)
            tables["q"].append(q)
            tables["v"].append(v)
            tables["u"].append(nominal)
            tables["K"].append(db["K"][idx])
            tables["desired_palm_load"].append(db["desired_palm_load"][idx])
            tables["delay"].append(delays[None])
            for k in ("look_gain", "support_style", "reference_style"):
                tables[k].append(db[k][source:source + 1])
            tables["quality"].append(np.array([row["drift_cm"] / 100.0]))
            offsets.append(offsets[-1] + n)
            sources.append(str(path))
    if not sources:
        raise ValueError("No successful trajectories with recorded actions")
    fields.update({prefix + k: np.concatenate(v) for k, v in tables.items()})
    fields[prefix + "offset"] = np.asarray(offsets)
    fields[prefix + "primary_count"] = np.asarray(primary_count)
    fields[prefix + "terminal_height_margin"] = np.asarray(0.015)
    fields[prefix + "impulse_control"] = np.asarray(True)
    if on_exhaustion:
        fields[prefix + "expansion_start"] = np.asarray(len(db["offset"]) - 1)
    (out / "ckpt").mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "ckpt/0000.npz", **fields)
    shutil.copyfile(checkpoint.parents[1] / "config.json", out / "config.json")
    (out / "training_sources.json").write_text(json.dumps(dict(base=str(checkpoint), training_examples=sources), indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rollouts", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--on-exhaustion", action="store_true", help="Query the added examples only if all original searches failed")
    args = p.parse_args()
    augment(args.checkpoint, args.rollouts, args.out, args.on_exhaustion)
