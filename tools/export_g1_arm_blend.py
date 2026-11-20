"""Package two averaged G1 actors as a deployable arm-only recovery policy.

python tools/export_g1_arm_blend.py BASE.npz STYLE.npz --out runs/g1_arm_recovery
python tools/render.py runs/g1_arm_recovery --out runs/g1_arm_recovery/preview
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instantamp import env
from instantamp.inference import load


def export(base, style, out, arm_weight):
    if not 0 <= arm_weight <= 1:
        raise ValueError("Arm weight must be in [0, 1]")
    if out.exists():
        raise FileExistsError(f"Choose a new output run directory: {out}")
    weights = np.array([arm_weight if any(part in joint for part in ("shoulder", "elbow", "wrist")) else 0.
                        for joint in env.DOF], np.float32)
    bundle = {"blend": weights, "gate": np.array(True)}
    configs = []
    for name, path in (("base", base), ("style", style)):
        config = json.loads((path.parents[1] / "config.json").read_text())
        if config.get("robot", "g1") != "g1":
            raise ValueError(f"Expected a G1 run: {path}")
        configs.append(config)
        with np.load(path, allow_pickle=False) as actor:
            for key in actor.files:
                if key.startswith(("avg/", "obs_norm/", "ode/")):
                    bundle[f"{name}/{key}"] = actor[key]
    if configs[0].get("obs_clip", 10.) != configs[1].get("obs_clip", 10.):
        raise ValueError("Both actors must use the same observation clip")
    metadata = json.loads(style.with_suffix(".json").read_text())
    metadata.update(policy_type="g1_gated_arm_blend", base=str(base), style=str(style), arm_weight=arm_weight)
    configs[0].update(log_dir=str(out), policy_type=metadata["policy_type"])
    out.mkdir(parents=True)
    try:
        directory = out / "ckpt"
        directory.mkdir()
        checkpoint = directory / "best.npz"
        np.savez(checkpoint, **bundle)
        load(checkpoint)
        checkpoint.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        (out / "config.json").write_text(json.dumps(configs[0], indent=2) + "\n")
    except Exception:
        shutil.rmtree(out)
        raise
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("style", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm-weight", type=float, default=.2)
    args = parser.parse_args()
    print(export(args.base, args.style, args.out, args.arm_weight))


if __name__ == "__main__":
    main()
