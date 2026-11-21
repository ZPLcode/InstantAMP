"""Fit a quiet-stance handoff from physically tested G1 landings and package it with the rise actor."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from instantamp.g1_capture import CaptureGate
from instantamp.inference import load


def fit(rows):
    features = np.asarray([row["feat"] for row in rows], dtype=np.float64)
    success = np.asarray([row["ok"] for row in rows], dtype=bool)
    if features.shape != (len(rows), 14) or not np.isfinite(features).all() or len(np.unique(success)) != 2:
        raise ValueError("Expected finite 14-feature landings with positive and negative outcomes")
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), .02)
    design = np.column_stack((np.clip((features - mean) / scale, -5., 5.), np.ones(len(rows))))

    def objective(weights):
        logits = design @ weights
        loss = (np.logaddexp(0., logits) - success * logits).sum() + .8 * (weights[:-1] ** 2).sum()
        grad = design.T @ (expit(logits) - success) + 1.6 * np.r_[weights[:-1], 0.]
        return loss, grad

    solution = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B")
    if not solution.success:
        raise ValueError(solution.message)
    return dict(mean=mean, scale=scale, weight=solution.x[:-1], bias=solution.x[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True, help="Existing three-stage G1 release directory")
    parser.add_argument("--rise-amp", type=Path, help="Full checkpoint exported by PPO and AMP after refining this controller")
    parser.add_argument("--labels", type=Path, nargs="+", required=True, help="JSONL physically tested handoff outcomes")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quiet-pose", type=Path, help="Seven-output quiet stance policy fitted with PPO and AMP")
    parser.add_argument("--threshold", type=float, default=.82)
    parser.add_argument("--confirm-frames", type=int, default=3)
    parser.add_argument("--blend-frames", type=int, default=20)
    parser.add_argument("--velocity-limits", type=float, nargs=4, metavar=("ENTRY_XY", "ENTRY_Z", "RELEASE_XY", "RELEASE_Z"),
                        help="Maximum sensor-derived planar and vertical support speeds during acquisition and release, in m/s")
    args = parser.parse_args()
    rows = []
    for path in sorted(args.labels):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    fitted = fit(rows)
    fitted["settings"] = np.array([args.threshold, args.confirm_frames, args.blend_frames, .81, 4., *(args.velocity_limits or [])], np.float32)
    checkpoint = args.controller / "ckpt" / "0000.npz"
    active = args.rise_amp or checkpoint
    with np.load(active, allow_pickle=False) as base:
        archive = dict(base)
    if args.rise_amp and "local_rise/residual/scale" not in archive:
        raise ValueError("Expected a trained local rise residual in the AMP checkpoint")
    archive.update({"capture_gate/" + name: np.asarray(value, dtype=np.float32) for name, value in fitted.items()})
    archive["quiet_stance"] = "sensor_wbc_v1"
    if args.quiet_pose:
        with np.load(args.quiet_pose, allow_pickle=False) as pose:
            archive["quiet_pose/weight"] = pose["weight"]
    CaptureGate.read(archive)
    meta = json.loads(checkpoint.with_suffix(".json").read_text())
    meta["quiet_stance"] = dict(kind="sensor_wbc_v1", capture_examples=len(rows), successes=sum(r["ok"] for r in rows),
                                threshold=args.threshold, confirmation_frames=args.confirm_frames,
                                blend_frames=args.blend_frames, base_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    if args.rise_amp:
        training = json.loads((active.parent.parent / "config.json").read_text())
        original = Path(training["base"])
        if not original.is_absolute():
            original = Path(__file__).resolve().parents[1] / original
        # A continued AMP run may start from an already refined residual. Verify all frozen
        # actors, observation calibration, local examples and routing against the declared base.
        with np.load(original, allow_pickle=False) as initial, np.load(checkpoint, allow_pickle=False) as declared:
            fixed_keys = {k for k in initial if not k.startswith("local_rise/residual/")}
            other_keys = {k for k in declared if not k.startswith("local_rise/residual/")}
            if fixed_keys != other_keys or any(not np.array_equal(initial[k], declared[k]) for k in fixed_keys):
                raise ValueError("The AMP training run used a different base controller")
        meta["rise_amp"] = {"checkpoint": active.name, "sha256": hashlib.sha256(active.read_bytes()).hexdigest(),
                            "action_rate_multiplier": training.get("action_rate_multiplier", 1.),
                            "seed": training["seed"]}
    if args.velocity_limits:
        meta["quiet_stance"]["velocity_limits_m_s"] = args.velocity_limits
    if args.quiet_pose:
        meta["quiet_stance"]["amp_pose_sha256"] = hashlib.sha256(args.quiet_pose.read_bytes()).hexdigest()
    args.out.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temp:
        run = Path(temp) / "run"
        ckpt = run / "ckpt" / "0000.npz"
        ckpt.parent.mkdir(parents=True)
        np.savez(ckpt, **archive)
        load(ckpt)
        ckpt.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        (run / "config.json").write_bytes((args.controller / "config.json").read_bytes())
        if args.out.exists():
            raise FileExistsError(args.out)
        run.rename(args.out)
    print(args.out / "ckpt" / ckpt.name)


if __name__ == "__main__":
    main()
