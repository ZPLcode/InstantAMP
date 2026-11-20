"""Export a local SuperMimic G1 actor or physically track a timed 50 Hz clip with it."""

import argparse
import json
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    # Small matrix-vector batches spend considerably more time synchronizing a
    # full CPU than computing their results. Allow an explicit BLAS setting.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np


def export(source, output):
    # Export runs in the source PyTorch environment. Inference only requires InstantAMP.
    import torch

    with torch.serialization.safe_globals([type(Path("."))]):
        ckpt = torch.load(source, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    if cfg.get("robot", "g1") != "g1" or cfg["policy"]["goal_dim"] != 689:
        raise ValueError("Expected an anchor-conditioned G1 SuperMimic policy")
    weights = {name: value.detach().cpu().numpy() for name, value in ckpt["policy"].items()}
    output.parent.mkdir(exist_ok=True, parents=True)
    with output.open("xb") as stream:
        np.savez(stream, **weights)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="task", required=True)
    e = sub.add_parser("export", help="Convert the source PyTorch checkpoint to a portable policy NPZ")
    e.add_argument("checkpoint", type=Path)
    e.add_argument("output", type=Path)
    r = sub.add_parser("run", help="Simulate the selected reference interval")
    r.add_argument("policy", type=Path)
    r.add_argument("clip", type=Path)
    r.add_argument("--start", type=int, default=0, help="First 50 Hz reference frame (inclusive)")
    r.add_argument("--end", type=int, help="Last 50 Hz reference frame (exclusive)")
    r.add_argument("--seconds", type=float, help="Rollout duration; defaults to the selected reference duration")
    r.add_argument("--legacy-contacts", action="store_true", help="Use the original G1 normal-only palm and knee contacts")
    r.add_argument("--out", type=Path, required=True, help="Output NPZ with physics states, reference, landmarks and controls")
    feedback = sub.add_parser("feedback", help="Run an exported 50 Hz clip feedback policy in native MuJoCo")
    feedback.add_argument("policy", type=Path)
    feedback.add_argument("case", help="Clip name in the exported policy")
    feedback.add_argument("--out", type=Path, required=True, help="Save the simulated qpos trajectory")
    args = p.parse_args()
    if args.task == "export":
        export(args.checkpoint, args.output)
        print(args.output)
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if args.task == "feedback":
        from instantamp.sim import build_scene
        from instantamp.timed_feedback import TimedFeedbackPolicy
        from tools.export_g1_tracking_policy import replay
        policy = TimedFeedbackPolicy.load(args.policy)
        if args.case not in policy.names:
            feedback.error(f"Choose a clip from {', '.join(policy.names)}")
        model = build_scene(g1_support_friction=policy.support_friction)
        qpos = replay(model, policy, policy.names.index(args.case))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("xb") as stream:
            np.savez_compressed(stream, qpos=qpos, fps=50, name=args.case)
        print(args.out)
        return
    from instantamp import env
    from instantamp.reference_tracker import Motion, ReferencePolicy, track
    motion = Motion.load(args.clip, args.start, args.end)
    steps = len(motion.qpos) - 1 if args.seconds is None else round(args.seconds / env.DT)
    if args.out.exists():
        raise FileExistsError(f"Output already exists at {args.out}")
    result = track([motion], ReferencePolicy(args.policy), steps, support_friction=not args.legacy_contacts)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    with args.out.open("xb") as stream:
        np.savez_compressed(stream, **{name: values[0] for name, values in result.items()})
    print(json.dumps({"out": str(args.out), "clip": str(args.clip), "dt": env.DT, "steps": steps,
                      "support_friction": not args.legacy_contacts}))


if __name__ == "__main__":
    main()
