"""Measure standing stability in recorded G1 get-up physics rollouts.

python tools/score_getup_stability.py RUN/rollouts/NNNN_avg_source_play_recovery.npz --out report.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instantamp.stability import score

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--skip-geometry", action="store_true", help="score all posture/load/drift metrics without limb FK")
    args = p.parse_args()
    report = score(args.archive, not args.skip_geometry)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))
