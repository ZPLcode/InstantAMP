"""Measure quiet ground pauses and time to rise for each real Zenbot motion opening."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.evaluate_zenbot_getup import DT, evaluate
from tools.play_zenbot_getup import ZenbotPlayer
from tools.score_zenbot_sequence import ground_timing


def summarize(rows, horizon):
    """Keep non-rises in the restricted mean and quantiles by scoring their full observation time."""
    rise = np.array([horizon if r["rise"] is None else r["rise"] for r in rows])
    quiet = [r["ground_timing"]["longest_quiet_ground_s"] for r in rows]
    lift = [r["ground_timing"]["first_pelvis_lift_s"] for r in rows]
    timing_ratio = [r["rise"] / r["reference_rise_s"] for r in rows if r["rise"] is not None and r["reference_rise_s"]]
    return dict(n=len(rows), rose=sum(r["rise"] is not None for r in rows),
                rose_by_ref_plus_1_5s=sum(r["rise"] is not None and r["reference_rise_s"] is not None
                                         and r["rise"] <= r["reference_rise_s"] + 1.5 for r in rows),
                held_first_attempt=sum(r["rise"] is not None and r["held"] and not r["refall"]
                                       and not r["pre_rise_attempts"]["retries"] for r in rows),
                rise_restricted_mean_s=float(np.mean(rise)), rise_capped_p90_s=float(np.percentile(rise, 90)),
                median_rise_to_reference_ratio_on_success=float(np.median(timing_ratio)) if timing_ratio else None,
                first_lift_capped_p90_s=float(np.percentile([horizon if v is None else v for v in lift], 90)),
                longest_quiet_ground_p90_s=float(np.percentile(quiet, 90)), longest_quiet_ground_max_s=max(quiet))


def reference_ground_timing(path, rise_s):
    with np.load(path, allow_pickle=False) as clip:
        root = clip["root_quat"][:, [3, 0, 1, 2]]
        q = np.column_stack((clip["root_pos"], root, clip["dof_pos"]))
        v = np.column_stack((clip["root_lin_vel"], clip["root_ang_vel"], clip["dof_vel"]))
    return ground_timing(q, v, None if rise_s is None else round(rise_s / DT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--seconds", type=float, default=12.)
    p.add_argument("--reference", type=Path, help="Override the motion opening directory when comparing teachers")
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 2. or len(set(args.seeds)) != len(args.seeds):
        p.error("Provide distinct seeds and an observation window of at least two seconds")
    horizon = round(args.seconds / DT) * DT
    player = ZenbotPlayer(args.checkpoint, args.reference, visuals=False)
    openings = [i for i, (_, offset) in enumerate(player.names) if offset == 0.]
    rows = []
    for seed in args.seeds:
        for trial in openings:
            row, _ = evaluate(player, trial, seed, round(horizon / DT), -1)
            rows.append(row)
        print(f"Seed {seed} complete", flush=True)
    report = dict(checkpoint=str(args.checkpoint), sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  reference=str(player.reference.resolve()),
                  seeds=args.seeds, horizon_s=horizon, summary=summarize(rows, horizon), openings={}, cases=rows)
    for i in openings:
        subset = [row for row in rows if row["trial"] == i]
        name = player.names[i][0]
        source_rise = subset[0]["reference_rise_s"]
        report["openings"][name] = dict(trial=i, reference_rise_s=source_rise,
                                       reference_ground_timing=reference_ground_timing(player.reference / name, source_rise),
                                       **summarize(subset, horizon))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
