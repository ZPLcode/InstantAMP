"""Evaluate saved scratch-training policies at fixed milestones in a separate CPU process."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MILESTONES = (0, 250, 500, 1000, 2000, 4000, 6000, 8000, 10000)
WM_MILESTONES = (1000, 3000, 6000, 10000)
FINAL_SEEDS = (932277, 1654679, 2718869, 3950477, 5969183)


def ready(checkpoint, iteration):
    # save_policy atomically renames the NPZ, then writes the small iteration JSON.
    if not checkpoint.is_file():
        return False
    try:
        metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return metadata.get("it") == iteration


def wm_checkpoint(run, iteration):
    """Find a WM policy by the learner's saved iteration. A live writer may still be writing a sidecar."""
    matches = []
    for sidecar in (run / "ckpt").glob("*.json"):
        try:
            info = json.loads(sidecar.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if info.get("it") == iteration:
            matches.append(sidecar.with_suffix(".npz"))
    if len(matches) > 1:
        raise ValueError(f"Multiple policy sidecars report iteration {iteration} under {run / 'ckpt'}")
    return matches[0] if matches and matches[0].is_file() else None


def evaluate(checkpoint, output, seeds, *, immediate_motors=False):
    # A one-shot rescore and a live monitor can reach the same milestone at the same time.
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _evaluate(checkpoint, output, seeds, immediate_motors=immediate_motors)


def _evaluate(checkpoint, output, seeds, *, immediate_motors):
    if output.is_file():
        previous = json.loads(output.read_text())
        if previous.get("immediate_motors", False) != immediate_motors:
            raise ValueError(f"Motor-delay mode differs from existing {output}; use a separate evaluation directory")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if previous["sha256"] == digest and previous["seeds"] == list(seeds):
            return
    temporary = output.with_suffix(".pending")
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    command = [sys.executable, "-m", "tools.evaluate_zenbot_scratch", str(checkpoint), "--out", str(temporary),
               "--seeds", *map(str, seeds)]
    if immediate_motors:
        command.append("--immediate-motors")
    subprocess.run(command, check=True, env=env)
    temporary.replace(output)
    print(f"Scored {checkpoint.name} using {len(seeds)} seed(s): {output}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, help="Defaults to a separate subdirectory of RUN for each motor-delay mode")
    p.add_argument("--milestones", type=int, nargs="+")
    p.add_argument("--wm", action="store_true", help="Use WM sidecar iterations; default milestones 1000, 3000, 6000, 10000")
    p.add_argument("--available", action="store_true", help="Score saved checkpoints now and return without waiting for later ones")
    p.add_argument("--immediate-motors", action="store_true", help="Run every evaluation with all five motor delays zero")
    args = p.parse_args()
    default_out = args.run / "milestones"
    immediate_out = args.run / "milestones_immediate_motors"
    out = args.out or (immediate_out if args.immediate_motors else default_out)
    if out.resolve() == (default_out if args.immediate_motors else immediate_out).resolve():
        p.error("The two motor-delay modes must use separate evaluation directories")
    out.mkdir(exist_ok=True, parents=True)
    milestones = args.milestones or (WM_MILESTONES if args.wm else MILESTONES)
    if sorted(set(milestones)) != list(milestones) or milestones[0] < 0:
        p.error("Use increasing nonnegative checkpoint iterations")
    if not args.wm and milestones[-1] != 10000:
        p.error("The scratch PPO benchmark ends at iteration 10000")
    for iteration in milestones:
        while True:
            if args.wm:
                checkpoint = wm_checkpoint(args.run, iteration)
            else:
                path = args.run / "ckpt" / f"{iteration:06}.npz"
                checkpoint = path if ready(path, iteration) else None
            if checkpoint is not None or args.available:
                break
            time.sleep(15)
        if checkpoint is None:
            print(f"Iteration {iteration} has no saved policy yet", flush=True)
            continue
        evaluate(checkpoint, out / f"{iteration:06}.json", (42,), immediate_motors=args.immediate_motors)
        if iteration == 10000 and not args.wm:
            evaluate(checkpoint, out / "010000_five_seeds.json", FINAL_SEEDS, immediate_motors=args.immediate_motors)


if __name__ == "__main__":
    main()
