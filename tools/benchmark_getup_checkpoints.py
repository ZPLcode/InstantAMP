"""Evaluate G1 player checkpoints on fixed development, extended and final reset suites.

    python -m tools.benchmark_getup_checkpoints --suite selection --checkpoint live=runs/g1_getup_amp_rate3/ckpt/0000.npz
    python -m tools.benchmark_getup_checkpoints --suite panel --follow runs/g1_getup_amp_rate3_10k/integrated

The panel covers nominal starts and the first development seed. Selection uses all six
development seeds. Long runs exercise the viewer impulse at 15 s over 90 s. The final
suite uses five separately reserved seeds. Keep it for a nominated policy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time


DEVELOPMENT = (4471, 18317, 29041, 41057, 56333, 81173)
FINAL = (90481, 101501, 108707, 119447, 126271)
SUITES = {"selection": DEVELOPMENT, "panel": (None, DEVELOPMENT[0]),
          "long": (None, 138203), "final": FINAL}
PROTOCOLS = ("source-play", "fallen")
WORKER_LIMIT = 2


@contextmanager
def exclusive(path):
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


@contextmanager
def worker_slot(root, limit=WORKER_LIMIT):
    """Share physical rollout worker slots between benchmark invocations."""
    while True:
        for i in range(limit):
            fp = (root / f".worker-{i}.lock").open("a")
            try:
                fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fp.close()
                continue
            try:
                yield
            finally:
                fp.close()
            return
        time.sleep(1)


def seed_name(seed):
    return "nominal" if seed is None else f"seed_{seed}"


def valid(folder, digest, seed, seconds, push, protocols):
    for proto in protocols:
        path = folder / f"{proto}-metrics.json"
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        if (data["checkpoint_sha256"] != digest or data["duration_s"] != seconds
                or data.get("reset_jitter_seed", -1) != (-1 if seed is None else seed)
                or data.get("push_s", -1.) != push or "tail" not in data["cases"][0]):
            return False
    return True


def run_seed(args, folder, checkpoint, digest, seed):
    seconds, push = (90, 15.) if args.suite == "long" else (30, -1.)
    protocols = ("source-play",) if args.suite == "long" else PROTOCOLS
    dest = folder / seed_name(seed)
    dest.mkdir(parents=True, exist_ok=True)
    with exclusive(dest / ".run.lock"), worker_slot(args.out):
        if valid(dest, digest, seed, seconds, push, protocols):
            return
        env = dict(os.environ, JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1",
                   OMP_NUM_THREADS="1", MALLOC_ARENA_MAX="2", PYTHONUNBUFFERED="1")
        with tempfile.TemporaryDirectory(prefix=".record-", dir=dest) as tmp:
            tmp = Path(tmp)
            # Upgrade existing raw captures without simulating them again.
            archives = [dest / f"{proto}.npz" for proto in protocols]
            command = [sys.executable, "-m", "tools.verify_getup_player"]
            if all(p.exists() for p in archives):
                command += ["score", *map(str, archives)]
                output = dest
            else:
                command += ["record", "--checkpoint", str(checkpoint), "--out", str(tmp),
                            "--seconds", str(seconds), "--protocol", *protocols]
                if seed is not None:
                    command += ["--jitter-seed", str(seed)]
                if push >= 0:
                    command += ["--push-at", str(push)]
                output = tmp
            print(f"{checkpoint}: {seed_name(seed)} ({seconds}s)", flush=True)
            with (dest / "player.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not valid(output, digest, seed, seconds, push, protocols):
                raise ValueError(f"Capture provenance mismatch in {dest}. Use a different label for a new policy.")
            if output != dest:
                for proto in protocols:
                    name = f"{proto}-metrics.json"
                    os.replace(tmp / name, dest / name)
                if args.keep_recordings:
                    for path in tmp.iterdir():
                        os.replace(path, dest / path.name)


def wilson(passed, total):
    z = 1.959963984540054
    frac = passed / total
    d = 1 + z*z/total
    mid = (frac + z*z/(2*total)) / d
    half = z / d * math.sqrt(frac*(1-frac)/total + z*z/(4*total**2))
    return [max(0., mid-half), min(1., mid+half)]


def bilateral_quiet(row):
    # An upright robot can still be stepping. This stricter endpoint uses the
    # measured final 10 s and both individual foot loads alongside the final hold.
    return (row["final_3s_supported_upright"] and row["tail"]["both_feet_loaded_fraction"] >= .90
            and row["tail"]["pelvis_xy_speed_rms_m_s"] < .08 and row["tail"]["foot_excursion_cm"] <= 3.
            and row["tail"]["torso_tilt_mean_deg"] < 15.)


def stats(rows):
    total = len(rows)
    supported = sum(r["final_3s_supported_upright"] for r in rows)
    clean = sum(r["final_3s_supported_upright"] and not r["post_rise_falls"] for r in rows)
    quiet_supported = sum(bilateral_quiet(r) for r in rows)
    def median(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return statistics.median(values) if values else None
    return dict(trials=total, final_supported=supported, final_supported_rate=supported/total,
                final_supported_wilson95=wilson(supported, total), clean_supported=clean,
                clean_supported_rate=clean/total, clean_supported_wilson95=wilson(clean,total),
                final_bilateral_quiet_10s=quiet_supported,
                final_bilateral_quiet_wilson95=wilson(quiet_supported,total),
                rose_and_no_refall=sum(r["rise_s"] is not None and not r["post_rise_falls"] for r in rows),
                with_refall=sum(bool(r["post_rise_falls"]) for r in rows),
                with_1s_quiet=sum(r["first_quiet_s"] is not None for r in rows),
                median_rise_s=median("rise_s"), median_rise_lag_s=median("rise_lag_s"),
                median_first_quiet_s=median("first_quiet_s"),
                mean_joint_rmse_deg=statistics.mean(r["joint_rmse_deg"] for r in rows),
                mean_leg_speed_p95_rad_s=statistics.mean(r["actual"]["leg_speed_p95_rad_s"] for r in rows),
                mean_knee_reversals=statistics.mean(r["actual"]["knee_reversals"] for r in rows),
                median_feet_path_ratio=median("feet_path_ratio"),
                mean_final_3s_load_fraction=statistics.mean(r["final_3s_loaded_fraction"] for r in rows),
                mean_tail_both_feet_fraction=statistics.mean(r["tail"]["both_feet_loaded_fraction"] for r in rows),
                median_tail_pelvis_speed_m_s=statistics.median(r["tail"]["pelvis_xy_speed_rms_m_s"] for r in rows))


def benchmark(args, label, checkpoint):
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    group = "selection" if args.suite == "panel" else args.suite
    folder = args.out / group / label
    folder.mkdir(parents=True, exist_ok=True)
    with exclusive(folder / ".identity.lock"):
        identity = folder / "checkpoint.sha256"
        if identity.exists() and identity.read_text().strip() != digest:
            raise ValueError(f"{label} already names another checkpoint in {folder}")
        identity.write_text(digest+"\n")
    seeds = SUITES[args.suite]
    if not args.summarize_only:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(lambda seed: run_seed(args, folder, checkpoint, digest, seed), seeds))
    protocols = ("source-play",) if args.suite == "long" else PROTOCOLS
    unique = "unique_clip_offsets"
    combined = {k: [] for k in (*protocols, unique)}
    per_seed = {}
    for seed in seeds:
        distinct, groups = {}, {}
        for proto in protocols:
            path = folder / seed_name(seed) / f"{proto}-metrics.json"
            rows = json.loads(path.read_text())["cases"]
            groups[proto] = rows
            for row in rows:
                key = (row["clip"], row["offset_s"])
                if (key in distinct and row.get("reset_state_sha256") != distinct[key].get("reset_state_sha256")):
                    raise ValueError(f"Shared source and fallen starts disagree for {key} in {path}")
                distinct[key] = row
        groups[unique] = list(distinct.values())
        per_seed[seed_name(seed)] = {k: stats(rows) for k, rows in groups.items()}
        for k, rows in groups.items():
            combined[k].extend(rows)
    report = dict(checkpoint=str(checkpoint), sha256=digest, suite=args.suite, seeds=seeds,
                  duration_s=90 if args.suite=="long" else 30, push_s=15 if args.suite=="long" else None,
                  interpretation="Wilson 95% intervals assume independent Bernoulli trials and are descriptive. Seeds vary initial perturbations of the same finite clip set. Source-play and fallen share eight clip/offset pairs. The unique count removes within-seed overlaps. See each seed separately for variation.",
                  bilateral_quiet_definition="Final supported upright hold, plus both foot loads >=0.10 bodyweight for >=90% of the final 10s, maximum ankle displacement <=3cm, pelvis horizontal RMS speed <0.08m/s and mean torso tilt <15deg.",
                  by_seed=per_seed, aggregate={k: stats(rows) for k, rows in combined.items()})
    output = folder / f"{args.suite}-summary.json"
    with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=folder) as stage:
        stage.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    os.replace(stage.name, output)
    print(output, {k: (v["final_supported"], v["clean_supported"], v["trials"])
                   for k,v in report["aggregate"].items()}, flush=True)


def paired(args):
    candidate, baseline = args.compare
    root = args.out / ("selection" if args.suite == "panel" else args.suite)
    groups, all_pairs = {}, []
    for seed in SUITES[args.suite]:
        name = seed_name(seed)
        states = []
        for label in args.compare:
            rows = {}
            for protocol in (("source-play",) if args.suite == "long" else PROTOCOLS):
                report = json.loads((root / label / name / f"{protocol}-metrics.json").read_text())
                rows.update({(r["clip"], r["offset_s"]):r for r in report["cases"]})
            states.append(rows)
        left, right = states
        if left.keys() != right.keys():
            raise ValueError(f"Clip/offset mismatch for {name}")
        pairs = []
        for key,r in left.items():
            old = right[key]
            for digest in ("source_sha256", "reset_state_sha256"):
                if r.get(digest) is None or r[digest] != old.get(digest):
                    raise ValueError(f"{digest} mismatch for {name}, {key}")
            pairs.append((r, old))
        groups[name] = pairs
        all_pairs.extend(pairs)
    endpoints = dict(bilateral_quiet_10s=bilateral_quiet,
                     rose_without_refall=lambda r:r["rise_s"] is not None and not r["post_rise_falls"],
                     final_supported=lambda r:r["final_3s_supported_upright"],
                     clean_supported=lambda r:r["final_3s_supported_upright"] and not r["post_rise_falls"])
    outcome = {(True, True): "both", (True, False): "candidate_only",
               (False, True): "baseline_only", (False, False): "neither"}
    def count(pairs):
        result = {}
        for metric,evaluate in endpoints.items():
            cells = dict(both=0, candidate_only=0, baseline_only=0, neither=0)
            for r,old in pairs:
                cells[outcome[bool(evaluate(r)), bool(evaluate(old))]] += 1
            result[metric] = cells
        return result
    report = dict(suite=args.suite, candidate=candidate, baseline=baseline, unique_pairs=len(all_pairs),
                  passed_cells=count(all_pairs), by_seed={k:count(x) for k,x in groups.items()},
                  provenance="Each pair has identical SHA-256 digests for both its source clip and native reset qpos/qvel.")
    output = root / f"{candidate}-vs-{baseline}-{args.suite}.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(output, report["passed_cells"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=SUITES, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/g1_getup_multiseed"))
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", nargs="+", metavar="LABEL=PATH")
    source.add_argument("--follow", type=Path, help="Wait for packaged milestones under DIR/NNNNNN/ckpt/0000.npz")
    source.add_argument("--compare", nargs=2, metavar=("CANDIDATE", "BASELINE"),
                        help="Compare two completed labelled suites with identical physical reset hashes")
    p.add_argument("--milestones", type=int, nargs="+", default=[1000,3000,6000,10000])
    p.add_argument("--jobs", type=int, choices=range(1,WORKER_LIMIT+1), default=2)
    p.add_argument("--keep-recordings", action="store_true", help="Retain large archives as well as per-case metrics")
    p.add_argument("--summarize-only", action="store_true", help="Recompute the summary from completed per-trial reports")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.compare:
        if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", label) for label in args.compare):
            p.error("--compare expects two checkpoint labels")
        paired(args)
    elif args.follow:
        if args.suite != "panel":
            p.error("--follow evaluates the panel for candidate selection")
        for step in args.milestones:
            name = f"{step:06d}"
            path = args.follow / name / "ckpt" / "0000.npz"
            print(f"Waiting for {path}", flush=True)
            while not path.is_file():
                time.sleep(15)
            benchmark(args, name, path)
    else:
        for spec in args.checkpoint:
            label, separator, path = spec.partition("=")
            if not separator or not re.fullmatch(r"[a-zA-Z0-9_-]+", label) or not path:
                p.error("--checkpoint expects each entry as LABEL=PATH")
            benchmark(args, label, Path(path))


if __name__ == "__main__":
    main()
