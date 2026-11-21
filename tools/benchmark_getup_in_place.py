"""Run matched 30s native G1 tests for standing in place.

    python -m tools.benchmark_getup_in_place --suite dev --checkpoint A=PATH_A B=PATH_B

The dev suite uses three previously seen development seeds. Run the final suite only
after selecting a policy; its five perturbation seeds were reserved for this endpoint.
The ground-phase suite uses five further seeds reserved for palm support before extension.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

from tools.benchmark_getup_checkpoints import exclusive, worker_slot
from tools.score_getup_in_place import summarize

DEV = (4471,18317,29041)
FINAL = (184903,198761,205759,224729,245083)
SUITES = {
    "dev": DEV,
    "final": FINAL,
    "ground-phase": (281117, 306407, 337219, 362393, 395189),
    "final-stability": (426119, 451213, 477241, 501287, 528191),
}


def paired(first, second):
    def keyed(rows):
        return {(r["clip"], r["offset_s"], r["reset_state_sha256"]): r for r in rows}
    left, right = keyed(first), keyed(second)
    if left.keys() != right.keys() or len(left) != len(first) or len(right) != len(second):
        raise ValueError("Paired reports require the same distinct physical starting states")
    identical = sum(r["trajectory_sha256"] == right[k]["trajectory_sha256"] for k,r in left.items())
    differences = [r["full_after_rise"]["max_cm"] - right[k]["full_after_rise"]["max_cm"]
                   for k,r in left.items() if "full_after_rise" in r and "full_after_rise" in right[k]]
    measured_attempts = all("attempts" in row for row in (*first,*second))
    def endpoints(row):
        result = {"rose":row["first_upright_s"] is not None,
                  "rose_without_refall":row["first_upright_s"] is not None and not row["post_rise_falls"],
                  "final_bilateral_quiet_10s":row["final_bilateral_quiet_10s"],
                  "throughout_stable":row["throughout_stable"]}
        for key in ("pelvis_within_cm", "pelvis_and_planted_feet_within_cm"):
            result.update({f"{key}_{limit}":outcome for limit,outcome in row[key].items()})
        if measured_attempts:
            single = result["rose_without_refall"] and row["attempts"]["retries"] == 0
            result["rose_without_retry_or_refall"] = single
            result.update({f"first_attempt_pelvis_and_feet_within_cm_{limit}":single and outcome
                           for limit,outcome in row["pelvis_and_planted_feet_within_cm"].items()})
        return result
    outcomes = {}
    names = ("neither", "second_only", "first_only", "both")
    for key,row in left.items():
        counterpart = endpoints(right[key])
        for endpoint,value in endpoints(row).items():
            counts = outcomes.setdefault(endpoint, dict.fromkeys(names,0))
            counts[names[2*int(value)+int(counterpart[endpoint])]] += 1
    return dict(distinct_paired_resets=len(first), identical_trajectory_and_support=identical,
                both_rose=len(differences), both_have_attempt_diagnostics=measured_attempts,
                mean_paired_peak_delta_cm=statistics.mean(differences) if differences else None,
                median_paired_peak_delta_cm=statistics.median(differences) if differences else None,
                passed=outcomes)


def verify(args, label, checkpoint, digest, seed):
    dest = args.out / args.suite / label / f"seed_{seed}"
    dest.mkdir(parents=True,exist_ok=True)
    result = dest/"in-place.json"
    with exclusive(dest/".run.lock"):
        if result.exists():
            data = json.loads(result.read_text())
            if (data["checkpoint_sha256"] != digest or data["reset_jitter_seed"] != seed
                    or data["duration_s"] != 30 or data["unique"]["trials"] != 17):
                raise ValueError(f"Cached physical run differs for {dest}")
            return data
        with worker_slot(args.out,args.jobs), tempfile.TemporaryDirectory(dir=dest) as temp:
            tmp = Path(temp)
            env = dict(os.environ,JAX_PLATFORMS="cpu",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
            print(f"{label} seed {seed}",flush=True)
            def run(command,log):
                with (dest/log).open("w") as stream:
                    subprocess.run([sys.executable,"-m",*command],stdout=stream,stderr=subprocess.STDOUT,
                                   env=env,check=True)
            run(["tools.verify_getup_player","record","--checkpoint",str(checkpoint),"--out",str(tmp),
                 "--jitter-seed",str(seed),"--seconds","30"],"record.log")
            run(["tools.score_getup_in_place",str(tmp/"source-play.npz"),str(tmp/"fallen.npz"),
                 "--out",str(tmp/"in-place.json")],"score.log")
            data = json.loads((tmp/"in-place.json").read_text())
            if data["unique"]["trials"] != 17 or data["checkpoint_sha256"] != digest:
                raise ValueError(f"Unexpected recording provenance in {dest}")
            if args.keep_recordings:
                for path in tmp.glob("*.npz"):
                    os.replace(path,dest/path.name)
                for path in tmp.glob("*.mjb"):
                    os.replace(path,dest/path.name)
                data["archives"] = ["source-play.npz","fallen.npz"]
            else:
                data["archives"] = []
            data["checkpoint"] = str(checkpoint)
            (tmp/"in-place.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
            os.replace(tmp/"in-place.json",result)
            return data


def main():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite",choices=SUITES,required=True)
    p.add_argument("--checkpoint",nargs="+",required=True,metavar="LABEL=PATH")
    p.add_argument("--out",type=Path,default=Path("runs/g1_getup_in_place"))
    p.add_argument("--jobs",type=int,choices=range(1,5),default=2)
    p.add_argument("--seeds",type=int,nargs="+",choices=DEV,
                   help="Run a subset of the registered development seeds")
    p.add_argument("--keep-recordings",action="store_true")
    args=p.parse_args()
    seeds = SUITES[args.suite]
    if args.seeds is not None:
        if args.suite != "dev" or len(set(args.seeds)) != len(args.seeds):
            p.error("--seeds requires distinct development seeds and --suite dev")
        seeds=tuple(args.seeds)
    tasks=[];sets={}
    for spec in args.checkpoint:
        label,separator,path=spec.partition("=")
        if not separator or not re.fullmatch(r"[a-zA-Z0-9_-]+",label) or label in sets:
            p.error("--checkpoint expects distinct LABEL=PATH entries")
        checkpoint=Path(path)
        with checkpoint.open("rb") as stream:
            digest=hashlib.file_digest(stream,"sha256").hexdigest()
        sets[label]=[]
        tasks.append((label,checkpoint,digest))
    # Interleave policies for the same seed so both sides of a comparison finish together.
    jobs=[(label,ck,digest,seed) for seed in seeds for label,ck,digest in tasks]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for job,data in zip(jobs,pool.map(lambda spec:verify(args,*spec),jobs)):
            sets[job[0]].append(data)
    initial_states=None
    for label,rows in sets.items():
        cases=[r for seed in rows for r in seed["cases"]]
        keys={(r["clip"],r["offset_s"],r["reset_state_sha256"]) for r in cases}
        if len(keys)!=17*len(seeds) or initial_states is not None and keys!=initial_states:
            raise ValueError("The compared policies must use identical, distinct physical resets")
        initial_states=keys
        summary=dict(checkpoint_sha256=rows[0]["checkpoint_sha256"],seeds=seeds,
                     by_seed={str(seed):data["unique"] for seed,data in zip(seeds,rows)},unique=summarize(cases))
        suffix = "" if seeds == SUITES[args.suite] else "-"+"-".join(map(str,seeds))
        output=args.out/args.suite/label/f"summary{suffix}.json"
        output.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
        print(output,summary["unique"],flush=True)
    if len(sets) == 2:
        first,second = sets
        all_cases = lambda rows: [r for seed in rows for r in seed["cases"]]
        report = dict(first=first, second=second, first_sha256=sets[first][0]["checkpoint_sha256"],
                      second_sha256=sets[second][0]["checkpoint_sha256"], seeds=seeds,
                      unique=paired(all_cases(sets[first]), all_cases(sets[second])),
                      by_seed={str(seed):paired(a["cases"],b["cases"])
                               for seed,a,b in zip(seeds,sets[first],sets[second])})
        output=args.out/args.suite/f"{first}-vs-{second}{suffix}.json"
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        print(output,report["unique"],flush=True)


if __name__=="__main__":
    main()
