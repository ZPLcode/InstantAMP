"""Compare two complete Zenbot release/evaluate.sh reports using only their saved JSON."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
from statistics import median


SEEDS = (932277, 1654679, 2718869, 3950477, 5969183)
OPENINGS = tuple(range(0, 22, 2))
STRESS = (3611, 6019, 7919, 9833, 13469, 16139, 19997, 25471, 32251, 41263,
          49871, 57467, 65353, 76367, 88339, 98731, 114269, 129779, 158329, 174893)
COMMANDS = {"forward": [1., 0., 0.], "back": [-.5, 0., 0.], "left": [0., .4, 0.],
            "right": [0., -.4, 0.], "turn_left": [0., 0., .6], "turn_right": [0., 0., -.6], "still": [0., 0., 0.]}
SUITES = {"continuous": (SEEDS, OPENINGS, 12.), "extended_standing": ((2339, 15497), (2, 14, 18), 72.),
          "difficult_openings": (STRESS, (16, 18), 12.)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def cases(rows, expected, fields=("seed", "trial")):
    actual = Counter(tuple(r[k] for k in fields) for r in rows)
    require(actual == Counter(expected), f"Missing, duplicate or unexpected {fields} cases")


def mid(values):
    numbers = [x for x in values if x is not None]
    return median(numbers) if numbers else None


def floor_metrics(rows):
    return {"cases": len(rows), "rose": sum(r["rise"] is not None for r in rows),
            "quiet_stand_at_end": sum(r["held"] for r in rows),
            "strict_first_rise_and_hold": sum(r.get("first_rise_held", r.get("first")) for r in rows),
            "rose_within_4s": sum(r["rise"] is not None and r["rise"] <= 4 for r in rows),
            "quiet_on_ground_at_least_1s": sum(r["ground_timing"]["longest_quiet_ground_s"] >= 1 for r in rows),
            "longest_quiet_ground_s": max(r["ground_timing"]["longest_quiet_ground_s"] for r in rows),
            "refalls": sum(r.get("refall", r.get("refell")) is True for r in rows),
            "median_rise_s_among_rises": mid(r["rise"] for r in rows)}


def read_suite(directory):
    def read(name):
        return json.loads((directory / name).read_text())
    metrics, protocol, shas = {}, {}, set()
    for name, (seeds, openings, seconds) in SUITES.items():
        d = read(name + "/summary.json")
        shas.add(d["sha256"])
        require(sorted(d["seeds"]) == sorted(seeds) and sorted(d["trials"]) == sorted(openings)
                and d["seconds"] == seconds and d["stage_seconds"] == 8, f"{name} seeds, trials or duration differ")
        floor, seq = d["floor"], d["transitions"]
        cases(floor, product(seeds, openings))
        cases(seq, [] if name == "difficult_openings" else product(seeds, openings))
        protocol[name] = [d["references"], sorted((r["seed"], r["trial"], r["clip"], r["offset"], r["lags"],
                                                  r["push_step"], r["reference_rise_s"]) for r in floor)]
        out = metrics[name] = floor_metrics(floor)
        if name == "difficult_openings":
            for trial in openings:
                metrics[f"difficult_opening_{trial}"] = floor_metrics([r for r in floor if r["trial"] == trial])
        if name == "extended_standing":
            held = [r["extended_stand"] for r in floor]
            require(all(r["seconds"] == 60. for r in held), "Extended hold duration differs")
            out.update(upright_full_60s=sum(r["upright_fraction"] == 1 for r in held),
                       bilateral_still_full_60s=sum(r["bilateral_still_fraction"] == 1 for r in held),
                       max_pelvis_departure_60s_cm=max(r["max_displacement_cm"] for r in held))
        if name != "continuous":
            continue
        settled = [r for r in floor if r["held"]]
        out.update(hands_loaded_then_transferred=sum(r["real_hand_load_s"] is not None and r["transfer_s"] is not None for r in floor),
                   median_reference_rise_s=mid(r["reference_rise_s"] for r in floor),
                   median_rise_style_deg=mid(r["rise_style_deg"] for r in floor),
                   median_standing_tilt_deg_among_held=mid(r["stance"]["tilt_deg"] for r in settled),
                   median_standing_joint_deg_among_held=mid(r["stance"]["joints_deg"] for r in settled),
                   median_postrise_pelvis_cm_among_held=mid(r["drift_cm"] for r in settled),
                   median_postrise_feet_cm_among_held=mid(r["feet_cm"] for r in settled))
        metrics["sequence"] = {"cases": len(seq), "from_held": sum(r["from_held"] for r in seq),
                               "whole_sequence_passes": sum(r["passed"] for r in seq)}
        for stage in ("walk", "run", "stop"):
            rows = [r["stages"][stage] for r in seq]
            out = metrics["sequence_" + stage] = {"cases": len(rows), "passed": sum(r["passed"] for r in rows),
                        "falls": sum(r["fell"] for r in rows), "median_final_forward_mps": mid(r["final_forward_mps"] for r in rows)}
            if stage != "stop":
                out["median_joint_style_deg_among_passed"] = mid(r["style"]["joint_distribution_mae_deg"] for r in rows if r["passed"])

    dirs = read("directions.json")
    require(len(dirs["checkpoints"]) == 1, "Expected one evaluated checkpoint")
    require(Counter(map(tuple, dirs["cases"])) == Counter(product(SEEDS, OPENINGS)), "Directional seeds/trials differ")
    require(dirs["commands"] == COMMANDS and [dirs[k] for k in ("floor_seconds", "stage_seconds", "tail_seconds",
            "yaw_max_radius_m")] == [12., 8., 3., .35], "Directional commands, duration or yaw threshold differ")
    checkpoint = dirs["checkpoints"][0]
    protocol["directions"] = [dirs["coordinate_order"], sorted((r["seed"], r["trial"], r["name"], r["clip"]) for r in checkpoint["rows"])]
    shas.add(checkpoint["sha256"])
    cases(checkpoint["rows"], product(SEEDS, OPENINGS, COMMANDS), ("seed", "trial", "name"))
    for name, cmd in COMMANDS.items():
        rows = [r for r in checkpoint["rows"] if r["name"] == name]
        require(all(r["command"] == cmd for r in rows), f"Wrong command for {name}")
        held = [r for r in rows if r["from_held"]]
        metrics["direction_" + name] = {"cases": len(rows), "from_held": len(held), "passed_from_held": sum(r["passed"] for r in held),
             "passed_all_starts": sum(r["passed"] for r in rows), "falls_all_starts": sum(r["fell"] for r in rows),
             "falls_from_held": sum(r["fell"] for r in held),
             **{f"median_{axis}_from_held": mid(r["mean_last_3s"][i] for r in held)
                for i, axis in enumerate(("vx_mps", "vy_mps", "yaw_rad_s"))}}
        if name.startswith("turn_"):
            metrics["direction_" + name]["median_max_radius_m_from_held"] = mid(r["max_departure_m"] for r in held)

    src = read("source_getup.json")
    require(sorted(src["seeds"]) == sorted(SEEDS) and [src[k] for k in ("seconds", "policy_hz", "physics_hz",
            "motor_delay_steps", "observation_width")] == [12., 50, 200, 0, 234], "Source contact protocol differs")
    cases(src["cases"], product(SEEDS, OPENINGS))
    protocol["source_contact"] = [src["xml_sha256"], sorted((r["seed"], r["trial"], r["clip"]) for r in src["cases"])]
    shas.add(src["sha256"])
    require(len(shas) == 1, "The five suites contain different policy hashes")
    metrics["source_contact"] = floor_metrics(src["cases"])
    return shas.pop(), protocol, metrics


def compare(baseline, candidate):
    base_sha, protocol, old = read_suite(baseline)
    new_sha, other_protocol, new = read_suite(candidate)
    for name in protocol:
        require(protocol[name] == other_protocol[name], f"{name} source settings, motion references or opening assignments differ")
    result = {}
    for group, metrics in old.items():
        result[group] = {}
        for name, before in metrics.items():
            after = new[group][name]
            delta = after - before if before is not None and after is not None else None
            result[group][name] = {"baseline": before, "candidate": after, "delta": delta}
    return {"baseline_sha256": base_sha, "candidate_sha256": new_sha, "metrics": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, help="Optional JSON output, for example RUN/comparison.json")
    args = parser.parse_args()
    try:
        text = json.dumps(compare(args.baseline, args.candidate), allow_nan=False, indent=2) + "\n"
        if args.out is None:
            print(text, end="")
        else:
            inputs = [*(name + "/summary.json" for name in SUITES), "directions.json", "source_getup.json"]
            require(args.out.resolve() not in {(root / path).resolve() for root in (args.baseline, args.candidate)
                                               for path in inputs}, "Output path would overwrite an input report")
            args.out.write_text(text)
    except (ValueError, KeyError, OSError) as e:
        parser.exit(1, f"Benchmark comparison failed: {e}\n")
