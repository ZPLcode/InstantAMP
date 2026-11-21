"""Compare physical Zenbot palm loads with the same original G1 clip and starting frame."""

import argparse
import json
from pathlib import Path

from tools.summarize_zenbot_getup import describe


def compare(g1_profile, directories, reference):
    g1 = json.loads(g1_profile.read_text())
    source = {(r["clip"], r["offset_s"]): r["source"] for r in g1["cases"]}
    out = {}
    for directory in directories:
        cases = []
        for f in sorted(directory.glob("[0-9]*_[0-9]*.json")):
            row = describe(f, reference)
            name, offset = row["clip"], row["offset"]
            g1_motion = source.get((name, offset))
            original_required = None if g1_motion is None else g1_motion["sustained_palm_before_extension"]
            actual = row["actual_phase"]
            cases.append(dict(clip=name, offset=offset, trial=row["trial"], seed=row["seed"],
                              original_g1_needs_palm=original_required, retargeted_marker_phase=row["reference_marker_phase"],
                              retargeted_surface_phase=row["reference_phase"], actual_phase=actual,
                              minimum_hand_landmark_m=row["minimum_hand_landmark_m"],
                              minimum_palm_surface_m=row["minimum_palm_surface_m"],
                              real_palm_precedes_extension=actual["hand_tick"] is not None and actual["knee_tick"] is not None))
        required = [r for r in cases if r["original_g1_needs_palm"]]
        surface = [r for r in cases if r["retargeted_surface_phase"]["hand_tick"] is not None and
                   r["retargeted_surface_phase"]["knee_tick"] is not None]
        out[str(directory)] = dict(cases=cases, trials=len(cases), original_g1_requires_palm=len(required),
                                   actual_before_knee_on_g1_required=sum(r["real_palm_precedes_extension"] for r in required),
                                   retargeted_surface_requires_palm=len(surface),
                                   actual_before_knee_on_retargeted_surface=sum(r["real_palm_precedes_extension"] for r in surface))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--g1-profile", required=True, type=Path)
    p.add_argument("--reference", default=Path("data/zenbot23_getup/recovery"), type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("directories", nargs="+", type=Path)
    args = p.parse_args()
    result = compare(args.g1_profile, args.directories, args.reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print({k: {s: v for s, v in d.items() if s != "cases"} for k, d in result.items()})
