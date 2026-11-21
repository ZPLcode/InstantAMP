"""Measure the longest continuous post-push loss of upright pose and of quiet dual-foot stance."""
import json
from pathlib import Path
import sys

import numpy as np


def longest(mask, dt):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    lengths = np.flatnonzero(edges < 0) - np.flatnonzero(edges > 0)
    return float(max(lengths, default=0) * dt)


for arg in sys.argv[1:]:
    directory = Path(arg)
    report = json.loads((directory / "summary.json").read_text())
    rows = []
    for row in report["cases"]:
        with np.load(directory / f'{row["trial"]:02}_{row["direction"]}.npz') as raw:
            q, v, load, f = (raw[k] for k in ("qpos", "qvel", "support_load_bw", "feet_xy"))
        dt = .02
        step = round(report["push_time_s"] / dt)
        up_cos = 1 - 2 * np.square(q[1:, 4:6]).sum(1)
        upright = (q[1:, 2] > .58) & (q[1:, 2] <= .78) & (up_cos > np.cos(np.deg2rad(30)))
        fv = np.linalg.norm(np.diff(f, axis=0, prepend=f[:1]), axis=-1) / dt
        both = (load[:, :2] >= .1).all(axis=1)
        quiet = upright & both & (np.linalg.norm(v[1:, :2], axis=1) < .08) & (fv.max(1) < .05)
        rows.append(dict(trial=row["trial"], direction=row["direction"],
                         longest_not_upright_s=longest(~upright[step:], dt),
                         longest_without_bilateral_load_s=longest(~both[step:], dt),
                         longest_not_quiet_s=longest(~quiet[step:], dt),
                         total_not_quiet_s=float((~quiet[step:]).sum() * dt)))
    out = dict(longest_not_upright_s=max(r["longest_not_upright_s"] for r in rows),
               longest_without_bilateral_load_s=max(r["longest_without_bilateral_load_s"] for r in rows),
               longest_not_quiet_s=max(r["longest_not_quiet_s"] for r in rows), cases=rows)
    (directory / "transients.json").write_text(json.dumps(out, indent=2) + "\n")
