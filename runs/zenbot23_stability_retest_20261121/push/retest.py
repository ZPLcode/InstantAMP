"""From repository root run JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python runs/zenbot23_stability_retest_20261121/push/retest.py."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

from tools.evaluate_zenbot_getup import DT, evaluate
from tools.play_zenbot_getup import RELEASE, ZenbotPlayer
from tools.score_zenbot_sequence import first_run

OUT = Path(__file__).resolve().parent
TRIALS = [4, 6, 9, 14]
DIRECTIONS = {"pos_x": (1, 0), "neg_x": (-1, 0), "pos_y": (0, 1), "neg_y": (0, -1)}
SPEED = 0.20
SEED = 42
PUSH_STEP = round(15 / DT)
STEPS = round(40 / DT)


class DirectionalPlayer(ZenbotPlayer):
    def push(self):
        self.native.data.qvel[:2] += self.velocity_change


def excursions(xy, anchor):
    return float(100 * np.linalg.norm(xy - anchor, axis=-1).max())


def analyze(arrays):
    q, v, f, loads = (arrays[k] for k in ("qpos", "qvel", "feet_xy", "support_load_bw"))
    tilt_cos = 1 - 2 * np.square(q[:, 4:6]).sum(axis=1)
    upright = (q[:, 2] > 0.58) & (q[:, 2] <= 0.78) & (tilt_cos > np.cos(np.deg2rad(30)))
    foot_speed = np.linalg.norm(np.diff(f, prepend=f[:1], axis=0), axis=-1).max(axis=1) / DT
    quiet = (upright[1:] & (loads[:, :2] >= 0.10).all(axis=1) &
             (np.linalg.norm(v[1:, :2], axis=1) < 0.08) & (foot_speed < 0.05))
    after = slice(PUSH_STEP, None)
    half = round(0.5 / DT)
    recovered = first_run(quiet[after], half)
    return dict(
        upright_at_push=bool(upright[PUSH_STEP]), quiet_before_1s=float(quiet[PUSH_STEP-50:PUSH_STEP].mean()),
        upright_all_after=bool(upright[after].all()), refell_after=bool(((q[after, 2] < 0.42) | (tilt_cos[after] < 0.55)).any()),
        post_push_min_height_m=float(q[after, 2].min()),
        post_push_peak_tilt_deg=float(np.rad2deg(np.arccos(np.clip(tilt_cos[after], -1, 1))).max()),
        time_to_confirm_quiet_for_0p5s=None if recovered is None else (recovered + half) * DT,
        post_push_max_pelvis_cm=excursions(q[after, :2], q[PUSH_STEP, :2]),
        post_push_max_foot_cm=excursions(f[after], f[PUSH_STEP - 1]),
        final_10s_quiet_fraction=float(quiet[-500:].mean()),
        final_10s_pelvis_excursion_cm=excursions(q[-501:, :2], q[-501, :2]),
        final_10s_foot_excursion_cm=excursions(f[-500:], f[-500]),
    )


def main():
    player = DirectionalPlayer(RELEASE)
    rows = []
    for trial in TRIALS:
        for name, direction in DIRECTIONS.items():
            player.velocity_change = np.asarray(direction) * SPEED
            case, arrays = evaluate(player, trial, SEED, STEPS, PUSH_STEP)
            case.update(direction=name, **analyze(arrays))
            prefix = OUT / f"{trial:02d}_{name}"
            np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
            prefix.with_suffix(".json").write_text(json.dumps(case, indent=2) + "\n")
            rows.append(case)
            print(trial, name, case["held"], case["refell_after"], case["post_push_max_pelvis_cm"], flush=True)
    key = "time_to_confirm_quiet_for_0p5s"
    summary = dict(n=len(rows), stood_before_push=sum(r["upright_at_push"] and r["rise"] is not None and r["rise"] < PUSH_STEP * DT for r in rows),
                   min_quiet_fraction_1s_before_push=min(r["quiet_before_1s"] for r in rows),
                   fully_upright_after_push=sum(r["upright_all_after"] for r in rows), refell_after_push=sum(r["refell_after"] for r in rows),
                   held=sum(r["held"] for r in rows), quiet_last_10s=sum(r["final_10s_quiet_fraction"] >= .90 for r in rows),
                   confirmed_half_second_quiet=sum(r[key] is not None for r in rows))
    for k in (key, "post_push_max_pelvis_cm", "post_push_max_foot_cm", "final_10s_pelvis_excursion_cm", "final_10s_foot_excursion_cm", "post_push_peak_tilt_deg"):
        values = [r[k] for r in rows if r[k] is not None]
        summary[k] = {"median": float(np.median(values)), "max": max(values)} if values else None
    report = dict(checkpoint=str(RELEASE), sha256=hashlib.sha256(RELEASE.read_bytes()).hexdigest(),
                  trials=TRIALS, seed=SEED, directions_world_frame=DIRECTIONS, delta_velocity_m_s=SPEED,
                  push_time_s=PUSH_STEP * DT, duration_s=STEPS * DT,
                  quiet="Root height (0.58,0.78] m, tilt <30 degrees, each foot >=10% body weight, root speed <0.08 m/s and maximum foot speed <0.05 m/s in XY.",
                  summary=summary, cases=rows)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=SPEED)
    parser.add_argument("--trials", type=int, nargs="+", default=TRIALS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    SPEED, TRIALS, OUT = args.speed, args.trials, args.out
    OUT.mkdir(parents=True, exist_ok=True)
    main()
