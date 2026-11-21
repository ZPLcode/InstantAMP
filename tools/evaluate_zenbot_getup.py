"""Score first-rise, planted standing and hand support through the interactive Zenbot player's physics."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.spatial.transform import Rotation

from instantamp import zenbot
from tools.play_zenbot_getup import REFERENCE, RELEASE, ZenbotPlayer
from tools.score_getup_in_place import pre_rise_attempts
from tools.score_zenbot_sequence import first_run, support_diagnostics

DT = 0.02


def evaluate(player, trial, seed, steps, push_step, plan_out=None):
    rng = np.random.default_rng([seed, trial, 7143])
    lags = np.random.default_rng(seed).integers(3, 8, (len(player.names), 5))[trial]
    q, v = (
        np.array(player.poses[trial], float),
        np.array(player.velocities[trial], float),
    )
    if seed != 42:
        q[7:] += np.deg2rad(rng.uniform(-2.0, 2.0, 23))
        rotation = Rotation.from_quat(q[3:7], scalar_first=True)
        tilt = Rotation.from_euler("xy", rng.uniform(-2.0, 2.0, 2), degrees=True)
        q[3:7] = (rotation * tilt).as_quat(scalar_first=True)
        v[:2] += rng.uniform(-0.07, 0.07, 2)
    sim = player.native
    sim.reset(q, v, lags)
    player.update_view()
    feet = [sim.model.body("robot_" + name).id for name in zenbot.FEET]
    Q = np.empty((steps + 1, 30))
    V = np.empty((steps + 1, 29))
    F = np.empty((steps, 2, 2))
    loads = np.empty(steps)
    support = np.empty((steps, 6))
    Q[0], V[0] = q, v
    for k in range(steps):
        if k == push_step:
            player.push()
        loads[k] = player.step()
        if k == 0 and plan_out is not None:
            tracker = sim.native_tracking
            if tracker is None or tracker.plan is None:
                raise ValueError(f"No measured-state motor plan from seed {seed}, trial {trial}")
            length = len(tracker.plan)
            plan_out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                plan_out / f"{seed}_{trial:02}.npz", state=np.r_[q, v], lags=tracker.lags,
                source=tracker.debug["matched"], score=tracker.debug["forecast_objective"],
                parameters=tracker.parameters, action=tracker.plan,
                q=tracker.planned_q[:length], v=tracker.planned_v[:length],
            )
        Q[k + 1], V[k + 1] = sim.data.qpos, sim.data.qvel
        F[k], support[k] = sim.data.xpos[feet, :2], sim.support_load
    up_cos = 1.0 - 2.0 * np.square(Q[:, 4:6]).sum(-1)
    upright = (Q[:, 2] > 0.58) & (Q[:, 2] <= 0.78) & (up_cos > 0.8660254)
    tick = first_run(upright, 10)
    duration = None if tick is None else tick * DT
    refall = drift = foot_drift = None
    if tick is not None:
        refall = bool(((Q[tick + 10 :, 2] < 0.42) | (up_cos[tick + 10 :] < 0.55)).any())
        drift = float(100 * np.linalg.norm(Q[tick:, :2] - Q[tick, :2], axis=-1).max())
        f0 = max(0, tick - 1)
        foot_drift = float(100 * np.linalg.norm(F[f0:] - F[f0], axis=-1).max())
    held = bool(
        upright[-100:].all()
        and (loads[-100:] >= 0.50).mean() >= 0.9
        and np.linalg.norm(V[-100:, :2], axis=1).max() < 0.15
    )
    clip, offset = player.names[trial]
    row = dict(
        trial=trial,
        seed=seed,
        push_step=push_step,
        clip=clip,
        offset=offset,
        lags=lags.tolist(),
        rise=duration,
        refall=refall,
        held=held,
        drift_cm=drift,
        feet_cm=foot_drift,
        throughout=bool(tick is not None and upright[tick:].all()),
        pre_rise_attempts=pre_rise_attempts(Q[:, 2], DT, duration),
    )
    row.update(support_diagnostics(Q, V, support, tick, clip, offset, player.reference))
    return row, dict(qpos=Q, qvel=V, feet_xy=F, loads=loads, support_load_bw=support)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=RELEASE)
    p.add_argument("--reference", type=Path, default=REFERENCE)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--trials", type=int, nargs="+", default=list(range(17)))
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument(
        "--push-time",
        type=float,
        help="Time of a +0.20 m/s world-X pelvis velocity impulse",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-plan-cache", action="store_true", help="Run the same planner from measured state on every reset")
    p.add_argument("--save-plans", type=Path, help="Save uncached motor plans generated from each exact reset state")
    p.add_argument(
        "--trajectories",
        action="store_true",
        help="Also store the raw 50 Hz state and real support loads",
    )
    args = p.parse_args()
    if args.seconds < 2.0 or (
        args.push_time is not None and not 0 <= args.push_time < args.seconds
    ):
        p.error(
            "Use a duration of at least two seconds and a push time inside the recording"
        )
    player = ZenbotPlayer(args.checkpoint, args.reference, use_plan_cache=not (args.no_plan_cache or args.save_plans))
    args.out.mkdir(parents=True, exist_ok=True)
    step = -1 if args.push_time is None else round(args.push_time / DT)
    for seed in args.seeds:
        for trial in args.trials:
            if not 0 <= trial < len(player.names):
                p.error(f"Trial must be between 0 and {len(player.names) - 1}")
            row, arrays = evaluate(player, trial, seed, round(args.seconds / DT), step, args.save_plans)
            prefix = args.out / f"{seed}_{trial:02}"
            if args.trajectories:
                np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
            prefix.with_suffix(".json").write_text(json.dumps(row, indent=2))
            print(
                seed,
                trial,
                "rise_s",
                row["rise"],
                "held",
                row["held"],
                "max_pelvis_cm",
                row["drift_cm"],
                flush=True,
            )


if __name__ == "__main__":
    main()
