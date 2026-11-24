"""Score walking, running and recovery with one Zenbot policy in native MuJoCo."""

import argparse
import hashlib
import json
import os
from pathlib import Path

if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from instantamp import env, zenbot
from instantamp.zenbot_trials import JointTrials, resolve_checkpoint


def _gait_trial(trials, speed, steps, window):
    trials.stand((speed, 0.0, 0.0))
    robot = trials.native
    data = robot.data
    velocities, upright, support, flight = [], [], [], []
    for _ in range(steps):
        robot.step()
        heading = np.arctan2(data.xmat[1, 3], data.xmat[1, 0])
        velocities.append(
            np.cos(heading) * data.qvel[0] + np.sin(heading) * data.qvel[1]
        )
        upright.append(
            data.qpos[2] > 0.55 and data.xmat[1, 8] > np.cos(np.pi / 6)
        )
        support.append(robot.support_load[:2].copy())
        flight.extend(robot.flight)

    feet = np.asarray(support[-window:]) > 0.08
    alternating = all(np.any(feet[:, s] & ~feet[:, 1 - s]) for s in (0, 1))
    airborne = np.asarray(flight[-window * env.DECIMATION :])
    edges = np.flatnonzero(np.diff(np.r_[False, airborne, False]))
    flights = int(np.sum((edges[1::2] - edges[::2]) * env.SIM_DT >= 0.01))
    actual = float(np.mean(velocities[-window:]))
    stable = all(upright[-window:])
    passed = stable and abs(actual - speed) < 0.25 * speed and alternating
    return dict(
        command_m_s=speed,
        measured_m_s=actual,
        upright=stable,
        alternating_feet=alternating,
        flight_periods_at_least_10ms=flights,
        passed=bool(passed and (speed < 2 or flights >= 2)),
    )


def evaluate(checkpoint, seed=42, reference=None):
    """Evaluate ten physical seconds per command and per fixed fallen start."""
    checkpoint = Path(checkpoint)
    trials = JointTrials(checkpoint, reference, seed)
    steps, hold = round(10.0 / env.DT), round(1.0 / env.DT)
    gait = [_gait_trial(trials, speed, steps, 2 * hold) for speed in (1.0, 2.5)]
    robot = trials.native
    results = []
    for i, (clip, offset) in enumerate(trials.names):
        trials.fallen(i)
        postures, loads = [], []
        for _ in range(steps):
            loads.append(robot.step())
            postures.append(
                zenbot.standing_posture(
                    robot.data.qpos[2], robot.data.xmat[1, 8]
                )
            )
        supported = (
            np.count_nonzero(np.asarray(loads[-hold:]) >= 0.5) >= 0.9 * hold
        )
        results.append(
            dict(
                clip=clip,
                offset_s=offset,
                success=bool(all(postures[-hold:]) and supported),
                final_height_m=float(robot.data.qpos[2]),
            )
        )
    metadata = checkpoint.with_suffix(".json")
    return dict(
        checkpoint=str(checkpoint),
        sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        iteration=json.loads(metadata.read_text()).get("it")
        if metadata.is_file()
        else None,
        seed=seed,
        duration_s=10.0,
        assistance_N=0,
        automatic_resets=False,
        teacher_inference=False,
        gait=gait,
        getup_successes=sum(t["success"] for t in results),
        getup_total=len(results),
        trials=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Exported policy NPZ or training run directory",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        help="Exact iteration when selecting from a run",
    )
    parser.add_argument(
        "--reference", type=Path, help="Recovery clips for this articulation"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint, args.iteration)
    result = evaluate(checkpoint, args.seed, args.reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        result["gait"],
        f"get-up {result['getup_successes']}/{result['getup_total']}",
    )


if __name__ == "__main__":
    main()
