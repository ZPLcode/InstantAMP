"""Score walking, running and recovery with one Zenbot policy in native MuJoCo."""

import argparse
import hashlib
import json
import os
from pathlib import Path

if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import mujoco
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


def recovery_outcome(heights, postures, loads, dt=env.DT, *, profile=zenbot, initial_height=None,
                     shoulder_heights=None, initial_shoulder_height=None):
    """Keep the final standing test and expose failed rises before it.

    Count an 8 cm drop from an armed peak. Zenbot26 must first rise 8 cm
    from the trough and reach 30 cm. Its mean shoulder height can independently arm
    and fall by 12 cm. Each subsequent attempt requires a fresh rise.
    """
    heights, postures, loads = map(np.asarray, (heights, postures, loads))
    hold = round(zenbot.SUPPORTED_STAND_SECONDS / dt)
    stable = np.array([
        postures[i:i + hold].all()
        and np.count_nonzero(loads[i:i + hold] >= zenbot.MIN_SUPPORT_LOAD) >= zenbot.MIN_SUPPORT_FRACTION * hold
        for i in range(len(heights) - hold + 1)
    ], dtype=bool)
    starts = np.flatnonzero(stable)
    success = bool(len(stable) and stable[-1])
    unsettled = np.flatnonzero(~stable)
    sustained_start = int(unsettled[-1] + 1) if len(unsettled) else 0
    shoulders = None if shoulder_heights is None else np.asarray(shoulder_heights)
    samples = zenbot.attempt_heights(heights, shoulders, xp=np)
    trough = samples[0].copy()
    if initial_height is not None:
        trough[0] = initial_height
    if shoulders is not None and initial_shoulder_height is not None:
        trough[1] = initial_shoulder_height
    peak = zenbot.initial_attempt_peak(trough, profile)
    setbacks = 0
    for sample in samples:
        trough, peak, hits, _ = zenbot.update_attempt_peaks(sample, trough, peak, profile, xp=np)
        if hits.any():
            setbacks += 1
            trough, peak = sample, np.zeros_like(peak)
    up, good, bad, refalls = False, 0, 0, 0
    debounce = max(1, round(zenbot.REFALL_SECONDS / dt))
    for posture, load in zip(postures, loads, strict=True):
        good = good + 1 if posture and load >= zenbot.MIN_SUPPORT_LOAD else 0
        bad = bad + 1 if not posture else 0
        if not up and good >= debounce:
            up = True
        elif up and bad >= debounce:
            refalls += 1
            up = False
    return dict(
        success=success,
        first_attempt_success=success and setbacks == 0 and refalls == 0,
        lift_setbacks=setbacks,
        refalls=refalls,
        # Samples describe the state after each control step.
        first_supported_stand_s=float((starts[0] + 1) * dt) if len(starts) else None,
        sustained_supported_stand_s=float((sustained_start + 1) * dt) if success else None,
        final_height_m=float(heights[-1]),
    )


def _mean_shoulder_height(robot, shoulder_ids):
    # mj_step leaves body positions at the beginning of the last physics substep.
    mujoco.mj_kinematics(robot.model, robot.data)
    return robot.data.xpos[shoulder_ids, 2].mean()


def _recovery_trial(robot, steps):
    heights, postures, loads, velocity, angular, torque, joint_velocity = ([] for _ in range(7))
    # Include the reset/release pose in Zenbot26's observed minimum before the first control step.
    initial_height = float(robot.data.qpos[2]) if robot.profile.MODEL is not zenbot.MODEL else None
    shoulder_ids = [robot.model.body("robot_" + name).id for name in getattr(robot.profile, "SHOULDERS", ())]
    shoulders = [_mean_shoulder_height(robot, shoulder_ids)] if shoulder_ids else []
    for _ in range(steps):
        loads.append(robot.step())
        if shoulder_ids:
            shoulders.append(_mean_shoulder_height(robot, shoulder_ids))
        heights.append(robot.data.qpos[2])
        postures.append(zenbot.standing_posture(robot.data.qpos[2], robot.data.xmat[1, 8]))
        velocity.append(robot.data.qvel[:3].copy())
        angular.append(robot.data.qvel[3:6].copy())
        torque.append(robot.motor_torque.copy())
        joint_velocity.append(robot.motor_velocity.copy())
    outcome = recovery_outcome(heights, postures, loads, profile=robot.profile, initial_height=initial_height,
                               shoulder_heights=shoulders[1:] if shoulders else None,
                               initial_shoulder_height=shoulders[0] if shoulders else None)
    # Include retries through the final settling interval, excluding later standing.
    settled = outcome["sustained_supported_stand_s"]
    end = steps if settled is None else min(steps, round(settled / env.DT) + round(1. / env.DT) - 1)
    if shoulders:
        trace = np.asarray(shoulders[:end + 1])
        lifted = np.flatnonzero(trace >= np.minimum.accumulate(trace) + robot.profile.SHOULDER_SETBACK)
        raised = trace[lifted[0]:] if len(lifted) else trace[:1]
        outcome["max_shoulder_drop_m"] = float((np.maximum.accumulate(raised) - raised).max())
    tau = np.asarray(torque[:end])
    qd = np.asarray(joint_velocity[:end])
    linear, angular = np.asarray(velocity[:end]), np.asarray(angular[:end])
    outcome.update(
        effort_window_s=end * env.DT,
        peak_upward_speed_m_s=float(np.maximum(linear[:, 2], 0.).max()),
        peak_roll_pitch_speed_rad_s=float(np.linalg.norm(angular[:, :2], axis=-1).max()),
        peak_joint_speed_rad_s=float(np.abs(qd).max()),
        peak_torque_nm=float(np.abs(tau).max()),
        rms_torque_nm=float(np.sqrt(np.mean(tau**2))),
        motor_saturation_fraction=float(np.mean(np.abs(tau) >= .95 * robot.profile.EFFORT)),
        absolute_joint_work_j=float(np.abs(tau * qd).sum() * env.SIM_DT),
    )
    return outcome


def _push_recovery_trials(trials, steps, forces=(300., 450., 600.)):
    """Recover from physical torso pushes with the live velocity/history intact."""
    robot = trials.native
    body = robot.model.body("robot_" + robot.profile.TORSO).id
    directions = {"forward": (1., 0., 0.), "backward": (-1., 0., 0.),
                  "left": (0., 1., 0.), "right": (0., -1., 0.)}
    results = []
    for force in forces:
        for name, direction in directions.items():
            trials.stand()
            for _ in range(round(2. / env.DT)):
                robot.step()
            vector = robot.data.xmat[body].reshape(3, 3) @ np.asarray(direction)
            vector[2] = 0.
            vector /= np.linalg.norm(vector)
            robot.data.xfrc_applied[body, :3] = force * vector
            try:
                for _ in range(round(.4 / env.DT)):
                    robot.step()
            finally:
                robot.data.xfrc_applied.fill(0.)
            # Start scoring on the floor, excluding the externally forced descent.
            wait = 0
            while robot.data.qpos[2] >= zenbot.FALLEN_HEIGHT and wait < round(2. / env.DT):
                robot.step()
                wait += 1
            fallen = bool(robot.data.qpos[2] < zenbot.FALLEN_HEIGHT)
            result = dict(direction=name, force_N=force, push_duration_s=.4, fall_observed=fallen)
            if fallen:
                result.update(_recovery_trial(robot, steps))
            results.append(result)
    return results


def evaluate(checkpoint, seed=42, reference=None, *, push_recovery=False):
    """Evaluate ten physical seconds per command and per fixed fallen start."""
    checkpoint = Path(checkpoint)
    trials = JointTrials(checkpoint, reference, seed)
    steps, hold = round(10.0 / env.DT), round(1.0 / env.DT)
    gait = [_gait_trial(trials, speed, steps, 2 * hold) for speed in (1.0, 2.5)]
    robot = trials.native
    results = []
    for i, (clip, offset) in enumerate(trials.names):
        trials.fallen(i)
        results.append(dict(clip=clip, offset_s=offset, **_recovery_trial(robot, steps)))
    metadata = checkpoint.with_suffix(".json")
    report = dict(
        checkpoint=str(checkpoint),
        sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        iteration=json.loads(metadata.read_text()).get("it")
        if metadata.is_file()
        else None,
        seed=seed,
        duration_s=10.0,
        attempt_lift_height_m=robot.profile.ATTEMPT_LIFT_HEIGHT,
        lift_setback_m=zenbot.ATTEMPT_SETBACK,
        assistance_N=0,
        automatic_resets=False,
        teacher_inference=False,
        gait=gait,
        getup_successes=sum(t["success"] for t in results),
        getup_first_attempt_successes=sum(t["first_attempt_success"] for t in results),
        getup_lift_setbacks=sum(t["lift_setbacks"] for t in results),
        getup_refalls=sum(t["refalls"] for t in results),
        getup_total=len(results),
        trials=results,
    )
    if hasattr(robot.profile, "SHOULDER_SETBACK"):
        report["shoulder_lift_and_setback_m"] = robot.profile.SHOULDER_SETBACK
    if push_recovery:
        pushed = _push_recovery_trials(trials, steps)
        report.update(
            push_recovery_trials=pushed,
            push_recovery_falls=sum(t["fall_observed"] for t in pushed),
            push_recovery_successes=sum(t.get("success", False) for t in pushed),
            push_recovery_first_attempt_successes=sum(t.get("first_attempt_success", False) for t in pushed),
        )
    return report


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
    parser.add_argument(
        "--push-recovery", action="store_true",
        help="Also apply 300/450/600 N torso pushes for 0.4 s in four directions, then score recovery without resetting",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint, args.iteration)
    result = evaluate(checkpoint, args.seed, args.reference, push_recovery=args.push_recovery)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        result["gait"],
        f"get-up {result['getup_successes']}/{result['getup_total']}",
        f"first attempt {result['getup_first_attempt_successes']}/{result['getup_total']}",
    )
    if args.push_recovery:
        print(f"pushed recovery {result['push_recovery_successes']}/{result['push_recovery_falls']}, "
              f"first attempt {result['push_recovery_first_attempt_successes']}/{result['push_recovery_falls']}")


if __name__ == "__main__":
    main()
