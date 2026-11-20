"""Convert physically executed G1 MPC plans into a finite bank of 50 Hz feedback policies.

Each controller requires measured qpos and qvel in its local planning world frame.
The exporter linearizes native MuJoCo motor and contact dynamics around the plan
and solves a finite-horizon linear quadratic regulator backward in time.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from scipy.linalg import solve

from instantamp import env
from instantamp.sim import build_scene
from instantamp.timed_feedback import TimedFeedbackPolicy
from tools.score_g1_tracking import score_tracking


def input_derivatives(q, velocity, target):
    """Map the constrained native servo's position and velocity derivatives to MuJoCo ctrl."""
    torque = env.KP * (target - q) - env.KD * velocity
    envelope = np.minimum(np.where(torque * velocity > 0, env.Y1, env.Y2), env.EFFORT)
    speed = np.abs(velocity)
    speed_scale = np.clip((env.X2 - speed) / (env.X2 - env.X1), 0, 1)
    active = np.abs(torque) < envelope * speed_scale
    rate_limited = (speed > env.X1) & (speed < env.X2)
    limit_derivative = -envelope / (env.X2 - env.X1) * np.sign(velocity) * rate_limited
    friction_derivative = env.FD + env.FS / env.VA * (1 - np.tanh(velocity / env.VA) ** 2)
    velocity_derivative = ((~active) * (env.KD + np.sign(torque) * limit_derivative) - friction_derivative) / env.KP
    return active, velocity_derivative


def linearize(qpath, vpath, targets, controls, support_friction):
    model = build_scene(g1_support_friction=support_friction)
    data, scratch = mujoco.MjData(model), mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = qpath[0], vpath[0]
    mujoco.mj_forward(model, data)
    n, joints = model.nv, model.nu
    micro_a = np.empty((2 * n, 2 * n))
    micro_b = np.empty((2 * n, joints))
    a_path = np.empty((len(targets), 2 * n, 2 * n))
    b_path = np.empty((len(targets), 2 * n, joints))
    input_scale = env.EFFORT / env.KP
    state_scale = np.sqrt(np.r_[[60., 60., 150.], [25.] * 3, [1.] * joints,
                                [.4] * 3, [.12] * 3, [.008] * joints])

    for t, target in enumerate(targets):
        a, b = np.eye(2 * n), np.zeros((2 * n, joints))
        for substep, command in enumerate(controls[t]):
            q, velocity = data.qpos[7:], data.qvel[6:]
            expected = env.actuator_ctrl(q, velocity, target, xp=np)
            if np.max(np.abs(command - expected)) > 3e-5:
                raise ValueError(f"Recorded and native G1 motor commands disagree at frame {t}, substep {substep}")
            data.ctrl[:] = command
            # Keep finite-difference perturbations out of the replay's contact solver warm starts.
            mujoco.mj_copyData(scratch, model, data)
            mujoco.mjd_transitionFD(model, scratch, .0002, False, micro_a, micro_b, None, None)
            active, velocity_derivative = input_derivatives(q, velocity, target)
            micro_a[:, 6:n] += micro_b * (~active)[None, :]
            micro_a[:, n+6:] += micro_b * velocity_derivative[None, :]
            micro_input = micro_b * (input_scale * active)[None, :]
            a, b = micro_a @ a, micro_a @ b + micro_input
            mujoco.mj_step(model, data)

        # New contacts can give unbounded one-sided derivatives. Cap amplification
        # in dimensionless cost-weighted coordinates before the Riccati recursion.
        scaled = a * state_scale[:, None] / state_scale[None, :]
        left, singular, right = np.linalg.svd(scaled, full_matrices=False)
        a_path[t] = (left * np.minimum(singular, 8.)) @ right
        b_path[t] = np.clip(b * state_scale[:, None], -15., 15.)
        if not np.allclose(data.qpos, qpath[t+1], rtol=0, atol=3e-7):
            raise ValueError(f"The saved plan does not replay in this contact model at frame {t}")
    return a_path, b_path, state_scale


def fit_case(data):
    a_path, b_path, scale = linearize(*data)
    n, joints = b_path.shape[1:]
    state_cost, effort_cost = np.eye(n), .35 * np.eye(joints)
    value = state_cost.copy()
    gains = np.empty((len(a_path), joints, n), dtype=np.float16)
    for t in reversed(range(len(a_path))):
        a, b = a_path[t], b_path[t]
        discounted = .975 * value
        gain = -solve(effort_cost + b.T @ discounted @ b, b.T @ discounted @ a,
                      assume_a="pos", check_finite=False)
        closed = a + b @ gain
        value = state_cost + closed.T @ discounted @ closed + gain.T @ effort_cost @ gain
        value = .5 * (value + value.T)
        gains[t] = gain * scale[None, :]
    return gains


def physically_evaluate(policy, source):
    model = build_scene(g1_support_friction=policy.support_friction)
    trials = []
    conditions = {"nominal": (0., 0.), "height_and_velocity": (.005, .08),
                  "opposite_velocity": (-.005, -.08)}
    for i, name in enumerate(policy.names):
        for label, (height_offset, velocity_offset) in conditions.items():
            robot = mujoco.MjData(model)
            robot.qpos[:], robot.qvel[:] = policy.qpos[i, 0], policy.qvel[i, 0]
            robot.qpos[2] += height_offset
            robot.qvel[0] += velocity_offset
            mujoco.mj_forward(model, robot)
            path = [robot.qpos.copy()]
            for tick in range(policy.targets.shape[1]):
                target = policy.target(i, tick, robot.qpos, robot.qvel)
                for _ in range(env.DECIMATION):
                    robot.ctrl[:] = env.actuator_ctrl(robot.qpos[7:], robot.qvel[6:], target, xp=np)
                    mujoco.mj_step(model, robot)
                path.append(robot.qpos.copy())
            path = np.stack(path)
            end = 1 + int(source["reference_end"][i])
            result = score_tracking(model, source["reference"][i, :end], path)
            result.update(clip=name, mode=label, max_abs_qpos_vs_plan=float(np.max(np.abs(path - policy.qpos[i]))))
            trials.append(result)
    summary = {}
    for condition in conditions:
        rows = [r for r in trials if r["mode"] == condition]
        medians = {key: float(np.median([r["rise_and_settle"][key] for r in rows]))
                   for key in rows[0]["rise_and_settle"]}
        timing = [abs(r["rise_time_error_s"]) for r in rows if r["rise_time_error_s"] is not None]
        summary[condition] = dict(rose=sum(r["rose_by_deadline"] for r in rows),
                                  held=sum(r["final_half_second_upright"] for r in rows),
                                  max_qpos_error=max(r["max_abs_qpos_vs_plan"] for r in rows),
                                  max_abs_rise_time_error_s=max(timing, default=None), median_error=medians)
    return dict(conditions=conditions, summary=summary, trials=trials)


def export(trajectory, output, workers, support_friction):
    if workers < 1:
        raise ValueError("workers must be positive")
    if output.exists():
        raise FileExistsError(f"Choose an unused output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(trajectory, allow_pickle=False) as archive:
        source = dict(archive)
    plans = [(q, v, t, c, support_friction) for q, v, t, c in
             zip(source["qpos"], source["qvel"], source["target"], source["ctrl"], strict=True)]
    with ProcessPoolExecutor(workers) as executor:
        feedback = np.stack(list(executor.map(fit_case, plans)))

    with TemporaryDirectory(dir=output.parent) as temp:
        folder = Path(temp) / "run"
        folder.mkdir()
        np.savez_compressed(folder / "policy.npz", names=source["names"], qpos=source["qpos"],
                            qvel=source["qvel"], targets=source["target"], gains=feedback,
                            version=1, dt=env.DT, correction_limit=.65, support_friction=support_friction)
        policy = TimedFeedbackPolicy.load(folder / "policy.npz")
        evaluation = physically_evaluate(policy, source)
        (folder / "validation.json").write_text(json.dumps(evaluation, indent=2, allow_nan=False) + "\n")
        folder.rename(output)
    return evaluation["summary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--baseline-contacts", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args.trajectory, args.out, args.workers, not args.baseline_contacts)), flush=True)
