"""Predict a short joint correction for a G1 disturbed after reaching a quiet stance."""

import mujoco
import numpy as np

from . import g1_getup
from .native_getup import advance


LEG_JOINTS = [g1_getup.DOF.index(f"{side}_{joint}") for side in ("left", "right")
              for joint in ("ankle_pitch", "ankle_roll", "hip_pitch", "hip_roll", "knee")]


def pulse_weight(frame, duration):
    return np.clip((duration - frame) / 4., 0., 1.)


def fit_stance_impulse(tracker, live, previous):
    """Fit bounded offsets on a copy of the currently loaded feet and the measured motor state."""
    model = tracker.model
    sim = mujoco.MjData(model)
    origin = live.qpos.copy()
    gain, strength = map(float, tracker.policy.params[tracker.starts[tracker.row] + tracker.cap_tick])
    solver = tracker.solver
    saved_data = solver.data
    eye = np.eye(len(LEG_JOINTS))

    def forecast(offset, duration, length):
        mujoco.mj_copyData(sim, model, live)
        last = previous.copy()
        positions, velocities = [], []
        for frame in range(length):
            residual = tracker._predicted_action(sim, last)
            solver.data = sim
            mujoco.mj_forward(model, sim)
            shift = tracker.loaded_balance.shift if tracker.loaded_balance is not None else None
            action = solver.action(tracker.footprint, tracker.pelvis, gain, strength, support_shift=shift)
            if frame < duration:
                action[LEG_JOINTS] += pulse_weight(frame, duration) * offset
            last = np.asarray(action, np.float32) + residual
            advance(model, sim, last)
            positions.append(sim.qpos.copy())
            velocities.append(sim.qvel.copy())
            if sim.qpos[2] < .48 or 1. - 2. * np.square(sim.qpos[4:6]).sum() < .50:
                break
        return np.asarray(positions), np.asarray(velocities)

    def motion_error(offset, duration):
        q, v = forecast(offset, duration, max(32, duration + 8))
        difference = np.empty(model.nv)
        terms = []
        for k in (min(len(q) - 1, duration + max(3, (32 - duration) // 2)), -1):
            mujoco.mj_differentiatePos(model, difference, 1., origin, q[k])
            terms.extend((8. * difference[:3], 1.3 * difference[3:6], 2. * v[k, :3], .35 * v[k, 3:6]))
        return np.concatenate(terms)

    try:
        for duration in (12, 22, 7, 35):
            offset = np.zeros(len(LEG_JOINTS))
            for _ in range(7):
                q, v = forecast(offset, duration, 150)
                upright = 1. - 2. * np.square(q[:, 4:6]).sum(axis=1)
                if (len(q) == 150 and q[:, 2].min() > .65 and upright.min() > np.cos(np.pi / 6)
                        and np.linalg.norm(q[:, :2] - origin[:2], axis=1).max() < .10
                        and np.linalg.norm(v[-25:, :2], axis=1).max() < .06):
                    return offset, duration
                error = motion_error(offset, duration)
                eps = .04
                jac = np.column_stack([(motion_error(offset + eps * axis, duration) - error) / eps
                                       for axis in eye])
                step = np.linalg.solve(jac.T @ jac + .018 * eye, -jac.T @ error - .002 * offset)
                # 0.90 normalized motor units correspond to 0.225 rad in the G1 position drives.
                proposals = [np.clip(offset + s * step, -.90, .90) for s in (1., .5, .25, .125)]
                scores = [np.linalg.norm(motion_error(candidate, duration)) + .007 * np.linalg.norm(candidate)
                          for candidate in proposals]
                best = int(np.argmin(scores))
                if scores[best] >= np.linalg.norm(error) + .007 * np.linalg.norm(offset):
                    break
                offset = proposals[best]
    finally:
        solver.data = saved_data
    return None
