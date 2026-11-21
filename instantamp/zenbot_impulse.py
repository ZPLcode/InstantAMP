"""Short-horizon motor correction for a measured velocity impulse in planted stance."""

import mujoco
import numpy as np

from . import zenbot


JOINTS = [zenbot.DOF.index(f"{side}_{joint}") for side in ("left", "right")
          for joint in ("ankle_pitch", "ankle_roll", "hip_pitch", "hip_roll", "knee")]


def predict_impulse_response(controller, initial):
    model = controller.model
    scratch = mujoco.MjData(model)
    center = initial.qpos[:2].copy()
    before = initial.qpos.copy()

    def command(offset):
        result = np.zeros(23, dtype=np.float32)
        result[JOINTS] = offset
        return result

    def forecast(offset, length, follow, config):
        mujoco.mj_copyData(scratch, model, initial)
        previous, older = controller.previous.copy(), controller.older.copy()
        pulse = command(offset)
        for tick in range(length):
            action = controller.standing(scratch, config, previous, controller.anchors)
            action += pulse * min(1.0, (length - tick) / 4)
            target = controller.integrate(scratch, action, previous, older)
            older, previous = previous, target
        q, v = np.empty((follow, 30)), np.empty((follow, 29))
        n = controller.stance_forecast(scratch, controller.anchors, previous, older, config, q, v)
        return q[:n], v[:n]

    def error(offset, length, config):
        follow = max(8, 32 - length)
        q, v = forecast(offset, length, follow, config)
        residual = []
        for tick in (min(len(q) - 1, follow // 2 - 1), len(q) - 1):
            diff = np.empty(model.nv)
            mujoco.mj_differentiatePos(model, diff, 1.0, before, q[tick])
            residual.extend((diff[:3] * 8, diff[3:6] * 1.3, v[tick, :3] * 2, v[tick, 3:6] * 0.35))
        return np.concatenate(residual)

    # Bound motor offsets in the same normalized units as the native 23-action policy.
    eye = np.eye(len(JOINTS))
    for config in (controller.parameters, (0.018, 0.8), (0.012, 0.65), (0.025, 1.15)):
        config = np.asarray(config, dtype=float)
        for length in (12, 22, 7, 35):
            offset = np.zeros(len(JOINTS))
            for _ in range(6):
                q, v = forecast(offset, length, 160, config)
                height = q[:, 2].min()
                upright = (1.0 - 2.0 * np.square(q[:, 4:6]).sum(axis=1)).min()
                drift = np.linalg.norm(q[:, :2] - center, axis=1).max()
                speed = np.linalg.norm(v[-30:, :2], axis=1).max()
                if len(q) == 160 and height > 0.56 and upright > 0.866 and drift < 0.12 and speed < 0.05:
                    return config, command(offset), length
                residual = error(offset, length, config)
                jac = np.column_stack([(error(offset + 0.06 * direction, length, config) - residual) / 0.06
                                       for direction in eye])
                step = np.linalg.solve(jac.T @ jac + 0.02 * eye, -jac.T @ residual - 0.008 * offset)
                options = [np.clip(offset + scale * step, -1.3, 1.3) for scale in (1, 0.5, 0.25, 0.125)]
                costs = [np.linalg.norm(error(option, length, config)) + 0.015 * np.linalg.norm(option)
                         for option in options]
                best = np.argmin(costs)
                if costs[best] >= np.linalg.norm(residual) + 0.015 * np.linalg.norm(offset):
                    break
                offset = options[best]
    return None
