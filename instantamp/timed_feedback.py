"""50 Hz state feedback for a selected, previously planned G1 reference clip."""

from dataclasses import dataclass
import operator
from pathlib import Path

import numpy as np

from . import env


def state_error(qpos, qvel, nominal_qpos, nominal_qvel):
    """Current minus nominal pose and velocity, using MuJoCo's free-joint tangent order."""
    w0 = nominal_qpos[3]
    axis0 = nominal_qpos[4:7]
    w = qpos[3]
    axis = qpos[4:7]

    # Quaternion product conjugate(nominal) * measured, on the shorter rotation arc.
    scalar = w0 * w + axis0 @ axis
    vector = w0 * axis - w * axis0 - np.cross(axis0, axis)
    if scalar < 0:
        scalar, vector = -scalar, -vector
    length = np.linalg.norm(vector)
    angle_scale = 2 * np.arctan2(length, scalar) / length if length > 1e-12 else 2.0
    return np.concatenate((
        qpos[:3] - nominal_qpos[:3],
        angle_scale * vector,
        qpos[7:] - nominal_qpos[7:],
        qvel - nominal_qvel,
    ))


@dataclass(frozen=True)
class TimedFeedbackPolicy:
    names: tuple[str, ...]
    qpos: np.ndarray
    qvel: np.ndarray
    targets: np.ndarray
    gains: np.ndarray
    correction_limit: float
    support_friction: bool

    @classmethod
    def load(cls, path: Path | str):
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != 1 or not np.isclose(float(data["dt"]), env.DT):
                raise ValueError("Unsupported G1 feedback export or control period")
            names = tuple(map(str, data["names"]))
            qpos, qvel, targets = (data[key] for key in ("qpos", "qvel", "targets"))
            gains = data["gains"].astype(np.float32)
            limit = float(data["correction_limit"])
            support_friction = bool(data["support_friction"])

        if targets.ndim != 3:
            raise ValueError("Expected joint targets with shape (clips, control steps, joints)")
        cases, ticks, joints = targets.shape
        if joints != env.J or cases == 0 or ticks == 0 or len(names) != cases or len(set(names)) != cases:
            raise ValueError("Invalid clip names or G1 joint targets")
        expected = (cases, ticks + 1)
        if (qpos.shape != (*expected, 7 + joints) or qvel.shape != (*expected, 6 + joints)
                or gains.shape != (cases, ticks, joints, 2 * (joints + 6))):
            raise ValueError("State and feedback gain dimensions must match the joint targets")
        if not 0 < limit <= 2 or not all(np.isfinite(x).all() for x in (qpos, qvel, targets, gains)):
            raise ValueError("Nonfinite policy parameters or invalid motor correction limit")
        return cls(names, qpos, qvel, targets, gains, limit, support_friction)

    def target(self, case, tick, qpos, qvel):
        """Return 29 position setpoints in radians. The state must use the original planning frame."""
        case = self.names.index(case) if isinstance(case, str) else operator.index(case)
        tick = operator.index(tick)
        if not 0 <= case < len(self.names) or not 0 <= tick < self.targets.shape[1]:
            raise IndexError("Requested clip or 50 Hz tick lies outside the exported trajectory")
        qpos, qvel = np.asarray(qpos), np.asarray(qvel)
        if qpos.shape != (7 + env.J,) or qvel.shape != (6 + env.J,):
            raise ValueError("Expected one measured G1 qpos (36) and qvel (35)")
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("Nonfinite measured G1 state")
        error = state_error(qpos, qvel, self.qpos[case, tick], self.qvel[case, tick])
        offset = self.gains[case, tick] @ error
        bounded = np.clip(offset, -self.correction_limit, self.correction_limit)
        return self.targets[case, tick] + bounded * (env.EFFORT / env.KP)
