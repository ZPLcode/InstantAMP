"""Zenbot articulation, actuator calibration and motion-bank conventions.

Joint arrays follow the hinge order in the model's MJCF. Motion Train exports
use a separate order, given by lab_order as indices into those arrays.
"""

import dataclasses
from pathlib import Path

import numpy as np

# Rows contain stiffness (N m/rad), damping (N m s/rad), rest angle (rad)
# and the actuator's delay group from Motion Train's control_parameters.json.
_LEG = {
    "hip_pitch": (180, 4, -0.174, 0),
    "hip_roll": (180, 4, 0, 0),
    "hip_yaw": (100, 2.6, 0, 0),
    "knee": (180, 4, 0.348, 1),
    "ankle_pitch": (70, 2.3, -0.174, 2),
    "ankle_roll": (60, 2, 0, 2),
}
_ARM = {
    "shoulder_pitch": (25, 2.3, 0.348, 3),
    "shoulder_roll": (25, 2.17, 0.279, 3),
    "shoulder_yaw": (25, 0.94, 0, 3),
    "elbow": (25, 1.15, 0.785, 4),
    "wrist_roll": (25, 0.61, 0, 3),
}
_WAIST_YAW = {"waist_yaw": (150, 4, 0, 0)}
_EXTRA = {
    "waist_roll": (80, 4, 0, 5),
    "waist_pitch": (80, 4, 0, 5),
    "head_yaw": (17.226699182516345, 0.6477238892626146, 0, 6),
}

# Per-group torque limits (N m) and maximum motoring speeds (rad/s).
_EFFORT = (130, 130, 120, 45, 45, 120, 12.5)
_SPEED = (31.4, 21.4, 18.3, 34.9, 30, 18.3, 47.12)


def _paired(rows):
    result = {}
    for side in ("left", "right"):
        for name, (kp, kd, rest, group) in rows.items():
            if side == "right" and name == "shoulder_roll":
                rest = -rest
            result[f"{side}_{name}"] = (kp, kd, rest, group)
    return result


@dataclasses.dataclass(frozen=True, eq=False)
class ZenbotModel:
    """Static properties shared by tasks that use the same articulation."""

    dof: tuple[str, ...]
    kp: np.ndarray
    kd: np.ndarray
    default_q: np.ndarray
    group: np.ndarray
    effort: np.ndarray
    speed: np.ndarray
    lab_order: np.ndarray
    mass: float
    head_body: str

    @property
    def mixed_task(self) -> str:
        return f"zenbot{len(self.dof)}_walk_run_recovery"

    @property
    def mixed_recovery_dir(self) -> str:
        return f"data/{self.mixed_task}/recovery"

    @property
    def xml_path(self) -> Path:
        assets = Path(__file__).resolve().parents[1] / "assets"
        return assets / f"zenbot_{len(self.dof)}dof_43kg.xml"

    @property
    def lab_joints(self) -> tuple[str, ...]:
        return tuple(self.dof[i] for i in self.lab_order)

    @property
    def wrist_indices(self) -> tuple[int, ...]:
        return tuple(i for i, name in enumerate(self.dof) if "wrist" in name)

    def motion_dirs(self) -> tuple[str, str]:
        """Return the upright and fallen reset banks for combined training."""
        return f"data/{self.mixed_task}/walk_run", self.mixed_recovery_dir


def _model(*, spine, lab_order, mass, head_body):
    joints = {**_paired(_LEG), **spine, **_paired(_ARM)}
    kp, kd, default_q, groups = np.array(list(joints.values()), np.float32).T
    groups = groups.astype(np.int32)
    effort = np.array(_EFFORT, np.float32)[groups]
    speed = np.array(_SPEED, np.float32)[groups]
    order = np.array(lab_order, dtype=np.intp)
    for array in (kp, kd, default_q, groups, effort, speed, order):
        array.setflags(write=False)
    return ZenbotModel(
        dof=tuple(joints), kp=kp, kd=kd, default_q=default_q, group=groups,
        effort=effort, speed=speed, lab_order=order, mass=mass,
        head_body=head_body,
    )


# lab_order follows Motion Train's usd2urdf mapping for each articulation.
ZENBOT23 = _model(
    spine=_WAIST_YAW,
    lab_order=(0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20,
               4, 10, 16, 21, 5, 11, 17, 22),
    mass=43.0678308,
    head_body="head_Link",
)
ZENBOT26 = _model(
    spine={**_WAIST_YAW, **_EXTRA},
    lab_order=(0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 16, 21, 4, 10,
               17, 22, 5, 11, 18, 23, 19, 24, 20, 25),
    mass=42.7090468,
    head_body="head_yaw_Link",
)
MODELS = (ZENBOT23, ZENBOT26)
MIXED_TASKS = tuple(model.mixed_task for model in MODELS)


def for_task(task: str) -> ZenbotModel:
    """Resolve a Zenbot task name from a CLI argument or saved configuration."""
    for model in MODELS:
        if task == model.mixed_task:
            return model
    raise ValueError(f"Unknown Zenbot task {task!r}")


def for_joint_count(count: int) -> ZenbotModel:
    """Resolve an articulation from the output width of a policy or clip."""
    for model in MODELS:
        if len(model.dof) == count:
            return model
    raise ValueError(f"Unsupported Zenbot joint count {count}")
