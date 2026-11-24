"""Checkpoint selection and reproducible starts for joint Zenbot policies."""

import json
from pathlib import Path

import numpy as np

from . import zenbot
from .native_zenbot import ZenbotNative
from .rot import mv3, quat_to_mat, tr3


def resolve_checkpoint(path, iteration=None):
    """Find a policy file or select a run's export by its learner iteration."""
    path = Path(path)
    if path.is_file():
        if iteration is not None:
            raise ValueError("--iteration requires a run directory")
        return path

    config = path / "config.json"
    if iteration is None and config.is_file():
        iteration = json.loads(config.read_text()).get("max_iterations") or None
    candidates = []
    for metadata in sorted((path / "ckpt").glob("*.json")):
        policy = metadata.with_suffix(".npz")
        if policy.is_file():
            step = json.loads(metadata.read_text())["it"]
            if iteration is None or step == iteration:
                candidates.append((step, policy))
    if not candidates:
        requested = (
            "latest iteration"
            if iteration is None
            else f"iteration {iteration}"
        )
        raise FileNotFoundError(f"No policy for {requested} in {path}/ckpt")
    return max(candidates)[1]


def fallen_starts(directory, lift=zenbot.RESET_LIFT):
    """Read fallen starts at 0 and 0.5 seconds, in native MuJoCo coordinates."""
    names, poses, velocities = [], [], []
    for path in sorted(Path(directory).glob("*.npz")):
        with np.load(path, allow_pickle=False) as clip:
            for seconds in (0.0, 0.5):
                i = round(seconds * float(clip["fps"]))
                if i >= len(clip["root_pos"]):
                    continue
                height = clip["root_pos"][i, 2] + lift
                if height > 0.5:
                    continue
                quat = clip["root_quat"][i, [3, 0, 1, 2]]
                angular = mv3(tr3(quat_to_mat(quat)), clip["root_ang_vel"][i])
                names.append((path.name, seconds))
                poses.append(np.r_[0.0, 0.0, height, quat, clip["dof_pos"][i]])
                velocities.append(
                    np.r_[clip["root_lin_vel"][i], angular, clip["dof_vel"][i]]
                )
    if not names:
        raise ValueError(f"No fallen starts in {directory}")
    return (
        names,
        np.asarray(poses, np.float32),
        np.asarray(velocities, np.float32),
    )


class JointTrials:
    """Run the same unassisted native policy from upright and fallen states."""

    def __init__(self, checkpoint, reference=None, seed=42):
        checkpoint = Path(checkpoint)
        config = checkpoint.parent.parent / "config.json"
        if (
            reference is None
            and checkpoint.parent.name == "ckpt"
            and config.is_file()
        ):
            reference = json.loads(config.read_text()).get("floor_dir")
        self.native = ZenbotNative(reference, checkpoint, seed)
        self.names, self.poses, self.velocities = fallen_starts(
            self.native.directory
        )
        self.delays = np.random.default_rng(seed).integers(
            3, 8, size=(len(self.names), self.native.motor_delay_groups)
        )

    def stand(self, command=(0.0, 0.0, 0.0)):
        """Reset to the model's neutral stance with a body-frame command."""
        profile = self.native.profile
        pose = np.r_[
            0.0,
            0.0,
            profile.DEFAULT_HEIGHT,
            1.0,
            0.0,
            0.0,
            0.0,
            profile.DEFAULT_Q,
        ]
        self.native.reset(
            pose, np.zeros(self.native.model.nv), self.delays[0], command
        )

    def fallen(self, trial):
        """Reset to a reproducible clip and offset with zero commanded velocity."""
        self.native.reset(
            self.poses[trial], self.velocities[trial], self.delays[trial]
        )
