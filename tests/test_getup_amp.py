"""Check the G1 recovery prior on untrimmed fall-and-get-up motions."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from instantamp.getup_amp import GetupDemos


class MotionFeatures:
    J = 1

    @staticmethod
    def amp_frame(state):
        return state[4]

    @staticmethod
    def is_up(height, rotation):
        return height > 0.60


def write_clip(path, height, first_joint_value):
    height = np.asarray(height, dtype=np.float32)
    n = len(height)
    pos = np.zeros((n, 3), np.float32)
    pos[:, 2] = height
    vel = np.gradient(pos, axis=0) * 50
    joints = (first_joint_value + np.arange(n, dtype=np.float32))[:, None]
    np.savez(path, fps=50, root_pos=pos, root_quat=np.tile([0., 0., 0., 1.], (n, 1)),
             dof_pos=joints, dof_vel=joints * 0, root_ang_vel=pos * 0, root_lin_vel=vel)
    return vel[:, 2]


def test_recovery_windows_keep_only_all_down_frames(tmp_path):
    write_clip(tmp_path / "sequence.npz", [.12, .30, .50, .70, .65, .42, .30, .15], 0)
    bank = GetupDemos(tmp_path, MotionFeatures(), 3)
    np.testing.assert_array_equal(bank.starts, [0, 5])
    assert bank.frames.shape == (8, 1)
