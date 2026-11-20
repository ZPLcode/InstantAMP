import mujoco
import numpy as np
import pytest

from instantamp import env
from instantamp.sim import build_scene
from instantamp.timed_feedback import TimedFeedbackPolicy, state_error


def test_state_error_uses_local_quaternion_and_short_rotation():
    model = build_scene()
    base = model.qpos0.copy()
    base[3:7] = np.array([.7, .3, -.2, .6]) / np.linalg.norm([.7, .3, -.2, .6])
    current = base.copy()
    desired = np.linspace(-.15, .15, model.nv)
    mujoco.mj_integratePos(model, current, desired, 1.)
    # Both signs represent exactly the same measured attitude.
    current[3:7] *= -1
    zeros = np.zeros(model.nv)
    error = state_error(current, desired, base, zeros)
    np.testing.assert_allclose(error[:model.nv], desired, atol=1e-12)
    np.testing.assert_array_equal(error[model.nv:], desired)


def test_finite_clip_uses_measured_state_and_clamps_target_corrections(tmp_path):
    targets = np.ones((1, 2, env.J))
    poses = np.zeros((1, 3, 7 + env.J))
    poses[:, :, 3] = 1.
    speeds = np.zeros((1, 3, 6 + env.J))
    gains = np.zeros((1, 2, env.J, 2 * (6 + env.J)), dtype=np.float16)
    gains[0, 1, 0, 2] = -8.
    path = tmp_path / "export.npz"
    np.savez(path, names=["clip"], qpos=poses, qvel=speeds, targets=targets, gains=gains,
             dt=env.DT, correction_limit=.5, support_friction=True, version=1)
    policy = TimedFeedbackPolicy.load(path)
    q = poses[0,1].copy()
    q[2] += .2
    output = policy.target("clip", 1, q, speeds[0,1])
    expected = targets[0,1].copy()
    expected[0] -= .5 * env.EFFORT[0] / env.KP[0]
    np.testing.assert_allclose(output, expected)
    for case, tick in ((-1,0), (1,0), (0,-1), (0,2)):
        with pytest.raises(IndexError):
            policy.target(case, tick, q, speeds[0,1])
