"""Both joint policies use the same observation, motor and contact pipeline."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import mujoco
import numpy as np
import pytest

from instantamp import env, zenbot_models
from instantamp.native_zenbot import ZenbotNative


@pytest.mark.parametrize("model", zenbot_models.MODELS, ids=lambda m: m.mixed_task)
@pytest.mark.parametrize("hold_first_command", [False, True])
def test_joint_actor_controls_native_articulation(tmp_path, model, hold_first_command):
    joints = len(model.dof)
    width = 3 * (9 + 3 * joints)
    weights = np.zeros((width, joints), np.float32)
    # The most recent forward command scales each joint's action differently.
    weights[24] = np.linspace(-0.7, 0.7, joints)
    checkpoint = tmp_path / "policy.npz"
    np.savez(checkpoint, **{
        "avg/0/w": weights,
        "avg/0/b": np.zeros(joints, np.float32),
        "obs_norm/mean": np.zeros(width, np.float32),
        "obs_norm/var": np.ones(width, np.float32),
        "obs_norm/raw": True,
        "zenbot/delay_uses_first_command": hold_first_command,
    })
    player = ZenbotNative(None, checkpoint)
    assert player.directory.as_posix().endswith(f"{model.mixed_task}/recovery")
    assert player.model.nu == joints
    np.testing.assert_allclose(player.model.body_mass.sum(), model.mass)
    pose = np.r_[0., 0., .72, 1., 0., 0., 0., model.default_q]
    velocity = np.zeros(player.model.nv)
    delays = np.arange(player.motor_delay_groups) % env.DECIMATION
    player.reset(pose, velocity, delays, command=(.5, 0., 0.))

    expected = mujoco.MjData(player.model)
    expected.qpos[:] = pose
    expected.qvel[:] = velocity
    mujoco.mj_forward(player.model, expected)
    action = .5 * weights[24]
    for substep in range(env.DECIMATION):
        arrived = hold_first_command | (substep >= delays[model.group])
        applied = np.where(arrived, action, 0.)
        target = model.default_q + player.profile.ACTION_SCALE * applied
        expected.ctrl[:] = player.profile.actuator_ctrl(
            expected.qpos[7:], expected.qvel[6:], target
        )
        mujoco.mj_step(player.model, expected)

    foot_load = player.step()
    np.testing.assert_allclose(player.previous, action)
    np.testing.assert_allclose(player.data.qpos, expected.qpos, atol=2e-7)
    np.testing.assert_allclose(player.data.qvel, expected.qvel, atol=1e-5)
    assert player.data.time == pytest.approx(env.DT)
    assert player.flight.shape == (env.DECIMATION,)
    assert foot_load == player.support_load[:2].sum()
