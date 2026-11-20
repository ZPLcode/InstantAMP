"""Regressions for the independent LeggedLab G1 get-up task."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from instantamp import amp, g1_getup
from instantamp.sim import Sim, unflatten
from instantamp.train import parse


def test_task_uses_its_own_bank_and_profile():
    cfg = parse(["--task", g1_getup.TASK_ID])
    assert cfg.robot == "g1_getup" and cfg.log_dir == "runs/g1_getup"
    assert cfg.floor_dir == cfg.amp_dir == "data/g1_getup/recovery"
    assert cfg.amp_getup == cfg.push == 0 and cfg.amp_lerp == .5 and cfg.amp_style_scale == 50.
    assert parse(["--robot", "g1_getup"]).task == cfg.task
    with pytest.raises(SystemExit):
        parse(["--task", cfg.task, "--robot", "g1"])


def test_reference_resets_drives_and_observations(tmp_path):
    for i, n in enumerate((18, 95)):
        t = np.arange(n, dtype=np.float32)
        pos = np.column_stack((t, np.full(n, i), .25 + t / 200))
        q = np.broadcast_to(g1_getup.DEFAULT_Q, (n, 29))
        np.savez(tmp_path / f"{i}.npz", root_pos=pos, root_quat=np.tile([0, 0, 0, 1], (n, 1)),
                 root_lin_vel=pos * 0, root_ang_vel=pos * 0, dof_pos=q, dof_vel=q * 0, fps=50)
    sim = Sim(1024, floor_dir=tmp_path, robot="g1_getup", physics=False)
    key = jax.random.key(8)
    qpos, qvel, _ = sim._fresh(key, sim.world(key))
    x, clip, z = np.asarray(qpos[:, :3]).T
    np.testing.assert_allclose(z, .35 + x / 200, atol=1e-6)
    assert .45 < clip.mean() < .55
    assert x[clip == 0].max() < 8 and x[clip == 1].max() < 85
    assert sim.episode_steps == 500 and sim.assist_newtons == 200.
    np.testing.assert_allclose(sim.m.dof_armature[6:], .01)
    np.testing.assert_allclose(sim.m.actuator_forcerange[:, 1], g1_getup.EFFORT)
    rotating = qvel[:2].at[:, :3].set(jnp.array([1., 0., 0.])).at[:, 4].set(2.)
    corrected = sim.reference_velocity(qpos[:2], rotating)
    np.testing.assert_allclose(corrected[:, 0], 1. - 2. * sim.root_com[2], atol=1e-6)
    spinning = unflatten(qpos[:2], corrected)
    privileged = sim.push_critic(None, spinning, qvel[:2, 6:])
    np.testing.assert_allclose(privileged[:, :3], rotating[:, :3], atol=1e-6)
    state = unflatten(qpos[:2], qvel[:2])
    action = jnp.ones((2, 29))
    np.testing.assert_allclose(sim.control_targets(action, 0., None, None), g1_getup.DEFAULT_Q[None] + .25 * action)
    assert sim.support_link[sim.m.geom("robot_left_shin_collision").id] >= 0
    assert sim.support_link[sim.m.geom("robot_left_shoulder_yaw_collision").id] == -1
    assert amp.Demos(tmp_path, sim, 10).sample(key, 2).shape == (2, 10, 85)
    frame = sim.obs_frame(state, action, None, None)
    history = sim.push_actor(None, frame)
    updated = sim.push_actor(history, frame + 1, jnp.array([True, False]))
    assert history.shape == (2, 570) and sim.push_critic(None, state, action).shape == (2, 590)
    np.testing.assert_allclose(updated[1, :12], history[1, :12])
    np.testing.assert_allclose(updated[1, 12:15], frame[1, :3] + 1)
    np.testing.assert_allclose(updated[0, :15], np.tile(frame[0, :3] + 1, 5))
    pos = state[0].at[:, 2].set(jnp.array([.65, .75]))
    state = (pos, *state[1:])
    terms = sim.reward_terms(state, state, None, action, action, torque=0 * action)
    np.testing.assert_allclose(terms["target_base_height"], [0., 1.])
    np.testing.assert_allclose(sum(sim.weighted(terms).values()), [0., .22], atol=1e-5)
