"""Robot selection, motion layout and delayed-target regressions, runnable on CPU."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from instantamp import env, nets, zenbot
from instantamp.sim import Sim, build_scene
from instantamp.train import parse
from tools.prepare_zenbot import convert, permutation


@pytest.fixture
def sim(tmp_path):
    # Non-symmetric joint angles catch a silent Isaac/MuJoCo column permutation.
    names = [joint + "_joint" for joint in reversed(zenbot.DOF)]
    q = np.tile(np.arange(23) / 100, (20, 1))
    pos = np.zeros((20, 3))
    pos[:, 2] = np.linspace(0.4, 0.69, len(q))
    wxyz = np.tile([1., 0., 0., 0.], (len(q), 1))
    wxyz[1::2] *= -1
    convert(tmp_path / "test.npz", pos, wxyz, q, permutation(names), 50)
    return Sim(2, walk_dir=str(tmp_path), floor_dir=str(tmp_path), robot="zenbot23_getup", physics=False)


def test_zenbot_model_and_banks(sim):
    m = sim.m
    assert (m.nq, m.nv, m.nu, sim.obs_dim, sim.amp_dim) == (30, 29, 23, 480, 73)
    assert m.body_mass.sum() == pytest.approx(43.0678308)
    np.testing.assert_allclose(sim.walk["q"][0], np.arange(22, -1, -1) / 100)
    np.testing.assert_allclose(sim.walk["w"], 0, atol=1e-7)
    key = jax.random.PRNGKey(1)
    poses, _, up = sim._fresh(key, sim.world(key))
    assert sim.walk["q"].shape == (1, 23)
    np.testing.assert_allclose(poses[:, 2], 0.42)
    np.testing.assert_allclose(poses[:, :2], 0)
    assert not up.any()
    np.testing.assert_allclose(m.actuator_ctrlrange[:, 1], zenbot.EFFORT)
    d = mujoco.MjData(m)
    d.qpos[2], d.qpos[7:] = zenbot.DEFAULT_HEIGHT, zenbot.DEFAULT_Q
    mujoco.mj_forward(m, d)
    for geom in sim.foot_geoms:
        rot = d.geom_xmat[geom].reshape(3, 3)
        bottom = d.geom_xpos[geom, 2] - abs(rot[2]) @ m.geom_size[geom]
        assert abs(bottom) < 0.005
    contacts = {frozenset((c.geom1, c.geom2)) for c in d.contact}
    assert all(frozenset((sim.floor_id, foot)) in contacts for foot in sim.foot_geoms)
    assert build_scene().nu == 29  # G1 and Zenbot coexist in one interpreter.


def test_substep_targets_and_model_input(sim):
    history = np.zeros((2, sim.history, sim.frame_dim), np.float32)
    history[:, -2, -41:-18] = 1
    history[:, -1, -41:-18] = 2
    action = jnp.full((2, 23), 3.)
    bias = jnp.full((2, 23), 0.01)
    context = np.zeros((2, 8), np.float32)
    context[:, 2:7] = [[3, 4, 5, 6, 7], [7, 6, 5, 4, 3]]
    targets = sim.control_targets(action, bias, jnp.array(history.reshape(2, -1)), jnp.array(context))
    expected = np.empty((2, 4, 23))
    for i, delays in enumerate(context[:, 2:7]):
        for j, group in enumerate(zenbot.GROUP):
            for t in range(4):
                ix = 2 + (t - int(delays[group])) // 4
                expected[i, t, j] = zenbot.DEFAULT_Q[j] + 0.25 * (1 + ix) - 0.01
    np.testing.assert_allclose(targets, expected.reshape(2, -1), atol=1e-6)
    q = jnp.broadcast_to(zenbot.DEFAULT_Q, (2, 23))
    z = jnp.zeros((2, 3))
    state = z, jnp.broadcast_to(jnp.eye(3), (2, 3, 3)), z, z, q, jnp.zeros_like(q)
    x = nets.wm_inputs(state, z[:, 0], jnp.zeros((2, 8)), targets, context)
    np.testing.assert_allclose(x[:, -(targets.shape[-1] + context.shape[-1]):-context.shape[-1]], (expected - np.asarray(q)[:, None]).reshape(2, -1), atol=1e-6)
    assert sim.amp_frame(state).shape == (2, 73)


def test_robot_cli_paths():
    cfg = parse(["--robot", "zenbot23_getup", "--amp_dir", "custom"])
    assert cfg.walk_dir == "data/zenbot23_getup/recovery" and cfg.amp_dir == "custom"
    assert cfg.amp_getup_dir == cfg.floor_dir == "data/zenbot23_getup/recovery"
    assert cfg.max_iterations == 100_000 and cfg.ppo_len == 24
    assert cfg.push == 0 and cfg.amp_getup == 0 and cfg.amp_style_scale == 50.
    assert parse([]).log_dir == "runs/instantamp"


def test_recovery_reward_and_observations(sim):
    pos = jnp.array([[0., 0., 0.30], [0., 0., 0.68]])
    rot = jnp.tile(jnp.eye(3), (2, 1, 1))
    q = jnp.tile(zenbot.DEFAULT_Q, (2, 1))
    z = jnp.zeros_like(q)
    v = jnp.zeros_like(pos)
    state = pos, rot, v, v, q, z
    terms = zenbot.reward_terms(state, state, z, z, z, sim.q_soft, env.DT)
    assert tuple(terms) == sim.terms
    np.testing.assert_allclose(terms["target_base_height"], [0., 1.])
    np.testing.assert_allclose(sum(sim.weighted(terms).values()), [0., 11. * env.DT])
    obs = sim.obs_frame(state, z + 2, v, None)
    disc = sim.amp_frame(state)
    np.testing.assert_allclose(obs[:, -41:-18], 2)
    np.testing.assert_allclose(obs[:, -18:], disc[:, -18:])
    np.testing.assert_allclose(disc[:, :6], [[1, 0, 0, 0, 0, 1]] * 2)
