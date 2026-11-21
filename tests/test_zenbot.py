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
    assert cfg.reference_weight == 5. and cfg.assist_newtons == zenbot.INITIAL_ASSIST
    continuation = parse(["--robot", cfg.robot, "--init_full", "prior/ckpt/full/0000.npz",
                          "--assist_newtons", "0", "--action_rate_l2_weight", "-0.015"])
    assert continuation.assist_newtons == 0 and continuation.action_rate_l2_weight == -.015
    with pytest.raises(ValueError, match="new log directory"):
        parse(["--robot", cfg.robot, "--log_dir", "prior", "--init_full", "prior/ckpt/full/0000.npz"])
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


def test_zenbot_reference_clip_phase_and_gradient(sim, tmp_path):
    with np.load(tmp_path / "test.npz") as clip:
        shifted = dict(clip)
    shifted["root_pos"] = shifted["root_pos"][:12].copy()
    shifted["root_pos"][:, 2] += 0.25
    shifted = {k: v[:12] if v.ndim else v for k, v in shifted.items()}
    np.savez(tmp_path / "two.npz", **shifted)
    reference = zenbot.RecoveryReference(tmp_path, sim.key_positions)
    np.testing.assert_array_equal(reference.lengths, [20, 12])
    state = reference.poses[reference.offsets[1] + 5]
    clip, step = jnp.array(1), jnp.array(5)
    assert reference.score(state, clip, step) == pytest.approx(1.)
    assert reference.score(state, 0, step) < 1.
    assert reference.score(state, clip, step + 4) < 1.
    assert reference.score(state, clip, 12) == 0.
    grad = jax.grad(lambda height: reference.score(state.at[0].set(height), clip, step))(state[0] - 0.12)
    assert np.isfinite(grad) and grad > 0


def test_zenbot_phase_resets_preserve_the_selected_reference_frame(sim, tmp_path):
    mixed = Sim(64, walk_dir=str(tmp_path), floor_dir=str(tmp_path), robot="zenbot23_getup",
                physics=False, reference_weight=10., zenbot_phase_resets=.7, zenbot_episode_steps=225)
    key = jax.random.PRNGKey(11)
    (qpos, _, _), (clip, start) = mixed._fresh(key, mixed.world(key), return_source=True)
    assert mixed.episode_steps == 225
    assert np.all(np.asarray(clip) == 0)
    assert 5 < int(jnp.sum(start == 0)) < 45
    np.testing.assert_allclose(qpos[:, 2], mixed.walk["z"][start] + zenbot.RESET_LIFT)
    np.testing.assert_allclose(qpos[:, 7:], mixed.walk["q"][start])
    with pytest.raises(ValueError, match="tracking reward"):
        Sim(2, walk_dir=str(tmp_path), floor_dir=str(tmp_path), robot="zenbot23_getup",
            physics=False, reference_weight=0., zenbot_phase_resets=.5)


def test_local_rise_uses_the_same_sensor_fields_in_flat_and_batched_histories(tmp_path):
    from instantamp.inference import load, policy_output
    from instantamp.zenbot_rise import features

    sensor = np.zeros((5, 5, 96), dtype=np.float32)
    sensor[:, :, 9] = np.linspace(-.9, .9, 5)[:, None]
    obs = sensor.reshape(5, 480)
    commands = np.arange(115, dtype=np.float32).reshape(5, 23) / 10.
    flat = {
        'zenbot_library/keys': np.asarray(features(jnp.asarray(obs))),
        'zenbot_library/joints': sensor[:, -1, 9:32],
        'zenbot_library/velocities': sensor[:, -1, 32:55],
        'zenbot_library/commands': commands,
        'zenbot_library/gain': np.float32(.35),
        'zenbot_library/residual_scale': np.float32(.06),
        'avg/0/w': np.zeros((480,23), dtype=np.float32),
        'avg/0/b': np.ones(23, dtype=np.float32),
        'obs_norm/mean': np.zeros(480, dtype=np.float32),
        'obs_norm/var': np.ones(480, dtype=np.float32),
    }
    checkpoint = tmp_path / 'local.npz'
    np.savez(checkpoint, **flat)
    policy, norm, ode = load(checkpoint)
    assert ode is None
    action = jax.jit(lambda x: policy_output(policy,norm,x))
    np.testing.assert_allclose(action(obs), commands + .06*np.tanh(1.), atol=2e-6)
    np.testing.assert_allclose(action(obs[2]), action(obs)[2], atol=2e-6)
    flat['zenbot_library/commands'] = commands[:-1]
    np.savez(checkpoint, **flat)
    with pytest.raises(ValueError, match='dimensions'):
        load(checkpoint)


def test_stance_uses_per_robot_memory_and_resets_on_done():
    from instantamp.inference import policy_memory, policy_step, reset_policy_memory
    from instantamp.zenbot_rise import ZenbotLibrary, ZenbotRisePolicy
    from instantamp.zenbot_stance import ZenbotStance, stance_key

    frame = np.zeros((2, 96), dtype=np.float32)
    frame[:, 3] = 1.
    frame[0, 8] = 1.
    frame[1, 8] = -1.
    frame[:, 9:32] = zenbot.DEFAULT_Q
    history = jnp.asarray(np.tile(frame, (1, 5)))
    archive = {
        "zenbot_stance/keys": stance_key(frame[:1]),
        "zenbot_stance/parameters": np.array([[.018, .80]]),
        "zenbot_stance/settings": np.array([.10, 0., 0.]),
    }
    controller = ZenbotStance.from_archive(archive)
    flat = {
        "keys": np.zeros((4, 192)), "commands": np.ones((4, 23)),
        "joints": np.tile(zenbot.DEFAULT_Q, (4, 1)), "velocities": np.zeros((4, 23)),
        "gain": np.float32(0.),
    }
    bank = ZenbotLibrary.from_archive({"zenbot_library/" + key: value for key, value in flat.items()})
    layers = [{"w": jnp.zeros((480, 23)), "b": jnp.zeros(23)}]
    policy = ZenbotRisePolicy(bank, layers, jnp.asarray(0.), stance=controller)
    initial = policy_memory(policy, history)
    out, state = jax.jit(lambda obs, mem: policy_step(policy, {"raw": True}, obs, mem))(history, initial)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(state[:, 0], [1., 0.])
    np.testing.assert_allclose(out[1], 1.)
    np.testing.assert_array_equal(reset_policy_memory(state, jnp.asarray([True, False])), 0.)
    assert policy_memory(policy._replace(stance=None), history) is None
    with pytest.raises(ValueError, match="Invalid Zenbot sensor stance"):
        ZenbotStance.from_archive({**archive, "zenbot_stance/parameters": np.zeros((2, 2))})
