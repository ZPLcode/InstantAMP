"""Regression tests for the Zenbot joint task and recovery motion prior."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from instantamp import zenbot
from instantamp.sim import Sim
from instantamp.train import parse
from tools.prepare_zenbot import convert


def test_joint_task_continues_from_a_full_checkpoint_in_a_new_run():
    args = ["--robot", zenbot.MIXED_TASK, "--init_full", "prior/ckpt/full/0015.npz", "--max_iterations", "10000"]
    cfg = parse([*args, "--log_dir", "continued"])
    assert cfg.init_full == "prior/ckpt/full/0015.npz" and cfg.max_iterations == 10000
    with pytest.raises(ValueError, match="new log directory"):
        parse([*args, "--log_dir", "prior"])


def test_recovery_pose_replay_pads_resets_and_gait_falls_in_a_wrapped_ring():
    from instantamp import buffer

    q = np.broadcast_to(np.arange(8)[:, None, None], (8, 4, 1)).astype(np.float32)
    end_q = q + 1
    end_q[5, 3] += 20
    t = np.broadcast_to(np.arange(60, 68)[:, None], (8, 4)).copy()
    t[-1, 1] = 0
    left = np.zeros((8, 4), np.int32)
    left[-3:, [0, 3]] = [[20], [19], [18]]
    left[-1, 1] = 20
    rec = {"s_qpos": q, "s_qvel": q * 0, "s1_qpos": end_q, "s1_qvel": q * 0, "t": t, "recovery_left": left,
           "ok": np.ones_like(t), "done": np.zeros_like(t)}
    rec["done"][-2, 1] = 1
    bank = {k: jnp.asarray(np.roll(v, 3, axis=0)) for k, v in rec.items()}
    bank.update(ptr=jnp.array(3), size=jnp.array(8))
    draw = jax.jit(lambda envs: buffer.sample_windows(bank, jax.random.PRNGKey(8), 1, 4, recent=1, envs=envs,
                                                      recovery_only=True, recovery_steps=20))
    fall, reset, no_recovery, jumped = (draw(jnp.array([i])) for i in range(4))
    np.testing.assert_array_equal(fall["s_qpos"][0, :, 0], [5, 5, 6, 7])
    np.testing.assert_array_equal(reset["s_qpos"][0, :, 0], [7, 7, 7, 7])
    np.testing.assert_array_equal(fall["mask"], 1.)
    np.testing.assert_array_equal(reset["mask"], 1.)
    np.testing.assert_array_equal(no_recovery["mask"], 0.)
    np.testing.assert_array_equal(jumped["mask"][0], [1., 1., 0., 0.])


def test_joint_task_keeps_command_and_motor_history_through_a_fall(tmp_path):
    from instantamp import amp

    cfg = parse(["--robot", zenbot.MIXED_TASK])
    assert cfg.walk_dir == cfg.amp_dir and cfg.floor_dir == cfg.amp_getup_dir != cfg.walk_dir
    motion = np.tile(zenbot.DEFAULT_Q, (25, 1))
    quat = np.tile([1., 0., 0., 0.], (25, 1))
    for name, height in (("walk_run", .69), ("recovery", .24)):
        pos = np.tile([0., 0., height], (25, 1))
        convert(tmp_path / name / "clip.npz", pos, quat, motion, list(range(23)), 50)
    sim = Sim(64, robot=cfg.robot, walk_dir=tmp_path / "walk_run", floor_dir=tmp_path / "recovery", physics=False)
    assert sim.mixed and sim.obs_dim == 390
    key = jax.random.PRNGKey(34)
    poses, _, up = sim._fresh(key, sim.world(key))
    assert up.any() and not up.all() and np.all(poses[~up, 2] < .4)
    rot = jnp.tile(jnp.eye(3), (2, 1, 1))
    np.testing.assert_array_equal(sim.next_upright(jnp.array([True, False]), jnp.array([.3, .69]), rot), [False, True])
    heights = jnp.array([.54, .54])
    np.testing.assert_array_equal(sim.next_upright(jnp.array([True, False]), heights, rot), [True, False])
    command = jnp.array([[1., 0., 0.], [2.5, 0., 0.]])
    zeros = jnp.zeros((2, 3))
    q = jnp.tile(zenbot.DEFAULT_Q, (2, 1))
    state = zeros, rot, zeros, zeros, q, jnp.zeros_like(q)
    frame = sim.obs_frame(state, jnp.ones_like(q), command, None)
    np.testing.assert_array_equal(frame[:, 6:9], command)
    hist = sim.push_actor(None, frame)
    xi = jnp.tile(jnp.array([1, 0, 4, 4, 4, 4, 4, 0]), (2, 1))
    targets = sim.control_targets(2 * jnp.ones_like(q), jnp.zeros_like(q), hist, xi).reshape(2, 4, 23)
    np.testing.assert_allclose(targets[:, 0], q + zenbot.ACTION_SCALE)
    assert sim.amp_frame(state).shape == (2, sim.amp_dim)
    for name in ("walk_run", "recovery"):
        demos = amp.Demos(tmp_path / name, sim, 10, per_clip=True)
        assert demos.sample(key, 3).shape == (3, 10, sim.amp_dim)


def test_amp_only_recovery_uses_physical_state_without_a_source_clip(tmp_path):
    from instantamp.train import InstantAMP

    pos = np.tile([0., 0., .25], (40, 1))
    quat = np.tile([1., 0., 0., 0.], (40, 1))
    q = np.tile(zenbot.DEFAULT_Q, (40, 1))
    convert(tmp_path / "floor.npz", pos, quat, q, list(range(23)), 50)
    sim = Sim(2, robot=zenbot.MIXED_TASK, walk_dir=tmp_path, floor_dir=tmp_path,
              recovery_amp_seconds=3.5, physics=False)
    assert not {"clip", "frame"} & sim.floor.keys()
    up = jnp.array([True, False])
    remaining = sim.next_recovery_interval(jnp.array([12, 0]), jnp.array([False, True]), up)
    np.testing.assert_array_equal(remaining, [11, 175])
    v = jnp.zeros((2, 3))
    joints = jnp.asarray(q[:2])
    state = jnp.asarray(pos[:2]), jnp.tile(jnp.eye(3), (2, 1, 1)), v, v, joints, jnp.zeros_like(joints)
    view = sim.view_state(state)
    args = state, state, view.kin, joints, joints, v, ~up, jnp.array([0., 1.]), sim.soft_contact(view.soles), jnp.zeros(2), sim.q_soft
    terms = sim.reward_terms(*args, recovery_active=remaining > 0)
    np.testing.assert_array_equal(terms["rise"], [0., 0.])
    np.testing.assert_array_equal(terms["track_lin"], [0., 0.])
    assert sim.weighted(terms)["termination"][1] < 0
    late = sim.reward_terms(*args, recovery_active=jnp.zeros(2, bool), recovery_deadline=jnp.ones(2, bool))
    assert late["rise"][1] > 0 and sim.weighted(late)["recovery_timeout_cost"][1] < 0
    xs = InstantAMP.__new__(InstantAMP)
    xs.sim, xs.cfg = sim, parse(["--robot", zenbot.MIXED_TASK])
    features = xs.ground_features(sim.amp_frame(state))
    np.testing.assert_allclose(features[:, 54:72:3], sim.key_positions(state[0], state[1], state[4])[..., 2], atol=1e-6)


def test_unpaired_amp_height_groups_sample_independent_motion_windows():
    from instantamp.amp import Demos

    bank = Demos.__new__(Demos)
    bank.window = 4
    bank.starts = jnp.array([0, 1, 2, 6, 7, 8, 12, 13, 14])
    bank.clip_lengths = jnp.array([3, 3, 3])
    heights = np.repeat([.2, .45, .68], 6)
    bank.frames = jnp.asarray(np.stack([heights, np.arange(18), heights, heights], axis=1))
    bank.prepare_ground_heights()
    windows = bank.sample_ground_heights(jax.random.PRNGKey(21), jnp.array([.31, .48, .64]))
    np.testing.assert_allclose(windows[:, -1, 0], [.2, .45, .68])
    np.testing.assert_allclose(np.diff(windows[:, :, 1], axis=1), np.ones((3, 3)))


def test_support_rsi_mixes_openings_with_independent_non_upright_states(tmp_path):
    n = 40
    pos = np.zeros((n, 3))
    pos[:, 2] = np.linspace(.20, .69, n)
    joints = np.tile(zenbot.DEFAULT_Q, (n, 1))
    joints[:, 3] = pos[:, 2]
    convert(tmp_path / "clip.npz", pos, np.tile([1., 0., 0., 0.], (n, 1)), joints, np.arange(23), 50)
    sim = Sim(256, robot=zenbot.MIXED_TASK, walk_dir=tmp_path, floor_dir=tmp_path, physics=False,
              reset_floor=1., recovery_amp_seconds=4., legged_support=True, support_rsi=.5)
    key = jax.random.PRNGKey(17)
    pose, velocity, up = sim._fresh(key, sim.world(key))
    heights = np.asarray(pose[:, 2])
    assert .35 < np.mean(np.isclose(heights, .22)) < .65 and len(np.unique(heights)) > 15
    assert not up.any() and not {"clip", "frame", "reference_start"} & sim.floor.keys()
    np.testing.assert_allclose(pose[:, 10], heights - zenbot.RESET_LIFT, atol=1e-6)
    with np.load(tmp_path / "clip.npz") as source:
        selected = np.abs(heights[:, None] - zenbot.RESET_LIFT - source["root_pos"][:, 2]).argmin(1)
        np.testing.assert_allclose(velocity[:, 6:], source["dof_vel"][selected])
        np.testing.assert_allclose(velocity[:, :3], source["root_lin_vel"][selected])
