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


def test_getup_preserves_out_of_range_pd_requests_at_every_delayed_substep(sim):
    n = len(zenbot.DOF)
    # Mix both extremes and feasible requests in each historical action.
    action = np.resize([80., -80., .1], (2, n)).astype(np.float32)
    history = np.zeros((2, sim.history, sim.frame_dim), np.float32)
    history[:, -1, 9+2*n:9+3*n] = np.roll(action, 1, axis=-1)
    history[:, -2, 9+2*n:9+3*n] = np.roll(action, 2, axis=-1)
    bias = np.linspace(-.04, .04, n, dtype=np.float32)[None].repeat(2, axis=0)
    context = np.zeros((2, 8), np.float32)
    context[:, 2:7] = [[3, 4, 5, 6, 7], [7, 6, 5, 4, 3]]
    args = tuple(map(jnp.asarray, (action, bias, history.reshape(2, -1), context)))
    raw = np.asarray(sim.control_targets(*args)).reshape(2, 4, n)
    lo, hi = sim.m.jnt_range[sim.m.actuator_trnid[:, 0]].T
    assert (raw < lo).any() and (raw > hi).any() and ((raw > lo) & (raw < hi)).any()
    commands = np.stack((np.roll(action, 2, axis=-1), np.roll(action, 1, axis=-1), action), axis=1)
    for world in range(len(context)):
        for substep in range(env.DECIMATION):
            delays = context[world, 2:7][zenbot.GROUP].astype(int)
            index = 2 + (substep - delays) // env.DECIMATION
            expected = zenbot.DEFAULT_Q + zenbot.ACTION_SCALE * commands[world, index, np.arange(n)] - bias[world]
            np.testing.assert_allclose(raw[world, substep], expected, atol=2e-7)

    # The physical actuator still enforces the torque and motoring speed limits.
    q = jnp.asarray(zenbot.DEFAULT_Q)
    far = q + 100.
    np.testing.assert_allclose(zenbot.actuator_ctrl(q, jnp.zeros_like(q), far), zenbot.EFFORT)
    np.testing.assert_allclose(zenbot.actuator_ctrl(q, zenbot.SPEED + 1., far), 0.)


def test_native_cold_start_uses_the_checkpoint_motor_queue_convention(tmp_path, sim):
    from instantamp.native_zenbot import ZenbotNative
    from instantamp.train import save_policy

    folder = tmp_path / "policies"
    folder.mkdir()
    layer = {"w": np.zeros((480, 23), np.float32), "b": np.full(23, 2., np.float32)}
    policy = {"avg": [layer], "pi": [layer],
              "obs_norm": {"mean": np.zeros(480), "var": np.ones(480), "raw": True}}
    for name, fill in (("old", False), ("new", True)):
        save_policy(str(folder / name), {**policy, "delay_uses_first_command": fill}, {})
    native = ZenbotNative(tmp_path, folder / "new.npz")
    key = jax.random.key(7)
    pose, velocity, _ = sim._fresh(key, sim.world(key))

    def first_state(lag):
        native.reset(pose[0], velocity[0], np.full(5, lag))
        native.step()
        return native.data.qvel.copy()

    fresh = first_state(7)
    np.testing.assert_allclose(fresh, first_state(0), atol=1e-8)
    native.replace_policy(folder / "old.npz")
    assert np.linalg.norm(first_state(7) - fresh) > 1.
    native.replace_policy(folder / "new.npz")
    np.testing.assert_allclose(first_state(7), fresh, atol=1e-8)


def test_native_checkpoint_swap_preserves_live_state_and_rolls_back_invalid_model(tmp_path, sim):
    from instantamp.native_zenbot import ZenbotNative

    checkpoints = tmp_path / "actors"
    checkpoints.mkdir()
    params = {"avg/0/w": np.zeros((480, 23), np.float32), "obs_norm/raw": True,
              "obs_norm/mean": np.zeros(480, np.float32), "obs_norm/var": np.ones(480, np.float32)}
    for name, bias in (("first", np.full(23, .25)), ("next", np.full(23, -.75)), ("bad", np.zeros(24))):
        np.savez(checkpoints / name, **params, **{"avg/0/b": bias.astype(np.float32)})
    large_bias = np.full(23, 80., dtype=np.float32)
    for name, tag in (("legacy", {}), ("bounded", {"zenbot/clip_pd_targets": True})):
        np.savez(checkpoints / name, **params, **{"avg/0/b": large_bias}, **tag)
    player = ZenbotNative(tmp_path, checkpoints / "first.npz")
    key = jax.random.PRNGKey(7)
    q, v, _ = sim._fresh(key, sim.world(key))
    player.reset(q[0], v[0], np.arange(3, 8))
    player.replace_policy(checkpoints / "first.npz")
    player.step()
    np.testing.assert_array_equal(player.previous, .25)

    control = player._dense_control
    model, data, time = player.model, player.data, player.data.time
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    previous, history, delays = player.previous, player.history, player.delays
    player.replace_policy(checkpoints / "next.npz")
    assert player._dense_control is control and player.model is model and player.data is data
    assert data.time == time and player.previous is previous and player.history is history and player.delays is delays
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    player.step()
    np.testing.assert_array_equal(player.previous, -.75)
    with pytest.raises(ValueError):
        player.replace_policy(checkpoints / "bad.npz")
    player.step()
    np.testing.assert_array_equal(player.previous, -.75)

    # Historical archives can still carry the obsolete target-clamp flag.
    bounded = ZenbotNative(tmp_path, checkpoints / "bounded.npz")
    bounded.reset(q[0], v[0], np.zeros(5, dtype=int))
    for name in (None, "legacy", "bounded"):
        if name:
            bounded.replace_policy(checkpoints / f"{name}.npz")
        action, _, _, targets = bounded.control(jnp.asarray(q[0]), jnp.asarray(v[0]), bounded.previous,
                                                bounded.history, bounded.memory, bounded.delays,
                                                bounded.zeros, bounded.command)
        np.testing.assert_array_equal(action, large_bias)
        expected = zenbot.DEFAULT_Q + zenbot.ACTION_SCALE * large_bias
        np.testing.assert_allclose(targets, np.tile(expected, (env.DECIMATION, 1)), atol=2e-7)


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


def test_joint_task_continues_from_a_full_checkpoint_in_a_new_run():
    args = ["--robot", zenbot.MIXED_TASK, "--init_full", "prior/ckpt/full/0015.npz", "--max_iterations", "10000"]
    cfg = parse([*args, "--log_dir", "continued"])
    assert cfg.init_full == "prior/ckpt/full/0015.npz" and cfg.max_iterations == 10000
    with pytest.raises(ValueError, match="new log directory"):
        parse([*args, "--log_dir", "prior"])


def test_rise_style_separates_timing_from_joint_shape():
    from tools.evaluate_zenbot_walk_recovery import rise_style

    pose = np.zeros((5, 30))
    pose[:, 2] = [.2, .3, .4, .5, .6]
    pose[:, 3] = 1.
    pose[:, 7:] = np.arange(5)[:, None] * .1
    reference = {"fps": 50, "root_pos": pose[:, :3], "root_quat": pose[:, [4, 5, 6, 3]], "dof_pos": pose[:, 7:]}
    faster = pose[::2]
    score = rise_style(faster, reference)
    assert score["policy_upright_s"] == .04 and score["reference_upright_s"] == .08
    assert score["phase_aligned_joint_mae_deg"] == pytest.approx(0.)
    no_rise = rise_style(faster[:-1], reference)
    assert no_rise["policy_upright_s"] is None and no_rise["phase_aligned_joint_mae_deg"] is None
    timed = rise_style(np.repeat(faster[-1:], len(pose), axis=0), reference)
    assert timed["source_time_joint_mae_deg"] > 10 and timed["source_time_pelvis_rmse_m"] > .2


def test_getup_summary_compares_at_reset_offset_and_conditions_on_observed_outcomes(tmp_path):
    from tools.evaluate_zenbot_getup import style_case, summarize_cases

    frames = 130
    pose = np.tile(np.r_[0., 0., .69, 1., 0., 0., 0., zenbot.DEFAULT_Q], (frames, 1))
    pose[:, 2] = np.linspace(.25, .69, frames)
    pose[:, 7] += np.linspace(.4, 0., frames)
    np.savez(tmp_path / "rise.npz", root_pos=pose[:, :3], root_quat=pose[:, [4, 5, 6, 3]],
             dof_pos=pose[:, 7:], fps=50.)
    row = {"seed": 17, "trial": 1, "clip": "rise.npz", "offset": .5, "rise": 1.5, "held": True, "refall": True}
    case = style_case(row, pose[25:], build_scene(robot_name="zenbot23_getup"), tmp_path)
    assert case["seed"] == 17 and case["offset_s"] == .5
    assert case["style"]["source_time_joint_mae_deg"] == pytest.approx(0.)
    assert case["style"]["phase_aligned_joint_mae_deg"] == pytest.approx(0.)
    failed = {**case, "rose": False, "held": False, "rise_s": None, "default_pose_mae_deg": 99.}
    stats = summarize_cases([case, failed])
    assert (stats["trials"], stats["rise"], stats["quiet_final_2s"]) == (2, 1, 1)
    assert stats["refalls_after_first_stand"] == 1
    assert stats["median_torso_tilt_deg_on_quiet_holds"] == pytest.approx(0.)
    assert stats["median_default_pose_mae_deg_on_quiet_holds"] == case["default_pose_mae_deg"]
    assert summarize_cases([failed])["median_rise_s_on_success"] is None


def test_support_sequence_uses_collision_surfaces_and_source_time():
    from tools.evaluate_zenbot_walk_recovery import limb_clearances, support_sequence

    model = build_scene(robot_name=zenbot.MIXED_TASK)
    stand = np.r_[0., 0., .69, 1., 0., 0., 0., zenbot.DEFAULT_Q]
    floor = stand.copy()
    floor[3:7] = [.5, .5, .5, .5]
    floor[2] -= limb_clearances(model, floor[None])[0, 4:].max() - .01
    ground_then_stand = np.array([floor] * 30 + [stand] * 12)
    ref = {"fps": 50., "root_pos": ground_then_stand[:, :3], "root_quat": ground_then_stand[:, [4, 5, 6, 3]],
           "dof_pos": ground_then_stand[:, 7:]}

    data = mujoco.MjData(model)
    data.qpos[:] = floor
    mujoco.mj_forward(model, data)
    distances = [mujoco.mj_geomDistance(model, data, model.geom("floor").id, model.geom("robot_" + name + "_collision").id,
                                        10., None) for name in zenbot.KEY_BODIES]
    np.testing.assert_allclose(limb_clearances(model, floor[None])[0], distances, atol=1e-7)
    same = support_sequence(model, ground_then_stand, ref)
    assert same["policy_duration_s"]["both_hands"] == pytest.approx(.6)
    assert same["source_time_iou"]["both_hands"] == pytest.approx(1.)
    assert same["knee_flexion_mae_deg_during_source_hand_support"] == pytest.approx(0.)
    faster = support_sequence(model, np.vstack([ground_then_stand[::2], [stand] * 12]), ref)
    assert faster["source_time_iou"]["both_hands"] == pytest.approx(.5)


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
    assert not sim.recovery and sim.assist_newtons == 0 and sim.obs_dim == 390
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
    assert sim.reference is None and not {"clip", "frame"} & sim.floor.keys()
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


def test_mixed_legged_support_samples_openings_and_switches_task_rewards(tmp_path):
    from instantamp.sim import unflatten

    joints = np.tile(zenbot.DEFAULT_Q, (40, 1))
    for i, start_height in enumerate((.20, .34)):
        pos = np.zeros((40, 3))
        pos[:, 2] = np.linspace(start_height, .69, 40)
        quat = np.tile([1., 0., 0., 0.], (40, 1))
        convert(tmp_path / f"clip_{i}.npz", pos, quat, joints, np.arange(23), 50)
    sim = Sim(12, robot=zenbot.MIXED_TASK, walk_dir=tmp_path, floor_dir=tmp_path, physics=False,
              reset_floor=1., recovery_amp_seconds=4., legged_support=True)
    key = jax.random.PRNGKey(17)
    pose, vel, up = sim._fresh(key, sim.world(key))
    np.testing.assert_allclose(np.unique(pose[:, 2]), [.22, .36], atol=1e-6)
    np.testing.assert_allclose(pose[:, 3:], np.broadcast_to(np.r_[1., 0., 0., 0., zenbot.DEFAULT_Q], (12, 27)))
    assert not up.any() and len(sim.floor["q"]) == 2 and sim.reference is None
    assert not {"clip", "frame"} & sim.floor.keys()

    state = unflatten(pose[:3].at[:, 2].set(jnp.array([.3, .68, .68])), vel[:3] * 0.)
    view = sim.view_state(state)
    active = jnp.array([True, True, False])
    a = jnp.zeros_like(state[4])
    torque = a + 1.
    terms = sim.reward_terms(state, state, view.kin, a, a, jnp.zeros((3, 3)), state[0][:, 2] < .58,
                             jnp.array([1., 0., 0.]), sim.soft_contact(view.soles), jnp.ones(3), sim.q_soft,
                             torque=torque, recovery_active=active)
    native = zenbot.reward_terms(state, state, a, a, torque, sim.q_soft, env.DT, com_offset=sim.root_com)
    for name, value in native.items():
        np.testing.assert_allclose(terms[name], jnp.where(active, value, 0.))
    for name in set(env.TERMS) - {"termination"}:
        np.testing.assert_array_equal(terms[name][:2], 0.)
    assert terms["target_base_height"][0] == 0 and terms["target_base_height"][1] == pytest.approx(1.)
    assert terms["track_lin"][2] > 0 and sim.weighted(terms)["termination"][0] < 0


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
    assert sim.reference is None and not up.any() and not {"clip", "frame", "reference_start"} & sim.floor.keys()
    np.testing.assert_allclose(pose[:, 10], heights - zenbot.RESET_LIFT, atol=1e-6)
    with np.load(tmp_path / "clip.npz") as source:
        selected = np.abs(heights[:, None] - zenbot.RESET_LIFT - source["root_pos"][:, 2]).argmin(1)
        np.testing.assert_allclose(velocity[:, 6:], source["dof_vel"][selected])
        np.testing.assert_allclose(velocity[:, :3], source["root_lin_vel"][selected])


def test_support_amp_gradient_remains_at_strongly_rejected_windows():
    from instantamp import support_discriminator as disc

    key = jax.random.PRNGKey(8)
    st = disc.init(key, 10, 3)
    st["net"][-1]["b"] = jnp.array([-12.])
    robot = jnp.zeros((4, 10, 3))
    demo = robot.at[..., 1].set(jnp.linspace(0., 1., 10))
    grad = jax.grad(lambda windows: disc.reward(st, windows, 5.).sum())(robot)
    assert jnp.isfinite(grad).all() and jnp.linalg.norm(grad) > .01
    frozen, _ = disc.train_step(st, demo, robot, jnp.zeros(4), key, .001)
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(st), jax.tree.leaves(frozen)))


def test_frozen_legged_prior_preprocessing_and_reward_tail(tmp_path):
    from instantamp import support_discriminator as disc

    path = tmp_path / "prior.npz"
    fields = {"norm/mean": np.ones(73, np.float32), "norm/var": np.full(73, .0001, np.float32),
              "norm/count": np.float32(12.)}
    for i, (a, b) in enumerate(((730, 1024), (1024, 512), (512, 1))):
        w = np.zeros((a, b), np.float32)
        w[0, 0] = 1.
        fields[f"net/{i}/w"], fields[f"net/{i}/b"] = w, np.zeros(b, np.float32)
    np.savez(path, **fields)
    head = disc.load_legged_prior(path)
    x = jnp.ones((3, 10, 73)).at[:, 0, 0].set(jnp.array([1.02, 2., 0.]))
    np.testing.assert_allclose(disc.legged_logit(head, x), [1., 10., -.001], atol=1e-5)
    raw = jnp.array([-8., -1.5, -1., 0., 1., 4.])
    np.testing.assert_allclose(disc.legged_generator_reward(raw[2:], 1.), [0., .75, 1., 0.])
    grad = jax.grad(lambda d: disc.legged_generator_reward(d, 1.).sum())(raw)
    assert (grad[:2] > .1).all() and grad[2] == pytest.approx(1.)
    with pytest.raises(ValueError, match="ten full-joint frames"):
        parse(["--robot", zenbot.MIXED_TASK, "--legged_prior_file", str(path)])
    assert parse(["--robot", zenbot.MIXED_TASK, "--recovery_amp_loss", "legged_support",
                  "--legged_prior_file", str(path)]).legged_prior_file == str(path)
    np.savez(path, **{**fields, "norm/var": -np.ones(73, np.float32)})
    with pytest.raises(ValueError, match="normalization"):
        disc.load_legged_prior(path)


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


def test_gait_handoff_uses_measured_tilt_then_waits_for_stable_posture():
    from instantamp.inference import ZenbotWalkRecovery, policy_memory, policy_step

    layer = lambda bias: [{"w": jnp.zeros((390, 23)), "b": jnp.full(23, bias)}]
    policy = ZenbotWalkRecovery(layer(-.5), layer(1.), (3.24, 20.))
    norms = (nets.rn_init(390), nets.rn_init(390))
    obs = jnp.zeros((1, 390)).at[:, -78 + 5].set(-1.)
    memory = policy_memory(policy, obs)
    step = jax.jit(lambda o, m: policy_step(policy, norms, o, m))
    action, memory = step(obs, memory)
    np.testing.assert_allclose(action, 1.)
    fallen = obs.at[:, -78 + 5].set(0.)
    action, memory = step(fallen, memory)
    np.testing.assert_allclose(action, -.5)
    for _ in range(17):
        action, memory = step(obs, memory)
    np.testing.assert_allclose(action, -.5)
    for _ in range(85):
        action, memory = step(obs, memory)
    np.testing.assert_allclose(action, 1.)


def test_packed_gait_preserves_ground_actor_and_settles_before_switch(tmp_path):
    from instantamp.inference import load, policy_memory, policy_step
    from instantamp.zenbot_ground_policy import ZenbotGroundPolicy

    archive = {"ode/dmax": 3., "ode/amax": 20., "ground_max_share": 1.,
               "gait/handoff_frames": 3, "gait/fall_upright": .8}
    for prefix, bias in (("", -2.), ("ground_base/", -.5), ("gait/", 1.)):
        archive.update({prefix + "avg/0/w": np.zeros((390, 23)), prefix + "avg/0/b": np.full(23, bias),
                        prefix + "obs_norm/mean": np.zeros(390), prefix + "obs_norm/var": np.ones(390)})
    archive["gait/pi/0/w"] = archive["gait/avg/0/w"]
    file = tmp_path / "compound.npz"
    np.savez(file, **archive)
    policy, norm, ode = load(file)
    assert ode is None and isinstance(policy.rise, ZenbotGroundPolicy)
    obs = jnp.zeros((1, 5, 78)).at[:, :, 5].set(-1.).reshape(1, 390)
    memory = policy_memory(policy, obs)
    for _ in range(2):
        action, memory = policy_step(policy, norm, obs, memory)
        np.testing.assert_allclose(action, -.5)
    for _ in range(6):
        action, memory = policy_step(policy, norm, obs, memory)
    np.testing.assert_allclose(action, 1.)
    down = obs.reshape(1, 5, 78).at[:, :, 5].set(1.).reshape(1, 390)
    action, _ = policy_step(policy, norm, down, memory)
    np.testing.assert_allclose(action, -2.)


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
