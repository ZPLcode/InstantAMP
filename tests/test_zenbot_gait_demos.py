import hashlib
import json
import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from instantamp import zenbot
from instantamp.zenbot_gait_demos import (
    DIRECTION_CLIPS,
    CommandGaitDemos,
    GlobalGaitDemos,
    StraightGaitDemos,
    command_reset_bank,
    directional_window_starts,
    motion_train_reference_tier,
    reference_command_index,
    source_command_speeds,
    straight_window_starts,
)
from tools.prepare_zenbot import prepare_locomotion


def clip(speed, heading=0., offset=0):
    speed = np.asarray(speed)
    n = len(speed)
    q = np.tile([0., 0., np.sin(heading / 2), np.cos(heading / 2)], (n, 1))
    velocity = np.column_stack((speed * np.cos(heading), speed * np.sin(heading), np.zeros(n)))
    return dict(fps=50., root_pos=np.c_[offset + np.arange(n), np.zeros((n, 2))], root_quat=q,
                root_lin_vel=velocity, root_ang_vel=np.zeros((n, 3)), dof_pos=np.zeros((n, 23)),
                dof_vel=np.zeros((n, 23)))


def test_window_selection_uses_local_forward_speed_and_rejects_stops_turns_and_other_gait():
    walk = clip(np.r_[np.ones(28), 0., np.ones(28)], heading=1.57)
    np.testing.assert_array_equal(straight_window_starts(walk, 28, "walk"), [0, 29])
    assert not straight_window_starts(walk, 28, "run").size
    run = clip(np.full(28, 2.75))
    np.testing.assert_array_equal(straight_window_starts(run, 28, "run"), [0])
    assert not straight_window_starts(run, 28, "walk").size
    assert not straight_window_starts(clip(np.full(28, -1.)), 28, "walk").size
    turn = clip(np.ones(28))
    turn["root_quat"][-1] = [0., 0., np.sin(.3), np.cos(.3)]
    assert not straight_window_starts(turn, 28, "walk").size


def test_sampler_draws_weighted_clips_then_contiguous_moving_windows(tmp_path):
    weights = {"Male1_B1_StandToWalk": 1., "Male2_B1_StandToWalk": 3.}
    for n, (name, _) in enumerate(weights.items()):
        speeds = np.r_[np.ones(31+n*18), np.zeros(30), np.ones(32+n*50)]
        np.savez(tmp_path / f"{name}.npz", **clip(speeds, offset=n*1000))
    (tmp_path / "sampling.json").write_text(json.dumps({"weights": weights}))
    bank = StraightGaitDemos(tmp_path, SimpleNamespace(J=23, amp_frame=lambda s: s[0]), 28, "walk")
    draw = np.asarray(jax.jit(lambda k: bank.sample(k, 2048))(jax.random.PRNGKey(11)))[:, :, 0]
    assert np.all(np.diff(draw, axis=1) == 1.)
    second = draw[:, 0] >= 1000
    assert abs(second.mean() - .75) < .04
    for n in range(2):
        t = draw[second == bool(n), 0] - n*1000
        assert np.all((t <= 31+n*18-28) | (t >= 61+n*18))
    with pytest.raises(ValueError, match="No valid straight run"):
        StraightGaitDemos(tmp_path, SimpleNamespace(J=23), 28, "run")


@pytest.mark.parametrize("frames", [10, 3])
def test_global_gait_uses_the_same_consecutive_frames_for_experts_and_rollouts(tmp_path, frames):
    from instantamp import amp, zenbot_source_gait
    from instantamp.train import InstantAMP, ZenbotMotionTrainWMConfig

    options = {} if frames == 10 else {"wm_gait_frames": frames}
    cfg = ZenbotMotionTrainWMConfig(wm_floor_amp=1, wm_global_gait=1, wm_fixed_floor_envs=1,
                                    wm_range_commands=1, wm_gait_rewards=1, gait_ls_scale=1.5, **options)
    learner = InstantAMP.__new__(InstantAMP)
    learner.cfg = cfg
    learner.sim = SimpleNamespace(robot=zenbot, J=23, recovery=False, amp_frame=None, amp_dim=74,
                                  key_positions=lambda p, r, q: jnp.broadcast_to(p[:, None], (len(p), 6, 3)))
    assert learner.gait_span == frames and learner.amp_span == 28
    weights = {"short": 1., "long": 1.}
    for name, length, offset in (("short", frames, 0.), ("long", frames + 20, 100.)):
        motion = clip(np.ones(length))
        motion["dof_pos"][:] = offset + np.arange(length)[:, None]
        motion["dof_vel"][:] = 50.
        np.savez(tmp_path / (name + ".npz"), **motion)
    (tmp_path / "sampling.json").write_text(json.dumps({"weights": weights}))
    bank = GlobalGaitDemos(tmp_path, learner.sim, learner.gait_span)
    sample = np.asarray(jax.jit(lambda k: bank.sample(k, 128))(jax.random.key(11)))
    assert sample.shape == (128, frames, 67)
    q = sample[..., 3]
    short = q[:, 0] < 100.
    assert short.any() and (~short).any()
    np.testing.assert_array_equal(q[short, 0], 0.)
    assert np.any(q[~short, 0] % 1. > .01)
    np.testing.assert_allclose(np.diff(q, axis=1), 1., atol=2e-5)
    np.testing.assert_allclose(sample[..., 26:49], 50., atol=1e-5)

    disc = amp.init(jax.random.key(1), frames, hidden=(8,), frame_dim=zenbot_source_gait.amp_frame_width(learner.sim.J))
    assert disc["net"][0]["w"].shape[0] == 67 * frames
    history = jax.random.normal(jax.random.key(2), (4, learner.amp_span, 74 + 67))
    style, floor = learner.rollout_gait_style(disc, history, None)
    np.testing.assert_allclose(style, amp.recovery_style(disc, history[:, -frames:, 74:], 1.5, motion_train=True))
    np.testing.assert_array_equal(floor, history[..., :74])


def test_directional_windows_use_the_pelvis_heading_and_signed_displacement():
    data = clip(np.full(28, -.5), heading=1.57)
    np.testing.assert_array_equal(directional_window_starts(data, 28, "male_walk__Walk_B4_-_Stand_to_Walk_Back_stageii"), [0])
    data["root_lin_vel"] = np.tile([-.4, 0., 0.], (28, 1))
    np.testing.assert_array_equal(directional_window_starts(data, 28, "droidupe1_side_left"), [0])
    assert not directional_window_starts(data, 28, "droidupe1_side_right").size
    yaw = 3.1 + np.arange(28) * .6 / 50.
    data["root_quat"] = np.column_stack((np.zeros((28, 2)), np.sin(yaw / 2), np.cos(yaw / 2)))
    np.testing.assert_array_equal(directional_window_starts(data, 28, "droidupe1_turn_left"), [0])
    assert not directional_window_starts(data, 28, "droidupe1_turn_right").size


def test_all_eight_amp_commands_draw_separate_contiguous_reference_windows(tmp_path):
    from instantamp import env
    from instantamp.train import InstantAMP

    names = ["stand_4s_120f", "Male1_B1_StandToWalk", "run__C3_-_run_stageii", *DIRECTION_CLIPS]
    for index, (name, cmd) in enumerate(zip(names, zenbot.DIRECTIONAL_COMMANDS)):
        motion = clip(np.full(64, cmd[0]), offset=index * 1000)
        motion["root_lin_vel"][:, 1] = cmd[1]
        motion["dof_pos"][:, 0] = np.sin(np.arange(64)/8.)
        motion["dof_vel"][:, 0] = (50./8.) * np.cos(np.arange(64)/8.)
        yaw = cmd[2] * np.arange(64) * env.DT
        motion["root_quat"][:, 2:] = np.column_stack((np.sin(yaw/2), np.cos(yaw/2)))
        np.savez(tmp_path / f"{name}.npz", **motion)
    sim = SimpleNamespace(J=23, robot=zenbot, recovery=False, mixed=True,
                          amp_frame=lambda state, _: jnp.column_stack((state[0], state[4][:, 0], state[5][:, 0])))
    demos = CommandGaitDemos(tmp_path, sim, 28)
    commands = np.tile(zenbot.DIRECTIONAL_COMMANDS, (5, 1))
    sample = jax.jit(demos.sample_commands)(jax.random.key(44), commands)
    np.testing.assert_array_equal(np.asarray(sample[..., 0]) // 1000, np.broadcast_to(np.tile(np.arange(8), 5)[:, None], (40, 28)))
    np.testing.assert_array_equal(np.diff(sample[..., 0], axis=1), 1.)
    assert (np.ptp(np.asarray(sample[:, 0, 0]).reshape(5, 8), axis=0) > 0).all()
    wrong = np.asarray(jax.jit(lambda k: demos.sample_commands(k, commands, mismatch=True))(jax.random.key(61)))
    assert (wrong[:, 0, 0] // 1000 != np.tile(np.arange(8), 5)).all()
    np.testing.assert_array_equal(np.diff(wrong[..., 0], axis=1), 1.)
    short = np.asarray(demos.sample_commands(jax.random.key(44), commands, reduced=True))
    np.testing.assert_array_equal(short[..., :3], sample[..., :3])
    np.testing.assert_allclose(np.ptp(short[..., 3], axis=1), .25 * np.ptp(sample[..., 3], axis=1), atol=1e-6)
    np.testing.assert_allclose(short[..., 4], .25 * sample[..., 4], atol=1e-6)
    learner = InstantAMP.__new__(InstantAMP)
    learner.cfg = SimpleNamespace(command_gait_amp=True, gait_amp_stride=3)
    learner.sim = SimpleNamespace(amp_joints=(0,))
    features = learner.gait_features(sample, commands)
    np.testing.assert_allclose(features[..., -3:], np.broadcast_to(commands[:, None], (40, 10, 3)))
    np.testing.assert_allclose(features[..., 1], 1 / env.DT)


def test_slow_turn_reference_rejects_a_fast_pivot_and_a_travelling_curve():
    name = "droidupe1_turn_right"
    for speed, forward, expected in ((-.6, 0., 1), (-1.4, 0., 0), (-.6, .5, 0)):
        motion = clip(np.full(28, forward))
        yaw = -3.1 + speed * np.arange(28) / 50.
        motion["root_quat"][:, 2:] = np.column_stack((np.sin(yaw/2), np.cos(yaw/2)))
        motion["root_lin_vel"][:, :2] = forward * np.column_stack((np.cos(yaw), np.sin(yaw)))
        assert len(directional_window_starts(motion, 28, name)) == 1
        assert len(directional_window_starts(motion, 28, name, match_speed=True)) == expected


def test_recorded_command_speeds_keep_slow_steps_and_fast_pivots_with_their_own_demos(tmp_path):
    from instantamp.sim import Sim

    names = ("stand_4s_120f", "Male1_B1_StandToWalk", "run__C3_-_run_stageii", *DIRECTION_CLIPS)
    expected = zenbot.DIRECTIONAL_COMMANDS.copy()
    expected[3:, :] = [[-.66, 0., 0.], [0., .17, 0.], [0., -.17, 0.], [0., 0., 1.2], [0., 0., -1.3]]
    for i, name in enumerate(names):
        speed = np.repeat([zenbot.DIRECTIONAL_COMMANDS[i], expected[i]], [100, 320], axis=0)
        yaw = np.cumsum(speed[:, 2] / 50.)
        d = clip(np.zeros(len(yaw)), offset=i * 1000)
        d["root_quat"][:, 2:] = np.column_stack((np.sin(yaw / 2), np.cos(yaw / 2)))
        d["root_lin_vel"][:, 0] = np.cos(yaw) * speed[:, 0] - np.sin(yaw) * speed[:, 1]
        d["root_lin_vel"][:, 1] = np.sin(yaw) * speed[:, 0] + np.cos(yaw) * speed[:, 1]
        np.savez(tmp_path / (name + ".npz"), **d)
    labels = source_command_speeds(tmp_path)
    np.testing.assert_allclose(labels[:8], zenbot.DIRECTIONAL_COMMANDS)
    np.testing.assert_allclose(labels[8:], expected, atol=1e-5)
    sim = SimpleNamespace(J=23, robot=zenbot, recovery=False, mixed=True, wm_command_speeds=labels,
                          amp_frame=lambda state, _: state[0])
    demos = CommandGaitDemos(tmp_path, sim, 28)
    same = np.asarray(demos.sample_commands(jax.random.key(5), labels))
    wrong = np.asarray(demos.sample_commands(jax.random.key(7), labels, mismatch=True))
    np.testing.assert_array_equal((same[:, 0, 0] // 1000).astype(int), np.tile(np.arange(8), 2))
    assert np.all((wrong[:, 0, 0] // 1000).astype(int) != np.tile(np.arange(8), 2))
    frames = []
    for path in sorted(tmp_path.glob("*.npz")):
        with np.load(path) as motion:
            frames.extend(motion["root_pos"][:, 0])
    reset = command_reset_bank(tmp_path, {"frame": np.array(frames)}, np.ones(len(frames), bool), True,
                               command_speeds=labels)
    for group, after_change in ((4, False), (12, True)):
        start, size = map(int, (reset["offsets"][group], reset["counts"][group]))
        left_frames = reset["frame"][start:start + size] - 4000
        assert (0 <= left_frames).all() and (left_frames < 420).all()
        assert bool(np.median(left_frames) > 100) == after_change
    sim = Sim.__new__(Sim)
    sim.gait_source_commands = sim.wm_floor_locomotion = True
    sim.wm_command_speeds = labels
    n = 8192
    key = jax.random.key(92)
    draw = sim.sample_wm_command(key, n)
    frequency = np.mean(np.asarray(draw.is_standing))
    assert .11 < frequency < .14
    assert np.any(np.isclose(np.asarray(draw.vel)[:, 1], .17, atol=1e-4))
    for group in range(8):
        requested = np.asarray(draw.vel)
        belongs = np.all(requested == labels[group], axis=1) | np.all(requested == labels[group + 8], axis=1)
        assert .10 < np.mean(belongs) < .15


def test_continuous_commands_match_motion_train_and_reach_the_actor():
    from instantamp import zenbot_source_gait
    from instantamp.sim import Sim

    sim = Sim.__new__(Sim)
    sim.wm_range_commands = sim.wm_floor_locomotion = True
    key, n = jax.random.key(861), 4096
    draw = jax.jit(lambda k: sim.sample_wm_command(k, n))(key)
    source = zenbot_source_gait.sample_command(key, n, stage=2)
    for name in ('vel', 'heading', 'is_heading', 'is_standing'):
        np.testing.assert_allclose(getattr(draw, name), getattr(source, name), atol=1e-6)
    assert np.all((draw.time_left >= 2.) & (draw.time_left < 4.))
    moving, stand = np.asarray(draw.is_heading), np.asarray(draw.is_standing)
    turn = ~moving & ~stand
    np.testing.assert_allclose([stand.mean(), turn.mean(), moving.mean()], [.05, .20, .75], atol=.02)
    rot = jnp.broadcast_to(jnp.eye(3), (n, 3, 3))
    actual = np.asarray(sim.wm_velocity_command(draw, rot, jnp.zeros(n, bool), jnp.ones(n)))
    assert np.unique(actual[moving, 0]).size > 2500
    assert actual[moving, 0].min() < -.99 and actual[moving, 0].max() > 2.99
    assert np.all(np.abs(actual[moving, 1]) <= .5)
    np.testing.assert_allclose(actual[moving, 2], np.clip(.5 * np.asarray(draw.heading)[moving], -1.5, 1.5), atol=1e-6)
    np.testing.assert_array_equal(actual[stand], 0.)
    np.testing.assert_array_equal(actual[turn, :2], 0.)
    assert np.max(np.abs(actual[turn, 2])) > 2.15


def test_global_gait_pure_sidestep_speed_range():
    from instantamp.sim import Sim

    sim = Sim.__new__(Sim)
    sim.wm_global_gait = sim.wm_range_commands = True
    sim.wm_side_share = .10
    n = 8192
    cmd = sim.sample_wm_command(jax.random.key(92), n)
    rotation = jnp.broadcast_to(jnp.eye(3), (n, 3, 3))
    velocity = np.asarray(sim.wm_velocity_command(cmd, rotation, jnp.ones(n, bool)))
    lateral = velocity[:, 1]
    for direction in (1, -1):
        side = ~np.asarray(cmd.is_heading) & (direction * lateral > 0.)
        np.testing.assert_allclose(side.mean(), .10, atol=.02)
        np.testing.assert_array_equal(velocity[side][:, [0, 2]], 0.)
        speed = direction * lateral[side]
        assert speed.min() < .02 and .98 < speed.max() < 1.
        np.testing.assert_allclose(np.quantile(speed, [.25, .50, .75]), [.25, .50, .75], atol=.05)


def test_continuous_reference_matches_translation_during_heading_corrections():
    commands = jnp.array([[[.82, .12, 1.4], [-.37, .1, -1.3], [.04, -.22, 1.3]],
                          [[0., 0., .04], [0., 0., 0.], [2.45, -.2, .9]]])
    np.testing.assert_array_equal(reference_command_index(commands, zenbot.DIRECTIONAL_COMMANDS),
                                  [[1, 3, 5], [6, 0, 2]])


def test_command_amp_can_reward_improvements_below_the_least_squares_cutoff():
    from instantamp import amp, nets

    def score(bias, soft_floor):
        st = {"net": [{"w": jnp.zeros((1, 1)), "b": bias[None]}], "norm": nets.rn_init(1)}
        return amp.recovery_style(st, jnp.zeros((1, 1, 1)), 6., soft_floor=soft_floor).sum()

    assert score(jnp.array(-1.1), False) == 0.
    assert 0. < score(jnp.array(-1.1), True) < score(jnp.array(-1.), True) < 1.
    assert jax.grad(lambda d: score(d, True))(jnp.array(-1.1)) > 0.
    np.testing.assert_allclose(score(jnp.array(1.), True), 6., rtol=1e-6)


def test_converter_uses_sole8_with_configured_weights_and_manifest_joint_order(tmp_path):
    source = tmp_path / "source"
    cfg = source / "robolab/robolab/tasks/manager_based/amp/zenbot_amp_env_cfg.py"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('MOTION_WEIGHTS = {"test": 2., "inactive": 0.}\n'
                   'cfg.motion_data_dir = os.path.join(ROBOLAB_ROOT_DIR, "data", "motions", "zenbot_amp_frax")\n')
    bank = source / "robolab/data/motions/zenbot_amp_frax_walk_sole8"
    bank.mkdir(parents=True)
    joints = np.arange(23)[None].repeat(30, axis=0)
    motion = dict(root_pos=np.zeros((30, 3)), root_rot=np.tile([1., 0., 0., 0.], (30, 1)),
                  dof_pos=joints, fps=50.)
    pkl = bank / "test.pkl"
    pkl.write_bytes(pickle.dumps(motion))
    manifest = dict(quaternion_order="wxyz", joint_names=[f"{n}_joint" for n in reversed(zenbot.DOF)],
                    clips=[dict(name="test", output_sha256=hashlib.sha256(pkl.read_bytes()).hexdigest())])
    (bank / "manifest.json").write_text(json.dumps(manifest))
    dest = tmp_path / "output"
    prepare_locomotion(source, dest)
    with np.load(dest / "walk_run/test.npz") as converted:
        np.testing.assert_array_equal(converted["dof_pos"][0], joints[0, ::-1])
    sampling = json.loads((dest / "walk_run/sampling.json").read_text())
    assert sampling["weights"] == {"test": 2.}
    assert sampling["source_bank"] == bank.name
    assert sampling["reference_sampling"] == "motion_train_command_speed"
    pkl.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Source hash"):
        prepare_locomotion(source, dest)


def test_original_frax_converter_preserves_joint_trajectory_and_checks_import_manifest(tmp_path):
    cfg = tmp_path / "robolab/robolab/tasks/manager_based/amp/zenbot_amp_env_cfg.py"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('MOTION_WEIGHTS = {"walk": 3.75}\n')
    bank = tmp_path / "robolab/data/motions/zenbot_amp_frax"
    bank.mkdir(parents=True)
    q = np.linspace(-.2, .3, 23 * 50).reshape(50, 23)
    pkl = bank / "walk.pkl"
    pkl.write_bytes(pickle.dumps(dict(root_pos=np.zeros((50, 3)), root_rot=np.tile([1., 0., 0., 0.], (50, 1)),
                                      dof_pos=q, fps=50.)))
    manifest = dict(joint_names=[f"{name}_joint" for name in reversed(zenbot.DOF)], quaternion_order="wxyz",
                    files={pkl.name: hashlib.sha256(pkl.read_bytes()).hexdigest()})
    (bank / "import_manifest.json").write_text(json.dumps(manifest))
    prepare_locomotion(tmp_path, tmp_path, bank.name)
    with np.load(tmp_path / "walk_run_frax/walk.npz") as result:
        np.testing.assert_allclose(result["dof_pos"], q[:, ::-1], atol=1.e-7)
    info = json.loads((tmp_path / "walk_run_frax/sampling.json").read_text())
    assert info["reference_sampling"] == "motion_train_command_speed"
    assert info["weights"] == {"walk": 3.75}
    pkl.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source hash"):
        prepare_locomotion(tmp_path, tmp_path, bank.name)


def test_slow_reference_rates_follow_command_and_preserve_clip_end_velocities(tmp_path):
    commands = jnp.array([[.1, 0, 0], [.3, 0, 0], [.375, 0, 0], [.375, 0, 0], [.45, 0, 0],
                          [0, 0, -.6], [0, 0, .75], [0, 0, .9], [-.2, 0, 0], [0, .2, 0], [2., 0, 0]])
    draw = jnp.array([.1, .9, .25, .75, .01, .9, .25, .01, .2, .2, .2])
    pools, rates = motion_train_reference_tier(commands, draw)
    np.testing.assert_array_equal(pools, [0, 8, 8, 1, 1, 7, 6, 6, 3, 4, 2])
    np.testing.assert_allclose(rates, [1., .35, .35, 1., 1., .5, .5, 1., 1., 1., 1.])

    frames = 21
    motion = clip(np.ones(frames))
    ticks = np.arange(frames)
    motion["dof_pos"][:] = ticks[:, None]
    yaw = ticks * .01
    motion["root_quat"][:, 2:] = np.stack((np.sin(yaw/2), np.cos(yaw/2)), axis=-1)
    np.savez(tmp_path / "first.npz", **motion)
    motion["dof_pos"] += 1000.
    np.savez(tmp_path / "second.npz", **motion)
    (tmp_path / "sampling.json").write_text(json.dumps({"weights": {"first": 1, "second": 1}}))
    sim = SimpleNamespace(J=23, key_positions=lambda p, r, q: jnp.broadcast_to(p[:, None], (len(p), 6, 3)))
    demos = GlobalGaitDemos(tmp_path, sim, 10)
    rate = jnp.array([.35, .5])
    start = (frames - 1) - 9 * rate
    windows = jax.jit(demos.at_time)(jnp.array([0, 0]), start, rate)
    np.testing.assert_allclose(windows[:, :, 3], start[:, None] + jnp.arange(10)*rate[:, None], atol=2.e-5)
    np.testing.assert_allclose(windows[:, :, 26], np.broadcast_to((rate*50)[:, None], (2, 10)), atol=2.e-4)
    np.testing.assert_allclose(windows[:, :, 2], np.broadcast_to((rate*.5)[:, None], (2, 10)), atol=2.e-5)


def test_step_width_moves_narrow_feet_across_the_heading_and_preserves_sole_orientation():
    import mujoco

    from tools.zenbot_step_width import MAX_FOOT_SHIFT, StepWidth

    widener = StepWidth()
    # Face world +Y, with both legs adducted into a narrow stance.
    pos = np.tile([0., 0., .72], (8, 1))
    pos[:, 1] = np.arange(8) / 50.
    quat = np.tile([2 ** -.5, 0., 0., 2 ** -.5], (8, 1))
    joints = np.tile(zenbot.DEFAULT_Q, (8, 1))
    joints[:, [1, 7]] = [-.14, .14]
    widened, shift = widener.adjust(pos, quat, joints, 50.)
    assert shift == MAX_FOOT_SHIFT
    np.testing.assert_array_equal(widened[:, 12:], joints[:, 12:])

    def soles(angles, i):
        widener.data.qpos[:] = np.r_[pos[i], quat[i], angles[i]]
        mujoco.mj_kinematics(widener.model, widener.data)
        return widener.data.geom_xpos[widener.boxes].copy(), widener.data.geom_xmat[widener.boxes].copy()

    for i in (0, len(joints) - 1):
        before, old_rotation = soles(joints, i)
        after, rotation = soles(widened, i)
        np.testing.assert_allclose(after[:, 0] - before[:, 0], [-shift, shift], atol=.001)
        np.testing.assert_allclose(after[:, 1:], before[:, 1:], atol=.004)
        np.testing.assert_allclose(rotation, old_rotation, atol=.003)
    assert widener.adjust(pos, quat, np.tile(zenbot.DEFAULT_Q, (8, 1)), 50.)[1] == 0.
    assert widener.adjust(np.zeros_like(pos), quat, joints, 50.)[1] == 0.


def test_footfall_correction_levels_loaded_feet_and_repairs_height_spike():
    from tools.zenbot_step_width import StepWidth

    rig = StepWidth()
    pos = np.tile([0., 0., .70], (40, 1))
    pos[:, 0] = np.arange(40) * .0008
    pos[19, 2] += .012
    quat = np.tile([1., 0., 0., 0.], (40, 1))
    joints = np.tile(zenbot.DEFAULT_Q, (40, 1))
    joints[:, [4, 10]] += .20
    root, angles = rig.correct_footfalls(pos, quat, joints, 50.)
    centers, rotations = rig.sole_poses(np.column_stack((root, quat, angles)))
    tilt = np.arccos(np.clip(rotations[:, :, 2, 2], -1., 1.))
    lowest = centers[:, :, 2] - (abs(rotations[:, :, 2, :]) * rig.model.geom_size[rig.boxes]).sum(axis=2)
    assert np.max(tilt) < np.deg2rad(1.)
    assert lowest.min() >= 0. and lowest.max() < .005
    assert np.max(abs(np.diff(root[:, 2], n=2))) < .0016
    np.testing.assert_array_equal(root[:, :2], pos[:, :2])


def test_level_sole_only_after_heel_roll_and_release_at_toe_off():
    from tools.zenbot_step_width import settled_support

    # Other foot is already planted. Right foot swings close to the ground,
    # touches down on its heel, settles, then leaves on its toe.
    height = np.zeros((30, 2))
    height[:5, 1] = .012
    speed = np.zeros_like(height)
    speed[:5, 1] = 1.4
    gap = np.zeros_like(height)
    gap[5:10, 1] = .085
    gap[23:, 1] = .085
    support = settled_support(height, gap, speed, 50.)
    np.testing.assert_allclose(support[:, 0], 1.)
    np.testing.assert_array_equal(support[:13, 1], 0.)
    assert 0. < support[13, 1] < 1.
    np.testing.assert_allclose(support[16:23, 1], 1.)
    np.testing.assert_array_equal(support[23:, 1], 0.)


def test_frame_repair_preserves_a_steady_ramp_and_bounds_an_isolated_jump():
    from tools.zenbot_step_width import limit_step_change

    velocity = np.full((39, 1), .08)
    acceleration = np.full((38, 1), .02)
    ramp = (.05 * np.arange(40))[:, None]
    np.testing.assert_array_equal(limit_step_change(ramp, acceleration, max_step=velocity), ramp)
    corrupted = ramp.copy()
    corrupted[20] += .2
    fixed = limit_step_change(corrupted, acceleration, max_step=velocity)
    assert np.max(abs(np.diff(fixed, axis=0))) <= .080001
    assert np.max(abs(np.diff(fixed, n=2, axis=0))) <= .020001
    np.testing.assert_array_equal(fixed[[0, -1]], ramp[[0, -1]])
