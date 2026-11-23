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
    CommandGaitDemos, DIRECTION_CLIPS, StraightGaitDemos, command_reset_bank,
    directional_window_starts, source_command_speeds, straight_window_starts,
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
    learner.J = 1
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


def test_recorded_low_speed_comparison_requires_measured_lateral_progress():
    from tools.evaluate_zenbot_directions import direction_metrics

    t = .02 * np.arange(201)
    poses = np.zeros((len(t), 30))
    poses[:, 2:4] = [.69, 1.]
    loads = np.full((len(t) - 1, 2), .5)
    command = np.array([0., .17, 0.])
    poses[:, 1] = .04 * t
    assert direction_metrics(poses, loads, command)["passed"]
    assert not direction_metrics(poses, loads, command, relative_active_speed=True)["passed"]
    poses[:, 1] = .17 * t
    assert direction_metrics(poses, loads, command, relative_active_speed=True)["passed"]


def test_command_amp_can_reward_improvements_below_the_least_squares_cutoff():
    from instantamp import amp, nets

    def score(bias, soft_floor):
        st = {"net": [{"w": jnp.zeros((1, 1)), "b": bias[None]}], "norm": nets.rn_init(1)}
        return amp.recovery_style(st, jnp.zeros((1, 1, 1)), 6., soft_floor=soft_floor).sum()

    assert score(jnp.array(-1.1), False) == 0.
    assert 0. < score(jnp.array(-1.1), True) < score(jnp.array(-1.), True) < 1.
    assert jax.grad(lambda d: score(d, True))(jnp.array(-1.1)) > 0.
    np.testing.assert_allclose(score(jnp.array(1.), True), 6., rtol=1e-6)


def test_converter_uses_weight_configured_bank_and_manifest_joint_order(tmp_path):
    source = tmp_path / "source"
    cfg = source / "robolab/robolab/tasks/manager_based/amp/zenbot_amp_env_cfg.py"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('MOTION_WEIGHTS = {"test": 2., "inactive": 0.}\n'
                   'cfg.motion_data_dir = os.path.join(ROBOLAB_ROOT_DIR, "data", "motions", "zenbot_amp_frax")\n')
    bank = source / "robolab/data/motions/zenbot_amp_frax"
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
    pkl.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Source hash"):
        prepare_locomotion(source, dest)
