"""Recovery and locomotion rewards for the joint Zenbot policy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_zenbot_motion_train_wm_sim import CpuState

from instantamp import amp, buffer as rb, env, zenbot
from instantamp.sim import Sim, unflatten
from instantamp.train import InstantAMP, parse


def test_joint_floor_amp_requires_online_motion_and_physical_rollouts():
    with pytest.raises(SystemExit):
        parse(["--robot", zenbot.MIXED_TASK, "--wm_motion_replay", "1"])
    with pytest.raises(ValueError, match="Floor-opening AMP"):
        parse(["--robot", zenbot.MIXED_TASK, "--ppo_frac", "0"])


@pytest.mark.parametrize("posture", [False, True])
def test_joint_floor_amp_commands_and_rewards_follow_current_pose(zenbot23_motion_banks, posture):
    sim = Sim(256, physics=False, robot=zenbot.MIXED_TASK, zenbot_motion_train_wm=True,
              wm_floor_amp=True, wm_lift_gate=.44, support_rsi=.35, reset_floor=.5, wm_posture_and_tracking=posture,
              randomize_world=False, obs_noise=False, push=False, prob_shove=0.)
    sim.dx = CpuState(jnp.zeros(30), jnp.zeros(29), jnp.zeros(29))
    s = sim.init(jax.random.key(44))
    lin, ang = "track_lin_vel_xy_exp", "track_ang_vel_z_exp"
    expected = {"stand", lin, ang, "action_rate", "lift", "first_attempt_failure"}
    if posture:
        expected.update(("flat_orientation_l2", "slow_speed_base_height_l2"))
    assert set(sim.weights) == expected and (sim.weights[lin], sim.weights[ang]) == (2., 1.5)
    assert .25 < np.mean(s.down) < .5 and np.all(s.was_up[sim.wm_gait_worlds])
    assert set(map(tuple, np.asarray(s.cmd.vel))) == set(map(tuple, zenbot.DIRECTIONAL_COMMANDS))
    assert np.all((s.cmd.time_left >= 2.) & (s.cmd.time_left <= 4.))
    np.testing.assert_array_equal(s.c_obs, s.cmd.vel)
    np.testing.assert_array_equal(s.motion_train_sensors.observation()[:, 24:27], s.cmd.vel)
    # The GPU collector donates both state fields on its first step.
    transition = jax.jit(lambda cmd, obs: (cmd + 1., obs - 1.), donate_argnums=(0, 1))
    advanced, observed = transition(s.cmd.vel, s.c_obs)
    np.testing.assert_allclose(advanced - observed, 2., atol=1e-6)
    assert s.recovery_left is None

    v = jnp.zeros((1, 3))
    q = sim.floor_stand_joints[None]
    a = jnp.zeros_like(q)
    pose = (jnp.array([[0.,0.,.68]]), jnp.eye(3)[None], v, v, q, a)
    def reward(command, speed, recovering, rotation=pose[1], spin=v):
        state = (pose[0], rotation, speed, spin, q, a)
        return sim.reward_terms(pose, state, sim.view_state(state).kin, a, a, command,
                                jnp.array([False]), jnp.zeros(1), jnp.ones((1, 2)), jnp.zeros(1),
                                sim.q_soft, recovery_active=jnp.array([recovering]))
    run = jnp.array([[2.5, 0., 0.]])
    np.testing.assert_allclose(reward(run, v, False)[lin], np.exp(-2.5**2 / .5**2), rtol=1e-6)
    assert reward(v, v, False)[lin].item() == reward(v, v, False)[ang].item() == 1.
    assert reward(run, run, False)["stand"].item() == 0.
    np.testing.assert_array_equal(reward(run, v, True)[lin], reward(run, v, False)[lin])
    np.testing.assert_array_equal(reward(run, v, True)[ang], reward(run, v, False)[ang])
    assert reward(v, v, False)["stand"].item() > .99
    assert jax.grad(lambda vx: reward(run, v.at[0,0].set(vx), False)[lin].sum())(0.) > 0.

    # Nonzero yaw, tilt and angular velocity distinguish pelvis axes, COM and upright scaling.
    rotation = np.array([[0., -.5, np.sqrt(.75)], [1., 0., 0.], [0., np.sqrt(.75), .5]])
    omega = np.array([1., -.6, .9])
    com_speed = np.array([.6, -.3, 2.])
    origin_speed = com_speed - np.cross(omega, sim.root_com)
    cmd = jnp.array([[.1, -.3, -.1]])
    scores = reward(cmd, jnp.asarray(rotation @ origin_speed)[None], False,
                    jnp.asarray(rotation)[None], jnp.asarray(rotation @ omega)[None])
    np.testing.assert_allclose(scores[lin], zenbot.gait_weight(.68, .5)*np.exp(-1.), rtol=1e-6)
    np.testing.assert_allclose(scores[ang], zenbot.gait_weight(.68, .5)*np.exp(-4.), rtol=1e-6)


def test_fixed_joint_collectors_keep_their_tasks_across_resets_and_double_tracking(zenbot23_motion_banks):
    sim = Sim(256, physics=False, robot=zenbot.MIXED_TASK, zenbot_motion_train_wm=True,
              wm_floor_amp=True, wm_range_commands=True, wm_posture_and_tracking=True,
              wm_gait_rewards=True, wm_lift_gate=.44, support_rsi=.35, reset_floor=.4,
              wm_fixed_floor_envs=True, wm_tracking_scale=2., randomize_world=False,
              obs_noise=False, push=False, prob_shove=0.)
    floor = ~np.asarray(sim.wm_gait_worlds)
    assert floor.sum() == round(.4 * sim.N)
    np.testing.assert_array_equal(floor[::2], floor[1::2])
    assert abs(floor[1:51:2].sum() - .4 * 25) <= 1
    assert sim.weights['track_lin_vel_xy_exp'] == 4. and sim.weights['track_ang_vel_z_exp'] == 3.
    assert sim.weights['stand'] == 6. and sim.weights['lift'] == .5
    sim.dx = CpuState(jnp.zeros(30), jnp.zeros(29), jnp.zeros(29))
    state = sim.init(jax.random.key(42))
    np.testing.assert_array_equal(state.down, floor)
    for seed in (7, 23):
        state = sim._reset(state, jnp.ones(sim.N, bool), jax.random.key(seed))
        np.testing.assert_array_equal(state.down, floor)
        assert state.recovery_left is None


def test_joint_task_samples_ground_to_stance_requests(zenbot23_motion_banks):
    sim = Sim(256, physics=False, robot=zenbot.MIXED_TASK, zenbot_motion_train_wm=True,
              wm_floor_amp=True, wm_lift_gate=.44, support_rsi=.35, reset_floor=2/3,
              floor_stop_command_share=.5,
              randomize_world=False, obs_noise=False, push=False, prob_shove=0.)
    sim.dx = CpuState(jnp.zeros(30), jnp.zeros(29), jnp.zeros(29))
    state = sim.init(jax.random.key(156))
    parked = np.asarray(state.cmd.time_left > 4.)
    assert .40 < np.mean(state.down) < .60
    assert .30 < np.mean(parked[np.asarray(state.down)]) < .70
    assert np.all(np.asarray(state.down)[parked])
    np.testing.assert_array_equal(np.asarray(state.motion_train_sensors.observation())[parked,24:27], 0.)
    assert np.any((~parked & np.asarray(state.down)) & np.any(np.asarray(state.cmd.vel) != 0., axis=1))
    assert state.motion_train_sensors.observation().shape == (sim.N, 234)


def test_range_forward_starts_include_standing_with_live_commands(zenbot23_motion_banks, monkeypatch):
    # This reset only needs standing and forward walking clips.
    monkeypatch.setattr("instantamp.zenbot_gait_demos.source_command_speeds", lambda _: zenbot.DIRECTIONAL_COMMANDS[:2])
    sim = Sim(1024, physics=False, robot=zenbot.MIXED_TASK, zenbot_motion_train_wm=True,
              wm_floor_amp=True, wm_range_commands=True, gait_source_commands=True, reset_floor=.5, obs_noise=False,
              randomize_world=False, push=False, prob_shove=0.)
    sim.dx = CpuState(jnp.zeros(30), jnp.zeros(29), jnp.zeros(29))
    s = sim.init(jax.random.key(236))
    forward = np.asarray(s.was_up & (s.cmd.vel[:, 0] > jnp.abs(s.cmd.vel[:, 1])))
    cold = np.all(np.asarray(s.data.qvel) == 0., axis=1)
    assert .18 < cold[forward].mean() < .32
    assert np.any(forward & ~cold)
    stand_rows = sim.walk["command_rows"]
    rows = stand_rows["row"][:int(stand_rows["counts"][0])]
    standing = set(map(tuple, np.asarray(sim.walk["q"][rows])))
    assert all(tuple(pose) in standing for pose in np.asarray(s.data.qpos)[forward & cold, 7:])
    np.testing.assert_array_equal(s.motion_train_sensors.frames[forward & cold, -1, 6:9], s.c_obs[forward & cold])
    np.testing.assert_array_equal(s.c_obs[forward & cold, 0], s.cmd.vel[forward & cold, 0])


def test_walk_to_stop_only_changes_expired_commands_while_moving():
    n = 8192
    sim = Sim.__new__(Sim)
    sim.wm_range_commands = sim.wm_floor_locomotion = True
    sim.root_com = jnp.zeros(3)
    row = jnp.arange(n) % 4
    # A 90-degree yaw makes positive world Y the robot's forward axis.
    q = jnp.tile(jnp.r_[jnp.array([0., 0., .69, 2**-.5, 0., 0., 2**-.5]), zenbot.DEFAULT_Q], (n, 1))
    state = unflatten(q, jnp.zeros((n, 29)).at[:, 1].set(row != 1))
    forward = jnp.tile(jnp.array([1., 0., 0.]), (n, 1))
    cmd = env.Cmd(forward, jnp.zeros(n), jnp.zeros(n, bool), jnp.zeros(n, bool),
                  jnp.where(row == 3, 1., env.DT))
    sample = jax.jit(sim.tick_wm_command)(cmd, jax.random.key(117), state, jnp.ones(n, bool), row == 2)
    stopped = np.asarray(sample.is_standing)
    np.testing.assert_allclose([stopped[row == i].mean() for i in range(4)], [.25, .05, .05, 0.], atol=.025)
    observed = sim.wm_velocity_command(sample, state[1], jnp.ones(n, bool))
    np.testing.assert_array_equal(observed[stopped], 0.)
    np.testing.assert_array_equal(observed[row == 3], forward[row == 3])
    assert np.all((sample.time_left[stopped] >= 2.) & (sample.time_left[stopped] < 4.))


def test_joint_floor_practice_includes_moving_low_support_and_cold_openings(zenbot23_motion_banks):
    sim = Sim(512, robot=zenbot.MIXED_TASK, physics=False, zenbot_motion_train_wm=True,
              wm_floor_amp=True, wm_lift_gate=.44, support_rsi=.35, reset_floor=2/3)
    (q, v), (clip, frame, _) = sim._wm_floor_start(jax.random.key(689), sim.floor)
    moving = np.asarray(frame) > 0
    assert .28 < moving.mean() < .42
    assert len(np.unique(clip[~moving])) == 11
    np.testing.assert_array_equal(v[~moving, 2:], 0.)
    z = np.asarray(q[:, 2])
    assert (z[moving] < .22).any() and (z[moving] > .44).any()
    assert (z <= zenbot.UP_HEIGHT).all()

    bank = sim.floor["phases"]
    by_source = {tuple(pair): i for i, pair in enumerate(zip(np.asarray(bank["clip"]), np.asarray(bank["frame"])))}
    rows = np.array([by_source[pair] for pair in zip(np.asarray(clip[moving]), np.asarray(frame[moving]))])
    _, _, velocity, angular, _, joint_speed = unflatten(q[moving], v[moving])
    np.testing.assert_allclose(velocity[:, 2], bank["v"][rows, 2], atol=1e-6)
    assert np.max(np.abs(velocity[:, :2] - bank["v"][rows, :2])) <= .071
    np.testing.assert_allclose(angular, bank["w"][rows], atol=2e-5)
    np.testing.assert_array_equal(joint_speed, bank["qd"][rows])


def test_recovery_sampling_finds_rare_intermediate_poses():
    steps, worlds = 32, 8
    states = jnp.zeros((steps, worlds, 1))
    gate = jnp.ones((steps, worlds)).at[18:21, 5].set(.7)
    buf = dict(s_qpos=states, s1_qpos=states, s_qvel=states, s1_qvel=states, gait_weight=gate,
               done=jnp.zeros_like(gate), ok=jnp.ones_like(gate), size=steps, ptr=0,
               t=jnp.broadcast_to(jnp.arange(steps)[:, None], gate.shape))
    windows = rb.sample_windows(buf, jax.random.key(43), 64, 5, recovery_only=True)
    np.testing.assert_array_equal(windows["gait_weight"][:, -1] < 1., True)
    np.testing.assert_array_equal(windows["mask"][:, -1], 1.)


def test_floor_discriminator_sees_the_same_reset_padding_as_the_policy():
    steps, worlds, frames = 20, 2, 5
    time = jnp.tile(jnp.arange(steps)[:,None], (1,worlds))
    states = time[..., None].astype(jnp.float32)
    # Pose suitability can change inside an episode without restarting its history.
    buf = dict(s_qpos=states, s1_qpos=states+1, s_qvel=states, s1_qvel=states+1, gait_weight=jnp.zeros_like(time),
               done=jnp.zeros_like(time), ok=jnp.ones_like(time), size=steps, ptr=0, t=time)
    windows = rb.sample_windows(buf, jax.random.key(73), 512, frames, recovery_only=True, recovery_steps=200)
    last = windows["t"][:, -1:]
    assert np.any(last == 1) and np.any(last == 10)
    np.testing.assert_array_equal(windows["t"], jnp.maximum(last+jnp.arange(1-frames,1), 0))
    np.testing.assert_array_equal(windows["mask"][:,-1], 1)


def test_quiet_supported_stand_beats_a_jump_and_a_floor_pose():
    pos = jnp.array([[0., 0., .15], [0., 0., .68], [0., 0., .68]])
    rotation = jnp.broadcast_to(jnp.eye(3), (3, 3, 3)).at[0, 2, 2].set(0.)
    vel = jnp.zeros((3, 3)).at[1, 2].set(1.)
    feet = jnp.array([[0., 0.], [0., 0.], [1., 1.]])
    score = zenbot.quiet_stand_score((pos, rotation, vel, jnp.zeros_like(vel), None, None), feet)
    np.testing.assert_allclose(score, [0., 0., 1.], atol=1.e-5)


def test_negative_windows_change_only_tempo_and_preserve_clip_boundaries():
    demos = amp.Demos.__new__(amp.Demos)
    demos.window = 28
    lengths = np.array([79, 55, 42])
    demos.clip_offsets = jnp.asarray(np.r_[0, np.cumsum(lengths[:-1])])
    demos.clip_lengths = jnp.asarray(lengths - demos.window + 1)
    demos.clip_probabilities = None
    frames = []
    for clip, length in enumerate(lengths):
        # The second source is still. The third can hold a regular window but is too short for double speed.
        row = np.ones((length, 11), np.float32)
        row[:, 0] = np.arange(length) * .02 if clip != 1 else 0.
        row[:, 6] = clip
        frames.append(row)
    demos.frames = jnp.asarray(np.concatenate(frames))
    raw, valid = jax.jit(lambda key: demos.sample_wrong_tempo(key, 256, 10, 3, 1))(jax.random.key(19))
    raw, valid = np.asarray(raw), np.asarray(valid)
    source_id = raw[..., 6]
    np.testing.assert_array_equal(source_id, np.broadcast_to(source_id[:, :1], source_id.shape))
    np.testing.assert_array_equal(valid, source_id[:, 0] == 0)
    fast = raw[:, 0, 1] == 2.
    assert (valid & fast).any() and (valid & ~fast).any()
    np.testing.assert_allclose(np.diff(raw[valid & fast, :, 0], axis=1), .12, atol=2e-6)
    np.testing.assert_allclose(np.diff(raw[~fast], axis=1), 0.)
    for column in (1, 2, 3, 4, -3, -2, -1):
        np.testing.assert_array_equal(raw[..., column], np.broadcast_to(2.*fast[:, None], source_id.shape))
    early, eligible = demos.sample_wrong_tempo(jax.random.key(19), 256, 10, 3, 1, freeze_start=True)
    np.testing.assert_array_equal(eligible, valid)
    np.testing.assert_allclose(early[fast], raw[fast])
    # In this clip joint 0 moves at one radian per second; the old middle anchor is 600 ms later.
    frozen = valid & ~fast
    np.testing.assert_allclose(raw[frozen, :, 0] - early[frozen, :, 0], .60, atol=2e-6)


def test_upright_mirror_uses_valid_supported_rows_and_one_sided_gradient():
    from types import SimpleNamespace

    from instantamp.zenbot_motion_train_obs import (
        mirror_action,
        mirror_observation,
    )

    learner = InstantAMP.__new__(InstantAMP)
    learner.cfg = SimpleNamespace(motion_train_wm=True, action_noise=.8, ppo_clip=.2, gait_mirror_weight=.2)
    x = jax.random.normal(jax.random.key(23), (3, 234))
    p = [{"w": .03 * jax.random.normal(jax.random.key(19), (234, 23)), "b": jnp.linspace(0., .2, 23)}]
    previous = jnp.zeros((3, 23))
    norm = {"raw": True}
    mu = learner.act(p, norm, x, previous)
    # Row 1 is invalid and row 2 is still recovering. Both have distinct inputs.
    batch = dict(obs=x, a_ode=previous, a=mu, mu=mu, adv=jnp.zeros(3),
                 mask=jnp.array([1., 0., 1.]), gait_mirror=jnp.array([True, True, False]))

    def expected(params):
        actual = learner.act(params, norm, mirror_observation(x[:1]), previous[:1])
        target = mirror_action(learner.act(params, norm, x[:1], previous[:1]))
        return .2 * jnp.square(actual - jax.lax.stop_gradient(target)).mean()

    value, grad = jax.value_and_grad(lambda params: learner.ppo_loss(params, norm, batch)[0])(p)
    target, expected_grad = jax.value_and_grad(expected)(p)
    assert float(value) > 0
    np.testing.assert_allclose(value, target, rtol=1e-5)
    for measured, wanted in zip(jax.tree.leaves(grad), jax.tree.leaves(expected_grad), strict=True):
        np.testing.assert_allclose(measured, wanted, rtol=2e-5, atol=2e-6)

    empty = {**batch, "gait_mirror": jnp.zeros(3, bool)}
    value, grads = jax.value_and_grad(lambda params: learner.ppo_loss(params, norm, empty)[0])(p)
    assert float(value) == 0 and all(np.all(np.asarray(g) == 0) for g in jax.tree.leaves(grads))


def test_upright_mirror_traces_the_complete_policy_iteration(zenbot23_motion_banks):
    if jax.devices()[0].platform not in ("gpu", "cuda"):
        pytest.skip("The physical MuJoCo Warp collector requires CUDA")
    cfg = parse(["--robot", zenbot.MIXED_TASK, "--motion_train_wm", "1", "--wm_floor_amp", "1",
                 "--gait_ls_scale", "10", "--gait_mirror_weight", ".2", "--wm_lift_gate", ".44",
                 "--num_envs", "32", "--pi_batch", "24", "--wm_batch", "16", "--amp_batch", "32",
                 "--buffer_steps", "64", "--env_steps", "4", "--ppo_len", "4", "--warmup_steps", "48"])
    learner = InstantAMP(cfg)
    carry = learner.init(jax.random.PRNGKey(42))
    state = carry["env"]
    state = state._replace(t=state.episode_limit - 1)
    _, tr = jax.jit(learner.sim.step)(state, jnp.zeros_like(state.a_last))
    events = tr.parts[:, learner.sim.terms.index("first_attempt_failure")]
    assert state.first_attempt.active.any()
    np.testing.assert_array_equal(events, -np.asarray(state.first_attempt.active, float))
    _, stats = jax.eval_shape(learner.pi_iter, carry)
    assert stats["gait_mirror_mse"].shape == ()
    assert stats["imag_gait_mirror_mse"].shape == stats["imag_gait_mirror_fraction"].shape == ()
