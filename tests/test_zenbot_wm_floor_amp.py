"""The floor-only hybrid learner rewards a supported stance and samples independent recovery windows."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from instantamp import amp, env, zenbot
from instantamp.sim import Sim, unflatten
from instantamp.train import InstantAMP, parse
from instantamp import buffer as rb
from instantamp.zenbot_wm_reference import PhysicalRecoveryReference
from test_zenbot_motion_train_wm_sim import CpuState, native_substeps


def test_all_resets_are_perturbed_ground_openings_even_after_episode_expiry(monkeypatch):
    def forbidden(*args):
        pytest.fail("Floor-only AMP must not load recorded motor trajectories")

    monkeypatch.setattr(PhysicalRecoveryReference, "__init__", forbidden)
    sim = Sim(256, robot=zenbot.MIXED_TASK, physics=False, zenbot_motion_train_wm=True,
              zenbot_wm_getup_only=True, wm_floor_amp=True)
    sim.dx = CpuState(jnp.zeros(30), jnp.zeros(29), jnp.zeros(29))
    state = sim.init(jax.random.key(13))
    for seed in (37, 59):
        state = sim._reset(state, jnp.ones(sim.N, bool), jax.random.key(seed))
        (q, v, up), (clips, frames) = sim._fresh(jax.random.key(seed), state.world, return_source=True)
        assert len(set(np.asarray(clips))) == 11
        np.testing.assert_array_equal(frames, 0)
        np.testing.assert_array_equal(up, False)
        np.testing.assert_array_equal(v[:, 2:], 0.)
        assert np.all(np.asarray(q)[:, 2] < .5)
        np.testing.assert_array_equal(state.was_up, False)
        np.testing.assert_array_equal(state.t, 0)
        np.testing.assert_array_equal(state.assist, 0.)
        assert state.reference_clip is None and state.motion_train_sensors.observation().shape == (256, 234)


def test_floor_amp_requires_online_motion_and_physical_rollouts():
    flags = ["--robot", zenbot.MIXED_TASK, "--motion_train_wm", "1", "--wm_getup_only", "1",
             "--wm_floor_amp", "1", "--wm_only_iters", "150", "--ppo_frac", ".1"]
    with pytest.raises(ValueError, match="Floor-opening AMP"):
        parse(flags + ["--wm_motion_replay", "1"])
    with pytest.raises(ValueError, match="motion-rate ramp"):
        parse(flags + ["--wm_rate_ramp", "2500"])
    learner = InstantAMP.__new__(InstantAMP)
    learner.cfg = parse(flags + ["--reference_weight", "18", "--wm_rate_ramp", "2500"])
    steps = jnp.array([0, 2500, 3750, 5000, 10000])
    np.testing.assert_allclose(learner.reference_rate_weight(steps), [0., 0., .5, 1., 1.])


def test_joint_floor_amp_exposes_eight_commands_and_uses_measured_support_to_release_locomotion():
    sim = Sim(256, physics=False, robot=zenbot.MIXED_TASK, zenbot_motion_train_wm=True,
              wm_floor_amp=True, wm_lift_gate=.44, support_rsi=.35, reset_floor=.5,
              randomize_world=False, obs_noise=False, push=False, prob_shove=0.)
    sim.dx = CpuState(jnp.zeros(30), jnp.zeros(29), jnp.zeros(29))
    s = sim.init(jax.random.key(44))
    assert set(sim.weights) == {"stand", "velocity", "action_rate", "lift"}
    assert sim.assist_newtons == 0. and sim.reference is None and not sim.reference_reward
    assert .25 < np.mean(s.down) < .5 and np.all(s.was_up[sim.wm_gait_worlds])
    assert set(map(tuple, np.asarray(s.cmd.vel))) == set(map(tuple, zenbot.DIRECTIONAL_COMMANDS))
    assert np.all((s.cmd.time_left >= 2.) & (s.cmd.time_left <= 4.))
    np.testing.assert_array_equal(s.c_obs, s.cmd.vel)
    np.testing.assert_array_equal(s.motion_train_sensors.observation()[:, 24:27], s.cmd.vel)
    # The GPU collector donates both state fields on its first step.
    transition = jax.jit(lambda cmd, obs: (cmd + 1., obs - 1.), donate_argnums=(0, 1))
    advanced, observed = transition(s.cmd.vel, s.c_obs)
    np.testing.assert_allclose(advanced - observed, 2., atol=1e-6)
    np.testing.assert_array_equal(s.recovery_left > 0, s.down)
    ready = sim.floor_gait_interval(jnp.array([200, 200, 0, 0]),
                                    jnp.array([True, True, True, False]), jnp.array([.9, 1., 0., 0.]))
    np.testing.assert_array_equal(ready, [200, 0, 0, 200])

    v = jnp.zeros((1, 3))
    q = sim.floor_stand_joints[None]
    a = jnp.zeros_like(q)
    pose = (jnp.array([[0.,0.,.68]]), jnp.eye(3)[None], v, v, q, a)
    def reward(command, speed, recovering):
        state = (*pose[:2], speed, *pose[3:])
        return sim.reward_terms(pose, state, sim.view_state(state).kin, a, a, command,
                                jnp.array([False]), jnp.zeros(1), jnp.ones((1, 2)), jnp.zeros(1),
                                sim.q_soft, recovery_active=jnp.array([recovering]), stand_blend=1.)
    run = jnp.array([[2.5, 0., 0.]])
    assert reward(run, run, False)["velocity"].item() == 1.
    for cmd in (run, jnp.array([[0., -.4, 0.]]), jnp.array([[-.5, 0., 0.]])):
        assert reward(cmd, v, False)["velocity"].item() == 0.
        assert reward(cmd, cmd, False)["velocity"].item() == 1.
        assert reward(cmd, -.5 * cmd, False)["velocity"].item() < 0.
    assert reward(run, run, False)["stand"].item() == 0.
    assert reward(run, v, True)["velocity"].item() == 0.
    assert reward(v, v, False)["stand"].item() > .99
    assert jax.grad(lambda vx: reward(run, v.at[0,0].set(vx), False)["velocity"].sum())(0.) > 0.


def test_joint_reward_averages_stride_velocity_and_samples_ground_to_stance_requests():
    sim = Sim(256, physics=False, robot=zenbot.MIXED_TASK, zenbot_motion_train_wm=True,
              wm_floor_amp=True, wm_lift_gate=.44, support_rsi=.35, reset_floor=2/3,
              gait_velocity_time_constant=.24, floor_stop_command_share=.5,
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
    prev = jnp.ones_like(state.gait_velocity) * 3.
    reset_mask = jnp.arange(sim.N)%2 == 0
    s = sim._reset(state._replace(gait_velocity=prev), reset_mask, jax.random.key(118))
    instant = sim.body_command_velocity(unflatten(s.data.qpos, s.data.qvel))
    np.testing.assert_allclose(s.gait_velocity[reset_mask], instant[reset_mask], atol=2e-6)
    np.testing.assert_array_equal(s.gait_velocity[~reset_mask], 3.)
    pose = unflatten(jnp.array([[0.,0.,.68,1.,0.,0.,0.]+[0.]*23]), jnp.zeros((1,29)))
    moving = (*pose[:2], jnp.array([[1.,0.,0.]]), *pose[3:])
    result = jnp.zeros((1,3))
    for _ in range(12):
        result = sim.body_command_velocity(moving, result)
    np.testing.assert_allclose(result, [[1.-np.exp(-1.),0.,0.]], atol=1e-6)


def test_amp_led_ground_practice_draws_unassisted_down_poses_on_both_sides_of_the_gate():
    sim = Sim(512, robot=zenbot.MIXED_TASK, physics=False, zenbot_motion_train_wm=True,
              zenbot_wm_getup_only=True, wm_floor_amp=True, wm_lift_gate=.44, support_rsi=.35)
    (qpos, qvel), (clip, frame, _) = sim._wm_floor_start(jax.random.key(689), sim.floor)
    middle = np.asarray(frame) > 0
    z = np.asarray(qpos[:, 2])
    assert sim.reference is None and set(sim.weights) == {"lift", "stand", "action_rate"}
    assert .28 < middle.mean() < .42
    assert len(set(np.asarray(clip)[~middle])) == 11
    assert (z[~middle] < .44).all() and (z[middle] <= zenbot.UP_HEIGHT).all()
    assert (z[middle] < .44).any() and (z[middle] > .54).any()
    np.testing.assert_array_equal(qvel[:, 2:], 0.)
    assert not sim.assist_newtons


def test_joint_floor_practice_includes_moving_low_support_and_cold_openings():
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


def test_recovery_sampling_finds_rare_transitions_to_supported_standing():
    steps, worlds = 32, 8
    states = jnp.zeros((steps, worlds, 1))
    gate = jnp.ones((steps, worlds)).at[18:21, 5].set(.7)
    buf = dict(s_qpos=states, s1_qpos=states, s_qvel=states, s1_qvel=states, stand_blend=gate,
               done=jnp.zeros_like(gate), ok=jnp.ones_like(gate), size=steps, ptr=0,
               t=jnp.broadcast_to(jnp.arange(steps)[:, None], gate.shape))
    windows = rb.sample_windows(buf, jax.random.key(43), 64, 5, recovery_only=True)
    np.testing.assert_array_equal(windows["stand_blend"][:, -1] < 1., True)
    np.testing.assert_array_equal(windows["mask"][:, -1], 1.)


def test_floor_discriminator_sees_the_same_reset_padding_as_the_policy():
    steps, worlds, frames = 20, 2, 5
    time = jnp.tile(jnp.arange(steps)[:,None], (1,worlds))
    states = time[..., None].astype(jnp.float32)
    # The recovery counter can restart when a robot refalls during the same physical episode.
    buf = dict(s_qpos=states, s1_qpos=states+1, s_qvel=states, s1_qvel=states+1, stand_blend=jnp.zeros_like(time),
               done=jnp.zeros_like(time), ok=jnp.ones_like(time), size=steps, ptr=0, t=time,
               recovery_left=jnp.where(time < 10, 200-time, 210-time))
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
    score = zenbot.floor_stand_score((pos, rotation, vel, jnp.zeros_like(vel), None, None), feet)
    np.testing.assert_allclose(score, [0., 0., 1.], atol=1.e-5)
    for z in (.15, .35, .57):
        low = zenbot.floor_stand_score((pos.at[:, 2].set(z), rotation, vel, vel, None, None), jnp.ones_like(feet))
        np.testing.assert_array_equal(low, 0.)


def test_amp_first_lift_waits_for_height_and_only_pays_for_progress():
    sim = Sim(1, robot=zenbot.MIXED_TASK, physics=False, zenbot_motion_train_wm=True,
              zenbot_wm_getup_only=True, wm_floor_amp=True, wm_lift_gate=.44)
    assert sim.reference is None and set(sim.weights) == {"lift", "stand", "action_rate"}
    def terms(z0, z1, joints=None, support=0.):
        p = jnp.array([[0., 0., z0]])
        q = sim.floor_stand_joints[None, :] if joints is None else joints
        v, actions = jnp.zeros_like(p), jnp.zeros_like(q)
        state = (p, jnp.eye(3)[None], v, v, q, actions)
        next_state = (p.at[:, 2].set(z1), *state[1:])
        return sim.weighted(sim.reward_terms(
            state, next_state, sim.view_state(next_state).kin, actions, actions, v, jnp.array([False]),
            jnp.zeros(1), jnp.ones((1, 2)), jnp.zeros(1), sim.q_soft, stand_blend=support))
    for a,b in ((.15,.32), (.41,.44), (.51,.51), (.6,.75)):
        np.testing.assert_allclose(terms(a,b)["lift"], 0.)
        np.testing.assert_allclose(terms(a,b)["stand"], 0.)
    fast = terms(.34,.67)["lift"]
    slow = sum(terms(a,b)["lift"] for a,b in ((.34,.44),(.44,.48),(.48,.54),(.54,.67)))
    np.testing.assert_allclose(fast, .5)
    np.testing.assert_allclose(slow, fast, atol=1e-6)
    np.testing.assert_allclose(fast + terms(.67,.34)["lift"], 0.)
    quiet = terms(zenbot.RISE_HEIGHT, zenbot.RISE_HEIGHT, support=1.)["stand"]
    np.testing.assert_allclose(quiet, sim.weights["stand"] * env.DT, atol=1e-6)
    crooked = sim.floor_stand_joints[None, :].at[:, 2:].add(.6)
    assert np.all(terms(.68,.68,crooked,support=1.)["stand"] < quiet)
    grad = jax.vmap(jax.grad(lambda h: zenbot.lift_progress(h,.44)))(jnp.array([.3,.44,.5,.58,.71]))
    assert grad[2] > 0.
    np.testing.assert_allclose(grad[jnp.array([0,1,3,4])], 0., atol=1e-5)


def test_floor_pose_score_keeps_the_assigned_clock_until_the_motion_ends(monkeypatch):
    def forbidden(*args):
        pytest.fail("Floor recovery uses only kinematic AMP references")

    monkeypatch.setattr(PhysicalRecoveryReference, "__init__", forbidden)
    sim = Sim(16, robot=zenbot.MIXED_TASK, physics=False, zenbot_motion_train_wm=True,
              zenbot_wm_getup_only=True, wm_floor_amp=True, reference_weight=18.)
    sim.dx = CpuState(jnp.zeros(30), jnp.zeros(29), jnp.zeros(29))

    def physics(*args):
        data, contact, hits, touch, _, acc = native_substeps(sim, *args)
        loads = jnp.zeros((sim.N, len(zenbot.SUPPORT_BODIES)))
        return data, contact, hits, touch, loads, acc

    sim.physics = physics
    s = sim.init(jax.random.key(15))
    key = jax.random.key(98)
    s = sim._reset(s, jnp.ones(sim.N, bool), key)
    _, (clip, frame) = sim._fresh(jax.random.split(key, 3)[0], s.world, return_source=True)
    np.testing.assert_array_equal(clip, s.reference_clip)
    np.testing.assert_array_equal(frame, s.t)
    assert s.motion_train_sensors.observation().shape == (16, 234)

    actions = jnp.zeros((sim.N, sim.J))
    post, transition = sim.step(s, actions)
    def rewarded(tr):
        return tr.parts[:, sim.terms.index("motion_reference")] / (.02 * sim.weights["motion_reference"])
    def expected_score(ref, tr, clip, frame, rate_weight=1.):
        current = unflatten(tr.qpos1, tr.qvel1)
        rate = ref.motion_rate(current[0], current[1], current[4], tr.qpos[..., :3], tr.qpos[..., 7:])
        return ref.score(sim.reference_pose(current), clip, frame, smooth_l1=True, rates=rate, rate_weight=rate_weight)

    np.testing.assert_allclose(rewarded(transition), expected_score(sim.reference, transition, clip, s.t + 1), atol=1e-6)
    for share in (0., .5):
        _, adjusted = sim.step(s, actions, reference_rate_weight=share)
        np.testing.assert_array_equal(adjusted.qpos1, transition.qpos1)
        expected = expected_score(sim.reference, transition, clip, s.t + 1, rate_weight=share)
        np.testing.assert_allclose(rewarded(adjusted), expected, atol=1e-6)
        assert np.all(rewarded(adjusted) > rewarded(transition))
    _, at_end = sim.step(s._replace(t=sim.reference.lengths[clip] - 1), actions)
    stand_ref = sim.standing_reference
    terminal = expected_score(stand_ref, at_end, jnp.zeros_like(clip), stand_ref.lengths[0] // 2, at_end.stats["stand_blend"])
    np.testing.assert_allclose(rewarded(at_end), terminal, atol=1e-6)
    assert np.all(np.asarray(terminal) < 0.)
    # At the opening, the fallen pose matches its own clip. Once late, it must get credit for moving.
    assert np.all(expected_score(sim.reference, transition, clip, s.t + 1, rate_weight=0.) > terminal)
    assert np.all(terminal > expected_score(stand_ref, at_end, jnp.zeros_like(clip), stand_ref.lengths[0] // 2))
    post_fall, falling = sim.step(s._replace(was_up=jnp.ones(sim.N, bool)), actions)
    np.testing.assert_array_equal(post_fall.reference_clip, clip)
    np.testing.assert_allclose(rewarded(falling), rewarded(transition), atol=1e-6)
    _, after_fall = sim.step(post_fall, actions)
    expected = expected_score(sim.reference, after_fall, clip, post_fall.t + 1)
    np.testing.assert_allclose(rewarded(after_fall), expected, atol=1e-6)
    target = sim.reference.at(clip, s.t)
    np.testing.assert_allclose(sim.reference.score(target, clip, s.t, smooth_l1=True), 1., atol=1e-6)
    assert np.all(sim.reference.score(target.at[:, 0].add(1.), clip, s.t, smooth_l1=True) < 0.)
    hold = zenbot.SupportedStand.empty(sim.N, .02)
    yes = jnp.ones(sim.N, bool)
    for _ in range(50):
        hold, blend = sim.advance_floor_stand(hold, yes, yes, clip, s.t)
    np.testing.assert_array_equal(blend, 0.)
    for _ in range(50):
        hold, blend = sim.advance_floor_stand(hold, yes, yes, clip, sim.reference.lengths[clip])
    np.testing.assert_array_equal(blend, 1.)

    with np.load(sim.standing_reference.files[0]) as data:
        mid = len(data["dof_pos"]) // 2
        qpos = np.r_[data["root_pos"][mid], data["root_quat"][mid][[3, 0, 1, 2]], data["dof_pos"][mid]]
    qpos = jnp.broadcast_to(qpos, (sim.N, len(qpos)))
    def standing_reward(q, blend):
        state = unflatten(q, jnp.zeros_like(s.data.qvel))
        end = sim.reference.lengths[clip]
        return sim.reward_terms(state, state, sim.view_state(state).kin, actions, actions, s.c_obs,
                                jnp.zeros(sim.N, bool), jnp.zeros(sim.N), jnp.ones((sim.N, 2)),
                                jnp.zeros(sim.N), sim.q_soft, clip=clip, step=end, stand_blend=blend)["motion_reference"]

    standing = standing_reward(qpos, 1.)
    np.testing.assert_allclose(standing, 1., atol=.001)
    np.testing.assert_allclose(standing_reward(qpos, .4), standing, atol=.001)
    np.testing.assert_allclose(standing_reward(qpos, 0.), standing, atol=.001)
    assert np.all(standing_reward(qpos.at[:, 20:].add(1.), 1.) < standing)


def test_timed_reference_distinguishes_fast_kicks_and_sliding_at_the_same_pose():
    from pathlib import Path
    from instantamp.rot import quat_to_mat

    ref = zenbot.RecoveryReference(Path("data/zenbot23_walk_run_recovery/recovery"), None,
                                   coupled_pose=True, with_rates=True)
    with np.load(ref.files[0]) as data:
        p, q = data["root_pos"], data["dof_pos"]
        rot = quat_to_mat(data["root_quat"][1:, [3, 0, 1, 2]])
    frames = jnp.arange(1, len(q))
    rates = ref.motion_rate(p[1:], rot, q[1:], p[:-1], q[:-1])
    pose = ref.at(0, frames)
    def score(rate):
        return ref.score(pose, 0, frames, smooth_l1=True, rates=rate)
    np.testing.assert_allclose(score(rates), 1., atol=1e-6)
    assert np.all(score(rates.at[:, 3].add(10.)) < .95)
    assert np.all(score(rates.at[:, 0].add(1.)) < .7)
    moving = rates.at[:, 0].add(1.)
    np.testing.assert_allclose(ref.score(pose, 0, frames, smooth_l1=True, rates=moving, rate_weight=0.), 1.)


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

    from instantamp.zenbot_motion_train_obs import mirror_action, mirror_observation

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


def test_upright_mirror_traces_the_complete_policy_iteration():
    if jax.devices()[0].platform not in ("gpu", "cuda"):
        pytest.skip("The physical MuJoCo Warp collector requires CUDA")
    cfg = parse(["--robot", zenbot.MIXED_TASK, "--motion_train_wm", "1", "--wm_floor_amp", "1",
                 "--gait_ls_scale", "10", "--gait_mirror_weight", ".2", "--wm_lift_gate", ".44",
                 "--num_envs", "32", "--pi_batch", "24", "--wm_batch", "16", "--amp_batch", "32",
                 "--buffer_steps", "64", "--env_steps", "4", "--ppo_len", "4", "--warmup_steps", "48"])
    learner = InstantAMP(cfg)
    carry = learner.init(jax.random.PRNGKey(42))
    _, stats = jax.eval_shape(learner.pi_iter, carry)
    assert stats["gait_mirror_mse"].shape == ()
