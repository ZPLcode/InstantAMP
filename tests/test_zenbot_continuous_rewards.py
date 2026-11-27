"""Physical-state reward contracts for both Zenbot articulations."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from instantamp import amp, env, zenbot, zenbot_gait_rewards, zenbot_models
from instantamp.sim import Sim, unflatten
from instantamp.train import InstantAMP, parse
from instantamp.training_sim import simulation_options


@pytest.fixture(params=zenbot_models.MODELS, ids=("23dof", "26dof"))
def joint_sim(request, tmp_path):
    """Build actual reward/kinematics code using tiny synthetic reset banks."""
    model = request.param
    n = 80
    joints = np.broadcast_to(model.default_q, (n, len(model.dof)))
    for bank, name in (("walk_run", "stand_4s_120f"), ("recovery", "recovery")):
        directory = tmp_path / bank
        directory.mkdir()
        pos = np.zeros((n, 3), np.float32)
        pos[:, 2] = .68 if bank == "walk_run" else np.linspace(.20, .68, n)
        np.savez(
            directory / f"{name}.npz", fps=50, root_pos=pos,
            root_quat=np.tile([0., 0., 0., 1.], (n, 1)),
            root_lin_vel=np.zeros((n, 3)), root_ang_vel=np.zeros((n, 3)),
            dof_pos=joints, dof_vel=np.zeros_like(joints),
        )
    cfg = parse([
        "--robot", model.mixed_task, "--num_envs", "8",
        "--walk_dir", str(tmp_path / "walk_run"),
        "--amp_dir", str(tmp_path / "walk_run"),
        "--floor_dir", str(tmp_path / "recovery"),
        "--amp_getup_dir", str(tmp_path / "recovery"),
    ])
    sim = Sim(**simulation_options(vars(cfg)), physics=False, obs_noise=False)
    return sim, cfg


def physical_state(sim, height=.68, pitch=0., velocity=(0., 0., 0.)):
    qpos = jnp.r_[0., 0., height, jnp.cos(pitch / 2), 0., jnp.sin(pitch / 2), 0., sim.robot.DEFAULT_Q][None]
    qvel = jnp.zeros((1, sim.J + 6)).at[0, :3].set(jnp.asarray(velocity))
    return unflatten(qpos, qvel)


def task_terms(sim, state, *, previous=None, command=(0., 0., 0.), contact=1., **history):
    action = jnp.full((1, sim.J), .1)
    return sim.reward_terms(
        state if previous is None else previous, state, sim.view_state(state).kin,
        action, jnp.zeros_like(action), jnp.asarray(command)[None],
        jnp.zeros(1, bool), jnp.zeros(1), jnp.broadcast_to(jnp.asarray(contact), (1, 2)),
        jnp.zeros(1), sim.q_soft, **history,
    )


def learner(sim, cfg):
    result = InstantAMP.__new__(InstantAMP)
    result.sim, result.cfg, result.J = sim, cfg, sim.J
    return result


def test_same_pose_has_same_task_and_critic_despite_recovery_history(joint_sim):
    sim, cfg = joint_sim
    state = physical_state(sim, height=.55, pitch=.55)
    old_history = dict(recovery_active=jnp.ones(1, bool))
    ready_history = dict(recovery_active=jnp.zeros(1, bool))
    for command in ((0., 0., 0.), (.5, 0., 0.)):
        before = task_terms(sim, state, command=command, **old_history)
        after = task_terms(sim, state, command=command, **ready_history)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)

    xs = learner(sim, cfg)
    args = (state, state[0][:, 2], jnp.zeros((1, sim.J)), jnp.zeros((1, 3)), jnp.zeros((1, 2)))
    before = xs.critic_in(*args, jnp.zeros(1, bool), jnp.array([1]), jnp.array([200]))
    after = xs.critic_in(*args, jnp.ones(1, bool), jnp.array([300]), jnp.array([0]))
    np.testing.assert_array_equal(before, after)


def test_quiet_stance_remains_rewarded_and_does_not_reward_lying(joint_sim):
    sim, _ = joint_sim
    state = physical_state(sim)
    quiet = task_terms(sim, state, recovery_active=False)["stand"]
    assert quiet.item() > .9
    assert task_terms(sim, physical_state(sim, height=.20, pitch=np.pi / 2))["stand"].item() == 0.
    assert task_terms(sim, state, contact=0.)["stand"].item() == 0.
    assert task_terms(sim, state, command=(2.5, 0., 0.))["stand"].item() < .01 * quiet.item()
    assert task_terms(sim, state, command=(0., 0., 1.))["stand"].item() < .01 * quiet.item()


def test_default_stand_pose_cost_only_shapes_quiet_double_support(joint_sim):
    sim, cfg = joint_sim
    assert sim.weights.get("stand_default_pose") == (-2. if sim.J == 26 else None)
    cfg = dataclasses.replace(cfg, wm_stand_pose_weight=-2.)
    sim = Sim(**simulation_options(vars(cfg)), physics=False, obs_noise=False)
    state = physical_state(sim)
    # The model's joint angles are the target even if the AMP reference differs.
    sim.floor_stand_joints = sim.floor_stand_joints + .1

    def terms(pose, offset, **kwargs):
        return task_terms(sim, (*pose[:4], pose[4] + offset, pose[5]), **kwargs)

    key = "stand_default_pose"
    delta = np.deg2rad(10.)
    assert terms(state, 0.)[key].item() == 0.
    perturbed = terms(state, delta, contact=(True, True))
    np.testing.assert_allclose(perturbed[key], .5, atol=1e-6)
    contact_terms = dict.fromkeys(sim.terms, jnp.zeros(1))
    np.testing.assert_allclose(sim.weighted({**contact_terms, **perturbed})[key], -env.DT, atol=1e-7)
    np.testing.assert_allclose(terms(state, -delta, contact=(1., .5))[key], .25, atol=1e-6)
    for command in ((.01, 0., 0.), (2.5, 0., 0.), (0., .1, 0.), (0., 0., .1)):
        assert terms(state, delta, command=command)[key].item() == 0.
    for contact in ((True, False), (False, False)):
        assert terms(state, delta, contact=contact)[key].item() == 0.
    for pose in (physical_state(sim, height=.50), physical_state(sim, pitch=.8)):
        assert terms(pose, delta)[key].item() == 0.
    pushed = physical_state(sim, velocity=(1., 0., 0.))
    assert terms(pushed, delta)[key].item() < .02 * perturbed[key].item()

    gradient = jax.jit(jax.grad(lambda offset: terms(state, offset)[key].sum()))
    assert gradient(0.).item() == 0.
    assert gradient(delta).item() > 0. and gradient(-delta).item() < 0.


def test_state_blend_is_bounded_continuous_and_independent_of_flight(joint_sim):
    sim, _ = joint_sim
    heights = jnp.linspace(.1, .9, 513)
    cosines = jnp.linspace(-1., 1., 513)
    height_curve = zenbot.gait_weight(heights, jnp.ones_like(heights))
    tilt_curve = zenbot.gait_weight(jnp.full_like(cosines, .68), cosines)
    for curve in (height_curve, tilt_curve):
        assert np.all(np.isfinite(curve))
        assert np.all((curve >= 0.) & (curve <= 1.))
        assert np.all(np.diff(curve) >= -1e-6)
        assert np.max(np.abs(np.diff(curve))) < .08
        np.testing.assert_allclose(np.asarray(curve)[[0, -1]], [0., 1.], atol=1e-6)
    derivatives = (
        jax.vmap(jax.grad(zenbot.gait_weight, 0))(heights, jnp.ones_like(heights)),
        jax.vmap(jax.grad(zenbot.gait_weight, 1))(jnp.full_like(cosines, .68), cosines),
    )
    assert all(np.all(np.isfinite(value)) for value in derivatives)
    assert all(np.max(np.asarray(value)) > 0. for value in derivatives)
    for height, velocity in ((.68, (0., 0., 0.)), (.80, (2.5, 0., 1.)), (.70, (2.5, 0., -1.))):
        np.testing.assert_allclose(sim.gait_weight(physical_state(sim, height=height, velocity=velocity)), 1.)


def test_motor_costs_and_task_style_mix_do_not_change_during_recovery(joint_sim):
    sim, cfg = joint_sim
    state = physical_state(sim)
    state = (*state[:5], jnp.full_like(state[5], .3))
    feet = sim.view_state(state).feet
    zeros = jnp.zeros(1)
    by_blend = []
    for gait in (0., .4, 1.):
        _, terms = zenbot_gait_rewards.reward_terms(
            state, feet, jnp.zeros((1, 3)), state[5] * 2., state[5] * 3., sim.q_soft,
            jnp.ones((1, 2), bool), zeros, zeros, jnp.zeros((1, 2)),
            zenbot_gait_rewards.StepHistory.empty(1), jnp.array([gait]),
        )
        by_blend.append(terms)
    for key in ("joint_vel_l2", "joint_acc_l2", "joint_torques_l2", "joint_energy", "joint_pos_limits"):
        np.testing.assert_array_equal(by_blend[0][key], by_blend[1][key], err_msg=key)
        np.testing.assert_array_equal(by_blend[0][key], by_blend[2][key], err_msg=key)

    xs = learner(sim, cfg)
    task = jnp.array([-.3, -.3, -.3])
    styles = jnp.array([1., 2., 3.])
    reward, style_reward = xs.mix_amp_reward(task, styles)
    np.testing.assert_allclose(style_reward, env.DT * .7 * styles)
    np.testing.assert_allclose(reward, .3 * task + env.DT * .7 * styles)
    increment = xs.mix_amp_reward(task + .2, styles)[0] - reward
    np.testing.assert_allclose(increment, (1. - cfg.amp_lerp) * .2, rtol=1e-6)


def test_two_motion_priors_have_shared_or_independent_reward_scales(joint_sim):
    sim, cfg = joint_sim
    xs = learner(sim, cfg)
    # Constant discriminator output isolates scale from different feature layouts.
    width = sim.amp_dim
    windows = jnp.zeros((2, cfg.amp_frames, width))
    st = amp.init(jax.random.key(21), cfg.amp_frames, hidden=(4,), frame_dim=width)
    st["net"] = jax.tree.map(jnp.zeros_like, st["net"])
    gait = xs.style(st, windows)
    recovery = xs.getup_style(st, windows)
    np.testing.assert_allclose(gait, .75 * cfg.gait_ls_scale)
    np.testing.assert_allclose(recovery, .75 * (10. if sim.J == 26 else cfg.gait_ls_scale))
    xs.cfg = dataclasses.replace(cfg, getup_amp_scale=0.)
    np.testing.assert_allclose(xs.getup_style(st, windows), gait)
    xs.cfg = dataclasses.replace(cfg, gait_ls_scale=3., getup_amp_scale=10.)
    np.testing.assert_allclose(xs.getup_style(st, windows), 7.5)
    np.testing.assert_allclose(xs.style(st, windows), 2.25)


def test_lift_ablation_changes_only_weighted_lifting_reward(joint_sim):
    sim, cfg = joint_sim
    reduced_cfg = dataclasses.replace(cfg, wm_lift_weight=cfg.wm_lift_weight * .4)
    reduced = Sim(**simulation_options(vars(reduced_cfg)), physics=False, obs_noise=False)
    before = physical_state(sim, height=.44)
    after = physical_state(sim, height=.51)
    assert reduced.weights == {**sim.weights, "lift": reduced_cfg.wm_lift_weight}
    # Contact-history terms are added by the collector, outside reward_terms.
    contact_terms = {key: jnp.zeros(1) for key in sim.terms}
    original = sim.weighted({**contact_terms, **task_terms(sim, after, previous=before)})
    actual = reduced.weighted({**contact_terms, **task_terms(reduced, after, previous=before)})
    for key in original:
        np.testing.assert_allclose(actual[key], original[key] * (.4 if key == "lift" else 1.), rtol=1e-6)


def test_gait_discriminator_weights_valid_windows_equally_across_poses(joint_sim):
    sim, cfg = joint_sim
    xs = learner(sim, cfg)
    pos = jnp.zeros((5, 10, 3)).at[..., 2].set(.68)
    # Include standing, rising and fallen poses as well as different histories.
    pos = pos.at[1, :5, 2].set(.20).at[2, -1, 2].set(.51).at[3, -1, 2].set(.20)
    rotation = jnp.broadcast_to(jnp.eye(3), (5, 10, 3, 3))
    window = {
        "mask": jnp.ones((5, 10)).at[4, -1].set(0.),
        "was_up": jnp.ones((5, 10), bool).at[1].set(False),
        "recovery_left": jnp.zeros((5, 10)).at[1].set(200.),
    }
    np.testing.assert_array_equal(xs.gait_amp_eligible(window, (pos, rotation)), [1, 1, 1, 1, 0])


def test_existing_startup_curve_and_sampling_defaults_are_unchanged(joint_sim):
    sim, cfg = joint_sim
    assert cfg.wm_lift_gate == .44
    assert (cfg.reset_floor, cfg.support_rsi, cfg.floor_stop_command_share) == (.4, .35, .5)
    np.testing.assert_allclose(zenbot.lift_progress(jnp.array([.10, .44, .51, .58, .70]), .44), [0., 0., .5, 1., 1.], atol=1e-6)
    assert sim.weights["lift"] == (.2 if sim.J == 26 else .5)
    for before, after in ((.20, .30), (.44, .51), (.51, .58)):
        terms = task_terms(sim, physical_state(sim, height=after), previous=physical_state(sim, height=before))
        expected = (zenbot.lift_progress(after, .44) - zenbot.lift_progress(before, .44)) / env.DT
        np.testing.assert_allclose(terms["lift"], expected, rtol=1e-6)


def test_first_attempt_is_charged_once_per_fall_and_allows_ground_repositioning(joint_sim):
    sim, cfg = joint_sim
    assert learner(sim, cfg).floor_reward_version == (5 if sim.J == 26 else 3)
    assert sim.weights["first_attempt_failure"] == (-2. if sim.J == 26 else -1.)
    assert not any(key.startswith("recovery_") for key in sim.weights)
    hold = round(1. / env.DT)

    def rollout(heights, cosines, supported, *, starts_down=True, gait=False):
        history = zenbot_gait_rewards.FirstAttempt.start(jnp.array(heights[0]), jnp.array(starts_down), sim.robot)
        def step(history, row):
            height, cosine, contact, done = row
            return history.advance(height, cosine, contact >= .5, done, jnp.asarray(gait), sim.robot)
        data = (jnp.array(heights), jnp.array(cosines), jnp.array(supported),
                jnp.arange(len(heights)) == len(heights) - 1)
        _, events = jax.lax.scan(step, history, data)
        return np.flatnonzero(np.asarray(events * env.DT) > .5).tolist()

    # A low retry detected only by the 26-DoF profile, then a successful rise.
    h = [.30, .16, .32, .23, .56] + [.67] * hold
    assert rollout(h, [1.] * len(h), [0.] * 5 + [1.] * hold) == ([3] if sim.J == 26 else [])
    # A fall bottoming out above 30 cm can still lead to an attempt.
    h = [.68, .42, .31, .39, .31] + [.67] * hold
    assert rollout(h, [1.] * len(h), [1.] * len(h), starts_down=False) == ([4] if sim.J == 26 else [])
    # A high initial pose followed by descent alone is not a failed rise on Zenbot26.
    h = [.48, .36, .25, .31, .22] + [.67] * hold
    assert rollout(h, [1.] * len(h), [1.] * len(h)) == ([] if sim.J == 26 else [1])
    # One setback followed by more retries and eventual success. A later fall gets a new attempt.
    h = [.20, .50, .39, .51, .38] + [.67] * hold + [.31, .50, .38]
    assert rollout(h, [1.] * len(h), [1.] * len(h)) == [2, len(h) - 1]
    # Tilt alone can spoil a briefly supported rise even at constant pelvis height.
    h = [.25] + [.68] * 26
    assert rollout(h, [0.] + [1.] * 12 + [0.] * 14, [1.] * len(h)) == [22]
    # Never rising, or reaching full height without support, fails at the deadline.
    for h in ([.26] * 65, [.26] + [.68] * 64):
        assert rollout(h, [1.] * len(h), [0.] * len(h)) == [64]
    h = [.20] + [.68] * hold
    assert rollout(h, [1.] * len(h), [0.] * 6 + [1.] * (hold - 5)) == []
    assert rollout(h, [1.] * len(h), [0.] * 7 + [1.] * (hold - 6)) == [hold]
    h = [.68, .47, .28, .27, .54] + [.68] * hold
    assert rollout(h, [1.] * len(h), [1.] * len(h), starts_down=False) == []
    assert rollout([.68, .28], [1., 0.], [1., 0.], starts_down=False, gait=True) == []


def test_shoulders_arm_before_pelvis_and_charge_only_once(joint_sim):
    sim, _ = joint_sim
    if sim.J != 26:
        assert sim.view_state(physical_state(sim)).shoulder_height is None
        return
    from mujoco import MjData, mj_kinematics
    native = MjData(sim.m)
    native.qpos[:] = np.r_[0., 0., .2, np.cos(.6), 0., np.sin(.6), 0., sim.robot.DEFAULT_Q]
    native.qpos[7 + sim.robot.DOF.index("waist_pitch")] = -.3
    mj_kinematics(sim.m, native)
    ids = [sim.m.body("robot_" + name).id for name in sim.robot.SHOULDERS]
    view = sim.view_state(unflatten(jnp.asarray(native.qpos), jnp.asarray(native.qvel)))
    np.testing.assert_allclose(view.shoulder_height, native.xpos[ids, 2].mean(), atol=1e-6)
    h = zenbot_gait_rewards.FirstAttempt.start(jnp.array(.17), jnp.array(True), sim.robot, shoulder_height=jnp.array(.45))
    def step(history, pair):
        pelvis, shoulders = pair
        return history.advance(pelvis, 0., jnp.array(False), jnp.array(False), jnp.array(False),
                               sim.robot, shoulder_height=shoulders)
    # Forced descent, a small bob, a shoulder-only setback, then a pelvis setback.
    _, events = jax.lax.scan(step, h, jnp.array([[.17,.28],[.16,.16],[.19,.25],[.17,.14],
                                                 [.20,.29],[.22,.20],[.21,.16],[.35,.41],[.24,.27]]))
    np.testing.assert_allclose(events * env.DT, [0.,0.,0.,0.,0.,0.,1.,0.,0.])
