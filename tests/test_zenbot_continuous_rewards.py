"""Physical-state reward contracts for both Zenbot articulations."""

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
        jnp.zeros(1, bool), jnp.zeros(1), jnp.full((1, 2), contact),
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
    np.testing.assert_allclose(style_reward, env.DT * cfg.amp_lerp * styles)
    increment = xs.mix_amp_reward(task + .2, styles)[0] - reward
    np.testing.assert_allclose(increment, (1. - cfg.amp_lerp) * .2, rtol=1e-6)


def test_two_motion_priors_share_reward_scale(joint_sim):
    sim, cfg = joint_sim
    xs = learner(sim, cfg)
    # Constant discriminator output isolates scale from different feature layouts.
    width = sim.amp_dim
    windows = jnp.zeros((2, cfg.amp_frames, width))
    st = amp.init(jax.random.key(21), cfg.amp_frames, hidden=(4,), frame_dim=width)
    st["net"] = jax.tree.map(jnp.zeros_like, st["net"])
    gait = xs.style(st, windows)
    recovery = xs.getup_style(st, windows)
    np.testing.assert_allclose(gait, recovery)
    np.testing.assert_allclose(recovery, .75 * cfg.gait_ls_scale)


def test_discriminator_suitability_uses_current_pose_without_old_history(joint_sim):
    sim, cfg = joint_sim
    xs = learner(sim, cfg)
    pos = jnp.zeros((4, 10, 3)).at[..., 2].set(.68)
    # Identical current stances have different preceding poses and old timer values.
    pos = pos.at[1, :5, 2].set(.20).at[2, -1, 2].set(.51)
    rotation = jnp.broadcast_to(jnp.eye(3), (4, 10, 3, 3))
    window = {
        "mask": jnp.ones((4, 10)).at[3, -1].set(0.),
        "was_up": jnp.ones((4, 10), bool).at[1].set(False),
        "recovery_left": jnp.zeros((4, 10)).at[1].set(200.),
    }
    np.testing.assert_allclose(xs.gait_amp_eligible(window, (pos, rotation)), [1., 1., .5, 0.], atol=1e-6)


def test_existing_startup_curve_and_sampling_defaults_are_unchanged(joint_sim):
    sim, cfg = joint_sim
    assert cfg.wm_lift_gate == .44
    assert (cfg.reset_floor, cfg.support_rsi, cfg.floor_stop_command_share) == (.4, .35, .5)
    np.testing.assert_allclose(zenbot.lift_progress(jnp.array([.10, .44, .51, .58, .70]), .44), [0., 0., .5, 1., 1.], atol=1e-6)
    assert sim.weights["lift"] == .5
    for before, after in ((.20, .30), (.44, .51), (.51, .58)):
        terms = task_terms(sim, physical_state(sim, height=after), previous=physical_state(sim, height=before))
        expected = (zenbot.lift_progress(after, .44) - zenbot.lift_progress(before, .44)) / env.DT
        np.testing.assert_allclose(terms["lift"], expected, rtol=1e-6)
