"""CPU integration contracts for the Motion Train dynamics switch."""

import json
from dataclasses import asdict
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from instantamp import env, nets, zenbot, zenbot_gait_rewards
from instantamp.sim import Sim, unflatten
from instantamp.train import InstantAMP, Zenbot23JointConfig, ZenbotMotionTrainWMConfig, parse
from instantamp.training_sim import simulation_options

pytestmark = pytest.mark.usefixtures("zenbot23_motion_banks")


class CpuData(NamedTuple):
    qpos: object
    qvel: object
    qacc_warmstart: object

    def replace(self, **values):
        return self._replace(**values)


def simulator(n=8, *, aligned=True, randomize=False):
    sim = Sim(
        n, physics=False, robot=zenbot.MIXED_TASK, zenbot_motion_train_wm=True,
        motion_train_randomization=aligned, randomize_world=randomize, obs_noise=False,
        random_start=False, reset_floor=.4, wm_floor_amp=True, wm_fixed_floor_envs=True,
        wm_global_gait=True, wm_gait_rewards=True, wm_posture_and_tracking=True,
        wm_lift_gate=.44,
    )
    sim.dx = CpuData(jnp.zeros(30), jnp.zeros(29), jnp.zeros(29))
    return sim


def test_new_recipe_enables_alignment_but_saved_legacy_config_remains_legacy():
    recipe = asdict(Zenbot23JointConfig())
    assert recipe["motion_train_randomization"] == 1
    assert simulation_options(recipe)["motion_train_randomization"] == 1
    assert ZenbotMotionTrainWMConfig().motion_train_randomization == 0
    legacy = {key: value for key, value in recipe.items() if key != "motion_train_randomization"}
    assert "motion_train_randomization" not in simulation_options(legacy)
    recipe["motion_train_randomization"] = 0
    assert simulation_options(recipe)["motion_train_randomization"] == 0


def test_full_learner_cannot_silently_change_dynamics_context(tmp_path):
    run = tmp_path / "old_run"
    (run / "ckpt/full").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"motion_train_wm": 1}))
    checkpoint = str(run / "ckpt/full/old.npz")
    with pytest.raises(ValueError, match="same motion_train_randomization context"):
        Zenbot23JointConfig(init_full=checkpoint, log_dir=str(tmp_path / "new_run"))
    continued = Zenbot23JointConfig(init_full=checkpoint, log_dir=str(tmp_path / "continued"),
                                   motion_train_randomization=0)
    assert continued.motion_train_randomization == 0


def test_aligned_nominal_world_preserves_actor_and_has_only_nonzero_motor_delays():
    sim = simulator(n=64)
    state = sim.init(jax.random.key(813))
    assert sim.terrain is None and sim.force_push is None
    assert state.world.terrain_origin is None and state.force_push is None
    assert sim.zero_motor_delay_probability == 0.
    assert np.asarray(state.delay_steps).min() == 3
    assert np.asarray(state.delay_steps).max() == 7
    assert state.motion_train_sensors.observation().shape == (64, 234)
    for name, expected in (("kp", zenbot.KP), ("kd", zenbot.KD)):
        np.testing.assert_array_equal(state.world.dynamics[name], np.broadcast_to(expected, (64, 23)))
    np.testing.assert_array_equal(state.world.body_ipos, np.broadcast_to(sim.m.body_ipos, (64, sim.m.nbody, 3)).astype(np.float32))
    context = sim.context(state)
    assert context.shape[1] > 8
    np.testing.assert_allclose(sim.context_root_com(context), state.world.body_ipos[:, 1], atol=1e-7)
    # The model-based effort estimate must use the same PD gains as physics.
    pose = unflatten(state.data.qpos, state.data.qvel)
    target = state.data.qpos[:, 7:] + .015
    expected = zenbot.actuator_ctrl(pose[4], pose[5], target, kp=state.world.dynamics["kp"], kd=state.world.dynamics["kd"])
    np.testing.assert_allclose(sim.estimated_torque(pose, jnp.tile(target, (1, 4)), context), expected, atol=1e-5)


def test_old_simulator_context_and_zero_delay_mixture_are_unchanged():
    sim = simulator(n=512, aligned=False)
    state = sim.init(jax.random.key(131))
    assert state.world.dynamics is None and state.world.terrain_origin is None
    assert state.force_push is None and sim.terrain is None
    assert sim.context(state).shape == (512, 8)
    assert sim.zero_motor_delay_probability == .5
    zero = np.all(np.asarray(state.delay_steps) == 0, axis=-1)
    assert .42 < zero.mean() < .58
    assert np.all(np.asarray(state.delay_steps)[~zero] >= 3)
    np.testing.assert_array_equal(sim.m.dof_frictionloss[sim.va[:4]], .986)


def test_episode_reset_keeps_startup_dynamics_and_terrain_but_resets_push_timer():
    sim = simulator(randomize=True)
    state = sim.init(jax.random.key(123))
    assert sim.terrain is not None and sim.force_push is not None
    assert state.force_push is not None
    assert np.asarray(sim.terrain.contains(state.data.qpos[:, :2])).all()
    assert np.any(np.asarray(state.world.dynamics["kp"]) != zenbot.KP)
    # Establish an active pulse to ensure reset cannot carry its force onward.
    old_push = state.force_push._replace(
        remaining=jnp.full(sim.N, .10), force_w=jnp.ones((sim.N, 3)),
        wait=jnp.full(sim.N, 2.),
    )
    state = state._replace(force_push=old_push)
    mask = jnp.arange(sim.N) % 2 == 0
    reset = sim._reset(state, mask, jax.random.key(456))
    for name, before in state.world.dynamics.items():
        np.testing.assert_array_equal(reset.world.dynamics[name], before)
    np.testing.assert_array_equal(reset.world.terrain_origin, state.world.terrain_origin)
    np.testing.assert_array_equal(np.asarray(reset.force_push.force_w)[mask], 0.)
    np.testing.assert_array_equal(np.asarray(reset.force_push.force_w)[~mask], 1.)
    assert np.all(np.asarray(reset.force_push.wait)[mask] >= 5.)
    # Startup privileged parameters round-trip through the actual model input.
    xi = sim.context(reset)
    assert np.isfinite(xi).all()
    np.testing.assert_allclose(sim.context_root_com(xi), reset.world.body_ipos[:, 1], atol=1e-7)
    pose = unflatten(reset.data.qpos, reset.data.qvel)
    target = pose[4] + .01
    expected = zenbot.actuator_ctrl(pose[4], pose[5], target, kp=state.world.dynamics["kp"], kd=state.world.dynamics["kd"])
    np.testing.assert_allclose(sim.estimated_torque(pose, target, xi), expected, atol=1e-5)


def test_raising_ground_and_robot_preserves_geometry_rewards_and_amp_height():
    sim = simulator(n=4)
    qpos = jnp.tile(jnp.r_[0., 0., .68, 1., 0., 0., 0., zenbot.DEFAULT_Q], (4, 1))
    qpos = qpos.at[:, 2].set(jnp.array([.24, .46, .64, .70]))
    qvel = jnp.zeros((4, 29)).at[:, 0].set(.2)
    state = unflatten(qpos, qvel)
    view = sim.view_state(state)
    weight = sim.gait_weight(state)
    amp_frame = sim._gait_amp_frame(state, sim.key_positions(state[0], state[1], state[4]))
    zeros, actions = jnp.zeros(4), jnp.zeros((4, 23))
    commands = jnp.zeros((4, 3))

    def reward(current, geometry):
        return sim.reward_terms(
            current, current, geometry.kin, actions, actions, commands,
            jnp.zeros(4, bool), zeros, jnp.ones((4, 2)), zeros, sim.q_soft,
        )

    def foot_reward(current, geometry):
        return zenbot_gait_rewards.reward_terms(
            current, geometry.feet, commands, actions, actions, sim.q_soft,
            jnp.ones((4, 2), bool), zeros, geometry.feet.predicted_slip,
            jnp.zeros((4, 2)), zenbot_gait_rewards.StepHistory.empty(4),
            sim.gait_weight(current), min_ankle_spacing=sim.min_ankle_spacing,
        )[1]

    before = reward(state, view)
    foot_before = foot_reward(state, view)
    # A flat plateau isolates height bookkeeping from slope/roughness changes.
    sim.terrain = SimpleNamespace(height_at=lambda xy: jnp.full(xy.shape[:-1], .35, dtype=xy.dtype))
    raised = unflatten(qpos.at[:, 2].add(.35), qvel)
    raised_view = sim.view_state(raised)
    for name in ("height", "soles", "clearance"):
        np.testing.assert_allclose(getattr(raised_view, name), getattr(view, name), atol=2e-7)
    np.testing.assert_allclose(raised_view.feet.bottom, view.feet.bottom, atol=2e-7)
    assert np.asarray(view.feet.predicted_slip).max() > 0.
    np.testing.assert_allclose(raised_view.feet.predicted_slip, view.feet.predicted_slip, atol=2e-6)
    np.testing.assert_allclose(sim.gait_weight(raised), weight, atol=2e-6)
    after = reward(raised, raised_view)
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_allclose(after[name], before[name], atol=3e-6, err_msg=name)
    foot_after = foot_reward(raised, raised_view)
    for name in foot_before:
        np.testing.assert_allclose(foot_after[name], foot_before[name], atol=3e-6, err_msg=name)
    raised_amp = sim._gait_amp_frame(raised, sim.key_positions(raised[0], raised[1], raised[4]))
    np.testing.assert_allclose(raised_amp, amp_frame, atol=2e-7)


def test_randomized_context_and_rough_terrain_have_finite_imagination_gradients():
    n = 2
    sim = simulator(n=n, randomize=True)
    state = sim.init(jax.random.key(142))
    # Put both worlds above actual rough patches, rather than accidentally only
    # exercising the flat share of the startup terrain distribution.
    indices = np.argwhere((sim.terrain.kinds > 0) & (sim.terrain.kinds < 7))[:n]
    origins = jnp.asarray(sim.terrain.origins[indices[:, 0], indices[:, 1]])
    qpos = jnp.tile(jnp.r_[0., 0., .70, 1., 0., 0., 0., zenbot.DEFAULT_Q], (n, 1))
    qpos = qpos.at[:, :3].add(origins)
    qvel = jnp.zeros((n, 29))
    command = jnp.array([[.3, 0., 0.], [0., .2, 0.]])
    state = state._replace(data=state.data.replace(qpos=qpos, qvel=qvel), c_obs=command)
    pose = unflatten(qpos, qvel)
    sensors = sim.motion_train_sensors(state, pose[1], command, jnp.ones(n, bool), jax.random.key(143))
    xi = sim.context(state)
    assert xi.shape == (n, 219)
    zeros = jnp.zeros(n)
    start = dict(
        s_qpos=qpos, s_qvel=qvel, xi=xi, bias=state.world.bias,
        obs=sensors.observation(), imu_delay=sensors.delay_s,
        current_imu=sensors.current_imu, previous_imu=sensors.previous_imu,
        a_last=state.a_last, a_ode=state.a_ode, c_obs=command,
        cmd=env.Cmd(command, zeros, zeros > 0, zeros > 0, zeros + 8.),
        t=jnp.full(n, 4, jnp.int32), was_up=jnp.ones(n, bool),
        gait_world=jnp.ones(n, bool), gait_stage=jnp.zeros(n, jnp.int32),
        episode_limit=jnp.full(n, 400, jnp.int32),
        step_history=zenbot_gait_rewards.StepHistory.empty(n)._asdict(),
    )
    learner = InstantAMP.__new__(InstantAMP)
    learner.sim, learner.B_im, learner.J = sim, n, 23
    learner.cfg = parse([
        "--robot", zenbot.MIXED_TASK, "--num_envs", "32", "--horizon", "2",
        "--imag_unroll", "1", "--mm_bf16", "none", "--imag_gait_mirror_weight", "0",
    ])
    assert learner.cfg.motion_train_randomization == 1
    pi = nets.mlp_init(jax.random.key(144), (234, 16, 23))
    view = sim.view_state(pose)
    target = jnp.tile(qpos[:, 7:], (1, 4))
    width = nets.wm_inputs(pose, view.height, view.soles, target, xi).shape[-1]
    wm, bounds = nets.wm_init(jax.random.key(145), width, 58, (16,))
    # Nonzero learned dynamics give a path through predicted pose/velocity,
    # terrain-relative geometry and the real physical reward implementation.
    wm["net"][-1]["w"] = jax.random.normal(jax.random.key(146), (16, 58)) * .001
    keys = jax.random.split(jax.random.key(147), 2)

    def objective(actor, dynamics):
        return learner.imagine(actor, dynamics, bounds, {"raw": True}, start, keys)[0]

    value, (actor_grad, model_grad) = jax.jit(jax.value_and_grad(objective, (0, 1)))(pi, wm)
    assert np.isfinite(value)
    for gradient in (actor_grad, model_grad):
        assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradient))
        assert sum(float(jnp.square(leaf).sum()) for leaf in jax.tree.leaves(gradient)) > 1e-12
