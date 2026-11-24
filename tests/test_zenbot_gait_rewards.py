from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from instantamp import env, zenbot, zenbot_gait_rewards as rewards
from instantamp.sim import Sim, unflatten


def test_completed_swing_earns_credit_once_and_uses_command_specific_clearance():
    cmd = jnp.array([[.8, 0., 0.], [0., .5, 0.], [0., 0., .8], [1.5, 0., 0.]])
    both = jnp.ones((4, 2), bool)
    ground = jnp.zeros((4, 2))
    left_up = both.at[:, 0].set(False)
    apex = ground.at[:, 0].set(.02)
    advance = jax.jit(lambda h, c, z: h.advance(c, z, cmd))
    h = rewards.StepHistory.empty(4)
    for _ in range(25):
        h, _, _ = advance(h, left_up, apex)
    h, landing, _ = advance(h, both, ground)
    np.testing.assert_array_equal(landing, 0.)  # midair reset
    for _ in range(25):
        h, landing, _ = advance(h, left_up, apex)
        np.testing.assert_array_equal(landing, 0.)
    h, landing, clearance = advance(h, both, ground)
    np.testing.assert_allclose(landing, [1., 1., 1., 0.])
    np.testing.assert_allclose(clearance, [.5**2, .2**2, (1.-.02/.035)**2, 0.], rtol=1e-5)
    h, landing, clearance = advance(h, both, ground)
    np.testing.assert_array_equal(landing + clearance, 0.)
    h = h.reset(jnp.array([True, False, False, True]))
    np.testing.assert_array_equal(h.seen[:, 0], [False, True, True, False])


def test_joint_costs_and_foot_yaw_crossing_and_recovery_masks():
    q = jnp.zeros((2, 23)).at[:, rewards.ANKLE_ROLL].set(np.deg2rad(15))
    v = jnp.ones_like(q)
    R = jnp.broadcast_to(jnp.eye(3), (2, 3, 3))
    state = (None, R, None, None, q, v)
    yaw = np.deg2rad(12.)
    foot = jnp.array([[np.cos(yaw), -np.sin(yaw), 0.], [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
    feet = rewards.FootGeometry(jnp.zeros((2, 2, 3)).at[:, 0, 1].set(-.1),
                                 jnp.broadcast_to(foot, (2, 2, 3, 3)), jnp.zeros((2, 4)), jnp.zeros(2), jnp.zeros((2, 2)))
    limits = jnp.broadcast_to(jnp.array([-1., 1.]), (23, 2)).at[rewards.ANKLE_ROLL, 1].set(.1)
    _, terms = rewards.reward_terms(state, feet, jnp.array([[.75, 0., 0.], [0., 0., 0.]]), 2*v, 3*v, limits,
                                     jnp.ones((2, 2), bool), jnp.array([2, 2]), jnp.array([.3, .5]), jnp.ones((2, 2)),
                                     rewards.StepHistory.empty(2), jnp.array([True, False]))
    assert set(terms) == set(rewards.WEIGHTS)
    np.testing.assert_allclose(terms["foot_slip"], [.3, .5])  # includes zero-command recovery
    np.testing.assert_allclose(terms["sound_suppression"], [2., 2.])
    for key, val in [('joint_vel_l2', 23.), ('joint_torques_l2', 92.), ('joint_acc_l2', 207.),
                     ('joint_energy', 46.), ('joint_pos_limits', 2*(np.deg2rad(15.)-.1))]:
        np.testing.assert_allclose(terms[key], [val, val], rtol=1e-5)
    for key, val in [('feet_crossing', .2), ('ankle_roll_l2', 2.),
                     ('feet_yaw_l2', 2*np.deg2rad(10.)**2), ('undesired_contacts', 2.)]:
        np.testing.assert_allclose(terms[key], [val, 0.], rtol=1e-5)


def test_global_gait_crossing_penalizes_spacing_below_15_cm(zenbot23_motion_banks):
    spacing = jnp.array([-.02, .12, .15, .18, .12])
    n = len(spacing)
    sim = Sim(n, robot=zenbot.MIXED_TASK, physics=False, zenbot_motion_train_wm=True,
              wm_floor_amp=True, wm_global_gait=True, wm_gait_rewards=True)
    qpos = jnp.broadcast_to(jnp.r_[jnp.array([0., 0., .68, 1., 0., 0., 0.]), zenbot.DEFAULT_Q], (n, 30))
    state = unflatten(qpos, jnp.zeros((n, 29)))
    feet = sim.view_state(state).feet
    positions = jnp.zeros_like(feet.positions).at[:, 0, 1].set(spacing)
    zeros = jnp.zeros(n)
    _, terms = rewards.reward_terms(
        state, feet._replace(positions=positions), jnp.zeros((n, 3)), state[5], state[5], sim.q_soft,
        jnp.ones((n, 2), bool), zeros, zeros, jnp.zeros((n, 2)), rewards.StepHistory.empty(n),
        jnp.array([True, True, True, True, False]), min_ankle_spacing=sim.min_ankle_spacing)
    np.testing.assert_allclose(sim.weights["feet_crossing"] * terms["feet_crossing"],
                               [-1.7, -.3, 0., 0., 0.], atol=1e-6)


def test_sound_suppression_gates_each_foot_and_caps_before_summing():
    n = 4
    q = jnp.zeros((n, 23))
    rot = jnp.broadcast_to(jnp.eye(3), (n, 3, 3))
    state = (None, rot, None, None, q, q)
    feet = rewards.FootGeometry(jnp.zeros((n, 2, 3)), jnp.stack([rot, rot], axis=1),
                                jnp.zeros((n, 2)), jnp.zeros(n), jnp.zeros((n, 2)))
    cmd = jnp.array([[1., 0., 0.], [0., 1.49, 2.], [1.2, .9, 0.], [0., 0., 2.]])
    contact = jnp.array([[True, False], [True, True], [True, True], [False, False]])
    def cost(acc):
        _, terms = rewards.reward_terms(state, feet, cmd, q, q, jnp.zeros((23, 2)), contact, jnp.zeros(n),
                                        jnp.zeros(n), acc, rewards.StepHistory.empty(n), jnp.ones(n, bool))
        return terms['sound_suppression']
    acc = jnp.tile(jnp.array([-3., 9.]), (n, 1))
    np.testing.assert_array_equal(jax.jit(cost)(acc), [9., 59., 0., 0.])
    np.testing.assert_array_equal(jax.grad(lambda a: cost(a).sum())(acc), [[-6., 0.], [-6., 0.], [0., 0.], [0., 0.]])


def test_foot_vertical_acceleration_matches_mujoco_com_jacobians_at_both_ends_of_substep(zenbot23_motion_banks, monkeypatch):
    # The kinematics check starts from a prescribed pose and needs no directional reset clips.
    monkeypatch.setattr("instantamp.zenbot_gait_demos.command_reset_bank", lambda *args, **kwargs: {})
    sim = Sim(1, robot=zenbot.MIXED_TASK, physics=False, zenbot_motion_train_wm=True,
              wm_floor_amp=True, wm_gait_rewards=True)
    m, ids = sim.m, sim.gait_bodies[:2]
    d = mujoco.MjData(m)
    d.qpos[2] = .68
    d.qpos[7:] = zenbot.DEFAULT_Q
    d.qvel[:] = np.linspace(-1., 1.3, m.nv)

    def native_vz():
        mujoco.mj_forward(m, d)
        jac = np.zeros((3, m.nv))
        result = []
        for body in ids:
            mujoco.mj_jacBodyCom(m, d, jac, None, body)
            result.append(jac[2] @ d.qvel)
        return np.asarray(result)

    before = native_vz()
    mujoco.mj_step(m, d)
    geometry = sim.view_state(unflatten(jnp.asarray(d.qpos), jnp.asarray(d.qvel))).feet
    data = SimpleNamespace(**{k: jnp.asarray(getattr(d, k))[None] for k in ('cvel', 'xipos', 'subtree_com')})
    accel = rewards.measured_foot_acc_z(data, ids, geometry.com_velocity_z[None])
    after = native_vz()
    np.testing.assert_allclose(geometry.com_velocity_z, after, atol=2e-6)
    np.testing.assert_allclose(accel[0], (after - before)/env.SIM_DT, atol=3e-4)


def test_terrain_contacts_sum_load_per_body_and_exclude_self_contacts():
    # Two small floor loads add on body 3. The self contact between 2 and 4 is ignored.
    pairs = jnp.array([[0, 1], [0, 3], [0, 3], [4, 2], [0, 4], [0, 4]])
    impl = SimpleNamespace(
        contact__worldid=jnp.zeros(6, int), contact__geom=pairs, nacon=jnp.array([5]),
        contact__efc_address=jnp.full((6, 4), -1).at[:, 0].set(jnp.arange(6)),
        contact__pos=jnp.zeros((6, 3)),
        efc__force=jnp.array([[1.2, .6, .6, 3., 1., 4.]]))
    data = SimpleNamespace(_impl=impl, cdof=jnp.ones((1, 1, 6)), qvel=jnp.ones((1, 1)), subtree_com=jnp.zeros((1, 5, 3)))
    feet, other, slip = rewards.measured_contacts(data, jnp.arange(5), np.array([1, 2]), np.array([3, 4]),
                                                  jnp.ones((2, 1)), jnp.array([400.]))
    np.testing.assert_array_equal(feet, [[True, False]])
    np.testing.assert_array_equal(other, [[True, False]])
    np.testing.assert_array_equal(slip, 0.)  # light brushes have no slip weight


def test_loaded_slip_uses_material_point_velocities_and_normal_load():
    # The COM is 0.2 m above the ground. With wy=1, vx=0.2 the foot rolls without sliding.
    # Add yaw: points at x=0 and x=0.2 then have tangent speeds 0 and 0.2 m/s.
    force = jnp.array([[0., 0.], [2.5, 7.5], [5., 15.]])
    impl = SimpleNamespace(
        contact__worldid=jnp.repeat(jnp.arange(3), 2), nacon=jnp.array([6]),
        contact__geom=jnp.tile(jnp.array([[0, 1], [1, 0]]), (3, 1)),
        contact__pos=jnp.tile(jnp.array([[0., 0., 0.], [.2, 0., 0.]]), (3, 1)),
        contact__efc_address=jnp.tile(jnp.array([[0], [1]]), (3, 1)), efc__force=force)
    twists = jnp.broadcast_to(jnp.eye(6), (3, 6, 6))
    def slip(yaw):
        speed = jnp.broadcast_to(jnp.array([0., 1., yaw, .2, 0., 0.]), (3, 6))
        data = SimpleNamespace(_impl=impl, cdof=twists, qvel=speed, subtree_com=jnp.zeros((3, 3, 3)).at[:, 1, 2].set(.2))
        return rewards.measured_contacts(data, jnp.arange(3), jnp.array([1, 2]), jnp.array([], int),
                                         jnp.ones((2, 6)), jnp.full(3, 100.))[-1]
    np.testing.assert_allclose(slip(0.), 0., atol=1e-7)
    np.testing.assert_allclose(slip(1.), [0., .375, .75], atol=1e-6)


def test_predicted_slip_ignores_airborne_feet_and_rolling_and_has_finite_rest_gradient():
    corners = jnp.array([[[.1, 0., -.02], [-.1, 0., -.02]]]*2)
    rotation = jnp.broadcast_to(jnp.eye(3), (1, 2, 3, 3))
    pos = jnp.array([[[0., 0., .02], [0., 0., .40]]])
    angular = jnp.array([[[0., 1., 0.], [0., 0., 0.]]])
    def cost(vx):
        velocity = jnp.array([[[vx, 0., 0.], [3., 1., 0.]]])
        return rewards.predicted_foot_slip(pos, rotation, velocity, angular, corners).sum()
    assert float(cost(.02)) == 0.
    assert float(jax.grad(cost)(.02)) == 0.
    np.testing.assert_allclose(cost(.22), 1., atol=1e-6)
    np.testing.assert_allclose(cost(.42), 3., atol=1e-6)  # linear tail


def test_posture_rewards_gate_on_speed_and_supported_stance():
    cmd = jnp.array([[0., 0., 0.], [.5, 0., 0.], [0., 0., .5], [0., 0., 1.5],
                     [0., 0., 0.], [0., 0., 0.]])
    n = len(cmd)
    pos = jnp.zeros((n, 3)).at[:, 2].set(.55)
    rotation = jnp.broadcast_to(jnp.eye(3), (n, 3, 3))
    velocity = jnp.zeros_like(pos).at[4, 1].set(1.5)
    gyro = jnp.zeros_like(pos)
    ready = jnp.arange(n) < n-1
    def evaluate(vel, rot):
        state = (pos, rot, vel, gyro)
        return rewards.posture_terms(state, cmd, ready, rewards.base_velocity(state, jnp.zeros(3)))
    terms = evaluate(velocity, rotation)
    assert set(terms) == set(rewards.POSTURE_WEIGHTS)
    np.testing.assert_allclose(terms['slow_speed_base_height_l2'], [1, 1/1.4, 1/1.4, 0, 0, 0], atol=1e-6)
    for term in terms.values():
        assert term[-1] == 0.
    tilted = rotation.at[0].set(jnp.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]))
    tilt = evaluate(velocity, tilted)
    assert tilt['flat_orientation_l2'][0] == 1.
    grads = jax.grad(lambda vel: sum(t.sum() for t in evaluate(vel, rotation).values()))(velocity)
    assert jnp.isfinite(grads).all()
