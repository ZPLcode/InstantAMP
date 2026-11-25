"""Physical-force distribution, timing, reset and frame conventions."""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from instantamp.zenbot_push import CLEAN, HIP, NORMAL, SHOULDER, STRONG, TimedPush


def scheduler():
    return TimedPush(8, tuple(range(1, 8)))


def test_episode_profiles_and_pulse_distributions():
    push = scheduler()
    state = jax.jit(lambda key: push.init(key, 65536))(jax.random.key(302))
    profile = np.asarray(state.profile)
    shares = np.bincount(profile, minlength=3) / len(profile)
    np.testing.assert_allclose(shares, [.70, .25, .05], atol=.006)
    clean = profile == CLEAN
    assert np.isinf(np.asarray(state.wait)[clean]).all()
    wait = np.asarray(state.wait)[~clean]
    assert wait.min() >= 5. and wait.max() < 10.
    # Make every disturbed world due in the next callback.
    state = state._replace(wait=jnp.where(state.profile == CLEAN, jnp.inf, .01))
    started = jax.jit(push.advance)(state, jax.random.key(11))
    np.testing.assert_array_equal(np.asarray(started.count), ~clean)
    np.testing.assert_array_equal(np.asarray(started.force_w)[clean], 0.)
    force = np.asarray(started.force_w)
    norm = np.linalg.norm(force, axis=-1)
    normal = profile == NORMAL
    shoulder = np.asarray(started.kind) == SHOULDER
    strong = profile == STRONG
    assert abs(shoulder[normal].mean() - .5) < .015
    assert shoulder[strong].all()
    for select, bounds in ((normal & shoulder, (50., 150.)),
                           (normal & ~shoulder, (30., 100.)), (strong, (150., 400.))):
        assert norm[select].min() >= bounds[0] - 1e-4
        assert norm[select].max() < bounds[1] + 1e-4
        assert abs(norm[select].mean() - np.mean(bounds)) < .025 * (bounds[1] - bounds[0])
    elevation = np.arcsin(force[~clean, 2] / norm[~clean])
    assert np.abs(elevation).max() <= np.deg2rad(10.) + 1e-6
    # Uniform azimuth should not select a privileged push direction.
    assert np.abs((force[~clean, :2] / norm[~clean, None]).mean(0)).max() < .015
    duration = np.asarray(started.remaining)
    assert duration[normal].min() >= .10 and duration[normal].max() < .20
    assert duration[strong].min() >= .08 and duration[strong].max() < .12
    hip_ids = np.asarray(started.body)[normal & ~shoulder]
    np.testing.assert_allclose(np.bincount(hip_ids - 2, minlength=6) / len(hip_ids), 1. / 6., atol=.012)


def test_pulse_duration_world_direction_and_start_to_start_timer():
    push = scheduler()
    state = push.init(jax.random.key(0), 2)._replace(
        profile=jnp.array([NORMAL, CLEAN]), wait=jnp.array([.02, jnp.inf]),
    )
    state = push.advance(state, jax.random.key(1))
    assert state.count[0] == 1 and state.count[1] == 0
    duration, interval = float(state.remaining[0]), float(state.wait[0])
    force = np.asarray(state.force_w[0]).copy()
    steps = int(np.ceil(duration / .02))
    for index in range(1, steps):
        state = push.advance(state, jax.random.key(index + 2))
        np.testing.assert_array_equal(state.force_w[0], force)
    state = push.advance(state, jax.random.key(20))
    assert state.remaining[0] == 0 and state.kind[0] == -1
    np.testing.assert_array_equal(state.force_w, 0.)
    np.testing.assert_allclose(state.wait[0], interval - steps * .02, atol=1e-5)
    # The timer counts through the pulse instead of adding its duration.
    state = state._replace(wait=jnp.array([.02, jnp.inf]))
    state = push.advance(state, jax.random.key(21))
    np.testing.assert_array_equal(state.count, [2, 0])


def test_partial_reset_clears_only_selected_worlds_and_resamples_membership():
    push = scheduler()
    state = push.init(jax.random.key(0), 4)._replace(
        profile=jnp.full(4, NORMAL), wait=jnp.zeros(4),
    )
    state = push.advance(state, jax.random.key(1))
    mask = jnp.array([True, False, True, False])
    reset = jax.jit(push.reset)(state, mask, jax.random.key(23))
    for old, new in zip(state, reset):
        np.testing.assert_array_equal(np.asarray(new)[~mask], np.asarray(old)[~mask])
    np.testing.assert_array_equal(np.asarray(reset.force_w)[mask], 0.)
    np.testing.assert_array_equal(np.asarray(reset.remaining)[mask], 0.)
    np.testing.assert_array_equal(np.asarray(reset.count)[mask], 0.)
    assert np.all(np.asarray(reset.wait)[mask] >= 5.)


def test_wrench_targets_correct_body_and_retains_world_force_when_body_rotates():
    push = scheduler()
    state = push.init(jax.random.key(0), 3)._replace(
        profile=jnp.array([NORMAL, NORMAL, CLEAN]), remaining=jnp.array([.1, .1, 0.]),
        kind=jnp.array([SHOULDER, HIP, -1]), body=jnp.array([1, 4, -1]),
        force_w=jnp.array([[100., 0., 0.], [0., 50., 0.], [0., 0., 0.]]),
        point=jnp.array([0, 0, 0]),
    )
    rotations = jnp.tile(jnp.eye(3), (3, 8, 1, 1))
    wrench = jax.jit(push.wrench)(state, rotations)
    np.testing.assert_allclose(wrench[0, 1], [100., 0., 0., 0., 29.817, -13.787], atol=1e-5)
    np.testing.assert_allclose(wrench[1, 4], [0., 50., 0., 0., 0., 0.])
    assert np.count_nonzero(np.linalg.norm(wrench, axis=-1)) == 2
    # A 90-degree torso yaw moves the left shoulder's lever arm but keeps F_w.
    rotations = rotations.at[0, 1].set(jnp.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]))
    rotated = push.wrench(state, rotations.reshape(3, 8, 9))
    np.testing.assert_allclose(rotated[0, 1], [100., 0., 0., 0., 29.817, -.8], atol=1e-5)


def test_world_wrench_order_has_expected_free_body_acceleration():
    # Exercise MuJoCo's force/torque convention rather than only array layout.
    model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
      <worldbody><body name="body"><freejoint/>
      <inertial pos="0 0 0" mass="2" diaginertia=".2 .3 .4"/>
      <geom type="sphere" size=".1" contype="0" conaffinity="0"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    push = TimedPush(2, (1, 1, 1, 1, 1, 1, 1))
    state = push.init(jax.random.key(0), 1)._replace(
        profile=jnp.array([NORMAL]), remaining=jnp.array([.1]), kind=jnp.array([SHOULDER]),
        body=jnp.array([1]), point=jnp.array([0]), force_w=jnp.array([[100., 0., 0.]]),
    )
    data.xfrc_applied[:] = np.asarray(push.wrench(state, jnp.asarray(data.xmat)[None]))[0]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qacc, [50., 0., 0., 0., 29.817 / .3, -13.787 / .4], rtol=1e-6)
