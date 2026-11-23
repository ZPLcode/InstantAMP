from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from instantamp import zenbot
from instantamp.sim import Sim
from instantamp.train_g1_getup_amp import GetUpAMP
from instantamp.zenbot_motion_train_obs import LAB_ORDER, advance_sensors, interpolate_delayed_imu, map_imu_delay


def test_imu_delay_matches_two_uniform_ranges_and_normalized_linear_interpolation():
    draws = jnp.array([[0., 0.], [.899, 1.], [.9, 0.], [.99, 1.]])
    np.testing.assert_allclose(map_imu_delay(draws), [.002, .010, .010, .035], atol=1e-8)
    last = np.array([2., 3., 4., 0., 1., 0.])
    now = np.array([4., 7., 8., 0., 0., -1.])
    older = np.array([0., 1., 2., 1., 0., 0.])
    delays = np.array([.002, .010, .020, .035])
    mixed = np.stack([.9 * now + .1 * last, (now + last) / 2, last, .25 * last + .75 * older])
    mixed[:, 3:] /= np.linalg.norm(mixed[:, 3:], axis=1, keepdims=True)
    args = [jnp.tile(x, (len(delays), 1)) for x in (now, last, older)]
    np.testing.assert_allclose(interpolate_delayed_imu(*args, jnp.asarray(delays)), mixed, atol=1e-6)


def test_actor_uses_corrupted_frames_once_and_refills_only_the_reset_world():
    count = 2
    command = jnp.array([[0., .4, 0.], [0., 0., -.6]])
    q = jnp.tile(zenbot.DEFAULT_Q, (count, 1))
    qd = jnp.tile(jnp.arange(23.), (count, 1))
    action = qd + 10.
    imu = jnp.tile(jnp.array([.1, .2, .3, 0., 0., -1.]), (count, 1))
    first = advance_sensors(None, imu, command, q, qd, action, jnp.ones(count, bool), jax.random.key(1))
    second = advance_sensors(first, imu, command, q, qd, action, jnp.zeros(count, bool), jax.random.key(2))
    third = advance_sensors(second, imu, command, q, qd, action, jnp.array([False, True]), jax.random.key(3))
    np.testing.assert_array_equal(first.frames[:, 0], first.frames[:, 2])
    np.testing.assert_array_equal(third.frames[1, 0], third.frames[1, 2])
    np.testing.assert_array_equal(third.frames[0, :-1], second.frames[0, 1:])
    assert second.delay_s[0] == first.delay_s[0] == third.delay_s[0]
    assert second.delay_s[1] != third.delay_s[1]
    frame = third.frames[:, -1]
    for actual, clean, bound in ((frame[:, :3], imu[:, :3], .28), (frame[:, 3:6], imu[:, 3:], .042),
                                 (frame[:, 9:32], 0., .03), (frame[:, 32:55], qd, 1.5)):
        residual = np.abs(actual - clean)
        assert (residual <= bound + 1e-6).all() and residual.max() > 0.
    np.testing.assert_array_equal(frame[:, 6:9], command)
    np.testing.assert_array_equal(frame[:, 55:], action)
    expected = np.concatenate([third.frames[..., sl].reshape(count, -1) for sl in
                               (slice(0, 3), slice(3, 6), slice(6, 9), 9 + LAB_ORDER, 32 + LAB_ORDER, 55 + LAB_ORDER)],
                              axis=-1)
    # Returning this observation to PPO must preserve the exact noise sampled during collection.
    state = SimpleNamespace(motion_train_sensors=third, obs=jnp.full((count, 495), 999.))
    np.testing.assert_array_equal(GetUpAMP.__new__(GetUpAMP).actor_input(state), expected)


def test_initial_imu_samples_can_be_donated_to_the_collector():
    imu = jnp.ones((2, 6))
    q = jnp.tile(zenbot.DEFAULT_Q, (2, 1))
    sensors = advance_sensors(None, imu, jnp.zeros((2, 3)), q, q, q, jnp.ones(2, bool),
                              jax.random.key(7))
    step = jax.jit(lambda now, last: (now + 1., last - 1.), donate_argnums=(0, 1))
    now, last = step(sensors.current_imu, sensors.previous_imu)
    np.testing.assert_array_equal(now, np.full((2, 6), 2.))
    np.testing.assert_array_equal(last, np.zeros((2, 6)))


def test_sim_reads_body_gyro_and_world_down_from_raw_physics():
    sim = Sim.__new__(Sim)
    sim.noisy_obs = False
    rotation = jnp.array([[[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]])
    qvel = jnp.zeros((1, 29)).at[0, 3:6].set(jnp.array([1., 2., 3.]))
    qpos = jnp.zeros((1, 30)).at[0, 7:].set(zenbot.DEFAULT_Q)
    state = SimpleNamespace(data=SimpleNamespace(qpos=qpos, qvel=qvel), a_last=jnp.zeros((1, 23)),
                            motion_train_sensors=None)
    result = sim.motion_train_sensors(state, rotation, jnp.zeros((1, 3)), jnp.array([True]), jax.random.key(4))
    np.testing.assert_array_equal(result.frames[0, -1, :6], [1., 2., 3., 1., 0., 0.])


def test_directional_training_uses_the_motion_train_policy_group_by_default():
    from instantamp.train_zenbot_getup_stand_amp import Settings

    assert Settings(directional_gait=1).motion_train_obs == 1
    assert Settings().motion_train_obs == 0


def test_motion_train_ppo_mean_anchor_and_gradients_match_float32():
    from instantamp.train_zenbot_getup_stand_amp import RiseStandAMP, Settings

    rng = np.random.default_rng(413)
    x = rng.normal(size=(8, 234)).astype(np.float32)
    first = rng.normal(scale=.05, size=(234, 37)).astype(np.float32)
    last = rng.normal(scale=.1, size=(37, 23)).astype(np.float32)
    action_grad = rng.normal(size=(8, 23)).astype(np.float32)
    pre = x @ first
    hidden = np.where(pre > 0., pre, np.expm1(pre))
    hidden_grad = (action_grad @ last.T) * np.where(pre > 0., 1., np.exp(pre))

    trainer = RiseStandAMP.__new__(RiseStandAMP)
    trainer.c = Settings(directional_gait=1)
    trainer.obs_norm = {"raw": True}
    trainer.anchor_pi = [{"w": jnp.asarray(w), "b": jnp.zeros(w.shape[1])} for w in (first, last)]
    with jax.default_matmul_precision("bfloat16"):  # global PPO default
        mean = jax.jit(trainer.actor_mean)(trainer.anchor_pi, x)
        anchor, _ = trainer.action_anchor_examples(None, jnp.ones(8), actor_obs=x)
        def loss(params, obs):
            return jnp.sum(trainer.actor_mean(params, obs) * action_grad)
        parameter_grad, input_grad = jax.jit(jax.grad(loss, argnums=(0, 1)))(trainer.anchor_pi, x)
    for actual, expected in ((mean, hidden @ last), (anchor, hidden @ last),
                             (parameter_grad[0]["w"], x.T @ hidden_grad),
                             (parameter_grad[1]["w"], hidden.T @ action_grad),
                             (input_grad, hidden_grad @ first.T)):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=3e-6)


def test_sagittal_mirror_follows_motion_train_channels_and_mjcf_actions():
    from instantamp.zenbot_motion_train_obs import mirror_action, mirror_observation

    rng = np.random.default_rng(83041)
    values = rng.normal(size=(3, 234)).astype(np.float32)
    actions = rng.normal(size=(3, 23)).astype(np.float32)
    # The saved Motion Train contract puts the waist between the hip pitches
    # and hip rolls. Controls sent to MuJoCo use the physical hinge order.
    left = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    right = [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    signs = np.array([1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1])

    def expected_lab(joints):
        result = joints.copy()
        result[..., left] = joints[..., right] * signs
        result[..., right] = joints[..., left] * signs
        result[..., 2] = -joints[..., 2]
        return result

    expected = values.copy()
    vector = expected[:, :27].reshape(-1, 3, 3, 3)
    vector *= np.array([[-1, 1, -1], [1, -1, 1], [1, -1, -1]])[:, None, :]
    expected[:, 27:] = expected_lab(values[:, 27:].reshape(-1, 3, 3, 23)).reshape(-1, 207)
    np.testing.assert_array_equal(mirror_observation(jnp.asarray(values)), expected)
    np.testing.assert_array_equal(mirror_observation(mirror_observation(jnp.asarray(values))), values)
    np.testing.assert_array_equal(np.asarray(mirror_action(jnp.asarray(actions)))[:, LAB_ORDER], expected_lab(actions[:, LAB_ORDER]))
