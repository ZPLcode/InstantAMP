from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from instantamp import amp, env, zenbot
from instantamp.sim import Sim
from instantamp.train_zenbot_getup_stand_amp import RiseStandAMP


def test_amp_weight_decreases_as_supported_standing_quality_increases(monkeypatch):
    z = jnp.array([.25, .69, .69])
    pos = jnp.zeros((3, 3)).at[:, 2].set(z)
    rotation = jnp.tile(jnp.eye(3), (3, 1, 1))
    zero = jnp.zeros_like(pos)
    q = jnp.tile(zenbot.DEFAULT_Q, (3, 1))
    state = (pos, rotation, zero, zero, q, jnp.zeros_like(q))
    standing, pose = zenbot.quiet_stand(state, jnp.array([450., 450., 0.]), jnp.full(3, 422.))
    np.testing.assert_array_equal(standing, [False, True, False])
    np.testing.assert_allclose(pose, [0., 1., 0.])
    _, moving_pose = zenbot.quiet_stand((*state[:-1], jnp.ones_like(q)), jnp.full(3, 450.), jnp.full(3, 422.))
    assert 0. < moving_pose[1] < pose[1]

    trainer = RiseStandAMP.__new__(RiseStandAMP)
    trainer.c = SimpleNamespace(default_stand=0)
    monkeypatch.setattr(amp, "recovery_style", lambda *args, **kwargs: jnp.full(3, 10.))
    amp_max = env.DT * 10.
    stand_max = env.DT * 15.
    style = trainer.motion_reward(None, None, SimpleNamespace(stats={"stand_quality": pose}))
    np.testing.assert_allclose(style, [amp_max, 0., amp_max])
    # The independent duration bonus must leave the instantaneous AMP handoff unchanged.
    transition_style = trainer.motion_reward(None, None, SimpleNamespace(
        stats={"stand_quality": moving_pose, "support_hold_score": jnp.array(1.)}))
    np.testing.assert_allclose(transition_style, amp_max * (1. - moving_pose), atol=1e-7)
    assert 0. < transition_style[1] < amp_max
    # Crossing into a supported, imperfect stand preserves a return at least
    # as good as the best AMP score, even before the robot has settled.
    assert amp_max < transition_style[1] + stand_max * moving_pose[1] < stand_max


def test_supported_hold_ramps_for_one_second_and_resets_per_world():
    memory = zenbot.SupportedStand.empty(2, env.DT)
    upright = zenbot.standing_posture(jnp.array([.69, .69]), jnp.ones(2))
    loaded = jnp.ones(2, dtype=bool)

    def step(state, _):
        return state.advance(upright, loaded)

    held, rewards = jax.lax.scan(step, memory, None, length=50)
    np.testing.assert_allclose(rewards[:, 0], np.arange(1, 51) * .02, atol=1e-6)
    np.testing.assert_array_equal(rewards[-1], [1., 1.])

    reset = held.reset(jnp.array([True, False]))
    np.testing.assert_array_equal(reset.upright_steps, [0, 50])
    np.testing.assert_array_equal(reset.loaded[0], False)
    np.testing.assert_array_equal(reset.loaded[1], True)
    _, scores = reset.advance(upright, loaded)
    np.testing.assert_allclose(scores, [.02, 1.])

    posture = zenbot.standing_posture(jnp.array([.25, .69]), jnp.ones(2))
    after_fall, scores = held.advance(posture, loaded)
    np.testing.assert_array_equal(scores, [0., 1.])
    _, scores = after_fall.advance(upright, loaded)
    np.testing.assert_allclose(scores, [.02, 1.])


def test_supported_hold_tolerates_five_unloaded_frames_in_a_full_window():
    upright = jnp.ones(1, dtype=bool)
    contact = jnp.arange(57)[:, None] < 50
    contact = contact.at[-1, 0].set(True)

    def step(state, loaded):
        return state.advance(upright, loaded)

    _, scores = jax.lax.scan(step, zenbot.SupportedStand.empty(1, env.DT), contact)
    np.testing.assert_array_equal(scores[49:55, 0], 1.)
    assert scores[55, 0] == 0.
    np.testing.assert_allclose(scores[56, 0], 44. / 45.)


def test_default_stand_scores_upright_pose_and_stillness_after_full_support():
    # Each row changes just one feature of an otherwise quiet, fully supported stand.
    n = 7
    pos = jnp.zeros((n, 3)).at[:, 2].set(zenbot.DEFAULT_HEIGHT)
    rotation = jnp.tile(jnp.eye(3), (n, 1, 1))
    angle = np.deg2rad(10.)
    rotation = rotation.at[1].set(jnp.array([[np.cos(angle), 0., np.sin(angle)],
                                             [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]]))
    q = jnp.tile(zenbot.DEFAULT_Q, (n, 1))
    q = q.at[2, zenbot.DOF.index("left_elbow")].add(np.deg2rad(45.))
    q = q.at[3, zenbot.DOF.index("right_hip_pitch")].add(np.deg2rad(-30.))
    linear = jnp.zeros_like(pos).at[4, 0].set(.2)
    angular = jnp.zeros_like(pos).at[5, 1].set(.5)
    speeds = jnp.zeros_like(q).at[6, zenbot.DOF.index("right_elbow")].set(1.)
    state = (pos, rotation, linear, angular, q, speeds)
    pose, hold = zenbot.default_stand_terms(state, jnp.ones(n))
    np.testing.assert_allclose([pose[0], hold[0]], 1., atol=1e-6)
    total = 15. * pose + 8. * hold
    assert np.all(np.asarray(total[1:]) < total[0])
    # Even when the feet remain loaded, sustained lean or one mispositioned
    # joint cannot earn full marks on the duration component.
    assert np.all(np.asarray(hold[1:4]) < hold[0])


def test_default_stand_crossfade_depends_on_each_worlds_time_on_its_feet(monkeypatch):
    # A low hand-supported pose and a pitched, mid-rise pose without foot
    # load must remain free to follow AMP. Other worlds have begun standing.
    height = jnp.array([.28, .54, .69, .69, .69])
    n = len(height)
    pos = jnp.zeros((n, 3)).at[:, 2].set(height)
    angle = jnp.deg2rad(jnp.array([75., 40., 0., 0., 0.]))
    c, s = jnp.cos(angle), jnp.sin(angle)
    rotation = jnp.tile(jnp.eye(3), (n, 1, 1))
    rotation = rotation.at[:, 0, 0].set(c).at[:, 0, 2].set(s)
    rotation = rotation.at[:, 2, 0].set(-s).at[:, 2, 2].set(c)
    q = jnp.tile(zenbot.DEFAULT_Q, (n, 1)).at[:2, zenbot.DOF.index("left_elbow")].add(.8)
    zero = jnp.zeros_like(pos)
    state = (pos, rotation, zero, zero, q, jnp.zeros_like(q))
    start = jnp.array([50, 50, 49, 25, 0])
    upright = zenbot.standing_posture(height, c)

    def step(memory, t):
        standing = upright & (t >= start)
        return memory.advance(standing, standing)

    _, scores = jax.lax.scan(step, zenbot.SupportedStand.empty(n, env.DT), jnp.arange(50))
    blend = scores[-1]
    np.testing.assert_allclose(blend, [0., 0., .02, .5, 1.], atol=1e-6)
    pose, hold = zenbot.default_stand_terms(state, blend)
    np.testing.assert_allclose(pose, blend, atol=1e-6)
    np.testing.assert_allclose(hold, blend, atol=1e-6)

    trainer = RiseStandAMP.__new__(RiseStandAMP)
    trainer.c = SimpleNamespace(default_stand=1)
    monkeypatch.setattr(amp, "recovery_style", lambda *args, **kwargs: jnp.full(n, 10.))
    transition = SimpleNamespace(stats={"stand_blend": blend, "stand_quality": jnp.array([0., 0., 1., 1., 1.])})
    style = trainer.motion_reward(None, None, transition)
    np.testing.assert_allclose(style / (10. * env.DT), 1. - blend, atol=1e-6)


def test_standing_amp_uses_support_reward_and_keeps_pd_target_limits_independent(tmp_path):
    q = np.tile(zenbot.DEFAULT_Q, (32, 1))
    pos = np.zeros((len(q), 3), np.float32)
    np.savez(tmp_path / "rise.npz", fps=50., root_pos=pos, root_quat=np.tile([0, 0, 0, 1], (len(q), 1)),
             root_lin_vel=pos, root_ang_vel=pos, dof_pos=q, dof_vel=np.zeros_like(q))
    options = dict(robot="zenbot23_getup", floor_dir=str(tmp_path), physics=False,
                   zenbot_getup_stand=True, zenbot_support_weight=8.)
    prior = Sim(1, **options, zenbot_default_stand=True)
    standing_amp = Sim(1, **options, zenbot_standing_amp=True, clip_pd_targets=True)
    assert prior.weights["stand_pose"] == 15.
    assert "stand_pose" not in standing_amp.weights
    assert "target_joint_deviation_l2" not in standing_amp.weights
    assert prior.weights["supported_hold"] == standing_amp.weights["supported_hold"] == 8.

    action = jnp.full((1, len(zenbot.DOF)), 100.)
    history = jnp.zeros((1, prior.obs_dim))
    context = jnp.zeros((1, 8))
    old = prior.control_targets(action, jnp.zeros_like(action), history, context).reshape(-1, prior.J)
    limited = standing_amp.control_targets(action, jnp.zeros_like(action), history, context).reshape(-1, prior.J)
    assert np.all(old > prior.joint_range[:, 1])
    np.testing.assert_allclose(limited, np.broadcast_to(standing_amp.joint_range[:, 1], limited.shape))


def test_single_discriminator_rewards_both_behaviors_with_supported_stance_blend(monkeypatch):
    trainer = RiseStandAMP.__new__(RiseStandAMP)
    trainer.condition_dim = 1
    trainer.c = SimpleNamespace(standing_amp=1)
    discriminator = object()
    hold = jnp.array([0., .25, .5, .75, 1.])
    motion = jnp.arange(5 * 10 * 3, dtype=jnp.float32).reshape(5, 10, 3)
    transition = SimpleNamespace(stats={"stand_blend": hold})

    def score(disc, conditioned, scale, leggedlab):
        assert disc is discriminator and scale == 10. and leggedlab
        condition = conditioned[..., -1]
        np.testing.assert_array_equal(condition, jnp.broadcast_to(condition[:, :1], condition.shape))
        return jnp.where(condition[:, 0] == 0., 2., 7.)

    monkeypatch.setattr(amp, "recovery_style", score)
    reward, replay, _ = trainer.rollout_motion(discriminator, motion, transition)
    np.testing.assert_allclose(reward / env.DT, [2., 3.25, 4.5, 5.75, 7.], rtol=1e-6)
    np.testing.assert_allclose(trainer.motion_reward(discriminator, motion, transition), reward)
    np.testing.assert_array_equal(replay[..., :-1], motion)
    np.testing.assert_array_equal(replay[..., -1], jnp.broadcast_to(hold[:, None], motion.shape[:2]))


def test_conditional_discriminator_independently_samples_expert_windows_in_policy_groups():
    class ExpertWindows:
        def __init__(self, offset):
            self.offset = offset

        def sample(self, key, count):
            start = self.offset + jax.random.randint(key, (count,), 0, 1000)
            window = start[:, None] + jnp.arange(28)
            return jnp.broadcast_to(window[..., None], (count, 28, 15)).astype(jnp.float32)

    trainer = RiseStandAMP.__new__(RiseStandAMP)
    trainer.c = SimpleNamespace(standing_amp=1)
    trainer.condition_dim = 1
    trainer.sim = SimpleNamespace(J=2)
    trainer.demos, trainer.stand_demos = ExpertWindows(1000), ExpertWindows(3000)
    hold = jnp.repeat(jnp.array([0., .5, 1.]), 64)
    replay = jnp.zeros((len(hold), 10, 16)).at[..., -1].set(hold[:, None])
    draw = jax.jit(lambda key: trainer.discriminator_examples(key, replay))
    expert, policy = map(np.asarray, draw(jax.random.PRNGKey(17)))
    labels = policy[:, 0, -1]
    np.testing.assert_array_equal(policy[..., -1], np.broadcast_to(labels[:, None], policy.shape[:2]))
    np.testing.assert_array_equal(expert[..., -1], np.broadcast_to(labels[:, None], expert.shape[:2]))
    np.testing.assert_array_equal(labels[:64], 0.)
    np.testing.assert_array_equal(labels[-64:], 1.)
    assert set(labels[64:128]) == {0., 1.}
    np.testing.assert_array_equal(expert[:, 0, 0] >= 3000, labels.astype(bool))
    assert np.unique(expert[:64, 0, 0]).size > 1
    another, _ = draw(jax.random.PRNGKey(18))
    assert not np.array_equal(another[:64, 0, 0], expert[:64, 0, 0])

    # Equal-sized banks with distinct offsets make it possible to detect
    # accidentally reusing the same random starts for both sources.
    same_key = jax.random.PRNGKey(19)
    rises, _ = trainer.discriminator_examples(same_key, replay.at[..., -1].set(0.))
    stands, _ = trainer.discriminator_examples(same_key, replay.at[..., -1].set(1.))
    assert not np.array_equal(rises[:, 0, 0] - 1000, stands[:, 0, 0] - 3000)
