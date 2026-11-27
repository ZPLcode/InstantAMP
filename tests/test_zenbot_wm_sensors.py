from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from instantamp import buffer, env, nets, zenbot, zenbot26, zenbot_gait_rewards
from instantamp.sim import Sim, unflatten
from instantamp.train import InstantAMP, parse
from instantamp.training_sim import simulation_options
from instantamp.zenbot_motion_train_obs import (
    advance_sensors,
    mirror_action,
    mirror_observation,
)
from instantamp.zenbot_wm_sensors import (
    pd_targets,
    predicted_sensors,
    restore_sensors,
)


def sensor_history(n, joints=23):
    imu = jnp.tile(jnp.array([.4, -.2, .15, 0., 0., -1.]), (n, 1))
    command = jnp.zeros((n, 3))
    raw = jnp.arange(n * joints, dtype=jnp.float32).reshape(n, joints) / 32
    history = None
    for t in range(3):
        history = advance_sensors(history, imu + .002*t, command, raw, raw, raw+t,
                                  jnp.full(n, t == 0), jax.random.key(31+t))
    return history


def poses(n, profile=zenbot):
    q = jnp.tile(jnp.r_[0., 0., .7, .5, .5, .5, .5, profile.DEFAULT_Q], (n, 1))
    nv = 6 + len(profile.DOF)
    v = jnp.arange(n * nv, dtype=jnp.float32).reshape(n, nv)/80
    return q, v


@pytest.mark.parametrize("noisy", [True, False])
@pytest.mark.parametrize("profile", [zenbot, zenbot26])
def test_restore_physical_sensor_frames_and_predict_like_sim(noisy, profile):
    seed = jax.random.key(432)
    joints = len(profile.DOF)
    original = sensor_history(2, joints)
    restored = restore_sensors(original.observation(), *original[:3])
    for a, b in zip(restored, original):
        np.testing.assert_array_equal(a, b)
    q, v = poses(2, profile)
    action = original.frames[:, -1, -joints:] + .25
    cmd = jnp.array([[0., 0., 0.], [-.5, 0., 0.]])
    sim = Sim.__new__(Sim)
    sim.noisy_obs = noisy
    state = SimpleNamespace(motion_train_sensors=original, data=SimpleNamespace(qpos=q, qvel=v), a_last=action)
    S = unflatten(q, v)
    actual = sim.motion_train_sensors(state, S[1], cmd, jnp.zeros(2, bool), seed)
    predicted = predicted_sensors(restored, S, cmd, action, seed, noisy)
    for a, b in zip(actual, predicted):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("profile", [zenbot, zenbot26])
def test_zero_and_grouped_delay_targets_and_gradients_equal_four_native_substeps(profile):
    n = 2
    joints = len(profile.DOF)
    frame_dim = 9 + 3 * joints
    history = sensor_history(n, joints)
    previous, older = history.frames[:, -1, -joints:], history.frames[:, -2, -joints:]
    act = jnp.flip(previous, -1) + .3
    bias = previous / 128
    groups = int(profile.GROUP.max()) + 1
    delays = jnp.stack((jnp.zeros(groups, jnp.int32), 7 - jnp.arange(groups) % 5))
    context = jnp.pad(delays, ((0, 0), (2, 1)))
    physical = Sim.__new__(Sim)
    physical.robot, physical.history, physical.frame_dim, physical.J = profile, 5, frame_dim, joints
    raw_frames = jnp.zeros((n, 5, frame_dim)).at[:, -2, -joints:].set(older).at[:, -1, -joints:].set(previous)
    actual = physical.control_targets(act, bias, raw_frames.reshape(n, -1), context)
    def expected(a, p, o):
        group_delay = delays[:, profile.GROUP]
        pieces = []
        for j in range(4):
            raw = jnp.where(group_delay <= j, a, jnp.where(group_delay <= j+4, p, o))
            pieces.append(profile.DEFAULT_Q + profile.ACTION_SCALE*raw-bias)
        return jnp.concatenate(pieces, axis=1)
    def modeled(a, p, o):
        return pd_targets(a, p, o, bias, context, profile=profile)
    np.testing.assert_array_equal(actual, expected(act, previous, older))
    np.testing.assert_array_equal(actual, modeled(act, previous, older))
    for variable in (0, 1, 2):
        true = jax.grad(lambda *args: expected(*args).sum(), argnums=variable)(act, previous, older)
        got = jax.grad(lambda *args: modeled(*args).sum(), argnums=variable)(act, previous, older)
        np.testing.assert_array_equal(got, true)


@pytest.mark.parametrize("profile,delays", [(zenbot, [3, 4, 5, 6, 7]),
                                           (zenbot26, [3, 4, 5, 6, 7, 7, 4])])
def test_cold_motor_queue_clamps_the_lag_to_commands_received_since_each_reset(profile, delays):
    joints = len(profile.DOF)
    actions = np.arange(3 * joints, dtype=np.float32).reshape(3, joints) / 16.
    delay_groups = np.asarray(delays, dtype=np.int32)
    if profile is zenbot26:
        # The waist yaw shares the hip delay. Roll/pitch share a separate
        # delay and the head has its own delayed PD group in Motion Train.
        indices = [profile.DOF.index(j) for j in ("waist_yaw", "waist_roll", "waist_pitch", "head_yaw")]
        np.testing.assert_array_equal(profile.GROUP[indices], [0, 5, 5, 6])
    context = jnp.broadcast_to(jnp.pad(delay_groups, (2, 1)), (3, len(delays) + 3))
    # Three worlds have received different numbers of control commands since their own resets.
    elapsed = np.arange(3)
    previous = np.vstack((np.zeros(joints), actions[:2])).astype(np.float32)
    older = np.vstack((np.zeros((2, joints)), actions[0])).astype(np.float32)
    targets = pd_targets(actions, previous, older, jnp.zeros_like(actions), context, elapsed=elapsed, profile=profile)
    for world, t in enumerate(elapsed):
        for sub in range(4):
            source = np.maximum(0, 4*t + sub - delay_groups[profile.GROUP]) // 4
            expected = profile.DEFAULT_Q + .25 * actions[source, np.arange(joints)]
            np.testing.assert_array_equal(targets.reshape(3, 4, joints)[world, sub], expected)


def test_real_record_resamples_only_newest_command_and_backpropagates_through_wm():
    n = 4
    xs = InstantAMP.__new__(InstantAMP)
    xs.B_im, xs.J = n, 23
    xs.amp, xs.ppo = False, False
    xs.cfg = SimpleNamespace(motion_train_wm=True, g1_style=False, num_envs=n, det_share=.5,
                             action_noise=0., imag_down=0., cmd_fresh=1., recovery_amp_seconds=4., mm_bf16='none', horizon=5,
                             imag_unroll=1, imag_remat='none', gamma=.99)
    sim = Sim.__new__(Sim)
    sim.mixed, sim.robot = True, zenbot
    sim.root_com = jnp.zeros(3)
    sim.noisy_obs, sim.episode_steps, sim.recovery_amp_steps = True, 400, 200
    sim.q_soft, sim.sole_frame = None, (0,1)
    zeros = jnp.zeros(n)
    command = env.Cmd(jnp.tile(jnp.array([1.,0.,0.]), (n,1)), zeros, zeros>1, zeros>1, zeros+9.)
    sim.zenbot_motion_train_wm = True
    sim.view_state = lambda s: SimpleNamespace(height=s[0][:,2], soles=jnp.zeros((n,2)),
                                               clearance=jnp.ones((n,2)), kin=None)
    sim.reward_terms = lambda s, s1, *args, **kwargs: {'output': s1[4][:, 0]}
    sim.weighted = lambda terms: terms
    xs.termination = lambda clearance: zeros
    xs.termination_down = lambda *args: zeros
    xs.sim = sim
    history = sensor_history(n)
    q, v = poses(n)
    S = unflatten(q,v)
    previous = history.frames[:,-1,55:]
    delay = jnp.array([[0]*5, [3,4,5,6,7], [0]*5, [7,6,5,4,3]])
    context = jnp.pad(delay, ((0,0),(2,1))).astype('f')
    sim.context = lambda _: context
    state = SimpleNamespace(cmd=command, motion_train_sensors=history, world=SimpleNamespace(bias=jnp.zeros_like(previous)),
                            was_up=jnp.array([False,True,True,True]), obs=jnp.zeros((n,390)),
                            a_last=previous,a_ode=previous,c_obs=jnp.zeros((n,3)),t=jnp.arange(n),
                            recovery_left=jnp.array([100, 5, 0, 0]))
    trans = SimpleNamespace(qpos=q,qvel=v,qpos1=q,qvel1=v,target=jnp.zeros((n,92)),reward=zeros,
                            terminated=zeros>1,truncated=zeros>1,diverged=zeros>1,stats={})
    rec, _ = xs.record(state, state, trans, previous, previous)
    np.testing.assert_array_equal(rec['obs'], history.observation())
    assert not any(x.shape==(n,3,78) for x in jax.tree.leaves(rec))

    pi = nets.mlp_init(jax.random.key(8), [234,16,23])
    sim.step = lambda state, applied, clean: (state, trans)
    xs.ingest = lambda c,flat: flat
    collected, _ = xs.collect_step(dict(pi=pi,env=state,obs_norm={'raw':True}),jax.random.key(11))
    np.testing.assert_array_equal(collected['obs'],history.observation())
    replay = buffer.add(buffer.init(1, collected, full_precision_obs=True), collected)
    assert replay["obs"].dtype == jnp.float32
    start = xs.sample_starts(replay, jax.random.key(3))
    row = start['t'].astype(int)
    expected = history.observation()[row]
    np.testing.assert_array_equal(start['obs'][:,:24], expected[:,:24])
    np.testing.assert_array_equal(start['obs'][:,27:], expected[:,27:])
    requested = sim.sample_wm_command(jax.random.split(jax.random.key(3), 3)[2], n)
    np.testing.assert_array_equal(start['obs'][:,24:27], jnp.where((row >= 2)[:,None],requested.vel,0.))

    wm_in = nets.wm_inputs(S, S[0][:,2], jnp.zeros((n,2)), trans.target, context).shape[-1]
    wm, wb = nets.wm_init(jax.random.key(4), wm_in, 58, (16,))
    # Give the learned dynamics a nonzero motor response. Reward depends only on the predicted joint.
    wm['net'][-1]['w'] = jax.random.normal(jax.random.key(13), (16,58))*.03
    keys = jax.random.split(jax.random.key(20), xs.cfg.horizon)
    def loss(weights):
        return xs.imagine(weights,wm,wb,{'raw':True},start,keys)[0]
    value, grad = jax.jit(jax.value_and_grad(loss))(pi)
    assert np.isfinite(value) and all(np.isfinite(x).all() for x in jax.tree.leaves(grad))
    assert float(jnp.linalg.norm(grad[0]['w'])) > 1.e-5


def test_imagined_mirror_only_updates_actor_on_live_commanded_gait(zenbot23_motion_banks, monkeypatch):
    # Forward, yaw, stop, recovering and already timing out. All remaining episodes end on step two.
    n = 5
    learner = InstantAMP.__new__(InstantAMP)
    cfg = learner.cfg = parse(["--robot", zenbot.MIXED_TASK, "--num_envs", "32", "--horizon", "2", "--imag_unroll", "1",
                               "--motion_train_randomization", "0"])
    learner.sim = Sim(**simulation_options(vars(cfg)), physics=False, obs_noise=False)
    learner.B_im, learner.J = n, 23
    monkeypatch.setattr(nets, "wm_step", lambda wm, bounds, state, *args, **kwargs: state)
    zeros = jnp.zeros(n)
    command = jnp.array([[.5, 0., 0.], [0., 0., -.5], [0., 0., 0.], [.5, 0., 0.], [.5, 0., 0.]])
    sensors = sensor_history(n)
    obs = sensors.observation().at[:, 24:27].set(command)
    q = jnp.tile(jnp.r_[0., 0., .70, 1., 0., 0., 0., zenbot.DEFAULT_Q], (n, 1)).at[3, 2].set(.20)
    previous = sensors.frames[:, -1, -23:]
    start = dict(s_qpos=q, s_qvel=jnp.zeros((n, 29)), xi=jnp.zeros((n, 8)), bias=jnp.zeros_like(previous),
                 obs=obs, imu_delay=sensors.delay_s, current_imu=sensors.current_imu, previous_imu=sensors.previous_imu,
                 a_last=previous, a_ode=previous, c_obs=command, cmd=env.Cmd(command, zeros, zeros > 0, zeros > 0, zeros+8.),
                 t=zeros.astype(int), was_up=jnp.ones(n, bool),
                 gait_world=jnp.array([True, True, True, False, True]),
                 first_attempt=zenbot_gait_rewards.FirstAttempt.start(q[:, 2], q[:, 2] < .44, zenbot)._asdict(),
                 step_history=zenbot_gait_rewards.StepHistory.empty(n)._asdict())
    pi = nets.mlp_init(jax.random.key(4), (234, 16, 23))
    keys = jax.random.split(jax.random.key(5), cfg.horizon)
    limits = jnp.array([2, 2, 2, 2, 1])
    norm = {"raw": True}

    def penalty(params, x, episode_limits):
        loss, stats = learner.imagine(params, None, None, norm, dict(start, obs=x, episode_limit=episode_limits), keys)
        return stats["imag_gait_mirror_mse"], (loss + stats["G"], stats["imag_gait_mirror_fraction"])

    def expected(params):
        target = mirror_action(learner.act(params, norm, obs[:2], previous[:2]))
        reflected = learner.act(params, norm, mirror_observation(obs[:2]), mirror_action(previous[:2]))
        return jnp.square(reflected - jax.lax.stop_gradient(target)).mean()

    evaluate = jax.jit(jax.value_and_grad(penalty, argnums=(0, 1), has_aux=True))
    (mse, (added, fraction)), (grad, obs_grad) = evaluate(pi, obs, limits)
    wanted, target_grad = jax.value_and_grad(expected)(pi)
    np.testing.assert_allclose([mse, added, fraction], [wanted, .2 * wanted, 2 / (2*n)], atol=2e-6)
    assert float(wanted) > 0
    np.testing.assert_array_equal(obs_grad, 0.)
    for actual, wanted_grad in zip(jax.tree.leaves(grad), jax.tree.leaves(target_grad), strict=True):
        np.testing.assert_allclose(actual, wanted_grad, atol=2e-6)
    (mse, (added, fraction)), grads = evaluate(pi, obs, jnp.ones_like(limits))
    np.testing.assert_allclose([mse, added, fraction], 0., atol=1e-7)
    assert all(np.all(np.asarray(g) == 0) for g in jax.tree.leaves(grads))


def test_imagined_fall_preserves_event_and_has_posture_gradient_only_from_standing():
    from instantamp.sim import MIXED_STAY_HEIGHT, MIXED_STAY_UP_COS

    sim = Sim.__new__(Sim)
    sim.robot, sim.mixed = zenbot, True
    grid = jnp.array([-.1, -.002, 0., .002, .12])
    h, z, was_up = jnp.meshgrid(MIXED_STAY_HEIGHT + grid, MIXED_STAY_UP_COS + grid,
                                jnp.array([False, True]), indexing="ij")
    def fall(height, cosine):
        rot = jnp.broadcast_to(jnp.eye(3), height.shape + (3, 3)).at[..., 2, 2].set(cosine)
        up = sim.next_upright(was_up, height, rot)
        return sim.wm_fall(was_up, up, height, rot), (was_up & ~up).astype(height.dtype)
    value, event = jax.jit(fall)(h, z)
    np.testing.assert_array_equal(value, event)
    # Return includes the original negative fall cost. Increasing height/verticality improves it near the boundary.
    dh, dz = jax.jit(jax.grad(lambda a,b: -2. * fall(a,b)[0].sum(), (0,1)))(h,z)
    np.testing.assert_array_equal(np.asarray(dh)[~was_up], 0.)
    np.testing.assert_array_equal(np.asarray(dz)[~was_up], 0.)
    assert dh[2,2,1] > 1. and dz[2,2,1] > 1.
    assert (np.asarray(dh)[was_up] >= 0.).all() and (np.asarray(dz)[was_up] >= 0.).all()


def test_wm_gait_discriminator_sees_still_after_recovery_and_legacy_keeps_moving_filter():
    xs = InstantAMP.__new__(InstantAMP)
    xs.sim = Sim.__new__(Sim)
    xs.sim.robot = zenbot
    n, ticks = 5, 10
    height = jnp.full((n, ticks, 3), .7).at[2, 5, 2].set(.1)
    rotation = jnp.broadcast_to(jnp.eye(3), (n, ticks, 3, 3))
    w = dict(mask=jnp.ones((n,ticks)).at[4,-1].set(0.),
             was_up=jnp.ones((n,ticks), bool), c_obs=jnp.zeros((n,ticks,3)).at[1,:,0].set(1.),
             recovery_left=jnp.zeros((n,ticks), int).at[3,0].set(1))
    xs.cfg = SimpleNamespace(motion_train_wm=True, recovery_amp_seconds=4.)
    np.testing.assert_array_equal(xs.gait_amp_eligible(w,(height,rotation)), [1,1,0,0,0])
    xs.cfg.motion_train_wm = False
    np.testing.assert_array_equal(xs.gait_amp_eligible(w,(height,rotation)), [0,1,0,0,0])


def test_extra_stand_is_one_independent_clip_and_preserves_source_sampling_weights(tmp_path):
    import json

    from instantamp import amp

    main = tmp_path / "ground"
    main.mkdir()
    stand = tmp_path / "stand_4s_120f.npz"
    def write(path, marker):
        n = 30
        q = np.full((n,23), marker, np.float32)
        np.savez(path, fps=50, root_pos=np.zeros((n,3)), root_quat=np.tile([0,0,0,1],(n,1)),
                 root_ang_vel=np.zeros((n,3)), root_lin_vel=np.zeros((n,3)), dof_pos=q, dof_vel=q*0)
    write(main/"rise1.npz", 1)
    write(main/"rise2.npz", 2)
    write(stand, 3)
    sim = SimpleNamespace(robot=zenbot, mixed=True, recovery=False, J=23,
                          amp_frame=lambda state, points: state[4])
    def bank(extra):
        return amp.Demos(main, sim, 28, per_clip=True, frame_fn=lambda s:s[4][:,:1], extra_clip=extra)
    old = bank(None)
    enriched = bank(stand)
    assert old.clip_probabilities is None and enriched.clip_probabilities is None
    np.testing.assert_array_equal(enriched.clip_lengths, [3,3,3])
    windows = enriched.frames[enriched.starts[:,None] + np.arange(28)]
    np.testing.assert_array_equal(windows.max(1), windows.min(1))
    chosen = np.asarray(enriched.sample(jax.random.key(103), 8192)[:,0,0])
    assert abs(np.mean(chosen==3.) - 1/3) < .02
    (main/"sampling.json").write_text(json.dumps({"weights": {"rise1":3., "rise2":1.}}))
    np.testing.assert_allclose(bank(None).clip_probabilities, [.75,.25])
    np.testing.assert_allclose(bank(stand).clip_probabilities, [.5,1/6,1/3])
    with pytest.raises(FileNotFoundError, match="Missing additional AMP reference"):
        bank(tmp_path/"missing.npz")
