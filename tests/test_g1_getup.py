"""Regressions for the independent LeggedLab G1 get-up task."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from instantamp import amp, g1_getup
from instantamp.sim import Sim, unflatten
from instantamp.train import parse


def test_task_uses_its_own_bank_and_profile():
    cfg = parse(["--task", g1_getup.TASK_ID])
    assert cfg.robot == "g1_getup" and cfg.log_dir == "runs/g1_getup"
    assert cfg.floor_dir == cfg.amp_dir == "data/g1_getup/recovery"
    assert cfg.amp_getup == cfg.push == 0 and cfg.amp_lerp == .5 and cfg.amp_style_scale == 50.
    assert cfg.ode == 0 and cfg.random_start == 1
    assert cfg.buffer_steps == cfg.amp_recent == 244
    assert cfg.reference_weight == 0 and cfg.init_policy == "" and cfg.assist_newtons == g1_getup.INITIAL_ASSIST
    assert parse(["--robot", "g1_getup"]).task == cfg.task
    with pytest.raises(SystemExit):
        parse(["--task", cfg.task, "--robot", "g1"])


def test_reference_resets_drives_and_observations(tmp_path):
    for i, n in enumerate((18, 95)):
        t = np.arange(n, dtype=np.float32)
        pos = np.column_stack((t, np.full(n, i), .25 + t / 200))
        q = np.broadcast_to(g1_getup.DEFAULT_Q, (n, 29))
        np.savez(tmp_path / f"{i}.npz", root_pos=pos, root_quat=np.tile([0, 0, 0, 1], (n, 1)),
                 root_lin_vel=pos * 0, root_ang_vel=pos * 0, dof_pos=q, dof_vel=q * 0, fps=50)
    sim = Sim(1024, floor_dir=tmp_path, robot="g1_getup", physics=False)
    # The named task penalizes the same raw action change three times as strongly as its source.
    delta = jnp.arange(29, dtype=jnp.float32) / 20.
    terms = {name: 0. for name in sim.terms}
    terms["action_rate_l2"] = jnp.sum(delta**2)
    np.testing.assert_allclose(sim.weighted(terms)["action_rate_l2"], .02 * (-.005 * 3) * np.sum(delta**2))
    key = jax.random.key(8)
    qpos, qvel, _ = sim._fresh(key, sim.world(key))
    x, clip, z = np.asarray(qpos[:, :3]).T
    np.testing.assert_allclose(z, .35 + x / 200, atol=1e-6)
    assert .45 < clip.mean() < .55
    assert x[clip == 0].max() < 8 and x[clip == 1].max() < 85
    assert sim.episode_steps == 500 and sim.assist_newtons == 200.
    assert sim.reference is None and "motion_reference" not in sim.terms
    np.testing.assert_allclose(sim.m.dof_armature[6:], .01)
    np.testing.assert_allclose(sim.m.actuator_forcerange[:, 1], g1_getup.EFFORT)
    rotating = qvel[:2].at[:, :3].set(jnp.array([1., 0., 0.])).at[:, 4].set(2.)
    corrected = sim.reference_velocity(qpos[:2], rotating)
    np.testing.assert_allclose(corrected[:, 0], 1. - 2. * sim.root_com[2], atol=1e-6)
    spinning = unflatten(qpos[:2], corrected)
    privileged = sim.push_critic(None, spinning, qvel[:2, 6:])
    np.testing.assert_allclose(privileged[:, :3], rotating[:, :3], atol=1e-6)
    state = unflatten(qpos[:2], qvel[:2])
    action = jnp.ones((2, 29))
    np.testing.assert_allclose(sim.control_targets(action, 0., None, None), g1_getup.DEFAULT_Q[None] + .25 * action)
    assert sim.support_link[sim.m.geom("robot_left_shin_collision").id] >= 0
    assert sim.support_link[sim.m.geom("robot_left_shoulder_yaw_collision").id] == -1
    # Source USD includes convex hand/pelvis collisions and a box plus five spheres per foot.
    import mujoco
    assert (sim.m.ngeom, sim.m.nmesh, len(sim.foot_geoms)) == (34, 15, 12)  # includes the ground
    foot_types = sim.m.geom_type[sim.foot_geoms]
    assert np.count_nonzero(foot_types == mujoco.mjtGeom.mjGEOM_BOX) == 2
    assert np.count_nonzero(foot_types == mujoco.mjtGeom.mjGEOM_SPHERE) == 10
    world = sim.world(key)
    ratio = world.body_mass[:, sim.torso] / sim.m.body_mass[sim.torso]
    np.testing.assert_allclose(world.body_inertia[:, sim.torso], ratio[:, None] * sim.m.body_inertia[sim.torso], rtol=1e-6)
    np.testing.assert_array_equal(world.geom_friction[:, 0, 0], 1.)
    assert amp.Demos(tmp_path, sim, 10).sample(key, 2).shape == (2, 10, 85)
    frame = sim.obs_frame(state, action, None, None)
    history = sim.push_actor(None, frame)
    updated = sim.push_actor(history, frame + 1, jnp.array([True, False]))
    assert history.shape == (2, 570) and sim.push_critic(None, state, action).shape == (2, 590)
    np.testing.assert_allclose(updated[1, :12], history[1, :12])
    np.testing.assert_allclose(updated[1, 12:15], frame[1, :3] + 1)
    np.testing.assert_allclose(updated[0, :15], np.tile(frame[0, :3] + 1, 5))
    pos = state[0].at[:, 2].set(jnp.array([.65, .75]))
    state = (pos, *state[1:])
    terms = sim.reward_terms(state, state, None, action, action, torque=0 * action)
    np.testing.assert_allclose(terms["target_base_height"], [0., 1.])
    np.testing.assert_allclose(sum(sim.weighted(terms).values()), [0., .22], atol=1e-5)


def test_named_reference_tracks_the_sampled_clip_and_holds_its_final_pose(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from typing import NamedTuple

    from instantamp import buffer, env, nets
    from instantamp.train import InstantAMP

    for clip, length in enumerate((23, 41)):
        frame = np.arange(length, dtype=np.float32)
        # x uniquely identifies the frame; y identifies the clip.
        pos = np.stack((frame, np.full_like(frame, clip), .25 + .5 * frame / (length - 1)), -1)
        q = np.tile(g1_getup.DEFAULT_Q, (length, 1))
        q[:, 0] += .02 * frame + .1 * clip
        np.savez(tmp_path / f"{clip}.npz", root_pos=pos, root_quat=np.tile([0, 0, 0, 1], (length, 1)),
                 root_lin_vel=pos * 0, root_ang_vel=pos * 0, dof_pos=q, dof_vel=q * 0, fps=50)

    cfg = parse(["--task", g1_getup.TASK_ID, "--reference_weight", "18", "--horizon", "1"])
    sim = Sim(8, floor_dir=tmp_path, robot=cfg.robot, physics=False, reference_weight=cfg.reference_weight)
    # Reset only reads/writes these fields. Real collision stepping uses the Warp backend.
    class ResetData(NamedTuple):
        qpos: jax.Array
        qvel: jax.Array
        qacc_warmstart: jax.Array

        def replace(self, **kwargs):
            return self._replace(**kwargs)

    sim.dx = ResetData(jnp.zeros(sim.m.nq), jnp.zeros(sim.m.nv), jnp.zeros(sim.m.nv))
    state = sim.init(jax.random.key(8))
    qpos, qvel = state.data.qpos, state.data.qvel
    clip, start = state.reference_clip, state.reference_start
    np.testing.assert_array_equal(state.t, np.zeros(sim.N))
    assert ((state.episode_limit >= 1) & (state.episode_limit <= sim.episode_steps)).all()
    assert len(np.unique(state.episode_limit)) > 1
    reset = jnp.arange(sim.N) % 2 == 0
    next_state = sim._reset(state, reset, jax.random.key(29))
    np.testing.assert_array_equal(next_state.episode_limit, np.where(reset, sim.episode_steps, state.episode_limit))
    np.testing.assert_array_equal(next_state.t, np.zeros(sim.N))
    np.testing.assert_array_equal(clip, qpos[:, 1])
    np.testing.assert_array_equal(start, qpos[:, 0])
    assert set(np.asarray(clip)) == {0, 1} and np.any(np.asarray(start) > 0)
    assert state.reference_yaw is None
    # Score an exactly matching pose at the next frame and well past the end.
    for step in (1, 100):
        idx = sim.reference.offsets[clip] + jnp.minimum(start + step, sim.reference.lengths[clip] - 1)
        pose = jnp.concatenate((sim.walk["xy"][idx], sim.walk["z"][idx, None], sim.walk["quat"][idx], sim.walk["q"][idx]), -1)
        target = unflatten(pose, qvel)
        zero = jnp.zeros_like(state.a_last)
        term = sim.reward_terms(target, target, None, zero, zero, torque=zero, clip=clip, reference_start=start, step=step)
        np.testing.assert_allclose(term["motion_reference"], 1., atol=1e-6)
    goal = sim.reference_pose(target)
    nearby = goal.at[:, 0].add(.03)
    fallen = goal.at[:, 0].set(.15).at[:, 1:4].set(jnp.array([0., 1., 0.]))
    final_frame = sim.reference.lengths[clip] - 1
    assert (sim.reference.score(nearby, clip, final_frame) > .98).all()
    assert (sim.reference.score(fallen, clip, final_frame) < .15).all()
    after_last = jnp.arange(sim.N) % 2 == 0
    boundary = final_frame - start + after_last.astype(jnp.int32)
    for scale in (1., .05, 0.):
        sim.reference_hold_scale = scale
        result = sim.reward_terms(target, target, None, zero, zero, torque=zero, clip=clip,
                                  reference_start=start, step=boundary)
        np.testing.assert_allclose(result["motion_reference"], np.where(after_last, scale, 1.), atol=1e-6)
        np.testing.assert_allclose(result["reference_pose_score"], 1., atol=1e-6)
    sim.reference_hold_scale = .05

    # One recorded collection step sampled back for imagination. Keep the pose
    # fixed so its expected reference reward can be evaluated independently.
    xs = InstantAMP.__new__(InstantAMP)
    xs.sim, xs.cfg, xs.B_im = sim, cfg, sim.N
    xs.amp = xs.ppo = xs.recovery_clock = False
    xs.act = lambda *args, **kwargs: jnp.zeros_like(state.a_last)
    monkeypatch.setattr(nets, "wm_step", lambda wm, wb, S, *args, **kwargs: S)
    tr = SimpleNamespace(qpos=qpos, qvel=qvel, qpos1=qpos, qvel1=qvel,
                         target=state.a_last, terminated=state.t > 0, truncated=state.t > 0,
                         reward=jnp.zeros(sim.N), stats={})
    rec, _ = xs.record(state, state, tr, state.a_last, state.a_last)
    assert "reference_yaw" not in rec
    flat = buffer.flatten({**rec, "ok": jnp.ones(sim.N)})
    replay = buffer.add(buffer.init(1, flat), flat)
    sampled = xs.sample_starts(replay, jax.random.key(12))
    np.testing.assert_array_equal(sampled["reference_start"], sampled["s_qpos"][:, 0])
    np.testing.assert_array_equal(sampled["reference_clip"], sampled["s_qpos"][:, 1])
    sim.terms = ("motion_reference",)
    _, stats = xs.imagine(None, None, None, None, sampled, jax.random.split(jax.random.key(15), 1))
    observed = unflatten(sampled["s_qpos"], sampled["s_qvel"])
    expected = sim.reference.score(sim.reference_pose(observed), sampled["reference_clip"],
                                   sampled["reference_start"] + sampled["t"] + 1)
    np.testing.assert_allclose(stats["r"], (expected * env.DT * cfg.reference_weight).mean(), rtol=1e-5)
    assert "episode_limit" in sampled
    # Two worlds can start on the same reference frame with different first episode
    # deadlines. Their target pose is identical for the step, even if one now ends.
    sampled["episode_limit"] = sampled["t"] + jnp.where(reset, 1, 2)
    _, shortened = xs.imagine(None, None, None, None, sampled, jax.random.split(jax.random.key(15), 1))
    np.testing.assert_allclose(shortened["r"], stats["r"])
    np.testing.assert_allclose(shortened["surv"], 0.5)
    # Compare per-world endpoints scored in the collector, replayed in real PPO,
    # and recomputed on a frozen world model at the same post-step time.
    from instantamp import critic
    xs.ppo = True
    xs.ppo_envs = np.arange(sim.N)
    cfg.ppo_len = cfg.amp_frames = 1
    state = state._replace(t=boundary - 1)
    observed = unflatten(qpos, qvel)
    terms = sim.reward_terms(observed, observed, None, zero, zero, torque=zero, clip=clip,
                             reference_start=start, step=boundary)
    tr.reward = sim.weighted(terms)["motion_reference"]
    rec, _ = xs.record(state, state, tr, zero, zero)
    row = buffer.flatten({**rec, "ok": jnp.ones(sim.N)})
    replay = buffer.add(buffer.add(buffer.init(2, row), row), row)
    monkeypatch.setattr(critic, "value", lambda _, observations: jnp.zeros(len(observations)))
    _, real = xs.ppo_batch({"buf": replay, "critic": None})
    start_state = buffer.nest(row)
    start_state["cmd"] = env.Cmd(**start_state["cmd"])
    _, imagined = xs.imagine(None, None, None, None, start_state, jax.random.split(jax.random.key(1), 1))
    expected = cfg.reference_weight * env.DT * sim.reference.score(sim.reference_pose(observed), clip, final_frame)
    expected *= jnp.where(after_last, .05, 1.)
    np.testing.assert_allclose(tr.reward, expected, rtol=1e-5)
    np.testing.assert_allclose([real["ppo_r"], imagined["r"]], expected.mean(), rtol=1e-5)
    cfg.rise_anchor = 1.
    clock = state._replace(reference_start=jnp.where(after_last, 0, final_frame - 1), t=jnp.zeros_like(state.t))
    terms = sim.reward_terms(observed, observed, None, zero, zero, torque=zero, clip=clip,
                             reference_start=clock.reference_start, step=1)
    tr.parts = sim.weighted(terms)["motion_reference"][:, None]
    rec, _ = xs.record(clock, clock, tr, zero, zero)
    wanted = np.where(after_last, np.asarray(terms["reference_pose_score"])**2, 0.)
    np.testing.assert_allclose(rec["rise_anchor_weight"], wanted, rtol=1e-5)
    row = buffer.flatten({**rec, "ok": jnp.ones(sim.N)})
    replay = buffer.add(buffer.add(buffer.init(2, row), row), row)
    xs.act = lambda pi, norm, obs, prev: obs[:, :sim.J] * pi + norm["mean"]
    teacher = {"pi": 2., "obs_norm": {"mean": .1}}
    training, _ = xs.ppo_batch({"buf": replay, "critic": None, "rise_teacher": teacher})
    np.testing.assert_allclose(training["rise_anchor_weight"], wanted, rtol=1e-5)
    np.testing.assert_allclose(training["anchor_mu"], 2. * training["obs"][:, :sim.J] + .1)


def test_initialize_actor_loads_the_averaged_export_and_matching_normalization(tmp_path):
    from instantamp.train import initialize_actor, save_policy

    def layer(value):
        return [{"w": jnp.full((3, 2), value), "b": jnp.full((2,), value)}]

    src = {"pi": layer(1.), "avg": layer(2.), "obs_norm": {"mean": jnp.arange(3.), "var": jnp.arange(3.) + 1}}
    dst = {"pi": layer(0.), "obs_norm": {"mean": jnp.zeros(3), "var": jnp.ones(3), "count": 0.}}
    path = str(tmp_path / "policy")
    save_policy(path, src, {}, (.1, None))
    loaded = initialize_actor(dst, path + ".npz", (.1, None))
    for name in ("pi", "avg"):
        for field in ("w", "b"):
            np.testing.assert_array_equal(loaded[name][0][field], src["avg"][0][field])
    for name in ("mean", "var"):
        np.testing.assert_array_equal(loaded["obs_norm"][name], src["obs_norm"][name])
    assert loaded["obs_norm"]["count"] == dst["obs_norm"]["count"]
    with pytest.raises(ValueError, match="ODE"):
        initialize_actor(dst, path + ".npz", None)
    with pytest.raises(ValueError, match="policy or observations"):
        initialize_actor({**dst, "pi": [{"w": jnp.zeros((4, 2)), "b": jnp.zeros(2)}]}, path + ".npz", (.1, None))


def test_raw_ppo_actor_preserves_fast_joint_observations(tmp_path):
    from instantamp.inference import load, policy_output
    from instantamp.train import InstantAMP, initialize_actor, save_policy

    actor = [{"w": jnp.eye(2), "b": jnp.zeros(2)}]
    stats = {"mean": np.zeros(2), "var": np.ones(2), "raw": True}
    output = str(tmp_path / "raw")
    save_policy(output, {"pi": actor, "avg": actor, "obs_norm": stats}, {})
    policy, stats, ode = load(output + ".npz")
    obs = jnp.asarray([[24., -16.]])
    assert ode is None
    np.testing.assert_array_equal(policy_output(policy, stats, obs), obs)
    template = [{"w": jnp.zeros((2, 2)), "b": jnp.zeros(2), "ln_g": jnp.ones(2), "ln_b": jnp.zeros(2)}]
    warm = initialize_actor({"pi": template, "obs_norm": stats, "opt_pi": {}}, output + ".npz", None)
    assert set(warm["pi"][0]) == {"w", "b"}
    assert set(warm["opt_pi"]["leaves"][0]) == {"w", "b"}
    np.testing.assert_array_equal(InstantAMP.norm_obs(None, warm["obs_norm"], obs), obs)


@pytest.mark.parametrize("full_raw", [False, True])
def test_native_continuation_replaces_only_actor_and_normalizer(tmp_path, full_raw):
    from instantamp import optim
    from instantamp.train import initialize_native_learner, save_full, save_policy

    def actor(value, raw):
        layer = {"w": jnp.full((3, 2), value), "b": jnp.full(2, value)}
        stats = {"mean": jnp.full(3, value), "var": jnp.ones(3), "count": jnp.array(0.)}
        if raw:
            stats["raw"] = jnp.array(True)
        else:
            layer.update(ln_g=jnp.ones(2), ln_b=jnp.zeros(2))
        pi = [layer]
        return {"pi": pi, "avg": pi, "obs_norm": stats,
                "opt_pi": optim.init(pi, jax.tree.map(lambda _: False, pi))}

    fresh = {**actor(0., False), "it": jnp.array(0), "env_steps": jnp.array(0.),
             "wm": jnp.zeros(2), "amp": jnp.zeros(2), "critic": jnp.zeros(2), "opt_wm": jnp.zeros(2)}
    source = {**jax.tree.map(lambda x: x + 5, fresh), **actor(7., full_raw)}
    source["opt_pi"] = jax.tree.map(lambda x: x + 31, source["opt_pi"])
    selected = actor(12., not full_raw)
    full = tmp_path / "ckpt" / "full" / "0012"
    full.parent.mkdir(parents=True)
    save_policy(str(full.parent.parent / full.name), source, {})
    save_full(str(full), source)
    override = tmp_path / "003000"
    save_policy(str(override), selected, {})

    original = initialize_native_learner(fresh, full.with_suffix(".npz"), None, None)
    assert original["opt_pi"]["t"] == 31 and "rise_teacher" not in original
    restored = initialize_native_learner(fresh, full.with_suffix(".npz"), override.with_suffix(".npz"), None, .5)
    for k in ("it", "env_steps", "wm", "amp", "critic", "opt_wm"):
        np.testing.assert_array_equal(restored[k], source[k])
    for k in ("pi", "avg", "obs_norm", "opt_pi"):
        assert jax.tree.structure(restored[k]) == jax.tree.structure(selected[k])
        for x, y in zip(jax.tree.leaves(restored[k]), jax.tree.leaves(selected[k])):
            np.testing.assert_array_equal(x, y)
    for k in ("pi", "obs_norm"):
        for x, y in zip(jax.tree.leaves(restored["rise_teacher"][k]), jax.tree.leaves(selected[k])):
            np.testing.assert_array_equal(x, y)
    save_policy(str(tmp_path / "inference"), restored, {})
    with np.load(tmp_path / "inference.npz") as archive:
        assert set(k.split("/")[0] for k in archive.files) == {"avg", "live", "obs_norm"}


def test_rise_anchor_weights_by_poststep_source_pose_and_valid_ppo_rows():
    from types import SimpleNamespace
    from instantamp.train import InstantAMP

    ref = SimpleNamespace(lengths=jnp.array([3, 2]), offsets=jnp.array([0, 3]),
                          poses=jnp.array([[.72], [.38], [.66], [.68], [.5]]))
    clip = jnp.array([0, 0, 1, 1, 1])
    start = jnp.array([0, 1, 0, 0, 0])
    t = jnp.array([0, 0, 0, 1, 0])
    weights = g1_getup.rise_anchor_weight(ref, clip, start + t + 1, jnp.array([.5, .7, 1., .8, .8]))
    np.testing.assert_allclose(weights, [.25, 0., 1., 0., .64])
    xs = InstantAMP.__new__(InstantAMP)
    xs.cfg = parse(["--task", g1_getup.TASK_ID, "--init_policy", "actor.npz", "--reference_weight", "8", "--rise_anchor", "2"])
    xs.act = lambda pi, norm, obs, prev: pi * obs
    obs = jnp.array([[2.], [40.], [1.], [50.], [60.]])
    batch = {"obs": obs, "a_ode": obs * 0, "a": obs, "mu": obs, "anchor_mu": obs * 0,
             "adv": jnp.zeros(5), "mask": jnp.array([1., 1., 1., 1., 0.]), "rise_anchor_weight": weights}
    (loss, stats), grad = jax.value_and_grad(xs.ppo_loss, has_aux=True)(1., None, batch)
    np.testing.assert_allclose([stats["rise_anchor_mse"], loss, grad], [1.6, 3.2, 6.4])
    assert parse(["--task", g1_getup.TASK_ID]).rise_anchor == 0.
    with pytest.raises(ValueError, match="real PPO"):
        parse(["--task", g1_getup.TASK_ID, "--rise_anchor", "1"])


def test_leggedlab_amp_normalizer_uses_policy_windows():
    st = amp.init(jax.random.key(11), frames=2, hidden=(8,), frame_dim=2)
    policy = jnp.asarray([[[-2., 4.], [2., 4.]], [[0., 4.], [0., 4.]]])
    expert = policy + 20.
    st, _ = amp.train_step(st, expert, policy, jnp.ones(2), jax.random.key(7),
                           .001, recovery=True, leggedlab=True)
    np.testing.assert_allclose(st["norm"]["mean"], [0., 4.], atol=1e-6)
    result = amp.recovery_features(st["norm"], jnp.array([[0., 5.]]), leggedlab=True)
    np.testing.assert_allclose(result, [[0., 100.]], rtol=1e-5)


def test_actor_resets_and_amp_fetch_continuous_native_motion(tmp_path):
    # Deliberately give the 50 Hz copy different data. The native 30 Hz knots
    # identify time in x and joint 0, and rotate at a constant world yaw rate.
    (tmp_path / "native").mkdir()
    for directory, fps, offset in ((tmp_path, 50, 100), (tmp_path / "native", 30, 0)):
        t = np.arange(fps + 1, dtype=np.float32) / fps
        pos = np.stack((t + offset, 0*t, .3 + .2*t), -1)
        quat = np.stack((0*t, 0*t, np.sin(t/2), np.cos(t/2)), -1)
        joints = np.tile(g1_getup.DEFAULT_Q, (len(t), 1))
        joints[:, 0] = t + offset
        np.savez(directory / "clip.npz", fps=fps, root_pos=pos, root_quat=quat,
                 root_lin_vel=pos*0, root_ang_vel=pos*0, dof_pos=joints, dof_vel=joints*0)
    sim = Sim(8, robot="g1_getup", floor_dir=tmp_path, physics=False)
    key = jax.random.key(18)
    qpos, _, _ = sim._fresh(key, sim.world(key))
    phase = np.asarray(qpos[:, 0])
    assert ((phase >= 0) & (phase <= .8)).all()
    assert np.any(abs(phase*30 - np.rint(phase*30)) > .05)
    np.testing.assert_allclose(qpos[:, 7], phase, atol=1e-6)
    np.testing.assert_allclose(qpos[:, 3], np.cos(phase/2), atol=1e-6)

    for window in (10, 24):
        obs = amp.Demos(tmp_path, sim, window).sample(key, 8)
        joint = obs[..., 9]  # 6 local orientation + 3 body gyro + joint positions
        assert (joint[:, 0] <= 1 - window * .02).all()
        np.testing.assert_allclose(np.diff(joint, axis=1), .02, atol=1e-6)
