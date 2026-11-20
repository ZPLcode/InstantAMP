import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from instantamp import amp, env
from instantamp.g1_phase_amp import G1PhaseDemos, update_paired
from instantamp.getup_amp import GetupDemos
from instantamp.rot import quat_to_mat
from instantamp.sim import Sim


def test_g1_paired_window_matches_physics_after_yaw_and_translation(tmp_path):
    for clip, n in enumerate((13, 15)):
        t = np.arange(n, dtype=np.float32)
        v = np.tile([.45, -.3, .1], (n, 1)).astype(np.float32)
        p = np.array([15.2, -3.5, .2]) + t[:, None] * env.DT * v
        ang = .3 + .04 * t
        quat = np.column_stack((0*t, 0*t, np.sin(ang/2), np.cos(ang/2)))
        q = np.broadcast_to(env.DEFAULT_Q, (n, env.J)).copy()
        q[:, 0] += t * .012 + .04 * clip
        q[:, 5:] += .4 * np.sin(.35*t[:, None])
        w = np.zeros_like(v)
        w[:, 2] = .04 / env.DT
        np.savez(tmp_path / f"rise_{clip}.npz", fps=50, root_pos=p, root_quat=quat, dof_pos=q,
                 dof_vel=np.zeros_like(q), root_lin_vel=v, root_ang_vel=w)
    sim = Sim(4, walk_dir=str(tmp_path), floor_dir=str(tmp_path), physics=False, reset_floor=.9,
              g1_style=True, reference_weight=10., getup_reset_frames=7)
    bank = G1PhaseDemos(sim, GetupDemos(tmp_path, sim, 4, segmented=True))
    yaw = -.62
    rot = np.asarray([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0., 0., 1.]], np.float32)
    first = 3
    elapsed = 7
    end = first + elapsed
    with np.load(tmp_path / "rise_1.npz") as clip:
        section = slice(end - 3, end + 1)
        origin = clip["root_pos"][first].copy()
        origin[2] = 0.
        state = ((clip["root_pos"][section] - origin) @ rot.T,
                 rot @ quat_to_mat(clip["root_quat"][section][:, [3, 0, 1, 2]]),
                 clip["root_lin_vel"][section] @ rot.T, clip["root_ang_vel"][section] @ rot.T,
                 clip["dof_pos"][section], clip["dof_vel"][section])
    feat = bank.frame(state)
    raw = jnp.broadcast_to(feat, (5, 4, bank.raw_dim))
    clips, starts, steps = jnp.array([1, 1, 1, -1, 0]), jnp.array([3, 3, 3, 3, 4]), jnp.array([7, 2, 12, 7, 8])
    pos, neg, valid = bank.paired_windows(raw, clips, starts, yaw, steps)
    np.testing.assert_array_equal(valid, [True, False, False, False, True])
    np.testing.assert_allclose(pos[0], neg[0], atol=2e-6)
    np.testing.assert_array_equal(pos[..., bank.raw_dim:], neg[..., bank.raw_dim:])
    assert pos.shape == (5, 4, 93)
    np.testing.assert_allclose(pos[0, -1, 0], env.DEFAULT_Q[0] + end * .012 + .04)
    assert not np.allclose(pos[0, :, 80:82], bank.paired_windows(raw, clips, starts, yaw + .9, steps)[1][0, :, 80:82])
    np.testing.assert_array_equal(bank.condition_robot(raw, clips, starts, yaw, steps)[0], neg)
    disc = bank.init(jax.random.key(8), hidden=(8,))
    assert float(disc["norm"]["count"]) > 0 and float(disc["norm"]["var"][83]) > 0
    def style(delta):
        shifted = raw.at[0, :, -2].add(delta)
        win, mask = bank.condition_robot(shifted, clips, starts, yaw, steps)
        return jnp.where(mask, amp.recovery_style(disc, win, 1.), 0.).sum()
    assert np.isfinite(jax.grad(style)(.001))

    # Saturate the classifier head. The frozen first-layer auxiliary still
    # differentiates a misplaced physical trajectory and peaks at exact alignment.
    last = dict(disc["net"][-1])
    last.update(w=jnp.zeros_like(last["w"]), b=jnp.full_like(last["b"], -9.))
    saturated = {**disc, "net": [*disc["net"][:-1], last]}
    def feature_style(delta):
        value, ok = bank.matched_feature_reward(saturated, raw.at[..., -2].add(delta),
                                                clips, starts, yaw, steps)
        return value[0]
    assert float(amp.recovery_style(saturated, neg[0], 1.)) == 0.
    np.testing.assert_allclose(feature_style(0.), 1., atol=1e-4)
    assert 0. < float(feature_style(.02)) < .999
    grad = float(jax.grad(feature_style)(.02))
    assert np.isfinite(grad) and abs(grad) > 1.e-5
    wrong_score, _ = bank.matched_feature_reward(saturated, raw, clips, starts, yaw, steps - 3)
    assert float(wrong_score[0]) < .999
    inactive, _ = bank.matched_feature_reward(saturated, raw, clips, starts, yaw, steps + 40)
    np.testing.assert_array_equal(inactive, 0.)
    wrong, distinct = bank.time_shifted_expert(clips, starts, steps, -3)
    assert bool(distinct[0]) and not bool(distinct[1])
    np.testing.assert_array_equal(wrong[..., bank.raw_dim:], pos[..., bank.raw_dim:])
    np.testing.assert_allclose(wrong[0, -1, 0], env.DEFAULT_Q[0] + (end-3)*.012 + .04)
    np.testing.assert_array_equal(bank.time_shifted_expert(clips, starts, steps, 0)[1], False)


def test_identical_valid_origins_for_both_amp_classes():
    st = amp.init(jax.random.key(0), 2, hidden=(8,), frame_dim=3)
    expert = jnp.stack([jnp.full((2, 3), -100.), jnp.ones((2, 3)), jnp.full((2, 3), 100.)])
    robot = expert.at[1].set(3.)
    step = jax.jit(lambda st, keep: update_paired(st, expert, robot, keep, jax.random.key(32), 1e-3))
    updated, metrics = step(st, jnp.array([False, True, False]))
    np.testing.assert_allclose(updated["norm"]["mean"], 2.)
    np.testing.assert_allclose(metrics["valid_fraction"], 1 / 3)
    frozen, _ = step(updated, jnp.zeros(3, bool))
    for old, new in zip(jax.tree.leaves(updated), jax.tree.leaves(frozen)):
        np.testing.assert_array_equal(new, old)


def test_full_orientation_survives_horizontal_fallen_pose_and_reset_yaw(tmp_path):
    n = 17
    source_yaw = .23 + np.arange(n)*.23
    c, s = np.cos(source_yaw/2), np.sin(source_yaw/2)
    # Pitch places the pelvis forward axis precisely vertical. Its left axis
    # still determines a stable horizontal basis at the source reset frame.
    quat = np.column_stack((-s, c, s, c)) / np.sqrt(2.)
    p = np.broadcast_to([1.5, -.8, .25], (n, 3)).copy()
    p[:, 0] += np.arange(n)*.02
    v = np.broadcast_to([1., 0., 0.], p.shape)
    ang = np.broadcast_to([0., 0., .23/env.DT], p.shape)
    q = np.broadcast_to(env.DEFAULT_Q, (n, env.J))
    np.savez(tmp_path / "horizontal.npz", fps=50, root_pos=p, root_quat=quat, dof_pos=q,
             dof_vel=np.zeros_like(q), root_lin_vel=v, root_ang_vel=ang)
    sim = Sim(1, walk_dir=str(tmp_path), floor_dir=str(tmp_path), physics=False, reset_floor=1.,
              g1_style=True, g1_phase_amp=True)
    demos = GetupDemos(tmp_path, sim, 4, segmented=True)
    old = G1PhaseDemos(sim, demos)
    bank = G1PhaseDemos(sim, demos, heading_features=True)
    assert old.frame_dim == 93 and bank.frame_dim == 99
    np.testing.assert_allclose(bank.frames[:, :82], old.frames)
    initial, end, aug = 2, 8, 2.67
    section = slice(end - 3, end + 1)
    def yaw_rotation(rad):
        c, s = np.cos(rad), np.sin(rad)
        return jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    rot = sim.reference.rotation[section]
    yaw = yaw_rotation(aug)
    origin = p[initial].copy()
    origin[2] = 0.
    state = ((p[section] - origin) @ yaw.T, yaw @ rot, v[section] @ yaw.T,
             ang[section] @ yaw.T, q[section], np.zeros_like(q[section]))
    expert, policy, valid = bank.paired_windows(bank.frame(state), 0, initial, aug, end-initial)
    assert valid
    np.testing.assert_allclose(policy, expert, atol=2e-6)
    np.testing.assert_allclose(policy[:, 82:88].reshape(-1, 3, 2).transpose(0, 2, 1)
                               @ policy[:, 82:88].reshape(-1, 3, 2),
                               np.broadcast_to(np.eye(2), (4, 2, 2)), atol=1e-6)
    wrong_rot = yaw_rotation(1.3) @ state[1]
    wrong_state = (state[0], wrong_rot, state[2], state[3], state[4], state[5])
    wrong, ok = bank.condition_robot(bank.frame(wrong_state), 0, initial, aug, end-initial)
    assert ok and np.max(np.abs(wrong[:, 82:88] - expert[:, 82:88])) > .8
    np.testing.assert_allclose(wrong[:, 88:], expert[:, 88:])
    st = bank.init(jax.random.key(11), hidden=(32,))
    true_score, _ = bank.matched_feature_reward(st, bank.frame(state), 0, initial, aug, end-initial)
    wrong_score, _ = bank.matched_feature_reward(st, bank.frame(wrong_state), 0, initial, aug, end-initial)
    assert true_score > .9999 and wrong_score < .99
    shifted, distinct = bank.time_shifted_expert(0, initial, end-initial, 3)
    assert distinct
    shifted_frames = slice(end, end + 4)
    reset_basis = yaw_rotation(-source_yaw[initial])
    expected_axes = (reset_basis @ sim.reference.rotation[shifted_frames])[:, :, :2].reshape(-1, 6)
    expected_xy = ((p[shifted_frames] - p[initial]) @ reset_basis.T)[:, :2]
    np.testing.assert_allclose(shifted[:, 82:88], expected_axes, atol=1e-6)
    np.testing.assert_allclose(shifted[:, bank.xy], expected_xy, atol=1e-6)
    np.testing.assert_array_equal(shifted[:, 88:], expert[:, 88:])
