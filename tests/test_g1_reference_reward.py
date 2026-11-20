"""Phase alignment, reset metadata and actor independence for G1 AMP training."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np

from instantamp import env
from instantamp.sim import Sim
from instantamp.train import parse


def native_clip(path, clip_id):
    count = 9
    t = np.arange(count, dtype=np.float32)
    position = np.stack((4 + .08 * t, -2 + .05 * t, .21 + .06 * clip_id + .008 * t), -1)
    angle = .23 * t
    quat = np.stack((0*t, np.sin(angle/2), 0*t, np.cos(angle/2)), -1)
    q = np.tile(env.DEFAULT_Q, (count, 1)).astype(np.float32)
    q[:, 0] += .13 * t
    np.savez(path, fps=50, root_pos=position, root_quat=quat, root_lin_vel=np.zeros((count, 3)),
             root_ang_vel=np.zeros((count, 3)), dof_pos=q, dof_vel=np.zeros_like(q))


def test_g1_reference_stays_on_reset_clock_and_world_yaw(tmp_path):
    for i in range(2):
        native_clip(tmp_path / f"rise_{i}.npz", i)
    sim = Sim(64, walk_dir=str(tmp_path), floor_dir=str(tmp_path), physics=False, reset_floor=.7,
              prob_shove=0., g1_style=True, reference_weight=25., getup_reset_frames=4)
    ref = sim.reference
    (qpos, _, _), (ci, first, yaw) = sim._fresh(jax.random.PRNGKey(32), sim.world(jax.random.PRNGKey(42)), return_source=True)
    valid = np.array(ci >= 0)
    assert valid.any() and (~valid).any()
    assert np.all(np.array(first[valid]) < 4)
    index = np.array(ref.offsets[ci[valid]] + first[valid])
    np.testing.assert_allclose(qpos[valid, 2], ref.position[index, 2] + .03)
    step = jnp.full_like(first, 2)
    # Transform an exact target by the same sampled heading and reset-frame translation.
    head = np.zeros((len(ci), 3, 3))
    c, s = np.cos(yaw), np.sin(yaw)
    head[:, 0, 0] = c
    head[:, 1, 1] = c
    head[:, 0, 1] = -s
    head[:, 1, 0] = s
    head[:, 2, 2] = 1
    ci0 = np.maximum(ci, 0)
    origins = np.array(ref.position[ref.offsets[ci0] + first])
    origins[:, 2] = 0.
    idx = ref.offsets[ci0] + first + step
    p = (head @ (np.array(ref.position[idx]) - origins)[..., None])[..., 0]
    r = head @ np.array(ref.rotation[idx])
    q = ref.joints[idx]
    points = sim.key_positions(p, r, q)
    st = (p, r, None, None, q, None)
    scorer = jax.jit(ref.score)
    expected = scorer(st, points, ci, first, yaw, step)
    np.testing.assert_allclose(expected, valid.astype(float), atol=2e-6)
    shifted = scorer(st, points, ci, first, yaw, step + 1)
    assert (np.array(shifted)[valid] < .99).all()
    np.testing.assert_array_equal(scorer(st, points, ci, first, yaw, step + 20), 0.)
    grad = jax.grad(lambda dp: scorer((p + dp, r, None, None, q, None), points, ci, first, yaw, step).sum())(jnp.full_like(qpos[:, :3], .08))
    assert jnp.isfinite(grad).all() and jnp.any(grad != 0)
    assert sim.obs_dim == env.OBS_DIM and sim.frame_dim == env.FRAME_DIM
    assert sim.terms[-1] == "motion_reference" and sim.terms[:-1] == env.TERMS


def test_reference_recipe_is_opt_in():
    c = parse(["--g1_style", "1"])
    assert c.reference_weight == c.g1_support_friction == 0 and c.getup_reset_frames == 20
    c = parse(["--g1_style", "1", "--reference_weight", "30", "--g1_support_friction", "1", "--getup_reset_frames", "1"])
    assert c.reference_weight == 30 and c.g1_support_friction == c.getup_reset_frames == 1


def test_g1_phase_amp_zero_scalar_and_native_episode_boundaries(tmp_path):
    native_clip(tmp_path / "rise.npz", 0)
    sim = Sim(3, walk_dir=str(tmp_path), floor_dir=str(tmp_path), physics=False, g1_style=True,
              g1_phase_amp=True, reference_weight=0., reset_floor=.5)
    assert sim.g1_reference and not sim.reference_reward
    assert sim.terms == env.TERMS
    clip = jnp.array([0, 0, -1])
    first = jnp.array([2, 2, 0])
    np.testing.assert_array_equal(sim.recovery_timeout(clip, first, jnp.array([56, 57, 1000])),
                                  [False, True, False])
    cmd = env.sample_cmd(jax.random.PRNGKey(5), 3)._replace(time_left=jnp.zeros(3))
    fresh = env.tick(cmd, jax.random.PRNGKey(2))
    filtered = sim.recovery_command(fresh, clip >= 0)
    np.testing.assert_array_equal(filtered.vel[:2], 0)
    np.testing.assert_array_equal(filtered.is_standing[:2], True)
    for original, overridden in zip(fresh, filtered):
        np.testing.assert_array_equal(original[-1], overridden[-1])

    rot = jnp.broadcast_to(jnp.eye(3), (3, 3, 3))
    pos = jnp.broadcast_to(jnp.array([0., 0., .8]), (3, 3))
    v = jnp.zeros_like(pos)
    q = jnp.broadcast_to(jnp.asarray(env.DEFAULT_Q), (3, sim.J))
    zero = jnp.zeros_like(q)
    state = (pos, rot, v, v, q, zero)
    terms = sim.reward_terms(state, state, sim.view_state(state).kin, zero, zero, v,
                             jnp.zeros(3, bool), jnp.zeros(3), jnp.ones((3, 2)), jnp.zeros(3), sim.q_soft,
                             clip=clip, reference_start=first, step=jnp.array([1, 7, 1]))
    for name in ("track_lin", "track_ang", "body_ang_vel_xy"):
        np.testing.assert_array_equal(terms[name], [0., 1., 1.])
    assert "motion_reference" not in terms
