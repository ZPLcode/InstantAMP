"""Legged G1 A/B feature correspondence, source timing and shared initialization."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from instantamp import amp, env, nets
from instantamp.g1_phase_amp import update_paired
from instantamp.getup_amp import GetupDemos, LEGGED_BODIES, LeggedRecoveryPrior
from instantamp.rot import quat_to_mat
from instantamp.sim import Sim
from instantamp.train import parse


def test_legged_clock_pair_and_equal_initial_discriminator(tmp_path):
    n = 24
    t = np.arange(n, dtype=np.float32)
    p = np.stack((.8 + .012*t, -.4 + .002*t, .25 + .03*t), -1)
    q = np.broadcast_to(env.DEFAULT_Q, (n, env.J)).copy()
    q[:, 8] += .014*t
    qd = np.zeros_like(q)
    qd[:, 8] = .7
    yaw, roll = -.32 + .005*t, 1.02 - .022*t
    cy, sy, cr, sr = np.cos(yaw/2), np.sin(yaw/2), np.cos(roll/2), np.sin(roll/2)
    quat = np.stack((cy*sr, sy*sr, sy*cr, cy*cr), -1)
    omega = np.tile([.6, .2, -.1], (n, 1))
    np.savez(tmp_path / 'rise.npz', fps=50, root_pos=p, root_quat=quat, dof_pos=q, dof_vel=qd,
             root_lin_vel=np.zeros_like(p), root_ang_vel=omega)
    sim = Sim(2, walk_dir=str(tmp_path), floor_dir=str(tmp_path), physics=False, reset_floor=1.,
              g1_style=True, g1_phase_amp=True, getup_reset_frames=1)
    demos = GetupDemos(tmp_path, sim, 10, segmented=True, legged=True)
    rotated = quat_to_mat(quat[:, [3, 0, 1, 2]])
    # Independent engine verifies the choice and order of all six body origins.
    mj = mujoco.MjData(sim.m)
    mj.qpos[:] = np.concatenate((p[0], quat[0, [3,0,1,2]], q[0]))
    mujoco.mj_forward(sim.m, mj)
    points = mj.xpos[[sim.m.body('robot_'+name).id for name in LEGGED_BODIES]]
    world_to_body = np.asarray(rotated[0])
    expected = np.concatenate(([1, 0, 0], [0, -np.sin(roll[0]), np.cos(roll[0])],
                               omega[0] @ world_to_body, q[0], qd[0], ((points - p[0]) @ world_to_body).ravel()))
    np.testing.assert_allclose(demos.frames[0], expected, atol=3e-6)
    # Shared policy FK and native arrays must agree, even after reset translation and yaw augmentation.
    angle = 1.7
    c, s = np.cos(angle), np.sin(angle)
    rz = np.array([[c,-s,0], [s,c,0], [0,0,1]], dtype=np.float32)
    state = ((p - np.array([p[4,0], p[4,1], 0])) @ rz.T, rz @ rotated,
             np.zeros_like(p), omega @ rz.T, q, qd)
    features = demos.policy_frames(state)
    np.testing.assert_allclose(features[:, :sim.amp_dim], sim.amp_frame(state), atol=3e-6)
    np.testing.assert_allclose(features[:, sim.amp_dim:], demos.frames, atol=3e-6)
    raw = jnp.broadcast_to(features[5:15, sim.amp_dim:], (5, 10, 85))
    clip, start, elapsed = jnp.array([0, 0, 0, -1, 0]), jnp.array([4, 4, 4, 4, 4]), jnp.array([10, 8, 9, 10, 20])
    a, b = (LeggedRecoveryPrior(sim, demos, timed) for timed in (False, True))
    ea, ra, active = a.pair(raw, clip, start, elapsed)
    eb, rb, valid = b.pair(raw, clip, start, elapsed)
    np.testing.assert_array_equal(active, [True, False, True, False, False])
    np.testing.assert_array_equal(active, valid)
    np.testing.assert_allclose(ea[0], ra[0], atol=3e-6)
    np.testing.assert_array_equal(ra, rb[..., :-1])
    np.testing.assert_array_equal(ea, eb[..., :-1])
    np.testing.assert_array_equal(eb[..., -1], rb[..., -1])
    np.testing.assert_allclose(rb[0, :, -1], np.arange(5,15)/(n-1))
    np.testing.assert_allclose(ea[2,-1], demos.frames[13])
    da, db = (prior.init(jax.random.key(3), hidden=(12,8)) for prior in (a, b))
    def predict(disc, frames):
        x = amp.normalize(disc['norm'], frames).reshape(len(frames), -1)
        return nets.mlp_apply(disc['net'], x, jax.nn.leaky_relu)
    np.testing.assert_allclose(predict(da, ra), predict(db, rb), atol=1e-6)
    np.testing.assert_allclose(b.style(db, raw, clip, start, elapsed, 3),
                               np.where(active, amp.recovery_style(db, rb, 3), 0.))
    np.testing.assert_array_equal(db['net'][0]['w'].reshape(10,86,-1)[:, -1], 0.)
    updated, _ = update_paired(db, eb, rb.at[..., 17].add(.4), valid, jax.random.key(7), 1e-4)
    assert np.max(np.abs(updated['net'][0]['w'].reshape(10,86,-1)[:, -1])) > 0
    assert sim.obs_dim == 384 and sim.J == 29


def test_cli_ab_and_old_defaults():
    assert parse([]).ckpt_iterations == 0
    assert parse(['--g1_style','1']).g1_getup_prior == 'native'
    for mode in ('legged', 'legged_time'):
        cfg = parse(['--g1_style','1','--g1_getup_prior', mode,'--ckpt_iterations','400'])
        assert cfg.ckpt_iterations == 400 and cfg.reference_weight == cfg.g1_phase_share == 0.
        with pytest.raises(ValueError, match='Legged G1 AMP'):
            parse(['--g1_style','1','--g1_getup_prior',mode,'--reference_weight','1'])
