"""CPU checks of the environment without the GPU simulator: the scene, the actuator model against MuJoCo, forward kinematics with
velocities against MuJoCo, and the shapes of the observation, the reward terms and the AMP frame.

    JAX_PLATFORMS=cpu python -m pytest tests
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from instantamp import amp, env  # noqa: E402
from instantamp.sim import (  # noqa: E402
    VelKinematics,
    build_scene,
    collider_points,
    unflatten,
)


def random_state(m, rng):
    d = mujoco.MjData(m)
    d.qpos[:] = 0
    d.qpos[2] = 0.78
    q = rng.normal(size=4)
    d.qpos[3:7] = q / np.linalg.norm(q)
    d.qpos[7:] = env.DEFAULT_Q + 0.3 * rng.normal(size=env.J)
    d.qvel[:] = rng.normal(size=m.nv)
    mujoco.mj_forward(m, d)
    return d


def test_scene():
    m = build_scene()
    assert m.nu == env.J and m.nq == 7 + env.J
    assert tuple(m.joint(m.actuator_trnid[i, 0]).name.removeprefix("robot_").removesuffix("_joint") for i in range(m.nu)) == env.DOF
    assert env.OBS_DIM == 384


def test_actuator_matches_mujoco():
    m = build_scene()
    rng = np.random.default_rng(0)
    for _ in range(5):
        d = random_state(m, rng)
        q, qd = jnp.asarray(d.qpos[7:], jnp.float32), jnp.asarray(d.qvel[6:], jnp.float32)
        tgt = jnp.asarray(env.DEFAULT_Q + 0.3 * rng.normal(size=env.J), jnp.float32)
        d.ctrl[:] = np.asarray(env.actuator_ctrl(q, qd, tgt))
        mujoco.mj_forward(m, d)
        tau = np.clip(np.asarray(env.actuator_torque(q, qd, tgt)), -env.EFFORT, env.EFFORT)
        np.testing.assert_allclose(d.actuator_force, tau, atol=1e-3)


def test_velocity_kinematics_matches_mujoco():
    m = build_scene()
    names = ["robot_torso_link", "robot_left_ankle_roll_link", "robot_right_wrist_yaw_link"]
    kin = VelKinematics(m, names)
    rng = np.random.default_rng(1)
    for _ in range(5):
        d = random_state(m, rng)
        S = unflatten(jnp.asarray(d.qpos[None], jnp.float32), jnp.asarray(d.qvel[None], jnp.float32))
        P, R, V, W = kin.forward_vel(*S)
        for i, n in enumerate(names):
            b = m.body(n).id
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, vel, 0)  # at the body origin, world frame
            np.testing.assert_allclose(np.asarray(P[0, i]), d.xpos[b], atol=1e-4)
            np.testing.assert_allclose(np.asarray(R[0, i]), d.xmat[b].reshape(3, 3), atol=1e-4)
            np.testing.assert_allclose(np.asarray(W[0, i]), vel[:3], atol=1e-3)
            np.testing.assert_allclose(np.asarray(V[0, i]), vel[3:], atol=1e-3)


def test_default_pose_stands_on_the_floor():
    m = build_scene()
    feet = [m.body(f"robot_{s}_ankle_roll_link").id for s in ("left", "right")]
    foot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in feet and m.geom_contype[g] > 0]
    assert len(foot_geoms) == env.N_FOOT_GEOMS
    kin = VelKinematics(m, [m.body(b).name for b in feet])
    local, frame, radius = collider_points(m, foot_geoms, feet)
    qpos = np.zeros(m.nq, np.float32)
    qpos[2], qpos[3], qpos[7:] = env.DEFAULT_HEIGHT, 1.0, env.DEFAULT_Q
    S = unflatten(jnp.asarray(qpos[None]), jnp.zeros((1, m.nv)))
    P, R, _, _ = kin.forward_vel(*S)
    from instantamp.rot import mv3

    z = np.asarray((P[0, frame] + mv3(R[0, frame], local))[:, 2]) - radius
    assert abs(z.min()) < 0.03, z.min()  # the soles touch the floor within a few cm


def test_rewards_and_amp_frame_shapes():
    N = 8
    k = jax.random.PRNGKey(0)
    S = (jnp.zeros((N, 3)).at[:, 2].set(0.78), jnp.tile(jnp.eye(3), (N, 1, 1)), jnp.zeros((N, 3)), jnp.zeros((N, 3)),
         jnp.tile(jnp.asarray(env.DEFAULT_Q), (N, 1)), jnp.zeros((N, env.J)))  # fmt: skip
    cmd = env.command(env.sample_cmd(k, N), S[1])
    frame = env.obs_frame(S, jnp.zeros((N, env.J)), cmd, k)
    assert frame.shape == (N, env.FRAME_DIM)
    kin = env.Kin(S[1], S[2], S[3], jnp.zeros((N, 2, 3)))
    terms = env.reward_terms(S, S, kin, jnp.zeros((N, env.J)), jnp.zeros((N, env.J)), cmd, jnp.zeros(N, bool), jnp.zeros(N), jnp.ones((N, 2)),
                             jnp.zeros(N), jnp.asarray(np.stack([-np.ones(env.J), np.ones(env.J)], -1) * 3))  # fmt: skip
    assert set(terms) == set(env.TERMS)
    assert all(v.shape == (N,) and bool(jnp.isfinite(v).all()) for v in terms.values())
    assert float(terms["rise"].max()) == 0.0  # up: no rise penalty
    amp.use_wrists(True)
    f = amp.frame(S, jnp.zeros((N, 4, 3)))
    assert f.shape == (N, amp.FRAME_DIM) == (N, 76)
