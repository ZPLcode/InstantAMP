"""Native Zenbot 26-DoF joint model, calibrated drives and independent delay groups."""
import hashlib
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from instantamp import zenbot26
from instantamp.sim import ZENBOT26_XML, Sim, build_scene


def test_scene_matches_native_retarget_articulation():
    # The FRAX bank records this digest for the original 26-DoF MJCF.
    assert hashlib.sha256(Path(ZENBOT26_XML).read_bytes()).hexdigest() == (
        "83af7b1e7efd69532ec240d6e53bfab61e659dd069b57411429348f00cd7b5a9")
    model = build_scene(robot_name=zenbot26.MIXED_TASK)
    assert (model.nq, model.nv, model.nu) == (33, 32, 26)
    hinges = model.actuator_trnid[:, 0]
    assert [model.joint(i).name for i in hinges] == ["robot_" + name + "_joint" for name in zenbot26.DOF]
    np.testing.assert_array_equal(model.jnt_qposadr[hinges], np.arange(7, 33))
    np.testing.assert_array_equal(model.jnt_dofadr[hinges], np.arange(6, 32))
    np.testing.assert_allclose(model.body_mass.sum(), zenbot26.MASS)
    np.testing.assert_allclose(model.actuator_ctrlrange, np.stack((-zenbot26.EFFORT, zenbot26.EFFORT), axis=-1))

    assert zenbot26.DOF[13:16] == ("waist_roll", "waist_pitch", "head_yaw")
    np.testing.assert_allclose(model.jnt_range[hinges[13:16]], [[-.52, .52], [-.52, .52], [-.78, .78]])
    np.testing.assert_allclose(zenbot26.EFFORT[13:16], [120., 120., 12.5])
    np.testing.assert_allclose(zenbot26.SPEED[13:16], [18.3, 18.3, 47.12])
    np.testing.assert_allclose(zenbot26.KP[13:16], [80., 80., 17.226699])
    np.testing.assert_allclose(zenbot26.KD[13:16], [4., 4., .647723889])
    q = jnp.zeros(26)
    np.testing.assert_allclose(zenbot26.actuator_ctrl(q, q, q + 100), zenbot26.EFFORT)
    np.testing.assert_allclose(zenbot26.actuator_ctrl(q, zenbot26.SPEED, q + 100), 0.)


def test_delayed_waist_and_head_actions_at_each_physics_step(tmp_path):
    # A tiny reset clip keeps the test independent of the local motion library.
    root = np.tile([0., 0., .25], (20, 1))
    quat = np.tile([0., 0., 0., 1.], (20, 1))
    q = np.zeros((20, 26))
    np.savez(tmp_path / "floor.npz", fps=50, root_pos=root, root_quat=quat,
             root_lin_vel=root * 0., root_ang_vel=root * 0., dof_pos=q, dof_vel=q)
    sim = Sim(1, robot=zenbot26.MIXED_TASK, walk_dir=tmp_path, floor_dir=tmp_path, physics=False)
    assert sim.motor_delay_groups == 7
    np.testing.assert_array_equal(zenbot26.GROUP[12:16], [0, 5, 5, 6])

    history = np.zeros((1, sim.history, sim.frame_dim), np.float32)
    actions = slice(9 + 2 * sim.J, 9 + 3 * sim.J)
    history[:, -2, actions] = 1.
    history[:, -1, actions] = 2.
    context = jnp.array([[1., 0., 5., 4., 4., 4., 4., 3., 7., 0.]])
    cmd = jnp.full((1, 26), 3.)
    target = sim.control_targets(cmd, jnp.zeros_like(cmd), jnp.asarray(history.reshape(1, -1)), context)
    delivered = (np.asarray(target).reshape(4, 26) - zenbot26.DEFAULT_Q) / zenbot26.ACTION_SCALE
    # Rows follow the 5ms substeps. Columns are waist yaw, roll, pitch and head yaw.
    np.testing.assert_allclose(delivered[:, 12:16], [[1., 2., 2., 1.], [2., 2., 2., 1.],
                                                    [2., 2., 2., 1.], [2., 3., 3., 2.]])
