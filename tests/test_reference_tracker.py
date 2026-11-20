import mujoco
import numpy as np

from instantamp import env
from instantamp.reference_tracker import Motion, motor_control
from instantamp.sim import build_scene


def test_reference_contact_settings_provide_tangential_ground_support():
    for enabled, palm_dimension in [(False, 1), (True, 3)]:
        model = build_scene(g1_support_friction=enabled)
        data = mujoco.MjData(model)
        # Place every collision shape below the plane so all contact-pair
        # dimensions can be inspected without depending on a motion file.
        data.qpos[2] = -1.
        mujoco.mj_forward(model, data)
        contacts = {model.geom(c.geom2).name: c.dim for c in data.contact if c.geom1 == model.geom("floor").id}
        assert contacts["robot_left_hand_collision"] == palm_dimension
        assert contacts["robot_left_shin_collision"] == palm_dimension
        assert contacts["robot_left_foot1_collision"] == 3


def test_source_servo_uses_native_motor_torque_and_speed_limits():
    rng = np.random.default_rng(58)
    q = rng.normal(size=(4, env.J))
    qd = rng.uniform(-38., 38., size=q.shape)
    actions = rng.normal(size=q.shape) * 5.
    target, ctrl = motor_control(q, qd, actions)
    np.testing.assert_allclose(ctrl, np.asarray(env.actuator_ctrl(q, qd, target)), rtol=2e-5, atol=2e-5)


def test_motion_end_keeps_pose_and_zeros_velocity():
    motion = Motion(np.arange(3 * 36).reshape(3, 36), np.ones((3, 35)))
    pose, vel = motion.frames(6)
    np.testing.assert_array_equal(pose[3:], np.tile(motion.qpos[-1], (3, 1)))
    np.testing.assert_array_equal(vel[:3], motion.qvel)
    np.testing.assert_array_equal(vel[3:], 0.)
