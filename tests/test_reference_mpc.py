import numpy as np
import pytest

from instantamp import env
from instantamp.reference_mpc import PlanningConfig, servo_target
from instantamp.sim import build_scene
from tools.score_g1_tracking import score_tracking


def test_target_realizes_commanded_feedback_before_motor_limits():
    rng = np.random.default_rng(51)
    q, v, ref_q, ref_v, offset = rng.normal(0., .1, size=(5, 3, env.J))
    config = PlanningConfig()
    target = servo_target(q, v, ref_q, ref_v, offset, config)
    torque = env.KP * (target - q) - env.KD * v
    expected = env.KP * config.position_gain * (ref_q + offset - q) + env.KD * config.velocity_gain * (ref_v - v)
    np.testing.assert_allclose(torque, expected, rtol=1e-6, atol=1e-5)
    with pytest.raises(ValueError):
        PlanningConfig(stride=25, horizon=24)


def test_same_clock_score_handles_quaternion_sign_and_early_reset():
    m = build_scene(g1_support_friction=True)
    ref = np.tile(np.r_[0., 0., .72, 1., 0., 0., 0., env.DEFAULT_Q], (60, 1))
    ref[:12, 2] = .32
    pose = ref.copy()
    pose[:, 3:7] *= -1
    exact = score_tracking(m, ref, pose)
    assert exact["rise_time_error_s"] == 0
    assert exact["completed_reference_window"] and exact["final_half_second_upright"]
    assert max(exact["rise_and_settle"].values()) < 1e-7
    ended = np.zeros(len(pose) - 1, bool)
    ended[3] = True
    stopped = score_tracking(m, ref, pose, ended=ended)
    assert not stopped["rose_by_deadline"] and not stopped["completed_reference_window"]
    assert stopped["rise_and_settle"]["pelvis_height_rmse_cm"] > 20
