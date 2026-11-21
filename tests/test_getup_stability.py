import numpy as np

from instantamp.stability import stability


def test_fall_recovery_and_supported_stance_on_one_timeline():
    dt = 0.02
    poses = np.zeros((301, 36))
    poses[:, 3] = 1.0
    poses[1:, 2] = 0.75
    poses[76:101, 2] = 0.4
    poses[101:, 0] = 0.12
    feet = np.ones(300)
    feet[-15:] = 0.0  # 135/150 loaded samples in the final three seconds.
    ended = np.zeros(300, dtype=bool)
    supports = np.zeros((300, 1))
    supports[75:85, 0] = 0.1
    measures = stability(poses, feet, ended, dt, support_bw=supports, support_names=["left_wrist_yaw_link"])
    assert measures["rise_s"] == 0.02
    assert measures["post_rise_falls"] == [{"fall_s": 1.52, "recovered_s": 2.02}]
    assert measures["longest_supported_upright_s"] == 4.0
    assert measures["final_3s_supported_upright"]
    assert measures["final_3s_xy_drift"]["max_cm"] == 0.0
    assert measures["after_settling_xy_drift"]["max_cm"] == 12.0
    assert measures["post_rise_limb_load"]["left_wrist_yaw_link"]["first_sustained_s"] == 1.52
    feet[-16:] = 0
    assert not stability(poses, feet, ended, dt)["final_3s_supported_upright"]
    ended[86] = True
    censored = stability(poses, feet, ended, dt)
    assert censored["post_rise_falls"] == [{"fall_s": 1.52, "recovered_s": None}]
    assert censored["final_3s_xy_drift"] is None
    assert not censored["final_3s_supported_upright"]
