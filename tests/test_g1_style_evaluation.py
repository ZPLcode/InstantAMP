from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from tools.evaluate_g1_style import foot_load_fraction
from tools.score_g1_tracking import _body_local


def test_sole_support_excludes_upper_body_and_padding_and_accepts_both_floor_orders():
    sim = SimpleNamespace(N=2, floor_id=0, foot_side=np.array([-1, 0, 1, -1]),
                          m=SimpleNamespace(body_mass=np.array([10.])))
    impl = SimpleNamespace(contact__geom=jnp.array([[0, 1], [2, 0], [0, 3], [0, 1]]),
                           contact__worldid=jnp.array([0, 1, 0, 1]), nacon=jnp.array([3]),
                           contact__efc_address=jnp.array([[0, 1], [0, -1], [2, -1], [1, -1]]),
                           efc__force=jnp.array([[20., 30., 60.], [55., 60., 0.]]))
    actual = foot_load_fraction(sim, SimpleNamespace(data=SimpleNamespace(_impl=impl)))
    np.testing.assert_allclose(np.asarray(actual) * 98.1, [50., 55.], rtol=1e-6)


def test_body_frame_landmarks_remove_pitch_and_global_translation():
    poses = np.zeros((2, 36))
    poses[0, 3] = 1.
    poses[1, :3] = [2, 0, 1.]
    poses[1, 3:7] = [np.sqrt(.5), 0., np.sqrt(.5), 0.]
    points = np.array([[[1., 0., 0.]], [[2., 0., 0.]]])
    np.testing.assert_allclose(_body_local(points, poses), [[[1., 0., 0.]], [[1., 0., 0.]]], atol=1e-6)


def test_floor_proximity_includes_proximal_knee_caps_and_raw_clip_times():
    from instantamp import env
    from instantamp.sim import build_scene
    from tools.score_g1_tracking import score_tracking

    model = build_scene()
    pose = np.r_[0., 0., env.DEFAULT_HEIGHT, 1., 0., 0., 0., env.DEFAULT_Q]
    sample = np.stack([pose, pose, pose])
    sample[:2, 2] = -1.  # All six geometry proxies are below the plane.
    score = score_tracking(model, sample, sample)["geometric_floor_proximity_2cm"]
    for name in ("left_knee", "right_knee", "left_hand", "right_hand"):
        assert score[name]["both_frames"] == score[name]["either_frames"] == 2
        assert score[name]["intersection_over_union"] == 1.
        assert score[name]["sim_last_near_s"] == score[name]["reference_last_near_s"] == .02


def test_single_clip_skips_offsets_after_first_lift_even_if_pelvis_drops_again():
    from tools.evaluate_g1_single import fallen_cases
    q = np.zeros((20, 36))
    q[:, 2] = .2
    q[:, 3:7] = [np.sqrt(.5), 0., np.sqrt(.5), 0.]
    q[4, 2] += .05
    assert sorted(set(t for t, _ in fallen_cases(q))) == [0, 3]
    assert sorted(set(t for t, _ in fallen_cases(q, exclusive_end=2))) == [0]


def test_single_clip_hand_release_records_absence_and_retouch():
    from tools.evaluate_g1_single import release_time
    assert release_time(np.array([False]*10)) == (None, "no_sustained_approach")
    assert release_time(np.array([True]*3 + [False]*4)) == (None, "no_confirmed_release")
    mask = np.array([True]*3 + [False]*5 + [True]*4 + [False]*5)
    assert release_time(mask) == (.24, "released")
