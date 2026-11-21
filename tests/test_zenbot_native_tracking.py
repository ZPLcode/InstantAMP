import ctypes as ct
import unittest

import mujoco
import numpy as np

from instantamp import zenbot
from instantamp.sim import build_scene
from instantamp.zenbot_native_tracking import backend, features, ptr


class ZenbotTrackingTest(unittest.TestCase):
    def test_expansion_is_reached_only_after_both_original_radii_fail(self):
        from instantamp.zenbot_native_tracking import NativeTracking

        for original_succeeds in (True, False):
            tracker = NativeTracking.__new__(NativeTracking)
            tracker.db = {"stand_radius": 0.18}
            tracker.position_limit = 0.18
            tracker._cached = lambda data: False
            visited = []

            def select(data, recovered=False, expansion=False):
                if expansion:
                    return ["extra"]
                return ["recovery"] if recovered else ["primary"]

            def plan(data, choices):
                visited.append((choices[0], tracker.position_limit))
                return choices[0] == "extra" or (original_succeeds and choices[0] == "recovery"
                                                  and tracker.position_limit > 0.26)

            tracker.select, tracker.plan_choices = select, plan
            self.assertTrue(tracker.prepare(None))
            expected = [("primary", 0.18), ("recovery", 0.18), ("primary", 0.27), ("recovery", 0.27)]
            self.assertEqual(visited, expected if original_succeeds else expected + [("extra", 0.18)])

    def test_hand_support_must_last_six_frames_before_sustained_knee_extension(self):
        from tools.summarize_zenbot_getup import phase

        joints = np.zeros((40, 23))
        joints[:24, [3, 9]] = 1.2
        palm = np.zeros(40, bool)
        palm[4:9] = True
        palm[25:] = True
        self.assertIsNone(phase(joints, palm, 36)["hand_tick"])
        palm[9] = True
        measured = phase(joints, palm, 36)
        self.assertEqual(measured["hand_tick"], 4)
        self.assertEqual(measured["knee_tick"], 24)
        self.assertAlmostEqual(measured["hand_s"], 0.12)

    def test_predictor_and_viewer_use_the_same_independent_joint_group_delays(self):
        model = build_scene(robot_name="zenbot23_getup")
        observed, predicted = mujoco.MjData(model), mujoco.MjData(model)
        observed.qpos[2] = 0.58
        observed.qpos[7:] = zenbot.DEFAULT_Q
        mujoco.mj_forward(model, observed)
        mujoco.mj_copyData(predicted, model, observed)
        lags = np.ascontiguousarray(np.array([3, 5, 4, 7, 6], dtype="i")[zenbot.GROUP])
        motor = np.ascontiguousarray(
            [zenbot.KP, zenbot.KD, zenbot.EFFORT, zenbot.SPEED], dtype="d"
        )
        old = previous = zenbot.DEFAULT_Q.astype("d")
        rng = np.random.default_rng(132)
        for _ in range(20):
            target = (
                zenbot.DEFAULT_Q + 0.25 * rng.normal(0, 0.3, 23).astype("f")
            ).astype("d")
            backend().motor_step(
                ct.c_void_p(model._address),
                ct.c_void_p(predicted._address),
                ptr(lags),
                ptr(motor),
                ptr(target),
                ptr(previous),
                ptr(old),
            )
            for sub in range(4):
                applied = np.where(
                    sub >= lags, target, np.where(sub + 4 >= lags, previous, old)
                )
                v = observed.qvel[6:]
                tau = np.clip(
                    zenbot.KP * (applied - observed.qpos[7:]) - zenbot.KD * v,
                    -zenbot.EFFORT,
                    zenbot.EFFORT,
                )
                observed.ctrl[:] = np.where(
                    (np.abs(v) >= zenbot.SPEED) & (tau * v > 0), 0.0, tau
                )
                mujoco.mj_step(model, observed)
            np.testing.assert_array_equal(predicted.qpos, observed.qpos)
            np.testing.assert_array_equal(predicted.qvel, observed.qvel)
            old, previous = previous, target

    def test_learned_feedback_inputs_exclude_world_translation(self):
        q = np.r_[np.zeros(3), 1.0, 0.0, 0.0, 0.0, zenbot.DEFAULT_Q]
        v = np.zeros(29)
        u = np.zeros(23)
        expected = features(q, v, u)
        self.assertEqual(expected.shape, (76,))
        np.testing.assert_array_equal(expected[3:6], [0.0, 0.0, 1.0])
        q[:3] = [3.0, 7.0, 0.7]
        np.testing.assert_array_equal(expected, features(q, v, u))
