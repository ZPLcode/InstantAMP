import unittest

import jax
import jax.numpy as jnp
import numpy as np

from instantamp.g1_tracking_capture import TrackingCapture


class TrackingCaptureTest(unittest.TestCase):
    def archive(self):
        count = 3
        observations = np.zeros((count, 96), np.float32)
        observations[:, [3, 8]] = 1.0
        qpos = np.zeros((count, 36), np.float64)
        qpos[:, 3] = 1.0
        fields = dict(
            observations=observations,
            qpos=qpos,
            qvel=np.zeros((count, 35)),
            action=np.full((count, 29), 0.3, np.float32),
            gain=np.zeros((count, 29, 70)),
            support_jacobian=np.zeros((count, 3, 32)),
            stance_parameters=np.ones((count, 2)),
            start=np.zeros(count, np.int32),
            end=np.full(count, count - 1, np.int32),
            settings=np.array([0.5, 1.0, 1.0, 0.75, 0.0]),
            residual_w=np.zeros((96, 29)),
            residual_b=np.ones(29),
            residual_scale=np.array(0.1),
        )
        return {"tracking_capture/" + key: value for key, value in fields.items()}

    def test_batched_exact_sensor_and_distant_sensor_keep_independent_state(self):
        for observed in (False, True):
            with self.subTest(observed=observed):
                archive = self.archive()
                if observed:
                    archive["tracking_capture/observer"] = np.zeros(4)
                    archive["tracking_capture/capture"] = np.full(3, 2, np.int32)
                tracker = TrackingCapture.from_archive(archive)
                obs = np.zeros((2, 570), np.float32)
                obs[:, [39, 44]] = 1
                obs[1, 161] = 3  # Beyond the permitted acquisition distance.
                fallback = jnp.ones((2, 29), jnp.float32)
                zero = tracker.memory(obs)
                output, memory = jax.jit(tracker.step)(obs, fallback, zero)
                np.testing.assert_allclose(output[0], 0.3 + 0.1 * np.tanh(1), rtol=1e-6)
                np.testing.assert_array_equal(output[1], fallback[1])
                self.assertEqual(memory[0, 0], 1)
                np.testing.assert_array_equal(memory[1], 0)
                np.testing.assert_array_equal(zero, 0)

    def test_sensor_branch_settings_require_metadata(self):
        archive = self.archive()
        archive["tracking_capture/observer"] = np.ones(4)
        with self.assertRaisesRegex(ValueError, "observer"):
            TrackingCapture.from_archive(archive)
        archive["tracking_capture/capture"] = np.full(3, 2)
        archive["tracking_capture/capture_search"] = np.array([4, 3, 20])
        archive["tracking_capture/branch_probe"] = np.array([2, 3])
        with self.assertRaisesRegex(ValueError, "selection"):
            TrackingCapture.from_archive(archive)

    def test_crouch_extension_uses_simulation_ticks(self):
        heights = [TrackingCapture._ramp_height(0.55, tick, 50) for tick in range(61)]
        self.assertEqual(heights[0], 0.55)
        self.assertAlmostEqual(heights[25], (0.55 + 0.748) / 2)
        np.testing.assert_array_equal(heights[50:], 0.748)
        self.assertTrue(np.all(np.diff(heights) >= 0.0))
        self.assertEqual(TrackingCapture._ramp_height(0.55, 0, 0), 0.748)


if __name__ == "__main__":
    unittest.main()
