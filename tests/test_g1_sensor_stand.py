import unittest

import jax
import mujoco
import numpy as np

from instantamp import g1_getup
from instantamp.g1_sensor_stand import SensorStance
from instantamp.native_getup import observation_terms


class SensorStanceTest(unittest.TestCase):
    def test_jitted_capture_preserves_native_storage_and_per_environment_action(self):
        weights = np.zeros((14, 7), np.float32)
        weights[-1, 0] = .3
        weights[0, 0] = .9
        policy = SensorStance.read({"quiet_pose/weight": weights})
        data = mujoco.MjData(policy.model)
        data.qpos[7:] = g1_getup.DEFAULT_Q
        mujoco.mj_forward(policy.model, data)
        links = [policy.model.body("robot_" + name).id for name in g1_getup.KEY_BODIES]
        terms = observation_terms(data, links, np.zeros(29))
        observation = np.concatenate([np.tile(term, 5) for term in terms]).astype(np.float32)
        obs = np.stack([observation] * 3)
        targets = np.zeros((3, 29), np.float32)
        targets[2, 3] = 3.
        step = jax.jit(policy)
        for _ in range(3):
            actions = np.asarray(step(obs, np.array([True, False, True]), targets))
            self.assertTrue(np.isfinite(actions).all())
            np.testing.assert_array_equal(actions[1], np.zeros(29))
            self.assertGreater(np.max(np.abs(actions[0] - actions[2])), 1.e-4)
        missing = {"quiet_pose/weight": np.zeros((13, 7))}
        with self.assertRaisesRegex(ValueError, "14-by-7"):
            SensorStance.read(missing)


if __name__ == "__main__":
    unittest.main()
