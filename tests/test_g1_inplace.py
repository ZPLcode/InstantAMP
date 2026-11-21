import unittest

import jax
import jax.numpy as jnp
import numpy as np

from instantamp import g1_inplace


class FirstRiseTests(unittest.TestCase):
    def test_first_continuous_rise_keeps_its_original_location_until_episode_end(self):
        state = g1_inplace.initial(2)
        contact = jnp.ones((2, 2), bool)
        rotation = jnp.broadcast_to(jnp.eye(3), (2, 3, 3))
        for frame in range(16):
            x = np.array([frame / 100.0, -frame / 100.0])
            z = [0.72 if frame > 2 else 0.4, 0.4 if frame in (3, 14) else 0.72]
            root = jnp.stack((x, jnp.zeros(2), jnp.array(z)), axis=1)
            feet = jnp.repeat(root[:, None, :], 2, axis=1)
            state, _ = g1_inplace.score(
                state,
                (root, rotation, jnp.zeros((2, 3))),
                feet,
                contact,
                jnp.array([False, False]),
            )
        np.testing.assert_array_equal(state.rose, [True, True])
        np.testing.assert_array_equal(state.fell, [False, True])
        np.testing.assert_allclose(state.root[:, 0], [0.03, -0.04], atol=1e-7)
        np.testing.assert_allclose(state.peak, [0.12, 0.11], atol=1e-7)
        reset, _ = g1_inplace.score(
            state,
            (root, rotation, jnp.zeros((2, 3))),
            feet,
            contact,
            jnp.array([True, False]),
        )
        np.testing.assert_array_equal(reset.rose, [False, True])
        np.testing.assert_array_equal(reset.fell, [False, True])
        np.testing.assert_array_equal(reset.root[0], [0.0, 0.0])

    def test_exact_and_out_of_distribution_sensor_evidence(self):
        obs = jnp.zeros((5, 570)).at[:, 39].set(1.0).at[:, 44].set(1.0)
        obs = obs.at[:, 161].set(jnp.arange(5) * 0.5)
        keys = np.asarray(g1_inplace.imitation_keys(obs))
        actions = np.broadcast_to(np.arange(5).astype(np.float32)[:, None], (5, 29))
        model = g1_inplace.LandingLibrary.read(
            {
                "in_place_demo/keys": keys,
                "in_place_demo/actions": actions,
                "in_place_demo/radius": np.array(0.1),
            }
        )
        command, confidence, _ = jax.jit(model.target)(obs)
        np.testing.assert_array_equal(command, actions)
        np.testing.assert_array_equal(confidence, 1.0)
        self.assertEqual(float(model.target(obs[0].at[161].set(4.0))[1]), 0.0)
        np.testing.assert_array_equal(model.residual_gate(obs), 0.0)
