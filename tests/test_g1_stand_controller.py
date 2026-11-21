from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from instantamp.inference import load, policy_memory, policy_step, reset_policy_memory
from tools.export_g1_stand_controller import export


class StandControllerTest(unittest.TestCase):
    def test_per_world_latch_uses_latest_ankles_and_resets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = []
            for n in range(3):
                path = root / f"actor{n}.npz"
                fields = {"obs_norm/raw": True, "obs_norm/mean": np.zeros(570), "obs_norm/var": np.ones(570)}
                for tag in ("avg", "live"):
                    fields[tag + "/0/w"] = np.zeros((570, 29), np.float32)
                    fields[tag + "/0/b"] = np.full(29, 2*n, np.float32)
                np.savez(path, **fields)
                files.append(path)
            result = export(*files, [.5, .6, .5, .62, .94, .75, 1.], root / "controller")
            controller, norm, ode = load(result)
            self.assertIsNone(ode)
            obs = jnp.zeros((2, 570)).at[:, 39].set(1.).at[:, 44].set(1.)
            # Old feet samples would switch both worlds if the five histories were misread.
            obs = obs.at[:, 482].set(-.85)
            obs = obs.at[0, 554].set(-.6)
            step = jax.jit(lambda x, mem: policy_step(controller, norm, x, mem))
            action, memory = step(obs, policy_memory(controller, obs))
            np.testing.assert_array_equal(action[:, 0], [1., 0.])
            obs = obs.at[0, 554].set(0.)
            action, memory = step(obs, memory)
            np.testing.assert_array_equal(action[:, 0], [2., 0.])
            obs = obs.at[0, 554].set(-.68).at[0, 557].set(-.67)
            action, memory = step(obs, memory)
            np.testing.assert_array_equal(action[:, 0], [4., 0.])
            np.testing.assert_array_equal(reset_policy_memory(memory, jnp.array([True, False])), 0.)

    def test_stance_filter_uses_encoders_and_preserves_action_history(self):
        from instantamp import g1_getup
        from instantamp.inference import stance_observation_filter

        transform = stance_observation_filter()
        widths = g1_getup.ACTOR_TERMS
        history = [np.zeros((5, w), np.float32) for w in widths]
        history[2][:] = np.linspace(-.15, .15, 29)
        history[4][:] = np.arange(5 * 29).reshape(5, 29)
        history[5][:] = 99.
        history[0][:, 0] = np.arange(5)  # a constant angular acceleration
        obs = jnp.asarray(np.concatenate([h.ravel() for h in history]))
        actual = jax.jit(transform)(obs)
        values = np.split(np.asarray(actual), 5 * np.cumsum(widths)[:-1])
        np.testing.assert_allclose(values[0].reshape(5, -1), history[0], atol=1e-6)
        np.testing.assert_array_equal(values[2].reshape(5, -1), history[2])
        np.testing.assert_array_equal(values[4].reshape(5, -1), history[4])
        terms = values[5].reshape(5, 6, 3)
        self.assertTrue(np.all(np.abs(terms) < 1.))
        self.assertLess(terms[0, 0, 2], -.4)
        np.testing.assert_allclose(jax.jit(transform)(jnp.stack([obs, obs])), np.stack([actual, actual]), atol=1e-6)

    def test_quiet_stance_confirmation_and_recapture(self):
        from instantamp.inference import G1StandController

        # The shared policy architecture is irrelevant to these state transitions.
        actors = tuple([{"w": jnp.zeros((570, 29)), "b": jnp.full(29, n, dtype=jnp.float32)}]
                       for n in (0, 1, 2))
        raw = tuple({"raw": True} for _ in actors)
        limits = jnp.array([.5, .6, 1., .62, .94, .75, 1.])
        guard = jnp.array([3., .52, .86, 1.65])
        # Model an encoder-derived ankle measurement with a different depth from the noisy landmarks.
        def filter_feet(obs):
            return obs.at[..., 554].set(-.67).at[..., 557].set(-.67)
        policy = G1StandController(actors, limits, filter_feet, guard)
        obs = jnp.zeros((2, 570)).at[:, 39].set(1.).at[:, 44].set(1.)
        step = jax.jit(lambda x, mem: policy_step(policy, raw, x, mem))
        action, mem = step(obs, policy_memory(policy, obs))
        # Filtered depth alone cannot advance the initial rise.
        np.testing.assert_array_equal(action[:, 0], [0., 0.])
        obs = obs.at[:, 554].set(-.56)
        action, mem = step(obs, mem)
        np.testing.assert_array_equal(mem, [[1., 0., 1.], [1., 0., 1.]])
        # An interruption restarts only the affected world's confirmation.
        gyroscope = obs.at[1, 12].set(1.)
        action, mem = step(gyroscope, mem)
        np.testing.assert_array_equal(mem[:, 2], [2., 0.])
        action, mem = step(obs, mem)
        np.testing.assert_array_equal(action[:, 0], [2., 1.])
        # Leaving the entry band is tolerated after takeover. Severe instability returns control immediately.
        action, mem = step(obs.at[0, 12].set(1.), mem)
        np.testing.assert_array_equal(action[:, 0], [2., 1.])
        action, mem = step(obs.at[0, 12].set(2.), mem)
        np.testing.assert_array_equal(mem[0], [1., 0., 0.])
        np.testing.assert_array_equal(action[:, 0], [1., 2.])
        for _ in range(3):
            action, mem = step(obs, mem)
        np.testing.assert_array_equal(action[:, 0], [2., 2.])
        np.testing.assert_array_equal(reset_policy_memory(mem, jnp.array([True, False]))[0], [0., 0., 0.])
