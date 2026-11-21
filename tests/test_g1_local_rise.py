import unittest

import jax
import jax.numpy as jnp
import numpy as np

from instantamp.g1_rise import LocalRise, features


class LocalRiseTest(unittest.TestCase):
    def test_known_neighborhood_blend_fallback_and_row_order(self):
        # Encode a single joint in the newest frame. Unit quaternions put the root upright.
        samples = jnp.zeros((3, 570)).at[:, 39].set(1.0).at[:, 44].set(1.0)
        samples = samples.at[:, 161].set(jnp.array([0.0, 9.0, 18.0]))
        keys, poses = features(samples, 1.0)
        arrays = dict(
            keys=np.asarray(keys),
            poses=np.asarray(poses),
            actions=np.tile(np.array([1.0, 2.0, 3.0], dtype=np.float32)[:, None], (1, 29)),
            gain=np.zeros((3, 29, 64), dtype=np.float16),
            settings=np.array([1.0, 2.0, 2.0], dtype=np.float32),
        )
        policy = LocalRise.read(arrays)
        fallback = jnp.full((3, 29), -4.0, dtype=jnp.float32)
        obs = samples.at[1, 161].set(2.5).at[2, 161].set(65.0)
        action = jax.jit(policy.apply)(obs, fallback)
        np.testing.assert_allclose(action[0], 1.0)
        self.assertTrue(np.all((action[1] > -4.0) & (action[1] < 1.0)))
        np.testing.assert_array_equal(action[2], fallback[2])
        np.testing.assert_array_equal(policy.apply(obs[0], fallback[0]), action[0])
        permuted = LocalRise.read({k: v if k == "settings" else v[[2, 0, 1]] for k, v in arrays.items()})
        np.testing.assert_allclose(permuted.apply(obs, fallback), action, atol=1e-6)

    def test_local_handoff_uses_only_the_observed_neighborhood(self):
        obs = jnp.zeros((3, 570)).at[:, 39].set(1.0).at[:, 44].set(1.0)
        obs = obs.at[:, 161].set(jnp.array([0.0, 1.0, 2.0]))
        keys, poses = features(obs, 0.65)
        arrays = dict(
            keys=np.asarray(keys),
            poses=np.asarray(poses),
            actions=np.zeros((3, 29), np.float32),
            gain=np.zeros((3, 29, 64), np.float16),
            settings=np.array([0.65, 2.0, 2.0], np.float32),
            handoff_keys=np.asarray(keys[1:2]),
            handoff_settings=np.array([0.4, 0.60, 0.78, 0.12], np.float32),
        )
        policy = LocalRise.read(arrays)
        gate = jax.jit(policy.handoff_thresholds)(obs, jnp.array([0.53, 0.55, 0.15]))
        np.testing.assert_allclose(gate, [[0.53, 0.60, 0.53], [0.55, 0.78, 0.55], [0.15, 0.12, 0.15]])

    def test_amp_residual_starts_at_zero_and_preserves_quiet_stance(self):
        from instantamp.g1_rise import rise_residual
        from instantamp.inference import G1StandController, policy_step

        x = jnp.zeros((3, 570)).at[:, 39].set(1.0).at[:, 44].set(1.0)
        keys, poses = features(x, 0.65)
        arrays = dict(
            keys=np.asarray(keys),
            poses=np.asarray(poses),
            actions=np.zeros((3, 29), np.float32),
            gain=np.zeros((3, 29, 64), np.float16),
            settings=np.array([0.65, 2.0, 2.0], np.float32),
        )
        residual = (jnp.zeros((172, 8)), jnp.zeros(8), jnp.zeros((8, 87)), jnp.zeros(87), jnp.array(0.5))
        np.testing.assert_array_equal(rise_residual(residual, x), 0.0)
        residual = (*residual[:3], residual[3].at[:29].set(1.0), residual[4])
        for k, v in zip(("w0", "b0", "w1", "b1", "scale"), residual):
            arrays["residual/" + k] = np.asarray(v)
        policy = LocalRise.read(arrays)
        np.testing.assert_allclose(policy.apply(x, jnp.zeros((3, 29))), 0.5 * np.tanh(1.0), rtol=1e-6)
        actors = tuple([{"w": jnp.zeros((570, 29)), "b": jnp.full(29, float(i))}] for i in range(2))
        bundle = G1StandController(actors, jnp.array([0.53, 0.55, 1.0]), local_rise=policy)
        action, _ = policy_step(bundle, ({"raw": True}, {"raw": True}), x, jnp.ones((3, 1)))
        np.testing.assert_array_equal(action, 1.0)

    def test_triple_action_rate_preserves_constant_bonus_and_curve(self):
        from instantamp.train_g1_rise_residual import action_rate_reward

        slew = jnp.array([0.0, 0.1, 1.0, 2.0, 20.0])
        original, p1 = action_rate_reward(slew, 1.0)
        triple, p3 = action_rate_reward(slew, 3.0)
        np.testing.assert_allclose(original, 2.0 / (1.0 + slew), rtol=1e-6)
        np.testing.assert_allclose(triple, 2.0 + 3.0 * (original - 2.0), rtol=1e-6)
        np.testing.assert_array_equal(p3, 3.0 * p1)
        self.assertEqual(float(triple[0]), 2.0)

    def test_training_snapshot_restores_optimizer_rng_and_physics_history(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from instantamp.train_g1_rise_residual import load_training_state, save_training_state

        rng = jax.random.key(37)
        physics = (jnp.array([[0.2, 0.5]], dtype=jnp.float32), jnp.array([9], dtype=jnp.int32))
        state = (physics, {"optimizer": jnp.array([0.3, -0.9])}, rng)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "training.npz"
            save_training_state(path, 500, 81.125, state)
            iteration, elapsed, loaded = load_training_state(path, state)
            self.assertEqual((iteration, elapsed), (500, 81.125))
            for a, b in zip(jax.tree.leaves(state[:2]), jax.tree.leaves(loaded[:2])):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(jax.random.normal(loaded[2], (16,)), jax.random.normal(rng, (16,)))
            with self.assertRaises(ValueError):
                load_training_state(path, (physics, {"optimizer": jnp.zeros(3)}, rng))

    def test_aligned_ankle_calibration_changes_only_qualified_worlds(self):
        from instantamp.inference import G1StandController, policy_memory, policy_step

        obs = jnp.zeros((2, 570)).at[:, 39].set(1.0).at[:, 44].set(1.0)
        # Both worlds reach 0.52 m, but only the first has both ankles at approximately equal height.
        obs = obs.at[:, 554].set(-0.52).at[0, 557].set(-0.51).at[1, 557].set(-0.35)
        keys, poses = features(jnp.concatenate([obs, obs[:1]], axis=0), 0.65)
        bank = LocalRise.read(
            dict(
                keys=np.asarray(keys),
                poses=np.asarray(poses),
                actions=np.zeros((3, 29), np.float32),
                gain=np.zeros((3, 29, 64), np.float16),
                settings=np.array([0.65, 2.0, 2.0], np.float32),
                aligned_handoff=np.array([0.96, 0.06, 0.51, 0.15], np.float32),
            )
        )
        actors = tuple([{"w": jnp.zeros((570, 29)), "b": jnp.full(29, float(i))}] for i in (0, 1))
        policy = G1StandController(actors, jnp.array([0.53, 0.55, 0.15]), local_rise=bank)
        _, memory = policy_step(policy, ({"raw": True}, {"raw": True}), obs, policy_memory(policy, obs))
        np.testing.assert_allclose(memory[:, 0], [0.15, 0.0])
