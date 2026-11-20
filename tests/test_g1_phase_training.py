import jax
import jax.numpy as jnp
import numpy as np
import pytest

from instantamp import buffer
from instantamp.train import G1StyleConfig


def test_recovery_replay_samples_only_complete_recent_native_windows():
    times = jnp.arange(10)[:, None]
    q = jnp.broadcast_to(times[:, :, None], (10, 3, 1)).astype(jnp.float32)
    bank = {"s_qpos": q, "s1_qpos": q + 1, "s_qvel": q * 0, "s1_qvel": q * 0,
            "reference_clip": jnp.broadcast_to(jnp.array([0, 0, -1]), (10, 3)),
            "reference_start": jnp.broadcast_to(jnp.array([0, 15, 0]), (10, 3)),
            "t": jnp.broadcast_to(times, (10, 3)),
            "done": jnp.zeros((10, 3)), "ok": jnp.ones((10, 3)),
            "ptr": jnp.array(0), "size": jnp.array(10)}
    batch = buffer.sample_windows(bank, jax.random.PRNGKey(5), 32, 3, recent=4,
                                  recovery_lengths=jnp.array([8]))
    np.testing.assert_array_equal(batch["mask"], 1.)
    assert (np.asarray(batch["reference_clip"]) == 0).all()
    assert (np.asarray(batch["reference_start"]) == 0).all()
    end = np.asarray(batch["t"][:, -1])
    assert ((end >= 5) & (end <= 7)).all()
    np.testing.assert_array_equal(np.diff(batch["t"], axis=1), 1)


def test_phase_discriminator_requires_identical_source_bank_and_learned_hold():
    with pytest.raises(ValueError, match="same segmented native bank"):
        G1StyleConfig(g1_phase_share=.7, amp_getup_dir="a_different_bank")
    with pytest.raises(ValueError, match="learned standing score"):
        G1StyleConfig(g1_phase_share=.7, getup_hold_score=.8)
