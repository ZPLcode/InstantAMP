from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from instantamp import amp


def test_support_window_uses_bilateral_clearance_and_motion_during_support():
    raw = jnp.zeros((5, amp.SUPPORT_FRAME_DIM)).at[:, 3].set(-1.)
    raw = raw.at[:, 0].set(jnp.linspace(.20, .40, 5))
    planted = amp.support_window(raw, .06)
    lifted = amp.support_window(raw.at[:, -1].set(.3), .06)
    held = amp.support_window(raw.at[:, 0].set(.20), .06)
    reversed_rise = amp.support_window(raw[::-1], .06)
    assert planted.shape == (5, amp.SUPPORT_FEATURE_DIM)
    assert planted[-1, -3] > lifted[-1, -3] > 0
    assert planted[-1, -2] > held[-1, -2] == 0
    np.testing.assert_allclose(planted[-1, -2], -reversed_rise[-1, -2], atol=1e-6)
    # Below-floor source intersections cannot be used to identify a demonstration.
    np.testing.assert_array_equal(planted, amp.support_window(raw.at[:, -6:].set(-.05), .06))
    gradient = jax.grad(lambda x: amp.support_window(x, .06).sum())(raw)
    assert np.isfinite(gradient).all()


def test_opening_windows_never_read_other_clips_and_cover_short_histories():
    bank = object.__new__(amp.Demos)
    bank.frames = jnp.array([10, 11, 12, 13, 14, 20, 21, 22])
    bank.window = 3
    bank.clip_offsets = jnp.array([0, 5])
    bank.clip_lengths = jnp.array([3, 1])
    bank.clip_probabilities = jnp.array([0., 1.])
    samples = np.asarray(bank.sample_opening_windows(jax.random.PRNGKey(314), 128))
    assert set(map(tuple, samples)) == {(20, 20, 20), (20, 20, 21), (20, 21, 22)}


def test_checkpoint_migration_decodes_source_before_replacing_the_recovery_head(tmp_path):
    from instantamp import support_discriminator
    from instantamp.train import load_full, mixed_recovery_restore_template, save_full

    sim = SimpleNamespace(amp_joints=range(23))
    cfg = SimpleNamespace(amp_frames=2, recovery_amp_loss="legged_support", recovery_amp_seconds=4., recovery_amp_stride=3)
    old_cfg = {"recovery_amp_loss": "wasserstein", "recovery_amp_seconds": 4., "recovery_amp_stride": 3}
    gait = {"net": jnp.array([2.])}
    old = {"amp": gait, "amp_up": {"net": jnp.array([3.])}, "it": jnp.asarray(15250)}
    save_full(str(tmp_path / "old"), old)
    k = jax.random.PRNGKey(0)
    dual = jax.eval_shape(lambda key: {
        "native": amp.init(key, 2, frame_dim=73),
        "support": support_discriminator.init(key, 2, amp.SUPPORT_FEATURE_DIM),
    }, k)
    fresh = {**old, "amp_up": dual}
    template, replacement = mixed_recovery_restore_template(fresh, cfg, old_cfg, sim)
    loaded = load_full(template, tmp_path / "old.npz")
    assert loaded["amp_up"]["net"][0] == 3 and loaded["it"] == 15250
    assert replacement is dual
    same, replacement = mixed_recovery_restore_template(fresh, cfg, vars(cfg), sim)
    assert same is fresh and replacement is None
    cfg_legacy = SimpleNamespace(**{**vars(cfg), "recovery_amp_seconds": 0.})
    reverse, replacement = mixed_recovery_restore_template(old, cfg_legacy, vars(cfg), sim)
    assert replacement is old["amp_up"]
    assert jax.tree.structure(reverse["amp_up"]) == jax.tree.structure(dual)
    assert all(a.shape == b.shape for a, b in zip(jax.tree.leaves(reverse["amp_up"]), jax.tree.leaves(dual)))


def test_native_discriminator_permutation_restores_joint_and_landmark_order():
    from instantamp import zenbot
    from tools.convert_zenbot_amp_discriminator import feature_order

    manifest = {"joint_names": [j + "_joint" for j in reversed(zenbot.DOF)], "key_body_names": zenbot.KEY_BODIES[::-1]}
    index = feature_order(manifest)
    np.testing.assert_array_equal(index[:9], np.arange(9))
    np.testing.assert_array_equal(index[9:32], np.arange(31, 8, -1))
    np.testing.assert_array_equal(index[32:55], np.arange(54, 31, -1))
    np.testing.assert_array_equal(index[55:].reshape(6, 3), np.arange(55, 73).reshape(6, 3)[::-1])
