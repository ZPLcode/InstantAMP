"""CPU regressions for the G1 recovery reward and its critic context."""
import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from instantamp import env
from instantamp.train import InstantAMP, initialize_g1_style, load_full, parse, save_full


def reward_model(*args):
    # Only exercise the reward functions; a GPU physics model is unnecessary.
    trainer = InstantAMP.__new__(InstantAMP)
    trainer.cfg = parse(list(args))
    trainer.sim = SimpleNamespace(recovery=trainer.cfg.robot == "zenbot23_getup")
    return trainer


def test_floor_start_gates_and_critic_handover():
    style = reward_model("--g1_style", "1", "--getup_duration_s", "3")
    baseline = reward_model()
    # Upright recovery tail, upright locomotion start, last active step, first
    # inactive step, and a fallen robot that still has time before its deadline.
    t = jnp.array([50, 50, 150, 151, 50])
    up = jnp.array([True, True, True, True, False])
    floor_start = jnp.array([True, False, True, True, True])
    np.testing.assert_array_equal(style.getup_gate(up, floor_start, t), [1, 0, 1, 0, 1])
    np.testing.assert_array_equal(baseline.getup_gate(up, None, t), [0, 0, 0, 0, 1])

    v = jnp.zeros((len(t), 3))
    q = jnp.zeros((len(t), env.J))
    state = v, jnp.broadcast_to(jnp.eye(3), (len(t), 3, 3)), v, v, q, q
    args = (state, v[:, 2], q, v, jnp.zeros((len(t), 4)), up, t)
    legacy = baseline.critic_in(*args)
    current = jax.jit(style.critic_in)(*args, floor_start)
    assert current.shape == legacy.shape
    np.testing.assert_array_equal(current[:, :-1], legacy[:, :-1])
    down_left = 1 - 50 / env.GETUP_STEPS
    np.testing.assert_allclose(legacy[:, -1], [0, 0, 0, 0, down_left], atol=1e-6)
    np.testing.assert_allclose(current[:, -1], [2/3, 0, 0, 0, down_left], atol=1e-6)


def test_lsgan_scale_is_specific_to_getup_head():
    style = reward_model("--g1_style", "1", "--getup_reward_scale", "20")
    baseline = reward_model()
    zenbot = reward_model("--robot", "zenbot23_getup")
    assert baseline.cfg.g1_style == zenbot.cfg.g1_style == 0
    # This mock linear head returns the window's only feature as its logit.
    logits = jnp.array([-1., 0., 1.])
    windows = logits.reshape(-1, 1, 1)
    head = {"net": [{"w": jnp.ones((1, 1)), "b": jnp.zeros(1)}],
            "norm": {"mean": jnp.zeros(1), "var": jnp.array([1. - 1.e-4])},
            "mu": jnp.zeros(2)}
    np.testing.assert_allclose(style.getup_style(head, windows), [0, 15, 20])
    original = 0.5 * (1 + np.tanh(logits / baseline.cfg.amp_style_scale))
    np.testing.assert_allclose(baseline.getup_style(head, windows), original, atol=1e-6)
    np.testing.assert_allclose(style.style(head, windows), original, atol=1e-6)  # walking head
    np.testing.assert_allclose(zenbot.style(head, windows), [0, 37.5, 50.])
    np.testing.assert_allclose(zenbot.getup_style(head, windows), 0.5 * (1 + np.tanh(logits / 50.)), atol=1e-6)


def test_disabled_recovery_head_releases_locomotion_and_invalid_duration_fails_early():
    style = reward_model("--g1_style", "1", "--amp_getup", "0")
    up, starts, t = jnp.array([True, False]), jnp.ones(2, bool), jnp.array([10, 80])
    np.testing.assert_array_equal(style.getup_gate(up, starts, t), [0, 0])
    v, q = jnp.zeros((2, 3)), jnp.zeros((2, env.J))
    state = v, jnp.broadcast_to(jnp.eye(3), (2, 3, 3)), v, v, q, q
    args = (state, v[:, 2], q, v, v, up, t)
    np.testing.assert_array_equal(style.critic_in(*args, starts), reward_model().critic_in(*args))
    for value in ("0", "-1", "nan", "inf"):
        with pytest.raises(ValueError, match="duration"):
            reward_model("--g1_style", "1", "--getup_duration_s", value)


def test_mild_getup_score_rewards_standing_only_after_an_entire_stable_window():
    style = reward_model("--g1_style", "1", "--getup_reward_scale", "4", "--getup_hold_score", "1", "--getup_style_ramp", "500")
    style.sim.amp_joints = np.arange(29)
    frames, features = 10, 78
    head = {"net": [{"w": jnp.zeros((frames * features, 1)), "b": jnp.array([-1.])}],
            "norm": {"mean": jnp.zeros(features), "var": jnp.ones(features)}, "ramp": 0.5}
    x = jnp.zeros((4, frames, features)).at[..., -2].set(0.75).at[..., 63].set(-1.)
    x = x.at[1, 0, -2].set(0.5).at[2, 4, -1].set(0.4).at[3, 7, 63].set(-0.7)
    np.testing.assert_allclose(style.getup_style(head, x), [2., 0., 0., 0.])


def test_real_action_anchor_preserves_unweighted_behavior_and_balances_rare_floor_states():
    style = reward_model("--g1_style", "1", "--getup_anchor", "1", "--init_g1_full", "baseline.npz")
    style.act = lambda pi, norm, obs, previous: previous + pi * obs
    b = {"obs": jnp.ones((4, 1)), "a_ode": jnp.full((4, 1), 2.),
         "a": jnp.full((4, 1), 2.), "mu": jnp.full((4, 1), 2.),
         "adv": jnp.zeros(4), "mask": jnp.array([1., 1., 1., 0.]),
         "down": jnp.array([True, False, False, True]), "early": jnp.array([True, True, False, False])}
    loss, _ = style.ppo_loss(0., {}, b)
    assert loss == 0.
    b["anchor_mu"] = jnp.array([[1.], [2.], [2.], [90.]])
    loss, aux = style.ppo_loss(0., {}, b)
    unit_kl = 1. / (2 * style.cfg.action_noise**2)
    np.testing.assert_allclose(loss, 0.5 * 0.35 * unit_kl)
    np.testing.assert_allclose(aux["anchor_kl"], unit_kl / 3.)
    b["early"] = jnp.zeros(4, bool)
    np.testing.assert_allclose(style.ppo_loss(0., {}, b)[0], 0.5 * unit_kl)
    assert jax.grad(lambda pi: style.ppo_loss(pi, {}, b)[0])(0.) > 0.


def test_frozen_teacher_survives_a_full_checkpoint(tmp_path):
    legacy = {k: jnp.array([float(i)]) for i, k in enumerate(("pi", "avg", "wm", "wm_b", "obs_norm", "amp", "amp_up", "critic"))}
    legacy_path = str(tmp_path / "legacy")
    save_full(legacy_path, legacy)
    fresh = {**legacy, "anchor": {"pi": jnp.zeros(1), "obs_norm": jnp.zeros(1)}, "amp_up": jnp.zeros(2)}
    carry = initialize_g1_style(fresh, legacy_path + ".npz")
    np.testing.assert_array_equal(carry["amp_up"], [0., 0.])
    changed = {**carry, "pi": carry["pi"] + 1., "obs_norm": carry["obs_norm"] + 2.}
    saved = str(tmp_path / "adapted")
    save_full(saved, changed)
    restored = load_full(fresh, saved + ".npz")
    np.testing.assert_allclose(restored["anchor"]["pi"], legacy["avg"])
    np.testing.assert_allclose(restored["anchor"]["obs_norm"], legacy["obs_norm"])
    np.testing.assert_allclose(restored["pi"], legacy["avg"] + 1.)
