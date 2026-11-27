"""Contracts shared by the two integrated Zenbot training tasks."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from instantamp import amp, nets, zenbot, zenbot_models
from instantamp.inference import load
from instantamp.train import (
    Config,
    InstantAMP,
    ZenbotJointConfig,
    action_log_prob,
    parse,
    save_policy,
)
from instantamp.zenbot_motion_train_obs import (
    mirror_action,
    motion_train_actor_mean,
)


def test_default_entry_point_uses_the_g1_combined_task():
    config = parse([])
    assert type(config) is Config
    assert config.robot == "g1" and config.ode and config.amp_getup
    assert 0 < config.reset_floor < 1


def test_joint_training_defaults_for_both_models():
    default = parse(["--robot", zenbot_models.ZENBOT23.mixed_task])
    assert isinstance(default, ZenbotJointConfig)
    assert default.robot == zenbot_models.ZENBOT23.mixed_task
    assert 0 < default.reset_floor < 1
    assert default.amp_getup and default.wm_global_gait and default.wm_floor_amp
    assert default.ppo_frac == .1 and default.ode == 0
    assert default.obs_clip == 10 and default.wm_initial_gait_stage == 0
    other = parse(["--robot", zenbot_models.ZENBOT26.mixed_task])
    assert (default.num_envs, default.max_iterations, default.wm_only_iters) == (2048, 20_000, 150)
    assert (default.ckpt_iterations, default.full_ckpt_keep, default.external_pushes) == (500, 0, 0)
    assert (default.getup_amp_scale, default.wm_stand_pose_weight, default.wm_lift_weight) == (0., 0., .5)
    assert (other.num_envs, other.max_iterations, other.wm_only_iters) == (4096, 100_000, 500)
    assert (other.ckpt_iterations, other.full_ckpt_keep, other.external_pushes) == (1000, 2, 1)
    assert (other.getup_amp_scale, other.wm_stand_pose_weight, other.wm_lift_weight) == (10., -2., .2)
    assert {key for key, value in vars(default).items() if value != getattr(other, key)} == {
        "robot", "walk_dir", "amp_dir", "floor_dir", "amp_getup_dir", "log_dir",
        "num_envs", "max_iterations", "wm_only_iters", "ckpt_iterations", "full_ckpt_keep",
        "external_pushes", "getup_amp_scale", "wm_stand_pose_weight", "wm_lift_weight",
    }
    for cfg in (default, other):
        assert cfg.motion_train_randomization == 1
        assert cfg.wm_hidden == "1024,1024,1024"
        assert (cfg.learn_action_std, cfg.action_noise, cfg.action_entropy_coef) == (1, 1., .01)
        assert (cfg.wm_tracking_scale, cfg.gait_ls_scale, cfg.action_rate_l2_weight) == (1., 1.5, -.05)
        assert cfg.amp_lerp == .7
        assert cfg.imag_gait_mirror_weight == .2
        assert (cfg.wm_gait_frames, cfg.gait_amp_stride, cfg.wm_lift_gate) == (10, 1, .44)
        assert (cfg.seed, cfg.minutes, cfg.init_policy, cfg.init_full) == (42, 0., "", "")
        assert cfg.walk_dir == cfg.amp_dir
        assert cfg.floor_dir == cfg.amp_getup_dir
        assert cfg.walk_dir != cfg.floor_dir


@pytest.mark.parametrize("robot", zenbot_models.MIXED_TASKS)
def test_zenbot_help_lists_run_options_and_legacy_overrides_still_work(robot, capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse(["--robot", robot, "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert set(re.findall(r"--[\w-]+", help_text)) == {
        "--help", "--robot", "--log_dir", "--num_envs", "--max_iterations", "--ckpt_iterations",
        "--seed", "--init_policy",
    }
    cfg = parse(["--robot", robot, "--wm_gait_frames", "3", "--getup_amp_scale", "4.5"])
    assert (cfg.wm_gait_frames, cfg.getup_amp_scale) == (3, 4.5)


@pytest.mark.parametrize("flags", [
    ["--robot", zenbot.MIXED_TASK, "--wm_getup_only", "1"],
    ["--robot", zenbot.MIXED_TASK, "--motion_train_wm", "0"],
    ["--robot", "zenbot23_getup"], ["--robot", "zenbot26_getup"],
    ["--robot", "g1_getup"], ["--robot", zenbot.MIXED_TASK, "--legged_real_ppo", "1"],
    ["--robot", "g1", "--g1_style", "1"],
])
def test_separate_training_tasks_are_rejected(flags):
    with pytest.raises(SystemExit):
        parse(flags)


def test_discriminator_waits_for_a_valid_policy_window():
    state = amp.init(jax.random.key(0), 2, frame_dim=4)
    demo = jax.random.normal(jax.random.key(1), (4, 2, 4))
    def update(weights):
        return amp.train_step_with_robot(state, demo, -demo, weights, jax.random.key(2), 1e-4)
    empty, metrics = update(jnp.zeros(4))
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(empty)):
        np.testing.assert_array_equal(before, after)
    assert all(float(v) == 0. for v in metrics.values())
    fitted, _ = update(jnp.array([0., 1., 0., 0.]))
    assert int(fitted["opt"]["t"]) == int(state["opt"]["t"]) + 1


@pytest.mark.parametrize("joints", [23, 26])
def test_motion_train_discriminator_clips_loss_gradient_before_adam_weight_decay(joints):
    width = 3 + 2 * joints + 18
    state = amp.init(jax.random.key(12), 10, hidden=(), frame_dim=width)
    state["net"][0]["w"] = jnp.full((10 * width, 1), 2.)
    demo = jnp.ones((2, 10, width))
    robot = -2. * demo
    weights, key = jnp.ones(2), jax.random.key(13)
    (_, _), raw = jax.value_and_grad(amp.recovery_loss, has_aux=True)(
        state["net"], state["norm"], demo, robot, weights, key, motion_train=True,
    )
    norm = np.sqrt(sum(np.square(np.asarray(g)).sum() for g in jax.tree.leaves(raw)))
    assert norm > 1.
    updated, _ = amp.train_step_with_robot(
        state, demo, robot, weights, key, 1e-4, recovery=True, motion_train=True,
    )
    # First-step Adam moments expose the actual gradient sent to the optimizer.
    clipped = {}
    for name, parameter in state["net"][0].items():
        moment = updated["opt"]["leaves"][0][name]["m"]
        clipped[name] = np.asarray(moment) / .1 - .1 * np.asarray(parameter)
        np.testing.assert_allclose(clipped[name], np.asarray(raw[0][name]) / (norm + 1e-6), atol=5e-7)
    np.testing.assert_allclose(np.sqrt(sum(np.square(g).sum() for g in clipped.values())), 1., atol=5e-6)
    np.testing.assert_allclose(updated["norm"]["mean"], robot.mean(axis=(0, 1)))


@pytest.mark.parametrize("flag,value", [("--ode", "1"), ("--ppo_frac", "1"), ("--ppo_frac", "-.1"),
                                         ("--obs_clip", "3"), ("--amp_frames", "1"),
                                         ("--recovery_amp_seconds", "0"), ("--imag_gait_mirror_weight", "-.1"),
                                         ("--getup_amp_scale", "-1"), ("--getup_amp_scale", "nan"),
                                         ("--wm_lift_weight", "-1"), ("--wm_lift_weight", "nan"),
                                         ("--full_ckpt_keep", "-1"),
                                         ("--imag_gait_mirror_weight", "nan")])
def test_incompatible_world_model_settings_fail_before_training(flag, value):
    with pytest.raises(ValueError):
        parse(["--robot", zenbot.MIXED_TASK, "--motion_train_wm", "1", flag, value])


@pytest.mark.parametrize("joints", [23, 26])
def test_learned_exploration_uses_recorded_behavior_probability(joints):
    learner = InstantAMP.__new__(InstantAMP)
    learner.cfg = ZenbotJointConfig()
    learner.act = lambda pi, norm, obs, prev: jnp.zeros_like(prev)
    old_std = np.linspace(.7, 1.1, joints, dtype=np.float32)
    actions = np.random.default_rng(4).normal(size=(3, joints)).astype(np.float32) * old_std
    adv = np.array([1., -.5, .25])

    def log_prob(a, std):
        return (-.5 * (a / std)**2 - np.log(std * np.sqrt(2 * np.pi))).sum(-1)

    behavior = action_log_prob(jnp.asarray(actions), 0., jnp.asarray(old_std))
    np.testing.assert_allclose(behavior, log_prob(actions, old_std), atol=6e-6)
    batch = dict(a=actions, a_ode=jnp.zeros_like(actions), obs=jnp.zeros((3, 3 * (9 + 3 * joints))),
                 mask=jnp.ones(3), adv=jnp.asarray(adv), behavior_log_prob=behavior, gait_mirror=jnp.ones(3))
    new_std = old_std + np.linspace(-.04, .04, joints, dtype=np.float32)

    def gain(a):
        ratio = np.exp(log_prob(a, new_std) - log_prob(actions, old_std))
        return np.minimum(ratio * adv, np.clip(ratio, .8, 1.2) * adv)

    loss = lambda std, b: learner.ppo_loss(None, None, b, std)[0]
    value, grad = jax.value_and_grad(loss)(jnp.asarray(new_std), batch)
    expected = -.5 * (gain(actions) + gain(np.asarray(mirror_action(actions)))).mean() - .01 * np.log(new_std).sum()
    np.testing.assert_allclose(value, expected, atol=4e-6)
    assert np.isfinite(grad).all() and np.ptp(grad) > .01
    empty = {**batch, "mask": jnp.zeros(3)}
    value, grad = jax.value_and_grad(loss)(jnp.asarray(new_std), empty)
    assert float(value) == 0
    np.testing.assert_array_equal(grad, np.zeros(joints))


@pytest.mark.parametrize("joints,width", [(23, 234), (26, 261)])
def test_exported_dense_actor_preserves_normalization_and_actions(tmp_path, joints, width):
    layers = nets.mlp_init(jax.random.key(41), (width, 16, 12, 8, joints))
    layers[0].update(ln_g=jnp.linspace(.6, 1.1, 16), ln_b=jnp.linspace(-.1, .2, 16))
    norm = nets.rn_init(width)
    norm["mean"] = jnp.linspace(-.5, .5, width)
    path = tmp_path / "policy"
    save_policy(str(path), dict(avg=layers, pi=layers, obs_norm=norm), {"it": 0})
    actor, stats, ode = load(path.with_suffix(".npz"))
    assert ode is None
    obs = np.random.default_rng(4).normal(size=(8, width)).astype(np.float32)
    expected = np.clip((obs - norm["mean"]) / np.sqrt(norm["var"] + .01**2), -10, 10)
    for index, layer in enumerate(layers):
        expected = np.asarray(expected) @ np.asarray(layer["w"]) + np.asarray(layer["b"])
        if index < len(layers) - 1:
            if "ln_g" in layer:
                centered = expected - expected.mean(axis=-1, keepdims=True)
                variance = (centered**2).mean(axis=-1, keepdims=True)
                expected = centered / np.sqrt(variance + 1e-6) * layer["ln_g"] + layer["ln_b"]
            expected = np.maximum(0, expected) + np.expm1(np.minimum(0, expected))
    np.testing.assert_allclose(motion_train_actor_mean(actor, stats, obs), expected,
                               atol=2e-6, rtol=2e-6)
