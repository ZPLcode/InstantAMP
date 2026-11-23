import pytest
import numpy as np
import jax
import jax.numpy as jnp

from instantamp import amp, nets, zenbot
from instantamp.train import ZenbotMotionTrainWMConfig, ZenbotWalkRunRecoveryConfig, parse, save_policy
from instantamp.zenbot_motion_train_obs import motion_train_actor_mean
from tools.train_zenbot_skill_actor import DenseActor


def test_selecting_the_world_model_profile_keeps_the_deployment_and_learner_contract():
    task = ["--robot", zenbot.MIXED_TASK]
    cfg = parse(task + ["--motion_train_wm", "1"])
    assert isinstance(cfg, ZenbotMotionTrainWMConfig)
    assert cfg.ode == cfg.legged_real_ppo == 0 and cfg.ppo_frac == .1
    assert cfg.init_policy == cfg.init_full == cfg.recovery_base_actor == ""
    assert cfg.obs_clip == 10 and cfg.action_rate_l2_weight == -.015
    assert cfg.amp_getup and cfg.amp_w > 0 and cfg.amp_gate == "none"
    assert type(parse(task)) is type(parse(task + ["--motion_train_wm", "0"])) is ZenbotWalkRunRecoveryConfig


def test_hybrid_getup_stage_starts_on_the_ground_and_keeps_the_same_actor():
    cfg = parse(["--robot", zenbot.MIXED_TASK, "--motion_train_wm", "1", "--wm_getup_only", "1", "--wm_motion_replay", "1"])
    assert (cfg.reset_floor, cfg.prob_shove, cfg.push) == (1., 0., 0)
    assert cfg.ppo_frac == .1 and cfg.ppo_w == 1. and cfg.ode == 0
    assert cfg.init_policy == cfg.init_full == ""
    assert parse(["--robot", zenbot.MIXED_TASK, "--motion_train_wm", "1", "--ppo_frac", "0"]).ppo_frac == 0


def test_standing_prior_waits_for_an_upright_policy_window():
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


@pytest.mark.parametrize("flag,value", [("--ode", "1"), ("--ppo_frac", "1"), ("--ppo_frac", "-.1"), ("--legged_real_ppo", "1"),
                                         ("--obs_clip", "3")])
def test_incompatible_world_model_settings_fail_before_training(flag, value):
    with pytest.raises(ValueError):
        parse(["--robot", zenbot.MIXED_TASK, "--motion_train_wm", "1", flag, value])


def test_source_mujoco_evaluator_matches_world_model_actor_with_learned_layer_norm(tmp_path):
    layers = nets.mlp_init(jax.random.key(41), (234, 16, 12, 8, 23))
    layers[0].update(ln_g=jnp.linspace(.6, 1.1, 16), ln_b=jnp.linspace(-.1, .2, 16))
    norm = nets.rn_init(234)
    norm["mean"] = jnp.linspace(-.5, .5, 234)
    path = tmp_path / "policy"
    save_policy(str(path), dict(avg=layers, pi=layers, obs_norm=norm), {"it": 0})
    evaluate = DenseActor(path.with_suffix(".npz"), allow_layer_norm=True)
    obs = np.random.default_rng(4).normal(size=(8, 234)).astype(np.float32)
    np.testing.assert_allclose([evaluate(row) for row in obs], motion_train_actor_mean(layers, norm, obs),
                               atol=2e-6, rtol=2e-6)
