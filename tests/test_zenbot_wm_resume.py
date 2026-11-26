import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from instantamp import nets, optim
from instantamp.train import (
    initialize_native_learner,
    load_full,
    parse,
    prune_full_checkpoints,
    save_full,
    save_policy,
)
from tools.resume_zenbot_wm import continuation_args


def test_full_checkpoint_retention_preserves_policies_and_incomplete_writes(tmp_path):
    full = tmp_path / "ckpt/full"
    full.mkdir(parents=True)
    for index in range(5):
        (full / f"{index:04d}.npz").touch()
        if index < 4:
            (full.parent / f"{index:04d}.npz").touch()
            (full.parent / f"{index:04d}.json").write_text("{}")
    (full / "0005.tmp.npz").touch()
    (full / "named_export.npz").touch()
    prune_full_checkpoints(full, 0)
    assert len(list(full.glob("*.npz"))) == 7
    prune_full_checkpoints(full, 2)
    assert {p.name for p in full.iterdir()} == {
        "0002.npz", "0003.npz", "0004.npz", "0005.tmp.npz", "named_export.npz",
    }
    assert len(list(full.parent.glob("*.npz"))) == 4
    assert len(list(full.parent.glob("*.json"))) == 4


def test_continue_archived_i_profile_with_full_directory_on_another_volume(tmp_path):
    run = tmp_path / "i"
    ckpt = run / "ckpt"
    ckpt.mkdir(parents=True)
    volume = tmp_path / "volume"
    volume.mkdir()
    (ckpt / "full").symlink_to(volume, target_is_directory=True)
    full = ckpt / "full/snapshot.npz"
    full.touch()
    policy = ckpt / full.name
    policy.touch()
    policy.with_suffix(".json").write_text(json.dumps({"it": 10000}))
    # Run I predates the mirror-consistency configuration field.
    config = dict(robot="zenbot23_walk_run_recovery", motion_train_wm=1,
                  gait_velocity_time_constant=.5, wm_floor_amp=1,
                  gait_ls_scale=10., gait_source_commands=1, gait_amp_stride=3)
    (run / "config.json").write_text(json.dumps(config))

    output = tmp_path / "next"
    flags = continuation_args(full, output, 12000)
    assert not any(flag.startswith("--gait_velocity_time_constant") for flag in flags)
    cfg = parse(flags)
    assert cfg.init_full == str(full) and cfg.init_policy == ""
    assert cfg.log_dir == str(output) and cfg.max_iterations == 12000
    assert cfg.gait_mirror_weight == 0
    assert cfg.imag_gait_mirror_weight == 0
    assert cfg.wm_hidden == "512,512,512"
    ranged = parse(continuation_args(full, output, 12000, range_commands=True, num_envs=4096))
    assert ranged.num_envs == 4096 and ranged.wm_range_commands == 1
    assert ranged.init_full == cfg.init_full
    with pytest.raises(ValueError, match="exceed the saved iteration 10000"):
        continuation_args(full, output, 10000)
    with pytest.raises(ValueError, match="full learner"):
        continuation_args(policy, output, 12000)
    config.update(wm_global_gait=1, wm_fixed_floor_envs=1, wm_range_commands=1,
                  wm_gait_rewards=1, gait_source_commands=0, gait_amp_stride=1)
    for settings, expected in (({}, 3), ({"wm_gait_frames": 10}, 10)):
        (run / "config.json").write_text(json.dumps({**config, **settings}))
        assert parse(continuation_args(full, output, 12000)).wm_gait_frames == expected
    config.update(wm_initial_gait_stage=0, imag_gait_mirror_weight=.12,
                  imu_replay_path="old_bank.npz", imu_replay_share=.25, imu_replay_scale=.1)
    (run / "config.json").write_text(json.dumps(config))
    assert parse(continuation_args(full, output, 12000)).imag_gait_mirror_weight == .12
    assert parse(continuation_args(full, output, 12000, imag_gait_mirror_weight=.2)).imag_gait_mirror_weight == .2
    del config["wm_initial_gait_stage"], config["imag_gait_mirror_weight"]
    config.update(robot="zenbot26_walk_run_recovery", wm_global_gait=1)
    (run / "config.json").write_text(json.dumps(config))
    cfg = parse(continuation_args(full, output, 12000))
    assert cfg.robot == config["robot"] and cfg.wm_initial_gait_stage == 2
    assert cfg.motion_train_randomization == 0
    assert cfg.external_pushes == 1
    assert (cfg.wm_tracking_scale, cfg.gait_ls_scale, cfg.action_rate_l2_weight) == (2., 10., -.03)
    config.update(motion_train_randomization=1, external_pushes=0)
    (run / "config.json").write_text(json.dumps(config))
    cfg = parse(continuation_args(full, output, 12000))
    assert (cfg.motion_train_randomization, cfg.external_pushes) == (1, 0)
    del config["gait_ls_scale"]
    (run / "config.json").write_text(json.dumps(config))
    assert parse(continuation_args(full, output, 12000)).gait_ls_scale == 1.5
    config.update(wm_getup_only=1)
    (run / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="only on floor recovery"):
        continuation_args(full, output, 12000)
    policy.unlink()
    with pytest.raises(FileNotFoundError, match="companion policy"):
        continuation_args(full, output, 12000)


def test_full_checkpoint_round_trips_exploration_and_reads_fixed_noise_runs(tmp_path):
    std = jnp.ones(23)
    fresh = dict(it=jnp.array(0), action_std=std, opt_std=optim.init(std, False))
    learned, state = optim.step(std, jnp.linspace(-1., 1., 23), fresh["opt_std"], False, .01)
    saved = dict(it=jnp.array(17), action_std=learned, opt_std=state)
    path = tmp_path / "snapshot"
    save_full(str(path), saved)
    restored = load_full(fresh, path.with_suffix(".npz"))
    for got, want in zip(jax.tree.leaves(restored), jax.tree.leaves(saved), strict=True):
        np.testing.assert_array_equal(got, want)
    save_full(str(path), dict(it=saved["it"]))
    restored = load_full(fresh, path.with_suffix(".npz"))
    np.testing.assert_array_equal(restored["action_std"], std)
    assert int(restored["it"]) == 17 and int(restored["opt_std"]["t"]) == 0


@pytest.mark.parametrize("saved_version", [None, 1])
def test_resume_resets_only_critic_when_the_reward_objective_changes(tmp_path, saved_version):
    actor = nets.mlp_init(jax.random.key(16), (3, 4, 2))
    fresh = {
        "pi": actor, "avg": actor, "obs_norm": nets.rn_init(3),
        "critic": {"value": jnp.array(0.), "opt": {"t": jnp.array(0)}},
        "wm": {"weight": jnp.array(0.)}, "amp": {"weight": jnp.array(0.)},
    }
    saved = jax.tree.map(lambda value: value + 7, fresh)
    directory = tmp_path / "ckpt"
    (directory / "full").mkdir(parents=True)
    full = directory / "full/policy"
    save_full(str(full), saved)
    exported = dict(saved)
    if saved_version is not None:
        exported["floor_reward_version"] = saved_version
    save_policy(str(directory / "policy"), exported, {})
    restored = initialize_native_learner(
        fresh, full.with_suffix(".npz"), "", None, floor_reward_version=1,
    )
    for key in ("pi", "avg", "obs_norm", "wm", "amp"):
        for actual, expected in zip(jax.tree.leaves(restored[key]), jax.tree.leaves(saved[key]), strict=True):
            np.testing.assert_array_equal(actual, expected)
    expected_critic = saved["critic"] if saved_version == 1 else fresh["critic"]
    for actual, expected in zip(jax.tree.leaves(restored["critic"]), jax.tree.leaves(expected_critic), strict=True):
        np.testing.assert_array_equal(actual, expected)
