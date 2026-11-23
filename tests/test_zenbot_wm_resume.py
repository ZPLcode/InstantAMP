import json

import pytest

from instantamp.train import parse
from tools.resume_zenbot_wm import continuation_args


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
    cfg = parse(continuation_args(full, output, 12000))
    assert cfg.init_full == str(full) and cfg.init_policy == ""
    assert cfg.log_dir == str(output) and cfg.max_iterations == 12000
    assert cfg.gait_velocity_time_constant == .5 and cfg.gait_mirror_weight == 0
    with pytest.raises(ValueError, match="exceed the saved iteration 10000"):
        continuation_args(full, output, 10000)
    with pytest.raises(ValueError, match="full learner"):
        continuation_args(policy, output, 12000)
    policy.unlink()
    with pytest.raises(FileNotFoundError, match="companion policy"):
        continuation_args(full, output, 12000)
