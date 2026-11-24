"""The scorer selects checkpoints by training iteration and uses the saved bank."""

import json

import numpy as np
import pytest

from instantamp.zenbot_trials import JointTrials, resolve_checkpoint


def test_run_selects_exact_final_iteration_and_replays_custom_bank(tmp_path):
    run = tmp_path / "run"
    ckpt = run / "ckpt"
    ckpt.mkdir(parents=True)
    bank = tmp_path / "motions"
    bank.mkdir()
    (run / "config.json").write_text(
        json.dumps({"floor_dir": str(bank), "max_iterations": 500})
    )

    # The largest filename is not necessarily the final learner iteration.
    for name, step in (("0002", 500), ("0999", 100)):
        export = ckpt / f"{name}.npz"
        np.savez(
            export,
            **{
                "avg/0/w": np.zeros((234, 23), np.float32),
                "avg/0/b": np.zeros(23, np.float32),
                "obs_norm/mean": np.zeros(234, np.float32),
                "obs_norm/var": np.ones(234, np.float32),
            },
        )
        export.with_suffix(".json").write_text(json.dumps({"it": step}))
    assert resolve_checkpoint(run) == ckpt / "0002.npz"
    assert resolve_checkpoint(run, 100) == ckpt / "0999.npz"
    with pytest.raises(FileNotFoundError, match="iteration 501"):
        resolve_checkpoint(run, 501)

    joints = np.zeros((26, 23))
    root = np.zeros((26, 3))
    np.savez(
        bank / "fall.npz",
        fps=50,
        root_pos=root,
        root_quat=np.tile([0.0, 0.0, 0.0, 1.0], (26, 1)),
        dof_pos=joints,
        dof_vel=joints,
        root_lin_vel=root,
        root_ang_vel=root,
    )
    trials = JointTrials(resolve_checkpoint(run))
    assert trials.names == [("fall.npz", 0.0), ("fall.npz", 0.5)]
    trials.fallen(1)
    assert np.isfinite(trials.native.step())
