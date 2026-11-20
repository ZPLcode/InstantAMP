import json

import numpy as np

from tools.prepare_g1_rises import prepare


def test_prepare_native_rises_separates_drop_and_holdout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    # The take includes one complete recovery and ends on a second, unrecovered fall.
    standing = np.full(36, 0.77)
    resting = np.full(50, 0.16)
    fall = np.linspace(0.77, 0.16, 20)
    rise = np.linspace(0.16, 0.77, 65)
    height = np.concatenate((standing, fall, resting, rise, standing, fall, resting))
    joints = np.repeat(np.arange(len(height))[:, None], 29, axis=1)
    root = np.column_stack((np.zeros((len(height), 2)), height))
    motion = dict(fps=50, root_quat=np.tile([0., 0., 0., 1.], (len(root), 1)),
                  root_lin_vel=np.gradient(root, axis=0) * 50, root_ang_vel=np.zeros_like(root),
                  dof_pos=joints, dof_vel=np.zeros_like(joints))
    for subject in ("subject1", "subject2"):
        np.savez(source / f"fallAndGetUp1_{subject}.npz", root_pos=root, **motion)

    output = tmp_path / "slices"
    assert prepare(source, output) == prepare(source, output) == {"train": 1, "heldout": 1}
    for split, name in (("train", "subject2"), ("heldout", "subject1")):
        records = json.loads((output / split / "manifest.json").read_text())["clips"]
        record, = records
        assert record["source_take"].endswith(name)
        assert len(standing) + len(fall) <= record["start_frame"] < record["stand_frame"]
        assert record["end_frame"] <= len(height) - len(fall) - len(resting)
        with np.load(output / split / record["file"]) as data:
            index = slice(record["start_frame"], record["end_frame"])
            np.testing.assert_array_equal(data["dof_pos"], joints[index])
            assert data["root_pos"][0, 2] < 0.4
            assert (data["root_pos"][-20:, 2] >= 0.7).all()
