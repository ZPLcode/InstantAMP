"""Integrity of the matched gait and recovery banks used by the joint recipe."""

import json

import pytest

from instantamp.zenbot_models import ZENBOT23, ZENBOT26
from tools import retarget_zenbot26


def test_verify_both_banks_rejects_changed_weights_and_stale_sources(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    clips = {}
    for bank in ("walk_run", "recovery"):
        for directory in (source, target):
            (directory / bank).mkdir(parents=True)
            (directory / bank / "sampling.json").write_text(
                '{"weights":{"clip":2}}'
            )
            (directory / bank / "clip.npz").write_bytes(directory.name.encode())
        relative = f"{bank}/clip.npz"
        clips[relative] = {
            "source_sha256": retarget_zenbot26.digest(source / relative),
            "output_sha256": retarget_zenbot26.digest(target / relative),
        }
    manifest = {
        "source_mjcf_sha256": retarget_zenbot26.digest(ZENBOT23.xml_path),
        "target_mjcf_sha256": retarget_zenbot26.digest(ZENBOT26.xml_path),
        "joint_names": [f"{joint}_joint" for joint in ZENBOT26.dof],
        "clips": clips,
    }
    (target / "manifest.json").write_text(json.dumps(manifest))

    retarget_zenbot26.verify(source, target)
    sampling = target / "walk_run/sampling.json"
    sampling.write_text('{"weights":{"clip":1}}')
    with pytest.raises(ValueError, match="sampling weights differ"):
        retarget_zenbot26.verify(source, target)
    sampling.write_text((source / "walk_run/sampling.json").read_text())
    (source / "recovery/clip.npz").write_bytes(b"updated")
    with pytest.raises(
        ValueError, match="checksum mismatch for recovery/clip.npz"
    ):
        retarget_zenbot26.verify(source, target)
