"""Assemble the published Zenbot weights from the three checkpoint parts in Git."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


RELEASE = Path(__file__).resolve().parents[1] / "runs/zenbot23_amp_ground_release/ckpt/0000.npz"


def resolve_checkpoint(checkpoint):
    target = Path(checkpoint).resolve()
    if target.is_file():
        return target
    if target != RELEASE:
        raise FileNotFoundError(f"Missing checkpoint {target}")

    manifest = target.parent.parent / "release.json"
    expected = json.loads(manifest.read_text())["sha256"]
    digest = hashlib.sha256()
    with TemporaryDirectory(dir=target.parent) as scratch:
        assembled = Path(scratch) / target.name
        with assembled.open("wb") as output:
            for n in range(1, 4):
                part = target.with_name(f"{target.name}.part{n:02d}")
                with part.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Zenbot checkpoint parts do not match SHA-256 in {manifest}")
        # Publish the verified file atomically without replacing another process's checkpoint.
        try:
            os.link(assembled, target)
        except FileExistsError:
            if not target.is_file():
                raise
    return target


if __name__ == "__main__":
    print(resolve_checkpoint(RELEASE))
