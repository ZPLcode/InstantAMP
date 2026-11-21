import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools import restore_zenbot_getup as restore


def test_restore_checksum_and_concurrent_publication(tmp_path, monkeypatch):
    target = tmp_path / "ckpt/0000.npz"
    target.parent.mkdir()
    monkeypatch.setattr(restore, "RELEASE", target)
    parts = [b"first", b"second", b"third"]
    combined = b"".join(parts)
    (tmp_path / "release.json").write_text(json.dumps({"sha256": hashlib.sha256(combined).hexdigest()}))
    for index, part in enumerate(parts, 1):
        target.with_name(f"0000.npz.part{index:02d}").write_bytes(part)

    last = target.with_name("0000.npz.part03")
    last.write_bytes(b"incorrect")
    with pytest.raises(ValueError, match="SHA-256"):
        restore.resolve_checkpoint(target)
    assert not target.exists()
    with pytest.raises(FileNotFoundError):
        restore.resolve_checkpoint(tmp_path / "custom.npz")

    last.write_bytes(parts[-1])
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(restore.resolve_checkpoint, [target] * 4)) == [target] * 4
    assert target.read_bytes() == combined

    target.write_bytes(b"existing local checkpoint")
    assert restore.resolve_checkpoint(target).read_bytes() == b"existing local checkpoint"
