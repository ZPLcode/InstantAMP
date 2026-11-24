"""Fixtures for integration tests that replay prepared motion banks."""

from pathlib import Path

import pytest


@pytest.fixture
def zenbot23_motion_banks():
    """Require the gait and recovery demos used by native reference tests."""
    directory = Path(__file__).resolve().parents[1] / "data/zenbot23_walk_run_recovery"
    if not all(any((directory / bank).glob("*.npz")) for bank in ("walk_run", "recovery")):
        pytest.skip("Prepare the Zenbot23 motion banks to run native replay tests")
    return directory
