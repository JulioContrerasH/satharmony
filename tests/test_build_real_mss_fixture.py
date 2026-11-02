"""Tests for the real-MSS fixture builder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from build_real_mss_fixture import array_sha256, choose  # noqa: E402


def test_choose_limits_and_deduplicates() -> None:
    """The sample never repeats a window and respects the requested size."""
    windows = [
        {"row": row, "col": col, "cloud_fraction": cloud}
        for row, cloud in enumerate([0.4] * 6 + [0.1] * 6 + [0.0] * 6)
        for col in range(1)
    ]
    chosen = choose(windows, limit=10, rng=np.random.default_rng(0))
    assert len(chosen) == 10
    assert len({(window["row"], window["col"]) for window in chosen}) == 10
    assert any(window["cloud_fraction"] >= 0.2 for window in chosen)
    assert any(window["cloud_fraction"] < 0.01 for window in chosen)


def test_array_sha256_is_deterministic() -> None:
    """The fixture hash depends on the bytes, not on the array object."""
    array = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert array_sha256(array) == array_sha256(array.copy())
    assert array_sha256(array) != array_sha256(array + 1)
