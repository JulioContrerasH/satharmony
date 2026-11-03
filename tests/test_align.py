"""Tests for subpixel alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as nd_shift

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.calibration.align import align_pair, apply_shift, phase_shift  # noqa: E402


def _texture(seed: int = 0, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(size, size))
    return gaussian_filter(raw, sigma=3.0).astype(np.float64)


def test_recovers_a_known_subpixel_shift() -> None:
    reference = _texture()
    moved = nd_shift(reference, (2.35, -1.70), order=3, mode="reflect")
    result = align_pair(reference, moved, max_shift=6.0, refine=False)
    assert result.dy == pytest.approx(-2.35, abs=0.3)
    assert result.dx == pytest.approx(1.70, abs=0.3)
    assert result.quality > 0.9


def test_rejects_a_shift_beyond_the_limit() -> None:
    reference = _texture()
    moved = nd_shift(reference, (0.0, 12.0), order=3, mode="reflect")
    result = align_pair(reference, moved, max_shift=6.0, refine=False)
    assert (result.dy, result.dx) == (0.0, 0.0)
    assert result.peak == 0.0


def test_apply_shift_moves_a_delta() -> None:
    image = np.zeros((32, 32), dtype=np.float64)
    image[10, 10] = 1.0
    moved = apply_shift(image, 2.0, -3.0, fill=0.0)
    row, col = np.unravel_index(int(np.argmax(moved)), moved.shape)
    assert (row, col) == (12, 7)


def test_identical_images_have_zero_shift() -> None:
    reference = _texture()
    dy, dx, _ = phase_shift(reference, reference)
    assert abs(dy) < 1e-6
    assert abs(dx) < 1e-6


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        phase_shift(np.zeros((8, 8)), np.zeros((8, 9)))
