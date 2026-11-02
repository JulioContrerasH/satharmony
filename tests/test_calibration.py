"""Tests for the fixed affine radiometry of the pseudo-MSS fixture.

The point of the correction is that it is **fixed**: one scale and one offset
per band, fitted once, applied identically to every chip.  These tests pin
that property, the algebra of the moment fit and the JSON contract, so the
retraining and the benchmarks cannot silently drift into a per-image
normalisation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.calibration import (  # noqa: E402
    DEFAULT_BANDS,
    SCALE_FACTOR,
    AffineRadiometry,
)


def _cube(seed: int = 0, chips: int = 8, bands: int = 4, size: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.02, 0.9, size=(chips, bands, size, size)).astype(np.float32)


def test_fit_recovers_a_known_affine() -> None:
    """A synthetic target ``2 * source + 0.1`` gives scale 2 and offset 0.1."""
    source = _cube()
    target = 2.0 * source + 0.1
    fitted = AffineRadiometry.fit_moments(source, target)
    assert np.allclose(fitted.scale, 2.0, atol=1e-5)
    assert np.allclose(fitted.offset, 0.1, atol=1e-5)


def test_apply_is_identical_per_chip_and_per_batch() -> None:
    """The mapping is fixed: batching chips cannot change a pixel's value."""
    rng = np.random.default_rng(7)
    scale = tuple(rng.uniform(1.2, 1.6, size=4))
    offset = tuple(rng.uniform(-0.04, 0.04, size=4))
    transform = AffineRadiometry(scale=scale, offset=offset)
    images = _cube(seed=3)
    batch = transform.apply(images)
    per_chip = np.stack([transform.apply(image) for image in images])
    assert np.array_equal(batch, per_chip)
    # And applying twice with the same coefficients is idempotent in value.
    assert np.array_equal(batch, transform.apply(images))


def test_apply_accepts_three_and_four_dimensions() -> None:
    """Both ``(C, H, W)`` and ``(N, C, H, W)`` are supported."""
    transform = AffineRadiometry(scale=(1.1, 1.2, 1.3, 1.4), offset=(0.0, 0.0, 0.0, 0.0))
    image = _cube(chips=1)[0]
    single = transform.apply(image)
    stacked = transform.apply(image[None])[0]
    assert single.shape == image.shape
    assert np.allclose(single, stacked)


def test_apply_uint16_round_trips_reflectance() -> None:
    """uint16 reflectance x 10000 survives the affine within one DN."""
    transform = AffineRadiometry(scale=(1.0, 1.0, 1.0, 1.0), offset=(0.01,) * 4)
    images = (np.ones((2, 4, 8, 8)) * 5000).astype(np.uint16)
    calibrated = transform.apply_uint16(images)
    assert calibrated.dtype == np.uint16
    expected = round((0.5 + 0.01) * SCALE_FACTOR)
    assert np.all(calibrated == expected)


def test_apply_uint16_clips_out_of_range_values() -> None:
    """Bright pixels saturate at 1 and dark pixels at 0."""
    transform = AffineRadiometry(
        scale=(2.0, 2.0, 2.0, 2.0), offset=(-0.2, -0.2, -0.2, -0.2)
    )
    images = np.zeros((4, 2, 2), dtype=np.uint16)
    images[:, 0, 0] = SCALE_FACTOR
    images[:, 0, 1] = 5000
    calibrated = transform.apply_uint16(images)
    assert calibrated.dtype == np.uint16
    assert np.all(calibrated[:, 0, 0] == SCALE_FACTOR)
    assert np.all(calibrated[:, 0, 1] == round(0.8 * SCALE_FACTOR))
    assert np.all(calibrated[:, 1, :] == 0)


def test_zero_variance_source_raises() -> None:
    """A constant band cannot be scaled by a moment fit."""
    source = np.ones((4, 8, 8), dtype=np.float32)
    target = _cube(chips=1)[0]
    with pytest.raises(ValueError, match="zero variance"):
        AffineRadiometry.fit_moments(source, target)


def test_band_mismatch_raises() -> None:
    """The transform refuses a cube with a different band count."""
    transform = AffineRadiometry(scale=(1.0, 1.0, 1.0, 1.0), offset=(0.0,) * 4)
    with pytest.raises(ValueError, match="4 bands"):
        transform.apply(np.zeros((3, 8, 8), dtype=np.float32))


def test_mask_is_used_for_the_moments() -> None:
    """Invalid pixels (zeros) do not drag the fitted mean down."""
    source = _cube(seed=11)
    source[:, :, :2, :] = 0.0
    target = 1.5 * source + 0.02
    valid = np.any(source > 0, axis=1)
    fitted = AffineRadiometry.fit_moments(source, target, valid_source=valid, valid_target=valid)
    assert np.allclose(fitted.scale, 1.5, atol=1e-4)
    assert np.allclose(fitted.offset, 0.02, atol=1e-4)


def test_json_round_trip(tmp_path: Path) -> None:
    """Every coefficient and every provenance field survives save/load."""
    transform = AffineRadiometry(
        scale=(1.3364, 1.4002, 1.5337, 1.4065),
        offset=(0.0204, 0.0195, -0.0332, -0.0209),
        bands=DEFAULT_BANDS,
        method="moment matching",
        split="train+validation",
        n_pixels=123,
    )
    path = transform.save(tmp_path / "radiometry.json")
    loaded = AffineRadiometry.load(path)
    assert loaded == transform
    payload = json.loads(path.read_text())
    assert payload["format"] == "satharmony.affine-radiometry/1"
    assert len(payload["scale"]) == 4


def test_missing_scale_in_json_raises() -> None:
    """A malformed JSON fails with the missing key, not a KeyError later."""
    with pytest.raises(ValueError, match="scale"):
        AffineRadiometry.from_dict({"offset": [0.0, 0.0, 0.0, 0.0]})
