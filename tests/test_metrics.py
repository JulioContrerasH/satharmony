"""Tests for the calibration metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.calibration.metrics import (  # noqa: E402
    bias_ci,
    c2st,
    edge_fwhm,
    rmse,
    sam,
    snr_db,
    summarize,
)


def test_sam_is_zero_for_identical_spectra() -> None:
    rng = np.random.default_rng(0)
    cube = rng.uniform(0.05, 0.8, size=(4, 16, 16))
    assert sam(cube, cube) == pytest.approx(0.0, abs=1e-6)


def test_sam_is_invariant_to_a_global_scale() -> None:
    rng = np.random.default_rng(1)
    cube = rng.uniform(0.05, 0.8, size=(4, 16, 16))
    assert sam(2.0 * cube, cube) == pytest.approx(0.0, abs=1e-6)


def test_rmse_matches_the_definition() -> None:
    obs = np.zeros((1, 4, 4))
    pred = np.full((1, 4, 4), 0.5)
    assert rmse(pred, obs) == pytest.approx(0.5)


def test_bias_ci_brackets_the_true_bias() -> None:
    rng = np.random.default_rng(2)
    obs = rng.uniform(0.1, 0.6, size=(4, 32, 32))
    pred = obs + 0.02 + rng.normal(0, 0.01, size=obs.shape)
    out = bias_ci(pred, obs, n_boot=500, rng=np.random.default_rng(3))
    assert out["bias"] == pytest.approx(0.02, abs=0.003)
    assert out["low"] < out["bias"] < out["high"]


def test_snr_is_finite_on_texture_and_infinite_on_constant() -> None:
    rng = np.random.default_rng(4)
    texture = gaussian_filter(rng.normal(size=(64, 64)), 2.0)
    assert np.isfinite(snr_db(texture))
    assert snr_db(np.ones((32, 32))) == float("inf")


def test_edge_fwhm_of_a_smoothed_step() -> None:
    sigma = 2.0
    step = np.zeros(81)
    step[40:] = 1.0
    profile = gaussian_filter(step, sigma)
    assert edge_fwhm(profile) == pytest.approx(2.355 * sigma, abs=1.0)


def test_c2st_separates_shifted_samples() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(5)
    real = rng.normal(0.0, 0.1, size=(4, 40, 40))
    same = rng.normal(0.0, 0.1, size=(4, 40, 40))
    shifted = rng.normal(0.5, 0.1, size=(4, 40, 40))
    assert c2st(real, same) < 0.7
    assert c2st(real, shifted) > 0.9


def test_summarize_has_the_three_numbers() -> None:
    rng = np.random.default_rng(6)
    cube = rng.uniform(0.1, 0.5, size=(4, 8, 8))
    out = summarize(cube, cube)
    assert set(out) == {"sam_deg", "rmse", "bias"}
    assert out["sam_deg"] == pytest.approx(0.0, abs=1e-6)
