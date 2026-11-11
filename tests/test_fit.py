"""Tests for profile fitting on synthetic pairs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.calibration.fit import (  # noqa: E402
    Pair,
    _deterministic_config,
    fit_profile,
)
from satharmony.emulators.s2_to_mss.degradations.spatial import (  # noqa: E402
    apply_spatial_degradation,
)
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402
from satharmony.emulators.s2_to_mss.sbaf import water_vapor_feature  # noqa: E402


def _pair(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s2 = rng.uniform(0.03, 0.85, size=(13, 48, 48)).astype(np.float32)
    config = _deterministic_config()
    # The fit adds the water-vapour term itself, so the synthetic prediction
    # must be built without it (same contract as fit_profile).
    config.spectral.water_vapor_scale = 0.0
    pred, _ = emulate_s2(s2, config=config)
    wv = apply_spatial_degradation(water_vapor_feature(s2)[None], config.spatial)[0]
    return s2, pred, wv, config


def test_fit_recovers_the_band_scales() -> None:
    s2, pred, _wv, _config = _pair()
    expected = np.array([1.02, 0.97, 1.08, 0.99], dtype=np.float32)
    mss = pred * expected[:, None, None]
    profile = fit_profile([Pair(s2=s2, mss=mss, name="synthetic")], n_boot=100)
    assert profile.band_scaling[:3] == pytest.approx(expected[:3], abs=1e-3)
    assert profile.n_pairs == 1


def test_fit_recovers_a_known_gamma() -> None:
    s2, pred, wv, _config = _pair()
    gamma_true = 0.05
    scale = 1.0
    mss = pred * scale
    mss[3] = (pred[3] - gamma_true * wv) * scale
    profile = fit_profile(
        [Pair(s2=s2, mss=mss, name="synthetic")],
        n_boot=100,
        gamma_grid=np.arange(0.0, 0.151, 0.001),
    )
    assert profile.gamma == pytest.approx(gamma_true, abs=0.01)


def test_profile_roundtrip_has_provenance() -> None:
    s2, pred, _wv, _config = _pair()
    profile = fit_profile([Pair(s2=s2, mss=pred, name="synthetic")], n_boot=10)
    payload = profile.to_dict()
    assert payload["schema_version"] == 1
    assert payload["n_pairs"] == 1
    assert set(payload["band_scaling"]) == {"B1", "B2", "B3", "B4"}


def test_empty_pairs_raise() -> None:
    with pytest.raises(ValueError, match="at least one pair"):
        fit_profile([])
