"""Tests for the versioned JSON profiles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.emulators.s2_to_mss.config import PipelineConfig  # noqa: E402
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402


def test_v1_profile_matches_hardcoded_config() -> None:
    """The JSON is the source of truth and still equals the hardcoded defaults."""
    loaded = PipelineConfig.from_profile("mss5_esa_gtc_v1")
    assert loaded.model_dump() == PipelineConfig.mss5_esa_gtc_v1().model_dump()


def test_legacy_profile_matches_hardcoded_config() -> None:
    loaded = PipelineConfig.from_profile("legacy_0_1_13")
    assert loaded.model_dump() == PipelineConfig.legacy_0_1_13().model_dump()


def test_profile_payload_has_provenance() -> None:
    path = REPO / "satharmony/profiles/mss5_esa_gtc_v1.json"
    payload = json.loads(path.read_text())
    for key in ("schema_version", "name", "created", "config", "provenance"):
        assert key in payload
    assert payload["provenance"]["package_sha"]
    assert payload["schema_version"] == 1


def test_unknown_profile_lists_alternatives() -> None:
    with pytest.raises(FileNotFoundError, match="available:"):
        PipelineConfig.from_profile("does_not_exist")


def test_profile_loads_from_path() -> None:
    path = REPO / "satharmony/profiles/mss5_esa_gtc_v1.json"
    assert PipelineConfig.from_profile(path).model_dump() == PipelineConfig.mss5_esa_gtc_v1().model_dump()


def test_emulation_from_json_is_bit_identical() -> None:
    """Loading the profile must not change a single DN."""
    rng = np.random.default_rng(0)
    s2 = rng.uniform(0.02, 0.9, size=(13, 48, 48)).astype(np.float32)

    from_json, _ = emulate_s2(
        s2, config=PipelineConfig.from_profile("mss5_esa_gtc_v1"), seed=7
    )
    from_code, _ = emulate_s2(s2, config=PipelineConfig.mss5_esa_gtc_v1(), seed=7)
    np.testing.assert_array_equal(from_json, from_code)


def test_v2_profile_loads_and_carries_the_gamma_scale() -> None:
    loaded = PipelineConfig.from_profile("mss5_esa_gtc_v2")
    assert loaded.spectral.water_vapor_scale == pytest.approx(1.0)
    assert loaded.final_scaling.band_scaling.green.mean == pytest.approx(0.971, abs=0.01)
    assert loaded.final_scaling.band_scaling.nir2.mean == pytest.approx(0.933, abs=0.001)


def _v2_payload() -> dict:
    return json.loads((REPO / "satharmony/profiles/mss5_esa_gtc_v2.json").read_text())


def test_v2_variation_is_the_stored_bootstrap_half_width() -> None:
    """A regeneration must not move the variation away from the interval."""
    payload = _v2_payload()
    for band in ("B1", "B2", "B3"):
        entry = payload["intervals"][band]
        expected = float(np.clip(2.0 * (entry["high"] - entry["low"]) / 3.92, 0.03, 0.12))
        assert payload["band_scaling"][band]["variation"] == pytest.approx(expected, abs=1e-9)
    assert payload["band_scaling"]["B4"]["variation"] == pytest.approx(0.072)
    assert payload["band_scaling"]["B4"]["mean"] == pytest.approx(0.933)


def test_v2_declares_the_frozen_gamma_and_the_provenance_of_its_numbers() -> None:
    payload = _v2_payload()
    assert payload["gamma"] == pytest.approx(0.112)
    assert payload["gamma_frozen"] is True
    assert payload["gamma_fitted"] == pytest.approx(0.14)
    assert payload["b4_fitted_scale"] == pytest.approx(0.942, abs=0.002)
    # The per-pair table must not leak the held-out deserts.
    names = {row["pair"] for row in payload["per_pair"]}
    assert names.isdisjoint(payload["validity"]["held_out_sites"])

    validity = payload["validity"]
    assert validity["scaled_rmse_outside"][0] == pytest.approx(0.134)
    assert validity["scaled_rmse_outside"][1] == pytest.approx(0.236)
    # The raw error is what a user gets; it must be larger than the rescaled one.
    assert validity["raw_rmse_outside"][0] > validity["scaled_rmse_outside"][0]
    assert validity["raw_rmse_outside"][1] > validity["scaled_rmse_outside"][1]
    desert = validity["desert_in_sample"]
    assert desert["v2_raw_rmse"][0] > desert["v2_scaled_rmse"][0]
    assert desert["v2_scaled_rmse"][0] >= desert["sbaf_scaled_rmse"][0]
    c2st = validity["c2st_held_out"]
    assert c2st["stochastic_median"] < c2st["deterministic_median"]
    assert 0.5 < c2st["stochastic_range"][0] < c2st["stochastic_range"][1] <= 1.0


def test_v2_sun_elevation_matches_the_canonical_measurement() -> None:
    payload = _v2_payload()
    measurement = json.loads(
        (REPO / "satharmony/profiles/measurements/sun_elevation_canonical.json").read_text()
    )
    stats = measurement["stats"]
    declared = payload["config"]["radiometric"]["sun_elevation_deg"]
    assert declared["min"] == pytest.approx(round(stats["min"], 2))
    assert declared["max"] == pytest.approx(round(stats["max"], 2))
    assert declared["mean"] == pytest.approx(round(stats["mean"], 2))
    assert declared["std"] == pytest.approx(round(stats["std"], 2))
    validity = payload["validity"]["sun_elevation"]
    assert validity["n_canonical"] == 3027
    assert validity["coverage"] > 0.95


def test_measurements_are_pinned_in_the_provenance() -> None:
    import hashlib

    payload = _v2_payload()
    for entry in payload["provenance"]["measurements"].values():
        path = REPO / "satharmony" / entry["path"]
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_sensor_physics_measurement_covers_four_bands() -> None:
    path = REPO / "satharmony/profiles/measurements/sensor_physics_sample.json"
    payload = json.loads(path.read_text())
    assert payload["n_scenes"] >= 20
    assert set(payload["bands"]) == {"B1", "B2", "B3", "B4"}
    for band in payload["bands"].values():
        assert band["stripe_amp"]["median"] > 0
        assert band["noise_rel"]["median"] > 0


def test_v2_per_pair_drops_the_unusable_b4() -> None:
    """Death Valley and Sonora have 5 levels in B4: their scale is meaningless."""
    payload = _v2_payload()
    rows = {row["pair"]: row for row in payload["per_pair"]}
    for site in ("death_valley", "sonora"):
        assert rows[site]["B4"] is None
        assert "5 levels" in rows[site]["note"]
    useful = [row["B4"] for row in payload["per_pair"] if row["B4"] is not None]
    assert len(useful) == 27
    assert float(np.std(useful, ddof=1)) == pytest.approx(0.072, abs=0.003)
    assert payload["validity"]["b4_unusable_sites"] == ["death_valley", "sonora"]
