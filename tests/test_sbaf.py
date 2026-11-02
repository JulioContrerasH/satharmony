"""Tests for the S2 -> MSS SBAF coefficients and spectral mode."""

import hashlib
import json

import numpy as np
import pytest

from satharmony import MSSEmulator, PipelineConfig
from satharmony.emulators.s2_to_mss.config import SpectralConfig
from satharmony.emulators.s2_to_mss.degradations.spectral import (
    apply_spectral_degradation,
)
from satharmony.emulators.s2_to_mss.sbaf import (
    MSS_BANDS,
    PREDICTOR_INDICES,
    PREDICTORS,
    S2_BAND_ORDER,
    SENSORS,
    load_coefficients,
    sbaf_transform,
    sensor_weights,
)


@pytest.fixture
def image_13():
    rng = np.random.default_rng(0)
    return rng.random((13, 32, 32), dtype=np.float32)


class TestCoefficients:
    def test_all_sensors_and_bands(self):
        coefficients = load_coefficients()
        assert tuple(coefficients["sensors"]) == SENSORS
        for sensor in SENSORS:
            block = coefficients["sensors"][sensor]
            assert np.asarray(block["weights"]).shape == (len(MSS_BANDS), len(PREDICTORS))
            assert np.asarray(block["offsets"]).shape == (len(MSS_BANDS),)

    def test_non_negative_weights(self):
        coefficients = load_coefficients()
        for sensor in SENSORS:
            weights = np.asarray(coefficients["sensors"][sensor]["weights"])
            assert np.all(weights >= 0.0)

    def test_row_sums_are_close_to_one(self):
        coefficients = load_coefficients()
        for sensor in SENSORS:
            sums = np.asarray(coefficients["sensors"][sensor]["sum_weights"])
            np.testing.assert_allclose(sums, 1.0, atol=0.02)

    def test_predictor_indices_match_package_order(self):
        assert PREDICTORS == ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A")
        assert PREDICTOR_INDICES == (0, 1, 2, 3, 4, 5, 6, 7, 12)
        assert S2_BAND_ORDER[PREDICTOR_INDICES[-1]] == "B8A"
        assert "B9" not in PREDICTORS

    def test_water_vapor_block(self):
        block = load_coefficients()["water_vapor"]
        assert block["predictors"] == ["B8A", "B9"]
        assert block["form"] == "rho_B8A * log(rho_B8A / rho_B9)"
        for sensor in SENSORS:
            weights = np.asarray(block["weights"][sensor], dtype=float)
            assert weights.shape == (len(MSS_BANDS),)
            assert np.all(weights[:3] == 0.0)
            assert 0.05 < weights[3] < 0.20

    def test_integrity_hash(self):
        coefficients = load_coefficients()
        payload = {
            sensor: {
                "weights": coefficients["sensors"][sensor]["weights"],
                "offsets": coefficients["sensors"][sensor]["offsets"],
            }
            for sensor in SENSORS
        } | {"water_vapor": coefficients["water_vapor"]["weights"]}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert digest == coefficients["coefficients_sha256"]

    def test_bad_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_coefficients(tmp_path / "missing.json")


class TestSBAFTransform:
    def test_shape_and_dtype(self, image_13):
        out = sbaf_transform(image_13)
        assert out.shape == (4, 32, 32)
        assert out.dtype == np.float32

    def test_flat_spectrum_identity(self):
        flat = np.full((13, 8, 8), 0.5, dtype=np.float32)
        out = sbaf_transform(flat)
        np.testing.assert_allclose(out, 0.5, atol=0.02)

    def test_manual_linear_combination(self, image_13):
        coefficients = load_coefficients()
        weights, offsets = sensor_weights("S2A", coefficients)
        expected = np.zeros((4, 32, 32), dtype=np.float32)
        for band in range(4):
            for k, index in enumerate(PREDICTOR_INDICES):
                expected[band] += weights[band, k] * image_13[index]
            expected[band] += offsets[band]
        np.clip(expected, 0.0, None, out=expected)
        out = sbaf_transform(
            image_13, sensor="S2A", coefficients=coefficients, water_vapor_scale=0.0
        )
        np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-6)

    def test_water_vapor_only_darkens_b4(self):
        from satharmony.emulators.s2_to_mss.sbaf import water_vapor_feature

        image = np.full((13, 8, 8), 0.5, dtype=np.float32)
        image[8] = 0.30  # B9 is absorbed relative to the B8A continuum
        coefficients = load_coefficients()
        gamma = float(coefficients["water_vapor"]["weights"]["S2A"][3])
        no_wv = sbaf_transform(image, coefficients=coefficients, water_vapor_scale=0.0)
        with_wv = sbaf_transform(image)
        np.testing.assert_array_equal(with_wv[:3], no_wv[:3])
        expected = no_wv[3] - gamma * water_vapor_feature(image)
        np.testing.assert_allclose(with_wv[3], expected, rtol=0.0, atol=1e-6)
        assert float(with_wv[3].mean()) < float(no_wv[3].mean())

    def test_missing_b9_is_safe(self):
        image = np.full((13, 4, 4), 0.5, dtype=np.float32)
        image[8] = 0.0
        out = sbaf_transform(image)
        assert np.all(np.isfinite(out))
        no_wv = sbaf_transform(image, water_vapor_scale=0.0)
        np.testing.assert_array_equal(out, no_wv)

    def test_water_vapor_scale_override(self):
        image = np.full((13, 4, 4), 0.5, dtype=np.float32)
        image[8] = 0.35
        base = sbaf_transform(image, water_vapor_scale=0.0)
        half = sbaf_transform(image, water_vapor_scale=0.056)
        full = sbaf_transform(image, water_vapor_scale=0.112)
        np.testing.assert_allclose(0.5 * (base[3] + full[3]), half[3], rtol=0.0, atol=1e-6)

    def test_sensor_switch_changes_output(self, image_13):
        a = sbaf_transform(image_13, sensor="S2A")
        b = sbaf_transform(image_13, sensor="S2B")
        c = sbaf_transform(image_13, sensor="S2C")
        assert not np.array_equal(a, b)
        assert not np.array_equal(a, c)
        assert not np.array_equal(b, c)

    def test_negative_values_are_clipped(self):
        zeros = np.zeros((13, 4, 4), dtype=np.float32)
        out = sbaf_transform(zeros)
        assert np.all(out >= 0.0)

    def test_unknown_sensor_raises(self, image_13):
        with pytest.raises(ValueError, match="Unknown S2 sensor"):
            sbaf_transform(image_13, sensor="S2D")

    def test_too_few_bands_raises(self):
        with pytest.raises(ValueError, match="at least 13"):
            sbaf_transform(np.zeros((12, 4, 4), dtype=np.float32))

    def test_non_3d_raises(self):
        with pytest.raises(ValueError, match="C, H, W"):
            sbaf_transform(np.zeros((13, 4), dtype=np.float32))


class TestSpectralModes:
    def test_legacy_is_bit_exact(self, image_13):
        config = SpectralConfig(mode="legacy", srf_adjustment=False)
        out = apply_spectral_degradation(image_13, config)
        expected = image_13[config.s2_bands].astype(np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_legacy_profile_keeps_legacy_mode(self):
        config = PipelineConfig.legacy_0_1_13()
        assert config.spectral.mode == "legacy"
        assert config.spectral.s2_bands == [2, 3, 6, 7]

    def test_calibrated_profile_uses_sbaf(self):
        config = PipelineConfig.mss5_esa_gtc_v1()
        assert config.spectral.mode == "sbaf"
        assert config.spectral.sensor == "S2A"

    def test_sbaf_mode_matches_transform(self, image_13):
        config = SpectralConfig(mode="sbaf", sensor="S2B")
        out = apply_spectral_degradation(image_13, config)
        expected = sbaf_transform(image_13, sensor="S2B")
        np.testing.assert_array_equal(out, expected)

    def test_sbaf_mode_respects_disabled(self, image_13):
        config = SpectralConfig(enabled=False, mode="sbaf")
        out = apply_spectral_degradation(image_13, config)
        np.testing.assert_array_equal(out, image_13)

    def test_invalid_mode_raises(self, image_13):
        config = SpectralConfig()
        config.mode = "nope"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown spectral mode"):
            apply_spectral_degradation(image_13, config)

    def test_pipeline_integration(self, image_13):
        config = PipelineConfig(seed=0)
        config.spectral.mode = "sbaf"
        config.spatial.enabled = False
        config.radiometric.enabled = False
        config.striping.enabled = False
        config.memory_effect.enabled = False
        config.coherent_noise.enabled = False
        config.random_noise.enabled = False
        config.scan_artifacts.enabled = False
        config.final_scaling.enabled = False
        out = MSSEmulator(config)(image_13)
        expected = sbaf_transform(image_13, sensor=config.spectral.sensor)
        np.testing.assert_array_equal(out, expected)
