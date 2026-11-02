"""Tests for satharmony package."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

import satharmony
from satharmony import MSSEmulator, PipelineConfig, aggregate_cloudsen12

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def s2_image_1024():
    """Synthetic S2 image 1024x1024."""
    np.random.seed(0)
    return np.random.rand(13, 1024, 1024).astype(np.float32)


@pytest.fixture
def s2_image_512():
    """Synthetic S2 image 512x512."""
    np.random.seed(0)
    return np.random.rand(13, 512, 512).astype(np.float32)


@pytest.fixture
def labels_1024():
    """Synthetic CloudSEN12 labels 1024x1024."""
    np.random.seed(0)
    return np.random.randint(0, 4, (1024, 1024), dtype=np.uint8)


@pytest.fixture
def labels_512():
    """Synthetic CloudSEN12 labels 512x512."""
    np.random.seed(0)
    return np.random.randint(0, 4, (512, 512), dtype=np.uint8)


# =============================================================================
# Test: MSSEmulator
# =============================================================================


class TestMSSEmulator:
    def test_output_shape_4_bands(self, s2_image_512):
        """Output should have 4 MSS bands."""
        emulator = MSSEmulator()
        mss = emulator(s2_image_512)
        assert mss.shape[0] == 4

    def test_output_downsampled(self, s2_image_512):
        """Output should be spatially downsampled."""
        emulator = MSSEmulator()
        mss = emulator(s2_image_512)
        assert mss.shape[1] < s2_image_512.shape[1]
        assert mss.shape[2] < s2_image_512.shape[2]

    def test_reproducibility_with_seed(self, s2_image_512):
        """Same seed should produce identical results."""
        config = PipelineConfig(seed=42)
        emulator = MSSEmulator(config)

        mss1 = emulator(s2_image_512)
        mss2 = emulator(s2_image_512)

        np.testing.assert_array_equal(mss1, mss2)

    def test_different_seeds_different_results(self, s2_image_512):
        """Different seeds should produce different results."""

        config1 = PipelineConfig(seed=42)
        config1.spatial.target_gsd.min = 70.0
        config1.spatial.target_gsd.max = 70.0

        config2 = PipelineConfig(seed=123)
        config2.spatial.target_gsd.min = 70.0
        config2.spatial.target_gsd.max = 70.0

        mss1 = MSSEmulator(config1)(s2_image_512)
        mss2 = MSSEmulator(config2)(s2_image_512)

        assert mss1.shape == mss2.shape
        assert not np.allclose(mss1, mss2)

    def test_output_dtype_float32(self, s2_image_512):
        """Default output should be float32."""
        config = PipelineConfig(seed=42)
        mss = MSSEmulator(config)(s2_image_512)
        assert mss.dtype == np.float32

    def test_output_dtype_uint8(self, s2_image_512):
        """uint8 output option."""
        config = PipelineConfig(seed=42, output_dtype="uint8")
        mss = MSSEmulator(config)(s2_image_512)
        assert mss.dtype == np.uint8


# =============================================================================
# Test: Label Aggregation
# =============================================================================


class TestLabelAggregation:
    def test_output_shape(self, labels_512):
        """Output shape should be input // scale_factor."""
        result = aggregate_cloudsen12(labels_512, scale_factor=6)
        expected_h = 512 // 6
        expected_w = 512 // 6
        assert result.shape == (expected_h, expected_w)

    def test_output_dtype(self, labels_512):
        """Output should be uint8."""
        result = aggregate_cloudsen12(labels_512, scale_factor=6)
        assert result.dtype == np.uint8

    def test_valid_classes(self, labels_512):
        """Output should only contain valid classes 0-3."""
        result = aggregate_cloudsen12(labels_512, scale_factor=6)
        assert np.all(result <= 3)

    def test_majority_clear(self):
        """Block with majority clear should return clear."""
        labels = np.zeros((6, 6), dtype=np.uint8)
        labels[0, 0] = 2  # One cloud pixel
        result = aggregate_cloudsen12(labels, scale_factor=6)
        assert result[0, 0] == 0  # Clear

    def test_majority_cloud(self):
        """Block with majority cloud should return cloud."""
        labels = np.full((6, 6), 2, dtype=np.uint8)  # All cloud
        result = aggregate_cloudsen12(labels, scale_factor=6)
        assert result[0, 0] == 2  # Cloud

    def test_conservative_cloud_over_shadow(self):
        """With more cloud than shadow in no-majority case, cloud should win."""
        # 20 cloud, 10 shadow, 6 clear = 36 total
        # No majority (need 18), n_cloudy=20 > n_shadow=10
        labels = np.zeros((6, 6), dtype=np.uint8)
        labels[:4, :5] = 2  # 20 cloud
        labels[4:, :5] = 3  # 10 shadow
        labels[:, 5] = 0  # 6 clear
        result = aggregate_cloudsen12(labels, scale_factor=6)
        assert result[0, 0] == 2  # Cloud (conservative: cloud > shadow)


# =============================================================================
# Test: High-level Functions
# =============================================================================


class TestEmulateS2:
    def test_returns_tuple(self, s2_image_512):
        """Should return (mss, scale_factor) tuple."""
        result = satharmony.emulate_s2(s2_image_512, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_scale_factor_in_range(self, s2_image_512):
        """Scale factor should be in expected range (6-8)."""
        _, sf = satharmony.emulate_s2(s2_image_512, seed=42)
        assert 6 <= sf <= 8

    def test_fixed_scale_factor(self, s2_image_512):
        """Fixed scale_factor should be respected."""
        mss, sf = satharmony.emulate_s2(s2_image_512, scale_factor=8, seed=42)
        assert sf == 8
        assert mss.shape[1] == 512 // 8


class TestEmulateLabels:
    def test_output_shape(self, labels_512):
        """Output shape matches scale_factor."""
        result = satharmony.emulate_labels(labels_512, scale_factor=7)
        assert result.shape == (512 // 7, 512 // 7)


class TestEmulateS2WithLabels:
    def test_returns_triple(self, s2_image_512, labels_512):
        """Should return (mss, labels, sf) triple."""
        result = satharmony.emulate_s2_with_labels(s2_image_512, labels_512, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_shapes_match(self, s2_image_512, labels_512):
        """Image and label shapes must match."""
        mss, mss_labels, _sf = satharmony.emulate_s2_with_labels(s2_image_512, labels_512, seed=42)
        assert mss.shape[1:] == mss_labels.shape

    def test_shapes_match_1024(self, s2_image_1024, labels_1024):
        """Shapes match for 1024x1024 input."""
        mss, mss_labels, _sf = satharmony.emulate_s2_with_labels(
            s2_image_1024, labels_1024, seed=42
        )
        assert mss.shape[1:] == mss_labels.shape

    def test_shapes_match_fixed_scale(self, s2_image_512, labels_512):
        """Shapes match with fixed scale_factor."""
        mss, mss_labels, sf = satharmony.emulate_s2_with_labels(
            s2_image_512, labels_512, scale_factor=8, seed=42
        )
        assert mss.shape[1:] == mss_labels.shape
        assert sf == 8

    def test_reproducibility(self, s2_image_512, labels_512):
        """Same seed produces identical results."""
        r1 = satharmony.emulate_s2_with_labels(s2_image_512, labels_512, seed=42)
        r2 = satharmony.emulate_s2_with_labels(s2_image_512, labels_512, seed=42)

        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])
        assert r1[2] == r2[2]


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    def test_non_divisible_size(self):
        """Non-divisible sizes should work (crops internally)."""
        image = np.random.rand(13, 500, 500).astype(np.float32)
        labels = np.random.randint(0, 4, (500, 500), dtype=np.uint8)

        mss, mss_labels, _sf = satharmony.emulate_s2_with_labels(image, labels, seed=42)
        assert mss.shape[1:] == mss_labels.shape

    def test_small_image(self):
        """Small images should work."""
        image = np.random.rand(13, 64, 64).astype(np.float32)
        labels = np.random.randint(0, 4, (64, 64), dtype=np.uint8)

        mss, mss_labels, _sf = satharmony.emulate_s2_with_labels(
            image, labels, scale_factor=8, seed=42
        )
        assert mss.shape == (4, 8, 8)
        assert mss_labels.shape == (8, 8)

    def test_all_same_label(self):
        """All same label should aggregate to that label."""
        labels = np.full((60, 60), 3, dtype=np.uint8)  # All shadow
        result = aggregate_cloudsen12(labels, scale_factor=6)
        assert np.all(result == 3)


# =============================================================================
# P0 regression tests
# =============================================================================


class TestP0Radiometry:
    def test_linear_output(self):
        from satharmony.emulators.s2_to_mss.config import RadiometricConfig, Range
        from satharmony.emulators.s2_to_mss.degradations.radiometric import (
            apply_radiometric_degradation,
        )

        cfg = RadiometricConfig()
        cfg.reflectance_boost_prob = 0.0
        cfg.saturation_model = "legacy_random"
        cfg.saturation_threshold = Range(min=0.9, max=0.9)
        cfg.decompress_after_quantization = True
        img = np.full((4, 8, 8), 0.25, dtype=np.float32)
        out = apply_radiometric_degradation(img, cfg)
        assert abs(float(out.mean()) - 0.25) < 0.01

    def test_legacy_is_compressed(self):
        from satharmony.emulators.s2_to_mss.config import RadiometricConfig, Range
        from satharmony.emulators.s2_to_mss.degradations.radiometric import (
            apply_radiometric_degradation,
        )

        cfg = RadiometricConfig()
        cfg.reflectance_boost_prob = 0.0
        cfg.saturation_model = "legacy_random"
        cfg.saturation_threshold = Range(min=0.9, max=0.9)
        cfg.decompress_after_quantization = False
        img = np.full((4, 8, 8), 0.25, dtype=np.float32)
        out = apply_radiometric_degradation(img, cfg)
        assert float(out.mean()) > 0.4

    def test_ramp_is_linear(self):
        from satharmony.emulators.s2_to_mss.config import RadiometricConfig, Range
        from satharmony.emulators.s2_to_mss.degradations.radiometric import (
            apply_radiometric_degradation,
        )

        cfg = RadiometricConfig()
        cfg.reflectance_boost_prob = 0.0
        cfg.saturation_model = "legacy_random"
        cfg.saturation_threshold = Range(min=0.9, max=0.9)
        cfg.decompress_after_quantization = True
        ramp = np.tile(np.linspace(0.02, 0.8, 256, dtype=np.float32), (4, 8, 1))
        out = apply_radiometric_degradation(ramp, cfg)
        x = ramp[0].ravel()
        y = out[0].ravel()
        slope = np.polyfit(x, y, 1)[0]
        assert 0.98 < slope < 1.02


class TestP0Spatial:
    def test_60m_from_1024(self):
        from satharmony.emulators.s2_to_mss.config import Range, SpatialConfig
        from satharmony.emulators.s2_to_mss.degradations.spatial import (
            apply_spatial_degradation,
        )

        cfg = SpatialConfig(sampling_gsd=60.0, psf_fwhm_m=Range(min=79.0, max=79.0))
        img = np.random.rand(4, 1024, 1024).astype(np.float32)
        out = apply_spatial_degradation(img, cfg)
        assert out.shape == (4, 170, 170)

    def test_psf_independent_of_sampling(self):
        from satharmony.emulators.s2_to_mss.config import Range, SpatialConfig
        from satharmony.emulators.s2_to_mss.degradations.spatial import (
            apply_spatial_degradation,
        )

        img = np.random.rand(4, 1024, 1024).astype(np.float32)
        cfg60 = SpatialConfig(sampling_gsd=60.0, psf_fwhm_m=Range(min=79.0, max=79.0))
        cfg80 = SpatialConfig(sampling_gsd=80.0, psf_fwhm_m=Range(min=79.0, max=79.0))
        assert apply_spatial_degradation(img, cfg60).shape == (4, 170, 170)
        assert apply_spatial_degradation(img, cfg80).shape == (4, 128, 128)

    def test_centered_decimation(self):
        from satharmony.emulators.s2_to_mss.config import Range, SpatialConfig
        from satharmony.emulators.s2_to_mss.degradations.spatial import (
            apply_spatial_degradation,
        )

        cfg = SpatialConfig(sampling_gsd=60.0, psf_fwhm_m=Range(min=79.0, max=79.0))
        ramp = np.tile(np.arange(600, dtype=np.float32) / 600, (4, 600, 1))
        out = apply_spatial_degradation(ramp, cfg)
        x_sample = out[0].mean(axis=0) * 600
        x_label = np.arange(out.shape[2]) * 6 + 2.5
        diff = (x_sample - x_label)[5:-5]
        assert abs(float(diff.mean())) < 0.1


class TestP0Labels:
    def test_thick_and_thin_encoding(self):
        labels = np.zeros((6, 6), dtype=np.uint8)
        labels[:4, :] = 1  # 24 thick pixels
        assert aggregate_cloudsen12(labels, scale_factor=6)[0, 0] == 1

        labels = np.zeros((6, 6), dtype=np.uint8)
        labels[:4, :] = 2  # 24 thin pixels
        assert aggregate_cloudsen12(labels, scale_factor=6)[0, 0] == 2

    def test_contaminated_block_not_clear(self):
        # 15 clear, 12 thick, 9 shadow: 58 % contaminated but clear is the
        # largest single class. New rule must not return clear.
        labels = np.zeros((6, 6), dtype=np.uint8)
        labels.ravel()[15:27] = 1
        labels.ravel()[27:36] = 3
        assert aggregate_cloudsen12(labels, scale_factor=6)[0, 0] == 1
        assert aggregate_cloudsen12(labels, scale_factor=6, legacy=True)[0, 0] == 0

    def test_priority_tie(self):
        # 12 thick, 12 shadow, 12 clear: tie -> thick (priority 1)
        labels = np.zeros((6, 6), dtype=np.uint8)
        labels.ravel()[13:25] = 1
        labels.ravel()[25:37] = 3
        assert aggregate_cloudsen12(labels, scale_factor=6)[0, 0] == 1


class TestCalibratedProfile:
    def test_profile_fields(self):
        cfg = PipelineConfig.mss5_esa_gtc_v1()
        assert cfg.spatial.sampling_gsd == 60.0
        assert cfg.spatial.center_decimation is True
        assert cfg.spatial.psf_fwhm_m is not None
        assert cfg.radiometric.decompress_after_quantization is True
        assert cfg.labels_legacy is False

    def test_profile_shape(self, s2_image_512):
        cfg = PipelineConfig.mss5_esa_gtc_v1()
        mss, sf = satharmony.emulate_s2(s2_image_512, seed=42, config=cfg)
        assert sf == 6
        assert mss.shape[1:] == (512 // 6, 512 // 6)

    def test_band_scaling_calibrated_on_pics(self):
        """G2: the five PICS give 0.956 / 0.933 / 0.867 / 0.933 per band."""
        scaling = PipelineConfig.mss5_esa_gtc_v1().final_scaling.band_scaling
        assert (scaling.green.mean, scaling.red.mean, scaling.nir1.mean, scaling.nir2.mean) == (
            pytest.approx(0.956),
            pytest.approx(0.933),
            pytest.approx(0.867),
            pytest.approx(0.933),
        )
        # The water-vapour term removes the 0.94 um absorption: B4 now sits
        # with B2 instead of below B3. The remaining offset is calibration.
        assert abs(scaling.nir2.mean - scaling.red.mean) < 0.005
        assert scaling.nir2.mean > scaling.nir1.mean

    def test_band_scaling_variation_covers_site_spread(self):
        """mean +- variation is a 2 sigma window over the five PICS.

        At most one site may sit outside it: Egypt-1 is the low outlier in all
        four bands (its MSS/date pair is the darkest relative to S2).
        """
        scaling = PipelineConfig.mss5_esa_gtc_v1().final_scaling.band_scaling
        per_site = {
            "green": [0.981, 0.963, 0.942, 0.923, 0.956],  # L4, Alg3, Nig1, Egy1, Ar2
            "red": [0.940, 0.933, 0.937, 0.886, 0.933],
            "nir1": [0.867, 0.859, 0.900, 0.839, 0.891],
            "nir2": [0.941, 0.909, 0.933, 0.871, 0.967],
        }
        for name, values in per_site.items():
            band = getattr(scaling, name)
            assert band.variation >= 0.03
            outside = [
                v
                for v in values
                if not band.mean - band.variation <= v <= band.mean + band.variation
            ]
            assert len(outside) <= 1, f"{name}: {outside} outside the 2 sigma window"
            assert band.mean <= max(values) and band.mean >= min(values)

    def test_sun_elevation_is_measured(self):
        """The 178 MSS/TM pairs give mean 42.8, std 12.25, range 7.6-62.3."""
        sun = PipelineConfig.mss5_esa_gtc_v1().radiometric.sun_elevation_deg
        assert sun.distribution == "truncated_normal"
        assert (sun.min, sun.max) == pytest.approx((7.6, 62.3))
        assert sun.mean == pytest.approx(42.8)
        assert sun.std == pytest.approx(12.25)
        np.random.seed(0)
        samples = np.array([sun.sample() for _ in range(10000)])
        assert samples.min() >= sun.min and samples.max() <= sun.max
        assert abs(samples.mean() - 42.8) < 0.5
        # Clipping the 5 % tail at 62.3 lowers the effective std to ~11.6;
        # the stored 12.25 is the measured population std.
        assert abs(samples.std() - 12.25) < 1.0

    def test_legacy_profile_does_not_use_sun_elevation(self):
        """The 0.1.13 profile keeps its random threshold and its old field."""
        legacy = PipelineConfig.legacy_0_1_13()
        assert legacy.radiometric.saturation_model == "legacy_random"
        sun = legacy.radiometric.sun_elevation_deg
        assert sun.distribution == "uniform"
        assert (sun.min, sun.max) == pytest.approx((25.0, 70.0))

    def test_legacy_profile_keeps_its_own_scaling(self):
        """Recalibrating the PICS must not touch legacy_0_1_13()."""
        legacy = PipelineConfig.legacy_0_1_13().final_scaling.band_scaling
        assert (legacy.green.mean, legacy.red.mean) == (0.60, 0.58)
        assert (legacy.nir1.mean, legacy.nir2.mean) == (0.60, 0.85)


# =============================================================================
# Gate 0: one regression test per audit error (07-auditoria-satharmony.md)
# =============================================================================


class TestGate0Audit:
    def test_cloudsen12_thick_thin_encoding(self):
        """2.7.1: CloudSEN12 Table 3 is 0 clear, 1 thick, 2 thin, 3 shadow."""
        from satharmony.emulators.s2_to_mss.labels import (
            CLASS_NAMES,
            CLEAR,
            SHADOW,
            THICK_CLOUD,
            THIN_CLOUD,
        )

        assert (CLEAR, THICK_CLOUD, THIN_CLOUD, SHADOW) == (0, 1, 2, 3)
        assert CLASS_NAMES[THICK_CLOUD] == "thick_cloud"
        assert CLASS_NAMES[THIN_CLOUD] == "thin_cloud"

        # A block that is all thick must stay thick (1) and all thin must
        # stay thin (2); 0.1.13 returned the swapped values.
        thick = np.full((6, 6), 1, dtype=np.uint8)
        thin = np.full((6, 6), 2, dtype=np.uint8)
        assert aggregate_cloudsen12(thick, scale_factor=6)[0, 0] == 1
        assert aggregate_cloudsen12(thin, scale_factor=6)[0, 0] == 2

        # Tie between thick and thin goes to thick (priority thick > thin).
        tie = np.zeros((6, 6), dtype=np.uint8)
        tie.ravel()[:18] = 1
        tie.ravel()[18:] = 2
        assert aggregate_cloudsen12(tie, scale_factor=6)[0, 0] == 1

    def test_contaminated_block_10_11_15_is_not_clear(self):
        """2.7.2: 58 % contaminated (10 thick, 11 shadow, 15 clear) is not clear."""
        labels = np.zeros((6, 6), dtype=np.uint8)
        labels.ravel()[:10] = 1  # thick
        labels.ravel()[10:21] = 3  # shadow
        # the remaining 15 pixels stay clear

        assert aggregate_cloudsen12(labels, scale_factor=6)[0, 0] == 3
        assert aggregate_cloudsen12(labels, scale_factor=6, legacy=True)[0, 0] == 0

        # The fraction is an explicit knob: at 60 % the same block is clear.
        assert (
            aggregate_cloudsen12(labels, scale_factor=6, min_contaminated_fraction=0.60)[0, 0] == 0
        )

    def test_sqrt_companding_is_inverted(self):
        """2.1: sqrt companding is quantized and then inverted to linear."""
        from satharmony.emulators.s2_to_mss.config import RadiometricConfig, Range
        from satharmony.emulators.s2_to_mss.degradations.radiometric import (
            apply_radiometric_degradation,
        )

        sat = 0.875
        cfg = RadiometricConfig()
        cfg.reflectance_boost_prob = 0.0
        cfg.saturation_model = "legacy_random"
        cfg.saturation_threshold = Range(min=sat, max=sat)
        cfg.decompress_after_quantization = True

        for rho in (0.02, 0.05, 0.1, 0.5):
            image = np.full((4, 8, 8), rho, dtype=np.float32)
            out = apply_radiometric_degradation(image, cfg)
            # One 6-bit step in the compressed domain, propagated linearly.
            step = 2.0 * np.sqrt(rho / sat) * sat / 63.0
            assert abs(float(out[0].mean()) - rho) <= step + 1e-6

        # Legacy path keeps the sqrt and is therefore compressed.
        cfg.decompress_after_quantization = False
        legacy = apply_radiometric_degradation(np.full((4, 8, 8), 0.05, dtype=np.float32), cfg)
        assert float(legacy[0].mean()) > 0.10

    def test_centered_decimation_centroid(self):
        """2.2: image and label blocks share the same subpixel centre."""
        from satharmony.emulators.s2_to_mss.config import Range, SpatialConfig
        from satharmony.emulators.s2_to_mss.degradations.spatial import (
            apply_spatial_degradation,
        )

        cfg = SpatialConfig(sampling_gsd=60.0, psf_fwhm_m=Range(min=79.0, max=79.0))
        n = 600
        ramp = np.tile(np.arange(n, dtype=np.float32) / (n - 1), (4, n, 1))
        out = apply_spatial_degradation(ramp, cfg)

        # Label block j is centred on j*sf + (sf-1)/2.
        x_label = np.arange(out.shape[2]) * 6 + 2.5
        y = out[0].mean(axis=0)
        slope, intercept = np.polyfit(x_label[5:-5], y[5:-5], 1)
        shift_px = -intercept / slope
        assert abs(float(shift_px)) < 0.05

    def test_default_sampling_is_60m_and_grid_is_explicit(self):
        """2.3: default output is 60 m and the grid cannot silently drift."""
        from satharmony.emulators.s2_to_mss.config import Range, SpatialConfig
        from satharmony.emulators.s2_to_mss.degradations.spatial import (
            apply_spatial_degradation,
        )

        cfg = PipelineConfig()
        assert cfg.spatial.sampling_gsd == 60.0
        assert cfg.spatial.center_decimation is True
        assert cfg.spatial.psf_fwhm_m is not None

        image = np.zeros((4, 1024, 1024), dtype=np.float32)
        assert apply_spatial_degradation(image, cfg.spatial).shape == (4, 170, 170)

        # PSF is independent of the 60 m grid: 82 m still gives 170x170.
        cfg82 = SpatialConfig(sampling_gsd=60.0, psf_fwhm_m=Range(min=82.0, max=82.0))
        assert apply_spatial_degradation(image, cfg82).shape == (4, 170, 170)

        # 65 m would round to 60/70 m; refuse instead of changing the GSD.
        cfg_bad = SpatialConfig(sampling_gsd=65.0, psf_fwhm_m=Range(min=79.0, max=79.0))
        with pytest.raises(ValueError, match="integer multiple"):
            apply_spatial_degradation(image, cfg_bad)

    def test_real_mss_units_contract(self):
        """4: real MSS must be TOA 0-1, never raw DN or surface reflectance."""
        from datetime import datetime

        from satharmony import assert_mss_toa_units, mss_to_toa
        from satharmony.corrections import MSSCalibration

        toa = np.linspace(0.0, 1.0, 4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
        assert assert_mss_toa_units(toa).dtype == np.float32

        # uint16 TOA files (0-10000) become 0-1 with the dataloader scale.
        raw_toa = (toa * 10000).astype(np.uint16)
        np.testing.assert_allclose(
            mss_to_toa(raw_toa, units="toa", scale=1 / 10000), toa, atol=1e-4
        )

        # Raw DN 0-255 is not TOA and must be rejected.
        dn = np.full((4, 8, 8), 100, dtype=np.uint8)
        with pytest.raises(ValueError, match="outside TOA"):
            assert_mss_toa_units(dn, units="toa")
        with pytest.raises(ValueError, match="surface reflectance"):
            assert_mss_toa_units(toa, units="sr")
        with pytest.raises(ValueError, match="MSSCalibration"):
            mss_to_toa(dn, units="dn")

        calib = MSSCalibration(
            spacecraft_id="LANDSAT_5",
            sensor_id="MSS",
            date_acquired=datetime(1985, 6, 1),
            sun_elevation=45.0,
            radiance_mult=[0.5, 0.5, 0.5, 0.5],
            radiance_add=[0.0, 0.0, 0.0, 0.0],
            qcal_min=[0, 0, 0, 0],
            qcal_max=[255, 255, 255, 255],
        )
        converted = mss_to_toa(dn, units="dn", calib=calib)
        assert np.isfinite(converted).all()
        assert float(converted.max()) < 1.5


# =============================================================================
# P1: physical saturation and 474 m striping (07-auditoria-satharmony.md)
# =============================================================================


class TestP1Saturation:
    def test_l5_threshold_matches_lmax_over_esun(self):
        """2.5: rho_sat = pi * LMAX * d^2 / (ESUN * sin(theta)), per band."""
        from satharmony.corrections import LMAX_MSS, saturation_reflectance

        sat = saturation_reflectance("L5_MSS", 90.0)
        expected = (
            np.pi * np.asarray(LMAX_MSS["L5_MSS"]) / np.asarray([1824.0, 1570.0, 1249.0, 853.4])
        )
        np.testing.assert_allclose(sat, expected, rtol=1e-5)

        # The audit's bands 2 and 3: 0.358 / sin(theta) and 0.372 / sin(theta).
        assert sat[1] == pytest.approx(0.358, abs=1e-3)
        assert sat[2] == pytest.approx(0.372, abs=1e-3)

    def test_lower_sun_raises_saturation_reflectance(self):
        """2.5: a lower sun needs a brighter scene to reach the same radiance."""
        from satharmony.corrections import saturation_reflectance

        sat30 = saturation_reflectance("L5_MSS", 30.0)
        sat45 = saturation_reflectance("L5_MSS", 45.0)
        sat70 = saturation_reflectance("L5_MSS", 70.0)
        assert np.all(sat30 > sat45)
        assert np.all(sat45 > sat70)
        # pi * 179 / (1570 * sin 30) = 0.716 for L5 band 2.
        assert sat30[1] == pytest.approx(0.716, abs=1e-3)

    def test_physical_threshold_is_band_dependent(self):
        """2.5: one random number for every band was the old bug."""
        from satharmony.corrections import saturation_reflectance

        sat = saturation_reflectance("L5_MSS", 90.0)
        assert len(np.unique(np.round(sat, 4))) == 4
        # L5: B1 0.462 > B4 0.453 > B3 0.372 > B2 0.358.
        assert sat[0] > sat[3] > sat[2] > sat[1]

    def test_invalid_saturation_inputs_raise(self):
        from satharmony.corrections import saturation_reflectance
        from satharmony.emulators.s2_to_mss.config import RadiometricConfig, Range
        from satharmony.emulators.s2_to_mss.degradations.radiometric import (
            apply_radiometric_degradation,
        )

        with pytest.raises(ValueError, match="unknown spacecraft"):
            saturation_reflectance("L7_MSS")
        with pytest.raises(ValueError, match="sun_elevation"):
            saturation_reflectance("L5_MSS", 0.0)
        with pytest.raises(ValueError, match="earth_sun_distance_au"):
            saturation_reflectance("L5_MSS", earth_sun_distance_au=0.0)

        # A 5-band image has no LMAX for the fifth band: fail instead of
        # reusing the L5 threshold.
        cfg = RadiometricConfig()
        cfg.reflectance_boost_prob = 0.0
        cfg.sun_elevation_deg = Range(min=45.0, max=45.0)
        with pytest.raises(ValueError, match="radiometric saturation"):
            apply_radiometric_degradation(np.full((5, 4, 4), 0.2, dtype=np.float32), cfg)

    def test_bright_cloud_clips_at_the_physical_threshold(self):
        """2.5: a cloud brighter than LMAX saturates at rho_sat in every band."""
        from satharmony.corrections import saturation_reflectance
        from satharmony.emulators.s2_to_mss.config import RadiometricConfig, Range
        from satharmony.emulators.s2_to_mss.degradations.radiometric import (
            apply_radiometric_degradation,
        )

        cfg = RadiometricConfig()
        cfg.reflectance_boost_prob = 0.0
        cfg.saturation_model = "physical"
        cfg.saturation_threshold = Range(min=0.99, max=0.99)  # ignored
        cfg.sun_elevation_deg = Range(min=45.0, max=45.0)

        image = np.full((4, 8, 8), 0.9, dtype=np.float32)
        out = apply_radiometric_degradation(image, cfg)
        sat = saturation_reflectance("L5_MSS", 45.0)
        for band in range(4):
            assert float(out[band].max()) <= float(sat[band]) + 1e-6
            np.testing.assert_allclose(out[band], sat[band], atol=1e-6)

        # The same scene at a lower sun saturates later.
        cfg_low = cfg.model_copy(deep=True)
        cfg_low.sun_elevation_deg = Range(min=30.0, max=30.0)
        out_low = apply_radiometric_degradation(image, cfg_low)
        np.testing.assert_allclose(out_low[1], saturation_reflectance("L5_MSS", 30.0)[1], atol=1e-6)
        assert float(out_low[1].mean()) > float(out[1].mean())

    def test_legacy_profile_keeps_random_saturation(self):
        """The 0.1.13 reproduction must not switch to the physical model."""
        cfg = PipelineConfig.legacy_0_1_13()
        assert cfg.radiometric.saturation_model == "legacy_random"
        assert PipelineConfig().radiometric.saturation_model == "physical"
        assert PipelineConfig.mss5_esa_gtc_v1().radiometric.saturation_model == "physical"


class TestP1Striping:
    def test_period_is_474m_not_six_rows(self):
        """2.6: 474 m / 60 m = 7.9 px, not 6 rows."""
        from satharmony.emulators.s2_to_mss.degradations.striping import (
            detector_line_index,
        )

        idx = detector_line_index((79, 16), sampling_gsd=60.0, yaw_deg=0.0)
        assert idx[0, 0] == 0
        assert idx[6, 0] != idx[0, 0]  # the old 6-row period
        assert idx[8, 0] == idx[0, 0]  # first return of the same detector
        assert (idx[7, 0], idx[8, 0]) == (5, 0)
        # 79 rows = 10 sweeps of 474 m.
        transitions = np.sum((idx[1:, 0] == 1) & (idx[:-1, 0] == 0))
        assert transitions == 10

    def test_scan_lines_are_tilted_not_row_aligned(self):
        """2.6: the trace is tilted ~9 deg from the image rows."""
        from satharmony.emulators.s2_to_mss.degradations.striping import (
            detector_phase,
        )

        phase = detector_phase((40, 120), sampling_gsd=60.0, yaw_deg=9.0)
        gy = float(phase[1, 0] - phase[0, 0])
        gx = float(phase[0, 1] - phase[0, 0])
        assert gx > 0.0
        # dy/dx of a scan line = -tan(yaw) = -0.158.
        assert gy / gx == pytest.approx(1.0 / np.tan(np.deg2rad(9.0)), rel=1e-3)

        # yaw=0 is the row-aligned legacy geometry.
        flat = detector_phase((40, 120), sampling_gsd=60.0, yaw_deg=0.0)
        assert float(flat[0, 1] - flat[0, 0]) == 0.0

    def test_striping_applied_along_scan_lines(self):
        """2.6: banding repeats every 7.9 rows and is not constant per column."""
        from satharmony.emulators.s2_to_mss.config import Range, StripingConfig
        from satharmony.emulators.s2_to_mss.degradations.striping import (
            apply_striping,
        )

        cfg = StripingConfig(
            gain_std=Range(min=0.2, max=0.2),
            offset_std=Range(min=0.0, max=0.0),
            per_band_variation=False,
            yaw_deg=0.0,
        )
        image = np.ones((4, 79, 32), dtype=np.float32)
        np.random.seed(0)
        out = apply_striping(image, cfg)

        rows = out[0].mean(axis=1)
        assert len(np.unique(np.round(rows, 4))) == 6  # six detectors
        assert not np.allclose(rows[:6], rows[6:12])  # not the old 6-row period
        assert np.allclose(out[0, :, 0], out[0, :, -1])  # yaw 0: columns equal

        cfg9 = cfg.model_copy(deep=True)
        cfg9.yaw_deg = 9.0
        np.random.seed(0)
        out9 = apply_striping(image, cfg9)
        assert not np.allclose(out9[0, :, 0], out9[0, :, -1])

    def test_legacy_profile_keeps_row_striping(self):
        cfg = PipelineConfig.legacy_0_1_13()
        assert cfg.striping.detector_geometry == "rows"
        assert PipelineConfig().striping.detector_geometry == "scan_lines"

    def test_row_geometry_reproduces_legacy_six_row_period(self):
        """legacy_0_1_13(): one detector per row, 6-row period."""
        from satharmony.emulators.s2_to_mss.config import Range, StripingConfig
        from satharmony.emulators.s2_to_mss.degradations.striping import (
            apply_striping,
        )

        cfg = StripingConfig(
            detector_geometry="rows",
            gain_std=Range(min=0.2, max=0.2),
            offset_std=Range(min=0.0, max=0.0),
            per_band_variation=False,
        )
        np.random.seed(0)
        out = apply_striping(np.ones((4, 12, 8), dtype=np.float32), cfg)
        rows = out[0].mean(axis=1)
        assert len(np.unique(np.round(rows, 4))) == 6
        np.testing.assert_allclose(rows[:6], rows[6:12])

    def test_geometry_rejects_impossible_grids(self):
        from satharmony.emulators.s2_to_mss.degradations.striping import (
            detector_line_index,
        )

        with pytest.raises(ValueError, match="sampling_gsd"):
            detector_line_index((4, 4), sampling_gsd=0.0)
        with pytest.raises(ValueError, match="scan_line_period_m"):
            detector_line_index((4, 4), scan_line_period_m=-1.0)
        with pytest.raises(ValueError, match="num_detectors"):
            detector_line_index((4, 4), num_detectors=0)


# =============================================================================
# Test: MTL metadata (Landsat 1-3 uses bands 4-7)
# =============================================================================


def _mtl_text(bands, spacecraft="LANDSAT_5"):
    """MTL excerpt with the keys parse_mtl reads, in the USGS layout."""
    lines = [
        "GROUP = PRODUCT_METADATA",
        f'  SPACECRAFT_ID = "{spacecraft}"',
        '  SENSOR_ID = "MSS"',
        "  DATE_ACQUIRED = 1993-12-05",
        "  SUN_ELEVATION = 31.21",
        "END_GROUP = PRODUCT_METADATA",
        "GROUP = RADIOMETRIC_RESCALING",
    ]
    for i, b in enumerate(bands):
        lines.append(f"  RADIANCE_MULT_BAND_{b} = {0.41 + i * 0.01:.2f}E+00")
        lines.append(f"  RADIANCE_ADD_BAND_{b} = -0.10")
        lines.append(f"  QUANTIZE_CAL_MIN_BAND_{b} = 1")
        lines.append(f"  QUANTIZE_CAL_MAX_BAND_{b} = 127")
    lines.append("END_GROUP = RADIOMETRIC_RESCALING")
    lines.append("END")
    return "\n".join(lines) + "\n"


class TestParseMTL:
    def test_landsat_5_reads_bands_1_to_4(self, tmp_path):
        mtl = tmp_path / "L5015_MTL.txt"
        mtl.write_text(_mtl_text([1, 2, 3, 4]))

        cal = satharmony.parse_mtl(mtl)

        assert cal.spacecraft_short == "L5_MSS"
        assert cal.radiance_mult == pytest.approx([0.41, 0.42, 0.43, 0.44])
        assert cal.radiance_add == [-0.10] * 4
        assert cal.qcal_min == [1] * 4
        assert cal.qcal_max == [127] * 4
        assert cal.doy == 339

    def test_landsat_1_to_3_reads_bands_4_to_7(self, tmp_path):
        """Audit error: the parser demanded BAND_1..4 and failed on L1-L3."""
        mtl = tmp_path / "L3015_MTL.txt"
        mtl.write_text(_mtl_text([4, 5, 6, 7], spacecraft="LANDSAT_3"))

        cal = satharmony.parse_mtl(mtl)

        assert cal.spacecraft_short == "L3_MSS"
        assert cal.radiance_mult == pytest.approx([0.41, 0.42, 0.43, 0.44])
        assert cal.sun_elevation == pytest.approx(31.21)

    def test_mtl_without_mss_bands_raises(self, tmp_path):
        mtl = tmp_path / "bad_MTL.txt"
        mtl.write_text('GROUP = PRODUCT_METADATA\n  SPACECRAFT_ID = "LANDSAT_5"\nEND\n')

        with pytest.raises(ValueError, match=r"BAND_1\.\.4"):
            satharmony.parse_mtl(mtl)


# =============================================================================
# Test: real Landsat 1-3 MTL files (assets/docs/mtl of the vault)
# =============================================================================

#: Real USGS Collection 2 L1 products, copied verbatim from the USGS archive
#: (``landsateuwest`` blob, the Collection 2 mirror USGS publishes through the
#: Planetary Computer STAC API) by ``scripts/fetch_mtl_landsat123.py``.
REAL_MTL_DIR = Path(
    os.environ.get(
        "SATHARMONY_MTL_DIR",
        Path(__file__).resolve().parents[2] / "assets" / "docs" / "mtl",
    )
)

REAL_MTLS = {
    "LM01_L1GS_043044_19780107_02_T2_MTL.txt": {
        "spacecraft": "L1_MSS",
        "date": "1978-01-07",
        "sun_elevation": 18.39795394,
        "radiance_mult": [None, 0.64843, 0.65236, 0.60866],
        "radiance_add": [None, -0.74843, -0.75236, -0.60866],
        "qcal": [(None, None), (1, 255), (1, 255), (1, 255)],
    },
    "LM02_L1GS_223027_19820312_02_T2_MTL.txt": {
        "spacecraft": "L2_MSS",
        "date": "1982-03-12",
        "sun_elevation": 33.05592204,
        "radiance_mult": [1.0598, 0.61496, 0.53386, 0.45787],
        "radiance_add": [-9.05984, 4.18504, 4.06614, 3.14213],
        "qcal": [(1, 255)] * 4,
    },
    "LM03_L1TP_077017_19830331_02_T2_MTL.txt": {
        "spacecraft": "L3_MSS",
        "date": "1983-03-31",
        "sun_elevation": 30.90851592,
        "radiance_mult": [1.0827, 0.72559, 0.62756, 0.50787],
        "radiance_add": [-6.48268, 0.97441, 2.57244, 0.19213],
        "qcal": [(1, 255)] * 4,
    },
}


def _real_mtl(name: str) -> Path:
    path = Path(REAL_MTL_DIR) / name
    if not path.exists():
        pytest.skip(f"real MTL not downloaded: {path}")
    return path


class TestParseRealMTL:
    """The parser against the real products, not a synthetic excerpt."""

    @pytest.mark.parametrize("name", sorted(REAL_MTLS))
    def test_real_landsat_1_to_3_mtl(self, name):
        expected = REAL_MTLS[name]
        cal = satharmony.parse_mtl(_real_mtl(name))

        assert cal.spacecraft_short == expected["spacecraft"]
        assert cal.date_acquired.date().isoformat() == expected["date"]
        assert cal.sun_elevation == pytest.approx(expected["sun_elevation"])
        assert cal.radiance_mult == pytest.approx(expected["radiance_mult"])
        assert cal.radiance_add == pytest.approx(expected["radiance_add"])
        assert list(zip(cal.qcal_min, cal.qcal_max, strict=True)) == expected["qcal"]

    def test_real_landsat_1_has_no_band_4_calibration(self):
        """Landsat 1 MSS band 4 is 'NULL' in the MTL: parse, do not radiance."""
        cal = satharmony.parse_mtl(
            _real_mtl("LM01_L1GS_043044_19780107_02_T2_MTL.txt")
        )

        assert cal.is_fully_calibrated is False
        assert cal.null_bands == (1,)
        assert cal.qcal_min[0] is None and cal.qcal_max[0] is None

        dn = np.full((4, 2, 2), 40, dtype=np.uint8)
        with pytest.raises(ValueError, match="no radiance calibration for band"):
            satharmony.dn_to_radiance(dn, cal)

    def test_real_landsat_2_and_3_convert_to_radiance(self):
        for name in ("LM02_L1GS_223027_19820312_02_T2_MTL.txt",
                     "LM03_L1TP_077017_19830331_02_T2_MTL.txt"):
            cal = satharmony.parse_mtl(_real_mtl(name))
            assert cal.is_fully_calibrated

            dn = np.full((4, 8, 8), 100, dtype=np.uint8)
            radiance = satharmony.dn_to_radiance(dn, cal)

            expected = np.array(cal.radiance_mult) * 100 + np.array(cal.radiance_add)
            np.testing.assert_allclose(radiance[:, 0, 0], expected, rtol=1e-6)
            assert np.isfinite(radiance).all()



# =============================================================================
# Test: legacy_0_1_13() reproduces the published 0.1.13 wheel bit for bit
# =============================================================================


def _legacy_reference_input():
    """Fixed input whose sha256 is pinned below (no RNG drift allowed)."""
    rng = np.random.default_rng(7)
    base = rng.random((13, 128, 128), dtype=np.float32)
    base[:, :32, :32] = 0.25  # flat-spectrum patch
    base[:, 32:64, 64:96] = 0.9  # bright cloud, saturates
    return np.ascontiguousarray(base, dtype=np.float32)


class TestLegacyBitForBit:
    """Hashes computed with the satharmony 0.1.13 wheel from PyPI.

    Input 80779ada..., MSS 12bc8d11..., labels 1446a21c...
    """

    def test_reference_input_is_unchanged(self):
        s2 = _legacy_reference_input()
        assert (
            hashlib.sha256(s2.tobytes()).hexdigest()
            == "80779adaea2bd78f1ddba7024749f8e542c0e358b78fa2333f37784fa950e460"
        )

    def test_mss_is_byte_identical_to_0_1_13(self):
        s2 = _legacy_reference_input()
        mss, scale_factor = satharmony.emulate_s2(
            s2, config=PipelineConfig.legacy_0_1_13(), seed=42
        )
        assert scale_factor == 7
        assert mss.shape == (4, 18, 18)
        assert (
            hashlib.sha256(np.ascontiguousarray(mss).tobytes()).hexdigest()
            == "12bc8d11a2a8cf1e5120f4241c5b47b5d1a7a53a7dd3ecb52cb39ccd74897853"
        )

    def test_labels_are_byte_identical_to_0_1_13(self):
        s2 = _legacy_reference_input()
        _, scale_factor = satharmony.emulate_s2(
            s2, config=PipelineConfig.legacy_0_1_13(), seed=42
        )
        labels = np.zeros((128, 128), dtype=np.uint8)
        labels[32:64, 64:96] = 1
        labels[96:, :] = 2
        out = satharmony.emulate_labels(labels, scale_factor=scale_factor)
        assert out.dtype == np.uint8
        assert (
            hashlib.sha256(np.ascontiguousarray(out).tobytes()).hexdigest()
            == "1446a21c4172e1842e368e51f481a4b98e9c6c405a7daeaa3ebdbada11f62126"
        )
