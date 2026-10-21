"""Tests for satharmony package."""

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
        labels[:4, :5] = 2   # 20 cloud
        labels[4:, :5] = 3   # 10 shadow
        labels[:, 5] = 0     # 6 clear
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
        mss, mss_labels, sf = satharmony.emulate_s2_with_labels(
            s2_image_512, labels_512, seed=42
        )
        assert mss.shape[1:] == mss_labels.shape

    def test_shapes_match_1024(self, s2_image_1024, labels_1024):
        """Shapes match for 1024x1024 input."""
        mss, mss_labels, sf = satharmony.emulate_s2_with_labels(
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

        mss, mss_labels, sf = satharmony.emulate_s2_with_labels(
            image, labels, seed=42
        )
        assert mss.shape[1:] == mss_labels.shape

    def test_small_image(self):
        """Small images should work."""
        image = np.random.rand(13, 64, 64).astype(np.float32)
        labels = np.random.randint(0, 4, (64, 64), dtype=np.uint8)

        mss, mss_labels, sf = satharmony.emulate_s2_with_labels(
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
