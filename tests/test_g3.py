"""Tests for the G3 work: CloudSEN12 chips, pseudo-MSS fixture, domain metrics
and the human-label package."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

import satharmony
from satharmony.datasets.cloudsen12 import (
    BAND_PERMUTATION,
    CLOUDSEN12_BAND_ORDER,
    PACKAGE_BAND_ORDER,
    array_sha256,
    class_fractions,
    is_shadow_rich,
    regime_of,
    select_window,
    to_package_order,
    valid_cells,
)
from satharmony.validation import (
    STRIPING_WAVELENGTH_PX,
    high_pass,
    ks_per_band,
    radial_psd,
    striping_metrics,
    wasserstein_per_band,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path("/data/databases/CLOUD_MSS/calibration/pseudo_mss_fixture")
LABELS = REPO.parent / "assets/md/fuentes/labels-humano"
DOMAIN = REPO.parent / "assets/md/fuentes/domain"

sys.path.insert(0, str(REPO / "scripts"))
from label_agreement import (  # noqa: E402
    cohen_kappa,
    compare_csv,
    pixel_kappa,
    read_annotation_csv,
    validate_fractions,
)

# =============================================================================
# Band order and label aggregation
# =============================================================================


class TestBandOrder:
    def test_orders_have_thirteen_bands(self):
        assert len(CLOUDSEN12_BAND_ORDER) == 13
        assert len(PACKAGE_BAND_ORDER) == 13

    def test_permutation_is_a_bijection(self):
        assert sorted(BAND_PERMUTATION) == list(range(13))

    def test_b8a_moves_to_the_end(self):
        assert CLOUDSEN12_BAND_ORDER[8] == "B8A"
        assert PACKAGE_BAND_ORDER[12] == "B8A"
        assert BAND_PERMUTATION[12] == 8
        assert BAND_PERMUTATION[8] == 9  # B9 stays the 945 nm band

    def test_flat_spectrum_is_identity(self):
        stack = np.full((13, 4, 4), 0.3, dtype=np.float32)
        assert np.array_equal(to_package_order(stack), stack)

    def test_each_band_keeps_its_value(self):
        stack = np.zeros((13, 2, 2), dtype=np.float32)
        for index in range(13):
            stack[index] = index + 1
        reordered = to_package_order(stack)
        for package_index, band in enumerate(PACKAGE_BAND_ORDER):
            expected = CLOUDSEN12_BAND_ORDER.index(band) + 1
            assert np.all(reordered[package_index] == expected)

    def test_wrong_band_count_raises(self):
        with pytest.raises(ValueError, match="13"):
            to_package_order(np.zeros((12, 4, 4)))


class TestLabelHelpers:
    def test_fractions_sum_to_one_and_include_contaminated(self):
        labels = np.array([[0, 0, 1, 2], [3, 3, 0, 0]], dtype=np.uint8)
        fractions = class_fractions(labels)
        assert pytest.approx(sum(fractions[name] for name in ("clear", "thick_cloud",
                                                              "thin_cloud", "cloud_shadow"))) == 1.0
        assert fractions["cloud_shadow"] == 0.25
        assert fractions["contaminated"] == 0.5

    def test_fractions_ignore_extra_values(self):
        labels = np.array([[0, 254, 254, 254]], dtype=np.uint8)
        fractions = class_fractions(labels, ignore=(254,))
        assert fractions["clear"] == 1.0

    def test_empty_raster_returns_zeros(self):
        fractions = class_fractions(np.array([], dtype=np.uint8))
        assert fractions["clear"] == 0.0
        assert fractions["contaminated"] == 0.0

    @pytest.mark.parametrize(
        ("contaminated", "expected"),
        [(0.0, "clear"), (0.01, "clear"), (0.05, "low_cloud"),
         (0.20, "mid_cloud"), (0.80, "high_cloud")],
    )
    def test_regime_bins(self, contaminated, expected):
        assert regime_of({"contaminated": contaminated}) == expected

    def test_shadow_flag(self):
        assert is_shadow_rich({"cloud_shadow": 0.12})
        assert not is_shadow_rich({"cloud_shadow": 0.05})

    def test_valid_cells_require_the_whole_block(self):
        valid = np.ones((12, 12), dtype=bool)
        assert valid_cells(valid, 6).all()
        valid[2, 3] = False
        cells = valid_cells(valid, 6)
        assert not cells[0, 0]
        assert cells[1, 1]

    def test_select_window_respects_the_valid_block(self):
        valid = np.zeros((20, 20), dtype=bool)
        valid[5:15, 6:16] = True
        rng = np.random.default_rng(0)
        window = select_window(valid, 8, rng)
        assert window is not None
        row, col = window
        assert valid[row : row + 8, col : col + 8].all()

    def test_select_window_returns_none_when_too_small(self):
        valid = np.ones((10, 10), dtype=bool)
        assert select_window(valid, 11, np.random.default_rng(0)) is None

    def test_select_window_is_seeded(self):
        valid = np.ones((20, 20), dtype=bool)
        first = select_window(valid, 8, np.random.default_rng(7))
        second = select_window(valid, 8, np.random.default_rng(7))
        assert first == second


class TestHash:
    def test_same_arrays_same_hash(self):
        image = np.arange(16, dtype=np.uint16).reshape(4, 4)
        labels = np.zeros((4, 4), dtype=np.uint8)
        assert array_sha256(image, labels) == array_sha256(image.copy(), labels.copy())

    def test_changed_pixel_changes_hash(self):
        image = np.zeros((4, 4), dtype=np.uint16)
        labels = np.zeros((4, 4), dtype=np.uint8)
        before = array_sha256(image, labels)
        image[2, 2] = 1
        assert array_sha256(image, labels) != before

    def test_hash_is_a_sha256_digest(self):
        digest = array_sha256(np.zeros((2, 2), np.uint16), np.zeros((2, 2), np.uint8))
        assert len(digest) == 64
        int(digest, 16)


# =============================================================================
# Fixture on disk
# =============================================================================


@pytest.mark.skipif(not (FIXTURE / "images.npy").exists(), reason="fixture not built")
class TestPseudoMSSFixture:
    @pytest.fixture(scope="class")
    def fixture(self):
        images = np.load(FIXTURE / "images.npy", mmap_mode="r")
        labels = np.load(FIXTURE / "labels.npy", mmap_mode="r")
        manifest = list(csv.DictReader((FIXTURE / "manifest.csv").open()))
        distribution = json.loads((FIXTURE / "label_distribution.json").read_text())
        metadata = json.loads((FIXTURE / "fixture.json").read_text())
        return images, labels, manifest, distribution, metadata

    def test_size_between_1k_and_2k(self, fixture):
        images, _, _, _, _ = fixture
        assert 1000 <= images.shape[0] <= 2000

    def test_shapes(self, fixture):
        images, labels, manifest, _, _ = fixture
        n = images.shape[0]
        assert images.shape[1:] == (4, 64, 64)
        assert labels.shape == (n, 64, 64)
        assert len(manifest) == n

    def test_chip_ids_are_unique(self, fixture):
        _, _, manifest, _, _ = fixture
        assert len({row["chip_id"] for row in manifest}) == len(manifest)

    def test_labels_only_have_cloudsen12_classes(self, fixture):
        _, labels, _, _, _ = fixture
        present = set(np.unique(np.asarray(labels)).tolist())
        assert present <= {0, 1, 2, 3}

    def test_images_are_reflectance_quantized(self, fixture):
        images, _, _, _, _ = fixture
        sample = np.asarray(images[:20])
        assert images.dtype == np.uint16
        assert sample.min() >= 0
        assert sample.max() <= 10000

    def test_manifest_hashes_match_the_arrays(self, fixture):
        images, labels, manifest, _, _ = fixture
        rng = np.random.default_rng(0)
        for index in rng.choice(len(manifest), size=min(25, len(manifest)), replace=False):
            row = manifest[int(index)]
            assert row["sha256"] == array_sha256(images[int(index)], labels[int(index)])

    def test_label_distribution_matches_the_arrays(self, fixture):
        _, labels, _, distribution, _ = fixture
        counts = distribution["pixels_per_class"]
        flat = np.asarray(labels).ravel()
        assert counts["clear"] == int(np.count_nonzero(flat == 0))
        assert counts["thick_cloud"] == int(np.count_nonzero(flat == 1))
        assert counts["thin_cloud"] == int(np.count_nonzero(flat == 2))
        assert counts["cloud_shadow"] == int(np.count_nonzero(flat == 3))
        assert sum(counts.values()) == flat.size

    def test_fixture_hash_matches_the_npy_file(self, fixture):
        _, _, _, _, metadata = fixture
        digest = hashlib.sha256((FIXTURE / "images.npy").read_bytes()).hexdigest()
        assert digest == metadata["images_sha256"]

    def test_regime_column_is_consistent(self, fixture):
        _, _, manifest, _, _ = fixture
        allowed = {"clear", "low_cloud", "mid_cloud", "high_cloud"}
        assert {row["regime"] for row in manifest} <= allowed


# =============================================================================
# Domain metrics
# =============================================================================


class TestWasserstein:
    def test_identical_distributions_are_zero(self):
        band = np.linspace(0.0, 1.0, 1000).reshape(1, -1)
        assert wasserstein_per_band(band, band.copy())[0] == pytest.approx(0.0)

    def test_shift_is_recovered(self):
        band = np.linspace(0.0, 1.0, 1000).reshape(1, -1)
        shifted = band + 0.1
        assert wasserstein_per_band(shifted, band)[0] == pytest.approx(0.1, abs=1e-9)

    def test_scale_is_linear(self):
        rng = np.random.default_rng(0)
        band = rng.random((1, 500))
        other = rng.random((1, 500))
        base = wasserstein_per_band(band, other)[0]
        assert wasserstein_per_band(2 * band, 2 * other)[0] == pytest.approx(2 * base)

    def test_masks_are_respected(self):
        real = np.array([[[0.1, 0.2, 99.0]]])
        pseudo = np.array([[[0.1, 0.2, 0.0]]])
        mask = np.array([[True, True, False]])
        assert wasserstein_per_band(pseudo, real, mask, mask)[0] == pytest.approx(0.0)

    def test_band_mismatch_raises(self):
        with pytest.raises(ValueError, match="bands"):
            wasserstein_per_band(np.zeros((2, 4, 4)), np.zeros((3, 4, 4)))


class TestKS:
    def test_identical_is_zero(self):
        band = np.linspace(0.0, 1.0, 500).reshape(1, -1)
        assert ks_per_band(band, band.copy())[0] == pytest.approx(0.0)

    def test_disjoint_supports_is_one(self):
        first = np.zeros((1, 200))
        second = np.ones((1, 200))
        assert ks_per_band(first, second)[0] == pytest.approx(1.0)


class TestStriping:
    def _sinusoid(self, wavelength=8.0, angle_deg=9.0, shape=(128, 128), amplitude=0.05):
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        angle = np.deg2rad(angle_deg)
        along = yy * np.cos(angle) + xx * np.sin(angle)
        return amplitude * np.sin(2 * np.pi * along / wavelength)

    def test_periodic_pattern_peaks_at_its_wavelength(self):
        band = self._sinusoid(wavelength=8.0)
        metrics = striping_metrics(band)
        assert metrics["wavelength_px"] == pytest.approx(8.0, abs=1.0)
        assert metrics["amplitude"] == pytest.approx(0.045, abs=0.01)
        assert metrics["stripe_index"] > 10

    def test_tilted_pattern_is_found(self):
        # The peak angle is the direction of the wave vector in the array
        # frame (y points down): |angle| = 90 + tilt.
        flat = striping_metrics(self._sinusoid(angle_deg=0.0))
        tilted = striping_metrics(self._sinusoid(angle_deg=20.0))
        assert abs(abs(flat["angle_deg"]) - 90.0) < 5.0
        assert abs(abs(tilted["angle_deg"]) - 110.0) < 10.0

    def test_white_noise_has_a_weaker_peak(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.05, (128, 128))
        striped = self._sinusoid(shape=(128, 128))
        noise_metrics = striping_metrics(noise)
        striped_metrics = striping_metrics(striped)
        assert striped_metrics["amplitude"] > 10 * noise_metrics["amplitude"]
        assert striped_metrics["stripe_index"] > noise_metrics["stripe_index"]

    def test_smooth_field_has_no_striping_peak(self):
        rng = np.random.default_rng(0)
        smooth = ndimage.gaussian_filter(rng.normal(0, 0.01, (250, 250)), 4)
        metrics = striping_metrics(smooth)
        assert metrics["stripe_index"] < 1.0

    def test_high_pass_removes_a_constant(self):
        band = np.full((32, 32), 0.4)
        assert np.allclose(high_pass(band), 0.0)

    def test_radial_psd_returns_increasing_wavelengths(self):
        wavelength, power = radial_psd(self._sinusoid())
        assert np.all(np.diff(wavelength) > 0)
        assert power.shape == wavelength.shape

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2D"):
            striping_metrics(np.zeros((2, 8, 8)))

    def test_striping_wavelength_constant(self):
        assert pytest.approx(7.9) == STRIPING_WAVELENGTH_PX


# =============================================================================
# Agreement
# =============================================================================


class TestAgreement:
    def test_perfect_agreement(self):
        labels = ["clear", "thick", "thin", "shadow"] * 5
        assert cohen_kappa(labels, labels) == pytest.approx(1.0)

    def test_known_confusion_matrix(self):
        # 2x2: po = 0.7, pe = 0.5 -> kappa = 0.4
        first = ["a"] * 40 + ["b"] * 10 + ["a"] * 20 + ["b"] * 30
        second = ["a"] * 40 + ["a"] * 10 + ["b"] * 20 + ["b"] * 30
        assert cohen_kappa(first, second) == pytest.approx(0.4)

    def test_independent_labels_have_near_zero_kappa(self):
        rng = np.random.default_rng(0)
        first = rng.choice(["a", "b"], size=5000)
        second = rng.choice(["a", "b"], size=5000)
        assert abs(cohen_kappa(first, second)) < 0.06

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            cohen_kappa(["a"], ["a", "b"])

    def test_single_class_is_undefined(self):
        assert np.isnan(cohen_kappa(["a"] * 10, ["a"] * 10))

    def test_pixel_kappa_ignores_unlabelled(self):
        first = np.array([[0, 1, 254, 255]], dtype=np.uint8)
        second = np.array([[0, 1, 2, 3]], dtype=np.uint8)
        result = pixel_kappa(first, second)
        assert result["n_pixels"] == 2
        assert result["kappa"] == pytest.approx(1.0)

    def test_pixel_iou_per_class(self):
        first = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        second = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        result = pixel_kappa(first, second)
        assert result["iou"]["thick"] == pytest.approx(0.5)
        assert result["iou"]["clear"] == pytest.approx(2 / 3)

    def test_pixel_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shapes"):
            pixel_kappa(np.zeros((2, 2), np.uint8), np.zeros((3, 3), np.uint8))

    def test_csv_comparison(self):
        first = {"a": {"dominant": "clear", "fractions": dict.fromkeys(
            ("clear", "thick", "thin", "shadow"), 25.0)},
                 "b": {"dominant": "thick", "fractions": dict.fromkeys(
                     ("clear", "thick", "thin", "shadow"), 25.0)}}
        second = {chip: {"dominant": row["dominant"], "fractions": dict(row["fractions"])}
                  for chip, row in first.items()}
        result = compare_csv(first, second)
        assert result["n_chips"] == 2
        assert result["kappa"] == pytest.approx(1.0)
        assert result["fraction_mae"] == pytest.approx(0.0)

    def test_fraction_sum_validation(self):
        rows = {"a": {"dominant": "clear",
                      "fractions": {"clear": 50.0, "thick": 20.0, "thin": 10.0, "shadow": 10.0}}}
        assert validate_fractions(rows)

    def test_empty_csv_templates_are_read_as_empty(self, tmp_path):
        path = tmp_path / "a.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["chip_id", "clase_dominante", "pct_clear"]
            )
            writer.writeheader()
            writer.writerow({"chip_id": "x"})
        assert read_annotation_csv(path) == {}


# =============================================================================
# Scripts and artefacts
# =============================================================================


class TestFixtureBuilderHelpers:
    @pytest.fixture(scope="class")
    def builder(self):
        import build_pseudo_mss_fixture as module

        return module

    def test_stable_seed_is_deterministic(self, builder):
        assert builder.stable_seed(1, "abc") == builder.stable_seed(1, "abc")
        assert builder.stable_seed(1, "abc") != builder.stable_seed(1, "abd")

    def test_scene_strata(self, builder):
        assert builder.scene_stratum(0.0) == "clear"
        assert builder.scene_stratum(0.05) == "low_cloud"
        assert builder.scene_stratum(0.30) == "mid_cloud"
        assert builder.scene_stratum(0.70) == "high_cloud"

    def test_profile_uses_scene_sensor_and_sun(self, builder):
        cfg = builder.profile_for_scene(
            {"s2_id": "S2B_MSIL1C_x", "s2_mean_solar_zenith_angle": 50.0}
        )
        assert cfg.spectral.sensor == "S2B"
        assert cfg.radiometric.sun_elevation_deg.min == pytest.approx(40.0)

    def test_profile_falls_back_to_s2a(self, builder):
        cfg = builder.profile_for_scene({})
        assert cfg.spectral.sensor == "S2A"


@pytest.mark.skipif(not (DOMAIN / "domain_wasserstein_summary.csv").exists(),
                    reason="domain comparison not run")
class TestDomainArtefacts:
    @pytest.fixture(scope="class")
    def summary(self):
        return list(csv.DictReader((DOMAIN / "domain_wasserstein_summary.csv").open()))

    def test_all_bands_and_modes_are_present(self, summary):
        assert {row["band"] for row in summary} == {"B1", "B2", "B3", "B4"}
        assert {"legacy", "sbaf"} <= {row["mode"] for row in summary}

    def test_distances_are_positive(self, summary):
        assert all(float(row["w1"]) >= 0 for row in summary)

    def test_w1_is_in_physical_units(self, summary):
        assert all(float(row["w1"]) < 0.2 for row in summary)


@pytest.mark.skipif(not (LABELS / "chips_index.csv").exists(),
                    reason="human label package not prepared")
class TestHumanLabelPackage:
    @pytest.fixture(scope="class")
    def index(self):
        return list(csv.DictReader((LABELS / "chips_index.csv").open()))

    def test_between_30_and_50_chips(self, index):
        assert 30 <= len(index) <= 50

    def test_every_chip_has_png_and_tif(self, index):
        for row in index:
            assert (LABELS / row["path_png"]).exists()
            assert (LABELS / row["path_tif"]).exists()

    def test_chip_ids_are_unique(self, index):
        assert len({row["chip_id"] for row in index}) == len(index)

    def test_annotation_templates_are_empty(self, index):
        for annotator in ("anotador1", "anotador2"):
            rows = list(csv.DictReader((LABELS / f"anotaciones_{annotator}.csv").open()))
            assert [row["chip_id"] for row in rows] == [row["chip_id"] for row in index]
            assert all(not row["clase_dominante"] for row in rows)

    def test_qa_hint_is_documented_as_a_hint(self, index):
        assert index[0]["notes"] == "QA hint only, not a label"


def test_top_level_exports():
    assert callable(satharmony.wasserstein_per_band)
    assert callable(satharmony.striping_metrics)
