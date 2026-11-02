"""Tests for G4b: loss levers, decision rule, TTA, encoders and the ablation grid.

The G4b work keeps the G4 data contract intact (same fixture, same scene split,
same labels) and only changes the model, the loss and the decision rule.  These
tests pin those contracts:

* the losses reduce to the G4 ones when their extra knobs are off;
* the per-class bias recovers a known bias and never makes the mean IoU worse
  than ``argmax`` on the split it was tuned on;
* the TTA helpers keep shapes and normalisation;
* the fixed split of the manifest is still the one used by the training script;
* a trained G4b checkpoint declares its loss, augmentation and class bias.

Torch is optional for the package, so the module is skipped without it.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from satharmony.models import (  # noqa: E402
    CLASS_NAMES,
    ModelConfig,
    SegLoss,
    TimmUNet,
    apply_class_bias,
    build_model,
    class_weights,
    confusion_matrix,
    count_parameters,
    focal_cross_entropy,
    load_model,
    mean_iou,
    predict_proba,
    predict_proba_dihedral,
    predict_proba_multiscale,
    save_checkpoint,
    soft_dice_loss,
    tune_class_bias,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path("/data/databases/CLOUD_MSS/calibration/pseudo_mss_fixture")
G4B_CHECKPOINT = Path("/data/databases/CLOUD_MSS/calibration/g4b/g4b_best.pt")
TINY = ModelConfig(base_channels=8, depth=2, norm_groups=4)
BAND_MEAN = np.full(4, 0.2, dtype=np.float32)
BAND_STD = np.full(4, 0.1, dtype=np.float32)

for extra in (REPO, REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


# =============================================================================
# Class weights and focal loss
# =============================================================================
class TestClassWeights:
    def test_sqrt_is_the_g4_baseline(self):
        counts = [1000, 200, 60, 80]
        weights = class_weights(counts, mode="sqrt")
        expected = 1.0 / np.sqrt(np.array(counts) / sum(counts))
        expected = expected / expected.mean()
        assert np.allclose(weights.numpy(), expected, rtol=1e-5)

    def test_inverse_is_stronger_on_the_minority_class(self):
        counts = [1000, 200, 60, 80]
        sqrt = class_weights(counts, mode="sqrt").numpy()
        inverse = class_weights(counts, mode="inverse").numpy()
        assert inverse[2] > sqrt[2]
        assert np.isclose(inverse.mean(), 1.0, rtol=1e-5)

    def test_none_is_all_ones(self):
        assert class_weights([5, 1, 2, 3], mode="none").tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_rejects_unknown_mode_and_negative_counts(self):
        with pytest.raises(ValueError, match="unknown"):
            class_weights([1, 2, 3, 4], mode="quadratic")
        with pytest.raises(ValueError, match="non-negative"):
            class_weights([-1, 2, 3, 4])


class TestFocalLoss:
    def test_gamma_zero_is_weighted_cross_entropy(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 4, 5, 5)
        target = torch.randint(0, 4, (2, 5, 5))
        weights = torch.tensor([0.5, 1.0, 2.0, 1.5])
        plain = torch.nn.functional.cross_entropy(logits, target, weight=weights)
        focal = focal_cross_entropy(logits, target, gamma=0.0, weight=weights)
        assert torch.allclose(plain, focal, rtol=1e-5)

    def test_focal_downweights_confident_pixels(self):
        # One pixel predicted with p=0.997: the focal loss is ~1e-5 of the CE.
        logits = torch.tensor([[[[5.0, 0.0]], [[-1.0, 0.0]], [[-3.0, 0.0]], [[-4.0, 0.0]]]])
        target = torch.tensor([[[0, 3]]])
        plain = focal_cross_entropy(logits, target, gamma=0.0)
        focal = focal_cross_entropy(logits, target, gamma=2.0)
        assert focal < plain
        assert focal > 0.0

    def test_label_smoothing_stays_finite(self):
        logits = torch.zeros(1, 4, 2, 2)
        target = torch.zeros(1, 2, 2, dtype=torch.long)
        loss = focal_cross_entropy(logits, target, gamma=1.0, label_smoothing=0.1)
        assert torch.isfinite(loss).all()
        assert loss > 0

    def test_rejects_negative_gamma(self):
        logits = torch.zeros(1, 4, 2, 2)
        target = torch.zeros(1, 2, 2, dtype=torch.long)
        with pytest.raises(ValueError, match="gamma"):
            focal_cross_entropy(logits, target, gamma=-1.0)


class TestDiceAndCombinedLoss:
    def test_dice_is_near_zero_for_a_perfect_prediction(self):
        logits = torch.full((1, 4, 4, 4), -10.0)
        target = torch.ones(1, 4, 4, dtype=torch.long)
        logits[0, 1] = 10.0
        loss = soft_dice_loss(logits, target, num_classes=4)
        assert float(loss) < 0.02

    def test_dice_is_near_one_when_the_only_reference_class_is_missed(self):
        # Two classes: every pixel is class 1 and the model says class 0, so
        # both Dice terms collapse and the loss goes to 16/17.
        logits = torch.full((1, 2, 4, 4), -10.0)
        target = torch.ones(1, 4, 4, dtype=torch.long)
        logits[0, 0] = 10.0
        loss = soft_dice_loss(logits, target, num_classes=2)
        assert float(loss) > 0.9

    def test_combined_loss_returns_the_three_terms(self):
        torch.manual_seed(1)
        criterion = SegLoss(4, weight=torch.tensor([0.5, 1.0, 2.0, 1.5]), gamma=2.0,
                            dice_weight=0.5)
        logits = torch.randn(2, 4, 8, 8, requires_grad=True)
        target = torch.randint(0, 4, (2, 8, 8))
        total, cross_entropy, dice = criterion(logits, target)
        assert torch.allclose(total, cross_entropy + 0.5 * dice, rtol=1e-5)
        total.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()

    def test_rejects_negative_weights(self):
        with pytest.raises(ValueError):
            SegLoss(4, dice_weight=-1.0)


# =============================================================================
# Decision rule
# =============================================================================
def _imbalanced_probabilities(bias_thin: float = 0.8, seed: int = 0):
    """Probabilities of a model that systematically under-predicts thin cloud."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 1.5, (40, 4, 16, 16)).astype(np.float32)
    logits[:, 2] -= 1.5
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = shifted / shifted.sum(axis=1, keepdims=True)
    target = logits.argmax(axis=1)
    log_p = np.log(np.maximum(probabilities, 1e-8))
    log_p[:, 2] -= bias_thin
    shifted = np.exp(log_p - log_p.max(axis=1, keepdims=True))
    return (shifted / shifted.sum(axis=1, keepdims=True)).astype(np.float32), target


class TestDecisionRule:
    def test_zero_bias_is_plain_argmax(self):
        probabilities, _ = _imbalanced_probabilities()
        assert np.array_equal(
            apply_class_bias(probabilities, [0, 0, 0, 0]), probabilities.argmax(axis=1)
        )

    def test_positive_bias_increases_recall_of_that_class(self):
        probabilities, target = _imbalanced_probabilities()
        base = confusion_matrix(probabilities.argmax(axis=1), target)
        biased = confusion_matrix(apply_class_bias(probabilities, [0, 0, 1.0, 0]), target)
        base_recall = base[2, 2] / base[2].sum()
        biased_recall = biased[2, 2] / biased[2].sum()
        assert biased_recall > base_recall

    def test_tuning_recovers_the_simulated_bias(self):
        probabilities, target = _imbalanced_probabilities(bias_thin=0.8)
        result = tune_class_bias(probabilities, target)
        assert np.isclose(result["bias"][2], 0.8, atol=0.15)
        assert result["mean_iou"] > result["baseline_mean_iou"]

    def test_tuning_never_hurts_on_the_split_it_was_tuned_on(self):
        probabilities, target = _imbalanced_probabilities(bias_thin=0.3, seed=3)
        result = tune_class_bias(probabilities, target)
        assert result["mean_iou"] >= result["baseline_mean_iou"] - 1e-9
        assert np.isclose(
            mean_iou(np.asarray(result["confusion_matrix"])), result["mean_iou"]
        )

    def test_is_deterministic(self):
        probabilities, target = _imbalanced_probabilities(seed=5)
        assert tune_class_bias(probabilities, target)["bias"] == tune_class_bias(
            probabilities, target
        )["bias"]

    def test_rejects_bad_shapes(self):
        probabilities, target = _imbalanced_probabilities()
        with pytest.raises(ValueError):
            apply_class_bias(probabilities, [0, 0, 0])
        with pytest.raises(ValueError):
            tune_class_bias(probabilities[:2], target)
        with pytest.raises(ValueError):
            tune_class_bias(probabilities, target, step=0.0)


# =============================================================================
# Test-time augmentation
# =============================================================================
class TestTTA:
    def test_dihedral_is_exact_for_a_per_pixel_model(self):
        # A 1x1 convolution is invariant to the eight dihedral transforms, so
        # averaging them must return exactly the single forward pass.  This
        # pins the bookkeeping of the inverse transforms.
        class PixelWise(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = ModelConfig(base_channels=8, depth=1, norm_groups=4)
                self.conv = torch.nn.Conv2d(4, 4, kernel_size=1, bias=True)

            def forward(self, x):
                return self.conv(x)

        torch.manual_seed(0)
        model = PixelWise().eval()
        image = np.random.default_rng(0).random((4, 8, 8)).astype(np.float32)
        single = predict_proba(model, image, BAND_MEAN, BAND_STD)
        averaged = predict_proba_dihedral(model, image, BAND_MEAN, BAND_STD)
        assert averaged.shape == single.shape
        assert np.allclose(averaged, single, atol=1e-6)

    def test_multiscale_keeps_shape_and_normalisation(self):
        model = build_model(TINY).eval()
        images = np.random.default_rng(0).random((2, 4, 32, 32)).astype(np.float32)
        probabilities = predict_proba_multiscale(
            model, images, BAND_MEAN, BAND_STD, scales=(1.0, 1.5), dihedral=False
        )
        assert probabilities.shape == (2, 4, 32, 32)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)

    def test_multiscale_restores_a_single_image(self):
        model = build_model(TINY).eval()
        image = np.random.default_rng(1).random((4, 25, 37)).astype(np.float32)
        probabilities = predict_proba_multiscale(
            model, image, BAND_MEAN, BAND_STD, scales=(1.0, 1.25), dihedral=False
        )
        assert probabilities.shape == (4, 25, 37)
        assert np.allclose(probabilities.sum(axis=0), 1.0, atol=1e-5)

    def test_rejects_bad_scales(self):
        model = build_model(TINY).eval()
        image = np.zeros((4, 16, 16), dtype=np.float32)
        with pytest.raises(ValueError):
            predict_proba_multiscale(model, image, BAND_MEAN, BAND_STD, scales=())
        with pytest.raises(ValueError):
            predict_proba_multiscale(model, image, BAND_MEAN, BAND_STD, scales=(0.0,))


# =============================================================================
# Pretrained encoder wrapper
# =============================================================================
class TestTimmEncoder:
    def test_forward_matches_the_input_resolution(self):
        pytest.importorskip("timm")
        model = TimmUNet(ModelConfig(encoder="resnet18", pretrained=False))
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(1, 4, 32, 32))
        assert logits.shape == (1, 4, 32, 32)
        assert count_parameters(model) > 1_000_000

    def test_rejects_size_not_multiple_of_32(self):
        pytest.importorskip("timm")
        model = TimmUNet(ModelConfig(encoder="resnet18", pretrained=False))
        with pytest.raises(ValueError, match="multiple of 32"):
            model(torch.zeros(1, 4, 30, 30))

    def test_checkpoint_roundtrip(self, tmp_path):
        pytest.importorskip("timm")
        model = TimmUNet(ModelConfig(encoder="resnet18", pretrained=False)).eval()
        info = save_checkpoint(
            tmp_path / "timm.pt", model, model.config, BAND_MEAN, BAND_STD, {"run": "x"}
        )
        reloaded, checkpoint = load_model(info["path"])
        assert isinstance(reloaded, TimmUNet)
        assert checkpoint.config.encoder == "resnet18"
        assert checkpoint.config.size_multiple == 32


# =============================================================================
# Training contracts (same fixture and split as G4)
# =============================================================================
@pytest.mark.skipif(not (FIXTURE / "manifest.csv").exists(), reason="fixture not available")
class TestTrainingContracts:
    def test_split_is_the_manifest_one(self):
        from train_cloud_baseline import split_indices

        with (FIXTURE / "manifest.csv").open() as handle:
            manifest = list(csv.DictReader(handle))
        splits = split_indices(manifest)
        assert {name: len(indices) for name, indices in splits.items()} == {
            "train": 1693,
            "validation": 113,
            "test": 194,
        }
        assert not set(splits["train"]) & set(splits["test"])
        assert not set(splits["validation"]) & set(splits["test"])

    def test_strong_augment_is_deterministic_and_keeps_labels_aligned(self):
        from train_cloud_baseline import augment_batch

        rng = np.random.default_rng(0)
        images = rng.random((4, 4, 32, 32)).astype(np.float32) * 0.5 + 0.1
        labels = rng.integers(0, 4, (4, 32, 32)).astype(np.uint8)
        first = augment_batch(images, labels, np.random.default_rng(7), "strong")
        second = augment_batch(images, labels, np.random.default_rng(7), "strong")
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])
        assert first[0].min() >= 0.0 and first[0].max() <= 1.0
        for index in range(4):
            assert np.bincount(first[1][index].ravel(), minlength=4).tolist() == \
                np.bincount(labels[index].ravel(), minlength=4).tolist()

    def test_flips_keep_the_image_label_correspondence(self):
        from train_cloud_baseline import augment_batch

        labels = np.tile(np.arange(4, dtype=np.uint8), (16, 4))
        images = np.zeros((2, 4, 16, 16), dtype=np.float32)
        for band in range(4):
            images[:, band] = labels.astype(np.float32) * 0.1
        out_images, out_labels = augment_batch(
            images, labels[None].repeat(2, axis=0), np.random.default_rng(1), "flips"
        )
        for index in range(2):
            assert np.allclose(out_images[index, 0], out_labels[index] * 0.1)
            assert np.allclose(out_images[index, 2], out_labels[index] * 0.1)

    def test_unknown_augment_mode_raises(self):
        from train_cloud_baseline import augment_batch

        with pytest.raises(ValueError, match="unknown augment"):
            augment_batch(np.zeros((1, 4, 8, 8), np.float32),
                          np.zeros((1, 8, 8), np.uint8),
                          np.random.default_rng(0), "warp")

    def test_thin_oversampling_only_upweights_thin_chips(self):
        from train_cloud_baseline import sampling_probabilities

        labels = np.zeros((4, 8, 8), dtype=np.uint8)
        labels[1, 0, 0] = 2  # only the second chip has a thin-cloud pixel
        indices = np.arange(4)
        assert sampling_probabilities(labels, indices, 1.0) is None
        probabilities = sampling_probabilities(labels, indices, 4.0)
        assert np.isclose(probabilities.sum(), 1.0)
        assert np.isclose(probabilities[1] / probabilities[0], 4.0)

    def test_band_statistics_are_in_reflectance_units(self):
        from train_cloud_baseline import band_statistics

        images = np.full((4, 4, 8, 8), 5000, dtype=np.uint16)
        mean, std = band_statistics(images, np.arange(4))
        assert np.allclose(mean, 0.5)
        assert np.allclose(std, 1.0)


# =============================================================================
# Evaluation helpers
# =============================================================================
class TestEvaluationHelpers:
    def test_class_fractions_sum_to_one(self):
        from evaluate_cloud_baseline import class_fractions

        mask = np.array([[0, 0, 1, 2], [3, 0, 0, 0]])
        valid = np.ones_like(mask, dtype=bool)
        fractions = class_fractions(mask, valid)
        total = sum(fractions[name] for name in CLASS_NAMES)
        assert np.isclose(total, 1.0)
        assert np.isclose(fractions["contaminated"],
                          fractions["thick_cloud"] + fractions["thin_cloud"]
                          + fractions["cloud_shadow"])

    def test_binary_iou_handles_empty_union(self):
        from evaluate_cloud_baseline import binary_iou

        mask = np.ones((3, 3), dtype=bool)
        assert np.isnan(binary_iou(np.zeros((3, 3)), np.zeros((3, 3)), mask))
        assert binary_iou(mask, mask, mask) == 1.0

    def test_choose_scales_uses_the_validation_split(self):
        from evaluate_cloud_baseline import choose_scales

        from satharmony.models import load_checkpoint

        model = build_model(TINY).eval()
        checkpoint = load_checkpoint(save_checkpoint(
            Path("/tmp") / "g4b_scale_test.pt", model, TINY, BAND_MEAN, BAND_STD
        )["path"])
        images = np.random.default_rng(2).random((2, 4, 16, 16)).astype(np.float32)
        labels = np.zeros((2, 16, 16), dtype=np.uint8)
        report = choose_scales(model, checkpoint, images, labels, "cpu", 2,
                               candidates=((1.0,), (1.0, 0.75)))
        assert "scales" in report and len(report["scales"]) >= 1
        assert report["val_mean_iou"] == max(
            candidate["mean_iou"] for candidate in report["candidates"].values()
        )

    def test_per_class_tables_have_one_block_per_model(self, tmp_path):
        from evaluate_cloud_baseline import write_per_class_csv, write_per_class_markdown

        predicted = np.array([0, 1, 2, 3, 0])
        result = {
            "confusion_matrix": confusion_matrix(predicted, predicted).tolist(),
            "per_class": {
                name: {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0,
                       "support": 1.0}
                for name in CLASS_NAMES
            },
            "mean_iou": 1.0,
            "overall_accuracy": 1.0,
            "cohen_kappa": 1.0,
        }
        csv_path, md_path = tmp_path / "t.csv", tmp_path / "t.md"
        write_per_class_csv(csv_path, {"g4b_bias": result, "ocm": result})
        write_per_class_markdown(md_path, {"g4b_bias": result, "ocm": result})
        rows = list(csv.DictReader(csv_path.open()))
        assert len(rows) == 2 * len(CLASS_NAMES)
        assert {row["model"] for row in rows} == {"g4b_bias", "ocm"}
        assert "mIoU / OA / kappa" in md_path.read_text()

    def test_pics_table_keeps_every_model_column(self, tmp_path):
        from evaluate_cloud_baseline import write_pics_csv

        fractions = {"clear": 0.9, "thick_cloud": 0.05, "thin_cloud": 0.03,
                     "cloud_shadow": 0.02, "contaminated": 0.1}
        row = {
            "site": "x", "label": "X", "mss_date": "1990-01-01",
            "sun_elevation_deg": 40.0, "n_valid_px": 100,
            "models": {
                name: {"fractions": fractions,
                       "vs_ocm": {"pixel_agreement": 0.9, "binary_iou": 0.5,
                                  "cohen_kappa": 0.4},
                       "qa_pixel_iou": 0.1}
                for name in ("g4", "g4b")
            },
            "ocm": {"fractions": fractions, "qa_pixel_iou": 0.05},
            "qa_pixel_reference": {"cloud_fraction": 0.02},
        }
        path = tmp_path / "pics.csv"
        write_pics_csv(path, [row], ["g4", "g4b"])
        header = next(csv.reader(path.open()))
        assert "g4_contaminated" in header and "g4b_contaminated" in header
        assert "ocm_contaminated" in header


# =============================================================================
# Trained G4b checkpoint (skipped when the data is not mounted)
# =============================================================================
@pytest.mark.skipif(not G4B_CHECKPOINT.exists(), reason="G4b checkpoint not available")
class TestTrainedG4bCheckpoint:
    def test_metadata_declares_the_levers(self):
        from satharmony.models import file_sha256

        model, checkpoint = load_model(G4B_CHECKPOINT)
        metadata = checkpoint.metadata
        assert Path(f"{G4B_CHECKPOINT}.sha256").read_text().split()[0] == \
            file_sha256(G4B_CHECKPOINT)
        assert metadata["loss"] in {"ce", "focal"}
        assert metadata["class_weight_mode"] in {"sqrt", "inverse", "none"}
        assert metadata["augment"] in {"flips", "gain", "full", "strong"}
        assert len(metadata["class_bias"]) == len(CLASS_NAMES)
        assert 0.0 < metadata["val_mean_iou"] <= 1.0
        assert model.training is False

    def test_predicts_on_a_fixture_chip(self):
        model, checkpoint = load_model(G4B_CHECKPOINT)
        images = np.load(FIXTURE / "images.npy", mmap_mode="r")
        chip = np.asarray(images[0], dtype=np.float32) / 10000.0
        probabilities = predict_proba(model, chip, checkpoint.band_mean, checkpoint.band_std)
        assert probabilities.shape == (4, 64, 64)
        assert np.allclose(probabilities.sum(axis=0), 1.0, atol=1e-5)
        mask = apply_class_bias(probabilities, checkpoint.metadata["class_bias"])
        assert mask.shape == (64, 64)
        assert set(np.unique(mask)) <= set(range(4))
        matrix = confusion_matrix(mask, mask)
        assert matrix.sum() == 64 * 64
        assert mean_iou(matrix) == 1.0
