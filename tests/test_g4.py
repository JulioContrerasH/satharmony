"""Integration tests for the G4 cloud baseline: model, checkpoint and metrics.

Torch is optional for the package, so the whole module is skipped when it is
not installed.  The tests use a two-level U-Net (a few thousand parameters) so
they run in under a second on CPU; the trained checkpoint and the fixture are
exercised only when the files are present.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from satharmony.models import (  # noqa: E402
    CHECKPOINT_FORMAT,
    CLASS_NAMES,
    ModelConfig,
    build_model,
    cohen_kappa,
    confusion_matrix,
    count_parameters,
    evaluate,
    file_sha256,
    iou_per_class,
    load_checkpoint,
    load_model,
    mean_iou,
    normalize_images,
    overall_accuracy,
    per_class_scores,
    predict_masks,
    predict_proba,
    save_checkpoint,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path("/data/databases/CLOUD_MSS/calibration/pseudo_mss_fixture")
CHECKPOINT = Path("/data/databases/CLOUD_MSS/calibration/g4/cloud_unet_mss_pseudo.pt")
TINY = ModelConfig(base_channels=8, depth=2, norm_groups=4)
BAND_MEAN = np.full(4, 0.2, dtype=np.float32)
BAND_STD = np.full(4, 0.1, dtype=np.float32)


# =============================================================================
# Architecture and forward
# =============================================================================
class TestModel:
    def test_parameters_are_positive_and_scale_with_width(self):
        narrow = count_parameters(build_model(ModelConfig(base_channels=8, depth=2)))
        wide = count_parameters(build_model(ModelConfig(base_channels=16, depth=2)))
        assert narrow > 0
        assert wide > narrow

    def test_forward_returns_logits_at_input_resolution(self):
        model = build_model(TINY)
        with torch.no_grad():
            logits = model(torch.zeros(2, 4, 32, 32))
        assert logits.shape == (2, 4, 32, 32)
        assert torch.isfinite(logits).all()

    def test_forward_rejects_size_not_multiple_of_depth(self):
        model = build_model(TINY)
        with pytest.raises(ValueError, match="multiple of 4"):
            model(torch.zeros(1, 4, 30, 30))

    def test_config_rejects_bad_groups_and_dropout(self):
        with pytest.raises(ValueError):
            ModelConfig(base_channels=10, norm_groups=8)
        with pytest.raises(ValueError):
            ModelConfig(dropout=1.0)

    def test_dropout_changes_training_output(self):
        torch.manual_seed(0)
        model = build_model(ModelConfig(base_channels=8, depth=2, dropout=0.5))
        model.train()
        x = torch.ones(1, 4, 16, 16)
        assert not torch.allclose(model(x), model(x))

    def test_eval_mode_is_deterministic(self):
        torch.manual_seed(0)
        model = build_model(TINY).eval()
        x = torch.randn(1, 4, 32, 32)
        with torch.no_grad():
            first, second = model(x), model(x)
        assert torch.allclose(first, second)


# =============================================================================
# Checkpoints
# =============================================================================
class TestCheckpoint:
    def test_roundtrip_preserves_everything(self, tmp_path):
        torch.manual_seed(1)
        model = build_model(TINY).eval()
        metadata = {"run_id": "test", "best_epoch": 7, "val_iou": {"clear": 0.9}}
        info = save_checkpoint(
            tmp_path / "model.pt", model, TINY, BAND_MEAN, BAND_STD, metadata
        )

        assert Path(info["path"]).exists()
        assert len(info["sha256"]) == 64
        assert info["bytes"] == Path(info["path"]).stat().st_size
        sidecar = Path(f"{info['path']}.sha256").read_text().split()[0]
        assert sidecar == info["sha256"] == file_sha256(info["path"])

        loaded = load_checkpoint(info["path"])
        assert loaded.config == TINY
        assert loaded.metadata["best_epoch"] == 7
        assert np.allclose(loaded.band_mean, BAND_MEAN)
        assert np.allclose(loaded.band_std, BAND_STD)
        assert loaded.sha256 == info["sha256"]

        reloaded, _ = load_model(info["path"])
        x = torch.from_numpy(np.random.default_rng(2).random((1, 4, 32, 32)).astype(np.float32))
        with torch.no_grad():
            assert torch.allclose(model(x), reloaded(x))

    def test_checkpoint_is_weights_only_safe(self, tmp_path):
        model = build_model(TINY)
        path = tmp_path / "model.pt"
        save_checkpoint(path, model, TINY, BAND_MEAN, BAND_STD, {"nested": [1, 2.5, "x"]})
        payload = torch.load(path, weights_only=True)
        assert payload["format"] == CHECKPOINT_FORMAT
        assert isinstance(payload["band_mean"], list)

    def test_rejects_foreign_payload(self, tmp_path):
        path = tmp_path / "other.pt"
        torch.save({"format": "something-else", "state_dict": {}}, path)
        with pytest.raises(ValueError, match="not a"):
            load_checkpoint(path)

    def test_metadata_is_json_serialisable(self, tmp_path):
        model = build_model(TINY)
        metadata = {"run_id": "x", "iou": {"clear": 0.5}, "n": 3}
        info = save_checkpoint(tmp_path / "m.pt", model, TINY, BAND_MEAN, BAND_STD, metadata)
        json.dumps(load_checkpoint(info["path"]).metadata)


# =============================================================================
# Inference
# =============================================================================
class TestInference:
    def test_normalize_images_uses_per_band_statistics(self):
        images = np.stack([np.full((2, 2), 0.2), np.full((2, 2), 0.4),
                           np.full((2, 2), 0.6), np.full((2, 2), 0.8)]).astype(np.float32)
        mean = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        std = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        out = normalize_images(images, mean, std)
        assert np.allclose(out[:, 0, 0], [0.0, 1.0, 4.0 / 3.0, 1.5])

    def test_predict_proba_sums_to_one_and_keeps_shape(self):
        model = build_model(TINY).eval()
        images = np.random.default_rng(3).random((2, 4, 30, 30)).astype(np.float32)
        probabilities = predict_proba(model, images, BAND_MEAN, BAND_STD, batch_size=1)
        assert probabilities.shape == (2, 4, 30, 30)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_masks_handles_single_image_and_odd_size(self):
        model = build_model(TINY).eval()
        image = np.random.default_rng(4).random((4, 25, 37)).astype(np.float32)
        mask = predict_masks(model, image, BAND_MEAN, BAND_STD)
        assert mask.shape == (25, 37)
        assert mask.min() >= 0 and mask.max() < 4

    def test_predict_restores_training_mode(self):
        model = build_model(TINY).train()
        image = np.random.default_rng(5).random((4, 16, 16)).astype(np.float32)
        predict_masks(model, image, BAND_MEAN, BAND_STD)
        assert model.training


# =============================================================================
# Metrics
# =============================================================================
class TestMetrics:
    def test_perfect_prediction(self):
        reference = np.array([[0, 0, 1], [1, 2, 3]])
        matrix = confusion_matrix(reference, reference)
        assert overall_accuracy(matrix) == 1.0
        assert mean_iou(matrix) == 1.0
        assert cohen_kappa(matrix) == 1.0
        assert all(np.isclose(value, 1.0) for value in iou_per_class(matrix).values())

    def test_known_confusion_values(self):
        # 4 clear pixels: 3 right, 1 predicted thick; 2 thick: 1 right, 1 clear.
        predicted = np.array([0, 0, 0, 1, 1, 0])
        reference = np.array([0, 0, 0, 0, 1, 1])
        matrix = confusion_matrix(predicted, reference)
        assert matrix.tolist() == [[3, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        iou = iou_per_class(matrix)
        assert np.isclose(iou["clear"], 3 / 5)
        assert np.isclose(iou["thick_cloud"], 1 / 3)
        assert np.isnan(iou["thin_cloud"])
        # The mean skips the two absent classes.
        assert np.isclose(mean_iou(matrix), (3 / 5 + 1 / 3) / 2)
        # Including them as zero halves the mean.
        assert np.isclose(mean_iou(matrix, ignore_absent=False), (3 / 5 + 1 / 3) / 4)

    def test_per_class_scores_are_consistent(self):
        predicted = np.array([0, 0, 1, 1, 2, 3])
        reference = np.array([0, 1, 1, 1, 2, 3])
        scores = per_class_scores(confusion_matrix(predicted, reference))
        assert list(scores) == list(CLASS_NAMES)
        assert scores["clear"]["support"] == 1
        assert np.isclose(scores["thick_cloud"]["recall"], 2 / 3)
        # Every thick-cloud prediction is thick; one thick pixel was missed.
        assert np.isclose(scores["thick_cloud"]["precision"], 1.0)
        assert np.isclose(scores["thin_cloud"]["f1"], 1.0)

    def test_ignore_mask_drops_pixels(self):
        predicted = np.array([0, 0, 0, 3])
        reference = np.array([0, 1, 1, 1])
        ignore = np.array([False, False, False, True])
        matrix = confusion_matrix(predicted, reference, ignore_mask=ignore)
        assert matrix.sum() == 3

    def test_evaluate_bundle_is_json_friendly(self):
        predicted = np.array([0, 1, 2, 3])
        result = evaluate(predicted, predicted)
        json.dumps(result)
        assert set(result) == {
            "confusion_matrix", "per_class", "mean_iou", "mean_iou_all_classes",
            "overall_accuracy", "cohen_kappa", "n_pixels",
        }
        assert result["n_pixels"] == 4

    def test_out_of_range_class_raises(self):
        with pytest.raises(ValueError):
            confusion_matrix(np.array([0, 4]), np.array([0, 1]))


# =============================================================================
# Trained artefacts (skipped when the data is not mounted)
# =============================================================================
@pytest.mark.skipif(not (FIXTURE / "manifest.csv").exists(), reason="fixture not available")
class TestFixtureContract:
    def test_splits_are_disjoint_and_cover_every_chip(self):
        import csv

        with (FIXTURE / "manifest.csv").open() as handle:
            manifest = list(csv.DictReader(handle))
        splits = {name: set() for name in ("train", "validation", "test")}
        for row in manifest:
            splits[row["split"]].add(row["chip_id"])
        assert len(splits["train"]) + len(splits["validation"]) + len(splits["test"]) == len(manifest)
        assert not (splits["train"] & splits["test"])
        assert not (splits["validation"] & splits["test"])
        assert not (splits["train"] & splits["validation"])

    def test_arrays_match_the_manifest(self):
        import csv

        images = np.load(FIXTURE / "images.npy", mmap_mode="r")
        labels = np.load(FIXTURE / "labels.npy", mmap_mode="r")
        assert images.shape == (2000, 4, 64, 64)
        assert labels.shape == (2000, 64, 64)
        assert images.dtype == np.uint16
        assert labels.dtype == np.uint8
        with (FIXTURE / "manifest.csv").open() as handle:
            manifest = list(csv.DictReader(handle))
        assert len(manifest) == len(images)
        assert set(np.unique(np.asarray(labels))) <= {0, 1, 2, 3}


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="trained checkpoint not available")
class TestTrainedCheckpoint:
    def test_hash_matches_sidecar(self):
        digest = file_sha256(CHECKPOINT)
        sidecar = Path(f"{CHECKPOINT}.sha256").read_text().split()[0]
        assert digest == sidecar

    def test_loads_and_predicts_on_a_fixture_chip(self):
        model, checkpoint = load_model(CHECKPOINT)
        assert checkpoint.config.base_channels >= 8
        images = np.load(FIXTURE / "images.npy", mmap_mode="r")
        chip = np.asarray(images[0], dtype=np.float32) / 10000.0
        probabilities = predict_proba(model, chip, checkpoint.band_mean, checkpoint.band_std)
        assert probabilities.shape == (4, 64, 64)
        assert np.allclose(probabilities.sum(axis=0), 1.0, atol=1e-5)


def test_importing_the_package_does_not_pull_torch():
    """The core package must stay NumPy-only; only ``satharmony.models`` needs torch."""
    code = "import sys, satharmony; assert 'torch' not in sys.modules, 'torch was imported'"
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
