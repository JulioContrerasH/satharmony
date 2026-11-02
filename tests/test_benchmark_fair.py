"""Tests for the fair MSS benchmark script.

The benchmark itself needs the fixture, the checkpoints and a GPU, so the
expensive part is not tested here.  These tests pin the pure pieces that decide
the numbers:

* the 4-class -> 3-class collapse, which is the only common encoding of the
  old QA4EO-2 checkpoints;
* the metric bundle, against a hand-computed confusion matrix;
* the input conventions tried for the old checkpoints (raw, 0.1.13 radiometry
  and per-chip z-score), because that block is what documents the caveat.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from benchmark_mss_common_test import (  # noqa: E402
    CLASS_NAMES_3,
    collapse_to_three,
    legacy_input_variants,
    pairwise_stats,
    per_chip_scores,
    score,
)


def test_collapse_to_three_merges_both_cloud_classes() -> None:
    """Thick and thin become cloud; shadow and clear keep their meaning."""
    mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    collapsed = collapse_to_three(mask)
    assert collapsed.tolist() == [[0, 1], [1, 2]]
    assert collapsed.dtype == np.uint8


def test_collapse_to_three_rejects_out_of_range_classes() -> None:
    """A 4 in the input is not a CloudSEN12 class and must fail loudly."""
    with pytest.raises(ValueError, match="classes must be in"):
        collapse_to_three(np.array([[0, 4]], dtype=np.uint8))


def test_score_matches_a_hand_computed_matrix() -> None:
    """One wrong pixel per class: IoU, OA and kappa against the arithmetic."""
    reference = np.array([0, 0, 1, 2])
    prediction = np.array([0, 1, 1, 2])
    result = score(prediction, reference, CLASS_NAMES_3)

    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert result["per_class"]["clear"]["iou"] == pytest.approx(1 / 2)
    assert result["per_class"]["cloud"]["iou"] == pytest.approx(1 / 2)
    assert result["per_class"]["cloud_shadow"]["iou"] == pytest.approx(1.0)
    assert result["mean_iou"] == pytest.approx(2 / 3)
    assert result["overall_accuracy"] == pytest.approx(0.75)
    assert result["cohen_kappa"] == pytest.approx(7 / 11)
    assert result["n_pixels"] == 4


def test_score_perfect_prediction() -> None:
    """A perfect mask has IoU, OA and kappa equal to one."""
    reference = np.array([0, 1, 2, 2])
    result = score(reference.copy(), reference, CLASS_NAMES_3)
    assert result["mean_iou"] == pytest.approx(1.0)
    assert result["overall_accuracy"] == pytest.approx(1.0)
    assert result["cohen_kappa"] == pytest.approx(1.0)


def test_score_ignores_absent_classes() -> None:
    """A class with no reference and no prediction does not drag the mean to zero."""
    reference = np.array([0, 1, 1])
    prediction = np.array([0, 1, 1])
    result = score(prediction, reference, CLASS_NAMES_3)
    assert np.isnan(result["per_class"]["cloud_shadow"]["iou"])
    assert result["mean_iou"] == pytest.approx(1.0)


def test_legacy_input_variants_without_reference() -> None:
    """Without a reference the block only offers raw and z-score."""
    images = np.random.default_rng(0).random((2, 4, 8, 8), dtype=np.float32)
    variants = legacy_input_variants(images, None)
    assert set(variants) == {"raw", "zscore"}
    assert np.array_equal(variants["raw"], images)


def test_legacy_input_variants_map_the_global_radiometry() -> None:
    """The affine maps the fixture to the reference per-band mean and std."""
    rng = np.random.default_rng(1)
    images = (0.1 + 0.2 * rng.random((3, 4, 16, 16))).astype(np.float32)
    reference = {
        "mean": [0.28, 0.25, 0.32, 0.29],
        "std": [0.12, 0.11, 0.11, 0.17],
    }
    variants = legacy_input_variants(images, reference)

    mapped = variants["legacy_radiometry"]
    assert np.allclose(mapped.mean(axis=(0, 2, 3)), reference["mean"], atol=1e-5)
    assert np.allclose(mapped.std(axis=(0, 2, 3)), reference["std"], atol=1e-4)

    zscore = variants["zscore"]
    assert np.allclose(zscore.mean(axis=(-2, -1)), 0.0, atol=1e-5)
    assert np.allclose(zscore.std(axis=(-2, -1)), 1.0, atol=1e-4)


def test_per_chip_scores_weights_every_chip_equally() -> None:
    """Two chips of different size: the per-chip mean is the mean of the two IoUs."""
    labels = np.zeros((2, 4, 4), dtype=np.uint8)
    masks = {"perfect": labels.copy(), "empty": labels.copy()}
    masks["empty"][1] = 1  # all cloud on a chip that is all clear
    scores = per_chip_scores(masks, labels)
    assert scores["perfect"].tolist() == [1.0, 1.0]
    assert scores["empty"].tolist() == [1.0, 0.0]


def test_pairwise_stats_counts_wins_and_losses() -> None:
    """A model that only ties is neither a win nor a loss."""
    per_chip = {
        "base": np.array([0.5, 0.5, 0.5]),
        "tie": np.array([0.5, 0.5, 0.5]),
        "better": np.array([0.8, 0.2, 0.5]),
        "worse": np.array([0.2, 0.8, 0.5]),
    }
    stats = pairwise_stats(per_chip, "base")
    assert stats["tie"]["win_share"] == 0.0
    assert stats["tie"]["loss_share"] == 0.0
    assert stats["better"]["win_share"] == pytest.approx(1 / 3)
    assert stats["better"]["loss_share"] == pytest.approx(1 / 3)
    assert stats["better"]["chips_won_by_010"] == 1
    assert stats["better"]["chips_lost_by_010"] == 1
    assert stats["worse"]["chips_won_by_010"] == 1
