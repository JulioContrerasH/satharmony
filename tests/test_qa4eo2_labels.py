"""Tests for the QA4EO-2 label and comparison scripts.

The heavy parts (the 42 scenes, the checkpoints, a GPU) are not tested here.
These tests pin the pieces that decide the numbers: the tile grid, the 4 -> 3
class merge, the valid mask, the aggregation rule of the v7 labels, the
boundary F1 and the block-level detection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from audit_layer_alignment import aggregate_labels_3class  # noqa: E402
from compare_qa4eo2_v7 import (  # noqa: E402
    boundary_f1,
    boundary_map,
    detection_blocks,
    detection_scores,
    false_positive_fraction,
    ocm_to_three,
    pool_detection,
    three_class_scores,
)
from qa4eo2_labels import (  # noqa: E402
    _tiles,
    probabilities_to_three,
    valid_mask,
)
from qa4eo2_labels_30m import (  # noqa: E402
    align_shift,
    duplicate_purity,
    four_class_argmax,
    iou_from_confusion,
    merge_thin_thick,
    upsample_reference,
)

from satharmony.models.metrics import confusion_matrix  # noqa: E402


# ── tiling ──────────────────────────────────────────────────────────────
def test_tiles_cover_the_whole_image() -> None:
    """Every pixel must belong to at least one tile, with no gap."""
    height, width, chunk, overlap = 1000, 700, 256, 64
    pad_top, pad_bottom, pad_left, pad_right, coords = _tiles(
        height, width, chunk, chunk - overlap
    )
    padded_h = height + pad_top + pad_bottom
    padded_w = width + pad_left + pad_right
    covered = np.zeros((padded_h, padded_w), dtype=bool)
    for row, col in coords:
        covered[row : row + chunk, col : col + chunk] = True
    assert covered.all()
    assert coords[0] == (0, 0)
    assert coords[-1][0] + chunk == padded_h
    assert coords[-1][1] + chunk == padded_w


def test_tiles_pad_small_images_to_one_tile() -> None:
    """An image smaller than the tile is centred in a single padded tile."""
    pad_top, pad_bottom, pad_left, pad_right, coords = _tiles(100, 80, 512, 256)
    assert coords == [(0, 0)]
    assert pad_top + pad_bottom + 100 == 512
    assert pad_left + pad_right + 80 == 512


# ── classes ─────────────────────────────────────────────────────────────
def test_probabilities_to_three_merges_thin_and_thick() -> None:
    """Thin + thick beat clear once summed; shadow stays its own class."""
    probabilities = np.zeros((4, 2, 2), dtype=np.float32)
    probabilities[0] = 0.30  # clear
    probabilities[1] = 0.15  # thin
    probabilities[2] = 0.40  # thick
    probabilities[3] = 0.15  # shadow
    assert probabilities_to_three(probabilities).tolist() == [[1, 1], [1, 1]]


def test_probabilities_to_three_keeps_three_class_models() -> None:
    """A 3-class volume is a plain argmax."""
    probabilities = np.zeros((3, 1, 3), dtype=np.float32)
    probabilities[0] = [0.8, 0.1, 0.1]
    probabilities[1] = [0.1, 0.8, 0.1]
    probabilities[2] = [0.1, 0.1, 0.8]
    assert probabilities_to_three(probabilities).tolist() == [[0, 1, 2]]


def test_probabilities_to_three_rejects_unmerged_four_classes() -> None:
    """The 3-class convention cannot hold thin and thick apart."""
    with pytest.raises(ValueError):
        probabilities_to_three(np.zeros((4, 2, 2), dtype=np.float32), merge_clouds=False)


def test_valid_mask_needs_every_band() -> None:
    """A pixel with one zero band is no data."""
    image = np.ones((4, 2, 2), dtype=np.float32)
    image[2, 0, 0] = 0.0
    mask = valid_mask(image)
    assert mask[0, 0] == np.False_
    assert mask[1, 1] == np.True_
    assert int(mask.sum()) == 3


def test_ocm_to_three_maps_thick_thin_and_shadow() -> None:
    """OCM 4-class order is clear, thick, thin, shadow."""
    assert ocm_to_three(np.array([[0, 1], [2, 3]], dtype=np.uint8)).tolist() == [[0, 1], [1, 2]]


# ── aggregation of the v7 labels ────────────────────────────────────────
def test_aggregation_majority_of_clear() -> None:
    """Three clear pixels and one cloud stay clear."""
    labels = np.array([[0, 0], [0, 1]], dtype=np.uint8)
    assert aggregate_labels_3class(labels).tolist() == [[0]]


def test_aggregation_no_majority_prefers_cloud_over_shadow() -> None:
    """Three of each in a 3 x 3 block: no majority, cloud wins the tie."""
    labels = np.array([[1, 1, 1], [2, 2, 2], [0, 0, 0]], dtype=np.uint8)
    assert aggregate_labels_3class(labels, scale=3).tolist() == [[1]]


def test_aggregation_no_majority_clear_beats_both_cloud_and_shadow() -> None:
    """Three clear, two cloud and four shadow: shadow dominates the tie."""
    labels = np.array([[0, 0, 0], [1, 1, 2], [2, 2, 2]], dtype=np.uint8)
    assert aggregate_labels_3class(labels, scale=3).tolist() == [[2]]


def test_aggregation_exact_half_tie_keeps_shadow() -> None:
    """The reference script assigns shadow last, so a 2-2 tie ends as shadow.

    This is the real behaviour of ``58_generate_mss60_and_label60.py``: the
    "cloud > shadow" order only decides the *no majority* branch, while the
    half-and-half case is overwritten by the shadow assignment.
    """
    labels = np.array([[1, 1], [2, 2]], dtype=np.uint8)
    assert aggregate_labels_3class(labels).tolist() == [[2]]


def test_aggregation_tie_prefers_shadow_over_clear() -> None:
    """Two shadow and two clear become shadow."""
    labels = np.array([[2, 2], [0, 0]], dtype=np.uint8)
    assert aggregate_labels_3class(labels).tolist() == [[2]]


# ── metrics ─────────────────────────────────────────────────────────────
def test_three_class_scores_rename_the_cloudsen12_keys() -> None:
    """Position 1 must be cloud and position 2 shadow, not the reverse."""
    matrix = confusion_matrix(
        np.array([0, 1, 2, 1]),
        np.array([0, 0, 2, 1]),
        num_classes=3,
    )
    scores = three_class_scores(matrix)
    assert scores["clear"]["support"] == 2
    assert scores["cloud"]["support"] == 1
    assert scores["cloud_shadow"]["support"] == 1
    assert scores["cloud"]["recall"] == 1.0
    assert scores["cloud_shadow"]["precision"] == 1.0
    assert scores["clear"]["recall"] == 0.5


def test_boundary_map_is_one_pixel() -> None:
    """A 3 x 3 square has an 8-pixel ring."""
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    assert int(boundary_map(mask).sum()) == 8


def test_boundary_f1_is_one_for_identical_masks() -> None:
    """Same mask, same boundary."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 6:14] = True
    assert boundary_f1(mask, mask) == pytest.approx(1.0)


def test_boundary_f1_is_zero_for_empty_and_full() -> None:
    """One mask with no boundary cannot be matched."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 6:14] = True
    assert boundary_f1(mask, np.zeros_like(mask)) == 0.0


def test_boundary_f1_tolerates_a_one_pixel_shift() -> None:
    """A one-pixel shift is inside the tolerance; a large one is not."""
    reference = np.zeros((40, 40), dtype=bool)
    reference[10:30, 10:30] = True
    near = np.zeros_like(reference)
    near[10:30, 11:31] = True
    far = np.zeros_like(reference)
    far[10:30, 20:40] = True
    assert boundary_f1(near, reference, tolerance=1) > 0.5
    assert boundary_f1(far, reference, tolerance=1) < 0.75


def test_false_positive_fraction_counts_only_reference_clear() -> None:
    """Cloud over the reference cloud is not a false positive."""
    reference = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    prediction = np.array([[1, 0], [1, 2]], dtype=np.uint8)
    values = false_positive_fraction(prediction, reference)
    assert values["cloud"] == pytest.approx(0.5)
    assert values["shadow"] == pytest.approx(0.0)
    assert values["any"] == pytest.approx(0.5)


def test_detection_blocks_need_half_valid_and_threshold() -> None:
    """Two of four blocks are valid; only the cloudy one is flagged."""
    reference = np.zeros((512, 512), dtype=np.uint8)
    reference[:256, :256] = 1
    prediction = np.zeros((512, 512), dtype=np.uint8)
    prediction[:256, :256] = 1
    valid = np.ones((512, 512), dtype=bool)
    valid[:256, 256:448] = False  # 75 % of that block is no data
    rows = detection_blocks(prediction, reference, valid)
    assert len(rows) == 3
    scores = detection_scores(rows)
    assert scores["cloud"]["tp"] == 1
    assert scores["cloud"]["precision"] == 1.0
    assert scores["cloud"]["recall"] == 1.0


def test_pool_detection_sums_counts_not_ratios() -> None:
    """Pooling must weight blocks, not scenes."""
    scenes = [
        {
            "detection": {
                "cloud": {"tp": 1, "fp": 0, "fn": 0, "blocks": 10, "present": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0},
                "cloud_shadow": {"tp": 0, "fp": 0, "fn": 0, "blocks": 10, "present": 0, "precision": float("nan"), "recall": float("nan"), "f1": float("nan")},
            }
        },
        {
            "detection": {
                "cloud": {"tp": 1, "fp": 9, "fn": 9, "blocks": 100, "present": 10, "precision": 0.1, "recall": 0.1, "f1": 0.1},
                "cloud_shadow": {"tp": 0, "fp": 0, "fn": 0, "blocks": 100, "present": 0, "precision": float("nan"), "recall": float("nan"), "f1": float("nan")},
            }
        },
    ]
    pooled = pool_detection(scenes)
    assert pooled["cloud"]["precision"] == pytest.approx(2 / 11)
    assert pooled["cloud"]["recall"] == pytest.approx(2 / 11)


# ── 30 m product ────────────────────────────────────────────────────────
def test_four_class_argmax_keeps_thin_and_thick_apart() -> None:
    """The 30 m product is 4-class: position 1 is thin, position 2 thick."""
    probabilities = np.zeros((4, 1, 4), dtype=np.float32)
    probabilities[1, 0, 0] = 0.9
    probabilities[2, 0, 1] = 0.9
    probabilities[3, 0, 2] = 0.9
    probabilities[0, 0, 3] = 0.9
    assert four_class_argmax(probabilities).tolist() == [[1, 2, 3, 0]]


def test_four_class_argmax_rejects_three_classes() -> None:
    """A 3-class volume cannot feed the 4-class product."""
    with pytest.raises(ValueError):
        four_class_argmax(np.zeros((3, 4, 4), dtype=np.float32))


def test_merge_thin_thick_maps_the_four_class_convention() -> None:
    """1 and 2 become cloud, 3 becomes shadow, 255 stays no data."""
    label = np.array([[0, 1, 2, 3, 255]], dtype=np.uint8)
    assert merge_thin_thick(label).tolist() == [[0, 1, 1, 2, 255]]


def test_align_shift_recovers_a_known_offset() -> None:
    """A 2 x 2 duplicate shifted by one row and one column must be measured."""
    original = np.random.default_rng(7).integers(1, 900, size=(3, 20, 24)).astype(np.float32)
    upsampled = np.repeat(np.repeat(original, 2, axis=1), 2, axis=2)
    aligned = np.zeros((3, 41, 49), dtype=np.float32)
    aligned[:, 1:41, 1:49] = upsampled
    score, dy, dx = align_shift(aligned, original, step=1, minimum_samples=10)
    assert score == pytest.approx(1.0)
    assert (dy, dx) == (1, 1)


def test_align_shift_recovers_no_offset() -> None:
    """The plain duplicate is (0, 0), the case the audit did not expect."""
    original = np.random.default_rng(3).integers(1, 900, size=(3, 20, 24)).astype(np.float32)
    upsampled = np.repeat(np.repeat(original, 2, axis=1), 2, axis=2)
    score, dy, dx = align_shift(upsampled, original, step=1, minimum_samples=10)
    assert score == pytest.approx(1.0)
    assert (dy, dx) == (0, 0)


def test_upsample_reference_follows_the_measured_shift() -> None:
    """Column 0 falls outside the source when the duplicate is shifted by one."""
    label = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    plain = upsample_reference(label, 0, 0, (4, 4))
    assert plain.tolist() == [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]
    shifted = upsample_reference(label, 0, 1, (4, 5))
    assert shifted.tolist() == [
        [255, 0, 0, 1, 1],
        [255, 0, 0, 1, 1],
        [255, 2, 2, 3, 3],
        [255, 2, 2, 3, 3],
    ]


def test_duplicate_purity_is_one_for_a_pure_upsample() -> None:
    """Every 2 x 2 block of a duplicated 60 m map is uniform."""
    label = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=np.uint8)
    assert duplicate_purity(label) == pytest.approx(1.0)


def test_duplicate_purity_drops_with_real_30m_detail() -> None:
    """One pixel off inside a block is not an upsample any more."""
    label = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=np.uint8)
    assert duplicate_purity(label) == pytest.approx(0.75)


def test_duplicate_purity_uses_the_measured_phase() -> None:
    """The blocks start at the offset the duplicated MSS carries."""
    label = np.array(
        [
            [9, 9, 9, 9, 9],
            [9, 0, 0, 1, 1],
            [9, 0, 0, 1, 1],
            [9, 2, 2, 3, 3],
            [9, 2, 2, 3, 3],
        ],
        dtype=np.uint8,
    )
    assert duplicate_purity(label, dy=0, dx=0) < 1.0
    assert duplicate_purity(label, dy=1, dx=1) == pytest.approx(1.0)


def test_duplicate_purity_skips_blocks_with_no_data() -> None:
    """A 2 x 2 block touching 255 is not scored."""
    label = np.array([[0, 255], [0, 255]], dtype=np.uint8)
    assert np.isnan(duplicate_purity(label))


def test_iou_from_confusion_is_nan_for_an_absent_class() -> None:
    """A class with no support gives NaN, not zero or a division error."""
    matrix = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 0]], dtype=np.int64)
    ious = iou_from_confusion(matrix)
    assert ious[0] == pytest.approx(1.0)
    assert ious[1] == pytest.approx(1.0)
    assert np.isnan(ious[2])
