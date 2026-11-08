"""Tests for the v7 proxy benchmark script.

The benchmark needs the 42 real scenes, the checkpoints and a GPU, so the
expensive part is not tested here.  These tests pin the pieces that decide the
numbers and the subset: the OCM 4 -> 3 class mapping, the chip regime, and the
manifest reader.
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

from benchmark_proxy_v7 import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_OUT,
    MODEL_LABELS,
    _chip_regime,
    _pick_rows,
    _pooled_score,
    _scene_summary,
    ocm_to_three,
    read_test_rows,
)


def test_ocm_to_three_maps_both_cloud_classes_and_shadow() -> None:
    """OCM thick/thin become cloud; shadow keeps its code."""
    mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    assert ocm_to_three(mask).tolist() == [[0, 1], [1, 2]]


def test_chip_regime_thresholds() -> None:
    """The four regimes use the same cut points as the tables."""
    assert _chip_regime(0.0) == "clear"
    assert _chip_regime(0.009) == "clear"
    assert _chip_regime(0.01) == "low_cloud"
    assert _chip_regime(0.199) == "low_cloud"
    assert _chip_regime(0.20) == "mid_cloud"
    assert _chip_regime(0.499) == "mid_cloud"
    assert _chip_regime(0.50) == "high_cloud"
    assert _chip_regime(1.0) == "high_cloud"


def test_read_test_rows_filters_and_rejects_empty(tmp_path: Path) -> None:
    """Only the requested split is returned; an empty split fails loudly."""
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "id_name,split,category\nscene_a,train,clear\nscene_b,test,normal\nscene_c,test,hard\n"
    )
    rows = read_test_rows(manifest)
    assert [row["id_name"] for row in rows] == ["scene_b", "scene_c"]
    with pytest.raises(ValueError, match="no rows"):
        read_test_rows(manifest, split="validation")


def test_pooled_score_ignores_invalid_pixels() -> None:
    """The invalid mask removes pixels from the confusion matrix."""
    reference = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    prediction = np.array([[0, 1], [2, 2]], dtype=np.uint8)
    valid = np.array([[True, True], [False, False]])
    result = _pooled_score(prediction, reference, valid)
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    assert result["mean_iou"] == pytest.approx(1.0)


def test_scene_summary_uses_scenes_as_the_unit() -> None:
    """A scene with a bad mIoU moves the mean as much as a good one."""
    per_scene = {
        "ocm": {"a": 0.5, "b": 0.5},
        "model": {"a": 0.6, "b": 0.2},
    }
    summary = _scene_summary(per_scene, "ocm")
    assert summary["model"]["mean_miou"] == pytest.approx(0.4)
    assert summary["model"]["median_delta"] == pytest.approx(-0.1)
    assert summary["model"]["win_share"] == pytest.approx(0.5)
    assert summary["model"]["loss_share"] == pytest.approx(0.5)
    assert CLASS_NAMES == ("clear", "cloud", "cloud_shadow")


def test_calibrated_models_are_labelled_and_kept_out_of_the_old_results() -> None:
    """The retrained rows have names and the run writes to a new directory."""
    assert MODEL_LABELS["g4_cal"] == "G4 calibrado"
    assert MODEL_LABELS["g4b_cal"] == "G4b calibrado"
    assert DEFAULT_OUT.name == "bench-proxy-v7-cal"


def test_pick_rows_prefers_the_calibrated_model() -> None:
    """The chips drawn are the ones where the calibrated G4b wins or loses."""
    labels = np.zeros((4, 8, 8), dtype=np.uint8)
    valid = np.ones((4, 8, 8), dtype=bool)
    meta = [
        {"scene": "a", "row": 0, "col": 0, "cloud_frac": 0.0, "shadow_frac": 0.0},
        {"scene": "b", "row": 0, "col": 8, "cloud_frac": 0.3, "shadow_frac": 0.0},
        {"scene": "c", "row": 8, "col": 0, "cloud_frac": 0.7, "shadow_frac": 0.0},
        {"scene": "d", "row": 8, "col": 8, "cloud_frac": 0.1, "shadow_frac": 0.5},
    ]
    masks = {
        "ocm_mss_60m": labels.copy(),
        "g4b_cal": np.full_like(labels, 1),
    }
    rows = _pick_rows(labels, masks, meta, valid)
    assert len(rows) == len(set(rows)) <= 6
    assert 3 in rows  # the shadow-rich chip is always drawn
