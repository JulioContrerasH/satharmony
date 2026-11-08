"""Tests for the 30 m label built from the canonical 60 m product.

The heavy parts (42 scenes, real rasters) are not tested here. These tests pin
the no-data contract of the upsampled product and the preservation of the
model-generated 30 m output.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from qa4eo2_labels import IGNORE, write_label  # noqa: E402
from qa4eo2_labels_30m_from_60m import (  # noqa: E402
    is_model_json,
    is_model_manifest,
    preserve_model_label,
    verify_upsampled,
)


def _write_mss(path: Path) -> dict:
    """4-band 4 x 5 MSS with one pixel of no data in every band."""
    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 5,
        "count": 4,
        "dtype": "uint16",
        "crs": "EPSG:32635",
        "transform": from_origin(0, 120, 30, 30),
        "nodata": 0,
    }
    image = np.full((4, 4, 5), 1000, dtype=np.uint16)
    image[:, 1, 1] = 0
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(image)
    return profile


def test_verify_upsampled_allows_extra_nodata_outside_the_60m_footprint(
    tmp_path: Path,
) -> None:
    """A 60 m no-data pixel is allowed where the 30 m MSS has data."""
    mss = tmp_path / "mss_aligned.tif"
    profile = _write_mss(mss)
    reference = np.zeros((4, 5), dtype=np.uint8)
    reference[0, 0] = IGNORE
    reference[0, 1] = 1
    reference[1, 1] = 2  # MSS no data, gets IGNORE below
    reference[3, 3] = IGNORE  # outside the 60 m footprint
    label = reference.copy()
    label[1, 1] = IGNORE
    label_path = tmp_path / "label_qa4eo_2.tif"
    write_label(label_path, label, profile)

    checks, extra = verify_upsampled(label_path, mss, reference)

    assert all(checks.values())
    assert extra == 2


def test_verify_upsampled_rejects_a_class_on_mss_nodata(tmp_path: Path) -> None:
    """A class where the MSS has no data must fail the check."""
    mss = tmp_path / "mss_aligned.tif"
    profile = _write_mss(mss)
    reference = np.zeros((4, 5), dtype=np.uint8)
    label = reference.copy()
    label[1, 1] = 0  # class on MSS no data
    label_path = tmp_path / "label_qa4eo_2.tif"
    write_label(label_path, label, profile)

    with pytest.raises(ValueError, match="nodata_covers_mss"):
        verify_upsampled(label_path, mss, reference)


def test_preserve_model_label_moves_only_the_model_output(tmp_path: Path) -> None:
    """The model 30 m label is moved aside; the upsampled one is left alone."""
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32635",
        "transform": from_origin(0, 60, 30, 30),
    }
    current = tmp_path / "label_qa4eo_2.tif"
    with rasterio.open(current, "w", **profile) as dst:
        dst.write(np.zeros((2, 2), dtype=np.uint8), 1)
        dst.update_tags(provenance="scripts/qa4eo2_labels_30m.py")

    model = preserve_model_label(tmp_path)

    assert model == tmp_path / "label_qa4eo_2_model30m.tif"
    assert model.exists()
    assert not current.exists()

    with rasterio.open(current, "w", **profile) as dst:
        dst.write(np.ones((2, 2), dtype=np.uint8), 1)
        dst.update_tags(method="upsample_60m")
    assert preserve_model_label(tmp_path) == model
    assert current.exists()


def test_is_model_manifest_requires_four_classes_and_no_method(tmp_path: Path) -> None:
    """The model manifest is the one with 4 classes and no method column."""
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id_name", "model_classes"])
        writer.writeheader()
        writer.writerow({"id_name": "a", "model_classes": "4"})
    assert is_model_manifest(path)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id_name", "model_classes", "method"])
        writer.writeheader()
        writer.writerow({"id_name": "a", "model_classes": "4", "method": "upsample_60m"})
    assert not is_model_manifest(path)


def test_is_model_json_checks_classes_and_method(tmp_path: Path) -> None:
    """The model JSON is the one with 4 classes and no method key."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"model_classes": 4}))
    assert is_model_json(path)
    path.write_text(json.dumps({"model_classes": 4, "method": "upsample_60m"}))
    assert not is_model_json(path)
    path.write_text(json.dumps({"model_classes": 3}))
    assert not is_model_json(path)
