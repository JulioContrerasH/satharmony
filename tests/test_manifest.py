"""Tests for the calibration manifest."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.manifests.manifest import (  # noqa: E402
    ManifestRecord,
    load_manifest,
    sha256_file,
    verify_manifest,
    write_manifest,
)


def _record(tmp_path: Path, name: str = "a") -> ManifestRecord:
    source = tmp_path / f"{name}_s2.tif"
    reference = tmp_path / f"{name}_mss.tif"
    source.write_bytes(b"s2")
    reference.write_bytes(b"mss")
    return ManifestRecord(
        id=name,
        kind="s2_mss",
        input_path=source,
        reference_path=reference,
        input_sha256=sha256_file(source),
        reference_sha256=sha256_file(reference),
        cloud_fraction=0.12,
        reason="calibration",
    )


def test_roundtrip(tmp_path: Path) -> None:
    records = [_record(tmp_path, "a"), _record(tmp_path, "b")]
    path = tmp_path / "manifest.csv"
    write_manifest(records, path)
    loaded = load_manifest(path)
    assert [r.id for r in loaded] == ["a", "b"]
    assert loaded[0].input_path == records[0].input_path
    assert loaded[0].kind == "s2_mss"
    assert loaded[0].cloud_fraction == pytest.approx(0.12)


def test_verify_detects_a_hash_mismatch(tmp_path: Path) -> None:
    record = _record(tmp_path)
    record.input_path.write_bytes(b"changed")
    problems = verify_manifest([record])
    assert any("hash mismatch" in problem for problem in problems)


def test_verify_detects_a_missing_file(tmp_path: Path) -> None:
    record = _record(tmp_path)
    record.reference_path.unlink()
    problems = verify_manifest([record])
    assert any("not found" in problem for problem in problems)


def test_missing_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("id,input_path\nx,y\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_manifest(path)


def test_empty_field_raises(tmp_path: Path) -> None:
    record = _record(tmp_path)
    path = tmp_path / "manifest.csv"
    write_manifest([record], path)
    text = path.read_text().replace("calibration", "")
    path.write_text(text)
    with pytest.raises(ValueError, match="empty"):
        load_manifest(path)


def test_unknown_kind_raises(tmp_path: Path) -> None:
    record = _record(tmp_path)
    path = tmp_path / "manifest.csv"
    write_manifest([record], path)
    path.write_text(path.read_text().replace("s2_mss", "other"))
    with pytest.raises(ValueError, match="unknown kind"):
        load_manifest(path)
