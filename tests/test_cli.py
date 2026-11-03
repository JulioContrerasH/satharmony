"""Tests for the command line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.calibration.fit import _deterministic_config  # noqa: E402
from satharmony.cli import main  # noqa: E402
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402
from satharmony.manifests.manifest import ManifestRecord, sha256_file, write_manifest  # noqa: E402


def _arrays(tmp_path: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    s2 = rng.uniform(0.03, 0.85, size=(13, 48, 48)).astype(np.float32)
    mss, _ = emulate_s2(s2, config=_deterministic_config())
    s2_path = tmp_path / "s2.npy"
    mss_path = tmp_path / "mss.npy"
    np.save(s2_path, s2)
    np.save(mss_path, mss)
    return s2_path, mss_path, s2, mss


def test_emulate_writes_the_expected_shape(tmp_path: Path, capsys) -> None:
    s2_path, _, _, _ = _arrays(tmp_path)
    out = tmp_path / "emulated.npy"
    assert main(["emulate", str(s2_path), "-o", str(out)]) == 0
    assert np.load(out).shape == (4, 8, 8)
    payload = json.loads(capsys.readouterr().out)
    assert payload["factor"] == 6


def test_align_reports_zero_for_identical(tmp_path: Path, capsys) -> None:
    _, mss_path, _, _ = _arrays(tmp_path)
    assert main(["align", str(mss_path), str(mss_path), "--no-ecc"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert abs(payload["dy"]) < 1e-6
    assert abs(payload["dx"]) < 1e-6


def test_metrics_prints_sam_and_rmse(tmp_path: Path, capsys) -> None:
    _, mss_path, _, _ = _arrays(tmp_path)
    assert main(["metrics", str(mss_path), str(mss_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert abs(payload["sam_deg"]) < 1e-5
    assert abs(payload["rmse"]) < 1e-9


def test_fit_writes_a_loadable_profile(tmp_path: Path, capsys) -> None:
    s2_path, mss_path, _s2, _mss = _arrays(tmp_path)
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        [
            ManifestRecord(
                id="synthetic",
                kind="s2_mss",
                input_path=s2_path,
                reference_path=mss_path,
                input_sha256=sha256_file(s2_path),
                reference_sha256=sha256_file(mss_path),
                cloud_fraction=0.0,
                reason="unit test",
            )
        ],
        manifest,
    )
    out = tmp_path / "profile.json"
    assert main(["fit", str(manifest), "-o", str(out), "--n-boot", "10"]) == 0
    payload = json.loads(out.read_text())
    assert payload["n_pairs"] == 1
    assert "config" in payload
    from satharmony.emulators.s2_to_mss.config import PipelineConfig

    loaded = PipelineConfig.from_profile(out)
    assert loaded.final_scaling.band_scaling.green.mean > 0


def test_verify_reports_ok(tmp_path: Path, capsys) -> None:
    s2_path, mss_path, _, _ = _arrays(tmp_path)
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        [
            ManifestRecord(
                id="synthetic",
                kind="s2_mss",
                input_path=s2_path,
                reference_path=mss_path,
                input_sha256=sha256_file(s2_path),
                reference_sha256=sha256_file(mss_path),
                cloud_fraction=0.0,
                reason="unit test",
            )
        ],
        manifest,
    )
    assert main(["verify", str(manifest)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
