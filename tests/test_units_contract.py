"""The public API declares and enforces its input units."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.emulators.s2_to_mss.config import PipelineConfig  # noqa: E402
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402


def test_dn_input_is_rejected() -> None:
    image = np.full((13, 48, 48), 255.0, dtype=np.float32)
    with pytest.raises(ValueError, match="TOA"):
        emulate_s2(image, config=PipelineConfig())


def test_wrong_rank_is_rejected() -> None:
    with pytest.raises(ValueError, match="13, H, W"):
        emulate_s2(np.zeros((48, 48), dtype=np.float32), config=PipelineConfig())


def test_toa_input_is_accepted() -> None:
    rng = np.random.default_rng(0)
    image = rng.uniform(0.02, 0.9, size=(13, 48, 48)).astype(np.float32)
    mss, _ = emulate_s2(image, config=PipelineConfig(seed=1))
    assert mss.shape == (4, 8, 8)


def test_align_and_metrics_share_the_contract() -> None:
    from satharmony.calibration.align import align_pair
    from satharmony.calibration.metrics import summarize

    dn = np.full((64, 64), 200.0, dtype=np.float32)
    with pytest.raises(ValueError, match="TOA"):
        align_pair(dn, dn)
    with pytest.raises(ValueError, match="TOA"):
        summarize(dn[None, ...], dn[None, ...])
