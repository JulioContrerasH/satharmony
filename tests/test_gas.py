"""The optional B3 gas scale only touches MSS3."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.emulators.s2_to_mss.config import PipelineConfig  # noqa: E402
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402


def test_gas_scale_changes_only_band_3() -> None:
    rng = np.random.default_rng(0)
    image = rng.uniform(0.05, 0.8, size=(13, 48, 48)).astype(np.float32)
    base = PipelineConfig(seed=3)
    base.radiometric.enabled = False
    base.striping.enabled = False
    base.random_noise.enabled = False
    base.coherent_noise.enabled = False
    base.memory_effect.enabled = False
    base.scan_artifacts.enabled = False
    base.final_scaling.enabled = False

    off, _ = emulate_s2(image, config=base)
    on = base.model_copy(deep=True)
    on.spectral.gas_mss3_scale = 1.054
    with_gas, _ = emulate_s2(image, config=on)

    np.testing.assert_array_equal(off[[0, 1, 3]], with_gas[[0, 1, 3]])
    assert not np.allclose(off[2], with_gas[2])
