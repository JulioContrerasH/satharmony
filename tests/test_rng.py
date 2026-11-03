"""The emulator must not touch NumPy's global random state."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.emulators.s2_to_mss.config import PipelineConfig  # noqa: E402
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402
from satharmony.emulators.s2_to_mss.pipeline import MSSEmulator  # noqa: E402


def _image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.02, 0.9, size=(13, 48, 48)).astype(np.float32)


def test_emulation_leaves_global_rng_untouched() -> None:
    np.random.seed(123)
    expected = np.random.random(8)

    np.random.seed(123)
    emulate_s2(_image(), config=PipelineConfig(seed=99))
    after = np.random.random(8)

    np.testing.assert_array_equal(expected, after)


def test_same_seed_same_bytes_across_emulators() -> None:
    config = PipelineConfig(seed=7)
    first = MSSEmulator(config)(_image())
    second = MSSEmulator(config)(_image())
    np.testing.assert_array_equal(first, second)


def test_external_generator_is_reproducible() -> None:
    image = _image()
    first, _ = emulate_s2(image, config=PipelineConfig(), rng=np.random.default_rng(5))
    second, _ = emulate_s2(image, config=PipelineConfig(), rng=np.random.default_rng(5))
    np.testing.assert_array_equal(first, second)


def test_external_generator_advances_between_calls() -> None:
    image = _image()
    rng = np.random.default_rng(5)
    first = MSSEmulator(PipelineConfig(), rng=rng)(image)
    second = MSSEmulator(PipelineConfig(), rng=rng)(image)
    assert not np.array_equal(first, second)
