"""Tests for the versioned JSON profiles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.emulators.s2_to_mss.config import PipelineConfig  # noqa: E402
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402


def test_v1_profile_matches_hardcoded_config() -> None:
    """The JSON is the source of truth and still equals the hardcoded defaults."""
    loaded = PipelineConfig.from_profile("mss5_esa_gtc_v1")
    assert loaded.model_dump() == PipelineConfig.mss5_esa_gtc_v1().model_dump()


def test_legacy_profile_matches_hardcoded_config() -> None:
    loaded = PipelineConfig.from_profile("legacy_0_1_13")
    assert loaded.model_dump() == PipelineConfig.legacy_0_1_13().model_dump()


def test_profile_payload_has_provenance() -> None:
    path = REPO / "satharmony/profiles/mss5_esa_gtc_v1.json"
    payload = json.loads(path.read_text())
    for key in ("schema_version", "name", "created", "config"):
        assert key in payload
    assert payload["schema_version"] == 1


def test_unknown_profile_lists_alternatives() -> None:
    with pytest.raises(FileNotFoundError, match="available:"):
        PipelineConfig.from_profile("does_not_exist")


def test_profile_loads_from_path() -> None:
    path = REPO / "satharmony/profiles/mss5_esa_gtc_v1.json"
    assert PipelineConfig.from_profile(path).model_dump() == PipelineConfig.mss5_esa_gtc_v1().model_dump()


def test_emulation_from_json_is_bit_identical() -> None:
    """Loading the profile must not change a single DN."""
    rng = np.random.default_rng(0)
    s2 = rng.uniform(0.02, 0.9, size=(13, 48, 48)).astype(np.float32)

    from_json, _ = emulate_s2(
        s2, config=PipelineConfig.from_profile("mss5_esa_gtc_v1"), seed=7
    )
    from_code, _ = emulate_s2(s2, config=PipelineConfig.mss5_esa_gtc_v1(), seed=7)
    np.testing.assert_array_equal(from_json, from_code)
