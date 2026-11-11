"""Tests for the multi-sensor registry and the random selection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.emulators.s2_to_mss import (  # noqa: E402
    MSS_SENSORS,
    PipelineConfig,
    SensorNotCalibrated,
    SensorSpec,
    available_sensors,
    calibrated_sensors,
    emulate_any,
    emulate_s2,
)
from satharmony.emulators.s2_to_mss.sensors import (  # noqa: E402
    choose_sensor,
    resolve_sensor,
)


def _image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.02, 0.9, size=(13, 32, 32)).astype(np.float32)


def test_registry_covers_the_five_mss_sensors() -> None:
    assert set(MSS_SENSORS) == {"L1_MSS", "L2_MSS", "L3_MSS", "L4_MSS", "L5_MSS"}
    for spec in MSS_SENSORS.values():
        assert spec.bands == 4
        assert spec.quantization_bits == 6
        assert spec.ifov_m in (79.0, 82.0)


def test_only_l5_is_calibrated_and_l4_is_experimental() -> None:
    assert calibrated_sensors() == ("L5_MSS",)
    assert available_sensors() == ("L4_MSS", "L5_MSS")
    assert MSS_SENSORS["L4_MSS"].status == "experimental"


def test_l5_matches_the_single_sensor_path_bit_for_bit() -> None:
    image = _image()
    expected, factor_expected = emulate_s2(
        image, config=PipelineConfig.from_profile("mss5_esa_gtc_v2"), seed=7
    )
    mss, factor, key = emulate_any(image, sensor="L5_MSS", seed=7)
    assert key == "L5_MSS"
    assert factor == factor_expected
    np.testing.assert_array_equal(mss, expected)


def test_random_is_deterministic_with_one_calibrated_sensor() -> None:
    image = _image()
    a, _, key_a = emulate_any(image, sensor="random", seed=11)
    b, _, key_b = emulate_any(image, sensor="random", seed=11)
    assert key_a == key_b == "L5_MSS"
    np.testing.assert_array_equal(a, b)


def test_uncalibrated_sensor_raises_and_lists_the_available_ones() -> None:
    with pytest.raises(SensorNotCalibrated, match="L5_MSS"):
        emulate_any(_image(), sensor="L1_MSS", seed=0)


def test_unknown_sensor_raises() -> None:
    with pytest.raises(SensorNotCalibrated, match="unknown sensor"):
        resolve_sensor("L9_MSS", np.random.default_rng(0))


def test_choose_sensor_reaches_every_candidate() -> None:
    pool = [
        SensorSpec("A", 1, 79.0, 6, 4, "a.json", "calibrated"),
        SensorSpec("B", 2, 79.0, 6, 4, "b.json", "calibrated"),
    ]
    rng = np.random.default_rng(0)
    picked = {choose_sensor(rng, pool).key for _ in range(40)}
    assert picked == {"A", "B"}


def test_sensor_subset_restricts_the_choice() -> None:
    with pytest.raises(SensorNotCalibrated):
        emulate_any(_image(), sensor="random", seed=0, sensors=["L1_MSS"])


def test_l4_runs_through_its_own_profile() -> None:
    mss, _, key = emulate_any(_image(), sensor="L4_MSS", seed=3)
    assert key == "L4_MSS"
    assert mss.shape == (4, 5, 5)
    assert PipelineConfig.from_profile("mss4_esa_gtc_v1").radiometric.satellite == "L4_MSS"


def test_random_over_two_available_sensors_reaches_both() -> None:
    image = _image()
    picked = {emulate_any(image, sensor="random", seed=seed, sensors=["L4_MSS", "L5_MSS"])[2] for seed in range(40)}
    assert picked == {"L4_MSS", "L5_MSS"}


def test_default_random_stays_on_calibrated_sensors() -> None:
    keys = {emulate_any(_image(), sensor="random", seed=seed)[2] for seed in range(20)}
    assert keys == {"L5_MSS"}


def test_profile_targets_the_declared_sensor() -> None:
    config = PipelineConfig.from_profile("mss5_esa_gtc_v2")
    assert config.radiometric.satellite == "L5_MSS"
