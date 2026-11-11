"""Emulate any calibrated MSS sensor, explicitly or at random.

This is the entry point for augmentation: the sensor is sampled from the
calibrated ones with the caller's seed, so a run repeats exactly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .config import PipelineConfig
from .functional import emulate_s2
from .sensors import (
    MSS_SENSORS,
    SensorNotCalibrated,
    SensorSpec,
    resolve_sensor,
)


def _candidates(sensors: Sequence[str] | None) -> list[SensorSpec] | None:
    if sensors is None:
        return None
    pool: list[SensorSpec] = []
    for name in sensors:
        spec = MSS_SENSORS.get(name.upper())
        if spec is None:
            raise SensorNotCalibrated(f"unknown sensor {name!r}")
        if not spec.calibrated:
            raise SensorNotCalibrated(f"{spec.key} has no calibrated profile yet")
        pool.append(spec)
    return pool


def emulate_any(
    image: NDArray[np.floating],
    sensor: str = "random",
    seed: int | None = None,
    sensors: Sequence[str] | None = None,
    scale_factor: int | None = None,
) -> tuple[NDArray[np.float32], int, str]:
    """Emulate ``image`` targeting one MSS sensor.

    Parameters
    ----------
    image:
        ``(13, H, W)`` S2 TOA reflectance in the package band order.
    sensor:
        A sensor key (``"L5_MSS"``) or ``"random"``. Random samples uniformly
        over ``sensors`` (default: every calibrated sensor).
    seed:
        Controls both the sensor choice and the stochastic degradations.
    sensors:
        Optional subset of sensor keys to sample from.
    scale_factor:
        Optional fixed downsampling factor.

    Returns
    -------
    ``(mss, factor, sensor_key)``.
    """
    rng = np.random.default_rng(seed)
    spec = resolve_sensor(sensor, rng, candidates=_candidates(sensors))

    config = PipelineConfig.from_profile(spec.profile)
    declared = config.radiometric.satellite
    if declared != spec.key:
        raise ValueError(
            f"profile {spec.profile!r} targets {declared!r}, not {spec.key!r}"
        )

    mss, factor = emulate_s2(image, scale_factor=scale_factor, seed=seed, config=config)
    return mss, factor, spec.key
