"""Registry of the Landsat MSS sensors the emulator can target.

The four MSS bands and the 6-bit quantization are common to Landsat 1-5.
What changes per spacecraft is the spectral response, the IFOV and the
radiometric constants. A sensor is ``calibrated`` only when a profile with
measured numbers exists; the rest are ``pending`` and the API refuses them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorSpec:
    """What the emulator knows about one MSS sensor."""

    key: str
    spacecraft: int
    ifov_m: float
    quantization_bits: int
    bands: int
    profile: str | None
    status: str

    @property
    def calibrated(self) -> bool:
        return self.status == "calibrated" and self.profile is not None

    @property
    def available(self) -> bool:
        """True when a profile exists, even if it is only experimental."""
        return self.profile is not None and self.status in ("calibrated", "experimental")


#: The five MSS sensors. ``ifov_m`` is the nominal instantaneous field of view
#: (79 m for Landsat 1-3, 82 m for Landsat 4-5); the exact PSF is 79-82 m.
MSS_SENSORS: dict[str, SensorSpec] = {
    "L1_MSS": SensorSpec("L1_MSS", 1, 79.0, 6, 4, None, "pending"),
    "L2_MSS": SensorSpec("L2_MSS", 2, 79.0, 6, 4, None, "pending"),
    "L3_MSS": SensorSpec("L3_MSS", 3, 79.0, 6, 4, None, "pending"),
    "L4_MSS": SensorSpec("L4_MSS", 4, 82.0, 6, 4, "mss4_esa_gtc_v1", "experimental"),
    "L5_MSS": SensorSpec("L5_MSS", 5, 82.0, 6, 4, "mss5_esa_gtc_v2", "calibrated"),
}


class SensorNotCalibrated(ValueError):
    """Raised when a sensor has no measured profile yet."""


def calibrated_sensors() -> tuple[str, ...]:
    """Keys of the sensors with a measured, validated profile."""
    return tuple(key for key, spec in MSS_SENSORS.items() if spec.calibrated)


def available_sensors() -> tuple[str, ...]:
    """Keys of the sensors with a profile, experimental ones included."""
    return tuple(key for key, spec in MSS_SENSORS.items() if spec.available)


def _available_message() -> str:
    available = ", ".join(available_sensors()) or "none"
    return f"calibrated sensors: {available}"


def choose_sensor(
    rng: np.random.Generator,
    candidates: Sequence[SensorSpec] | None = None,
) -> SensorSpec:
    """Pick one sensor uniformly. The caller controls determinism via ``rng``."""
    if candidates is None:
        pool = [MSS_SENSORS[key] for key in calibrated_sensors()]
    else:
        pool = list(candidates)
    if not pool:
        raise SensorNotCalibrated("no sensor to choose from")
    return pool[int(rng.integers(len(pool)))]


def resolve_sensor(
    sensor: str,
    rng: np.random.Generator,
    candidates: Sequence[SensorSpec] | None = None,
) -> SensorSpec:
    """Turn a user string into a calibrated :class:`SensorSpec`.

    ``"random"`` samples among the candidates (default: the calibrated ones).
    Any other value must be a known, calibrated sensor key.
    """
    if sensor.lower() == "random":
        return choose_sensor(rng, candidates)

    key = sensor.upper()
    spec = MSS_SENSORS.get(key)
    if spec is None:
        raise SensorNotCalibrated(f"unknown sensor {sensor!r}; {_available_message()}")
    if not spec.available:
        raise SensorNotCalibrated(f"{key} has no calibrated profile yet; {_available_message()}")
    if candidates is not None and spec not in candidates:
        raise SensorNotCalibrated(f"{key} is not in the requested candidates")
    return spec
