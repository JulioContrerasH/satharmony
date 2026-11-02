"""Spectral Band Adjustment Factors (SBAF) for Sentinel-2 -> Landsat MSS.

The coefficients are fitted in ``scripts/fit_sbaf.py`` from official SRFs and
a documented spectral library.  They are embedded in the package as
``sbaf_coefficients.json`` so that the runtime does not depend on the vault
layout or on any external file.

Model (per MSS band ``m``)::

    rho_MSS(m) = sum_k w[m, k] * rho_S2(k) + c[m] - g[m] * WV

``WV`` is the water-vapour absorption depth inside the MSS 0.8-1.1 um band::

    WV = rho_B8A * log(rho_B8A / rho_B9)

B8A (0.865 um) is the continuum next to the 0.94 um absorption and B9
(0.945 um) samples the absorption itself.  The correction is only applied
to MSS4 (the NIR2 band), whose response covers 0.8-1.1 um.  ``g[4]`` is
calibrated on the five CEOS PICS in ``scripts/validate_sbaf_pics.py``.

The linear weights are fitted with a non-negative least squares (``w >= 0``)
with a free intercept.  Predictors are the nine S2 bands whose SRF overlaps
the 400-1100 nm MSS range: B1, B2, B3, B4, B5, B6, B7, B8, B8A.  B9 is not a
surface predictor: a 10-predictor fit that used B9 directly over-predicted
the surface B9 signal and under-predicted MSS4 by 30-48 % on the PICS, so it
enters only through the water-vapour term.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

#: S2 bands in the order used by the package's 13-band image stack.
S2_BAND_ORDER: tuple[str, ...] = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B9",
    "B10",
    "B11",
    "B12",
    "B8A",
)

#: Predictors used by the linear SBAF, in the order of the coefficient columns.
PREDICTORS: tuple[str, ...] = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
)

#: Index of each predictor in the 13-band S2 stack.
PREDICTOR_INDICES: tuple[int, ...] = tuple(S2_BAND_ORDER.index(b) for b in PREDICTORS)

#: Bands that build the water-vapour feature: continuum and absorption.
WATER_VAPOR_BANDS: tuple[str, ...] = ("B8A", "B9")
WATER_VAPOR_FORM = "rho_B8A * log(rho_B8A / rho_B9)"
B8A_INDEX = S2_BAND_ORDER.index(WATER_VAPOR_BANDS[0])
B9_INDEX = S2_BAND_ORDER.index(WATER_VAPOR_BANDS[1])
#: Guard against a fill/near-zero B9 producing an unbounded correction.
MAX_WATER_VAPOR_RATIO = 10.0

SENSORS: tuple[str, ...] = ("S2A", "S2B", "S2C")
MSS_BANDS: tuple[str, ...] = ("B1", "B2", "B3", "B4")

_COEFFICIENTS_PATH = Path(__file__).with_name("sbaf_coefficients.json")


@lru_cache(maxsize=4)
def load_coefficients(path: str | Path | None = None) -> dict:
    """Load and validate the embedded SBAF coefficients.

    Parameters
    ----------
    path:
        Optional JSON path.  ``None`` loads the file shipped with the package.

    Returns
    -------
    dict
        Parsed JSON with ``sensors``, ``predictors``, ``predictor_indices``,
        ``mss_bands``, the water-vapour block and integrity metadata.
    """
    source = Path(path) if path is not None else _COEFFICIENTS_PATH
    with source.open() as fh:
        coefficients = json.load(fh)

    if tuple(coefficients["predictors"]) != PREDICTORS:
        raise ValueError(
            "SBAF predictor order does not match the package band order: "
            f"{coefficients['predictors']} != {PREDICTORS}"
        )
    if tuple(coefficients["predictor_indices"]) != PREDICTOR_INDICES:
        raise ValueError("SBAF predictor indices do not match the package band order")
    if tuple(coefficients["mss_bands"]) != MSS_BANDS:
        raise ValueError("SBAF MSS band order is not B1-B4")

    for sensor, block in coefficients["sensors"].items():
        weights = np.asarray(block["weights"], dtype=float)
        offsets = np.asarray(block["offsets"], dtype=float)
        if weights.shape != (len(MSS_BANDS), len(PREDICTORS)):
            raise ValueError(f"Bad SBAF weight shape for {sensor}: {weights.shape}")
        if offsets.shape != (len(MSS_BANDS),):
            raise ValueError(f"Bad SBAF offset shape for {sensor}: {offsets.shape}")
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(offsets)):
            raise ValueError(f"Non-finite SBAF coefficient for {sensor}")
        if np.any(weights < -1e-12):
            raise ValueError(f"Negative SBAF weight for {sensor}")

    water_vapor = coefficients.get("water_vapor")
    if water_vapor is not None:
        if tuple(water_vapor.get("predictors", ())) != WATER_VAPOR_BANDS:
            raise ValueError(
                "Water-vapour predictors must be "
                f"{WATER_VAPOR_BANDS}, got {water_vapor.get('predictors')}"
            )
        if water_vapor.get("form") != WATER_VAPOR_FORM:
            raise ValueError("Unexpected water-vapour feature form")
        for sensor in SENSORS:
            weights = np.asarray(water_vapor["weights"][sensor], dtype=float)
            if weights.shape != (len(MSS_BANDS),):
                raise ValueError(f"Bad water-vapour weight shape for {sensor}: {weights.shape}")
            if not np.all(np.isfinite(weights)):
                raise ValueError(f"Non-finite water-vapour weight for {sensor}")
            if np.any(weights < -1e-12):
                raise ValueError(f"Negative water-vapour weight for {sensor}")
    return coefficients


def sensor_weights(sensor: str, coefficients: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(weights, offsets)`` for a sensor as float32 arrays."""
    coefficients = coefficients or load_coefficients()
    try:
        block = coefficients["sensors"][sensor]
    except KeyError as exc:
        raise ValueError(f"Unknown S2 sensor {sensor!r}; expected one of {SENSORS}") from exc
    return (
        np.asarray(block["weights"], dtype=np.float32),
        np.asarray(block["offsets"], dtype=np.float32),
    )


def water_vapor_weights(sensor: str, coefficients: dict | None = None) -> np.ndarray:
    """Return the per-MSS-band water-vapour coefficients for a sensor.

    A missing block is treated as zero, so older coefficient files still load.
    """
    coefficients = coefficients or load_coefficients()
    if sensor not in SENSORS:
        raise ValueError(f"Unknown S2 sensor {sensor!r}; expected one of {SENSORS}")
    block = coefficients.get("water_vapor")
    if block is None:
        return np.zeros(len(MSS_BANDS), dtype=np.float32)
    return np.asarray(block["weights"][sensor], dtype=np.float32)


def water_vapor_feature(image: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    """Water-vapour absorption depth at 0.94 um, same shape as one S2 band.

    ``rho_B8A * log(rho_B8A / rho_B9)`` is positive when B9 is absorbed
    relative to the B8A continuum.  Pixels with a missing or non-finite B9
    get zero correction instead of a spurious large optical depth; the ratio
    is capped at :data:`MAX_WATER_VAPOR_RATIO` so a fill value cannot blow up.
    """
    if image.ndim != 3:
        raise ValueError(f"image must be (C, H, W), got shape {image.shape}")
    b8a = image[B8A_INDEX].astype(np.float32, copy=False)
    b9 = image[B9_INDEX].astype(np.float32, copy=False)
    valid = np.isfinite(b8a) & np.isfinite(b9) & (b8a > floor) & (b9 > floor)
    ratio = np.where(valid, b8a / np.maximum(b9, floor), 1.0)
    ratio = np.clip(ratio, 1.0, MAX_WATER_VAPOR_RATIO)
    return np.where(valid, b8a * np.log(ratio), 0.0).astype(np.float32, copy=False)


def sbaf_transform(
    image: np.ndarray,
    sensor: str = "S2A",
    coefficients: dict | None = None,
    clip: bool = True,
    water_vapor_scale: float | None = None,
) -> np.ndarray:
    """Apply the S2 -> MSS SBAF to a 13-band S2 image.

    Parameters
    ----------
    image:
        Array ``(C, H, W)`` with S2 bands in :data:`S2_BAND_ORDER` order.
        At least 13 bands are required.
    sensor:
        One of ``"S2A"``, ``"S2B"``, ``"S2C"``.
    coefficients:
        Optional pre-loaded coefficient dictionary.
    clip:
        Clip negative reflectances to zero.  The NNLS intercept can make a
        very dark pixel slightly negative.
    water_vapor_scale:
        ``None`` uses the calibrated water-vapour coefficient, ``0`` disables
        the correction and a positive float overrides it (used to calibrate
        the coefficient on the PICS).

    Returns
    -------
    numpy.ndarray
        Array ``(4, H, W)`` float32 with MSS-like bands B1-B4.
    """
    if image.ndim != 3:
        raise ValueError(f"image must be (C, H, W), got shape {image.shape}")
    if image.shape[0] < len(S2_BAND_ORDER):
        raise ValueError(
            f"image needs at least {len(S2_BAND_ORDER)} S2 bands, got {image.shape[0]}"
        )
    if sensor not in SENSORS:
        raise ValueError(f"Unknown S2 sensor {sensor!r}; expected one of {SENSORS}")

    weights, offsets = sensor_weights(sensor, coefficients)
    selected = image[list(PREDICTOR_INDICES)].astype(np.float32, copy=False)
    output = np.tensordot(weights, selected, axes=(1, 0))
    output = output + offsets[:, None, None]

    wv_weights = water_vapor_weights(sensor, coefficients)
    if water_vapor_scale is not None:
        wv_weights = wv_weights * np.float32(water_vapor_scale)
    if np.any(wv_weights):
        feature = water_vapor_feature(image)
        output = output - wv_weights[:, None, None] * feature[None, :, :]

    if clip:
        np.clip(output, 0.0, None, out=output)
    return output.astype(np.float32, copy=False)
