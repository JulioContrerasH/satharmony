"""Units contract for real Landsat MSS imagery.

Three spaces are easy to confuse:

* ``dn``: raw digital numbers. The MSS L1 product ships 0-255 (7-bit after
  the on-board 6-bit compression is decompressed on the ground).
* ``toa``: top-of-atmosphere reflectance, dimensionless 0-1. This is what
  ``corrections.dn_to_toa`` returns, what the emulator returns, and what the
  finetune dataloader feeds the model after dividing the GeoTIFF by 10000.
* ``sr``: surface reflectance, also 0-1 but atmospherically corrected. It is
  **not** interchangeable with TOA and the emulator must not see it.

The 0.1.13 dataloader silently divided by 10000, which happens to be right
for the uint16 TOA files but would also accept raw DN or SR. These helpers
make the contract explicit and raise instead of training on the wrong space.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .corrections.landsat.metadata import MSSCalibration
from .corrections.landsat.mss import dn_to_toa

MSSUnits = Literal["dn", "toa", "sr"]

# TOA reflectance is 0-1; a unit swap (DN 0-255) blows past this bound.
TOA_MIN = -0.01
TOA_MAX = 1.5


def assert_mss_toa_units(
    array: np.ndarray,
    *,
    units: MSSUnits = "toa",
    name: str = "MSS",
) -> np.ndarray:
    """Validate that ``array`` is TOA reflectance and return it as float32.

    Call this in the real-MSS dataloader right after the ``/ 10000`` step.
    A DN array (0-255) or a surface-reflectance array is rejected.

    Args:
        array: MSS array (C, H, W) or (H, W).
        units: ``"toa"`` is accepted; ``"dn"`` and ``"sr"`` raise with the
            conversion that must be applied first.
        name: Label used in error messages.

    Returns:
        The array as float32.
    """
    if units == "sr":
        raise ValueError(
            f"{name}: surface reflectance (SR) is not TOA reflectance. "
            "The emulator is calibrated in TOA; do not feed SR."
        )
    if units == "dn":
        raise ValueError(
            f"{name}: raw DN cannot be compared with emulated TOA. "
            "Convert with corrections.dn_to_toa() first."
        )
    if units != "toa":
        raise ValueError(f"{name}: unknown units {units!r}; use 'dn', 'toa' or 'sr'.")

    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"{name}: empty array")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: non-finite values; not a reflectance array")

    lo = float(arr.min())
    hi = float(arr.max())
    if lo < TOA_MIN or hi > TOA_MAX:
        raise ValueError(
            f"{name}: values in [{lo:.3f}, {hi:.3f}] are outside TOA "
            f"reflectance [{TOA_MIN}, {TOA_MAX}]. If these are DN, divide "
            "by 10000 (uint16 TOA files) or call corrections.dn_to_toa()."
        )
    return arr


def mss_to_toa(
    array: np.ndarray,
    *,
    units: MSSUnits = "toa",
    scale: float = 1.0,
    calib: MSSCalibration | None = None,
    nodata: int = 0,
) -> np.ndarray:
    """Convert real MSS data to TOA reflectance with an explicit units tag.

    Args:
        array: (C, H, W) MSS array.
        units: ``"dn"`` (needs ``calib``), ``"toa"`` (already 0-1, optionally
            scaled) or ``"sr"`` (rejected).
        scale: Multiplier applied before validation, e.g. ``1 / 10000`` for
            the uint16 TOA files used by the finetune dataloader.
        calib: ``MSSCalibration`` parsed from the MTL; required for DN.
        nodata: DN value for nodata; only used with ``units="dn"``.

    Returns:
        TOA reflectance as float32.
    """
    if units == "dn":
        if calib is None:
            raise ValueError("units='dn' needs an MSSCalibration parsed from the MTL")
        return dn_to_toa(np.asarray(array), calib, nodata=nodata)
    if units == "sr":
        raise ValueError("surface reflectance is not accepted; the emulator is TOA")
    return assert_mss_toa_units(np.asarray(array) * scale, units="toa")


def assert_toa_units(
    array: np.ndarray,
    *,
    name: str = "TOA",
    upper: float = 2.0,
) -> np.ndarray:
    """Reject arrays that are clearly DN instead of TOA reflectance.

    Args:
        array: Reflectance array of any shape.
        name: Label used in the error message.
        upper: Highest plausible TOA value; clouds can exceed 1, DN cannot.

    Returns:
        The array unchanged.

    Raises:
        ValueError: If values fall below -0.01 or above ``upper``.
    """
    data = np.asarray(array, dtype=np.float64)
    if data.size == 0:
        raise ValueError(f"{name}: empty array")
    low, high = float(np.nanmin(data)), float(np.nanmax(data))
    if low < -0.01 or high > upper:
        raise ValueError(
            f"{name} expects TOA reflectance in [0, 1]; got [{low:.3g}, {high:.3g}] (DN?)"
        )
    return array
