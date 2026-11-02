"""MSS radiometric corrections: DN → Radiance → TOA Reflectance."""

import numpy as np
from numpy.typing import NDArray

from .constants import ESUN_MSS, LMAX_MSS, earth_sun_distance
from .metadata import MSSCalibration


def dn_to_radiance(
    dn: NDArray[np.uint8],
    calib: MSSCalibration,
    nodata: int = 0,
) -> NDArray[np.float32]:
    """
    Convert Digital Numbers to at-sensor radiance.

    Formula: L_λ = M_L * DN + A_L

    Args:
        dn: Array (C, H, W) with DN values [0-255]
        calib: MSSCalibration from MTL file
        nodata: DN value for nodata pixels (default: 0)

    Returns:
        Radiance in W/(m² sr μm), with nodata pixels set to 0
    """
    calib.require_full_calibration()
    dn = dn.astype(np.float32)

    # Create nodata mask (pixels where ALL bands are nodata)
    nodata_mask = np.all(dn == nodata, axis=0)

    radiance = np.zeros_like(dn)

    for band in range(dn.shape[0]):
        radiance[band] = calib.radiance_mult[band] * dn[band] + calib.radiance_add[band]

    # Restore nodata
    radiance[:, nodata_mask] = 0

    return radiance


def radiance_to_toa(
    radiance: NDArray[np.float32],
    calib: MSSCalibration,
    nodata: float = 0.0,
) -> NDArray[np.float32]:
    """
    Convert radiance to TOA reflectance.

    Formula: p_λ = (π * L_λ * d²) / (ESUN_λ * sin(θ_SE))

    Args:
        radiance: Array (C, H, W) in W/(m² sr μm)
        calib: MSSCalibration from MTL file
        nodata: Radiance value for nodata pixels (default: 0.0)

    Returns:
        TOA reflectance [0-1+], with nodata pixels set to 0
    """
    # Get ESUN for this spacecraft
    esun = ESUN_MSS[calib.spacecraft_short]

    # Earth-Sun distance correction
    d = earth_sun_distance(calib.doy)

    # Solar zenith angle correction
    sun_elev_rad = np.deg2rad(calib.sun_elevation)

    # Create nodata mask
    nodata_mask = np.all(radiance == nodata, axis=0)

    toa = np.zeros_like(radiance)
    for band in range(radiance.shape[0]):
        toa[band] = (np.pi * radiance[band] * d**2) / (esun[band] * np.sin(sun_elev_rad))

    # Restore nodata
    toa[:, nodata_mask] = 0

    return toa


def saturation_reflectance(
    spacecraft: str = "L5_MSS",
    sun_elevation: float = 45.0,
    earth_sun_distance_au: float = 1.0,
) -> NDArray[np.float32]:
    """TOA reflectance at which the MSS product reaches its top code.

    Uses the same equation as ``radiance_to_toa`` but with LMAX instead of
    the scene radiance::

        rho_sat = pi * LMAX * d^2 / (ESUN * sin(theta))

    The result depends on the band (LMAX and ESUN differ) and on the scene
    sun elevation, which is what the 0.1.13 emulator ignored by drawing a
    single random threshold in [0.8, 0.95] for every band.

    Note:
        LMAX is the rescaling range of the product, so ``rho_sat`` is the
        top of the dynamic range (DN 255), not the detector clip at the raw
        6-bit count 63 (USGS DFCB p. 18). The emulator uses it as the upper
        bound of the product; the true detector clip is lower and varies
        between the six detectors.

    Args:
        spacecraft: ``L1_MSS`` ... ``L5_MSS``.
        sun_elevation: Sun elevation above the horizon, degrees (0, 90].
        earth_sun_distance_au: Earth-Sun distance in AU (1.0 = mean).

    Returns:
        (4,) float32 array with the saturation reflectance per MSS band.
    """
    if spacecraft not in LMAX_MSS:
        raise ValueError(
            f"unknown spacecraft {spacecraft!r}; expected one of {sorted(LMAX_MSS)}"
        )
    if not 0.0 < sun_elevation <= 90.0:
        raise ValueError(f"sun_elevation must be in (0, 90], got {sun_elevation}")
    if earth_sun_distance_au <= 0.0:
        raise ValueError(
            f"earth_sun_distance_au must be positive, got {earth_sun_distance_au}"
        )

    lmax = np.asarray(LMAX_MSS[spacecraft], dtype=np.float64)
    esun = np.asarray(ESUN_MSS[spacecraft], dtype=np.float64)
    sin_sun = np.sin(np.deg2rad(sun_elevation))
    rho = np.pi * lmax * earth_sun_distance_au**2 / (esun * sin_sun)
    return rho.astype(np.float32)


def dn_to_toa(
    dn: NDArray[np.uint8],
    calib: MSSCalibration,
    nodata: int = 0,
) -> NDArray[np.float32]:
    """
    Convert DN directly to TOA reflectance (wrapper).

    Args:
        dn: Array (C, H, W) with DN values
        calib: MSSCalibration from MTL file
        nodata: DN value for nodata pixels (default: 0)

    Returns:
        TOA reflectance (C, H, W) float32, with nodata preserved as 0
    """
    radiance = dn_to_radiance(dn, calib, nodata=nodata)
    return radiance_to_toa(radiance, calib, nodata=0.0)
