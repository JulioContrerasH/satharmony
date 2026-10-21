"""MSS radiometric corrections: DN → Radiance → TOA Reflectance."""

import numpy as np
from numpy.typing import NDArray

from .constants import ESUN_MSS, earth_sun_distance
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
