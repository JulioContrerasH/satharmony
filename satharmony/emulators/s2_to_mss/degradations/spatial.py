"""Spatial degradation: PSF convolution + downsampling."""

import numpy as np
from scipy.ndimage import gaussian_filter

from ..config import SpatialConfig

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # 0.4247


def apply_spatial_degradation(image: np.ndarray, config: SpatialConfig) -> np.ndarray:
    """Degrade spatial resolution from S2 10m to the MSS grid.

    The output grid (``sampling_gsd``, 60 m for ESA GTC) and the sensor PSF
    (``psf_fwhm_m``, ~79-82 m for MSS) are independent. The image is blurred
    with the PSF and then averaged over each output cell, so the sample is
    centred on the cell and matches the label aggregation.

    Args:
        image: Input array (C, H, W) float32 [0, 1+]
        config: SpatialConfig with parameters

    Returns:
        Degraded array (C, H', W') where H' = H // scale_factor
    """
    if not config.enabled:
        return image

    # Output grid and scale factor
    if config.sampling_gsd is not None:
        target_gsd = config.sampling_gsd
        ratio = target_gsd / config.input_gsd
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(
                f"sampling_gsd={target_gsd} m is not an integer multiple of "
                f"input_gsd={config.input_gsd} m; the output grid would be "
                f"{round(ratio) * config.input_gsd} m, not {target_gsd} m"
            )
    else:
        target_gsd = config.target_gsd.sample()
    scale_factor = max(1, round(target_gsd / config.input_gsd))

    # PSF in input pixels
    if config.psf_fwhm_m is not None:
        fwhm_m = config.psf_fwhm_m.sample()
        sigma_pixels = fwhm_m * FWHM_TO_SIGMA / config.input_gsd
    else:
        # Legacy 0.1.13: sigma expressed in units of the scale factor
        sigma_pixels = config.psf_sigma.sample() * scale_factor

    C, H, W = image.shape
    new_h = H // scale_factor
    new_w = W // scale_factor

    blurred = np.empty_like(image)
    for c in range(C):
        blurred[c] = gaussian_filter(image[c], sigma=sigma_pixels)

    if config.center_decimation or config.method == "average":
        # Average pool after the PSF: sample centred on each output cell
        cropped = blurred[:, : new_h * scale_factor, : new_w * scale_factor]
        downsampled = cropped.reshape(
            C, new_h, scale_factor, new_w, scale_factor
        ).mean(axis=(2, 4))
    else:
        # Legacy 0.1.13: point sampling at the top-left corner of each block
        downsampled = blurred[
            :, : new_h * scale_factor : scale_factor, : new_w * scale_factor : scale_factor
        ]

    return downsampled.astype(np.float32)
