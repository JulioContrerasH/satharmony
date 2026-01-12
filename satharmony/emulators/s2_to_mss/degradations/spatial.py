"""Spatial degradation: PSF convolution + downsampling."""

import numpy as np
from scipy.ndimage import gaussian_filter

from ..config import SpatialConfig


def apply_spatial_degradation(image: np.ndarray, config: SpatialConfig) -> np.ndarray:
    """Degrade spatial resolution from S2 10m to MSS ~60-80m.

    Args:
        image: Input array (C, H, W) float32 [0, 1+]
        config: SpatialConfig with parameters

    Returns:
        Degraded array (C, H', W') where H' = H // scale_factor
    """
    if not config.enabled:
        return image

    # Sample parameters
    target_gsd = config.target_gsd.sample()
    psf_sigma = config.psf_sigma.sample()

    # Calculate downsampling factor
    scale_factor = int(round(target_gsd / config.input_gsd))

    # PSF sigma in input pixels
    sigma_pixels = psf_sigma * scale_factor

    C, H, W = image.shape

    # Output dimensions (floor division, same as labels.py)
    new_h = H // scale_factor
    new_w = W // scale_factor

    # Apply PSF (gaussian blur) per band
    blurred = np.empty_like(image)
    for c in range(C):
        blurred[c] = gaussian_filter(image[c], sigma=sigma_pixels)

    # Downsample (crop first to ensure consistent dimensions)
    if config.method == "gaussian":
        # Crop then subsample
        downsampled = blurred[
            :, : new_h * scale_factor : scale_factor, : new_w * scale_factor : scale_factor
        ]
    else:
        # Average pooling
        cropped = blurred[:, : new_h * scale_factor, : new_w * scale_factor]
        downsampled = cropped.reshape(C, new_h, scale_factor, new_w, scale_factor).mean(axis=(2, 4))

    return downsampled.astype(np.float32)
