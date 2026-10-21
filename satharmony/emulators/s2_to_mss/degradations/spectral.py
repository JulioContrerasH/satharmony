"""Spectral degradation: S2 band selection + SRF adjustment."""

import numpy as np

from ..config import SpectralConfig


def apply_spectral_degradation(image: np.ndarray, config: SpectralConfig) -> np.ndarray:
    """Select and adjust S2 bands to match MSS spectral response.

    Default mapping (S2 -> MSS):
        B3 (560nm) -> MSS1 Green
        B4 (665nm) -> MSS2 Red
        B7 (783nm) -> MSS3 NIR1
        B8 (842nm) -> MSS4 NIR2

    Args:
        image: Input array (C, H, W) with S2 bands
        config: SpectralConfig with band indices

    Returns:
        Array (4, H, W) with MSS-like bands
    """
    if not config.enabled:
        return image

    selected = image[config.s2_bands]

    if config.srf_adjustment:
        noise_std = config.srf_noise_std.sample()
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, selected.shape)
            selected = selected + noise * selected

    return selected.astype(np.float32)
