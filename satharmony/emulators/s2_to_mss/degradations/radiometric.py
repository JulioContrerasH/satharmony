"""Radiometric degradation: sqrt compression + quantization + saturation."""

import numpy as np

from ..config import RadiometricConfig


def apply_radiometric_degradation(image: np.ndarray, config: RadiometricConfig) -> np.ndarray:
    """Apply MSS radiometric characteristics.

    MSS had:
        - 6-bit quantization (64 levels)
        - Sqrt compression for bands 1-3
        - Heavy saturation on bright targets

    Args:
        image: Input array (C, H, W) float32 [0, 1+]
        config: RadiometricConfig

    Returns:
        Degraded array (C, H, W) float32
    """
    if not config.enabled:
        return image

    result = image.copy()

    # Optional reflectance boost (increases saturation)
    if np.random.random() < config.reflectance_boost_prob:
        boost = config.reflectance_boost.sample()
        result = result * boost

    # Saturation threshold
    sat_threshold = config.saturation_threshold.sample()

    # Sqrt compression for specified bands
    if config.sqrt_compression:
        for b in config.sqrt_bands:
            if b < result.shape[0]:
                result[b] = np.sqrt(np.clip(result[b], 0, None))

    # Normalize to [0, 1] for quantization
    result = np.clip(result, 0, sat_threshold) / sat_threshold

    # Quantize to n-bit
    levels = 2**config.quantization_bits
    result = np.floor(result * (levels - 1)) / (levels - 1)

    # Mark saturated pixels (optional: keep at max)
    result = np.clip(result, 0, 1)

    return result.astype(np.float32)
