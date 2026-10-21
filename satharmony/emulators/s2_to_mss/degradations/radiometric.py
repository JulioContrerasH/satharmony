"""Radiometric degradation: 6-bit quantization + sqrt companding + saturation."""

import numpy as np

from ..config import RadiometricConfig


def apply_radiometric_degradation(image: np.ndarray, config: RadiometricConfig) -> np.ndarray:
    """Apply MSS radiometric characteristics.

    The real MSS records 6 bits after a roughly square-root companding. The
    L1 TOA product is decompanded back to linear reflectance, so a faithful
    emulator must quantize in the compressed domain and then invert the
    companding. The legacy 0.1.13 path skipped that last step.

    Args:
        image: Input array (C, H, W) float32 [0, 1+]
        config: RadiometricConfig

    Returns:
        Degraded array (C, H, W) float32, linear reflectance.
    """
    if not config.enabled:
        return image

    result = image.copy()

    # Optional reflectance boost (increases saturation)
    if np.random.random() < config.reflectance_boost_prob:
        boost = config.reflectance_boost.sample()
        result = result * boost

    sat_threshold = config.saturation_threshold.sample()
    levels = 2**config.quantization_bits

    # ── Legacy 0.1.13 path (sqrt never inverted) ────────────────────────────
    if not config.decompress_after_quantization:
        if config.sqrt_compression:
            for b in config.sqrt_bands:
                if b < result.shape[0]:
                    result[b] = np.sqrt(np.clip(result[b], 0, None))
        result = np.clip(result, 0, sat_threshold) / sat_threshold
        result = np.floor(result * (levels - 1)) / (levels - 1)
        return np.clip(result, 0, 1).astype(np.float32)

    # ── Correct path: compress -> quantize -> decompress ────────────────────
    result = np.clip(result, 0, sat_threshold)
    compressed = result / sat_threshold

    if config.sqrt_compression:
        for b in config.sqrt_bands:
            if b < result.shape[0]:
                compressed[b] = np.sqrt(np.clip(compressed[b], 0, 1))

    quantized = np.floor(compressed * (levels - 1)) / (levels - 1)

    output = quantized * sat_threshold
    if config.sqrt_compression:
        for b in config.sqrt_bands:
            if b < result.shape[0]:
                output[b] = quantized[b] ** 2 * sat_threshold

    return np.clip(output, 0, 1).astype(np.float32)
