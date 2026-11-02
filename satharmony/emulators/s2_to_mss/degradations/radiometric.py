"""Radiometric degradation: 6-bit quantization + sqrt companding + saturation."""

import numpy as np

from ....corrections.landsat.mss import saturation_reflectance
from ..config import RadiometricConfig


def _band_saturation(config: RadiometricConfig, bands: int) -> np.ndarray | float:
    """Saturation threshold per band, shaped (bands, 1, 1) for broadcasting.

    ``"physical"`` uses ``rho_sat = pi * LMAX * d^2 / (ESUN * sin(theta))``
    for the configured spacecraft and a scene sun elevation drawn from
    ``sun_elevation_deg``. ``"legacy_random"`` draws one threshold for all
    bands, as 0.1.13 did.
    """
    if config.saturation_model == "legacy_random":
        return float(config.saturation_threshold.sample())

    sun_elevation = config.sun_elevation_deg.sample()
    rho_sat = saturation_reflectance(
        spacecraft=config.satellite,
        sun_elevation=sun_elevation,
        earth_sun_distance_au=config.earth_sun_distance_au,
    )
    if bands > rho_sat.shape[0]:
        raise ValueError(
            f"radiometric saturation has {rho_sat.shape[0]} bands "
            f"({config.satellite}) but the image has {bands}"
        )
    return rho_sat[:bands].reshape(-1, 1, 1)


def apply_radiometric_degradation(image: np.ndarray, config: RadiometricConfig) -> np.ndarray:
    """Apply MSS radiometric characteristics.

    The real MSS records 6 bits after a roughly square-root companding. The
    L1 TOA product is decompanded back to linear reflectance, so a faithful
    emulator must quantize in the compressed domain and then invert the
    companding. The legacy 0.1.13 path skipped that last step.

    Saturation is physical by default: the threshold comes from LMAX, ESUN
    and the sun elevation, per band, so a cloud saturates at the same
    reflectance in the sqrt bands and in the linear band.

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

    sat = _band_saturation(config, result.shape[0])
    levels = 2**config.quantization_bits

    # ── Legacy 0.1.13 path (sqrt never inverted, floor bias kept) ───────────
    if not config.decompress_after_quantization:
        if config.sqrt_compression:
            for b in config.sqrt_bands:
                if b < result.shape[0]:
                    result[b] = np.sqrt(np.clip(result[b], 0, None))
        result = np.clip(result, 0, sat) / sat
        result = np.floor(result * (levels - 1)) / (levels - 1)
        return np.clip(result, 0, 1).astype(np.float32)

    # ── Correct path: compress -> quantize -> decompress ────────────────────
    # Saturation is applied in linear reflectance (not in sqrt space), which
    # is where Lmax lives, and it is per band.
    result = np.clip(result, 0, sat)
    compressed = result / sat

    if config.sqrt_compression:
        for b in config.sqrt_bands:
            if b < result.shape[0]:
                compressed[b] = np.sqrt(np.clip(compressed[b], 0, 1))

    # Round to nearest code instead of floor: floor biases every pixel half a
    # step down, which is what made shadows darker than the real MSS.
    quantized = np.rint(compressed * (levels - 1)) / (levels - 1)

    output = quantized * sat
    if config.sqrt_compression:
        for b in config.sqrt_bands:
            if b < result.shape[0]:
                # Inverse of sqrt: linear reflectance = (q ** 2) * Lmax
                sat_b = sat if np.ndim(sat) == 0 else sat[b]
                output[b] = quantized[b] ** 2 * sat_b

    return np.clip(output, 0, 1).astype(np.float32)
