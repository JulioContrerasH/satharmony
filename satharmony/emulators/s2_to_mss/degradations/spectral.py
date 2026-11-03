"""Spectral degradation: legacy band selection or SBAF adjustment."""

import numpy as np

from ..config import SpectralConfig
from ..sbaf import sbaf_transform


def apply_spectral_degradation(
    image: np.ndarray, config: SpectralConfig, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Select and adjust S2 bands to match MSS spectral response.

    Two modes are available:

    ``"legacy"`` (default)
        Select the four S2 bands in ``config.s2_bands``.  The default mapping
        is B3 (560 nm) -> MSS1 Green, B4 (665 nm) -> MSS2 Red,
        B7 (783 nm) -> MSS3 NIR1 and B8 (842 nm) -> MSS4 NIR2.  This path is
        bit-for-bit the 0.1.13/0.2.0 behaviour.

    ``"sbaf"``
        Apply the fitted S2 -> MSS spectral band adjustment factors (see
        :mod:`satharmony.emulators.s2_to_mss.sbaf`).  Each MSS band is a
        non-negative linear combination of the nine S2 VNIR bands plus an
        offset, using the official SRF of the selected sensor.

    Args:
        image: Input array (C, H, W) with S2 bands
        config: SpectralConfig with mode, sensor and band indices

    Returns:
        Array (4, H, W) with MSS-like bands
    """
    if not config.enabled:
        return image
    generator = rng if rng is not None else np.random.default_rng()

    if config.mode == "sbaf":
        selected = sbaf_transform(
            image, sensor=config.sensor, water_vapor_scale=config.water_vapor_scale
        )
    elif config.mode == "legacy":
        selected = image[config.s2_bands]
    else:
        raise ValueError(
            f"Unknown spectral mode {config.mode!r}; expected 'legacy' or 'sbaf'"
        )

    if config.gas_mss3_scale != 1.0:
        selected = selected.copy()
        selected[2] = selected[2] * config.gas_mss3_scale

    if config.srf_adjustment:
        noise_std = config.srf_noise_std.sample(generator)
        if noise_std > 0:
            noise = generator.normal(0, noise_std, selected.shape)
            selected = selected + noise * selected

    return selected.astype(np.float32)
