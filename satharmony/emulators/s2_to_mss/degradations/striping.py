"""Detector striping: 6-detector gain/offset mismatch."""

import numpy as np

from ..config import StripingConfig


def apply_striping(image: np.ndarray, config: StripingConfig) -> np.ndarray:
    """Apply 6-detector striping artifact.

    MSS used 6 parallel detectors per band. Each has slightly
    different gain/offset creating periodic banding.

    Args:
        image: Input array (C, H, W) float32
        config: StripingConfig

    Returns:
        Array with striping artifact
    """
    if not config.enabled:
        return image

    C, H, W = image.shape
    n_det = config.num_detectors

    # Sample gain/offset per detector
    gain_std = config.gain_std.sample()
    offset_std = config.offset_std.sample()

    result = image.copy()

    for c in range(C):
        # Per-band variation or same for all bands
        if config.per_band_variation or c == 0:
            gains = 1.0 + np.random.normal(0, gain_std, n_det)
            offsets = np.random.normal(0, offset_std, n_det)

        # Apply to each detector's lines
        for d in range(n_det):
            result[c, d::n_det, :] = result[c, d::n_det, :] * gains[d] + offsets[d]

    return result.astype(np.float32)
