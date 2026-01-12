"""Memory effect: bright target recovery artifact."""

import numpy as np

from ..config import MemoryEffectConfig


def apply_memory_effect(image: np.ndarray, config: MemoryEffectConfig) -> np.ndarray:
    """Apply detector memory effect.

    After bright pixels, detectors show exponential recovery
    creating trailing light/dark bands along scan direction.

    Args:
        image: Input array (C, H, W) float32
        config: MemoryEffectConfig

    Returns:
        Array with memory effect artifact
    """
    if not config.enabled:
        return image

    C, H, W = image.shape

    threshold = config.trigger_threshold.sample()
    decay = config.decay_pixels.sample()
    amplitude = config.amplitude.sample()

    result = image.copy()

    for c in range(C):
        # Find bright pixels that trigger effect
        bright_mask = image[c] > threshold

        if not bright_mask.any():
            continue

        # Process each row (scan line)
        for row in range(H):
            line = image[c, row]
            bright_cols = np.where(line > threshold)[0]

            if len(bright_cols) == 0:
                continue

            # Apply exponential decay after each bright region
            effect = np.zeros(W)
            for col in bright_cols:
                trigger_val = line[col] - threshold
                x = np.arange(W) - col
                decay_curve = trigger_val * amplitude * np.exp(-x / decay)
                decay_curve[x < 0] = 0
                effect = np.maximum(effect, decay_curve)

            # Alternating positive/negative (overshoot)
            result[c, row] = result[c, row] - effect * 0.5 + np.roll(effect, decay // 2) * 0.3

    return result.astype(np.float32)
