"""Noise: coherent (periodic) + random (gaussian/poisson)."""

import numpy as np

from ..config import CoherentNoiseConfig, RandomNoiseConfig


def apply_coherent_noise(image: np.ndarray, config: CoherentNoiseConfig) -> np.ndarray:
    """Apply periodic electronic noise.

    Fixed frequency interference from detector readout.
    MSS typical: 1/6, 1/3, 1/2 cycles/line.

    Args:
        image: Input array (C, H, W) float32
        config: CoherentNoiseConfig

    Returns:
        Array with coherent noise
    """
    if not config.enabled:
        return image

    C, H, W = image.shape
    amplitude = config.amplitude.sample()

    # Generate noise pattern
    x = np.arange(W)
    noise = np.zeros((H, W))

    for freq in config.frequencies:
        phase = np.random.uniform(0, 2 * np.pi) if config.random_phase else 0
        noise += np.sin(2 * np.pi * freq * x + phase)

    # Normalize and scale
    noise = noise / len(config.frequencies) * amplitude

    # Apply to all bands
    result = image + noise[np.newaxis, :, :]

    return result.astype(np.float32)


def apply_random_noise(image: np.ndarray, config: RandomNoiseConfig) -> np.ndarray:
    """Apply gaussian and/or poisson noise.

    Args:
        image: Input array (C, H, W) float32
        config: RandomNoiseConfig

    Returns:
        Array with random noise
    """
    if not config.enabled:
        return image

    snr_db = config.snr_db.sample()
    snr_linear = 10 ** (snr_db / 20)

    # Signal power (mean of image)
    signal_power = np.mean(image**2)
    noise_std = np.sqrt(signal_power) / snr_linear

    if config.noise_type == "gaussian":
        noise = np.random.normal(0, noise_std, image.shape)
        result = image + noise

    elif config.noise_type == "poisson":
        # Scale for poisson, then scale back
        scale = 1.0 / (noise_std**2 + 1e-10)
        noisy = np.random.poisson(np.clip(image * scale, 0, None))
        result = noisy / scale

    else:  # mixed
        w = config.poisson_weight.sample()

        # Gaussian component
        gaussian = np.random.normal(0, noise_std * (1 - w), image.shape)

        # Poisson component
        scale = 1.0 / ((noise_std * w) ** 2 + 1e-10)
        poisson = np.random.poisson(np.clip(image * scale, 0, None)) / scale - image

        result = image + gaussian + poisson

    return result.astype(np.float32)
