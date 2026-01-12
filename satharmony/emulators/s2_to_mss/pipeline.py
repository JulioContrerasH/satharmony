"""MSS Emulator Pipeline."""

import numpy as np

from .config import PipelineConfig
from .degradations import (
    apply_coherent_noise,
    apply_memory_effect,
    apply_radiometric_degradation,
    apply_random_noise,
    apply_scan_artifacts,
    apply_spatial_degradation,
    apply_spectral_degradation,
    apply_striping,
)


class MSSEmulator:
    """Pipeline to emulate Landsat MSS from Sentinel-2."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply MSS emulation pipeline.

        Args:
            image: S2 L1C array (C, H, W) float32, reflectance [0, 1+]

        Returns:
            MSS-like array (4, H', W') float32
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        x = image.copy()

        # 1. Spectral (always)
        if self._should_apply(self.config.spectral):
            x = apply_spectral_degradation(x, self.config.spectral)

        # 2. Spatial (always)
        if self._should_apply(self.config.spatial):
            x = apply_spatial_degradation(x, self.config.spatial)

        # 3. Radiometric (always)
        if self._should_apply(self.config.radiometric):
            x = apply_radiometric_degradation(x, self.config.radiometric)

        # 4. Striping
        if self._should_apply(self.config.striping):
            x = apply_striping(x, self.config.striping)

        # 5. Memory effect
        if self._should_apply(self.config.memory_effect):
            x = apply_memory_effect(x, self.config.memory_effect)

        # 6. Coherent noise
        if self._should_apply(self.config.coherent_noise):
            x = apply_coherent_noise(x, self.config.coherent_noise)

        # 7. Random noise
        if self._should_apply(self.config.random_noise):
            x = apply_random_noise(x, self.config.random_noise)

        # 8. Scan artifacts
        if self._should_apply(self.config.scan_artifacts):
            x = apply_scan_artifacts(x, self.config.scan_artifacts)

        # Output dtype
        if self.config.output_dtype == "uint8":
            x = (np.clip(x, 0, 1) * 255).astype(np.uint8)

        return x

    def _should_apply(self, cfg) -> bool:
        """Check if degradation should be applied."""
        return cfg.enabled and np.random.random() < cfg.probability
