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
        self._labels = None  # Para pasar labels al módulo de saturación

    def __call__(self, image: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
        """Apply MSS emulation pipeline."""
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        x = image.copy()
        current_labels = labels  # Labels a resolución actual

        if self._should_apply(self.config.spectral):
            x = apply_spectral_degradation(x, self.config.spectral)

        if self._should_apply(self.config.spatial):
            x = apply_spatial_degradation(x, self.config.spatial)
            # Downsample labels también si existen
            if current_labels is not None:
                scale = image.shape[1] // x.shape[1]
                from .labels import aggregate_cloudsen12

                current_labels = aggregate_cloudsen12(
                    current_labels, scale_factor=scale,
                    merge_cloud_classes=False,
                    legacy=self.config.labels_legacy,
                )

        if self._should_apply(self.config.radiometric):
            x = apply_radiometric_degradation(x, self.config.radiometric)

        if self._should_apply(self.config.striping):
            x = apply_striping(x, self.config.striping)

        if self._should_apply(self.config.memory_effect):
            x = apply_memory_effect(x, self.config.memory_effect)

        if self._should_apply(self.config.coherent_noise):
            x = apply_coherent_noise(x, self.config.coherent_noise)

        if self._should_apply(self.config.random_noise):
            x = apply_random_noise(x, self.config.random_noise)

        if self._should_apply(self.config.scan_artifacts):
            x = apply_scan_artifacts(x, self.config.scan_artifacts)

        # Final linear scaling
        if self.config.final_scaling.enabled:
            scales = self.config.final_scaling.band_scaling.sample()
            for b, scale in enumerate(scales):
                if b < x.shape[0]:
                    x[b] = x[b] * scale

        if self.config.output_dtype == "uint8":
            x = (np.clip(x, 0, 1) * 255).astype(np.uint8)

        return x

    def _should_apply(self, cfg) -> bool:
        return cfg.enabled and np.random.random() < cfg.probability
