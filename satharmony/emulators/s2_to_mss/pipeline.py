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


class _LegacyRandom:
    """Adapter that keeps the 0.1.13 global RandomState sequence.

    The legacy profile promises a byte-for-byte reproduction of the published
    0.1.13 wheel, which used ``np.random.seed`` and the module-level
    functions.  Everything else uses a local ``Generator``.
    """

    def random(self) -> float:
        return float(np.random.random())

    def normal(self, *args, **kwargs):
        return np.random.normal(*args, **kwargs)

    def uniform(self, *args, **kwargs):
        return np.random.uniform(*args, **kwargs)

    def integers(self, low: int, high: int):
        return np.random.randint(low, high)

    def poisson(self, *args, **kwargs):
        return np.random.poisson(*args, **kwargs)

    def choice(self, *args, **kwargs):
        return np.random.choice(*args, **kwargs)


class MSSEmulator:
    """Pipeline to emulate Landsat MSS from Sentinel-2."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.config = config or PipelineConfig()
        self._external_rng = rng
        self._legacy = bool(self.config.legacy_rng)
        self.rng = rng

    def __call__(self, image: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
        """Apply MSS emulation pipeline.

        ``labels`` is accepted for backwards compatibility but is not used
        here: label aggregation happens once in ``emulate_s2_with_labels``
        with the same scale factor as the image. The old code aggregated the
        labels here and then discarded them.
        """
        if self._external_rng is None:
            if self._legacy:
                np.random.seed(self.config.seed)
                self.rng = _LegacyRandom()
            else:
                self.rng = np.random.default_rng(self.config.seed)
        if self.rng is None:
            raise RuntimeError("no random generator available")

        x = image.copy()

        if self._should_apply(self.config.spectral):
            x = apply_spectral_degradation(x, self.config.spectral, self.rng)

        if self._should_apply(self.config.spatial):
            x = apply_spatial_degradation(x, self.config.spatial, self.rng)

        if self.config.radiometric.decompress_after_quantization:
            # Physical order: artifacts and noise before 6-bit quantization
            if self._should_apply(self.config.striping):
                x = apply_striping(
                    x, self.config.striping, sampling_gsd=self._output_gsd, rng=self.rng
                )
            if self._should_apply(self.config.memory_effect):
                x = apply_memory_effect(x, self.config.memory_effect, self.rng)
            if self._should_apply(self.config.coherent_noise):
                x = apply_coherent_noise(x, self.config.coherent_noise, self.rng)
            if self._should_apply(self.config.random_noise):
                x = apply_random_noise(x, self.config.random_noise, self.rng)
            if self._should_apply(self.config.scan_artifacts):
                x = apply_scan_artifacts(x, self.config.scan_artifacts, self.rng)
            if self._should_apply(self.config.radiometric):
                x = apply_radiometric_degradation(x, self.config.radiometric, self.rng)
        else:
            # Legacy 0.1.13 order: quantize first, then artifacts/noise
            if self._should_apply(self.config.radiometric):
                x = apply_radiometric_degradation(x, self.config.radiometric, self.rng)
            if self._should_apply(self.config.striping):
                x = apply_striping(
                    x, self.config.striping, sampling_gsd=self._output_gsd, rng=self.rng
                )
            if self._should_apply(self.config.memory_effect):
                x = apply_memory_effect(x, self.config.memory_effect, self.rng)
            if self._should_apply(self.config.coherent_noise):
                x = apply_coherent_noise(x, self.config.coherent_noise, self.rng)
            if self._should_apply(self.config.random_noise):
                x = apply_random_noise(x, self.config.random_noise, self.rng)
            if self._should_apply(self.config.scan_artifacts):
                x = apply_scan_artifacts(x, self.config.scan_artifacts, self.rng)

        # Final linear scaling
        if self.config.final_scaling.enabled:
            scales = self.config.final_scaling.band_scaling.sample(self.rng)
            for b, scale in enumerate(scales):
                if b < x.shape[0]:
                    x[b] = x[b] * scale

        if self.config.output_dtype == "uint8":
            x = (np.clip(x, 0, 1) * 255).astype(np.uint8)

        return x

    def _should_apply(self, cfg) -> bool:
        return cfg.enabled and self.rng.random() < cfg.probability

    @property
    def _output_gsd(self) -> float | None:
        """Pixel size of the grid striping is applied on, in metres."""
        spatial = self.config.spatial
        if not spatial.enabled:
            return spatial.input_gsd
        return spatial.sampling_gsd
