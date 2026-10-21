"""MSS Emulator Configuration."""

from typing import Literal

import numpy as np
from pydantic import BaseModel


class Range(BaseModel):
    """Uniform range for stochastic sampling."""

    min: float
    max: float

    def sample(self) -> float:
        return np.random.uniform(self.min, self.max)


class RangeInt(BaseModel):
    """Integer range for stochastic sampling."""

    min: int
    max: int

    def sample(self) -> int:
        return np.random.randint(self.min, self.max + 1)


class CenteredRange(BaseModel):
    """Range centered on mean with controlled variation."""

    mean: float
    variation: float

    def sample(self) -> float:
        return self.mean + np.random.uniform(-self.variation, self.variation)


class BandScaleRange(BaseModel):
    """Per-band scaling."""

    green: CenteredRange = CenteredRange(mean=0.60, variation=0.20)
    red: CenteredRange = CenteredRange(mean=0.58, variation=0.20)
    nir1: CenteredRange = CenteredRange(mean=0.60, variation=0.20)
    nir2: CenteredRange = CenteredRange(mean=0.85, variation=0.20)

    def sample(self) -> list[float]:
        return [
            self.green.sample(),
            self.red.sample(),
            self.nir1.sample(),
            self.nir2.sample(),
        ]


class SpectralConfig(BaseModel):
    """S2 bands -> MSS bands mapping."""

    enabled: bool = True
    probability: float = 1.0
    s2_bands: list[int] = [2, 3, 6, 7]
    srf_adjustment: bool = False
    srf_noise_std: Range = Range(min=0.0, max=0.02)


class SpatialConfig(BaseModel):
    """PSF + downsampling.

    ``sampling_gsd`` is the output grid (60 m for the ESA GTC product).
    ``psf_fwhm_m`` is the sensor IFOV (79-82 m for MSS), independent of the
    sampling grid. Set both to ``None`` to reproduce the 0.1.13 behaviour,
    which used ``target_gsd`` and ``psf_sigma`` coupled to the scale factor.
    """

    enabled: bool = True
    probability: float = 1.0
    input_gsd: float = 10.0
    sampling_gsd: float | None = 60.0
    psf_fwhm_m: Range | None = Range(min=79.0, max=82.0)
    center_decimation: bool = True
    method: Literal["average", "gaussian"] = "average"
    # Legacy 0.1.13 fields (used when sampling_gsd/psf_fwhm_m are None).
    target_gsd: Range = Range(min=60.0, max=80.0)
    psf_sigma: Range = Range(min=0.4, max=0.6)


class RadiometricConfig(BaseModel):
    """Quantization 6-bit + sqrt compression + saturation.

    With ``decompress_after_quantization=True`` the sqrt companding is
    inverted after the 6-bit quantization, so the output stays linear in
    reflectance (as the MSS L1 TOA product is).
    """

    enabled: bool = True
    probability: float = 1.0
    quantization_bits: int = 6
    sqrt_compression: bool = True
    sqrt_bands: list[int] = [0, 1, 2]
    decompress_after_quantization: bool = True
    saturation_threshold: Range = Range(min=0.8, max=0.95)
    reflectance_boost: Range = Range(min=1.0, max=1.3)
    reflectance_boost_prob: float = 0.3


class StripingConfig(BaseModel):
    """6-detector gain/offset mismatch."""

    enabled: bool = True
    probability: float = 0.8
    num_detectors: int = 6
    gain_std: Range = Range(min=0.01, max=0.05)
    offset_std: Range = Range(min=0.005, max=0.02)
    per_band_variation: bool = True


class MemoryEffectConfig(BaseModel):
    """Bright target recovery artifact."""

    enabled: bool = True
    probability: float = 0.4
    trigger_threshold: Range = Range(min=0.6, max=0.85)
    decay_pixels: RangeInt = RangeInt(min=50, max=200)
    amplitude: Range = Range(min=0.02, max=0.08)


class CoherentNoiseConfig(BaseModel):
    """Periodic electronic noise."""

    enabled: bool = True
    probability: float = 0.3
    frequencies: list[float] = [1 / 6, 1 / 3, 1 / 2]
    amplitude: Range = Range(min=0.005, max=0.02)
    random_phase: bool = True


class RandomNoiseConfig(BaseModel):
    """Gaussian + shot noise."""

    enabled: bool = True
    probability: float = 0.7
    snr_db: Range = Range(min=25.0, max=45.0)
    noise_type: Literal["gaussian", "poisson", "mixed"] = "mixed"
    poisson_weight: Range = Range(min=0.2, max=0.5)


class ScanArtifactsConfig(BaseModel):
    """Line dropouts and transmission errors."""

    enabled: bool = True
    probability: float = 0.2
    dropout_prob: Range = Range(min=0.001, max=0.01)
    max_consecutive_lines: RangeInt = RangeInt(min=1, max=3)


class FinalScalingConfig(BaseModel):
    """Linear scaling with variation for training robustness."""

    enabled: bool = True
    band_scaling: BandScaleRange = BandScaleRange()


class PipelineConfig(BaseModel):
    """Main config combining all degradations."""

    spectral: SpectralConfig = SpectralConfig()
    spatial: SpatialConfig = SpatialConfig()
    radiometric: RadiometricConfig = RadiometricConfig()
    striping: StripingConfig = StripingConfig()
    memory_effect: MemoryEffectConfig = MemoryEffectConfig()
    coherent_noise: CoherentNoiseConfig = CoherentNoiseConfig()
    random_noise: RandomNoiseConfig = RandomNoiseConfig()
    scan_artifacts: ScanArtifactsConfig = ScanArtifactsConfig()
    final_scaling: FinalScalingConfig = FinalScalingConfig()
    labels_legacy: bool = False

    seed: int | None = None
    output_dtype: Literal["float32", "uint8"] = "float32"

    @classmethod
    def legacy_0_1_13(cls) -> "PipelineConfig":
        """Return the configuration used to train the 0.1.13 model.

        Reproduces the old (buggy) radiometry, non-centred decimation,
        scale-coupled PSF and the old CloudSEN12 label rule.
        """
        cfg = cls()
        cfg.spatial.method = "gaussian"
        cfg.spatial.center_decimation = False
        cfg.spatial.sampling_gsd = None
        cfg.spatial.psf_fwhm_m = None
        cfg.spatial.target_gsd = Range(min=60.0, max=80.0)
        cfg.spatial.psf_sigma = Range(min=0.4, max=0.6)
        cfg.radiometric.decompress_after_quantization = False
        cfg.labels_legacy = True
        return cfg
