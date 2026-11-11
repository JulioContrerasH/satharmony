"""MSS Emulator Configuration."""

import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

PROFILE_DIR = Path(__file__).resolve().parents[2] / "profiles"


def _rng_or_default(rng: np.random.Generator | None) -> np.random.Generator:
    """Return ``rng`` or a fresh generator when the caller did not pass one."""
    return rng if rng is not None else np.random.default_rng()


class Range(BaseModel):
    """Range for stochastic sampling.

    ``distribution="uniform"`` (default) keeps the historical behaviour.
    ``distribution="truncated_normal"`` draws from a normal distribution with
    ``mean`` and ``std`` clipped to ``[min, max]``; it is used for the sun
    elevation, whose measured distribution is not uniform.
    """

    min: float
    max: float
    distribution: Literal["uniform", "truncated_normal"] = "uniform"
    mean: float | None = None
    std: float | None = None

    def sample(self, rng: np.random.Generator | None = None) -> float:
        generator = _rng_or_default(rng)
        if self.distribution == "truncated_normal":
            center = self.mean if self.mean is not None else 0.5 * (self.min + self.max)
            spread = self.std if self.std is not None else 0.25 * (self.max - self.min)
            return float(np.clip(generator.normal(center, spread), self.min, self.max))
        return float(generator.uniform(self.min, self.max))


#: Sun-elevation distribution measured on the 178 MSS/TM calibration pairs of
#: the legacy bank. Kept for the v1 profile.
MEASURED_SUN_ELEVATION = {
    "min": 7.6,
    "max": 62.3,
    "distribution": "truncated_normal",
    "mean": 42.8,
    "std": 12.25,
}

#: Sun-elevation distribution of the canonical cloud set (2,886 of the 3,027
#: aligned scenes; 141 have no GEE/MTL metadata). Rebuilt by
#: ``scripts/measure_sun_elevation_canonical.py``.
MEASURED_SUN_ELEVATION_CANONICAL = {
    "min": 4.02,
    "max": 62.73,
    "distribution": "truncated_normal",
    "mean": 41.51,
    "std": 12.95,
}


class RangeInt(BaseModel):
    """Integer range for stochastic sampling."""

    min: int
    max: int

    def sample(self, rng: np.random.Generator | None = None) -> int:
        return int(_rng_or_default(rng).integers(self.min, self.max + 1))


class CenteredRange(BaseModel):
    """Range centered on mean with controlled variation."""

    mean: float
    variation: float

    def sample(self, rng: np.random.Generator | None = None) -> float:
        generator = _rng_or_default(rng)
        return float(self.mean + generator.uniform(-self.variation, self.variation))


class BandScaleRange(BaseModel):
    """Per-band scaling."""

    green: CenteredRange = CenteredRange(mean=0.60, variation=0.20)
    red: CenteredRange = CenteredRange(mean=0.58, variation=0.20)
    nir1: CenteredRange = CenteredRange(mean=0.60, variation=0.20)
    nir2: CenteredRange = CenteredRange(mean=0.85, variation=0.20)

    def sample(self, rng: np.random.Generator | None = None) -> list[float]:
        generator = _rng_or_default(rng)
        return [
            self.green.sample(generator),
            self.red.sample(generator),
            self.nir1.sample(generator),
            self.nir2.sample(generator),
        ]


class SpectralConfig(BaseModel):
    """S2 bands -> MSS bands mapping.

    ``mode="legacy"`` keeps the 0.1.13 band selection (B3/B4/B7/B8) and is the
    default, so existing callers do not change.  ``mode="sbaf"`` uses the
    fitted spectral band adjustment factors with the SRF of ``sensor``.
    """

    enabled: bool = True
    probability: float = 1.0
    mode: Literal["legacy", "sbaf"] = "legacy"
    sensor: Literal["S2A", "S2B", "S2C"] = "S2A"
    s2_bands: list[int] = [2, 3, 6, 7]
    srf_adjustment: bool = False
    srf_noise_std: Range = Range(min=0.0, max=0.02)
    #: Multiplier on the packaged B4 water-vapour coefficient (1.0 = package).
    water_vapor_scale: float = 1.0
    #: Optional B3 gas scale (1.0 = off; 6SV suggests about 1.054).
    gas_mss3_scale: float = 1.0


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

    ``saturation_model="physical"`` derives the per-band threshold from
    LMAX, ESUN and the scene sun elevation
    (``pi * LMAX * d^2 / (ESUN * sin(theta))``). The 0.1.13 model drew a
    single random threshold in [0.8, 0.95] shared by every band, which meant
    a different reflectance in the sqrt bands than in the linear band.
    ``"legacy_random"`` keeps that behaviour for ``legacy_0_1_13()``.
    """

    enabled: bool = True
    probability: float = 1.0
    quantization_bits: int = 6
    sqrt_compression: bool = True
    sqrt_bands: list[int] = [0, 1, 2]
    decompress_after_quantization: bool = True
    saturation_model: Literal["physical", "legacy_random"] = "physical"
    satellite: str = "L5_MSS"
    # Scene sun elevation. Measured on the 178 MSS/TM pairs used for the
    # calibration (7.64-62.33 deg, mean 42.80, std 12.25); the old 25-70
    # range was a placeholder. A truncated normal keeps the winter tail
    # without pretending the distribution is uniform.
    sun_elevation_deg: Range = Range(**MEASURED_SUN_ELEVATION)
    earth_sun_distance_au: float = 1.0
    # Legacy 0.1.13 threshold, only read with saturation_model="legacy_random".
    saturation_threshold: Range = Range(min=0.8, max=0.95)
    reflectance_boost: Range = Range(min=1.0, max=1.3)
    reflectance_boost_prob: float = 0.3


class StripingConfig(BaseModel):
    """6-detector gain/offset mismatch.

    The MSS scans six lines per mirror sweep, with the detectors spaced
    79 m along track: one period is 474 m (Handbook p. 85), i.e. 7.9 px on
    the 60 m ESA grid and not six image rows. The scan lines are tilted
    with respect to the rows because the ground track of the 99° orbit is
    not north-south (Taranik p. 36). ``detector_geometry="rows"`` keeps the
    0.1.13 row-aligned pattern for ``legacy_0_1_13()``.
    """

    enabled: bool = True
    probability: float = 0.8
    num_detectors: int = 6
    gain_std: Range = Range(min=0.01, max=0.05)
    offset_std: Range = Range(min=0.005, max=0.02)
    per_band_variation: bool = True
    detector_geometry: Literal["scan_lines", "rows"] = "scan_lines"
    sampling_gsd: float = 60.0
    scan_line_period_m: float = 474.0
    yaw_deg: float = 9.0


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
    #: Reproduce 0.1.13 bit for byte, including its global RandomState.
    legacy_rng: bool = False

    @classmethod
    def from_profile(cls, profile: str | Path) -> "PipelineConfig":
        """Load a versioned JSON profile by name or path.

        Args:
            profile: Profile name without ``.json`` (resolved inside
                ``satharmony/profiles/``) or a path to a JSON file.

        Returns:
            The validated configuration.

        Raises:
            FileNotFoundError: If neither the name nor the path exists.
        """
        path = Path(profile)
        if not path.exists():
            path = PROFILE_DIR / f"{profile}.json"
        if not path.exists():
            available = sorted(p.stem for p in PROFILE_DIR.glob("*.json"))
            raise FileNotFoundError(
                f"profile {profile!r} not found; available: {', '.join(available)}"
            )
        payload = json.loads(path.read_text())
        return cls.model_validate(payload.get("config", payload))

    @classmethod
    def mss5_esa_gtc_v1(cls) -> "PipelineConfig":
        """Perfil calibrado provisional (MSS L5, grilla ESA GTC 60 m).

        - muestreo fijo 60 m y PSF 79-82 m (IFOV MSS)
        - decimacion centrada y radiometria lineal (P0)
        - saturacion fisica conservadora, sin boost aleatorio
        - striping/ruido en rangos medidos en MSS real
        - SBAF S2A con las SRF oficiales (``mode="sbaf"``)
        - termino de vapor de agua en B4 (``rho_B8A * log(rho_B8A / rho_B9)``)
        - elevacion solar con la distribucion medida en 178 pares MSS/TM
        - ``final_scaling.band_scaling`` calibrado en 5 PICS (Libya-4,
          Algeria-3, Niger-1, Egypt-1, Arabia-2), no a ojo

        Origen de las constantes (regenerable): ``sbaf_pics_table.md`` y
        ``sbaf_pics_regression.json`` del vault, escritos por
        ``scripts/validate_sbaf_pics.py``. La tabla da la pendiente, el offset
        y el IC bootstrap por banda; el JSON es la version que se puede leer
        desde codigo.
        """
        cfg = cls()
        cfg.spatial.sampling_gsd = 60.0
        cfg.spatial.psf_fwhm_m = Range(min=79.0, max=82.0)
        cfg.spatial.center_decimation = True
        cfg.spatial.method = "average"
        cfg.radiometric.decompress_after_quantization = True
        cfg.radiometric.saturation_model = "physical"
        cfg.radiometric.satellite = "L5_MSS"
        cfg.radiometric.reflectance_boost_prob = 0.0
        cfg.radiometric.sun_elevation_deg = Range(**MEASURED_SUN_ELEVATION)
        cfg.striping.detector_geometry = "scan_lines"
        cfg.striping.gain_std = Range(min=0.005, max=0.02)
        cfg.striping.offset_std = Range(min=0.002, max=0.008)
        cfg.random_noise.snr_db = Range(min=30.0, max=40.0)
        # Calibrated on five CEOS PICS (15 km, cloud-free S2A L1C paired with
        # L5 MSS L1TP at the same sun height): Libya-4 (CEOS), Algeria-3,
        # Niger-1, Egypt-1 and Arabia-2. The mean is the median per-band scale
        # <SBAF, MSS> / <SBAF, SBAF>; the variation is 2x the site-to-site
        # standard deviation, floored at 0.03. Egypt-1 is the low outlier in
        # all four bands (surface/calibration); it stays in the table but the
        # median is used.
        #
        # B4 used to land at 0.820 because the linear predictors stop at
        # 0.88 um and cannot see the 0.94 um water-vapour absorption inside
        # the MSS 0.8-1.1 um band. The SBAF now subtracts
        # gamma * rho_B8A * log(rho_B8A / rho_B9) from B4 (gamma=0.112,
        # calibrated on the same PICS). That cuts the B4 TOA bias from
        # 17-28 % to 4-15 % and lifts the B4 scale to 0.933, in line with
        # B2 (0.933): the remaining offset is sensor calibration, not water
        # vapour.
        cfg.final_scaling.band_scaling = BandScaleRange(
            green=CenteredRange(mean=0.956, variation=0.044),
            red=CenteredRange(mean=0.933, variation=0.045),
            nir1=CenteredRange(mean=0.867, variation=0.050),
            nir2=CenteredRange(mean=0.933, variation=0.072),
        )
        cfg.spectral.mode = "sbaf"
        cfg.spectral.sensor = "S2A"
        cfg.labels_legacy = False
        return cfg

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
        cfg.radiometric.saturation_model = "legacy_random"
        # Unused by legacy_random, but keep the 0.1.13 field value so the
        # legacy config is unchanged end to end.
        cfg.radiometric.sun_elevation_deg = Range(min=25.0, max=70.0)
        cfg.striping.detector_geometry = "rows"
        cfg.labels_legacy = True
        cfg.legacy_rng = True
        return cfg
