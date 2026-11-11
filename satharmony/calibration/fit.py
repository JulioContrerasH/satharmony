"""Fit a versioned calibration profile from S2/MSS pairs or PICS."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..emulators.s2_to_mss.config import (
    BandScaleRange,
    CenteredRange,
    PipelineConfig,
    Range,
)
from ..emulators.s2_to_mss.degradations.spatial import apply_spatial_degradation
from ..emulators.s2_to_mss.functional import emulate_s2
from ..emulators.s2_to_mss.sbaf import (
    load_coefficients,
    water_vapor_feature,
)

BANDS = ("B1", "B2", "B3", "B4")


def package_gamma(coefficients: str | None = None) -> float:
    """Water-vapour weight of the coefficient file the profile actually uses.

    ``to_config`` turns the fitted gamma into a multiplier of that weight, so it
    must read the same file the spectral stage will load. Reading the shipped
    Landsat 5 file for a Landsat 2 profile inflated B4 by 15 points.
    """
    path = None
    if coefficients:
        path = Path(coefficients)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / "emulators" / "s2_to_mss" / path.name
    payload = load_coefficients(path)
    return float(payload["water_vapor"]["weights"]["S2A"][3])


@dataclass
class Pair:
    """One S2/MSS scene pair on a common grid.

    Attributes:
        s2: Sentinel-2 TOA ``(13, H, W)`` in the package band order.
        mss: Landsat MSS TOA ``(4, h, w)`` at the output grid.
        valid: Optional mask at the MSS grid.
        name: Identifier used in the per-pair report.
    """

    s2: NDArray[np.floating]
    mss: NDArray[np.floating]
    valid: NDArray[np.bool_] | None = None
    name: str = ""


@dataclass(frozen=True)
class Profile:
    """Fitted per-band scale and B4 water-vapour coefficient."""

    name: str
    band_scaling: tuple[float, ...]
    gamma: float
    n_pairs: int
    variation: tuple[float, ...] = (0.03, 0.03, 0.03, 0.03)
    per_pair: list[dict[str, Any]] = field(default_factory=list)
    intervals: dict[str, dict[str, float]] = field(default_factory=dict)
    coefficients: str | None = None
    schema_version: int = 1
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready payload with provenance fields."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created": self.created,
            "n_pairs": self.n_pairs,
            "band_scaling": {
                band: {"mean": scale, "variation": variation}
                for band, scale, variation in zip(
                    BANDS, self.band_scaling, self.variation, strict=True
                )
            },
            "gamma": self.gamma,
            "intervals": self.intervals,
            "per_pair": self.per_pair,
        }

    def to_config(self, base: PipelineConfig | None = None) -> PipelineConfig:
        """Return a pipeline config with the fitted scale and gamma."""
        config = (base or PipelineConfig.mss5_esa_gtc_v1()).model_copy(deep=True)
        config.final_scaling.band_scaling = BandScaleRange(
            green=CenteredRange(mean=self.band_scaling[0], variation=self.variation[0]),
            red=CenteredRange(mean=self.band_scaling[1], variation=self.variation[1]),
            nir1=CenteredRange(mean=self.band_scaling[2], variation=self.variation[2]),
            nir2=CenteredRange(mean=self.band_scaling[3], variation=self.variation[3]),
        )
        config.spectral.water_vapor_scale = (
            self.gamma / package_gamma(self.coefficients) if package_gamma(self.coefficients) else 0.0
        )
        return config


def _deterministic_config(base: PipelineConfig | None = None) -> PipelineConfig:
    """Base profile with every stochastic stage off."""
    config = (base or PipelineConfig.mss5_esa_gtc_v1()).model_copy(deep=True)
    config.spectral.mode = "sbaf"
    config.spatial.psf_fwhm_m = Range(min=80.0, max=80.0)
    config.radiometric.enabled = False
    config.striping.enabled = False
    config.memory_effect.enabled = False
    config.coherent_noise.enabled = False
    config.random_noise.enabled = False
    config.scan_artifacts.enabled = False
    config.final_scaling.enabled = False
    return config


def _valid_mask(mss: NDArray[np.floating], valid: NDArray[np.bool_] | None) -> NDArray[np.bool_]:
    mask = np.all(np.isfinite(mss), axis=0) & np.all(mss > 0, axis=0) & np.all(mss < 1.5, axis=0)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    return mask


def _band_scale(pred: NDArray[np.floating], obs: NDArray[np.floating], mask: NDArray[np.bool_]) -> float:
    p = pred[mask].astype(np.float64)
    y = obs[mask].astype(np.float64)
    return float(np.dot(p, y) / max(np.dot(p, p), 1e-12))


def _bootstrap_median(
    values: NDArray[np.floating],
    n_boot: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        value = float(values[0])
        return {"mean": value, "low": value, "high": value, "n": 1}
    index = rng.integers(0, values.size, size=(n_boot, values.size))
    medians = np.median(values[index], axis=1)
    alpha = 100 * (1 - level) / 2
    low, high = np.percentile(medians, [alpha, 100 - alpha])
    return {
        "mean": float(np.median(values)),
        "low": float(low),
        "high": float(high),
        "n": int(values.size),
    }


def fit_profile(
    pairs: Iterable[Pair],
    base: PipelineConfig | None = None,
    name: str = "fitted",
    method: Literal["pics", "pairs"] = "pics",
    n_boot: int = 1000,
    gamma_grid: NDArray[np.floating] | None = None,
    gamma_rule: Literal["match_scale", "min_error", "off"] = "match_scale",
    seed: int = 0,
) -> Profile:
    """Fit per-band scale and B4 gamma from aligned pairs.

    The spectral matrix is not refitted here: it comes from the SRF library
    and lives in the package coefficients.  What this function learns from the
    data is the sensor-to-sensor scale and the water-vapour term that the
    linear SBAF cannot see.

    Args:
        pairs: Aligned S2/MSS pairs.  The MSS is assumed on the emulator grid.
        base: Base profile; defaults to ``mss5_esa_gtc_v1()``.
        name: Profile name written into the JSON.
        method: ``"pics"`` or ``"pairs"`` (same fit; kept for provenance).
        n_boot: Bootstrap replicates for the per-band interval.
        gamma_grid: Candidate gamma values; defaults to 0..0.30 step 0.005.
        gamma_rule: How gamma is chosen. ``"match_scale"`` (default) forces the
            B4 gain to equal the B1-B3 gain, which assumes the sensor has one
            gain for all bands. ``"min_error"`` takes the value that minimises
            the B4 error with a single global gain. ``"off"`` keeps the gain of
            every band free and drops the water-vapour term, which is what the
            auxiliary banks need: their water-vapour feature barely varies, so
            gamma cannot be identified and fitting it would be fitting noise.
        seed: Seed for the bootstrap.

    Returns:
        The fitted profile.

    Raises:
        ValueError: If no pair has enough valid pixels.
    """
    config = _deterministic_config(base)
    # The fit adds the water-vapour term itself, so the pipeline must not.
    config.spectral.water_vapor_scale = 0.0
    grid = (
        np.arange(0.0, 0.301, 0.005)
        if gamma_grid is None
        else np.asarray(gamma_grid, dtype=float)
    )
    rng = np.random.default_rng(seed)

    per_pair: list[dict[str, Any]] = []
    scales = {band: [] for band in BANDS}
    wv_by_pair: list[NDArray[np.floating]] = []
    preds: list[NDArray[np.floating]] = []
    masks: list[NDArray[np.bool_]] = []
    mss4_values: list[NDArray[np.floating]] = []
    count = 0

    for pair in pairs:
        s2 = np.asarray(pair.s2, dtype=np.float32)
        mss = np.asarray(pair.mss, dtype=np.float32)
        pred, _ = emulate_s2(s2, config=config)
        if pred.shape != mss.shape:
            raise ValueError(
                f"{pair.name}: emulated {pred.shape} != MSS {mss.shape}; "
                "the pair must already be on the output grid"
            )
        mask = _valid_mask(mss, pair.valid)
        if mask.sum() < 10:
            raise ValueError(f"{pair.name}: only {int(mask.sum())} valid pixels")
        wv = water_vapor_feature(s2)
        wv = apply_spatial_degradation(wv[None], config.spatial)[0]
        preds.append(pred)
        wv_by_pair.append(wv)
        masks.append(mask)
        mss4_values.append(mss[3])
        count += 1

        row = {"pair": pair.name, "valid_pixels": int(mask.sum())}
        for index, band in enumerate(BANDS[:3]):
            scale = _band_scale(pred[index], mss[index], mask)
            scales[band].append(scale)
            row[band] = scale
        per_pair.append(row)

    if count == 0:
        raise ValueError("fit_profile needs at least one pair")

    # B1-B3 do not depend on gamma; their pooled median is the target.
    target = float(np.median([value for band in BANDS[:3] for value in scales[band]]))

    def gamma_scales(gamma: float) -> list[float]:
        values = []
        for pred, wv, mask, mss4 in zip(preds, wv_by_pair, masks, mss4_values, strict=True):
            band4 = pred[3] - gamma * wv
            values.append(_band_scale(band4, mss4, mask))
        return values

    def gamma_errors(gamma: float) -> list[float]:
        """Per-pair B4 error with one global gain, so gamma stays identifiable."""
        scale = float(np.median(gamma_scales(gamma)))
        errors = []
        for pred, wv, mask, mss4 in zip(preds, wv_by_pair, masks, mss4_values, strict=True):
            band4 = pred[3] - gamma * wv
            residual = scale * band4[mask] - mss4[mask]
            errors.append(float(np.sqrt(np.mean(residual**2)) / np.mean(mss4[mask])))
        return errors

    if gamma_rule == "off":
        gamma = 0.0
    elif gamma_rule == "min_error":
        # The water-vapour term is a correction, so its coefficient is the one
        # that minimises the B4 error. Forcing the B4 gain to match B1-B3 gave
        # the wrong gamma on Landsat 2, where B4 really has a different gain.
        pooled = np.array([np.median(gamma_errors(gamma)) for gamma in grid])
        gamma = float(grid[int(np.argmin(pooled))])
    else:
        medians = np.array([np.median(gamma_scales(gamma)) for gamma in grid])
        gamma = float(grid[int(np.argmin(np.abs(medians - target)))])

    b4_scales = gamma_scales(gamma)
    for row, value in zip(per_pair, b4_scales, strict=True):
        row["B4"] = value
        row["gamma"] = gamma

    scales["B4"] = b4_scales

    summary = {band: _bootstrap_median(np.array(values), n_boot, rng) for band, values in scales.items()}
    means = tuple(summary[band]["mean"] for band in BANDS)
    variations = tuple(
        float(np.clip(2.0 * (summary[band]["high"] - summary[band]["low"]) / 3.92, 0.03, 0.12))
        for band in BANDS
    )
    return Profile(
        name=name,
        band_scaling=means,
        gamma=gamma,
        n_pairs=count,
        variation=variations,
        per_pair=per_pair,
        intervals={band: summary[band] for band in BANDS},
        coefficients=config.spectral.coefficients,
    )
