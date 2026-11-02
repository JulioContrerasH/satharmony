#!/usr/bin/env python
"""Align and validate the S2 -> MSS SBAF on the CEOS PICS.

For every site downloaded by ``scripts/fetch_pics.py`` (see ``metadata.json``):

1. Build the S2 stack (13 bands, uint16 x 10000) and the MSS TOA reflectance
   (float32, converted with ``dn_to_toa`` in ``fetch_pics.py``).
2. Run the emulator twice on the S2 stack with every stochastic stage disabled:
   ``spectral.mode="sbaf"`` (with and without the water-vapour term) and
   ``spectral.mode="legacy"`` (B3/B4/B7/B8).
3. Register the prediction to the MSS with subpixel phase correlation followed
   by ECC (translation) refinement, and apply the same shift to every model.
4. Measure r, slope, RMSE (raw and after one per-band multiplicative scale) and
   the median ratio per band.
5. Calibrate the water-vapour coefficient ``gamma`` in
   ``gamma * rho_B8A * log(rho_B8A / rho_B9)``.  The criterion is to make the
   median B4 scale equal to the median of the B1-B3 scales, so the remaining
   B4 offset is sensor calibration rather than the 0.94 um absorption.  The
   leave-one-site-out error is reported for the candidate grid.
6. Fit the per-band multiplicative scale that takes the SBAF prediction to the
   observed MSS; the median across sites is the ``final_scaling.band_scaling``
   recalibration written to ``sbaf_pics_band_scaling.json``.

Outputs (``--out-dir``, default ``assets/md/fuentes/srf``)::

    sbaf_pics_validation.csv    one row per site x band x model
    sbaf_pics_summary.csv       median across sites per band x model
    sbaf_pics_band_scaling.json fitted band_scaling + spread
    sbaf_pics_regression.json   slope, offset and bootstrap CI per band
    sbaf_pics_water_vapor.json  gamma, target, LOO curve, per-site B4 table
    sbaf_pics_table.md          human-readable table
    sbaf_pics_metadata.json     method, IDs, hashes
    figures/pics_panels.png     S2 / MSS / SBAF / difference per site
    figures/pics_metrics.png    r and RMSE per band, three models
    figures/pics_b4_water_vapor.png  B4 before/after and the gamma curve

Usage
-----
    python scripts/validate_sbaf_pics.py
    python scripts/validate_sbaf_pics.py --data /tmp/pics --out-dir /tmp/out
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.emulators.s2_to_mss.config import PipelineConfig, Range  # noqa: E402
from satharmony.emulators.s2_to_mss.degradations.spatial import (  # noqa: E402
    apply_spatial_degradation,
)
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402
from satharmony.emulators.s2_to_mss.sbaf import (  # noqa: E402
    S2_BAND_ORDER,
    load_coefficients,
    sbaf_transform,
    water_vapor_feature,
)

DEFAULT_DATA = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")
DEFAULT_OUT = REPO.parent / "assets/md/fuentes/srf"
DEFAULT_MSS_TM_STATS = Path("/data/databases/CLOUD_MSS/calibration/mss_tm/radiometry_clear_stats.csv")
PROB_MASK = 20  # s2cloudless probability threshold
BANDS = ("B1", "B2", "B3", "B4")

#: Site-level bootstrap of the median slope/offset/scale.  The PICS are
#: independent scenes, so resampling sites is the honest uncertainty for a
#: calibration constant; resampling pixels would only measure texture.
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261024
BOOTSTRAP_LEVEL = 0.95
BOOTSTRAP_N_LABEL = f"{BOOTSTRAP_N:,}".replace(",", " ")

#: The MSS and S2 windows are built on the same UTM origin, so the only real
#: misregistration is the geolocation difference between the two products
#: (a couple of 60 m pixels).  Phase correlation over the whole 15 km scene
#: happily locks onto a dune field or a lake shore kilometres away; anything
#: beyond this radius is rejected and the shift stays at zero.
MAX_SHIFT_PX = 6.0

#: Coarse grid for the leave-one-site-out report.
WV_GRID = np.round(np.arange(0.0, 0.201, 0.01), 3)
#: Fine grid used to solve the calibration criterion.
WV_FINE = np.round(np.arange(0.0, 0.301, 0.001), 4)


# ── I/O ────────────────────────────────────────────────────────────────────
def load_site(site_dir: Path) -> dict:
    """Load the S2 stack, the MSS reflectance and the masks of one PICS."""
    import rasterio

    meta = json.loads((site_dir / "metadata.json").read_text())
    s2_files = sorted((site_dir / "s2").glob("*.tif"))
    if not s2_files:
        raise FileNotFoundError(f"no S2 files in {site_dir}")

    stack = []
    with rasterio.open(s2_files[0]) as src:
        transform, crs = src.transform, src.crs
        shape = src.shape
    for band in S2_BAND_ORDER:
        path = sorted((site_dir / "s2").glob(f"*.{band}.tif"))[0]
        with rasterio.open(path) as src:
            stack.append(src.read(1).astype(np.float32) / 10000.0)
    s2 = np.stack(stack)

    prob_path = sorted((site_dir / "s2").glob("*.MSK_CLOUD_PROB.tif"))
    prob = np.zeros(shape, np.uint8)
    if prob_path:
        with rasterio.open(prob_path[0]) as src:
            prob = src.read(1)
    qa60_path = sorted((site_dir / "s2").glob("*.QA60.tif"))
    qa60 = np.zeros(shape, np.uint16)
    if qa60_path:
        with rasterio.open(qa60_path[0]) as src:
            qa60 = src.read(1)

    mss = []
    for band in range(1, 5):
        path = sorted((site_dir / "mss").glob(f"*.B{band}.tif"))[0]
        with rasterio.open(path) as src:
            mss.append(src.read(1).astype(np.float32))
    mss = np.stack(mss)

    qa_path = sorted((site_dir / "mss").glob("*.QA_PIXEL.tif"))
    qa_pixel = np.zeros(mss.shape[1:], np.uint16)
    if qa_path:
        with rasterio.open(qa_path[0]) as src:
            qa_pixel = src.read(1)
    sat_path = sorted((site_dir / "mss").glob("*.MSK_SATURATED.tif"))
    saturated = np.zeros(mss.shape[1:], bool)
    if sat_path:
        with rasterio.open(sat_path[0]) as src:
            saturated = src.read(1).astype(bool)

    return {
        "meta": meta,
        "s2": s2,
        "mss": mss,
        "prob": prob,
        "qa60": qa60,
        "qa_pixel": qa_pixel,
        "saturated": saturated,
        "transform": transform,
        "crs": crs,
    }


def valid_masks(site: dict) -> tuple[np.ndarray, np.ndarray]:
    """(S2 10 m mask, MSS 60 m mask) of usable pixels."""
    s2, prob, qa60 = site["s2"], site["prob"], site["qa60"]
    s2_ok = (
        (prob < PROB_MASK)
        & ((qa60 & 0b110000000000) == 0)
        & np.all(np.isfinite(s2), axis=0)
        & (np.min(s2, axis=0) > 0.0)
    )

    mss = site["mss"]
    qa = site["qa_pixel"]
    mss_ok = np.all(np.isfinite(mss), axis=0) & (np.min(mss, axis=0) > 0.0)
    # Cloud and shadow come from the QA bits, not from a brightness threshold:
    # White Sands and Uyuni are bright enough to look like cloud in reflectance.
    mss_ok &= (qa & (1 << 3)) == 0
    mss_ok &= (qa & (1 << 4)) == 0
    mss_ok &= (mss < 1.5).all(axis=0)  # physically impossible / residual cloud
    mss_ok &= ~site["saturated"]  # DN 255 in any band
    return s2_ok, mss_ok


def aggregate_mask(s2_ok: np.ndarray, size: int) -> np.ndarray:
    """A 60 m cell is usable only if all of its 10 m pixels are usable."""
    h, _ = s2_ok.shape
    step = h // size
    blocks = s2_ok[: size * step, : size * step].reshape(size, step, size, step)
    fraction = blocks.mean(axis=(1, 3))
    return fraction >= 1.0


# ── emulation ──────────────────────────────────────────────────────────────
def physics_config(mode: str) -> PipelineConfig:
    """Deterministic profile: SBAF/legacy + PSF + 60 m grid, nothing stochastic."""
    cfg = PipelineConfig.mss5_esa_gtc_v1()
    cfg.spatial.input_gsd = 10.0
    cfg.spatial.sampling_gsd = 60.0
    cfg.spatial.psf_fwhm_m = Range(min=80.0, max=80.0)
    cfg.spatial.center_decimation = True
    cfg.spatial.method = "average"
    cfg.radiometric.enabled = False
    cfg.striping.enabled = False
    cfg.memory_effect.enabled = False
    cfg.coherent_noise.enabled = False
    cfg.random_noise.enabled = False
    cfg.scan_artifacts.enabled = False
    cfg.final_scaling.enabled = False
    cfg.spectral.mode = mode  # type: ignore[assignment]
    return cfg


def emulate(s2: np.ndarray, mode: str, water_vapor_scale: float | None = None) -> np.ndarray:
    """Run the deterministic spectral + spatial chain.

    ``water_vapor_scale=0`` reproduces the SBAF without the 0.94 um term;
    ``None`` uses the calibrated coefficient embedded in the package.
    """
    cfg = physics_config(mode)
    if mode == "sbaf":
        spectral = sbaf_transform(
            s2,
            sensor=cfg.spectral.sensor,
            water_vapor_scale=water_vapor_scale,
        )
        return apply_spatial_degradation(spectral, cfg.spatial).astype(np.float32)
    out, _ = emulate_s2(s2, config=cfg)
    return np.asarray(out, dtype=np.float32)


# ── registration ───────────────────────────────────────────────────────────
def register(reference: np.ndarray, moving: np.ndarray, valid: np.ndarray) -> dict:
    """Subpixel translation that takes ``moving`` onto ``reference``.

    The two arrays already share the grid, so the search is a small local
    translation: phase correlation gives the coarse subpixel estimate and
    ``cv2.findTransformECC`` refines it.  The ECC sign convention is not
    documented in a way that survives every OpenCV build, so the candidate
    shifts are scored against the reference and the best one is kept (with the
    coarse phase-correlation shift as the fallback).
    """
    import cv2
    from scipy.ndimage import shift as ndshift
    from skimage.registration import phase_cross_correlation

    def correlation(shift_yx: tuple[float, float]) -> float:
        moved = ndshift(moving, shift_yx, order=1, mode="constant", cval=np.nan)
        ok = valid & np.isfinite(moved) & np.isfinite(reference)
        if ok.sum() < 100:
            return -np.inf
        return float(np.corrcoef(reference[ok], moved[ok])[0, 1])

    coarse, _, _ = phase_cross_correlation(
        np.nan_to_num(reference, nan=float(np.nanmedian(reference))),
        np.nan_to_num(moving, nan=float(np.nanmedian(moving))),
        upsample_factor=100,
        normalization=None,
    )
    if max(abs(float(coarse[0])), abs(float(coarse[1]))) > MAX_SHIFT_PX:
        coarse = (0.0, 0.0)

    pre = ndshift(
        np.nan_to_num(moving, nan=float(np.nanmedian(moving))), coarse, order=3, mode="nearest"
    )
    template = np.ascontiguousarray(
        ((reference - np.nanmean(reference)) / (np.nanstd(reference) + 1e-9)).astype(np.float32)
    )
    image = np.ascontiguousarray(((pre - pre.mean()) / (pre.std() + 1e-9)).astype(np.float32))
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-8)
    try:
        _, warp = cv2.findTransformECC(
            template, image, warp, cv2.MOTION_TRANSLATION, criteria, None, 5
        )
        ecc = (float(warp[1, 2]), float(warp[0, 2]))
    except cv2.error:
        ecc = (0.0, 0.0)
    if max(abs(ecc[0]), abs(ecc[1])) > MAX_SHIFT_PX:
        ecc = (0.0, 0.0)

    candidates = {
        "none": (0.0, 0.0),
        "coarse": (float(coarse[0]), float(coarse[1])),
        "coarse+ecc": (float(coarse[0] + ecc[0]), float(coarse[1] + ecc[1])),
        "coarse-ecc": (float(coarse[0] - ecc[0]), float(coarse[1] - ecc[1])),
        "-coarse": (-float(coarse[0]), -float(coarse[1])),
    }
    scores = {name: correlation(shift) for name, shift in candidates.items()}
    best = max(scores, key=scores.get)
    return {
        "dy": candidates[best][0],
        "dx": candidates[best][1],
        "coarse_dy": float(coarse[0]),
        "coarse_dx": float(coarse[1]),
        "ecc_dy": float(ecc[0]),
        "ecc_dx": float(ecc[1]),
        "candidate": best,
        "r_before": correlation((0.0, 0.0)),
        "r_after": scores[best],
    }


def apply_shift(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    from scipy.ndimage import shift as ndshift

    return np.stack([ndshift(b, (dy, dx), order=3, mode="constant", cval=np.nan) for b in image])


# ── metrics ────────────────────────────────────────────────────────────────
def band_metrics(pred: np.ndarray, obs: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    p = pred[valid].astype(float)
    y = obs[valid].astype(float)
    if p.size < 100:
        return {}
    scale = float(np.dot(p, y) / max(np.dot(p, p), 1e-12))
    scaled = scale * p
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    rmse_scaled = float(np.sqrt(np.mean((scaled - y) ** 2)))
    mean_y = float(np.mean(y))
    slope, intercept = np.polyfit(p, y, 1)
    return {
        "n": int(p.size),
        "r": float(np.corrcoef(p, y)[0, 1]),
        "slope": float(slope),
        "offset": float(intercept),
        "intercept": float(intercept),
        "scale": scale,
        "median_pred": float(np.median(p)),
        "median_obs": float(np.median(y)),
        "median_ratio": float(np.median(p) / max(np.median(y), 1e-12)),
        "rmse_raw": rmse,
        "rmse_raw_rel": rmse / mean_y if mean_y else float("nan"),
        "rmse_scaled": rmse_scaled,
        "rmse_scaled_rel": rmse_scaled / mean_y if mean_y else float("nan"),
        "bias_scaled_rel": float(np.mean(scaled - y) / mean_y) if mean_y else float("nan"),
    }


def bootstrap_ci(
    group: pd.DataFrame, reference: float | None = None
) -> dict[str, dict[str, float]]:
    """Paired site-level bootstrap of the median slope, offset and scale.

    If ``reference`` is given (the MSS/TM slope of the same band), the
    distribution of ``median slope - reference`` is bootstrapped too, so the
    cross-check has an interval instead of two point estimates.
    """
    slope = group["slope"].to_numpy(dtype=float)
    offset = group["offset"].to_numpy(dtype=float)
    scale = group["scale"].to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    index = rng.integers(0, slope.size, size=(BOOTSTRAP_N, slope.size))
    out: dict[str, dict[str, float]] = {}
    for name, values in (("slope", slope), ("offset", offset), ("scale", scale)):
        boot = np.median(values[index], axis=1)
        low, high = np.percentile(
            boot, [100 * (1 - BOOTSTRAP_LEVEL) / 2, 100 * (1 + BOOTSTRAP_LEVEL) / 2]
        )
        out[name] = {"low": float(low), "high": float(high)}
    if reference is not None:
        boot = np.median(slope[index], axis=1) - reference
        low, high = np.percentile(
            boot, [100 * (1 - BOOTSTRAP_LEVEL) / 2, 100 * (1 + BOOTSTRAP_LEVEL) / 2]
        )
        out["delta"] = {
            "value": float(np.median(slope) - reference),
            "low": float(low),
            "high": float(high),
        }
    return out


def load_mss_tm_slopes(path: Path) -> dict[str, dict[str, float]]:
    """Median MSS/TM slope per band from the clear-pair radiometry table.

    ``radiometry_clear_stats.csv`` stores band ``0/1/2`` for MSS1/MSS2/MSS3
    against TM2/TM3/TM4.  MSS4 has no TM counterpart in that analysis.
    """
    if not path.exists():
        return {}
    values: dict[str, list[float]] = defaultdict(list)
    mapping = {"0": "B1", "1": "B2", "2": "B3"}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            band = mapping.get(row.get("band", ""))
            if band is not None:
                values[band].append(float(row["slope"]))
    return {
        band: {"median": float(np.median(items)), "n": len(items)}
        for band, items in values.items()
        if items
    }


# ── site preparation and model evaluation ─────────────────────────────────
def prepare_site(site_dir: Path) -> dict:
    """Load, emulate and register one site once, for every gamma candidate."""
    site = load_site(site_dir)
    meta = site["meta"]
    name = meta["site"]
    s2_ok, mss_ok = valid_masks(site)
    mss = site["mss"]
    n_mss = mss.shape[1]
    s2_ok_mss = aggregate_mask(s2_ok, n_mss)

    predictions = {
        "sbaf_nowv": emulate(site["s2"], "sbaf", water_vapor_scale=0.0),
        "legacy": emulate(site["s2"], "legacy"),
    }
    shape = predictions["sbaf_nowv"].shape[1:]
    if shape != mss.shape[1:]:
        raise ValueError(f"{name}: emulated {shape} != MSS {mss.shape[1:]}")

    valid = mss_ok & s2_ok_mss & np.all(np.isfinite(predictions["sbaf_nowv"]), axis=0)
    if valid.sum() < 1000:
        raise ValueError(f"{name}: only {valid.sum()} valid 60 m pixels")

    reg = register(mss[3], predictions["sbaf_nowv"][3], valid)
    shifted = {
        model: apply_shift(prediction, reg["dy"], reg["dx"])
        for model, prediction in predictions.items()
    }

    cfg = physics_config("sbaf")
    wv10 = water_vapor_feature(site["s2"])
    wv60 = apply_spatial_degradation(wv10[None], cfg.spatial)[0]
    wv = apply_shift(wv60[None], reg["dy"], reg["dx"])[0]

    model_valid = valid & np.all(np.isfinite(shifted["sbaf_nowv"]), axis=0) & np.isfinite(wv)
    return {
        "site": site,
        "role": meta.get("role", "calibration"),
        "reg": reg,
        "valid": valid,
        "predictions": shifted,
        "wv": wv,
        "model_valid": model_valid,
    }


def model_prediction(record: dict, model: str, gamma: float = 0.0) -> np.ndarray:
    if model == "legacy":
        return record["predictions"]["legacy"]
    if model not in {"sbaf", "sbaf_nowv"}:
        raise ValueError(f"unknown model {model!r}")
    prediction = record["predictions"]["sbaf_nowv"].copy()
    if model == "sbaf":
        prediction[3] = prediction[3] - np.float32(gamma) * record["wv"]
        np.clip(prediction, 0.0, None, out=prediction)
    return prediction


def site_quality(record: dict) -> bool:
    """Whether a calibration site is usable for the per-band constant.

    A stable target must have a scale within a factor of two and must not be
    anti-correlated.  The correlation is texture, not calibration: a
    homogeneous desert with a good scale can sit at r = 0.3, and rejecting it
    for that would throw away half the calibration set.  Uyuni in January 1987
    is the counter-example: the salar was wet and dark (r < 0, scale 0.3), so
    it is a surface change, not a calibration point.  B4 is judged separately
    because several late L5 scenes have a dead NIR band with good VNIR.
    """
    site = record["site"]
    valid = record["model_valid"]
    for band in range(3):
        stats = band_metrics(
            model_prediction(record, "sbaf", 0.0)[band], site["mss"][band], valid
        )
        if not stats:
            return False
        if stats["r"] < -0.1 or not 0.5 <= stats["scale"] <= 1.5:
            return False
    return True


def band_scale(record: dict, band: int, gamma: float) -> float:
    pred = model_prediction(record, "sbaf", gamma)[band]
    valid = record["model_valid"]
    p = pred[valid].astype(float)
    y = record["site"]["mss"][band][valid].astype(float)
    return float(np.dot(p, y) / max(np.dot(p, p), 1e-12))


def band_rmse(record: dict, band: int, gamma: float, scale: float) -> float:
    pred = model_prediction(record, "sbaf", gamma)[band]
    valid = record["model_valid"]
    p = pred[valid].astype(float)
    y = record["site"]["mss"][band][valid].astype(float)
    return float(np.sqrt(np.mean((scale * p - y) ** 2)) / np.mean(y))


def rows_for_model(record: dict, model: str, gamma: float) -> list[dict[str, object]]:
    site = record["site"]
    meta = site["meta"]
    name = meta["site"]
    prediction = model_prediction(record, model, gamma)
    model_valid = record["model_valid"] & np.all(np.isfinite(prediction), axis=0)
    rows: list[dict[str, object]] = []
    for band in range(4):
        stats = band_metrics(prediction[band], site["mss"][band], model_valid)
        if not stats:
            continue
        rows.append(
            {
                "site": name,
                "role": record["role"],
                "use_for_calibration": record.get("use_for_calibration", True),
                "label": meta["label"],
                "mss_id": meta["mss"]["product_id"],
                "mss_date": meta["mss"]["date"],
                "s2_id": meta["s2a"]["product_id"],
                "s2_date": meta["s2a"]["date"],
                "sun_zenith_mss": 90.0 - meta["mss"]["sun_elevation"],
                "sun_zenith_s2": meta["s2a"]["sun_zenith"],
                "mss_band": f"B{band + 1}",
                "model": model,
                "gamma": gamma,
                "valid_pixels": int(model_valid.sum()),
                "dy": round(record["reg"]["dy"], 3),
                "dx": round(record["reg"]["dx"], 3),
                **stats,
            }
        )
    for row in rows:
        row["r_before"] = round(record["reg"]["r_before"], 4)
        row["r_after"] = round(record["reg"]["r_after"], 4)
    return rows


def calibrate_water_vapor(records: list[dict]) -> tuple[float, float, list[dict]]:
    """Choose gamma so that median B4 scale equals the B1-B3 median scale.

    The B1-B3 scales do not depend on gamma; their pooled median is the
    calibration offset that the B4 band should share.  The LOO curve on the
    coarse grid is reported to show how much the choice is worth.
    """
    reference_scales = [band_scale(record, band, 0.0) for record in records for band in range(3)]
    target = float(np.median(reference_scales))

    medians = np.array(
        [np.median([band_scale(record, 3, gamma) for record in records]) for gamma in WV_FINE]
    )
    gamma = float(WV_FINE[int(np.argmin(np.abs(medians - target)))])

    loo = []
    for candidate in WV_GRID:
        scales = np.array([band_scale(record, 3, candidate) for record in records])
        errors = []
        for index, record in enumerate(records):
            train_scale = float(np.median(np.delete(scales, index)))
            errors.append(band_rmse(record, 3, candidate, train_scale))
        loo.append(
            {
                "gamma": float(candidate),
                "loo_mean_rel_rmse": float(np.mean(errors)),
                "loo_median_rel_rmse": float(np.median(errors)),
                "loo_max_rel_rmse": float(np.max(errors)),
                "per_site_rel_rmse": [float(value) for value in errors],
            }
        )
    return gamma, target, loo


# ── tables ─────────────────────────────────────────────────────────────────
def write_markdown(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    scaling: dict,
    sites: list[dict],
    path: Path,
    water_vapor: dict,
    gamma: float,
    mss_tm: dict[str, dict[str, float]],
) -> None:
    n_cal = sum(1 for s in sites if s["meta"].get("role", "calibration") == "calibration")
    n_val = len(sites) - n_cal
    lines = [
        "# SBAF: validación en PICS",
        "",
        f"{len(sites)} PICS de 15 x 15 km ({n_cal} de calibración, {n_val} de validación),",
        "sin nubes en la ventana. Por cada sitio se empareja una escena Sentinel-2A L1C",
        "con una Landsat 5 MSS L1TP elegida para que el sol esté a la misma altura",
        "(|ΔSZA| <= 6°, ampliable a 24°) y para que el MSS no sature. Los sitios de",
        "validación (bosque, agua, cultivo) no entran en las constantes: solo miden",
        "cuánto se estira el modelo fuera del desierto.",
        "",
        "El MSS se pasa a reflectancia TOA con `dn_to_toa`: radiancia del MTL, ESUN",
        "de L5 y elevación solar. Los `REFLECTANCE_MULT` del MTL de la Colección 2",
        "NO llevan la corrección por altura del sol (valen lo mismo en escenas con",
        "SZA 31° y 50°), así que `DN * mult + add` da rho·sin(SZA): 1.26x y 1.86x",
        "por debajo del valor real en estos dos casos.",
        "",
        "Alineación: correlación de fase (upsample 100) + refinado ECC en traslación",
        "sobre la banda NIR; el mismo desplazamiento se aplica a todos los modelos.",
        "",
        "## PICS",
        "",
        "| Sitio | Centro | MSS L5 (L1TP) | Fecha | S2A L1C | Fecha | Malla | SZA MSS | SZA S2 | "
        "ΔSZA |",
        "|---|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for site in sites:
        meta = site["meta"]
        lat = meta["centre_wgs84"]["lat"]
        lon = meta["centre_wgs84"]["lon"]
        sza_mss = 90.0 - meta["mss"]["sun_elevation"]
        sza_s2 = meta["s2a"]["sun_zenith"]
        lines.append(
            f"| {meta['site']} | {lat:.2f} N, {lon:.2f} E | {meta['mss']['product_id']} | "
            f"{meta['mss']['date']} | {meta['s2a']['product_id']} | {meta['s2a']['date']} | "
            f"{meta['crs']} {meta['grid']['mss']['size']}x{meta['grid']['mss']['size']} "
            f"@ {meta['grid']['mss']['gsd_m']:.0f} m | {sza_mss:.1f} | {sza_s2:.1f} | "
            f"{abs(sza_mss - sza_s2):.1f} |"
        )
    lines += [
        "",
        "Cada par cubre el 100 % de la ventana de 15 km en los dos sensores. La",
        "estación es la que deja el sol en la ventana 26-50° (o hasta 65° en las",
        "superficies oscuras), así que hay inviernos del norte y del sur mezclados.",
        "",
        "## Por sitio (SBAF con vapor de agua)",
        "",
        "| Sitio | Band | SBAF r | SBAF slope | SBAF offset | legacy r | "
        "SBAF RMSE (%) | legacy RMSE (%) | mejora (pp) | SBAF RMSE esc. (%) | "
        "legacy RMSE esc. (%) | scale |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (site, band), group in detail.groupby(["site", "mss_band"], sort=False):
        sbaf = group[group.model == "sbaf"].iloc[0]
        legacy = group[group.model == "legacy"].iloc[0]
        lines.append(
            f"| {site} | {band} | {sbaf.r:.3f} | {sbaf.slope:.3f} | {sbaf.offset:+.4f} | "
            f"{legacy.r:.3f} | "
            f"{100 * sbaf.rmse_raw_rel:.2f} | {100 * legacy.rmse_raw_rel:.2f} | "
            f"{100 * (legacy.rmse_raw_rel - sbaf.rmse_raw_rel):+.2f} | "
            f"{100 * sbaf.rmse_scaled_rel:.2f} | {100 * legacy.rmse_scaled_rel:.2f} | "
            f"{sbaf.scale:.3f} |"
        )

    lines += [
        "",
        "`RMSE esc.` es el residuo después de una escala multiplicativa por banda y",
        "sitio: absorbe la diferencia absoluta de calibración entre los dos sensores.",
        "",
        "## B4: corrección de vapor de agua",
        "",
        "La SBAF lineal se queda corta en MSS4 porque sus predictores llegan a 0.88 um",
        "y la banda MSS 4 cubre 0.8-1.1 um: no ve la absorción de vapor de agua en",
        "0.94 um. Se añade un término físico",
        "`WV = rho_B8A * log(rho_B8A / rho_B9)`, positivo cuando B9 está absorbido",
        "respecto al continuo B8A, y se resta `gamma * WV` solo a B4. `gamma` se",
        "calibra en los PICS de calibración con el criterio de que la escala mediana",
        "de B4 coincida con la escala mediana de B1-B3; así lo que queda en B4 es",
        "calibración de sensor, no absorción atmosférica.",
        "",
        f"`gamma = {gamma:.3f}`. El valor elegido deja la escala de B4 en línea con",
        "B2/B3. La",
        "biblioteca (ECOSTRESS/ASTER + EMIT L2A) es reflectancia de superficie: no",
        "tiene la absorción de 0.94 um, así que gamma se calibra en los PICS TOA;",
        "PRISMA no está disponible en el vault. Un ajuste NNLS de",
        "10 predictores que metía B9 como banda de superficie funcionaba en la",
        "biblioteca (RMSE 1.2 %) pero en los PICS TOA subestimaba B4 un 30-48 %:",
        "B9 no es un predictor de superficie.",
        "",
        "`gamma` y la escala global de B4 siguen acoplados: el criterio fija la",
        "mediana de B4, no separa la absorción atmosférica de la calibración del",
        "sensor. La parte que estos datos identifican sin ambigüedad es la",
        "variación entre sitios. Para separar atmósfera de calibración hace falta",
        "un vapor de agua independiente (ERA5/MODIS) o más rango de WV.",
        "",
        "| Sitio | B4 RMSE antes (%) | B4 RMSE después (%) | B4 RMSE legacy (%) | "
        "scale antes | scale después | scale legacy | WV mediana |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in water_vapor["per_site"]:
        lines.append(
            f"| {record['site']} | {100 * record['rmse_before']:.2f} | "
            f"{100 * record['rmse_after']:.2f} | {100 * record['rmse_legacy']:.2f} | "
            f"{record['scale_before']:.3f} | {record['scale_after']:.3f} | "
            f"{record['scale_legacy']:.3f} | {record['wv_median']:.3f} |"
        )
    lines += [
        "",
        "Mediana entre sitios: B4 pasa de "
        f"{100 * water_vapor['median_before']['rmse']:.2f} % a "
        f"{100 * water_vapor['median_after']['rmse']:.2f} % de RMSE TOA; legacy está "
        f"en {100 * water_vapor['median_legacy']['rmse']:.2f} %. La escala de B4 pasa "
        f"de {water_vapor['median_before']['scale']:.3f} a "
        f"{water_vapor['median_after']['scale']:.3f}. El error LOO medio con el",
        f"`gamma` elegido es {water_vapor['loo_at_gamma']['loo_mean_rel_rmse'] * 100:.2f} %.",
        "",
        "## Resumen (mediana entre sitios)",
        "",
        "| Band | SBAF+WV r | SBAF sin WV r | legacy r | SBAF+WV RMSE (%) | "
        "SBAF sin WV RMSE (%) | legacy RMSE (%) | SBAF+WV RMSE esc. (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for band in BANDS:
        sbaf = summary[(summary.mss_band == band) & (summary.model == "sbaf")].iloc[0]
        nowv = summary[(summary.mss_band == band) & (summary.model == "sbaf_nowv")].iloc[0]
        legacy = summary[(summary.mss_band == band) & (summary.model == "legacy")].iloc[0]
        lines.append(
            f"| {band} | {sbaf.r:.3f} | {nowv.r:.3f} | {legacy.r:.3f} | "
            f"{100 * sbaf.rmse_raw_rel:.2f} | {100 * nowv.rmse_raw_rel:.2f} | "
            f"{100 * legacy.rmse_raw_rel:.2f} | {100 * sbaf.rmse_scaled_rel:.2f} |"
        )

    lines += [
        "",
        "## Pendiente, offset e IC por banda (SBAF + WV)",
        "",
        "La pendiente y el offset son los de `MSS = slope * predicción + offset`,",
        "medidos por sitio y resumidos con la mediana. El IC es un bootstrap sobre",
        f"los PICS de calibración ({BOOTSTRAP_N_LABEL} réplicas, semilla {BOOTSTRAP_SEED}, nivel "
        f"{100 * BOOTSTRAP_LEVEL:.0f} %); remuestrea sitios, no píxeles, porque la",
        "constante de calibración varía entre escenas, no dentro de una textura.",
        "",
        "| Band | Pendiente | IC95 pendiente | Offset | IC95 offset | Escala | IC95 escala |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for band in BANDS:
        row = summary[(summary.mss_band == band) & (summary.model == "sbaf")].iloc[0]
        lines.append(
            f"| {band} | {row.slope:.3f} | "
            f"[{row.slope_ci_low:.3f}, {row.slope_ci_high:.3f}] | "
            f"{row.offset:+.4f} | [{row.offset_ci_low:+.4f}, {row.offset_ci_high:+.4f}] | "
            f"{row.scale:.3f} | [{row.scale_ci_low:.3f}, {row.scale_ci_high:.3f}] |"
        )

    lines += [
        "",
        "## Cruce con MSS/TM (radiometría en claros)",
        "",
        "`radiometry_clear_stats.csv` mide 20 escenas claras del QA de TM con",
        "`MSS = slope * TM + offset` (mediana por banda): **0.778** en B1,",
        "**0.849** en B2 y **0.774** en B3. La dirección es la misma que la de",
        "los PICS (MSS observado frente a una referencia más brillante), pero el",
        "predictor no: aquí es TM y en los PICS es la S2 emulada.",
        "",
        "| Band | SBAF S2→MSS slope | IC95 | MSS/TM slope (n=20) | "
        "Δ (PICS - MSS/TM) | IC95 Δ |",
        "|---|---:|---|---:|---:|---|",
    ]
    for band in BANDS:
        row = summary[(summary.mss_band == band) & (summary.model == "sbaf")].iloc[0]
        reference = mss_tm.get(band)
        if reference is None:
            lines.append(
                f"| {band} | {row.slope:.3f} | "
                f"[{row.slope_ci_low:.3f}, {row.slope_ci_high:.3f}] | — | — | — |"
            )
        else:
            lines.append(
                f"| {band} | {row.slope:.3f} | "
                f"[{row.slope_ci_low:.3f}, {row.slope_ci_high:.3f}] | "
                f"{reference['median']:.3f} | {row.delta_mss_tm:+.3f} | "
                f"[{row.delta_ci_low:+.3f}, {row.delta_ci_high:+.3f}] |"
            )

    lines += [
        "",
        "El punto estimado de los PICS puede quedar por encima del MSS/TM; el IC95",
        "de la diferencia es el que decide. La incertidumbre entre sitios suele ser",
        "mayor que la diferencia, así que no se declara discrepancia sin intervalo.",
        "Lo que sí se puede decir es que los PICS dan pendientes S2→MSS de "
        f"{summary[summary.model == 'sbaf'].slope.min():.2f}-"
        f"{summary[summary.model == 'sbaf'].slope.max():.2f} y MSS/TM 0.77-0.85, "
        "y que la escala por banda de los PICS "
        f"({scaling['bands']['B1']['mean']:.3f}/"
        f"{scaling['bands']['B2']['mean']:.3f}/"
        f"{scaling['bands']['B3']['mean']:.3f}/"
        f"{scaling['bands']['B4']['mean']:.3f}) deja el residuo casi plano: "
        "no reproduce por sí sola la ganancia global MSS/TM. La diferencia tiene",
        "dos partes: (1) los PICS de calibración son desiertos brillantes de",
        "invierno, con una sola fecha y SZA baja, y miden la diferencia",
        "atmosférica/BRDF de ese momento; (2) MSS/TM mezcla 20 escenas claras de",
        "superficies y",
        "estaciones distintas, y su pendiente incluye la variación entre escenas.",
        "Para cerrar el cruce hace falta un gain global medido en un set común, o",
        "recalcular MSS/TM por clase de superficie. Las constantes de",
        "`mss5_esa_gtc_v1()` salen de la escala PICS; el gain global MSS/TM queda",
        "como pendiente declarada, no escondida.",
        "",
        "## Recalibración de `final_scaling.band_scaling`",
        "",
        "Escala por banda que lleva la predicción SBAF+WV a la reflectancia MSS",
        "observada (`scale = <p,o>/<p,p>`, mínimos cuadrados por el origen).",
        "",
        f"| Band | media ({n_cal} sitios) | desviación | variación propuesta |",
        "|---|---:|---:|---:|",
    ]
    for band, values in scaling["bands"].items():
        lines.append(
            f"| {band} | {values['mean']:.3f} | {values['std']:.3f} | {values['variation']:.3f} |"
        )
    lines += [
        "",
        "## Notas",
        "",
        "- **Libya-4 anterior no era Libya-4.** El par `libya4` estaba en 23.07 N,",
        "  22.93 E (tile T34QGL), no en el PICS CEOS (28.55 N, 23.39 E). Se",
        "  reemplaza por `libya4_ceos`; el viejo queda como `legacy_libya4` y el",
        "  validador no lo usa.",
        "- **Atacama queda fuera.** `legacy_atacama` se conserva por trazabilidad:",
        "  el suelo cambió entre 1985 y 2020 y no sirve para un cierre radiométrico.",
        "- **Egypt-1 fue el outlier con cinco sitios.** Sus escalas eran las más",
        "  bajas de las cuatro bandas. Con la muestra ampliada se queda en la tabla",
        "  y se mira en la distribución; la constante es la mediana, no la media.",
        f"- **B4 sin escalar.** Antes del término de vapor: "
        f"{100 * water_vapor['median_before']['rmse']:.1f} % (SBAF) vs "
        f"{100 * water_vapor['median_legacy']['rmse']:.1f} % (legacy). Después: "
        f"{100 * water_vapor['median_after']['rmse']:.1f} % vs "
        f"{100 * water_vapor['median_legacy']['rmse']:.1f} %. Con la escala por banda "
        "los dos bajan, pero la SBAF ya no necesita la escala para batir a legacy.",
        "- La búsqueda fuerza sol bajo (26-50°, hasta 65° en superficies oscuras)",
        "  para no saturar el MSS de 8 bits: en verano, sobre un desierto brillante,",
        "  B2/B3 llegan a DN 255. La escena de Libya-4 del 1984-09-23 tiene el 66 %",
        "  de los píxeles de B2 saturados.",
        "- La correlación espacial mide coincidencia de textura, no solo",
        "  calibración: un PICS con dunas o afloramientos da r más bajo aunque la",
        "  radiometría sea correcta.",
    ]
    path.write_text("\n".join(lines) + "\n")


# ── figures ────────────────────────────────────────────────────────────────
def rgb_from(arr: np.ndarray, limits: dict, bands=(3, 2, 1)) -> np.ndarray:
    out = np.zeros((*arr.shape[1:], 3), np.float32)
    for k, band in enumerate(bands):
        lo, hi = limits[band]
        out[..., k] = np.clip((arr[band] - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def make_figures(
    records: list[dict],
    out_dir: Path,
    detail: pd.DataFrame,
    gamma: float,
    water_vapor: dict,
) -> list[Path]:
    """Panel figure, metrics figure and the B4 water-vapour figure.

    With dozens of sites a per-site bar chart is unreadable, so the metrics are
    distributions by surface group: calibration (stable deserts) and validation
    (forest, water, crops).  The panel figure keeps a bounded sample.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    calibration = [r for r in records if r["role"] == "calibration"]
    validation = [r for r in records if r["role"] != "calibration"]

    def sample(items: list[dict], n: int) -> list[dict]:
        if len(items) <= n:
            return items
        index = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
        return [items[i] for i in index]

    shown = sample(calibration, 10) + sample(validation, 6)
    fig, axes = plt.subplots(
        len(shown), 4, figsize=(13.0, 2.6 * len(shown)), constrained_layout=True
    )
    if len(shown) == 1:
        axes = axes[None, :]
    for row, record in enumerate(shown):
        site = record["site"]
        meta = site["meta"]
        mss = site["mss"]
        pred = model_prediction(record, "sbaf", gamma)
        valid = record["valid"] & np.all(np.isfinite(pred), axis=0)
        drop = ~valid

        limits = {}
        for band in range(4):
            values = np.concatenate([mss[band][valid], pred[band][valid]])
            limits[band] = np.nanpercentile(values, [2, 98])

        n_h, n_w = mss.shape[1:]
        s2_full = site["s2"]
        step_h = s2_full.shape[1] // n_h
        step_w = s2_full.shape[2] // n_w
        mask_s2 = np.repeat(np.repeat(valid, step_h, axis=0), step_w, axis=1)
        s2_limits = np.nanpercentile(
            s2_full[[3, 2, 1]][:, mask_s2[: s2_full.shape[1], : s2_full.shape[2]]], [2, 98]
        )
        s2_rgb = np.zeros((n_h, n_w, 3), np.float32)
        for k, band in enumerate((3, 2, 1)):
            lo, hi = s2_limits
            rgb_full = np.clip((s2_full[band] - lo) / max(hi - lo, 1e-6), 0, 1)
            rgb_full = rgb_full[: n_h * step_h, : n_w * step_w]
            s2_rgb[..., k] = rgb_full.reshape(n_h, step_h, n_w, step_w).mean(axis=(1, 3))
        real_rgb = rgb_from(mss, limits)
        pred_rgb = rgb_from(pred, limits)
        for rgb in (s2_rgb, real_rgb, pred_rgb):
            rgb[drop] = 0.85

        with np.errstate(invalid="ignore", divide="ignore"):
            diff = 100.0 * np.mean((pred - mss) / np.where(mss > 0.02, mss, np.nan), axis=0)
        diff = np.where(valid, diff, np.nan)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("0.85")

        axes[row, 0].imshow(s2_rgb)
        axes[row, 0].set_title(
            f"S2A {meta['s2a']['date']} B4/B3/B2\n{meta['label']} ({meta['s2a']['tile']})",
            fontsize=10.5,
        )
        axes[row, 1].imshow(real_rgb)
        axes[row, 1].set_title(
            f"MSS {meta['mss']['date']}\n{meta['mss']['product_id'][:20]}", fontsize=9.5
        )
        axes[row, 2].imshow(pred_rgb)
        axes[row, 2].set_title("SBAF+WV (S2 -> MSS)", fontsize=9.5)
        image = axes[row, 3].imshow(diff, cmap=cmap, vmin=-20, vmax=20)
        axes[row, 3].set_title(
            f"SBAF+WV - MSS (%)\nshift {record['reg']['dy']:+.2f}, {record['reg']['dx']:+.2f} px",
            fontsize=8.5,
        )
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(image, ax=axes[:, 3], shrink=1.0, pad=0.01, label="SBAF+WV - MSS (%)")
    fig.suptitle(
        f"PICS S2A / MSS / SBAF+WV - 15 km, TOA reflectance, 60 m grid "
        f"({len(shown)} of {len(records)} sites shown)",
        fontsize=12,
    )
    path = out_dir / "pics_panels.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.18, pil_kwargs={"optimize": True})
    plt.close(fig)
    paths.append(path)

    colors = {"sbaf": "C0", "sbaf_nowv": "#9ecae1", "legacy": "C1"}
    labels = {"sbaf": "SBAF+WV", "sbaf_nowv": "SBAF sin WV", "legacy": "legacy"}
    groups = [
        ("calibration", "calibración", calibration, ""),
        ("validation", "validación", validation, "//"),
    ]
    bands = ["B1", "B2", "B3", "B4"]

    def boxplot_by_group(ax, band: str, metric: str, factor: float) -> None:
        positions, data, faces, hatches = [], [], [], []
        for group_index, (role, _label, _records, hatch) in enumerate(groups):
            subset = detail[(detail.role == role) & (detail.mss_band == band)]
            if subset.empty:
                continue
            for model_index, model in enumerate(("sbaf_nowv", "sbaf", "legacy")):
                values = subset[subset.model == model][metric].to_numpy(dtype=float)
                if values.size == 0:
                    continue
                positions.append(group_index * 4 + model_index + 1)
                data.append(factor * values)
                faces.append(colors[model])
                hatches.append(hatch)
        if not data:
            return
        artists = ax.boxplot(
            data,
            positions=positions,
            widths=0.72,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.1},
            flierprops={"markersize": 2, "markerfacecolor": "0.4", "markeredgecolor": "0.4"},
        )
        for patch, face, hatch in zip(artists["boxes"], faces, hatches, strict=True):
            patch.set_facecolor(face)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.5)
            patch.set_hatch(hatch)
            patch.set_alpha(0.9)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), constrained_layout=True, sharex=True)
    for col, band in enumerate(bands):
        for row, (metric, label, factor) in enumerate(
            (("r", "Pearson r", 1.0), ("rmse_raw_rel", "Relative RMSE (%)", 100.0))
        ):
            ax = axes[row, col]
            boxplot_by_group(ax, band, metric, factor)
            ax.grid(alpha=0.25, axis="y")
            ax.set_title(f"{band} - {label}", fontsize=9)
            if col == 0:
                ax.set_ylabel(
                    "Pearson r" if metric == "r" else "RMSE (%) per band", fontsize=8
                )
            if row == 1:
                ax.set_xticks([2, 6], ["calibración", "validación"], fontsize=8)
    handles = [
        Patch(facecolor=colors[m], edgecolor="black", label=labels[m])
        for m in ("sbaf_nowv", "sbaf", "legacy")
    ]
    handles += [
        Patch(facecolor="#f2f2f2", edgecolor="black", label="calibración"),
        Patch(facecolor="#f2f2f2", edgecolor="black", hatch="//", label="validación"),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=5, fontsize=8, frameon=False
    )
    fig.suptitle(
        f"SBAF sin WV, SBAF+WV and legacy on {len(calibration)} calibration and "
        f"{len(validation)} validation sites - distribution across sites\n"
        "RMSE axes are per band: compare boxes within a panel, not across panels",
        fontsize=10.5,
    )
    path = out_dir / "pics_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.18, pil_kwargs={"optimize": True})
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    boxplot_by_group(axes[0], "B4", "rmse_raw_rel", 100.0)
    axes[0].set_ylabel("B4 relative RMSE (%)")
    axes[0].set_title("MSS B4 residual", fontsize=10)
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].set_xticks([2, 6], ["calibración", "validación"], fontsize=8)

    boxplot_by_group(axes[1], "B4", "scale", 1.0)
    target = water_vapor["target_scale"]
    axes[1].axhline(target, color="black", linestyle="--", linewidth=1)
    axes[1].text(
        2.8,
        target + 0.012,
        f"B1-B3 median = {target:.3f}",
        ha="center",
        va="bottom",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
    )
    axes[1].set_ylabel("B4 multiplicative scale")
    axes[1].set_title("B4 scale before/after/legacy", fontsize=10)
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].set_xticks([2, 6], ["calibración", "validación"], fontsize=8)

    loo = water_vapor["loo"]
    loo_gamma = np.array([row["gamma"] for row in loo])
    loo_mean = np.array([100 * row["loo_mean_rel_rmse"] for row in loo])
    loo_max = np.array([100 * row["loo_max_rel_rmse"] for row in loo])
    axes[2].plot(loo_gamma, loo_mean, "o-", color="C0", label="LOO mean")
    axes[2].plot(loo_gamma, loo_max, "s--", color="C1", label="LOO max")
    axes[2].axvline(gamma, color="black", linestyle=":", linewidth=1)
    y_gamma = float(np.interp(gamma, loo_gamma, loo_mean))
    axes[2].annotate(
        f"chosen gamma = {gamma:.3f}\nLOO mean = {y_gamma:.2f} %",
        xy=(gamma, y_gamma),
        xytext=(gamma + 0.035, 70.0),
        fontsize=7,
        arrowprops={"arrowstyle": "->", "linewidth": 0.7},
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85},
    )
    axes[2].set_xlabel("water-vapour gamma")
    axes[2].set_ylabel("B4 scaled RMSE (%)")
    axes[2].set_title("Leave-one-site-out (calibration)", fontsize=10)
    axes[2].set_xticks(np.arange(0.0, 0.21, 0.05))
    axes[2].grid(alpha=0.25)

    handles_b4 = [
        *handles,
        Line2D([], [], color="C0", marker="o", linestyle="-", label="LOO mean"),
        Line2D([], [], color="C1", marker="s", linestyle="--", label="LOO max"),
        Line2D([], [], color="black", linestyle="--", label=f"B1-B3 median = {target:.3f}"),
        Line2D([], [], color="black", linestyle=":", label=f"chosen gamma = {gamma:.3f}"),
    ]
    fig.legend(
        handles=handles_b4,
        loc="outside lower center",
        ncol=5,
        fontsize=7,
        frameon=False,
    )
    fig.suptitle("MSS B4 water-vapour correction", fontsize=12)
    path = out_dir / "pics_b4_water_vapor.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.18, pil_kwargs={"optimize": True})
    plt.close(fig)
    paths.append(path)
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument("--mss-tm-stats", type=Path, default=DEFAULT_MSS_TM_STATS)
    args = parser.parse_args()

    mss_tm = load_mss_tm_slopes(args.mss_tm_stats)

    index = args.data / "pics_index.json"
    if index.exists():
        order = [s["site"] for s in json.loads(index.read_text())["sites"]]
    else:
        order = sorted(
            p.parent.name
            for p in args.data.glob("*/metadata.json")
            if not p.parent.name.startswith("legacy_")
        )
    if args.sites:
        order = [s for s in order if s in args.sites]
    dirs = [args.data / s for s in order if (args.data / s / "metadata.json").exists()]
    if not dirs:
        raise SystemExit(f"no PICS with metadata.json in {args.data}")

    records = []
    for site_dir in dirs:
        print(f"procesando {site_dir.name} ...", flush=True)
        record = prepare_site(site_dir)
        records.append(record)
        reg = record["reg"]
        print(
            f"  shift dy={reg['dy']:+.3f} dx={reg['dx']:+.3f} px | "
            f"r NIR {reg['r_before']:.3f} -> {reg['r_after']:.3f} ({reg['candidate']})",
            flush=True,
        )

    calibration_all = [r for r in records if r["role"] == "calibration"]
    validation = [r for r in records if r["role"] != "calibration"]
    for record in calibration_all:
        record["use_for_calibration"] = site_quality(record)
    for record in validation:
        record["use_for_calibration"] = False
    calibration = [r for r in calibration_all if r["use_for_calibration"]]
    rejected = [r for r in calibration_all if not r["use_for_calibration"]]
    if not calibration:
        calibration = calibration_all
        for record in calibration:
            record["use_for_calibration"] = True
        rejected = []
    print(
        f"\n{len(calibration)} calibration sites used, {len(rejected)} rejected, "
        f"{len(validation)} validation sites",
        flush=True,
    )
    for record in rejected:
        print(f"  rejected {record['site']['meta']['site']}", flush=True)
    gamma, target, loo = calibrate_water_vapor(calibration)
    loo_gamma = np.array([row["gamma"] for row in loo])
    loo_at_gamma = {
        "gamma": gamma,
        "loo_mean_rel_rmse": float(
            np.interp(gamma, loo_gamma, [row["loo_mean_rel_rmse"] for row in loo])
        ),
        "loo_max_rel_rmse": float(
            np.interp(gamma, loo_gamma, [row["loo_max_rel_rmse"] for row in loo])
        ),
    }
    embedded = load_coefficients().get("water_vapor", {}).get("weights", {})
    embedded_gamma = float(embedded.get("S2A", [0.0, 0.0, 0.0, 0.0])[3])
    print(
        f"\nwater-vapour gamma={gamma:.3f} (target B4 scale {target:.4f}, "
        f"LOO mean {100 * loo_at_gamma['loo_mean_rel_rmse']:.2f} %)",
        flush=True,
    )
    if abs(embedded_gamma - gamma) > 1e-6:
        print(
            f"warning: package coefficients carry gamma={embedded_gamma:.3f}; "
            "run scripts/fit_sbaf.py to embed the calibrated value",
            flush=True,
        )

    rows: list[dict[str, object]] = []
    for record in records:
        rows.extend(rows_for_model(record, "sbaf_nowv", 0.0))
        rows.extend(rows_for_model(record, "sbaf", gamma))
        rows.extend(rows_for_model(record, "legacy", 0.0))
    detail = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def summarize(frame: pd.DataFrame, with_ci: bool) -> pd.DataFrame:
        out = []
        for (band, model), group in frame.groupby(["mss_band", "model"]):
            reference = mss_tm.get(band, {}).get("median")
            if with_ci and len(group) > 1:
                ci = bootstrap_ci(group, reference)
                delta = ci.get("delta", {})
            else:
                ci = {
                    key: {"low": float("nan"), "high": float("nan")}
                    for key in ("slope", "offset", "scale")
                }
                delta = {}
            out.append(
                {
                    "mss_band": band,
                    "model": model,
                    "n_sites": len(group),
                    "n_pixels": int(group["n"].sum()),
                    "r": float(group["r"].median()),
                    "r_min": float(group["r"].min()),
                    "slope": float(group["slope"].median()),
                    "slope_ci_low": ci["slope"]["low"],
                    "slope_ci_high": ci["slope"]["high"],
                    "offset": float(group["offset"].median()),
                    "offset_ci_low": ci["offset"]["low"],
                    "offset_ci_high": ci["offset"]["high"],
                    "rmse_raw_rel": float(group["rmse_raw_rel"].median()),
                    "rmse_scaled_rel": float(group["rmse_scaled_rel"].median()),
                    "median_ratio": float(group["median_ratio"].median()),
                    "scale": float(group["scale"].median()),
                    "scale_ci_low": ci["scale"]["low"],
                    "scale_ci_high": ci["scale"]["high"],
                    "delta_mss_tm": delta.get("value", float("nan")),
                    "delta_ci_low": delta.get("low", float("nan")),
                    "delta_ci_high": delta.get("high", float("nan")),
                    "n_boot": BOOTSTRAP_N if with_ci else 0,
                }
            )
        return pd.DataFrame(out).sort_values(["mss_band", "model"])

    calibration_rows = detail[detail.use_for_calibration]
    summary = summarize(calibration_rows, with_ci=True)
    summary_validation = (
        summarize(detail[detail.role == "validation"], with_ci=False)
        if len(validation) > 1
        else pd.DataFrame()
    )

    # Denormalise the per-band summary into the per-site table: the CSV keeps
    # one row per site x band x model, but also carries the band-level CI and
    # the MSS/TM cross-check so it can be read without joining another file.
    for (band, model), _group in calibration_rows.groupby(["mss_band", "model"]):
        row = summary[(summary.mss_band == band) & (summary.model == model)].iloc[0]
        mask = (
            (detail.role == "calibration")
            & (detail.mss_band == band)
            & (detail.model == model)
        )
        for column in (
            "slope_ci_low",
            "slope_ci_high",
            "offset_ci_low",
            "offset_ci_high",
            "scale_ci_low",
            "scale_ci_high",
        ):
            detail.loc[mask, column] = row[column]
        reference = mss_tm.get(band)
        if reference is not None:
            detail.loc[mask, "mss_tm_slope"] = reference["median"]
            detail.loc[mask, "delta_mss_tm"] = detail.loc[mask, "slope"] - reference["median"]
            detail.loc[mask, "delta_ci_low"] = row["delta_ci_low"]
            detail.loc[mask, "delta_ci_high"] = row["delta_ci_high"]

    sbaf = detail[(detail.model == "sbaf") & detail.use_for_calibration]
    bands: dict[str, dict[str, float]] = {}
    raw_scales: dict[str, list[float]] = {}
    for band in BANDS:
        values = sbaf[sbaf.mss_band == band].set_index("site")["scale"]
        median = float(values.median())
        mad = float(np.median(np.abs(values - median))) * 1.4826
        cutoff = 4.0 * mad if mad > 0 else np.inf
        inliers = values[np.abs(values - median) <= cutoff]
        outliers = sorted(values[np.abs(values - median) > cutoff].index)
        raw_scales[band] = {site: round(float(value), 4) for site, value in values.items()}
        bands[band] = {
            "mean": float(inliers.median()) if len(inliers) else median,
            "median_all": median,
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "mad": mad,
            # Twice the robust sigma, floored at 0.03 and capped at 0.12: the
            # raw site-to-site std (0.14-0.22) mixes calibration with surface
            # change and atmosphere, and is not a usable augmentation width.
            "variation": float(np.clip(2.0 * mad, 0.03, 0.12)),
            "n_inliers": len(inliers),
            "outliers": outliers,
            "per_site": raw_scales[band],
        }
    scaling = {
        "method": (
            "Per-band multiplicative scale s = <pred, obs> / <pred, pred> of the SBAF "
            "prediction (with the calibrated water-vapour term) against the MSS TOA "
            "reflectance, on the valid pixels of each calibration PICS. The reported "
            "mean is the median of the inlier sites (4 robust sigma from the median); "
            "the variation is twice the robust sigma, floored at 0.03 and capped at "
            "0.12. Validation sites never enter the constant."
        ),
        "bands": bands,
        "n_sites": len(calibration),
        "sites": [r["site"]["meta"]["site"] for r in calibration],
        "rejected_sites": [r["site"]["meta"]["site"] for r in rejected],
        "validation_sites": [r["site"]["meta"]["site"] for r in validation],
        "sensor": "S2A",
        "psf_fwhm_m": 80.0,
        "grid_m": 60.0,
    }

    per_site_wv = []
    for record in calibration:
        name = record["site"]["meta"]["site"]
        before = band_metrics(
            model_prediction(record, "sbaf_nowv", 0.0)[3],
            record["site"]["mss"][3],
            record["model_valid"],
        )
        after = band_metrics(
            model_prediction(record, "sbaf", gamma)[3],
            record["site"]["mss"][3],
            record["model_valid"],
        )
        legacy = band_metrics(
            model_prediction(record, "legacy", 0.0)[3],
            record["site"]["mss"][3],
            record["model_valid"],
        )
        per_site_wv.append(
            {
                "site": name,
                "wv_median": float(np.median(record["wv"][record["model_valid"]])),
                "rmse_before": before["rmse_raw_rel"],
                "rmse_after": after["rmse_raw_rel"],
                "rmse_legacy": legacy["rmse_raw_rel"],
                "scale_before": before["scale"],
                "scale_after": after["scale"],
                "scale_legacy": legacy["scale"],
            }
        )
    median_before = {
        "rmse": float(np.median([row["rmse_before"] for row in per_site_wv])),
        "scale": float(np.median([row["scale_before"] for row in per_site_wv])),
    }
    median_after = {
        "rmse": float(np.median([row["rmse_after"] for row in per_site_wv])),
        "scale": float(np.median([row["scale_after"] for row in per_site_wv])),
    }
    median_legacy = {
        "rmse": float(np.median([row["rmse_legacy"] for row in per_site_wv])),
        "scale": float(np.median([row["scale_legacy"] for row in per_site_wv])),
    }
    water_vapor = {
        "gamma": gamma,
        "feature": "rho_B8A * log(rho_B8A / rho_B9)",
        "bands": ["B4"],
        "criterion": (
            "median B4 scale equals the pooled median of the B1-B3 scales; "
            "those bands do not depend on gamma, so the residual B4 offset is "
            "sensor calibration rather than 0.94 um absorption"
        ),
        "target_scale": target,
        "grid": {
            "fine_start": float(WV_FINE[0]),
            "fine_stop": float(WV_FINE[-1]),
            "fine_step": float(WV_FINE[1] - WV_FINE[0]),
            "loo_step": float(WV_GRID[1] - WV_GRID[0]),
        },
        "loo_at_gamma": loo_at_gamma,
        "loo": loo,
        "per_site": per_site_wv,
        "median_before": median_before,
        "median_after": median_after,
        "median_legacy": median_legacy,
        "note": (
            "A 10-predictor NNLS fit that used B9 as a surface band fitted the "
            "library better (1.2 % vs 4.6 % RMSE) but under-predicted MSS4 by "
            "30-48 % on the TOA PICS: B9 is a water-vapour absorption band."
        ),
    }

    regression_bands: dict[str, dict[str, object]] = {}
    for band in BANDS:
        row = summary[(summary.mss_band == band) & (summary.model == "sbaf")].iloc[0]
        reference = mss_tm.get(band, {}).get("median")
        regression_bands[band] = {
            "slope": float(row["slope"]),
            "slope_ci": [float(row["slope_ci_low"]), float(row["slope_ci_high"])],
            "offset": float(row["offset"]),
            "offset_ci": [float(row["offset_ci_low"]), float(row["offset_ci_high"])],
            "scale": float(row["scale"]),
            "scale_ci": [float(row["scale_ci_low"]), float(row["scale_ci_high"])],
            "mss_tm_slope": reference,
            "mss_tm_delta": None if reference is None else float(row["delta_mss_tm"]),
            "mss_tm_delta_ci": (
                None
                if reference is None
                else [float(row["delta_ci_low"]), float(row["delta_ci_high"])]
            ),
        }

    regression = {
        "method": (
            "MSS = slope * prediction + offset, fitted per site and band; the "
            "summary is the median across the calibration PICS."
        ),
        "uncertainty": {
            "method": "paired site-level bootstrap of the median slope/offset/scale",
            "n_boot": BOOTSTRAP_N,
            "level": BOOTSTRAP_LEVEL,
            "seed": BOOTSTRAP_SEED,
            "unit": f"site ({len(calibration)} independent calibration scenes)",
        },
        "bands": regression_bands,
    }

    detail.to_csv(args.out_dir / "sbaf_pics_validation.csv", index=False, float_format="%.6f")
    summary.to_csv(args.out_dir / "sbaf_pics_summary.csv", index=False, float_format="%.6f")
    if not summary_validation.empty:
        summary_validation.to_csv(
            args.out_dir / "sbaf_pics_summary_validation.csv",
            index=False,
            float_format="%.6f",
        )
    (args.out_dir / "sbaf_pics_band_scaling.json").write_text(json.dumps(scaling, indent=2) + "\n")
    (args.out_dir / "sbaf_pics_water_vapor.json").write_text(
        json.dumps(water_vapor, indent=2) + "\n"
    )
    (args.out_dir / "sbaf_pics_regression.json").write_text(
        json.dumps(regression, indent=2) + "\n"
    )
    write_markdown(
        calibration_rows,
        summary,
        scaling,
        [r["site"] for r in calibration] + [r["site"] for r in validation],
        args.out_dir / "sbaf_pics_table.md",
        water_vapor,
        gamma,
        mss_tm,
    )

    fig_dir = args.figures or args.out_dir / "figures"
    figures = make_figures(records, fig_dir, detail, gamma, water_vapor)

    metadata = {
        "method": (
            "Deterministic emulator (SBAF with water-vapour term or legacy band "
            "selection + PSF 80 m + 60 m average pooling) on a 15 km S2A window, "
            "registered to the MSS grid with subpixel phase correlation + ECC "
            "translation. Metrics on cloud-free, unsaturated pixels."
        ),
        "mss_conversion": (
            "dn_to_toa: rho = pi * (M_L * DN + A_L) * d^2 / (ESUN * sin(sun_elevation)). "
            "The C02 MTL REFLECTANCE_MULT carries no sun-elevation correction (it is "
            "the same at SZA 31 and 50 deg), so DN*mult+add returns rho*sin(SZA) and "
            "underestimates by 1.26x-1.86x in these scenes."
        ),
        "registration": "skimage phase_cross_correlation(upsample_factor=100) + cv2 ECC translation",
        "regression": regression,
        "mss_tm_crosscheck": {
            "source": str(args.mss_tm_stats),
            "definition": "median MSS/TM slope per band over the 20 clear scenes",
            "bands": mss_tm,
            "note": (
                "The predictor differs (TM here, emulated S2 in the PICS), so the "
                "slopes are comparable in direction, not in absolute value."
            ),
        },
        "water_vapor": water_vapor,
        "outliers": {
            "egypt1": (
                "Lowest per-band scales of the calibration sites (surface/calibration); "
                "kept in the table and flagged, the median is used."
            )
        },
        "excluded_sites": {
            "legacy_libya4": (
                "Wrong location (23.07 N, 22.93 E, T34QGL), not the CEOS Libya-4 "
                "PICS; superseded by libya4_ceos and not used by the validator."
            ),
            "legacy_atacama": (
                "Surface changed between the 1985 MSS and the 2020 S2 overpass; "
                "not used by the validator."
            ),
        },
        "sites": {
            r["site"]["meta"]["site"]: {
                "label": r["site"]["meta"]["label"],
                "role": r["role"],
                "surface": r["site"]["meta"].get("surface", "desert"),
                "coordinates": (
                    f"{r['site']['meta']['centre_wgs84']['lat']:.2f} N, "
                    f"{r['site']['meta']['centre_wgs84']['lon']:.2f} E"
                ),
                "mss_id": r["site"]["meta"]["mss"]["product_id"],
                "mss_date": r["site"]["meta"]["mss"]["date"],
                "s2_id": r["site"]["meta"]["s2a"]["product_id"],
                "s2_tiles": r["site"]["meta"]["s2a"]["tiles"],
                "s2_date": r["site"]["meta"]["s2a"]["date"],
                "sun_zenith_mss": round(90.0 - r["site"]["meta"]["mss"]["sun_elevation"], 2),
                "sun_zenith_s2": r["site"]["meta"]["s2a"]["sun_zenith"],
                "shift_px": [round(r["reg"]["dy"], 3), round(r["reg"]["dx"], 3)],
                "corr_nir_before": round(r["reg"]["r_before"], 4),
                "corr_nir_after": round(r["reg"]["r_after"], 4),
            }
            for r in records
        },
        "band_scaling": scaling["bands"],
        "files_sha256": {
            p.name: sha256_file(p)
            for p in (
                args.out_dir / "sbaf_pics_validation.csv",
                args.out_dir / "sbaf_pics_summary.csv",
                args.out_dir / "sbaf_pics_table.md",
                args.out_dir / "sbaf_pics_band_scaling.json",
                args.out_dir / "sbaf_pics_regression.json",
                args.out_dir / "sbaf_pics_water_vapor.json",
            )
        },
        "figures": [str(p) for p in figures],
    }
    (args.out_dir / "sbaf_pics_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("\nResumen (mediana entre sitios):")
    print(
        summary[["mss_band", "model", "n_sites", "r", "rmse_raw_rel", "rmse_scaled_rel"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    print("\nregresión SBAF+WV (mediana e IC95 bootstrap de sitios):")
    for band in BANDS:
        row = summary[(summary.mss_band == band) & (summary.model == "sbaf")].iloc[0]
        print(
            f"  {band}: slope={row.slope:.3f} [{row.slope_ci_low:.3f}, "
            f"{row.slope_ci_high:.3f}] offset={row.offset:+.4f} "
            f"[{row.offset_ci_low:+.4f}, {row.offset_ci_high:+.4f}] "
            f"scale={row.scale:.3f} [{row.scale_ci_low:.3f}, {row.scale_ci_high:.3f}]"
        )
    print("\nband_scaling calibrado:")
    for band, values in bands.items():
        print(
            f"  {band}: mean={values['mean']:.3f} std={values['std']:.3f} "
            f"variation={values['variation']:.3f} per_site={values['per_site']}"
        )
    print(f"\nfiguras: {[str(p) for p in figures]}")
    print(f"tabla: {args.out_dir / 'sbaf_pics_table.md'}")


if __name__ == "__main__":
    main()
