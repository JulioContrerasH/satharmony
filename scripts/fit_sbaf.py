#!/usr/bin/env python
"""Fit the S2 -> Landsat MSS Spectral Band Adjustment Factors (SBAF).

Model
-----
For every MSS band ``m`` and Sentinel-2 sensor ``s``::

    rho_MSS(m) ~= sum_k w[m, k] * rho_S2(s, k) + c[m] - g[m] * WV

``WV = rho_B8A * log(rho_B8A / rho_B9)`` is the water-vapour absorption
depth at 0.94 um.  It is calibrated on the five CEOS PICS by
``scripts/validate_sbaf_pics.py`` and only enters MSS4; B9 is not used as a
surface predictor (a 10-predictor fit with B9 under-predicted MSS4 by
30-48 % on TOA PICS).

The linear fit is a non-negative least squares with a free intercept
(``scipy.optimize.lsq_linear``).  Non-negativity keeps the coefficients
interpretable and avoids the wild +/-3 weights that unconstrained OLS gives
for the strongly collinear NIR bands.  The weights are *not* forced to sum to
one; the observed row sums are reported (they are 1.00 +/- 0.01).

Inputs
------
- Official SRFs: ``assets/md/fuentes/srf/s2_srf_long.csv`` and
  ``l5mss_rsr_long.csv`` (already extracted from the ESA/USGS originals).
- ECOSTRESS/ASTER VNIR library (475 spectra covering 400-1100 nm).
- EMIT L2A cloud spectra (30 pixels, one overcast granule).
- Documented synthetic shadow spectra (30, derived from the library).

Outputs
-------
- ``sbaf_weights.csv``        long table with weights and offsets.
- ``sbaf_residuals.csv``      per-band fit and legacy residuals.
- ``sbaf_library_bands.csv``  band-integrated library used by the fit.
- ``sbaf_metadata.json``      method, hashes, counts, notes.
- ``sbaf_coefficients.json``  runtime coefficients embedded in the package.
- ``figures/sbaf_weights_residuals.png``

The water-vapour coefficient is read from
``sbaf_pics_water_vapor.json`` (written by ``validate_sbaf_pics.py``) and
embedded in the package JSON.  Run the validator first; without that file
the water-vapour term is written with zero weight.

Usage
-----
    python scripts/validate_sbaf_pics.py
    python scripts/fit_sbaf.py
    python scripts/fit_sbaf.py --out-dir /tmp/sbaf --package-json /tmp/sbaf.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import lsq_linear
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRF = ROOT / "assets/md/fuentes/srf"
DEFAULT_LIBRARY = ROOT / "assets/docs/srf/library"
DEFAULT_OUT = ROOT / "assets/md/fuentes/srf"
DEFAULT_PACKAGE_JSON = ROOT / "satharmony/satharmony/emulators/s2_to_mss/sbaf_coefficients.json"

SENSORS = ("S2A", "S2B", "S2C")
PREDICTORS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A")
# Index of each predictor in the 13-band S2 stack used by satharmony:
# [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B8A]
# B9 is deliberately excluded: it is a water-vapour absorption band and its
# 0.735 weight for MSS4 degraded the Atacama PICS validation (r 0.63 vs 0.76).
PREDICTOR_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 12)
MSS_BANDS = ("B1", "B2", "B3", "B4")
LEGACY_BAND = {"B1": "B3", "B2": "B4", "B3": "B7", "B4": "B8"}
GRID = np.arange(400.0, 1100.0 + 0.5, 1.0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_srf_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sensor", "band", "wavelength_nm", "response"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} is missing columns {required - set(df.columns)}")
    return df


def srf_array(table: pd.DataFrame, sensor: str, band: str) -> np.ndarray:
    row = table[(table.sensor == sensor) & (table.band == band)]
    if row.empty:
        raise ValueError(f"No SRF for {sensor} {band}")
    # Extend to the common grid; zero outside the measured response.
    response = np.interp(
        GRID, row.wavelength_nm.to_numpy(), row.response.to_numpy(), left=0.0, right=0.0
    )
    area = trapezoid(response, GRID)
    if area <= 0:
        raise ValueError(f"SRF for {sensor} {band} has zero area")
    return response / area


def load_library(library_dir: Path) -> pd.DataFrame:
    """Return long table: spectrum_id, source, wavelength_nm, reflectance."""
    files = [
        (library_dir / "ecostress/spectra_visible.csv", "ecostress"),
        (library_dir / "emit/emit_cloud_spectra.csv", "emit_cloud"),
        (library_dir / "shadow/shadow_synthetic.csv", "shadow_synthetic"),
    ]
    frames = []
    for path, source in files:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        cols = ["spectrum_id", "wavelength_nm", "reflectance"]
        if "class" in df.columns:
            df = df.rename(columns={"class": "spectrum_class"})
            cols.append("spectrum_class")
        else:
            df["spectrum_class"] = source
            cols.append("spectrum_class")
        df = df[cols].copy()
        df["source"] = source
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["wavelength_nm", "reflectance"])
    out = out[(out.wavelength_nm >= 400.0) & (out.wavelength_nm <= 1100.0)]
    return out


def interpolate_library(long: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Interpolate every spectrum to GRID; return matrix (n, len(GRID))."""
    pivot = long.pivot_table(
        index="wavelength_nm",
        columns="spectrum_id",
        values="reflectance",
        aggfunc="mean",
    )
    meta = long.drop_duplicates("spectrum_id").set_index("spectrum_id")[
        ["source", "spectrum_class"]
    ]
    rows: list[np.ndarray] = []
    ids: list[str] = []
    sources: list[str] = []
    classes: list[str] = []
    for sid in sorted(pivot.columns):
        y = pivot[sid].to_numpy(dtype=float)
        x = pivot.index.to_numpy(dtype=float)
        good = np.isfinite(y)
        if good.sum() < 50:
            continue
        values = np.interp(GRID, x[good], y[good])
        if values.min() < -0.01 or values.max() > 1.5:
            continue
        rows.append(values)
        ids.append(sid)
        sources.append(meta.loc[sid, "source"])
        classes.append(meta.loc[sid, "spectrum_class"])
    if not rows:
        raise ValueError("No usable spectra after interpolation")
    return np.vstack(rows), ids, sources, classes


def band_integrate(
    spectra: np.ndarray, table: pd.DataFrame, sensor: str, bands: tuple[str, ...]
) -> np.ndarray:
    columns = []
    for band in bands:
        response = srf_array(table, sensor, band)
        columns.append(trapezoid(spectra * response[None, :], GRID, axis=1))
    return np.column_stack(columns)


def nnls_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.column_stack([x, np.ones(len(x))])
    lower = np.r_[np.zeros(x.shape[1]), -np.inf]
    upper = np.r_[np.full(x.shape[1], np.inf), np.inf]
    result = lsq_linear(design, y, bounds=(lower, upper), lsmr_tol="auto")
    return result.x[:-1], float(result.x[-1])


def residual_stats(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - y
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "n": len(y),
        "rmse": rmse,
        "rel_rmse": float(rmse / y.mean()) if y.mean() else float("nan"),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot else float("nan"),
        "bias": float(np.mean(residual)),
        "max_abs": float(np.max(np.abs(residual))),
    }


def legacy_stats(x_legacy: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Best-case single-band linear mapping (slope + intercept)."""
    design = np.column_stack([x_legacy, np.ones(len(x_legacy))])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = design @ coef
    stats = residual_stats(y, pred)
    stats["slope"] = float(coef[0])
    stats["offset"] = float(coef[1])
    return stats


def cross_validate(x: np.ndarray, y: np.ndarray, folds: int = 5) -> float:
    kfold = KFold(n_splits=folds, shuffle=True, random_state=0)
    errors = []
    for train, test in kfold.split(x):
        weights, offset = nnls_fit(x[train], y[train])
        pred = x[test] @ weights + offset
        errors.append(float(np.sqrt(np.mean((pred - y[test]) ** 2))))
    return float(np.mean(errors))


def plot_results(
    weights: dict[str, np.ndarray],
    offsets: dict[str, np.ndarray],
    residuals: pd.DataFrame,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.0),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )

    w = weights["S2A"]
    ax = axes[0]
    im = ax.imshow(w, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(0.05, w.max()))
    ax.set_xticks(range(len(PREDICTORS)), PREDICTORS, rotation=0)
    ax.set_yticks(range(len(MSS_BANDS)), [f"MSS {b}" for b in MSS_BANDS])
    ax.set_xlabel("Sentinel-2 predictor")
    ax.set_ylabel("Landsat MSS target")
    ax.set_title("SBAF weights, S2A (NNLS, w >= 0)")
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            if w[i, j] >= 0.005:
                ax.text(
                    j,
                    i,
                    f"{w[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if w[i, j] < 0.55 * w.max() else "white",
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="weight")

    ax = axes[1]
    width = 0.38
    x = np.arange(len(MSS_BANDS))
    s2a = residuals[(residuals.sensor == "S2A") & (residuals.model == "sbaf")]
    legacy = residuals[(residuals.sensor == "S2A") & (residuals.model == "legacy")]
    ax.bar(x - width / 2, s2a.rmse.values * 100, width, label="SBAF", color="#1f77b4")
    ax.bar(
        x + width / 2,
        legacy.rmse.values * 100,
        width,
        label="legacy B3/B4/B7/B8",
        color="#bbbbbb",
    )
    for xi, (a, b) in enumerate(zip(s2a.rmse.values, legacy.rmse.values, strict=True)):
        ax.text(
            xi - width / 2,
            a * 100,
            f"{a * 100:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            xi + width / 2,
            b * 100,
            f"{b * 100:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x, [f"MSS {b}" for b in MSS_BANDS])
    ax.set_ylabel("RMSE (reflectance x 100)")
    ax.set_title("Residual on the spectral library (S2A)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "SBAF S2 -> MSS: fitted weights and residual vs legacy band selection",
        fontsize=11,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--srf-dir", type=Path, default=DEFAULT_SRF)
    parser.add_argument("--library-dir", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--package-json", type=Path, default=DEFAULT_PACKAGE_JSON)
    parser.add_argument(
        "--water-vapor-json",
        type=Path,
        default=DEFAULT_OUT / "sbaf_pics_water_vapor.json",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    water_vapor_gamma = 0.0
    water_vapor_source = None
    if args.water_vapor_json.exists():
        water_vapor = json.loads(args.water_vapor_json.read_text())
        water_vapor_gamma = float(water_vapor["gamma"])
        try:
            water_vapor_source = str(args.water_vapor_json.resolve().relative_to(ROOT))
        except ValueError:
            water_vapor_source = str(args.water_vapor_json)
        print(f"water-vapour gamma {water_vapor_gamma:.4f} from {water_vapor_source}")
    else:
        print("warning: no water-vapour calibration file; the MSS4 correction will be zero")
    water_vapor_weights = {sensor: [0.0, 0.0, 0.0, water_vapor_gamma] for sensor in SENSORS}
    water_vapor_block = {
        "form": "rho_B8A * log(rho_B8A / rho_B9)",
        "predictors": ["B8A", "B9"],
        "weights": water_vapor_weights,
        "calibration": {
            "source": water_vapor_source,
            "gamma": water_vapor_gamma,
            "method": (
                "Calibrated on the five CEOS PICS by anchoring the median B4 "
                "scale to the median of the B1-B3 scales; see "
                "validate_sbaf_pics.py and sbaf_pics_water_vapor.json."
            ),
        },
    }

    s2_table = load_srf_table(args.srf_dir / "s2_srf_long.csv")
    mss_table = load_srf_table(args.srf_dir / "l5mss_rsr_long.csv")

    long = load_library(args.library_dir)
    spectra, ids, sources, classes = interpolate_library(long)
    print(f"library: {spectra.shape[0]} spectra x {spectra.shape[1]} wavelengths")
    print(pd.Series(sources).value_counts().to_dict())

    # Band-integrated library, for transparency and reproducibility.
    band_columns: dict[str, np.ndarray] = {}
    for sensor in SENSORS:
        values = band_integrate(spectra, s2_table, sensor, PREDICTORS)
        for k, band in enumerate(PREDICTORS):
            band_columns[f"{sensor}_{band}"] = values[:, k]
    mss_values = band_integrate(spectra, mss_table, "L5MSS", MSS_BANDS)
    for k, band in enumerate(MSS_BANDS):
        band_columns[f"L5MSS_{band}"] = mss_values[:, k]
    library_bands = pd.DataFrame({"spectrum_id": ids, "source": sources, "spectrum_class": classes})
    for name, values in band_columns.items():
        library_bands[name] = values
    library_bands.to_csv(out_dir / "sbaf_library_bands.csv", index=False, float_format="%.8f")
    print(f"wrote {out_dir / 'sbaf_library_bands.csv'}")

    all_weights: dict[str, np.ndarray] = {}
    all_offsets: dict[str, np.ndarray] = {}
    weight_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    sensor_blocks: dict[str, dict[str, object]] = {}

    for sensor in SENSORS:
        x = band_integrate(spectra, s2_table, sensor, PREDICTORS)
        weights = np.zeros((len(MSS_BANDS), len(PREDICTORS)))
        offsets = np.zeros(len(MSS_BANDS))
        sensor_blocks[sensor] = {"weights": [], "offsets": [], "sum_weights": [], "residuals": {}}
        for j, mss_band in enumerate(MSS_BANDS):
            y = mss_values[:, j]
            weights[j], offsets[j] = nnls_fit(x, y)
            pred = x @ weights[j] + offsets[j]
            stats = residual_stats(y, pred)
            stats["cv_rmse"] = cross_validate(x, y)
            legacy_band = LEGACY_BAND[mss_band]
            legacy_index = PREDICTORS.index(legacy_band)
            legacy = legacy_stats(x[:, legacy_index], y)
            improvement = 100.0 * (1.0 - stats["rmse"] / legacy["rmse"])
            stats["legacy_rmse"] = legacy["rmse"]
            stats["legacy_rel_rmse"] = legacy["rel_rmse"]
            stats["legacy_r2"] = legacy["r2"]
            stats["legacy_band"] = legacy_band
            stats["improvement_pct"] = improvement
            sensor_blocks[sensor]["residuals"][mss_band] = stats
            for k, predictor in enumerate(PREDICTORS):
                weight_rows.append(
                    {
                        "sensor": sensor,
                        "mss_band": mss_band,
                        "predictor": predictor,
                        "weight": float(weights[j, k]),
                        "offset": float(offsets[j]),
                        "sum_weights": float(weights[j].sum()),
                    }
                )
            residual_rows.append(
                {
                    "sensor": sensor,
                    "mss_band": mss_band,
                    **{
                        k: stats[k]
                        for k in (
                            "n",
                            "rmse",
                            "rel_rmse",
                            "r2",
                            "bias",
                            "max_abs",
                            "cv_rmse",
                            "legacy_band",
                            "legacy_rmse",
                            "legacy_rel_rmse",
                            "legacy_r2",
                            "improvement_pct",
                        )
                    },
                }
            )
        all_weights[sensor] = weights
        all_offsets[sensor] = offsets
        sensor_blocks[sensor]["weights"] = weights.tolist()
        sensor_blocks[sensor]["offsets"] = offsets.tolist()
        sensor_blocks[sensor]["sum_weights"] = weights.sum(axis=1).tolist()

    weights_df = pd.DataFrame(weight_rows)
    residuals_df = pd.DataFrame(residual_rows)
    weights_df.to_csv(out_dir / "sbaf_weights.csv", index=False, float_format="%.8f")
    residuals_df.to_csv(out_dir / "sbaf_residuals.csv", index=False, float_format="%.8f")
    print(f"wrote {out_dir / 'sbaf_weights.csv'}")
    print(f"wrote {out_dir / 'sbaf_residuals.csv'}")

    # Reprojection-free summary for the note.
    print("\nResidual summary (S2A):")
    print(
        residuals_df[residuals_df.sensor == "S2A"][
            ["mss_band", "rmse", "rel_rmse", "r2", "legacy_rmse", "improvement_pct"]
        ].to_string(index=False, float_format=lambda v: f"{v:.5f}")
    )

    coefficients_sha = sha256_json(
        {
            sensor: {
                "weights": sensor_blocks[sensor]["weights"],
                "offsets": sensor_blocks[sensor]["offsets"],
            }
            for sensor in SENSORS
        }
        | {"water_vapor": water_vapor_weights}
    )
    input_hashes = {
        "s2_srf_long.csv": sha256_file(args.srf_dir / "s2_srf_long.csv"),
        "l5mss_rsr_long.csv": sha256_file(args.srf_dir / "l5mss_rsr_long.csv"),
        "ecostress/spectra_visible.csv": sha256_file(
            args.library_dir / "ecostress/spectra_visible.csv"
        ),
        "emit/emit_cloud_spectra.csv": sha256_file(
            args.library_dir / "emit/emit_cloud_spectra.csv"
        ),
        "shadow/shadow_synthetic.csv": sha256_file(
            args.library_dir / "shadow/shadow_synthetic.csv"
        ),
    }
    counts = pd.Series(sources).value_counts().to_dict()
    metadata = {
        "version": "1.0.0",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": (
            "Non-negative least squares with free intercept: "
            "rho_MSS(m) = sum_k w[m,k] * rho_S2(k) + c[m] - g[m] * WV, "
            "w >= 0, WV = rho_B8A * log(rho_B8A / rho_B9). "
            "Band values are SRF-integrated reflectance over 400-1100 nm; "
            "the water-vapour coefficient is calibrated on the five PICS."
        ),
        "predictors": list(PREDICTORS),
        "predictor_indices": list(PREDICTOR_INDICES),
        "mss_bands": list(MSS_BANDS),
        "sensors": [*SENSORS],
        "library": {
            "n_spectra": spectra.shape[0],
            "counts": {k: int(v) for k, v in counts.items()},
            "grid_nm": [float(GRID[0]), float(GRID[-1]), float(GRID[1] - GRID[0])],
            "note": (
                "ECOSTRESS/ASTER VNIR library (475), EMIT L2A cloud spectra "
                "(30, overcast granule 2022-08-10), documented synthetic "
                "shadow sample (30). The synthetic shadow is not measured. "
                "B9 is not a surface predictor: a 10-band variant that used "
                "it (weight 0.735 for MSS4) fitted the library better "
                "(1.2% vs 4.6%) but under-predicted MSS4 by 30-48 % on the "
                "TOA PICS. B9 enters only through the water-vapour term, "
                "which is calibrated on the PICS."
            ),
        },
        "water_vapor": water_vapor_block,
        "residuals": {sensor: sensor_blocks[sensor]["residuals"] for sensor in SENSORS},
        "input_hashes_sha256": input_hashes,
        "coefficients_sha256": coefficients_sha,
        "figure": str((out_dir / "figures/sbaf_weights_residuals.png").relative_to(out_dir)),
    }

    package = {
        "version": metadata["version"],
        "created": metadata["created"],
        "method": metadata["method"],
        "predictors": list(PREDICTORS),
        "predictor_indices": list(PREDICTOR_INDICES),
        "mss_bands": list(MSS_BANDS),
        "sensors": {
            sensor: {
                "weights": sensor_blocks[sensor]["weights"],
                "offsets": sensor_blocks[sensor]["offsets"],
                "sum_weights": sensor_blocks[sensor]["sum_weights"],
                "residuals": sensor_blocks[sensor]["residuals"],
            }
            for sensor in SENSORS
        },
        "library": metadata["library"],
        "water_vapor": water_vapor_block,
        "input_hashes_sha256": input_hashes,
        "coefficients_sha256": coefficients_sha,
    }

    (out_dir / "sbaf_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    args.package_json.parent.mkdir(parents=True, exist_ok=True)
    args.package_json.write_text(json.dumps(package, indent=2) + "\n")
    print(f"wrote {out_dir / 'sbaf_metadata.json'}")
    print(f"wrote {args.package_json}")

    # The plot needs two rows per band/model; build them explicitly.
    plot_rows = []
    for _, row in residuals_df.iterrows():
        plot_rows.append(
            {
                "sensor": row.sensor,
                "mss_band": row.mss_band,
                "model": "sbaf",
                "rmse": row.rmse,
            }
        )
        plot_rows.append(
            {
                "sensor": row.sensor,
                "mss_band": row.mss_band,
                "model": "legacy",
                "rmse": row.legacy_rmse,
            }
        )
    plot_results(
        all_weights,
        all_offsets,
        pd.DataFrame(plot_rows),
        out_dir / "figures/sbaf_weights_residuals.png",
    )
    print(f"wrote {out_dir / 'figures/sbaf_weights_residuals.png'}")


if __name__ == "__main__":
    main()
