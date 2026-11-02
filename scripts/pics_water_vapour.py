#!/usr/bin/env python
"""Water vapour over the calibration sites: ERA5 column + 6SV gas scale.

Two things are computed per site and per acquisition date (MSS and S2):

1. **Total column water vapour from ERA5** (``ECMWF/ERA5_LAND/HOURLY`` through Earth
   Engine, ~9 km, surface dewpoint + 2 m temperature + surface pressure).
   GEE does not expose ``total_column_water_vapour``, so the column is
   integrated from the surface specific humidity with an exponential humidity
   profile of scale height ``HQ_M`` metres::

       q0 = 0.622 * es(Td) / (p - 0.378 * es(Td))
       TCWV = q0 * p / (Rd * T) * HQ

   The Magnus formula gives ``es``. The scale-height assumption is the only
   free parameter; the script reports the sensitivity to +-25 %.

2. **The gas scale each band would see**, interpolated from the 6SV table
   (``gas_bands.json``): the MSS band transmittance at the MSS date's column
   over the reflectance-weighted predictor transmittance at the S2 date's
   column. This is the part of ``band_scaling`` that is atmosphere, not sensor.

The point of the exercise is the water-vapour ``gamma`` of the SBAF: the five
desert PICS all sit in a narrow, dry range, so they cannot separate the gamma
from the sensor calibration. The output states how much column difference
would be needed to explain the B3/B4 deficits and compares it with ERA5.

Usage
-----
    /data/users/julio/.conda/envs/deep/bin/python scripts/pics_water_vapour.py
    ... --site libya4_ceos --out /tmp/wv.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.emulators.s2_to_mss.sbaf import (  # noqa: E402
    MSS_BANDS,
    PREDICTORS,
    load_coefficients,
)

VAULT = REPO.parent
SRF_DIR = VAULT / "assets" / "md" / "fuentes" / "srf"
PICS_DIR = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")
GEE_PROJECT = "ee-contrerasnetk"

HQ_M = 2000.0
RD = 287.05
SITES = ("libya4_ceos", "algeria3", "niger1", "egypt1", "arabia2")


def saturation_vapour_pressure(td_c: float) -> float:
    """Magnus formula, hPa."""
    return 6.112 * np.exp(17.62 * td_c / (243.12 + td_c))


def tcwv_cm(td_c: float, t_c: float, p_pa: float, hq_m: float = HQ_M) -> float:
    """Column water vapour (cm) from surface dewpoint, temperature and pressure."""
    es = saturation_vapour_pressure(td_c) * 100.0
    q0 = 0.622 * es / (p_pa - 0.378 * es)
    rho_v = q0 * p_pa / (RD * (t_c + 273.15))
    return rho_v * hq_m / 10.0  # kg/m2 -> cm


def era5_surface(lat: float, lon: float, day: str) -> dict:
    """Daily ERA5 surface fields at the site centre."""
    import ee

    ee.Initialize(project=GEE_PROJECT)
    date = ee.Date(day)
    # ERA5-Land hourly (~9 km) covers every date in the set; ERA5/DAILY stops
    # before 2021 and has no TCWV band in Earth Engine. The 24 hourly fields of
    # the acquisition day are averaged.
    image = (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterDate(date, date.advance(1, "day"))
        .select(["dewpoint_temperature_2m", "temperature_2m", "surface_pressure"])
        .mean()
    )
    values = image.reduceRegion(
        ee.Reducer.mean(), ee.Geometry.Point([lon, lat]), 11132, bestEffort=True
    ).getInfo()
    td = float(values["dewpoint_temperature_2m"]) - 273.15
    t2m = float(values["temperature_2m"]) - 273.15
    pressure = float(values["surface_pressure"])
    return {
        "dewpoint_c": round(td, 2),
        "temperature_c": round(t2m, 2),
        "pressure_hpa": round(pressure / 100.0, 1),
        "tcwv_cm": round(tcwv_cm(td, t2m, pressure), 3),
        "tcwv_cm_hq15": round(tcwv_cm(td, t2m, pressure, 1500.0), 3),
        "tcwv_cm_hq25": round(tcwv_cm(td, t2m, pressure, 2500.0), 3),
    }


def gas_table(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """TCWV grid and per-band transmittances (MSS, S2)."""
    payload = json.loads(path.read_text())
    grid = np.array(sorted(float(k) for k in payload["bands"]))
    mss = {
        band: np.array([payload["bands"][f"{w:g}"]["mss"][band] for w in grid])
        for band in MSS_BANDS
    }
    s2 = {
        band: np.array([payload["bands"][f"{w:g}"]["s2"][band] for w in grid])
        for band in PREDICTORS
    }
    return grid, mss, s2


def interpolate(grid: np.ndarray, values: np.ndarray, point: float) -> float:
    return float(np.interp(point, grid, values))


def predictor_transmittance(
    grid: np.ndarray, s2: dict[str, np.ndarray], weights: np.ndarray,
    reflectance: np.ndarray, wv_cm: float,
) -> float:
    """Reflectance-weighted mean S2 transmittance for one MSS band's predictors."""
    t = np.array([interpolate(grid, s2[b], wv_cm) for b in PREDICTORS])
    surface = reflectance * weights
    return float((surface * t).sum() / surface.sum())


def site_s2_reflectance(site: str) -> np.ndarray | None:
    import rasterio

    folder = PICS_DIR / site / "s2"
    if not folder.exists():
        return None
    bands = []
    for band in PREDICTORS:
        files = sorted(folder.glob(f"*.{band}.tif"))
        if not files:
            return None
        with rasterio.open(files[0]) as src:
            data = src.read(1).astype(np.float32) / 10000.0
        data[data <= 0] = np.nan
        bands.append(np.nanmedian(data))
    return np.asarray(bands, dtype=float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=list(SITES))
    parser.add_argument("--gas", type=Path, default=SRF_DIR / "gas_bands.json")
    parser.add_argument("--scaling", type=Path, default=SRF_DIR / "sbaf_pics_band_scaling.json")
    parser.add_argument("--out", type=Path, default=SRF_DIR / "pics_water_vapour.json")
    args = parser.parse_args(argv)

    index = json.loads((PICS_DIR / "pics_index.json").read_text())
    meta = {s["site"]: s for s in index["sites"]}
    grid, t_mss, t_s2 = gas_table(args.gas)
    coefficients = load_coefficients()
    weights = np.asarray(coefficients["sensors"]["S2A"]["weights"], dtype=float)

    scaling = json.loads(args.scaling.read_text()) if args.scaling.exists() else None

    out: dict = {"sites": {}, "hq_m": HQ_M, "gas_grid_cm": grid.tolist()}
    for site in args.sites:
        entry = meta.get(site)
        if entry is None:
            print(f"  {site}: not in pics_index.json, skipped")
            continue
        lat = entry["centre_wgs84"]["lat"]
        lon = entry["centre_wgs84"]["lon"]
        dates = {
            "mss": entry["mss"]["date"],
            "s2": entry["s2a"]["date"][:4] + "-" + entry["s2a"]["date"][4:6] + "-" + entry["s2a"]["date"][6:],
        }
        fields = {key: era5_surface(lat, lon, day) for key, day in dates.items()}
        reflectance = site_s2_reflectance(site)
        if reflectance is None:
            print(f"  {site}: no S2 files, skipped")
            continue

        scales = {}
        for i, band in enumerate(MSS_BANDS):
            w = weights[i]
            t_pred_mss = predictor_transmittance(grid, t_s2, w, reflectance, fields["mss"]["tcwv_cm"])
            t_pred_s2 = predictor_transmittance(grid, t_s2, w, reflectance, fields["s2"]["tcwv_cm"])
            t_band = interpolate(grid, t_mss[band], fields["mss"]["tcwv_cm"])
            observed = (
                scaling["bands"][band]["per_site"].get(site) if scaling else None
            )
            scales[band] = {
                "t_mss": round(t_band, 5),
                "t_pred_mss": round(t_pred_mss, 5),
                "t_pred_s2": round(t_pred_s2, 5),
                "s_gas": round(t_band / t_pred_s2, 5),
                "s_gas_if_same_date": round(t_band / t_pred_mss, 5),
                "observed_scale": observed,
                "corrected": round(observed / (t_band / t_pred_s2), 5) if observed else None,
            }

        out["sites"][site] = {
            "lat": lat,
            "lon": lon,
            "dates": dates,
            "era5": fields,
            "delta_tcwv_cm": round(fields["mss"]["tcwv_cm"] - fields["s2"]["tcwv_cm"], 3),
            "bands": scales,
        }
        print(
            f"  {site:13s} MSS {dates['mss']} {fields['mss']['tcwv_cm']:.2f} cm | "
            f"S2 {dates['s2']} {fields['s2']['tcwv_cm']:.2f} cm | "
            f"B3 s_gas {scales['B3']['s_gas']:.4f} -> {scales['B3']['corrected']}"
        )

    # Site-to-site diagnostics on the corrected scales.
    summary = {}
    for band in MSS_BANDS:
        values = [
            s["bands"][band]["corrected"]
            for s in out["sites"].values()
            if s["bands"][band]["corrected"] is not None
        ]
        if values:
            summary[band] = {
                "n": len(values),
                "median": round(float(np.median(values)), 4),
                "std": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else None,
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
            }
    out["corrected_summary"] = summary
    out["tcwv_range_cm"] = [
        round(min(s["era5"]["mss"]["tcwv_cm"] for s in out["sites"].values()), 3),
        round(max(s["era5"]["mss"]["tcwv_cm"] for s in out["sites"].values()), 3),
    ]

    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print("\ncorrected scales (observed / s_gas):")
    for band, block in summary.items():
        print(
            f"  {band}: median {block['median']:.4f}  std {block['std']}  "
            f"range [{block['min']:.4f}, {block['max']:.4f}]"
        )
    print(f"\nTCWV range over the PICS (MSS dates): {out['tcwv_range_cm']} cm")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
