#!/usr/bin/env python
"""Turn the 6SV band transmittances into a per-band bias of the SBAF prediction.

The SBAF predicts ``rho_MSS(m) = sum_k w[m,k] * rho_S2(k) + c[m]``. Applied at
TOA, that linear combination also predicts the *gaseous transmission* as the
reflectance-weighted mean of the S2 band transmissions. The MSS band has its
own transmission, because O2-A (0.76 um) and the H2O bands (0.72, 0.82, 0.94 um)
fall inside a broad MSS band and only partly inside the S2 bands.

So the gas-induced scale of band ``m`` is::

    s_gas(m) = T_MSS(m) / sum_k (w[m,k] * rho_k * T_S2(k)) / sum_k (w[m,k] * rho_k)

with ``rho_k`` the median TOA reflectance of the site in each S2 predictor.
Any residual after dividing the observed scale by ``s_gas`` is sensor
calibration, which is what ``final_scaling.band_scaling`` should hold.

Usage
-----
    /data/users/julio/.conda/envs/deep/bin/python scripts/fit_gas_correction.py
    ... --tcwv 1.5 --out /tmp/gas_correction.json
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
SITES = ("libya4_ceos", "algeria3", "niger1", "egypt1", "arabia2")

#: Reference water-vapour column: the median of the PICS, from ERA5 daily
#: surface dewpoint and pressure (``scripts/pics_water_vapour.py``). Updated by
#: that script; 1.1 cm is the desert-winter value used when it is missing.
DEFAULT_TCWV = 1.1


def site_s2_reflectance(site: str) -> np.ndarray | None:
    """Median TOA reflectance of the S2 predictors over the usable pixels."""
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


def gas_scale(
    band: int,
    t_mss: dict[str, float],
    t_s2: dict[str, float],
    weights: np.ndarray,
    reflectance: np.ndarray,
) -> float:
    """Scale that gaseous absorption alone imposes on band ``band`` (1-based)."""
    w = weights[band - 1]
    surface = reflectance * w
    predicted = float((surface * np.array([t_s2[b] for b in PREDICTORS])).sum() / surface.sum())
    return float(t_mss[MSS_BANDS[band - 1]] / predicted)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gas", type=Path, default=SRF_DIR / "gas_bands.json")
    parser.add_argument("--scaling", type=Path, default=SRF_DIR / "sbaf_pics_band_scaling.json")
    parser.add_argument("--regression", type=Path, default=SRF_DIR / "sbaf_pics_regression.json")
    parser.add_argument("--tcwv", type=float, default=None, help="reference TCWV in cm")
    parser.add_argument("--out", type=Path, default=SRF_DIR / "gas_correction.json")
    args = parser.parse_args(argv)

    gas = json.loads(args.gas.read_text())
    coefficients = load_coefficients()
    weights = np.asarray(coefficients["sensors"]["S2A"]["weights"], dtype=float)

    reflectances = {site: site_s2_reflectance(site) for site in SITES}
    usable = {k: v for k, v in reflectances.items() if v is not None}
    if not usable:
        raise SystemExit(f"no S2 reflectance found under {PICS_DIR}")
    source = (
        "median of the five PICS"
        if len(usable) == len(SITES)
        else f"median of {len(usable)} PICS ({', '.join(sorted(usable))})"
    )
    reflectance = np.median(np.stack(list(usable.values())), axis=0)

    tcwv = args.tcwv if args.tcwv is not None else DEFAULT_TCWV
    key = min(gas["bands"], key=lambda k: abs(float(k) - tcwv))
    block = gas["bands"][key]
    scales = {
        band: gas_scale(band, block["mss"], block["s2"], weights, reflectance)
        for band in (1, 2, 3, 4)
    }

    # Curve over the whole TCWV grid, for the sensitivity and the figure.
    curve = {
        k: {
            band: gas_scale(band, b["mss"], b["s2"], weights, reflectance)
            for band in (1, 2, 3, 4)
        }
        for k, b in gas["bands"].items()
    }

    scaling = json.loads(args.scaling.read_text()) if args.scaling.exists() else None
    observed = (
        {band: scaling["bands"][band]["mean"] for band in ("B1", "B2", "B3", "B4")}
        if scaling
        else None
    )
    regression = json.loads(args.regression.read_text()) if args.regression.exists() else None
    ci = (
        {band: regression["bands"][band]["scale_ci"] for band in ("B1", "B2", "B3", "B4")}
        if regression
        else None
    )

    corrected = {
        band: (observed[band] / scales[int(band[1])]) if observed else None
        for band in ("B1", "B2", "B3", "B4")
    }
    pooled = (
        float(np.median([corrected[b] for b in ("B1", "B2", "B4")])) if observed else None
    )

    payload = {
        "method": "s_gas(m) = T_MSS(m) / reflectance-weighted mean of the S2 predictor "
        "transmissions. 'corrected' = observed band scale / s_gas(m): what is left for "
        "the sensor once the atmospheric absorption the linear SBAF cannot represent is "
        "removed.",
        "gas_source": str(args.gas),
        "sensor_s2": "S2A",
        "weights_source": str(Path(coefficients.get("source", "sbaf_coefficients.json"))),
        "reflectance_source": source,
        "reflectance_predictors": dict(zip(PREDICTORS, map(float, reflectance), strict=True)),
        "reference_tcwv_cm": float(key),
        "requested_tcwv_cm": tcwv,
        "s_gas": {f"B{b}": v for b, v in scales.items()},
        "observed_scale": observed,
        "scale_ci": ci,
        "corrected_scale": corrected,
        "pooled_corrected": pooled,
        "curve": curve,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"reference TCWV {key} cm  ({source})")
    print(f"{'band':4s} {'T_MSS':>7s} {'s_gas':>7s} {'obs':>7s} {'obs/s_gas':>10s}")
    for band in ("B1", "B2", "B3", "B4"):
        i = int(band[1])
        print(
            f"{band:4s} {block['mss'][band]:7.4f} {scales[i]:7.4f} "
            f"{(observed[band] if observed else float('nan')):7.4f} "
            f"{(corrected[band] if observed else float('nan')):10.4f}"
        )
    if pooled:
        print(f"pooled B1/B2/B4 corrected: {pooled:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
