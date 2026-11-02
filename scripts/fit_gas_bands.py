#!/usr/bin/env python
"""Band-integrated gaseous transmittance with 6SV, for MSS vs Sentinel-2.

The SBAF predicts an MSS band as a linear combination of Sentinel-2 bands.
That works for the *surface* spectrum, but the atmosphere does not follow the
same linear combination: O2-A (0.76 um) and H2O (0.72, 0.82, 0.94 um) sit
inside the broad MSS bands and only partially inside the narrow S2 bands the
SBAF uses. The result is a multiplicative bias per band that is not a sensor
calibration error and that changes with the water-vapour column.

This script runs 6SV monochromatically from 400 to 1100 nm for a grid of
total-column water vapour values, integrates the gaseous transmittance with
the official SRFs of both sensors (``assets/md/fuentes/srf``) and writes the
per-band transmittances. ``fit_gas_correction.py`` turns them into the scale
bias per band, using the SBAF weights and the reflectance of each PICS.

6SV lives in its own conda env because Py6S needs the Fortran binary::

    /data/users/julio/miniconda3/bin/conda create -p ~/.conda/envs/sixs \\
        -c conda-forge sixs python=3.12 && pip install Py6S

Usage
-----
    /data/users/julio/.conda/envs/sixs/bin/python scripts/fit_gas_bands.py
    /data/users/julio/.conda/envs/sixs/bin/python scripts/fit_gas_bands.py --step-nm 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

VAULT = REPO.parent
SRF_DIR = VAULT / "assets" / "md" / "fuentes" / "srf"
OUT_DIR = SRF_DIR

MSS_SRF = SRF_DIR / "l5mss_rsr_long.csv"
S2_SRF = SRF_DIR / "s2_srf_long.csv"

MSS_BANDS = ("B1", "B2", "B3", "B4")
S2_BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9")

#: Water-vapour column grid, cm. 0.25 is a dry desert; 6 cm a humid summer.
TCWV_CM = (0.10, 0.25, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00)

#: Fixed gases. Mid-latitude summer ozone column, US standard.
OZONE_ATM_CM = 0.30

#: Geometry: median sun zenith of the five PICS pairs (~45 deg) and a near
#: nadir view. The gas transmittance scales with the air mass, so the ratio
#: between MSS and S2 bands is what matters and it barely moves with geometry.
SOLAR_ZENITH = 45.0
VIEW_ZENITH = 8.0
DAY, MONTH = 15, 7


def find_sixs() -> str:
    """Path of the 6SV executable, in its own env first."""
    candidates = [
        Path.home() / ".conda" / "envs" / "sixs" / "bin" / "sixs",
        Path("/data/users/julio/.conda/envs/sixs/bin/sixs"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    found = shutil.which("sixs")
    if found:
        return found
    raise SystemExit(
        "6SV executable not found. Create the env: conda create -p "
        "~/.conda/envs/sixs -c conda-forge sixs python=3.12 && pip install Py6S"
    )


def load_srf(path: Path, sensor: str, bands: tuple[str, ...]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read a long-format SRF CSV into ``{band: (wl_nm, response)}``."""
    table: dict[str, list[tuple[float, float]]] = {band: [] for band in bands}
    with path.open() as handle:
        header = handle.readline()
        columns = [name.strip() for name in header.split(",")]
        idx = {name: i for i, name in enumerate(columns)}
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) != len(columns) or parts[idx["sensor"]] != sensor:
                continue
            band = parts[idx["band"]]
            if band not in table:
                continue
            table[band].append((float(parts[idx["wavelength_nm"]]), float(parts[idx["response"]])))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for band, rows in table.items():
        if not rows:
            continue
        arr = np.asarray(sorted(rows))
        out[band] = (arr[:, 0], arr[:, 1] / arr[:, 1].sum())
    return out


def run_6s(wavelengths_nm: np.ndarray, tcwv_cm: float) -> np.ndarray:
    """Total gaseous transmittance for each wavelength (sun-surface-sensor)."""
    from Py6S import AeroProfile, AtmosProfile, Geometry, SixS, Wavelength

    sixs = SixS(path=find_sixs())
    sixs.atmos_profile = AtmosProfile.UserWaterAndOzone(tcwv_cm, OZONE_ATM_CM)
    sixs.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    sixs.geometry = Geometry.User()
    sixs.geometry.solar_z = SOLAR_ZENITH
    sixs.geometry.view_z = VIEW_ZENITH
    sixs.geometry.day = DAY
    sixs.geometry.month = MONTH
    sixs.ground_reflectance = __import__("Py6S").GroundReflectance.HomogeneousLambertian(0.3)

    out = np.empty(wavelengths_nm.size)
    for i, wl_nm in enumerate(wavelengths_nm):
        sixs.wavelength = Wavelength(wl_nm / 1000.0)
        sixs.run()
        out[i] = sixs.outputs.transmittance_global_gas.total
    return out


def integrate(srf: tuple[np.ndarray, np.ndarray], wl_nm: np.ndarray, tgas: np.ndarray) -> float:
    """SRF-weighted mean of the gaseous transmittance (linear interpolation)."""
    wl_srf, resp = srf
    inside = (wl_srf >= wl_nm[0]) & (wl_srf <= wl_nm[-1])
    t = np.interp(wl_srf[inside], wl_nm, tgas)
    weight = resp[inside]
    return float((t * weight).sum() / weight.sum())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-nm", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "gas_bands.json")
    parser.add_argument("--tcwv", nargs="+", type=float, default=list(TCWV_CM))
    args = parser.parse_args(argv)

    wavelengths = np.arange(400.0, 1100.0 + 1e-9, args.step_nm)
    mss = load_srf(MSS_SRF, "L5MSS", MSS_BANDS)
    s2 = load_srf(S2_SRF, "S2A", S2_BANDS)
    print(f"SRFs: MSS {sorted(mss)} | S2 {sorted(s2)} | {wavelengths.size} wavelengths")

    results: dict[str, dict[str, float]] = {}
    for tcwv in args.tcwv:
        tgas = run_6s(wavelengths, tcwv)
        results[f"{tcwv:g}"] = {
            "mss": {band: integrate(srf, wavelengths, tgas) for band, srf in mss.items()},
            "s2": {band: integrate(srf, wavelengths, tgas) for band, srf in s2.items()},
        }
        print(
            f"  TCWV {tcwv:4.2f} cm  "
            + "  ".join(f"MSS{b}={results[f'{tcwv:g}']['mss'][b]:.4f}" for b in MSS_BANDS)
        )

    payload = {
        "sensor_mss": "L5MSS",
        "sensor_s2": "S2A",
        "method": "6SV 1.1 monochromatic 400-1100 nm, integrated with the official SRFs "
        "(1 nm response curves). Total gaseous transmittance (sun-surface-sensor) of "
        "H2O + O2 + O3 + CO2 + CH4 + NO2 + CO.",
        "geometry": {
            "solar_zenith": SOLAR_ZENITH,
            "view_zenith": VIEW_ZENITH,
            "day": DAY,
            "month": MONTH,
            "ozone_atm_cm": OZONE_ATM_CM,
            "aerosol": "continental",
        },
        "step_nm": args.step_nm,
        "tcwv_cm": list(args.tcwv),
        "bands": results,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
