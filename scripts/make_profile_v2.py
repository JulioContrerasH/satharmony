#!/usr/bin/env python
"""Fit the v2 profile from the 49-pixel S2/MSS calibration set."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.calibration.fit import Pair, fit_profile  # noqa: E402
from satharmony.emulators.s2_to_mss.config import (  # noqa: E402
    MEASURED_SUN_ELEVATION_CANONICAL,
)
from satharmony.emulators.s2_to_mss.sbaf import S2_BAND_ORDER  # noqa: E402

PICS = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")
HELD_OUT = ["libya4_ceos", "sudan1", "oman1", "kyzylkum", "bonneville"]
OUT = REPO / "satharmony/profiles/mss5_esa_gtc_v2.json"


def read_band(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as handle:
        return handle.read(1).astype(np.float32)


def load_pair(site: str) -> Pair:
    root = PICS / site
    s2 = np.stack(
        [
            read_band(sorted((root / "s2").glob(f"*.{band}.tif"))[0])
            / 10000.0
            for band in S2_BAND_ORDER
        ]
    )
    mss = np.stack(
        [read_band(sorted((root / "mss").glob(f"*.B{band}.tif"))[0]) for band in range(1, 5)]
    )
    return Pair(s2=s2, mss=mss, name=site)


def main() -> None:
    index = json.loads((PICS / "pics_index.json").read_text())
    sites = [
        entry["site"]
        for entry in index["sites"]
        if entry.get("role", "calibration") == "calibration"
        and entry["site"] != "uyuni"
        and entry["site"] not in HELD_OUT
    ]
    print(f"{len(sites)} calibration sites")
    pairs = (load_pair(site) for site in sites)
    profile = fit_profile(pairs, name="mss5_esa_gtc_v2", n_boot=1000)
    fitted_gamma = float(profile.gamma)
    fitted_b4 = float(profile.band_scaling[3])
    # B4 stays frozen at v1: the gas term is still a hypothesis and the fitted
    # B4 is not better on the held-out deserts.
    intervals = dict(profile.intervals)
    intervals["B4"] = {"mean": 0.933, "low": 0.916, "high": 0.977, "n": 5}
    profile = replace(
        profile,
        band_scaling=(*profile.band_scaling[:3], 0.933),
        variation=(*profile.variation[:3], 0.072),
        gamma=0.112,
        intervals=intervals,
    )
    payload = profile.to_dict()
    payload["config"] = profile.to_config().model_dump(mode="json")
    # The sun-elevation distribution is the canonical one (2,886 of the 3,027
    # aligned scenes), not the 178 pairs of the legacy bank that v1 used.
    payload["config"]["radiometric"]["sun_elevation_deg"] = dict(
        MEASURED_SUN_ELEVATION_CANONICAL
    )
    payload["gamma_frozen"] = True
    payload["gamma_fitted"] = fitted_gamma
    payload["b4_fitted_scale"] = fitted_b4
    index_path = PICS / "pics_index.json"
    payload["validity"] = {
        "calibration_domain": "desert and salt flats",
        "outside_domain": ["forest", "water", "crops", "wetland"],
        "held_out_sites": HELD_OUT,
        "held_out_disjoint": True,
        # Medians across sites, same convention as the SBAF tables in
        # assets/md/fuentes/srf/pics_49/: "scaled" applies one in-sample
        # multiplicative factor per site and band; "raw" uses the fixed
        # profile scaling and is the number a user would get.
        "desert_in_sample": {
            "sbaf_scaled_rmse": [0.053, 0.062],
            "v2_scaled_rmse": [0.057, 0.070],
            "v2_raw_rmse": [0.082, 0.147],
            "n_sites": 29,
            "method": "29 calibration sites, median across sites",
        },
        "scaled_rmse_outside": [0.134, 0.236],
        "raw_rmse_outside": [0.272, 0.600],
        "n_validation_sites": 14,
        "c2st_held_out": {
            "deterministic_median": 0.999,
            "deterministic_min": 0.956,
            "stochastic_median": 0.880,
            "stochastic_range": [0.810, 0.920],
            "verdict": "distinguishable: a per-pixel classifier separates emulated from real",
        },
        "variation_note": (
            "B1-B3 variation is the bootstrap half-width (0.031/0.030/0.030). "
            "The build tagged v0.2.0 carried 0.12, inconsistent with the "
            "stated 2.7-3.2 % interval."
        ),
        "note": (
            "The 5.3-6.2 % is the SBAF calibration error with an in-sample "
            "per-site factor, not the v2 output. With v2 the same convention "
            "gives 5.7-7.0 %; without the free factor the desert error is "
            "8.2-14.7 %. Outside the domain the scaled error is 13.4-23.6 % "
            "and the raw error 27.2-60.0 %. B4 scaling 0.933 and gamma 0.112 "
            f"stay frozen at v1: the 29-site fit gives gamma {fitted_gamma:.3f} "
            f"and B4 {fitted_b4:.3f}, and the gas term is still a hypothesis."
        ),
        "v1_note": "v1 was fitted on 5 CEOS PICS, one of which (libya4_ceos) is in the held-out set; the comparison is conservative against v2.",
    }
    payload["provenance"] = {
        "package_sha": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source": "MSS_S2/pics/pics_index.json",
        "source_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        "scales",
        [round(value, 4) for value in profile.band_scaling],
        "gamma",
        round(profile.gamma, 4),
    )
    print(OUT)


if __name__ == "__main__":
    main()
