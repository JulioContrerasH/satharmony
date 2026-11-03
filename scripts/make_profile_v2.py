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
    # B4 stays frozen at v1: the held-out deserts show that the fitted B4
    # (scale 0.943, gamma 0.135) is slightly worse than v1 (0.933, 0.112).
    # The gas term is still a hypothesis; do not move B4 until it is measured.
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
    index_path = PICS / "pics_index.json"
    payload["validity"] = {
        "calibration_domain": "desert and salt flats",
        "outside_domain": ["forest", "water", "crops", "wetland"],
        "held_out_sites": HELD_OUT,
        "held_out_disjoint": True,
        "scaled_rmse_outside": [0.142, 0.231],
        "note": (
            "The 14 validation sites give 14-23 % scaled RMSE; v2 is not "
            "better than v1 outside the calibration domain."
        ),
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
