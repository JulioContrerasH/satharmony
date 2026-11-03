#!/usr/bin/env python
"""Fit the v2 profile from the 49-pixel S2/MSS calibration set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.calibration.fit import Pair, fit_profile  # noqa: E402
from satharmony.emulators.s2_to_mss.sbaf import S2_BAND_ORDER  # noqa: E402

PICS = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")
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
        if entry.get("role", "calibration") == "calibration" and entry["site"] != "uyuni"
    ]
    print(f"{len(sites)} calibration sites")
    pairs = (load_pair(site) for site in sites)
    profile = fit_profile(pairs, name="mss5_esa_gtc_v2", n_boot=1000)
    payload = profile.to_dict()
    payload["config"] = profile.to_config().model_dump(mode="json")
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
