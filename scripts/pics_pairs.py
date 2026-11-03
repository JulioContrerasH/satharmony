#!/usr/bin/env python
"""Load S2/MSS PICS pairs with the mask of usable pixels."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.calibration.fit import Pair  # noqa: E402
from satharmony.emulators.s2_to_mss.sbaf import S2_BAND_ORDER  # noqa: E402

PICS = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")


def read_band(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as handle:
        return handle.read(1).astype(np.float32)


def load_pair(site: str) -> Pair:
    """S2 TOA in [0, 1] and MSS TOA for one PICS site."""
    root = PICS / site
    s2 = np.stack(
        [
            read_band(sorted((root / "s2").glob(f"*.{band}.tif"))[0]) / 10000.0
            for band in S2_BAND_ORDER
        ]
    )
    mss = np.stack(
        [read_band(sorted((root / "mss").glob(f"*.B{band}.tif"))[0]) for band in range(1, 5)]
    )
    return Pair(s2=s2, mss=mss, name=site)


def valid_mask(site: str, mss: np.ndarray) -> np.ndarray:
    """Pixels that are not saturated and stay inside the TOA range."""
    saturated = read_band(sorted((PICS / site / "mss").glob("*.MSK_SATURATED.tif"))[0])
    mask = saturated == 0
    mask &= np.all(np.isfinite(mss), axis=0)
    mask &= np.all(mss > 0.0, axis=0)
    mask &= np.all(mss < 1.5, axis=0)
    return mask
