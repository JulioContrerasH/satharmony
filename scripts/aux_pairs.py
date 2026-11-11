#!/usr/bin/env python
"""Load the auxiliary site pairs (S2 L1C composite + MSS TOA) and fit them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.calibration.fit import Pair  # noqa: E402
from satharmony.emulators.s2_to_mss.sbaf import S2_BAND_ORDER  # noqa: E402

ROOT = Path("/data/databases/CLOUD_MSS/MSS_S2")


def root(sensor: str) -> Path:
    return ROOT / f"aux_{sensor.lower()}"


def load_pair(sensor: str, name: str) -> Pair:
    """S2 TOA in [0, 1] and MSS TOA for one auxiliary site, with its clear mask."""
    base = root(sensor)
    s2 = tifffile.imread(base / "s2" / f"{name}.tif").astype(np.float32) / 10000.0
    mss = tifffile.imread(base / "mss" / f"{name}.tif").astype(np.float32)
    mask = valid_mask(sensor, name, mss)
    return Pair(s2=s2, mss=mss, valid=mask, name=name)


def valid_mask(sensor: str, name: str, mss: np.ndarray | None = None) -> np.ndarray:
    """Pixels clear in the MSS window and filled in the S2 composite."""
    base = root(sensor)
    mask = tifffile.imread(base / "mask" / f"{name}.tif").astype(bool)
    reference = mss if mss is not None else tifffile.imread(base / "mss" / f"{name}.tif")
    mask &= np.all(np.isfinite(reference), axis=0)
    mask &= np.all(reference > 0.0, axis=0)
    mask &= np.all(reference < 1.5, axis=0)
    return mask


def sites(sensor: str) -> list[dict]:
    path = root(sensor) / "meta.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    return [row for row in rows if (root(sensor) / "mss" / f"{row['site']}.tif").exists()]


__all__ = ["S2_BAND_ORDER", "load_pair", "root", "sites", "valid_mask"]
