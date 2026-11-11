#!/usr/bin/env python
"""Fetch one Landsat 4 MSS scene per PICS, on the exact PICS grid.

The S2 side is reused from ``MSS_S2/pics/<site>/s2``. Only the MSS scene is
new: a cloud-free Landsat 4 pass over the same WRS path/row, resampled by GEE
onto the PICS UTM grid (250 x 250 at 60 m), so no alignment is needed.

Outputs
-------
- ``pics_l4/dn/<site>.tif``   raw DN, 4 bands (CubeXpress).
- ``pics_l4/mss/<site>/mss.tif``  TOA reflectance, (4, 250, 250), float32.
- ``pics_l4/meta.json``       asset, date, sun elevation, cloud cover per site.

Usage::

    python scripts/fetch_pics_l4.py --limit 1
    python scripts/fetch_pics_l4.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.corrections.landsat.constants import LMAX_MSS  # noqa: E402
from satharmony.corrections.landsat.metadata import MSSCalibration  # noqa: E402
from satharmony.corrections.landsat.mss import dn_to_toa  # noqa: E402

PICS = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")
OUT = Path("/data/databases/CLOUD_MSS/MSS_S2/pics_l4")
PROJECT = "ee-contrerasnetk"
CLOUD_MAX = 20.0
QCAL_MAX = 255


def find_scene(entry: dict):
    """Cloud-free Landsat 4 pass at the PICS path/row, closest in sun height.

    Matching the L5 sun elevation keeps the surface brightness comparable and
    avoids the summer saturation of the desert targets.
    """
    import ee

    path = int(entry["mss"]["wrs_path"])
    row = int(entry["mss"]["wrs_row"])
    target_sun = float(entry["mss"].get("sun_elevation") or 45.0)
    for collection in ("T2", "T1"):
        scenes = (
            ee.ImageCollection(f"LANDSAT/LM04/C02/{collection}")
            .filterDate("1982-01-01", "1994-01-01")
            .filter(ee.Filter.eq("WRS_PATH", path))
            .filter(ee.Filter.eq("WRS_ROW", row))
            .filter(ee.Filter.lt("CLOUD_COVER", CLOUD_MAX))
        )
        ids = scenes.aggregate_array("system:index").getInfo()
        if not ids:
            continue
        suns = scenes.aggregate_array("SUN_ELEVATION").getInfo()
        clouds = scenes.aggregate_array("CLOUD_COVER").getInfo()
        best = min(range(len(ids)), key=lambda i: abs(float(suns[i]) - target_sun))
        sid = ids[best]
        date = f"{sid[-8:-4]}-{sid[-4:-2]}-{sid[-2:]}"
        return sid, float(suns[best]), date, float(clouds[best]), collection
    return None


def to_toa(dn: np.ndarray, date: str, sun: float) -> np.ndarray:
    """DN -> TOA with the Landsat 4 constants (no MTL needed)."""
    calib = MSSCalibration(
        spacecraft_id="LANDSAT_4",
        sensor_id="MSS",
        date_acquired=datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        sun_elevation=sun,
        radiance_mult=[value / QCAL_MAX for value in LMAX_MSS["L4_MSS"]],
        radiance_add=[0.0, 0.0, 0.0, 0.0],
        qcal_min=[0, 0, 0, 0],
        qcal_max=[QCAL_MAX] * 4,
    )
    return dn_to_toa(dn.astype(np.uint8), calib)


def main() -> int:
    import cubexpress
    import ee

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--nworkers", type=int, default=8)
    parser.add_argument("--convert-only", action="store_true")
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    index = json.loads((PICS / "pics_index.json").read_text())
    entries = index["sites"]
    if args.limit:
        entries = entries[: args.limit]

    meta_path = OUT / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else []
    known = {row["site"] for row in meta}

    if not args.convert_only:
        rows = []
        for entry in entries:
            site = entry["site"]
            if site in known:
                continue
            found = find_scene(entry)
            if found is None:
                print(f"no L4 scene: {site}", flush=True)
                continue
            asset, sun, date, cloud, collection = found
            image = ee.Image(f"LANDSAT/LM04/C02/{collection}/{asset}")
            transform = cubexpress.RasterTransform(
                crs=entry["crs"],
                translate_x=entry["origin_utm"]["x"],
                translate_y=entry["origin_utm"]["y"],
                scale_x=60.0,
                scale_y=-60.0,
                width=entry["grid"]["mss"]["size"],
                height=entry["grid"]["mss"]["size"],
            )
            rows.append(
                cubexpress.RequestRow(
                    id=site,
                    raster_transform=transform,
                    image=image.select(["B1", "B2", "B3", "B4"]),
                    bands=("B1", "B2", "B3", "B4"),
                )
            )
            meta.append(
                {
                    "site": site,
                    "asset": f"LANDSAT/LM04/C02/{collection}/{asset}",
                    "date": date,
                    "sun_elevation": sun,
                    "cloud_cover": cloud,
                }
            )
            print(f"{site}: {asset} {date} cloud={meta[-1]['cloud_cover']:.1f}", flush=True)

        if rows:
            (OUT / "dn").mkdir(parents=True, exist_ok=True)
            print(cubexpress.express(cubexpress.RequestTable(rows), OUT / "dn", nworkers=args.nworkers))
            meta_path.write_text(json.dumps(meta, indent=2))

    (OUT / "mss").mkdir(parents=True, exist_ok=True)
    for row in meta:
        source = OUT / "dn" / f"{row['site']}.tif"
        if not source.exists():
            continue
        dn = tifffile.imread(source)
        if dn.ndim == 3 and dn.shape[-1] == 4:
            dn = np.moveaxis(dn, -1, 0)
        if dn.max() > QCAL_MAX:
            print(f"{row['site']}: DN max {dn.max()} > {QCAL_MAX}, skipped")
            continue
        toa = to_toa(dn, row["date"], row["sun_elevation"])
        target = OUT / "mss" / row["site"] / "mss.tif"
        target.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(target, toa.astype(np.float32))
        print(f"{row['site']}: TOA {toa.shape} max {toa.max():.3f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
