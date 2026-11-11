#!/usr/bin/env python
"""Fetch one Landsat 1-3 MSS scene per PICS, on the exact PICS grid.

Landsat 1-3 used WRS-1, so the scenes are found by geometry, not by the
WRS-2 path/row of the L5 PICS. The band order is B4-B7; it is renamed to
B1-B4 so the rest of the pipeline is unchanged.

Usage::

    python scripts/fetch_pics_l123.py --sensor L1 --limit 1
    python scripts/fetch_pics_l123.py --sensor L1
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
PROJECT = "ee-contrerasnetk"
CLOUD_MAX = 20.0
BANDS = ("B4", "B5", "B6", "B7")
NAMES = ("B1", "B2", "B3", "B4")


def output(sensor: str) -> Path:
    return Path(f"/data/databases/CLOUD_MSS/MSS_S2/pics_{sensor.lower()}")


def find_scene(entry: dict, sensor: str):
    import ee

    centre = entry["centre_wgs84"]
    point = ee.Geometry.Point(centre["lon"], centre["lat"]).buffer(10000)
    target_sun = float(entry["mss"].get("sun_elevation") or 45.0)
    platform = f"LM0{sensor[-1]}"
    scenes = (
        ee.ImageCollection(f"LANDSAT/{platform}/C02/T1")
        .filterDate("1972-07-01", "1984-01-01")
        .filterBounds(point)
        .filter(ee.Filter.lt("CLOUD_COVER", CLOUD_MAX))
    )
    ids = scenes.aggregate_array("system:index").getInfo()
    if not ids:
        return None
    suns = scenes.aggregate_array("SUN_ELEVATION").getInfo()
    clouds = scenes.aggregate_array("CLOUD_COVER").getInfo()
    best = min(range(len(ids)), key=lambda i: abs(float(suns[i]) - target_sun))
    sid = ids[best]
    date = f"{sid[-8:-4]}-{sid[-4:-2]}-{sid[-2:]}"
    return sid, float(suns[best]), date, float(clouds[best])


def to_toa(dn: np.ndarray, date: str, sun: float, sensor: str) -> np.ndarray:
    """DN -> TOA with the Landsat 1-3 constants. QCALMAX from the data."""
    qcal = 127 if dn.max() <= 127 else 255
    calib = MSSCalibration(
        spacecraft_id=f"LANDSAT_{sensor[-1]}",
        sensor_id="MSS",
        date_acquired=datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        sun_elevation=sun,
        radiance_mult=[value / qcal for value in LMAX_MSS[f"{sensor}_MSS"]],
        radiance_add=[0.0, 0.0, 0.0, 0.0],
        qcal_min=[0, 0, 0, 0],
        qcal_max=[qcal] * 4,
    )
    return dn_to_toa(dn.astype(np.uint8), calib)


def main() -> int:
    import cubexpress
    import ee

    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", choices=["L1", "L2", "L3"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--nworkers", type=int, default=8)
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    out = output(args.sensor)
    index = json.loads((PICS / "pics_index.json").read_text())
    entries = index["sites"]
    if args.limit:
        entries = entries[: args.limit]

    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else []
    known = {row["site"] for row in meta}

    rows = []
    for entry in entries:
        site = entry["site"]
        if site in known:
            continue
        found = find_scene(entry, args.sensor)
        if found is None:
            continue
        asset, sun, date, cloud = found
        platform = f"LM0{args.sensor[-1]}"
        image = ee.Image(f"LANDSAT/{platform}/C02/T1/{asset}").select(list(BANDS), list(NAMES))
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
                image=image,
                bands=NAMES,
            )
        )
        meta.append(
            {
                "site": site,
                "sensor": args.sensor,
                "asset": f"LANDSAT/LM0{args.sensor[-1]}/C02/T1/{asset}",
                "date": date,
                "sun_elevation": sun,
                "cloud_cover": cloud,
            }
        )
        print(f"{site}: {asset} {date} cloud={cloud:.1f}", flush=True)

    if rows:
        (out / "dn").mkdir(parents=True, exist_ok=True)
        print(cubexpress.express(cubexpress.RequestTable(rows), out / "dn", nworkers=args.nworkers))
        meta_path.write_text(json.dumps(meta, indent=2))

    (out / "mss").mkdir(parents=True, exist_ok=True)
    for row in meta:
        source = out / "dn" / f"{row['site']}.tif"
        if not source.exists():
            continue
        dn = tifffile.imread(source)
        if dn.ndim == 3 and dn.shape[-1] == 4:
            dn = np.moveaxis(dn, -1, 0)
        toa = to_toa(dn, row["date"], row["sun_elevation"], args.sensor)
        target = out / "mss" / row["site"] / "mss.tif"
        target.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(target, toa.astype(np.float32))
        print(f"{row['site']}: TOA {toa.shape} max {toa.max():.3f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
