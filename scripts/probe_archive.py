#!/usr/bin/env python
"""Probe the Landsat archive before downloading anything.

For every candidate point and every MSS platform, report how many cloud-free
scenes exist, how far the sun elevation is from the target, and the scene ids.
Metadata only: nothing is downloaded, so a wrong hunch costs seconds.

Points come from two places:

- the 49 PICS of ``pics_index.json`` (validates the probe against the counts
  already known from the fetchers);
- an optional arid sample generated in GEE (``--arid N``), used to estimate
  what happens if the PICS bank grows.

Usage::

    python scripts/probe_archive.py                     # only the 49 PICS
    python scripts/probe_archive.py --arid 300          # PICS + 300 arid points
    python scripts/probe_archive.py --arid 300 --platforms L2 L3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PICS = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")
OUT = Path("/data/databases/CLOUD_MSS/MSS_S2/archive_probe.json")
PROJECT = "ee-contrerasnetk"
CLOUD_MAX = 20.0
SUN_WINDOW = 10.0
MAX_SCENES_PER_POINT = 12

RANGES = {
    "L1": ("1972-07-23", "1978-01-06"),
    "L2": ("1975-01-22", "1982-02-25"),
    "L3": ("1978-03-05", "1983-09-18"),
    "L4": ("1982-07-16", "1993-12-31"),
    "L5": ("1984-03-01", "2013-06-30"),
}
#: L1-L3 used WRS-1 and only T1 exists; L4/L5 fall back to T2.
COLLECTIONS = {"L1": ("T1",), "L2": ("T1",), "L3": ("T1",), "L4": ("T2", "T1"), "L5": ("T2", "T1")}


def pics_points() -> list[dict]:
    index = json.loads((PICS / "pics_index.json").read_text())
    rows = []
    for entry in index["sites"]:
        centre = entry["centre_wgs84"]
        rows.append(
            {
                "name": f"pics:{entry['site']}",
                "lon": float(centre["lon"]),
                "lat": float(centre["lat"]),
                "sun": float(entry["mss"].get("sun_elevation") or 45.0),
            }
        )
    return rows


def arid_points(count: int) -> list[dict]:
    """Sample flat, dry, low land: where an MSS scene is likely to be useful.

    NDVI from MODIS (2001-2020 median), slope from SRTM, and elevation. The
    sample is random but seeded, so the same run gives the same points.
    """
    import ee

    ndvi = (
        ee.ImageCollection("MODIS/061/MOD13A2")
        .select("NDVI")
        .filterDate("2001-01-01", "2021-01-01")
        .median()
        .multiply(0.0001)
    )
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
    elev = ee.Image("USGS/SRTMGL1_003")
    mask = ndvi.lt(0.15).And(slope.lt(3.0)).And(elev.lt(2500)).selfMask()
    region = ee.Geometry.Rectangle([-180, -56, 180, 72], geodesic=False)
    sample = mask.sample(region=region, scale=40000, projection="EPSG:4326", numPixels=count, seed=7, geometries=True)
    rows = []
    for feature in sample.getInfo()["features"]:
        coords = feature["geometry"]["coordinates"]
        rows.append({"name": f"arid:{coords[0]:.2f},{coords[1]:.2f}", "lon": coords[0], "lat": coords[1], "sun": 45.0})
    return rows


def collection(platform: str):
    """Cloud-free collection for the platform, T2 first when it exists."""
    import ee

    start, end = RANGES[platform]
    for tier in COLLECTIONS[platform]:
        coll = (
            ee.ImageCollection(f"LANDSAT/LM0{platform[-1]}/C02/{tier}")
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", CLOUD_MAX))
        )
        try:
            if coll.size().getInfo():
                return coll
        except Exception:
            continue
    return None


def probe(points: list[dict], platforms: list[str], batch: int) -> list[dict]:
    import ee

    colls = {platform: collection(platform) for platform in platforms}
    colls = {key: value for key, value in colls.items() if value is not None}
    features = ee.FeatureCollection(
        [
            ee.Feature(ee.Geometry.Point(point["lon"], point["lat"]), {"name": point["name"], "sun": point["sun"]})
            for point in points
        ]
    )

    def annotate(feature):
        target = ee.Number(feature.get("sun"))
        out = feature
        for platform, coll in colls.items():
            near = coll.filterBounds(feature.geometry())
            markers = near.map(lambda image: image.set("sun_dist", ee.Number(image.get("SUN_ELEVATION")).subtract(target).abs()))
            ordered = markers.sort("sun_dist")
            out = out.set(f"{platform}_n", ordered.size())
            out = out.set(f"{platform}_sun", ordered.aggregate_array("SUN_ELEVATION").slice(0, MAX_SCENES_PER_POINT))
            out = out.set(f"{platform}_cloud", ordered.aggregate_array("CLOUD_COVER").slice(0, MAX_SCENES_PER_POINT))
            out = out.set(f"{platform}_ids", ordered.aggregate_array("system:index").slice(0, MAX_SCENES_PER_POINT))
        return out

    rows = []
    for start in range(0, len(points), batch):
        chunk = features.toList(batch, start)
        payload = ee.FeatureCollection(chunk).map(annotate).getInfo()["features"]
        for feature in payload:
            props = feature["properties"]
            coords = feature["geometry"]["coordinates"]
            row = {"name": props["name"], "lon": coords[0], "lat": coords[1], "sun": props["sun"], "sensors": {}}
            for platform in colls:
                suns = [float(value) for value in (props.get(f"{platform}_sun") or [])]
                clouds = [float(value) for value in (props.get(f"{platform}_cloud") or [])]
                ids = props.get(f"{platform}_ids") or []
                usable = [
                    {"id": sid, "sun": sun, "cloud": cloud}
                    for sid, sun, cloud in zip(ids, suns, clouds, strict=False)
                    if abs(sun - float(props["sun"])) <= SUN_WINDOW
                ]
                row["sensors"][platform] = {"n": int(props.get(f"{platform}_n") or 0), "usable": usable}
            rows.append(row)
        print(f"probed {len(rows)}/{len(points)}", flush=True)
    return rows


def summarise(rows: list[dict], platforms: list[str]) -> None:
    pics = [row for row in rows if row["name"].startswith("pics:")]
    arid = [row for row in rows if row["name"].startswith("arid:")]
    for group, subset in (("PICS", pics), ("arid", arid)):
        if not subset:
            continue
        print(f"\n{group}: {len(subset)} points")
        for platform in platforms:
            with_scene = sum(1 for row in subset if row["sensors"][platform]["usable"])
            print(f"  {platform}: {with_scene} points with a usable scene ({with_scene / len(subset):.0%})")


def main() -> int:
    import ee

    parser = argparse.ArgumentParser()
    parser.add_argument("--arid", type=int, default=0)
    parser.add_argument("--platforms", nargs="+", default=["L1", "L2", "L3", "L4", "L5"])
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    points = pics_points()
    if args.arid:
        points += arid_points(args.arid)
    print(f"{len(points)} points", flush=True)

    rows = probe(points, args.platforms, args.batch)
    args.out.write_text(json.dumps({"sun_window": SUN_WINDOW, "cloud_max": CLOUD_MAX, "points": rows}, indent=2))
    summarise(rows, args.platforms)
    print(f"\nwritten {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
