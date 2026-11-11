#!/usr/bin/env python
"""Probe the archive measuring clear sky inside the PICS window, not on the scene.

The Landsat metadata cloud cover is computed over the whole 185 km scene, so a
scene tagged 60 % cloud can still be perfectly clear over our 15 km window.
That filter was throwing away usable scenes. Here the clear fraction is
measured on the exact window with QA_PIXEL, which is the number that matters.

Nothing is downloaded.

Usage::

    python scripts/probe_window.py --platforms L1 L2 L3 L4
    python scripts/probe_window.py --points /tmp/arid.json --platforms L2 L3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PICS = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")
OUT = Path("/data/databases/CLOUD_MSS/MSS_S2/window_probe.json")
PROJECT = "ee-contrerasnetk"
MAX_CANDIDATES = 10

RANGES = {
    "L1": ("1972-07-23", "1978-01-06"),
    "L2": ("1975-01-22", "1982-02-25"),
    "L3": ("1978-03-05", "1983-09-18"),
    "L4": ("1982-07-16", "1993-12-31"),
}
TIERS = {"L1": ("T1",), "L2": ("T1",), "L3": ("T1",), "L4": ("T2", "T1")}


def pics_windows() -> list[dict]:
    """The exact 250 x 250 mss grid of every PICS, as lon/lat corners."""
    index = json.loads((PICS / "pics_index.json").read_text())
    rows = []
    for entry in index["sites"]:
        size = entry["grid"]["mss"]["size"] * 60.0
        x, y = entry["origin_utm"]["x"], entry["origin_utm"]["y"]
        rows.append(
            {
                "name": f"pics:{entry['site']}",
                "crs": entry["crs"],
                "xmin": x,
                "ymax": y,
                "xmax": x + size,
                "ymin": y - size,
                "sun": float(entry["mss"].get("sun_elevation") or 45.0),
            }
        )
    return rows


def collection(platform: str) -> tuple[str, object] | None:
    """Asset id and collection for the platform, first tier that has data."""
    import ee

    start, end = RANGES[platform]
    for tier in TIERS[platform]:
        asset = f"LANDSAT/LM0{platform[-1]}/C02/{tier}"
        coll = ee.ImageCollection(asset).filterDate(start, end)
        try:
            if coll.size().getInfo():
                return asset, coll
        except Exception:
            continue
    return None


def clear_fraction(image, region) -> float:
    """Fraction of the window that is neither cloud, dilated cloud nor fill.

    Bit 6 ("clear") is not populated in the MSS products, so the project
    convention is used: cloud = dilated (bit 1) or cloud (bit 3), plus fill.
    """
    import ee

    qa = image.select("QA_PIXEL")
    cloud = qa.bitwiseAnd(1 << 1).neq(0).Or(qa.bitwiseAnd(1 << 3).neq(0))
    fill = qa.bitwiseAnd(1).neq(0)
    clear = cloud.Or(fill).Not()
    value = clear.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=60, maxPixels=1e6, bestEffort=False
    ).get("QA_PIXEL").getInfo()
    return float(value) if value is not None else 0.0


def probe_point(row: dict, colls: dict, sun_window: float, min_clear: float) -> dict:
    import ee

    region = ee.Geometry.Rectangle(
        [row["xmin"], row["ymin"], row["xmax"], row["ymax"]], proj=row["crs"], geodesic=False
    )
    out = {"name": row["name"], "sun": row["sun"], "sensors": {}}
    for platform, (asset, coll) in colls.items():
        near = coll.filterBounds(region)
        count = near.size().getInfo()
        if not count:
            out["sensors"][platform] = {"n": 0, "usable": []}
            continue
        suns = near.aggregate_array("SUN_ELEVATION").getInfo()
        ids = near.aggregate_array("system:index").getInfo()
        clouds = near.aggregate_array("CLOUD_COVER").getInfo()
        order = sorted(range(len(ids)), key=lambda i: abs(float(suns[i]) - row["sun"]))[:MAX_CANDIDATES]
        usable = []
        for index in order:
            sun = float(suns[index])
            if abs(sun - row["sun"]) > sun_window:
                continue
            image = ee.Image(f"{asset}/{ids[index]}")
            clear = clear_fraction(image, region)
            if clear >= min_clear:
                sid = ids[index]
                usable.append(
                    {
                        "id": sid,
                        "date": f"{sid[-8:-4]}-{sid[-4:-2]}-{sid[-2:]}",
                        "sun": sun,
                        "scene_cloud": float(clouds[index]),
                        "window_clear": clear,
                    }
                )
        out["sensors"][platform] = {"n": int(count), "usable": usable}
    return out


def main() -> int:
    import ee

    parser = argparse.ArgumentParser()
    parser.add_argument("--platforms", nargs="+", default=["L1", "L2", "L3", "L4"])
    parser.add_argument("--points", type=Path, default=None, help="JSON list of windows (arid sample)")
    parser.add_argument("--sun-window", type=float, default=15.0)
    parser.add_argument("--min-clear", type=float, default=0.98)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    rows = pics_windows()
    if args.points:
        rows += json.loads(args.points.read_text())
    colls = {platform: collection(platform) for platform in args.platforms}
    colls = {key: value for key, value in colls.items() if value is not None}
    print(f"{len(rows)} windows, collections {list(colls)}", flush=True)

    results = []
    for row in rows:
        results.append(probe_point(row, colls, args.sun_window, args.min_clear))
        usable = sum(1 for platform in colls if results[-1]["sensors"][platform]["usable"])
        print(f"{row['name']}: {usable} sensor(s) usable", flush=True)

    args.out.write_text(
        json.dumps(
            {"sun_window": args.sun_window, "min_clear": args.min_clear, "points": results},
            indent=2,
        )
    )
    print(f"\n{len(results)} windows written to {args.out}")
    for platform in colls:
        hits = sum(1 for row in results if row["sensors"][platform]["usable"])
        print(f"  {platform}: {hits}/{len(results)} with a clear scene in the window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
