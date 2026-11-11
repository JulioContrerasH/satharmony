#!/usr/bin/env python
"""Pick usable sites out of the enumerated MSS archive.

A scene is a candidate only if it is low cloud, its sun is comparable to the
L5 profile (25-60 deg), and the ground under it is flat and dry. Distance
filtering keeps the sites spread out, so the fit is not ten sites on the same
dune.

Usage::

    python scripts/score_archive_sites.py --scenes /tmp/l2_scenes.json \
        --min-distance 1.0 --out /tmp/l2_sites.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = "ee-contrerasnetk"
TARGET_SUN = 41.5
MAX_CLOUD = 40.0


def aridity(points: list[dict]) -> list[dict]:
    """Attach MODIS NDVI, SRTM slope and elevation to every point."""
    import ee

    ndvi = (
        ee.ImageCollection("MODIS/061/MOD13A2")
        .select("NDVI")
        .filterDate("2001-01-01", "2021-01-01")
        .median()
        .multiply(0.0001)
        .rename("ndvi")
    )
    dem = ee.Image("USGS/SRTMGL1_003")
    stack = ndvi.addBands(ee.Terrain.slope(dem).rename("slope")).addBands(dem.rename("elev"))
    collection = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point(point["lon"], point["lat"]), {"idx": index}) for index, point in enumerate(points)]
    )
    sample = stack.sampleRegions(collection=collection, scale=1000, tileScale=4, geometries=False)
    values = {row["properties"]["idx"]: row["properties"] for row in sample.getInfo()["features"]}
    for index, point in enumerate(points):
        row = values.get(index, {})
        point["ndvi"] = row.get("ndvi")
        point["slope"] = row.get("slope")
        point["elev"] = row.get("elev")
    return points


def thin(points: list[dict], min_distance: float) -> list[dict]:
    """Greedy: best first, then only points far enough from those kept."""
    ordered = sorted(points, key=lambda row: (row["cloud"] or 100) + abs((row["sun"] or 0) - TARGET_SUN))
    kept: list[dict] = []
    for point in ordered:
        if all(abs(point["lon"] - other["lon"]) > min_distance or abs(point["lat"] - other["lat"]) > min_distance for other in kept):
            kept.append(point)
    return kept


def main() -> int:
    import ee

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--min-distance", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    scenes = json.loads(args.scenes.read_text())
    candidates = [
        scene
        for scene in scenes
        if scene["cloud"] is not None
        and scene["cloud"] < MAX_CLOUD
        and scene["sun"] is not None
        and 25.0 <= scene["sun"] <= 60.0
    ]
    # One scene per ~25 km cell before spending a GEE call per point.
    cell: dict[tuple[float, float], dict] = {}
    for scene in candidates:
        key = (round(scene["lon"] * 4) / 4, round(scene["lat"] * 4) / 4)
        current = cell.get(key)
        score = (scene["cloud"] or 100) + abs((scene["sun"] or 0) - TARGET_SUN)
        if current is None or score < (current["cloud"] or 100) + abs((current["sun"] or 0) - TARGET_SUN):
            cell[key] = scene
    points = list(cell.values())
    print(f"{len(candidates)} scenes -> {len(points)} cells to sample", flush=True)

    aridity(points)
    dry = [point for point in points if (point["ndvi"] or 1) < 0.15 and (point["slope"] or 99) < 3.0 and (point["elev"] or 9999) < 2500]
    print(f"{len(dry)} dry and flat", flush=True)
    kept = thin(dry, args.min_distance)
    print(f"{len(kept)} after the {args.min_distance} deg spacing", flush=True)
    args.out.write_text(json.dumps(kept, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
