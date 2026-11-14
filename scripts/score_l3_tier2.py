#!/usr/bin/env python
"""Score L3 Tier 2 candidates in chunks so Earth Engine does not time out.

``score_archive_sites.py`` samples MODIS/SRTM over every 0.25 deg cell at once.
With 45k Tier 2 scenes that is thousands of points in a single call and the
server gives up. Here the best candidates are taken first, one per 0.5 deg cell,
and the sampling runs in batches of 250 with a retry.

    python scripts/score_l3_tier2.py --scenes /tmp/l3_t2_scenes.json \
        --top 1500 --out /tmp/l3_t2_sites.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROJECT = "ee-contrerasnetk"
TARGET_SUN = 41.5
MAX_CLOUD = 40.0
BATCH = 250


def score(point: dict) -> float:
    return (point["cloud"] or 100) + abs((point["sun"] or 0) - TARGET_SUN)


def thin(points: list[dict], min_distance: float) -> list[dict]:
    kept: list[dict] = []
    for point in sorted(points, key=score):
        if all(
            abs(point["lon"] - other["lon"]) > min_distance
            or abs(point["lat"] - other["lat"]) > min_distance
            for other in kept
        ):
            kept.append(point)
    return kept


def aridity(points: list[dict], batch: int = BATCH) -> None:
    """Attach NDVI, slope and elevation, in batches, retrying on timeouts."""
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

    for start in range(0, len(points), batch):
        chunk = points[start : start + batch]
        collection = ee.FeatureCollection(
            [
                ee.Feature(ee.Geometry.Point(p["lon"], p["lat"]), {"idx": start + i})
                for i, p in enumerate(chunk)
            ]
        )
        for attempt in range(3):
            try:
                sample = stack.sampleRegions(
                    collection=collection, scale=1000, tileScale=8, geometries=False
                ).getInfo()
                break
            except Exception as error:  # EE raises generic timeouts
                print(f"  lote {start}: {type(error).__name__} (intento {attempt + 1})", flush=True)
                time.sleep(5 + 5 * attempt)
        else:
            print(f"  lote {start}: se deja sin NDVI", flush=True)
            continue
        for row in sample["features"]:
            props = row["properties"]
            index = int(props["idx"])
            points[index].update(
                {"ndvi": props.get("ndvi"), "slope": props.get("slope"), "elev": props.get("elev")}
            )
        print(f"  lote {start + len(chunk)}/{len(points)}", flush=True)


def main() -> int:
    import ee

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--top", type=int, default=1500)
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
    cell: dict[tuple[float, float], dict] = {}
    for scene in candidates:
        key = (round(scene["lon"] * 2) / 2, round(scene["lat"] * 2) / 2)
        current = cell.get(key)
        if current is None or score(scene) < score(current):
            cell[key] = scene
    print(f"{len(candidates)} escenas -> {len(cell)} celdas de 0.5 grados", flush=True)

    points = sorted(cell.values(), key=score)[: args.top]
    print(f"se puntúan las {len(points)} mejores", flush=True)
    aridity(points)

    dry = [
        point
        for point in points
        if (point.get("ndvi") or 1) < 0.15
        and (point.get("slope") or 99) < 3.0
        and (point.get("elev") or 9999) < 2500
    ]
    print(f"{len(dry)} secos y planos", flush=True)
    kept = thin(dry, args.min_distance)
    print(f"{len(kept)} tras separar {args.min_distance} grados", flush=True)
    args.out.write_text(json.dumps(kept, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
