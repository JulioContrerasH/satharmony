#!/usr/bin/env python
"""Pick Landsat 1-3 scenes over wet and forested ground.

The desert bank was built on ``C02/T1``, which for Landsat 1-3 holds a few
thousand repatriated scenes.  ``C02/T2`` holds hundreds of thousands, so the
outside-domain bank is possible.  This script writes the site list that
``fetch_aux_sites.py`` consumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee

PROJECT = "ee-contrerasnetk"
CLOUD_MAX = 40.0
SUN_MIN, SUN_MAX = 25.0, 60.0

# Forest, water and wetland.  One point each, spread over the archive.
SITES = {
    "great_lakes": (43.5, -87.0),
    "everglades": (25.6, -80.8),
    "amazon": (-3.0, -60.0),
    "pantanal": (-17.5, -57.0),
    "boreal_canada": (55.0, -105.0),
    "congo": (0.5, 22.0),
    "titicaca": (-15.8, -69.3),
    "okavango": (-19.3, 22.9),
    "borneo": (0.5, 114.0),
    "victoria": (-1.0, 33.0),
    "siberia": (60.0, 90.0),
}


def pick(lat: float, lon: float, platform: str, tier: str) -> dict | None:
    region = ee.Geometry.Point(lon, lat).buffer(12000)
    collection = (
        ee.ImageCollection(f"LANDSAT/{platform}/C02/{tier}")
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUD_COVER", CLOUD_MAX))
    )
    ids = collection.aggregate_array("system:index").getInfo()
    if not ids:
        return None
    suns = collection.aggregate_array("SUN_ELEVATION").getInfo()
    clouds = collection.aggregate_array("CLOUD_COVER").getInfo()
    candidates = [
        index
        for index, sun in enumerate(suns)
        if sun is not None and SUN_MIN <= float(sun) <= SUN_MAX
    ] or list(range(len(ids)))
    best = min(candidates, key=lambda index: float(clouds[index] or 100.0))
    return {
        "id": ids[best],
        "asset": f"LANDSAT/{platform}/C02/{tier}",
        "sun": float(suns[best]),
        "cloud": float(clouds[best] or 0.0),
        "lon": lon,
        "lat": lat,
        "tier": tier,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", required=True, choices=["L1", "L2", "L3"])
    parser.add_argument("--tiers", default="T2,T1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    platform = f"LM0{args.sensor[-1]}"
    rows = []
    for label, (lat, lon) in SITES.items():
        for tier in args.tiers.split(","):
            found = pick(lat, lon, platform, tier)
            if found is None:
                continue
            found["name_hint"] = label
            rows.append(found)
            print(
                f"{label:14s} {tier} {found['id']} sun={found['sun']:.1f} cloud={found['cloud']:.0f}",
                flush=True,
            )
            break
        else:
            print(f"{label:14s} no scene", flush=True)

    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\n{len(rows)} sites -> {args.out}")


if __name__ == "__main__":
    main()
