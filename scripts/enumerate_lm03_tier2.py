#!/usr/bin/env python
"""Enumerate L3 scenes of Tier 2, filtered on the server.

Tier 1 of Landsat 3 is 180 scenes; Tier 2 is 156.671. The desert bank for L3
came only from Tier 1, which is why it could not grow past 14 sites.

The filter runs in Earth Engine, not after the download: only scenes with low
cloud and a sun comparable to the L5 profile are transferred. Without it the
payload is ~156k features and minutes of transfer for nothing.

Usage::

    python scripts/enumerate_lm03_tier2.py --out /tmp/l3_t2_scenes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = "ee-contrerasnetk"
ASSET = "LANDSAT/LM03/C02/T2"
START, END = "1978-03-05", "1983-09-18"
MAX_CLOUD = 25.0
SUN = (25.0, 60.0)
CHUNK = 1000


def main() -> int:
    import ee

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    collection = (
        ee.ImageCollection(ASSET)
        .filterDate(START, END)
        .filter(ee.Filter.lt("CLOUD_COVER", MAX_CLOUD))
        .filter(ee.Filter.gte("SUN_ELEVATION", SUN[0]))
        .filter(ee.Filter.lte("SUN_ELEVATION", SUN[1]))
    )
    total = collection.size().getInfo()
    print(f"{total} escenas candidatas en {ASSET}", flush=True)

    def to_feature(image):
        centre = image.geometry().centroid(1)
        return ee.Feature(
            centre,
            {
                "id": image.get("system:index"),
                "sun": image.get("SUN_ELEVATION"),
                "cloud": image.get("CLOUD_COVER"),
            },
        )

    features = collection.map(to_feature)
    rows: list[dict] = []
    for start in range(0, total, CHUNK):
        payload = features.toList(CHUNK, start).getInfo()
        for feature in payload:
            props = feature["properties"]
            lon, lat = feature["geometry"]["coordinates"]
            rows.append(
                {
                    "id": props["id"],
                    "lon": round(lon, 5),
                    "lat": round(lat, 5),
                    "sun": props.get("sun"),
                    "cloud": props.get("cloud"),
                    "asset": ASSET,
                }
            )
        print(f"{len(rows)}/{total}", flush=True)

    args.out.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} escenas -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
