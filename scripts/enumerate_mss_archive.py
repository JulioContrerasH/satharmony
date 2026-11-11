#!/usr/bin/env python
"""List every MSS scene of a platform with its centre, sun and cloud cover.

The whole Landsat 2 archive is ~3,000 scenes, so it can be enumerated. That
turns the problem around: instead of hoping that a PICS has a scene, pick the
sites where the archive already has one.

Usage::

    python scripts/enumerate_mss_archive.py --platform L2 --out /tmp/l2_scenes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = "ee-contrerasnetk"
RANGES = {
    "L1": ("1972-07-23", "1978-01-06"),
    "L2": ("1975-01-22", "1982-02-25"),
    "L3": ("1978-03-05", "1983-09-18"),
    "L4": ("1982-07-16", "1993-12-31"),
}
TIERS = {"L1": ("T1",), "L2": ("T1",), "L3": ("T1",), "L4": ("T1", "T2")}


def scenes(asset: str, start: str, end: str, chunk: int = 800) -> list[dict]:
    import ee

    collection = ee.ImageCollection(asset).filterDate(start, end)

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
    total = features.size().getInfo()
    rows = []
    for start_index in range(0, total, chunk):
        payload = features.toList(chunk, start_index).getInfo()
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
                    "asset": asset,
                }
            )
        print(f"{asset}: {len(rows)}/{total}", flush=True)
    return rows


def main() -> int:
    import ee

    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True, choices=sorted(RANGES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    start, end = RANGES[args.platform]
    rows: list[dict] = []
    for tier in TIERS[args.platform]:
        rows += scenes(f"LANDSAT/LM0{args.platform[-1]}/C02/{tier}", start, end)
    args.out.write_text(json.dumps(rows, indent=1))
    print(f"{args.platform}: {len(rows)} scenes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
