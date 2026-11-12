#!/usr/bin/env python
"""Check the Landsat 1-3 archive: both tiers, and whether wet sites have scenes.

The earlier bank was built on ``C02/T1`` only and filtered to dry ground.  This
script answers two questions with counts: how many scenes exist per tier, and
whether any of them sit over forest or water.
"""

from __future__ import annotations

import json
from pathlib import Path

import ee

PROJECT = "ee-contrerasnetk"
PLATFORMS = ("LM01", "LM02", "LM03")
TIERS = ("T1", "T2")

# Wet and forested targets, not the desert bank.
SITES = {
    "amazon": (-3.0, -60.0),
    "congo": (0.5, 22.0),
    "borneo": (0.5, 114.0),
    "great_lakes": (43.5, -87.0),
    "pantanal": (-17.5, -57.0),
    "titicaca": (-15.8, -69.3),
    "victoria": (-1.0, 33.0),
    "okavango": (-19.3, 22.9),
    "everglades": (25.6, -80.8),
    "siberia": (60.0, 90.0),
    "boreal_canada": (55.0, -105.0),
    "atlantic": (40.0, -40.0),
}


def main() -> None:
    ee.Initialize(project=PROJECT)
    counts: dict[str, int] = {}
    for platform in PLATFORMS:
        for tier in TIERS:
            name = f"LANDSAT/{platform}/C02/{tier}"
            try:
                collection = ee.ImageCollection(name)
                total = int(collection.size().getInfo())
            except Exception as error:
                print(f"{name}: {error}")
                counts[f"{platform}_{tier}"] = -1
                continue
            counts[f"{platform}_{tier}"] = total
            print(f"{name}: {total} scenes", flush=True)

    print("\nscenes within 12 km, cloud < 40:", flush=True)
    found: dict[str, dict[str, int]] = {}
    for label, (lat, lon) in SITES.items():
        region = ee.Geometry.Point(lon, lat).buffer(12000)
        row = {}
        for platform in PLATFORMS:
            for tier in TIERS:
                if counts.get(f"{platform}_{tier}", 0) <= 0:
                    continue
                collection = (
                    ee.ImageCollection(f"LANDSAT/{platform}/C02/{tier}")
                    .filterBounds(region)
                    .filter(ee.Filter.lt("CLOUD_COVER", 40))
                )
                row[f"{platform}_{tier}"] = int(collection.size().getInfo())
        found[label] = row
        print(label, row, flush=True)

    out = {"counts": counts, "sites": {k: list(v) for k, v in SITES.items()}, "found": found}
    Path("/tmp/probe_l123_wet.json").write_text(json.dumps(out, indent=2))
    print("\nany wet site with a scene:", any(any(v > 0 for v in row.values()) for row in found.values()))


if __name__ == "__main__":
    main()
