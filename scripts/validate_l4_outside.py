#!/usr/bin/env python
"""Measure the out-of-domain error of the Landsat 4 profile.

The L4 profile was fitted on 11 stable deserts.  This script runs it on the
PICS that are outside the calibration domain (forest, water, crops, wetland)
and that also have a Landsat 4 scene, so the L4 row of the README carries a
number instead of a promise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.emulators.s2_to_mss.config import PipelineConfig  # noqa: E402
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402
from scripts.holdout_v2 import deterministic  # noqa: E402
from scripts.make_profile_l4 import PICS, PICS_L4, load_pair  # noqa: E402

OUT = REPO.parent / "assets/md/fuentes/srf/pics_49/validation_l4_outside.json"
PROFILE = "mss4_esa_gtc_v1"
BANDS = ("B1", "B2", "B3", "B4")


def sites() -> list[str]:
    index = json.loads((PICS / "pics_index.json").read_text())
    out = []
    for entry in index["sites"]:
        if entry.get("role") != "validation":
            continue
        if (PICS_L4 / "mss" / entry["site"] / "mss.tif").exists():
            out.append(entry["site"])
    return out


def main() -> None:
    config = PipelineConfig.from_profile(PROFILE)
    names = sites()
    print(f"{len(names)} out-of-domain sites with Landsat 4", flush=True)
    rows = []
    for site in names:
        pair = load_pair(site)
        pred, _ = emulate_s2(pair.s2, config=deterministic(config))
        mask = np.isfinite(pair.mss).all(0) & (pair.mss > 0).all(0) & (pair.mss < 1.5).all(0)
        row: dict[str, float | str] = {"site": site, "valid_pixels": int(mask.sum())}
        for index, band in enumerate(BANDS):
            reference = pair.mss[index][mask].astype(np.float64)
            estimate = pred[index][mask].astype(np.float64)
            scale = float(np.dot(estimate, reference) / max(np.dot(estimate, estimate), 1e-12))
            row[f"{band}_raw"] = float(
                np.sqrt(np.mean((estimate - reference) ** 2)) / np.mean(reference)
            )
            row[f"{band}_rescaled"] = float(
                np.sqrt(np.mean((scale * estimate - reference) ** 2)) / np.mean(reference)
            )
        rows.append(row)
        print(site, [round(float(row[f"{band}_rescaled"]), 3) for band in BANDS], flush=True)

    summary = {
        band: {
            "raw_median": float(np.median([float(r[f"{band}_raw"]) for r in rows])),
            "rescaled_median": float(np.median([float(r[f"{band}_rescaled"]) for r in rows])),
        }
        for band in BANDS
    }
    payload = {
        "profile": PROFILE,
        "sites": names,
        "note": "out-of-domain PICS that also have a Landsat 4 scene",
        "summary": summary,
        "per_site": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print({band: round(summary[band]["rescaled_median"], 3) for band in BANDS})
    print(OUT)


if __name__ == "__main__":
    main()
