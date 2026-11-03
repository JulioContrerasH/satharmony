#!/usr/bin/env python
"""In-sample desert error of the v2 profile on its 29 calibration sites.

This is the number that must be compared with the 5.3-6.2 % in the README:
it is fitted, not validated.  The held-out deserts are in ``c2st_v2.json``.
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
from scripts.make_profile_v2 import HELD_OUT  # noqa: E402
from scripts.pics_pairs import PICS, load_pair, valid_mask  # noqa: E402

OUT = REPO.parent / "assets/md/fuentes/srf/pics_49/desert_v2_insample.json"
PROFILE = "mss5_esa_gtc_v2"
BANDS = ("B1", "B2", "B3", "B4")


def calibration_sites() -> list[str]:
    index = json.loads((PICS / "pics_index.json").read_text())
    return [
        entry["site"]
        for entry in index["sites"]
        if entry.get("role", "calibration") == "calibration"
        and entry["site"] != "uyuni"
        and entry["site"] not in HELD_OUT
    ]


def main() -> None:
    sites = calibration_sites()
    print(f"{len(sites)} calibration sites", flush=True)
    config = deterministic(PipelineConfig.from_profile(PROFILE))
    rows = []
    for site in sites:
        pair = load_pair(site)
        valid = valid_mask(site, pair.mss)
        pred, _ = emulate_s2(pair.s2, config=config)
        row = {"site": site}
        for index, band in enumerate(BANDS):
            reference = pair.mss[index][valid].astype(np.float64)
            estimate = pred[index][valid].astype(np.float64)
            scale = float(np.dot(estimate, reference) / max(np.dot(estimate, estimate), 1e-12))
            row[band] = float(
                np.sqrt(np.mean((estimate - reference) ** 2)) / np.mean(reference)
            )
            row[f"{band}_rescaled"] = float(
                np.sqrt(np.mean((scale * estimate - reference) ** 2)) / np.mean(reference)
            )
        rows.append(row)
        print(site, [round(row[band], 3) for band in BANDS], flush=True)

    medians = {
        band: {
            "median": float(np.median([row[band] for row in rows])),
            "rescaled_median": float(np.median([row[f"{band}_rescaled"] for row in rows])),
            "min": float(np.min([row[band] for row in rows])),
            "max": float(np.max([row[band] for row in rows])),
        }
        for band in BANDS
    }
    payload = {
        "profile": PROFILE,
        "sites": sites,
        "note": "in-sample: these 29 sites produced the v2 band scaling",
        "results": {"median_across_sites": medians, "per_site": rows},
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print({band: round(medians[band]["median"], 4) for band in BANDS})
    print(OUT)


if __name__ == "__main__":
    main()
