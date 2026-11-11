#!/usr/bin/env python
"""C2ST of the Landsat 4 profile on its two held-out deserts."""

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
from scripts.c2st_v2 import band_rmse_rel, c2st, summarize  # noqa: E402
from scripts.holdout_v2 import deterministic  # noqa: E402
from scripts.make_profile_l4 import load_pair  # noqa: E402

OUT = REPO.parent / "assets/md/fuentes/srf/pics_49/c2st_l4.json"
PROFILE = "mss4_esa_gtc_v1"
HELD_OUT = ["libya4_ceos", "oman1"]


def main() -> None:
    configs = {
        "deterministic": (deterministic(PipelineConfig.from_profile(PROFILE)), None),
        "stochastic_seed0": (PipelineConfig.from_profile(PROFILE), 0),
    }
    per_site: dict[str, list[dict[str, object]]] = {name: [] for name in configs}
    for site in HELD_OUT:
        pair = load_pair(site)
        mask = np.isfinite(pair.mss).all(0) & (pair.mss > 0).all(0) & (pair.mss < 1.5).all(0)
        for name, (config, seed) in configs.items():
            pred, _ = emulate_s2(pair.s2, config=config, seed=seed)
            row: dict[str, object] = {
                "site": site,
                "valid_pixels": int(mask.sum()),
                "c2st": c2st(pred, pair.mss, valid=mask),
                "rmse_rel": band_rmse_rel(pred, pair.mss, mask),
                **summarize(pred, pair.mss, mask),
            }
            per_site[name].append(row)
            print(name, site, "c2st", round(float(row["c2st"]), 3), flush=True)

    summary = {}
    for name, rows in per_site.items():
        summary[name] = {
            "c2st_median": float(np.median([float(row["c2st"]) for row in rows])),
            "rmse_rel_median": np.median(
                np.stack([np.array(row["rmse_rel"], dtype=float) for row in rows]), axis=0
            )
            .round(4)
            .tolist(),
        }
    payload = {
        "profile": PROFILE,
        "held_out_sites": HELD_OUT,
        "note": "only two held-out deserts: the C2ST is reported with that caveat",
        "summary": summary,
        "per_site": per_site,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
