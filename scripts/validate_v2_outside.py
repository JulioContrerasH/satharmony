#!/usr/bin/env python
"""Measure the out-of-domain error of the v2 profile on the 14 validation sites.

The 14-23 % figure in the profile came from the SBAF validation with an
in-sample per-band scale.  This script repeats the measurement with the v2
emulator and reports both the raw error and the rescaled one, so the two are
directly comparable.
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
from scripts.pics_pairs import PICS, load_pair, valid_mask  # noqa: E402

OUT = REPO.parent / "assets/md/fuentes/srf/pics_49/validation_v2_outside.json"
PROFILE = "mss5_esa_gtc_v2"
BANDS = ("B1", "B2", "B3", "B4")


def validation_sites() -> list[str]:
    index = json.loads((PICS / "pics_index.json").read_text())
    return [entry["site"] for entry in index["sites"] if entry.get("role") == "validation"]


def per_site_stats(sites: list[str], config: PipelineConfig, seed: int | None) -> dict:
    """Per-site errors and their median, matching the old SBAF table."""
    rows = []
    for site in sites:
        pair = load_pair(site)
        valid = valid_mask(site, pair.mss)
        pred, _ = emulate_s2(pair.s2, config=config, seed=seed)
        row = {"site": site, "valid_pixels": int(valid.sum())}
        for index, band in enumerate(BANDS):
            reference = pair.mss[index][valid].astype(np.float64)
            estimate = pred[index][valid].astype(np.float64)
            scale = float(np.dot(estimate, reference) / max(np.dot(estimate, estimate), 1e-12))
            row[f"{band}_rmse_rel"] = float(
                np.sqrt(np.mean((estimate - reference) ** 2)) / np.mean(reference)
            )
            row[f"{band}_rmse_rel_rescaled"] = float(
                np.sqrt(np.mean((scale * estimate - reference) ** 2)) / np.mean(reference)
            )
        rows.append(row)
        print(
            site,
            [round(row[f"{band}_rmse_rel_rescaled"], 3) for band in BANDS],
            flush=True,
        )

    medians = {}
    for band in BANDS:
        raw = np.array([row[f"{band}_rmse_rel"] for row in rows])
        rescaled = np.array([row[f"{band}_rmse_rel_rescaled"] for row in rows])
        medians[band] = {
            "rmse_rel_median": float(np.median(raw)),
            "rmse_rel_rescaled_median": float(np.median(rescaled)),
            "rmse_rel_rescaled_min": float(rescaled.min()),
            "rmse_rel_rescaled_max": float(rescaled.max()),
        }
    return {"median_across_sites": medians, "per_site": rows}


def main() -> None:
    sites = validation_sites()
    print(f"{len(sites)} validation sites", flush=True)
    configs = {
        "deterministic": (deterministic(PipelineConfig.from_profile(PROFILE)), None),
        "stochastic_seed0": (PipelineConfig.from_profile(PROFILE), 0),
    }
    results = {name: per_site_stats(sites, config, seed) for name, (config, seed) in configs.items()}

    payload = {
        "profile": PROFILE,
        "sites": sites,
        "method": (
            "v2 emulation with saturation and range mask; per-site relative "
            "RMSE, median across the 14 sites; rmse_rel_rescaled applies one "
            "in-sample per-site factor, the same convention as the old SBAF "
            "table (sbaf_pics_summary_validation.csv)"
        ),
        "results": results,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    for name in configs:
        bands = results[name]["median_across_sites"]
        print(
            name,
            {
                band: (
                    round(bands[band]["rmse_rel_median"], 3),
                    round(bands[band]["rmse_rel_rescaled_median"], 3),
                )
                for band in BANDS
            },
        )
    print(OUT)


if __name__ == "__main__":
    main()
