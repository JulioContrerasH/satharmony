#!/usr/bin/env python
"""C2ST of the v2 profile on the five held-out deserts.

Runs the classifier two-sample test described in ``calibration.metrics`` with
both a deterministic and a seeded stochastic emulation, and writes the table
that backs the claim in the profile.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.calibration.metrics import c2st, summarize  # noqa: E402
from satharmony.emulators.s2_to_mss.config import PipelineConfig  # noqa: E402
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402
from scripts.holdout_v2 import deterministic  # noqa: E402
from scripts.make_profile_v2 import HELD_OUT  # noqa: E402
from scripts.pics_pairs import load_pair, valid_mask  # noqa: E402

OUT = REPO.parent / "assets/md/fuentes/srf/pics_49/c2st_v2.json"
PROFILE = "mss5_esa_gtc_v2"


def band_rmse_rel(pred: np.ndarray, obs: np.ndarray, valid: np.ndarray) -> list[float]:
    values = []
    for index in range(obs.shape[0]):
        reference = obs[index][valid].astype(np.float64)
        predicted = pred[index][valid].astype(np.float64)
        values.append(float(np.sqrt(np.mean((predicted - reference) ** 2)) / np.mean(reference)))
    return values


def main() -> None:
    configs = {
        "deterministic": (deterministic(PipelineConfig.from_profile(PROFILE)), None),
        "stochastic_seed0": (PipelineConfig.from_profile(PROFILE), 0),
    }
    per_site: dict[str, list[dict[str, object]]] = {name: [] for name in configs}
    for site in HELD_OUT:
        pair = load_pair(site)
        valid = valid_mask(site, pair.mss)
        for name, (config, seed) in configs.items():
            pred, _ = emulate_s2(pair.s2, config=config, seed=seed)
            row: dict[str, object] = {
                "site": site,
                "valid_pixels": int(valid.sum()),
                "c2st": c2st(pred, pair.mss, valid=valid),
                "rmse_rel": band_rmse_rel(pred, pair.mss, valid),
                **summarize(pred, pair.mss, valid),
            }
            per_site[name].append(row)
            print(
                name,
                site,
                "c2st",
                round(float(row["c2st"]), 3),
                "rmse_rel",
                [round(value, 3) for value in row["rmse_rel"]],
                flush=True,
            )

    summary = {}
    for name, rows in per_site.items():
        c2st_values = np.array([float(row["c2st"]) for row in rows])
        rmse_rel = np.stack([np.array(row["rmse_rel"], dtype=float) for row in rows])
        summary[name] = {
            "c2st_median": float(np.median(c2st_values)),
            "c2st_min": float(c2st_values.min()),
            "c2st_max": float(c2st_values.max()),
            "rmse_rel_median": np.median(rmse_rel, axis=0).round(4).tolist(),
            "sam_deg_median": float(np.median([float(row["sam_deg"]) for row in rows])),
            "bias_median": float(np.median([float(row["bias"]) for row in rows])),
        }

    payload = {
        "profile": PROFILE,
        "held_out_sites": HELD_OUT,
        "method": (
            "deterministic and stochastic (seed 0) v2 emulation; saturation, "
            "non-finite and out-of-range pixels dropped; c2st is 3-fold "
            "stratified accuracy over per-pixel spectra"
        ),
        "summary": summary,
        "per_site": per_site,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
