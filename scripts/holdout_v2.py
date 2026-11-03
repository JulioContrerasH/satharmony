#!/usr/bin/env python
"""Hold out five desert PICS, fit on the rest, compare v1 vs the fit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.calibration.fit import _deterministic_config, fit_profile  # noqa: E402
from satharmony.emulators.s2_to_mss.config import PipelineConfig  # noqa: E402
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402
from satharmony.validation.domain import ks_per_band, wasserstein_per_band  # noqa: E402
from scripts.make_profile_v2 import load_pair  # noqa: E402

PICS = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")
OUT = REPO.parent / "assets/md/fuentes/srf/pics_49/holdout_v2.json"


def deterministic(config: PipelineConfig) -> PipelineConfig:
    config = _deterministic_config(config)
    config.final_scaling.enabled = True
    for band in (
        config.final_scaling.band_scaling.green,
        config.final_scaling.band_scaling.red,
        config.final_scaling.band_scaling.nir1,
        config.final_scaling.band_scaling.nir2,
    ):
        band.variation = 0.0
    return config


def main() -> None:
    index = json.loads((PICS / "pics_index.json").read_text())
    sites = [
        entry["site"]
        for entry in index["sites"]
        if entry.get("role", "calibration") == "calibration" and entry["site"] != "uyuni"
    ]
    held = sites[::7]
    train = [site for site in sites if site not in held]
    print(f"train {len(train)}, held {len(held)}: {' '.join(held)}", flush=True)

    profile = fit_profile((load_pair(site) for site in train), name="holdout")
    configs = {
        "v1": deterministic(PipelineConfig.from_profile("mss5_esa_gtc_v1")),
        "fit": deterministic(profile.to_config()),
    }
    scores = {name: {"w1": [], "ks": []} for name in configs}
    for site in held:
        pair = load_pair(site)
        for name, config in configs.items():
            pred, _ = emulate_s2(pair.s2, config=config)
            scores[name]["w1"].append(wasserstein_per_band(pred, pair.mss))
            scores[name]["ks"].append(ks_per_band(pred, pair.mss))
        print(site, flush=True)

    summary = {
        name: {
            "w1": np.nanmean(np.stack(scores[name]["w1"]), axis=0).round(5).tolist(),
            "ks": np.nanmean(np.stack(scores[name]["ks"]), axis=0).round(5).tolist(),
        }
        for name in configs
    }
    summary["held_out"] = held
    summary["n_train"] = len(train)
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
