#!/usr/bin/env python
"""Compare v1 and v2 on the validation PICS; test the optional gas scale."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.calibration.fit import _deterministic_config  # noqa: E402
from satharmony.emulators.s2_to_mss.config import PipelineConfig  # noqa: E402
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402
from satharmony.validation.domain import ks_per_band, wasserstein_per_band  # noqa: E402
from scripts.make_profile_v2 import load_pair  # noqa: E402

PICS = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")
OUT = REPO.parent / "assets/md/fuentes/srf/pics_49/profile_comparison.json"


def deterministic(profile: str) -> PipelineConfig:
    config = _deterministic_config(PipelineConfig.from_profile(profile))
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
    sites = [entry["site"] for entry in index["sites"] if entry.get("role") == "validation"]
    profiles = {"v1": deterministic("mss5_esa_gtc_v1"), "v2": deterministic("mss5_esa_gtc_v2")}
    scores = {name: {"w1": [], "ks": []} for name in profiles}
    gas = {"off": [], "on": []}

    for site in sites:
        pair = load_pair(site)
        for name, config in profiles.items():
            pred, _ = emulate_s2(pair.s2, config=config)
            scores[name]["w1"].append(wasserstein_per_band(pred, pair.mss))
            scores[name]["ks"].append(ks_per_band(pred, pair.mss))
        for label, scale in (("off", 1.0), ("on", 1.054)):
            config = profiles["v2"].model_copy(deep=True)
            config.spectral.gas_mss3_scale = scale
            pred, _ = emulate_s2(pair.s2, config=config)
            gas[label].append(float(np.nanmean(pred[2] - pair.mss[2])))
        print(site, flush=True)

    summary = {}
    for name in profiles:
        summary[name] = {
            "w1": np.nanmean(np.stack(scores[name]["w1"]), axis=0).round(5).tolist(),
            "ks": np.nanmean(np.stack(scores[name]["ks"]), axis=0).round(5).tolist(),
        }
    summary["gas_mss3_bias"] = {
        "off": float(np.nanmean(gas["off"])),
        "on": float(np.nanmean(gas["on"])),
    }
    summary["n_validation_sites"] = len(sites)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
