#!/usr/bin/env python
"""Bootstrap the per-site v2-v1 delta on held-out deserts."""

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

HELD = ["libya4_ceos", "sudan1", "oman1", "kyzylkum", "bonneville"]
OUT = REPO.parent / "assets/md/fuentes/srf/pics_49/holdout_delta.json"


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
    configs = {
        "v1": deterministic(PipelineConfig.from_profile("mss5_esa_gtc_v1")),
        "v2": deterministic(PipelineConfig.from_profile("mss5_esa_gtc_v2")),
    }
    w1 = {name: [] for name in configs}
    ks = {name: [] for name in configs}
    for site in HELD:
        pair = load_pair(site)
        for name, config in configs.items():
            pred, _ = emulate_s2(pair.s2, config=config)
            w1[name].append(wasserstein_per_band(pred, pair.mss))
            ks[name].append(ks_per_band(pred, pair.mss))
        print(site, flush=True)

    rng = np.random.default_rng(0)
    index = rng.integers(0, len(HELD), size=(10000, len(HELD)))
    out = {"sites": HELD, "bands": ["B1", "B2", "B3", "B4"], "delta": {}}
    for label, values in (("w1", w1), ("ks", ks)):
        first = np.stack(values["v1"])
        second = np.stack(values["v2"])
        delta = second - first
        boot = np.median(delta[index], axis=1)
        out["delta"][label] = {
            "per_site": delta.round(6).tolist(),
            "median": np.median(delta, axis=0).round(6).tolist(),
            "low": np.percentile(boot, 2.5, axis=0).round(6).tolist(),
            "high": np.percentile(boot, 97.5, axis=0).round(6).tolist(),
        }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
