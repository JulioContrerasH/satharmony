#!/usr/bin/env python
"""Fit a Landsat 1-3 MSS profile from the PICS and the fetched scenes.

Same pipeline as the L4 profile: the SBAF is that sensor's own (fitted from
its SRF and the spectral library), the scale and gamma are fitted here, and a
site is used only when the sensor/L5 ratio stays within 15 % of the median.

Usage::

    python scripts/make_profile_sensor.py --sensor L1_MSS --pics pics_l1 \
        --coefficients sbaf_coefficients_l1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.calibration.fit import Pair, fit_profile  # noqa: E402
from satharmony.emulators.s2_to_mss.config import (  # noqa: E402
    MEASURED_SUN_ELEVATION_CANONICAL,
    PipelineConfig,
)
from satharmony.emulators.s2_to_mss.sbaf import S2_BAND_ORDER  # noqa: E402
from scripts.make_profile_l4 import load_l5, site_error  # noqa: E402
from scripts.make_profile_v2 import read_band  # noqa: E402

PICS = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")
DATA = Path("/data/databases/CLOUD_MSS/MSS_S2")
HELD_OUT = ["libya4_ceos", "sudan1", "oman1", "kyzylkum", "bonneville"]
BANDS = ("B1", "B2", "B3", "B4")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pair(site: str, pics_dir: Path) -> Pair:
    root = PICS / site
    s2 = np.stack(
        [
            read_band(sorted((root / "s2").glob(f"*.{band}.tif"))[0]) / 10000.0
            for band in S2_BAND_ORDER
        ]
    )
    mss = tifffile.imread(pics_dir / "mss" / site / "mss.tif").astype(np.float32)
    return Pair(s2=s2, mss=mss, name=site)


def stable_sites(candidates: list[str], pics_dir: Path, tolerance: float = 0.15):
    ratios = {}
    for site in candidates:
        l4 = tifffile.imread(pics_dir / "mss" / site / "mss.tif").astype(np.float32)
        l5 = load_l5(site)
        ratios[site] = [float(l4[b].mean() / l5[b].mean()) for b in range(4)]
    medians = np.median(np.array(list(ratios.values())), axis=0)
    keep, reject = [], []
    for site, ratio in ratios.items():
        deviation = max(abs(ratio[b] / medians[b] - 1.0) for b in range(4))
        (keep if deviation <= tolerance else reject).append(site)
    return keep, reject, [float(value) for value in medians]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", required=True, choices=["L1_MSS", "L2_MSS", "L3_MSS"])
    parser.add_argument("--pics", required=True)
    parser.add_argument("--coefficients", required=True)
    args = parser.parse_args()

    pics_dir = DATA / args.pics
    meta = json.loads((pics_dir / "meta.json").read_text())
    available = {row["site"] for row in meta if (pics_dir / "mss" / row["site"] / "mss.tif").exists()}
    index = json.loads((PICS / "pics_index.json").read_text())
    calibration = [
        entry["site"]
        for entry in index["sites"]
        if entry.get("role", "calibration") == "calibration"
        and entry["site"] != "uyuni"
        and entry["site"] not in HELD_OUT
    ]
    candidates = [site for site in calibration if site in available]
    held_candidates = [site for site in HELD_OUT if site in available]
    stable, unstable, median_ratio = stable_sites(candidates + held_candidates, pics_dir)
    sites = [site for site in candidates if site in stable]
    held = [site for site in held_candidates if site in stable]
    print(f"{len(sites)} fit sites, {len(held)} held out, {len(unstable)} unstable")
    if len(sites) < 5:
        raise SystemExit(f"only {len(sites)} stable sites; not enough for a profile")

    base = PipelineConfig.from_profile("mss5_esa_gtc_v2")
    base.radiometric.satellite = args.sensor
    base.spectral.coefficients = args.coefficients
    name = f"mss{args.sensor[1]}_esa_gtc_v1"
    profile = fit_profile(
        (load_pair(site, pics_dir) for site in sites), base=base, name=name, n_boot=1000
    )
    config = profile.to_config(base=base)
    config.radiometric.satellite = args.sensor
    config.spectral.coefficients = args.coefficients
    config.radiometric.sun_elevation_deg = type(config.radiometric.sun_elevation_deg)(
        **MEASURED_SUN_ELEVATION_CANONICAL
    )

    in_sample = {site: site_error(load_pair(site, pics_dir), config) for site in sites}
    held_out = {site: site_error(load_pair(site, pics_dir), config) for site in held}
    median = {band: float(np.median([row[band] for row in in_sample.values()])) for band in BANDS}
    held_median = (
        {band: float(np.median([row[band] for row in held_out.values()])) for band in BANDS}
        if held_out
        else {}
    )
    status = "calibrated" if max(median.values()) <= 0.08 else "experimental"

    payload = profile.to_dict()
    payload["config"] = config.model_dump(mode="json")
    payload["sensor"] = args.sensor
    payload["validity"] = {
        "sensor": args.sensor,
        "status": status,
        "calibration_domain": "desert and salt flats",
        "outside_domain": ["forest", "water", "crops", "wetland"],
        "held_out_sites": held,
        "n_sites": len(sites),
        "n_held_out": len(held),
        "rejected_unstable_sites": sorted(unstable),
        "sensor_l5_median_ratio": median_ratio,
        "stability_tolerance": 0.15,
        "desert_in_sample": {
            "median_rel_rmse": median,
            "method": "per-site relative RMSE = RMSE / mean(reference), median across sites",
        },
        "held_out": {"median_rel_rmse": held_median},
        "sun_elevation": dict(MEASURED_SUN_ELEVATION_CANONICAL),
        "coverage_note": (
            "Landsat 1-3 used WRS-1 and their archive is sparse; the profile is "
            "fitted only on the PICS that have a scene at a comparable sun height."
        ),
    }
    payload["provenance"] = {
        "package_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "source": f"MSS_S2/pics (S2) + MSS_S2/{args.pics} ({args.sensor})",
        "meta_sha256": sha256(pics_dir / "meta.json"),
        "assets": {row["site"]: row["asset"] for row in meta if row["site"] in available},
    }
    out = REPO / "satharmony" / "profiles" / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"scales {[round(v, 3) for v in profile.band_scaling]} gamma {profile.gamma:.3f}")
    print(f"in-sample {[round(median[b], 3) for b in BANDS]}")
    print(f"held-out  {[round(held_median.get(b, float('nan')), 3) for b in BANDS]}")
    print(f"status {status}; written {out}")


if __name__ == "__main__":
    main()
