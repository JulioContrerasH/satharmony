#!/usr/bin/env python
"""Fit the Landsat 4 MSS profile from the PICS and the L4 scenes.

The S2 side is the existing PICS. The MSS side is the L4 scene fetched by
``fetch_pics_l4.py`` on the same 250 x 250 grid, so no alignment is involved.
The spectral matrix is the package SBAF (L4/L5 SRFs are close); what the fit
learns is the per-band scale and the B4 water-vapour term for L4.

Usage::

    python scripts/make_profile_l4.py
"""

from __future__ import annotations

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
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402
from satharmony.emulators.s2_to_mss.sbaf import S2_BAND_ORDER  # noqa: E402
from scripts.holdout_v2 import deterministic  # noqa: E402
from scripts.make_profile_v2 import read_band  # noqa: E402

PICS = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")
PICS_L4 = Path("/data/databases/CLOUD_MSS/MSS_S2/pics_l4")
OUT = REPO / "satharmony/profiles/mss4_esa_gtc_v1.json"
HELD_OUT = ["libya4_ceos", "sudan1", "oman1", "kyzylkum", "bonneville"]
BANDS = ("B1", "B2", "B3", "B4")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pair(site: str) -> Pair:
    root = PICS / site
    s2 = np.stack(
        [
            read_band(sorted((root / "s2").glob(f"*.{band}.tif"))[0]) / 10000.0
            for band in S2_BAND_ORDER
        ]
    )
    mss = tifffile.imread(PICS_L4 / "mss" / site / "mss.tif").astype(np.float32)
    return Pair(s2=s2, mss=mss, name=site)


def load_l5(site: str) -> np.ndarray:
    root = PICS / site / "mss"
    return np.stack(
        [read_band(sorted(root.glob(f"*.B{band}.tif"))[0]) for band in range(1, 5)]
    )


def stable_sites(candidates: list[str], tolerance: float = 0.15) -> tuple[list[str], list[str], list[float]]:
    """Keep the sites where L4 and L5 see the same surface.

    The ratio of the L4 to the L5 mean reflectance per band is normalised by
    its median across sites (a uniform radiometric offset is absorbed by the
    profile scale). A site whose ratio departs more than ``tolerance`` from
    that median changed surface, or its window is hazy.
    """
    ratios = {}
    for site in candidates:
        l4 = tifffile.imread(PICS_L4 / "mss" / site / "mss.tif").astype(np.float32)
        l5 = load_l5(site)
        ratios[site] = [float(l4[b].mean() / l5[b].mean()) for b in range(4)]
    medians = np.median(np.array(list(ratios.values())), axis=0)
    keep, reject = [], []
    for site, ratio in ratios.items():
        deviation = max(abs(ratio[b] / medians[b] - 1.0) for b in range(4))
        (keep if deviation <= tolerance else reject).append(site)
    return keep, reject, [float(value) for value in medians]


def site_error(pair: Pair, config: PipelineConfig) -> dict[str, float]:
    pred, _ = emulate_s2(pair.s2, config=deterministic(config))
    mask = np.isfinite(pair.mss).all(0) & (pair.mss > 0).all(0)
    row: dict[str, float] = {}
    for index, band in enumerate(BANDS):
        reference = pair.mss[index][mask].astype(np.float64)
        estimate = pred[index][mask].astype(np.float64)
        row[band] = float(np.sqrt(np.mean((estimate - reference) ** 2)) / np.mean(reference))
    return row


def main() -> None:
    meta = json.loads((PICS_L4 / "meta.json").read_text())
    available = {row["site"] for row in meta if (PICS_L4 / "mss" / row["site"] / "mss.tif").exists()}
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
    stable, unstable, median_ratio = stable_sites(candidates + held_candidates)
    sites = [site for site in candidates if site in stable]
    held = [site for site in held_candidates if site in stable]
    missing = [site for site in calibration if site not in available]
    print(f"{len(sites)} fit sites, {len(held)} held out, {len(missing)} missing, {len(unstable)} unstable")
    print("rejected as unstable:", unstable)
    if len(sites) < 8:
        raise SystemExit(f"only {len(sites)} sites; not enough for a profile")

    base = PipelineConfig.from_profile("mss5_esa_gtc_v2")
    base.radiometric.satellite = "L4_MSS"
    base.spectral.coefficients = "sbaf_coefficients_l4.json"
    profile = fit_profile((load_pair(site) for site in sites), base=base, name="mss4_esa_gtc_v1", n_boot=1000, gamma_rule="off")
    config = profile.to_config(base=base)
    config.radiometric.satellite = "L4_MSS"
    config.radiometric.sun_elevation_deg = type(config.radiometric.sun_elevation_deg)(
        **MEASURED_SUN_ELEVATION_CANONICAL
    )

    in_sample = {site: site_error(load_pair(site), config) for site in sites}
    held_out = {site: site_error(load_pair(site), config) for site in held}
    median = {band: float(np.median([row[band] for row in in_sample.values()])) for band in BANDS}
    held_median = {band: float(np.median([row[band] for row in held_out.values()])) for band in BANDS} if held_out else {}

    payload = profile.to_dict()
    payload["config"] = config.model_dump(mode="json")
    payload["sensor"] = "L4_MSS"
    payload["validity"] = {
        "sensor": "L4_MSS",
        "status": "experimental",
        "band_gain": (
            "free per band, water-vapour term off: with gamma the B4 gain was forced "
            "up and the error doubled, and gamma is not identifiable on these sites"
        ),
        "calibration_domain": "desert and salt flats",
        "outside_domain": ["forest", "water", "crops", "wetland"],
        "held_out_sites": held,
        "n_sites": len(sites),
        "n_held_out": len(held),
        "missing_in_gee": missing,
        "rejected_unstable_sites": sorted(unstable),
        "l4_l5_median_ratio": median_ratio,
        "stability_tolerance": 0.15,
        "stability_note": (
            "A site is used only when its L4/L5 mean ratio per band stays within "
            "15 % of the median across sites; the rest changed surface or have a "
            "hazy window."
        ),
        "coverage_note": (
            "Landsat 4 does not cover every PICS path/row in GEE; the profile is "
            "fitted on the sites that have a cloud-free L4 scene at the same sun height."
        ),
        "desert_in_sample": {
            "median_rel_rmse": median,
            "method": "per-site relative RMSE = RMSE / mean(reference), median across sites",
        },
        "held_out": {"median_rel_rmse": held_median},
        "sun_elevation": dict(MEASURED_SUN_ELEVATION_CANONICAL),
    }
    payload["provenance"] = {
        "package_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "source": "MSS_S2/pics (S2) + MSS_S2/pics_l4 (MSS L4)",
        "meta_sha256": sha256(PICS_L4 / "meta.json"),
        "l4_assets": {row["site"]: row["asset"] for row in meta if row["site"] in available},
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"scales {[round(v, 3) for v in profile.band_scaling]} gamma {profile.gamma:.3f}")
    print(f"in-sample {[round(median[b], 3) for b in BANDS]}")
    print(f"held-out  {[round(held_median.get(b, float('nan')), 3) for b in BANDS]}")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
