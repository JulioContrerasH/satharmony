#!/usr/bin/env python
"""Fit a Landsat 1-3 MSS profile on the auxiliary site bank.

The bank is built from the archive (``fetch_aux_sites.py``), so every site has
a scene. The stability check does not need Landsat 5 here: the per-site optimal
scale is compared with the median across sites, and a site whose scale departs
more than 15 % changed surface or has an artefact.

A quarter of the sites is held out before the fit and never seen by it.

Usage::

    python scripts/make_profile_aux.py --sensor L2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.calibration.fit import fit_profile  # noqa: E402
from satharmony.emulators.s2_to_mss.config import (  # noqa: E402
    MEASURED_SUN_ELEVATION_CANONICAL,
    PipelineConfig,
)
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402
from scripts.aux_pairs import load_pair, root, sites, valid_mask  # noqa: E402
from scripts.holdout_v2 import deterministic  # noqa: E402

BANDS = ("B1", "B2", "B3", "B4")
TOLERANCE = 0.15
HELD_OUT_EVERY = 4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def site_metrics(sensor: str, name: str, config: PipelineConfig) -> dict[str, float]:
    """Relative RMSE with the global config and with a free per-site scale."""
    pair = load_pair(sensor, name)
    mask = valid_mask(sensor, name, pair.mss)
    prediction, _ = emulate_s2(pair.s2, config=config)
    row: dict[str, float] = {"masked": float(mask.mean())}
    for index, band in enumerate(BANDS):
        reference = pair.mss[index][mask].astype(np.float64)
        estimate = prediction[index][mask].astype(np.float64)
        if reference.size < 100:
            row[band] = float("nan")
            row[f"{band}_scaled"] = float("nan")
            row[f"{band}_scale"] = float("nan")
            continue
        row[band] = float(np.sqrt(np.mean((estimate - reference) ** 2)) / np.mean(reference))
        scale = float((estimate * reference).sum() / max((estimate * estimate).sum(), 1e-12))
        row[f"{band}_scale"] = scale
        row[f"{band}_scaled"] = float(np.sqrt(np.mean((scale * estimate - reference) ** 2)) / np.mean(reference))
    return row


def stable(names: list[str], metrics: dict[str, dict[str, float]]) -> tuple[list[str], list[str], dict[str, float]]:
    scales = {name: [metrics[name][f"{band}_scale"] for band in BANDS] for name in names}
    medians = {band: float(np.median([scales[name][index] for name in names])) for index, band in enumerate(BANDS)}
    keep, reject = [], []
    for name in names:
        deviation = max(abs(scales[name][index] / medians[band] - 1.0) for index, band in enumerate(BANDS))
        (keep if deviation <= TOLERANCE else reject).append(name)
    return keep, reject, medians


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", required=True, choices=["L1", "L2", "L3"])
    parser.add_argument("--name", default=None)
    parser.add_argument("--min-sites", type=int, default=6)
    args = parser.parse_args()

    sensor = f"{args.sensor}_MSS"
    profile_name = args.name or f"mss{args.sensor[1]}_esa_gtc_v2"
    coefficients = f"sbaf_coefficients_{args.sensor.lower()}.json"
    rows = sites(args.sensor)
    if len(rows) < args.min_sites:
        raise SystemExit(f"only {len(rows)} sites downloaded for {args.sensor}")
    names = [row["site"] for row in rows]

    base = PipelineConfig.from_profile("mss5_esa_gtc_v2")
    base.radiometric.satellite = sensor
    base.spectral.coefficients = coefficients
    base.radiometric.sun_elevation_deg = type(base.radiometric.sun_elevation_deg)(
        **MEASURED_SUN_ELEVATION_CANONICAL
    )

    print(f"{len(names)} sites on disk", flush=True)
    metrics = {name: site_metrics(args.sensor, name, base) for name in names}
    usable = [name for name in names if metrics[name]["masked"] >= 0.9 and all(np.isfinite(metrics[name][b]) for b in BANDS)]
    keep, reject, medians = stable(usable, metrics)
    print(f"{len(keep)} stable, {len(reject)} rejected, {len(names) - len(usable)} too cloudy", flush=True)
    print("rejected:", reject, flush=True)
    if len(keep) < args.min_sites:
        raise SystemExit(f"only {len(keep)} stable sites; not enough")

    # A held-out split needs enough sites to keep the fit meaningful; below
    # eight the profile is fitted on everything and the `validity` says so.
    if len(keep) >= 8:
        held = [name for index, name in enumerate(keep) if index % HELD_OUT_EVERY == HELD_OUT_EVERY - 1]
    else:
        held = []
    train = [name for name in keep if name not in held]
    print(f"train {len(train)}, held out {len(held)}", flush=True)

    base.spectral.enabled = True
    profile = fit_profile((load_pair(args.sensor, name) for name in train), base=base, name=profile_name, n_boot=1000, gamma_rule="off")
    config = profile.to_config(base=base)
    config.radiometric.satellite = sensor
    config.spectral.coefficients = coefficients
    config.radiometric.sun_elevation_deg = type(config.radiometric.sun_elevation_deg)(
        **MEASURED_SUN_ELEVATION_CANONICAL
    )
    eval_config = deterministic(config)

    train_metrics = {name: site_metrics(args.sensor, name, eval_config) for name in train}
    held_metrics = {name: site_metrics(args.sensor, name, eval_config) for name in held}
    in_sample = {band: float(np.nanmedian([row[band] for row in train_metrics.values()])) for band in BANDS}
    in_scaled = {band: float(np.nanmedian([row[f"{band}_scaled"] for row in train_metrics.values()])) for band in BANDS}
    held_out = {band: float(np.nanmedian([row[band] for row in held_metrics.values()])) for band in BANDS} if held else {}
    held_scaled = (
        {band: float(np.nanmedian([row[f"{band}_scaled"] for row in held_metrics.values()])) for band in BANDS}
        if held
        else {}
    )
    worst = max(held_out.values()) if held_out else max(in_sample.values())
    status = "calibrated" if max(in_sample.values()) <= 0.08 and worst <= 0.12 else "experimental"

    payload = profile.to_dict()
    payload["config"] = config.model_dump(mode="json")
    payload["sensor"] = sensor
    payload["validity"] = {
        "sensor": sensor,
        "status": status,
        "calibration_domain": "desert, steppe and salt flats",
        "outside_domain": ["forest", "water", "crops", "wetland"],
        "site_source": (
            "auxiliary bank selected from the MSS archive itself (one scene per site), "
            "not CEOS PICS; each site needs only a clear same-season Sentinel-2 composite."
        ),
        "n_sites": len(train),
        "n_held_out": len(held),
        "held_out_sites": held,
        "rejected_unstable_sites": reject,
        "stability_tolerance": TOLERANCE,
        "median_scale": medians,
        "latitude_range": [
            float(min(row["lat"] for row in rows)),
            float(max(row["lat"] for row in rows)),
        ],
        "desert_in_sample": {
            "median_rel_rmse": in_sample,
            "median_rel_rmse_free_scale": in_scaled,
            "method": "per-site relative RMSE = RMSE / mean(reference), median across sites",
        },
        "held_out": {
            "median_rel_rmse": held_out,
            "median_rel_rmse_free_scale": held_scaled,
        },
        "sun_elevation": dict(MEASURED_SUN_ELEVATION_CANONICAL),
        "coverage_note": (
            "The Landsat 1-3 archive is sparse and was enumerated globally: every scene "
            "of the platform was listed and only the dry, flat, low-cloud ones were kept. "
            "The limit is the archive, not the number of candidate sites."
        ),
        "status_rule": "calibrated when the worst in-sample band is <= 8 % and the worst held-out band <= 12 %",
    }
    payload["provenance"] = {
        "package_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "source": f"MSS_S2/aux_{args.sensor.lower()} (S2 L1C + MSS {sensor})",
        "meta_sha256": sha256(root(args.sensor) / "meta.json"),
        "assets": {row["site"]: row["mss_asset"] for row in rows},
    }
    out = REPO / "satharmony" / "profiles" / f"{profile_name}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"scales {[round(v, 3) for v in profile.band_scaling]} gamma {profile.gamma:.3f}")
    print(f"in-sample {[round(in_sample[b], 3) for b in BANDS]} free-scale {[round(in_scaled[b], 3) for b in BANDS]}")
    print(f"held-out  {[round(held_out.get(b, float('nan')), 3) for b in BANDS]}")
    print(f"status {status}; written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
