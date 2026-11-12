#!/usr/bin/env python
"""How far is the wet-bank MSS from the S2 grid?

The wet bank pairs a Landsat 1-3 scene (1973-1983, Tier 2) with an S2 composite
(2015+).  Tier 2 geolocation is coarse and the dates are decades apart, so the
error measured without registration mixes the emulator with a geometric offset.
This script estimates the subpixel translation and reports the error before and
after applying it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import shift as ndshift

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.emulators.s2_to_mss.config import PipelineConfig  # noqa: E402
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402
from scripts.holdout_v2 import deterministic  # noqa: E402
from scripts.validate_sbaf_pics import register  # noqa: E402

ROOT = Path("/data/databases/CLOUD_MSS/MSS_S2")
OUT = REPO.parent / "assets/md/fuentes/srf/pics_49"
BANDS = ("B1", "B2", "B3", "B4")
MIN_CLEAR = 0.9


def band_error(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray, dy: float, dx: float) -> list[float]:
    values = []
    for index in range(obs.shape[0]):
        reference = obs[index]
        estimate = pred[index]
        if dy or dx:
            estimate = ndshift(estimate, (dy, dx), order=1, mode="nearest")
        ok = mask & np.isfinite(estimate)
        values.append(
            float(
                np.sqrt(np.mean((estimate[ok] - reference[ok]) ** 2))
                / np.mean(reference[ok])
            )
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", required=True, choices=["L1", "L2", "L3"])
    args = parser.parse_args()

    base = ROOT / f"wet_{args.sensor.lower()}"
    config = deterministic(PipelineConfig.from_profile(f"mss{args.sensor[-1]}_esa_gtc_v2"))
    names = sorted(path.stem for path in (base / "mss").glob("*.tif"))
    rows = []
    for name in names:
        clear = float(tifffile.imread(base / "mask" / f"{name}.tif").astype(bool).mean())
        if clear < MIN_CLEAR:
            continue
        s2 = tifffile.imread(base / "s2" / f"{name}.tif").astype(np.float32) / 10000.0
        mss = tifffile.imread(base / "mss" / f"{name}.tif").astype(np.float32)
        mask = tifffile.imread(base / "mask" / f"{name}.tif").astype(bool)
        mask &= np.all(np.isfinite(mss), axis=0) & np.all(mss > 0, axis=0) & np.all(mss < 1.5, axis=0)
        pred, _ = emulate_s2(s2, config=config)

        reference = mss[3].astype(np.float64)
        moving = pred[3].astype(np.float64)
        found = register(reference, moving, mask)
        dy, dx = float(found["dy"]), float(found["dx"])
        before = band_error(pred, mss, mask, 0.0, 0.0)
        after = band_error(pred, mss, mask, dy, dx)
        ratio = [
            float(np.mean(mss[index][mask]) / max(np.mean(pred[index][mask]), 1e-9))
            for index in range(4)
        ]
        rows.append(
            {
                "site": name,
                "shift_dy": dy,
                "shift_dx": dx,
                "shift_px": float(np.hypot(dy, dx)),
                "corr_before": found.get("corr", None),
                "ratio_mss_over_pred": ratio,
                "error_before": before,
                "error_after": after,
            }
        )
        print(
            f"{name:22s} shift {dy:+.2f},{dx:+.2f} px  "
            f"err {np.mean(before):.3f} -> {np.mean(after):.3f}",
            flush=True,
        )

    if rows:
        medians = np.median(np.array([row["ratio_mss_over_pred"] for row in rows]), axis=0)
        for row in rows:
            deviation = max(
                abs(float(row["ratio_mss_over_pred"][b] / medians[b] - 1.0)) for b in range(4)
            )
            row["ratio_deviation"] = float(deviation)
            row["stable"] = bool(deviation <= 0.15)
            print(
                f"  {row['site']:22s} desviacion {deviation:6.1%} "
                f"{'OK' if row['stable'] else 'CAMBIO'}",
                flush=True,
            )
        stable = [row for row in rows if row["stable"]]
        shifts = np.array([row["shift_px"] for row in rows])
        payload = {
            "sensor": args.sensor,
            "n_sites": len(rows),
            "shift_px_median": float(np.median(shifts)),
            "shift_px_max": float(shifts.max()),
            "summary_before": {
                band: float(np.median([row["error_before"][index] for row in rows]))
                for index, band in enumerate(BANDS)
            },
            "summary_after": {
                band: float(np.median([row["error_after"][index] for row in rows]))
                for index, band in enumerate(BANDS)
            },
            "n_stable": len(stable),
            "stable_sites": [row["site"] for row in stable],
            "summary_stable": {
                band: float(np.median([row["error_after"][index] for row in stable]))
                for index, band in enumerate(BANDS)
            }
            if stable
            else None,
            "stability_rule": "MSS/prediction mean ratio per band within 15 % of the median across sites",
            "note": "translation estimated on B4 with phase correlation + ECC; no rotation or scale",
            "per_site": rows,
        }
        target = OUT / f"alignment_wet_{args.sensor.lower()}.json"
        target.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"median shift {np.median(shifts):.2f} px, max {shifts.max():.2f} px")
        print(target)


if __name__ == "__main__":
    main()
