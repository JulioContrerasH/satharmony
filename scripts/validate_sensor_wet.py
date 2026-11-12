#!/usr/bin/env python
"""Out-of-domain error of the L1-L3 profiles on the wet bank.

The desert bank cannot answer this: it was filtered to dry ground.  These sites
are forest, water and wetland, so they are outside the calibration domain by
definition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.emulators.s2_to_mss.config import PipelineConfig  # noqa: E402
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402
from satharmony.emulators.s2_to_mss.sbaf import S2_BAND_ORDER  # noqa: E402
from scripts.holdout_v2 import deterministic  # noqa: E402

ROOT = Path("/data/databases/CLOUD_MSS/MSS_S2")
OUT = REPO.parent / "assets/md/fuentes/srf/pics_49"
BANDS = ("B1", "B2", "B3", "B4")
MIN_CLEAR = 0.9


def load(sensor: str, name: str):
    base = ROOT / f"wet_{sensor.lower()}"
    s2 = tifffile.imread(base / "s2" / f"{name}.tif").astype(np.float32) / 10000.0
    mss = tifffile.imread(base / "mss" / f"{name}.tif").astype(np.float32)
    mask = tifffile.imread(base / "mask" / f"{name}.tif").astype(bool)
    mask &= np.all(np.isfinite(mss), axis=0) & np.all(mss > 0, axis=0) & np.all(mss < 1.5, axis=0)
    if s2.shape[0] < len(S2_BAND_ORDER):
        raise ValueError(f"{name}: {s2.shape[0]} S2 bands")
    return s2, mss, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", required=True, choices=["L1", "L2", "L3"])
    args = parser.parse_args()

    base = ROOT / f"wet_{args.sensor.lower()}"
    profile = f"mss{args.sensor[-1]}_esa_gtc_v2"
    config = deterministic(PipelineConfig.from_profile(profile))
    names = sorted(path.stem for path in (base / "mss").glob("*.tif"))
    print(f"{len(names)} wet sites for {args.sensor}", flush=True)

    rows = []
    for name in names:
        s2, mss, mask = load(args.sensor, name)
        clear = float(tifffile.imread(base / "mask" / f"{name}.tif").astype(bool).mean())
        if clear < MIN_CLEAR:
            print(f"{name}: skipped, only {clear:.2%} clear", flush=True)
            continue
        pred, _ = emulate_s2(s2, config=config)
        row: dict[str, float | str] = {"site": name, "clear": clear, "valid_pixels": int(mask.sum())}
        for index, band in enumerate(BANDS):
            reference = mss[index][mask].astype(np.float64)
            estimate = pred[index][mask].astype(np.float64)
            scale = float(np.dot(estimate, reference) / max(np.dot(estimate, estimate), 1e-12))
            row[f"{band}_raw"] = float(
                np.sqrt(np.mean((estimate - reference) ** 2)) / np.mean(reference)
            )
            row[f"{band}_rescaled"] = float(
                np.sqrt(np.mean((scale * estimate - reference) ** 2)) / np.mean(reference)
            )
        rows.append(row)
        print(name, [round(float(row[f"{band}_rescaled"]), 3) for band in BANDS], flush=True)

    if not rows:
        raise SystemExit("no site with a clear window")
    summary = {
        band: {
            "raw_median": float(np.median([float(r[f"{band}_raw"]) for r in rows])),
            "rescaled_median": float(np.median([float(r[f"{band}_rescaled"]) for r in rows])),
        }
        for band in BANDS
    }
    payload = {
        "profile": profile,
        "bank": str(base),
        "min_clear": MIN_CLEAR,
        "n_sites": len(rows),
        "summary": summary,
        "per_site": rows,
    }
    target = OUT / f"validation_{args.sensor.lower()}_wet.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print({band: round(summary[band]["rescaled_median"], 3) for band in BANDS})
    print(target)


if __name__ == "__main__":
    main()
