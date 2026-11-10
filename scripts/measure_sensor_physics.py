#!/usr/bin/env python
"""Sensor physics of the canonical MSS set, measured on the 60 m raw scenes.

The profile's striping, noise and saturation ranges came from the 178 pairs of
the legacy bank.  This script measures the same quantities on a seeded random
sample of the 3,027 aligned scenes, always on the original 60 m MSS (the
aligned product is warped to the 30 m TM grid and would mix resampling into the
noise).

Per scene and band, on clear pixels:

- noise: robust std of the high-pass residual, relative to the band median.
- striping: dominant period and amplitude of the column/row means.
- saturation: fraction of pixels above 0.80 and 0.95 reflectance.

Usage
-----
    python scripts/measure_sensor_physics.py
    python scripts/measure_sensor_physics.py --n 120 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
ROOT = Path("/data/databases/CLOUD_MSS/MSS_TM")
ALIGNED = ROOT / "pairs_4k_aligned/final_cloud_4k.csv"
BATCH1 = ROOT / "pairs_4k/mss"
ZEROS = ROOT / "pairs_4k/zeros_1100/staging/mss"
OUT = PROJECT / "assets/md/fuentes/srf"


def load_ids() -> list[tuple[str, str]]:
    with ALIGNED.open() as fh:
        return [(row["id"], row["source"]) for row in csv.DictReader(fh)]


def paths(site: str, source: str) -> tuple[Path, Path]:
    if source == "batch1":
        return BATCH1 / f"{site}.tif", BATCH1 / f"{site}_cloud_mask.tif"
    return ZEROS / f"{site}.tif", ZEROS / f"{site}_cloud_mask.tif"


def period_and_amplitude(profile: np.ndarray, pixel_m: float = 60.0) -> tuple[float, float]:
    detrended = profile - gaussian_filter(profile, 25, mode="nearest")
    spectrum = np.abs(np.fft.rfft(detrended))
    spectrum[0] = 0.0
    if spectrum.max() <= 0:
        return float("nan"), 0.0
    peak = int(spectrum.argmax())
    period = len(profile) / peak if peak else float("nan")
    amplitude = float(2.0 * spectrum[peak] / len(profile) / max(np.median(profile), 1e-6))
    return period * pixel_m, amplitude


def measure(site: str, source: str) -> dict | None:
    mss_path, mask_path = paths(site, source)
    if not mss_path.exists() or not mask_path.exists():
        return None
    cube = tifffile.imread(mss_path).astype(np.float32) / 10000.0
    if cube.ndim == 3 and cube.shape[-1] == 4:
        cube = np.moveaxis(cube, -1, 0)
    mask = tifffile.imread(mask_path)
    h, w = cube.shape[1:]
    y0, y1 = int(0.15 * h), int(0.85 * h)
    x0, x1 = int(0.15 * w), int(0.85 * w)
    cube = cube[:, y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]
    clear = (mask == 0) & np.all(cube > 0, axis=0)
    if clear.sum() < 5000:
        return None

    row = {"id": site, "source": source, "clear_frac": round(float(clear.mean()), 4)}
    for band in range(cube.shape[0]):
        reference = cube[band]
        pixels = reference[clear]
        median = float(np.median(pixels))
        residual = reference - gaussian_filter(reference, 1.0)
        mad = float(np.median(np.abs(residual[clear] - np.median(residual[clear]))))
        noise = 1.4826 * mad
        masked = np.where(clear, reference, np.nan)
        columns = np.nanmean(masked, axis=0)
        rows = np.nanmean(masked, axis=1)
        columns = np.where(np.isfinite(columns), columns, median)
        rows = np.where(np.isfinite(rows), rows, median)
        period_col, amp_col = period_and_amplitude(columns)
        period_row, amp_row = period_and_amplitude(rows)
        if amp_col >= amp_row:
            period, amplitude = period_col, amp_col
        else:
            period, amplitude = period_row, amp_row
        row[f"B{band + 1}_median"] = round(median, 4)
        row[f"B{band + 1}_noise_rel"] = round(noise / max(median, 1e-6), 4)
        row[f"B{band + 1}_stripe_period_m"] = round(float(period), 1)
        row[f"B{band + 1}_stripe_amp"] = round(amplitude, 4)
        row[f"B{band + 1}_sat80"] = round(float((pixels > 0.80).mean()), 4)
        row[f"B{band + 1}_sat95"] = round(float((pixels > 0.95).mean()), 4)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=90)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)

    scenes = load_ids()
    rng = np.random.default_rng(args.seed)
    batch = [s for s in scenes if s[1] == "batch1"]
    zeros = [s for s in scenes if s[1] == "zeros"]
    take = round(args.n * 2 / 3)
    sample = list(rng.choice(len(batch), size=min(take, len(batch)), replace=False))
    sample = [batch[i] for i in sample]
    sample += [zeros[i] for i in rng.choice(len(zeros), size=min(args.n - len(sample), len(zeros)), replace=False)]

    rows = []
    for index, (site, source) in enumerate(sample, 1):
        try:
            result = measure(site, source)
        except (OSError, ValueError) as error:
            print(f"  {site}: {error}", flush=True)
            result = None
        if result:
            rows.append(result)
        if index % 15 == 0:
            print(f"  {index}/{len(sample)}", flush=True)

    if not rows:
        print("no scenes measured")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    table = OUT / "sensor_physics_sample.csv"
    with table.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"n_scenes": len(rows), "seed": args.seed, "bands": {}}
    for band in range(1, 5):
        block = {}
        for key, label in (
            ("noise_rel", "noise_rel"),
            ("stripe_amp", "stripe_amp"),
            ("stripe_period_m", "stripe_period_m"),
            ("sat80", "sat80"),
            ("sat95", "sat95"),
        ):
            values = np.array([row[f"B{band}_{key}"] for row in rows], dtype=float)
            values = values[np.isfinite(values)]
            block[label] = {
                "median": round(float(np.median(values)), 4),
                "p25": round(float(np.percentile(values, 25)), 4),
                "p75": round(float(np.percentile(values, 75)), 4),
            }
        summary["bands"][f"B{band}"] = block
    (OUT / "sensor_physics_sample.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"written {table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
