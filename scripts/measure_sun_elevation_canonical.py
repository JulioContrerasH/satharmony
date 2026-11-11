#!/usr/bin/env python
"""Sun-elevation distribution of the canonical cloud set (3,027 scenes).

The v1 profile used 178 pairs curated from the legacy bank.  Of those, 145 are
in the final aligned set, 29 fell into ``pairs_3k_aligned_not_used`` and 4 were
rejected before alignment, so the distribution was measured on a population
that is not the one the cloud model uses.

This script rebuilds it from ``pairs_4k_aligned/final_cloud_4k.csv``:

- zeros: ``pairs_4k/zeros_1100/selected.csv`` (has ``sun_elevation``).
- batch 1: the MSS/TM match tables under ``assets/md/metadata/``.
- leftovers: GEE ``SUN_ELEVATION`` of the scene asset, trying L5 then L4.

Outputs
-------
- ``sun_elevation_canonical.csv``   id, sun_elevation, source.
- ``sun_elevation_canonical.json``  coverage, stats, percentiles, input hashes.

Usage
-----
    python scripts/measure_sun_elevation_canonical.py
    python scripts/measure_sun_elevation_canonical.py --no-gee
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
ALIGNED = Path("/data/databases/CLOUD_MSS/MSS_TM/pairs_4k_aligned/final_cloud_4k.csv")
ZEROS = Path("/data/databases/CLOUD_MSS/MSS_TM/pairs_4k/zeros_1100/selected.csv")
METADATA = PROJECT / "assets/md/metadata"
OUT = PROJECT / "assets/md/fuentes/srf"
PERCENTILES = (0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 100.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_ids() -> list[str]:
    with ALIGNED.open() as fh:
        return [row["id"] for row in csv.DictReader(fh)]


def from_zeros() -> dict[str, float]:
    out: dict[str, float] = {}
    if not ZEROS.exists():
        return out
    with ZEROS.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("sun_elevation"):
                out[row["id"]] = float(row["sun_elevation"])
    return out


def from_metadata(wanted: set[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for path in sorted(METADATA.glob("*.csv")):
        try:
            with path.open() as fh:
                reader = csv.DictReader(fh)
                if not reader.fieldnames or "sun_elevation" not in reader.fieldnames:
                    continue
                key = "id_name" if "id_name" in reader.fieldnames else "id"
                for row in reader:
                    site = row.get(key)
                    if site in wanted and site not in out and row.get("sun_elevation"):
                        out[site] = float(row["sun_elevation"])
        except (OSError, csv.Error, ValueError):
            continue
    return out


def from_gee(missing: list[str]) -> dict[str, float]:
    import ee  # imported only when needed

    ee.Initialize(project="ee-contrerasnetk")
    out: dict[str, float] = {}
    for index, site in enumerate(missing, 1):
        # The canonical id is ``date_row_path``; the GEE asset is
        # ``<sat>_<path><row>_<date>`` (path first).
        date, row, path = site.split("_")
        stamp = date.replace("-", "")
        for sat in ("LM05", "LM04"):
            for collection in ("T2", "T1"):
                asset = f"LANDSAT/{sat}/C02/{collection}/{sat}_{path}{row}_{stamp}"
                try:
                    value = ee.Image(asset).get("SUN_ELEVATION").getInfo()
                except Exception:  # any GEE failure means "try next"
                    value = None
                if value is not None:
                    out[site] = float(value)
                    break
            if site in out:
                break
        if index % 25 == 0:
            print(f"  GEE {index}/{len(missing)}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gee", action="store_true", help="skip the GEE fallback")
    args = parser.parse_args(argv)

    ids = canonical_ids()
    wanted = set(ids)
    values = from_zeros()
    values = {k: v for k, v in values.items() if k in wanted}
    values.update(from_metadata(wanted - set(values)))
    missing = [site for site in ids if site not in values]
    print(f"{len(ids)} canonical scenes, {len(values)} with metadata, {len(missing)} missing", flush=True)

    if missing and not args.no_gee:
        values.update(from_gee(missing))

    OUT.mkdir(parents=True, exist_ok=True)
    table = OUT / "sun_elevation_canonical.csv"
    with table.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "sun_elevation"])
        for site in ids:
            if site in values:
                writer.writerow([site, f"{values[site]:.6f}"])

    sample = np.array([values[site] for site in ids if site in values])
    payload = {
        "n_canonical": len(ids),
        "n_measured": int(sample.size),
        "coverage": round(sample.size / len(ids), 4),
        "source": {
            "zeros": str(ZEROS),
            "metadata": str(METADATA),
            "missing": "GEE SUN_ELEVATION (LM05 then LM04)",
        },
        "inputs_sha256": {
            "final_cloud_4k.csv": sha256(ALIGNED),
            "zeros_selected.csv": sha256(ZEROS) if ZEROS.exists() else None,
        },
        "stats": {
            "n": int(sample.size),
            "min": float(sample.min()),
            "max": float(sample.max()),
            "mean": float(sample.mean()),
            "std": float(sample.std(ddof=1)),
            "median": float(np.median(sample)),
        },
        "percentiles": {str(p): float(np.percentile(sample, p)) for p in PERCENTILES},
    }
    json_path = OUT / "sun_elevation_canonical.json"
    json_path.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload["stats"], indent=1))
    print(f"written {table} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
