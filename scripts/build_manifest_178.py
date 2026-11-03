#!/usr/bin/env python
"""Build the 178-pair MSS/TM manifest with hashes and provenance."""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.manifests.manifest import (  # noqa: E402
    ManifestRecord,
    sha256_file,
    write_manifest,
)

SELECTION = Path("/data/databases/CLOUD_MSS/calibration/mss_tm/selection.csv")
ROOT = Path("/data/databases/CLOUD_MSS/pairs_optimization")
OUT = REPO / "manifests/manifest_178.csv"


def _one(row: dict[str, str]) -> ManifestRecord:
    scene = ROOT / row["id_name"]
    mss = scene / "mss_aligned.tif"
    tm = scene / "tm_aligned.tif"
    if not (mss.exists() and tm.exists()):
        raise FileNotFoundError(f"{scene}: missing aligned rasters")
    return ManifestRecord(
        id=row["id_name"],
        kind="mss_tm",
        input_path=mss,
        reference_path=tm,
        input_sha256=sha256_file(mss),
        reference_sha256=sha256_file(tm),
        cloud_fraction=float(row["cloud_frac"]),
        reason=f"178-pair calibration, cloud bin {row['bin']}",
    )


def main() -> None:
    with SELECTION.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_one, row): row["id_name"] for row in rows}
        for future in as_completed(futures):
            records.append(future.result())
            print(f"{len(records):3d} {futures[future]}", flush=True)
    records.sort(key=lambda record: record.id)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(records, OUT)
    print(f"wrote {OUT} ({len(records)} pairs)")


if __name__ == "__main__":
    main()
