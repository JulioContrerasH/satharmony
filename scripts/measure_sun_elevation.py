#!/usr/bin/env python
"""Measure the sun-elevation distribution of the MSS/TM calibration pairs.

The emulator used a placeholder ``sun_elevation_deg = 25-70``.  This script
reads the 178 MSS/TM pairs selected for the radiometric analysis, joins their
GEE/MTL metadata, and writes the measured distribution used by
``PipelineConfig.mss5_esa_gtc_v1()``.

The 178 ``id_name`` values are in
``/data/databases/CLOUD_MSS/calibration/mss_tm/selection.csv``.  The matching
``sun_elevation`` is in the MSS/TM match tables under
``assets/md/metadata/``; the v2 file covers all 178 pairs.

Outputs
-------
- ``sun_elevation_pairs.csv``         one row per pair with the value used.
- ``sun_elevation_distribution.json`` n, quantiles, mean/std, histogram.
- ``figures/sun_elevation_hist.png``  histogram with mean/median.

Usage
-----
    python scripts/measure_sun_elevation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
DEFAULT_SELECTION = Path("/data/databases/CLOUD_MSS/calibration/mss_tm/selection.csv")
DEFAULT_METADATA = PROJECT / "assets/md/metadata"
DEFAULT_OUT = PROJECT / "assets/md/fuentes/srf"
CANDIDATES = (
    "mss_tm_esa_gee_matches_filter_dtm_finetune_v2.csv",
    "mss_tm_esa_gee_matches_filter_dtm_finetune_v4.csv",
    "mss_tm_esa_gee_matches_filter_dtm_finetune.csv",
)
PERCENTILES = (0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 100.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pairs(selection_path: Path, metadata_dir: Path) -> tuple[pd.DataFrame, Path]:
    selection = pd.read_csv(selection_path)
    if "id_name" not in selection:
        raise ValueError(f"{selection_path} has no id_name column")
    wanted = set(selection["id_name"])

    source = None
    matches = None
    for name in CANDIDATES:
        path = metadata_dir / name
        if not path.exists():
            continue
        table = pd.read_csv(path, usecols=["id_name", "sun_elevation", "time_gee_mss"])
        table = table.drop_duplicates("id_name")
        if wanted.issubset(set(table["id_name"])):
            source, matches = path, table
            break

    if source is None or matches is None:
        raise FileNotFoundError(
            "no metadata table in "
            f"{metadata_dir} covers all {len(wanted)} selected id_name values; "
            f"tried {CANDIDATES}"
        )

    pairs = selection.merge(matches, on="id_name", how="left", validate="one_to_one")
    missing = pairs["sun_elevation"].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} pairs have no sun_elevation")
    return pairs, source


def stats(values: np.ndarray) -> dict[str, float | int]:
    quantiles = np.percentile(values, PERCENTILES)
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        **{f"p{int(q):02d}": float(value) for q, value in zip(PERCENTILES, quantiles, strict=True)},
    }


def plot(values: np.ndarray, path: Path) -> dict[str, list[float] | list[int]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts, edges = np.histogram(values, bins=20)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.hist(values, bins=edges, color="#4c78a8", edgecolor="black", linewidth=0.5)
    ax.axvline(
        float(np.mean(values)),
        color="C1",
        linestyle="--",
        linewidth=1.2,
        label=f"mean = {np.mean(values):.1f}°",
    )
    ax.axvline(
        float(np.median(values)),
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"median = {np.median(values):.1f}°",
    )
    ax.set_xlabel("Sun elevation (degrees)")
    ax.set_ylabel("MSS/TM pairs")
    ax.set_title("Sun elevation of the 178 MSS/TM calibration pairs")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.1, pil_kwargs={"optimize": True})
    plt.close(fig)
    return {
        "bin_edges": [float(edge) for edge in edges],
        "counts": [int(count) for count in counts],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    pairs, source = load_pairs(args.selection, args.metadata_dir)
    values = pairs["sun_elevation"].to_numpy(dtype=float)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        column
        for column in ("id_name", "time_gee_mss", "sun_elevation", "category", "split")
        if column in pairs
    ]
    pairs[columns].sort_values("sun_elevation").to_csv(
        args.out_dir / "sun_elevation_pairs.csv", index=False, float_format="%.6f"
    )

    distribution = plot(values, args.out_dir / "figures/sun_elevation_hist.png")
    summary = stats(values)
    try:
        metadata_source = str(source.resolve().relative_to(PROJECT))
    except ValueError:
        metadata_source = str(source)
    payload = {
        "n": summary["n"],
        "selection": str(args.selection),
        "metadata_source": metadata_source,
        "selection_sha256": sha256(args.selection),
        "metadata_sha256": sha256(source),
        "stats": summary,
        "histogram": distribution,
        "method": (
            "Merged the 178 id_name values of the MSS/TM selection with the "
            "GEE match table on id_name and took the SUN_ELEVATION of the MSS "
            "scene (the same value used by dn_to_toa)."
        ),
        "note": (
            "The 178 pairs are the QA-approved analysis set, not the whole ESA "
            "archive. They are winter/autumn scenes (SZA 28-82°), so the summer "
            "tail is under-represented; the emulator uses a truncated normal "
            "with these measured moments for augmentation, not a uniform range."
        ),
    }
    (args.out_dir / "sun_elevation_distribution.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    print(f"pairs: {summary['n']}")
    print(
        "sun elevation: "
        f"min={summary['min']:.2f} p5={summary['p05']:.2f} "
        f"median={summary['median']:.2f} mean={summary['mean']:.2f} "
        f"std={summary['std']:.2f} p95={summary['p95']:.2f} max={summary['max']:.2f}"
    )
    print(f"wrote {args.out_dir / 'sun_elevation_distribution.json'}")
    print(f"wrote {args.out_dir / 'sun_elevation_pairs.csv'}")
    print(f"wrote {args.out_dir / 'figures/sun_elevation_hist.png'}")


if __name__ == "__main__":
    main()
