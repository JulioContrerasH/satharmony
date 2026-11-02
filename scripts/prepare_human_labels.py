#!/usr/bin/env python
"""Prepare 60 m real-MSS chips for human cloud/shadow labelling.

The chips come from the MSS L1 scenes paired with TM (``pairs/*_mss.tif``,
native 60 m grid).  Scenes used to calibrate the emulator (the 178 pairs in
``assets/md/fuentes/srf/sun_elevation_pairs.csv``) are excluded, so the
human labels can act as an independent test set later.

Nothing is labelled here: the script only writes the images, a georeferenced
4-band GeoTIFF per chip, an index CSV with a QA *hint* (the old cloud mask,
used only to pick scenes that contain clouds) and empty annotation CSVs for
two annotators.

Outputs (``--out``)::

    chips/<chip_id>.png           false colour (MSS4, MSS2, MSS1), 3x upscaled
    chips/<chip_id>.tif           4-band uint16 60 m GeoTIFF
    chips_index.csv               chip metadata and QA hint (not a label)
    anotaciones_anotador1.csv     empty template
    anotaciones_anotador2.csv     empty template
    anotador.html                 offline mask painter
    instrucciones.md              protocol for the two annotators
    mascaras/anotador1/           output folder for painted masks
    mascaras/anotador2/

Usage
-----
    python scripts/prepare_human_labels.py
    python scripts/prepare_human_labels.py --chips 40 --chip-size 128
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_PAIRS = Path("/data/databases/CLOUD_MSS/pairs")
DEFAULT_QA = Path("/data/databases/CLOUD_MSS/pairs_optimization")
DEFAULT_CALIBRATION = REPO.parent / "assets/md/fuentes/srf/sun_elevation_pairs.csv"
DEFAULT_OUT = REPO.parent / "assets/md/fuentes/labels-humano"

#: Chips per old-QA cloud stratum.  The QA is only a hint for scene choice.
STRATA = ("clear", "low_cloud", "mid_cloud", "high_cloud")
QUOTAS = {"clear": 8, "low_cloud": 8, "mid_cloud": 12, "high_cloud": 12}

#: Target QA cloud fraction of the window for each stratum.  Picking the
#: cloudiest window instead would give 100 % cloud chips and no edges.
TARGETS = {"clear": 0.0, "low_cloud": 0.08, "mid_cloud": 0.30, "high_cloud": 0.70}

ANNOTATION_COLUMNS = (
    "chip_id",
    "clase_dominante",
    "pct_clear",
    "pct_thick",
    "pct_thin",
    "pct_shadow",
    "pct_dudoso",
    "comentario",
)


def stratum_of(fraction: float) -> str:
    """Old-QA cloud fraction -> scene stratum (selection only)."""
    if fraction < 0.02:
        return "clear"
    if fraction < 0.15:
        return "low_cloud"
    if fraction < 0.50:
        return "mid_cloud"
    return "high_cloud"


def calibration_ids(csv_path: Path) -> set[str]:
    """Scene ids used to calibrate the emulator; these must stay out."""
    if not csv_path.exists():
        return set()
    with csv_path.open() as fh:
        return {row["id_name"] for row in csv.DictReader(fh)}


def scene_candidates(
    pairs: Path,
    qa: Path,
    excluded: set[str],
    pool_size: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Sample paired scenes and read their old-QA cloud fractions.

    Reading all 2.7k cloud masks would be ~12 minutes, so a random pool is
    scored at a decimated resolution (it is only used to balance the scene
    choice).  The 178 calibration scenes are always excluded.
    """
    import rasterio

    available = [
        qa_dir
        for qa_dir in sorted(qa.iterdir())
        if qa_dir.is_dir()
        and qa_dir.name not in excluded
        and (pairs / f"{qa_dir.name}_mss.tif").exists()
        and (qa_dir / "cloud_mask_v3.tif").exists()
    ]
    rng.shuffle(available)
    rows = []
    for qa_dir in available[:pool_size]:
        mss_path = pairs / f"{qa_dir.name}_mss.tif"
        cloud_path = qa_dir / "cloud_mask_v3.tif"
        with rasterio.open(cloud_path) as src:
            cloud = src.read(1, out_shape=(512, 512))
        cloud_fraction = float(np.count_nonzero(cloud == 1) / cloud.size)
        shadow = np.zeros(cloud.shape, np.uint8)
        shadow_path = qa_dir / "shadow_zone.tif"
        if shadow_path.exists():
            with rasterio.open(shadow_path) as src:
                shadow = src.read(1, out_shape=(512, 512))
        rows.append(
            {
                "scene_id": qa_dir.name,
                "mss_path": mss_path,
                "cloud_path": cloud_path,
                "shadow_path": shadow_path if shadow_path.exists() else None,
                "cloud_fraction": cloud_fraction,
                "shadow_fraction": float(np.count_nonzero(shadow == 1) / shadow.size),
                "stratum": stratum_of(cloud_fraction),
            }
        )
    return rows


def qa_fraction_on_mss(qa_path: Path, mss_path: Path) -> np.ndarray:
    """Reproject a 0/1 QA mask to the MSS 60 m grid as a fraction per cell."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(mss_path) as dst:
        height, width = dst.shape
        transform, crs = dst.transform, dst.crs
    with rasterio.open(qa_path) as src:
        source = src.read(1).astype(np.float32)
        source_transform, source_crs = src.transform, src.crs
    destination = np.zeros((height, width), dtype=np.float32)
    reproject(
        source=source,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=transform,
        dst_crs=crs,
        resampling=Resampling.average,
    )
    return destination


def _window_sums(values: np.ndarray, size: int) -> np.ndarray:
    """Sum of every ``size x size`` window, via a summed-area table."""
    integral = np.zeros((values.shape[0] + 1, values.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = values.cumsum(axis=0).cumsum(axis=1)
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )


def best_window(
    score: np.ndarray,
    valid: np.ndarray,
    size: int,
    target: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Valid window whose mean score is closest to ``target``.

    ``target=0`` for clear chips; otherwise the window is chosen to contain a
    cloud fraction near the stratum target, so the chips show edges and
    shadows instead of a uniform overcast.  Ties are broken at random.
    """
    sums = _window_sums(score.astype(np.float64), size)
    invalid = _window_sums((~valid).astype(np.float64), size)
    rows, cols = np.nonzero(invalid == 0)
    if rows.size == 0:
        raise ValueError("no fully valid window")
    fractions = sums[rows, cols] / (size * size)
    distance = np.abs(fractions - target)
    best = np.flatnonzero(distance <= distance.min() + 1e-6)
    index = int(best[int(rng.integers(best.size))])
    return int(rows[index]), int(cols[index])


def write_png(mss: np.ndarray, path: Path, scale: int = 3) -> None:
    """False colour (MSS4, MSS2, MSS1) with a per-chip percentile stretch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = np.stack([mss[3], mss[1], mss[0]], axis=-1).astype(np.float32)
    stretched = np.zeros_like(rgb)
    for band in range(3):
        values = rgb[..., band]
        low, high = np.percentile(values, (2.0, 98.0))
        if high <= low:
            high = low + 1.0
        stretched[..., band] = np.clip((values - low) / (high - low), 0.0, 1.0)
    display = np.repeat(np.repeat(stretched, scale, axis=0), scale, axis=1)
    fig = plt.figure(figsize=(display.shape[1] / 100.0, display.shape[0] / 100.0), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(display, interpolation="nearest")
    ax.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_tif(mss: np.ndarray, path: Path, transform, crs) -> None:
    """Georeferenced 4-band uint16 GeoTIFF of the chip."""
    import rasterio

    profile = {
        "driver": "GTiff",
        "height": mss.shape[1],
        "width": mss.shape[2],
        "count": mss.shape[0],
        "dtype": "uint16",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mss.astype(np.uint16))


def write_annotation_csv(path: Path, chip_ids: list[str]) -> None:
    """Empty two-annotator template; no label is written by this script."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ANNOTATION_COLUMNS))
        writer.writeheader()
        for chip_id in chip_ids:
            writer.writerow({"chip_id": chip_id})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--chips", type=int, default=40, help="30-50 recommended")
    parser.add_argument("--chip-size", type=int, default=128)
    parser.add_argument("--pool", type=int, default=400, help="scenes scored for selection")
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()

    if not 30 <= args.chips <= 50:
        print("warning: the protocol expects 30-50 chips", file=sys.stderr)

    import rasterio

    rng = np.random.default_rng(args.seed)
    excluded = calibration_ids(args.calibration)
    candidates = scene_candidates(args.pairs, args.qa, excluded, args.pool, rng)
    print(f"{len(candidates)} sampled non-calibration scenes with MSS and QA")

    by_stratum: dict[str, list[dict]] = {name: [] for name in STRATA}
    for row in candidates:
        by_stratum[row["stratum"]].append(row)
    for pool in by_stratum.values():
        rng.shuffle(pool)

    quota = dict(QUOTAS)
    # Scale the quotas to the requested total.
    scale = args.chips / sum(quota.values())
    quota = {name: max(1, round(value * scale)) for name, value in quota.items()}
    while sum(quota.values()) < args.chips:
        quota["mid_cloud"] += 1
    while sum(quota.values()) > args.chips:
        quota["mid_cloud"] -= 1

    chips = []
    for stratum in STRATA:
        for row in by_stratum[stratum]:
            if sum(1 for chip in chips if chip["stratum"] == stratum) >= quota[stratum]:
                break
            try:
                cloud60 = qa_fraction_on_mss(row["cloud_path"], row["mss_path"])
            except Exception as exc:
                print(f"  skip {row['scene_id']}: {exc}", file=sys.stderr)
                continue
            with rasterio.open(row["mss_path"]) as src:
                mss = src.read()
                transform, crs = src.transform, src.crs
            valid = np.all(mss > 0, axis=0)
            if valid.mean() < 0.5:
                continue
            try:
                row_idx, col_idx = best_window(
                    cloud60.astype(np.float64),
                    valid,
                    args.chip_size,
                    TARGETS[row["stratum"]],
                    rng,
                )
            except ValueError:
                continue
            chip = mss[
                :,
                row_idx : row_idx + args.chip_size,
                col_idx : col_idx + args.chip_size,
            ]
            if chip.shape[1:] != (args.chip_size, args.chip_size):
                continue
            chip_cloud = float(
                cloud60[
                    row_idx : row_idx + args.chip_size,
                    col_idx : col_idx + args.chip_size,
                ].mean()
            )
            # Local transform of the chip.
            from rasterio.transform import Affine

            chip_transform = transform * Affine.translation(col_idx, row_idx)
            chips.append(
                {
                    "chip_id": row["scene_id"],
                    "scene_id": row["scene_id"],
                    "stratum": row["stratum"],
                    "cloud_hint": chip_cloud,
                    "shadow_hint": row["shadow_fraction"],
                    "row": row_idx,
                    "col": col_idx,
                    "image": chip,
                    "transform": chip_transform,
                    "crs": crs,
                }
            )
            print(
                f"  {row['scene_id']} {row['stratum']}: window ({row_idx},{col_idx}), "
                f"QA cloud hint {chip_cloud:.1%}"
            )
            if len(chips) >= args.chips:
                break
        if len(chips) >= args.chips:
            break

    if len(chips) < 30:
        print(f"only {len(chips)} chips could be prepared; check the inputs", file=sys.stderr)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    chip_ids = []
    index_rows = []
    for chip in chips:
        chip_id = chip["chip_id"]
        chip_ids.append(chip_id)
        png_path = out / "chips" / f"{chip_id}.png"
        tif_path = out / "chips" / f"{chip_id}.tif"
        write_png(chip["image"], png_path)
        write_tif(chip["image"], tif_path, chip["transform"], chip["crs"])
        index_rows.append(
            {
                "chip_id": chip_id,
                "path_png": str(png_path.relative_to(out)),
                "path_tif": str(tif_path.relative_to(out)),
                "scene_id": chip["scene_id"],
                "row": chip["row"],
                "col": chip["col"],
                "size_px": args.chip_size,
                "gsd_m": 60,
                "qa_stratum_hint": chip["stratum"],
                "qa_cloud_frac_hint": round(chip["cloud_hint"], 4),
                "qa_shadow_frac_hint": round(chip["shadow_hint"], 4),
                "notes": "QA hint only, not a label",
            }
        )

    with (out / "chips_index.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    for annotator in ("anotador1", "anotador2"):
        write_annotation_csv(out / f"anotaciones_{annotator}.csv", chip_ids)
        (out / "mascaras" / annotator).mkdir(parents=True, exist_ok=True)
        (out / "mascaras" / annotator / ".gitkeep").touch()

    print(f"wrote {len(chips)} chips to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
