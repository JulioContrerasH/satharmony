#!/usr/bin/env python
"""Build a pseudo-MSS fixture from CloudSEN12 high-quality chips.

For every selected CloudSEN12 chip (10 m, 13 bands, strong labels):

1. Read the stack in CloudSEN12 order (B1..B8, B8A, B9, B10, B11, B12) and
   reorder it to the emulator order (B1..B12, B8A).
2. Emulate Landsat MSS with the calibrated profile
   (:meth:`PipelineConfig.mss5_esa_gtc_v1`): SBAF, PSF 79-82 m, 60 m grid,
   physical saturation, striping and noise.  The scene sun elevation and the
   S2 sensor (A/B/C) come from the CloudSEN12 metadata, not from the default
   distribution.
3. Aggregate the CloudSEN12 labels 10 m -> 60 m with the conservative rule
   (``aggregate_cloudsen12``): a class needs >=50 % of the block; a
   no-majority block with >=50 % contaminated pixels takes its most frequent
   contaminated class.
4. Cut one seeded window of ``--chip-size`` cells from the fully valid
   84 x 84 block, and record the class fractions and the regime.

Outputs (``--out``)::

    images.npy            (N, 4, S, S) uint16, reflectance x 10000
    labels.npy            (N, S, S) uint8, 0 clear / 1 thick / 2 thin / 3 shadow
    manifest.csv          one row per chip with fractions, window, seed, hash
    label_distribution.json  chip and pixel counts per class
    fixture.json          build parameters, array hashes and versions

Usage
-----
    python scripts/build_pseudo_mss_fixture.py --limit 40 --workers 2
    python scripts/build_pseudo_mss_fixture.py --chips 2000 --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.datasets.cloudsen12 import (  # noqa: E402
    SCALE_FACTOR,
    array_sha256,
    class_fractions,
    is_shadow_rich,
    read_scene,
    regime_of,
    select_window,
)
from satharmony.emulators.s2_to_mss.config import PipelineConfig, Range  # noqa: E402
from satharmony.emulators.s2_to_mss.functional import emulate_s2_with_labels  # noqa: E402

DEFAULT_DATA = Path("/data/databases/CMIXII/CLOUDSEN12_1K")
DEFAULT_CSV = Path("/data/databases/CMIXII/dataset.csv")
DEFAULT_OUT = Path("/data/databases/CLOUD_MSS/calibration/pseudo_mss_fixture")

#: Final chips per cloud regime.  Sums to the default ``--chips``.
REGIME_QUOTAS = {
    "clear": 0.24,
    "low_cloud": 0.25,
    "mid_cloud": 0.26,
    "high_cloud": 0.25,
}

#: Scene-level cloud strata used to pick candidates before emulation.
CANDIDATE_STRATA = {
    "clear": 0.20,
    "low_cloud": 0.24,
    "mid_cloud": 0.26,
    "high_cloud": 0.30,
}


def stable_seed(base: int, key: str) -> int:
    """Deterministic 32-bit seed from a base seed and a scene id."""
    digest = hashlib.sha256(f"{base}:{key}".encode()).hexdigest()
    return int(digest[:8], 16)


def scene_cloud_fraction(metadata: dict) -> float:
    """Contaminated fraction of a CloudSEN12 scene, from its metadata."""
    return (
        float(metadata.get("thick_percentage", 0.0))
        + float(metadata.get("thin_percentage", 0.0))
        + float(metadata.get("cloud_shadow_percentage", 0.0))
    ) / 100.0


def scene_stratum(fraction: float) -> str:
    """Scene-level stratum used only to balance the candidate list."""
    if fraction < 0.02:
        return "clear"
    if fraction < 0.15:
        return "low_cloud"
    if fraction < 0.50:
        return "mid_cloud"
    return "high_cloud"


def sensor_from_metadata(metadata: dict) -> str:
    """``S2A``/``S2B``/``S2C`` from the CloudSEN12 ``s2_id``."""
    s2_id = str(metadata.get("s2_id", "S2A"))
    for sensor in ("S2A", "S2B", "S2C"):
        if s2_id.startswith(sensor):
            return sensor
    return "S2A"


def profile_for_scene(metadata: dict) -> PipelineConfig:
    """Calibrated profile with the real sensor and sun elevation of the scene."""
    cfg = PipelineConfig.mss5_esa_gtc_v1()
    cfg.spectral.sensor = sensor_from_metadata(metadata)  # type: ignore[assignment]
    zenith = metadata.get("s2_mean_solar_zenith_angle")
    if zenith is not None:
        elevation = float(np.clip(90.0 - float(zenith), 1.0, 89.0))
        cfg.radiometric.sun_elevation_deg = Range(
            min=elevation, max=elevation, distribution="uniform"
        )
    return cfg


def process_scene(task: tuple[Path, str, int, int, int]) -> dict | None:
    """Read, emulate and cut one chip.  Runs in a worker process."""
    data_dir, scene_id, chip_size, seed, scale_factor = task
    scene_dir = data_dir / scene_id
    try:
        scene = read_scene(scene_dir, scale_factor=scale_factor)
    except (FileNotFoundError, OSError):
        return None

    metadata = scene.metadata
    cfg = profile_for_scene(metadata)
    rng = np.random.default_rng(stable_seed(seed, scene_id))

    mss, labels60, sf = emulate_s2_with_labels(
        scene.s2,
        scene.labels,
        seed=stable_seed(seed, f"emulate:{scene_id}"),
        config=cfg,
    )
    if sf != scale_factor:
        raise ValueError(f"{scene_id}: scale factor {sf} != {scale_factor}")

    window = select_window(scene.valid60, chip_size, rng)
    if window is None:
        return None
    row, col = window
    image = mss[:, row : row + chip_size, col : col + chip_size]
    labels = labels60[row : row + chip_size, col : col + chip_size]
    if labels.shape != (chip_size, chip_size):
        raise ValueError(f"{scene_id}: chip {labels.shape} does not match {chip_size}")

    fractions = class_fractions(labels)
    return {
        "scene_id": scene_id,
        "split": str(metadata.get("tortilla:data_split", "train")),
        "sensor": sensor_from_metadata(metadata),
        "sun_elevation_deg": float(cfg.radiometric.sun_elevation_deg.min),
        "window_row": row,
        "window_col": col,
        "seed": seed,
        "image": image.astype(np.float32),
        "labels": labels.astype(np.uint8),
        "fractions": fractions,
        "regime": regime_of(fractions),
        "shadow_rich": is_shadow_rich(fractions),
    }


def pick_candidates(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Balance the candidate scenes by scene-level cloud stratum."""
    rng = np.random.default_rng(seed)
    by_stratum: dict[str, list[dict]] = {}
    for row in rows:
        by_stratum.setdefault(scene_stratum(scene_cloud_fraction(row["metadata"])), []).append(row)

    selected: list[dict] = []
    for stratum, weight in CANDIDATE_STRATA.items():
        pool = by_stratum.get(stratum, [])
        take = min(len(pool), round(n * weight))
        if take:
            idx = rng.choice(len(pool), size=take, replace=False)
            selected.extend(pool[int(i)] for i in np.sort(idx))
    # Fill the remainder from any stratum.
    if len(selected) < n:
        chosen = {row["scene_id"] for row in selected}
        rest = [row for row in rows if row["scene_id"] not in chosen]
        rng.shuffle(rest)
        selected.extend(rest[: n - len(selected)])
    return selected[:n]


def select_chips(chips: list[dict], total: int, seed: int) -> list[dict]:
    """Keep the requested total, respecting the regime quotas."""
    rng = np.random.default_rng(seed + 1)
    by_regime: dict[str, list[dict]] = {}
    for chip in chips:
        by_regime.setdefault(chip["regime"], []).append(chip)
    for pool in by_regime.values():
        rng.shuffle(pool)

    quotas = {regime: round(total * share) for regime, share in REGIME_QUOTAS.items()}
    quotas["high_cloud"] += total - sum(quotas.values())  # rounding remainder

    selected = []
    leftover = []
    for regime, quota in quotas.items():
        pool = by_regime.get(regime, [])
        selected.extend(pool[:quota])
        leftover.extend(pool[quota:])
    if len(selected) < total:
        rng.shuffle(leftover)
        selected.extend(leftover[: total - len(selected)])
    return selected


def write_fixture(chips: list[dict], out_dir: Path, build: dict) -> dict:
    """Write the .npy arrays, manifest and JSON metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chips = sorted(chips, key=lambda chip: chip["scene_id"])
    shape = chips[0]["image"].shape
    images = np.empty((len(chips), *shape), dtype=np.uint16)
    labels = np.empty((len(chips), *chips[0]["labels"].shape), dtype=np.uint8)

    rows = []
    for i, chip in enumerate(chips):
        image = np.clip(chip["image"], 0.0, 1.0)
        quantized = np.rint(image * 10000.0).astype(np.uint16)
        images[i] = quantized
        labels[i] = chip["labels"]
        fractions = chip["fractions"]
        rows.append(
            {
                "chip_id": chip["scene_id"],
                "split": chip["split"],
                "regime": chip["regime"],
                "shadow_rich": int(chip["shadow_rich"]),
                "sensor": chip["sensor"],
                "sun_elevation_deg": round(chip["sun_elevation_deg"], 3),
                "window_row": chip["window_row"],
                "window_col": chip["window_col"],
                "seed": chip["seed"],
                "clear_frac": round(fractions["clear"], 6),
                "thick_frac": round(fractions["thick_cloud"], 6),
                "thin_frac": round(fractions["thin_cloud"], 6),
                "shadow_frac": round(fractions["cloud_shadow"], 6),
                "contaminated_frac": round(fractions["contaminated"], 6),
                "sha256": array_sha256(quantized, chip["labels"]),
            }
        )

    np.save(out_dir / "images.npy", images)
    np.save(out_dir / "labels.npy", labels)

    import csv

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    classes = labels.ravel()
    pixel_counts = {
        name: int(np.count_nonzero(classes == code))
        for code, name in {0: "clear", 1: "thick_cloud", 2: "thin_cloud", 3: "cloud_shadow"}.items()
    }
    total_pixels = int(classes.size)
    regime_counts = Counter(row["regime"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)

    distribution = {
        "n_chips": len(chips),
        "chip_size_px": int(shape[1]),
        "chips_per_regime": dict(sorted(regime_counts.items())),
        "chips_per_split": dict(sorted(split_counts.items())),
        "chips_shadow_rich": int(sum(row["shadow_rich"] for row in rows)),
        "pixels_per_class": pixel_counts,
        "pixel_fraction_per_class": {
            name: round(count / total_pixels, 6) for name, count in pixel_counts.items()
        },
        "mean_chip_fraction_per_class": {
            name: round(float(np.mean([row[key] for row in rows])), 6)
            for name, key in {
                "clear": "clear_frac",
                "thick_cloud": "thick_frac",
                "thin_cloud": "thin_frac",
                "cloud_shadow": "shadow_frac",
            }.items()
        },
    }
    (out_dir / "label_distribution.json").write_text(json.dumps(distribution, indent=2) + "\n")

    metadata = {
        **build,
        "images_sha256": _file_sha256(out_dir / "images.npy"),
        "labels_sha256": _file_sha256(out_dir / "labels.npy"),
        "images_shape": [int(v) for v in images.shape],
        "labels_shape": [int(v) for v in labels.shape],
        "label_distribution": distribution,
    }
    (out_dir / "fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidates(data: Path, csv_path: Path, label_type: str = "high") -> list[dict]:
    """Read the dataset CSV and keep the chips with the requested label type."""
    import csv

    rows = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("label_type") != label_type:
                continue
            scene_id = Path(row["input"]).parent.name
            meta_path = data / scene_id / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                metadata = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                continue
            rows.append({"scene_id": scene_id, "metadata": metadata})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--chips", type=int, default=2000, help="final fixture size")
    parser.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="scenes to emulate before selecting (default: chips x 1.35)",
    )
    parser.add_argument("--chip-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="pilot mode: process at most this many scenes, write everything",
    )
    parser.add_argument("--label-type", default="high")
    args = parser.parse_args()

    if args.chip_size > 84:
        parser.error("--chip-size cannot exceed the 84 fully valid 60 m cells")

    t0 = time.time()
    rows = load_candidates(args.data, args.dataset_csv, args.label_type)
    if not rows:
        parser.error(f"no {args.label_type} chips found in {args.dataset_csv}")
    n_candidates = args.candidates or round(args.chips * 1.35)
    if args.limit is not None:
        n_candidates = min(n_candidates, args.limit)
        args.chips = min(args.chips, args.limit)
    candidates = pick_candidates(rows, n_candidates, args.seed)
    print(
        f"{len(rows)} {args.label_type} chips available; "
        f"{len(candidates)} candidates, target {args.chips} chips"
    )

    tasks = [
        (args.data, row["scene_id"], args.chip_size, args.seed, SCALE_FACTOR)
        for row in candidates
    ]
    chips: list[dict] = []
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_scene, task): task[1] for task in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                chip = future.result()
            except Exception as exc:  # report and keep going
                failures += 1
                print(f"  failed {futures[future]}: {exc}", file=sys.stderr)
                continue
            if chip is not None:
                chips.append(chip)
            if done % 100 == 0:
                print(f"  {done}/{len(tasks)} scenes, {len(chips)} chips, {failures} failed")

    if not chips:
        print("no chips produced", file=sys.stderr)
        return 1

    selected = chips if args.limit is not None else select_chips(chips, args.chips, args.seed)

    build = {
        "created_by": "scripts/build_pseudo_mss_fixture.py",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(args.data),
        "dataset_csv": str(args.dataset_csv),
        "label_type": args.label_type,
        "profile": "mss5_esa_gtc_v1",
        "labels": "aggregate_cloudsen12 conservative rule, 10 m -> 60 m",
        "band_order": "CloudSEN12 B1..B8,B8A,B9,B10,B11,B12 -> package B1..B12,B8A",
        "chip_size_px": args.chip_size,
        "n_candidates_emulated": len(chips),
        "n_scenes_failed": failures,
        "seed": args.seed,
        "elapsed_s": round(time.time() - t0, 1),
    }
    metadata = write_fixture(selected, args.out, build)

    print(
        f"wrote {metadata['images_shape'][0]} chips to {args.out} "
        f"in {time.time() - t0:.0f} s"
    )
    print(json.dumps(metadata["label_distribution"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
