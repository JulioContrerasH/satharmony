#!/usr/bin/env python
"""Build a real-MSS fixture from the v7 train/validation scenes.

The G4/G4b models were trained on CloudSEN12 emulated to MSS.  The v7 scenes
already have MSS at 60 m and a 3-class label (`label_refined_mss`) that agrees
with the QA4EO-2 product.  Training on those real pixels is the direct way to
close the domain gap that makes the emulated models predict cloud over bright,
clear surfaces.

Inputs
------
``assets/md/manifest-finetune-v7.csv`` rows with ``split`` in
``train``/``val``.  The test split is never read.

Label mapping
-------------
`label_refined_mss` is 3-class: 0 clear, 1 cloud, 2 shadow.  The model keeps
the package's 4-class head, so the mapping is 0 -> 0 clear, 1 -> 1 thick cloud,
2 -> 3 shadow.  The thin-cloud class stays empty; the loss gives it zero weight
and its logit is suppressed in the checkpoint.  The 3-class comparison merges
thick into cloud, so nothing is lost for the current metric.

Outputs
-------
``images.npy`` ``(N, 4, 64, 64)`` uint16, ``labels.npy`` ``(N, 64, 64)``
uint8, ``manifest.csv`` one row per chip, ``fixture.json`` with hashes and
build parameters.

Usage
-----
    python scripts/build_real_mss_fixture.py --limit 4 --out /tmp/real_mss_smoke
    python scripts/build_real_mss_fixture.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_MANIFEST = REPO.parent / "assets/md/manifest-finetune-v7.csv"
DEFAULT_DATA = Path("/data/databases/CLOUD_MSS")
DEFAULT_OUT = Path("/data/databases/CLOUD_MSS/calibration/real_mss_fixture")
CHIP = 64
STRIDE = 64
MAX_PER_SCENE = 32
MIN_VALID = 0.7
SEED = 20261024


def array_sha256(array: np.ndarray) -> str:
    """SHA-256 of the C-order bytes of an array."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_label_on_mss(
    mss_path: Path, label_path: Path
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read MSS and its label on the MSS grid, reprojecting the label if needed."""
    with rasterio.open(mss_path) as src:
        mss = src.read()
        profile = src.profile.copy()
    with rasterio.open(label_path) as src:
        label = src.read(1)
        if (
            label.shape == mss.shape[-2:]
            and src.transform == profile["transform"]
            and src.crs == profile["crs"]
        ):
            return mss, label, profile
        destination = np.zeros(mss.shape[-2:], dtype=np.uint8)
        reproject(
            source=label,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            resampling=Resampling.nearest,
        )
        return mss, destination, profile


def candidates(
    mss: np.ndarray, label: np.ndarray, minimum_valid: float
) -> list[dict]:
    """Non-overlapping 64 x 64 windows with their valid/cloud fractions."""
    height, width = mss.shape[-2:]
    valid = np.all(mss > 0, axis=0)
    out = []
    for row in range(0, height - CHIP + 1, STRIDE):
        for col in range(0, width - CHIP + 1, STRIDE):
            valid_window = valid[row : row + CHIP, col : col + CHIP]
            fraction_valid = float(valid_window.mean())
            if fraction_valid < minimum_valid:
                continue
            cloud = float((label[row : row + CHIP, col : col + CHIP] == 1).mean())
            shadow = float((label[row : row + CHIP, col : col + CHIP] == 2).mean())
            out.append(
                {
                    "row": row,
                    "col": col,
                    "valid_fraction": round(fraction_valid, 6),
                    "cloud_fraction": round(cloud, 6),
                    "shadow_fraction": round(shadow, 6),
                }
            )
    return out


def choose(windows: list[dict], limit: int, rng: np.random.Generator) -> list[dict]:
    """Stratified sample of windows: cloud, mixed and clear."""
    if not windows:
        return []
    high = [w for w in windows if w["cloud_fraction"] >= 0.2]
    mid = [w for w in windows if 0.01 <= w["cloud_fraction"] < 0.2]
    clear = [w for w in windows if w["cloud_fraction"] < 0.01]
    quota = {"high": round(0.4 * limit), "mid": round(0.4 * limit)}
    quota["clear"] = limit - quota["high"] - quota["mid"]
    chosen: list[dict] = []
    used: set[tuple[int, int]] = set()
    for pool, key in ((high, "high"), (mid, "mid"), (clear, "clear")):
        if not pool or quota[key] <= 0:
            continue
        take = min(quota[key], len(pool))
        indices = rng.choice(len(pool), size=take, replace=False)
        for index in indices:
            window = pool[int(index)]
            key_rc = (window["row"], window["col"])
            if key_rc in used:
                continue
            used.add(key_rc)
            chosen.append(window)
    remaining = [w for w in windows if (w["row"], w["col"]) not in used]
    rng.shuffle(remaining)
    chosen.extend(remaining[: limit - len(chosen)])
    return chosen


def build_scene(
    scene: str, split: str, data: Path, limit: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    """Read one scene, choose chips and map the 3-class label to the 4-class head."""
    folder = data / "pairs_optimization" / scene
    mss, label, _ = read_label_on_mss(
        folder / "mss_original.tif", folder / "label_refined_mss.tif"
    )
    valid = np.all(mss > 0, axis=0)
    windows = candidates(mss, label, MIN_VALID)
    seed = SEED + int(hashlib.sha256(scene.encode()).hexdigest()[:8], 16)
    chosen = choose(windows, limit, np.random.default_rng(seed))
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    rows: list[dict] = []
    for window in chosen:
        row, col = window["row"], window["col"]
        image = mss[:, row : row + CHIP, col : col + CHIP]
        target = label[row : row + CHIP, col : col + CHIP]
        mapped = np.zeros_like(target, dtype=np.uint8)
        mapped[target == 1] = 1
        mapped[target == 2] = 3
        mapped[~valid[row : row + CHIP, col : col + CHIP]] = 0
        images.append(image.astype(np.uint16, copy=False))
        labels.append(mapped)
        rows.append(
            {
                "id_name": scene,
                "split": split,
                "row": row,
                "col": col,
                "valid_fraction": window["valid_fraction"],
                "cloud_fraction": window["cloud_fraction"],
                "shadow_fraction": window["shadow_fraction"],
            }
        )
    return images, labels, rows


def main(argv: list[str] | None = None) -> int:
    """Build the fixture and its manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="first N scenes (smoke test)")
    parser.add_argument("--chips", type=int, default=MAX_PER_SCENE)
    args = parser.parse_args(argv)

    with args.manifest.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] in ("train", "val")]
    if args.limit:
        rows = rows[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    all_images: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    manifest_rows: list[dict] = []
    started = time.time()
    for number, row in enumerate(rows, 1):
        scene = row["id_name"]
        split = "validation" if row["split"] == "val" else "train"
        try:
            images, labels, chips = build_scene(scene, split, args.data, args.chips)
        except Exception as error:
            print(f"[{number}/{len(rows)}] {scene} skipped: {error}")
            continue
        if not images:
            print(f"[{number}/{len(rows)}] {scene} no valid chips")
            continue
        all_images.extend(images)
        all_labels.extend(labels)
        manifest_rows.extend(chips)
        print(
            f"[{number}/{len(rows)}] {scene} {split} {len(images)} chips | "
            f"{time.time() - started:.0f}s"
        )

    if not all_images:
        raise SystemExit("no chips built")
    images = np.stack(all_images)
    labels = np.stack(all_labels)
    np.save(args.out / "images.npy", images)
    np.save(args.out / "labels.npy", labels)
    fields = list(manifest_rows[0])
    with (args.out / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    counts = np.bincount(labels.ravel(), minlength=4).tolist()
    fixture = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "chips": len(images),
        "scenes": len({row["id_name"] for row in manifest_rows}),
        "chip_size": CHIP,
        "stride": STRIDE,
        "max_per_scene": args.chips,
        "min_valid": MIN_VALID,
        "label_mapping": "0 clear, 1 cloud -> 1 thick, 2 shadow -> 3 shadow; thin empty",
        "class_counts": counts,
        "images_sha256": array_sha256(images),
        "labels_sha256": array_sha256(labels),
    }
    (args.out / "fixture.json").write_text(json.dumps(fixture, indent=1))
    (args.out / "label_distribution.json").write_text(
        json.dumps({"class_counts": counts}, indent=1)
    )
    print(
        f"done: {len(images)} chips from {fixture['scenes']} scenes | "
        f"classes {counts} | {time.time() - started:.0f}s"
    )
    print(f"fixture: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
