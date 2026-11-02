#!/usr/bin/env python
"""Average the G4 probabilities over four tile-grid offsets.

The grid-shift test shows that the prediction depends on where the 512 tiles
fall.  Averaging four offsets (0/96/192/288 px) smooths the tile seams without
changing the model or the labels.

Usage
-----
    python scripts/inference_offset_ensemble.py --scenes 1987-05-02_018_182
    python scripts/inference_offset_ensemble.py --split test
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from compare_qa4eo2_v7 import (  # noqa: E402
    DEFAULT_G4_REAL,
    collapse_to_three,
    read_mss60,
    reference_path,
    score_scene,
    three_class_scores,
)
from qa4eo2_labels import predict_proba_unet, valid_mask  # noqa: E402

from satharmony.models import load_model  # noqa: E402
from satharmony.models.metrics import cohen_kappa, mean_iou, overall_accuracy  # noqa: E402

DEFAULT_DATA = Path("/data/databases/CLOUD_MSS")
DEFAULT_MANIFEST = REPO.parent / "assets/md/manifest-finetune-v7.csv"
DEFAULT_OUT = Path("/tmp/inference_offset_ensemble")
OFFSETS = (0, 96, 192, 288)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_G4_REAL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save", nargs="*", default=[])
    args = parser.parse_args(argv)

    if args.scenes:
        scenes = args.scenes
    else:
        with args.manifest.open(newline="") as handle:
            scenes = [
                row["id_name"]
                for row in csv.DictReader(handle)
                if row["split"] == args.split
            ]
    args.out.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint, device=args.device)
    rows: list[dict] = []
    pooled = np.zeros((3, 3), dtype=np.int64)
    started = time.time()
    for number, scene in enumerate(scenes, 1):
        folder = args.data / "pairs_optimization" / scene
        image, _ = read_mss60(folder / "mss_original.tif")
        with rasterio.open(reference_path(folder)) as src:
            reference = src.read(1)
        valid = valid_mask(image) & (reference != 255)
        accumulated = None
        for offset in OFFSETS:
            probabilities = predict_proba_unet(
                model,
                checkpoint,
                image,
                args.device,
                dihedral=True,
                pad_half=True,
                pad_offset=offset,
            )
            accumulated = probabilities if accumulated is None else accumulated + probabilities
        prediction = collapse_to_three((accumulated / len(OFFSETS)).argmax(axis=0))
        scored = score_scene(prediction, reference, valid)
        scores = three_class_scores(scored["confusion"])
        pooled += scored["confusion"]
        rows.append(
            {
                "scene": scene,
                "mean_iou": round(scored["mean_iou"], 6),
                "overall_accuracy": round(scored["overall_accuracy"], 6),
                "kappa": round(scored["kappa"], 6),
                "iou_clear": round(scores["clear"]["iou"], 6),
                "iou_cloud": round(scores["cloud"]["iou"], 6),
                "iou_shadow": round(scores["cloud_shadow"]["iou"], 6),
            }
        )
        if scene in args.save:
            np.save(args.out / f"{scene}.npy", prediction)
        print(
            f"[{number}/{len(scenes)}] {scene} mIoU {rows[-1]['mean_iou']:.4f} "
            f"OA {rows[-1]['overall_accuracy']:.4f} | {time.time() - started:.0f}s"
        )
    with (args.out / "offset_ensemble.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"POOLED mIoU {mean_iou(pooled):.4f} OA {overall_accuracy(pooled):.4f} "
        f"kappa {cohen_kappa(pooled):.4f}"
    )
    print(f"written: {args.out / 'offset_ensemble.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
