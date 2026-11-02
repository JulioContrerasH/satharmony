#!/usr/bin/env python
"""Ablation of the sliding-window inference of the real-MSS G4 model.

The model is trained on 64 x 64 chips but the comparison uses 512 x 512 tiles
with a Hann window and dihedral TTA.  The border of a scene sits on the zero
edge of the Hann window because the current tiler only pads enough to fit the
last tile.  This script compares that scheme with a half-tile pad (the scene
border falls in the centre of the first/last tile), with and without TTA, and
with smaller tiles.

Usage
-----
    python scripts/inference_ablation.py --scenes 1992-07-15_002_024
    python scripts/inference_ablation.py --scenes 1992-07-15_002_024 1985-07-05_041_192
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

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
from qa4eo2_labels import _hann, _tiles, valid_mask  # noqa: E402

from satharmony.models import load_model  # noqa: E402

DEFAULT_DATA = Path("/data/databases/CLOUD_MSS")
DEFAULT_OUT = Path("/tmp/inference_ablation")
SCENES = ("1992-07-15_002_024", "1985-07-05_041_192", "1987-05-02_018_182")

VARIANTS = {
    "base_512_128_tta": (512, 128, True, False, "reflect"),
    "half_512_128_tta": (512, 128, True, True, "reflect"),
    "half_512_128_no_tta": (512, 128, False, True, "reflect"),
    "half_256_64_tta": (256, 64, True, True, "reflect"),
    "half_128_32_tta": (128, 32, True, True, "reflect"),
    "half_512_128_tta_edge": (512, 128, True, True, "edge"),
}


def predict_variant(
    model,
    checkpoint,
    image: np.ndarray,
    chunk: int,
    overlap: int,
    dihedral: bool,
    pad_half: bool,
    pad_mode: str,
    batch_size: int = 8,
    device: str = "cuda:0",
) -> np.ndarray:
    """Sliding-window probabilities with the requested tiling variant."""
    mean = np.asarray(checkpoint.band_mean, dtype=np.float32)
    std = np.asarray(checkpoint.band_std, dtype=np.float32)
    normalized = ((image - mean[:, None, None]) / std[:, None, None]).astype(np.float32)
    height, width = image.shape[-2:]
    step = chunk - overlap

    if pad_half:
        base = chunk // 2
        padded = np.pad(
            normalized,
            ((0, 0), (base, base), (base, base)),
            mode=pad_mode,
        )
        padded_h, padded_w = padded.shape[-2:]
        extra_h = (step - (padded_h - chunk) % step) % step
        extra_w = (step - (padded_w - chunk) % step) % step
        padded = np.pad(
            padded,
            ((0, 0), (0, extra_h), (0, extra_w)),
            mode=pad_mode,
        )
        coords = [
            (row, col)
            for row in range(0, padded.shape[-2] - chunk + 1, step)
            for col in range(0, padded.shape[-1] - chunk + 1, step)
        ]
        crop_top, crop_left = base, base
    else:
        pad_top, pad_bottom, pad_left, pad_right, coords = _tiles(
            height, width, chunk, step
        )
        padded = np.pad(
            normalized,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode=pad_mode,
        )
        crop_top, crop_left = pad_top, pad_left

    number = int(model.config.num_classes)
    total = np.zeros((number, *padded.shape[-2:]), dtype=np.float32)
    weight = np.zeros(padded.shape[-2:], dtype=np.float32)
    window = _hann(chunk)

    def forward(tiles: torch.Tensor) -> torch.Tensor:
        if not dihedral:
            return torch.softmax(model(tiles), dim=1)
        accumulated = None
        for code in range(8):
            transformed = tiles
            if code & 1:
                transformed = transformed.flip(-1)
            if code & 2:
                transformed = transformed.flip(-2)
            if code & 4:
                transformed = transformed.rot90(1, dims=(-2, -1))
            probabilities = torch.softmax(model(transformed), dim=1)
            if code & 4:
                probabilities = probabilities.rot90(-1, dims=(-2, -1))
            if code & 2:
                probabilities = probabilities.flip(-2)
            if code & 1:
                probabilities = probabilities.flip(-1)
            accumulated = probabilities if accumulated is None else accumulated + probabilities
        return accumulated / 8

    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            batch = coords[start : start + batch_size]
            tiles = np.stack(
                [padded[:, row : row + chunk, col : col + chunk] for row, col in batch]
            )
            probabilities = forward(
                torch.from_numpy(tiles).float().to(device)
            ).cpu().numpy()
            for index, (row, col) in enumerate(batch):
                window_slice = np.s_[:, row : row + chunk, col : col + chunk]
                total[window_slice] += probabilities[index] * window
                weight[row : row + chunk, col : col + chunk] += window

    probabilities = total / np.maximum(weight, 1e-8)[None]
    return probabilities[
        :, crop_top : crop_top + height, crop_left : crop_left + width
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_G4_REAL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save", nargs="*", default=[])
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint, device=args.device)
    rows: list[dict] = []
    pooled: dict[str, np.ndarray] = {}
    for scene in args.scenes:
        folder = args.data / "pairs_optimization" / scene
        image, _ = read_mss60(folder / "mss_original.tif")
        with rasterio.open(reference_path(folder)) as src:
            reference = src.read(1)
        valid = valid_mask(image) & (reference != 255)
        for name in args.variants:
            chunk, overlap, dihedral, pad_half, pad_mode = VARIANTS[name]
            started = time.time()
            probabilities = predict_variant(
                model,
                checkpoint,
                image,
                chunk,
                overlap,
                dihedral,
                pad_half,
                pad_mode,
                device=args.device,
            )
            prediction = collapse_to_three(probabilities.argmax(axis=0))
            scored = score_scene(prediction, reference, valid)
            scores = three_class_scores(scored["confusion"])
            row = {
                "scene": scene,
                "variant": name,
                "mean_iou": round(scored["mean_iou"], 6),
                "overall_accuracy": round(scored["overall_accuracy"], 6),
                "kappa": round(scored["kappa"], 6),
                "iou_clear": round(scores["clear"]["iou"], 6),
                "iou_cloud": round(scores["cloud"]["iou"], 6),
                "iou_shadow": round(scores["cloud_shadow"]["iou"], 6),
                "boundary_cloud": round(scored["boundary_f1_cloud"], 6),
                "boundary_shadow": round(scored["boundary_f1_shadow"], 6),
                "seconds": round(time.time() - started, 2),
            }
            rows.append(row)
            pooled.setdefault(name, np.zeros((3, 3), dtype=np.int64))
            pooled[name] += scored["confusion"]
            print(
                f"{scene} {name:24s} mIoU {row['mean_iou']:.4f} "
                f"OA {row['overall_accuracy']:.4f} "
                f"cloud {row['iou_cloud']:.3f} shadow {row['iou_shadow']:.3f} "
                f"{row['seconds']}s"
            )
            if name in args.save:
                mask_dir = args.out / "masks"
                mask_dir.mkdir(exist_ok=True)
                np.save(mask_dir / f"{name}_{scene}.npy", prediction)

    with (args.out / "inference_ablation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, matrix in pooled.items():
        from satharmony.models.metrics import cohen_kappa, mean_iou, overall_accuracy

        print(
            f"POOLED {name:24s} mIoU {mean_iou(matrix):.4f} "
            f"OA {overall_accuracy(matrix):.4f} kappa {cohen_kappa(matrix):.4f}"
        )
    print(f"written: {args.out / 'inference_ablation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
