#!/usr/bin/env python
"""Test whether the G4 prediction depends on the tile grid.

The same scene is inferred with four different half-pad offsets.  If the
prediction is stable, the masks agree; if the rectangular blocks move with the
grid, the inference is the problem.  This is the cheap test proposed by Fable
before changing the blend.
"""

from __future__ import annotations

import argparse
import csv
import sys
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
)
from qa4eo2_labels import predict_proba_unet, valid_mask  # noqa: E402

from satharmony.models import load_model  # noqa: E402
from satharmony.models.metrics import confusion_matrix, mean_iou, overall_accuracy  # noqa: E402

DEFAULT_DATA = Path("/data/databases/CLOUD_MSS")
DEFAULT_OUT = Path("/tmp/inference_grid_shift")
SCENES = ("1992-07-15_002_024", "1985-07-05_041_192", "1987-05-02_018_182")
OFFSETS = (0, 96, 192, 288)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--offsets", nargs="+", type=int, default=list(OFFSETS))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_G4_REAL)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint, device=args.device)
    rows: list[dict] = []
    for scene in args.scenes:
        folder = args.data / "pairs_optimization" / scene
        image, _ = read_mss60(folder / "mss_original.tif")
        with rasterio.open(reference_path(folder)) as src:
            reference = src.read(1)
        valid = valid_mask(image) & (reference != 255)
        masks = {}
        for offset in args.offsets:
            probabilities = predict_proba_unet(
                model,
                checkpoint,
                image,
                args.device,
                dihedral=True,
                pad_half=True,
                pad_offset=offset,
            )
            mask = collapse_to_three(probabilities.argmax(axis=0))
            masks[offset] = mask
            np.save(args.out / f"{scene}_offset{offset}.npy", mask)
        for i, first in enumerate(args.offsets):
            for second in args.offsets[i + 1 :]:
                matrix = confusion_matrix(masks[first], masks[second], num_classes=3, ignore_mask=~valid)
                rows.append(
                    {
                        "scene": scene,
                        "first": first,
                        "second": second,
                        "mean_iou": round(mean_iou(matrix), 6),
                        "overall_accuracy": round(overall_accuracy(matrix), 6),
                    }
                )
                print(
                    f"{scene} {first:3d} vs {second:3d} | "
                    f"IoU {rows[-1]['mean_iou']:.4f} | OA {rows[-1]['overall_accuracy']:.4f}"
                )
    with (args.out / "grid_shift.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written: {args.out / 'grid_shift.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
