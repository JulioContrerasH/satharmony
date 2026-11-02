#!/usr/bin/env python
"""Greedy ensemble selection for the G4b models, on the validation split only.

The G4b ablations did not beat the G4 checkpoint one by one: the ceiling of a
single 4-band, 60 m U-Net on 1.7k chips sits around 0.56 validation mIoU.  Their
errors are partly independent, though, so averaging their softmax does help.
This script picks the subset:

1. Compute the validation probabilities of every candidate with dihedral TTA
   and cache them (``val_probs_<name>.npz``).
2. Forward selection: start from the best single model, then add the candidate
   that maximises the validation mean IoU of the average, until no candidate
   improves it by more than ``--min-gain``.
3. Write ``ensemble.json`` with the members, their SHA-256 and the validation
   metrics.  The test split is **never** read here; the chosen subset is scored
   once by ``scripts/evaluate_cloud_baseline.py --ensemble``.

Usage
-----
    python scripts/select_g4b_ensemble.py
    python scripts/select_g4b_ensemble.py --candidates g4 b1_long b7_thin
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.models import (  # noqa: E402
    confusion_matrix,
    file_sha256,
    iou_per_class,
    load_model,
    mean_iou,
    predict_proba_dihedral,
)

DEFAULT_FIXTURE = Path("/data/databases/CLOUD_MSS/calibration/pseudo_mss_fixture")
DEFAULT_CANDIDATES = Path("/data/databases/CLOUD_MSS/calibration/g4b")
DEFAULT_G4 = Path("/data/databases/CLOUD_MSS/calibration/g4/cloud_unet_mss_pseudo.pt")
DEFAULT_OUT = REPO.parent / "assets/md/fuentes/g4b"


def validation_probabilities(
    path: Path,
    images: np.ndarray,
    labels: np.ndarray,
    cache: Path,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Dihedral-TTA probabilities of one checkpoint on the validation split.

    Args:
        path: Checkpoint file.
        images: ``(N, 4, H, W)`` validation reflectance, 0-1.
        labels: ``(N, H, W)`` validation classes (only used for the cache key).
        cache: ``.npy`` file with the probabilities.
        device: Torch device.
        batch_size: Images per pass.

    Returns:
        ``(N, 4, H, W)`` float16 probabilities.
    """
    if cache.exists():
        return np.load(cache)
    model, checkpoint = load_model(path, device=device)
    probabilities = predict_proba_dihedral(
        model, images, checkpoint.band_mean, checkpoint.band_std,
        batch_size=batch_size, device=device,
    ).astype(np.float16)
    np.save(cache, probabilities)
    return probabilities


def metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Mean IoU and per-class IoU of an average probability map."""
    matrix = confusion_matrix(probabilities.argmax(axis=1), labels)
    return {
        "mean_iou": mean_iou(matrix),
        "iou": iou_per_class(matrix),
        "overall_accuracy": float(np.trace(matrix) / max(matrix.sum(), 1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--candidates-dir", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--g4", type=Path, default=DEFAULT_G4)
    parser.add_argument("--candidates", nargs="+", default=None,
                        help="checkpoint stems to consider (default: every .pt)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-gain", type=float, default=5e-4)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="where the validation probabilities are cached "
             "(default: <candidates-dir>/val_probs, outside the vault)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    manifest = list(csv.DictReader((args.fixture / "manifest.csv").open()))
    labels = np.load(args.fixture / "labels.npy", mmap_mode="r")
    val_index = [index for index, row in enumerate(manifest) if row["split"] == "validation"]
    images = np.load(args.fixture / "images.npy", mmap_mode="r")
    val_images = np.asarray(images[val_index], dtype=np.float32) / 10000.0
    val_labels = np.asarray(labels[val_index]).astype(np.uint8)

    available = {}
    if args.g4.exists():
        available["g4"] = args.g4
    for path in sorted(args.candidates_dir.glob("*.pt")):
        available[path.stem] = path
    names = args.candidates or list(available)
    missing = [name for name in names if name not in available]
    if missing:
        raise FileNotFoundError(f"unknown checkpoints: {missing}")
    print(f"{len(names)} candidates: {names}")

    cache_dir = args.cache_dir or (args.candidates_dir / "val_probs")
    cache_dir.mkdir(parents=True, exist_ok=True)
    probabilities = {}
    for name in names:
        probabilities[name] = validation_probabilities(
            available[name], val_images, val_labels,
            cache_dir / f"{name}.npy", device, args.batch_size,
        ).astype(np.float32)

    singles = {name: metrics(probabilities[name], val_labels) for name in names}
    for name, result in sorted(singles.items(), key=lambda item: -item[1]["mean_iou"]):
        print(
            f"  {name:14s} val mIoU {result['mean_iou']:.4f} "
            f"(thin {result['iou']['thin_cloud']:.3f})"
        )

    order = sorted(names, key=lambda name: -singles[name]["mean_iou"])
    selected = [order[0]]
    current = probabilities[order[0]].copy()
    best = singles[order[0]]["mean_iou"]
    steps = [{"added": order[0], "mean_iou": best,
              "thin_cloud": singles[order[0]]["iou"]["thin_cloud"]}]
    remaining = list(order[1:])
    while remaining:
        trials = {}
        for name in remaining:
            candidate = (current * len(selected) + probabilities[name]) / (len(selected) + 1)
            trials[name] = metrics(candidate, val_labels)["mean_iou"]
        winner = max(trials, key=trials.get)
        if trials[winner] < best + args.min_gain:
            break
        selected.append(winner)
        current = (current * (len(selected) - 1) + probabilities[winner]) / len(selected)
        best = trials[winner]
        steps.append({
            "added": winner,
            "mean_iou": best,
            "thin_cloud": metrics(current, val_labels)["iou"]["thin_cloud"],
        })
        remaining.remove(winner)
        print(f"  + {winner:14s} → val mIoU {best:.4f}")

    final = metrics(current, val_labels)
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_split": "validation (113 chips), dihedral TTA, scales 1.0",
        "rule": "greedy forward selection of the softmax average",
        "min_gain": args.min_gain,
        "members": [
            {
                "name": name,
                "path": str(available[name]),
                "sha256": file_sha256(available[name]),
            }
            for name in selected
        ],
        "single_val": {name: {"mean_iou": result["mean_iou"],
                              "thin_cloud": result["iou"]["thin_cloud"]}
                       for name, result in singles.items()},
        "steps": steps,
        "val": final,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ensemble.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"selected {selected} → val mIoU {final['mean_iou']:.4f} "
        f"(thin {final['iou']['thin_cloud']:.3f})"
    )
    print(f"wrote {args.out / 'ensemble.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
