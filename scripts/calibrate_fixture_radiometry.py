#!/usr/bin/env python
"""Fit a fixed per-band radiometry for the pseudo-MSS fixture against real MSS.

Why
---
The G4 / G4b U-Nets are trained on the pseudo-MSS fixture and evaluated on the
real MSS of the v7 pairs.  The fixture is 1.3-1.5x darker and less variable
than the real reflectance, so after the checkpoint's fixed normalisation every
real pixel sits above the training distribution.  The G4b report measured it
but did not correct it.

What this script does
---------------------
1. Reads the real ``mss_original.tif`` (uint16 TOA x 10000) and the 60 m label
   of a set of v7 scenes.  By default the split is ``train,val``: the
   test scenes are **not** used to fit anything, so the proxy benchmark stays
   clean.  ``--check-split`` additionally fits on another split (e.g. ``test``)
   only to report how stable the coefficients are; it never applies it.
2. Computes per-band mean and standard deviation over the valid pixels
   (``any(band) > 0``) of the emulated fixture and of the real scenes.
3. Fits ONE fixed affine per band by moment matching::

       scale  = std_real / std_fixture
       offset = mean_real - scale * mean_fixture

   No per-image, per-chip or per-scene statistic is used.
4. With ``--out-fixture`` writes a calibrated copy of the fixture: the same
   labels and manifest, ``images.npy`` re-quantised to uint16 after the affine
   and clipped to [0, 1], and a ``fixture.json`` that records the transform and
   the hash of both the raw and the calibrated arrays.

Outputs
-------
``radiometry_calibration.json``  coefficients, moments, split and hashes.
``calibration.md``               the per-band table for the note.
``<out-fixture>/``               calibrated fixture (optional).

Usage
-----
    python scripts/calibrate_fixture_radiometry.py --check-split test
    python scripts/calibrate_fixture_radiometry.py \
        --out-fixture /data/databases/CLOUD_MSS/calibration/pseudo_mss_fixture_cal
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.calibration import DEFAULT_BANDS, SCALE_FACTOR, AffineRadiometry  # noqa: E402
from satharmony.datasets.cloudsen12 import array_sha256  # noqa: E402
from satharmony.models import file_sha256  # noqa: E402

DEFAULT_MANIFEST = REPO.parent / "assets/md/manifest-finetune-v7.csv"
DEFAULT_PAIRS = Path("/data/databases/CLOUD_MSS/pairs_optimization")
DEFAULT_FIXTURE = Path("/data/databases/CLOUD_MSS/calibration/pseudo_mss_fixture")
DEFAULT_OUT = REPO.parent / "assets/md/fuentes/calibracion-fixture"

#: Class codes of ``label_refined_mss.tif`` (0 clear, 1 cloud, 2 shadow).
REAL_CLASSES = {"clear": 0, "cloud": 1, "shadow": 2}

#: Class codes of the fixture labels (0 clear, 1 thick, 2 thin, 3 shadow).
FIXTURE_CLASSES = {"clear": 0, "cloud": (1, 2), "shadow": 3, "all": None}


def read_rows(manifest: Path, splits: list[str]) -> list[dict[str, str]]:
    """Manifest rows of the requested splits, in manifest order.

    Args:
        manifest: v7 manifest CSV.
        splits: Split names (``train``, ``validation``, ``test``).

    Returns:
        Selected rows.

    Raises:
        ValueError: If a split is unknown or empty.
    """
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    known = {row["split"] for row in rows}
    missing = sorted(set(splits) - known)
    if missing:
        raise ValueError(f"{manifest} has no split(s) {missing}; known: {sorted(known)}")
    selected = [row for row in rows if row["split"] in set(splits)]
    if not selected:
        raise ValueError(f"{manifest} has no rows for splits {splits}")
    return selected


def _scene_moments(task: tuple[str, str, int]) -> tuple[str, dict | None]:
    """Accumulate per-band moments of one real scene, per class and globally.

    The scene is decimated on read (``step``) so 238 scenes fit in a few
    seconds.  Decimation is deterministic and identical for the label and the
    image; only the summary statistics are kept.

    Args:
        task: ``(scene, pairs_root, step)``.

    Returns:
        ``(scene, {"sum", "sq", "count"} or None)`` with arrays shaped
        ``(4 classes, 4 bands)``: clear, cloud, shadow and all.
    """
    scene, pairs_root, step = task
    directory = Path(pairs_root) / scene
    try:
        with rasterio.open(directory / "label_refined_mss.tif") as source:
            label = source.read(1)[::step, ::step]
        with rasterio.open(directory / "mss_original.tif") as source:
            # Full read, then decimate both arrays with the same slicing: the
            # ``out_shape`` reader resamples on its own grid, which can shift
            # the MSS relative to the label and mix the per-class statistics.
            mss = (source.read().astype(np.float32) / 10000.0)[:, ::step, ::step]
    except (FileNotFoundError, OSError, rasterio.errors.RasterioIOError):
        return scene, None

    height = min(mss.shape[1], label.shape[0])
    width = min(mss.shape[2], label.shape[1])
    mss = mss[:, :height, :width]
    label = label[:height, :width]
    valid = np.any(mss > 0, axis=0)

    sums = np.zeros((4, mss.shape[0]))
    squares = np.zeros((4, mss.shape[0]))
    counts = np.zeros(4, dtype=np.int64)
    masks = [
        valid & (label == REAL_CLASSES["clear"]),
        valid & (label == REAL_CLASSES["cloud"]),
        valid & (label == REAL_CLASSES["shadow"]),
        valid,
    ]
    for index, mask in enumerate(masks):
        if not mask.any():
            continue
        block = mss[:, mask]
        sums[index] = block.sum(axis=1)
        squares[index] = np.square(block, dtype=np.float64).sum(axis=1)
        counts[index] = mask.sum()
    return scene, {"sum": sums, "sq": squares, "count": counts}


def real_moments(
    rows: list[dict[str, str]], pairs_root: Path, step: int, workers: int
) -> dict:
    """Pool the per-scene moments of a split into moments per class.

    Args:
        rows: Manifest rows of the split.
        pairs_root: Directory with ``<scene>/mss_original.tif``.
        step: Decimation factor on read.
        workers: Worker processes.

    Returns:
        ``{"<class>": {"mean": (4,), "std": (4,), "n": int}, ...}`` for the
        four classes plus ``"all"``, and the list of scenes actually read.
    """
    total = {
        "sum": np.zeros((4, 4)),
        "sq": np.zeros((4, 4)),
        "count": np.zeros(4, dtype=np.int64),
    }
    used: list[str] = []
    tasks = [(row["id_name"], str(pairs_root), step) for row in rows]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for scene, moments in pool.map(_scene_moments, tasks):
            if moments is None:
                print(f"  skipped {scene}: unreadable", flush=True)
                continue
            total["sum"] += moments["sum"]
            total["sq"] += moments["sq"]
            total["count"] += moments["count"]
            used.append(scene)
    names = ("clear", "cloud", "shadow", "all")
    out: dict[str, dict] = {}
    for index, name in enumerate(names):
        count = int(total["count"][index])
        if count == 0:
            raise ValueError(f"split has no valid {name} pixels")
        mean = total["sum"][index] / count
        variance = np.maximum(total["sq"][index] / count - mean**2, 0.0)
        out[name] = {
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "n": count,
        }
    out["_scenes"] = used  # type: ignore[assignment]
    return out


def fixture_moments(fixture: Path) -> dict:
    """Per-band moments of the fixture, per class and globally.

    Args:
        fixture: Directory with ``images.npy`` and ``labels.npy``.

    Returns:
        Same structure as :func:`real_moments`.
    """
    images = np.load(fixture / "images.npy").astype(np.float32) / SCALE_FACTOR
    labels = np.load(fixture / "labels.npy")
    out: dict[str, dict] = {}
    for name, codes in FIXTURE_CLASSES.items():
        mask = np.ones_like(labels, dtype=bool) if codes is None else np.isin(labels, codes)
        if not mask.any():
            raise ValueError(f"fixture has no {name} pixels")
        values = np.stack([images[:, band][mask] for band in range(images.shape[1])])
        out[name] = {
            "mean": values.mean(axis=1).tolist(),
            "std": values.std(axis=1).tolist(),
            "n": int(mask.sum()),
        }
    out["_chips"] = int(images.shape[0])  # type: ignore[assignment]
    return out


def build_calibrated_fixture(
    source: Path, target: Path, radiometry: AffineRadiometry, chunk: int = 250
) -> dict:
    """Write a calibrated copy of the fixture.

    The labels, the manifest and the label distribution are copied byte for
    byte: only the radiometry changes, so the split is identical.

    Args:
        source: Raw fixture directory.
        target: Output directory.
        radiometry: Fixed transform to apply.
        chunk: Chips per pass.

    Returns:
        Summary with the array hashes and the clipped fraction per band.
    """
    target.mkdir(parents=True, exist_ok=True)
    images = np.load(source / "images.npy", mmap_mode="r")
    out = np.empty_like(np.asarray(images))
    clipped_high = np.zeros(images.shape[1], dtype=np.int64)
    clipped_low = np.zeros(images.shape[1], dtype=np.int64)
    for start in range(0, images.shape[0], chunk):
        block = np.asarray(images[start : start + chunk])
        reflectance = block.astype(np.float32) / SCALE_FACTOR
        scale, offset = radiometry.as_arrays()
        raw = reflectance * scale.reshape(1, -1, 1, 1) + offset.reshape(1, -1, 1, 1)
        clipped_high += np.sum(raw > 1.0, axis=(0, 2, 3))
        clipped_low += np.sum(raw < 0.0, axis=(0, 2, 3))
        out[start : start + chunk] = radiometry.apply_uint16(block)
        print(f"  calibrated {min(start + chunk, images.shape[0])}/{images.shape[0]}", end="\r")

    print()
    np.save(target / "images.npy", out)
    shutil.copyfile(source / "labels.npy", target / "labels.npy")
    shutil.copyfile(source / "manifest.csv", target / "manifest.csv")
    for optional in ("label_distribution.json",):
        if (source / optional).exists():
            shutil.copyfile(source / optional, target / optional)

    metadata = json.loads((source / "fixture.json").read_text())
    calibrated_labels = np.load(target / "labels.npy")
    metadata["source_fixture"] = str(source)
    metadata["source_images_sha256"] = metadata.get("images_sha256")
    metadata["images_sha256"] = array_sha256(out, calibrated_labels)
    metadata["calibration"] = radiometry.to_dict()
    metadata["calibration"]["clipped_fraction"] = (
        (clipped_high + clipped_low) / (images.shape[0] * images.shape[2] * images.shape[3])
    ).tolist()
    (target / "fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "path": str(target),
        "images_sha256": metadata["images_sha256"],
        "source_images_sha256": metadata["source_images_sha256"],
        "clipped_fraction": metadata["calibration"]["clipped_fraction"],
        "labels_sha256": file_sha256(target / "labels.npy"),
    }


def write_markdown(path: Path, payload: dict) -> None:
    """Write the per-band table used in the note."""
    bands = payload["bands"]
    fixture = payload["fixture_moments"]
    real = payload["real"]["all"]
    clear = payload["real"]["clear"]
    cloud = payload["real"]["cloud"]
    shadow = payload["real"]["shadow"]
    lines = [
        "# Calibración radiométrica fija del pseudo-MSS",
        "",
        f"Escala y offset por banda, ajustados una sola vez con los momentos de "
        f"todos los píxeles válidos de `{payload['split']}` "
        f"({payload['n_scenes']} escenas). Nada se calcula por imagen.",
        "",
        "| Banda | Fixture media | Fixture std | Real media | Real std | Escala | Offset |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, band in enumerate(bands):
        lines.append(
            f"| {band} | {fixture['all']['mean'][index]:.4f} | "
            f"{fixture['all']['std'][index]:.4f} | {real['mean'][index]:.4f} | "
            f"{real['std'][index]:.4f} | {payload['scale'][index]:.4f} | "
            f"{payload['offset'][index]:+.4f} |"
        )
    lines += [
        "",
        "Qué queda después de la transformación (medias por clase):",
        "",
        "| Clase | Fixture calibrado | Real |",
        "|---|---:|---:|",
    ]
    calibrated = payload["calibrated_class_means"]
    for name, real_class in (("clear", clear), ("cloud", cloud), ("shadow", shadow)):
        for index, band in enumerate(bands):
            lines.append(
                f"| {name} {band} | {calibrated[name][index]:.3f} | "
                f"{real_class['mean'][index]:.3f} |"
            )
    lines += [
        "",
        "La transformación es afín por banda y no puede cambiar la separación "
        "entre clases: la sombra del fixture queda más brillante que la real.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pairs-root", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-fixture", type=Path, default=None)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--check-split", default="test")
    parser.add_argument("--step", type=int, default=6, help="read decimation")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    start = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    rows = read_rows(args.manifest, splits)
    print(f"{len(rows)} scenes from splits {splits}, step {args.step}")

    print("fixture moments")
    fixture = fixture_moments(args.fixture)
    print("real moments")
    real = real_moments(rows, args.pairs_root, args.step, args.workers)
    bands = list(DEFAULT_BANDS)

    radiometry = AffineRadiometry.from_moments(
        np.asarray(fixture["all"]["mean"]),
        np.asarray(fixture["all"]["std"]),
        np.asarray(real["all"]["mean"]),
        np.asarray(real["all"]["std"]),
        bands=tuple(bands),
        method="moment matching over all valid pixels (mean and std)",
        source="emulated pseudo-MSS fixture",
        target="real v7 Landsat MSS (mss_original.tif / 10000)",
        split="+".join(splits),
        n_pixels=int(fixture["all"]["n"] + real["all"]["n"]),
    )
    scale, offset = radiometry.as_arrays()
    print(f"scale {np.round(scale, 4).tolist()}")
    print(f"offset {np.round(offset, 4).tolist()}")

    calibrated_class_means = {
        name: (np.asarray(fixture[name]["mean"]) * scale + offset).tolist()
        for name in ("clear", "cloud", "shadow", "all")
    }
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.perf_counter() - start, 1),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "pairs_root": str(args.pairs_root),
        "fixture": str(args.fixture),
        "fixture_sha256": file_sha256(args.fixture / "images.npy"),
        "fixture_moments": fixture,
        "split": "+".join(splits),
        "n_scenes": len(real["_scenes"]),
        "scenes": real["_scenes"],
        "step": args.step,
        "bands": bands,
        "scale": scale.tolist(),
        "offset": offset.tolist(),
        "radiometry": radiometry.to_dict(),
        "real": {name: real[name] for name in ("clear", "cloud", "shadow", "all")},
        "real_by_split": {},
        "calibrated_class_means": calibrated_class_means,
        "check_split": None,
    }

    if args.check_split:
        check_rows = read_rows(args.manifest, [args.check_split])
        print(f"stability check on {len(check_rows)} {args.check_split} scenes")
        check = real_moments(check_rows, args.pairs_root, args.step, min(args.workers, 8))
        check_transform = AffineRadiometry.from_moments(
            np.asarray(fixture["all"]["mean"]),
            np.asarray(fixture["all"]["std"]),
            np.asarray(check["all"]["mean"]),
            np.asarray(check["all"]["std"]),
        )
        train_mean = np.asarray(real["all"]["mean"])
        train_std = np.asarray(real["all"]["std"])
        payload["check_split"] = {
            "split": args.check_split,
            "scenes": check["_scenes"],
            "real": {name: check[name] for name in ("clear", "cloud", "shadow", "all")},
            "scale": list(check_transform.scale),
            "offset": list(check_transform.offset),
            "scale_delta": (np.asarray(check_transform.scale) - scale).tolist(),
            "shift_sigma": (
                (np.asarray(check["all"]["mean"]) - train_mean) / train_std
            ).tolist(),
            "std_ratio": (np.asarray(check["all"]["std"]) / train_std).tolist(),
        }
        print(
            "  check split sits at "
            f"{np.round(payload['check_split']['shift_sigma'], 3).tolist()} sigma "
            "of the calibrated training domain"
        )
        print(
            f"  check scale {np.round(check_transform.scale, 4).tolist()} "
            f"(delta {np.round(np.asarray(check_transform.scale) - scale, 4).tolist()})"
        )

    if args.out_fixture is not None:
        print(f"writing calibrated fixture to {args.out_fixture}")
        payload["calibrated_fixture"] = build_calibrated_fixture(
            args.fixture, args.out_fixture, radiometry
        )
        radiometry.save(args.out_fixture / "radiometry_calibration.json")

    radiometry.save(args.out / "radiometry_calibration.json")
    write_markdown(args.out / "calibration.md", payload)
    (args.out / "calibration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"done in {time.perf_counter() - start:.1f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
