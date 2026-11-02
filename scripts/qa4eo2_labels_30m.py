#!/usr/bin/env python
"""QA4EO-2 labels on the 30 m aligned MSS grid, four classes.

What this script produces
-------------------------
``label_qa4eo_2.tif`` for every scene of the v7 test split, on the grid of
``pairs_optimization/<id>/mss_aligned.tif``: 4 MSS bands, 30 m, same CRS,
transform and shape as the input.  Classes ``0 clear``, ``1 thin cloud``,
``2 thick cloud``, ``3 cloud shadow``; ``255`` no data, written as the raster
``nodata`` value.

Why 30 m and what changes
-------------------------
The 60 m product of ``scripts/qa4eo2_labels.py`` sits on ``mss_original.tif``.
Everything that works on the TM grid (``tm_aligned``, ``label.tif``, the v7
labels) needs a label on ``mss_aligned.tif``.

``mss_aligned.tif`` is not a new observation: it is ``mss_original.tif``
duplicated 2 x 2 nearest with a one-column (15 m) shift.  :func:`align_shift`
measures that shift exactly for every scene instead of assuming it.  The model
therefore sees the same radiometry, but its receptive field covers half the
ground distance it was trained on, so the 30 m output is **not** the 2 x 2
upsample of the 60 m label.  The manifest records the agreement between the two
so the difference is measured, not assumed.

Which checkpoint
----------------
``train/mlstac_export/unet.ckpt`` (sha256 ``920ac779...``, epoch 55) is the
4-class export of January: 0 clear, 1 thin, 2 thick, 3 shadow.  It is the model
that generated ``first_label``.  ``QA4EO-2/unet.ckpt`` (``bb8d3fa1...``) is the
3-class fine-tune used by the 60 m product; it cannot tell thin from thick.

Input convention
----------------
Raw TOA 0-1 (DN / 10000), no z-score, ``merge_clouds=False`` so the four
classes survive, no physical post-rules: the same convention as
``scripts/qa4eo2_labels.py`` and ``assets/codes/mss/QA4EO-2/load.py``.

Sliding window
--------------
512 x 512 tiles, Hann window of power 3, 50 % overlap, cloud probabilities
blended with the per-tile max (``cloud_max_alpha=0.6``), the same
``predict_large`` scheme as the 60 m product, so the two differ only in grid
and checkpoint.

The 60 m product is preserved
-----------------------------
Before a 60 m ``label_qa4eo_2.tif`` is replaced it is renamed to
``label_qa4eo_2_60m.tif`` in the same folder, and its sha256 is checked against
the one recorded by the 60 m run.  Nothing is deleted.

Usage
-----
    python scripts/qa4eo2_labels_30m.py --limit 1 --overwrite   # smoke test
    python scripts/qa4eo2_labels_30m.py                         # 42 scenes
    python scripts/qa4eo2_labels_30m.py --verify-only           # re-check only
"""

from __future__ import annotations

import argparse
import csv
import json
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

from qa4eo2_labels import (  # noqa: E402
    IGNORE,
    load_qa4eo2,
    predict_proba_qa4eo2,
    read_test_rows,
    sha256_file,
    valid_mask,
    write_label,
)

DEFAULT_MANIFEST = REPO.parent / "assets/md/manifest-finetune-v7.csv"
DEFAULT_DATA = Path("/data/databases/CLOUD_MSS")
DEFAULT_QA4EO2_JAN = Path("/data/users/julio/isp_clouds/mss/train/mlstac_export/unet.ckpt")
DEFAULT_OUT = REPO.parent / "assets/md/fuentes/qa4eo2-labels"

#: sha256 of the 4-class January export; the run refuses another checkpoint.
EXPECTED_SHA256 = "920ac77982e059ba6300f757d5588284cb983f4a5430d05b8103f95101e3470a"

#: Class names of the 4-class export, in output-channel order.
CLASS_NAMES = ("clear", "thin_cloud", "thick_cloud", "cloud_shadow")

#: Three-class convention used by the 60 m product and by every metric.
CLASS_NAMES_3 = ("clear", "cloud", "cloud_shadow")

#: Name of the 30 m product and of the preserved 60 m product.
LABEL_30M = "label_qa4eo_2.tif"
LABEL_60M = "label_qa4eo_2_60m.tif"


# ── IO ──────────────────────────────────────────────────────────────────
def read_mss30(path: Path) -> tuple[np.ndarray, dict]:
    """Read a 30 m aligned MSS stack as physical TOA 0-1.

    Args:
        path: ``mss_aligned.tif`` (4 bands, uint16 DN, 30 m).

    Returns:
        ``((4, H, W) float32, profile)``; the profile is the rasterio profile
        of the source, so the label inherits the exact grid.
    """
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32) / 10000.0
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=IGNORE)
    return image, profile


def is_60m(path: Path) -> bool:
    """True if the raster is a single-band 60 m label."""
    with rasterio.open(path) as src:
        return src.count == 1 and abs(float(src.transform.a) - 60.0) < 1e-6


def backup_60m(folder: Path, expected_hash: str | None = None) -> Path | None:
    """Preserve the 60 m label before the 30 m one takes its name.

    Args:
        folder: Scene folder in ``pairs_optimization``.
        expected_hash: sha256 recorded by the 60 m run, if known.

    Returns:
        Path of ``label_qa4eo_2_60m.tif``, or ``None`` if there is no 60 m
        label to preserve.

    Raises:
        RuntimeError: If a 60 m label is present and the backup already exists
            with different content, so the caller can look instead of losing it.
    """
    current = folder / LABEL_30M
    backup = folder / LABEL_60M
    if backup.exists():
        if current.exists() and is_60m(current):
            if sha256_file(current) == sha256_file(backup):
                current.unlink()  # exact duplicate of the backup
            else:
                raise RuntimeError(f"{current} is 60 m and differs from {backup}")
        return backup
    if not current.exists() or not is_60m(current):
        return None
    current.rename(backup)
    digest = sha256_file(backup)
    if expected_hash and digest != expected_hash:
        raise RuntimeError(f"{backup} sha256 {digest} != manifest {expected_hash}")
    return backup


# ── inference ───────────────────────────────────────────────────────────
def four_class_argmax(probabilities: np.ndarray) -> np.ndarray:
    """Argmax of a 4-class probability volume, no merging.

    Args:
        probabilities: ``(4, H, W)`` probabilities.

    Returns:
        ``(H, W)`` uint8 with 0 clear, 1 thin, 2 thick, 3 shadow.
    """
    if probabilities.shape[0] != 4:
        raise ValueError(f"expected 4 classes, got {probabilities.shape[0]}")
    return probabilities.argmax(axis=0).astype(np.uint8)


def qa4eo2_four_classes(
    model: torch.nn.Module,
    image: np.ndarray,
    device: str,
    **kwargs,
) -> np.ndarray:
    """Four classes of one scene, no-data pixels set to ``IGNORE``.

    Args:
        model: 4-class QA4EO-2 model.
        image: ``(4, H, W)`` reflectance, 0-1.
        device: Torch device.
        **kwargs: Passed to :func:`predict_proba_qa4eo2`.

    Returns:
        ``(H, W)`` uint8 classes with 255 where the MSS has no data.
    """
    probabilities = predict_proba_qa4eo2(model, image, device, **kwargs)
    classes = four_class_argmax(probabilities)
    classes[~valid_mask(image)] = IGNORE
    return classes


# ── grid and comparison ─────────────────────────────────────────────────
def align_shift(
    aligned: np.ndarray,
    original: np.ndarray,
    step: int = 16,
    minimum: float = 0.9999,
    minimum_samples: int = 1000,
    hint: tuple[int, int] | None = None,
) -> tuple[float, int, int]:
    """Measure how ``aligned`` duplicates ``original`` 2 x 2 nearest.

    Finds ``(dy, dx)`` such that ``aligned[i, j] == original[(i - dy) // 2,
    (j - dx) // 2]`` for every valid pixel.  The offset is not the same in
    every scene: the audit found ``(0, 0)``, ``(0, 1)`` and ``(1, 1)``, so it is
    measured here instead of assumed.  Checked on a regular subsample.

    Args:
        aligned: ``(C, H, W)`` 30 m stack.
        original: ``(C, h, w)`` 60 m stack.
        step: Subsampling stride in pixels of the 30 m grid.
        minimum: Fraction of exactly equal samples required.
        minimum_samples: Samples that must be valid before a shift is scored.
        hint: ``(dy, dx)`` from the geotransforms.  The cropping of
            ``mss_aligned`` to the TM grid can move the duplicate by tens of
            pixels, so the offsets are searched around this hint instead of
            around zero.

    Returns:
        ``(score, dy, dx)``.

    Raises:
        RuntimeError: If no shift reproduces the original within ``minimum``.
    """
    height, width = aligned.shape[-2:]
    src_h, src_w = original.shape[-2:]
    # Both parities, so a half-pixel row shift cannot hide behind the stride.
    rows = np.unique(np.concatenate([np.arange(0, height, step), np.arange(1, height, step)]))
    cols = np.unique(np.concatenate([np.arange(0, width, step), np.arange(1, width, step)]))
    best: tuple[float, int, int] | None = None
    second = 0.0
    row_offsets = range(hint[0] - 1, hint[0] + 2) if hint else range(-2, 3)
    col_offsets = range(hint[1] - 1, hint[1] + 2) if hint else range(-2, 3)
    for dy in row_offsets:
        raw_r = rows - dy
        keep_r = (raw_r >= 0) & (raw_r // 2 < src_h)
        for dx in col_offsets:
            raw_c = cols - dx
            keep_c = (raw_c >= 0) & (raw_c // 2 < src_w)
            if keep_r.sum() < 8 or keep_c.sum() < 8:
                continue
            sample = aligned[:, rows[keep_r][:, None], cols[keep_c][None, :]]
            source = original[
                :,
                (raw_r[keep_r] // 2)[:, None],
                (raw_c[keep_c] // 2)[None, :],
            ]
            valid = np.all(source > 0, axis=0) & np.all(sample > 0, axis=0)
            if valid.sum() < minimum_samples:
                continue
            equal = float(np.all(sample[:, valid] == source[:, valid], axis=0).mean())
            if equal >= 1.0 - 1e-12:
                return 1.0, dy, dx
            if best is None or equal > best[0]:
                second = best[0] if best else 0.0
                best = (equal, dy, dx)
            else:
                second = max(second, equal)
    if best is None or best[0] < minimum:
        raise RuntimeError(f"no 2x nearest shift matches the original (best {best})")
    if best[0] - second < 0.1:
        raise RuntimeError(f"ambiguous 2x nearest shift: {best} vs {second:.6f}")
    return best


def upsample_reference(
    label: np.ndarray, dy: int, dx: int, shape: tuple[int, int]
) -> np.ndarray:
    """Nearest 2 x 2 upsample of a 60 m label with the measured shift.

    Args:
        label: ``(h, w)`` 60 m label.
        dy: Row shift from :func:`align_shift`.
        dx: Column shift from :func:`align_shift`.
        shape: Target ``(H, W)`` of the 30 m grid.

    Returns:
        ``(H, W)`` label, ``IGNORE`` outside the source footprint.
    """
    height, width = shape
    raw_r = np.arange(height) - dy
    raw_c = np.arange(width) - dx
    ok_r = (raw_r >= 0) & (raw_r // 2 < label.shape[0])
    ok_c = (raw_c >= 0) & (raw_c // 2 < label.shape[1])
    rows = np.clip(raw_r // 2, 0, label.shape[0] - 1)
    cols = np.clip(raw_c // 2, 0, label.shape[1] - 1)
    out = label[rows[:, None], cols[None, :]].copy()
    out[~ok_r, :] = IGNORE
    out[:, ~ok_c] = IGNORE
    return out


def merge_thin_thick(label: np.ndarray) -> np.ndarray:
    """Collapse thin and thick cloud into the 3-class convention.

    Args:
        label: ``(H, W)`` with 0/1/2/3/255.

    Returns:
        ``(H, W)`` with 0 clear, 1 cloud, 2 shadow, 255 no data.
    """
    out = label.copy()
    out[label == 2] = 1
    out[label == 3] = 2
    return out


def confusion_3x3(reference: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Confusion matrix of two 3-class labels, rows reference, columns prediction."""
    index = reference[valid].astype(np.int64) * 3 + prediction[valid].astype(np.int64)
    return np.bincount(index, minlength=9).reshape(3, 3)


def iou_from_confusion(matrix: np.ndarray) -> list[float]:
    """Per-class IoU of a confusion matrix, NaN when the class is absent."""
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - intersection
    return [float(t / u) if u else float("nan") for t, u in zip(intersection, union, strict=True)]


def duplicate_purity(label: np.ndarray, dy: int = 0, dx: int = 0) -> float:
    """Fraction of 2 x 2 blocks that are uniform, at the duplication phase.

    ``mss_aligned`` is a 2 x 2 nearest duplicate, so a label that has no detail
    at 30 m is made of uniform 2 x 2 blocks.  The blocks are anchored at the
    measured ``(dy, dx)`` phase, and blocks touching no data are skipped.  A
    value of 1.0 means the prediction is exactly an upsample of a 60 m map; the
    lower it goes, the more the model resolves at 30 m.

    Args:
        label: ``(H, W)`` classes with ``IGNORE`` for no data.
        dy: Row phase from :func:`align_shift`.
        dx: Column phase from :func:`align_shift`.

    Returns:
        Fraction in ``[0, 1]``, NaN if there is no complete block.
    """
    sub = label[dy % 2 :, dx % 2 :]
    valid = sub != IGNORE
    height = sub.shape[0] - sub.shape[0] % 2
    width = sub.shape[1] - sub.shape[1] % 2
    if height == 0 or width == 0:
        return float("nan")
    blocks = sub[:height, :width].reshape(height // 2, 2, width // 2, 2)
    complete = valid[:height, :width].reshape(height // 2, 2, width // 2, 2).all(axis=(1, 3))
    uniform = (blocks == blocks[:, :1, :, :1]).all(axis=(1, 3))
    return float(uniform[complete].mean()) if complete.any() else float("nan")


def verify_label(label_path: Path, mss_path: Path) -> dict:
    """Check a label against its MSS: grid, no data and classes.

    Args:
        label_path: Written ``label_qa4eo_2.tif``.
        mss_path: ``mss_aligned.tif`` it was predicted from.

    The contract for no data is the **union** of the per-band nodata: a pixel
    is no data as soon as one MSS band is 0, because the model needs the four
    of them.  GDAL's dataset mask only flags a pixel when *all* bands are 0, so
    the difference is returned as a count instead of failing the check.

    Returns:
        ``(checks, extra_invalid)``: the booleans and how many pixels the label
        marks as no data beyond GDAL's dataset mask.

    Raises:
        ValueError: If any check fails.
    """
    with rasterio.open(mss_path) as src:
        mss_shape = src.shape
        mss_transform = src.transform
        mss_crs = src.crs
        mss_res = src.res
        image = src.read()
        declared_mask = src.dataset_mask()
    with rasterio.open(label_path) as src:
        label = src.read(1)
        label_shape = src.shape
        label_transform = src.transform
        label_crs = src.crs
        label_res = src.res
        label_nodata = src.nodata
        label_dtype = src.dtypes[0]
        label_count = src.count

    invalid = np.any(image == 0, axis=0)
    declared = declared_mask == 0
    checks = {
        "shape_ok": label_shape == mss_shape,
        "transform_ok": tuple(label_transform)[:6] == tuple(mss_transform)[:6],
        "crs_ok": label_crs == mss_crs,
        "res_ok": all(
            abs(a - b) < 1e-9 for a, b in zip(label_res, mss_res, strict=True)
        ),
        "count_ok": label_count == 1,
        "dtype_ok": label_dtype == "uint8",
        "nodata_value_ok": label_nodata == IGNORE,
        "nodata_union_ok": bool(np.array_equal(label == IGNORE, invalid)),
        "nodata_declared_subset_ok": bool(np.all((label == IGNORE)[declared])),
        "classes_ok": bool(set(np.unique(label)).issubset({0, 1, 2, 3, IGNORE})),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{label_path} failed {failed}")
    return checks, int(np.count_nonzero((label == IGNORE) & ~declared))


# ── manifests ───────────────────────────────────────────────────────────
def load_60m_hashes(out_dir: Path) -> dict[str, str]:
    """sha256 of the 60 m labels from the manifest of the 60 m run."""
    for name in ("labels_manifest_60m.csv", "labels_manifest.csv"):
        path = out_dir / name
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows and rows[0].get("model_classes", "").strip() == "3":
            return {row["id_name"]: row["label_sha256"] for row in rows}
    return {}


def archive_60m_manifest(out_dir: Path) -> None:
    """Rename the 60 m manifest so the canonical one is always the 30 m run."""
    for name in ("labels_manifest.csv", "labels_manifest.json"):
        source = out_dir / name
        target = out_dir / f"labels_manifest_60m{source.suffix}"
        if source.exists() and not target.exists():
            source.rename(target)
            print(f"60 m manifest preserved as {target.name}")


def main(argv: list[str] | None = None) -> int:
    """Write the 30 m labels, verify them and refresh the manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--qa4eo2", type=Path, default=DEFAULT_QA4EO2_JAN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="first N scenes only (smoke test)")
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    rows = read_test_rows(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    old_hashes = load_60m_hashes(args.out)
    archive_60m_manifest(args.out)

    model = None
    model_hash = sha256_file(args.qa4eo2)
    if model_hash != EXPECTED_SHA256:
        raise SystemExit(
            f"{args.qa4eo2} sha256 {model_hash} is not the 4-class export {EXPECTED_SHA256}"
        )
    if not args.verify_only:
        model = load_qa4eo2(args.qa4eo2, args.device)
        if model.num_classes != 4:
            raise SystemExit(f"{args.qa4eo2} has {model.num_classes} classes, expected 4")
        print(f"{len(rows)} scenes | {args.qa4eo2.name} sha256 {model_hash[:12]}… | {args.device}")

    records: list[dict] = []
    started = time.time()
    for number, row in enumerate(rows, 1):
        scene = row["id_name"]
        folder = args.data / "pairs_optimization" / scene
        mss_path = folder / "mss_aligned.tif"
        original_path = folder / "mss_original.tif"
        label_path = folder / LABEL_30M
        scene_started = time.time()

        backup = backup_60m(folder, old_hashes.get(scene))

        with rasterio.open(mss_path) as src:
            mss_shape = src.shape
        needs_run = args.overwrite or not label_path.exists()
        if not needs_run:
            with rasterio.open(label_path) as src:
                if src.shape != mss_shape:
                    needs_run = True
        if needs_run and model is not None:
            image, profile = read_mss30(mss_path)
            classes = qa4eo2_four_classes(
                model,
                image,
                args.device,
                chunk=args.chunk,
                overlap=args.overlap,
                batch_size=args.batch_size,
            )
            write_label(
                label_path,
                classes,
                profile,
                tags={
                    "model": "QA4EO-2 4-class (January export)",
                    "checkpoint": str(args.qa4eo2),
                    "checkpoint_sha256": model_hash,
                    "input": "mss_aligned.tif 30 m, TOA DN/10000, 4 bands, raw (no z-score)",
                    "classes": "0 clear, 1 thin_cloud, 2 thick_cloud, 3 cloud_shadow, 255 nodata",
                    "inference": f"sliding window {args.chunk}/{args.overlap}, hann^3, "
                    f"cloud_max_alpha 0.6, merge_clouds=False",
                    "grid": "mss_aligned.tif, 30 m, same transform and shape",
                    "provenance": "scripts/qa4eo2_labels_30m.py",
                },
            )
            status = "written"
        else:
            status = "verified" if model is None else "skipped"

        checks, nodata_delta = verify_label(label_path, mss_path)
        with rasterio.open(label_path) as src:
            label = src.read(1)
            label_transform = [round(float(value), 6) for value in tuple(src.transform)[:6]]
            label_crs = str(src.crs)
        with rasterio.open(folder / "label.tif") as src:
            label_v7 = src.read(1)
        with rasterio.open(original_path) as src:
            original = src.read().astype(np.float32) / 10000.0

        with rasterio.open(mss_path) as src30:
            res30 = abs(float(src30.res[0]))
            origin30 = (float(src30.transform.c), float(src30.transform.f))
        with rasterio.open(original_path) as src60:
            origin60 = (float(src60.transform.c), float(src60.transform.f))
        hint = (
            round((origin30[1] - origin60[1]) / res30),
            round((origin30[0] - origin60[0]) / res30),
        )
        image30, _ = read_mss30(mss_path)
        try:
            score, dy, dx = align_shift(image30, original, hint=hint)
            align_ok = True
        except RuntimeError as error:
            print(f"    alignment not measurable: {error}")
            score, dy, dx = float("nan"), hint[0], hint[1]
            align_ok = False
        del image30, original
        if backup is not None and align_ok:
            with rasterio.open(backup) as src:
                label60 = src.read(1)
            reference = upsample_reference(label60, dy, dx, label.shape)
            prediction = merge_thin_thick(label)
            valid = (prediction != IGNORE) & (reference != IGNORE)
            matrix = confusion_3x3(reference, prediction, valid)
            ious = iou_from_confusion(matrix)
            agreement = float(np.trace(matrix) / matrix.sum()) if matrix.sum() else float("nan")
        else:
            ious = [float("nan")] * 3
            agreement = float("nan")
        purity = duplicate_purity(label, dy, dx)

        total = int(label.size)
        values, counts = np.unique(label, return_counts=True)
        counts = {int(value): int(count) for value, count in zip(values, counts, strict=True)}
        valid_pixels = total - counts.get(IGNORE, 0)
        records.append(
            {
                "id_name": scene,
                "split": row.get("split", ""),
                "category": row.get("category", ""),
                "label_path": str(label_path.relative_to(args.data)),
                "label_sha256": sha256_file(label_path),
                "label_bytes": label_path.stat().st_size,
                "shape": list(label.shape),
                "crs": label_crs,
                "transform": label_transform,
                "dtype": "uint8",
                "nodata": IGNORE,
                "nodata_dataset_mask_delta": nodata_delta,
                "classes_present": " ".join(str(value) for value in sorted(np.unique(label))),
                "valid_fraction": round(valid_pixels / total, 6),
                "clear_fraction": round(counts.get(0, 0) / total, 6),
                "thin_fraction": round(counts.get(1, 0) / total, 6),
                "thick_fraction": round(counts.get(2, 0) / total, 6),
                "shadow_fraction": round(counts.get(3, 0) / total, 6),
                "cloud_fraction_3c": round((counts.get(1, 0) + counts.get(2, 0)) / total, 6),
                "model_sha256": model_hash or "verify-only",
                "model_classes": 4,
                "grid": "mss_aligned.tif 30 m",
                "mss_aligned_path": str(mss_path.relative_to(args.data)),
                "mss_aligned_sha256": sha256_file(mss_path),
                "mss_aligned_shape": list(mss_shape),
                "label_60m_path": str(backup.relative_to(args.data)) if backup else "",
                "label_60m_sha256": sha256_file(backup) if backup else "",
                "align_dy": dy,
                "align_dx": dx,
                "align_ok": align_ok,
                "align_score": round(score, 9) if align_ok else "",
                "agreement_3c_with_60m": round(agreement, 6),
                "iou_clear_3c": round(ious[0], 6),
                "iou_cloud_3c": round(ious[1], 6),
                "iou_shadow_3c": round(ious[2], 6),
                "duplicate_purity": round(purity, 6),
                "grid_ok": all(checks.values()),
                "label_v7_all_zero": not bool(label_v7.any()),
                "elapsed_s": round(time.time() - scene_started, 1),
            }
        )
        print(
            f"[{number}/{len(rows)}] {scene} {status} | {label.shape} | "
            f"agreement 60m {agreement:.4f} | 2x2 purity {purity:.3f} | "
            f"{records[-1]['elapsed_s']}s"
        )

    fields = list(records[0]) if records else []
    if args.limit:
        print(f"limit mode: {len(records)} scenes processed, manifests not written")
        return 0
    with (args.out / "labels_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    def mean(key: str) -> float:
        """Mean over the scenes where the metric exists, NaN if none."""
        values = [
            float(row[key])
            for row in records
            if isinstance(row[key], (int, float)) and not np.isnan(float(row[key]))
        ]
        return round(float(np.mean(values)), 6) if values else float("nan")

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "checkpoint": str(args.qa4eo2),
        "checkpoint_sha256": model_hash,
        "model_classes": 4,
        "grid": "mss_aligned.tif, 30 m, 4 bands, same transform and shape",
        "input": "TOA DN/10000, raw, no z-score",
        "inference": f"chunk {args.chunk}, overlap {args.overlap}, hann^3, "
        f"cloud_max_alpha 0.6, merge_clouds=False",
        "classes": "0 clear, 1 thin_cloud, 2 thick_cloud, 3 cloud_shadow, 255 nodata",
        "scenes": len(records),
        "elapsed_s": round(time.time() - started, 1),
        "means": {
            "valid_fraction": mean("valid_fraction"),
            "clear_fraction": mean("clear_fraction"),
            "thin_fraction": mean("thin_fraction"),
            "thick_fraction": mean("thick_fraction"),
            "shadow_fraction": mean("shadow_fraction"),
            "cloud_fraction_3c": mean("cloud_fraction_3c"),
            "agreement_3c_with_60m": mean("agreement_3c_with_60m"),
            "duplicate_purity": mean("duplicate_purity"),
        },
        "rows": records,
    }
    (args.out / "labels_manifest.json").write_text(json.dumps(summary, indent=1))

    comparison_fields = [
        "id_name",
        "category",
        "label_60m_sha256",
        "label_sha256",
        "agreement_3c_with_60m",
        "iou_clear_3c",
        "iou_cloud_3c",
        "iou_shadow_3c",
        "duplicate_purity",
        "label_v7_all_zero",
    ]
    with (args.out / "comparison_60m_vs_30m.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in comparison_fields} for row in records)

    print(
        f"done: {len(records)} labels | mean agreement with 60 m "
        f"{summary['means']['agreement_3c_with_60m']:.4f} | "
        f"2x2 purity {summary['means']['duplicate_purity']:.3f}"
    )
    print(f"manifest: {args.out / 'labels_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
