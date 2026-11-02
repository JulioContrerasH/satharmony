#!/usr/bin/env python
"""Proxy benchmark on the real v7 fine-tune test split (42 scenes).

Why this script exists
----------------------
The fair benchmark (``benchmark_mss_common_test.py``) puts every model on the
CloudSEN12 pseudo-MSS test, where the labels are CloudSEN12 (S2) aggregated to
60 m.  That test is common and independent of the models, but it is not real
Landsat MSS.  The only real-MSS labels in the project are the v7 fine-tune
labels, and those are **pseudo-labels**: clouds from a LightGBM trained on the
QA4EO-2 predictions (``label.tif``) plus shadows from a TM CNN.  They are not an
independent reference.

This script uses them anyway, as a **proxy**, because they are the only real
MSS labels available.  Everything that can bias the comparison is written out:

* the labels' provenance and the circularity for QA4EO-2 / its fine-tune;
* the subset (a deterministic chip grid over the 42 test scenes);
* the input convention of each model (physical TOA for the MSS models,
  per-patch z-score inside OmniCloudMask);
* a **second, independent reference**: the TM ``QA_PIXEL`` bits 3/1 (cloud)
  and 4 (shadow), warped to the 60 m MSS grid.  It is coarse and its shadow is
  weak, but it does not come from QA4EO-2.

Subset
------
The 42 ``test`` scenes of ``manifest-finetune-v7.csv``.  Each scene is cut into
128 x 128 chips on a regular grid with a 512 px stride (one chip every four),
keeping the chips with at least 50 % valid pixels.  That is ~30 chips per
scene, ~5 % of the test area, spread over every scene.  The grid, the stride
and the minimum valid fraction are recorded in ``run_benchmark.json``.

Models
------
``ocm_mss_60m``    OmniCloudMask on MSS bands (red, green, NIR2) at 60 m.
``ocm_mss_nir1``   Same with MSS3 as NIR (band-choice check).
``qa4eo2``         published QA4EO-2, raw TOA 0-1 (its best input convention).
``finetune``       best QA4EO-2 fine-tune, raw TOA 0-1.
``finetune_tta``   Same with the eight dihedral transforms.
``g4`` / ``g4b``   satharmony U-Nets, argmax + dihedral TTA, 4 -> 3 classes.
``g4_cal`` / ``g4b_cal``  Same models retrained on the calibrated fixture.
``g4b_ens``        softmax average of the four G4b ensemble members.

``--reuse`` loads every mask already stored in ``predictions.npz`` and computes
only the missing models, so the before/after run does not spend the OCM pass
twice.  The calibrated models were trained on ``pseudo_mss_fixture_cal``: the
fixture with one fixed scale and offset per band fitted on the train+val real
MSS (``scripts/calibrate_fixture_radiometry.py``).  Their checkpoint still
holds a fixed per-band mean/std, now computed on the calibrated chips.

There is **no** ``ocm_s2_10m`` row: the 42 v7 scenes are Landsat MSS/TM
(1972-2013) and have no Sentinel-2.  The 0.795 of the fair benchmark is
CloudSEN12 and does not belong to this subset.  The script says so instead of
inventing it.

Normalisation (the point of the task)
-------------------------------------
The MSS U-Nets see **physical TOA 0-1**: ``mss_original.tif`` is uint16 DN and
the dataloader divides by 10000 (``finetune/dataloader.py``), then the
checkpoint applies its **fixed** per-band training mean/std.  No per-chip
z-score is applied.

OmniCloudMask does the opposite: ``channel_norm`` in the installed package
(1.7.1) z-scores **each band of each patch** with the patch's nonzero mean and
std.  OCM never sees physical TOA, so its input distribution is not the same
as the U-Net's.  The script records this in the JSON and in the report.

Outputs (``--out``)
-------------------
``metrics_3class.csv/.md``   pooled per-class IoU vs the v7 proxy labels.
``metrics_qa_pixel.csv/.md`` same predictions vs TM QA_PIXEL (independent).
``metrics_perscene.md``      42 scenes as the sampling unit, win/loss vs OCM.
``metrics_perchip.md``       chips as the unit (secondary).
``metrics_regime.md``        pooled mIoU by label cloud fraction.
``chips.csv``                the chip grid and its label fractions.
``predictions.npz``          masks of every model (reused with --reuse).
``run_benchmark.json``       every number, hashes and the configuration.
``figures/fig_contact_sheet.png``

Usage
-----
    python scripts/benchmark_proxy_v7.py --max-chips 60 --no-figures
    python scripts/benchmark_proxy_v7.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.warp import Resampling, reproject

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from benchmark_mss_common_test import (  # noqa: E402
    DEFAULT_ENSEMBLE,
    DEFAULT_FINETUNE,
    DEFAULT_G4,
    DEFAULT_G4B,
    DEFAULT_QA4EO2,
    _chip_miou,
    collapse_to_three,
    evaluate_ensemble,
    evaluate_unet,
    legacy_predict,
    load_legacy_model,
    ocm_mask,
    score,
)
from evaluate_cloud_baseline import false_colour  # noqa: E402

from satharmony import assert_mss_toa_units  # noqa: E402
from satharmony.models import file_sha256  # noqa: E402

DEFAULT_MANIFEST = REPO.parent / "assets/md/manifest-finetune-v7.csv"
DEFAULT_OUT = REPO.parent / "assets/md/fuentes/bench-proxy-v7-cal"
DEFAULT_G4_CAL = Path("/data/databases/CLOUD_MSS/calibration/g4_cal/cloud_unet_mss_pseudo_cal.pt")
DEFAULT_G4B_CAL = Path("/data/databases/CLOUD_MSS/calibration/g4b_cal/g4b_best_cal.pt")
DEFAULT_RADIOMETRY = (
    REPO.parent / "assets/md/fuentes/calibracion-fixture/radiometry_calibration.json"
)

CLASS_NAMES = ("clear", "cloud", "cloud_shadow")
PRETTY = {"clear": "Claro", "cloud": "Nube", "cloud_shadow": "Sombra"}

#: MSS band indices: (green, red, NIR1, NIR2) -> 0, 1, 2, 3.
MSS_GREEN, MSS_RED, MSS_NIR1, MSS_NIR2 = 0, 1, 2, 3

#: OmniCloudMask order is (red, green, NIR).
OCM_BANDS = (MSS_RED, MSS_GREEN, MSS_NIR2)
OCM_BANDS_NIR1 = (MSS_RED, MSS_GREEN, MSS_NIR1)

MODEL_LABELS = {
    "ocm_mss_60m": "OCM @ MSS 60 m",
    "ocm_mss_nir1": "OCM @ MSS 60 m (NIR1)",
    "qa4eo2": "QA4EO-2 publicado",
    "finetune": "QA4EO-2 fine-tune",
    "finetune_tta": "QA4EO-2 fine-tune + TTA",
    "g4": "G4 U-Net",
    "g4b": "G4b U-Net",
    "g4b_ens": "G4b ensemble",
    "g4_cal": "G4 calibrado",
    "g4b_cal": "G4b calibrado",
}

COLORS = ["#f2f2f2", "#3a3a3a", "#a8a8a8"]
CMAP = ListedColormap(COLORS)

LEGACY_NAMES = ("qa4eo2", "finetune", "finetune_tta")


# ── Data ─────────────────────────────────────────────────────────────────
def read_test_rows(manifest: Path, split: str = "test") -> list[dict[str, str]]:
    """Manifest rows of one split.

    Args:
        manifest: v7 manifest CSV.
        split: Split name (``test`` by default).

    Returns:
        The rows of that split, in manifest order.

    Raises:
        ValueError: If the split is empty.
    """
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"{manifest} has no rows for split {split!r}")
    return selected


def _read_mss_aligned(path: Path, reference: dict) -> np.ndarray:
    """Read a 4-band MSS scene on the label grid, as physical TOA 0-1.

    Five of the 42 test scenes have a one-pixel (or larger) offset between
    ``mss_original.tif`` and ``label_refined_mss.tif``.  Reprojecting with
    nearest neighbour puts the MSS on the label grid without touching the
    radiometry; the scenes that already match are read directly.

    Args:
        path: ``mss_original.tif`` (uint16 DN at 60 m).
        reference: Profile of the label raster.

    Returns:
        ``(4, H, W)`` float32 reflectance in 0-1, H and W from the label.
    """
    with rasterio.open(path) as source:
        same_grid = (
            source.transform == reference["transform"]
            and source.crs == reference["crs"]
            and (source.height, source.width) == (reference["height"], reference["width"])
        )
        if same_grid:
            array = source.read().astype(np.float32)
        else:
            array = np.empty(
                (source.count, reference["height"], reference["width"]), dtype=np.float32
            )
            for index in range(source.count):
                reproject(
                    source=source.read(index + 1).astype(np.float32),
                    destination=array[index],
                    src_transform=source.transform,
                    src_crs=source.crs,
                    dst_transform=reference["transform"],
                    dst_crs=reference["crs"],
                    resampling=Resampling.nearest,
                )
    toa = array / 10000.0
    return assert_mss_toa_units(toa, name=str(path))


def _read_qa_pixel(path: Path, reference: dict) -> np.ndarray:
    """TM ``QA_PIXEL`` as a 3-class mask on the 60 m label grid.

    Bits 3 (cloud) and 1 (dilated cloud) are cloud; bit 4 is shadow, with
    priority over cloud.  This is the same rule as
    ``assets/codes/mss/analysis/qa_pixel_compare.py``.  The 30 m QA is
    averaged to 60 m and thresholded at 0.5.

    Args:
        path: ``pairs_tm/<id>_tm_original.tif``; QA_PIXEL is its last band.
        reference: Profile of the label raster.

    Returns:
        ``(H, W)`` uint8 classes: 0 clear, 1 cloud, 2 shadow.
    """
    with rasterio.open(path) as source:
        qa = source.read(source.count).astype(np.uint16)
        cloud = (((qa >> 3) & 1) | ((qa >> 1) & 1)).astype(np.float32)
        shadow = ((qa >> 4) & 1).astype(np.float32)
        height, width = reference["height"], reference["width"]
        cloud_out = np.empty((height, width), dtype=np.float32)
        shadow_out = np.empty((height, width), dtype=np.float32)
        for array, out in ((cloud, cloud_out), (shadow, shadow_out)):
            reproject(
                source=array,
                destination=out,
                src_transform=source.transform,
                src_crs=source.crs,
                dst_transform=reference["transform"],
                dst_crs=reference["crs"],
                resampling=Resampling.average,
            )
    classes = np.zeros((height, width), dtype=np.uint8)
    classes[(cloud_out > 0.5) & ~(shadow_out > 0.5)] = 1
    classes[shadow_out > 0.5] = 2
    return classes


def build_chips(
    rows: list[dict[str, str]],
    chip_size: int,
    stride: int,
    min_valid: float,
    max_chips: int | None = None,
) -> dict:
    """Deterministic chip grid over every test scene.

    A chip is kept when at least ``min_valid`` of its pixels have any MSS band
    above zero.  Nothing is sampled at random: the grid, the stride and the
    threshold fully determine the subset.

    Args:
        rows: Test-scene manifest rows.
        chip_size: Chip side in 60 m pixels.
        stride: Grid step in pixels (512 = one chip every four).
        min_valid: Minimum valid-pixel fraction to keep a chip.
        max_chips: Keep only the first N chips (smoke runs).

    Returns:
        Dict with ``images`` ``(N, 4, s, s)`` TOA, ``labels`` ``(N, s, s)``
        uint8, ``qa`` ``(N, s, s)`` uint8, ``valid`` ``(N, s, s)`` bool and
        ``meta`` (one dict per chip).
    """
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    qa_masks: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    meta: list[dict] = []

    for row in rows:
        scene = row["id_name"]
        directory = Path("/data/databases/CLOUD_MSS/pairs_optimization") / scene
        label_path = directory / "label_refined_mss.tif"
        mss_path = directory / "mss_original.tif"
        tm_path = Path("/data/databases/CLOUD_MSS/pairs_tm") / f"{scene}_tm_original.tif"
        with rasterio.open(label_path) as label_source:
            reference = label_source.profile
            label_scene = label_source.read(1)
        mss_scene = _read_mss_aligned(mss_path, reference)
        qa_scene = _read_qa_pixel(tm_path, reference)

        for y in range(0, reference["height"] - chip_size + 1, stride):
            for x in range(0, reference["width"] - chip_size + 1, stride):
                label = label_scene[y : y + chip_size, x : x + chip_size]
                valid = np.any(mss_scene[:, y : y + chip_size, x : x + chip_size] > 0, axis=0)
                if valid.mean() < min_valid:
                    continue
                images.append(mss_scene[:, y : y + chip_size, x : x + chip_size].copy())
                labels.append(label.copy())
                qa_masks.append(qa_scene[y : y + chip_size, x : x + chip_size].copy())
                valid_masks.append(valid.copy())
                counts = np.bincount(label[valid], minlength=3) / max(valid.sum(), 1)
                meta.append(
                    {
                        "scene": scene,
                        "category": row["category"],
                        "row": y,
                        "col": x,
                        "valid_frac": float(valid.mean()),
                        "clear_frac": float(counts[0]),
                        "cloud_frac": float(counts[1]),
                        "shadow_frac": float(counts[2]),
                    }
                )
                if max_chips is not None and len(meta) >= max_chips:
                    break
            if max_chips is not None and len(meta) >= max_chips:
                break
        print(f"  {scene}: {sum(m['scene'] == scene for m in meta)} chips")
        if max_chips is not None and len(meta) >= max_chips:
            break

    return {
        "images": np.stack(images).astype(np.float32),
        "labels": np.stack(labels).astype(np.uint8),
        "qa": np.stack(qa_masks).astype(np.uint8),
        "valid": np.stack(valid_masks),
        "meta": meta,
    }


# ── Models ───────────────────────────────────────────────────────────────
def ocm_to_three(mask: np.ndarray) -> np.ndarray:
    """CloudSEN12 4-class OCM mask -> clear / cloud / shadow.

    Args:
        mask: 0 clear, 1 thick, 2 thin, 3 shadow.

    Returns:
        uint8 mask with 0 clear, 1 cloud, 2 shadow.
    """
    return np.where(mask == 3, 2, np.where((mask == 1) | (mask == 2), 1, 0)).astype(np.uint8)


def _chunked(function, images: np.ndarray, chunk: int) -> np.ndarray:
    """Run a batch function in chunks and concatenate the masks."""
    outputs = [
        function(images[start : start + chunk]) for start in range(0, images.shape[0], chunk)
    ]
    return np.concatenate(outputs, axis=0)


def evaluate_all_models(
    data: dict,
    args: argparse.Namespace,
    device: str,
    cached: dict[str, np.ndarray] | None = None,
) -> dict:
    """Run every model on the chip stack, reusing cached masks when possible.

    Args:
        data: Output of :func:`build_chips`.
        args: Parsed CLI arguments.
        device: Torch device.
        cached: Masks already computed for the same chip grid.  A model in
            ``cached`` is not run again; only the missing ones are computed.

    Returns:
        ``{model: (N, s, s) uint8}`` 3-class masks.
    """
    import torch

    images = data["images"]
    masks: dict[str, np.ndarray] = dict(cached or {})
    for name in sorted(masks):
        print(f"[{name}] reused from predictions.npz")

    def done(name: str) -> bool:
        return name in masks

    for name, bands in (("ocm_mss_60m", OCM_BANDS), ("ocm_mss_nir1", OCM_BANDS_NIR1)):
        if done(name) or (name == "ocm_mss_nir1" and args.skip_ocm_nir1):
            continue
        start = time.perf_counter()
        masks[name] = np.asarray(
            [ocm_to_three(ocm_mask(image[list(bands)], device)) for image in images]
        )
        print(f"[{name}] {time.perf_counter() - start:.1f} s")

    for name, path in (("g4", args.g4), ("g4b", args.g4b)):
        if done(name):
            continue
        start = time.perf_counter()
        prediction = _chunked(
            lambda batch, checkpoint=path: evaluate_unet(checkpoint, batch, device), images, 128
        )
        masks[name] = collapse_to_three(prediction)
        print(f"[{name}] {time.perf_counter() - start:.1f} s")

    if not args.skip_calibrated:
        for name, path in (("g4_cal", args.g4_cal), ("g4b_cal", args.g4b_cal)):
            if done(name) or not Path(path).exists():
                if not Path(path).exists():
                    print(f"[{name}] missing checkpoint {path}, skipped")
                continue
            start = time.perf_counter()
            prediction = _chunked(
                lambda batch, checkpoint=path: evaluate_unet(checkpoint, batch, device),
                images,
                128,
            )
            masks[name] = collapse_to_three(prediction)
            print(f"[{name}] {time.perf_counter() - start:.1f} s")

    if not args.skip_ensemble and not done("g4b_ens"):
        start = time.perf_counter()
        prediction = _chunked(
            lambda batch: evaluate_ensemble(tuple(args.ensemble), batch, device), images, 64
        )
        masks["g4b_ens"] = collapse_to_three(prediction)
        print(f"[g4b_ens] {time.perf_counter() - start:.1f} s")

    if not args.skip_legacy:
        for name, path in (("qa4eo2", args.qa4eo2), ("finetune", args.finetune)):
            if done(name):
                continue
            if not Path(path).exists():
                print(f"[{name}] missing checkpoint {path}, skipped")
                continue
            model = load_legacy_model(Path(path), device)
            chunk = 8 if name == "finetune" else 16
            start = time.perf_counter()
            masks[name] = _chunked(
                lambda batch, net=model: legacy_predict(net, batch, device, tta=False),
                images,
                chunk,
            )
            print(f"[{name}] {time.perf_counter() - start:.1f} s")
            if name == "finetune" and not done("finetune_tta"):
                start = time.perf_counter()
                masks["finetune_tta"] = _chunked(
                    lambda batch, net=model: legacy_predict(net, batch, device, tta=True), images, 8
                )
                print(f"[finetune_tta] {time.perf_counter() - start:.1f} s")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return masks


# ── Metrics ──────────────────────────────────────────────────────────────
def _pooled_score(prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> dict:
    """``score`` over the valid pixels of a stack."""
    return score(prediction[valid], reference[valid], CLASS_NAMES)


def _per_scene_scores(
    masks: dict[str, np.ndarray], labels: np.ndarray, valid: np.ndarray, meta: list[dict]
) -> dict[str, dict[str, float]]:
    """Pooled mIoU per scene for every model."""
    scenes = sorted({item["scene"] for item in meta})
    index = {scene: position for position, scene in enumerate(scenes)}
    positions = np.asarray([index[item["scene"]] for item in meta])
    out: dict[str, dict[str, float]] = {}
    for name, mask in masks.items():
        per_scene: dict[str, float] = {}
        for scene, number in index.items():
            keep = positions == number
            per_scene[scene] = _pooled_score(mask[keep], labels[keep], valid[keep])["mean_iou"]
        out[name] = per_scene
    return out


def _scene_summary(per_scene: dict[str, dict[str, float]], reference: str) -> dict[str, dict]:
    """Mean, median and win/loss share over scenes against one model."""
    baseline = np.asarray(list(per_scene[reference].values()))
    out: dict[str, dict] = {}
    for name, values in per_scene.items():
        array = np.asarray(list(values.values()))
        delta = array - baseline
        out[name] = {
            "mean_miou": float(np.mean(array)),
            "median_miou": float(np.median(array)),
            "win_share": float(np.mean(delta > 0.01)),
            "loss_share": float(np.mean(delta < -0.01)),
            "median_delta": float(np.median(delta)),
            "scenes_won_by_010": int(np.sum(delta > 0.1)),
            "scenes_lost_by_010": int(np.sum(delta < -0.1)),
        }
    return out


def _chip_regime(cloud_fraction: float) -> str:
    """Chip regime from its cloud fraction."""
    if cloud_fraction < 0.01:
        return "clear"
    if cloud_fraction < 0.20:
        return "low_cloud"
    if cloud_fraction < 0.50:
        return "mid_cloud"
    return "high_cloud"


# ── Tables ───────────────────────────────────────────────────────────────
def write_metrics_csv(path: Path, results: dict[str, dict]) -> None:
    """One row per model and class plus a summary row."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "class", "support", "precision", "recall", "f1", "iou"])
        for model, result in results.items():
            for class_name in CLASS_NAMES:
                values = result["per_class"][class_name]
                writer.writerow(
                    [
                        model,
                        class_name,
                        int(values["support"]),
                        *[f"{values[key]:.6f}" for key in ("precision", "recall", "f1", "iou")],
                    ]
                )
            writer.writerow(
                [
                    model,
                    "SUMMARY",
                    "",
                    "",
                    "",
                    f"mIoU={result['mean_iou']:.6f}",
                    f"OA={result['overall_accuracy']:.6f} kappa={result['cohen_kappa']:.6f}",
                ]
            )


def write_metrics_markdown(path: Path, results: dict[str, dict], title: str, note: str) -> None:
    """Markdown table: one row per model, one column per class IoU."""
    lines = [f"# {title}", "", note, ""]
    lines.append(
        "| Modelo | " + " | ".join(PRETTY[name] for name in CLASS_NAMES) + " | mIoU | OA | kappa |"
    )
    lines.append("|---" + "|---:" * (len(CLASS_NAMES) + 3) + "|")
    for model, result in results.items():
        cells = [
            "—"
            if not np.isfinite(result["per_class"][name]["iou"])
            else f"{result['per_class'][name]['iou']:.3f}"
            for name in CLASS_NAMES
        ]
        lines.append(
            f"| {MODEL_LABELS.get(model, model)} | "
            + " | ".join(cells)
            + f" | **{result['mean_iou']:.3f}** | {result['overall_accuracy']:.3f} | "
            f"{result['cohen_kappa']:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def write_before_after(
    path: Path,
    results: dict[str, dict],
    pairs: tuple[tuple[str, str], ...] = (("g4", "g4_cal"), ("g4b", "g4b_cal")),
) -> None:
    """Before/after table for the models retrained on the calibrated fixture.

    The pair shares architecture, loss, split and seed; the only difference is
    the fixture radiometry.  The table is written only when both rows exist.

    Args:
        path: Markdown output.
        results: ``{model: score(...)}`` of the run.
        pairs: ``(before, after)`` model names.
    """
    lines = [
        "# Antes y después de calibrar el fixture",
        "",
        "Misma arquitectura, misma loss, mismo split y misma semilla; lo único que cambia "
        "es la radiometría del fixture. Delta = después - antes.",
        "",
        "| Modelo | Claro | Nube | Sombra | mIoU | Δ mIoU | Δ OA | Δ kappa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    wrote = False
    for before, after in pairs:
        if before not in results or after not in results:
            continue
        wrote = True
        old, new = results[before], results[after]
        cells = []
        for name in CLASS_NAMES:
            iou_old = old["per_class"][name]["iou"]
            iou_new = new["per_class"][name]["iou"]
            cells.append(f"{iou_old:.3f} → {iou_new:.3f}")
        lines.append(
            f"| {MODEL_LABELS.get(before, before)} → {MODEL_LABELS.get(after, after)} | "
            + " | ".join(cells)
            + f" | {old['mean_iou']:.3f} → {new['mean_iou']:.3f} | "
            f"{new['mean_iou'] - old['mean_iou']:+.3f} | "
            f"{new['overall_accuracy'] - old['overall_accuracy']:+.3f} | "
            f"{new['cohen_kappa'] - old['cohen_kappa']:+.3f} |"
        )
    if not wrote:
        return
    lines += [
        "",
        "El fixture calibrado no cambia las etiquetas: la mejora no puede venir de una "
        "etiqueta distinta, solo de la radiometría de entrada.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


# ── Figures ──────────────────────────────────────────────────────────────
def _pick_rows(
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    meta: list[dict],
    valid: np.ndarray,
) -> list[int]:
    """Six rows: one chip per regime, the shadow-rich one and the biggest gap."""
    reference = next(
        (name for name in ("g4b_cal", "g4b_ens", "g4b") if name in masks), None
    )
    per_chip: dict[str, np.ndarray] = {}
    for name in (reference, "ocm_mss_60m"):
        if name and name in masks:
            per_chip[name] = np.asarray(
                [
                    _chip_miou(
                        masks[name][position][valid[position]], labels[position][valid[position]]
                    )
                    for position in range(len(meta))
                ]
            )
    picked: list[int] = []
    for regime in ("clear", "low_cloud", "mid_cloud", "high_cloud"):
        candidates = [
            position
            for position, item in enumerate(meta)
            if _chip_regime(item["cloud_frac"]) == regime
        ]
        if candidates:
            if reference in per_chip:
                picked.append(max(candidates, key=lambda position: per_chip[reference][position]))
            else:
                picked.append(candidates[0])
    shadow = max(range(len(meta)), key=lambda position: meta[position]["shadow_frac"])
    picked.append(shadow)
    if {reference, "ocm_mss_60m"} <= set(per_chip):
        delta = per_chip[reference] - per_chip["ocm_mss_60m"]
        picked.append(int(np.argmax(delta)))
    unique = list(dict.fromkeys(picked))
    return unique[:6]


def figure_contact_sheet(
    path: Path,
    data: dict,
    masks: dict[str, np.ndarray],
    rows: list[int],
) -> None:
    """MSS, proxy label, QA_PIXEL and the available models on the same chips."""
    figure_models = [
        name
        for name in (
            "ocm_mss_60m",
            "qa4eo2",
            "g4",
            "g4b",
            "g4_cal",
            "g4b_cal",
            "g4b_ens",
        )
        if name in masks
    ][:6]
    short = {
        "ocm_mss_60m": "OCM @ MSS",
        "qa4eo2": "QA4EO-2",
        "g4": "G4",
        "g4b": "G4b",
        "g4_cal": "G4 cal",
        "g4b_cal": "G4b cal",
        "g4b_ens": "G4b ensemble",
    }
    columns = ["MSS falso color", "Label proxy", "TM QA_PIXEL"] + [
        short[name] for name in figure_models
    ]
    figure, axes = plt.subplots(
        len(rows), len(columns), figsize=(2.0 * len(columns), 2.0 * len(rows)), squeeze=False
    )
    for row_position, index in enumerate(rows):
        meta = data["meta"][index]
        panels = [
            false_colour(data["images"][index]),
            CMAP(data["labels"][index]),
            CMAP(data["qa"][index]),
        ]
        panels += [CMAP(masks[name][index]) for name in figure_models]
        for column, panel in enumerate(panels):
            axis = axes[row_position, column]
            axis.imshow(panel, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if row_position == 0:
                axis.set_title(columns[column], fontsize=9)
            if column == 0:
                axis.set_ylabel(f"{meta['scene']}\n{meta['row']},{meta['col']}", fontsize=7)
    handles = [
        Patch(facecolor=color, edgecolor="#999999", label=name)
        for color, name in zip(COLORS, ("Claro", "Nube", "Sombra"), strict=True)
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    figure.suptitle(
        "Proxy v7: 128 x 128 chips at 60 m. Label = old pseudo-labels (QA4EO-2 clouds + TM shadows)",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chip-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--min-valid", type=float, default=0.5)
    parser.add_argument("--g4", type=Path, default=DEFAULT_G4)
    parser.add_argument("--g4b", type=Path, default=DEFAULT_G4B)
    parser.add_argument("--g4-cal", type=Path, default=DEFAULT_G4_CAL)
    parser.add_argument("--g4b-cal", type=Path, default=DEFAULT_G4B_CAL)
    parser.add_argument("--ensemble", type=Path, nargs="+", default=list(DEFAULT_ENSEMBLE))
    parser.add_argument("--radiometry", type=Path, default=DEFAULT_RADIOMETRY)
    parser.add_argument("--qa4eo2", type=Path, default=DEFAULT_QA4EO2)
    parser.add_argument("--finetune", type=Path, default=DEFAULT_FINETUNE)
    parser.add_argument("--max-chips", type=int, default=None)
    parser.add_argument("--skip-ocm-nir1", action="store_true")
    parser.add_argument("--skip-ensemble", action="store_true")
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--skip-calibrated", action="store_true")
    parser.add_argument("--reuse", action="store_true", help="reuse predictions.npz")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    rows = read_test_rows(args.manifest)
    print(f"{len(rows)} test scenes, chip {args.chip_size} stride {args.stride}")
    data = build_chips(rows, args.chip_size, args.stride, args.min_valid, args.max_chips)
    print(
        f"{len(data['meta'])} chips, {data['images'].shape}, "
        f"{data['valid'].mean() * 100:.1f} % valid pixels, device={device}"
    )

    predictions_path = args.out / "predictions.npz"
    cached: dict[str, np.ndarray] = {}
    if args.reuse and predictions_path.exists():
        archive = np.load(predictions_path)
        cached_chips = int(archive["_n_chips"]) if "_n_chips" in archive.files else -1
        if cached_chips != len(data["meta"]):
            raise ValueError(
                f"{predictions_path} has {cached_chips} chips, the grid built "
                f"{len(data['meta'])}; rerun without --reuse"
            )
        cached = {name: archive[name] for name in archive.files if not name.startswith("_")}
        print(f"loaded {len(cached)} cached model masks from {predictions_path}")
    masks = evaluate_all_models(data, args, device, cached)
    np.savez_compressed(
        predictions_path,
        _n_chips=np.asarray(len(data["meta"])),
        _chip_size=np.asarray(args.chip_size),
        _stride=np.asarray(args.stride),
        **masks,
    )

    labels, valid, qa = data["labels"], data["valid"], data["qa"]

    # Primary reference: the v7 proxy labels.
    results = {name: _pooled_score(mask, labels, valid) for name, mask in masks.items()}
    for name, result in results.items():
        print(
            f"[{name}] proxy mIoU {result['mean_iou']:.4f} | OA {result['overall_accuracy']:.4f}"
            f" | kappa {result['cohen_kappa']:.4f}"
        )

    write_metrics_csv(args.out / "metrics_3class.csv", results)
    write_metrics_markdown(
        args.out / "metrics_3class.md",
        results,
        "Proxy v7 — 3 clases contra los pseudo-labels del fine-tune",
        "Referencia: `label_refined_mss.tif` (nubes de un LightGBM entrenado con las "
        "predicciones de QA4EO-2 + sombras de la CNN de TM, agregadas a 60 m). **No es una "
        "referencia independiente**: QA4EO-2 y su fine-tune están sesgados a su favor.",
    )

    write_before_after(args.out / "metrics_before_after.md", results)

    # Secondary, independent reference: TM QA_PIXEL.
    qa_results = {name: _pooled_score(mask, qa, valid) for name, mask in masks.items()}
    write_metrics_csv(args.out / "metrics_qa_pixel.csv", qa_results)
    write_metrics_markdown(
        args.out / "metrics_qa_pixel.md",
        qa_results,
        "Proxy v7 — la misma predicción contra TM QA_PIXEL (referencia independiente)",
        "Referencia: bits 3/1 (nube) y 4 (sombra) del QA_PIXEL de TM, 30 m -> 60 m. No viene de "
        "QA4EO-2. Es gruesa y su sombra es débil; sirve para ver si el orden se mantiene, no "
        "como ground truth.",
    )

    # Per scene: the 42 scenes are the sampling unit.
    per_scene = _per_scene_scores(masks, labels, valid, data["meta"])
    scene_summary = _scene_summary(per_scene, "ocm_mss_60m") if "ocm_mss_60m" in per_scene else {}
    lines = [
        "# Por escena (42 escenas del test v7)",
        "",
        "Cada escena pesa lo mismo, aunque tenga más o menos chips. *Gana* = +0,01 mIoU o más "
        "contra OCM @ MSS; *pierde* = -0,01 o menos.",
        "",
        "| Modelo | mIoU medio | mIoU mediano | Gana | Pierde | Delta mediano | "
        "Gana >0,1 | Pierde >0,1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in scene_summary.items():
        lines.append(
            f"| {MODEL_LABELS.get(name, name)} | {values['mean_miou']:.3f} | "
            f"{values['median_miou']:.3f} | {100 * values['win_share']:.0f} % | "
            f"{100 * values['loss_share']:.0f} % | {values['median_delta']:+.3f} | "
            f"{values['scenes_won_by_010']} | {values['scenes_lost_by_010']} |"
        )
    (args.out / "metrics_perscene.md").write_text("\n".join(lines) + "\n")

    # Per chip.
    per_chip = {
        name: np.asarray(
            [
                _chip_miou(mask[position][valid[position]], labels[position][valid[position]])
                for position in range(len(data["meta"]))
            ]
        )
        for name, mask in masks.items()
    }
    baseline = per_chip["ocm_mss_60m"]
    lines = [
        "# Por chip (secundario)",
        "",
        "Un chip pesa lo mismo que otro. *Gana* = +0,01 mIoU o más; *pierde* = -0,01 o menos.",
        "",
        "| Modelo | mIoU medio | mIoU mediano | Gana | Pierde | Delta mediano |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in per_chip.items():
        if name == "ocm_mss_60m":
            continue
        delta = values - baseline
        lines.append(
            f"| {MODEL_LABELS.get(name, name)} | {values.mean():.3f} | {np.median(values):.3f} | "
            f"{100 * np.mean(delta > 0.01):.0f} % | {100 * np.mean(delta < -0.01):.0f} % | "
            f"{np.median(delta):+.3f} |"
        )
    (args.out / "metrics_perchip.md").write_text("\n".join(lines) + "\n")

    # By cloud regime.
    lines = [
        "# Por régimen de nube (mIoU, 3 clases)",
        "",
        "Régimen por la fracción de nube del label proxy del chip.",
        "",
        "| Régimen | Chips | " + " | ".join(MODEL_LABELS[name] for name in masks) + " |",
        "|---|---:|" + "---:|" * len(masks),
    ]
    for regime in ("clear", "low_cloud", "mid_cloud", "high_cloud"):
        indices = [
            position
            for position, item in enumerate(data["meta"])
            if _chip_regime(item["cloud_frac"]) == regime
        ]
        if not indices:
            continue
        cells = [
            f"{_pooled_score(masks[name][indices], labels[indices], valid[indices])['mean_iou']:.3f}"
            for name in masks
        ]
        lines.append(f"| {regime} | {len(indices)} | " + " | ".join(cells) + " |")
    (args.out / "metrics_regime.md").write_text("\n".join(lines) + "\n")

    # False positives on chips that the proxy label says are clean.
    clean = [
        position
        for position, item in enumerate(data["meta"])
        if item["cloud_frac"] < 0.01 and item["shadow_frac"] < 0.01
    ]
    false_positives = {}
    lines = [
        "# Falsos positivos en chips claros",
        "",
        f"{len(clean)} chips con menos de 1 % de nube y 1 % de sombra en el label proxy. "
        "La columna *pooled* pesa los píxeles; las demás tratan cada chip igual, porque un "
        "solo chip saturado no debe esconder que el resto está bien.",
        "",
        "| Modelo | Nube (pooled) | Nube (≥90 % válido) | Mediana | p90 | Máx | "
        "Chips >10 % | Chips >50 % | Sombra (pooled) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    clean_interior = [position for position in clean if data["meta"][position]["valid_frac"] >= 0.9]
    for name, mask in masks.items():
        cloud = np.asarray(
            [float(np.mean(mask[position][valid[position]] == 1)) for position in clean]
        )
        shadow = float(np.mean(mask[clean][valid[clean]] == 2))
        false_positives[name] = {
            "n_chips": len(clean),
            "cloud_pooled": float(np.mean(mask[clean][valid[clean]] == 1)),
            "cloud_pooled_valid90": float(
                np.mean(mask[clean_interior][valid[clean_interior]] == 1)
            ),
            "cloud_median": float(np.median(cloud)),
            "cloud_p90": float(np.percentile(cloud, 90)),
            "cloud_max": float(np.max(cloud)),
            "chips_over_10pct": int(np.sum(cloud > 0.10)),
            "chips_over_50pct": int(np.sum(cloud > 0.50)),
            "shadow_pooled": shadow,
            "contaminated_pooled": float(np.mean(mask[clean][valid[clean]] != 0)),
        }
        lines.append(
            f"| {MODEL_LABELS.get(name, name)} | {100 * false_positives[name]['cloud_pooled']:.2f} % | "
            f"{100 * false_positives[name]['cloud_pooled_valid90']:.2f} % | "
            f"{100 * np.median(cloud):.2f} % | {100 * np.percentile(cloud, 90):.2f} % | "
            f"{100 * np.max(cloud):.2f} % | {int(np.sum(cloud > 0.10))} | "
            f"{int(np.sum(cloud > 0.50))} | {100 * shadow:.2f} % |"
        )
    (args.out / "metrics_false_positives.md").write_text("\n".join(lines) + "\n")

    # How well the two references agree with each other.
    label_vs_qa = _pooled_score(qa, labels, valid)
    print(f"label vs QA_PIXEL: mIoU {label_vs_qa['mean_iou']:.4f}")

    # Chip list.
    with (args.out / "chips.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data["meta"][0].keys()))
        writer.writeheader()
        writer.writerows(data["meta"])

    # JSON.
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.perf_counter() - start, 1),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "split": "test",
        "n_scenes": len(rows),
        "n_chips": len(data["meta"]),
        "chip_size": args.chip_size,
        "stride": args.stride,
        "min_valid": args.min_valid,
        "valid_pixel_fraction": float(data["valid"].mean()),
        "chip_band_mean": np.asarray(data["images"].mean(axis=(0, 2, 3))).tolist(),
        "chip_band_std": np.asarray(data["images"].std(axis=(0, 2, 3))).tolist(),
        "normalisation": {
            "mss_unet": (
                "physical TOA 0-1 (uint16 DN / 10000) then the checkpoint's fixed per-band "
                "mean/std; no per-chip z-score"
            ),
            "qa4eo2": "raw TOA 0-1; no z-score",
            "ocm": (
                "internal channel_norm: per-band z-score of each patch with its own nonzero "
                "mean/std (omnicloudmask 1.7.1); OCM never sees physical TOA"
            ),
        },
        "labels": {
            "primary": "label_refined_mss.tif (v7 fine-tune labels)",
            "provenance": (
                "clouds from a LightGBM trained on the QA4EO-2 predictions (label.tif) + "
                "shadows from the TM CNN, aggregated to 60 m; not independent"
            ),
            "secondary": "TM QA_PIXEL bits 3/1 (cloud) and 4 (shadow), 30 m -> 60 m",
        },
        "checkpoints": {
            "g4": {"path": str(args.g4), "sha256": file_sha256(args.g4)},
            "g4b": {"path": str(args.g4b), "sha256": file_sha256(args.g4b)},
            "g4_cal": {
                "path": str(args.g4_cal),
                "sha256": file_sha256(args.g4_cal) if Path(args.g4_cal).exists() else None,
            },
            "g4b_cal": {
                "path": str(args.g4b_cal),
                "sha256": file_sha256(args.g4b_cal) if Path(args.g4b_cal).exists() else None,
            },
            "ensemble": [
                {"path": str(path), "sha256": file_sha256(path)} for path in args.ensemble
            ],
            "qa4eo2": {"path": str(args.qa4eo2), "sha256": file_sha256(args.qa4eo2)},
            "finetune": {"path": str(args.finetune), "sha256": file_sha256(args.finetune)},
        },
        "fixture_radiometry": (
            json.loads(Path(args.radiometry).read_text())
            if Path(args.radiometry).exists()
            else None
        ),
        "scores_proxy": results,
        "scores_qa_pixel": qa_results,
        "label_vs_qa_pixel": label_vs_qa,
        "false_positives_clear": false_positives,
        "per_scene_miou": per_scene,
        "per_scene_summary": scene_summary,
    }
    (args.out / "run_benchmark.json").write_text(json.dumps(payload, indent=2) + "\n")

    if not args.no_figures:
        rows_to_draw = _pick_rows(labels, masks, data["meta"], valid)
        figure_contact_sheet(
            args.out / "figures" / "fig_contact_sheet.png", data, masks, rows_to_draw
        )
        print(f"contact sheet: {len(rows_to_draw)} chips")

    print(f"done in {time.perf_counter() - start:.1f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
