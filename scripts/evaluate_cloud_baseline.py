#!/usr/bin/env python
"""Evaluate the G4/G4b cloud models: pseudo test, real PICS, OmniCloudMask.

The script answers four questions with the same metric code:

1. How well does each checkpoint do on the held-out CloudSEN12 test split
   (194 chips, scene-level split)?  With plain argmax and with the per-class
   bias of the decision rule.
2. Does multi-scale TTA help?  ``--scales auto`` picks the scale set with the
   best validation mIoU and only then touches the test split.
3. How does OmniCloudMask (OCM) compare on the *same* pseudo test split?  OCM
   sees the Sentinel-2 chips it expects (B04, B03, B8A at 10 m) and its 10 m
   mask is aggregated to 60 m with the same conservative rule used to build the
   reference labels.
4. How do the models behave **zero-shot** on the five real PICS (Landsat 5 MSS
   L1TP at 60 m)?  There are no human labels yet, so only predictions and
   diagnostics are reported there.

Outputs (``--out``)::

    run_eval.json                 every number of the run
    metrics_pseudo_test.csv/.md   per-class table for every model and OCM
    metrics_pics.csv/.md          per-site predictions and diagnostics
    confusion_<model>_pseudo.npz  confusion matrices (counts)
    ocm_pseudo_test.npz           cached OCM masks on the test chips
    figures/fig_pseudo_predictions.png
    figures/fig_pics_predictions.png
    figures/fig_confusion_pseudo.png

Usage
-----
    python scripts/evaluate_cloud_baseline.py
    python scripts/evaluate_cloud_baseline.py --checkpoint /path/g4b.pt \
        --baseline-checkpoint /path/g4.pt --out ../assets/md/fuentes/g4b --reuse-ocm
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from build_pseudo_mss_fixture import profile_for_scene, stable_seed  # noqa: E402

from satharmony.datasets.cloudsen12 import array_sha256  # noqa: E402
from satharmony.emulators.s2_to_mss.functional import emulate_s2_with_labels  # noqa: E402
from satharmony.emulators.s2_to_mss.labels import (  # noqa: E402
    aggregate_cloudsen12,
)
from satharmony.models import (  # noqa: E402
    CLASS_NAMES,
    apply_class_bias,
    cohen_kappa,
    confusion_matrix,
    evaluate,
    iou_per_class,
    load_model,
    mean_iou,
    overall_accuracy,
    predict_proba_multiscale,
    tune_class_bias,
)

DEFAULT_FIXTURE = Path("/data/databases/CLOUD_MSS/calibration/pseudo_mss_fixture")
DEFAULT_CHECKPOINT = Path("/data/databases/CLOUD_MSS/calibration/g4b/g4b_best.pt")
DEFAULT_BASELINE = Path("/data/databases/CLOUD_MSS/calibration/g4/cloud_unet_mss_pseudo.pt")
DEFAULT_OUT = REPO.parent / "assets/md/fuentes/g4b"
DEFAULT_PICS = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")
DEFAULT_CLOUDSEN12 = Path("/data/databases/CMIXII/CLOUDSEN12_1K")
DEFAULT_LABELS = REPO.parent / "assets/md/fuentes/labels-humano"

#: CloudSEN12 s2.tif band indices (B1..B8, B8A, B9, B10, B11, B12) of the three
#: bands OmniCloudMask expects (red, green, NIR).
OCM_CLOUDSEN12_INDEX = {"red": 3, "green": 2, "nir": 8}

#: The same three bands in the PICS per-band GeoTIFFs.
OCM_PICS_BANDS = ("B4", "B3", "B8A")

CHIP_SIZE = 64
SCALE_FACTOR = 6
PICS_GSD = 60.0
PROB_MASK = 20
QA_PIXEL_CLOUD = 0b1000  # bit 3 of the Landsat Collection 2 QA_PIXEL

#: Scale sets tried by ``--scales auto`` on the validation split.
SCALE_CANDIDATES: tuple[tuple[float, ...], ...] = ((1.0,), (1.0, 1.5))

PRETTY = {
    "clear": "Clear",
    "thick_cloud": "Thick cloud",
    "thin_cloud": "Thin cloud",
    "cloud_shadow": "Cloud shadow",
}
CLASS_COLORS = ["#f2f2f2", "#08306b", "#6baed6", "#525252"]
CLASS_CMAP = ListedColormap(CLASS_COLORS)
ERROR_CMAP = ListedColormap(["#2ca25f", "#de2d26"])


# ── Fixture and labels ────────────────────────────────────────────────────
def read_manifest(fixture: Path) -> list[dict[str, str]]:
    """Rows of the fixture manifest."""
    with (fixture / "manifest.csv").open() as handle:
        return list(csv.DictReader(handle))


def human_labels_state(labels_dir: Path) -> dict:
    """Whether the two annotation CSVs contain any human label.

    Args:
        labels_dir: Directory with ``anotaciones_anotador1.csv`` and
            ``anotaciones_anotador2.csv``.

    Returns:
        Dict with the number of filled rows per annotator and an ``available``
        flag; empty CSVs mean there is nothing to score against.
    """
    state = {"available": False, "annotators": {}}
    for name in ("anotaciones_anotador1.csv", "anotaciones_anotador2.csv"):
        path = labels_dir / name
        if not path.exists():
            state["annotators"][name] = {"exists": False, "filled_rows": 0}
            continue
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        filled = sum(1 for row in rows if (row.get("clase_dominante") or "").strip())
        state["annotators"][name] = {"exists": True, "filled_rows": filled}
        state["available"] = state["available"] or filled > 0
    return state


def verify_fixture_chips(
    chips: list[dict[str, str]],
    images_path: Path,
    labels_path: Path,
    data_dir: Path,
    seed: int,
) -> dict:
    """Re-run the emulation of a few chips and compare their hashes.

    This proves that the Sentinel-2 crops used for the OmniCloudMask comparison
    are the exact ones that produced the fixture chips.

    Args:
        chips: Manifest rows to verify.
        images_path: ``images.npy`` of the fixture.
        labels_path: ``labels.npy`` of the fixture.
        data_dir: CloudSEN12 root.
        seed: Base seed of the fixture build.

    Returns:
        Dict with ``checked``, ``matched`` and per-chip results.
    """
    from satharmony.datasets.cloudsen12 import read_scene

    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    manifest = read_manifest(images_path.parent)
    position = {row["chip_id"]: index for index, row in enumerate(manifest)}
    results = []
    for chip in chips:
        index = position[chip["chip_id"]]
        scene = read_scene(data_dir / chip["chip_id"])
        config = profile_for_scene(scene.metadata)
        mss, labels60, _ = emulate_s2_with_labels(
            scene.s2,
            scene.labels,
            seed=stable_seed(seed, f"emulate:{chip['chip_id']}"),
            config=config,
        )
        row, col = int(chip["window_row"]), int(chip["window_col"])
        quantized = np.rint(np.clip(mss[:, row : row + CHIP_SIZE, col : col + CHIP_SIZE], 0, 1)
                            * 10000).astype(np.uint16)
        digest = array_sha256(quantized, labels60[row : row + CHIP_SIZE, col : col + CHIP_SIZE])
        results.append(
            {
                "chip_id": chip["chip_id"],
                "matched": digest == chip["sha256"],
                "matches_fixture": bool(
                    np.array_equal(quantized, np.asarray(images[index]))
                    and np.array_equal(
                        labels60[row : row + CHIP_SIZE, col : col + CHIP_SIZE],
                        np.asarray(labels[index]),
                    )
                ),
            }
        )
    return {
        "checked": len(results),
        "matched": int(sum(item["matched"] for item in results)),
        "chips": results,
    }


# ── OmniCloudMask ─────────────────────────────────────────────────────────
def ocm_predict(array: np.ndarray, device: str) -> np.ndarray:
    """Run OmniCloudMask on a 3-band (red, green, NIR) 10 m array.

    Args:
        array: ``(3, H, W)`` float32 reflectance, 0-1.
        device: Torch device string.

    Returns:
        ``(H, W)`` uint8 classes with 0 clear, 1 thick, 2 thin, 3 shadow.
    """
    from omnicloudmask import predict_from_array

    height, width = array.shape[-2:]
    patch = min(1000, max(64, 8 * (min(height, width) // 8)))
    return predict_from_array(
        np.ascontiguousarray(array, dtype=np.float32),
        patch_size=patch,
        inference_device=device,
    )[0].astype(np.uint8)


def ocm_version() -> str:
    """Installed OmniCloudMask version, or ``unknown``."""
    try:
        import omnicloudmask

        return str(getattr(omnicloudmask, "__version__", "unknown"))
    except ImportError:
        return "missing"


def read_cloudsen12_crop(
    scene_dir: Path, row: int, col: int, size: int = CHIP_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Read the 10 m Sentinel-2 and label windows of one fixture chip.

    Args:
        scene_dir: CloudSEN12 chip directory (``s2.tif``, ``target.tif``).
        row: Window row on the 60 m grid.
        col: Window column on the 60 m grid.
        size: Window size in 60 m cells.

    Returns:
        ``(s2, labels10)`` with s2 ``(13, size*6, size*6)`` float32 reflectance
        in CloudSEN12 band order and labels uint8 with 99 replaced by 0.
    """
    import rasterio
    from rasterio.windows import Window

    window = Window(col * SCALE_FACTOR, row * SCALE_FACTOR, size * SCALE_FACTOR,
                    size * SCALE_FACTOR)
    with rasterio.open(scene_dir / "s2.tif") as source:
        s2 = source.read(window=window).astype(np.float32) / 10000.0
    with rasterio.open(scene_dir / "target.tif") as source:
        raw = source.read(1, window=window)
    labels = np.where(raw == 99, 0, raw).astype(np.uint8)
    return s2, labels


def ocm_pseudo_test(
    chips: list[dict[str, str]],
    data_dir: Path,
    cache: Path,
    device: str,
    reuse: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """OCM masks for every pseudo-test chip, aggregated to 60 m.

    Args:
        chips: Test-split manifest rows.
        data_dir: CloudSEN12 root.
        cache: ``.npz`` path for the 60 m masks.
        device: Torch device.
        reuse: Load the cache when it exists.

    Returns:
        ``(predictions, crosstab)``; predictions ``(N, 64, 64)`` uint8 at 60 m
        and the 10 m cross-tabulation against the CloudSEN12 labels
        (reference rows, OCM columns).
    """
    if reuse and cache.exists():
        payload = np.load(cache)
        if list(payload["chip_ids"]) == [chip["chip_id"] for chip in chips]:
            print(f"reusing OCM cache {cache}")
            return payload["predictions"], payload["crosstab"]

    predictions = np.zeros((len(chips), CHIP_SIZE, CHIP_SIZE), dtype=np.uint8)
    crosstab = np.zeros((4, 4), dtype=np.int64)
    for index, chip in enumerate(chips, start=1):
        s2, labels10 = read_cloudsen12_crop(
            data_dir / chip["chip_id"], int(chip["window_row"]), int(chip["window_col"])
        )
        mask10 = ocm_predict(
            np.stack([s2[OCM_CLOUDSEN12_INDEX["red"]], s2[OCM_CLOUDSEN12_INDEX["green"]],
                      s2[OCM_CLOUDSEN12_INDEX["nir"]]]),
            device,
        )
        crosstab += confusion_matrix(mask10, labels10)
        predictions[index - 1] = aggregate_cloudsen12(mask10, scale_factor=SCALE_FACTOR)
        if index % 25 == 0 or index == len(chips):
            print(f"  OCM {index}/{len(chips)} chips")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        predictions=predictions,
        crosstab=crosstab,
        chip_ids=np.array([chip["chip_id"] for chip in chips]),
    )
    return predictions, crosstab


# ── Real PICS ─────────────────────────────────────────────────────────────
def read_band(directory: Path, band: str, dtype=np.float32) -> np.ndarray:
    """Read the single GeoTIFF that matches ``*.{band}.tif`` in ``directory``."""
    import rasterio

    matches = sorted(directory.glob(f"*.{band}.tif"))
    if not matches:
        raise FileNotFoundError(f"no {band} band in {directory}")
    with rasterio.open(matches[0]) as source:
        return source.read(1).astype(dtype)


def load_pics_site(site_dir: Path) -> dict:
    """Load MSS TOA, S2 OCM bands, QA and saturation of one PICS.

    Args:
        site_dir: Site directory with ``mss/``, ``s2/`` and ``metadata.json``.

    Returns:
        Dict with ``metadata``, ``mss`` ``(4, 250, 250)`` TOA reflectance,
        ``s2`` ``(3, 1500, 1500)`` reflectance in OCM order, ``qa_pixel`` and
        ``saturated``.
    """
    import rasterio

    metadata = json.loads((site_dir / "metadata.json").read_text())
    mss = np.stack([read_band(site_dir / "mss", f"B{band}") for band in range(1, 5)])
    s2 = np.stack([read_band(site_dir / "s2", band) / 10000.0 for band in OCM_PICS_BANDS])
    qa_pixel = read_band(site_dir / "mss", "QA_PIXEL", dtype=np.uint16)
    saturated = read_band(site_dir / "mss", "MSK_SATURATED", dtype=np.uint8).astype(bool)
    with rasterio.open(sorted((site_dir / "mss").glob("*.B1.tif"))[0]) as source:
        transform, crs = source.transform, source.crs
    return {
        "metadata": metadata,
        "mss": mss,
        "s2": s2,
        "qa_pixel": qa_pixel,
        "saturated": saturated,
        "transform": transform,
        "crs": crs,
    }


def mss_valid_mask(site: dict) -> np.ndarray:
    """Usable MSS pixels: finite, positive, below 1 and not saturated."""
    mss = site["mss"]
    valid = np.all(np.isfinite(mss), axis=0) & (np.min(mss, axis=0) > 0.0)
    valid &= (mss < 1.0).all(axis=0)
    return valid & ~site["saturated"]


def binary_iou(pred: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    """IoU of two boolean masks over the valid pixels."""
    pred = np.asarray(pred, dtype=bool) & mask
    reference = np.asarray(reference, dtype=bool) & mask
    union = np.count_nonzero(pred | reference)
    if union == 0:
        return float("nan")
    return float(np.count_nonzero(pred & reference) / union)


def class_fractions(mask: np.ndarray, mask_valid: np.ndarray) -> dict[str, float]:
    """Fraction of the valid pixels in every class."""
    values = np.asarray(mask)[mask_valid]
    total = max(values.size, 1)
    fractions = {name: float(np.count_nonzero(values == code) / total)
                 for code, name in enumerate(CLASS_NAMES)}
    fractions["contaminated"] = (
        fractions["thick_cloud"] + fractions["thin_cloud"] + fractions["cloud_shadow"]
    )
    return fractions


def evaluate_pics_site(
    site: dict, model_masks: dict[str, np.ndarray], ocm60: np.ndarray
) -> dict:
    """Predictions and diagnostics for one real PICS and every model.

    The site is declared cloud-free (CEOS PICS, QA), so any cloud prediction is
    a false alarm.  There are no human labels, therefore no accuracy is
    claimed: only fractions, model/OCM agreement and agreement with the
    Landsat QA_PIXEL cloud bit (a weak, non-human reference).

    Args:
        site: Output of :func:`load_pics_site`.
        model_masks: ``{model_name: (250, 250) classes}`` at 60 m.
        ocm60: ``(250, 250)`` OCM classes aggregated to 60 m.

    Returns:
        One row of the PICS table.
    """
    valid = mss_valid_mask(site)
    qa_cloud = (site["qa_pixel"] & QA_PIXEL_CLOUD) != 0
    qa_binary = qa_cloud & valid
    ocm_fractions = class_fractions(ocm60, valid)
    ocm_binary = ocm60 != 0

    models = {}
    for name, mask in model_masks.items():
        binary = mask != 0
        models[name] = {
            "fractions": class_fractions(mask, valid),
            "vs_ocm": {
                "pixel_agreement": float(
                    np.count_nonzero((mask == ocm60) & valid) / max(valid.sum(), 1)
                ),
                "binary_iou": binary_iou(binary, ocm_binary, valid),
                "cohen_kappa": cohen_kappa(
                    confusion_matrix(mask, ocm60, ignore_mask=~valid)
                ),
            },
            "qa_pixel_iou": binary_iou(binary, qa_binary, valid),
        }
    return {
        "site": site["metadata"]["site"],
        "label": site["metadata"].get("label", site["metadata"]["site"]),
        "mss_date": site["metadata"]["mss"].get("date"),
        "sun_elevation_deg": site["metadata"]["mss"].get("sun_elevation"),
        "n_valid_px": int(valid.sum()),
        "models": models,
        "ocm": {
            "fractions": ocm_fractions,
            "qa_pixel_iou": binary_iou(ocm_binary, qa_binary, valid),
        },
        "qa_pixel_reference": {
            "cloud_fraction": float(np.count_nonzero(qa_binary) / max(valid.sum(), 1)),
        },
    }


# ── Tables ────────────────────────────────────────────────────────────────
def write_per_class_csv(path: Path, results: dict[str, dict]) -> None:
    """Write the per-class table of several models."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "class", "support", "precision", "recall", "f1", "iou"])
        for model_name, result in results.items():
            for class_name in CLASS_NAMES:
                scores = result["per_class"][class_name]
                writer.writerow(
                    [model_name, class_name,
                     *[f"{scores[key]:.6f}"
                       for key in ("support", "precision", "recall", "f1", "iou")]]
                )


def write_per_class_markdown(path: Path, results: dict[str, dict]) -> None:
    """Markdown version of the per-class table."""
    lines = [
        "| Modelo | Clase | Soporte | Precisión | Recall | F1 | IoU |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name, result in results.items():
        for class_name in CLASS_NAMES:
            scores = result["per_class"][class_name]
            lines.append(
                f"| {model_name} | {PRETTY[class_name]} | {int(scores['support']):,} | "
                f"{scores['precision']:.4f} | {scores['recall']:.4f} | "
                f"{scores['f1']:.4f} | {scores['iou']:.4f} |"
            )
        lines.append(
            f"| {model_name} | **mIoU / OA / kappa** | | | | "
            f"**{result['mean_iou']:.4f} / {result['overall_accuracy']:.4f} / "
            f"{result['cohen_kappa']:.4f}** | |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_pics_csv(path: Path, rows: list[dict], model_names: list[str]) -> None:
    """Write one row per PICS with every model and OCM fractions."""
    fields = ["site", "label", "mss_date", "sun_elevation_deg", "n_valid_px"]
    for name in model_names:
        fields += [f"{name}_{key}" for key in
                   ("clear", "thick", "thin", "shadow", "contaminated")]
    fields += [f"ocm_{key}" for key in ("clear", "thick", "thin", "shadow", "contaminated")]
    for name in model_names:
        fields += [f"{name}_agreement_ocm", f"{name}_iou_ocm", f"{name}_kappa_ocm"]
    fields += ["qa_cloud_fraction"]
    fields += [f"{name}_iou_qa" for name in model_names]
    fields.append("ocm_iou_qa")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for row in rows:
            values = [row["site"], row["label"], row["mss_date"],
                      f"{row['sun_elevation_deg']:.3f}", row["n_valid_px"]]
            for name in model_names:
                fractions = row["models"][name]["fractions"]
                values += [f"{fractions[key]:.4f}" for key in
                           ("clear", "thick_cloud", "thin_cloud", "cloud_shadow",
                            "contaminated")]
            fractions = row["ocm"]["fractions"]
            values += [f"{fractions[key]:.4f}" for key in
                       ("clear", "thick_cloud", "thin_cloud", "cloud_shadow", "contaminated")]
            for name in model_names:
                vs_ocm = row["models"][name]["vs_ocm"]
                values += [f"{vs_ocm['pixel_agreement']:.4f}",
                           f"{vs_ocm['binary_iou']:.4f}",
                           f"{vs_ocm['cohen_kappa']:.4f}"]
            values.append(f"{row['qa_pixel_reference']['cloud_fraction']:.4f}")
            for name in model_names:
                values.append(f"{row['models'][name]['qa_pixel_iou']:.4f}")
            values.append(f"{row['ocm']['qa_pixel_iou']:.4f}")
            writer.writerow(values)


def write_pics_markdown(path: Path, rows: list[dict], model_names: list[str]) -> None:
    """Markdown table of the PICS diagnostics."""
    header = "| Sitio | Fecha MSS | " + " | ".join(
        f"{name} nube+sombra" for name in model_names
    ) + " | OCM nube+sombra | " + " | ".join(
        f"Acuerdo {name}-OCM" for name in model_names
    ) + " | IoU nube vs QA (" + "/".join(model_names) + "/OCM) |"
    lines = [header, "|---" * (5 + 2 * len(model_names)) + "|"]
    for row in rows:
        model_cells = " | ".join(
            f"{100 * row['models'][name]['fractions']['contaminated']:.2f} %"
            for name in model_names
        )
        agreement_cells = " | ".join(
            f"{row['models'][name]['vs_ocm']['pixel_agreement']:.3f}"
            for name in model_names
        )
        qa_cells = "/".join(
            f"{row['models'][name]['qa_pixel_iou']:.3f}" for name in model_names
        )
        lines.append(
            f"| {row['label']} | {row['mss_date']} | {model_cells} | "
            f"{100 * row['ocm']['fractions']['contaminated']:.2f} % | "
            f"{agreement_cells} | {qa_cells}/{row['ocm']['qa_pixel_iou']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


# ── Figures ───────────────────────────────────────────────────────────────
def false_colour(image: np.ndarray, bands: tuple[int, int, int] = (3, 1, 0)) -> np.ndarray:
    """Percentile-stretched false colour from a 4-band reflectance array."""
    rgb = np.stack([image[index] for index in bands], axis=-1).astype(np.float32)
    low, high = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    return np.clip((rgb - low) / max(high - low, 1e-6), 0, 1)


def legend_handles() -> list[Patch]:
    """Patch handles for the four classes."""
    return [Patch(facecolor=color, edgecolor="#999999", label=PRETTY[name])
            for name, color in zip(CLASS_NAMES, CLASS_COLORS, strict=True)]


def figure_pseudo_test(
    path: Path,
    chips: list[dict[str, str]],
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> list[dict]:
    """Qualitative figure of the pseudo test: six regimes and error maps.

    Returns:
        The rows drawn (chip id, regime, per-chip mIoU) for the report.
    """
    test_index = list(range(len(chips)))
    per_chip_iou = []
    for index in test_index:
        matrix = confusion_matrix(predictions[index], labels[index])
        iou = iou_per_class(matrix)
        per_chip_iou.append((float(np.nanmean(list(iou.values()))), index))

    picked: list[tuple[str, int]] = []
    for regime in ("clear", "low_cloud", "mid_cloud", "high_cloud"):
        candidates = [index for index, chip in enumerate(chips) if chip["regime"] == regime]
        if candidates:
            best = max(candidates, key=lambda index: per_chip_iou[index][0])
            picked.append((regime, best))
    shadow_candidates = [index for index, chip in enumerate(chips) if chip["shadow_rich"] == "1"]
    if shadow_candidates:
        picked.append(("shadow-rich", shadow_candidates[0]))
    worst = min(per_chip_iou)[1]
    picked.append(("worst", worst))

    fig, axes = plt.subplots(
        len(picked), 4, figsize=(13.0, 3.1 * len(picked)), constrained_layout=True
    )
    columns = ["MSS false colour", "Reference", "Prediction", "Error"]
    rows_out = []
    for row, (regime, index) in enumerate(picked):
        chip = chips[index]
        iou = per_chip_iou[index][0]
        image = np.asarray(images[index], dtype=np.float32) / 10000.0
        axes[row, 0].imshow(false_colour(image))
        axes[row, 0].set_ylabel(f"{chip['chip_id'].split('__')[-1]}\n{regime}", fontsize=8)
        axes[row, 1].imshow(labels[index], cmap=CLASS_CMAP, vmin=0, vmax=3,
                            interpolation="nearest")
        axes[row, 2].imshow(predictions[index], cmap=CLASS_CMAP, vmin=0, vmax=3,
                            interpolation="nearest")
        axes[row, 3].imshow((predictions[index] != labels[index]).astype(np.uint8),
                            cmap=ERROR_CMAP, vmin=0, vmax=1, interpolation="nearest")
        axes[row, 3].set_title(f"mIoU {iou:.3f}", fontsize=8)
        for column in range(4):
            if row == 0:
                axes[row, column].set_title(columns[column], fontsize=9)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        rows_out.append(
            {
                "chip_id": chip["chip_id"],
                "regime": regime,
                "panel": regime,
                "miou": iou,
                "clear_frac": float(chip["clear_frac"]),
                "shadow_frac": float(chip["shadow_frac"]),
            }
        )
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=4, fontsize=8,
               frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return rows_out


def figure_pics(
    path: Path,
    sites: list[dict],
    model_masks: dict[str, list[np.ndarray]],
    ocm_masks: list[np.ndarray],
    model_probabilities: dict[str, list[np.ndarray]],
    rows: list[dict],
    main_model: str,
) -> None:
    """Five PICS: false colour, main probability, masks and disagreement."""
    names = [main_model, *[name for name in model_masks if name != main_model][:1]]
    columns = ["MSS false colour", f"P({main_model} contaminated)"] + [
        f"{name} mask" for name in names
    ] + ["OCM mask", f"{main_model} vs OCM"]
    fig, axes = plt.subplots(
        len(sites), len(columns), figsize=(3.0 * len(columns), 3.2 * len(sites)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row, site in enumerate(sites):
        axes[row, 0].imshow(false_colour(site["mss"]))
        axes[row, 0].set_ylabel(
            f"{rows[row]['label']}\n{main_model} "
            f"{100 * rows[row]['models'][main_model]['fractions']['contaminated']:.1f} %",
            fontsize=8,
        )
        probability = model_probabilities[main_model][row]
        image = axes[row, 1].imshow(1.0 - probability[0], cmap="magma", vmin=0, vmax=1)
        for column, name in enumerate(names, start=2):
            axes[row, column].imshow(model_masks[name][row], cmap=CLASS_CMAP, vmin=0, vmax=3,
                                     interpolation="nearest")
        axes[row, len(columns) - 2].imshow(ocm_masks[row], cmap=CLASS_CMAP, vmin=0, vmax=3,
                                           interpolation="nearest")
        disagreement = (model_masks[main_model][row] != ocm_masks[row]).astype(np.uint8)
        axes[row, len(columns) - 1].imshow(disagreement, cmap=ERROR_CMAP, vmin=0, vmax=1,
                                           interpolation="nearest")
        axes[row, len(columns) - 1].set_title(
            f"disagree {100 * disagreement.mean():.1f} %", fontsize=8
        )
        for column in range(len(columns)):
            if row == 0:
                axes[row, column].set_title(columns[column], fontsize=9)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.colorbar(image, ax=[axes[row, 1] for row in range(len(sites))], shrink=0.6,
                 label="P(cloud or shadow)")
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=4, fontsize=8,
               frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_confusion(path: Path, results: dict[str, dict]) -> None:
    """Normalised confusion matrices of every model and OCM on the pseudo test."""
    fig, axes = plt.subplots(1, len(results), figsize=(6.6 * len(results), 5.6),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, (name, result) in zip(axes, results.items(), strict=True):
        matrix = np.asarray(result["confusion_matrix"], dtype=np.float64)
        normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        image = axis.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
        axis.set_xticks(range(4), [PRETTY[item] for item in CLASS_NAMES], rotation=30,
                        ha="right", fontsize=8.5)
        axis.set_yticks(range(4), [PRETTY[item] for item in CLASS_NAMES], fontsize=8.5)
        axis.set_xlabel("Prediction", fontsize=9)
        axis.set_ylabel("Reference", fontsize=9)
        axis.set_title(f"{name} — mIoU {result['mean_iou']:.3f}", fontsize=10.5)
        for i in range(4):
            for j in range(4):
                axis.text(j, i, f"{100 * normalised[i, j]:.0f}", ha="center", va="center",
                          fontsize=8,
                          color="white" if normalised[i, j] > 0.55 else "#222222")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03, label="Row fraction")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_class_iou(path: Path, results: dict[str, dict]) -> None:
    """Grouped bars of the per-class IoU of every model.

    This is the figure that shows where the gap is: the thin-cloud bar of the
    MSS models against the OmniCloudMask one, with the mean IoU in the legend.

    Args:
        path: Destination PNG.
        results: ``{model_name: evaluate() result}`` in plotting order.
    """
    names = list(results)
    positions = np.arange(len(CLASS_NAMES), dtype=np.float64)
    width = 0.8 / max(len(names), 1)
    palette = ["#9ecae1", "#2171b5", "#6baed6", "#fdae6b", "#756bb1", "#31a354"]
    fig, axis = plt.subplots(figsize=(8.0, 3.8), constrained_layout=True)
    for index, name in enumerate(names):
        values = [results[name]["per_class"][cls]["iou"] for cls in CLASS_NAMES]
        offset = (index - (len(names) - 1) / 2) * width
        bars = axis.bar(
            positions + offset, values, width,
            color=palette[index % len(palette)], edgecolor="#333333", linewidth=0.4,
            label=f"{name} (mIoU {results[name]['mean_iou']:.3f})",
        )
        for bar, value in zip(bars, values, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                      f"{value:.2f}", ha="center", va="bottom", fontsize=6.5)
    axis.set_xticks(positions, [PRETTY[cls] for cls in CLASS_NAMES], fontsize=9)
    axis.set_ylabel("IoU", fontsize=9)
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.3, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper center",
                bbox_to_anchor=(0.5, -0.14))
    axis.set_title("Pseudo-MSS test: IoU per class", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Model evaluation ──────────────────────────────────────────────────────
def choose_scales(
    model,
    checkpoint,
    images: np.ndarray,
    labels: np.ndarray,
    device: str,
    batch_size: int,
    candidates: tuple[tuple[float, ...], ...] = SCALE_CANDIDATES,
) -> dict:
    """Pick the scale set with the best validation mIoU.

    The choice is made on the validation split only; the test split is not
    involved.

    Args:
        model: Loaded model.
        checkpoint: Its :class:`LoadedCheckpoint` (for the band statistics).
        images: ``(N, 4, H, W)`` validation reflectance, 0-1.
        labels: ``(N, H, W)`` validation classes.
        device: Torch device.
        batch_size: Images per pass.
        candidates: Scale sets to try.

    Returns:
        Dict with ``scales``, ``val_mean_iou`` per candidate and ``chosen``.
    """
    scores = {}
    for scales in candidates:
        probabilities = predict_proba_multiscale(
            model, images, checkpoint.band_mean, checkpoint.band_std,
            scales=scales, dihedral=True, batch_size=batch_size, device=device,
        )
        matrix = confusion_matrix(probabilities.argmax(axis=1), labels)
        scores[",".join(f"{scale:g}" for scale in scales)] = {
            "scales": list(scales),
            "mean_iou": mean_iou(matrix),
        }
    chosen = max(scores.values(), key=lambda item: item["mean_iou"])
    return {"candidates": scores, "scales": chosen["scales"],
            "val_mean_iou": chosen["mean_iou"]}


def evaluate_checkpoint(
    name: str,
    path: Path,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    device: str,
    batch_size: int,
    scales_mode: str = "auto",
    bias_mode: str = "auto",
) -> dict:
    """Predictions, metrics and decision rule of one checkpoint.

    Args:
        name: Label used in the tables.
        path: Checkpoint file.
        test_images: ``(N, 4, H, W)`` test reflectance, 0-1.
        test_labels: ``(N, H, W)`` test classes.
        val_images: ``(N, 4, H, W)`` validation reflectance.
        val_labels: ``(N, H, W)`` validation classes.
        device: Torch device.
        batch_size: Images per pass.
        scales_mode: ``auto`` (pick on validation), ``none`` (1.0) or a
            comma-separated list such as ``1.0,1.5``.
        bias_mode: ``auto`` (checkpoint metadata, tuned on validation when
            absent), ``none`` or ``val`` (always tune on validation).

    Returns:
        Dict with the model, checkpoint info, probabilities, masks and metrics.
    """
    model, checkpoint = load_model(path, device=device)
    print(f"[{name}] checkpoint {path} sha256 {checkpoint.sha256[:16]}")

    scale_report = None
    if scales_mode == "auto":
        scale_report = choose_scales(
            model, checkpoint, val_images, val_labels, device, batch_size
        )
        scales = tuple(scale_report["scales"])
    elif scales_mode == "none":
        scales = (1.0,)
    else:
        scales = tuple(float(value) for value in scales_mode.split(","))
    print(f"[{name}] scales {scales}")

    probabilities = predict_proba_multiscale(
        model, test_images, checkpoint.band_mean, checkpoint.band_std,
        scales=scales, dihedral=True, batch_size=batch_size, device=device,
    )
    argmax_mask = probabilities.argmax(axis=1).astype(np.uint8)
    argmax_result = evaluate(argmax_mask, test_labels)

    stored_bias = checkpoint.metadata.get("class_bias")
    if bias_mode == "none":
        bias = [0.0] * len(CLASS_NAMES)
        bias_source = "argument"
    elif bias_mode == "auto" and stored_bias is not None:
        bias = list(stored_bias)
        bias_source = "checkpoint"
    else:
        val_probabilities = predict_proba_multiscale(
            model, val_images, checkpoint.band_mean, checkpoint.band_std,
            scales=scales, dihedral=True, batch_size=batch_size, device=device,
        )
        tuned = tune_class_bias(val_probabilities, val_labels)
        bias = tuned["bias"]
        bias_source = "validation"
        print(
            f"[{name}] bias tuned on validation {np.round(bias, 2).tolist()} → "
            f"val mIoU {tuned['mean_iou']:.4f} (argmax {tuned['baseline_mean_iou']:.4f})"
        )

    bias_mask = apply_class_bias(probabilities, bias)
    bias_result = evaluate(bias_mask, test_labels)
    print(
        f"[{name}] test mIoU argmax {argmax_result['mean_iou']:.4f} | "
        f"bias {bias_result['mean_iou']:.4f} | OA {bias_result['overall_accuracy']:.4f} | "
        + ", ".join(f"{cls} {value:.3f}"
                    for cls, value in iou_per_class(bias_result["confusion_matrix"]).items())
    )
    return {
        "name": name,
        "path": str(path),
        "sha256": checkpoint.sha256,
        "metadata": checkpoint.metadata,
        "scales": list(scales),
        "scale_report": scale_report,
        "class_bias": [float(value) for value in bias],
        "class_bias_source": bias_source,
        "probabilities": probabilities,
        "argmax_mask": argmax_mask,
        "bias_mask": bias_mask,
        "argmax": argmax_result,
        "metrics": bias_result,
        "model": model,
        "checkpoint": checkpoint,
    }


def prediction_for(
    evaluation: dict,
    images: np.ndarray,
    scales: tuple[float, ...],
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Probabilities of one evaluated model or ensemble on ``images``.

    Args:
        evaluation: Entry of the ``evaluations`` dict.
        images: ``(N, 4, H, W)`` or ``(4, H, W)`` reflectance, 0-1.
        scales: Scale set chosen for that model.
        batch_size: Images per pass.
        device: Torch device.

    Returns:
        Probabilities with the class axis first.
    """
    members = evaluation.get("member_models")
    if not members:
        return predict_proba_multiscale(
            evaluation["model"], images, evaluation["checkpoint"].band_mean,
            evaluation["checkpoint"].band_std, scales=scales, dihedral=True,
            batch_size=batch_size, device=device,
        )
    accumulated = None
    for model, checkpoint in members:
        probabilities = predict_proba_multiscale(
            model, images, checkpoint.band_mean, checkpoint.band_std, scales=scales,
            dihedral=True, batch_size=batch_size, device=device,
        )
        accumulated = probabilities if accumulated is None else accumulated + probabilities
    average = accumulated / len(members)
    return average / np.maximum(average.sum(axis=-3, keepdims=True), 1e-8)


def evaluate_ensemble(
    name: str,
    paths: list[Path],
    test_images: np.ndarray,
    test_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    device: str,
    batch_size: int,
    scales_mode: str = "auto",
    bias_mode: str = "none",
) -> dict:
    """Average the softmax of several checkpoints and score the result.

    The members are trained on the same fixture and split but differ in loss,
    capacity or seed, so their errors are partly independent.  The scale set
    and the optional per-class bias are chosen on the validation split, never
    on the test split.

    Args:
        name: Label used in the tables.
        paths: Checkpoint files to average.
        test_images: ``(N, 4, H, W)`` test reflectance, 0-1.
        test_labels: ``(N, H, W)`` test classes.
        val_images: ``(N, 4, H, W)`` validation reflectance.
        val_labels: ``(N, H, W)`` validation classes.
        device: Torch device.
        batch_size: Images per pass.
        scales_mode: ``auto``, ``none`` or a comma-separated list.
        bias_mode: ``auto``/``val`` tune on validation, ``none`` disables it.

    Returns:
        Same shape as :func:`evaluate_checkpoint`.
    """
    if len(paths) < 2:
        raise ValueError("an ensemble needs at least two checkpoints")
    members = [load_model(path, device=device) for path in paths]
    print(f"[{name}] ensemble of {len(paths)} checkpoints")
    for path, (_, checkpoint) in zip(paths, members, strict=True):
        print(f"[{name}]   {path} sha256 {checkpoint.sha256[:16]}")

    def averaged(images: np.ndarray, scales: tuple[float, ...]) -> np.ndarray:
        accumulated = None
        for model, checkpoint in members:
            probabilities = predict_proba_multiscale(
                model, images, checkpoint.band_mean, checkpoint.band_std,
                scales=scales, dihedral=True, batch_size=batch_size, device=device,
            )
            accumulated = probabilities if accumulated is None else accumulated + probabilities
        average = accumulated / len(members)
        return average / np.maximum(average.sum(axis=1, keepdims=True), 1e-8)

    scale_report = None
    if scales_mode == "auto":
        scores = {}
        for scales in SCALE_CANDIDATES:
            matrix = confusion_matrix(
                averaged(val_images, scales).argmax(axis=1), val_labels
            )
            scores[",".join(f"{scale:g}" for scale in scales)] = {
                "scales": list(scales), "mean_iou": mean_iou(matrix),
            }
        chosen = max(scores.values(), key=lambda item: item["mean_iou"])
        scales = tuple(chosen["scales"])
        scale_report = {"candidates": scores, "scales": chosen["scales"],
                        "val_mean_iou": chosen["mean_iou"]}
    elif scales_mode == "none":
        scales = (1.0,)
    else:
        scales = tuple(float(value) for value in scales_mode.split(","))
    print(f"[{name}] scales {scales}")

    probabilities = averaged(test_images, scales)
    argmax_mask = probabilities.argmax(axis=1).astype(np.uint8)
    argmax_result = evaluate(argmax_mask, test_labels)
    if bias_mode == "none":
        bias = [0.0] * len(CLASS_NAMES)
        bias_source = "argument"
    else:
        val_probabilities = averaged(val_images, scales)
        tuned = tune_class_bias(val_probabilities, val_labels)
        bias = tuned["bias"]
        bias_source = "validation"
        print(
            f"[{name}] bias tuned on validation {np.round(bias, 2).tolist()} → "
            f"val mIoU {tuned['mean_iou']:.4f} (argmax {tuned['baseline_mean_iou']:.4f})"
        )
    bias_mask = apply_class_bias(probabilities, bias)
    bias_result = evaluate(bias_mask, test_labels)
    print(
        f"[{name}] test mIoU argmax {argmax_result['mean_iou']:.4f} | "
        f"bias {bias_result['mean_iou']:.4f} | OA {bias_result['overall_accuracy']:.4f} | "
        + ", ".join(f"{cls} {value:.3f}"
                    for cls, value in iou_per_class(bias_result["confusion_matrix"]).items())
    )
    return {
        "name": name,
        "path": [str(path) for path in paths],
        "sha256": [checkpoint.sha256 for _, checkpoint in members],
        "metadata": {"members": len(paths)},
        "scales": list(scales),
        "scale_report": scale_report,
        "class_bias": [float(value) for value in bias],
        "class_bias_source": bias_source,
        "probabilities": probabilities,
        "argmax_mask": argmax_mask,
        "bias_mask": bias_mask,
        "argmax": argmax_result,
        "metrics": bias_result,
        "model": members[0][0],
        "checkpoint": members[0][1],
        "member_models": members,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-name", default="g4b")
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pics", type=Path, default=DEFAULT_PICS)
    parser.add_argument("--cloudsen12", type=Path, default=DEFAULT_CLOUDSEN12)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--verify", type=int, default=3, help="fixture chips to re-emulate")
    parser.add_argument("--scales", default="auto",
                        help="auto, none or a comma-separated list like 1.0,1.5")
    parser.add_argument("--bias", choices=("auto", "none", "val"), default="auto")
    parser.add_argument(
        "--ensemble",
        action="append",
        default=[],
        metavar="NAME=PATH1,PATH2",
        help="average several checkpoints; may be repeated",
    )
    parser.add_argument("--reuse-ocm", action="store_true")
    parser.add_argument("--skip-ocm", action="store_true")
    parser.add_argument("--skip-pics", action="store_true")
    args = parser.parse_args()

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    out = args.out
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(args.fixture)
    images = np.load(args.fixture / "images.npy", mmap_mode="r")
    labels = np.load(args.fixture / "labels.npy", mmap_mode="r")
    test_chips = [row for row in manifest if row["split"] == "test"]
    test_index = [index for index, row in enumerate(manifest) if row["split"] == "test"]
    val_index = [index for index, row in enumerate(manifest) if row["split"] == "validation"]
    test_images = np.asarray(images[test_index], dtype=np.float32) / 10000.0
    test_labels = np.asarray(labels[test_index], dtype=np.uint8)
    val_images = np.asarray(images[val_index], dtype=np.float32) / 10000.0
    val_labels = np.asarray(labels[val_index]).astype(np.uint8)
    print(
        f"{len(test_chips)} test chips, {len(val_index)} validation chips, device={device}"
    )

    verification = {"checked": 0, "matched": 0, "chips": []}
    if args.verify > 0:
        fixture_seed = int(json.loads((args.fixture / "fixture.json").read_text())["seed"])
        verification = verify_fixture_chips(
            test_chips[: args.verify],
            args.fixture / "images.npy",
            args.fixture / "labels.npy",
            args.cloudsen12,
            fixture_seed,
        )
        print(f"fixture hash verification: {verification['matched']}/{verification['checked']}")

    evaluations = {}
    if not args.skip_baseline and Path(args.baseline_checkpoint).exists():
        evaluations["g4"] = evaluate_checkpoint(
            "g4", args.baseline_checkpoint, test_images, test_labels, val_images,
            val_labels, device, args.batch_size, args.scales, args.bias,
        )
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    evaluations[args.checkpoint_name] = evaluate_checkpoint(
        args.checkpoint_name, args.checkpoint, test_images, test_labels, val_images,
        val_labels, device, args.batch_size, args.scales, args.bias,
    )

    for specification in args.ensemble:
        if "=" not in specification:
            raise ValueError(f"--ensemble needs NAME=PATH1,PATH2, got {specification}")
        name, raw_paths = specification.split("=", 1)
        paths = [Path(item) for item in raw_paths.split(",") if item]
        evaluations[name] = evaluate_ensemble(
            name, paths, test_images, test_labels, val_images, val_labels, device,
            args.batch_size, args.scales, "none" if args.bias == "none" else "val",
        )

    results: dict[str, dict] = {}
    for name, evaluation in evaluations.items():
        results[f"{name}_argmax"] = evaluation["argmax"]
        if any(evaluation["class_bias"]):
            results[f"{name}_bias"] = evaluation["metrics"]

    for name, evaluation in evaluations.items():
        np.savez_compressed(
            out / f"confusion_{name}_pseudo.npz",
            matrix=np.asarray(evaluation["metrics"]["confusion_matrix"], dtype=np.int64),
            argmax=np.asarray(evaluation["argmax"]["confusion_matrix"], dtype=np.int64),
        )

    per_regime = {}
    main_name = args.checkpoint_name
    main_evaluation = evaluations[main_name]
    main_predictions = main_evaluation["bias_mask"]
    for regime in ("clear", "low_cloud", "mid_cloud", "high_cloud"):
        indices = [position for position, chip in enumerate(test_chips) if chip["regime"] == regime]
        if not indices:
            continue
        matrix = confusion_matrix(main_predictions[indices], test_labels[indices])
        per_regime[regime] = {
            "n_chips": len(indices),
            "mean_iou": mean_iou(matrix),
            "overall_accuracy": overall_accuracy(matrix),
            "iou": iou_per_class(matrix),
        }

    ocm_result = None
    ocm_crosstab = None
    ocm_predictions = None
    if not args.skip_ocm:
        ocm_predictions, ocm_crosstab = ocm_pseudo_test(
            test_chips, args.cloudsen12, out / "ocm_pseudo_test.npz", device, args.reuse_ocm
        )
        ocm_result = evaluate(ocm_predictions, test_labels)
        np.savez_compressed(out / "confusion_ocm_pseudo.npz",
                            matrix=np.asarray(ocm_result["confusion_matrix"], dtype=np.int64))
        results["ocm"] = ocm_result
        print(
            f"[ocm] test mIoU {ocm_result['mean_iou']:.4f} | OA "
            f"{ocm_result['overall_accuracy']:.4f} | "
            + ", ".join(f"{name} {value:.3f}" for name, value in
                        iou_per_class(ocm_result["confusion_matrix"]).items())
        )
        print("OCM 10 m cross-tabulation (rows reference, columns OCM):")
        print(np.asarray(ocm_crosstab))

    write_per_class_csv(out / "metrics_pseudo_test.csv", results)
    write_per_class_markdown(out / "metrics_pseudo_test.md", results)

    figure_pseudo_test(
        figures / "fig_pseudo_predictions.png",
        test_chips,
        np.asarray(images[test_index]),
        test_labels,
        main_predictions,
    )
    confusion_panels = {
        key: results[key]
        for key in ("g4_argmax", f"{main_name}_argmax", f"{main_name}_bias", "ocm")
        if key in results
    }
    if f"{main_name}_bias" not in confusion_panels:
        confusion_panels[f"{main_name}_bias"] = results[f"{main_name}_bias"] if (
            f"{main_name}_bias" in results
        ) else results[f"{main_name}_argmax"]
    figure_confusion(figures / "fig_confusion_pseudo.png", confusion_panels)
    iou_panels = {
        key: results[key]
        for key in ("g4_argmax", f"{main_name}_argmax", "ocm")
        if key in results
    }
    figure_class_iou(figures / "fig_class_iou_pseudo.png", iou_panels)

    pics_rows: list[dict] = []
    if not args.skip_pics:
        site_dirs = [path for path in sorted(args.pics.iterdir())
                     if (path / "metadata.json").exists()]
        model_masks: dict[str, list[np.ndarray]] = {name: [] for name in evaluations}
        probability_maps: dict[str, list[np.ndarray]] = {name: [] for name in evaluations}
        ocm_masks: list[np.ndarray] = []
        sites: list[dict] = []
        for site_dir in site_dirs:
            site = load_pics_site(site_dir)
            masks = {}
            for name, evaluation in evaluations.items():
                probabilities = prediction_for(
                    evaluation, site["mss"], tuple(evaluation["scales"]),
                    args.batch_size, device,
                )
                mask = apply_class_bias(probabilities, evaluation["class_bias"])
                masks[name] = mask.astype(np.uint8)
                model_masks[name].append(masks[name])
                probability_maps[name].append(probabilities)
            ocm60 = aggregate_cloudsen12(ocm_predict(site["s2"], device),
                                         scale_factor=SCALE_FACTOR)
            ocm_masks.append(ocm60)
            row = evaluate_pics_site(site, masks, ocm60)
            pics_rows.append(row)
            sites.append(site)
            summary = ", ".join(
                f"{name} {100 * row['models'][name]['fractions']['contaminated']:.2f} %"
                for name in evaluations
            )
            print(
                f"  {row['site']}: {summary}, OCM "
                f"{100 * row['ocm']['fractions']['contaminated']:.2f} %"
            )
        write_pics_csv(out / "metrics_pics.csv", pics_rows, list(evaluations))
        write_pics_markdown(out / "metrics_pics.md", pics_rows, list(evaluations))
        figure_pics(figures / "fig_pics_predictions.png", sites, model_masks, ocm_masks,
                    probability_maps, pics_rows, main_name)

    human = human_labels_state(args.labels)
    run = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": time.perf_counter() - t0,
        "inference": {
            "tta": "dihedral (8)",
            "scales_mode": args.scales,
            "bias_mode": args.bias,
        },
        "checkpoints": {
            name: {
                "path": evaluation["path"],
                "sha256": evaluation["sha256"],
                "members": len(evaluation.get("member_models", [])),
                "scales": evaluation["scales"],
                "scale_report": evaluation["scale_report"],
                "class_bias": evaluation["class_bias"],
                "class_bias_source": evaluation["class_bias_source"],
                "metadata": evaluation["metadata"],
            }
            for name, evaluation in evaluations.items()
        },
        "fixture": {
            "path": str(args.fixture),
            "images_sha256": json.loads((args.fixture / "fixture.json").read_text()).get(
                "images_sha256"
            ),
            "n_test_chips": len(test_chips),
            "n_validation_chips": len(val_index),
        },
        "verification": verification,
        "pseudo_test": {
            "model": main_evaluation["metrics"],
            "model_argmax": main_evaluation["argmax"],
            "ocm": ocm_result,
            "per_regime_model": per_regime,
            "ocm_version": ocm_version() if ocm_result is not None else None,
            "ocm_10m_crosstab": None if ocm_crosstab is None
            else np.asarray(ocm_crosstab).tolist(),
        },
        "pics": {"human_labels": human, "sites": pics_rows},
        "figures": sorted(path.name for path in figures.glob("*.png")),
    }
    (out / "run_eval.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"wrote {out / 'run_eval.json'} ({run['elapsed_s']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
