#!/usr/bin/env python
"""Fair cloud/shadow benchmark on Landsat MSS: OmniCloudMask vs the MSS models.

Why this script exists
----------------------
The G4/G4b report compares OmniCloudMask (OCM) at **0.722 mIoU** with the
U-Net at **0.518-0.532**.  That comparison is not like for like: OCM was fed
the Sentinel-2 chips at 10 m (B04, B03, B8A) and only then aggregated to 60 m,
while the U-Nets were fed the emulated MSS at 60 m.  OCM saw six times more
spatial detail than the model it was compared with.

This script puts every model on the *same* test:

* reference: the CloudSEN12 test split, labels aggregated 10 m -> 60 m with
  ``aggregate_cloudsen12`` (the same 194 chips and the same labels as G4/G4b);
* grid: the 60 m labels;
* metrics: one implementation, the same confusion matrix for everybody.

Models
------
``ocm_s2_10m``      OmniCloudMask on B04/B03/B8A at 10 m, mask aggregated to 60 m.
                    This is the published 0.722 number; kept as an upper bound.
``ocm_mss_60m``     OmniCloudMask on the MSS at 60 m (red=MSS2, green=MSS1,
                    NIR=MSS4).  Same input information as the MSS models.
``ocm_mss_nir1``    Same with MSS3 as NIR (a robustness check of the band choice).
``ocm_mss_upsampled`` OCM on the 60 m MSS resampled to 10 m.  Documented because
                    it is the obvious "give OCM its native GSD" idea and it
                    does not work: interpolated 60 m pixels look like cloud.
``g4``, ``g4b``, ``g4b_ens``  the satharmony U-Nets, argmax + dihedral TTA,
                    exactly the decision rule of ``evaluate_cloud_baseline.py``.
``qa4eo2``          the published QA4EO-2 checkpoint (3 classes: clear, cloud,
                    shadow; 4 MSS bands).
``finetune``        the best fine-tune of QA4EO-2 on the v7 labels (3 classes),
                    with and without dihedral TTA.

The old checkpoints are also re-scored under two alternative input conventions,
because their training code and the published inference code disagree:

``legacy_radiometry``  the fixture mapped to the radiometry of the 0.1.13
                    emulator (per-band affine from 30 test scenes, stored in
                    ``legacy_radiometry.json``).
``zscore``          per-chip, per-band z-score, which is what the old training
                    dataloader applied with ``DynamicZScore``.

Neither variant improves the old models (see ``metrics_legacy_inputs.md``), so
the caveat is documented with numbers instead of argued.

The old QA4EO-2 checkpoints cannot split thick from thin cloud, so there are
two tables: the 4-class one for the models that have four outputs, and a
3-class one (thick + thin -> cloud) where every model appears.  The 3-class
table is the only fully common comparison.

Outputs (``--out``)::

    metrics_4class.csv / .md     per-class table of the 4-class models
    metrics_3class.csv / .md     the common 3-class table, every model
    metrics_perchip.md           mean over chips and win rate against OCM on MSS
    metrics_legacy_inputs.md     raw / 0.1.13 radiometry / z-score for QA4EO-2
    metrics_pics.csv / .md       real PICS: fraction flagged as contaminated
    ocm_pseudo_test.npz          cached OCM masks at 10 m (reused across runs)
    predictions.npz              per-chip masks of the test split
    run_benchmark.json           every number, the checkpoint hashes and the
                                 fixture hashes of the run
    figures/fig_contact_sheet.png
    figures/fig_pics_real_mss.png

Usage
-----
    python scripts/benchmark_mss_common_test.py --max-chips 24 --no-figures
    python scripts/benchmark_mss_common_test.py --out ../assets/md/fuentes/bench-fair
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
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from evaluate_cloud_baseline import (  # noqa: E402
    CHIP_SIZE,
    SCALE_FACTOR,
    class_fractions,
    false_colour,
    human_labels_state,
    load_pics_site,
    mss_valid_mask,
    ocm_pseudo_test,
    read_manifest,
)

from satharmony.emulators.s2_to_mss.labels import aggregate_cloudsen12  # noqa: E402
from satharmony.models import (  # noqa: E402
    file_sha256,
    load_model,
    predict_proba_multiscale,
)
from satharmony.models.metrics import (  # noqa: E402
    cohen_kappa,
    confusion_matrix,
    overall_accuracy,
)

DEFAULT_FIXTURE = Path("/data/databases/CLOUD_MSS/calibration/pseudo_mss_fixture")
DEFAULT_CLOUDSEN12 = Path("/data/databases/CMIXII/CLOUDSEN12_1K")
DEFAULT_G4 = Path("/data/databases/CLOUD_MSS/calibration/g4/cloud_unet_mss_pseudo.pt")
DEFAULT_G4B = Path("/data/databases/CLOUD_MSS/calibration/g4b/g4b_best.pt")
DEFAULT_ENSEMBLE = (
    Path("/data/databases/CLOUD_MSS/calibration/g4b/b8_wide_lowlr.pt"),
    Path("/data/databases/CLOUD_MSS/calibration/g4/cloud_unet_mss_pseudo.pt"),
    Path("/data/databases/CLOUD_MSS/calibration/g4b/b5_strong.pt"),
    Path("/data/databases/CLOUD_MSS/calibration/g4b/b4_deep.pt"),
)
DEFAULT_QA4EO2 = Path("/data/users/julio/isp_clouds/mss/QA4EO-2/unet.ckpt")
DEFAULT_FINETUNE = Path(
    "/data/users/julio/isp_clouds/mss/finetune/checkpoints_finetune_last_prev"
    "/ft-epoch=28-val_iou=0.7965.ckpt"
)
DEFAULT_PICS = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")
DEFAULT_LABELS = REPO.parent / "assets/md/fuentes/labels-humano"
DEFAULT_OUT = REPO.parent / "assets/md/fuentes/bench-fair"

#: Output classes of the package (CloudSEN12 order).
CLASS_NAMES_4 = ("clear", "thick_cloud", "thin_cloud", "cloud_shadow")

#: Common coarse classes: the old QA4EO-2 checkpoints only have these three.
CLASS_NAMES_3 = ("clear", "cloud", "cloud_shadow")

#: MSS band indices in the emulated stack: B1 green, B2 red, B3 NIR1, B4 NIR2.
MSS_GREEN, MSS_RED, MSS_NIR1, MSS_NIR2 = 0, 1, 2, 3

#: OmniCloudMask expects (red, green, NIR).  The NIR can be either MSS3 (0.7-0.8
#: um) or MSS4 (0.8-1.1 um, the band that contains the B8A window).
OCM_MSS_BANDS = (MSS_RED, MSS_GREEN, MSS_NIR2)
OCM_MSS_BANDS_NIR1 = (MSS_RED, MSS_GREEN, MSS_NIR1)

COLORS_3 = ["#f2f2f2", "#3a3a3a", "#a8a8a8"]
CMAP_3 = ListedColormap(COLORS_3)

PRETTY = {
    "clear": "Clear",
    "cloud": "Cloud",
    "thick_cloud": "Thick cloud",
    "thin_cloud": "Thin cloud",
    "cloud_shadow": "Shadow",
}

MODEL_LABELS = {
    "ocm_s2_10m": "OCM @ S2 10 m",
    "ocm_mss_60m": "OCM @ MSS 60 m",
    "ocm_mss_nir1": "OCM @ MSS 60 m (NIR1)",
    "ocm_mss_upsampled": "OCM @ MSS ups. 10 m",
    "g4": "G4 U-Net",
    "g4b": "G4b U-Net",
    "g4b_ens": "G4b ensemble",
    "qa4eo2": "QA4EO-2 (published)",
    "finetune": "QA4EO-2 fine-tune",
    "finetune_tta": "QA4EO-2 fine-tune + TTA",
}

LEGACY_NAMES = ("qa4eo2", "finetune", "finetune_tta")


# ── Metrics ──────────────────────────────────────────────────────────────
def collapse_to_three(mask: np.ndarray) -> np.ndarray:
    """Map CloudSEN12 4-class masks to clear / cloud / shadow.

    Args:
        mask: Classes 0 clear, 1 thick, 2 thin, 3 shadow (any shape).

    Returns:
        uint8 mask with 0 clear, 1 cloud (thick or thin), 2 shadow.

    Raises:
        ValueError: If a class outside 0-3 appears.
    """
    mask = np.asarray(mask)
    if mask.size and (mask.min() < 0 or mask.max() > 3):
        raise ValueError("mask classes must be in [0, 3]")
    cloud = (mask == 1) | (mask == 2)
    return np.where(mask == 3, 2, np.where(cloud, 1, 0)).astype(np.uint8)


def score(pred: np.ndarray, reference: np.ndarray, names: tuple[str, ...]) -> dict:
    """Per-class scores, mIoU, overall accuracy and kappa.

    Args:
        pred: Predicted classes.
        reference: Reference classes with the same encoding as ``pred``.
        names: Class names in encoding order.

    Returns:
        JSON-friendly dict with ``confusion_matrix``, ``per_class``,
        ``mean_iou``, ``overall_accuracy`` and ``cohen_kappa``.
    """
    matrix = confusion_matrix(pred, reference, num_classes=len(names))
    matrix_f = np.asarray(matrix, dtype=np.float64)
    true_positive = np.diag(matrix_f)
    reference_sum = matrix_f.sum(axis=1)
    predicted_sum = matrix_f.sum(axis=0)
    union = reference_sum + predicted_sum - true_positive
    per_class = {}
    for index, name in enumerate(names):
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = true_positive[index] / union[index] if union[index] else np.nan
            precision = (
                true_positive[index] / predicted_sum[index] if predicted_sum[index] else np.nan
            )
            recall = (
                true_positive[index] / reference_sum[index] if reference_sum[index] else np.nan
            )
            if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = np.nan
        per_class[name] = {
            "support": float(reference_sum[index]),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
        }
    finite = [values["iou"] for values in per_class.values() if np.isfinite(values["iou"])]
    return {
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "mean_iou": float(np.mean(finite)) if finite else float("nan"),
        "overall_accuracy": overall_accuracy(matrix),
        "cohen_kappa": cohen_kappa(matrix),
        "n_pixels": int(matrix.sum()),
    }


def legacy_input_variants(images: np.ndarray, reference: dict | None) -> dict[str, np.ndarray]:
    """Input conventions tried for the QA4EO-2 checkpoints.

    Args:
        images: ``(N, 4, H, W)`` fixture reflectance, 0-1.
        reference: Optional dict with per-band ``mean`` and ``std`` of the
            0.1.13 emulator.

    Returns:
        ``{name: (N, 4, H, W)}`` with ``raw`` and, when the reference exists,
        ``legacy_radiometry``; always includes ``zscore``.
    """
    variants = {"raw": images}
    if reference is not None:
        new_mean = images.mean(axis=(0, 2, 3))
        new_std = images.std(axis=(0, 2, 3))
        reference_mean = np.asarray(reference["mean"], dtype=np.float32)
        reference_std = np.asarray(reference["std"], dtype=np.float32)
        gain = reference_std / np.maximum(new_std, 1e-8)
        offset = reference_mean - new_mean * gain
        variants["legacy_radiometry"] = (
            images * gain[None, :, None, None] + offset[None, :, None, None]
        ).astype(np.float32)
    chip_mean = images.mean(axis=(-2, -1), keepdims=True)
    chip_std = images.std(axis=(-2, -1), keepdims=True)
    variants["zscore"] = ((images - chip_mean) / (chip_std + 1e-8)).astype(np.float32)
    return variants


def per_chip_scores(masks: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, np.ndarray]:
    """Per-chip 3-class mIoU of every model.

    The mean over chips answers a different question than the pooled mIoU: it
    gives every chip the same weight, so a handful of huge easy scenes cannot
    hide a model that fails on many small ones.

    Args:
        masks: ``{model: (N, H, W)}`` predictions.
        labels: ``(N, H, W)`` reference classes (4-class encoding).

    Returns:
        ``{model: (N,)}`` per-chip mean IoU over the classes present.
    """
    reference = collapse_to_three(labels)
    out = {}
    for name, mask in masks.items():
        prediction = mask if name in LEGACY_NAMES else collapse_to_three(mask)
        out[name] = np.asarray(
            [_chip_miou(prediction[position], reference[position])
             for position in range(reference.shape[0])]
        )
    return out


def pairwise_stats(per_chip: dict[str, np.ndarray], reference: str) -> dict[str, dict]:
    """Win / loss rate of every model against one reference, chip by chip.

    Args:
        per_chip: ``{model: (N,)}`` per-chip mIoU.
        reference: Model used as the baseline of the comparison.

    Returns:
        ``{model: {...}}`` with mean, median, win/loss share and the number of
        chips won or lost by more than 0.1 mIoU.
    """
    baseline = per_chip[reference]
    stats = {}
    for name, values in per_chip.items():
        if name == reference:
            continue
        delta = values - baseline
        stats[name] = {
            "mean_miou": float(np.mean(values)),
            "median_miou": float(np.median(values)),
            "win_share": float(np.mean(delta > 0.01)),
            "loss_share": float(np.mean(delta < -0.01)),
            "median_delta": float(np.median(delta)),
            "chips_won_by_010": int(np.sum(delta > 0.1)),
            "chips_lost_by_010": int(np.sum(delta < -0.1)),
        }
    return stats


# ── Data ─────────────────────────────────────────────────────────────────
def load_test_split(fixture: Path, max_chips: int | None = None) -> dict:
    """Test chips, reflectance and reference labels of the pseudo-MSS fixture.

    Args:
        fixture: Fixture directory with ``manifest.csv``, ``images.npy`` and
            ``labels.npy``.
        max_chips: Keep only the first N test chips (smoke runs).

    Returns:
        Dict with ``chips`` (manifest rows), ``all_chips`` (every test row,
        used to reuse the OCM cache), ``images`` ``(N, 4, 64, 64)``
        reflectance 0-1 and ``labels`` ``(N, 64, 64)`` classes.
    """
    manifest = read_manifest(fixture)
    images = np.load(fixture / "images.npy", mmap_mode="r")
    labels = np.load(fixture / "labels.npy", mmap_mode="r")
    index = [position for position, row in enumerate(manifest) if row["split"] == "test"]
    all_chips = [manifest[position] for position in index]
    if max_chips is not None:
        index = index[:max_chips]
    return {
        "chips": [manifest[position] for position in index],
        "all_chips": all_chips,
        "index": index,
        "images": np.asarray(images[index], dtype=np.float32) / 10000.0,
        "labels": np.asarray(labels[index], dtype=np.uint8),
    }


# ── Models ───────────────────────────────────────────────────────────────
def ocm_mask(array: np.ndarray, device: str) -> np.ndarray:
    """OmniCloudMask mask of one image, with a patch size that fits the image.

    ``omnicloudmask.predict_from_array`` defaults to a 300 px overlap, which is
    larger than a 64 x 64 MSS chip and raises.  The patch is rounded down to a
    multiple of eight and the overlap set to a quarter of it; on a chip that
    fits in one patch the result is independent of both (verified in the pilot).

    Args:
        array: ``(3, H, W)`` reflectance, 0-1, in OCM order (red, green, NIR).
        device: Torch device.

    Returns:
        ``(H, W)`` uint8 classes.
    """
    import warnings

    from omnicloudmask import predict_from_array

    size = min(array.shape[-2:])
    patch = min(1000, max(64, 8 * (size // 8)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return predict_from_array(
            np.ascontiguousarray(array, dtype=np.float32),
            patch_size=patch,
            patch_overlap=max(8, patch // 4),
            inference_device=device,
        )[0].astype(np.uint8)


def ocm_on_mss(images: np.ndarray, bands: tuple[int, int, int], device: str) -> np.ndarray:
    """Run OmniCloudMask on 60 m MSS chips, one forward per chip.

    Args:
        images: ``(N, 4, 64, 64)`` MSS reflectance, 0-1.
        bands: Three band indices in OmniCloudMask order (red, green, NIR).
        device: Torch device.

    Returns:
        ``(N, 64, 64)`` uint8 classes.
    """
    masks = np.zeros((images.shape[0], *images.shape[-2:]), dtype=np.uint8)
    for position, image in enumerate(images):
        masks[position] = ocm_mask(image[list(bands)], device)
    return masks


def ocm_on_mss_upsampled(
    images: np.ndarray, bands: tuple[int, int, int], device: str, factor: int = 6
) -> np.ndarray:
    """OCM on the 60 m MSS bilinearly resampled to 10 m, mask back at 60 m.

    The interpolation invents no information: it only changes the pixel grid
    so OCM sees its native ground sampling distance.  The 10 m mask is
    aggregated back to the 60 m grid with :func:`aggregate_cloudsen12`.

    Args:
        images: ``(N, 4, 64, 64)`` MSS reflectance, 0-1.
        bands: Three band indices in OmniCloudMask order.
        device: Torch device.
        factor: Upsampling factor (6 for 60 m -> 10 m).

    Returns:
        ``(N, 64, 64)`` uint8 classes.
    """
    from scipy.ndimage import zoom

    masks = np.zeros((images.shape[0], *images.shape[-2:]), dtype=np.uint8)
    for position, image in enumerate(images):
        coarse = np.ascontiguousarray(image[list(bands)], dtype=np.float32)
        fine = zoom(coarse, (1, factor, factor), order=1).astype(np.float32)
        prediction = ocm_mask(fine, device)
        masks[position] = aggregate_cloudsen12(prediction, scale_factor=factor)
    return masks


def evaluate_unet(path: Path, images: np.ndarray, device: str) -> np.ndarray:
    """Argmax of a satharmony U-Net with dihedral TTA at 1.0 scale.

    This is the decision rule of ``evaluate_cloud_baseline.py`` with
    ``--scales none --bias none``, which is how the G4/G4b test numbers
    (0.518 / 0.520) were produced.

    Args:
        path: Checkpoint.
        images: ``(N, 4, H, W)`` reflectance, 0-1.
        device: Torch device.

    Returns:
        ``(N, H, W)`` uint8 classes.
    """
    model, checkpoint = load_model(path, device=device)
    probabilities = predict_proba_multiscale(
        model,
        images,
        checkpoint.band_mean,
        checkpoint.band_std,
        scales=(1.0,),
        dihedral=True,
        batch_size=32,
        device=device,
    )
    return probabilities.argmax(axis=1).astype(np.uint8)


def evaluate_ensemble(paths: tuple[Path, ...], images: np.ndarray, device: str) -> np.ndarray:
    """Average the softmax of several U-Nets and take the argmax.

    Args:
        paths: Checkpoints to average.
        images: ``(N, 4, H, W)`` reflectance, 0-1.
        device: Torch device.

    Returns:
        ``(N, H, W)`` uint8 classes.
    """
    total = None
    for path in paths:
        model, checkpoint = load_model(path, device=device)
        probabilities = predict_proba_multiscale(
            model,
            images,
            checkpoint.band_mean,
            checkpoint.band_std,
            scales=(1.0,),
            dihedral=True,
            batch_size=32,
            device=device,
        )
        total = probabilities if total is None else total + probabilities
    average = total / len(paths)
    average = average / np.maximum(average.sum(axis=-3, keepdims=True), 1e-8)
    return average.argmax(axis=1).astype(np.uint8)


# ── Legacy QA4EO-2 ───────────────────────────────────────────────────────
def load_legacy_model(path: Path, device: str):
    """Load a QA4EO-2 checkpoint (4 MSS bands, 3 classes).

    The class mirrors ``assets/codes/mss/QA4EO-2/load.py``: a
    ``segmentation_models_pytorch`` U-Net with an EfficientNet-B3 encoder and
    SCSE attention, no encoder weights.  It is redefined here so the
    benchmark does not import a file from the vault.

    Args:
        path: Lightning ``.ckpt``.
        device: Torch device.

    Returns:
        The model in eval mode.
    """
    import pytorch_lightning as pl
    import segmentation_models_pytorch as smp

    class LegacyMSS(pl.LightningModule):
        """QA4EO-2 MSS U-Net, 3 classes: clear, cloud, shadow."""

        def __init__(
            self,
            in_channels: int = 4,
            num_classes: int = 3,
            encoder: str = "efficientnet-b3",
            lr: float = 3e-4,
            weight_decay: float = 1e-4,
            encoder_lr_factor: float = 0.3,
            freeze_encoder_epochs: int = 0,
            freeze_bn: bool = False,
        ):
            super().__init__()
            self.save_hyperparameters()
            self.model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=in_channels,
                classes=num_classes,
                encoder_depth=5,
                activation=None,
                decoder_attention_type="scse",
            )

        def forward(self, tensor):
            return self.model(tensor)

    model = LegacyMSS.load_from_checkpoint(str(path), map_location=device)
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _dihedral(tensor, code: int):
    """Apply one of the eight dihedral transforms to ``(N, C, H, W)``."""
    if code & 1:
        tensor = tensor.flip(-1)
    if code & 2:
        tensor = tensor.flip(-2)
    if code & 4:
        tensor = tensor.rot90(1, dims=(-2, -1))
    return tensor


def _dihedral_inverse(tensor, code: int):
    """Undo :func:`_dihedral`."""
    if code & 4:
        tensor = tensor.rot90(-1, dims=(-2, -1))
    if code & 2:
        tensor = tensor.flip(-2)
    if code & 1:
        tensor = tensor.flip(-1)
    return tensor


def legacy_predict(model, images: np.ndarray, device: str, tta: bool = False) -> np.ndarray:
    """Class mask of a QA4EO-2 checkpoint.

    Args:
        model: Loaded legacy model.
        images: ``(N, 4, H, W)`` MSS reflectance, 0-1.
        device: Torch device.
        tta: Average the softmax over the eight dihedral transforms.

    Returns:
        ``(N, H, W)`` uint8 classes: 0 clear, 1 cloud, 2 shadow.
    """
    import torch

    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(images, dtype=np.float32)).to(device)
        if not tta:
            return model(tensor).argmax(dim=1).cpu().numpy().astype(np.uint8)
        accumulated = None
        for code in range(8):
            probabilities = model(_dihedral(tensor, code)).softmax(dim=1)
            probabilities = _dihedral_inverse(probabilities, code)
            accumulated = probabilities if accumulated is None else accumulated + probabilities
        return (accumulated / 8).argmax(dim=1).cpu().numpy().astype(np.uint8)


# ── Tables ───────────────────────────────────────────────────────────────
def write_table_csv(path: Path, results: dict[str, dict], names: tuple[str, ...]) -> None:
    """One row per model and class plus a summary row; CSV."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "class", "support", "precision", "recall", "f1", "iou"])
        for model_name, result in results.items():
            for class_name in names:
                values = result["per_class"][class_name]
                writer.writerow(
                    [
                        model_name,
                        class_name,
                        int(values["support"]),
                        *[f"{values[key]:.6f}" for key in ("precision", "recall", "f1", "iou")],
                    ]
                )
            writer.writerow(
                [
                    model_name,
                    "SUMMARY",
                    "",
                    "",
                    "",
                    f"mIoU={result['mean_iou']:.6f}",
                    f"OA={result['overall_accuracy']:.6f} kappa={result['cohen_kappa']:.6f}",
                ]
            )


def write_table_markdown(
    path: Path, results: dict[str, dict], names: tuple[str, ...], title: str
) -> None:
    """Markdown table: one row per model, one column per class IoU."""
    lines = [f"# {title}", ""]
    header = "| Modelo | " + " | ".join(PRETTY[name] for name in names) + " | mIoU | OA | kappa |"
    lines.append(header)
    lines.append("|---" + "|---:" * (len(names) + 3) + "|")
    for model_name, result in results.items():
        cells = []
        for class_name in names:
            value = result["per_class"][class_name]["iou"]
            cells.append("—" if not np.isfinite(value) else f"{value:.3f}")
        lines.append(
            f"| {MODEL_LABELS.get(model_name, model_name)} | "
            + " | ".join(cells)
            + f" | **{result['mean_iou']:.3f}** | {result['overall_accuracy']:.3f} | "
            f"{result['cohen_kappa']:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


# ── Figures ──────────────────────────────────────────────────────────────
def _chip_miou(pred: np.ndarray, reference: np.ndarray) -> float:
    """Mean IoU of one chip over the classes present."""
    matrix = confusion_matrix(pred, reference)
    values = []
    for index in range(matrix.shape[0]):
        true_positive = matrix[index, index]
        union = matrix[index].sum() + matrix[:, index].sum() - true_positive
        if union:
            values.append(true_positive / union)
    return float(np.mean(values)) if values else float("nan")


def pick_contact_rows(
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    chips: list[dict],
    count: int = 8,
) -> list[int]:
    """Choose the chips shown in the contact sheet.

    Four chips are the best G4b case of each cloud regime, one is shadow-rich,
    and the last ones are the extreme cases of the comparison that matters:
    the chip where the old fine-tune beats OCM the most and the chip where it
    loses the most.  The sheet therefore shows both the typical behaviour and
    the best case for the old model.

    Args:
        labels: ``(N, H, W)`` reference classes.
        masks: ``{model: (N, H, W)}`` predictions.
        chips: Manifest rows of the test split.
        count: Number of rows.

    Returns:
        Indices into the test split, in display order.
    """
    per_chip = {}
    for name in ("g4b", "finetune", "ocm_mss_60m"):
        if name not in masks:
            continue
        cloud = name in ("qa4eo2", "finetune", "finetune_tta")
        reference = collapse_to_three(labels) if cloud else labels
        predictions = collapse_to_three(masks[name]) if cloud else masks[name]
        per_chip[name] = np.asarray(
            [_chip_miou(predictions[position], reference[position])
             for position in range(reference.shape[0])]
        )

    picked: list[int] = []
    if "g4b" in per_chip:
        for regime in ("clear", "low_cloud", "mid_cloud", "high_cloud"):
            candidates = [
                position for position, chip in enumerate(chips) if chip["regime"] == regime
            ]
            if candidates:
                picked.append(max(candidates, key=lambda position: per_chip["g4b"][position]))
    shadow = [position for position, chip in enumerate(chips) if chip["shadow_rich"] == "1"]
    if shadow:
        picked.append(shadow[0])
    if "finetune" in per_chip and "ocm_mss_60m" in per_chip:
        delta = per_chip["finetune"] - per_chip["ocm_mss_60m"]
        picked.append(int(np.argmax(delta)))
        picked.append(int(np.argmin(delta)))
    if "ocm_mss_60m" in per_chip:
        picked.append(int(np.argmin(per_chip["ocm_mss_60m"])))
    unique = list(dict.fromkeys(picked))
    return unique[:count]


def chip_label(chip_id: str) -> str:
    """Short label of a CloudSEN12 chip: date and Sentinel-2 tile.

    ``ROI_00032__20190330T032539_20190330T033305_T49TFH`` becomes
    ``20190330 T49TFH``.
    """
    parts = chip_id.split("__")
    tail = parts[-1].split("_")
    date = tail[0][:8] if tail and len(tail[0]) >= 8 else tail[0]
    return f"{date} {tail[-1]}" if tail else chip_id


def figure_contact_sheet(
    path: Path,
    images: np.ndarray,
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    chips: list[dict],
    rows: list[int],
) -> None:
    """Three-class contact sheet: MSS, reference and the MSS models.

    Args:
        path: Destination PNG.
        images: ``(N, 4, H, W)`` MSS reflectance, 0-1.
        labels: ``(N, H, W)`` reference classes (4-class encoding).
        masks: ``{model: (N, H, W)}`` 4-class predictions.
        chips: Manifest rows of the test split.
        rows: Test-split indices to draw.
    """
    mask3 = {name: collapse_to_three(masks[name]) for name in masks if name not in LEGACY_NAMES}
    reference3 = collapse_to_three(labels)
    columns = [
        "MSS false colour",
        "Reference",
        "OCM @ S2 10 m",
        "OCM @ MSS 60 m",
        "G4b U-Net",
        "QA4EO-2 fine-tune",
        "fine-tune vs OCM",
    ]
    fig, axes = plt.subplots(
        len(rows), len(columns), figsize=(17.5, 2.35 * len(rows)), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    comparison = ListedColormap(["#f0f0f0", "#2ca25f", "#de2d26"])
    for row, position in enumerate(rows):
        chip = chips[position]
        axes[row, 0].imshow(
            false_colour(images[position], bands=(MSS_NIR2, MSS_RED, MSS_GREEN))
        )
        composition = " ".join(
            f"{label} {100 * float(chip[key]):.0f}"
            for key, label in (
                ("clear_frac", "clear"),
                ("thick_frac", "thick"),
                ("thin_frac", "thin"),
                ("shadow_frac", "shadow"),
            )
        )
        axes[row, 0].set_ylabel(
            f"{chip_label(chip['chip_id'])}\n{chip['regime']}\n{composition}",
            fontsize=6.5,
        )
        axes[row, 1].imshow(
            reference3[position], cmap=CMAP_3, vmin=0, vmax=2, interpolation="nearest"
        )
        for column, name in zip(
            range(2, 6), ("ocm_s2_10m", "ocm_mss_60m", "g4b", "finetune"), strict=True
        ):
            prediction = mask3.get(name, masks.get(name))[position]
            iou = score(prediction, reference3[position], CLASS_NAMES_3)["mean_iou"]
            axes[row, column].imshow(
                prediction, cmap=CMAP_3, vmin=0, vmax=2, interpolation="nearest"
            )
            text = f"mIoU {iou:.2f}"
            axes[row, column].set_title(
                f"{columns[column]}\n{text}" if row == 0 else text,
                fontsize=8.5 if row == 0 else 6.5,
                pad=1.5,
            )
        old = masks["finetune"][position] == reference3[position]
        ocm = mask3["ocm_mss_60m"][position] == reference3[position]
        delta = np.where(old & ~ocm, 1, np.where(~old & ocm, 2, 0)).astype(np.uint8)
        axes[row, 6].imshow(delta, cmap=comparison, vmin=0, vmax=2, interpolation="nearest")
        gained = round(float((delta == 1).mean()) * 100)
        lost = round(float((delta == 2).mean()) * 100)
        text = f"+{gained} / -{lost} pp"
        axes[row, 6].set_title(
            f"{columns[6]}\n{text}" if row == 0 else text,
            fontsize=8.5 if row == 0 else 6.5,
            pad=1.5,
        )
        for column in range(len(columns)):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=COLORS_3[0], edgecolor="#999999"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS_3[1], edgecolor="#999999"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS_3[2], edgecolor="#999999"),
        Rectangle((0, 0), 1, 1, facecolor="#2ca25f", edgecolor="#999999"),
        Rectangle((0, 0), 1, 1, facecolor="#de2d26", edgecolor="#999999"),
    ]
    legend_labels = ["Clear", "Cloud", "Shadow", "fine-tune right / OCM wrong",
                     "OCM right / fine-tune wrong"]
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=5, fontsize=8,
               frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_pics(
    path: Path,
    site_rows: list[dict],
    ocm_s2: list[np.ndarray],
    ocm_mss: list[np.ndarray],
    unet: list[np.ndarray],
    legacy: list[np.ndarray],
    sites: list[dict],
) -> None:
    """Real PICS: MSS false colour and the four masks, no labels.

    Args:
        path: Destination PNG.
        site_rows: One dict per site with the contaminated fractions.
        ocm_s2: OCM masks on the 10 m S2 (aggregated to 60 m).
        ocm_mss: OCM masks on the real MSS at 60 m.
        unet: G4b masks.
        legacy: Fine-tune masks (3 classes).
        sites: Output of ``load_pics_site``.
    """
    columns = [
        "Real MSS false colour",
        "OCM @ S2 10 m",
        "OCM @ MSS 60 m",
        "G4b U-Net",
        "QA4EO-2 fine-tune",
    ]
    fig, axes = plt.subplots(
        len(sites), len(columns), figsize=(13.0, 2.7 * len(sites)), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    for row, site in enumerate(sites):
        axes[row, 0].imshow(
            false_colour(site["mss"], bands=(MSS_NIR2, MSS_RED, MSS_GREEN))
        )
        axes[row, 0].set_ylabel(site_rows[row]["label"], fontsize=8)
        series = [
            collapse_to_three(ocm_s2[row]),
            collapse_to_three(ocm_mss[row]),
            collapse_to_three(unet[row]),
            legacy[row],
        ]
        for column, series_mask in enumerate(series, start=1):
            axes[row, column].imshow(
                series_mask, cmap=CMAP_3, vmin=0, vmax=2, interpolation="nearest"
            )
            fraction = float(np.mean(series_mask != 0))
            axes[row, column].set_title(
                f"cloud+shadow {100 * fraction:.1f} %", fontsize=7.5, pad=1.5
            )
        for column in range(len(columns)):
            if row == 0:
                axes[row, column].set_title(columns[column], fontsize=8.5)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=COLORS_3[0], edgecolor="#999999"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS_3[1], edgecolor="#999999"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS_3[2], edgecolor="#999999"),
    ]
    fig.legend(handles, ["Clear", "Cloud", "Shadow"], loc="outside lower center", ncol=3,
               fontsize=8, frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--cloudsen12", type=Path, default=DEFAULT_CLOUDSEN12)
    parser.add_argument("--g4", type=Path, default=DEFAULT_G4)
    parser.add_argument("--g4b", type=Path, default=DEFAULT_G4B)
    parser.add_argument("--ensemble", type=Path, nargs="+", default=list(DEFAULT_ENSEMBLE))
    parser.add_argument("--qa4eo2", type=Path, default=DEFAULT_QA4EO2)
    parser.add_argument("--finetune", type=Path, default=DEFAULT_FINETUNE)
    parser.add_argument("--pics", type=Path, default=DEFAULT_PICS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--legacy-reference",
        type=Path,
        default=DEFAULT_OUT / "legacy_radiometry.json",
        help="per-band mean/std of the 0.1.13 emulator for the sensitivity block",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-chips", type=int, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--skip-pics", action="store_true")
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--skip-ocm-upsampled", action="store_true")
    parser.add_argument("--reuse-ocm", action="store_true", default=True)
    args = parser.parse_args()

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    out = args.out
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    data = load_test_split(args.fixture, args.max_chips)
    chips, images, labels = data["chips"], data["images"], data["labels"]
    print(f"{len(chips)} test chips, {images.shape}, device={device}")

    masks: dict[str, np.ndarray] = {}
    results4: dict[str, dict] = {}
    results3: dict[str, dict] = {}

    # ── OmniCloudMask ────────────────────────────────────────────────────
    ocm_s2_full, _ = ocm_pseudo_test(
        data["all_chips"], args.cloudsen12, out / "ocm_pseudo_test.npz", device, args.reuse_ocm
    )
    masks["ocm_s2_10m"] = ocm_s2_full[: len(chips)]

    masks["ocm_mss_60m"] = ocm_on_mss(images, OCM_MSS_BANDS, device)
    masks["ocm_mss_nir1"] = ocm_on_mss(images, OCM_MSS_BANDS_NIR1, device)
    if not args.skip_ocm_upsampled:
        masks["ocm_mss_upsampled"] = ocm_on_mss_upsampled(images, OCM_MSS_BANDS, device)

    # ── satharmony U-Nets ────────────────────────────────────────────────
    masks["g4"] = evaluate_unet(args.g4, images, device)
    masks["g4b"] = evaluate_unet(args.g4b, images, device)
    if args.ensemble:
        masks["g4b_ens"] = evaluate_ensemble(tuple(args.ensemble), images, device)

    # ── QA4EO-2 published and fine-tuned ─────────────────────────────────
    legacy_models = {}
    if not args.skip_legacy:
        for name, path in (("qa4eo2", args.qa4eo2), ("finetune", args.finetune)):
            if not Path(path).exists():
                print(f"[{name}] missing checkpoint {path}, skipped")
                continue
            model = load_legacy_model(Path(path), device)
            legacy_models[name] = model
            masks[name] = legacy_predict(model, images, device, tta=False)
            if name == "finetune":
                masks["finetune_tta"] = legacy_predict(model, images, device, tta=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Scores ───────────────────────────────────────────────────────────
    for name, mask in masks.items():
        is_three = name in LEGACY_NAMES
        if is_three:
            results3[name] = score(mask, collapse_to_three(labels), CLASS_NAMES_3)
        else:
            results4[name] = score(mask, labels, CLASS_NAMES_4)
            results3[name] = score(
                collapse_to_three(mask), collapse_to_three(labels), CLASS_NAMES_3
            )
        reference = results4.get(name, results3[name])
        print(
            f"[{name}] 4c mIoU {reference['mean_iou']:.4f} | 3c mIoU "
            f"{results3[name]['mean_iou']:.4f} | OA {results3[name]['overall_accuracy']:.4f} | "
            f"kappa {results3[name]['cohen_kappa']:.4f}"
        )

    # ── Old checkpoints: alternative input conventions ───────────────────
    legacy_sensitivity: dict[str, dict] = {}
    reference = None
    if args.legacy_reference.exists():
        reference = json.loads(args.legacy_reference.read_text())
    if legacy_models:
        variants = legacy_input_variants(images, reference)
        for name, model in legacy_models.items():
            legacy_sensitivity[name] = {}
            for variant_name, variant_images in variants.items():
                prediction = legacy_predict(model, variant_images, device, tta=False)
                legacy_sensitivity[name][variant_name] = score(
                    prediction, collapse_to_three(labels), CLASS_NAMES_3
                )
            best = max(legacy_sensitivity[name].values(), key=lambda item: item["mean_iou"])
            print(
                f"[{name}] input variants: "
                + ", ".join(
                    f"{variant} {result['mean_iou']:.4f}"
                    for variant, result in legacy_sensitivity[name].items()
                )
                + f" (best {best['mean_iou']:.4f})"
            )
        lines = [
            "# QA4EO-2: convención de entrada (3 clases, mismo test)",
            "",
            "| Modelo | Entrada | mIoU | OA | kappa |",
            "|---|---|---:|---:|---:|",
        ]
        for name, variants in legacy_sensitivity.items():
            for variant_name, result in variants.items():
                lines.append(
                    f"| {MODEL_LABELS.get(name, name)} | {variant_name} | "
                    f"{result['mean_iou']:.4f} | {result['overall_accuracy']:.4f} | "
                    f"{result['cohen_kappa']:.4f} |"
                )
        (out / "metrics_legacy_inputs.md").write_text("\n".join(lines) + "\n")

    # ── Per chip: mean over chips and win rate against OCM on MSS ────────
    per_chip = per_chip_scores(masks, labels)
    pairwise = pairwise_stats(per_chip, "ocm_mss_60m") if "ocm_mss_60m" in per_chip else {}
    if pairwise:
        lines = [
            "# Por chip (3 clases): media, mediana y cara a cara con OCM @ MSS 60 m",
            "",
            "Un chip pesa lo mismo que cualquier otro. *Gana* = +0,01 mIoU o más;",
            "*pierde* = -0,01 o menos.",
            "",
            "| Modelo | mIoU medio | mIoU mediano | Gana | Pierde | Delta mediano | "
            "Gana >0,1 | Pierde >0,1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, values in pairwise.items():
            lines.append(
                f"| {MODEL_LABELS.get(name, name)} | {values['mean_miou']:.3f} | "
                f"{values['median_miou']:.3f} | {100 * values['win_share']:.0f} % | "
                f"{100 * values['loss_share']:.0f} % | {values['median_delta']:+.3f} | "
                f"{values['chips_won_by_010']} | {values['chips_lost_by_010']} |"
            )
        (out / "metrics_perchip.md").write_text("\n".join(lines) + "\n")

    # ── Per regime (3 classes, the common denominator) ───────────────────
    per_regime = {}
    for regime in ("clear", "low_cloud", "mid_cloud", "high_cloud"):
        indices = [position for position, chip in enumerate(chips) if chip["regime"] == regime]
        if not indices:
            continue
        reference = collapse_to_three(labels[indices])
        per_regime[regime] = {
            "n_chips": len(indices),
            "models": {
                name: score(
                    mask[indices] if name in LEGACY_NAMES else collapse_to_three(mask[indices]),
                    reference,
                    CLASS_NAMES_3,
                )["mean_iou"]
                for name, mask in masks.items()
            },
        }

    # ── Tables ───────────────────────────────────────────────────────────
    write_table_csv(out / "metrics_4class.csv", results4, CLASS_NAMES_4)
    write_table_markdown(
        out / "metrics_4class.md",
        results4,
        CLASS_NAMES_4,
        "Test común pseudo-MSS 60 m — 4 clases (solo modelos con gruesa/fina)",
    )
    write_table_csv(out / "metrics_3class.csv", results3, CLASS_NAMES_3)
    write_table_markdown(
        out / "metrics_3class.md",
        results3,
        CLASS_NAMES_3,
        "Test común pseudo-MSS 60 m — 3 clases (claro / nube / sombra), todos los modelos",
    )

    # ── Figures ──────────────────────────────────────────────────────────
    rows: list[int] = []
    if not args.no_figures:
        rows = pick_contact_rows(labels, masks, chips)
        figure_contact_sheet(
            figures_dir / "fig_contact_sheet.png", images, labels, masks, chips, rows
        )
        print(f"contact sheet: {len(rows)} chips")

    # ── Real PICS ────────────────────────────────────────────────────────
    pics_rows: list[dict] = []
    if not args.skip_pics and args.pics.exists():
        sites = []
        ocm_s2_masks = []
        ocm_mss_masks = []
        unet_masks = []
        legacy_masks = []
        for site_dir in sorted(path for path in args.pics.iterdir() if path.is_dir()):
            if not (site_dir / "metadata.json").exists():
                continue
            site = load_pics_site(site_dir)
            valid = mss_valid_mask(site)
            ocm60 = aggregate_cloudsen12(ocm_mask(site["s2"], device),
                                         scale_factor=SCALE_FACTOR)
            mss_mask = ocm_on_mss(site["mss"][None], OCM_MSS_BANDS, device)[0]
            unet_mask = evaluate_unet(args.g4b, site["mss"][None], device)[0]
            if "finetune" in legacy_models:
                legacy_mask = legacy_predict(
                    legacy_models["finetune"], site["mss"][None], device, tta=False
                )[0]
            else:
                legacy_mask = np.zeros_like(unet_mask)
            sites.append(site)
            ocm_s2_masks.append(ocm60)
            ocm_mss_masks.append(mss_mask)
            unet_masks.append(unet_mask)
            legacy_masks.append(legacy_mask)
            fractions = class_fractions(mss_mask, valid)
            pics_rows.append(
                {
                    "site": site["metadata"]["site"],
                    "label": site["metadata"].get("label", site["metadata"]["site"]),
                    "mss_date": site["metadata"]["mss"].get("date"),
                    "sun_elevation_deg": site["metadata"]["mss"].get("sun_elevation"),
                    "n_valid_px": int(valid.sum()),
                    "ocm_s2_10m": class_fractions(ocm60, valid)["contaminated"],
                    "ocm_mss_60m": fractions["contaminated"],
                    "g4b": class_fractions(unet_mask, valid)["contaminated"],
                    "finetune": class_fractions(legacy_mask, valid)["contaminated"],
                }
            )
            printed = pics_rows[-1]
            print(
                f"  PICS {printed['label']}: OCM S2 {100 * printed['ocm_s2_10m']:.1f} %, "
                f"OCM MSS {100 * printed['ocm_mss_60m']:.1f} %, G4b "
                f"{100 * printed['g4b']:.1f} %, FT {100 * printed['finetune']:.1f} %"
            )
        if pics_rows:
            with (out / "metrics_pics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(pics_rows[0]))
                writer.writeheader()
                writer.writerows(pics_rows)
            lines = [
                "| Sitio | Fecha MSS | OCM @ S2 10 m | OCM @ MSS 60 m | G4b | QA4EO-2 fine-tune |",
                "|---|---|---:|---:|---:|---:|",
            ]
            for row in pics_rows:
                lines.append(
                    f"| {row['label']} | {row['mss_date']} | "
                    f"{100 * row['ocm_s2_10m']:.2f} % | {100 * row['ocm_mss_60m']:.2f} % | "
                    f"{100 * row['g4b']:.2f} % | {100 * row['finetune']:.2f} % |"
                )
            (out / "metrics_pics.md").write_text("\n".join(lines) + "\n")
            if not args.no_figures:
                figure_pics(
                    figures_dir / "fig_pics_real_mss.png",
                    pics_rows,
                    ocm_s2_masks,
                    ocm_mss_masks,
                    unet_masks,
                    legacy_masks,
                    sites,
                )

    # ── Record ───────────────────────────────────────────────────────────
    human = human_labels_state(args.labels)
    fixture_meta = json.loads((args.fixture / "fixture.json").read_text())
    run = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": time.perf_counter() - start,
        "test": {
            "fixture": str(args.fixture),
            "images_sha256": fixture_meta.get("images_sha256"),
            "labels_sha256": fixture_meta.get("labels_sha256"),
            "n_chips": len(chips),
            "grid": f"{CHIP_SIZE} px at 60 m",
            "decision_rule": "argmax, dihedral TTA (8), scale 1.0 in the satharmony "
            "U-Nets; OCM and QA4EO-2 without TTA (finetune_tta adds it)",
        },
        "models": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
                "classes": 3 if name in LEGACY_NAMES else 4,
            }
            for name, path in (
                ("g4", args.g4),
                ("g4b", args.g4b),
                ("qa4eo2", args.qa4eo2),
                ("finetune", args.finetune),
            )
            if Path(path).exists()
        },
        "ensemble": [str(path) for path in args.ensemble],
        "ensemble_sha256": [file_sha256(path) for path in args.ensemble],
        "scores_4class": results4,
        "scores_3class": results3,
        "per_regime_3class": per_regime,
        "pairwise_vs_ocm_mss": pairwise,
        "legacy_input_sensitivity": legacy_sensitivity,
        "pics": {"human_labels": human, "sites": pics_rows},
        "contact_sheet_chips": [
            {"chip_id": chips[position]["chip_id"], "regime": chips[position]["regime"]}
            for position in rows
        ],
    }
    (out / "run_benchmark.json").write_text(json.dumps(run, indent=2) + "\n")
    np.savez_compressed(out / "predictions.npz", **masks)
    print(f"wrote {out / 'run_benchmark.json'} ({run['elapsed_s']:.1f}s)")
    print(f"human labels available: {human['available']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
