#!/usr/bin/env python
"""Train the G4b cloud / shadow model on the pseudo-MSS fixture.

Same fixture and same scene split as the G4 baseline; the differences are the
levers that target the thin-cloud hole:

* **Loss**: class-weighted focal cross-entropy (``--focal-gamma``) plus soft
  Dice, with the class priors selectable (``--class-weight-mode``).
* **Capacity**: wider/deeper U-Net or a ``timm`` encoder, optionally with
  ImageNet weights (``--encoder``/``--pretrained``).
* **Training length**: up to ``--epochs`` with early stopping on validation
  mIoU (``--patience``).
* **Decision rule**: after the best epoch, a per-class bias is tuned on the
  validation split (never on test) and stored in the checkpoint metadata.
* **Augmentation**: ``--augment strong`` adds gamma jitter and read noise on
  top of the flips, rotations and per-band gain.

Outputs (``--out``)::

    run_train.json     full config, per-epoch log, best metrics, tuned bias
    run_train.md       human-readable summary
    train_log.csv      one row per epoch
    <ckpt>             best weights (torch.save, weights_only-safe)
    <ckpt>.sha256      file digest

The test split is never touched: it is evaluated by
``scripts/evaluate_cloud_baseline.py``.

Usage
-----
    python scripts/train_cloud_baseline.py
    python scripts/train_cloud_baseline.py --epochs 150 --focal-gamma 2 \
        --class-weight-mode inverse --checkpoint /tmp/g4b_a.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.models import (  # noqa: E402
    CLASS_NAMES,
    ModelConfig,
    SegLoss,
    apply_class_bias,
    build_model,
    class_weights,
    confusion_matrix,
    count_parameters,
    iou_per_class,
    load_checkpoint,
    mean_iou,
    overall_accuracy,
    predict_proba_dihedral,
    save_checkpoint,
    tune_class_bias,
)

DEFAULT_FIXTURE = Path("/data/databases/CLOUD_MSS/calibration/pseudo_mss_fixture")
DEFAULT_CHECKPOINT = Path("/data/databases/CLOUD_MSS/calibration/g4/cloud_unet_mss_pseudo.pt")
DEFAULT_OUT = REPO.parent / "assets/md/fuentes/g4"

SEED = 20261024
TRAIN = {
    "epochs": 60,
    "batch_size": 32,
    "learning_rate": 2e-3,
    "min_learning_rate": 1e-5,
    "weight_decay": 1e-4,
    "dice_weight": 0.5,
    "patience": 15,
    "augment": "full",
}

AUGMENT_DESCRIPTION = {
    "flips": "flip h/v, rot90",
    "gain": "flip h/v, rot90, per-band gain 0.95-1.05",
    "full": "flip h/v, rot90, per-band gain 0.95-1.05, offset +/-0.01",
    "strong": (
        "flip h/v, rot90, per-band gain 0.90-1.10, offset +/-0.02, "
        "gamma 0.85-1.15, read noise sigma 0.004"
    ),
}


def load_fixture(fixture: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    """Read the arrays and the manifest of the pseudo-MSS fixture.

    The arrays are loaded fully into RAM (73 MB): the training loop copies
    random batches, and memory mapping those copies was the main cost of the
    G4 run.

    Args:
        fixture: Directory with ``images.npy``, ``labels.npy`` and
            ``manifest.csv``.

    Returns:
        ``(images, labels, manifest)`` with images ``(N, 4, 64, 64)`` uint16,
        labels ``(N, 64, 64)`` uint8 and the manifest rows in array order.

    Raises:
        FileNotFoundError: If any of the three files is missing.
    """
    images_path = fixture / "images.npy"
    labels_path = fixture / "labels.npy"
    manifest_path = fixture / "manifest.csv"
    for path in (images_path, labels_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"missing fixture file: {path}")
    images = np.load(images_path)
    labels = np.load(labels_path)
    with manifest_path.open() as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != len(images):
        raise ValueError(f"manifest has {len(manifest)} rows, arrays have {len(images)}")
    return images, labels, manifest


def split_indices(manifest: list[dict[str, str]]) -> dict[str, np.ndarray]:
    """Indices of every split, sorted for reproducibility.

    Args:
        manifest: Rows of ``manifest.csv``.

    Returns:
        ``{"train": idx, "validation": idx, "test": idx}``.
    """
    splits = {}
    for name in ("train", "validation", "test"):
        splits[name] = np.array(
            [index for index, row in enumerate(manifest) if row["split"] == name], dtype=np.int64
        )
    return splits


def band_statistics(images: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean and standard deviation over the training chips.

    Args:
        images: ``(N, 4, H, W)`` uint16 reflectance x 10000.
        indices: Training indices.

    Returns:
        ``(mean, std)`` float32 arrays of shape ``(4,)`` in reflectance units.
    """
    block = np.asarray(images[indices], dtype=np.float32) / 10000.0
    mean = block.mean(axis=(0, 2, 3))
    std = block.std(axis=(0, 2, 3))
    std = np.where(std > 1e-6, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def sampling_probabilities(
    labels: np.ndarray, indices: np.ndarray, thin_oversample: float = 1.0
) -> np.ndarray | None:
    """Per-chip sampling probability that oversamples chips with thin cloud.

    Thin cloud is 6 % of the pixels and appears in a minority of the chips, so
    a plain shuffle sees it far less often than clear.  Multiplying the
    probability of the chips that contain thin cloud is a data-level lever that
    does not touch the labels.

    Args:
        labels: ``(N, H, W)`` class array.
        indices: Training indices.
        thin_oversample: Factor applied to the chips with any thin-cloud pixel;
            ``1`` disables the oversampling.

    Returns:
        Normalised probability per training chip, or ``None`` when disabled.
    """
    if thin_oversample <= 1.0:
        return None
    block = np.asarray(labels[indices])
    has_thin = (block == 2).any(axis=(1, 2)).astype(np.float64)
    weights = 1.0 + (thin_oversample - 1.0) * has_thin
    return (weights / weights.sum()).astype(np.float64)


def augment_batch(
    images: np.ndarray, labels: np.ndarray, rng: np.random.Generator, mode: str = "full"
) -> tuple[np.ndarray, np.ndarray]:
    """Physically mild augmentation for reflectance chips.

    Flips and 90-degree rotations are exact symmetries of a top-of-atmosphere
    image.  The radiometric terms only model small calibration differences:
    per-band gain, a small offset, a gamma (haze proxy) and read noise in the
    ``strong`` mode.  No elastic or crop augmentation is used, so clouds keep
    their size relative to the 60 m grid.

    Args:
        images: ``(B, 4, H, W)`` float32 reflectance.
        labels: ``(B, H, W)`` uint8 classes.
        rng: Seeded generator.
        mode: ``flips``, ``gain``, ``full`` or ``strong``.

    Returns:
        New ``(images, labels)`` arrays.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    if mode not in AUGMENT_DESCRIPTION:
        raise ValueError(f"unknown augment mode: {mode}")
    out_images = np.empty_like(images)
    out_labels = np.empty_like(labels)
    for index in range(images.shape[0]):
        image = images[index]
        label = labels[index]
        if rng.random() < 0.5:
            image, label = image[:, :, ::-1], label[:, ::-1]
        if rng.random() < 0.5:
            image, label = image[:, ::-1, :], label[::-1, :]
        rotations = int(rng.integers(4))
        if rotations:
            image = np.rot90(image, rotations, axes=(1, 2))
            label = np.rot90(label, rotations, axes=(0, 1))
        if mode == "flips":
            out_images[index] = image
            out_labels[index] = label
            continue
        low, high = (0.90, 1.10) if mode == "strong" else (0.95, 1.05)
        gain = rng.uniform(low, high, size=(4, 1, 1)).astype(np.float32)
        if mode in {"full", "strong"}:
            spread = 0.02 if mode == "strong" else 0.01
            offset = rng.uniform(-spread, spread, size=(4, 1, 1)).astype(np.float32)
            image = image * gain + offset
        else:
            image = image * gain
        if mode == "strong":
            gamma = float(rng.uniform(0.85, 1.15))
            image = np.clip(image, 0.0, 1.0) ** gamma
            image = image + rng.normal(0.0, 0.004, size=image.shape).astype(np.float32)
        out_images[index] = np.clip(image, 0.0, 1.0)
        out_labels[index] = label
    return out_images, out_labels


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    criterion: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict:
    """Loss and per-class IoU over a split, plain argmax.

    Args:
        model: The segmentation model.
        images: ``(N, 4, H, W)`` uint16 reflectance x 10000.
        labels: ``(N, H, W)`` uint8 classes.
        indices: Chips of the split.
        band_mean: Per-band training mean.
        band_std: Per-band training standard deviation.
        criterion: Loss module.
        device: Torch device.
        batch_size: Images per pass.

    Returns:
        Dict with ``loss``, ``mean_iou``, ``overall_accuracy`` and ``iou``.
    """
    model.eval()
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    total_loss = 0.0
    n_pixels = 0
    for start in range(0, len(indices), batch_size):
        batch_index = indices[start : start + batch_size]
        reflectance = np.asarray(images[batch_index], dtype=np.float32) / 10000.0
        batch_images = torch.from_numpy(
            (reflectance - band_mean.reshape(1, 4, 1, 1)) / band_std.reshape(1, 4, 1, 1)
        ).to(device)
        batch_labels = torch.from_numpy(np.asarray(labels[batch_index]).astype(np.int64)).to(device)
        logits = model(batch_images)
        total, _, _ = criterion(logits, batch_labels)
        total_loss += float(total) * batch_labels.numel()
        n_pixels += batch_labels.numel()
        predicted = logits.argmax(dim=1).cpu().numpy()
        matrix += confusion_matrix(predicted, batch_labels.cpu().numpy())
    return {
        "loss": total_loss / max(n_pixels, 1),
        "mean_iou": mean_iou(matrix),
        "overall_accuracy": overall_accuracy(matrix),
        "iou": iou_per_class(matrix),
        "confusion_matrix": matrix.tolist(),
    }


def validate_tta(
    model: torch.nn.Module,
    images: np.ndarray,
    indices: np.ndarray,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    device: torch.device,
    batch_size: int,
    labels: np.ndarray | None = None,
) -> dict:
    """Mean IoU of a split with the eight dihedral transforms averaged.

    Args:
        model: The segmentation model.
        images: ``(N, 4, H, W)`` uint16 reflectance x 10000.
        indices: Chips of the split.
        band_mean: Per-band training mean.
        band_std: Per-band training standard deviation.
        device: Torch device.
        batch_size: Images per pass.
        labels: ``(N, H, W)`` uint8 classes; required.

    Returns:
        Dict with ``mean_iou``, ``overall_accuracy``, ``iou`` and the
        confusion matrix.
    """
    if labels is None:
        raise ValueError("validate_tta needs the reference labels")
    reflectance = np.asarray(images[indices], dtype=np.float32) / 10000.0
    probabilities = predict_proba_dihedral(
        model, reflectance, band_mean, band_std, batch_size=batch_size, device=device
    )
    predicted = probabilities.argmax(axis=1).astype(np.uint8)
    matrix = confusion_matrix(predicted, np.asarray(labels[indices]).astype(np.uint8))
    return {
        "mean_iou": mean_iou(matrix),
        "overall_accuracy": overall_accuracy(matrix),
        "iou": iou_per_class(matrix),
        "confusion_matrix": matrix.tolist(),
    }


def git_state() -> dict:
    """Commit hash and dirty flag of the package repository."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(REPO), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def write_markdown(path: Path, run: dict) -> None:
    """Write the human-readable summary of the run."""
    best = run["best"]
    config = run["model"]
    decision = run["decision_rule"]
    lines = [
        f"# G4b — {run['model']['name']} sobre el pseudo-MSS",
        "",
        f"- Run: `{run['run_id']}` ({run['created_utc']})",
        f"- Arquitectura: {config['name']}, {config['parameters']:,} parámetros "
        f"(base {config['base_channels']}, depth {config['depth']}, "
        f"encoder {config['encoder']})",
        f"- Loss: {run['loss']['name']} + Dice {run['loss']['dice_weight']}; "
        f"pesos por clase {run['loss']['class_weight_mode']} "
        f"{[round(value, 3) for value in run['loss']['class_weights']]}",
        f"- Semilla: {run['seed']}; data: `{run['data']['fixture']}`",
        f"- Splits: train {run['split']['train']}, val {run['split']['validation']}, "
        f"test {run['split']['test']} (por escena CloudSEN12)",
        f"- Checkpoint: `{run['checkpoint']['path']}` "
        f"({run['checkpoint']['bytes']:,} bytes)",
        f"- SHA-256: `{run['checkpoint']['sha256']}`",
        "",
        f"Mejor época: {best['epoch']} — val mIoU argmax {best['mean_iou']:.4f}, "
        f"OA {best['overall_accuracy']:.4f}",
        f"Bias por clase ajustado en validación: {decision['bias']} → "
        f"val mIoU {decision['mean_iou']:.4f} (argmax {decision['baseline_mean_iou']:.4f})",
        "",
        "| Clase | IoU val argmax | IoU val con bias |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {name} | {best['iou'][name]:.4f} | {decision['iou'][name]:.4f} |"
        for name in CLASS_NAMES
    )
    lines += [
        "",
        "## Log por época",
        "",
        "| Época | lr | train loss | val loss | val mIoU | val OA | s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run["epochs"]:
        lines.append(
            f"| {row['epoch']} | {row['learning_rate']:.2e} | {row['train_loss']:.4f} | "
            f"{row['val_loss']:.4f} | {row['val_mean_iou']:.4f} | "
            f"{row['val_overall_accuracy']:.4f} | {row['seconds']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAIN["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=TRAIN["learning_rate"])
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--encoder", default=None, help="timm encoder name (e.g. resnet18)")
    parser.add_argument("--pretrained", action="store_true", help="ImageNet weights")
    parser.add_argument("--weight-decay", type=float, default=TRAIN["weight_decay"])
    parser.add_argument(
        "--augment", choices=tuple(AUGMENT_DESCRIPTION), default=TRAIN["augment"]
    )
    parser.add_argument("--loss", choices=("ce", "focal"), default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--class-weight-mode", choices=("sqrt", "inverse", "none"), default="sqrt"
    )
    parser.add_argument("--dice-weight", type=float, default=TRAIN["dice_weight"])
    parser.add_argument("--patience", type=int, default=TRAIN["patience"])
    parser.add_argument(
        "--thin-oversample",
        type=float,
        default=1.0,
        help="sampling weight multiplier for chips that contain thin cloud",
    )
    parser.add_argument(
        "--select-on",
        choices=("plain", "tta"),
        default="plain",
        help="validation mIoU used to keep the best epoch",
    )
    parser.add_argument("--no-tune-bias", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-threads", type=int, default=4, help="CPU threads when on CPU")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)

    t0 = time.perf_counter()
    images, labels, manifest = load_fixture(args.fixture)
    splits = split_indices(manifest)
    band_mean, band_std = band_statistics(images, splits["train"])
    counts = np.bincount(np.asarray(labels[splits["train"]]).ravel(), minlength=len(CLASS_NAMES))
    weights = class_weights(counts, mode=args.class_weight_mode).numpy()
    weights[counts == 0] = 0.0
    config = ModelConfig(
        base_channels=args.base_channels,
        depth=args.depth,
        dropout=args.dropout,
        encoder=args.encoder,
        pretrained=args.pretrained,
    )
    model = build_model(config).to(device)
    if args.init_checkpoint is not None:
        initial = load_checkpoint(args.init_checkpoint)
        if initial.config != config:
            raise SystemExit(
                f"init checkpoint architecture {initial.config} != requested {config}"
            )
        model.load_state_dict(initial.state_dict)
        print(f"initialized from {args.init_checkpoint} ({initial.sha256[:12]}…)")

    print(
        f"{len(splits['train'])} train / {len(splits['validation'])} val / "
        f"{len(splits['test'])} test chips; {count_parameters(model):,} parameters; "
        f"device={device}"
    )
    print("class weights:", np.round(weights, 3).tolist())

    criterion = SegLoss(
        num_classes=len(CLASS_NAMES),
        weight=torch.from_numpy(weights),
        gamma=args.focal_gamma if args.loss == "focal" else 0.0,
        dice_weight=args.dice_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=TRAIN["min_learning_rate"]
    )
    rng = np.random.default_rng(args.seed)
    sample_probabilities = sampling_probabilities(
        labels, splits["train"], args.thin_oversample
    )
    if sample_probabilities is not None:
        print(
            f"thin oversampling x{args.thin_oversample}: "
            f"{np.count_nonzero(sample_probabilities > 1.0 / len(splits['train']))} chips "
            f"with thin cloud"
        )

    best_score = -np.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict | None = None
    best_tta: dict | None = None
    epoch_log: list[dict] = []
    stale = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        if sample_probabilities is None:
            order = rng.permutation(splits["train"])
        else:
            order = rng.choice(
                splits["train"], size=len(splits["train"]), replace=True,
                p=sample_probabilities,
            )
        running_loss = 0.0
        running_ce = 0.0
        running_dice = 0.0
        n_seen = 0
        for start in range(0, len(order), args.batch_size):
            batch_index = order[start : start + args.batch_size]
            reflectance = np.asarray(images[batch_index], dtype=np.float32) / 10000.0
            batch_labels = np.asarray(labels[batch_index]).astype(np.uint8)
            reflectance, batch_labels = augment_batch(
                reflectance, batch_labels, rng, args.augment
            )
            batch_images = torch.from_numpy(
                (reflectance - band_mean.reshape(1, 4, 1, 1)) / band_std.reshape(1, 4, 1, 1)
            ).to(device)
            targets = torch.from_numpy(batch_labels.astype(np.int64)).to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, cross_entropy, dice = criterion(model(batch_images), targets)
            loss.backward()
            optimizer.step()

            n_seen += targets.numel()
            running_loss += float(loss) * targets.numel()
            running_ce += float(cross_entropy) * targets.numel()
            running_dice += float(dice) * targets.numel()
        scheduler.step()

        validation = validate(
            model,
            images,
            labels,
            splits["validation"],
            band_mean,
            band_std,
            criterion,
            device,
            args.batch_size,
        )
        tta_metrics = None
        if args.select_on == "tta":
            tta_metrics = validate_tta(
                model, images, splits["validation"], band_mean, band_std, device,
                args.batch_size, labels=labels,
            )
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": running_loss / max(n_seen, 1),
            "train_cross_entropy": running_ce / max(n_seen, 1),
            "train_dice": running_dice / max(n_seen, 1),
            "val_loss": validation["loss"],
            "val_mean_iou": validation["mean_iou"],
            "val_overall_accuracy": validation["overall_accuracy"],
            "val_iou": validation["iou"],
            "val_mean_iou_tta": None if tta_metrics is None else tta_metrics["mean_iou"],
            "seconds": time.perf_counter() - epoch_start,
        }
        epoch_log.append(row)
        score = validation["mean_iou"] if tta_metrics is None else tta_metrics["mean_iou"]
        print(
            f"epoch {epoch:3d} | lr {row['learning_rate']:.1e} | "
            f"train {row['train_loss']:.4f} | val {row['val_loss']:.4f} | "
            f"val mIoU {row['val_mean_iou']:.4f}"
            + (f" | tta {score:.4f}" if tta_metrics is not None else "")
            + f" | {row['seconds']:.1f}s"
        )

        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_metrics = validation
            best_tta = tta_metrics
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch} (patience {args.patience})")
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("training produced no checkpoint")

    model.load_state_dict(best_state)
    val_images = np.asarray(images[splits["validation"]], dtype=np.float32) / 10000.0
    val_labels = np.asarray(labels[splits["validation"]]).astype(np.uint8)
    val_probabilities = predict_proba_dihedral(
        model, val_images, band_mean, band_std, batch_size=args.batch_size, device=device
    )
    if args.no_tune_bias:
        decision = {
            "bias": [0.0] * len(CLASS_NAMES),
            "mean_iou": best_metrics["mean_iou"],
            "baseline_mean_iou": best_metrics["mean_iou"],
            "iou": best_metrics["iou"],
        }
    else:
        decision = tune_class_bias(val_probabilities, val_labels)
    if counts[2] == 0:
        decision["bias"][2] = -10.0
    predicted = apply_class_bias(val_probabilities, decision["bias"])
    matrix = confusion_matrix(predicted, val_labels)
    decision["confusion_matrix"] = matrix.tolist()
    decision["mean_iou"] = mean_iou(matrix)
    decision["iou"] = iou_per_class(matrix)
    decision["calibration_split"] = "validation"
    print(
        f"val bias {np.round(decision['bias'], 2).tolist()} → mIoU "
        f"{decision['mean_iou']:.4f} (argmax {decision['baseline_mean_iou']:.4f})"
    )

    fixture_json = json.loads((args.fixture / "fixture.json").read_text())
    elapsed = time.perf_counter() - t0
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    checkpoint_info = save_checkpoint(
        args.checkpoint,
        model,
        config,
        band_mean,
        band_std,
        metadata={
            "run_id": run_id,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "val_mean_iou": best_metrics["mean_iou"],
            "val_iou": best_metrics["iou"],
            "val_mean_iou_tta": None if best_tta is None else best_tta["mean_iou"],
            "class_bias": decision["bias"],
            "class_bias_val_mean_iou": decision["mean_iou"],
            "class_names": list(CLASS_NAMES),
            "loss": args.loss,
            "focal_gamma": args.focal_gamma if args.loss == "focal" else 0.0,
            "class_weight_mode": args.class_weight_mode,
            "augment": args.augment,
            "encoder": args.encoder,
            "pretrained": args.pretrained,
            "fixture_images_sha256": fixture_json.get("images_sha256"),
            "fixture_labels_sha256": fixture_json.get("labels_sha256"),
        },
    )

    run = {
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": elapsed,
        "seed": args.seed,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git": git_state(),
        "model": {
            "name": "TimmUNet" if config.encoder else "CloudUNet",
            "in_channels": config.in_channels,
            "num_classes": config.num_classes,
            "base_channels": config.base_channels,
            "depth": config.depth,
            "norm_groups": config.norm_groups,
            "encoder": config.encoder,
            "pretrained": config.pretrained,
            "parameters": count_parameters(model),
        },
        "loss": {
            "name": args.loss,
            "focal_gamma": args.focal_gamma if args.loss == "focal" else 0.0,
            "dice_weight": args.dice_weight,
            "class_weight_mode": args.class_weight_mode,
            "class_weights": weights.tolist(),
        },
        "data": {
            "fixture": str(args.fixture),
            "images_sha256": fixture_json.get("images_sha256"),
            "labels_sha256": fixture_json.get("labels_sha256"),
            "chip_size": fixture_json.get("chip_size_px"),
            "units": "reflectance x 10000 (uint16), divided by 10000 before the model",
        },
        "split": {name: len(indices) for name, indices in splits.items()},
        "normalisation": {
            "band_mean": band_mean.tolist(),
            "band_std": band_std.tolist(),
            "bands": ["green", "red", "nir1", "nir2"],
        },
        "train": {
            **TRAIN,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "augment": args.augment,
            "augment_description": AUGMENT_DESCRIPTION[args.augment],
            "dropout": args.dropout,
            "patience": args.patience,
            "select_on": args.select_on,
            "thin_oversample": args.thin_oversample,
        },
        "epochs": epoch_log,
        "best": {"epoch": best_epoch, **best_metrics},
        "best_tta": best_tta,
        "decision_rule": decision,
        "checkpoint": checkpoint_info,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run_train.json").write_text(json.dumps(run, indent=2) + "\n")
    write_markdown(args.out / "run_train.md", run)
    with (args.out / "train_log.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["epoch", "learning_rate", "train_loss", "val_loss", "val_mean_iou",
             "val_overall_accuracy", "val_mean_iou_tta", "seconds"]
        )
        for row in epoch_log:
            writer.writerow(
                [row["epoch"], f"{row['learning_rate']:.6g}", f"{row['train_loss']:.6f}",
                 f"{row['val_loss']:.6f}", f"{row['val_mean_iou']:.6f}",
                 f"{row['val_overall_accuracy']:.6f}",
                 "" if row["val_mean_iou_tta"] is None else f"{row['val_mean_iou_tta']:.6f}",
                 f"{row['seconds']:.2f}"]
            )

    print(
        f"best epoch {best_epoch}: val mIoU {best_metrics['mean_iou']:.4f} "
        f"({', '.join(f'{k} {v:.3f}' for k, v in best_metrics['iou'].items())})"
    )
    print(f"checkpoint {checkpoint_info['path']} sha256 {checkpoint_info['sha256']}")
    print(f"log {args.out / 'run_train.json'} ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
