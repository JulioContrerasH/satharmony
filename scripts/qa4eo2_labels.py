#!/usr/bin/env python
"""QA4EO-2 labels on the native 60 m Landsat MSS grid.

What this script produces
-------------------------
``label_qa4eo_2.tif`` for every scene of the v7 test split, on the grid of
``mss_original.tif`` (4 bands, 60 m, TOA DN/10000).  That is the native MSS
grid and the grid the published QA4EO-2 was trained on (CloudSEN12 emulated to
60 m), so no resampling happens between the input and the label.

Classes: ``0 clear``, ``1 cloud``, ``2 cloud shadow``, ``255`` no data (the
four MSS bands are not all positive).  ``255`` is written as the raster's
``nodata`` value.

Which checkpoint
----------------
``QA4EO-2/unet.ckpt`` (sha256 ``bb8d3fa1...``) is byte-identical to
``finetune/unet.ckpt`` and to the ``unet.ckpt`` currently hosted in
``huggingface.co/isp-uv-es/QA4EO-2``.  It is **3-class** (clear, cloud,
shadow) with the ``encoder_lr_factor`` / ``freeze_bn`` hyper-parameters of
``finetune/model_finetune.py``.  The 4-class export of ``train/`` is
``train/mlstac_export/unet.ckpt`` (sha256 ``920ac77...``, epoch 55); it is the
older published version and is used as the *old QA4EO-2* row of the
comparison, not to generate the labels.

Input convention
----------------
Raw TOA 0-1, no z-score, ``merge_clouds=True``, no physical post-rules: the
same convention as ``assets/codes/mss/QA4EO-2/load.py``,
``assets/codes/mss/17_inference_all.py`` and
``assets/codes/mss/finetune/trainer_fig.py`` (the runs that produced the
first labels and the published Table 5).

Sliding window
--------------
512 x 512 tiles, Hann window of power 3, 50 % overlap, cloud probabilities
blended with the per-tile max (``cloud_max_alpha=0.6``), exactly the
``predict_large`` of ``QA4EO-2/load.py``.

Usage
-----
    python scripts/qa4eo2_labels.py --limit 2          # smoke test
    python scripts/qa4eo2_labels.py                    # 42 scenes
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from satharmony.models import load_model  # noqa: E402

DEFAULT_MANIFEST = REPO.parent / "assets/md/manifest-finetune-v7.csv"
DEFAULT_DATA = Path("/data/databases/CLOUD_MSS")
DEFAULT_QA4EO2 = Path("/data/users/julio/isp_clouds/mss/QA4EO-2/unet.ckpt")
DEFAULT_QA4EO2_JAN = Path("/data/users/julio/isp_clouds/mss/train/mlstac_export/unet.ckpt")
DEFAULT_OUT = REPO.parent / "assets/md/fuentes/qa4eo2-labels"

#: MSS band order in every ``pairs_optimization`` stack.
MSS_BANDS = ("green", "red", "nir1", "nir2")

#: Coarse classes of the product.
CLASS_NAMES = ("clear", "cloud", "cloud_shadow")

#: Value written for pixels where the MSS has no data.
IGNORE = 255


# ── IO ──────────────────────────────────────────────────────────────────
def read_mss60(path: Path) -> tuple[np.ndarray, dict]:
    """Read a 60 m MSS stack as physical TOA 0-1.

    Args:
        path: ``mss_original.tif`` (4 bands, uint16 DN, 60 m).

    Returns:
        ``((4, H, W) float32, profile)``; the profile is the rasterio profile
        of the source so the label inherits the exact grid.
    """
    with rasterio.open(path) as src:
        image = (src.read().astype(np.float32) / 10000.0)
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=IGNORE)
    return image, profile


def valid_mask(image: np.ndarray) -> np.ndarray:
    """Pixels with data in **all** MSS bands.

    Args:
        image: ``(C, H, W)`` reflectance.

    Returns:
        ``(H, W)`` boolean, True where every band is strictly positive.
    """
    return np.all(np.asarray(image) > 0, axis=0)


def write_label(path: Path, classes: np.ndarray, profile: dict, tags: dict | None = None) -> None:
    """Write a single-band uint8 label with ``255`` as no data.

    Args:
        path: Destination GeoTIFF.
        classes: ``(H, W)`` classes in ``{0, 1, 2, 255}``.
        profile: Profile of the source MSS (grid, CRS, transform).
        tags: Optional raster tags (model, hash, conventions).
    """
    profile = dict(profile)
    profile.update(count=1, dtype="uint8", nodata=IGNORE, compress="deflate", predictor=2)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(classes.astype(np.uint8), 1)
        if tags:
            dst.update_tags(**tags)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ── QA4EO-2 inference ───────────────────────────────────────────────────
def load_qa4eo2(path: Path, device: str) -> torch.nn.Module:
    """Build the U-Net of a QA4EO-2 checkpoint from its own state dict.

    The architecture is read from the weights (number of input channels and of
    output classes), so the 3-class March checkpoint and the 4-class January
    export both load without guessing.

    Args:
        path: Lightning ``.ckpt``.
        device: Torch device.

    Returns:
        The model in eval mode with gradients disabled.
    """
    import segmentation_models_pytorch as smp

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    head = state["model.segmentation_head.0.weight"]
    num_classes = int(head.shape[0])
    in_channels = int(state["model.encoder._conv_stem.weight"].shape[1])
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
        encoder_depth=5,
        activation=None,
        decoder_attention_type="scse",
    )
    model.load_state_dict({key.removeprefix("model."): value for key, value in state.items()})
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.num_classes = num_classes
    return model


def _hann(size: int, power: int = 3) -> np.ndarray:
    """2-D Hann window of ``size`` x ``size`` raised to ``power``."""
    one = np.hanning(size)
    return (np.outer(one, one) ** power).astype(np.float32)


def _tiles(
    height: int, width: int, chunk: int, step: int
) -> tuple[int, int, int, int, list[tuple[int, int]]]:
    """Padding and tile origins of a sliding window.

    Args:
        height: Image height.
        width: Image width.
        chunk: Tile size.
        step: Tile stride (``chunk - overlap``).

    Returns:
        ``(pad_top, pad_bottom, pad_left, pad_right, coords)`` with the four
        padding widths and the top-left corner of every tile in the padded
        image.
    """
    if height <= chunk:
        pad_top, pad_bottom = (chunk - height) // 2, chunk - height - (chunk - height) // 2
    else:
        extra = (step - (height - chunk) % step) % step
        pad_top, pad_bottom = extra // 2, extra - extra // 2
    if width <= chunk:
        pad_left, pad_right = (chunk - width) // 2, chunk - width - (chunk - width) // 2
    else:
        extra = (step - (width - chunk) % step) % step
        pad_left, pad_right = extra // 2, extra - extra // 2
    padded_h, padded_w = height + pad_top + pad_bottom, width + pad_left + pad_right
    coords = [
        (row, col)
        for row in range(0, padded_h - chunk + 1, step)
        for col in range(0, padded_w - chunk + 1, step)
    ]
    return pad_top, pad_bottom, pad_left, pad_right, coords


def predict_proba_qa4eo2(
    model: torch.nn.Module,
    image: np.ndarray,
    device: str,
    chunk: int = 512,
    overlap: int = 256,
    batch_size: int = 8,
    cloud_max_alpha: float = 0.6,
) -> np.ndarray:
    """Sliding-window probabilities, the ``predict_large`` of QA4EO-2.

    Args:
        model: Model from :func:`load_qa4eo2`.
        image: ``(4, H, W)`` reflectance, 0-1.
        device: Torch device.
        chunk: Tile size.
        overlap: Overlap between tiles in pixels.
        batch_size: Tiles per forward pass.
        cloud_max_alpha: Weight of the per-tile max for cloud classes.

    Returns:
        ``(num_classes, H, W)`` float32 probabilities.
    """
    classes = int(model.num_classes)
    num_classes_in = classes
    height, width = image.shape[-2:]
    step = chunk - overlap
    pad_top, pad_bottom, pad_left, pad_right, coords = _tiles(height, width, chunk, step)
    padded = np.pad(
        image,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="reflect",
    )
    _, padded_h, padded_w = padded.shape
    cloud_classes = [1] if num_classes_in == 3 else [1, 2]
    total = np.zeros((num_classes_in, padded_h, padded_w), dtype=np.float32)
    peak = np.zeros((num_classes_in, padded_h, padded_w), dtype=np.float32)
    weight = np.zeros((padded_h, padded_w), dtype=np.float32)
    window = _hann(chunk)

    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            batch = coords[start : start + batch_size]
            tiles = np.stack([padded[:, row : row + chunk, col : col + chunk] for row, col in batch])
            tensor = torch.from_numpy(tiles).float().to(device)
            probabilities = torch.softmax(model(tensor), dim=1).cpu().numpy()
            for index, (row, col) in enumerate(batch):
                window_slice = np.s_[:, row : row + chunk, col : col + chunk]
                total[window_slice] += probabilities[index] * window
                weight[row : row + chunk, col : col + chunk] += window
                peak[window_slice] = np.maximum(peak[window_slice], probabilities[index])

    weight = np.maximum(weight, 1e-8)
    blended = total / weight
    for index in cloud_classes:
        blended[index] = cloud_max_alpha * peak[index] + (1.0 - cloud_max_alpha) * blended[index]
    return blended[:, pad_top : pad_top + height, pad_left : pad_left + width]


def probabilities_to_three(probabilities: np.ndarray, merge_clouds: bool = True) -> np.ndarray:
    """Argmax of a QA4EO-2 probability volume, as clear / cloud / shadow.

    Args:
        probabilities: ``(C, H, W)`` probabilities, ``C`` 3 or 4.
        merge_clouds: Sum thin + thick clouds before the argmax.

    Returns:
        ``(H, W)`` uint8 with 0 clear, 1 cloud, 2 shadow.
    """
    if probabilities.shape[0] == 3:
        return probabilities.argmax(axis=0).astype(np.uint8)
    if not merge_clouds:
        raise ValueError("4-class output needs merge_clouds=True for the 3-class convention")
    merged = np.stack(
        [
            probabilities[0],
            probabilities[1] + probabilities[2],
            probabilities[3],
        ]
    )
    return merged.argmax(axis=0).astype(np.uint8)


def qa4eo2_classes(
    model: torch.nn.Module,
    image: np.ndarray,
    device: str,
    **kwargs,
) -> np.ndarray:
    """Classes of one scene, no-data pixels set to ``IGNORE``.

    Args:
        model: QA4EO-2 model.
        image: ``(4, H, W)`` reflectance.
        device: Torch device.
        **kwargs: Passed to :func:`predict_proba_qa4eo2`.

    Returns:
        ``(H, W)`` uint8 classes with 255 where the MSS has no data.
    """
    probabilities = predict_proba_qa4eo2(model, image, device, **kwargs)
    classes = probabilities_to_three(probabilities)
    classes[~valid_mask(image)] = IGNORE
    return classes


# ── satharmony U-Nets ───────────────────────────────────────────────────
def predict_proba_unet(
    model,
    checkpoint,
    image: np.ndarray,
    device: str,
    chunk: int = 512,
    overlap: int = 128,
    batch_size: int = 8,
    dihedral: bool = True,
    pad_half: bool = True,
    pad_offset: int = 0,
    window_power: int = 3,
    pad_offsets: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Sliding-window probabilities of a satharmony U-Net, with TTA.

    The model is fully convolutional but a whole 3200 x 3200 scene does not fit
    in memory at the first encoder level, so the scene is cut into tiles and
    blended with a Hann window, the same scheme used for QA4EO-2.

    Args:
        model: Model from :func:`satharmony.models.load_model`.
        checkpoint: Its ``LoadedCheckpoint`` (band mean/std).
        image: ``(3, H, W)`` reflectance, 0-1.
        device: Torch device.
        chunk: Tile size.
        overlap: Overlap between tiles.
        batch_size: Tiles per forward pass.
        dihedral: Average the eight dihedral transforms inside every tile.
        pad_half: Pad half a tile on every side so the scene border falls in
            the centre of the first/last tile instead of on the zero edge of
            the Hann window.
        pad_offset: Extra pixels added to the half-tile pad. It shifts the tile
            grid relative to the scene and is used to test whether the
            prediction depends on the tiling.
        window_power: Exponent of the 2-D Hann window. Lower values blend the
            tiles over a wider region; the default 3 is the sharpest.
        pad_offsets: If given, average the probabilities over these half-tile
            offsets. The tile grid moves relative to the scene and the seams
            average out. The default ``None`` uses only ``pad_offset``.

    Returns:
        ``(3, H, W)`` float32 probabilities.
    """
    if pad_offsets is not None and len(pad_offsets) > 1:
        accumulated = None
        for offset in pad_offsets:
            probabilities = predict_proba_unet(
                model,
                checkpoint,
                image,
                device,
                chunk=chunk,
                overlap=overlap,
                batch_size=batch_size,
                dihedral=dihedral,
                pad_half=pad_half,
                pad_offset=offset,
                window_power=window_power,
                pad_offsets=None,
            )
            accumulated = probabilities if accumulated is None else accumulated + probabilities
        return accumulated / len(pad_offsets)

    mean = np.asarray(checkpoint.band_mean, dtype=np.float32)
    std = np.asarray(checkpoint.band_std, dtype=np.float32)
    normalized = ((image - mean[:, None, None]) / std[:, None, None]).astype(np.float32)
    height, width = image.shape[-2:]
    step = chunk - overlap
    if pad_half:
        base = chunk // 2 + pad_offset
        padded = np.pad(
            normalized, ((0, 0), (base, base), (base, base)), mode="reflect"
        )
        extra_h = (step - (padded.shape[-2] - chunk) % step) % step
        extra_w = (step - (padded.shape[-1] - chunk) % step) % step
        padded = np.pad(
            padded, ((0, 0), (0, extra_h), (0, extra_w)), mode="reflect"
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
            normalized, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect"
        )
        crop_top, crop_left = pad_top, pad_left
    _, padded_h, padded_w = padded.shape
    number = int(model.config.num_classes) if hasattr(model, "config") else 3
    total = np.zeros((number, padded_h, padded_w), dtype=np.float32)
    weight = np.zeros((padded_h, padded_w), dtype=np.float32)
    window = _hann(chunk, power=window_power)

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
            tiles = np.stack([padded[:, row : row + chunk, col : col + chunk] for row, col in batch])
            probabilities = forward(torch.from_numpy(tiles).float().to(device)).cpu().numpy()
            for index, (row, col) in enumerate(batch):
                window_slice = np.s_[:, row : row + chunk, col : col + chunk]
                total[window_slice] += probabilities[index] * window
                weight[row : row + chunk, col : col + chunk] += window

    probabilities = total / np.maximum(weight, 1e-8)[None]
    return probabilities[
        :, crop_top : crop_top + height, crop_left : crop_left + width
    ]


def load_unet(path: Path, device: str):
    """Load a satharmony checkpoint and its metadata."""
    return load_model(path, device=device)


# ── manifest ────────────────────────────────────────────────────────────
def read_test_rows(manifest: Path, split: str = "test") -> list[dict[str, str]]:
    """Rows of one split of ``manifest-finetune-v7.csv``, in file order."""
    with manifest.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == split]
    if not rows:
        raise ValueError(f"{manifest} has no rows for split {split!r}")
    return rows


def main(argv: list[str] | None = None) -> int:
    """Write ``label_qa4eo_2.tif`` for every test scene and a hash manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--qa4eo2", type=Path, default=DEFAULT_QA4EO2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="first N scenes only (smoke test)")
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    rows = read_test_rows(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} scenes | device {args.device}")

    model = load_qa4eo2(args.qa4eo2, args.device)
    model_hash = sha256_file(args.qa4eo2)
    print(f"QA4EO-2 {args.qa4eo2} sha256 {model_hash[:12]}… {model.num_classes} classes")

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.time()
    for number, row in enumerate(rows, 1):
        scene = row["id_name"]
        folder = args.data / "pairs_optimization" / scene
        mss_path = folder / "mss_original.tif"
        label_path = folder / "label_qa4eo_2.tif"
        if label_path.exists() and not args.overwrite:
            print(f"[{number}/{len(rows)}] {scene} exists, skipped")
        else:
            image, profile = read_mss60(mss_path)
            classes = qa4eo2_classes(
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
                    "model": "QA4EO-2",
                    "checkpoint": str(args.qa4eo2),
                    "checkpoint_sha256": model_hash,
                    "input": "mss_original.tif 60 m, TOA DN/10000, 4 bands, raw (no z-score)",
                    "classes": "0 clear, 1 cloud, 2 cloud_shadow, 255 nodata",
                    "inference": f"sliding window {args.chunk}/{args.overlap}, hann^3, "
                    f"cloud_max_alpha 0.6, merge_clouds",
                    "provenance": "scripts/qa4eo2_labels.py",
                },
            )
            print(f"[{number}/{len(rows)}] {scene} written")

        with rasterio.open(label_path) as src:
            label = src.read(1)
            label_transform = [round(float(value), 6) for value in tuple(src.transform)[:6]]
            label_crs = str(src.crs)
        total = label.size
        values, counts = np.unique(label, return_counts=True)
        counts = {int(value): int(count) for value, count in zip(values, counts, strict=True)}
        records.append(
            {
                "id_name": scene,
                "split": row.get("split", ""),
                "bin": row.get("bin", ""),
                "label_path": str(label_path.relative_to(args.data)),
                "label_sha256": sha256_file(label_path),
                "label_bytes": label_path.stat().st_size,
                "shape": list(label.shape),
                "crs": label_crs,
                "transform": label_transform,
                "valid_fraction": round(1.0 - counts.get(IGNORE, 0) / total, 6),
                "clear_fraction": round(counts.get(0, 0) / total, 6),
                "cloud_fraction": round(counts.get(1, 0) / total, 6),
                "shadow_fraction": round(counts.get(2, 0) / total, 6),
                "model_sha256": model_hash,
                "model_classes": int(model.num_classes),
            }
        )

    (args.out / "labels_manifest.json").write_text(
        json.dumps(
            {
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "manifest": str(args.manifest),
                "manifest_sha256": sha256_file(args.manifest),
                "checkpoint": str(args.qa4eo2),
                "checkpoint_sha256": model_hash,
                "model_classes": int(model.num_classes),
                "grid": "mss_original.tif, 60 m, 4 bands, native MSS",
                "input": "TOA DN/10000, raw, no z-score",
                "inference": f"chunk {args.chunk}, overlap {args.overlap}, hann^3, "
                f"cloud_max_alpha 0.6, merge_clouds",
                "scenes": len(records),
                "elapsed_s": round(time.time() - started, 1),
                "rows": records,
            },
            indent=1,
        )
    )

    fields = list(records[0]) if records else []
    with (args.out / "labels_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    cloud = np.mean([row["cloud_fraction"] for row in records]) if records else float("nan")
    shadow = np.mean([row["shadow_fraction"] for row in records]) if records else float("nan")
    print(f"done: {len(records)} labels, mean cloud {cloud:.3f}, shadow {shadow:.3f}")
    print(f"manifest: {args.out / 'labels_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
