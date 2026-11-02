#!/usr/bin/env python
"""Visual comparison of the QA4EO-2 label and the two MSS cloud models.

Each page shows one scene with five panels: MSS false colour, TM natural
colour, the canonical QA4EO-2 3-class label, OmniCloudMask and the real-MSS G4
model.  The label panels use opaque CloudSEN12 colours; clear is transparent
and no data is grey.

The canonical QA4EO-2 product is 3-class: thin and thick cloud are merged into
``cloud``.  The 4-class January model is a different product and is not used
here, so the label panel is the same raster that the metrics are computed
against.

Usage
-----
    python scripts/visualize_g4_real_scenes.py --limit 1
    python scripts/visualize_g4_real_scenes.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from compare_qa4eo2_v7 import (  # noqa: E402
    DEFAULT_G4_REAL,
    OCM_BANDS,
    Predictor,
    ocm_mask,
    ocm_to_three,
    read_mss60,
)

DEFAULT_OUT = REPO.parent / "assets/md/fuentes/g4-real-viz"
DEFAULT_COMPARISON = REPO.parent / "assets/md/fuentes/qa4eo2-comparison-real/per_scene.csv"
DEFAULT_DATA = Path("/data/databases/CLOUD_MSS")

#: Best G4 real, worst G4 real and two scenes where one model wins clearly.
SCENES = (
    "1992-05-06_028_199",  # best G4 real 0.824
    "1987-05-08_025_192",  # best G4 real 0.757
    "1988-07-17_029_172",  # best G4 real 0.756
    "1992-07-15_002_024",  # worst G4 real 0.072
    "1985-07-05_041_192",  # worst G4 real 0.295
    "1987-05-02_018_182",  # worst G4 real 0.307 (OCM 0.500)
    "1993-03-15_022_182",  # OCM 1.000 vs G4 0.429
    "1989-06-18_026_188",  # G4 0.582 vs OCM 0.214
)

#: Canonical 3-class colours: cloud yellow, shadow red, clear transparent.
LABEL_COLORS_3 = ListedColormap(
    [(1, 1, 1, 0), (0.95, 0.80, 0.10, 1), (0.85, 0.15, 0.15, 1)]
)
LABEL_NORM_3 = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)


def stretch(rgb: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Per-band percentile stretch to uint8 RGB."""
    out = np.empty_like(rgb, dtype=np.float32)
    for index in range(rgb.shape[0]):
        band = rgb[index].astype(np.float32)
        lo, hi = np.percentile(band, (low, high))
        out[index] = np.clip((band - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (np.transpose(out, (1, 2, 0)) * 255).astype(np.uint8)


def resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an RGB uint8 array with bilinear interpolation."""
    return np.asarray(
        Image.fromarray(image).resize((size, size), resample=Image.Resampling.BILINEAR)
    ).copy()


def resize_label(label: np.ndarray, size: int) -> np.ndarray:
    """Resize a class map with nearest interpolation."""
    return np.asarray(
        Image.fromarray(label.astype(np.uint8)).resize(
            (size, size), resample=Image.Resampling.NEAREST
        )
    )


def read_tm_rgb(path: Path) -> np.ndarray:
    """TM natural colour: B3 red, B2 green, B1 blue."""
    with rasterio.open(path) as src:
        return src.read([3, 2, 1]).astype(np.float32)


def load_scene_metrics(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    """Per-scene metrics keyed by scene and model."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row["id_name"], {})[row["model"]] = row
    return out


def get_masks(
    scene: str,
    data: Path,
    out: Path,
    device: str,
    predictor: Predictor,
    recompute: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mss, label3, ocm3, g4real3) for one scene, caching the models."""
    mask_dir = out / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ocm": mask_dir / f"{scene}_ocm.npy",
        "g4real": mask_dir / f"{scene}_g4real.npy",
    }
    folder = data / "pairs_optimization" / scene
    mss, _ = read_mss60(folder / "mss_original.tif")
    with rasterio.open(folder / "label_qa4eo_2_60m.tif") as src:
        label3 = src.read(1)
    if label3.shape != mss.shape[-2:]:
        raise ValueError(f"{scene}: canonical label {label3.shape} != MSS {mss.shape[-2:]}")
    if recompute or not all(path.exists() for path in paths.values()):
        with rasterio.open(folder / "mss_original.tif") as src:
            image = src.read().astype(np.float32) / 10000.0
        ocm3 = ocm_to_three(ocm_mask(image[list(OCM_BANDS)], device))
        g4real3 = predictor.predict(image)
        np.save(paths["ocm"], ocm3.astype(np.uint8))
        np.save(paths["g4real"], g4real3.astype(np.uint8))
    return mss, label3, np.load(paths["ocm"]), np.load(paths["g4real"])


def panel(
    ax,
    image: np.ndarray,
    title: str,
    cmap=None,
    norm=None,
    mask: np.ndarray | None = None,
) -> None:
    """Draw one image panel without axes; mask is drawn in grey."""
    if cmap is None:
        ax.imshow(image, interpolation="nearest")
    else:
        data = np.ma.masked_where(mask, image) if mask is not None else image
        ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")
        if mask is not None and np.any(mask) and not np.all(mask):
            ax.contour(
                (~mask).astype(float), levels=[0.5], colors="0.25", linewidths=0.9
            )
    ax.set_title(title, fontsize=11.5, pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("0.15")
        spine.set_linewidth(1.1)


def page(
    scene: str,
    mss: np.ndarray,
    tm: np.ndarray,
    label3: np.ndarray,
    ocm3: np.ndarray,
    g4real3: np.ndarray,
    metrics: dict[str, dict[str, str]],
    size: int,
) -> plt.Figure:
    """Build the five-panel page of one scene."""
    mss_rgb = resize_rgb(stretch(mss[[2, 1, 0]]), size)
    tm_rgb = resize_rgb(stretch(tm), size)
    label3 = resize_label(label3, size)
    ocm3 = resize_label(ocm3, size)
    g4real3 = resize_label(g4real3, size)
    valid = resize_label(np.all(mss > 0, axis=0).astype(np.uint8), size).astype(bool)
    tm_valid = resize_label(np.all(tm > 0, axis=0).astype(np.uint8), size).astype(bool)
    mss_rgb[~valid] = 184
    tm_rgb[~tm_valid] = 184
    mask3 = (~valid) | (label3 == 255)
    mask_models = ~valid
    cmap = LABEL_COLORS_3.copy()
    cmap.set_bad((0.72, 0.72, 0.72, 1))

    g4 = metrics.get("g4_real", {})
    ocm = metrics.get("ocm_mss", {})
    g4_iou = float(g4.get("mean_iou", "nan"))
    ocm_iou = float(ocm.get("mean_iou", "nan"))
    g4_oa = float(g4.get("overall_accuracy", "nan"))
    ocm_oa = float(ocm.get("overall_accuracy", "nan"))

    def fraction(mask: np.ndarray, value: int) -> float:
        total = int(np.count_nonzero(mask != 255))
        return float(np.count_nonzero(mask == value) / total) if total else 0.0

    label_cloud = fraction(label3, 1)
    label_shadow = fraction(label3, 2)
    ocm_cloud = fraction(ocm3, 1)
    ocm_shadow = fraction(ocm3, 2)
    g4_cloud = fraction(g4real3, 1)
    g4_shadow = fraction(g4real3, 2)

    label_title = (
        f"QA4EO-2 3-class | cloud {label_cloud:.1%}, shadow {label_shadow:.1%}"
    )
    ocm_title = f"OmniCloudMask | cloud {ocm_cloud:.1%}, shadow {ocm_shadow:.1%}"
    g4_title = f"G4 real (MSS) | cloud {g4_cloud:.1%}, shadow {g4_shadow:.1%}"
    warning = ""
    if label_cloud + label_shadow < 0.001:
        warning = (
            "Canonical label is all clear: the mIoU is trivial. "
            "The label is the problem, not the models."
        )
    elif label_cloud + label_shadow > 0.95:
        warning = (
            "Canonical label is almost all cloud/shadow: the mIoU is trivial. "
            "The label is the problem, not the models."
        )

    fig, axes = plt.subplots(1, 5, figsize=(25, 5.6))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.77, bottom=0.10, wspace=0.03)
    panel(axes[0], mss_rgb, "MSS false colour (NIR-R-G)")
    panel(axes[1], tm_rgb, "TM natural colour (R-G-B)")
    panel(axes[2], label3, label_title, cmap, LABEL_NORM_3, mask3)
    panel(axes[3], ocm3, ocm_title, cmap, LABEL_NORM_3, mask_models)
    panel(axes[4], g4real3, g4_title, cmap, LABEL_NORM_3, mask_models)

    fig.suptitle(
        f"{scene} | G4 real mIoU {g4_iou:.3f} (OA {g4_oa:.3f}) | "
        f"OCM mIoU {ocm_iou:.3f} (OA {ocm_oa:.3f})",
        fontsize=14.5,
        y=0.985,
    )
    legend = [
        Patch(facecolor=(1, 1, 1), edgecolor="0.5", label="clear (transparent)"),
        Patch(facecolor=(0.95, 0.80, 0.10), edgecolor="0.4", label="cloud"),
        Patch(facecolor=(0.85, 0.15, 0.15), edgecolor="0.4", label="shadow"),
        Patch(facecolor=(0.72, 0.72, 0.72), edgecolor="0.4", label="no data"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.text(
        0.5,
        0.91,
        "Canonical QA4EO-2 3-class label: thin and thick are merged into cloud. "
        "Model panels are 3-class. Clear is transparent; no data is grey.",
        ha="center",
        fontsize=11.0,
        color="0.15",
    )
    if warning:
        fig.text(
            0.5,
            0.85,
            warning,
            ha="center",
            va="center",
            fontsize=12,
            color="#b00020",
            weight="bold",
            bbox={"facecolor": "white", "edgecolor": "#b00020", "boxstyle": "round,pad=0.3"},
        )
    return fig


def main(argv: list[str] | None = None) -> int:
    """Render the selected scenes to PDF and PNG pages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--size", type=int, default=900)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args(argv)

    scenes = SCENES[: args.limit] if args.limit else SCENES
    args.out.mkdir(parents=True, exist_ok=True)
    pages_dir = args.out / "pages"
    pages_dir.mkdir(exist_ok=True)
    metrics = load_scene_metrics(args.comparison)
    predictor = Predictor("g4_real", "unet", DEFAULT_G4_REAL, args.device)

    page_images: list[Path] = []
    pdf_path = args.out / "g4_real_scenes.pdf"
    for number, scene in enumerate(scenes, 1):
        folder = args.data / "pairs_optimization" / scene
        mss, label3, ocm3, g4real3 = get_masks(
            scene, args.data, args.out, args.device, predictor, args.recompute
        )
        tm = read_tm_rgb(folder / "tm_aligned.tif")
        fig = page(
            scene, mss, tm, label3, ocm3, g4real3, metrics.get(scene, {}), args.size
        )
        png_path = pages_dir / f"{number:02d}_{scene}.png"
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        page_images.append(png_path)
        print(f"[{number}/{len(scenes)}] {scene} -> {png_path.name}")

    pdf_pages = [Image.open(path).convert("RGB") for path in page_images]
    pdf_pages[0].save(
        pdf_path,
        save_all=True,
        append_images=pdf_pages[1:],
        resolution=110,
    )

    thumb_width = 1400
    thumbs = []
    for path in page_images:
        image = Image.open(path).convert("RGB")
        ratio = thumb_width / image.width
        thumbs.append(image.resize((thumb_width, int(image.height * ratio))))
    columns = 2
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumbs[0].height), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * thumb_width, (index // columns) * thumb.height))
    sheet.save(args.out / "contact_sheet.png")
    print(f"PDF: {pdf_path}")
    print(f"contact sheet: {args.out / 'contact_sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
