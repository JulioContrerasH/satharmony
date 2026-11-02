#!/usr/bin/env python
"""Contact sheet of comparison 1, recomputed chip by chip.

The full comparison does not keep the predictions (7 models x 42 scenes x
10 Mpx would be 3 GB), so the sheet is redrawn from the same six chips chosen
by ``compare_qa4eo2_v7.pick_chips``.  Each chip is predicted on a 1024 x 1024
window and cropped to its centre, so the model sees context around the chip
and the result is visually the same as the full-scene sliding window.

Usage
-----
    python scripts/figure_qa4eo2_contact.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from compare_qa4eo2_v7 import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_G4,
    DEFAULT_G4_CAL,
    DEFAULT_G4B,
    DEFAULT_G4B_CAL,
    DEFAULT_LABELS,
    DEFAULT_MANIFEST,
    DEFAULT_OUT,
    DEFAULT_QA4EO2_JAN,
    Predictor,
    false_colour,
    pick_chips,
    read_test_rows,
)

#: Colour of the three classes plus no data.  Cloud is a pale blue instead of
#: white so it does not read as background; no data is a flat grey so a model
#: painting "cloud" over the MSS border cannot pass for a real detection.
LABEL_COLOURS = ["#2E7D32", "#B3E5FC", "#424242"]
NODATA_COLOUR = "#CFCFCF"
PALETTE = np.array(
    [
        [46, 125, 50],    # clear
        [179, 229, 252],  # cloud
        [66, 66, 66],     # cloud shadow
        [207, 207, 207],  # no data
    ],
    dtype=np.uint8,
)


def to_rgb(classes: np.ndarray, nodata: np.ndarray) -> np.ndarray:
    """Class map to an RGB image with no data in grey.

    Args:
        classes: ``(H, W)`` classes 0/1/2.
        nodata: ``(H, W)`` boolean, True where the MSS has no data.

    Returns:
        ``(H, W, 3)`` uint8 image.
    """
    index = np.clip(classes.astype(np.int16), 0, 2)
    index[nodata] = 3
    return PALETTE[index]

#: Column titles.
MODEL_TITLES = {
    "ocm_mss": "OmniCloudMask @ MSS 60 m",
    "qa4eo2_jan4": "QA4EO-2 January (4 classes)",
    "g4": "G4 U-Net",
    "g4b": "G4b U-Net",
    "g4_cal": "G4 U-Net calibrated",
    "g4b_cal": "G4b U-Net calibrated",
}

REGISTRY = {
    "ocm_mss": ("ocm", None),
    "qa4eo2_jan4": ("qa4eo2", DEFAULT_QA4EO2_JAN),
    "g4": ("unet", DEFAULT_G4),
    "g4b": ("unet", DEFAULT_G4B),
    "g4_cal": ("unet", DEFAULT_G4_CAL),
    "g4b_cal": ("unet", DEFAULT_G4B_CAL),
}


def read_window(path: Path, row: int, col: int, size: int) -> np.ndarray:
    """Read a square window with zero padding outside the raster."""
    with rasterio.open(path) as src:
        return (
            src.read(
                window=rasterio.windows.Window(col, row, size, size),
                boundless=True,
                fill_value=0,
            ).astype(np.float32)
            / 10000.0
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", nargs="+", default=list(REGISTRY))
    args = parser.parse_args(argv)

    rows = read_test_rows(args.manifest)
    chips = pick_chips(rows, args.data, args.labels, size=512, count=6)
    print(f"{len(chips)} chips")

    context = 1024
    pad = (context - 512) // 2
    for name in args.models:
        kind, path = REGISTRY[name]
        predictor = Predictor(name, kind, path, args.device)
        for chip in chips:
            row = max(chip["row"] - pad, 0)
            col = max(chip["col"] - pad, 0)
            image = read_window(args.data / "pairs_optimization" / chip["scene"] / "mss_original.tif", row, col, context)
            prediction = predictor.predict(image)
            offset_row = chip["row"] - row
            offset_col = chip["col"] - col
            chip.setdefault("masks", {})[name] = prediction[
                offset_row : offset_row + 512, offset_col : offset_col + 512
            ].copy()
        print(f"{name} done")
        del predictor

    columns = ["MSS false colour", "QA4EO-2 label", *[MODEL_TITLES[name] for name in args.models]]
    figure, axes = plt.subplots(
        len(chips),
        len(columns),
        figsize=(2.15 * len(columns), 2.1 * len(chips)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for index, chip in enumerate(chips):
        nodata = chip["reference"] == 255
        axes[index, 0].imshow(false_colour(chip["image"]))
        axes[index, 1].imshow(to_rgb(chip["reference"], nodata), interpolation="nearest")
        for column, name in enumerate(args.models, start=2):
            axes[index, column].imshow(
                to_rgb(chip["masks"][name], nodata), interpolation="nearest"
            )
        axes[index, 0].set_ylabel(chip["label"], fontsize=7.5)
        for column in range(len(columns)):
            axes[index, column].set_xticks([])
            axes[index, column].set_yticks([])
            for spine in axes[index, column].spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#888888")
            if index == 0:
                axes[index, column].set_title(columns[column], fontsize=7.5, pad=4)
    handles = [
        Patch(facecolor=LABEL_COLOURS[0], edgecolor="#888888", label="clear"),
        Patch(facecolor=LABEL_COLOURS[1], edgecolor="#888888", label="cloud"),
        Patch(facecolor=LABEL_COLOURS[2], edgecolor="#888888", label="cloud shadow"),
        Patch(facecolor=NODATA_COLOUR, edgecolor="#888888", label="no data"),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "figures" / "contact_sheet.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=150)
    plt.close(figure)
    print(f"written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
