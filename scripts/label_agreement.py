#!/usr/bin/env python
"""Measure agreement between the two human annotators.

Two annotation paths are supported and can be used together:

* **Masks** (``mascaras/anotador1/<chip_id>.png`` and ``anotador2``): 128 x
  128 PNGs whose pixel value is the class (0 clear, 1 thick, 2 thin, 3
  shadow, 254 unlabelled, 255 doubtful).  The script computes pixel-wise
  Cohen's kappa and per-class IoU, ignoring 254/255.
* **CSV** (``anotaciones_anotador1.csv`` and ``_2``): per-chip dominant
  class and estimated class percentages.  The script computes kappa on the
  dominant class and the mean absolute error of the percentages.

The script never invents labels.  With empty templates it reports that the
agreement is pending and exits 0.  The gate in the plan is kappa >= 0.7.

Usage
-----
    python scripts/label_agreement.py --labels-dir ../assets/md/fuentes/labels-humano
    python scripts/label_agreement.py --labels-dir ... --masks
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CLASS_NAMES = ("clear", "thick", "thin", "shadow")
UNLABELLED = 254
DOUBTFUL = 255
FRACTION_COLUMNS = {
    "clear": "pct_clear",
    "thick": "pct_thick",
    "thin": "pct_thin",
    "shadow": "pct_shadow",
}


def cohen_kappa(first, second) -> float:
    """Cohen's kappa for two label sequences.

    Args:
        first: Sequence of hashable labels.
        second: Sequence of the same length.

    Returns:
        Kappa in [-1, 1]; NaN when both annotators used a single identical
        class (chance agreement is 1, kappa undefined).

    Raises:
        ValueError: If the lengths differ or the sequences are empty.
    """
    first = list(first)
    second = list(second)
    if len(first) != len(second):
        raise ValueError(f"length mismatch: {len(first)} != {len(second)}")
    if not first:
        raise ValueError("empty label sequences")

    labels = sorted(set(first) | set(second))
    index = {label: i for i, label in enumerate(labels)}
    n = len(first)
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for a, b in zip(first, second, strict=True):
        matrix[index[a], index[b]] += 1

    observed = np.trace(matrix) / n
    expected = float(matrix.sum(axis=0) @ matrix.sum(axis=1)) / (n * n)
    if np.isclose(expected, 1.0):
        return float("nan")
    return float((observed - expected) / (1.0 - expected))


def pixel_kappa(first: np.ndarray, second: np.ndarray) -> dict:
    """Pixel-wise kappa and per-class IoU for two class masks.

    Args:
        first: 2D integer mask.
        second: 2D integer mask of the same shape.

    Returns:
        Dictionary with ``kappa``, ``n_pixels``, ``agreement`` and ``iou``
        keyed by class name.  Pixels equal to 254 or 255 in either mask are
        ignored.

    Raises:
        ValueError: If the shapes differ.
    """
    if first.shape != second.shape:
        raise ValueError(f"mask shapes differ: {first.shape} != {second.shape}")
    keep = (first != UNLABELLED) & (second != UNLABELLED)
    keep &= (first != DOUBTFUL) & (second != DOUBTFUL)
    a = first[keep].ravel()
    b = second[keep].ravel()
    if a.size == 0:
        return {"kappa": float("nan"), "n_pixels": 0, "agreement": float("nan"), "iou": {}}

    iou = {}
    for code, name in enumerate(CLASS_NAMES):
        predicted, truth = (a == code), (b == code)
        union = int(np.count_nonzero(predicted | truth))
        intersection = int(np.count_nonzero(predicted & truth))
        iou[name] = intersection / union if union else float("nan")

    return {
        "kappa": cohen_kappa(a, b),
        "n_pixels": int(a.size),
        "agreement": float(np.mean(a == b)),
        "iou": iou,
    }


def read_annotation_csv(path: Path) -> dict[str, dict]:
    """Read one annotator CSV, keeping only the rows with a class."""
    rows = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            chip_id = (row.get("chip_id") or "").strip()
            dominant = (row.get("clase_dominante") or "").strip().lower()
            if not chip_id or not dominant:
                continue
            fractions = {}
            for name, column in FRACTION_COLUMNS.items():
                value = (row.get(column) or "").strip()
                fractions[name] = float(value) if value else float("nan")
            rows[chip_id] = {"dominant": dominant, "fractions": fractions}
    return rows


def validate_fractions(rows: dict[str, dict], tol: float = 2.0) -> list[str]:
    """Return the chips whose percentages do not add up to 100."""
    problems = []
    for chip_id, row in rows.items():
        values = [v for v in row["fractions"].values() if np.isfinite(v)]
        if values and abs(sum(values) - 100.0) > tol:
            problems.append(f"{chip_id}: percentages sum to {sum(values):.0f}")
    return problems


def compare_csv(first: dict[str, dict], second: dict[str, dict]) -> dict:
    """Kappa on the dominant class and fraction MAE for two CSV annotations."""
    common = sorted(set(first) & set(second))
    if not common:
        return {"n_chips": 0}

    pairs = [
        (first[chip]["dominant"], second[chip]["dominant"])
        for chip in common
        if "dudoso" not in (first[chip]["dominant"], second[chip]["dominant"])
    ]
    kappa = cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]) if pairs else float("nan")

    errors = []
    per_class = {}
    for name in CLASS_NAMES:
        differences = [
            abs(first[chip]["fractions"][name] - second[chip]["fractions"][name])
            for chip in common
            if np.isfinite(first[chip]["fractions"].get(name, np.nan))
            and np.isfinite(second[chip]["fractions"].get(name, np.nan))
        ]
        per_class[name] = float(np.mean(differences)) if differences else float("nan")
        errors.extend(differences)

    return {
        "n_chips": len(common),
        "n_chips_kappa": len(pairs),
        "kappa": kappa,
        "fraction_mae": float(np.mean(errors)) if errors else float("nan"),
        "fraction_mae_per_class": per_class,
        "sum_check": {
            "anotador1": validate_fractions(first),
            "anotador2": validate_fractions(second),
        },
    }


def load_masks(folder: Path) -> dict[str, np.ndarray]:
    """Load every PNG mask in a folder as a 2D uint8 array."""
    from PIL import Image

    masks = {}
    if not folder.exists():
        return masks
    for path in sorted(folder.glob("*.png")):
        with Image.open(path) as image:
            masks[path.stem] = np.asarray(image.convert("L"), dtype=np.uint8)
    return masks


def compare_masks(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict:
    """Pooled pixel kappa and per-class IoU over the common chips."""
    common = sorted(set(first) & set(second))
    if not common:
        return {"n_chips": 0}
    pooled_first, pooled_second = [], []
    kappas = []
    intersection = dict.fromkeys(CLASS_NAMES, 0)
    union = dict.fromkeys(CLASS_NAMES, 0)
    n_pixels = 0
    agreement = 0.0
    for chip in common:
        result = pixel_kappa(first[chip], second[chip])
        if result["n_pixels"] == 0:
            continue
        kappas.append(result["kappa"])
        n_pixels += result["n_pixels"]
        agreement += result["agreement"] * result["n_pixels"]
        keep = (first[chip] != UNLABELLED) & (second[chip] != UNLABELLED)
        keep &= (first[chip] != DOUBTFUL) & (second[chip] != DOUBTFUL)
        pooled_first.append(first[chip][keep].ravel())
        pooled_second.append(second[chip][keep].ravel())
        for code, name in enumerate(CLASS_NAMES):
            union[name] += int(
                np.count_nonzero(
                    ((first[chip] == code) | (second[chip] == code)) & keep
                )
            )
            intersection[name] += int(
                np.count_nonzero((first[chip] == code) & (second[chip] == code))
            )
    if n_pixels == 0:
        return {"n_chips": 0}
    finite = [value for value in kappas if np.isfinite(value)]
    pooled = cohen_kappa(
        np.concatenate(pooled_first), np.concatenate(pooled_second)
    )
    return {
        "n_chips": len(common),
        "n_pixels": n_pixels,
        "kappa": float(pooled),
        "kappa_median_per_chip": float(np.median(finite)) if finite else float("nan"),
        "kappa_per_chip_iqr": (
            float(np.percentile(finite, 75) - np.percentile(finite, 25))
            if finite
            else float("nan")
        ),
        "agreement": agreement / n_pixels,
        "iou": {
            name: (intersection[name] / union[name] if union[name] else float("nan"))
            for name in CLASS_NAMES
        },
    }


def write_report(path: Path, report: dict) -> None:
    """Write the agreement report as markdown."""
    lines = ["# Acuerdo entre anotadores", ""]
    has_data = any(
        (report.get(section) or {}).get("n_chips") for section in ("masks", "csv")
    )
    if not has_data:
        lines += [
            "Pendiente: todavía no hay anotaciones.",
            "",
            "Rellena `anotaciones_anotador1.csv` y `anotaciones_anotador2.csv`, o deja",
            "las máscaras en `mascaras/anotador1/` y `mascaras/anotador2/`, y vuelve a",
            "correr `scripts/label_agreement.py`.",
            "",
        ]
        path.write_text("\n".join(lines))
        return

    for section, title in (("masks", "Máscaras (píxel a píxel)"), ("csv", "CSV (clase dominante)")):
        data = report.get(section)
        if not data or not data.get("n_chips"):
            continue
        lines += [f"## {title}", ""]
        if section == "masks":
            lines += [
                f"- Chips: {data['n_chips']}",
                f"- Píxeles válidos: {data['n_pixels']}",
                f"- Kappa (píxeles agrupados): {data['kappa']:.3f}",
                f"- Kappa por chip (mediana): {data['kappa_median_per_chip']:.3f}",
                f"- Acuerdo bruto: {data['agreement']:.3f}",
                "- IoU por clase: " + ", ".join(
                    f"{name} {value:.3f}" for name, value in data["iou"].items()
                ),
            ]
        else:
            lines += [
                f"- Chips: {data['n_chips']} (kappa sobre {data['n_chips_kappa']})",
                f"- Kappa (clase dominante): {data['kappa']:.3f}",
                f"- MAE de porcentajes: {data['fraction_mae']:.1f} p.p.",
            ]
            for name, value in data["fraction_mae_per_class"].items():
                lines.append(f"- MAE {name}: {value:.1f} p.p.")
            problems = data["sum_check"]["anotador1"] + data["sum_check"]["anotador2"]
            if problems:
                lines += ["", "Porcentajes que no suman 100:", ""]
                lines += [f"- {problem}" for problem in problems]
        lines.append("")

    kappa = report.get("masks", {}).get("kappa", report.get("csv", {}).get("kappa"))
    if kappa is not None and np.isfinite(kappa):
        verdict = "cumple" if kappa >= 0.7 else "no cumple"
        lines += [f"**Gate kappa ≥ 0.7: {verdict} ({kappa:.3f}).**", ""]
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--masks", action="store_true", help="also compare PNG masks")
    parser.add_argument("--first", default="anotador1")
    parser.add_argument("--second", default="anotador2")
    args = parser.parse_args()

    labels_dir = args.labels_dir
    report: dict = {"labels_dir": str(labels_dir)}

    csv_first = labels_dir / f"anotaciones_{args.first}.csv"
    csv_second = labels_dir / f"anotaciones_{args.second}.csv"
    if csv_first.exists() and csv_second.exists():
        report["csv"] = compare_csv(
            read_annotation_csv(csv_first), read_annotation_csv(csv_second)
        )

    if args.masks:
        report["masks"] = compare_masks(
            load_masks(labels_dir / "mascaras" / args.first),
            load_masks(labels_dir / "mascaras" / args.second),
        )

    if not any(report.get(section, {}).get("n_chips") for section in ("csv", "masks")):
        print("pending: no annotations found; nothing to compare")
        write_report(labels_dir / "acuerdo.md", report)
        (labels_dir / "acuerdo.json").write_text(json.dumps(report, indent=2) + "\n")
        return 0

    (labels_dir / "acuerdo.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    write_report(labels_dir / "acuerdo.md", report)
    if "masks" in report and report["masks"].get("n_chips"):
        print(f"masks: n={report['masks']['n_chips']} kappa={report['masks']['kappa']:.3f}")
    if "csv" in report and report["csv"].get("n_chips"):
        print(f"csv: n={report['csv']['n_chips']} kappa={report['csv']['kappa']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
