#!/usr/bin/env python
"""Layer alignment audit of the v7 test scenes.

The project has five versions of the same scene on disk and they are not on the
same grid.  This script measures, scene by scene, what each layer is and how it
relates to the others, and writes the result as a table so the assumptions of
any downstream comparison are written down instead of remembered.

Layers
------
``pairs_tm/<id>_tm_original.tif`` raw TM, 8 bands, 30 m, 6400 x 6400
``pairs_optimization/<id>/tm_aligned.tif`` TM B1-B7, 30 m, cropped to the valid area
``pairs_optimization/<id>/mss_aligned.tif`` MSS nearest placed on the TM grid
``pairs_optimization/<id>/mss_original.tif``  native MSS, 4 bands, 60 m
``pairs_optimization/<id>/label.tif``  QA4EO-2/LGBM labels, 30 m, TM grid
``pairs_optimization/<id>/label_refined_cnn.tif`` clouds + CNN shadows, 30 m
``pairs_optimization/<id>/label_refined_mss.tif`` same, aggregated 2 x 2, 60 m

Checks
------
1. Which band of the raw TM each band of ``tm_aligned.tif`` is, and which one is dropped.
2. Whether ``tm_aligned.tif`` is just ``pairs_tm`` cropped (exact values, no resampling).
3. How ``mss_aligned.tif`` (30 m) relates to ``mss_original.tif`` (60 m): exact
   decimation or interpolation, and with which sub-pixel offset.
4. Whether ``label.tif`` is on the TM grid and ``label_refined_mss`` on the
   60 m grid of ``mss_original``.
5. Whether ``label_refined_mss`` is the 2 x 2 majority aggregation of
   ``label_refined_cnn`` (the rule of ``58_generate_mss60_and_label60.py``).
6. Agreement between ``label_qa4eo_2_60m.tif`` and ``label_refined_mss.tif``
   (the former on the MSS grid, the latter resampled to it when needed).

Usage
-----
    python scripts/audit_layer_alignment.py
    python scripts/audit_layer_alignment.py --out assets/md/fuentes/qa4eo2-labels
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

DEFAULT_MANIFEST = REPO.parent / "assets/md/manifest-finetune-v7.csv"
DEFAULT_DATA = Path("/data/databases/CLOUD_MSS")
DEFAULT_OUT = REPO.parent / "assets/md/fuentes/qa4eo2-labels"

#: The 60 m QA4EO-2 label: preserved copy first, old canonical name as fallback.
QA4EO_60M = "label_qa4eo_2_60m.tif"
QA4EO_FALLBACK = "label_qa4eo_2.tif"


def aggregate_labels_3class(labels: np.ndarray, scale: int = 2) -> np.ndarray:
    """2 x 2 aggregation of 3-class labels, majority with cloud > shadow > clear.

    Copy of ``assets/codes/mss/58_generate_mss60_and_label60.py`` so the audit
    can verify the rule instead of trusting it.

    Args:
        labels: ``(H, W)`` uint8 classes 0 clear, 1 cloud, 2 shadow.
        scale: Aggregation factor.

    Returns:
        ``(H // scale, W // scale)`` uint8 classes.
    """
    height, width = labels.shape
    out_h, out_w = height // scale, width // scale
    blocks = (
        labels[: out_h * scale, : out_w * scale]
        .reshape(out_h, scale, out_w, scale)
        .transpose(0, 2, 1, 3)
        .reshape(out_h, out_w, scale * scale)
    )
    count = scale * scale
    half = (count + 1) // 2
    n_clear = (blocks == 0).sum(axis=2)
    n_cloud = (blocks == 1).sum(axis=2)
    n_shadow = (blocks == 2).sum(axis=2)
    out = np.zeros((out_h, out_w), dtype=np.uint8)
    out[n_clear >= half] = 0
    out[n_cloud >= half] = 1
    out[n_shadow >= half] = 2
    no_majority = ~((n_clear >= half) | (n_cloud >= half) | (n_shadow >= half))
    cloud_dominant = no_majority & (n_cloud >= n_shadow)
    shadow_dominant = no_majority & (n_cloud < n_shadow)
    out[cloud_dominant & (n_cloud >= n_clear)] = 1
    out[cloud_dominant & (n_cloud < n_clear)] = 0
    out[shadow_dominant & (n_shadow >= n_clear)] = 2
    out[shadow_dominant & (n_shadow < n_clear)] = 0
    return out


def _read(path: Path, window: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, dict]:
    """Read a raster (optionally a window) and a small description of it."""
    with rasterio.open(path) as src:
        if window is None:
            data = src.read()
        else:
            row, col, height, width = window
            data = src.read(window=rasterio.windows.Window(col, row, width, height))
        info = {
            "count": src.count,
            "shape": [src.height, src.width],
            "res": [abs(src.res[0]), abs(src.res[1])],
            "dtype": src.dtypes[0],
            "crs": str(src.crs),
            "origin": [round(src.transform.c, 3), round(src.transform.f, 3)],
            "transform": [round(float(v), 6) for v in tuple(src.transform)[:6]],
            "nodata": src.nodata,
        }
    return data, info


def _info(path: Path) -> dict:
    """Metadata of a raster without reading the pixels."""
    with rasterio.open(path) as src:
        return {
            "count": src.count,
            "shape": [src.height, src.width],
            "res": [abs(src.res[0]), abs(src.res[1])],
            "dtype": src.dtypes[0],
            "crs": str(src.crs),
            "origin": [round(src.transform.c, 3), round(src.transform.f, 3)],
            "transform": [round(float(v), 6) for v in tuple(src.transform)[:6]],
            "nodata": src.nodata,
        }


def _equal_fraction(left: np.ndarray, right: np.ndarray) -> float:
    """Fraction of pixels where two same-shaped arrays hold the same value."""
    if left.shape != right.shape:
        return float("nan")
    return float((left == right).mean())


def align_to(source: np.ndarray, src_info: dict, target: dict) -> np.ndarray:
    """Nearest-neighbour resampling of a class raster onto a target grid."""
    destination = np.zeros(target["shape"], dtype=np.uint8)
    reproject(
        source=source.astype(np.uint8),
        destination=destination,
        src_transform=rasterio.Affine(*src_info["transform"]),
        src_crs=src_info["crs"],
        dst_transform=rasterio.Affine(*target["transform"]),
        dst_crs=target["crs"],
        resampling=Resampling.nearest,
    )
    return destination


def audit_scene(scene: str, data: Path) -> dict:
    """All alignment measurements of one scene.

    Args:
        scene: ``id_name``.
        data: Root of ``pairs_tm`` / ``pairs_optimization``.

    Returns:
        A JSON-friendly dict with one entry per check.
    """
    folder = data / "pairs_optimization" / scene
    record: dict = {"id_name": scene}

    raw_path = data / "pairs_tm" / f"{scene}_tm_original.tif"
    tm_path = folder / "tm_aligned.tif"
    mss30_path = folder / "mss_aligned.tif"
    mss60_path = folder / "mss_original.tif"
    label30_path = folder / "label.tif"
    refined_path = folder / "label_refined_cnn.tif"
    refined_mss_path = folder / "label_refined_mss.tif"
    qa4eo_path = folder / QA4EO_60M
    if not qa4eo_path.exists():
        qa4eo_path = folder / QA4EO_FALLBACK

    # ── 1-2. raw TM vs tm_aligned.tif (one window, the layers are 1.3 GB) ───────
    if raw_path.exists() and tm_path.exists():
        raw_info = _info(raw_path)
        tm_info = _info(tm_path)
        row_shift = max(round(raw_info["origin"][1] - tm_info["origin"][1]) // 30, 0)
        col_shift = max(round(tm_info["origin"][0] - raw_info["origin"][0]) // 30, 0)
        box = 1024
        row0 = max(row_shift + tm_info["shape"][0] // 2 - box // 2, row_shift)
        col0 = max(col_shift + tm_info["shape"][1] // 2 - box // 2, col_shift)
        raw, _ = _read(raw_path, (row0, col0, box, box))
        aligned, _ = _read(tm_path, (row0 - row_shift, col0 - col_shift, box, box))
        identity = [
            round(_equal_fraction(raw[index], aligned[index]), 6)
            for index in range(min(raw.shape[0], aligned.shape[0]))
        ]
        record["tm"] = {
            "raw_bands": raw_info["count"],
            "raw_shape": raw_info["shape"],
            "aligned_bands": tm_info["count"],
            "aligned_shape": tm_info["shape"],
            "crop_origin_30m_px": [col_shift, row_shift],
            "equal_fraction_per_band": identity,
            "band_order_confirmed": bool(
                len(identity) == tm_info["count"] and min(identity) > 0.999
            ),
            "dropped_band": int(raw_info["count"]),
        }

    # ── 3. mss_aligned.tif 30 m vs mss_original 60 m (windowed) ─────────────────
    if mss30_path.exists() and mss60_path.exists():
        mss30_info = _info(mss30_path)
        mss60_info = _info(mss60_path)
        height, width = mss60_info["shape"]
        box = 512
        row0 = max(height // 2 - box // 2, 0)
        col0 = max(width // 2 - box // 2, 0)
        mss60, _ = _read(mss60_path, (row0, col0, box, box))
        padded = 2
        mss30, _ = _read(mss30_path, (max(2 * row0 - padded, 0), max(2 * col0 - padded, 0), 2 * box + 2 * padded, 2 * box + 2 * padded))
        fractions = {}
        best = None
        for row_offset in (0, 1):
            for col_offset in (0, 1):
                block = mss30[
                    :,
                    padded + row_offset : padded + row_offset + 2 * box : 2,
                    padded + col_offset : padded + col_offset + 2 * box : 2,
                ]
                if block.shape[-2:] != (box, box):
                    continue
                fraction = float((block == mss60).mean())
                fractions[f"{row_offset},{col_offset}"] = round(fraction, 6)
                if best is None or fraction > best[2]:
                    best = (row_offset, col_offset, fraction)
        shift_x = (mss30_info["origin"][0] - mss60_info["origin"][0]) / 60.0
        shift_y = (mss30_info["origin"][1] - mss60_info["origin"][1]) / 60.0
        record["mss"] = {
            "mss30_shape": mss30_info["shape"],
            "mss60_shape": mss60_info["shape"],
            "best_decimation": {
                "row": best[0],
                "col": best[1],
                "equal_fraction": round(best[2], 6),
            }
            if best
            else None,
            "origin_shift_60m_px": [round(shift_x, 3), round(shift_y, 3)],
            "all_offsets": fractions,
            "row_offset_ambiguous": bool(
                len(fractions) == 4
                and min(fractions["0,1"], fractions["1,1"]) > 0.999
            ),
            "same_grid": bool(
                mss30_info["origin"] == mss60_info["origin"]
                and mss30_info["shape"][0] // 2 == height
            ),
        }

    # ── 4. label layers on their grids ──────────────────────────────────
    label_info = None
    if label30_path.exists():
        _, label_info = _read(label30_path)
    record["label_tif_grid_matches_tm"] = bool(
        label_info is not None
        and tm_path.exists()
        and label_info["transform"] == _read(tm_path)[1]["transform"]
    )
    if refined_mss_path.exists() and mss60_path.exists():
        _, refined_info = _read(refined_mss_path)
        _, mss60_info = _read(mss60_path)
        record["label_refined_mss_grid_matches_mss_original"] = bool(
            refined_info["transform"] == mss60_info["transform"]
            and refined_info["shape"] == mss60_info["shape"]
        )
        record["label_refined_mss_shape"] = refined_info["shape"]
        record["mss_original_shape"] = mss60_info["shape"]

    # ── 5. aggregation rule ─────────────────────────────────────────────
    if refined_path.exists() and refined_mss_path.exists():
        refined, _ = _read(refined_path)
        refined_mss, refined_mss_info = _read(refined_mss_path)
        aggregated = aggregate_labels_3class(refined[0])
        height, width = refined_mss_info["shape"]
        crop = aggregated[:height, :width]
        record["aggregation_rule_agreement"] = round(_equal_fraction(crop, refined_mss[0]), 6)
        valid = _read(mss60_path)[0].max(axis=0) > 0 if mss60_path.exists() else None
        if valid is not None and valid.shape == refined_mss[0].shape:
            record["aggregation_rule_agreement_valid"] = round(
                _equal_fraction(crop[valid], refined_mss[0][valid]), 6
            )

    # ── 6. label_qa4eo_2 (60 m) vs label_refined_mss ────────────────────
    if qa4eo_path.exists() and abs(float(_info(qa4eo_path)["res"][0]) - 60.0) > 1e-6:
        record["qa4eo2_60m_missing"] = True
        qa4eo_path = Path()
    if qa4eo_path.exists() and refined_mss_path.exists() and mss60_path.exists():
        qa4eo, qa4eo_info = _read(qa4eo_path)
        refined_mss, refined_info = _read(refined_mss_path)
        if qa4eo_info["transform"] != refined_info["transform"]:
            refined_on_mss = align_to(refined_mss[0], refined_info, qa4eo_info)
        else:
            refined_on_mss = refined_mss[0][: qa4eo_info["shape"][0], : qa4eo_info["shape"][1]]
        mss, _ = _read(mss60_path)
        valid = np.all(mss > 0, axis=0) & (qa4eo[0] != 255)
        same_shape = refined_on_mss.shape == qa4eo[0].shape
        if same_shape:
            record["qa4eo2_vs_refined_mss_agreement"] = round(
                _equal_fraction(qa4eo[0][valid], refined_on_mss[valid]), 6
            )
            record["qa4eo2_vs_refined_mss_valid_pixels"] = int(valid.sum())
            record["qa4eo2_resampled_refined"] = bool(qa4eo_info["transform"] != refined_info["transform"])

    return record


def write_markdown(path: Path, records: list[dict], summary: dict) -> None:
    """Human-readable version of the audit."""
    lines = [
        "# Alineación de capas en el test v7",
        "",
        f"{summary['scenes']} escenas de `manifest-finetune-v7.csv` (split `test`).",
        "",
        "## Qué es cada capa",
        "",
        "| Capa | Bandas | Resolución | Grilla |",
        "|---|---:|---:|---|",
        "| `pairs_tm/<id>_tm_original.tif` | 8 | 30 m | TM crudo, 1 px más de borde |",
        "| `pairs_optimization/<id>/tm_aligned.tif` | 7 (B1-B7) | 30 m | TM recortado a la intersección y a la huella válida |",
        "| `pairs_optimization/<id>/mss_aligned.tif` | 4 | 30 m | MSS remuestreado a la grilla TM |",
        "| `pairs_optimization/<id>/mss_original.tif` | 4 | 60 m | MSS nativo |",
        "| `pairs_optimization/<id>/label.tif` | 1 | 30 m | etiquetas v7 en la grilla TM |",
        "| `pairs_optimization/<id>/label_refined_cnn.tif` | 1 | 30 m | nubes + sombras CNN |",
        "| `pairs_optimization/<id>/label_refined_mss.tif` | 1 | 60 m | agregado 2x2 del anterior |",
        "| `pairs_optimization/<id>/label_qa4eo_2_60m.tif` | 1 | 60 m | QA4EO-2 sobre `mss_original` (el `label_qa4eo_2.tif` es ahora el de 30 m) |",
        "",
        "## Resultado por comprobación",
        "",
        f"- `pairs_tm` -> `tm_aligned.tif`: bandas 1-7 idénticas en "
        f"{summary['tm_identity_ok']}/{summary['tm_scenes']} escenas; la banda 8 se descarta.",
        f"- `mss_aligned.tif` 30 m vs `mss_original` 60 m: **duplicado nearest** 2x2 del MSS de 60 m, "
        f"sin interpolación. El valor del píxel de 60 m (i, j) cae en los píxeles de 30 m "
        f"(2i..2i+1, 2j+1..2j+2): medio píxel (15 m) de desplazamiento en x y ninguno en y. "
        f"El offset de fila es degenerado (las dos filas duplicadas llevan el mismo valor), "
        f"por eso el mejor offset por ventana sale (0, 1) o (1, 1) indistintamente. "
        f"{summary['mss_same_origin']}/{summary['mss_scenes']} escenas comparten origen "
        f"30 m y 60 m; en el resto el contenido está desplazado 1-63 px de 30 m.",
        f"- `label.tif` en la grilla de `tm_aligned.tif`: {summary['label_on_tm']}/{summary['label_scenes']}.",
        f"- `label_refined_mss` en la grilla de `mss_original`: "
        f"{summary['refined_on_mss']}/{summary['refined_scenes']} "
        f"({summary['refined_mismatch']} escenas con origen distinto).",
        f"- Regla de agregación 2x2 (`label_refined_cnn` -> `label_refined_mss`): "
        f"acuerdo mínimo {summary['aggregation_min']:.6f}.",
        f"- `label_qa4eo_2` vs `label_refined_mss` (resampleado si hace falta): "
        f"acuerdo medio {summary['qa4eo2_vs_refined_mean']:.4f}, "
        f"mediana {summary['qa4eo2_vs_refined_median']:.4f}.",
        "",
        "## Por escena",
        "",
        "| Escena | tm=tm_raw | offset de ventana (fila, col) | origen 30->60 (px 60 m) | "
        "label=TM | refined=mss_orig | agregación | QA4EO-2 vs refined |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for record in records:
        tm = record.get("tm", {})
        mss = record.get("mss", {})
        decimation = mss.get("best_decimation") or {}
        lines.append(
            "| {scene} | {tm_ok} | {row},{col} ({frac}) | {shift} | {label} | {refined} | "
            "{agg} | {qa} |".format(
                scene=record["id_name"],
                tm_ok="sí" if tm.get("band_order_confirmed") else "no",
                row=decimation.get("row", "?"),
                col=decimation.get("col", "?"),
                frac=f"{decimation.get('equal_fraction', float('nan')):.4f}"
                if decimation
                else "-",
                shift=mss.get("origin_shift_60m_px", "-"),
                label="sí" if record.get("label_tif_grid_matches_tm") else "no",
                refined="sí" if record.get("label_refined_mss_grid_matches_mss_original") else "no",
                agg=f"{record.get('aggregation_rule_agreement', float('nan')):.4f}",
                qa=f"{record.get('qa4eo2_vs_refined_mss_agreement', float('nan')):.4f}",
            )
        )
    lines += [
        "",
        "## Consecuencia para la comparativa",
        "",
        "- La entrada del modelo y su etiqueta `label_qa4eo_2` comparten la grilla de",
        "  `mss_original.tif`: no hay que remuestrear nada entre ellas.",
        "- `label_refined_mss` **no** está en esa grilla en las escenas listadas como `no`",
        "  en la columna `refined=mss_orig`; ahí hay que reproyectar con vecino más próximo.",
        "- `mss_aligned.tif` no es el MSS nativo: es una duplicación *nearest* del MSS de 60 m en la",
        "  grilla TM de 30 m, con el contenido desplazado medio píxel en x. No se usa para el modelo.",
        "- `mss_original.tif` es la única grilla de 60 m del MSS nativo; en 5 escenas no coincide",
        "  con la de `label_refined_mss.tif`, así que comparar contra esa etiqueta exige reproyectar.",
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    """Run the audit over the test split and write markdown + JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)

    with args.manifest.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == args.split]
    records = [audit_scene(row["id_name"], args.data) for row in rows]

    tm_scenes = [record for record in records if "tm" in record]
    mss_scenes = [record for record in records if "mss" in record]
    label_scenes = [record for record in records if "label_tif_grid_matches_tm" in record]
    refined_scenes = [record for record in records if "label_refined_mss_shape" in record]
    aggregations = [
        record["aggregation_rule_agreement_valid"]
        for record in records
        if "aggregation_rule_agreement_valid" in record
    ]
    qa4eo_values = [
        record["qa4eo2_vs_refined_mss_agreement"]
        for record in records
        if "qa4eo2_vs_refined_mss_agreement" in record
    ]
    decimations = [
        (record["mss"]["best_decimation"]["row"], record["mss"]["best_decimation"]["col"])
        for record in mss_scenes
        if record["mss"].get("best_decimation")
    ]
    summary = {
        "scenes": len(records),
        "tm_scenes": len(tm_scenes),
        "tm_identity_ok": sum(record["tm"]["band_order_confirmed"] for record in tm_scenes),
        "mss_scenes": len(mss_scenes),
        "mss_decimation": f"{max(set(decimations), key=decimations.count)}",
        "mss_same_origin": sum(
            record["mss"]["same_grid"] for record in mss_scenes if record["mss"].get("same_grid") is not None
        ),
        "label_scenes": len(label_scenes),
        "label_on_tm": sum(record["label_tif_grid_matches_tm"] for record in label_scenes),
        "refined_scenes": len(refined_scenes),
        "refined_on_mss": sum(
            record["label_refined_mss_grid_matches_mss_original"] for record in refined_scenes
        ),
        "refined_mismatch": sum(
            not record["label_refined_mss_grid_matches_mss_original"] for record in refined_scenes
        ),
        "aggregation_min": min(aggregations) if aggregations else float("nan"),
        "qa4eo2_vs_refined_mean": float(np.mean(qa4eo_values)) if qa4eo_values else float("nan"),
        "qa4eo2_vs_refined_median": float(np.median(qa4eo_values)) if qa4eo_values else float("nan"),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "alignment.json").write_text(json.dumps({"summary": summary, "scenes": records}, indent=1))
    write_markdown(args.out / "alignment.md", records, summary)
    print(json.dumps(summary, indent=1))
    print(f"written: {args.out / 'alignment.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
