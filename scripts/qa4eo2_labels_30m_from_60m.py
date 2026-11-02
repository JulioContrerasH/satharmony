#!/usr/bin/env python
"""30 m QA4EO-2 labels by upsampling the canonical 60 m product.

Why this exists
---------------
``scripts/qa4eo2_labels_30m.py`` ran the 4-class January QA4EO-2 on
``mss_aligned.tif``.  That model was trained at 60 m; on the 2 x 2 duplicate it
sees the same radiance with half the ground context and predicts cloud over
clear, bright scenes (note 27).  The 30 m grid has no information the 60 m
product does not have, so the defensible 30 m label is the 60 m label upsampled
2 x 2 nearest with the duplication shift measured per scene.

What is written
---------------
``label_qa4eo_2.tif`` on the grid of ``mss_aligned.tif``: one band, ``uint8``,
``0 clear``, ``1 cloud``, ``2 cloud_shadow``, ``255`` no data.  It is the
3-class canonical product, not the 4-class January model output.  The model
output is preserved as ``label_qa4eo_2_model30m.tif`` in the same folder.

The 60 m product (``label_qa4eo_2_60m.tif``) is never modified.  The model
manifests and the model comparison are preserved with a ``_model30m`` suffix.

Usage
-----
    python scripts/qa4eo2_labels_30m_from_60m.py --limit 1 --overwrite
    python scripts/qa4eo2_labels_30m_from_60m.py
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

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from qa4eo2_labels import (  # noqa: E402
    IGNORE,
    read_test_rows,
    sha256_file,
    valid_mask,
    write_label,
)
from qa4eo2_labels_30m import (  # noqa: E402
    align_shift,
    duplicate_purity,
    upsample_reference,
)

DEFAULT_MANIFEST = REPO.parent / "assets/md/manifest-finetune-v7.csv"
DEFAULT_DATA = Path("/data/databases/CLOUD_MSS")
DEFAULT_OUT = REPO.parent / "assets/md/fuentes/qa4eo2-labels"

LABEL_30M = "label_qa4eo_2.tif"
LABEL_60M = "label_qa4eo_2_60m.tif"
MODEL_30M = "label_qa4eo_2_model30m.tif"


def is_model_manifest(path: Path) -> bool:
    """True if the CSV is the 30 m model manifest (4 classes, no method)."""
    if not path.exists():
        return False
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False
    first = rows[0]
    return first.get("model_classes") == "4" and "method" not in first


def is_model_json(path: Path) -> bool:
    """True if the JSON summary is the 30 m model run (4 classes, no method)."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return data.get("model_classes") == 4 and "method" not in data


def load_model_shifts(out_dir: Path) -> dict[str, tuple[int, int, bool]]:
    """Measured duplication shift per scene from the 30 m model manifest."""
    candidates = [out_dir / "labels_manifest_model30m.csv", out_dir / "labels_manifest.csv"]
    for path in candidates:
        if not is_model_manifest(path):
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        shifts: dict[str, tuple[int, int, bool]] = {}
        for row in rows:
            if not row.get("align_dy") or not row.get("align_dx"):
                continue
            shifts[row["id_name"]] = (
                int(row["align_dy"]),
                int(row["align_dx"]),
                row.get("align_ok", "True") == "True",
            )
        return shifts
    return {}


def archive_model_outputs(out_dir: Path) -> None:
    """Preserve the manifests and comparison of the 30 m model run."""
    pairs = (
        ("labels_manifest.csv", "labels_manifest_model30m.csv"),
        ("labels_manifest.json", "labels_manifest_model30m.json"),
        ("comparison_60m_vs_30m.csv", "comparison_60m_vs_30m_model.csv"),
        ("verification_30m.json", "verification_30m_model.json"),
    )
    for source_name, target_name in pairs:
        source = out_dir / source_name
        target = out_dir / target_name
        if not source.exists() or target.exists():
            continue
        if source_name == "labels_manifest.csv" and not is_model_manifest(source):
            continue
        if source_name == "labels_manifest.json" and not is_model_json(source):
            continue
        source.rename(target)
        print(f"model output preserved as {target.name}")


def preserve_model_label(folder: Path) -> Path | None:
    """Move the 30 m model label aside before writing the upsampled one."""
    current = folder / LABEL_30M
    model = folder / MODEL_30M
    if model.exists():
        if current.exists() and sha256_file(current) == sha256_file(model):
            current.unlink()
        return model
    if not current.exists():
        return None
    with rasterio.open(current) as src:
        provenance = src.tags().get("provenance", "")
    if not provenance.endswith("qa4eo2_labels_30m.py"):
        return None
    current.rename(model)
    return model


def hint_from_transforms(mss_path: Path, original_path: Path) -> tuple[int, int]:
    """Integer 30 m offset between the aligned and original upper-left corners."""
    with rasterio.open(mss_path) as src:
        res = abs(float(src.res[0]))
        x30, y30 = float(src.transform.c), float(src.transform.f)
    with rasterio.open(original_path) as src:
        x60, y60 = float(src.transform.c), float(src.transform.f)
    return round((y30 - y60) / res), round((x30 - x60) / res)


def scene_shift(
    scene: str,
    mss_path: Path,
    original_path: Path,
    image: np.ndarray,
    shifts: dict[str, tuple[int, int, bool]],
) -> tuple[int, int, bool]:
    """Measured shift from the model manifest, or measured here if missing."""
    if scene in shifts:
        return shifts[scene]
    hint = hint_from_transforms(mss_path, original_path)
    with rasterio.open(original_path) as src:
        original = src.read().astype(np.float32) / 10000.0
    try:
        _, dy, dx = align_shift(image, original, hint=hint)
        return dy, dx, True
    except RuntimeError:
        return hint[0], hint[1], False


def verify_upsampled(
    label_path: Path, mss_path: Path, reference: np.ndarray
) -> tuple[dict[str, bool], int]:
    """Check an upsampled label against its MSS and its 60 m reference.

    The label is allowed to mark extra no data outside the 60 m footprint:
    those pixels have MSS data but no 60 m label, so a class would be invented.
    Every MSS no-data pixel must be ``IGNORE``, and every ``IGNORE`` pixel must
    be either MSS no data or 60 m no data.
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
    expected_ignore = (reference == IGNORE) | invalid
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
        "nodata_covers_mss": bool(np.all((label == IGNORE)[invalid])),
        "nodata_matches_reference": bool(np.array_equal(label == IGNORE, expected_ignore)),
        "classes_ok": bool(set(np.unique(label)).issubset({0, 1, 2, IGNORE})),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{label_path} failed {failed}")
    return checks, int(np.count_nonzero((label == IGNORE) & ~declared))


def main(argv: list[str] | None = None) -> int:
    """Write, verify and manifest the upsampled 30 m labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="first N scenes only (smoke test)")
    args = parser.parse_args(argv)

    rows = read_test_rows(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    shifts = load_model_shifts(args.out)
    if not args.limit:
        archive_model_outputs(args.out)

    records: list[dict] = []
    started = time.time()
    for number, row in enumerate(rows, 1):
        scene = row["id_name"]
        folder = args.data / "pairs_optimization" / scene
        mss_path = folder / "mss_aligned.tif"
        original_path = folder / "mss_original.tif"
        label_path = folder / LABEL_30M
        source_path = folder / LABEL_60M
        scene_started = time.time()

        model_path = preserve_model_label(folder)
        with rasterio.open(mss_path) as src:
            image = src.read().astype(np.float32) / 10000.0
            profile = src.profile.copy()
            shape = src.shape
        with rasterio.open(source_path) as src:
            label60 = src.read(1)
            label60_hash = sha256_file(source_path)

        dy, dx, align_ok = scene_shift(scene, mss_path, original_path, image, shifts)
        reference = upsample_reference(label60, dy, dx, shape)
        label = reference.copy()
        label[~valid_mask(image)] = IGNORE
        write_label(
            label_path,
            label,
            profile,
            tags={
                "method": "upsample_60m",
                "source_label": LABEL_60M,
                "source_label_sha256": label60_hash,
                "shift": f"{dy},{dx}",
                "align_ok": str(align_ok),
                "grid": "mss_aligned.tif, 30 m, same transform and shape",
                "classes": "0 clear, 1 cloud, 2 cloud_shadow, 255 nodata",
                "provenance": "scripts/qa4eo2_labels_30m_from_60m.py",
            },
        )

        checks, nodata_delta = verify_upsampled(label_path, mss_path, reference)
        valid = (label != IGNORE) & (reference != IGNORE)
        agreement = float((label[valid] == reference[valid]).mean()) if valid.any() else float("nan")
        purity = duplicate_purity(label, dy, dx)
        with rasterio.open(folder / "label.tif") as src:
            label_v7 = src.read(1)

        values, counts = np.unique(label, return_counts=True)
        counts = {int(value): int(count) for value, count in zip(values, counts, strict=True)}
        total = int(label.size)
        valid_pixels = total - counts.get(IGNORE, 0)
        records.append(
            {
                "id_name": scene,
                "split": row.get("split", ""),
                "category": row.get("category", ""),
                "method": "upsample_60m",
                "label_path": str(label_path.relative_to(args.data)),
                "label_sha256": sha256_file(label_path),
                "label_bytes": label_path.stat().st_size,
                "shape": list(label.shape),
                "dtype": "uint8",
                "nodata": IGNORE,
                "nodata_dataset_mask_delta": nodata_delta,
                "classes_present": " ".join(str(value) for value in sorted(np.unique(label))),
                "valid_fraction": round(valid_pixels / total, 6),
                "clear_fraction": round(counts.get(0, 0) / total, 6),
                "cloud_fraction_3c": round(counts.get(1, 0) / total, 6),
                "shadow_fraction": round(counts.get(2, 0) / total, 6),
                "source_label_path": str(source_path.relative_to(args.data)),
                "source_label_sha256": label60_hash,
                "model30_path": str(model_path.relative_to(args.data)) if model_path else "",
                "model30_sha256": sha256_file(model_path) if model_path else "",
                "align_dy": dy,
                "align_dx": dx,
                "align_ok": align_ok,
                "agreement_3c_with_60m": round(agreement, 6),
                "duplicate_purity": round(purity, 6),
                "grid_ok": all(checks.values()),
                "label_v7_all_zero": not bool(label_v7.any()),
                "elapsed_s": round(time.time() - scene_started, 1),
            }
        )
        print(
            f"[{number}/{len(rows)}] {scene} | agreement {agreement:.4f} | "
            f"purity {purity:.3f} | model30 {'yes' if model_path else 'no'} | "
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
        values = [
            float(row[key])
            for row in records
            if isinstance(row[key], (int, float)) and not np.isnan(float(row[key]))
        ]
        return round(float(np.mean(values)), 6) if values else float("nan")

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "upsample_60m",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "source": "label_qa4eo_2_60m.tif (3 classes, canonical 60 m product)",
        "grid": "mss_aligned.tif, 30 m, same transform and shape",
        "classes": "0 clear, 1 cloud, 2 cloud_shadow, 255 nodata",
        "scenes": len(records),
        "elapsed_s": round(time.time() - started, 1),
        "means": {
            "valid_fraction": mean("valid_fraction"),
            "clear_fraction": mean("clear_fraction"),
            "cloud_fraction_3c": mean("cloud_fraction_3c"),
            "shadow_fraction": mean("shadow_fraction"),
            "agreement_3c_with_60m": mean("agreement_3c_with_60m"),
            "duplicate_purity": mean("duplicate_purity"),
        },
        "rows": records,
    }
    (args.out / "labels_manifest.json").write_text(json.dumps(summary, indent=1))

    comparison_fields = [
        "id_name",
        "category",
        "source_label_sha256",
        "label_sha256",
        "agreement_3c_with_60m",
        "duplicate_purity",
        "align_dy",
        "align_dx",
        "align_ok",
        "label_v7_all_zero",
    ]
    with (args.out / "comparison_60m_vs_30m.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in comparison_fields} for row in records)

    verification = {
        "method": "upsample_60m",
        "scenes": len(records),
        "failed": [row["id_name"] for row in records if not row["grid_ok"]],
        "bytes_total": sum(int(row["label_bytes"]) for row in records),
        "unique_hashes": len({row["label_sha256"] for row in records}),
        "classes_union": sorted(
            {int(value) for row in records for value in row["classes_present"].split()}
        ),
        "nodata_beyond_dataset_mask_px": sum(
            int(row["nodata_dataset_mask_delta"]) for row in records
        ),
        "checks": [
            "shape",
            "transform",
            "crs",
            "res",
            "count",
            "dtype",
            "nodata",
            "nodata_covers_mss",
            "nodata_matches_reference",
            "classes",
            "sha",
            "bytes",
            "source",
            "tags",
        ],
    }
    (args.out / "verification_30m.json").write_text(json.dumps(verification, indent=1))

    print(
        f"done: {len(records)} labels | mean agreement {summary['means']['agreement_3c_with_60m']:.4f} | "
        f"purity {summary['means']['duplicate_purity']:.3f}"
    )
    print(f"manifest: {args.out / 'labels_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
