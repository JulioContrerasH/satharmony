#!/usr/bin/env python
"""Download the VNIR subset of the ECOSTRESS/ASTER spectral library.

The library is used only to fit the SBAF coefficients in ``fit_sbaf.py``.
It is not imported by the package.

Source
------
ECOSTRESS Spectral Library v1.0 (formerly ASTER Spectral Library v2.0)
https://speclib.jpl.nasa.gov/

The selection is a documented subset, not the full 6139-spectrum library:
soils, water/ice/snow, man-made materials, non-photosynthetic vegetation,
vegetation (grass/shrub/tree) and rocks (igneous/metamorphic/sedimentary).
Only spectra that cover 400-1100 nm are kept in the cleaned CSV.

Usage
-----
    python scripts/fetch_srf_library.py --out ../assets/docs/srf/library

The ``--out`` default is relative to the repository root.  Existing files are
not downloaded again.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
from pathlib import Path

import requests

BASE = "https://speclib.jpl.nasa.gov"
LIST_URL = f"{BASE}/ecospeclib_list"
DATA_URL = f"{BASE}/ecospeclibdata"

# category -> classes to harvest.  "All" plus the useful sub-classes; the
# endpoint returns at most 100 rows, so several class queries are needed for
# vegetation and rocks.
CATEGORIES = {
    "soil": ["All"],
    "water": ["All"],
    "manmade": ["All"],
    "nonphotosyntheticvegetation": [
        "All",
        "leaves",
        "bark",
        "branches",
        "flowers",
        "lichen",
        "needles",
    ],
    "vegetation": ["All", "grass", "shrub", "tree"],
    "rock": ["All", "igneous", "metamorphic", "sedimentary"],
}

VIS_MIN_NM = 400.0
VIS_MAX_NM = 1100.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def list_category(category: str, classes: list[str]) -> list[str]:
    names: set[str] = set()
    for cls in classes:
        form = {
            "searchtype": category,
            "classsel": cls,
            "subclass": "All",
            "mname": "",
            "xstart": "",
            "xstop": "",
            "maxhits": "100",
            "wavelength": "Any",
        }
        resp = requests.post(LIST_URL, data=form, timeout=120)
        resp.raise_for_status()
        names.update(re.findall(r"[a-zA-Z0-9._-]+\.spectrum\.txt", resp.text))
    return sorted(names)


def fetch_one(name: str, dest: Path) -> tuple[str, int]:
    path = dest / name
    if path.exists() and path.stat().st_size > 0:
        return name, path.stat().st_size
    resp = requests.get(f"{DATA_URL}/{name}", timeout=180)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return name, len(resp.content)


def parse_header(lines: list[str]) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key and len(key) < 40:
            header[key] = value.strip()
    return header


def parse_spectrum(path: Path) -> tuple[dict[str, str], list[tuple[float, float]]]:
    """Return (header, [(wavelength_nm, reflectance), ...])."""
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    # Metadata ends at the first blank line followed by a numeric row.
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) == 2:
            try:
                float(parts[0])
                float(parts[1])
            except ValueError:
                continue
            data_start = i
            break
    if data_start is None:
        raise ValueError(f"No numeric data in {path.name}")
    header = parse_header(lines[:data_start])
    rows: list[tuple[float, float]] = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if x <= 0 or x > 1000:
            continue
        rows.append((x * 1000.0, y))
    y_units = header.get("Y Units", "").lower()
    # Some headers omit the space after the colon ("Y Units:Reflectance
    # (percent)"), and a few spectra are already stored as fractions.
    y_max = max((abs(y) for _, y in rows), default=0.0)
    if "percent" in y_units or y_max > 2.0:
        rows = [(x, y / 100.0) for x, y in rows]
    # ascending wavelength
    rows.sort(key=lambda row: row[0])
    return header, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "assets/docs/srf/library",
        help="library root (default: <project>/assets/docs/srf/library)",
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root: Path = args.out
    raw = root / "ecostress" / "raw"
    manifest_dir = root / "manifests"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    lists: dict[str, list[str]] = {}
    for category, classes in CATEGORIES.items():
        names = list_category(category, classes)
        lists[category] = names
        print(f"{category:32s} {len(names):4d} spectra")

    all_names = sorted({name for names in lists.values() for name in names})
    print(f"unique spectra: {len(all_names)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda n: fetch_one(n, raw), all_names))
    print(f"downloaded {len(results)} files in {time.time() - started:.0f}s")

    manifest_rows: list[dict[str, object]] = []
    visible_rows: list[str] = []
    visible_rows.append("spectrum_id,wavelength_nm,reflectance\n")
    kept = 0
    for name in all_names:
        path = raw / name
        try:
            header, rows = parse_spectrum(path)
        except Exception as exc:  # report and skip malformed files
            print(f"skip {name}: {exc}")
            continue
        if not rows:
            continue
        waves = [row[0] for row in rows]
        covered = waves[0] <= VIS_MIN_NM + 5 and waves[-1] >= VIS_MAX_NM - 5
        in_range = [(x, y) for x, y in rows if VIS_MIN_NM <= x <= VIS_MAX_NM]
        finite = all(abs(y) < 1.0e30 for _, y in in_range)
        category = next(
            (cat for cat, names in lists.items() if name in names),
            "unknown",
        )
        manifest_rows.append(
            {
                "category": category,
                "filename": name,
                "type": header.get("Type", ""),
                "class": header.get("Class", ""),
                "subclass": header.get("Subclass", ""),
                "wavelength_range": header.get("Wavelength Range", ""),
                "wavelength_min_nm": round(min(waves), 2),
                "wavelength_max_nm": round(max(waves), 2),
                "n_points": len(rows),
                "vnir_covered": bool(covered and finite and len(in_range) > 50),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "url": f"{DATA_URL}/{name}",
            }
        )
        if covered and finite and len(in_range) > 50:
            kept += 1
            for x, y in in_range:
                visible_rows.append(f"{name},{x:.4f},{y:.6f}\n")

    manifest_path = manifest_dir / "ecostress_manifest.csv"
    fields = list(manifest_rows[0].keys())
    with manifest_path.open("w") as fh:
        fh.write(",".join(fields) + "\n")
        for row in manifest_rows:
            fh.write(",".join(str(row[f]) for f in fields) + "\n")

    visible_path = root / "ecostress" / "spectra_visible.csv"
    visible_path.write_text("".join(visible_rows))
    print(f"kept {kept}/{len(manifest_rows)} VNIR spectra -> {visible_path}")
    print(f"manifest -> {manifest_path}")

    metadata = {
        "source": "ECOSTRESS Spectral Library v1.0 (formerly ASTER v2.0)",
        "url": "https://speclib.jpl.nasa.gov/",
        "downloaded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "categories": {k: len(v) for k, v in lists.items()},
        "unique_spectra": len(all_names),
        "vnir_spectra": kept,
        "visible_csv": str(visible_path.relative_to(root)),
        "manifest_csv": str(manifest_path.relative_to(root)),
    }
    (manifest_dir / "ecostress_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
