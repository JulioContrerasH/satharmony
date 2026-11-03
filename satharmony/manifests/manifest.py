"""Manifest of calibration pairs with provenance and hashes.

The same schema covers the two calibration families:

* ``kind="s2_mss"``: ``input_path`` is a 13-band Sentinel-2 scene and
  ``reference_path`` the 4-band MSS scene on the output grid.
* ``kind="mss_tm"``: ``input_path`` is MSS and ``reference_path`` is TM.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REQUIRED_COLUMNS = (
    "id",
    "kind",
    "input_path",
    "reference_path",
    "input_sha256",
    "reference_sha256",
    "cloud_fraction",
    "reason",
)

Kind = Literal["s2_mss", "mss_tm"]


@dataclass(frozen=True)
class ManifestRecord:
    """One calibration pair and the hashes that pin its inputs."""

    id: str
    kind: Kind
    input_path: Path
    reference_path: Path
    input_sha256: str
    reference_sha256: str
    cloud_fraction: float
    reason: str


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    """Read and validate a CSV manifest.

    Args:
        path: CSV with the columns in :data:`REQUIRED_COLUMNS`.

    Returns:
        The records, with relative paths resolved against the CSV directory.

    Raises:
        ValueError: If a required column is missing or a field is empty.
    """
    manifest = Path(path)
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        records = []
        for row in reader:
            empty = [key for key in REQUIRED_COLUMNS if not str(row[key]).strip()]
            if empty:
                raise ValueError(f"manifest row {row.get('id', '?')} has empty {empty}")
            if row["kind"] not in ("s2_mss", "mss_tm"):
                raise ValueError(f"unknown kind {row['kind']!r} in row {row['id']}")
            records.append(
                ManifestRecord(
                    id=row["id"],
                    kind=row["kind"],  # type: ignore[arg-type]
                    input_path=(manifest.parent / row["input_path"]).resolve(),
                    reference_path=(manifest.parent / row["reference_path"]).resolve(),
                    input_sha256=row["input_sha256"],
                    reference_sha256=row["reference_sha256"],
                    cloud_fraction=float(row["cloud_fraction"]),
                    reason=row["reason"],
                )
            )
    return records


def verify_manifest(
    records: list[ManifestRecord], check_hashes: bool = True
) -> list[str]:
    """Return one message per missing file or hash mismatch."""
    problems: list[str] = []
    for record in records:
        for label, path, expected in (
            ("input", record.input_path, record.input_sha256),
            ("reference", record.reference_path, record.reference_sha256),
        ):
            if not path.exists():
                problems.append(f"{record.id}: {label} file not found: {path}")
                continue
            if check_hashes and sha256_file(path) != expected:
                problems.append(f"{record.id}: {label} hash mismatch")
    return problems


def write_manifest(records: list[ManifestRecord], path: str | Path) -> None:
    """Write the records to CSV with a stable column order."""
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record.id,
                    "kind": record.kind,
                    "input_path": record.input_path,
                    "reference_path": record.reference_path,
                    "input_sha256": record.input_sha256,
                    "reference_sha256": record.reference_sha256,
                    "cloud_fraction": record.cloud_fraction,
                    "reason": record.reason,
                }
            )
