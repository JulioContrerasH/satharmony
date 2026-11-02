#!/usr/bin/env python
"""Build the manifest of the 178 MSS/TM calibration pairs.

The manifest is the single table that ties every scene of
``calibration/mss_tm/selection.csv`` to the files it was built from:

* ``mss_path`` / ``tm_path`` are relative to ``/data/databases/CLOUD_MSS``;
* ``mss_sha256`` / ``tm_sha256`` pin the bytes actually read by the
  calibration scripts (``pairs_optimization/<id>/mss_original.tif`` and
  ``pairs_optimization/<id>/tm_aligned.tif``);
* ``pair_sha256`` is ``sha256(mss_sha256 + "\\n" + tm_sha256)`, so a single
  value identifies the pair without hashing both files again;
* ``cc`` comes from ``alignment_metrics.csv`` and ``min_valid`` /
  ``bad_band`` from ``mss_quality.csv``;
* ``split`` keeps the v7 split for the 79 scenes that already had one and
  fills the other 99 to reach 120 train / 30 val / 28 test, stratified by
  the cloud-fraction bin.  No v7 test scene moves into train.

The two scenes rejected by ``min_valid < 0.7`` are appended with
``status=discarded`` and their reason, so the manifest documents the
180 -> 178 cut.  The earlier visual/ECC rejects are kept in
``qa_rejected.csv``; the four that are still in ``selection.csv`` carry
their reason in ``qa_flag``.

Usage
-----
    python scripts/build_manifest_pairs.py
    python scripts/build_manifest_pairs.py --workers 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CLOUD_MSS = Path("/data/databases/CLOUD_MSS")
DEFAULT_CALIBRATION = DEFAULT_CLOUD_MSS / "calibration/mss_tm"
DEFAULT_OLD_MANIFEST = PROJECT / "assets/md/manifest-finetune-v7.csv"
DEFAULT_OUT = PROJECT / "assets/md/metadata/manifest-finetune-v7.csv"

#: Exact split targets agreed in the 10/10 plan (120 / 30 / 28), stratified
#: by the cloud-fraction bin of ``selection.csv``.
SPLIT_TARGETS = {
    "0-2": (40, 10, 10),
    "2-10": (19, 5, 4),
    "10-30": (20, 5, 5),
    "30-60": (20, 5, 5),
    "60-90": (21, 5, 4),
}
SPLIT_ORDER = ("train", "val", "test")
BIN_ORDER = ("0-2", "2-10", "10-30", "30-60", "60-90")
SPLIT_SEED = 20261024

FIELDS = [
    "id_name",
    "status",
    "split",
    "split_source",
    "bin",
    "cloud_frac",
    "qa_cloud",
    "qa_shadow",
    "cc",
    "cc_ok",
    "min_valid",
    "bad_band",
    "mss_path",
    "mss_sha256",
    "mss_bytes",
    "tm_path",
    "tm_sha256",
    "tm_bytes",
    "pair_sha256",
    "discard_reason",
    "qa_flag",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a small CSV as a list of dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in 8 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_digest(mss_sha: str, tm_sha: str) -> str:
    """Stable digest of the pair built from the two file digests."""
    return hashlib.sha256(f"{mss_sha}\n{tm_sha}".encode()).hexdigest()


def cloud_bin(cloud_frac: float) -> str:
    """Cloud-fraction bin used by ``selection.csv``."""
    for low, high, name in (
        (-0.001, 0.02, "0-2"),
        (0.02, 0.10, "2-10"),
        (0.10, 0.30, "10-30"),
        (0.30, 0.60, "30-60"),
        (0.60, 0.90, "60-90"),
    ):
        if low <= cloud_frac < high:
            return name
    raise ValueError(f"cloud fraction outside the calibration bins: {cloud_frac}")


def relative_to(root: Path, path: Path) -> str:
    """POSIX path relative to ``root``, the form stored in the manifest."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def assign_splits(rows: list[dict[str, object]], old_splits: dict[str, str]) -> None:
    """Assign ``split`` in place, preserving v7 splits and filling the rest."""
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["bin"])].append(row)

    rng = random.Random(SPLIT_SEED)
    for bin_name in BIN_ORDER:
        group = sorted(groups[bin_name], key=lambda row: str(row["id_name"]))
        counts: Counter[str] = Counter()
        pending = []
        for row in group:
            old = old_splits.get(str(row["id_name"]), "")
            if old in SPLIT_ORDER:
                row["split"] = old
                row["split_source"] = "v7"
                counts[old] += 1
            else:
                pending.append(row)

        needs = {
            split: SPLIT_TARGETS[bin_name][index] - counts[split]
            for index, split in enumerate(SPLIT_ORDER)
        }
        if any(value < 0 for value in needs.values()):
            raise ValueError(f"bin {bin_name}: v7 split already exceeds the target: {needs}")
        if sum(needs.values()) != len(pending):
            raise ValueError(
                f"bin {bin_name}: {len(pending)} new scenes but {sum(needs.values())} slots"
            )

        rng.shuffle(pending)
        start = 0
        for split in SPLIT_ORDER:
            for row in pending[start : start + needs[split]]:
                row["split"] = split
                row["split_source"] = f"new-seed-{SPLIT_SEED}"
            start += needs[split]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud-mss", type=Path, default=DEFAULT_CLOUD_MSS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--old-manifest", type=Path, default=DEFAULT_OLD_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    selection = read_csv(args.calibration / "selection.csv")
    alignment = {row["id_name"]: row for row in read_csv(args.calibration / "alignment_metrics.csv")}
    quality = {row["id_name"]: row for row in read_csv(args.calibration / "mss_quality.csv")}
    rejected = {row["id_name"]: row for row in read_csv(args.calibration / "qa_rejected_tmqa.csv")}
    qa_previous = {row["id_name"]: row for row in read_csv(args.calibration / "qa_rejected.csv")}
    old_splits = (
        {
            row["id_name"]: row["split"]
            for row in read_csv(args.old_manifest)
            if row.get("split") in SPLIT_ORDER
        }
        if args.old_manifest.exists()
        else {}
    )

    rows: list[dict[str, object]] = []
    for record in selection:
        scene = record["id_name"]
        metrics = alignment[scene]
        stats = quality[scene]
        previous = qa_previous.get(scene)
        rows.append(
            {
                "id_name": scene,
                "status": "kept",
                "bin": record["bin"],
                "cloud_frac": record["cloud_frac"],
                "qa_cloud": record["qa_cloud"],
                "qa_shadow": record["qa_shadow"],
                "cc": metrics["cc"],
                "cc_ok": str(float(metrics["cc"]) >= 0.85),
                "min_valid": stats["min_valid"],
                "bad_band": stats["bad_band"],
                "mss_path": relative_to(
                    args.cloud_mss, args.cloud_mss / "pairs_optimization" / scene / "mss_original.tif"
                ),
                "tm_path": relative_to(
                    args.cloud_mss, args.cloud_mss / "pairs_optimization" / scene / "tm_aligned.tif"
                ),
                "discard_reason": "",
                "qa_flag": previous["reason"] if previous else "",
                "notes": "en qa_rejected.csv; sigue en selection.csv" if previous else "",
            }
        )

    for scene, record in rejected.items():
        stats = quality.get(scene, {})
        previous = qa_previous.get(scene)
        rows.append(
            {
                "id_name": scene,
                "status": "discarded",
                "split": "",
                "split_source": "",
                "bin": cloud_bin(float(record["cloud_frac"])),
                "cloud_frac": record["cloud_frac"],
                "qa_cloud": "",
                "qa_shadow": "",
                "cc": alignment.get(scene, {}).get("cc", ""),
                "cc_ok": "",
                "min_valid": record["min_valid"],
                "bad_band": stats.get("bad_band", ""),
                "mss_path": relative_to(
                    args.cloud_mss, args.cloud_mss / "pairs_optimization" / scene / "mss_original.tif"
                ),
                "tm_path": relative_to(
                    args.cloud_mss, args.cloud_mss / "pairs_optimization" / scene / "tm_aligned.tif"
                ),
                "discard_reason": f"min_valid={float(record['min_valid']):.3f} < 0.7",
                "qa_flag": previous["reason"] if previous else "",
                "notes": "reemplazo del mismo bin ya incluido en selection.csv",
            }
        )

    kept = [row for row in rows if row["status"] == "kept"]
    if len(kept) != 178:
        raise SystemExit(f"expected 178 kept pairs, found {len(kept)}")
    assign_splits(kept, old_splits)

    def hash_pair(row: dict[str, object]) -> dict[str, object]:
        mss = args.cloud_mss / str(row["mss_path"])
        tm = args.cloud_mss / str(row["tm_path"])
        for path in (mss, tm):
            if not path.exists():
                raise SystemExit(f"missing file: {path}")
        row["mss_sha256"] = sha256_file(mss)
        row["mss_bytes"] = mss.stat().st_size
        row["tm_sha256"] = sha256_file(tm)
        row["tm_bytes"] = tm.stat().st_size
        row["pair_sha256"] = pair_digest(str(row["mss_sha256"]), str(row["tm_sha256"]))
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(hash_pair, rows))

    split_rank = {split: rank for rank, split in enumerate(("", *SPLIT_ORDER))}
    bin_rank = {bin_name: rank for rank, bin_name in enumerate(BIN_ORDER)}
    rows.sort(
        key=lambda row: (
            row["status"] != "kept",
            split_rank[str(row["split"])],
            bin_rank[str(row["bin"])],
            str(row["id_name"]),
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(str(row["split"]) for row in kept)
    cc_ok = sum(str(row["cc_ok"]) == "True" for row in kept)
    total_bytes = sum(int(row["mss_bytes"]) + int(row["tm_bytes"]) for row in rows)
    print(f"manifest: {args.out}")
    print(f"rows: {len(rows)} ({len(kept)} kept, {len(rows) - len(kept)} discarded)")
    print(f"splits: {dict(counts)}")
    print(f"cc >= 0.85: {cc_ok}/{len(kept)}")
    print(f"hashed bytes: {total_bytes / 1e9:.2f} GB")
    print(f"v7 split preserved: {sum(str(row['split_source']) == 'v7' for row in kept)}")
    if counts != Counter({"train": 120, "val": 30, "test": 28}):
        raise SystemExit(f"unexpected split counts: {counts}")


if __name__ == "__main__":
    main()
