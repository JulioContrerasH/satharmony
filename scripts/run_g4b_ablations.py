#!/usr/bin/env python
"""Run the G4b ablation grid on the pseudo-MSS validation split.

Every ablation trains on the *same* fixed scene split (the fixture manifest),
keeps the best epoch by validation mIoU (dihedral TTA) and tunes the per-class
bias on the same validation split.  The test split is never touched here: the
winner is evaluated once by ``scripts/evaluate_cloud_baseline.py``.

The grid isolates one lever at a time over the G4 baseline:

===========  ==========================================================
name         what changes
===========  ==========================================================
g4_ref       G4 baseline: 60 epochs, CE sqrt + Dice, no focal
b1_long      more epochs (120) on the same architecture and loss
b2_focal     focal (gamma 2) + inverse-frequency class weights
b3_wide      focal + wider encoder (base 64)
b4_deep      deeper encoder (base 48, depth 5), CE sqrt
b5_strong    strong augmentation (gamma, noise, wider gain), CE sqrt
b6_pretrain  focal + ImageNet ResNet-18 encoder
b7_thin      thin-cloud chip oversampling x4, CE sqrt
b8_wide_lowlr wider encoder (base 64) with lr 1e-3 and 150 epochs
===========  ==========================================================

Outputs (``--out``)::

    ablations.json       every number, one object per run
    ablations.md         sorted table for the report
    logs/<name>.log      stdout of each training run
    <name>.pt            checkpoints (``--checkpoint-dir``)

Usage
-----
    python scripts/run_g4b_ablations.py --jobs 3
    python scripts/run_g4b_ablations.py --only b2_focal b3_wide --jobs 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO / "scripts/train_cloud_baseline.py"
DEFAULT_OUT = REPO.parent / "assets/md/fuentes/g4b"
DEFAULT_CHECKPOINTS = Path("/data/databases/CLOUD_MSS/calibration/g4b")

#: name -> (description, CLI overrides)
ABLATIONS: dict[str, tuple[str, dict[str, object]]] = {
    "g4_ref": (
        "G4 baseline: U-Net 32/4, CE sqrt + Dice, 60 épocas",
        {
            "epochs": 60,
            "patience": 15,
            "loss": "ce",
            "class_weight_mode": "sqrt",
            "augment": "full",
            "base_channels": 32,
            "depth": 4,
            "select_on": "plain",
        },
    ),
    "b1_long": (
        "Más épocas: misma arquitectura y loss, 120 épocas",
        {
            "epochs": 120,
            "patience": 25,
            "loss": "ce",
            "class_weight_mode": "sqrt",
            "augment": "full",
            "base_channels": 32,
            "depth": 4,
            "select_on": "tta",
        },
    ),
    "b2_focal": (
        "Focal gamma 2 + pesos inversos (1/frecuencia)",
        {
            "epochs": 120,
            "patience": 25,
            "loss": "focal",
            "focal_gamma": 2.0,
            "class_weight_mode": "inverse",
            "augment": "full",
            "base_channels": 32,
            "depth": 4,
            "select_on": "tta",
        },
    ),
    "b3_wide": (
        "Focal + encoder más ancho (base 64)",
        {
            "epochs": 120,
            "patience": 25,
            "loss": "focal",
            "focal_gamma": 2.0,
            "class_weight_mode": "inverse",
            "augment": "full",
            "base_channels": 64,
            "depth": 4,
            "select_on": "tta",
        },
    ),
    "b4_deep": (
        "Encoder más profundo (base 48, depth 5), CE sqrt + Dice",
        {
            "epochs": 120,
            "patience": 25,
            "loss": "ce",
            "class_weight_mode": "sqrt",
            "augment": "full",
            "base_channels": 48,
            "depth": 5,
            "learning_rate": 1.5e-3,
            "select_on": "tta",
        },
    ),
    "b5_strong": (
        "Augment fuerte (gamma, ruido, ganancia amplia), CE sqrt + Dice",
        {
            "epochs": 120,
            "patience": 25,
            "loss": "ce",
            "class_weight_mode": "sqrt",
            "augment": "strong",
            "base_channels": 32,
            "depth": 4,
            "select_on": "tta",
        },
    ),
    "b7_thin": (
        "Oversampling de los chips con nube fina (x4), CE sqrt + Dice",
        {
            "epochs": 120,
            "patience": 25,
            "loss": "ce",
            "class_weight_mode": "sqrt",
            "augment": "full",
            "base_channels": 32,
            "depth": 4,
            "select_on": "tta",
            "thin_oversample": 4.0,
        },
    ),
    "b8_wide_lowlr": (
        "Encoder ancho (base 64) con lr 1e-3 y 150 épocas, CE sqrt + Dice",
        {
            "epochs": 150,
            "patience": 30,
            "loss": "ce",
            "class_weight_mode": "sqrt",
            "augment": "full",
            "base_channels": 64,
            "depth": 4,
            "learning_rate": 1e-3,
            "seed": 20261025,
            "select_on": "tta",
        },
    ),
    "b6_pretrain": (
        "Focal + encoder ResNet-18 con pesos ImageNet",
        {
            "epochs": 100,
            "patience": 25,
            "loss": "focal",
            "focal_gamma": 2.0,
            "class_weight_mode": "inverse",
            "augment": "full",
            "base_channels": 32,
            "depth": 4,
            "select_on": "tta",
            "encoder": "resnet18",
            "pretrained": True,
            "learning_rate": 5e-4,
        },
    ),
}


def build_command(name: str, overrides: dict[str, object], out: Path,
                  checkpoint_dir: Path, device: str) -> list[str]:
    """Translate an ablation entry into the training CLI."""
    command = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        "--out", str(out / "runs" / name),
        "--checkpoint", str(checkpoint_dir / f"{name}.pt"),
        "--device", device,
    ]
    for key, value in overrides.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        command.extend([flag, str(value)])
    return command


def summarise(name: str, description: str, run_dir: Path, seconds: float,
              returncode: int) -> dict:
    """Read ``run_train.json`` and flatten the numbers the report needs."""
    payload_path = run_dir / "run_train.json"
    result: dict = {
        "name": name,
        "description": description,
        "seconds": seconds,
        "returncode": returncode,
        "ok": False,
    }
    if not payload_path.exists():
        return result
    run = json.loads(payload_path.read_text())
    best = run["best"]
    decision = run["decision_rule"]
    result.update(
        {
            "ok": returncode == 0,
            "model": run["model"]["name"],
            "parameters": run["model"]["parameters"],
            "base_channels": run["model"]["base_channels"],
            "depth": run["model"]["depth"],
            "encoder": run["model"]["encoder"],
            "loss": run["loss"]["name"],
            "focal_gamma": run["loss"]["focal_gamma"],
            "class_weight_mode": run["loss"]["class_weight_mode"],
            "augment": run["train"]["augment"],
            "epochs_requested": run["train"]["epochs"],
            "epochs_run": len(run["epochs"]),
            "best_epoch": best["epoch"],
            "val_mean_iou": best["mean_iou"],
            "val_overall_accuracy": best["overall_accuracy"],
            "val_iou": best["iou"],
            "val_mean_iou_tta": (run.get("best_tta") or {}).get("mean_iou"),
            "val_iou_tta": (run.get("best_tta") or {}).get("iou"),
            "class_bias": decision["bias"],
            "val_mean_iou_bias": decision["mean_iou"],
            "val_iou_bias": decision["iou"],
            "checkpoint": run["checkpoint"]["path"],
            "checkpoint_sha256": run["checkpoint"]["sha256"],
            "elapsed_s": run["elapsed_s"],
        }
    )
    return result


def run_one(name: str, entry: tuple[str, dict[str, object]], out: Path,
            checkpoint_dir: Path, device: str) -> dict:
    """Train one ablation and return its summary."""
    description, overrides = entry
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(name, overrides, out, checkpoint_dir, device)
    print(f"[{name}] {' '.join(command)}", flush=True)
    start = time.perf_counter()
    with (log_dir / f"{name}.log").open("w") as handle:
        handle.write(f"$ {' '.join(command)}\n\n")
        handle.flush()
        completed = subprocess.run(
            command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    seconds = time.perf_counter() - start
    result = summarise(name, description, out / "runs" / name, seconds, completed.returncode)
    print(
        f"[{name}] rc={completed.returncode} val mIoU {result.get('val_mean_iou', float('nan')):.4f} "
        f"bias {result.get('val_mean_iou_bias', float('nan')):.4f} ({seconds:.0f}s)",
        flush=True,
    )
    return result


def write_markdown(path: Path, results: list[dict]) -> None:
    """Sorted table of the ablations."""
    ranked = sorted(
        [row for row in results if row.get("ok")],
        key=lambda row: row["val_mean_iou_bias"],
        reverse=True,
    )
    lines = [
        "# G4b — ablaciones (validación, split por escena fijo)",
        "",
        "El test no se toca: todo se elige con la validación (113 chips). La columna "
        "«mIoU bias» ajusta el sesgo por clase en la misma validación.",
        "",
        "| # | Ablación | mIoU val | mIoU val TTA | mIoU val + bias | Claro | Gruesa | Fina | Sombra | Params | Épocas |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for position, row in enumerate(ranked, start=1):
        iou_bias = row["val_iou_bias"]
        lines.append(
            f"| {position} | `{row['name']}` — {row['description']} | "
            f"{row['val_mean_iou']:.4f} | "
            f"{row['val_mean_iou_tta'] if row['val_mean_iou_tta'] is not None else float('nan'):.4f} | "
            f"**{row['val_mean_iou_bias']:.4f}** | "
            f"{iou_bias['clear']:.3f} | {iou_bias['thick_cloud']:.3f} | "
            f"{iou_bias['thin_cloud']:.3f} | {iou_bias['cloud_shadow']:.3f} | "
            f"{row['parameters']:,} | {row['best_epoch']}/{row['epochs_run']} |"
        )
    failed = [row for row in results if not row.get("ok")]
    if failed:
        lines += ["", "## Fallidas", ""]
        lines.extend(f"- `{row['name']}` (rc {row['returncode']})" for row in failed)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--only", nargs="+", choices=tuple(ABLATIONS), default=None)
    args = parser.parse_args()

    names = args.only or list(ABLATIONS)
    args.out.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(
                run_one, name, ABLATIONS[name], args.out, args.checkpoint_dir, args.device
            ): name
            for name in names
        }
        results.extend(future.result() for future in as_completed(futures))

    summary_path = args.out / "ablations.json"
    previous: dict[str, dict] = {}
    if summary_path.exists():
        for row in json.loads(summary_path.read_text()).get("results", []):
            previous[row["name"]] = row
    for row in results:
        previous[row["name"]] = row
    merged = sorted(previous.values(), key=lambda row: row["name"])
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": time.perf_counter() - started,
        "device": args.device,
        "split": "fixture manifest (train 1693 / validation 113 / test 194)",
        "selection": "best epoch by validation mIoU with dihedral TTA",
        "results": merged,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(args.out / "ablations.md", merged)
    print(f"wrote {args.out / 'ablations.md'} ({payload['elapsed_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
