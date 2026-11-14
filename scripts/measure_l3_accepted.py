#!/usr/bin/env python
"""Medir el perfil de L3 publicado sobre el conjunto aceptado completo.

No reajusta nada: coge ``mss3_esa_gtc_v2`` tal como se publica y lo evalúa en
los 24 sitios aceptados (6 del banco viejo + 18 de Tier 2). Reserva cada 4º
sitio para que el número de reserva no sea el de un subconjunto distinto.

    python scripts/measure_l3_accepted.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.emulators.s2_to_mss.config import PipelineConfig  # noqa: E402
from scripts.holdout_v2 import deterministic  # noqa: E402
from scripts.make_profile_aux import BANDS, site_metrics  # noqa: E402

OLD_OK = [
    "l3_025.1E_18.7N",
    "l3_033.9E_30.2N",
    "l3_038.4E_31.6N",
    "l3_096.9W_59.8N",
    "l3_106.7E_43.0N",
    "l3_107.8W_31.6N",
]
NEW_OK = [
    "l3_142.9E_27.3S",
    "l3_041.4E_35.9N",
    "l3_103.3W_27.3N",
    "l3_107.8W_31.6N",
    "l3_113.5W_31.6N",
    "l3_085.4E_50.1N",
    "l3_107.6E_41.6N",
    "l3_142.3E_24.4S",
    "l3_036.2E_17.3N",
    "l3_106.8W_34.5N",
    "l3_117.1W_38.8N",
    "l3_078.5E_47.3N",
    "l3_032.9E_15.9N",
    "l3_139.4E_24.4S",
    "l3_102.2W_25.9N",
    "l3_002.5E_25.9N",
    "l3_139.0E_26.0S",
    "l3_140.5E_26.0S",
]


def main() -> int:
    names = sorted(set(OLD_OK) | set(NEW_OK))
    print(f"{len(names)} sitios aceptados", flush=True)
    config = deterministic(PipelineConfig.from_profile("mss3_esa_gtc_v2"))
    metrics = {name: site_metrics("L3", name, config) for name in names}
    usable = [
        name
        for name in names
        if metrics[name]["masked"] >= 0.9 and all(np.isfinite(metrics[name][b]) for b in BANDS)
    ]
    print(f"{len(usable)} con máscara >= 90 %", flush=True)
    held = [name for index, name in enumerate(usable) if index % 4 == 3]
    train = [name for name in usable if name not in held]

    def median(rows: list[str]) -> dict[str, float]:
        return {band: float(np.nanmedian([metrics[name][band] for name in rows])) for band in BANDS}

    def median_scaled(rows: list[str]) -> dict[str, float]:
        return {
            band: float(np.nanmedian([metrics[name][f"{band}_scaled"] for name in rows]))
            for band in BANDS
        }

    in_sample = median(train)
    held_out = median(held)
    print(f"train {len(train)}: " + ", ".join(f"{b} {100 * v:.1f} %" for b, v in in_sample.items()))
    print(f"reserva {len(held)}: " + ", ".join(f"{b} {100 * v:.1f} %" for b, v in held_out.items()))
    print("con factor libre por sitio: "
          + ", ".join(f"{b} {100 * v:.1f} %" for b, v in median_scaled(train).items()))

    payload = {
        "profile": "mss3_esa_gtc_v2",
        "sites_accepted": names,
        "sites_used": usable,
        "train": train,
        "held_out": held,
        "in_sample_rel_rmse": in_sample,
        "held_out_rel_rmse": held_out,
        "in_sample_rel_rmse_free_scale": median_scaled(train),
        "held_out_rel_rmse_free_scale": median_scaled(held),
        "per_site": {name: dict(metrics[name]) for name in names},
        "method": "shipped profile, deterministic config, no refit; per-site relative RMSE, median across sites",
    }
    out = REPO.parent / "assets/md/fuentes/srf/pics_49/validation_l3_accepted.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
