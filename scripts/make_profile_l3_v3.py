#!/usr/bin/env python
"""Reajustar el perfil de L3 con el banco ampliado (Tier 2) y medir cuánto cambia.

Usa los mismos pasos que ``make_profile_aux.py`` (estabilidad al 15 %, reserva
cada 4º sitio, ``fit_profile`` con ganancia libre por banda) pero solo con los
sitios que pasaron la revisión: los 6 aceptados del banco viejo y los 18
nuevos aceptados de Tier 2. Los 12 rechazados van en la lista de exclusión.

No sobrescribe el perfil: deja el candidato en ``--out`` y muestra el cambio
banda por banda frente al perfil publicado.

    python scripts/make_profile_l3_v3.py --out /tmp/mss3_v3_candidate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.calibration.fit import fit_profile  # noqa: E402
from satharmony.emulators.s2_to_mss.config import (  # noqa: E402
    MEASURED_SUN_ELEVATION_CANONICAL,
    PipelineConfig,
)
from scripts.aux_pairs import load_pair, sites  # noqa: E402
from scripts.holdout_v2 import deterministic  # noqa: E402
from scripts.make_profile_aux import BANDS, TOLERANCE, site_metrics, stable  # noqa: E402

SENSOR = "L3"
PROFILE = "mss3_esa_gtc_v2"
COEFFICIENTS = "sbaf_coefficients_l3.json"

ACCEPTED_NEW = [
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
REJECTED = [
    # banco viejo, rechazados por Julio
    "l3_054.8E_17.2N",
    "l3_003.3E_28.8N",
    "l3_017.3E_27.4N",
    "l3_058.9E_41.6N",
    "l3_036.8E_30.2N",
    "l3_029.1E_28.8N",
    "l3_095.0W_58.4N",
    "l3_100.6W_44.5N",
    "l3_010.3E_33.0N",
    # nuevos rechazados: superficie cambiada o nube que la máscara no ve
    "l3_106.4W_31.6N",
    "l3_110.4E_41.7N",
    "l3_134.3E_27.3S",
    "l3_157.8E_57.1N",
]


def base_config() -> PipelineConfig:
    base = PipelineConfig.from_profile("mss5_esa_gtc_v2")
    base.radiometric.satellite = f"{SENSOR}_MSS"
    base.spectral.coefficients = COEFFICIENTS
    base.radiometric.sun_elevation_deg = type(base.radiometric.sun_elevation_deg)(
        **MEASURED_SUN_ELEVATION_CANONICAL
    )
    return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/mss3_v3_candidate.json"))
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    names = [row["site"] for row in sites(SENSOR) if row["site"] in ACCEPTED_NEW or row["site"] not in REJECTED]
    names = [name for name in names if name not in REJECTED]
    print(f"{len(names)} sitios aceptados", flush=True)

    base = base_config()
    metrics = {name: site_metrics(SENSOR, name, base) for name in names}
    usable = [
        name
        for name in names
        if metrics[name]["masked"] >= 0.9 and all(np.isfinite(metrics[name][b]) for b in BANDS)
    ]
    print(f"{len(usable)} con máscara >= 90 %", flush=True)
    keep, reject, medians = stable(usable, metrics)
    print(f"{len(keep)} estables, {len(reject)} fuera de estabilidad", flush=True)
    if reject:
        print("inestables:", reject, flush=True)

    held = [name for index, name in enumerate(keep) if index % 4 == 3]
    train = [name for name in keep if name not in held]
    print(f"train {len(train)}, reserva {len(held)}", flush=True)

    base.spectral.enabled = True
    profile = fit_profile(
        (load_pair(SENSOR, name) for name in train),
        base=base,
        name=f"{PROFILE}_v3",
        n_boot=args.n_boot,
        gamma_rule="off",
    )
    config = profile.to_config(base=base)
    config.radiometric.satellite = f"{SENSOR}_MSS"
    config.spectral.coefficients = COEFFICIENTS
    config.radiometric.sun_elevation_deg = type(config.radiometric.sun_elevation_deg)(
        **MEASURED_SUN_ELEVATION_CANONICAL
    )
    eval_config = deterministic(config)

    train_metrics = {name: site_metrics(SENSOR, name, eval_config) for name in train}
    held_metrics = {name: site_metrics(SENSOR, name, eval_config) for name in held}
    in_sample = {b: float(np.nanmedian([row[b] for row in train_metrics.values()])) for b in BANDS}
    held_out = {b: float(np.nanmedian([row[b] for row in held_metrics.values()])) for b in BANDS}

    old = json.loads((REPO / "satharmony/profiles" / f"{PROFILE}.json").read_text())
    print("\nbanda   perfil actual   ajuste nuevo   cambio")
    for band in BANDS:
        was = old["band_scaling"][band]["mean"]
        now = profile.to_dict()["band_scaling"][band]["mean"]
        print(f"{band:5s} {was:14.3f} {now:14.3f} {100 * (now / was - 1):+7.1f} %")
    print(f"\nRMSE relativo en muestra: { {b: round(v, 3) for b, v in in_sample.items()} }")
    print(f"RMSE relativo en reserva: { {b: round(v, 3) for b, v in held_out.items()} }")
    print(f"peor en reserva {max(held_out.values()):.3f}")
    print(f"n_pairs train {len(train)}, antes {old['n_pairs']}")

    payload = profile.to_dict()
    payload["config"] = config.model_dump(mode="json")
    payload["candidate_of"] = PROFILE
    payload["selection"] = {
        "accepted": sorted(train),
        "held_out": sorted(held),
        "rejected": REJECTED,
        "tolerance": TOLERANCE,
        "median_scale_all_usable": medians,
    }
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"candidato -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
