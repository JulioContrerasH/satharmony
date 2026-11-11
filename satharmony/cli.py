"""Command line interface for satharmony."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .calibration.align import align_pair
from .calibration.fit import Pair, fit_profile
from .calibration.metrics import c2st, summarize
from .emulators.s2_to_mss.config import PipelineConfig
from .emulators.s2_to_mss.functional import emulate_s2
from .emulators.s2_to_mss.multisensor import emulate_any
from .emulators.s2_to_mss.sensors import SensorNotCalibrated
from .manifests.manifest import load_manifest, verify_manifest


def _read_array(path: str | Path) -> np.ndarray:
    """Read ``.npy`` or a multi-band GeoTIFF."""
    source = Path(path)
    if source.suffix == ".npy":
        return np.load(source)
    if source.suffix in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("reading GeoTIFF needs the [io] extra (rasterio)") from error
        with rasterio.open(source) as handle:
            return handle.read().astype(np.float32)
    raise ValueError(f"unsupported array format: {source.suffix}")


def _write_array(array: np.ndarray, path: str | Path) -> None:
    target = Path(path)
    if target.suffix != ".npy":
        raise ValueError("only .npy output is supported without rasterio")
    np.save(target, array)


def _emulate(args: argparse.Namespace) -> int:
    image = _read_array(args.input)
    if args.sensor:
        try:
            mss, factor, sensor = emulate_any(image, sensor=args.sensor, seed=args.seed)
        except SensorNotCalibrated as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        config = PipelineConfig.from_profile(args.profile)
        mss, factor = emulate_s2(image, config=config, seed=args.seed)
        sensor = config.radiometric.satellite
    _write_array(mss, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "shape": list(mss.shape),
                "factor": factor,
                "sensor": sensor,
            }
        )
    )
    return 0


def _align(args: argparse.Namespace) -> int:
    reference = _read_array(args.reference)
    moving = _read_array(args.moving)
    if reference.ndim == 3:
        reference = reference[0]
    if moving.ndim == 3:
        moving = moving[0]
    result = align_pair(reference, moving, max_shift=args.max_shift, refine=not args.no_ecc)
    print(
        json.dumps(
            {
                "dy": result.dy,
                "dx": result.dx,
                "quality": result.quality,
                "method": result.method,
            }
        )
    )
    return 0


def _read_mask(path: str | Path) -> np.ndarray:
    """Read a mask where nonzero pixels are used."""
    mask = _read_array(path)
    if mask.ndim == 3:
        mask = mask[0]
    return np.asarray(mask) > 0


def _metrics(args: argparse.Namespace) -> int:
    prediction = _read_array(args.prediction)
    observation = _read_array(args.observation)
    valid = _read_mask(args.mask) if args.mask else None
    result = summarize(prediction, observation, valid)
    if args.c2st:
        result["c2st"] = c2st(
            prediction,
            observation,
            n_splits=args.c2st_splits,
            seed=args.seed,
            valid=valid,
        )
    print(json.dumps(result, indent=2))
    return 0


def _fit(args: argparse.Namespace) -> int:
    records = load_manifest(args.manifest)
    if args.verify:
        problems = verify_manifest(records)
        if problems:
            print(json.dumps({"problems": problems}, indent=2), file=sys.stderr)
            return 1
    pairs = []
    for record in records:
        if record.kind != "s2_mss":
            continue
        pairs.append(
            Pair(
                s2=_read_array(record.input_path),
                mss=_read_array(record.reference_path),
                name=record.id,
            )
        )
    if not pairs:
        raise ValueError("the manifest has no s2_mss records")
    profile = fit_profile(
        pairs,
        name=args.name,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    payload = profile.to_dict()
    payload["config"] = profile.to_config().model_dump(mode="json")
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "n_pairs": len(pairs), "gamma": profile.gamma}))
    return 0


def _verify(args: argparse.Namespace) -> int:
    problems = verify_manifest(load_manifest(args.manifest), check_hashes=not args.no_hashes)
    print(json.dumps({"ok": not problems, "problems": problems}, indent=2))
    return 0 if not problems else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(prog="satharmony", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    emulate = sub.add_parser("emulate", help="S2 -> MSS with a versioned profile")
    emulate.add_argument("input")
    emulate.add_argument("-o", "--output", required=True)
    emulate.add_argument("--profile", default="mss5_esa_gtc_v1")
    emulate.add_argument("--seed", type=int, default=None)
    emulate.add_argument(
        "--sensor",
        default=None,
        help="MSS sensor key (L1_MSS..L5_MSS) or 'random'; default: the profile sensor",
    )
    emulate.set_defaults(func=_emulate)

    align = sub.add_parser("align", help="subpixel shift between two rasters")
    align.add_argument("reference")
    align.add_argument("moving")
    align.add_argument("--max-shift", type=float, default=6.0)
    align.add_argument("--no-ecc", action="store_true")
    align.set_defaults(func=_align)

    metrics = sub.add_parser("metrics", help="SAM, RMSE, bias and optional C2ST")
    metrics.add_argument("prediction")
    metrics.add_argument("observation")
    metrics.add_argument("--mask", help="raster where nonzero pixels are used")
    metrics.add_argument("--c2st", action="store_true", help="add the classifier two-sample test")
    metrics.add_argument("--c2st-splits", type=int, default=3)
    metrics.add_argument("--seed", type=int, default=0)
    metrics.set_defaults(func=_metrics)

    fit = sub.add_parser("fit", help="fit a profile from an s2_mss manifest")
    fit.add_argument("manifest")
    fit.add_argument("-o", "--output", required=True)
    fit.add_argument("--name", default="fitted")
    fit.add_argument("--n-boot", type=int, default=1000)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--verify", action="store_true")
    fit.set_defaults(func=_fit)

    verify = sub.add_parser("verify", help="check a manifest and its hashes")
    verify.add_argument("manifest")
    verify.add_argument("--no-hashes", action="store_true")
    verify.set_defaults(func=_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
