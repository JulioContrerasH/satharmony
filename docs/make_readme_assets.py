#!/usr/bin/env python
"""Build the figures used by the README.

Reads the PICS locally (deep) and the validation JSONs from the vault, and
writes everything into docs/assets. Safe to re-run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import tifffile
    from PIL import Image
except ImportError as error:  # pragma: no cover - docs only
    raise SystemExit(
        "the README figures need tifffile and Pillow: pip install tifffile pillow"
    ) from error

REPO = Path(__file__).resolve().parents[1]
VAULT = REPO.parents[0]
sys.path.insert(0, str(REPO))

from satharmony.calibration.fit import _deterministic_config  # noqa: E402
from satharmony.emulators.s2_to_mss.config import PipelineConfig  # noqa: E402
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402
from satharmony.emulators.s2_to_mss.sbaf import S2_BAND_ORDER  # noqa: E402
from satharmony.validation.domain import radial_psd  # noqa: E402

PICS = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")
SITE = "libya2"
OUT = REPO / "docs" / "assets"
EVIDENCE = REPO / "docs" / "validation"

INK = "#1f2328"
MUTED = "#6b7280"
TEAL = "#0f766e"
ORANGE = "#c2410c"
GRID = "#e5e7eb"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#d0d7de",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def read_band(path: Path) -> np.ndarray:
    return tifffile.imread(path).astype(np.float32)


def load_pair(site: str) -> tuple[np.ndarray, np.ndarray]:
    root = PICS / site
    s2 = np.stack(
        [
            read_band(sorted((root / "s2").glob(f"*.{band}.tif"))[0]) / 10000.0
            for band in S2_BAND_ORDER
        ]
    )
    mss = np.stack(
        [read_band(sorted((root / "mss").glob(f"*.B{band}.tif"))[0]) for band in range(1, 5)]
    )
    return s2, mss


def deterministic(config: PipelineConfig) -> PipelineConfig:
    config = _deterministic_config(config)
    config.final_scaling.enabled = True
    for band in (
        config.final_scaling.band_scaling.green,
        config.final_scaling.band_scaling.red,
        config.final_scaling.band_scaling.nir1,
        config.final_scaling.band_scaling.nir2,
    ):
        band.variation = 0.0
    return config


def block_mean(image: np.ndarray, factor: int) -> np.ndarray:
    _, h, w = image.shape
    h2, w2 = h // factor, w // factor
    blocks = image[:, : h2 * factor, : w2 * factor].reshape(
        3, h2, factor, w2, factor
    ).mean(axis=(2, 4))
    return np.transpose(blocks, (1, 2, 0))


def stretch(rgb: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    out = np.empty_like(rgb, dtype=np.float64)
    for index in range(3):
        band = rgb[..., index]
        low, high = np.percentile(band, [lo, hi])
        out[..., index] = np.clip((band - low) / max(high - low, 1e-6), 0, 1)
    return out


def stretch_with(rgb: np.ndarray, limits: list[tuple[float, float]]) -> np.ndarray:
    out = np.empty_like(rgb, dtype=np.float64)
    for index, (low, high) in enumerate(limits):
        band = rgb[..., index]
        out[..., index] = np.clip((band - low) / max(high - low, 1e-6), 0, 1)
    return out


def false_color(mss: np.ndarray) -> np.ndarray:
    return np.stack([mss[3], mss[1], mss[0]], axis=-1)


def example_figure(s2: np.ndarray, mss: np.ndarray, pred: np.ndarray) -> None:
    true_color = block_mean(s2[[3, 2, 1]], 6)
    real = false_color(mss)
    emulated = false_color(pred)
    limits = [tuple(np.percentile(real[..., index], [2, 98])) for index in range(3)]

    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.5))
    panels = (
        (stretch(true_color), "Sentinel-2, true colour (10 m)"),
        (stretch_with(emulated, limits), "Emulated MSS, deterministic (60 m)"),
        (stretch_with(real, limits), "Landsat 5 MSS, false colour (60 m)"),
    )
    for axis, (image, title) in zip(axes, panels, strict=True):
        axis.imshow(image, interpolation="nearest")
        axis.set_title(title, fontsize=9, pad=6)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.text(
        0.5,
        -0.01,
        "Libya-2 PICS. Red: MSS4 (0.9 um), green: MSS2, blue: MSS1. The two MSS "
        "panels share one per-band stretch, taken from the real scene. "
        "The emulated panel is the deterministic core; the stochastic profile adds sensor noise.",
        ha="center",
        fontsize=7.5,
        color=MUTED,
    )
    figure.savefig(OUT / "example.png", dpi=200)
    plt.close(figure)


def gif(s2: np.ndarray, mss: np.ndarray, pred: np.ndarray) -> None:
    real = stretch(false_color(mss))
    emulated = stretch(false_color(pred))
    frames = []
    for image in (real, emulated):
        array = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        frames.append(Image.fromarray(array).resize((400, 400), Image.NEAREST))
    frames[0].save(
        OUT / "example.gif",
        save_all=True,
        append_images=frames[1:],
        duration=[1200, 1200],
        loop=0,
        optimize=True,
    )


def load_evidence(name: str) -> dict:
    return json.loads((VAULT / "assets/md/fuentes/srf/pics_49" / f"{name}.json").read_text())


def validation_figure(desert: dict, outside: dict, c2st: dict) -> None:
    bands = ["MSS1", "MSS2", "MSS3", "MSS4"]
    x = np.arange(4)
    width = 0.38

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.25]})

    for axis, values, title in (
        (axes[0], desert, "Desert, 29 calibration sites"),
        (axes[1], outside, "Outside the domain, 14 sites"),
    ):
        scaled = values["scaled"]
        raw = values["raw"]
        axis.bar(x - width / 2, np.array(scaled) * 100, width, color=TEAL, label="with one scale per site")
        axis.bar(x + width / 2, np.array(raw) * 100, width, color="#99f6e4", label="no free scale")
        axis.set_xticks(x, bands)
        axis.set_ylabel("scaled RMSE (%)")
        axis.set_title(title, fontsize=9, loc="left")
        axis.yaxis.grid(True, color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    top = 1.12 * max(
        max(desert["scaled"]), max(desert["raw"]), max(outside["scaled"]), max(outside["raw"])
    ) * 100
    axes[0].set_ylim(0, top)
    axes[1].set_ylim(0, top)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")

    axis = axes[2]
    sites = [row["site"] for row in c2st["per_site"]["deterministic"]]
    det = [row["c2st"] for row in c2st["per_site"]["deterministic"]]
    stochastic = [row["c2st"] for row in c2st["per_site"]["stochastic_seed0"]]
    positions = np.arange(len(sites))
    axis.scatter(positions, det, s=26, color=ORANGE, zorder=3, label="deterministic")
    axis.scatter(positions, stochastic, s=26, color=TEAL, zorder=3, label="stochastic, seed 0")
    axis.axhline(0.5, color=MUTED, linestyle="--", linewidth=0.8)
    axis.axhline(0.75, color="#b91c1c", linestyle=":", linewidth=0.8)
    axis.text(-0.45, 0.515, "0.5 indistinguishable", fontsize=6.5, color=MUTED, ha="left")
    axis.text(-0.45, 0.765, "0.75 flag", fontsize=6.5, color="#b91c1c", ha="left")
    axis.set_xticks(positions, [site.replace("_", "\n") for site in sites], fontsize=7)
    axis.set_ylim(0.4, 1.02)
    axis.set_ylabel("C2ST accuracy")
    axis.set_title("Real vs emulated, 5 held-out sites", fontsize=9, loc="left")
    axis.yaxis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=7, loc="lower right")

    figure.tight_layout()
    figure.savefig(OUT / "validation.png", dpi=200)
    plt.close(figure)


def psd_figure(mss: np.ndarray, pred: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for axis, index, name in zip(axes, (0, 3), ("MSS1", "MSS4"), strict=True):
        real_wave, real_power = radial_psd(mss[index])
        emu_wave, emu_power = radial_psd(pred[index])
        axis.loglog(real_wave, real_power, color=INK, linewidth=1.2, label="Landsat 5 MSS")
        axis.loglog(emu_wave, emu_power, color=ORANGE, linewidth=1.2, label="emulated")
        axis.set_title(f"{name}, high-pass PSD", fontsize=9, loc="left")
        axis.set_xlabel("wavelength (pixels)")
        axis.grid(True, color=GRID, linewidth=0.6, which="major")
        axis.set_axisbelow(True)
    axes[0].set_ylabel("power")
    axes[0].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(OUT / "psd.png", dpi=200)
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    s2, mss = load_pair(SITE)
    profile = PipelineConfig.from_profile("mss5_esa_gtc_v2")
    pred, factor = emulate_s2(s2, config=profile, seed=42)
    smooth, _ = emulate_s2(s2, config=deterministic(profile))
    print(SITE, "grid", mss.shape, "pred", pred.shape, "factor", factor)

    example_figure(s2, mss, smooth)
    gif(s2, mss, smooth)
    psd_figure(mss, smooth)

    desert = load_evidence("desert_v2_insample")
    outside = load_evidence("validation_v2_outside")["results"]["deterministic"]["median_across_sites"]
    c2st = load_evidence("c2st_v2")
    bands = ("B1", "B2", "B3", "B4")
    desert_medians = desert["results"]["median_across_sites"]
    desert_summary = {
        "scaled": [desert_medians[band]["rescaled_median"] for band in bands],
        "raw": [desert_medians[band]["median"] for band in bands],
    }
    outside_summary = {
        "scaled": [outside[band]["rmse_rel_rescaled_median"] for band in bands],
        "raw": [outside[band]["rmse_rel_median"] for band in bands],
    }
    validation_figure(desert_summary, outside_summary, c2st)

    for name in ("c2st_v2", "desert_v2_insample", "validation_v2_outside"):
        source = VAULT / "assets/md/fuentes/srf/pics_49" / f"{name}.json"
        (EVIDENCE / f"{name}.json").write_text(source.read_text())

    print("written", sorted(path.name for path in OUT.iterdir()))


if __name__ == "__main__":
    main()
