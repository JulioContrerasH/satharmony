#!/usr/bin/env python
"""Compare the pseudo-MSS domain with the real MSS on the five PICS.

For every CEOS PICS downloaded by ``scripts/fetch_pics.py``:

1. Build the S2 stack (13 bands, TOA reflectance) and the real MSS TOA
   reflectance (60 m, 4 bands, already converted by ``fetch_pics.py``).
2. Emulate three pseudo-MSS variants on the same grid:
   ``legacy`` (0.1.13 profile), ``sbaf`` (mss5_esa_gtc_v1, full stochastic)
   and ``sbaf_physics`` (SBAF + PSF + 60 m + calibrated band scaling, with
   radiometry and sensor artifacts disabled).
3. Measure the Wasserstein-1 distance and the KS statistic per band against
   the real MSS, over the same valid pixels.  ``sbaf`` and ``legacy`` are run
   with several seeds; the reported distance is the median over seeds.
4. Measure the striping with the 2D power spectrum: peak wavelength and the
   power fraction in the 6-10 px annulus (474 m / 60 m ~ 7.9 px).

Outputs (``--out-dir``, default ``assets/md/fuentes/domain``)::

    domain_wasserstein.csv          one row per site x band x mode x seed
    domain_wasserstein_summary.csv  median across sites and seeds
    domain_striping.csv             one row per site x band x source x seed
    domain_striping_summary.csv     median across sites and seeds
    domain_table.md                 human-readable tables
    domain_metadata.json            method, inputs, hashes, parameters
    figures/domain_pseudo_vs_real.png

Usage
-----
    python scripts/compare_pseudo_domain.py
    python scripts/compare_pseudo_domain.py --data /tmp/pics --out-dir /tmp/out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.emulators.s2_to_mss.config import (  # noqa: E402
    BandScaleRange,
    CenteredRange,
    PipelineConfig,
    Range,
)
from satharmony.emulators.s2_to_mss.functional import emulate_s2  # noqa: E402
from satharmony.emulators.s2_to_mss.sbaf import S2_BAND_ORDER  # noqa: E402
from satharmony.validation.domain import (  # noqa: E402
    STRIPING_WAVELENGTH_PX,
    ks_per_band,
    radial_psd,
    striping_metrics,
    wasserstein_per_band,
)

DEFAULT_DATA = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")
DEFAULT_OUT = REPO.parent / "assets/md/fuentes/domain"
BANDS = ("B1", "B2", "B3", "B4")
SITES = ("libya4_ceos", "algeria3", "niger1", "egypt1", "arabia2")
MODES = ("legacy", "sbaf", "sbaf_physics")
PROB_MASK = 20  # s2cloudless probability threshold


# ── I/O ────────────────────────────────────────────────────────────────────
def load_site(site_dir: Path) -> dict:
    """Load the S2 stack, the real MSS reflectance and the masks of one PICS."""
    import rasterio

    meta = json.loads((site_dir / "metadata.json").read_text())
    s2_files = sorted((site_dir / "s2").glob("*.tif"))
    if not s2_files:
        raise FileNotFoundError(f"no S2 files in {site_dir}")

    with rasterio.open(s2_files[0]) as src:
        shape = src.shape
    stack = []
    for band in S2_BAND_ORDER:
        path = sorted((site_dir / "s2").glob(f"*.{band}.tif"))[0]
        with rasterio.open(path) as src:
            stack.append(src.read(1).astype(np.float32) / 10000.0)
    s2 = np.stack(stack)

    prob = np.zeros(shape, np.uint8)
    prob_path = sorted((site_dir / "s2").glob("*.MSK_CLOUD_PROB.tif"))
    if prob_path:
        with rasterio.open(prob_path[0]) as src:
            prob = src.read(1)

    mss = []
    for band in range(1, 5):
        path = sorted((site_dir / "mss").glob(f"*.B{band}.tif"))[0]
        with rasterio.open(path) as src:
            mss.append(src.read(1).astype(np.float32))
    mss = np.stack(mss)

    saturated = np.zeros(mss.shape[1:], bool)
    sat_path = sorted((site_dir / "mss").glob("*.MSK_SATURATED.tif"))
    if sat_path:
        with rasterio.open(sat_path[0]) as src:
            saturated = src.read(1).astype(bool)

    return {"meta": meta, "s2": s2, "mss": mss, "prob": prob, "saturated": saturated}


def valid_masks(site: dict) -> tuple[np.ndarray, np.ndarray]:
    """(S2 10 m mask, MSS 60 m mask) of usable pixels."""
    s2 = site["s2"]
    s2_ok = (
        (site["prob"] < PROB_MASK)
        & np.all(np.isfinite(s2), axis=0)
        & (np.min(s2, axis=0) > 0.0)
    )
    mss = site["mss"]
    mss_ok = np.all(np.isfinite(mss), axis=0) & (np.min(mss, axis=0) > 0.0)
    mss_ok &= (mss < 1.0).all(axis=0)
    mss_ok &= ~site["saturated"]
    return s2_ok, mss_ok


def aggregate_mask(s2_ok: np.ndarray, size: int) -> np.ndarray:
    """A 60 m cell is usable only if all of its 10 m pixels are usable."""
    h, _ = s2_ok.shape
    step = h // size
    blocks = s2_ok[: size * step, : size * step].reshape(size, step, size, step)
    return blocks.mean(axis=(1, 3)) >= 1.0


# ── profiles ───────────────────────────────────────────────────────────────
def full_profile(mode: str, meta: dict) -> PipelineConfig:
    """Stochastic profile for ``legacy`` or ``sbaf`` with the scene geometry."""
    cfg = PipelineConfig.legacy_0_1_13() if mode == "legacy" else PipelineConfig.mss5_esa_gtc_v1()
    sun = meta["mss"].get("sun_elevation")
    if sun is not None:
        cfg.radiometric.sun_elevation_deg = Range(
            min=float(sun), max=float(sun), distribution="uniform"
        )
    distance = meta["mss"].get("earth_sun_distance")
    if distance is not None:
        cfg.radiometric.earth_sun_distance_au = float(distance)
    return cfg


def physics_profile() -> PipelineConfig:
    """SBAF + PSF + 60 m + fixed calibrated scaling; no sensor artifacts."""
    cfg = PipelineConfig.mss5_esa_gtc_v1()
    cfg.radiometric.enabled = False
    cfg.striping.enabled = False
    cfg.memory_effect.enabled = False
    cfg.coherent_noise.enabled = False
    cfg.random_noise.enabled = False
    cfg.scan_artifacts.enabled = False
    base = PipelineConfig.mss5_esa_gtc_v1().final_scaling.band_scaling
    cfg.final_scaling.band_scaling = BandScaleRange(
        green=CenteredRange(mean=base.green.mean, variation=0.0),
        red=CenteredRange(mean=base.red.mean, variation=0.0),
        nir1=CenteredRange(mean=base.nir1.mean, variation=0.0),
        nir2=CenteredRange(mean=base.nir2.mean, variation=0.0),
    )
    return cfg


def emulate_variant(s2: np.ndarray, meta: dict, mode: str, seed: int) -> np.ndarray:
    """Run one variant and return the 60 m pseudo-MSS cube."""
    cfg = physics_profile() if mode == "sbaf_physics" else full_profile(mode, meta)
    mss, _ = emulate_s2(s2, scale_factor=6, seed=seed, config=cfg)
    return np.asarray(mss, dtype=np.float32)


# ── metrics ────────────────────────────────────────────────────────────────
def distance_rows(
    pseudo: np.ndarray,
    real: np.ndarray,
    valid: np.ndarray,
    site: str,
    mode: str,
    seed: int,
) -> list[dict]:
    """W1 and KS per band for one site, mode and seed."""
    w1 = wasserstein_per_band(pseudo, real, valid, valid)
    ks = ks_per_band(pseudo, real, valid, valid)
    rows = []
    for band_idx, band in enumerate(BANDS):
        p = pseudo[band_idx][valid]
        r = real[band_idx][valid]
        rows.append(
            {
                "site": site,
                "band": band,
                "mode": mode,
                "seed": seed,
                "w1": float(w1[band_idx]),
                "w1_rel": float(w1[band_idx] / np.median(r)),
                "ks": float(ks[band_idx]),
                "median_pseudo": float(np.median(p)),
                "median_real": float(np.median(r)),
                "mean_pseudo": float(np.mean(p)),
                "mean_real": float(np.mean(r)),
                "n_pixels": int(p.size),
            }
        )
    return rows


def striping_rows(
    cube: np.ndarray, site: str, band_names: tuple[str, ...], source: str, seed: int
) -> list[dict]:
    """Striping metrics per band of one cube."""
    rows = []
    for band_idx, band in enumerate(band_names):
        metrics = striping_metrics(cube[band_idx])
        rows.append({"site": site, "band": band, "source": source, "seed": seed, **metrics})
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ── figure ────────────────────────────────────────────────────────────────
def plot_domain(
    histograms: dict[str, dict[str, np.ndarray]],
    w1_median: dict[str, dict[str, float]],
    psd: dict[str, dict[str, np.ndarray]],
    path: Path,
) -> None:
    """Two rows: pooled histograms and mean radial PSD per band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    colors = {"real": "black", "sbaf": "#1f77b4", "legacy": "#d95f02", "sbaf_physics": "#8E44AD"}
    labels = {
        "real": "Real MSS",
        "sbaf": "Pseudo MSS (SBAF, full)",
        "legacy": "Pseudo MSS (legacy, full)",
        "sbaf_physics": "Pseudo (physics only)",
    }
    zorders = {"real": 10, "sbaf": 5, "legacy": 4, "sbaf_physics": 6}

    fig = plt.figure(figsize=(13.0, 8.2), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.45, h_pad=0.35)
    grid = fig.add_gridspec(
        4, 5,
        height_ratios=[0.075, 0.065, 1.0, 1.0],
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.30],
    )
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.set_axis_off()
    title_ax.text(
        0.5,
        0.5,
        "Pseudo-MSS vs real MSS (5 CEOS PICS aggregated): distributions and power spectra",
        ha="center",
        va="center",
        fontsize=11,
    )
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.set_axis_off()

    axes = np.empty((2, 4), dtype=object)
    for col in range(4):
        axes[0, col] = fig.add_subplot(grid[2, col])
        axes[1, col] = fig.add_subplot(grid[3, col])

    for band_idx, band in enumerate(BANDS):
        ax = axes[0, band_idx]
        real_values = histograms["real"][band]
        bins = np.linspace(
            np.percentile(real_values, 0.2), np.percentile(real_values, 99.8), 60
        )
        # Plot non-real first so the real black line stays on top.
        for mode in ("sbaf", "legacy", "sbaf_physics"):
            ax.hist(
                histograms[mode][band],
                bins=bins,
                density=True,
                histtype="step",
                color=colors[mode],
                lw=1.1,
                ls="-" if mode != "sbaf_physics" else "--",
                zorder=zorders[mode],
            )
        ax.hist(
            real_values,
            bins=bins,
            density=True,
            histtype="step",
            color=colors["real"],
            lw=2.5,
            zorder=zorders["real"],
        )
        ax.set_yscale("log")
        ax.set_title(f"MSS {band}")
        ax.set_xlabel("TOA reflectance")
        if band_idx == 0:
            ax.set_ylabel("Density")

        ax = axes[1, band_idx]
        for source in ("sbaf", "legacy", "sbaf_physics"):
            frequency, power = psd[source][band]
            ax.plot(
                1.0 / frequency,
                power / np.median(power[frequency < 0.03]),
                color=colors[source],
                lw=1.1,
                ls="-" if source != "sbaf_physics" else "--",
                zorder=zorders[source],
            )
        frequency, power = psd["real"][band]
        ax.plot(
            1.0 / frequency,
            power / np.median(power[frequency < 0.03]),
            color=colors["real"],
            lw=2.5,
            zorder=zorders["real"],
        )
        ax.axvline(STRIPING_WAVELENGTH_PX, color="0.15", ls=":", lw=1.2, zorder=1)
        ax.text(
            STRIPING_WAVELENGTH_PX * 1.04,
            0.55,
            "474 m\n(7.9 px)",
            transform=ax.get_xaxis_transform(),
            color="0.1",
            fontsize=7,
            va="top",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Fit xlim to the actual wavelength range of the data.
        all_freqs = np.concatenate(
            [psd[s][band][0] for s in ("real", "sbaf", "legacy", "sbaf_physics")]
        )
        ax.set_xlim(0.9 / all_freqs.max(), 1.1 / all_freqs.min())
        # Avoid overlapping log labels: 1,2,5 ticks only, no minor labels.
        import matplotlib.ticker as mticker
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1, 2, 5], numticks=6))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_title(f"Power spectrum {band}")
        ax.set_xlabel("Wavelength (pixels)")
        if band_idx == 0:
            ax.set_ylabel("PSD / low-frequency median")
        ax.tick_params(axis="x", rotation=25, labelsize=8)

    # W1 metrics moved outside the plotting axes into a dedicated gutter panel.
    w1_ax = fig.add_subplot(grid[2, 4])
    w1_ax.set_axis_off()
    w1_lines = ["W1 distance", "─" * 16]
    for band in BANDS:
        s = w1_median["sbaf"][band]
        legacy = w1_median["legacy"][band]
        change = 100 * (legacy - s) / legacy if legacy else float("nan")
        w1_lines.append(f"{band}  SBAF {s:.4f}")
        w1_lines.append(f"    leg. {legacy:.4f}")
        w1_lines.append(f"    Δ   {change:+.0f}%")
    w1_ax.text(
        0.08,
        0.98,
        "\n".join(w1_lines),
        transform=w1_ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        linespacing=1.35,
        family="monospace",
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.95},
    )

    handles = [
        plt.Line2D(
            [], [], color=colors[key], lw=1.5,
            ls="-" if key != "sbaf_physics" else "--",
            label=labels[key],
        )
        for key in ("real", "sbaf", "legacy", "sbaf_physics")
    ]
    legend_ax.legend(handles=handles, loc="center", ncol=4, frameon=False, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--sites", nargs="+", default=list(SITES))
    args = parser.parse_args()

    t0 = time.time()
    distance_frames: list[pd.DataFrame] = []
    striping_records: list[dict] = []
    histograms: dict[str, dict[str, list[np.ndarray]]] = {}
    psd_curves: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}

    for site_name in args.sites:
        site = load_site(args.data / site_name)
        meta = site["meta"]
        s2_ok, mss_ok = valid_masks(site)
        real = site["mss"]
        n_mss = real.shape[1]
        valid = mss_ok & aggregate_mask(s2_ok, n_mss)
        if valid.sum() < 1000:
            raise ValueError(f"{site_name}: only {valid.sum()} valid 60 m pixels")
        print(f"{site_name}: {valid.sum()} valid pixels")

        striping_records.extend(striping_rows(real, site_name, BANDS, "real", -1))
        histograms.setdefault("real", {})
        psd_curves.setdefault("real", {})
        for band_idx, band in enumerate(BANDS):
            histograms["real"].setdefault(band, []).append(real[band_idx][valid])
            psd_curves["real"].setdefault(band, []).append(radial_psd(real[band_idx]))

        for mode in MODES:
            seeds = [0] if mode == "sbaf_physics" else list(range(args.seeds))
            histograms.setdefault(mode, {})
            psd_curves.setdefault(mode, {})
            for seed in seeds:
                pseudo = emulate_variant(site["s2"], meta, mode, seed)
                if pseudo.shape[1:] != real.shape[1:]:
                    raise ValueError(
                        f"{site_name}/{mode}: pseudo {pseudo.shape} != real {real.shape}"
                    )
                distance_frames.append(
                    pd.DataFrame(distance_rows(pseudo, real, valid, site_name, mode, seed))
                )
                striping_records.extend(striping_rows(pseudo, site_name, BANDS, mode, seed))
                for band_idx, band in enumerate(BANDS):
                    psd_curves[mode].setdefault(band, []).append(radial_psd(pseudo[band_idx]))
                    if seed == 0:
                        histograms[mode].setdefault(band, []).append(pseudo[band_idx][valid])
            print(f"  {mode}: {len(seeds)} seed(s)")

    distances = pd.concat(distance_frames, ignore_index=True)
    striping = pd.DataFrame(striping_records)

    summary = (
        distances.groupby(["band", "mode"], as_index=False)
        .agg(
            w1=("w1", "median"),
            w1_iqr=("w1", lambda x: float(np.percentile(x, 75) - np.percentile(x, 25))),
            w1_rel=("w1_rel", "median"),
            ks=("ks", "median"),
            median_pseudo=("median_pseudo", "median"),
            median_real=("median_real", "median"),
            n_sites=("site", "nunique"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["band", "mode"])
    )
    striping_summary = (
        striping.groupby(["band", "source"], as_index=False)
        .agg(
            wavelength_px=("wavelength_px", "median"),
            wavelength_m=("wavelength_m", "median"),
            amplitude=("amplitude", "median"),
            amplitude_iqr=(
                "amplitude",
                lambda x: float(np.percentile(x, 75) - np.percentile(x, 25)),
            ),
            stripe_index=("stripe_index", "median"),
            n_sites=("site", "nunique"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["band", "source"])
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    distances.to_csv(out_dir / "domain_wasserstein.csv", index=False, float_format="%.6f")
    summary.to_csv(out_dir / "domain_wasserstein_summary.csv", index=False, float_format="%.6f")
    striping.to_csv(out_dir / "domain_striping.csv", index=False, float_format="%.6f")
    striping_summary.to_csv(
        out_dir / "domain_striping_summary.csv", index=False, float_format="%.6f"
    )

    # Interpolate the PSD curves on a common frequency grid and take the median.
    common = np.linspace(0.02, 0.5, 60)
    psd_median: dict[str, dict[str, np.ndarray]] = {}
    for source in ("real", "sbaf", "legacy", "sbaf_physics"):
        psd_median[source] = {}
        for band in BANDS:
            curves = [
                np.interp(common, np.sort(1.0 / wavelength), power[np.argsort(1.0 / wavelength)])
                for wavelength, power in psd_curves[source][band]
            ]
            psd_median[source][band] = (common, np.mean(np.stack(curves), axis=0))

    hist = {
        source: {band: np.concatenate(values) for band, values in bands.items()}
        for source, bands in histograms.items()
    }
    w1_median = {
        mode: {
            band: float(summary[(summary["band"] == band) & (summary["mode"] == mode)]["w1"].iloc[0])
            for band in BANDS
        }
        for mode in ("sbaf", "legacy")
    }
    plot_domain(hist, w1_median, psd_median, out_dir / "figures/domain_pseudo_vs_real.png")

    _write_table(out_dir / "domain_table.md", summary, striping_summary)
    metadata = {
        "created_by": "scripts/compare_pseudo_domain.py",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": str(args.data),
        "sites": args.sites,
        "seeds": args.seeds,
        "modes": {
            "legacy": "PipelineConfig.legacy_0_1_13(), scale factor fixed at 6",
            "sbaf": "PipelineConfig.mss5_esa_gtc_v1(), full stochastic",
            "sbaf_physics": (
                "mss5_esa_gtc_v1 with radiometry/artifacts off and fixed band scaling"
            ),
        },
        "metrics": {
            "w1": "scipy.stats.wasserstein_distance on the valid 60 m pixels, per band",
            "ks": "scipy.stats.ks_2samp statistic",
            "striping": (
                "2D power spectrum: peak wavelength in 5-12 px and power fraction "
                "in the 6-10 px annulus"
            ),
        },
        "pics": {
            site: json.loads((args.data / site / "metadata.json").read_text())["site"]
            for site in args.sites
        },
        "outputs": {
            name: sha256_file(out_dir / name)
            for name in (
                "domain_wasserstein.csv",
                "domain_wasserstein_summary.csv",
                "domain_striping.csv",
                "domain_striping_summary.csv",
            )
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "domain_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {out_dir} in {time.time() - t0:.0f} s")
    print(summary.to_string(index=False))
    return 0


def _write_table(
    path: Path,
    summary: pd.DataFrame,
    striping_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Pseudo-MSS vs real MSS on the five PICS",
        "",
        "W1 is the Wasserstein-1 distance in TOA reflectance, median across sites and",
        "seeds. `W1 rel` divides W1 by the median real reflectance of the band. The",
        "gate is that SBAF is closer to the real MSS than legacy in all four bands.",
        "",
        "## Wasserstein and KS",
        "",
        "| Band | Mode | W1 | IQR | W1 rel | KS | median pseudo | median real |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['band']} | {row['mode']} | {row['w1']:.4f} | {row['w1_iqr']:.4f} | "
            f"{row['w1_rel']:.3f} | {row['ks']:.3f} | {row['median_pseudo']:.3f} | "
            f"{row['median_real']:.3f} |"
        )
    lines += [
        "",
        "## Striping (median across sites and seeds)",
        "",
        "Amplitude of the 474 m / 60 m = 7.9 px component in the 2D spectrum,",
        "in reflectance units (x100 for % reflectance), and the ratio between that",
        "annulus and its neighbouring background. The per-site angle is in",
        "`domain_striping.csv`: it depends on the orbit heading and is not averaged.",
        "",
        "| Band | Source | Peak (px) | Peak (m) | Amplitude | IQR | Stripe index |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in striping_summary.iterrows():
        lines.append(
            f"| {row['band']} | {row['source']} | {row['wavelength_px']:.2f} | "
            f"{row['wavelength_m']:.0f} | {row['amplitude']:.4f} | "
            f"{row['amplitude_iqr']:.4f} | {row['stripe_index']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
