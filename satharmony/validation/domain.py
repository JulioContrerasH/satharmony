"""Domain metrics: pseudo-MSS against real MSS.

Two questions are answered here:

* Is the marginal distribution of the emulated reflectance close to the
  real one?  The Wasserstein-1 distance (a.k.a. earth mover's distance) is
  used per band because it is in reflectance units, unlike KS.
* Does the emulator reproduce the MSS detector striping?  The 474 m
  along-track period is a bump in the 2D power spectrum at ~7.9 px on the
  60 m grid.  The movie window is tilted, so the metric looks for the peak
  in an annulus instead of assuming row alignment.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, stats

#: MSS scan-line period along the ground trace at 60 m (474 m / 60 m).
STRIPING_WAVELENGTH_PX = 474.0 / 60.0

#: Annulus (in pixels) where the striping peak is searched.
STRIPING_MIN_WAVELENGTH_PX = 5.0
STRIPING_MAX_WAVELENGTH_PX = 12.0


def wasserstein_per_band(
    pseudo: NDArray[np.floating],
    real: NDArray[np.floating],
    valid_pseudo: NDArray[np.bool_] | None = None,
    valid_real: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Wasserstein-1 distance per band between two reflectance cubes.

    Args:
        pseudo: Emulated cube ``(bands, H, W)``.
        real: Real cube, must have the same number of bands (not the same
            shape: the distance is distributional).
        valid_pseudo: Optional mask on ``pseudo``.
        valid_real: Optional mask on ``real``.

    Returns:
        Array ``(bands,)`` with the distance in reflectance units.

    Raises:
        ValueError: If the band counts differ or a mask has a wrong shape.
    """
    if pseudo.shape[0] != real.shape[0]:
        raise ValueError(
            f"pseudo has {pseudo.shape[0]} bands but real has {real.shape[0]}"
        )
    distances = []
    for band in range(pseudo.shape[0]):
        p = _valid_values(pseudo[band], valid_pseudo, "pseudo")
        r = _valid_values(real[band], valid_real, "real")
        if p.size == 0 or r.size == 0:
            distances.append(float("nan"))
            continue
        distances.append(float(stats.wasserstein_distance(p, r)))
    return np.asarray(distances, dtype=np.float64)


def ks_per_band(
    pseudo: NDArray[np.floating],
    real: NDArray[np.floating],
    valid_pseudo: NDArray[np.bool_] | None = None,
    valid_real: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Two-sample Kolmogorov-Smirnov statistic per band.

    The KS statistic is unitless and lies in [0, 1]; it is reported next to
    the Wasserstein distance because it is sensitive to the shape of the
    distribution rather than to the bulk shift.

    Args:
        pseudo: Emulated cube ``(bands, H, W)``.
        real: Real cube.
        valid_pseudo: Optional mask on ``pseudo``.
        valid_real: Optional mask on ``real``.

    Returns:
        Array ``(bands,)`` with the KS statistic.
    """
    if pseudo.shape[0] != real.shape[0]:
        raise ValueError(
            f"pseudo has {pseudo.shape[0]} bands but real has {real.shape[0]}"
        )
    out = []
    for band in range(pseudo.shape[0]):
        p = _valid_values(pseudo[band], valid_pseudo, "pseudo")
        r = _valid_values(real[band], valid_real, "real")
        if p.size < 2 or r.size < 2:
            out.append(float("nan"))
            continue
        out.append(float(stats.ks_2samp(p, r).statistic))
    return np.asarray(out, dtype=np.float64)


def _valid_values(
    band: NDArray[np.floating], mask: NDArray[np.bool_] | None, name: str
) -> NDArray[np.float64]:
    if mask is not None:
        if mask.shape != band.shape:
            raise ValueError(f"{name} mask {mask.shape} does not match band {band.shape}")
        band = band[mask & np.isfinite(band)]
    else:
        band = band[np.isfinite(band)]
    return np.asarray(band, dtype=np.float64)


def high_pass(band: NDArray[np.floating], sigma_px: float = 3.0) -> NDArray[np.float64]:
    """Remove the low-frequency terrain so the striping is visible.

    Args:
        band: 2D reflectance array.
        sigma_px: Standard deviation of the Gaussian low-pass in pixels.

    Returns:
        ``band - gaussian_filter(band, sigma_px)`` as float64.
    """
    data = np.asarray(band, dtype=np.float64)
    return data - ndimage.gaussian_filter(data, sigma_px)


def power_spectrum(band: NDArray[np.floating], sigma_px: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shifted 2D power spectrum of the high-passed band.

    Args:
        band: 2D reflectance array.
        sigma_px: High-pass scale.

    Returns:
        ``(power, fy, fx)`` with frequencies in cycles/pixel, shifted so the
        zero frequency is at the centre.
    """
    resid = high_pass(band, sigma_px=sigma_px)
    resid = resid - float(np.mean(resid))
    h, w = resid.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    spectrum = np.fft.fftshift(np.fft.fft2(resid * window))
    power = np.abs(spectrum) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    return power, fy, fx


def striping_metrics(
    band: NDArray[np.floating],
    period_px: float = STRIPING_WAVELENGTH_PX,
    tolerance_px: float = 0.8,
    sigma_px: float = 3.0,
) -> dict[str, float]:
    """Amplitude of the 474 m component in the 2D spectrum of one band.

    The metric is targeted at the known detector period instead of taking the
    global spectral peak: on the PICS the dunes and the 6-bit quantization
    produce stronger peaks elsewhere, so ``argmax`` would measure terrain.
    Within ``period_px +- tolerance_px`` the strongest Fourier bin gives the
    sinusoid amplitude in reflectance units; ``stripe_index`` compares that
    annulus with a neighbouring background.

    Args:
        band: 2D reflectance array (the valid area of a chip or PICS).
        period_px: Along-track detector period in pixels (474 m / 60 m).
        tolerance_px: Half width of the searched annulus.
        sigma_px: High-pass scale that removes the terrain signal.

    Returns:
        Dictionary with ``wavelength_px`` (peak inside the target annulus),
        ``wavelength_m``, ``angle_deg`` (direction of the wave vector in the
        array frame), ``amplitude`` (reflectance) and ``stripe_index``
        (target power over the neighbouring background).
    """
    if band.ndim != 2:
        raise ValueError(f"band must be 2D, got shape {band.shape}")
    if period_px <= tolerance_px:
        raise ValueError("period_px must be larger than tolerance_px")
    if tolerance_px <= 0:
        raise ValueError("tolerance_px must be positive")

    power, fy, fx = power_spectrum(band, sigma_px=sigma_px)
    grid_fy, grid_fx = np.meshgrid(fy, fx, indexing="ij")
    freq = np.hypot(grid_fy, grid_fx)
    with np.errstate(divide="ignore"):
        wavelength = np.where(freq > 0, 1.0 / np.maximum(freq, 1e-12), np.inf)

    target = np.abs(wavelength - period_px) <= tolerance_px
    if not np.any(target):
        raise ValueError("the search annulus is empty; the array is too small")
    background = (
        ((wavelength >= 5.0) & (wavelength < period_px - tolerance_px))
        | ((wavelength > period_px + tolerance_px) & (wavelength <= 12.0))
    )

    flat = np.flatnonzero(target)
    peak_index = int(flat[int(np.argmax(power.ravel()[flat]))])
    i, j = np.unravel_index(peak_index, power.shape)
    peak_power = float(power[i, j])
    window_sum = float(np.outer(np.hanning(band.shape[0]), np.hanning(band.shape[1])).sum())
    amplitude = 2.0 * np.sqrt(peak_power) / window_sum

    background_power = float(np.mean(power[background])) if np.any(background) else float("nan")
    mean_target = float(np.mean(power[target]))

    return {
        "wavelength_px": float(wavelength[i, j]),
        "wavelength_m": float(wavelength[i, j] * 60.0),
        "angle_deg": float(np.degrees(np.arctan2(grid_fy[i, j], grid_fx[i, j]))),
        "amplitude": amplitude,
        "stripe_index": mean_target / background_power if background_power else float("nan"),
    }


def radial_psd(
    band: NDArray[np.floating],
    n_bins: int = 48,
    sigma_px: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged power spectrum.

    Args:
        band: 2D reflectance array.
        n_bins: Number of radial bins between the DC and the Nyquist
            frequency.
        sigma_px: High-pass scale.

    Returns:
        ``(wavelength_px, mean_power)`` for bins with at least one sample,
        sorted by increasing wavelength (decreasing frequency).
    """
    power, fy, fx = power_spectrum(band, sigma_px=sigma_px)
    grid_fy, grid_fx = np.meshgrid(fy, fx, indexing="ij")
    freq = np.hypot(grid_fy, grid_fx).ravel()
    values = power.ravel()
    keep = (freq > 0) & (freq <= 0.5)
    freq, values = freq[keep], values[keep]

    edges = np.linspace(0.0, 0.5, n_bins + 1)
    which = np.digitize(freq, edges) - 1
    valid = (which >= 0) & (which < n_bins)
    which, freq, values = which[valid], freq[valid], values[valid]

    sums = np.bincount(which, weights=values, minlength=n_bins)
    counts = np.bincount(which, minlength=n_bins)
    ok = counts > 0
    centres = 0.5 * (edges[:-1] + edges[1:])[ok]
    mean_power = sums[ok] / counts[ok]
    order = np.argsort(centres)[::-1]  # longest wavelength (lowest frequency) first
    return 1.0 / centres[order], mean_power[order]
