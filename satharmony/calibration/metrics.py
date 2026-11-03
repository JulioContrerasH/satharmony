"""Metrics for radiometric and spatial validation of the emulator."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from ..units import assert_toa_units


def _as_pixels(
    array: NDArray[np.floating], valid: NDArray[np.bool_] | None = None
) -> NDArray[np.float64]:
    """Flatten ``(C, H, W)`` or ``(H, W)`` to ``(n_pixels, C)``."""
    data = np.asarray(array, dtype=np.float64)
    if data.ndim == 2:
        data = data[None, ...]
    if data.ndim != 3:
        raise ValueError(f"expected (C, H, W) or (H, W), got {data.shape}")
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != data.shape[1:]:
            raise ValueError(f"mask {mask.shape} does not match {data.shape[1:]}")
        data = data[:, mask]
    return data.reshape(data.shape[0], -1).T


def sam(
    prediction: NDArray[np.floating],
    observation: NDArray[np.floating],
    valid: NDArray[np.bool_] | None = None,
) -> float:
    """Mean spectral angle in degrees between two reflectance cubes."""
    pred = _as_pixels(prediction, valid)
    obs = _as_pixels(observation, valid)
    if pred.shape != obs.shape:
        raise ValueError(f"shape mismatch: {pred.shape} != {obs.shape}")
    numerator = np.sum(pred * obs, axis=1)
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(obs, axis=1)
    cosine = np.clip(numerator / np.maximum(denominator, 1e-12), -1.0, 1.0)
    return float(np.degrees(np.mean(np.arccos(cosine))))


def rmse(
    prediction: NDArray[np.floating],
    observation: NDArray[np.floating],
    valid: NDArray[np.bool_] | None = None,
) -> float:
    """Root mean squared error over all bands and valid pixels."""
    pred = _as_pixels(prediction, valid)
    obs = _as_pixels(observation, valid)
    if pred.shape != obs.shape:
        raise ValueError(f"shape mismatch: {pred.shape} != {obs.shape}")
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def bias_ci(
    prediction: NDArray[np.floating],
    observation: NDArray[np.floating],
    valid: NDArray[np.bool_] | None = None,
    n_boot: int = 1000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Mean bias and bootstrap confidence interval over pixels.

    Args:
        prediction: Predicted reflectance.
        observation: Reference reflectance.
        valid: Optional spatial mask.
        n_boot: Bootstrap replicates.
        level: Confidence level.
        rng: Optional generator for the bootstrap.

    Returns:
        ``{"bias", "low", "high", "n"}``.
    """
    pred = _as_pixels(prediction, valid)
    obs = _as_pixels(observation, valid)
    if pred.shape != obs.shape:
        raise ValueError(f"shape mismatch: {pred.shape} != {obs.shape}")
    difference = (pred - obs).ravel()
    generator = rng if rng is not None else np.random.default_rng()
    index = generator.integers(0, difference.size, size=(n_boot, difference.size))
    medians = difference[index].mean(axis=1)
    alpha = 100 * (1 - level) / 2
    low, high = np.percentile(medians, [alpha, 100 - alpha])
    return {
        "bias": float(difference.mean()),
        "low": float(low),
        "high": float(high),
        "n": int(difference.size),
    }


def snr_db(
    image: NDArray[np.floating],
    valid: NDArray[np.bool_] | None = None,
    sigma_px: float = 3.0,
) -> float:
    """Signal-to-noise ratio from the low/high-pass split of one band.

    The signal is the Gaussian-smoothed image and the noise is the residual;
    this is a texture-aware proxy, not a detector model.
    """
    data = np.asarray(image, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"snr_db expects a 2D band, got {data.shape}")
    low = gaussian_filter(data, sigma=sigma_px)
    high = data - low
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        low, high = low[mask], high[mask]
    signal = float(np.std(low))
    noise = float(np.std(high))
    if noise <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(signal / noise))


def edge_fwhm(profile: NDArray[np.floating]) -> float:
    """FWHM in pixels of the derivative of a monotonic edge profile."""
    values = np.asarray(profile, dtype=np.float64).ravel()
    if values.size < 5:
        raise ValueError("edge profile is too short")
    values = (values - values.min()) / max(values.max() - values.min(), 1e-12)
    derivative = np.gradient(values)
    derivative = np.clip(derivative, 0, None)
    peak = int(np.argmax(derivative))
    half = derivative[peak] / 2.0
    left = peak
    while left > 0 and derivative[left] > half:
        left -= 1
    right = peak
    while right < derivative.size - 1 and derivative[right] > half:
        right += 1
    if right == left:
        return 0.0
    return float(right - left)


def c2st(
    real: NDArray[np.floating],
    emulated: NDArray[np.floating],
    n_splits: int = 3,
    seed: int = 0,
) -> float:
    """Classifier two-sample test accuracy (0.5 = indistinguishable).

    A logistic regression is trained to separate the two samples; the score
    is cross-validated accuracy.  Requires scikit-learn.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("c2st needs scikit-learn") from error

    first = _as_pixels(real)
    second = _as_pixels(emulated)
    if first.shape[1] != second.shape[1]:
        raise ValueError("real and emulated must have the same number of bands")
    features = np.vstack([first, second])
    labels = np.r_[np.zeros(len(first)), np.ones(len(second))]
    model = LogisticRegression(max_iter=1000)
    scores = cross_val_score(model, features, labels, cv=n_splits)
    return float(np.mean(scores))


def summarize(
    prediction: NDArray[np.floating],
    observation: NDArray[np.floating],
    valid: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """One-line radiometric summary: SAM, RMSE and bias."""
    assert_toa_units(prediction, name="metrics prediction")
    assert_toa_units(observation, name="metrics observation")
    return {
        "sam_deg": sam(prediction, observation, valid),
        "rmse": rmse(prediction, observation, valid),
        "bias": float(np.mean(_as_pixels(prediction, valid) - _as_pixels(observation, valid))),
    }
