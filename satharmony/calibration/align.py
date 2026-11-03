"""Subpixel alignment of an emulated MSS prediction to a real MSS scene.

The two rasters already share a UTM grid; the only real offset is the
geolocation difference between the products, a few 60 m pixels at most.  The
estimator is phase correlation with a parabolic subpixel peak, optionally
refined with OpenCV ECC when it is installed.  A hard ``max_shift`` keeps the
correlator from locking onto a dune field or a lake shore kilometres away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import shift as nd_shift

from ..units import assert_toa_units


@dataclass(frozen=True)
class AlignResult:
    """Translation that takes ``moving`` onto ``reference``.

    Attributes:
        dy: Row shift in pixels, positive down.
        dx: Column shift in pixels, positive right.
        quality: Pearson correlation after the shift on the valid pixels.
        method: ``"phase"`` or ``"phase+ecc"``.
        peak: Phase-correlation peak height, normalized to [0, 1].
    """

    dy: float
    dx: float
    quality: float
    method: str
    peak: float


def _window(shape: tuple[int, int]) -> NDArray[np.float64]:
    h, w = shape
    return np.outer(np.hanning(h), np.hanning(w))


def _parabolic(corr: NDArray[np.float64], row: int, col: int, axis: int) -> float:
    """Subpixel offset from the three samples around the peak."""
    n = corr.shape[axis]
    index = row if axis == 0 else col
    if index <= 0 or index >= n - 1:
        return 0.0
    if axis == 0:
        left, center, right = corr[row - 1, col], corr[row, col], corr[row + 1, col]
    else:
        left, center, right = corr[row, col - 1], corr[row, col], corr[row, col + 1]
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return 0.0
    return float(0.5 * (left - right) / denominator)


def phase_shift(
    reference: NDArray[np.floating],
    moving: NDArray[np.floating],
    max_shift: float | None = None,
) -> tuple[float, float, float]:
    """Phase-correlation shift and peak height.

    Args:
        reference: Fixed image, 2D.
        moving: Image to register, same shape.
        max_shift: Reject shifts larger than this many pixels.

    Returns:
        ``(dy, dx, peak)``.  The shift is clipped to zero when it exceeds
        ``max_shift``.

    Raises:
        ValueError: If the shapes differ or are not 2D.
    """
    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    if ref.ndim != 2 or mov.ndim != 2:
        raise ValueError("phase_shift expects two 2D arrays")
    if ref.shape != mov.shape:
        raise ValueError(f"shape mismatch: {ref.shape} != {mov.shape}")

    window = _window(ref.shape)
    ref = (ref - np.nanmean(ref)) * window
    mov = (mov - np.nanmean(mov)) * window
    ref = np.nan_to_num(ref, nan=0.0)
    mov = np.nan_to_num(mov, nan=0.0)

    cross = np.fft.rfft2(ref) * np.conj(np.fft.rfft2(mov))
    cross /= np.abs(cross) + 1e-12
    corr = np.fft.irfft2(cross, s=ref.shape)

    row, col = np.unravel_index(int(np.argmax(corr)), corr.shape)
    peak = float(corr[row, col] / (np.std(corr) + 1e-12))
    # The cross-power A * conj(B) peaks at the shift of the reference
    # relative to the moving image; the translation to apply is the negative.
    peak_y = float(row) + _parabolic(corr, row, col, axis=0)
    peak_x = float(col) + _parabolic(corr, row, col, axis=1)
    h, w = ref.shape
    if peak_y > h / 2:
        peak_y -= h
    if peak_x > w / 2:
        peak_x -= w
    dy = peak_y
    dx = peak_x

    if max_shift is not None and max(abs(dy), abs(dx)) > max_shift:
        return 0.0, 0.0, 0.0
    return dy, dx, peak


def apply_shift(
    image: NDArray[np.floating], dy: float, dx: float, fill: float = np.nan
) -> NDArray[np.floating]:
    """Shift a 2D or 3D array by ``(dy, dx)`` without wrapping."""
    data = np.asarray(image, dtype=np.float64)
    if data.ndim == 2:
        return nd_shift(data, (dy, dx), order=3, mode="constant", cval=fill)
    return np.stack(
        [nd_shift(band, (dy, dx), order=3, mode="constant", cval=fill) for band in data]
    )


def _correlation(
    reference: NDArray[np.floating],
    moving: NDArray[np.floating],
    valid: NDArray[np.bool_] | None,
) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    ok = np.isfinite(ref) & np.isfinite(mov) if valid is None else valid & np.isfinite(ref) & np.isfinite(mov)
    if ok.sum() < 100:
        return float("nan")
    return float(np.corrcoef(ref[ok], mov[ok])[0, 1])


def align_pair(
    reference: NDArray[np.floating],
    moving: NDArray[np.floating],
    valid: NDArray[np.bool_] | None = None,
    max_shift: float = 6.0,
    refine: bool = True,
) -> AlignResult:
    """Align ``moving`` to ``reference`` and score the result.

    Args:
        reference: Real MSS band, 2D.
        moving: Emulated MSS band, 2D, same shape.
        valid: Optional mask of usable pixels.
        max_shift: Reject shifts beyond this many pixels.
        refine: Try OpenCV ECC around the phase estimate when available.

    Returns:
        The translation, its post-shift correlation and the method used.
    """
    assert_toa_units(reference, name="align reference")
    assert_toa_units(moving, name="align moving")
    dy, dx, peak = phase_shift(reference, moving, max_shift=max_shift)
    method = "phase"
    if refine:
        try:
            import cv2
        except ImportError:
            cv2 = None
        if cv2 is not None:
            template = np.ascontiguousarray(
                ((reference - np.nanmean(reference)) / (np.nanstd(reference) + 1e-9)).astype(
                    np.float32
                )
            )
            image = np.ascontiguousarray(
                ((moving - np.nanmean(moving)) / (np.nanstd(moving) + 1e-9)).astype(np.float32)
            )
            warp = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
            try:
                _, warp = cv2.findTransformECC(
                    template, image, warp, cv2.MOTION_TRANSLATION, criteria, None, 5
                )
                ecc_dy, ecc_dx = float(warp[1, 2]), float(warp[0, 2])
                if max(abs(ecc_dy), abs(ecc_dx)) <= max_shift:
                    dy, dx = dy + ecc_dy, dx + ecc_dx
                    method = "phase+ecc"
            except cv2.error:
                pass

    shifted = apply_shift(moving, dy, dx)
    quality = _correlation(reference, shifted, valid)
    return AlignResult(dy=dy, dx=dx, quality=quality, method=method, peak=peak)
