"""Detector striping: 6-detector gain/offset mismatch."""

import numpy as np

from ..config import StripingConfig


def detector_phase(
    shape: tuple[int, int],
    *,
    sampling_gsd: float = 60.0,
    scan_line_period_m: float = 474.0,
    num_detectors: int = 6,
    yaw_deg: float = 9.0,
) -> np.ndarray:
    """Position along the ground trace in units of one detector line.

    The six MSS detectors are spaced 79 m along track, so one mirror sweep
    covers 474 m (Handbook p. 85) and the striping period is
    ``scan_line_period_m`` along the trace, not six image rows. The trace
    is tilted by ``yaw_deg`` because the ground track of the 99° orbit is
    not north-south (Taranik p. 36).

    Args:
        shape: Output grid ``(H, W)``.
        sampling_gsd: Output pixel size in metres.
        scan_line_period_m: Along-track distance of one mirror sweep (474 m).
        num_detectors: Detectors per band (6).
        yaw_deg: Rotation of the scan lines with respect to the image rows.

    Returns:
        (H, W) float32 array; ``floor(phase) % num_detectors`` is the
        detector index.
    """
    if sampling_gsd <= 0:
        raise ValueError(f"sampling_gsd must be positive, got {sampling_gsd}")
    if scan_line_period_m <= 0:
        raise ValueError(
            f"scan_line_period_m must be positive, got {scan_line_period_m}"
        )
    if num_detectors < 1:
        raise ValueError(f"num_detectors must be >= 1, got {num_detectors}")

    h, w = shape
    spacing_px = (scan_line_period_m / num_detectors) / sampling_gsd
    yaw = np.deg2rad(yaw_deg)
    yy, xx = np.mgrid[0:h, 0:w]
    along_track = yy * np.cos(yaw) + xx * np.sin(yaw)
    return (along_track / spacing_px).astype(np.float32)


def detector_line_index(
    shape: tuple[int, int],
    *,
    sampling_gsd: float = 60.0,
    scan_line_period_m: float = 474.0,
    num_detectors: int = 6,
    yaw_deg: float = 9.0,
) -> np.ndarray:
    """Detector index (0..num_detectors-1) for every output pixel."""
    phase = detector_phase(
        shape,
        sampling_gsd=sampling_gsd,
        scan_line_period_m=scan_line_period_m,
        num_detectors=num_detectors,
        yaw_deg=yaw_deg,
    )
    return (np.floor(phase).astype(np.int16) % num_detectors).astype(np.int16)


def apply_striping(
    image: np.ndarray,
    config: StripingConfig,
    sampling_gsd: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply 6-detector striping artifact.

    MSS used 6 parallel detectors per band. Each has slightly different
    gain/offset creating periodic banding every 474 m along the ground
    trace (7.9 px at 60 m), tilted with respect to the image rows. The
    legacy 0.1.13 pattern repeated every 6 rows instead.

    Args:
        image: Input array (C, H, W) float32
        config: StripingConfig
        sampling_gsd: Output pixel size; defaults to ``config.sampling_gsd``.

    Returns:
        Array with striping artifact
    """
    if not config.enabled:
        return image

    generator = rng if rng is not None else np.random.default_rng()
    C, H, W = image.shape
    n_det = config.num_detectors

    # Sample gain/offset per detector
    gain_std = config.gain_std.sample(generator)
    offset_std = config.offset_std.sample(generator)

    result = image.copy()

    if config.detector_geometry == "rows":
        # Legacy 0.1.13: one detector per image row, 6-row period.
        for c in range(C):
            # Per-band variation or same for all bands
            if config.per_band_variation or c == 0:
                gains = 1.0 + generator.normal(0, gain_std, n_det)
                offsets = generator.normal(0, offset_std, n_det)

            # Apply to each detector's lines
            for d in range(n_det):
                result[c, d::n_det, :] = result[c, d::n_det, :] * gains[d] + offsets[d]

        return result.astype(np.float32)

    gsd = config.sampling_gsd if sampling_gsd is None else sampling_gsd
    detector = detector_line_index(
        (H, W),
        sampling_gsd=gsd,
        scan_line_period_m=config.scan_line_period_m,
        num_detectors=n_det,
        yaw_deg=config.yaw_deg,
    )

    for c in range(C):
        if config.per_band_variation or c == 0:
            gains = 1.0 + generator.normal(0, gain_std, n_det)
            offsets = generator.normal(0, offset_std, n_det)
        result[c] = result[c] * gains[detector] + offsets[detector]

    return result.astype(np.float32)
