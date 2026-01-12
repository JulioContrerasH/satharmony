"""High-level functions for S2 to MSS emulation."""

import numpy as np
from numpy.typing import NDArray

from .config import PipelineConfig, Range
from .labels import aggregate_cloudsen12
from .pipeline import MSSEmulator


def emulate_s2(
    image: NDArray[np.floating],
    scale_factor: int | None = None,
    seed: int | None = None,
    config: PipelineConfig | None = None,
) -> tuple[NDArray[np.float32], int]:
    """Emulate MSS from S2 image.

    Args:
        image: S2 array (C, H, W) float32, reflectance [0, 1+]
        scale_factor: Fixed downsampling factor. If None, sampled from config.
        seed: Random seed for reproducibility.
        config: PipelineConfig (optional).

    Returns:
        Tuple of (mss_image, scale_factor_used)

    Example:
        >>> mss, sf = emulate_s2(s2_image, seed=42)
        >>> labels_mss = aggregate_cloudsen12(labels, scale_factor=sf)
    """
    cfg = config or PipelineConfig()

    if seed is not None:
        cfg.seed = seed

    # Fix scale_factor if provided
    if scale_factor is not None:
        target_gsd = scale_factor * cfg.spatial.input_gsd
        cfg.spatial.target_gsd = Range(min=target_gsd, max=target_gsd)

    # Run emulation
    emulator = MSSEmulator(cfg)
    mss = emulator(image)

    # Calculate actual scale_factor used
    sf_used = image.shape[1] // mss.shape[1]

    return mss, sf_used


def emulate_labels(
    labels: NDArray[np.uint8],
    scale_factor: int,
) -> NDArray[np.uint8]:
    """Aggregate CloudSEN12 labels to MSS resolution.

    Args:
        labels: Input (H, W) uint8 with CloudSEN12 classes
        scale_factor: Downsampling factor (must match image emulation)

    Returns:
        Aggregated labels (H', W')
    """
    return aggregate_cloudsen12(labels, scale_factor=scale_factor)


def emulate_s2_with_labels(
    image: NDArray[np.floating],
    labels: NDArray[np.uint8],
    scale_factor: int | None = None,
    seed: int | None = None,
    config: PipelineConfig | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.uint8], int]:
    """Emulate S2 image AND labels together, guaranteeing same dimensions.

    This is the recommended function when you have both image and labels,
    as it ensures consistent scale_factor for both outputs.

    Args:
        image: S2 array (C, H, W) float32, reflectance [0, 1+]
        labels: CloudSEN12 labels (H, W) uint8
        scale_factor: Fixed downsampling factor. If None, sampled from config.
        seed: Random seed for reproducibility.
        config: PipelineConfig (optional).

    Returns:
        Tuple of (mss_image, mss_labels, scale_factor_used)
        Guaranteed: mss_image.shape[1:] == mss_labels.shape

    Example:
        >>> mss, mss_labels, sf = emulate_s2_with_labels(s2_image, labels, seed=42)
        >>> assert mss.shape[1:] == mss_labels.shape  # Always true
    """
    # Emulate image first
    mss, sf_used = emulate_s2(
        image,
        scale_factor=scale_factor,
        seed=seed,
        config=config,
    )

    # Aggregate labels with SAME scale_factor
    mss_labels = aggregate_cloudsen12(labels, scale_factor=sf_used)

    # Verify dimensions match
    assert (
        mss.shape[1:] == mss_labels.shape
    ), f"Shape mismatch: image {mss.shape[1:]} vs labels {mss_labels.shape}"

    return mss, mss_labels, sf_used
