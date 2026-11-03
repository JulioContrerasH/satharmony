"""High-level functions for S2 to MSS emulation."""

import numpy as np
from numpy.typing import NDArray

from ...units import assert_toa_units
from .config import PipelineConfig, Range
from .labels import aggregate_cloudsen12
from .pipeline import MSSEmulator


def emulate_s2(
    image: NDArray[np.floating],
    scale_factor: int | None = None,
    seed: int | None = None,
    config: PipelineConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float32], int]:
    """Emulate MSS from an S2 TOA image.

    The input is ``(13, H, W)`` in the package band order
    (B1..B12, B8A) with TOA reflectance in [0, 1].  Values above 2 or below
    -0.01 are rejected: they almost always mean the caller passed DN.

    Raises:
        ValueError: If the input is not 3D or is outside the TOA range.
    """
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"expected (13, H, W) S2 TOA, got {array.shape}")
    assert_toa_units(array, name="emulate_s2")

    cfg = config.model_copy(deep=True) if config is not None else PipelineConfig()

    if seed is not None:
        cfg.seed = seed

    if scale_factor is not None:
        target_gsd = scale_factor * cfg.spatial.input_gsd
        # In the new profile the output grid is explicit; in the legacy
        # profile (sampling_gsd=None) the scale factor drives target_gsd.
        if cfg.spatial.sampling_gsd is not None:
            cfg.spatial.sampling_gsd = target_gsd
        cfg.spatial.target_gsd = Range(min=target_gsd, max=target_gsd)

    if rng is None and seed is not None and not cfg.legacy_rng:
        rng = np.random.default_rng(seed)

    emulator = MSSEmulator(cfg, rng=rng)
    mss = emulator(image)

    sf_used = image.shape[1] // mss.shape[1]

    return mss, sf_used


def emulate_labels(
    labels: NDArray[np.uint8],
    scale_factor: int,
    merge_cloud_classes: bool = False,
    legacy: bool = False,
) -> NDArray[np.uint8]:
    """Aggregate CloudSEN12 labels to MSS resolution."""
    return aggregate_cloudsen12(
        labels,
        scale_factor=scale_factor,
        merge_cloud_classes=merge_cloud_classes,
        legacy=legacy,
    )


def emulate_s2_with_labels(
    image: NDArray[np.floating],
    labels: NDArray[np.uint8],
    scale_factor: int | None = None,
    seed: int | None = None,
    config: PipelineConfig | None = None,
    merge_cloud_classes: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.uint8], int]:
    """Emulate S2 image AND labels together."""
    cfg = config if config is not None else PipelineConfig()

    mss, sf_used = emulate_s2(
        image,
        scale_factor=scale_factor,
        seed=seed,
        config=cfg,
        rng=rng,
    )

    mss_labels = aggregate_cloudsen12(
        labels,
        scale_factor=sf_used,
        merge_cloud_classes=merge_cloud_classes,
        legacy=cfg.labels_legacy,
    )

    assert mss.shape[1:] == mss_labels.shape

    return mss, mss_labels, sf_used
