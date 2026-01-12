"""CloudSEN12 label aggregation for S2→MSS emulation (10m → 60m)."""

import numpy as np
from numpy.typing import NDArray

# CloudSEN12 class encoding
CLEAR = 0
THIN_CLOUD = 1
CLOUD = 2
SHADOW = 3


def aggregate_cloudsen12(
    labels: NDArray[np.uint8],
    scale_factor: int = 6,
    classes: tuple[int, int, int, int] = (CLEAR, THIN_CLOUD, CLOUD, SHADOW),
) -> NDArray[np.uint8]:
    """Aggregate CloudSEN12 labels from 10m to coarser resolution.

    Uses conservative criteria from Roy et al.: given balanced mix of labels,
    precedence is cloud > shadow > thin_cloud > clear.

    Reference: Fig. 2 in "Cloud and cloud shadow..." (Roy et al.)

    Args:
        labels: Input array (H, W) uint8 with class values 0-3
        scale_factor: Downsampling factor (6 for 10m→60m, 3 for 10m→30m)
        classes: Tuple of (clear, thin_cloud, cloud, shadow) class values

    Returns:
        Aggregated labels (H', W') where H' = H // scale_factor

    Note:
        If H or W not divisible by scale_factor, image is cropped.
        E.g., 1024 with scale_factor=6 → crops to 1020 → output 170x170
    """
    clear, thin_cloud, cloud, shadow = classes
    H, W = labels.shape
    sf = scale_factor

    # Crop to multiple of scale_factor
    new_h = H // sf
    new_w = W // sf
    cropped = labels[: new_h * sf, : new_w * sf]

    # Reshape to (new_h, sf, new_w, sf) then transpose to (new_h, new_w, sf, sf)
    blocks = cropped.reshape(new_h, sf, new_w, sf).transpose(0, 2, 1, 3)
    # Now (new_h, new_w, sf*sf) for counting
    blocks = blocks.reshape(new_h, new_w, sf * sf)

    n = sf * sf  # Total pixels per block (e.g., 36 for 6x6)
    half_n = n // 2

    # Count each class per block
    n_clear = np.sum(blocks == clear, axis=2)
    n_thin = np.sum(blocks == thin_cloud, axis=2)
    n_cloud = np.sum(blocks == cloud, axis=2)
    n_shadow = np.sum(blocks == shadow, axis=2)
    n_cloudy = n_cloud + n_thin

    # Initialize output
    out = np.full((new_h, new_w), clear, dtype=np.uint8)

    # Check majority (any class >= n/2)
    has_majority = (
        (n_clear >= half_n) | (n_thin >= half_n) | (n_cloud >= half_n) | (n_shadow >= half_n)
    )

    # Assign majority class where exists
    out[n_clear >= half_n] = clear
    out[n_thin >= half_n] = thin_cloud
    out[n_cloud >= half_n] = cloud
    out[n_shadow >= half_n] = shadow

    # For non-majority pixels, apply conservative criteria (Fig. 2)
    no_maj = ~has_majority

    # Branch: n_cloudy >= n_shadow
    cloudy_dom = no_maj & (n_cloudy >= n_shadow)
    shadow_dom = no_maj & (n_cloudy < n_shadow)

    # Cloudy dominant branch
    cloudy_vs_clear = cloudy_dom & (n_cloudy >= n_clear)
    out[cloudy_vs_clear & (n_cloud >= n_thin)] = cloud
    out[cloudy_vs_clear & (n_cloud < n_thin)] = thin_cloud
    out[cloudy_dom & (n_cloudy < n_clear)] = clear

    # Shadow dominant branch
    shadow_cloudy_clear = shadow_dom & (n_cloudy >= n_clear)
    out[shadow_cloudy_clear] = shadow

    shadow_clear_dom = shadow_dom & (n_cloudy < n_clear)
    out[shadow_clear_dom & (n_shadow >= n_clear)] = shadow
    out[shadow_clear_dom & (n_shadow < n_clear)] = clear

    return out
