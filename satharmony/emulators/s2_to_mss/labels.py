"""CloudSEN12 label aggregation for S2→MSS emulation (10m → 60m)."""

import numpy as np
from numpy.typing import NDArray

# CloudSEN12 class encoding (Aybar et al. 2022, Table 3).
# 0 = clear, 1 = thick cloud, 2 = thin cloud, 3 = cloud shadow.
# The 0.1.13 code had 1 = thin and 2 = thick, so the two were swapped.
CLEAR = 0
THICK_CLOUD = 1
THIN_CLOUD = 2
SHADOW = 3

CLASS_NAMES: dict[int, str] = {
    CLEAR: "clear",
    THICK_CLOUD: "thick_cloud",
    THIN_CLOUD: "thin_cloud",
    SHADOW: "cloud_shadow",
}

# Fail loudly if a future edit reintroduces the 0.1.13 swap.
if (CLEAR, THICK_CLOUD, THIN_CLOUD, SHADOW) != (0, 1, 2, 3):
    raise RuntimeError("CloudSEN12 encoding must be 0=clear, 1=thick, 2=thin, 3=shadow")


def aggregate_cloudsen12(
    labels: NDArray[np.uint8],
    scale_factor: int = 6,
    classes: tuple[int, int, int, int] = (CLEAR, THICK_CLOUD, THIN_CLOUD, SHADOW),
    merge_cloud_classes: bool = False,
    legacy: bool = False,
    min_contaminated_fraction: float = 0.5,
) -> NDArray[np.uint8]:
    """Aggregate CloudSEN12 labels from 10m to coarser resolution.

    New rule (default): a class with >= 50 % of the block wins. If no class
    reaches 50 %, a block with >= 50 % contaminated pixels (thick + thin +
    shadow) is assigned to the most frequent contaminated class; ties follow
    the CloudSEN12 priority thick > shadow > thin. Otherwise the block is
    clear. This avoids labelling a 58 %-contaminated block as clear.

    ``legacy=True`` reproduces the 0.1.13 rule and class encoding
    (1=thin, 2=thick), which swapped thick and thin and could return clear
    for contaminated blocks.

    Args:
        labels: Input array (H, W) uint8 with CloudSEN12 classes 0-3.
        scale_factor: Downsampling factor (6 for 10m→60m, 3 for 10m→30m).
        classes: Tuple of (clear, thick_cloud, thin_cloud, shadow) values.
        merge_cloud_classes: If True, thick+thin → 1, shadow → 2.
        legacy: Reproduce the 0.1.13 behaviour.
        min_contaminated_fraction: Fraction of thick+thin+shadow pixels that
            makes a no-majority block take its most frequent contaminated
            class instead of clear. The 0.1.13 rule effectively used 1.0
            (clear won whenever it was the largest single class).

    Returns:
        Aggregated labels (H', W') where H' = H // scale_factor.
        If merge_cloud_classes=True: {0: clear, 1: cloud, 2: shadow}.
        If False: {0: clear, 1: thick, 2: thin, 3: shadow}.

    Note:
        If H or W is not divisible by scale_factor, the image is cropped.
        E.g. 1024 with scale_factor=6 → 1020 → 170x170.
    """
    H, W = labels.shape
    sf = scale_factor

    new_h = H // sf
    new_w = W // sf
    cropped = labels[: new_h * sf, : new_w * sf]

    blocks = cropped.reshape(new_h, sf, new_w, sf).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(new_h, new_w, sf * sf)

    n = sf * sf
    half_n = n // 2
    out = np.full((new_h, new_w), CLEAR, dtype=np.uint8)

    if legacy:
        # ── 0.1.13: 1=thin, 2=thick; old conservative rule ──────────────────
        clear, thin_cloud, cloud, shadow = 0, 1, 2, 3
        n_clear = np.sum(blocks == clear, axis=2)
        n_thin = np.sum(blocks == thin_cloud, axis=2)
        n_cloud = np.sum(blocks == cloud, axis=2)
        n_shadow = np.sum(blocks == shadow, axis=2)
        n_cloudy = n_cloud + n_thin

        has_majority = (
            (n_clear >= half_n) | (n_thin >= half_n) | (n_cloud >= half_n) | (n_shadow >= half_n)
        )
        out[n_clear >= half_n] = clear
        out[n_thin >= half_n] = thin_cloud
        out[n_cloud >= half_n] = cloud
        out[n_shadow >= half_n] = shadow

        no_maj = ~has_majority
        cloudy_dom = no_maj & (n_cloudy >= n_shadow)
        shadow_dom = no_maj & (n_cloudy < n_shadow)

        cloudy_vs_clear = cloudy_dom & (n_cloudy >= n_clear)
        out[cloudy_vs_clear & (n_cloud >= n_thin)] = cloud
        out[cloudy_vs_clear & (n_cloud < n_thin)] = thin_cloud
        out[cloudy_dom & (n_cloudy < n_clear)] = clear

        shadow_cloudy_clear = shadow_dom & (n_cloudy >= n_clear)
        out[shadow_cloudy_clear] = shadow
        shadow_clear_dom = shadow_dom & (n_cloudy < n_clear)
        out[shadow_clear_dom & (n_shadow >= n_clear)] = shadow
        out[shadow_clear_dom & (n_shadow < n_clear)] = clear

        if merge_cloud_classes:
            out[out == thin_cloud] = 1
            out[out == cloud] = 1
            out[out == shadow] = 2
        return out

    # ── New rule: CloudSEN12 Table 3 encoding and priority ──────────────────
    if not 0.0 < min_contaminated_fraction <= 1.0:
        raise ValueError("min_contaminated_fraction must be in (0, 1]")

    # For odd block sizes (e.g. sf=5) require ceil(n/2) to avoid two winners.
    half_n = (n + 1) // 2
    min_contaminated = int(np.ceil(min_contaminated_fraction * n - 1e-12))
    clear, thick, thin, shadow = classes
    n_clear = np.sum(blocks == clear, axis=2)
    n_thick = np.sum(blocks == thick, axis=2)
    n_thin = np.sum(blocks == thin, axis=2)
    n_shadow = np.sum(blocks == shadow, axis=2)

    has_majority = (
        (n_clear >= half_n) | (n_thick >= half_n) | (n_thin >= half_n) | (n_shadow >= half_n)
    )
    # Assign in increasing priority order; thick wins ties (last assignment)
    out[n_clear >= half_n] = clear
    out[n_thin >= half_n] = thin
    out[n_shadow >= half_n] = shadow
    out[n_thick >= half_n] = thick

    no_maj = ~has_majority
    contaminated = n_thick + n_thin + n_shadow
    contam_dom = no_maj & (contaminated >= min_contaminated)
    max_count = np.maximum.reduce([n_thick, n_thin, n_shadow])
    out[contam_dom & (n_thin == max_count)] = thin
    out[contam_dom & (n_shadow == max_count)] = shadow
    out[contam_dom & (n_thick == max_count)] = thick
    # Remaining no-majority blocks stay clear.

    if merge_cloud_classes:
        out[out == thick] = 1
        out[out == thin] = 1
        out[out == shadow] = 2

    return out
