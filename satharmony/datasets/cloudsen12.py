"""CloudSEN12 chips: band order, labels, valid area and stable hashes.

CloudSEN12 (Aybar et al. 2022) ships every 10 m chip as a 13-band
``s2.tif`` and a ``target.tif`` with the 0=clear, 1=thick, 2=thin,
3=cloud-shadow encoding.  The stack order of the distribucion is::

    B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12

while the emulator expects::

    B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 B12 B8A

so the two must not be confused.  ``B8A`` is the water-vapour continuum
used by :mod:`satharmony.emulators.s2_to_mss.sbaf`; putting B9 there would
silently feed the absorption band as the continuum.

The CloudSEN12 chips are padded: only the central ~509 x 509 pixels of the
1024 x 1024 grid hold data (the rest is nodata 99 in ``target.tif``).  After
the 10 m -> 60 m aggregation that leaves an 84 x 84 block of fully valid
cells, which is the only region a chip may come from.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..emulators.s2_to_mss.labels import CLASS_NAMES, aggregate_cloudsen12
from ..emulators.s2_to_mss.sbaf import S2_BAND_ORDER

#: Band order of the CloudSEN12 13-band stack.
CLOUDSEN12_BAND_ORDER: tuple[str, ...] = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
)

#: Band order expected by :func:`satharmony.emulators.s2_to_mss.sbaf.sbaf_transform`.
PACKAGE_BAND_ORDER: tuple[str, ...] = tuple(S2_BAND_ORDER)

#: For every package band, its index in the CloudSEN12 stack.
BAND_PERMUTATION: tuple[int, ...] = tuple(
    CLOUDSEN12_BAND_ORDER.index(band) for band in PACKAGE_BAND_ORDER
)

#: Nodata value used by CloudSEN12 ``target.tif``.
LABEL_NODATA = 99

#: CloudSEN12 chip edge in pixels (10 m).
CHIP_SIZE_PX = 1024

#: Scale factor between the 10 m chip and the 60 m ESA grid.
SCALE_FACTOR = 6

#: Fully valid cells left after the 10 m -> 60 m aggregation.
VALID_CELLS = 84


@dataclass(frozen=True)
class CloudSEN12Scene:
    """One CloudSEN12 chip on its 10 m grid and the 60 m labels."""

    scene_id: str
    s2: NDArray[np.float32]
    labels: NDArray[np.uint8]
    labels60: NDArray[np.uint8]
    valid60: NDArray[np.bool_]
    metadata: dict


def to_package_order(image: NDArray[np.floating]) -> NDArray[np.float32]:
    """Reorder a CloudSEN12 stack to the emulator band order.

    Args:
        image: Array ``(13, H, W)`` with CloudSEN12 band order.

    Returns:
        Array ``(13, H, W)`` float32 in :data:`PACKAGE_BAND_ORDER`.

    Raises:
        ValueError: If the stack does not have 13 bands.
    """
    if image.ndim != 3 or image.shape[0] != len(CLOUDSEN12_BAND_ORDER):
        raise ValueError(
            "CloudSEN12 stack must be (13, H, W), "
            f"got {image.shape}; expected {len(CLOUDSEN12_BAND_ORDER)} bands"
        )
    return np.ascontiguousarray(image[list(BAND_PERMUTATION)], dtype=np.float32)


def read_scene(scene_dir: str | Path, scale_factor: int = SCALE_FACTOR) -> CloudSEN12Scene:
    """Read one CloudSEN12 chip and aggregate its labels to 60 m.

    ``s2.tif`` is stored as uint16 DN at 10 m; dividing by 10000 gives TOA
    reflectance, which is the unit the emulator and the SBAF expect.
    ``target.tif`` carries nodata 99 outside the real projection; blocks with
    any nodata are excluded instead of being counted as clear.

    Args:
        scene_dir: Chip directory with ``s2.tif`` and ``target.tif``.
        scale_factor: Aggregation factor (6 for 10 m -> 60 m).

    Returns:
        A :class:`CloudSEN12Scene` with reflectance, labels on both grids and
        the boolean mask of fully valid 60 m cells.

    Raises:
        FileNotFoundError: If either raster is missing.
    """
    import rasterio

    scene_dir = Path(scene_dir)
    s2_path = scene_dir / "s2.tif"
    label_path = scene_dir / "target.tif"
    if not s2_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"{scene_dir} is not a CloudSEN12 chip directory")

    with rasterio.open(s2_path) as src:
        stack = src.read().astype(np.float32) / 10000.0
    with rasterio.open(label_path) as src:
        raw = src.read(1)

    valid10 = raw != LABEL_NODATA
    labels = np.where(valid10, raw, 0).astype(np.uint8)
    labels60 = aggregate_cloudsen12(labels, scale_factor=scale_factor)
    valid60 = valid_cells(valid10, scale_factor=scale_factor)

    metadata: dict = {}
    meta_path = scene_dir / "metadata.json"
    if meta_path.exists():
        import json

        metadata = json.loads(meta_path.read_text())

    return CloudSEN12Scene(
        scene_id=scene_dir.name,
        s2=to_package_order(stack),
        labels=labels,
        labels60=labels60,
        valid60=valid60,
        metadata=metadata,
    )


def valid_cells(valid10: NDArray[np.bool_], scale_factor: int = SCALE_FACTOR) -> NDArray[np.bool_]:
    """60 m cells whose 10 m block is entirely valid.

    Args:
        valid10: Boolean mask on the 10 m grid.
        scale_factor: Aggregation factor.

    Returns:
        Boolean mask ``(H // scale_factor, W // scale_factor)``.  A cell is
        True only when all of its ``scale_factor ** 2`` pixels are valid.
    """
    sf = scale_factor
    h, w = valid10.shape
    new_h, new_w = h // sf, w // sf
    blocks = valid10[: new_h * sf, : new_w * sf].reshape(new_h, sf, new_w, sf)
    return blocks.all(axis=(1, 3))


def select_window(
    valid60: NDArray[np.bool_],
    size: int,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    """Pick a random fully valid ``size x size`` window of 60 m cells.

    Args:
        valid60: Boolean mask of fully valid cells.
        size: Window edge in cells.
        rng: Seeded generator, so the fixture is reproducible.

    Returns:
        ``(row, col)`` of the window top-left corner, or ``None`` when no
        window fits inside the valid block.
    """
    h, w = valid60.shape
    if size > h or size > w:
        return None
    candidates = [
        (r, c)
        for r in range(0, h - size + 1)
        for c in range(0, w - size + 1)
        if valid60[r : r + size, c : c + size].all()
    ]
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def class_fractions(labels: NDArray[np.integer], ignore: tuple[int, ...] = ()) -> dict[str, float]:
    """Fraction of every CloudSEN12 class in a label raster.

    Args:
        labels: Label raster with classes 0-3.
        ignore: Extra values to drop (e.g. 255 from an annotation mask).

    Returns:
        Dictionary keyed by :data:`CLASS_NAMES` plus ``"contaminated"``
        (thick + thin + shadow).  Fractions sum to 1 over the counted
        pixels; an empty raster returns zeros.
    """
    values = np.asarray(labels).ravel()
    keep = np.ones(values.shape, dtype=bool)
    for value in ignore:
        keep &= values != value
    values = values[keep]
    total = int(values.size)
    counts = {name: int(np.count_nonzero(values == code)) for code, name in CLASS_NAMES.items()}
    if total:
        fractions = {name: count / total for name, count in counts.items()}
    else:
        fractions = dict.fromkeys(CLASS_NAMES.values(), 0.0)
    fractions["contaminated"] = (
        fractions["thick_cloud"] + fractions["thin_cloud"] + fractions["cloud_shadow"]
    )
    return fractions


def regime_of(fractions: dict[str, float]) -> str:
    """Cloud regime of a chip, from the contaminated pixel fraction.

    The bins are ``clear`` (<2 %), ``low_cloud`` (2-15 %), ``mid_cloud``
    (15-50 %) and ``high_cloud`` (>=50 %).  Shadow is a separate flag
    (:func:`is_shadow_rich`) so it does not steal chips from the cloud bins.

    Args:
        fractions: Output of :func:`class_fractions`.

    Returns:
        One of ``clear``, ``low_cloud``, ``mid_cloud`` or ``high_cloud``.
    """
    contaminated = fractions.get("contaminated", 0.0)
    if contaminated < 0.02:
        return "clear"
    if contaminated < 0.15:
        return "low_cloud"
    if contaminated < 0.50:
        return "mid_cloud"
    return "high_cloud"


def is_shadow_rich(fractions: dict[str, float], threshold: float = 0.10) -> bool:
    """Whether a chip has enough shadow pixels to be useful for that class."""
    return fractions.get("cloud_shadow", 0.0) >= threshold


def array_sha256(image: NDArray[np.integer], labels: NDArray[np.integer]) -> str:
    """Stable SHA-256 over the arrays, their dtype and their shape.

    Hashing the raw bytes of a ``.npy``/``.npz`` file is not reproducible
    because the container stores timestamps; this hash depends only on the
    values.

    Args:
        image: Chip image array.
        labels: Chip label array.

    Returns:
        Hex digest.
    """
    digest = hashlib.sha256()
    for array in (image, labels):
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
