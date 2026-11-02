"""Fixed per-band affine radiometry between emulated and real Landsat MSS.

The emulated pseudo-MSS is internally consistent but not on the same
reflectance scale as the real MSS of the v7 pairs.  Over the 238 train and
validation scenes, the real TOA is brighter and more variable than the
fixture: the per-band mean is 1.33-1.53x larger and the standard deviation,
1.40-1.53x larger.  A model trained on the fixture therefore sees every real
pixel above its training distribution, and a per-image z-score would fix that
only by throwing away physical TOA.

This module applies one **fixed** scale and offset per band::

    real ~ scale * pseudo + offset

The coefficients are fitted once, on a split that does not include the test
scenes, and stored in JSON.  Every chip, every scene and every inference uses
the same numbers; nothing is computed per image.

The correction changes radiometry, not the input convention: the model still
sees TOA 0-1 and its checkpoint still carries a fixed per-band mean/std.  That
mean/std is recomputed on the calibrated training chips when the model is
retrained.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Band names in the emulator order (MSS green, red, NIR1, NIR2).
DEFAULT_BANDS = ("green", "red", "nir1", "nir2")

#: Calibrated reflectance is stored as uint16 with this factor.
SCALE_FACTOR = 10000

#: The emulated sensor saturates at reflectance 1.0; the affine map clips there.
TOA_MAX = 1.0


def _as_cube(array: NDArray[np.floating] | NDArray[np.integer], name: str) -> NDArray:
    """Accept ``(C, H, W)`` or ``(N, C, H, W)`` and return a float64 view."""
    data = np.asarray(array, dtype=np.float64)
    if data.ndim == 3:
        return data[None, ...]
    if data.ndim == 4:
        return data
    raise ValueError(f"{name}: expected (C, H, W) or (N, C, H, W), got {data.shape}")


def _band_moments(
    array: NDArray[np.floating] | NDArray[np.integer],
    valid: NDArray[np.bool_] | None,
    name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Per-band mean and standard deviation over the valid pixels.

    Args:
        array: Reflectance cube ``(C, H, W)`` or ``(N, C, H, W)``.
        valid: Optional mask broadcastable to the cube's spatial shape
            ``(H, W)`` or ``(N, H, W)``.
        name: Label for error messages.

    Returns:
        ``(mean, std, n_pixels)``, each band independent.

    Raises:
        ValueError: If the cube is empty or a band has zero variance.
    """
    cube = _as_cube(array, name)
    if cube.size == 0:
        raise ValueError(f"{name}: empty cube")
    if valid is None:
        values = cube.reshape(cube.shape[0], cube.shape[1], -1).transpose(0, 2, 1)
        values = values.reshape(-1, cube.shape[1])
    else:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != cube.shape[:-3] + cube.shape[-2:]:
            try:
                mask = np.broadcast_to(mask, cube.shape[:-3] + cube.shape[-2:])
            except ValueError as error:
                raise ValueError(
                    f"{name}: mask {mask.shape} does not match cube {cube.shape}"
                ) from error
        values = cube.transpose(0, 2, 3, 1)[mask]  # (n, C)
    if values.size == 0:
        raise ValueError(f"{name}: no valid pixels")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    zero = np.flatnonzero(std <= 1e-12)
    if zero.size:
        raise ValueError(f"{name}: zero variance in band(s) {zero.tolist()}")
    return mean, std, int(values.shape[0])


@dataclass(frozen=True)
class AffineRadiometry:
    """One fixed scale and offset per band, in the emulator band order.

    Attributes:
        scale: Multiplicative factor per band.
        offset: Additive term per band, in reflectance units.
        bands: Band names, same order as ``scale`` and ``offset``.
        method: How the coefficients were obtained.
        source: Domain the coefficients are applied to.
        target: Domain they map into.
        split: Scenes used to fit them (documentation only).
        n_pixels: Valid pixels behind the fit (documentation only).
    """

    scale: tuple[float, ...]
    offset: tuple[float, ...]
    bands: tuple[str, ...] = DEFAULT_BANDS
    method: str = "moment matching over all valid pixels"
    source: str = "emulated pseudo-MSS"
    target: str = "real v7 Landsat MSS"
    split: str = "train+validation"
    n_pixels: int = 0

    def __post_init__(self) -> None:
        if len(self.scale) != len(self.offset) or len(self.scale) != len(self.bands):
            raise ValueError(
                "scale, offset and bands must have the same length "
                f"({len(self.scale)}, {len(self.offset)}, {len(self.bands)})"
            )
        if not np.all(np.isfinite(self.scale)) or not np.all(np.isfinite(self.offset)):
            raise ValueError("scale and offset must be finite")

    @property
    def n_bands(self) -> int:
        """Number of bands the transform expects."""
        return len(self.scale)

    def as_arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(scale, offset)`` as float64 arrays shaped ``(C,)``."""
        return (
            np.asarray(self.scale, dtype=np.float64),
            np.asarray(self.offset, dtype=np.float64),
        )

    @classmethod
    def from_moments(
        cls,
        source_mean: NDArray[np.floating],
        source_std: NDArray[np.floating],
        target_mean: NDArray[np.floating],
        target_std: NDArray[np.floating],
        **metadata: Any,
    ) -> AffineRadiometry:
        """Fit the affine map from per-band moments of both domains.

        Matching the first two moments gives ``scale = std_t / std_s`` and
        ``offset = mean_t - scale * mean_s``.

        Args:
            source_mean: Per-band mean of the emulated domain.
            source_std: Per-band standard deviation of the emulated domain.
            target_mean: Per-band mean of the real domain.
            target_std: Per-band standard deviation of the real domain.
            **metadata: Passed to the dataclass (``bands``, ``method``, ...).

        Returns:
            The fitted transform.

        Raises:
            ValueError: If the inputs disagree in length or a source std is
                zero.
        """
        mean_s = np.asarray(source_mean, dtype=np.float64).ravel()
        std_s = np.asarray(source_std, dtype=np.float64).ravel()
        mean_t = np.asarray(target_mean, dtype=np.float64).ravel()
        std_t = np.asarray(target_std, dtype=np.float64).ravel()
        if not (mean_s.size == std_s.size == mean_t.size == std_t.size):
            raise ValueError(
                "source and target moments must have the same number of bands "
                f"({mean_s.size}, {std_s.size}, {mean_t.size}, {std_t.size})"
            )
        if np.any(std_s <= 1e-12):
            raise ValueError("source std is zero in at least one band")
        scale = std_t / std_s
        offset = mean_t - scale * mean_s
        return cls(
            scale=tuple(float(value) for value in scale),
            offset=tuple(float(value) for value in offset),
            **metadata,
        )

    @classmethod
    def fit_moments(
        cls,
        source: NDArray[np.floating] | NDArray[np.integer],
        target: NDArray[np.floating] | NDArray[np.integer],
        valid_source: NDArray[np.bool_] | None = None,
        valid_target: NDArray[np.bool_] | None = None,
        **metadata: Any,
    ) -> AffineRadiometry:
        """Fit the affine map from two reflectance cubes.

        Args:
            source: Emulated cube ``(C, H, W)`` or ``(N, C, H, W)``.
            target: Real cube, same band count, any spatial shape.
            valid_source: Optional mask for ``source``.
            valid_target: Optional mask for ``target``.
            **metadata: Passed to :meth:`from_moments`.

        Returns:
            The fitted transform.

        Raises:
            ValueError: If the band counts differ.
        """
        mean_s, std_s, n_s = _band_moments(source, valid_source, "source")
        mean_t, std_t, n_t = _band_moments(target, valid_target, "target")
        if mean_s.size != mean_t.size:
            raise ValueError(
                f"source has {mean_s.size} bands but target has {mean_t.size}"
            )
        metadata.setdefault("n_pixels", int(n_s + n_t))
        return cls.from_moments(mean_s, std_s, mean_t, std_t, **metadata)

    def apply(
        self,
        reflectance: NDArray[np.floating] | NDArray[np.integer],
        clip: bool = True,
    ) -> NDArray[np.float32]:
        """Apply ``scale * x + offset`` band-wise.

        Args:
            reflectance: Cube ``(C, H, W)`` or ``(N, C, H, W)`` in reflectance
                units (0-1), **not** uint16 DN.
            clip: Clip the result to ``[0, 1]``.  On by default: the affine
                pushes ~1 % of the pixels past the emulated saturation and a
                few dark NIR pixels below zero.

        Returns:
            float32 cube with the same shape.

        Raises:
            ValueError: If the band count does not match.
        """
        array = np.asarray(reflectance)
        squeeze = array.ndim == 3
        cube = _as_cube(array, "reflectance")
        if cube.shape[1] != self.n_bands:
            raise ValueError(
                f"transform has {self.n_bands} bands, input has {cube.shape[1]}"
            )
        scale, offset = self.as_arrays()
        out = cube * scale.reshape(1, -1, 1, 1) + offset.reshape(1, -1, 1, 1)
        if clip:
            out = np.clip(out, 0.0, TOA_MAX)
        result = out.astype(np.float32)
        return result[0] if squeeze else result

    def apply_uint16(
        self,
        images: NDArray[np.integer],
        clip: bool = True,
    ) -> NDArray[np.uint16]:
        """Apply the transform to a uint16 reflectance-x-10000 array.

        Args:
            images: ``(N, C, H, W)`` or ``(C, H, W)`` uint16, reflectance
                times :data:`SCALE_FACTOR`.
            clip: Clip to ``[0, 1]`` before quantising (default on).

        Returns:
            uint16 array with the same shape.

        Raises:
            ValueError: If the input is not integer or the band count differs.
        """
        array = np.asarray(images)
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"expected a uint16 array, got {array.dtype}")
        reflectance = array.astype(np.float32) / SCALE_FACTOR
        calibrated = self.apply(reflectance, clip=clip)
        quantised = np.rint(calibrated * SCALE_FACTOR)
        return np.clip(quantised, 0, SCALE_FACTOR).astype(np.uint16)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation, with the metadata that explains the fit."""
        return {
            "format": "satharmony.affine-radiometry/1",
            "bands": list(self.bands),
            "scale": [float(value) for value in self.scale],
            "offset": [float(value) for value in self.offset],
            "method": self.method,
            "source": self.source,
            "target": self.target,
            "split": self.split,
            "n_pixels": int(self.n_pixels),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AffineRadiometry:
        """Rebuild a transform from :meth:`to_dict` output.

        Args:
            payload: Mapping with ``scale``, ``offset`` and optional metadata.

        Returns:
            The transform.

        Raises:
            ValueError: If a required key is missing.
        """
        for key in ("scale", "offset"):
            if key not in payload:
                raise ValueError(f"radiometry JSON is missing {key!r}")
        return cls(
            scale=tuple(float(value) for value in payload["scale"]),
            offset=tuple(float(value) for value in payload["offset"]),
            bands=tuple(payload.get("bands", DEFAULT_BANDS)),
            method=str(payload.get("method", "moment matching")),
            source=str(payload.get("source", "emulated pseudo-MSS")),
            target=str(payload.get("target", "real v7 Landsat MSS")),
            split=str(payload.get("split", "")),
            n_pixels=int(payload.get("n_pixels", 0)),
        )

    def save(self, path: Path) -> Path:
        """Write the transform as JSON and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> AffineRadiometry:
        """Read a transform written by :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text()))
