"""Landsat MTL metadata parser."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class MSSCalibration:
    """MSS radiometric calibration parameters.

    A coefficient is ``None`` when the MTL carries ``"NULL"`` for that band.
    That is not a corner case: the real ``LM01_L1GS_043044_19780107`` product
    has no radiance or reflectance calibration for band 4 (its green band),
    so the scene can be parsed and described but not converted to radiance.
    """

    spacecraft_id: str  # "LANDSAT_5"
    sensor_id: str  # "MSS"
    date_acquired: datetime
    sun_elevation: float

    # Radiance rescaling (per band)
    radiance_mult: list[float | None]  # [B1, B2, B3, B4] (bands 4-7 for L1-L3)
    radiance_add: list[float | None]

    # Dynamic range (per band)
    qcal_min: list[int | None]
    qcal_max: list[int | None]

    @property
    def doy(self) -> int:
        """Day of year."""
        return self.date_acquired.timetuple().tm_yday

    @property
    def spacecraft_short(self) -> str:
        """L1_MSS, L5_MSS, etc."""
        num = self.spacecraft_id.split("_")[1]  # "LANDSAT_5" → "5"
        return f"L{num}_MSS"

    @property
    def null_bands(self) -> tuple[int, ...]:
        """1-based position of the bands without radiance calibration.

        Position 1 is the first MSS band of the product: band 1 on Landsat 4/5,
        band 4 on Landsat 1-3.
        """
        return tuple(i + 1 for i, value in enumerate(self.radiance_mult) if value is None)

    @property
    def is_fully_calibrated(self) -> bool:
        """True when all four bands have radiance coefficients."""
        return not self.null_bands

    def require_full_calibration(self) -> None:
        """Raise if any band is ``"NULL"`` in the MTL.

        Raises:
            ValueError: naming the missing bands and the product.
        """
        missing = self.null_bands
        if missing:
            raise ValueError(
                f"{self.spacecraft_short} product of {self.date_acquired.date()} has no "
                f"radiance calibration for band(s) {missing} (MTL value 'NULL'); "
                "cannot convert DN to radiance"
            )


def mss_band_indices(data: dict[str, str]) -> tuple[int, int, int, int]:
    """MSS band numbers as stored in the MTL keys.

    Landsat 4/5 MSS uses bands 1-4; Landsat 1-3 MSS uses bands 4-7
    (USGS LSDS-1416, "Landsat 1 through Landsat 3 captured images using
    bands 4 through 7").
    """
    for order in ((1, 2, 3, 4), (4, 5, 6, 7)):
        if f"RADIANCE_MULT_BAND_{order[0]}" in data:
            return order
    raise ValueError(
        "MTL has no MSS radiance keys: expected BAND_1..4 (Landsat 4/5) "
        "or BAND_4..7 (Landsat 1-3)"
    )


def _as_float(value: str) -> float | None:
    """MTL number, or ``None`` for the literal ``"NULL"``."""
    text = value.strip().strip('"')
    return None if text.upper() == "NULL" else float(text)


def _as_int(value: str) -> int | None:
    number = _as_float(value)
    return None if number is None else int(number)


def parse_mtl(mtl_path: str | Path) -> MSSCalibration:
    """
    Parse Landsat MTL file.

    Reads the radiometric rescaling of MSS from Landsat 4/5 (bands 1-4) or
    Landsat 1-3 (bands 4-7). A ``"NULL"`` coefficient is kept as ``None`` so a
    partially calibrated scene can still be read; converting it to radiance
    raises through :meth:`MSSCalibration.require_full_calibration`.

    Args:
        mtl_path: Path to *_MTL.txt file

    Returns:
        MSSCalibration object
    """
    mtl_path = Path(mtl_path)

    with Path.open(mtl_path) as f:
        lines = f.readlines()

    # Simple key-value extraction
    data = {}
    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip().strip('"')

    # Landsat 1-3 MSS numbers its bands 4-7; Landsat 4/5 uses 1-4.
    bands = mss_band_indices(data)

    # Extract radiance rescaling
    rad_mult = [_as_float(data[f"RADIANCE_MULT_BAND_{b}"]) for b in bands]

    rad_add = [_as_float(data[f"RADIANCE_ADD_BAND_{b}"]) for b in bands]

    # Extract quantization range
    qcal_min = [_as_int(data[f"QUANTIZE_CAL_MIN_BAND_{b}"]) for b in bands]

    qcal_max = [_as_int(data[f"QUANTIZE_CAL_MAX_BAND_{b}"]) for b in bands]

    return MSSCalibration(
        spacecraft_id=data["SPACECRAFT_ID"],
        sensor_id=data["SENSOR_ID"],
        date_acquired=datetime.fromisoformat(data["DATE_ACQUIRED"]),
        sun_elevation=float(data["SUN_ELEVATION"]),
        radiance_mult=rad_mult,
        radiance_add=rad_add,
        qcal_min=qcal_min,
        qcal_max=qcal_max,
    )
