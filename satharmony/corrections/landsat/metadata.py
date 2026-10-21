"""Landsat MTL metadata parser."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class MSSCalibration:
    """MSS radiometric calibration parameters."""

    spacecraft_id: str  # "LANDSAT_5"
    sensor_id: str  # "MSS"
    date_acquired: datetime
    sun_elevation: float

    # Radiance rescaling (per band)
    radiance_mult: list[float]  # [B1, B2, B3, B4]
    radiance_add: list[float]

    # Dynamic range (per band)
    qcal_min: list[int]
    qcal_max: list[int]

    @property
    def doy(self) -> int:
        """Day of year."""
        return self.date_acquired.timetuple().tm_yday

    @property
    def spacecraft_short(self) -> str:
        """L1_MSS, L5_MSS, etc."""
        num = self.spacecraft_id.split("_")[1]  # "LANDSAT_5" → "5"
        return f"L{num}_MSS"


def parse_mtl(mtl_path: str | Path) -> MSSCalibration:
    """
    Parse Landsat MTL file.

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

    # Extract radiance rescaling
    rad_mult = [
        float(data["RADIANCE_MULT_BAND_1"]),
        float(data["RADIANCE_MULT_BAND_2"]),
        float(data["RADIANCE_MULT_BAND_3"]),
        float(data["RADIANCE_MULT_BAND_4"]),
    ]

    rad_add = [
        float(data["RADIANCE_ADD_BAND_1"]),
        float(data["RADIANCE_ADD_BAND_2"]),
        float(data["RADIANCE_ADD_BAND_3"]),
        float(data["RADIANCE_ADD_BAND_4"]),
    ]

    # Extract quantization range
    qcal_min = [
        int(data["QUANTIZE_CAL_MIN_BAND_1"]),
        int(data["QUANTIZE_CAL_MIN_BAND_2"]),
        int(data["QUANTIZE_CAL_MIN_BAND_3"]),
        int(data["QUANTIZE_CAL_MIN_BAND_4"]),
    ]

    qcal_max = [
        int(data["QUANTIZE_CAL_MAX_BAND_1"]),
        int(data["QUANTIZE_CAL_MAX_BAND_2"]),
        int(data["QUANTIZE_CAL_MAX_BAND_3"]),
        int(data["QUANTIZE_CAL_MAX_BAND_4"]),
    ]

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
