"""MSS Emulator: Sentinel-2 to Landsat MSS degradation."""

from .config import PipelineConfig
from .functional import emulate_labels, emulate_s2, emulate_s2_with_labels
from .labels import aggregate_cloudsen12
from .multisensor import emulate_any
from .pipeline import MSSEmulator
from .sbaf import load_coefficients, sbaf_transform
from .sensors import (
    MSS_SENSORS,
    SensorNotCalibrated,
    SensorSpec,
    calibrated_sensors,
)

__all__ = [
    "MSS_SENSORS",
    "MSSEmulator",
    "PipelineConfig",
    "SensorNotCalibrated",
    "SensorSpec",
    "aggregate_cloudsen12",
    "calibrated_sensors",
    "emulate_any",
    "emulate_labels",
    "emulate_s2",
    "emulate_s2_with_labels",
    "load_coefficients",
    "sbaf_transform",
]
