"""SatHarmony: Sensor harmonization and emulation for Earth Observation."""

from .emulators.s2_to_mss import (
    MSSEmulator,
    PipelineConfig,
    aggregate_cloudsen12,
    emulate_labels,
    emulate_s2,
    emulate_s2_with_labels,
)

__all__ = [
    # Classes
    "MSSEmulator",
    "PipelineConfig",
    # Functions
    "emulate_s2",
    "emulate_labels",
    "emulate_s2_with_labels",
    "aggregate_cloudsen12",
]

__version__ = "0.1.0"
