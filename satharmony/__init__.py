"""SatHarmony: Sensor harmonization and emulation for Earth Observation."""

from .corrections import (
    ESUN_MSS,
    MSSCalibration,
    dn_to_radiance,
    dn_to_toa,
    parse_mtl,
    radiance_to_toa,
)
from .emulators.s2_to_mss import (
    MSSEmulator,
    PipelineConfig,
    aggregate_cloudsen12,
    emulate_labels,
    emulate_s2,
    emulate_s2_with_labels,
)

__all__ = [
    # Emulation
    "MSSEmulator",
    "PipelineConfig",
    "emulate_s2",
    "emulate_labels",
    "emulate_s2_with_labels",
    "aggregate_cloudsen12",
    # Corrections
    "parse_mtl",
    "MSSCalibration",
    "dn_to_toa",
    "dn_to_radiance",
    "radiance_to_toa",
    "ESUN_MSS",
]


try:
    from importlib.metadata import version

    __version__ = version("satharmony")
except Exception:
    __version__ = "0.0.0-dev"
