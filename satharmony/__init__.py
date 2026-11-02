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
from .units import assert_mss_toa_units, mss_to_toa
from .validation import (
    ks_per_band,
    radial_psd,
    striping_metrics,
    wasserstein_per_band,
)

__all__ = [
    "ESUN_MSS",
    "MSSCalibration",
    # Emulation
    "MSSEmulator",
    "PipelineConfig",
    "aggregate_cloudsen12",
    # Units
    "assert_mss_toa_units",
    "dn_to_radiance",
    "dn_to_toa",
    "emulate_labels",
    "emulate_s2",
    "emulate_s2_with_labels",
    "ks_per_band",
    "mss_to_toa",
    # Corrections
    "parse_mtl",
    "radial_psd",
    "radiance_to_toa",
    "striping_metrics",
    "wasserstein_per_band",
]


try:
    from importlib.metadata import version

    __version__ = version("satharmony")
except Exception:
    __version__ = "0.0.0-dev"
