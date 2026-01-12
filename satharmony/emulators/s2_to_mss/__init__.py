"""MSS Emulator: Sentinel-2 to Landsat MSS degradation."""

from .config import PipelineConfig
from .functional import emulate_labels, emulate_s2, emulate_s2_with_labels
from .labels import aggregate_cloudsen12
from .pipeline import MSSEmulator

__all__ = [
    "MSSEmulator",
    "PipelineConfig",
    "aggregate_cloudsen12",
    "emulate_s2",
    "emulate_labels",
    "emulate_s2_with_labels",
]
