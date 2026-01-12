"""Degradation functions for MSS emulation."""

from .memory_effect import apply_memory_effect
from .noise import apply_coherent_noise, apply_random_noise
from .radiometric import apply_radiometric_degradation
from .scan_artifacts import apply_scan_artifacts
from .spatial import apply_spatial_degradation
from .spectral import apply_spectral_degradation
from .striping import apply_striping

__all__ = [
    "apply_spectral_degradation",
    "apply_spatial_degradation",
    "apply_radiometric_degradation",
    "apply_striping",
    "apply_memory_effect",
    "apply_coherent_noise",
    "apply_random_noise",
    "apply_scan_artifacts",
]
