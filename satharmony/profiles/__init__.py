"""Versioned calibration profiles shipped inside the package."""

from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parent

__all__ = ["PROFILE_DIR"]
