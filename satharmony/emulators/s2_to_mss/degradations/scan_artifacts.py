"""Scan artifacts: line dropouts and transmission errors."""

import numpy as np

from ..config import ScanArtifactsConfig


def apply_scan_artifacts(image: np.ndarray, config: ScanArtifactsConfig) -> np.ndarray:
    """Apply scan line dropouts and transmission errors.

    Data loss during downlink caused missing or corrupted lines.

    Args:
        image: Input array (C, H, W) float32
        config: ScanArtifactsConfig

    Returns:
        Array with scan artifacts
    """
    if not config.enabled:
        return image

    C, H, W = image.shape
    dropout_prob = config.dropout_prob.sample()
    max_lines = config.max_consecutive_lines.sample()

    result = image.copy()

    row = 0
    while row < H:
        if np.random.random() < dropout_prob:
            # Number of consecutive lines to drop
            n_lines = np.random.randint(1, max_lines + 1)
            n_lines = min(n_lines, H - row)

            # Random dropout type
            dropout_type = np.random.choice(["zero", "previous", "noise"])

            if dropout_type == "zero":
                result[:, row : row + n_lines, :] = 0
            elif dropout_type == "previous" and row > 0:
                result[:, row : row + n_lines, :] = result[:, row - 1 : row, :]
            else:  # noise
                result[:, row : row + n_lines, :] = np.random.uniform(0, 0.1, (C, n_lines, W))

            row += n_lines
        else:
            row += 1

    return result.astype(np.float32)
