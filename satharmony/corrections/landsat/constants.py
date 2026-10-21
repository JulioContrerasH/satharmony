"""Landsat MSS radiometric constants."""

import numpy as np

# ESUN values from Chander et al. (2009) - W/(m² μm)
ESUN_MSS = {
    "L1_MSS": [1823, 1559, 1276, 880.1],
    "L2_MSS": [1829, 1539, 1268, 886.6],
    "L3_MSS": [1839, 1555, 1291, 887.9],
    "L4_MSS": [1827, 1569, 1260, 866.4],
    "L5_MSS": [1824, 1570, 1249, 853.4],
}


def earth_sun_distance(doy: int) -> float:
    """
    Calculate Earth-Sun distance correction factor.

    Args:
        doy: Day of year (1-366)

    Returns:
        Distance in astronomical units
    """
    return 1 - 0.01672 * np.cos(0.9856 * (doy - 4) * np.pi / 180)
