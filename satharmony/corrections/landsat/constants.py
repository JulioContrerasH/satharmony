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

# LMAX rescaling factors [W/(m² sr μm)] for the four MSS bands, NLAPS
# processing (Chander et al. 2009, Table 2, p. 16). LMAX is the radiance
# mapped to QCALMAX = 127, i.e. the top of the product dynamic range, not
# the detector saturation point, which is the raw 6-bit count 63
# (USGS DFCB, LSDS-1416 v3.0, p. 18). The values are internally consistent
# with the published Grescale = (LMAX - LMIN) / 127.
LMAX_MSS = {
    "L1_MSS": [248.0, 200.0, 176.0, 153.0],
    "L2_MSS": [263.0, 176.0, 152.0, 130.333],
    "L3_MSS": [259.0, 179.0, 149.0, 128.0],
    "L4_MSS": [238.0, 164.0, 142.0, 116.0],
    "L5_MSS": [268.0, 179.0, 148.0, 123.0],
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
