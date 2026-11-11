# Roadmap

One feature per phase. A phase closes when its validation gate passes; it is
not a wish list.

## P0 - Single-sensor release (closed)

- Profile `mss5_esa_gtc_v2`: SBAF, band scales, gamma, PSF, quantization,
  artifacts, canonical sun elevation.
- Validation: desert 5.7-7.0 %, outside 13.4-23.6 %, C2ST 0.999 / 0.884.
- Tag `v0.2.0`, not on PyPI yet.

## P1 - Sensor registry and selection (this feature)

- A registry for Landsat 1-5 MSS with calibrated/pending status.
- `sensor=` accepts a sensor key or `random`; random is seeded.
- Augmentation helper defaults to random.
- Uncalibrated sensors raise a typed error.
- No new calibration data.

Gate: registry complete, L5 output bit-identical, random deterministic,
uncalibrated sensor raises, suite green.

## P2 - Landsat 4 calibration

- Download the 49 PICS MSS scenes from Landsat 4 (S2 already in the vault).
- Fit SBAF and scales for L4; validate on the 5 held-out deserts.
- Measure whether L4 and L5 differ by less than 3 % in scaled RMSE.
- Decision: one generic profile or one per sensor.

Gate: L4 profile with `validity`, C2ST and RMSE measured, and the L4/L5
comparison written down.

## P3 - Landsat 1-3 calibration

- Spectral response curves for L1-L3.
- MSS scenes at the 49 PICS (no TM on board; the sensor physics cannot use the
  same-second reference).
- SBAF shape from the SRF and the spectral library; scale from the PICS.

Gate: same numbers as P2, with the missing TM reference declared.

## P4 - Cloud model (separate product)

- Out of `satharmony`. Uses the emulator and the MSS/TM bank.

## Release

- PyPI upload only after P1 is merged and the user provides a token.
