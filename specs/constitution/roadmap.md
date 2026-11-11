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

## P2 - Landsat 4 calibration (experimental)

- 32 of the 49 PICS have a Landsat 4 scene in GEE; 20 have one at a comparable
  sun height; 11 survive the L4/L5 surface-stability check (15 %).
- The L4 SBAF was refitted from the L4 SRF and the spectral library.
- In-sample 9.5-13.3 %, held out 12.8-35 %; L5 is 5.7-7.0 %. B4 is the weak band.
- Shipped as `experimental`; `random` does not pick it by default.

Gate: profile with `validity` and the L4/L5 comparison written down. Done.
Next: a dedicated B4 term for L4 before calling it calibrated.

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
