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

- 32 of the 49 PICS have a Landsat 4 scene in GEE; 20 at a comparable sun
  height; 11 survive the L4/L5 surface-stability check (15 %).
- The L4 SBAF was refitted from the L4 SRF and the spectral library.
- In-sample 6.3-11.7 %, held out 9.4-12.5 %.
- **Calibrated** on 2026-11-12: outside the domain 13.9-22.3 % and C2ST 0.929,
  with two held-out deserts recorded as a caveat. `random` samples L4 and L5.

Gate: profile with `validity` and the four measurements. Done.

## P3 - Landsat 1-3 calibration (done through P4)

- SRFs converted; SBAF fitted for L1, L2 and L3.
- The CEOS PICS only carry 6 / 3 / 1 usable scenes for L1 / L2 / L3, so the
  sensor scale is fitted on the archive-inversion bank instead.
- L1: 15 + 4 sites, 8.5-10.8 %. L2: 18 + 6, 9.3-10.3 %. L3: 6 + 0, 14.8-16.1 %.
- All three experimental. Their outside-domain test is **not measurable** with
  the wet bank: the pairs are 40 years apart over surfaces that change, only
  1-2 windows pass the stability filter, and those sensors carried no TM, so
  there is no same-second reference (G11).

Gate: profile with `validity` for every sensor. Done; L4 and L5 carry the four
measurements, L1-L3 carry the in-sample fit and the documented gap.

## P4 - Site bank from the archive, not from a list

The wall was never the code: it was hoping that a list of desert sites happens
to contain early Landsat scenes. Enumerate the archive, keep dry flat low-cloud
scenes, and download only those.

Tier 1 was the wrong level to enumerate: it holds 2.8k / 3.1k / 180 scenes for
L1 / L2 / L3, while Tier 2 holds 173k / 365k / 157k. Both tiers are now part of
the bank.

Gate: the four scripts in G9 run end to end and produce a profile.
Done for L1, L2 and L3; applicable to L4 and TM later.

## P5 - Wet bank and the honest limit

Forest, water and wetland sites from Tier 2, with the scene id and the shift
written down. The measurement failed for a reason worth keeping: the surface
changed between the two dates, not the geometry. See G11.

