# Validation - multi-sensor MSS

Every gate is a command or a measured number. A gate that cannot run is not
a gate.

## G1 - Registry complete

```bash
python -c "from satharmony.emulators.s2_to_mss.sensors import MSS_SENSORS; assert len(MSS_SENSORS) == 5"
```

Pass: five keys, each with spacecraft, IFOV, quantization bits and status.

## G2 - L5 bit-identical

```bash
pytest tests/test_multisensor.py -q
```

Pass: `emulate_any(image, sensor="L5_MSS", seed=7)` equals
`emulate_s2(image, config=profile_v2, seed=7)` byte for byte.

## G3 - Random is deterministic

Pass: two calls with `seed=11` give the same sensor key and the same bytes;
two calls with different seeds may differ (with one calibrated sensor they
will not, which is also correct).

## G4 - No silent fallback

Pass: `sensor="L4_MSS"` raises `SensorNotCalibrated`, and the message lists the
calibrated sensors.

## G5 - CLI

```bash
satharmony emulate scene.npy -o out.npy --sensor random --seed 0
```

Pass: writes the output and prints the chosen sensor; exit code 0.

## G6 - Suite and build

```bash
pytest -q
ruff check satharmony scripts tests
python -m build
python -m twine check dist/*
```

Pass: all tests green, ruff clean, wheel and sdist PASSED.

## G7 - L4 calibration (P2)

Result (2026-11-11):
- `mss4_esa_gtc_v1.json` exists with `validity`, marked `experimental`.
- 11 stable fit sites of 29 PICS candidates: 9 have no L4 scene in GEE and 9
  changed surface or have a hazy window (L4/L5 ratio beyond 15 %).
- In-sample 6.3-11.7 %, held out 9.4-12.5 % (two sites).
- Early runs forced the B4 gain to equal B1-B3 and the B4 error doubled
  (13.8 % in sample, 34.7 % held out). With the gain free per band it is 6.3 %
  and 9.4 %. The weak band was the constraint, not the sensor.
- Outside the domain (8 PICS with an L4 scene): 13.9-22.3 % rescaled, 25.6-35.2 %
  raw. `pantanal` and `great_salt_lake` carry the tail because the water extent
  changed; the profile does not target them.
- C2ST on its two held-out deserts: 0.929 deterministic, 0.921 stochastic. Two
  sites is weak evidence, so the number is stored with that caveat.
- Status: **calibrated** (2026-11-12). It carries the four measurements L5
  carries. The held-out split is two deserts and stays written in `validity`
  as a caveat. Consequence: `random` now samples L4 and L5.

## G8 - L1-L3 calibration (P3) - solved by changing the bank

Result (2026-11-11):
- SRFs for L1-L3 converted from the existing Excel files; SBAF fitted from
  each SRF and the spectral library (`sbaf_coefficients_l1..l3.json`).
- The CEOS PICS cannot carry L1-L3: with a clear window and a sun within 15 deg
  of the L5 reference, 6 PICS have an L1 scene, 3 an L2 and 1 an L3.
- Phase D (below) fixes it. Final profiles, all `experimental`:
  L1 15 fit + 4 held out, 8.5-10.8 %; L2 18 + 6, 9.3-10.3 %; L3 6 + 0,
  14.8-16.1 %.
- L1-L3 carried no TM, so the sensor physics has no same-second reference.

## G9 - Archive-inversion bank (P4)

```bash
python scripts/enumerate_mss_archive.py --platform L2 --out /tmp/l2_scenes.json
python scripts/score_archive_sites.py --scenes /tmp/l2_scenes.json --out /tmp/l2_sites.json
python scripts/fetch_aux_sites.py --sensor L2 --sites /tmp/l2_sites.json
python scripts/make_profile_aux.py --sensor L2
```

Pass: a profile with `validity`, `site_source` naming the auxiliary bank, and a
held-out split when the bank has at least eight stable sites.

Measured:
- Archive size: L1 2,800 scenes, L2 3,050, L3 180. The mission is the archive.
- Dry, flat, low-cloud candidates at 1 deg spacing: L1 306, L2 198, L3 20.
- Downloaded 30 / 30 / 14 sites; stable after the self-consistent 15 % scale
  check: 19 / 24 / 6.
- Held out one in four for L1 and L2 (the split needs eight sites), none for L3.

## G10 - Evaluation must use the profile it claims to measure

Pass: `pytest tests/test_fit.py -q` and a desert re-run whose numbers match the
`validity` block.

What was wrong: `_deterministic_config` zeroed `spectral.water_vapor_scale`, so
every "deterministic" metric measured a model without the B4 water-vapour term
that the profile applies. The fix keeps the term and moves the zeroing into
`fit_profile`, which adds the term itself. Consequences measured on L5:
B4 in-sample 14.7 % -> 9.7 %, C2ST deterministic 0.999 -> 0.8985.

## Evidence to keep

- The command and its output for each gate.
- The profile `validity` block.
- The L4/L5 comparison table.
