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

## G7 - L4 calibration (P2) - measured, below the L5 bar

Result (2026-11-11):
- `mss4_esa_gtc_v1.json` exists with `validity`, marked `experimental`.
- 11 stable fit sites (of 29 calibration): 9 have no L4 scene in GEE, 9 changed
  surface or have a hazy window (L4/L5 ratio beyond 15 %).
- In-sample relative RMSE 9.5-13.3 %; held out 12.8-35 % (two sites).
- L5 is 5.7-7.0 %. L4 is roughly twice as bad and B4 is the weak band.
- Decision: **per-sensor profiles**, not a generic one. L4 stays experimental
  until its B4 term is refitted; `random` does not pick it by default.

## G8 - L1-L3 calibration (P3)

Same as G7, plus an explicit note that these sensors had no TM on board, so
the sensor physics has no same-second reference.

## Evidence to keep

- The command and its output for each gate.
- The profile `validity` block.
- The L4/L5 comparison table.
