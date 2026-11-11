# Plan - multi-sensor MSS

## Approach

The degradation chain does not change. What changes per spacecraft is a small
set of parameters. The emulator gets a sensor registry that says, for each
MSS sensor, which parameters are calibrated and which profile to load.

Selection happens once, at the start of a call, and is part of the seed. A
random sensor therefore repeats exactly with the same seed.

## Design decisions

- **Sensor keys**: `L1_MSS`, `L2_MSS`, `L3_MSS`, `L4_MSS`, `L5_MSS`.
- **One profile per sensor**, same schema. The existing L5 profile keeps its
  name; new ones follow `mss<N>_esa_gtc_v1.json`.
- **Random is over calibrated sensors only.** Sampling an uncalibrated sensor
  would fabricate numbers. With only L5 calibrated, `random` returns L5; the
  behaviour is correct and becomes useful as P2 and P3 land.
- **Backward compatibility**: the library default stays the profile's sensor.
  The augmentation helper defaults to `sensor="random"`.
- **The SBAF is per (S2 sensor, MSS sensor) pair**, because the target SRF
  changes the weights. The registry stores which pairs exist.
- **No silent fallback.** An uncalibrated sensor raises `SensorNotCalibrated`
  with the list of available ones.

## Phases

### A. Registry and selection (no new data)

1. `sensors.py`: `SensorSpec` and `MSS_SENSORS` with five entries, each with
   spacecraft, IFOV, quantization bits, radiometric constants source, profile
   name or `None`, and status (`calibrated` / `pending`).
2. `choose_sensor(rng, candidates)` and `resolve_sensor(name, rng)`.
3. `emulate_any(image, sensor="random", seed=None, sensors=None)` returns
   `(mss, factor, sensor_key)` and loads the sensor profile.
4. CLI: `emulate --sensor <key|random>`; prints the chosen sensor.
5. Tests: registry, determinism, L5 bit-identity, typed error.
6. Docs: README section and `validity` note.

### B. Landsat 4 calibration

1. For each of the 49 PICS, find an L4 MSS scene over the same site
   (1982-1993) and download it. S2 is reused.
2. Fit SBAF and band scales; write `mss4_esa_gtc_v1.json` with `validity`.
3. Validate on the 5 held-out deserts: scaled RMSE, raw RMSE, C2ST, PSD.
4. Compare L4 against L5 on the same sites.

### C. Landsat 1-3 calibration

1. Obtain the L1-L3 MSS spectral response curves.
2. Build the SBAF shape from the SRF and the spectral library.
3. Calibrate the scale on the 49 PICS; declare the missing TM reference.
4. Validate as in B.

## Order and dependencies

A has no data dependency and ships first. B depends on A and on 49 downloads.
C depends on the SRFs, which are an external download.

## Risks

- L4 has fewer cloud-free scenes at some PICS than L5.
- L1-L3 SRFs may need manual transcription from USGS documents.
- A generic profile may be good enough; the L4/L5 comparison decides, not
  intuition.
