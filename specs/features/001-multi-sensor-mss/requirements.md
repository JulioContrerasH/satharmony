# Requirements - multi-sensor MSS

## Functional

1. The package exposes the five MSS sensors: `L1_MSS`, `L2_MSS`, `L3_MSS`,
   `L4_MSS`, `L5_MSS`.
2. `sensor=` accepts one of those keys or `random`.
3. `random` samples uniformly over the calibrated sensors, using the caller's
   seed. The same seed gives the same sensor and the same bytes.
4. The call reports which sensor was used.
5. An uncalibrated sensor raises a typed error that lists the calibrated ones.
   It never falls back silently.
6. The augmentation entry point defaults to `sensor="random"`.
7. The CLI accepts `--sensor` and prints the chosen sensor.
8. The existing L5 path is bit-identical to the current release for the same
   seed and profile.

## Non-functional

1. No new runtime dependency.
2. Selection is O(1) and happens before any array work.
3. The registry is data, not code branches: adding a sensor is adding an entry.
4. Import time does not grow noticeably (no new heavy imports).
5. Python 3.10-3.13 remain supported.

## Data and calibration

1. A sensor is `calibrated` only when its profile exists, has a `validity`
   block, and its numbers were measured.
2. A sensor without a profile is `pending`, and the API refuses to emulate it.
3. The SBAF is keyed by (S2 sensor, MSS sensor). Until an MSS sensor has its
   own SBAF, it stays pending.
4. No profile ships with numbers copied from another spacecraft.

## Documentation

1. README: how to choose a sensor and what `random` means today.
2. `validity`: which sensor the profile targets.
3. A spec file per feature, committed before the code.

## Out of scope

- Interpolating a sensor from another one.
- Emulating TM, ETM+ or OLI.
- Cloud labels.
