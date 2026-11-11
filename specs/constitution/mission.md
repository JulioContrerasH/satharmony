# Mission

`satharmony` turns a Sentinel-2 TOA scene into what a Landsat MSS sensor would
have recorded: four broad bands, 60-80 m, 6 bits, and the sensor defects of the
1980s.

## Why it exists

Sentinel-2 (2015+) and Landsat MSS (1972-2013) never flew together. There is no
same-day S2/MSS pair anywhere, so a cloud model trained on Sentinel-2 cannot be
applied to the MSS archive without a bridge. `satharmony` is that bridge: it
manufactures MSS-looking imagery from S2, with the radiometry, the point spread
function, the quantization and the artifacts of the real sensor.

## Who uses it

- The `cloud-mss-esa` project: pretraining a cloud model on emulated MSS.
- Researchers who need MSS-like data where only S2 exists.

## What is out of scope

- Cloud detection itself. That is a separate model with its own labels.
- Any sensor other than Landsat MSS (TM, ETM+, OLI, Sentinel-2 output).
- Claiming same-day S2/MSS truth. The emulator is a model, not a measurement.

## What "valid" means here

A profile is valid only inside its declared domain. Today that domain is TOA,
60 m, four MSS bands, desert and salt flats, Landsat 5. Outside it the profile
is not calibrated and the README says so.

## Multi-sensor extension (this feature)

The four MSS bands and the 6-bit quantization are common to Landsat 1-5. What
changes per spacecraft is the spectral response, the IFOV, the radiometric
constants and the sensor defects. The mission does not change: the same
emulator must be able to target any MSS, chosen explicitly or sampled at
random for augmentation.
