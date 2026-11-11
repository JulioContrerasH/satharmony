# Tech stack

## Runtime

- Python 3.10 to 3.13.
- `numpy` for arrays, `pydantic` v2 for the configuration schema.
- No `rasterio` inside the emulator: the input is an array, the I/O belongs to
  the caller. This keeps the wheel small and importable under load.
- `tifffile` is used only by scripts that read TIFFs, never by the package.

## Configuration

- Every calibrated number lives in a versioned JSON profile under
  `satharmony/profiles/`.
- The JSON is the source of truth; the Python defaults mirror it and a test
  fails if they drift.
- Provenance is stored in the profile: source hashes and the SHA-256 of every
  measurement file.

## Selection logic

- `PipelineConfig` describes the degradation chain.
- A sensor registry declares, per MSS sensor, what is calibrated and what is
  not. It never guesses.
- Selection is deterministic given a seed.

## Quality gates

- `pytest` for behaviour; `ruff check` for style.
- `python -m build` and `twine check` before every tag.
- A profile ships only with its `validity` block and a test that reads it.

## What we do not use

- No GPU, no torch in the emulator.
- No cloud service at import time.
- No silent fallback: an uncalibrated sensor raises, it does not approximate.
