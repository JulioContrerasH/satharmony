# Changelog

## 0.2.0 (2026-11-03)

- Versioned JSON profiles: `mss5_esa_gtc_v1`, `mss5_esa_gtc_v2` and
  `legacy_0_1_13`, loaded with `PipelineConfig.from_profile()`. The v2 profile
  is fitted from the 49-pixel S2/MSS calibration set (34 calibration sites,
  14 validation sites).
- The emulator uses a local `numpy.random.Generator` per call. Only the legacy
  profile keeps the global RandomState, because it promises a byte-for-byte
  reproduction of 0.1.13.
- `satharmony.calibration` adds subpixel alignment (phase correlation + ECC),
  `fit_profile()` with bootstrap intervals, and SAM/RMSE/bias/SNR/FWHM/C2ST
  metrics.
- `satharmony.manifests` reads and verifies CSV pair manifests with SHA-256,
  cloud fraction and reason. `manifests/manifest_178.csv` pins the 178 MSS/TM
  calibration pairs.
- `satharmony` CLI with `emulate`, `align`, `metrics`, `fit` and `verify`.
- Optional `gas_mss3_scale` for the 6SV O2-A correction; off by default.

- `parse_mtl` survives a real MTL with `"NULL"` coefficients: the Landsat 1
  product `LM01_L1GS_043044_19780107` has no radiance or reflectance
  calibration for band 4, and the parser used to die on
  `float("NULL")`. `"NULL"` is now `None`, `MSSCalibration.null_bands` and
  `is_fully_calibrated` describe the gap, `require_full_calibration()` raises
  with the band list, and `dn_to_radiance` calls it before doing arithmetic.
- Three real Landsat 1-3 `*_MTL.txt` (1978, 1982, 1983) pinned in
  `tests/test_satharmony.py::TestParseRealMTL`; the files live in the vault
  (`assets/docs/mtl/`) and the tests skip with a message when they are absent.
  `scripts/fetch_mtl_landsat123.py` downloads them from the USGS Collection 2
  L1 mirror and reports why the USGS M2M and ESA EOWEB routes are closed
  (`/login-token` needs a USGS ERS account; EOWEB answers 403 `not authorized`
  after a successful CAS login).
- `scripts/fit_gas_bands.py` + `scripts/fit_gas_correction.py`: 6SV
  band-integrated gaseous transmittance for the MSS and S2 SRFs, and the
  per-band bias it imposes on the SBAF prediction. At 1 cm of water vapour the
  MSS4 transmittance is 0.881 while the predictor-weighted S2 transmittance is
  0.8815 for B4 and 0.9857 versus 0.9395 for B3: the linear SBAF cannot see the
  O2-A core at 0.76 um. With the gas term, B3 moves from 0.867 to 0.914 and its
  95 % CI from [0.8385, 0.9004] to [0.884, 0.950], which contains the 0.93
  gate; without it, it does not.
- `scripts/pics_water_vapour.py`: ERA5-Land surface dewpoint and pressure per
  PICS and acquisition date, integrated to a water-vapour column with a 2 km
  humidity scale height. The five PICS span 0.46-1.83 cm, so the SBAF gamma
  cannot be separated from the sensor calibration with them alone. Egypt-1 is
  the minimum of the five in all four bands (p = 1/625): it is not a usable
  reference, but removing it only moves B3 from 0.867 to 0.879, so the profile
  is left untouched until v2.

- `satharmony.corrections.landsat.parse_mtl` reads the MSS bands of Landsat 1-3
  MTLs, where the sensor bands are numbered 4-7, as well as 1-4 on Landsat 4/5.
  Before it raised `KeyError` on every L1-L3 MTL (audit of 0.1.13).
- `satharmony.datasets` (optional, needs rasterio through the `io` extra) reads
  CloudSEN12 S2/label pairs with the band order corrected (`B8A` sits at index
  8, not 12) and `scripts/build_pseudo_mss_fixture.py` turns them into a
  2 000-chip pseudo-MSS fixture at 60 m (64x64, 526 chips with shadow >= 10 %)
  with per-chip hashes in `manifest.csv`.
- `satharmony.validation` adds Wasserstein per band, radial PSD and striping
  metrics, and `scripts/compare_pseudo_domain.py` measures pseudo against real
  MSS on the five PICS: W1 = 0.0064 / 0.0099 / 0.0177 / 0.0175 for SBAF versus
  0.117 / 0.098 / 0.080 / 0.049 for legacy, SBAF winning on 20/20 site x band
  and reproducing the 474 m striping that legacy does not.
- `scripts/prepare_human_labels.py` and `scripts/label_agreement.py` ship the
  40-chip human-label package (60 m chips outside the 178 MSS/TM pairs), the
  empty annotation CSVs at 10 m and the kappa/agreement tooling. No chip is
  annotated yet.
- Packaging: `Requires-Python` is `>=3.10,<3.14`, the same range the CI matrix
  tests; `rasterio` is declared as an optional dependency behind the `io` extra
  (it was only in a Poetry group, so the extra was empty in the wheel); the
  SBAF coefficients JSON is shipped inside the wheel.
- CI: 3.10-3.13 matrix, a separate job that runs `tests/test_g4.py` with CPU
  torch, a ruff job and a wheel build checked with `twine check`.
- 184 tests, ruff clean, 92 % coverage. The legacy profile is now pinned
  against the published 0.1.13 wheel with SHA-256 hashes of the MSS array and
  the labels (it reproduces both byte for byte), not only by config fields.
- G4 baseline: `satharmony.models` (optional, needs torch) adds a 4-band U-Net
  (`CloudUNet`, 7.24 M parameters) with GroupNorm, weights-only
  checkpoints plus a SHA-256 sidecar, dihedral TTA inference and NumPy
  segmentation metrics (confusion matrix, per-class IoU, mIoU, kappa).
  `scripts/train_cloud_baseline.py` and `scripts/evaluate_cloud_baseline.py`
  reproduce the run: 1693/113/194 scene split, val mIoU 0.563, pseudo-test mIoU
  0.518 with TTA (OmniCloudMask 0.722 on the same split). `import satharmony`
  stays NumPy-only; the package gains no hard dependency.
- `mss5_esa_gtc_v1().final_scaling.band_scaling` recalibrated on five CEOS
  PICS (Libya-4, Algeria-3, Niger-1, Egypt-1, Arabia-2): 0.956 / 0.933 / 0.867
  / 0.933, with the variation from the site-to-site spread (0.044 / 0.045 /
  0.050 / 0.072). The previous 0.72 / 0.70 / 0.67 / 0.68 were a guess and were
  compensating a wrong MSS reflectance conversion. `legacy_0_1_13()` untouched.
- B4 water-vapour correction: ``rho_MSS4 -= gamma * rho_B8A *
  log(rho_B8A / rho_B9)``, with ``gamma=0.112`` calibrated on the five PICS.
  B4 TOA RMSE drops from 22.1 % to 7.4 % (legacy 14.1 %), and the B4 scale
  rises from 0.820 to 0.933, in line with B2. A 10-predictor NNLS using B9 as
  a surface band was rejected: it under-predicts MSS4 by 30-48 % on TOA PICS.
- Sun elevation: the 25-70 placeholder is replaced by the distribution measured
  on the 178 MSS/TM pairs (7.64-62.33 deg, mean 42.80, std 12.25, truncated
  normal) in both the default config and `mss5_esa_gtc_v1()`. The legacy
  profile keeps the old field and still uses `legacy_random`.
- `scripts/fetch_pics.py`: downloads the five PICS from Earth Engine
  (`ee-contrerasnetk`) on a shared 15 km UTM grid, cloud-free and unsaturated.
  MSS is converted with `dn_to_toa`, not with the MTL `REFLECTANCE_MULT`, which
  carries no sun-elevation correction (it returns rho*sin(SZA)).
- `scripts/validate_sbaf_pics.py`: subpixel registration (phase correlation +
  ECC), per-band metrics, SBAF vs legacy, and the band_scaling fit. Result on
  the five PICS: r 0.81-0.86 (NIR 0.90 on four of five), RMSE without scale
  5.3 / 7.7 / 15.6 / 7.4 % for SBAF+WV vs 6.3 / 11.2 / 21.6 / 14.1 % for
  legacy; after one scale per band both drop to 2.3-2.7 %.
- SBAF spectral mode (`SpectralConfig.mode="sbaf"`): each MSS band is a
  non-negative linear combination of nine S2 VNIR bands plus an offset,
  fitted from the official SRFs and a 535-spectrum library. Coefficients are
  embedded in `sbaf_coefficients.json`; `mode="legacy"` keeps B3/B4/B7/B8
  bit for bit. `mss5_esa_gtc_v1()` now uses SBAF; `legacy_0_1_13()` does not.
- Physical saturation: per-band threshold from LMAX, ESUN and the scene sun
  elevation (`pi * LMAX * d^2 / (ESUN * sin(theta))`), replacing the single
  random 0.8-0.95 threshold that meant different reflectances in the sqrt and
  linear bands. LMAX (Chander Table 2) and `saturation_reflectance()` added to
  `corrections.landsat`; `legacy_random` preserved in `legacy_0_1_13()`.
- Striping: 474 m period along the ground trace (7.9 px at 60 m) and ~9 deg
  tilt from the image rows, instead of six row-aligned bands. The applied grid
  follows `spatial.sampling_gsd`; row geometry preserved in `legacy_0_1_13()`.
### Known limitations
- The cloud baseline is trained and evaluated on pseudo-MSS built from
  CloudSEN12. The test split is independent of training, but its labels are
  CloudSEN12, not human labels on real MSS: the 0.518 mIoU is a pseudo-test
  number, not an accuracy on real MSS imagery. The 40 human-labelled chips are
  prepared, not annotated, so no kappa is reported.
- No real Landsat 1-3 MTL was available in the working data. The L1-L3 band
  selection (bands 4-7) is covered by a synthetic MTL test; three real Landsat
  5 MTLs parse correctly but exercise bands 1-4, so they do not replace it.
- OmniCloudMask (0.722 mIoU on the same pseudo-test) is fed B04/B03/B8A at 10 m,
  not the 4-band 60 m MSS grid.
- `mss5_esa_gtc_v1()` is calibrated on five desert PICS, all winter/autumn with
  SZA 50-59 deg. A humid or summer PICS and an independent water-vapour source
  (ERA5/MODIS) are missing, so `gamma=0.112` and the B4 scale are degenerate
  with the site-to-site spread.

- New calibrated profile `PipelineConfig.mss5_esa_gtc_v1()`.
- Physical order: striping, memory, coherent/random noise and scan artifacts
  before 6-bit quantization in the new profile; legacy order preserved.
- 60 m sampling with PSF 79-82 m; centered decimation; linear radiometry.
- CloudSEN12 Table 3 label encoding; odd-block aggregation fixed.
- 35 tests; wheel builds.

- P0: linear radiometry (sqrt companded then decompressed), centered
  decimation, independent 60 m sampling and PSF, CloudSEN12 labels.
- `PipelineConfig.legacy_0_1_13()` reproduces 0.1.13.

## 0.1.13
- Initial published package (QA4EO-2 first version).
