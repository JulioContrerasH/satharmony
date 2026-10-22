# Changelog

## 0.3.0 (2026-10-21)
- New calibrated profile `PipelineConfig.mss5_esa_gtc_v1()`.
- Physical order: striping, memory, coherent/random noise and scan artifacts
  before 6-bit quantization in the new profile; legacy order preserved.
- 60 m sampling with PSF 79-82 m; centered decimation; linear radiometry.
- CloudSEN12 Table 3 label encoding; odd-block aggregation fixed.
- 35 tests; wheel builds.

## 0.2.0 (2026-10-21)
- P0: linear radiometry (sqrt companded then decompressed), centered
  decimation, independent 60 m sampling and PSF, CloudSEN12 labels.
- `PipelineConfig.legacy_0_1_13()` reproduces 0.1.13.

## 0.1.13
- Initial published package (QA4EO-2 first version).
