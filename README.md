# 🛰️ SatHarmony

**Sensor emulation and harmonization for Earth Observation.**

[![PyPI](https://img.shields.io/pypi/v/satharmony.svg)](https://pypi.org/project/satharmony/)
[![Python](https://img.shields.io/pypi/pyversions/satharmony.svg)](https://pypi.org/project/satharmony/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

SatHarmony provides physically-based sensor emulation to create synthetic historical satellite imagery from modern sensors. Currently supports **Sentinel-2 → Landsat MSS** emulation with realistic degradation modeling.

## Installation

```bash
pip install satharmony

# With GeoTIFF I/O support
pip install satharmony[io]
```

## Quick Start

```python
import numpy as np
import satharmony

# Load your Sentinel-2 image (C, H, W) and CloudSEN12 labels (H, W)
s2_image = ...      # shape: (13, 512, 512), float32, [0, 1]
s2_labels = ...     # shape: (512, 512), uint8, classes 0-3

# Emulate both together (recommended - guarantees matching dimensions)
mss, mss_labels, scale_factor = satharmony.emulate_s2_with_labels(
    s2_image, s2_labels, seed=42
)
# mss.shape:        (4, 85, 85)   # 512 @ 10 m -> 60 m grid
# mss_labels.shape: (85, 85)

# Or emulate separately (must pass scale_factor to labels)
mss, sf = satharmony.emulate_s2(s2_image, seed=42)
mss_labels = satharmony.emulate_labels(s2_labels, scale_factor=sf)
```

## Features

The MSS emulator applies physically-motivated degradations:

| Degradation | Description |
|-------------|-------------|
| **Spectral** | Band selection (B3, B4, B7, B8 → MSS 4 bands) |
| **Spatial** | PSF convolution (MSS IFOV ~79-82 m) + 60 m sampling |
| **Radiometric** | 6-bit quantization with sqrt companding (decompressed back to linear reflectance) |
| **Striping** | 6-detector gain/offset mismatch artifacts |
| **Memory Effect** | Bright target recovery trailing |
| **Noise** | Coherent (periodic) + random (gaussian/poisson) |
| **Scan Artifacts** | Line dropouts and transmission errors |

## API Reference

```python
# Combined emulation (recommended)
mss, labels, sf = satharmony.emulate_s2_with_labels(image, labels, seed=42)

# Independent emulation
mss, sf = satharmony.emulate_s2(image, seed=42)
mss, sf = satharmony.emulate_s2(image, scale_factor=8)  # fixed scale
labels = satharmony.emulate_labels(labels, scale_factor=sf)

# Low-level access
from satharmony import MSSEmulator, PipelineConfig

config = PipelineConfig()
config.spatial.sampling_gsd = 60.0          # output grid (ESA GTC)
config.spatial.psf_fwhm_m = Range(79, 82)   # sensor IFOV, independent of grid

emulator = MSSEmulator(config)
mss = emulator(image)

# Reproduce the 0.1.13 behaviour used by the published model
config_legacy = PipelineConfig.legacy_0_1_13()
```

## Behavior changes in 0.2.0

- Radiometry is linear again: the sqrt companding is inverted after 6-bit quantization.
- Decimation is centred (average pooling after the PSF), matching the label grid.
- `sampling_gsd` (60 m) and `psf_fwhm_m` (79-82 m) are independent.
- Labels use the CloudSEN12 encoding (0 clear, 1 thick, 2 thin, 3 shadow) and a
  conservative contamination rule.
- `PipelineConfig.legacy_0_1_13()` reproduces the 0.1.13 output bit for bit
  (same seed and input).

## Label Aggregation

CloudSEN12 labels (10m) are aggregated to the MSS grid. The default rule uses
the CloudSEN12 Table 3 encoding and priority (thick > shadow > thin > clear):
a class with >= 50 % of the block wins; if no class reaches 50 %, a block with
>= 50 % contaminated pixels is assigned to the most frequent contaminated
class; otherwise it is clear.

- **Classes**: 0=clear, 1=thick_cloud, 2=thin_cloud, 3=shadow
- **Legacy**: `aggregate_cloudsen12(..., legacy=True)` reproduces the 0.1.13 rule.

## Citation

```bibtex
@software{satharmony2025,
  author  = {Contreras, Julio},
  title   = {SatHarmony: Sensor Emulation for Earth Observation},
  year    = {2025},
  url     = {https://github.com/JulioContrerasH/satharmony}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.