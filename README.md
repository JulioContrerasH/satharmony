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
# mss.shape:        (4, 73, 73)
# mss_labels.shape: (73, 73)

# Or emulate separately (must pass scale_factor to labels)
mss, sf = satharmony.emulate_s2(s2_image, seed=42)
mss_labels = satharmony.emulate_labels(s2_labels, scale_factor=sf)
```

## Features

The MSS emulator applies physically-motivated degradations:

| Degradation | Description |
|-------------|-------------|
| **Spectral** | Band selection (B3, B4, B7, B8 → MSS 4 bands) |
| **Spatial** | PSF convolution + downsampling (10m → 60-80m) |
| **Radiometric** | 6-bit quantization, sqrt compression, saturation |
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
config.spatial.target_gsd.min = 60.0
config.spatial.target_gsd.max = 60.0  # fixed 60m

emulator = MSSEmulator(config)
mss = emulator(image)
```

## Label Aggregation

CloudSEN12 labels (10m) are aggregated using conservative criteria from [Roy et al.](https://doi.org/10.1016/j.rse.2024.114439):

- **Precedence**: cloud > shadow > thin_cloud > clear
- **Classes**: 0=clear, 1=thin_cloud, 2=cloud, 3=shadow

## Citation

```bibtex
@software{satharmony2025,
  author  = {Contreras, Julio},
  title   = {SatHarmony: Sensor Emulation for Earth Observation},
  year    = {2025},
  url     = {https://github.com/IPL-UV/satharmony}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.