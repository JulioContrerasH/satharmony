# 🛰️ SatHarmony

**Sensor emulation and harmonization for Earth Observation.**

[![PyPI](https://img.shields.io/pypi/v/satharmony.svg)](https://pypi.org/project/satharmony/)
[![Python](https://img.shields.io/pypi/pyversions/satharmony.svg)](https://pypi.org/project/satharmony/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

SatHarmony provides physically-based sensor emulation to create synthetic historical satellite imagery from modern sensors. Currently supports **Sentinel-2 → Landsat MSS** emulation with realistic degradation modeling.

## Installation

Python 3.10 to 3.13 (`Requires-Python >=3.10,<3.14`, the range the CI matrix
tests).

```bash
pip install satharmony

# With GeoTIFF I/O support (rasterio, needed by satharmony.datasets)
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
| **Spectral** | Legacy band selection (B3/B4/B7/B8) or SBAF: non-negative linear mixing of nine S2 VNIR bands plus a water-vapour term for B4, per sensor |
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

# Spectral mode: "legacy" (B3/B4/B7/B8, default) or "sbaf" (fitted SBAF)
config.spectral.mode = "sbaf"
config.spectral.sensor = "S2A"              # S2A, S2B or S2C

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

## Calibrated profile (0.4.0)

```python
from satharmony import PipelineConfig
config = PipelineConfig.mss5_esa_gtc_v1()  # 60 m, PSF 79-82 m, linear radiometry
```

The profile uses the SBAF with a water-vapour correction in B4 and the sun
elevation distribution measured on the 178 MSS/TM pairs (7.6-62.3 deg,
mean 42.8, std 12.25).

## Cloud / shadow baseline (G4, optional)

`satharmony.models` contains the 4-band U-Net used as the cloud-mask baseline
of the pseudo-MSS fixture (G4). It needs torch, which is **not** a core
dependency:

```bash
pip install torch
```

```python
import numpy as np
from satharmony.models import load_model, predict_masks

model, checkpoint = load_model("cloud_unet_mss_pseudo.pt")
mss = np.load("mss_chip.npy").astype(np.float32) / 10000.0   # (4, 64, 64) TOA
mask = predict_masks(model, mss, checkpoint.band_mean, checkpoint.band_std)
# 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow
```

`import satharmony` never imports torch; only `satharmony.models` does.

## Status and limitations (0.4.0)

- The cloud baseline is trained on pseudo-MSS from CloudSEN12 and evaluated on a
  scene split of that pseudo-test: mIoU 0.518, versus 0.722 for OmniCloudMask
  fed with 10 m B04/B03/B8A. It is a reproducible baseline, not a validated
  cloud mask on real MSS.
- The 40 human-labelled chips are prepared (60 m chips, empty CSVs, annotation
  tooling), but they are not annotated: no accuracy and no kappa on real MSS
  yet.
- `mss5_esa_gtc_v1()` is calibrated on five desert PICS, all winter/autumn with
  SZA 50-59 deg. ERA5 puts their water-vapour column at 0.46-1.83 cm, so
  `gamma = 0.112` and the B4 scale stay degenerate with the site-to-site
  spread: the gamma is fitted, not measured. A humid or summer PICS is still
  missing (Makgadikgadi was rejected: s2cloudless flags 28-58 % of the pan as
  cloud and the window is not pseudo-invariant).
- `parse_mtl` is tested against three real Landsat 1-3 MTLs (1978, 1982, 1983)
  and one of them, the Landsat 1 scene, has `"NULL"` for the whole band 4:
  `parse_mtl` keeps it as `None` and `dn_to_radiance` refuses to convert with a
  message naming the band.
- The SBAF is linear and has no atmosphere: the O2-A core at 0.76 um inside MSS
  band 3 is invisible to it, which is why `band_scaling.nir1` is 0.867 and not
  0.93. 6SV quantifies the gap (0.953 of gas scale at 1 cm); the gas term is
  not in the emulator yet.
- The paper figures are not in this release: the numbers in
  `version_to_paper/` still mix the old test v7 and the QA4EO-2 model.

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