# WSI Image Processing

Whole-slide image (WSI) preprocessing pipeline for computational pathology:
**patch extraction → patch filtering → stain normalization**.

Provides both CPU (numpy / OpenCV) and GPU (PyTorch / CUDA) implementations.
The GPU path uses all visible CUDA devices and is numerically equivalent to
the CPU path.

## ⚠️ Data & Privacy Notice

- This repository contains **no patient data** and **no reference templates**.
- Users must provide their own SVS, GeoJSON, and reference patch directories.
- Users are responsible for ensuring that any data they process is handled
  in accordance with applicable ethics and privacy regulations.
- Do not commit SVS / TIF / GeoJSON / normalized outputs to GitHub.

## Installation

System dependency (OpenSlide):

```bash
# Ubuntu / Debian
sudo apt install libopenslide0

# macOS
brew install openslide