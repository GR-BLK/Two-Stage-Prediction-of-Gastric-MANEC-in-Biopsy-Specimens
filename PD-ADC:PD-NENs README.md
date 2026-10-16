# WSI + Clinical Features Multimodal Fusion Pipeline

Whole-slide image (WSI) + clinical feature multimodal classification pipeline
for ADC vs Non-ADC, with per-model threshold optimization and comprehensive
reporting (BCa bootstrap CIs, Wilson CIs, DCA, Brier score).

## Data & Privacy Notice

- This repository contains **no patient data** and **no model weights**.
- Users must provide their own patch feature directories and clinical Excel
  files. Do not commit any patient data, SVS/TIF files, or clinical tables.
- Users are responsible for ensuring that any data they process is handled
  in accordance with applicable ethics and privacy regulations.
- Follows TRIPOD reporting guidelines.

## Installation

```bash
pip install -r requirements.txt