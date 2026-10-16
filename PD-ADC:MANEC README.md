# WSI + Clinical Features Multimodal Fusion Pipeline

A multimodal classification pipeline that combines whole-slide image (WSI)
patch-level features with tabular clinical features for binary
classification. Multiple fusion architectures are supported, including gated
fusion, confidence-guided fusion, and contrastive fusion. The pipeline
performs rigorous internal cross-validation and external validation with
comprehensive reporting: BCa bootstrap confidence intervals, Wilson
confidence intervals, decision curve analysis, and Brier scores.

---

## Data & Privacy Notice

- This repository contains **no patient data** and **no model weights**.
- Users must provide their own patch feature directories and clinical Excel
  files.
- Do **not** commit any of the following to GitHub:
  - Whole-slide images (`.svs`, `.tif`, `.ndpi`, `.mrxs`, `.scn`)
  - GeoJSON annotation files
  - Clinical Excel / CSV tables
  - Extracted patch features (`.npz`, `.npy`)
  - Trained model checkpoints (`.pkl`, `.pth`, `.pt`, `.bin`)
  - Prediction CSVs, ROC/DCA figures, or any pipeline output
- Users are responsible for ensuring that any data they process is handled in
  accordance with applicable ethics and privacy regulations.
- The pipeline follows TRIPOD reporting guidelines.

---

## Class Label Mapping

The binary task in this pipeline is **class A vs class B**. Class C is
excluded at load time.

| Directory name | Original label | Binary role | Binary label |
|----------------|----------------|-------------|--------------|
| `A`            | MANEC          | Negative    | 0            |
| `B`            | ADC            | Positive    | 1            |
| `C`            | NEC            | Excluded    | —            |

The class directory names and label mapping are defined by the following
constants in `multimodal_fusion_binary.py`:

```python
CLASSES = ['A', 'B']
CLASSES_ORIGINAL = ['B', 'A']
POSITIVE_CLASS = 'B'
NEGATIVE_CLASS = 'A'
LABEL_MAP_BINARY = {'B': 1, 'A': 0}