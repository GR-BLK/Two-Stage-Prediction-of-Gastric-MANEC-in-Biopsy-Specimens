# WSI + Clinical Features Contrastive Multimodal Fusion Pipeline

A multimodal classification pipeline that combines whole-slide image (WSI)
patch-level features with tabular clinical features for binary
classification. Multiple fusion architectures are supported, including
contrastive fusion with improved modality dropout, gated fusion, and
confidence-guided fusion. The pipeline performs rigorous internal
cross-validation and external validation with comprehensive reporting:
BCa bootstrap confidence intervals, Wilson confidence intervals, decision
curve analysis, and Brier scores.

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
- Users are responsible for ensuring that any data they process is handled
  in accordance with applicable ethics and privacy regulations.
- The pipeline follows TRIPOD reporting guidelines.

---

## Class Label Mapping

The binary task in this pipeline is **class B vs class C**. Class A is
excluded at load time.

| Directory name | Original label | Binary role | Binary label |
|----------------|----------------|-------------|--------------|
| `A`            | ADC            | Excluded    | —            |
| `B`            | MANEC          | Positive    | 1            |
| `C`            | NEC            | Negative    | 0            |

The class directory names and label mapping are defined by the following
constants in `contrastive_fusion_binary.py`:

```python
CLASSES = ['C', 'B']
CLASSES_ORIGINAL = ['A', 'B', 'C']
POSITIVE_CLASS = 'B'
NEGATIVE_CLASS = 'C'
EXCLUDED_CLASS = 'A'
LABEL_MAP_BINARY = {'C': 0, 'B': 1}