#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
WSI + Clinical Features Multimodal Gated Fusion Classification Pipeline (binary A vs B).

This module implements a complete multimodal classification pipeline that
combines whole-slide image (WSI) patch-level features with tabular clinical
features for binary classification. Multiple fusion architectures are
supported (gated fusion, confidence-guided fusion, contrastive fusion), and
the pipeline performs rigorous internal cross-validation plus external
validation with comprehensive reporting (BCa bootstrap CIs, Wilson CIs,
decision curve analysis, Brier score).

Class label mapping
-------------------
    A  -> negative class (formerly MANEC)
    B  -> positive class (formerly ADC)
    C  -> excluded class (formerly NEC); filtered out at load time

Global random seed: 2026 (full reproducibility).

Methodological guarantees
-------------------------
- Zero data leakage: all preprocessing is fitted on the training set only.
- Clinical preprocessing: binary kept as-is, continuous standardized,
  multi-class one-hot encoded.
- Compliant with TRIPOD reporting guidelines.

Data privacy notice
-------------------
This repository contains no patient data and no model weights. Users must
provide their own patch feature directories and clinical Excel files, and
are responsible for ensuring that any data they process is handled in
accordance with applicable ethics and privacy regulations.
"""

import argparse
import multiprocessing
import os
import pickle
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
)
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ============================================================================
# Hardware detection (utilize all available resources)
# ============================================================================
N_CPUS = multiprocessing.cpu_count()
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

os.environ['OMP_NUM_THREADS'] = str(N_CPUS)
os.environ['MKL_NUM_THREADS'] = str(N_CPUS)
os.environ['OPENBLAS_NUM_THREADS'] = str(N_CPUS)
os.environ['NUMEXPR_NUM_THREADS'] = str(N_CPUS)
torch.set_num_threads(N_CPUS)

if N_GPUS > 0:
    print(f"\n  [Hardware] Detected {N_GPUS} GPU(s): "
          f"{[torch.cuda.get_device_name(i) for i in range(N_GPUS)]}")
    print(f"  [Hardware] Detected {N_CPUS} CPU threads (fully utilized)")
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    USE_MULTI_GPU = True
else:
    print(f"\n  [Hardware] No GPU detected; running on CPU")
    print(f"  [Hardware] Detected {N_CPUS} CPU threads (fully utilized)")
    DEVICE = torch.device('cpu')
    torch.backends.cudnn.benchmark = False
    USE_MULTI_GPU = False


# ============================================================================
# Global configuration
# ============================================================================
SEED = 2026
N_FOLDS = 10
N_CLASSES = 2

# Binary class labels
#   A  -> negative class (formerly MANEC)
#   B  -> positive class (formerly ADC)
CLASSES = ['A', 'B']
CLASSES_ORIGINAL = ['B', 'A']
POSITIVE_CLASS = 'B'
NEGATIVE_CLASS = 'A'

# Patch-level features (raw patch features, CLAM attention pooling)
PATCH_FEAT_DIM = 1536
CLAM_HIDDEN_DIM = 128
CLAM_DROPOUT = 0.2
STAT_INNER_CV_FOLDS = 5
C_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# Fusion and training hyperparameters
FUSION_HIDDEN_DIM = 512
BATCH_SIZE = 8
MAX_EPOCHS = 100
PATIENCE = 15
LR = 5e-5
WEIGHT_DECAY = 1e-3
MIXUP_ALPHA = 0.2

# LASSO clinical feature selection (LogisticRegression L1)
ENABLE_LASSO_SELECTION = True
LASSO_CV_MAX_ITER = 5000
MIN_LASSO_FEATURES = 5

# Clinical feature schema
# ---------------------------------------------------------------------------
# These constants define the expected column names in the clinical Excel file.
# Rename your columns accordingly, or adjust these constants to match your
# data schema before running the pipeline.
# ---------------------------------------------------------------------------
BINARY_FEATURES = [
    'sex', 'abdominal_pain', 'abdominal_distension', 'nausea', 'vomiting',
    'acid_reflux', 'belching', 'poor_appetite', 'diarrhea', 'constipation',
    'hematemesis', 'melena', 'dysphagia', 'fatigue', 'weight_loss',
    'cough', 'fever', 'hypertension', 'diabetes', 'family_history_of_cancer',
]
CONTINUOUS_FEATURES = ['age']
MULTICLASS_FEATURES = ['location', 'gross_type']

MODAL_ALIGN_LAMBDA = 0.0
CONTRAST_LOSS_WEIGHT = 0.1

# Post-processing: per-model threshold optimization on each model's own OOF
DEFAULT_THRESHOLD = 0.5
MODEL_THRESHOLDS = {}

# Bootstrap confidence intervals
N_BOOTSTRAP = 10000

# Post-processing
ENABLE_THRESHOLD_OPTIMIZATION = True
# Threshold search tolerance: among thresholds whose Youden index is within
# this tolerance of the maximum, choose the one with the highest specificity
# on the model's own OOF predictions, in order to maximize correct
# identification of class A (negative class, clinically distinct from class B).
YOUDEN_TOLERANCE = 0.02

# Cache version: after fixing the LR double-standardization leak, older caches
# (version < 2) are automatically invalidated and recomputed.
LR_LEAK_FIX_VERSION = 2


def setup_output_dirs(output_dir):
    for subdir in ['folds', 'models', 'predictions', 'figures', 'baselines']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)


# ============================================================================
# Reproducibility
# ============================================================================
def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


set_seed(SEED)


# ============================================================================
# Label mapping
# ---------------------------------------------------------------------------
# Category sub-directory names map to binary labels as follows:
#   B  -> positive class (1)
#   A  -> negative class (0)
#   C  -> excluded from the binary task (filtered at load time)
# Adjust this dictionary to match your data layout.
# ============================================================================
LABEL_MAP_BINARY = {'B': 1, 'A': 0}


def to_binary_labels(categories):
    return np.array([LABEL_MAP_BINARY[c] for c in categories], dtype=np.int64)


# ============================================================================
# Stratified bootstrap BCa 95% CI for AUC
# ============================================================================
def auc_bca_ci(y_true, y_prob, n_bootstrap=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    """Compute the stratified bootstrap BCa 95% confidence interval for AUC.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (0/1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    n_bootstrap : int
        Number of bootstrap resamples.
    alpha : float
        Significance level.
    seed : int
        Random seed.

    Returns
    -------
    (auc, ci_lower, ci_upper) : tuple of float
    """
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    obs_auc = roc_auc_score(y_true, y_prob)

    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    n_pos = len(idx_pos)
    n_neg = len(idx_neg)

    boot_aucs = []
    for _ in range(n_bootstrap):
        sample_pos = rng.choice(idx_pos, size=n_pos, replace=True)
        sample_neg = rng.choice(idx_neg, size=n_neg, replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        try:
            boot_aucs.append(roc_auc_score(y_true[sample_idx], y_prob[sample_idx]))
        except Exception:
            continue
    boot_aucs = np.array(boot_aucs)

    z0 = norm.ppf(np.mean(boot_aucs < obs_auc))

    n = len(y_true)
    jack_aucs = []
    all_idx = np.arange(n)
    for i in range(n):
        jack_idx = np.delete(all_idx, i)
        try:
            jack_aucs.append(roc_auc_score(y_true[jack_idx], y_prob[jack_idx]))
        except ValueError:
            continue
    jack_aucs = np.array(jack_aucs)
    if len(jack_aucs) < 3:
        ci_lower = np.quantile(boot_aucs, norm.cdf(2 * z0 + norm.ppf(alpha / 2)))
        ci_upper = np.quantile(boot_aucs, norm.cdf(2 * z0 + norm.ppf(1 - alpha / 2)))
        return obs_auc, ci_lower, ci_upper
    jack_mean = np.mean(jack_aucs)
    num = np.sum((jack_mean - jack_aucs) ** 3)
    den = np.sum((jack_mean - jack_aucs) ** 2)
    a = num / (6 * (den ** 1.5)) if den != 0 else 0.0

    z_alpha2 = norm.ppf(alpha / 2)
    z_1_alpha2 = norm.ppf(1 - alpha / 2)

    p1 = norm.cdf(z0 + (z0 + z_alpha2) / (1 - a * (z0 + z_alpha2)))
    p2 = norm.cdf(z0 + (z0 + z_1_alpha2) / (1 - a * (z0 + z_1_alpha2)))

    ci_lower = np.quantile(boot_aucs, p1)
    ci_upper = np.quantile(boot_aucs, p2)

    return obs_auc, ci_lower, ci_upper


# ============================================================================
# General stratified bootstrap BCa 95% CI (any metric)
# ============================================================================
def bca_ci_general(y_true, y_prob, metric_fn, n_bootstrap=N_BOOTSTRAP,
                   alpha=0.05, seed=SEED):
    """Compute the stratified bootstrap BCa 95% CI for an arbitrary metric.

    Parameters
    ----------
    metric_fn : callable(y_true, y_prob) -> float
    """
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    obs = metric_fn(y_true, y_prob)

    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    n_pos = len(idx_pos)
    n_neg = len(idx_neg)

    boot_vals = []
    for _ in range(n_bootstrap):
        sample_pos = rng.choice(idx_pos, size=n_pos, replace=True)
        sample_neg = rng.choice(idx_neg, size=n_neg, replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        try:
            boot_vals.append(metric_fn(y_true[sample_idx], y_prob[sample_idx]))
        except Exception:
            continue
    boot_vals = np.array(boot_vals)

    z0 = norm.ppf(np.mean(boot_vals < obs))

    n = len(y_true)
    jack_vals = []
    all_idx = np.arange(n)
    for i in range(n):
        jack_idx = np.delete(all_idx, i)
        try:
            jack_vals.append(metric_fn(y_true[jack_idx], y_prob[jack_idx]))
        except Exception:
            jack_vals.append(np.nan)
    jack_vals = np.array(jack_vals)
    jack_vals = jack_vals[~np.isnan(jack_vals)]
    if len(jack_vals) < 3:
        ci_lower = np.quantile(boot_vals, alpha / 2)
        ci_upper = np.quantile(boot_vals, 1 - alpha / 2)
        return obs, ci_lower, ci_upper
    jack_mean = np.mean(jack_vals)
    num = np.sum((jack_mean - jack_vals) ** 3)
    den = np.sum((jack_mean - jack_vals) ** 2)
    a = num / (6 * (den ** 1.5)) if den != 0 else 0.0

    z_alpha2 = norm.ppf(alpha / 2)
    z_1_alpha2 = norm.ppf(1 - alpha / 2)

    p1 = norm.cdf(z0 + (z0 + z_alpha2) / (1 - a * (z0 + z_alpha2)))
    p2 = norm.cdf(z0 + (z0 + z_1_alpha2) / (1 - a * (z0 + z_1_alpha2)))

    ci_lower = np.quantile(boot_vals, max(0, min(1, p1)))
    ci_upper = np.quantile(boot_vals, max(0, min(1, p2)))

    return obs, ci_lower, ci_upper


def compute_brier_score(y_true, y_prob):
    """Brier score: mean squared error between predicted probability and label."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return np.mean((y_prob - y_true) ** 2)


def wilson_ci(numerator, denominator, alpha=0.05):
    """Wilson score interval for a proportion.

    Returns
    -------
    (p, ci_lower, ci_upper) : tuple of float
    """
    if denominator == 0:
        return float('nan'), float('nan'), float('nan')
    p = numerator / denominator
    n = denominator
    z = norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, center - margin, center + margin


def round_to_4sf(value):
    """Round a float to 4 significant figures."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float('nan')
    if value == 0:
        return 0.0
    return float(f"{value:.4g}")


def compute_oof_metrics_with_ci(y_true, y_prob, n_bootstrap=N_BOOTSTRAP,
                                alpha=0.05, seed=SEED):
    """Compute all OOF metrics with BCa bootstrap CIs.

    Returns a dictionary with keys: brier, brier_ci, auc_ci.
    """
    results = {}
    results['brier'] = compute_brier_score(y_true, y_prob)
    results['brier_ci'] = bca_ci_general(y_true, y_prob,
        lambda yt, yp: compute_brier_score(yt, yp), n_bootstrap, alpha, seed)
    results['auc_ci'] = bca_ci_general(y_true, y_prob,
        lambda yt, yp: roc_auc_score(yt, yp), n_bootstrap, alpha, seed)
    return results


def compute_oof_wilson_cis(y_true, y_prob, threshold=None):
    """Compute Wilson CIs for OOF sensitivity, specificity, accuracy, F1, PPV, NPV.

    F1 is treated as a proportion 2TP / (2TP + FP + FN). Classification uses
    the given threshold; when omitted, ``ACTIVE_THRESHOLD`` is used.
    """
    if threshold is None:
        threshold = ACTIVE_THRESHOLD
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'sensitivity': wilson_ci(tp, tp + fn),
        'specificity': wilson_ci(tn, tn + fp),
        'accuracy': wilson_ci(tp + tn, tp + tn + fp + fn),
        'f1': wilson_ci(2 * tp, 2 * tp + fp + fn),
        'ppv': wilson_ci(tp, tp + fp),
        'npv': wilson_ci(tn, tn + fn),
    }


# ============================================================================
# Loss: unified standard binary cross-entropy (BCEWithLogitsLoss, no weights)
# ============================================================================
ACTIVE_THRESHOLD = DEFAULT_THRESHOLD


class BCEWithLogitsLossWrapper(nn.Module):
    """Unified standard BCE loss: take the positive-class logit and apply BCE.

    Equivalent to ``torch.nn.BCEWithLogitsLoss(pos_weight=None)``, adapted for
    two-class logits.
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        # logits: (N, 2); targets: 0/1 hard labels or mixup soft labels in [0, 1]
        return self.bce(logits[:, 1], targets.float())


def prob_to_class(prob, threshold=None):
    """Map probabilities to classes: prob >= threshold -> positive class (B).

    When ``threshold`` is omitted, the global fallback ``ACTIVE_THRESHOLD``
    is used. Per-model callers should pass the model's own OOF-optimized
    threshold.
    """
    if threshold is None:
        threshold = ACTIVE_THRESHOLD
    prob = np.asarray(prob)
    return (prob >= threshold).astype(int)


# ============================================================================
# Data loading
# ============================================================================
def load_patch_features(patch_dir):
    """Load patch-level raw features from ``*_features.npz`` files.

    Each ``.npz`` file contains a ``features`` array of shape
    ``(n_patches, PATCH_FEAT_DIM)``.

    Class directory names must be one of ``B`` / ``A`` / ``C``. Samples in
    directory ``C`` are excluded from the binary A vs B task.

    Parameters
    ----------
    patch_dir : str
        Root directory containing category sub-directories (B / A / C), each
        holding ``*_features.npz`` files.

    Returns
    -------
    case_ids : list of str
    patch_list : list of np.ndarray, each of shape (n_patches_i, PATCH_FEAT_DIM)
    categories : np.ndarray of str
    """
    import glob

    case_ids = []
    patch_list = []
    categories = []

    npz_files = sorted(glob.glob(os.path.join(patch_dir, '*', '*_features.npz')))

    if len(npz_files) == 0:
        raise FileNotFoundError(
            f"No .npz files found at: {patch_dir}/*/*_features.npz")

    for npz_path in npz_files:
        basename = os.path.basename(npz_path)
        case_id = basename.rsplit('_features', 1)[0]

        dir_name = os.path.basename(os.path.dirname(npz_path))
        category = dir_name  # 'B', 'A', or 'C'

        data = np.load(npz_path)
        patch_features = data['features'].astype(np.float32)

        case_ids.append(case_id)
        patch_list.append(patch_features)
        categories.append(category)

    categories = np.array(categories)

    # Binary task is B vs A: drop C samples consistently across training,
    # validation, and external test.
    keep_mask = np.isin(categories, ['B', 'A'])
    if keep_mask.sum() != len(categories):
        n_drop = int((~keep_mask).sum())
        case_ids = [c for c, k in zip(case_ids, keep_mask) if k]
        patch_list = [p for p, k in zip(patch_list, keep_mask) if k]
        categories = categories[keep_mask]
        print(f"  [Label filter] Excluded {n_drop} class-C cases; "
              f"keeping only A and B: {len(case_ids)} cases remaining")

    n_patches = [p.shape[0] for p in patch_list]
    print(f"  Loaded patch features: {len(case_ids)} cases, "
          f"{PATCH_FEAT_DIM} dims / patch")
    print(f"  Patches per case: min={min(n_patches)}, max={max(n_patches)}, "
          f"median={np.median(n_patches):.0f}")
    print(f"  Class distribution: "
          f"{dict(zip(*np.unique(categories, return_counts=True)))}")
    return case_ids, patch_list, categories


def compute_mean_pooled_features(patch_list):
    """Mean-pool raw patches into fixed-length feature vectors.

    Returns
    -------
    features : np.ndarray of shape (n_cases, PATCH_FEAT_DIM)
    """
    features = np.array([np.mean(p, axis=0) for p in patch_list], dtype=np.float32)
    return features


# ============================================================================
# CLAM attention pooling (variable-length patch sequences)
# ============================================================================
class CLAMAttentionPooling(nn.Module):
    """CLAM-style attention pooling over variable-length patch sequences.

    Input
    -----
    patches : list of tensors (n_i, D) or padded tensor (B, max_n, D)
    mask    : optional padded-mask tensor (B, max_n)

    Output
    ------
    pooled : tensor of shape (B, D)
    attn   : attention weights (list or tensor of shape (B, max_n))
    """
    def __init__(self, input_dim=PATCH_FEAT_DIM, hidden_dim=CLAM_HIDDEN_DIM,
                 dropout=CLAM_DROPOUT):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, patches, mask=None):
        if isinstance(patches, list):
            return self._forward_list(patches)
        return self._forward_padded(patches, mask)

    def _forward_list(self, patch_list):
        pooled, attn_weights = [], []
        for p in patch_list:
            p = p.to(next(self.attention[0].parameters()).device)
            A = self.attention(p)
            A = torch.softmax(A, dim=0)
            pooled.append((A * p).sum(dim=0))
            attn_weights.append(A.squeeze(-1))
        return torch.stack(pooled), attn_weights

    def _forward_padded(self, patches_padded, mask):
        B, N, D = patches_padded.shape
        patches_flat = patches_padded.view(B * N, D)
        A = self.attention(patches_flat).view(B, N)
        A = A.masked_fill(~mask, float('-inf'))
        A = torch.softmax(A, dim=1)
        pooled = (A.unsqueeze(-1) * patches_padded).sum(dim=1)
        return pooled, A


def load_clinical_features(excel_path):
    """Load clinical features from an Excel file.

    The file must contain a ``case_id`` column plus all columns listed in
    ``BINARY_FEATURES``, ``CONTINUOUS_FEATURES``, and ``MULTICLASS_FEATURES``.
    """
    df = pd.read_excel(excel_path)
    column_rename = {
        'seq_no': 'case_id',
        'endoscopic_type': 'gross_type',
        'lesion_location': 'location',
    }
    df = df.rename(columns={k: v for k, v in column_rename.items()
                            if k in df.columns})
    required_cols = (['case_id'] + BINARY_FEATURES + CONTINUOUS_FEATURES
                     + MULTICLASS_FEATURES)
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Clinical feature table is missing required columns: "
                         f"{missing_cols}")
    na_total = df[required_cols].isna().sum().sum()
    if na_total > 0:
        warnings.warn(f"Clinical features contain {na_total} missing values. "
                      f"Please complete upstream QC and imputation first.")
    print(f"  Loaded clinical features: {len(df)} cases, "
          f"{len(required_cols) - 1} features")
    print(f"    Binary: {len(BINARY_FEATURES)} | Continuous: "
          f"{len(CONTINUOUS_FEATURES)} | Multi-class: {len(MULTICLASS_FEATURES)}")
    return df


# ============================================================================
# Fold splitting
# ============================================================================
def generate_stratified_folds(case_ids, categories, n_folds=N_FOLDS,
                              seed=SEED, save_path=None):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    case_ids_arr = np.array(case_ids)
    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(case_ids_arr, categories)):
        train_cases = case_ids_arr[train_idx].tolist()
        val_cases = case_ids_arr[val_idx].tolist()
        train_cats = categories[train_idx]
        val_cats = categories[val_idx]
        folds.append({
            'fold': fold_idx,
            'train_cases': train_cases,
            'val_cases': val_cases,
            'train_dist': {c: int(np.sum(train_cats == c))
                           for c in CLASSES_ORIGINAL},
            'val_dist': {c: int(np.sum(val_cats == c))
                         for c in CLASSES_ORIGINAL},
        })
        print(f"  Fold {fold_idx}: train={len(train_cases)} val={len(val_cases)}")
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(folds, f)
        print(f"  {n_folds}-fold split saved: {save_path}")
    return folds


def load_or_create_folds(case_ids, categories, output_dir):
    folds_path = os.path.join(output_dir, 'folds', 'stratified_folds.pkl')
    if os.path.exists(folds_path):
        with open(folds_path, 'rb') as f:
            folds = pickle.load(f)
        print(f"  Loaded existing fold split: {folds_path}")
        return folds
    return generate_stratified_folds(case_ids, categories, save_path=folds_path)


# ============================================================================
# Branch C: clinical feature preprocessor
# ============================================================================
class ClinicalPreprocessor:
    """Fit / transform clinical features with zero data leakage."""

    def __init__(self, binary_cols=None, continuous_cols=None,
                 multiclass_cols=None, drop_first=True):
        self.binary_cols = binary_cols or BINARY_FEATURES
        self.continuous_cols = continuous_cols or CONTINUOUS_FEATURES
        self.multiclass_cols = multiclass_cols or MULTICLASS_FEATURES
        self.drop_first = drop_first
        self.scaler = None
        self.multiclass_maps = {}
        self.multiclass_modes = {}
        self.multiclass_drop_indices = {}
        self.output_dim = None

    def fit(self, df_train):
        if self.continuous_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(df_train[self.continuous_cols].values)
            print(f"    Continuous features standardized: {self.continuous_cols}")
        for col in self.multiclass_cols:
            train_vals = df_train[col].dropna().values
            unique_cats = sorted(np.unique(train_vals).tolist())
            self.multiclass_maps[col] = {cat: idx for idx, cat in enumerate(unique_cats)}
            self.multiclass_modes[col] = pd.Series(train_vals).mode().iloc[0]
            if self.drop_first and len(unique_cats) > 1:
                self.multiclass_drop_indices[col] = 0
            else:
                self.multiclass_drop_indices[col] = -1
            print(f"    Multi-class [{col}] fitted: {len(unique_cats)} classes "
                  f"(drop_first={self.drop_first})")
        binary_dim = len(self.binary_cols)
        cont_dim = len(self.continuous_cols)
        multi_dim = sum(
            max(len(m) - 1 if self.drop_first else len(m), 0)
            for m in self.multiclass_maps.values()
        )
        self.output_dim = binary_dim + cont_dim + multi_dim
        print(f"    Clinical feature output dim: {self.output_dim}")
        return self

    def transform(self, df):
        n_samples = len(df)
        features_list = []
        if self.binary_cols:
            features_list.append(df[self.binary_cols].values.astype(np.float32))
        if self.continuous_cols and self.scaler is not None:
            features_list.append(
                self.scaler.transform(df[self.continuous_cols].values).astype(np.float32)
            )
        for col in self.multiclass_cols:
            cat_map = self.multiclass_maps[col]
            mode_val = self.multiclass_modes[col]
            n_cats = len(cat_map)
            one_hot = np.zeros((n_samples, n_cats), dtype=np.float32)
            for i, val in enumerate(df[col].values):
                if val not in cat_map:
                    val = mode_val
                one_hot[i, cat_map[val]] = 1.0
            drop_idx = self.multiclass_drop_indices.get(col, -1)
            if drop_idx >= 0:
                keep_cols = [c for c in range(n_cats) if c != drop_idx]
                one_hot = one_hot[:, keep_cols]
            features_list.append(one_hot)
        return np.hstack(features_list).astype(np.float32)

    def save(self, save_dir, fold_idx):
        save_path = os.path.join(save_dir,
                                 f'clinical_preprocessor_fold_{fold_idx}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump({
                'binary_cols': self.binary_cols,
                'continuous_cols': self.continuous_cols,
                'multiclass_cols': self.multiclass_cols,
                'drop_first': self.drop_first,
                'scaler': self.scaler,
                'multiclass_maps': self.multiclass_maps,
                'multiclass_modes': self.multiclass_modes,
                'multiclass_drop_indices': self.multiclass_drop_indices,
                'output_dim': self.output_dim
            }, f)

    def load(self, load_dir, fold_idx):
        load_path = os.path.join(load_dir,
                                 f'clinical_preprocessor_fold_{fold_idx}.pkl')
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            self.binary_cols = data['binary_cols']
            self.continuous_cols = data['continuous_cols']
            self.multiclass_cols = data['multiclass_cols']
            self.drop_first = data.get('drop_first', True)
            self.scaler = data['scaler']
            self.multiclass_maps = data['multiclass_maps']
            self.multiclass_modes = data['multiclass_modes']
            self.multiclass_drop_indices = data.get('multiclass_drop_indices', {})
            self.output_dim = data['output_dim']
        return self

    def save_final(self, save_dir, prefix='final'):
        save_path = os.path.join(save_dir, f'{prefix}_clinical_preprocessor.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump({
                'binary_cols': self.binary_cols,
                'continuous_cols': self.continuous_cols,
                'multiclass_cols': self.multiclass_cols,
                'drop_first': self.drop_first,
                'scaler': self.scaler,
                'multiclass_maps': self.multiclass_maps,
                'multiclass_modes': self.multiclass_modes,
                'multiclass_drop_indices': self.multiclass_drop_indices,
                'output_dim': self.output_dim
            }, f)


# ============================================================================
# LASSO clinical feature selector (fitted on the training fold only)
# ============================================================================
class LassoClinicalSelector:
    """LASSO LogisticRegression selector for clinical features.

    Uses ``LogisticRegression(penalty='l1', solver='saga')`` with an inner
    5-fold CV over lambda (C = 1 / lambda); retains features with
    ``|coef| > 1e-8``.
    """
    def __init__(self, enable=True, seed=SEED):
        self.enable = enable
        self.seed = seed
        self.model = None
        self.best_lambda = None
        self.selected_mask = None
        self.original_dim = None
        self.selected_dim = None

    def fit(self, X_train, y_train):
        if not self.enable:
            self.original_dim = X_train.shape[1]
            self.selected_dim = self.original_dim
            self.selected_mask = np.ones(self.original_dim, dtype=bool)
            print(f"  LASSO selection disabled; keeping all "
                  f"{self.original_dim} clinical features")
            return self
        self.original_dim = X_train.shape[1]
        n_samples = X_train.shape[0]
        if n_samples < 10:
            print(f"  Sample size ({n_samples}) too small; skipping LASSO")
            self.selected_dim = self.original_dim
            self.selected_mask = np.ones(self.original_dim, dtype=bool)
            return self

        lambda_values = np.logspace(-5, 5, 50)
        inner_skf = StratifiedKFold(n_splits=min(5, n_samples - 1),
                                    shuffle=True, random_state=self.seed)
        best_lambda = None
        best_deviance = float('inf')

        for lam in lambda_values:
            deviances = []
            for inner_train_idx, inner_val_idx in inner_skf.split(X_train, y_train):
                X_inner_train = X_train[inner_train_idx]
                y_inner_train = y_train[inner_train_idx]
                X_inner_val = X_train[inner_val_idx]
                y_inner_val = y_train[inner_val_idx]

                model = LogisticRegression(
                    penalty='l1', C=1.0 / lam, solver='saga',
                    max_iter=LASSO_CV_MAX_ITER, random_state=self.seed
                )
                model.fit(X_inner_train, y_inner_train)

                y_prob_inner = model.predict_proba(X_inner_val)
                eps = 1e-15
                y_prob_inner = np.clip(y_prob_inner, eps, 1 - eps)

                log_likelihood = 0
                for i in range(len(y_inner_val)):
                    log_likelihood += np.log(y_prob_inner[i, y_inner_val[i]])
                deviance = -2 * log_likelihood / len(y_inner_val)
                deviances.append(deviance)

            avg_deviance = np.mean(deviances)
            if avg_deviance < best_deviance:
                best_deviance = avg_deviance
                best_lambda = lam

        self.best_lambda = best_lambda
        self.model = LogisticRegression(
            penalty='l1', C=1.0 / best_lambda, solver='saga',
            max_iter=LASSO_CV_MAX_ITER, random_state=self.seed
        )
        self.model.fit(X_train, y_train)

        coef = self.model.coef_
        if coef.ndim == 2:
            coef = coef[0]
        self.selected_mask = np.abs(coef) > 1e-8
        self.selected_dim = self.selected_mask.sum()

        if self.selected_dim < MIN_LASSO_FEATURES:
            orig_selected = self.selected_dim
            k = max(MIN_LASSO_FEATURES, self.original_dim // 3)
            top_indices = np.argsort(np.abs(coef))[-k:]
            self.selected_mask = np.zeros(self.original_dim, dtype=bool)
            self.selected_mask[top_indices] = True
            self.selected_dim = k
            print(f"  LASSO retained only {orig_selected} features "
                  f"(< {MIN_LASSO_FEATURES}); falling back to top-{k} by |coef|")

        print(f"  LASSO selection: {self.original_dim} -> {self.selected_dim} "
              f"(lambda={self.best_lambda:.6f}, "
              f"{self.selected_dim / self.original_dim * 100:.1f}%)")
        return self

    def transform(self, X):
        if not self.enable or self.selected_mask is None:
            return X
        return X[:, self.selected_mask]

    def fit_transform(self, X_train, y_train):
        self.fit(X_train, y_train)
        return self.transform(X_train)

    def get_selected_indices(self):
        if self.selected_mask is None:
            return list(range(self.original_dim or 0))
        return np.where(self.selected_mask)[0].tolist()

    def save(self, save_dir, fold_idx):
        save_path = os.path.join(save_dir, f'lasso_selector_fold_{fold_idx}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump({
                'enable': self.enable, 'selected_mask': self.selected_mask,
                'original_dim': self.original_dim, 'selected_dim': self.selected_dim,
                'best_lambda': self.best_lambda,
            }, f)

    def load(self, load_dir, fold_idx):
        load_path = os.path.join(load_dir, f'lasso_selector_fold_{fold_idx}.pkl')
        if not os.path.exists(load_path):
            self.enable = False
            return self
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            self.enable = data['enable']
            self.selected_mask = data['selected_mask']
            self.original_dim = data['original_dim']
            self.selected_dim = data['selected_dim']
            self.best_lambda = data.get('best_lambda')
        return self

    def save_final(self, save_dir, prefix='final'):
        save_path = os.path.join(save_dir, f'{prefix}_lasso_selector.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump({
                'enable': self.enable, 'selected_mask': self.selected_mask,
                'original_dim': self.original_dim, 'selected_dim': self.selected_dim,
                'best_lambda': self.best_lambda,
            }, f)

    def load_final(self, load_dir, prefix='final'):
        load_path = os.path.join(load_dir, f'{prefix}_lasso_selector.pkl')
        if not os.path.exists(load_path):
            self.enable = False
            return self
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            self.enable = data['enable']
            self.selected_mask = data['selected_mask']
            self.original_dim = data['original_dim']
            self.selected_dim = data['selected_dim']
            self.best_lambda = data.get('best_lambda')
        return self


# ============================================================================
# ThyFusionNet: gated attention fusion
# ============================================================================
class GatedFeatureFusion(nn.Module):
    """ThyFusionNet Gated Feature Fusion (GFF) module.

    Reference
    ---------
    ThyFusionNet: a CNN-transformer framework with spatial-aware sparse
    attention for multimodal thyroid disease diagnosis.

    Both modalities jointly determine the gate weight.
    """
    def __init__(self, feat_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = feat_dim

        self.gate_net = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim)
        )

        # Initialize the gate to be biased toward the image modality.
        nn.init.constant_(self.gate_net[-2].bias, 1.0)

    def forward(self, img_feat: torch.Tensor, cli_feat: torch.Tensor):
        concat = torch.cat([img_feat, cli_feat], dim=1)
        gate = self.gate_net(concat)
        fused = gate * img_feat + (1 - gate) * cli_feat
        fused = self.fusion_proj(fused)
        return fused, gate


class ThyFusionNetFusionModel(nn.Module):
    """Integrate GFF into the pipeline with CLAM attention pooling."""

    def __init__(self, clinical_input_dim, common_dim=64, n_classes=2,
                 fusion_hidden=128, seed=2026, **kwargs):
        super().__init__()
        torch.manual_seed(seed)

        self.clam_pooling = CLAMAttentionPooling()

        self.img_encoder = nn.Sequential(
            nn.Linear(PATCH_FEAT_DIM, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden, common_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.cli_encoder = nn.Sequential(
            nn.Linear(clinical_input_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden, common_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.gff = GatedFeatureFusion(feat_dim=common_dim)

        self.classifier = nn.Sequential(
            nn.Linear(common_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, patches, clinical_features=None, masks=None,
                mixup_lam=None, mixup_perm=None):
        if isinstance(patches, list):
            pooled, _ = self.clam_pooling(patches)
        else:
            pooled, _ = self.clam_pooling(patches, masks)

        img_feat = self.img_encoder(pooled)
        cli_feat = self.cli_encoder(clinical_features)

        fused, gate = self.gff(img_feat, cli_feat)

        logits = self.classifier(fused)

        return logits, None, img_feat, cli_feat, gate, torch.tensor(0.0)


# ============================================================================
# MedPatch: confidence-guided multi-stage fusion
# ============================================================================
class ConfidenceGuidedFusion(nn.Module):
    """MedPatch: Confidence-Guided Multi-Stage Fusion.

    Reference
    ---------
    MedPatch: Confidence-Guided Multi-Stage Fusion for Multimodal Clinical Data.
    Official code: https://github.com/nyuad-cai/MedPatch
    """
    def __init__(self, feat_dim: int, n_classes: int = 2):
        super().__init__()

        self.joint_fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(feat_dim, feat_dim)
        )
        self.joint_classifier = nn.Linear(feat_dim, n_classes)

        self.img_classifier = nn.Linear(feat_dim, n_classes)
        self.cli_classifier = nn.Linear(feat_dim, n_classes)

        self.img_confidence = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.cli_confidence = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

        # Initialize confidence near 0.5.
        nn.init.constant_(self.img_confidence[-2].bias, 0.0)
        nn.init.constant_(self.cli_confidence[-2].bias, 0.0)

    def forward(self, img_feat: torch.Tensor, cli_feat: torch.Tensor):
        img_conf = self.img_confidence(img_feat)
        cli_conf = self.cli_confidence(cli_feat)

        total_conf = img_conf + cli_conf + 1e-8
        img_conf_norm = img_conf / total_conf
        cli_conf_norm = cli_conf / total_conf

        concat = torch.cat([img_feat, cli_feat], dim=1)
        joint_feat = self.joint_fusion(concat)
        joint_logits = self.joint_classifier(joint_feat)

        img_logits = self.img_classifier(img_feat)
        cli_logits = self.cli_classifier(cli_feat)
        late_logits = img_conf_norm * img_logits + cli_conf_norm * cli_logits

        fusion_weight = torch.sigmoid(img_conf * cli_conf * 5)
        logits = fusion_weight * late_logits + (1 - fusion_weight) * joint_logits

        return logits, (img_conf, cli_conf)


class MedPatchFusionModel(nn.Module):
    """MedPatch multi-stage fusion with CLAM attention pooling."""

    def __init__(self, clinical_input_dim, common_dim=64, n_classes=2,
                 fusion_hidden=128, seed=2026, **kwargs):
        super().__init__()
        torch.manual_seed(seed)

        self.clam_pooling = CLAMAttentionPooling()

        self.img_encoder = nn.Sequential(
            nn.Linear(PATCH_FEAT_DIM, fusion_hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(fusion_hidden, common_dim)
        )

        self.cli_encoder = nn.Sequential(
            nn.Linear(clinical_input_dim, fusion_hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(fusion_hidden, common_dim)
        )

        self.medpatch_fusion = ConfidenceGuidedFusion(
            feat_dim=common_dim, n_classes=n_classes
        )

    def forward(self, patches, clinical_features=None, masks=None,
                mixup_lam=None, mixup_perm=None):
        if isinstance(patches, list):
            pooled, _ = self.clam_pooling(patches)
        else:
            pooled, _ = self.clam_pooling(patches, masks)

        img_feat = self.img_encoder(pooled)
        cli_feat = self.cli_encoder(clinical_features)

        logits, confidence = self.medpatch_fusion(img_feat, cli_feat)

        return logits, None, img_feat, cli_feat, confidence, torch.tensor(0.0)


# ============================================================================
# Baseline models
# ============================================================================
class Baseline4_ImageBranchAOnly(nn.Module):
    """Image patch features only, with CLAM attention pooling."""

    def __init__(self, patch_dim=None, fusion_hidden=FUSION_HIDDEN_DIM,
                 n_classes=N_CLASSES, seed=SEED, **kwargs):
        super().__init__()
        set_seed(seed)
        self.clam_pooling = CLAMAttentionPooling()
        self.net = nn.Sequential(
            nn.Linear(PATCH_FEAT_DIM, fusion_hidden), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(fusion_hidden, fusion_hidden // 2), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(fusion_hidden // 2, n_classes)
        )

    def forward(self, patches, clinical_features=None, masks=None,
                mixup_lam=None, mixup_perm=None):
        if isinstance(patches, list):
            pooled, _ = self.clam_pooling(patches)
        else:
            pooled, _ = self.clam_pooling(patches, masks)
        if mixup_lam is not None and mixup_perm is not None and self.training:
            pooled = mixup_lam * pooled + (1 - mixup_lam) * pooled[mixup_perm]
        return self.net(pooled), [], None, None, None, torch.tensor(0.0)


class Baseline_ClinicalOnlyMLP(nn.Module):
    """Clinical-only MLP for a fair NN comparison with the image-only MLP."""

    def __init__(self, stat_dim=None, patch_dim=None,
                 clinical_input_dim=None, n_classes=N_CLASSES, seed=SEED, **kwargs):
        super().__init__()
        set_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(clinical_input_dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, patches=None, clinical_features=None, **kwargs):
        return self.net(clinical_features), [], None, None, None, torch.tensor(0.0)


# ============================================================================
# Contrastive multimodal fusion with improved modality dropout
# ============================================================================
class ImprovedModalityDropout(nn.Module):
    """MICCAI 2025: Contrastive Multimodal Fusion with Improved Modality Dropout.

    Reference
    ---------
    Official code: https://github.com/omron-sinicx/medical-modality-dropout

    Core mechanism
    --------------
    1. During training, randomly drop an entire modality with probability p.
    2. Single-modality encoders plus a multimodal fusion encoder.
    3. Contrastive learning aligns single-modality and multimodal representations.
    """
    def __init__(self, feat_dim: int, dropout_prob: float = 0.2,
                 temperature: float = 0.07):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.temperature = temperature

        self.img_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim)
        )
        self.cli_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim)
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim)
        )

        self.fusion_encoder = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, img_feat: torch.Tensor, cli_feat: torch.Tensor,
                training: bool = True):
        B, D = img_feat.shape
        device = img_feat.device

        if training:
            img_mask = torch.rand(B, 1, device=device) > self.dropout_prob
            cli_mask = torch.rand(B, 1, device=device) > self.dropout_prob
            both_dropped = ~(img_mask | cli_mask)
            img_mask = img_mask | both_dropped
            cli_mask = cli_mask | both_dropped
        else:
            img_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
            cli_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        img_feat_dropped = img_feat * img_mask.float()
        cli_feat_dropped = cli_feat * cli_mask.float()

        concat = torch.cat([img_feat_dropped, cli_feat_dropped], dim=1)
        fusion_feat = self.fusion_encoder(concat)

        if training:
            img_proj = self.img_proj(img_feat)
            cli_proj = self.cli_proj(cli_feat)
            fusion_proj = self.fusion_proj(fusion_feat)
            loss_img = self._contrastive_loss(fusion_proj, img_proj)
            loss_cli = self._contrastive_loss(fusion_proj, cli_proj)
            contrast_loss = (loss_img + loss_cli) / 2
        else:
            contrast_loss = torch.tensor(0.0, device=device)

        return fusion_feat, contrast_loss, (img_mask, cli_mask)

    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """InfoNCE contrastive loss."""
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim = torch.mm(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(sim, labels)


class ContrastiveFusionModel(nn.Module):
    """Contrastive fusion model with improved modality dropout."""

    def __init__(self, clinical_input_dim, common_dim=64, n_classes=2,
                 fusion_hidden=128, dropout_prob=0.2, seed=2026, **kwargs):
        super().__init__()
        torch.manual_seed(seed)

        self.clam_pooling = CLAMAttentionPooling()

        self.img_encoder = nn.Sequential(
            nn.Linear(PATCH_FEAT_DIM, fusion_hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(fusion_hidden, common_dim)
        )

        self.cli_encoder = nn.Sequential(
            nn.Linear(clinical_input_dim, fusion_hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(fusion_hidden, common_dim)
        )

        self.modality_dropout = ImprovedModalityDropout(
            feat_dim=common_dim, dropout_prob=dropout_prob
        )

        self.classifier = nn.Sequential(
            nn.Linear(common_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, patches, clinical_features=None, masks=None,
                mixup_lam=None, mixup_perm=None):
        if isinstance(patches, list):
            pooled, _ = self.clam_pooling(patches)
        else:
            pooled, _ = self.clam_pooling(patches, masks)

        img_feat = self.img_encoder(pooled)
        cli_feat = self.cli_encoder(clinical_features)

        fused, contrast_loss, _ = self.modality_dropout(
            img_feat, cli_feat, training=self.training
        )

        logits = self.classifier(fused)

        return logits, None, img_feat, cli_feat, None, contrast_loss


# ============================================================================
# Dataset and collate
# ============================================================================
class CaseDataset(Dataset):
    def __init__(self, case_ids, patch_list, clinical_features, labels):
        self.case_ids = case_ids
        self.patch_list = [torch.FloatTensor(p) for p in patch_list]
        self.clinical_features = [torch.FloatTensor(f) for f in clinical_features]
        self.labels = [torch.LongTensor([l]) for l in labels]

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        return {
            'case_id': self.case_ids[idx],
            'patches': self.patch_list[idx],
            'clinical': self.clinical_features[idx],
            'label': self.labels[idx].item()
        }


def balanced_collate_fn(batch):
    patches = [b['patches'] for b in batch]
    clinical = torch.stack([b['clinical'] for b in batch])
    labels = torch.LongTensor([b['label'] for b in batch])
    case_ids = [b['case_id'] for b in batch]
    return {'case_id': case_ids, 'patches': patches,
            'clinical': clinical, 'label': labels}


def padded_collate_fn(batch):
    """Pad variable-length patch sequences to a fixed shape for DataParallel."""
    clinical = torch.stack([b['clinical'] for b in batch])
    labels = torch.LongTensor([b['label'] for b in batch])
    case_ids = [b['case_id'] for b in batch]

    patch_list = [b['patches'] for b in batch]
    lengths = torch.LongTensor([p.shape[0] for p in patch_list])
    max_len = lengths.max().item()

    if max_len == 0:
        patches_padded = torch.zeros(len(batch), 1, patch_list[0].shape[1])
        masks = torch.zeros(len(batch), 1, dtype=torch.bool)
    else:
        patches_padded = torch.zeros(len(batch), max_len, patch_list[0].shape[1])
        for i, p in enumerate(patch_list):
            patches_padded[i, :p.shape[0]] = p
        masks = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        'case_id': case_ids,
        'patches_padded': patches_padded, 'masks': masks,
        'patches_raw': patch_list,
        'clinical': clinical, 'label': labels
    }


class MultiGPUModelWrapper(nn.Module):
    """Wrap the base model so DataParallel can scatter/gather variable-length inputs."""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, patches, clinical_features, masks=None, **kwargs):
        output = self.base_model(patches=patches,
                                 clinical_features=clinical_features, masks=masks)
        logits, attentions, img_code, cli_code, gate = output[:5]
        extra = output[5:] if len(output) > 5 else []
        device = logits.device
        extra = [e.to(device) if isinstance(e, torch.Tensor) else e for e in extra]
        return (logits, attentions, img_code, cli_code, gate, *extra)


COLLATE_FN = padded_collate_fn if USE_MULTI_GPU else balanced_collate_fn


def wrap_model_for_multi_gpu(model):
    """Wrap the model for DataParallel, handling variable-length inputs."""
    if USE_MULTI_GPU:
        model = MultiGPUModelWrapper(model)
        model = nn.DataParallel(model)
    return model


def build_criterion(device=None):
    if device is None:
        device = DEVICE
    return BCEWithLogitsLossWrapper(pos_weight=None)


def compute_metrics(y_true, y_pred_prob, y_pred_class):
    metrics = {}
    prob_pos = y_pred_prob[:, 1] if y_pred_prob.ndim > 1 else y_pred_prob
    metrics['accuracy'] = accuracy_score(y_true, y_pred_class)
    metrics['auc'] = roc_auc_score(y_true, prob_pos)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics['f1'] = f1_score(y_true, y_pred_class)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred_class)
    return metrics


def _is_padded_batch(batch):
    return 'patches_padded' in batch


def _get_model_inputs(batch, device):
    """Unify input handling for single-GPU and multi-GPU modes."""
    clinical = batch['clinical'].to(device)
    labels = batch['label'].to(device)
    if _is_padded_batch(batch):
        return clinical, labels, batch['patches_padded'].to(device), batch['masks'].to(device)
    else:
        patches = [p.to(device) for p in batch['patches']]
        return clinical, labels, patches, None


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in dataloader:
        clinical, labels, patches_or_padded, masks = _get_model_inputs(batch, device)
        optimizer.zero_grad()

        model_inner = model.module if isinstance(model, nn.DataParallel) else model
        use_mixup = MIXUP_ALPHA > 0 and isinstance(model_inner, Baseline4_ImageBranchAOnly)
        if use_mixup and _is_padded_batch(batch):
            use_mixup = False
        if use_mixup:
            lam_mix = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
            lam_mix = max(lam_mix, 1 - lam_mix)
            perm = torch.randperm(labels.size(0), device=device)
            mixed_labels = lam_mix * labels.float() + (1 - lam_mix) * labels[perm].float()

            output = model(
                patches=patches_or_padded, clinical_features=clinical,
                masks=masks, mixup_lam=lam_mix, mixup_perm=perm
            )
            logits, attentions, img_code, cli_code, gate = output[:5]
            contrast_loss = output[5] if len(output) > 5 else torch.tensor(0.0, device=device)
            loss_cls = criterion(logits, mixed_labels)
        else:
            output = model(patches=patches_or_padded, clinical_features=clinical, masks=masks)
            logits, attentions, img_code, cli_code, gate = output[:5]
            contrast_loss = output[5] if len(output) > 5 else torch.tensor(0.0, device=device)
            loss_cls = criterion(logits, labels)

        if img_code is not None and cli_code is not None and MODAL_ALIGN_LAMBDA > 0:
            cos_sim = F.cosine_similarity(img_code, cli_code, dim=1)
            loss_align = torch.mean(1 - cos_sim) * MODAL_ALIGN_LAMBDA
        else:
            loss_align = torch.tensor(0.0, device=device)

        loss_attn = torch.tensor(0.0, device=device)

        if isinstance(contrast_loss, torch.Tensor) and contrast_loss.dim() > 0:
            contrast_loss = contrast_loss.mean()
        loss_contrast = contrast_loss * CONTRAST_LOSS_WEIGHT

        loss = loss_cls + loss_align + loss_attn + loss_contrast
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.append(logits.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    return avg_loss, all_preds, all_labels


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_gates = [], [], []
    for batch in dataloader:
        clinical, labels, patches_or_padded, masks = _get_model_inputs(batch, device)
        output = model(patches=patches_or_padded, clinical_features=clinical, masks=masks)
        logits, _, img_code, cli_code, gate = output[:5]
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_preds.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        if gate is not None:
            if isinstance(gate, tuple):
                g = torch.cat(gate, dim=1).cpu().numpy()
            else:
                g = gate.cpu().numpy()
            all_gates.append(np.atleast_1d(g))

    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    logits_np = np.asarray(all_preds)
    prob_pos = 1.0 / (1.0 + np.exp(-logits_np[:, 1]))
    pred_prob = np.column_stack([1.0 - prob_pos, prob_pos])
    pred_class = prob_to_class(prob_pos)
    gate_vals = np.concatenate(all_gates).flatten() if all_gates else np.array([])

    metrics = compute_metrics(all_labels, pred_prob, pred_class)
    metrics['loss'] = total_loss / len(dataloader)
    metrics['mean_gate'] = float(np.mean(gate_vals)) if len(gate_vals) > 0 else 0.0
    return metrics, pred_prob, pred_class, gate_vals


def _make_inner_split_loaders(train_loader, inner_val_ratio=0.2, seed=SEED):
    """Split the training fold into an inner 80/20 split for early stopping.

    The external validation fold must not be used for early stopping, as that
    would cause information leakage.
    """
    ds = train_loader.dataset
    n = len(ds)
    y = np.array([int(ds.labels[i].item()) for i in range(n)])
    idx = np.arange(n)
    tr_idx, va_idx = train_test_split(
        idx, test_size=inner_val_ratio, stratify=y, random_state=seed
    )
    inner_train_ds = torch.utils.data.Subset(ds, tr_idx)
    inner_val_ds = torch.utils.data.Subset(ds, va_idx)
    collate = train_loader.collate_fn
    bs = train_loader.batch_size
    inner_train_loader = DataLoader(inner_train_ds, batch_size=bs, shuffle=True,
                                    collate_fn=collate,
                                    generator=torch.Generator().manual_seed(seed))
    inner_val_loader = DataLoader(inner_val_ds, batch_size=bs, shuffle=False,
                                  collate_fn=collate)
    return inner_train_loader, inner_val_loader


def train_model(model, train_loader, val_loader=None, device=DEVICE,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR,
                weight_decay=WEIGHT_DECAY, fold_idx=None, inner_val_ratio=0.2,
                use_inner_split=True):
    """Unified neural-network training and model selection.

    For CV folds, an inner 80/20 split of the training fold is used for early
    stopping (``use_inner_split=True``); the outer validation fold is used
    only for final evaluation.

    For the final model, if an 80/20 split has already been prepared outside,
    pass ``use_inner_split=False`` and provide ``val_loader`` directly.
    """
    criterion = build_criterion(device)
    seed = SEED if fold_idx is None else SEED + fold_idx
    if use_inner_split and val_loader is not None:
        train_loader, val_loader = _make_inner_split_loaders(
            train_loader, inner_val_ratio, seed
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_auc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    desc = f"    Fold {fold_idx} train" if fold_idx is not None else "    Train"

    for epoch in tqdm(range(max_epochs), desc=desc, unit="epoch", leave=False):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        val_auc = val_metrics['auc']

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_epoch, best_auc


# ============================================================================
# Image-only linear LR: single-fold training and evaluation
# ============================================================================
def select_lr_best_c(X_train, y_train, C_grid=C_GRID, n_inner=STAT_INNER_CV_FOLDS,
                     seed=SEED):
    """Inner n_inner-fold CV inside the training fold, selecting C by AUC."""
    rng = np.random.RandomState(seed)
    score_by_c = {}
    for C in C_grid:
        scores = []
        try:
            skf = StratifiedKFold(n_splits=n_inner, shuffle=True,
                                  random_state=int(rng.randint(0, 10**6)))
        except Exception:
            skf = StratifiedKFold(n_splits=max(
                2, min(n_inner, len(np.unique(y_train))))
                if len(np.unique(y_train)) > 1 else 2)
        for tr_local, va_local in skf.split(X_train, y_train):
            sc_local = StandardScaler().fit(X_train[tr_local])
            m = LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=5000,
                                   random_state=int(rng.randint(0, 10**6)),
                                   class_weight=None)
            m.fit(sc_local.transform(X_train[tr_local]), y_train[tr_local])
            p = m.predict_proba(sc_local.transform(X_train[va_local]))[:, 1]
            score = roc_auc_score(y_train[va_local], p)
            scores.append(score)
        score_by_c[C] = float(np.mean(scores)) if scores else -1.0
    return max(score_by_c, key=score_by_c.get)


def search_model_threshold(y_true, y_prob):
    """Per-model threshold optimization on its own OOF predictions.

    Within a small tolerance of the maximum Youden index, choose the threshold
    with the highest specificity, favoring correct identification of class A.
    Returns ``DEFAULT_THRESHOLD`` when threshold optimization is disabled or
    when the positive class is underrepresented.
    """
    if not ENABLE_THRESHOLD_OPTIMIZATION:
        return DEFAULT_THRESHOLD
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob)
    if len(np.unique(yt)) < 2:
        return DEFAULT_THRESHOLD
    return float(search_optimal_threshold(yt, yp)['threshold'])


def _train_eval_lr_fold(fold_info, case_ids, patch_list_all, binary_labels,
                        fold_idx, output_dir):
    """Train LogisticRegression on image mean-pooled features for a single fold."""
    import pickle as pkl
    summary_path = os.path.join(output_dir, 'folds', f'fold_summary_lr_{fold_idx}.pkl')
    if os.path.exists(summary_path):
        with open(summary_path, 'rb') as f:
            cached = pkl.load(f)
        if cached.get('leak_fix_version', 1) >= LR_LEAK_FIX_VERSION:
            return cached
        print(f"  Fold {fold_idx} cache predates the leak fix; recomputing ...")

    case_to_idx = {cid: i for i, cid in enumerate(case_ids)}

    train_idx = [case_to_idx[c] for c in fold_info['train_cases']]
    val_idx = [case_to_idx[c] for c in fold_info['val_cases']]

    train_patches = [patch_list_all[i] for i in train_idx]
    val_patches = [patch_list_all[i] for i in val_idx]
    X_train = compute_mean_pooled_features(train_patches)
    X_val = compute_mean_pooled_features(val_patches)
    y_train = binary_labels[train_idx]
    y_val = binary_labels[val_idx]

    # Fix the double-standardization leak: pass raw data to select_lr_best_c,
    # which standardizes each inner fold independently. Once C is chosen,
    # fit the final scaler and LR on the whole training fold.
    best_c = select_lr_best_c(X_train, y_train, C_GRID,
                              STAT_INNER_CV_FOLDS, seed=SEED + fold_idx)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lr = LogisticRegression(penalty='l1', solver='saga', C=best_c, max_iter=5000,
                            random_state=SEED, class_weight=None)
    lr.fit(X_train_scaled, y_train)

    y_prob = lr.predict_proba(X_val_scaled)
    y_pred = prob_to_class(y_prob[:, 1])

    auc = roc_auc_score(y_val, y_prob[:, 1])
    acc = accuracy_score(y_val, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    pkl.dump(scaler, open(os.path.join(output_dir, 'folds',
                                        f'lr_scaler_fold_{fold_idx}.pkl'), 'wb'))
    pkl.dump(lr, open(os.path.join(output_dir, 'folds',
                                    f'lr_model_fold_{fold_idx}.pkl'), 'wb'))

    val_cases = fold_info['val_cases']
    pd.DataFrame({
        'case_id': val_cases,
        'true_label': [POSITIVE_CLASS if y == 1 else NEGATIVE_CLASS for y in y_val],
        'pred_label': [POSITIVE_CLASS if p == 1 else NEGATIVE_CLASS for p in y_pred],
        'prob_A': y_prob[:, 0],
        'prob_B': y_prob[:, 1],
    }).to_csv(os.path.join(output_dir, 'predictions',
                            f'val_prediction_lr_fold_{fold_idx}.csv'), index=False)

    summary = {
        'fold_idx': fold_idx, 'auc': auc, 'accuracy': acc,
        'sensitivity': sens, 'specificity': spec, 'f1': f1, 'auc_gap': 0.0,
        'leak_fix_version': LR_LEAK_FIX_VERSION,
    }

    pkl.dump(summary, open(summary_path, 'wb'))
    print(f"  Fold {fold_idx} LR result: Val AUC={auc:.4f} | "
          f"Sens={sens:.4f} | Spec={spec:.4f}")
    return summary


# ============================================================================
# Post-processing: threshold optimization
# ============================================================================
def search_optimal_threshold(y_true, y_prob, youden_tol=YOUDEN_TOLERANCE):
    """Search the optimal threshold.

    Within a tolerance of the maximum Youden index, choose the threshold with
    the highest specificity on the model's own OOF predictions, favoring
    correct identification of class A.

    Procedure
    ---------
    1. Sweep a grid of thresholds from 0.01 to 0.99 (step 0.001), computing the
       Youden index (sensitivity + specificity - 1) at each.
    2. Keep thresholds whose Youden index is within ``youden_tol`` of the
       maximum.
    3. Among the candidates, choose the threshold with the highest specificity;
       ties are broken by the larger threshold (more conservative, fewer false
       positives to class B).
    """
    thresholds = np.arange(0.01, 0.99, 0.001)
    records = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        records.append((thr, sens, spec, sens + spec - 1.0))
    max_youden = max(r[3] for r in records)
    candidates = [r for r in records if r[3] >= max_youden - youden_tol]
    best = max(candidates, key=lambda r: (r[2], r[0]))
    return {'threshold': best[0], 'sensitivity': best[1], 'specificity': best[2],
            'youden': best[3], 'max_youden': max_youden, 'youden_tol': youden_tol}


# ============================================================================
# Decision curve analysis (DCA)
# ============================================================================
def calculate_dca(y_true, y_prob, thresholds=np.arange(0.01, 1.00, 0.01)):
    n = len(y_true)
    net_benefits = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits.append(nb)
    return thresholds, np.array(net_benefits)


# ============================================================================
# Clinical LR baseline (used by both run_baselines_cv and cached refresh)
# ============================================================================
def run_clinical_lr_baseline(folds, patch_list_all, clinical_df, categories_all,
                             case_ids, folds_dir, output_dir):
    """Clinical-only LASSO LR baseline: 10-fold OOF training + per-model
    threshold optimization.

    Preprocessing mirrors the main LR pipeline. The double-standardization
    leak is fixed by passing raw features to ``select_lr_best_c`` (each inner
    fold standardizes independently), then fitting a final scaler on the
    whole training fold before training the final LR.
    """
    case_to_idx = {cid: i for i, cid in enumerate(case_ids)}
    print(f"\n  --- Clinical only (LASSO LogisticRegression) ---")
    fold_results_clin = []
    oof_y_true_clin = []
    oof_y_prob_clin = []
    oof_case_ids_clin = []

    for fold_info in tqdm(folds, desc="    Clinical only (LASSO LR)", unit="fold"):
        fold_idx = fold_info['fold']
        train_cases = fold_info['train_cases']
        val_cases = fold_info['val_cases']
        train_indices = [case_to_idx[c] for c in train_cases if c in case_to_idx]
        val_indices = [case_to_idx[c] for c in val_cases if c in case_to_idx]
        y_train_bin = to_binary_labels(categories_all[train_indices])
        y_val_bin = to_binary_labels(categories_all[val_indices])

        try:
            clin_pre = ClinicalPreprocessor().load(folds_dir, fold_idx)
        except FileNotFoundError:
            train_clin_df_tmp2 = clinical_df[
                clinical_df['case_id'].isin(train_cases)
            ].reset_index(drop=True)
            clin_pre = ClinicalPreprocessor()
            clin_pre.fit(train_clin_df_tmp2)
            clin_pre.save(folds_dir, fold_idx)
        train_clin_df = clinical_df[
            clinical_df['case_id'].isin(train_cases)
        ].reset_index(drop=True)
        val_clin_df = clinical_df[
            clinical_df['case_id'].isin(val_cases)
        ].reset_index(drop=True)
        train_clin_map = dict(zip(train_clin_df['case_id'],
                                  clin_pre.transform(train_clin_df)))
        val_clin_map = dict(zip(val_clin_df['case_id'],
                                clin_pre.transform(val_clin_df)))
        X_train_clin = np.array([train_clin_map[c] for c in train_cases
                                 if c in train_clin_map])
        X_val_clin = np.array([val_clin_map[c] for c in val_cases
                               if c in val_clin_map])

        lasso_sel = LassoClinicalSelector(enable=ENABLE_LASSO_SELECTION)
        X_train_clin_sel = lasso_sel.fit_transform(X_train_clin, y_train_bin)
        X_val_clin_sel = lasso_sel.transform(X_val_clin)

        # Fix the double-standardization leak: pick C using raw features
        # (inner folds standardize independently), then fit the final scaler.
        best_c_clin = select_lr_best_c(X_train_clin_sel, y_train_bin, C_GRID,
                                       STAT_INNER_CV_FOLDS, seed=SEED + fold_idx)
        clin_scaler = StandardScaler().fit(X_train_clin_sel)
        X_train_clin_sel = clin_scaler.transform(X_train_clin_sel)
        X_val_clin_sel = clin_scaler.transform(X_val_clin_sel)

        lr_model = LogisticRegression(
            penalty='l1', C=best_c_clin, solver='saga',
            max_iter=LASSO_CV_MAX_ITER, random_state=SEED, class_weight=None
        )
        lr_model.fit(X_train_clin_sel, y_train_bin)
        y_val_prob = lr_model.predict_proba(X_val_clin_sel)
        val_auc = roc_auc_score(y_val_bin, y_val_prob[:, 1])

        fold_results_clin.append({
            'fold': fold_idx, 'val_auc': val_auc,
            'val_prob_pos': y_val_prob[:, 1], 'y_val': np.array(y_val_bin),
        })
        oof_y_true_clin.extend(y_val_bin)
        oof_y_prob_clin.extend(y_val_prob[:, 1])
        oof_case_ids_clin.extend(val_cases)

    thr_clin = search_model_threshold(oof_y_true_clin, oof_y_prob_clin)
    for fr in fold_results_clin:
        yc_fr = prob_to_class(fr['val_prob_pos'], thr_clin)
        yt_fr = fr['y_val']
        tn_fr, fp_fr, fn_fr, tp_fr = confusion_matrix(yt_fr, yc_fr).ravel()
        fr['sensitivity'] = tp_fr / (tp_fr + fn_fr) if (tp_fr + fn_fr) > 0 else 0.0
        fr['specificity'] = tn_fr / (tn_fr + fp_fr) if (tn_fr + fp_fr) > 0 else 0.0
        fr['val_f1'] = f1_score(yt_fr, yc_fr)

    mean_auc_clin = np.mean([r['val_auc'] for r in fold_results_clin])
    print(f"    Mean Val AUC: {mean_auc_clin:.4f} | "
          f"Optimal threshold: {thr_clin:.4f}")

    all_results_clin = {
        'baseline_clinical_only': {
            'display_name': 'Clinical only (LASSO LR)',
            'fold_results': fold_results_clin,
            'mean_auc': mean_auc_clin, 'threshold': thr_clin
        }
    }
    oof_preds_clin = {
        'baseline_clinical_only': {
            'case_ids': oof_case_ids_clin,
            'y_true': np.array(oof_y_true_clin),
            'y_prob': np.array(oof_y_prob_clin),
            'display_name': 'Clinical only (LASSO LR)',
            'threshold': thr_clin,
        }
    }
    oof_ci_results_clin = {
        'baseline_clinical_only': compute_oof_metrics_with_ci(
            oof_preds_clin['baseline_clinical_only']['y_true'],
            oof_preds_clin['baseline_clinical_only']['y_prob'])
    }
    return all_results_clin, oof_preds_clin, oof_ci_results_clin


# ============================================================================
# Baseline models: 10-fold cross-validation
# ============================================================================
def run_baselines_cv(folds, patch_list_all, clinical_df,
                     categories_all, case_ids, output_dir):
    print(f"\n{'=' * 60}")
    print(f"  Baseline models - 10-fold cross-validation")
    print(f"{'=' * 60}")

    case_to_idx = {cid: i for i, cid in enumerate(case_ids)}
    baselines = {
        'baseline_stat_only': (
            Baseline4_ImageBranchAOnly,
            'Image patch features only (MLP)'
        ),
        'baseline_thyfusionnet': (
            ThyFusionNetFusionModel,
            'Clinical + patch ThyFusionNet gated fusion'
        ),
        'baseline_clinical_mlp': (
            Baseline_ClinicalOnlyMLP,
            'Clinical only (MLP)'
        ),
        'baseline_contrastive_fusion': (
            ContrastiveFusionModel,
            'Clinical + patch contrastive fusion'
        ),
        'baseline_medpatch_fusion': (
            MedPatchFusionModel,
            'Clinical + patch MedPatch confidence fusion'
        ),
    }
    folds_dir = os.path.join(output_dir, 'folds')
    all_results = {}
    oof_preds = {}
    oof_ci_results = {}

    for key, (ModelClass, display_name) in baselines.items():
        print(f"\n  --- {display_name} ---")
        fold_results = []
        oof_y_true = []
        oof_y_prob = []
        oof_case_ids = []

        for fold_info in tqdm(folds, desc=f"    {display_name}", unit="fold"):
            fold_idx = fold_info['fold']
            train_cases = fold_info['train_cases']
            val_cases = fold_info['val_cases']
            train_indices = [case_to_idx[c] for c in train_cases
                             if c in case_to_idx]
            val_indices = [case_to_idx[c] for c in val_cases
                           if c in case_to_idx]
            y_train_bin = to_binary_labels(categories_all[train_indices])
            y_val_bin = to_binary_labels(categories_all[val_indices])

            try:
                clin_pre = ClinicalPreprocessor().load(folds_dir, fold_idx)
            except FileNotFoundError:
                train_clin_df_tmp = clinical_df[
                    clinical_df['case_id'].isin(train_cases)
                ].reset_index(drop=True)
                clin_pre = ClinicalPreprocessor()
                clin_pre.fit(train_clin_df_tmp)
                clin_pre.save(folds_dir, fold_idx)
            clinical_dim = clin_pre.output_dim

            train_patches = [patch_list_all[i] for i in train_indices]
            val_patches = [patch_list_all[i] for i in val_indices]

            train_clin_df = clinical_df[
                clinical_df['case_id'].isin(train_cases)
            ].reset_index(drop=True)
            val_clin_df = clinical_df[
                clinical_df['case_id'].isin(val_cases)
            ].reset_index(drop=True)
            train_clin_map = dict(zip(train_clin_df['case_id'],
                                      clin_pre.transform(train_clin_df)))
            val_clin_map = dict(zip(val_clin_df['case_id'],
                                    clin_pre.transform(val_clin_df)))
            X_train_clin = np.array([train_clin_map[c] for c in train_cases
                                     if c in train_clin_map])
            X_val_clin = np.array([val_clin_map[c] for c in val_cases
                                   if c in val_clin_map])

            lasso_sel = LassoClinicalSelector(enable=ENABLE_LASSO_SELECTION)
            X_train_clin = lasso_sel.fit_transform(X_train_clin, y_train_bin)
            X_val_clin = lasso_sel.transform(X_val_clin)
            clinical_dim = lasso_sel.selected_dim

            train_dataset = CaseDataset(train_cases, train_patches,
                                        X_train_clin, y_train_bin)
            val_dataset = CaseDataset(val_cases, val_patches,
                                      X_val_clin, y_val_bin)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                      shuffle=True, collate_fn=COLLATE_FN,
                                      generator=torch.Generator().manual_seed(SEED))
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                    shuffle=False, collate_fn=COLLATE_FN)

            model = ModelClass(clinical_input_dim=clinical_dim, seed=SEED).to(DEVICE)
            if USE_MULTI_GPU:
                model = wrap_model_for_multi_gpu(model)
            model, best_epoch, best_auc = train_model(model, train_loader,
                                                     val_loader, DEVICE,
                                                     fold_idx=fold_idx)
            val_metrics, val_prob, val_class, _ = evaluate(
                model, val_loader, build_criterion(), DEVICE)

            fold_results.append({
                'fold': fold_idx, 'val_auc': val_metrics['auc'],
                'val_f1': val_metrics['f1'],
                'sensitivity': val_metrics['sensitivity'],
                'specificity': val_metrics['specificity'],
                'val_prob_pos': val_prob[:, 1], 'y_val': np.array(y_val_bin)
            })
            oof_y_true.extend(y_val_bin)
            oof_y_prob.extend(val_prob[:, 1])
            oof_case_ids.extend(val_cases)

        thr_model = search_model_threshold(oof_y_true, oof_y_prob)
        for fr in fold_results:
            yc_fr = prob_to_class(fr['val_prob_pos'], thr_model)
            yt_fr = fr['y_val']
            tn_fr, fp_fr, fn_fr, tp_fr = confusion_matrix(yt_fr, yc_fr).ravel()
            fr['sensitivity'] = tp_fr / (tp_fr + fn_fr) if (tp_fr + fn_fr) > 0 else 0.0
            fr['specificity'] = tn_fr / (tn_fr + fp_fr) if (tn_fr + fp_fr) > 0 else 0.0
            fr['val_f1'] = f1_score(yt_fr, yc_fr)

        mean_auc = np.mean([r['val_auc'] for r in fold_results])
        print(f"    Mean Val AUC: {mean_auc:.4f} | "
              f"Optimal threshold: {thr_model:.4f}")
        all_results[key] = {'display_name': display_name,
                            'fold_results': fold_results,
                            'mean_auc': mean_auc, 'threshold': thr_model}
        oof_preds[key] = {
            'case_ids': oof_case_ids,
            'y_true': np.array(oof_y_true),
            'y_prob': np.array(oof_y_prob),
            'display_name': display_name,
            'threshold': thr_model,
        }
        oof_ci_results[key] = compute_oof_metrics_with_ci(
            oof_preds[key]['y_true'], oof_preds[key]['y_prob'])

    clin_results, clin_oof, clin_ci = run_clinical_lr_baseline(
        folds, patch_list_all, clinical_df, categories_all, case_ids,
        folds_dir, output_dir)
    all_results.update(clin_results)
    oof_preds.update(clin_oof)
    oof_ci_results.update(clin_ci)

    with open(os.path.join(output_dir, 'baselines', 'baselines_cv.pkl'), 'wb') as f:
        pickle.dump({'results': all_results, 'oof_preds': oof_preds,
                     'oof_ci_results': oof_ci_results}, f)
    return all_results, oof_preds, oof_ci_results


# ============================================================================
# Comprehensive Excel summary
# ============================================================================
def _generate_comprehensive_summary(fusion_oof, ext_state, baseline_cv_results,
                                     baseline_oof_preds,
                                     baseline_oof_ci_results=None,
                                     main_oof_metrics_ci=None,
                                     output_dir=None):
    """Generate the comprehensive comparison Excel (internal CV + external test)."""
    import openpyxl

    def _fmt_ci(ci_tuple):
        if ci_tuple is None or len(ci_tuple) < 3:
            return ''
        return f'{round_to_4sf(ci_tuple[1])}-{round_to_4sf(ci_tuple[2])}'

    def _build_oof_row(name, oof_pred, ci_results=None):
        oof_prob = np.array(oof_pred['y_prob'])
        oof_auc = roc_auc_score(oof_pred['y_true'], oof_prob)
        oof_m = compute_metrics(
            oof_pred['y_true'],
            np.column_stack([1 - oof_prob, oof_prob]),
            prob_to_class(oof_prob, threshold=oof_pred.get('threshold'))
        )
        row = {
            'Model': name, 'Source': 'Internal OOF',
            'AUC': round_to_4sf(oof_auc),
            'AUC_95CI': '',
            'Sensitivity': round_to_4sf(oof_m['sensitivity']),
            'Specificity': round_to_4sf(oof_m['specificity']),
            'F1': round_to_4sf(oof_m['f1']),
            'Accuracy': round_to_4sf(oof_m['accuracy']),
            'PPV': round_to_4sf(oof_m['ppv']),
            'NPV': round_to_4sf(oof_m['npv']),
        }
        if ci_results is not None:
            row['AUC_95CI'] = _fmt_ci(ci_results.get('auc_ci'))
            row['Brier'] = round_to_4sf(ci_results.get('brier'))
            row['Brier_95CI'] = _fmt_ci(ci_results.get('brier_ci'))
        else:
            row['AUC_95CI'] = ''
            row['Brier'] = row['Brier_95CI'] = ''
        wilson = compute_oof_wilson_cis(oof_pred['y_true'], oof_pred['y_prob'],
                                        threshold=oof_pred.get('threshold'))
        row['Sensitivity_95CI'] = _fmt_ci(wilson.get('sensitivity'))
        row['Specificity_95CI'] = _fmt_ci(wilson.get('specificity'))
        row['Accuracy_95CI'] = _fmt_ci(wilson.get('accuracy'))
        row['F1_95CI'] = _fmt_ci(wilson.get('f1'))
        row['PPV_95CI'] = _fmt_ci(wilson.get('ppv'))
        row['NPV_95CI'] = _fmt_ci(wilson.get('npv'))
        return row

    def _build_ext_row(name, ext_metrics, ext_auc_ci, ext_brier_ci=None,
                       ext_wilson_cis=None, f1_ci=None):
        row = {
            'Model': name, 'Source': 'External test',
            'AUC': round_to_4sf(ext_auc_ci[0]),
            'Sensitivity': round_to_4sf(ext_metrics['sensitivity']),
            'Specificity': round_to_4sf(ext_metrics['specificity']),
            'F1': round_to_4sf(ext_metrics['f1']),
            'Accuracy': round_to_4sf(ext_metrics['accuracy']),
            'PPV': round_to_4sf(ext_metrics.get('ppv', float('nan'))),
            'NPV': round_to_4sf(ext_metrics.get('npv', float('nan'))),
            'AUC_95CI': _fmt_ci(ext_auc_ci),
            'Sensitivity_95CI': '', 'Specificity_95CI': '', 'Accuracy_95CI': '',
            'F1_95CI': '', 'PPV_95CI': '', 'NPV_95CI': '',
            'Brier': '', 'Brier_95CI': '',
        }
        if ext_brier_ci is not None:
            row['Brier'] = round_to_4sf(ext_brier_ci[0])
            row['Brier_95CI'] = (f'{round_to_4sf(ext_brier_ci[1])}-'
                                 f'{round_to_4sf(ext_brier_ci[2])}')
        if f1_ci is not None:
            row['F1_95CI'] = _fmt_ci(f1_ci)
        if ext_wilson_cis is not None:
            _wilson_col = {'sensitivity': 'Sensitivity', 'specificity': 'Specificity',
                           'ppv': 'PPV', 'npv': 'NPV', 'accuracy': 'Accuracy'}
            for k in ['sensitivity', 'specificity', 'ppv', 'npv', 'accuracy']:
                row[f'{_wilson_col[k]}_95CI'] = _fmt_ci(ext_wilson_cis.get(k))
        return row

    rows_internal = []
    rows_external = []
    main_threshold = fusion_oof.get('threshold')

    # --- Internal validation ---
    cv_csv_path = os.path.join(output_dir, 'predictions', '10fold_cv_summary.csv')
    if os.path.exists(cv_csv_path):
        cv_table = pd.read_csv(cv_csv_path)
        numeric = cv_table.iloc[:-1]
        aucs = numeric['auc_mean'].astype(float)
        fold_metrics = []
        for fi in range(10):
            fp_ = os.path.join(output_dir, 'predictions',
                               f'val_prediction_lr_fold_{fi}.csv')
            if not os.path.exists(fp_):
                continue
            df_ = pd.read_csv(fp_)
            yt_ = (df_['true_label'] == POSITIVE_CLASS).astype(int).to_numpy()
            p1_ = df_['prob_B'].to_numpy()
            yc_ = prob_to_class(p1_, threshold=main_threshold)
            cm_ = confusion_matrix(yt_, yc_).ravel()
            tn_, fp2_, fn_, tp_ = cm_
            fold_metrics.append({
                'sens': tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0,
                'spec': tn_ / (tn_ + fp2_) if (tn_ + fp2_) > 0 else 0.0,
                'f1': 2 * tp_ / (2 * tp_ + fp2_ + fn_) if (2 * tp_ + fp2_ + fn_) > 0 else 0.0,
                'acc': (tp_ + tn_) / (tp_ + tn_ + fp2_ + fn_) if (tp_ + tn_ + fp2_ + fn_) > 0 else 0.0,
            })
        sens = np.mean([m['sens'] for m in fold_metrics])
        specs = np.mean([m['spec'] for m in fold_metrics])
        f1s = np.mean([m['f1'] for m in fold_metrics])
        accs = np.mean([m['acc'] for m in fold_metrics])
        rows_internal.append({
            'Model': 'Image only linear LR (main)',
            'Source': 'Internal 10-fold CV',
            'AUC': round_to_4sf(aucs.mean()),
            'AUC_95CI': '',
            'Sensitivity': round_to_4sf(sens),
            'Specificity': round_to_4sf(specs),
            'F1': round_to_4sf(f1s),
            'Accuracy': round_to_4sf(accs),
            'PPV': float('nan'), 'NPV': float('nan'),
            'Brier': '', 'Brier_95CI': '',
        })

    oof_prob = np.array(fusion_oof['y_prob'])
    oof_auc = roc_auc_score(fusion_oof['y_true'], oof_prob)
    oof_m = compute_metrics(
        fusion_oof['y_true'],
        np.column_stack([1 - oof_prob, oof_prob]),
        prob_to_class(oof_prob, threshold=main_threshold)
    )
    main_oof_row = {
        'Model': 'Image only linear LR (main)',
        'Source': 'Internal OOF',
        'AUC': round_to_4sf(oof_auc),
        'AUC_95CI': '',
        'Sensitivity': round_to_4sf(oof_m['sensitivity']),
        'Specificity': round_to_4sf(oof_m['specificity']),
        'F1': round_to_4sf(oof_m['f1']),
        'Accuracy': round_to_4sf(oof_m['accuracy']),
        'PPV': round_to_4sf(oof_m['ppv']),
        'NPV': round_to_4sf(oof_m['npv']),
    }
    if main_oof_metrics_ci is not None:
        main_oof_row['AUC_95CI'] = _fmt_ci(main_oof_metrics_ci.get('auc_ci'))
        main_oof_row['Brier'] = round_to_4sf(main_oof_metrics_ci.get('brier'))
        main_oof_row['Brier_95CI'] = _fmt_ci(main_oof_metrics_ci.get('brier_ci'))
    else:
        main_oof_row['AUC_95CI'] = ''
        main_oof_row['Brier'] = main_oof_row['Brier_95CI'] = ''
    wilson = compute_oof_wilson_cis(fusion_oof['y_true'], fusion_oof['y_prob'],
                                    threshold=main_threshold)
    main_oof_row['Sensitivity_95CI'] = _fmt_ci(wilson.get('sensitivity'))
    main_oof_row['Specificity_95CI'] = _fmt_ci(wilson.get('specificity'))
    main_oof_row['Accuracy_95CI'] = _fmt_ci(wilson.get('accuracy'))
    main_oof_row['F1_95CI'] = _fmt_ci(wilson.get('f1'))
    main_oof_row['PPV_95CI'] = _fmt_ci(wilson.get('ppv'))
    main_oof_row['NPV_95CI'] = _fmt_ci(wilson.get('npv'))
    rows_internal.append(main_oof_row)

    baseline_order = [
        ('baseline_clinical_only', 'Clinical only LASSO LR (baseline)'),
        ('baseline_stat_only', 'Image patch features only (baseline)'),
        ('baseline_clinical_mlp', 'Clinical only MLP (baseline)'),
        ('baseline_thyfusionnet', 'Clinical + patch ThyFusionNet gated fusion (baseline)'),
        ('baseline_contrastive_fusion', 'Clinical + patch contrastive fusion (baseline)'),
        ('baseline_medpatch_fusion', 'Clinical + patch MedPatch confidence fusion (baseline)'),
    ]
    for key, name in baseline_order:
        if key not in baseline_cv_results:
            continue
        fr = baseline_cv_results[key]['fold_results']
        fold_aucs = [r['val_auc'] for r in fr]
        fold_sens = [r.get('sensitivity', float('nan')) for r in fr]
        fold_specs = [r.get('specificity', float('nan')) for r in fr]
        fold_f1s = [r.get('val_f1', float('nan')) for r in fr]
        rows_internal.append({
            'Model': name, 'Source': 'Internal 10-fold CV',
            'AUC': round_to_4sf(np.nanmean(fold_aucs)),
            'AUC_95CI': '',
            'Sensitivity': (round_to_4sf(np.nanmean(fold_sens))
                            if not all(np.isnan(fold_sens)) else float('nan')),
            'Specificity': (round_to_4sf(np.nanmean(fold_specs))
                            if not all(np.isnan(fold_specs)) else float('nan')),
            'F1': (round_to_4sf(np.nanmean(fold_f1s))
                   if not all(np.isnan(fold_f1s)) else float('nan')),
            'PPV': float('nan'), 'NPV': float('nan'),
            'Brier': '', 'Brier_95CI': '',
        })

    for key, name in baseline_order:
        if key in baseline_oof_preds:
            ci = baseline_oof_ci_results.get(key) if baseline_oof_ci_results else None
            rows_internal.append(_build_oof_row(name, baseline_oof_preds[key], ci))

    # --- External test ---
    ext_metrics = ext_state['ext_metrics']
    ext_auc_ci = ext_state['ext_auc_ci']
    ext_brier_ci = ext_state.get('ext_brier_ci')
    ext_wilson_cis = ext_state.get('ext_wilson_cis')
    ext_f1_ci = ext_state.get('ext_f1_ci')
    rows_external.append(_build_ext_row(
        'Image only linear LR (main)', ext_metrics, ext_auc_ci,
        ext_brier_ci, ext_wilson_cis, ext_f1_ci
    ))

    pred_dir = os.path.join(output_dir, 'predictions')
    xlsx_path = os.path.join(pred_dir, 'all_models_comprehensive_summary.xlsx')

    cols_int = ['Model', 'Source', 'AUC', 'AUC_95CI',
                'Sensitivity', 'Sensitivity_95CI',
                'Specificity', 'Specificity_95CI',
                'F1', 'F1_95CI',
                'Accuracy', 'Accuracy_95CI',
                'PPV', 'PPV_95CI', 'NPV', 'NPV_95CI',
                'Brier', 'Brier_95CI']
    cols_ext = cols_int

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        if rows_internal:
            pd.DataFrame(rows_internal)[cols_int].to_excel(
                writer, sheet_name='Internal', index=False)
        if rows_external:
            pd.DataFrame(rows_external)[cols_ext].to_excel(
                writer, sheet_name='External', index=False)

    wb = openpyxl.load_workbook(xlsx_path)
    max_col_letter = 'V' if rows_external else 'R'
    for ws in wb.worksheets:
        for col_letter in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                           'S', 'T', 'U', 'V']:
            if col_letter <= max_col_letter:
                ws.column_dimensions[col_letter].width = 20
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row,
                                min_col=4, max_col=22):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = '0.0000'
    wb.save(xlsx_path)


def plot_external_figures(ext_state, output_dir, threshold=None):
    """External test set: ROC / confusion matrix / decision curve analysis."""
    plt.rcParams['axes.unicode_minus'] = False
    figures_dir = os.path.join(output_dir, 'figures')
    if threshold is None:
        threshold = ext_state.get('threshold', ACTIVE_THRESHOLD)
    y_true = ext_state['fusion_preds']['true']
    fusion_prob = ext_state['fusion_preds']['prob'][:, 1]
    fusion_auc, fusion_ci_l, fusion_ci_u = ext_state['ext_auc_ci']

    model_data = {'Image only linear LR': fusion_prob}

    # === ROC curve ===
    fig, ax = plt.subplots(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_true, fusion_prob)
    ax.plot(fpr, tpr, color='#d7191c', linestyle='-', lw=2,
            label=f'Image only linear LR '
                  f'(AUC={fusion_auc:.3f}, '
                  f'95%CI {fusion_ci_l:.3f}-{fusion_ci_u:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve on External Test Set (class {POSITIVE_CLASS})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, frameon=True, edgecolor='gray')
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'external_roc_comparison.tiff'),
                dpi=1000, bbox_inches='tight')
    plt.close()
    print("  External ROC figure saved (TIFF 1000 dpi)")

    # === Confusion matrix ===
    n_models = len(model_data)
    n_cols = min(n_models, 3)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    model_names = list(model_data.keys())
    vmax = len(y_true)
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        prob = model_data[name]
        y_pred = prob_to_class(prob, threshold)
        cm = confusion_matrix(y_true, y_pred)
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues,
                  vmin=0, vmax=vmax)
        ax.set_title(name, fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASSES, fontsize=10)
        ax.set_yticklabels(CLASSES, fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14)
    for idx in range(n_models, n_rows * n_cols):
        axes[idx].set_visible(False)
    fig.suptitle(f'Confusion Matrix on External Test Set '
                 f'(threshold={threshold:.4f})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'external_confusion_matrix_comparison.tiff'),
                dpi=1000, bbox_inches='tight')
    plt.close()
    print("  External confusion matrix figure saved")

    # === Decision curve analysis ===
    fig, ax = plt.subplots(figsize=(9, 7))
    thresholds = np.arange(0.01, 1.00, 0.01)
    prevalence = y_true.mean()
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    treat_none = np.zeros_like(thresholds)
    ax.plot(thresholds, treat_all, 'k--', lw=1.8, label='Treat all')
    ax.plot(thresholds, treat_none, 'k:', lw=1.8, label='Treat none')
    _, nb = calculate_dca(y_true, fusion_prob, thresholds)
    ax.plot(thresholds, nb, color='#d7191c', linestyle='-', lw=2,
            label='Image only linear LR')
    ax.set_xlabel('Threshold Probability', fontsize=13)
    ax.set_ylabel('Net Benefit', fontsize=13)
    ax.set_title('Decision Curve Analysis on External Test Set',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.3, 0.6)
    ax.legend(loc='upper right', fontsize=10, frameon=True,
              edgecolor='gray', fancybox=False)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'external_dca_curves.tiff'),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  External decision curve analysis figure saved")


def _percentile_auc_ci(y_true, y_prob, n_bootstrap=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    """BCa bootstrap AUC CI, consistent with the internal validation set."""
    _, lo, hi = bca_ci_general(y_true, y_prob,
                                lambda yt, yp: roc_auc_score(yt, yp),
                                n_bootstrap, alpha, seed)
    return lo, hi


def plot_internal_oof_roc(fusion_oof, baseline_oof_preds, output_dir):
    """Internal OOF summary: main model + baselines ROC, same format as external."""
    plt.rcParams['axes.unicode_minus'] = False
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    color_cycle = ['#d7191c', '#1a9641', '#fdae61', '#d01c8b',
                   '#4575b4', '#a6d96a', '#7b3294']

    models = [(fusion_oof['display_name'],
               np.asarray(fusion_oof['y_true']),
               np.asarray(fusion_oof['y_prob']))]
    for key, oof in baseline_oof_preds.items():
        if key == 'baseline_image_linear':
            continue
        models.append((oof['display_name'],
                       np.asarray(oof['y_true']),
                       np.asarray(oof['y_prob'])))

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, (name, y_true, y_prob) in enumerate(models):
        auc = roc_auc_score(y_true, y_prob)
        ci_l, ci_u = _percentile_auc_ci(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, color=color_cycle[idx % len(color_cycle)],
                linestyle='-', lw=2,
                label=f'{name} (AUC={auc:.3f}, 95%CI {ci_l:.3f}-{ci_u:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve on Internal OOF Summary (class {POSITIVE_CLASS})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, frameon=True, edgecolor='gray')
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'internal_oof_roc_comparison.tiff'),
                dpi=1000, bbox_inches='tight')
    plt.close()
    print("  Internal OOF ROC figure saved (TIFF 1000 dpi)")


# ============================================================================
# Command-line interface
# ============================================================================
def build_parser():
    parser = argparse.ArgumentParser(
        prog='multimodal_fusion_binary',
        description='WSI + clinical features multimodal gated fusion '
                    'classification pipeline (binary A vs B).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--sl-dir', type=Path, required=True,
                        help='SL (training) data root directory.')
    parser.add_argument('--fy-dir', type=Path, required=True,
                        help='FY (external test) data root directory.')
    parser.add_argument('--sl-clinical-file', type=Path, required=True,
                        help='SL clinical Excel file.')
    parser.add_argument('--fy-clinical-file', type=Path, required=True,
                        help='FY clinical Excel file.')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Root directory for all outputs.')

    parser.add_argument('--n-folds', type=int, default=N_FOLDS,
                        help='Number of CV folds.')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Global random seed.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Mini-batch size.')
    parser.add_argument('--max-epochs', type=int, default=MAX_EPOCHS,
                        help='Maximum training epochs.')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                        help='Early-stopping patience.')
    parser.add_argument('--n-bootstrap', type=int, default=N_BOOTSTRAP,
                        help='Number of bootstrap resamples for CIs.')
    parser.add_argument('--no-multi-gpu', action='store_true',
                        help='Disable DataParallel even if multiple GPUs are available.')
    parser.add_argument('--no-threshold-optimization', action='store_true',
                        help='Disable per-model threshold optimization '
                             '(use 0.5 everywhere).')
    parser.add_argument('--no-lasso', action='store_true',
                        help='Disable LASSO clinical feature selection.')
    return parser


def main():
    args = build_parser().parse_args()

    global N_FOLDS, SEED, BATCH_SIZE, MAX_EPOCHS, PATIENCE, N_BOOTSTRAP
    global ENABLE_LASSO_SELECTION, ENABLE_THRESHOLD_OPTIMIZATION, USE_MULTI_GPU

    N_FOLDS = args.n_folds
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    MAX_EPOCHS = args.max_epochs
    PATIENCE = args.patience
    N_BOOTSTRAP = args.n_bootstrap
    ENABLE_LASSO_SELECTION = not args.no_lasso
    ENABLE_THRESHOLD_OPTIMIZATION = not args.no_threshold_optimization
    if args.no_multi_gpu:
        USE_MULTI_GPU = False

    set_seed(SEED)

    SL_DIR = args.sl_dir
    FY_DIR = args.fy_dir
    SL_CLINICAL_FILE = args.sl_clinical_file
    FY_CLINICAL_FILE = args.fy_clinical_file
    OUTPUT_DIR = args.output_dir

    if not SL_DIR.is_dir():
        print(f"ERROR: SL directory not found: {SL_DIR}", file=sys.stderr)
        return 1
    if not FY_DIR.is_dir():
        print(f"ERROR: FY directory not found: {FY_DIR}", file=sys.stderr)
        return 1
    if not SL_CLINICAL_FILE.is_file():
        print(f"ERROR: SL clinical file not found: {SL_CLINICAL_FILE}",
              file=sys.stderr)
        return 1
    if not FY_CLINICAL_FILE.is_file():
        print(f"ERROR: FY clinical file not found: {FY_CLINICAL_FILE}",
              file=sys.stderr)
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_output_dirs(OUTPUT_DIR)

    t_start = datetime.now()
    print("=" * 70)
    print("  WSI + clinical features multimodal gated fusion pipeline")
    print(f"  Global seed: {SEED} | Started: "
          f"{t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Available GPUs: {N_GPUS} | NN baselines use DataParallel "
          f"(USE_MULTI_GPU={USE_MULTI_GPU}); main model runs 10 folds serially")
    print("=" * 70)

    # Step 1: Load data
    step_start = datetime.now()
    print("\n[Step 1] Loading training patch features ...")
    sl_patch_dir = os.path.join(SL_DIR, 'SL-patch_features')
    case_ids, patch_list_all, categories_all = load_patch_features(sl_patch_dir)
    clinical_df = load_clinical_features(SL_CLINICAL_FILE)
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Step 2: 10-fold split
    step_start = datetime.now()
    print("\n[Step 2] Generating stratified 10-fold split ...")
    folds = load_or_create_folds(case_ids, categories_all, OUTPUT_DIR)
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Step 3-8: Main model (image-only linear LR) 10-fold CV
    step_start = datetime.now()
    print(f"\n[Step 3-8] Main model (image-only Linear LR) 10-fold CV ...")

    binary_labels = np.array([LABEL_MAP_BINARY[cat] for cat in categories_all])

    all_fold_results = []
    skipped = []
    for fold_info in tqdm(folds, desc="  10-fold training (Linear LR)", unit="fold"):
        fold_idx = fold_info['fold']
        summary_path = os.path.join(OUTPUT_DIR, 'folds',
                                    f'fold_summary_lr_{fold_idx}.pkl')
        if os.path.exists(summary_path):
            with open(summary_path, 'rb') as f:
                _cached_fold = pickle.load(f)
            if _cached_fold.get('leak_fix_version', 1) >= LR_LEAK_FIX_VERSION:
                all_fold_results.append(_cached_fold)
                skipped.append(fold_idx)
                continue
            print(f"  Fold {fold_idx} cache predates the leak fix; recomputing ...")
        fold_sum = _train_eval_lr_fold(fold_info, case_ids, patch_list_all,
                                        binary_labels, fold_idx, OUTPUT_DIR)
        all_fold_results.append(fold_sum)

    if skipped:
        print(f"  [Resume] Completed folds: {sorted(skipped)}")

    all_fold_results.sort(key=lambda x: x['fold_idx'])

    metrics_list = ['auc', 'accuracy', 'sensitivity', 'specificity', 'f1']
    summary_data = {'Fold': [r['fold_idx'] for r in all_fold_results]}
    for m in metrics_list:
        summary_data[f'{m}_mean'] = [r[m] for r in all_fold_results]
    summary_data['AUC_gap'] = [r.get('auc_gap', 0) for r in all_fold_results]
    cv_summary_df = pd.DataFrame(summary_data)

    mean_row = {'Fold': 'Mean+-Std'}
    for m in metrics_list:
        vals = [r[m] for r in all_fold_results]
        mean_row[f'{m}_mean'] = f'{np.mean(vals):.4f}+-{np.std(vals):.4f}'
    cv_summary_df = pd.concat([cv_summary_df, pd.DataFrame([mean_row])],
                              ignore_index=True)
    cv_summary_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions',
                                       '10fold_cv_summary.csv'), index=False)

    fusion_oof_y_true = []
    fusion_oof_y_prob = []
    fusion_oof_case_ids = []
    for fold_idx in range(N_FOLDS):
        csv_path = os.path.join(OUTPUT_DIR, 'predictions',
                                f'val_prediction_lr_fold_{fold_idx}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            fusion_oof_case_ids.extend(df['case_id'].tolist())
            fusion_oof_y_true.extend(
                (df['true_label'] == POSITIVE_CLASS).astype(int).tolist())
            fusion_oof_y_prob.extend(df['prob_B'].tolist())

    oof_auc = roc_auc_score(fusion_oof_y_true, fusion_oof_y_prob)
    _, ci_l, ci_u = auc_bca_ci(fusion_oof_y_true, fusion_oof_y_prob)

    fusion_oof = {
        'case_ids': fusion_oof_case_ids,
        'y_true': fusion_oof_y_true,
        'y_prob': fusion_oof_y_prob,
        'display_name': 'Image only linear LR (main)',
        'oof_auc': oof_auc,
        'auc_ci': (oof_auc, ci_l, ci_u),
    }

    print("\n  [OOF metrics] Computing main model Brier Score / 95% CI ...")
    main_oof_metrics_ci = compute_oof_metrics_with_ci(fusion_oof['y_true'],
                                                      fusion_oof['y_prob'])
    print(f"    Brier: {main_oof_metrics_ci['brier']:.4g} "
          f"(95% CI: {main_oof_metrics_ci['brier_ci'][1]:.4g}-"
          f"{main_oof_metrics_ci['brier_ci'][2]:.4g})")

    print(f"\n  10-fold CV summary:")
    print(f"    OOF AUC: {oof_auc:.4f} (95% CI: {ci_l:.4f} - {ci_u:.4f})")
    print(f"  10-fold summary saved: predictions/10fold_cv_summary.csv")
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Post-processing: optimal threshold search
    step_start = datetime.now()
    print(f"\n[Post-processing] Optimal threshold search ...")

    oof_prob = np.array(fusion_oof['y_prob'])
    oof_true = np.array(fusion_oof['y_true'])

    optimal_threshold = search_model_threshold(oof_true, oof_prob)
    fusion_oof['threshold'] = optimal_threshold
    print(f"    Main model (image-only linear LR) optimal threshold "
          f"(max specificity within Youden tolerance) = "
          f"{optimal_threshold:.4f}")

    for fi in range(N_FOLDS):
        fp_ = os.path.join(OUTPUT_DIR, 'predictions',
                           f'val_prediction_lr_fold_{fi}.csv')
        if os.path.exists(fp_):
            df_ = pd.read_csv(fp_)
            yc_ = prob_to_class(df_['prob_B'].to_numpy(), optimal_threshold)
            df_['pred_label'] = [POSITIVE_CLASS if p == 1 else NEGATIVE_CLASS
                                 for p in yc_]
            df_.to_csv(fp_, index=False)

    _cv_old = None
    _cv_path = os.path.join(OUTPUT_DIR, 'predictions', '10fold_cv_summary.csv')
    if os.path.exists(_cv_path):
        try:
            _cv_old = pd.read_csv(_cv_path)
        except Exception:
            _cv_old = None
    _cv_gap = {}
    if _cv_old is not None and 'Fold' in _cv_old.columns:
        for _, _r in _cv_old.iterrows():
            try:
                _cv_gap[int(_r['Fold'])] = _r.get('AUC_gap', 0)
            except Exception:
                pass
    _cv_rows = []
    for fi in range(N_FOLDS):
        fp_ = os.path.join(OUTPUT_DIR, 'predictions',
                           f'val_prediction_lr_fold_{fi}.csv')
        if not os.path.exists(fp_):
            continue
        df_ = pd.read_csv(fp_)
        yt_ = (df_['true_label'] == POSITIVE_CLASS).astype(int).to_numpy()
        p1_ = df_['prob_B'].to_numpy()
        yc_ = prob_to_class(p1_, optimal_threshold)
        auc_fold = roc_auc_score(yt_, p1_)
        cm_ = confusion_matrix(yt_, yc_).ravel()
        tn_, fp2_, fn_, tp_ = cm_
        sens_ = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        spec_ = tn_ / (tn_ + fp2_) if (tn_ + fp2_) > 0 else 0.0
        acc_ = (tp_ + tn_) / (tp_ + tn_ + fp2_ + fn_) if (tp_ + tn_ + fp2_ + fn_) > 0 else 0.0
        f1_ = f1_score(yt_, yc_)
        _cv_rows.append((fi, auc_fold, acc_, sens_, spec_, f1_,
                         _cv_gap.get(fi, 0)))
    if _cv_rows:
        _cv_df = pd.DataFrame(_cv_rows, columns=['Fold', 'auc_mean',
                                                 'accuracy_mean',
                                                 'sensitivity_mean',
                                                 'specificity_mean',
                                                 'f1_mean', 'AUC_gap'])
        _mean_cv = {'Fold': 'Mean+-Std'}
        for _m, _i in [('auc_mean', 1), ('accuracy_mean', 2),
                       ('sensitivity_mean', 3), ('specificity_mean', 4),
                       ('f1_mean', 5)]:
            _vals = [r[_i] for r in _cv_rows]
            _mean_cv[_m] = f'{np.mean(_vals):.4f}+-{np.std(_vals):.4f}'
        _mean_cv['AUC_gap'] = float('nan')
        _cv_df = pd.concat([_cv_df, pd.DataFrame([_mean_cv])],
                           ignore_index=True)
        _cv_df.to_csv(_cv_path, index=False)
        print(f"    10fold_cv_summary.csv recomputed at the main model "
              f"threshold ({optimal_threshold:.4f})")

    threshold_config = {
        'optimal_threshold': optimal_threshold,
        'criterion': 'max specificity within Youden tolerance (favor A)',
        'per_model': True,
    }
    pickle.dump(threshold_config,
                open(os.path.join(OUTPUT_DIR, 'models', 'threshold_config.pkl'), 'wb'))
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Step 9: External test inference
    step_start = datetime.now()
    print(f"\n[Step 9] External test inference (main model) ...")

    ext_state_path = os.path.join(OUTPUT_DIR, 'predictions', 'external_test_state.pkl')
    if (os.path.exists(ext_state_path)
            and pickle.load(open(ext_state_path, 'rb')).get('leak_fix_version', 1)
            >= LR_LEAK_FIX_VERSION):
        with open(ext_state_path, 'rb') as f:
            ext_state = pickle.load(f)
        print("  Existing external test results found, loading ...")
        _fp_c = ext_state['fusion_preds']
        _yt_c = np.asarray(_fp_c['true'])
        _p1_c = np.asarray(_fp_c['prob'])[:, 1]
        _yc_c = prob_to_class(_p1_c, optimal_threshold)
        _fp_c['class'] = _yc_c
        ext_state['threshold'] = optimal_threshold
        _cm_c = confusion_matrix(_yt_c, _yc_c).ravel()
        _tn_c, _fp2_c, _fn_c, _tp_c = _cm_c
        ext_state['ext_metrics'].update({
            'accuracy': accuracy_score(_yt_c, _yc_c),
            'sensitivity': _tp_c / (_tp_c + _fn_c) if (_tp_c + _fn_c) > 0 else 0.0,
            'specificity': _tn_c / (_tn_c + _fp2_c) if (_tn_c + _fp2_c) > 0 else 0.0,
            'f1': f1_score(_yt_c, _yc_c),
        })
        ext_state['ext_wilson_cis'] = {
            'sensitivity': wilson_ci(_tp_c, _tp_c + _fn_c),
            'specificity': wilson_ci(_tn_c, _tn_c + _fp2_c),
            'ppv': wilson_ci(_tp_c, _tp_c + _fp2_c),
            'npv': wilson_ci(_tn_c, _tn_c + _fn_c),
            'accuracy': wilson_ci(_tp_c + _tn_c,
                                  _tp_c + _tn_c + _fp2_c + _fn_c),
            'f1': wilson_ci(2 * _tp_c, 2 * _tp_c + _fp2_c + _fn_c),
        }
        ext_state['ext_f1_ci'] = None
        _ext_csv_path = os.path.join(OUTPUT_DIR, 'predictions',
                                      'main_external_predictions.csv')
        if os.path.exists(_ext_csv_path):
            _edf_c = pd.read_csv(_ext_csv_path)
            _edf_c['pred_label'] = [POSITIVE_CLASS if p == 1 else NEGATIVE_CLASS
                                    for p in _yc_c]
            _edf_c.to_csv(_ext_csv_path, index=False)
        print("    Re-computed external predictions and metrics with the "
              "latest main model threshold")
    else:
        sl_mean_pooled = compute_mean_pooled_features(patch_list_all)
        best_c_ext = select_lr_best_c(sl_mean_pooled, binary_labels, C_GRID,
                                       STAT_INNER_CV_FOLDS, seed=SEED)
        sl_scaler = StandardScaler().fit(sl_mean_pooled)
        X_sl_scaled = sl_scaler.transform(sl_mean_pooled)
        sl_lr = LogisticRegression(penalty='l1', solver='saga', C=best_c_ext,
                                   max_iter=5000, random_state=SEED,
                                   class_weight=None)
        sl_lr.fit(X_sl_scaled, binary_labels)

        pickle.dump(sl_scaler, open(os.path.join(OUTPUT_DIR, 'models',
                                                 'final_lr_scaler.pkl'), 'wb'))
        pickle.dump(sl_lr, open(os.path.join(OUTPUT_DIR, 'models',
                                             'final_lr_model.pkl'), 'wb'))

        fy_patch_dir = os.path.join(FY_DIR, 'FY-patch_features')
        fy_case_ids, fy_patch_list, fy_categories = load_patch_features(fy_patch_dir)
        fy_binary = np.array([LABEL_MAP_BINARY[cat] for cat in fy_categories])

        fy_mean_pooled = compute_mean_pooled_features(fy_patch_list)
        fy_scaled = sl_scaler.transform(fy_mean_pooled)
        fy_prob = sl_lr.predict_proba(fy_scaled)
        fy_pred = prob_to_class(fy_prob[:, 1], optimal_threshold)

        ext_auc = roc_auc_score(fy_binary, fy_prob[:, 1])
        _, ext_ci_l, ext_ci_u = auc_bca_ci(fy_binary, fy_prob[:, 1])

        ext_acc = accuracy_score(fy_binary, fy_pred)
        tn, fp, fn, tp = confusion_matrix(fy_binary, fy_pred).ravel()
        ext_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        ext_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ext_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        ext_brier = compute_brier_score(fy_binary, fy_prob[:, 1])
        ext_brier_ci = bca_ci_general(fy_binary, fy_prob[:, 1],
            lambda yt, yp: compute_brier_score(yt, yp), N_BOOTSTRAP, 0.05, SEED)
        ext_wilson_cis = {
            'sensitivity': wilson_ci(tp, tp + fn),
            'specificity': wilson_ci(tn, tn + fp),
            'ppv': wilson_ci(tp, tp + fp),
            'npv': wilson_ci(tn, tn + fn),
            'accuracy': wilson_ci(tp + tn, tp + tn + fp + fn),
            'f1': wilson_ci(2 * tp, 2 * tp + fp + fn),
        }
        ext_f1_ci = bca_ci_general(fy_binary, fy_prob[:, 1],
            lambda yt, yp: f1_score(yt, prob_to_class(yp, optimal_threshold)),
            N_BOOTSTRAP, 0.05, SEED)

        ext_state = {
            'ext_metrics': {
                'auc': ext_auc, 'accuracy': ext_acc,
                'sensitivity': ext_sens, 'specificity': ext_spec, 'f1': ext_f1
            },
            'ext_auc_ci': (ext_auc, ext_ci_l, ext_ci_u),
            'ext_brier_ci': (ext_brier, ext_brier_ci[1], ext_brier_ci[2]),
            'ext_wilson_cis': ext_wilson_cis,
            'ext_f1_ci': ext_f1_ci,
            'threshold': optimal_threshold,
            'fusion_preds': {
                'true': fy_binary,
                'prob': fy_prob,
                'class': fy_pred,
                'case_ids': fy_case_ids,
                'display_name': 'Image only linear LR (main)',
            },
            'leak_fix_version': LR_LEAK_FIX_VERSION,
        }

        pd.DataFrame({
            'case_id': fy_case_ids,
            'true_label': [POSITIVE_CLASS if y == 1 else NEGATIVE_CLASS
                           for y in fy_binary],
            'pred_label': [POSITIVE_CLASS if p == 1 else NEGATIVE_CLASS
                           for p in fy_pred],
            'prob_A': fy_prob[:, 0],
            'prob_B': fy_prob[:, 1],
        }).to_csv(os.path.join(OUTPUT_DIR, 'predictions',
                               'main_external_predictions.csv'), index=False)

        pickle.dump(ext_state, open(ext_state_path, 'wb'))

    # External test: complete PPV/NPV
    ext_state.setdefault('threshold', optimal_threshold)
    _ext_m = ext_state['ext_metrics']
    _fp_ = ext_state['fusion_preds']
    _yt_ext = np.asarray(_fp_['true'])
    _p1_ext = np.asarray(_fp_['prob'])[:, 1]
    _pred_cls = np.asarray(_fp_.get('class',
                                     prob_to_class(_p1_ext, optimal_threshold)))
    _cm_ext = confusion_matrix(_yt_ext, _pred_cls).ravel()
    _tne, _fpe, _fne, _tpe = _cm_ext
    _ext_m['ppv'] = _tpe / (_tpe + _fpe) if (_tpe + _fpe) > 0 else 0
    _ext_m['npv'] = _tne / (_tne + _fne) if (_tne + _fne) > 0 else 0
    _ew = ext_state.setdefault('ext_wilson_cis', {})
    _ew['ppv'] = wilson_ci(_tpe, _tpe + _fpe)
    _ew['npv'] = wilson_ci(_tne, _tne + _fne)
    if 'ext_f1_ci' not in ext_state or ext_state.get('ext_f1_ci') is None:
        ext_state['ext_f1_ci'] = bca_ci_general(_yt_ext, _p1_ext,
            lambda yt, yp: f1_score(yt, prob_to_class(yp, optimal_threshold)),
            N_BOOTSTRAP, 0.05, SEED)

    pickle.dump(ext_state, open(ext_state_path, 'wb'))

    print("\n  === External test set ===")
    print(f"    PPV: {_ext_m['ppv']:.4f} | NPV: {_ext_m['npv']:.4f} "
          f"(threshold={optimal_threshold:.4f})")
    print(f"    AUC: {_ext_m['auc']:.4f} | Accuracy: {_ext_m['accuracy']:.4f} | "
          f"Sensitivity: {_ext_m['sensitivity']:.4f} | "
          f"Specificity: {_ext_m['specificity']:.4f}")
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Step 10: Baseline models + visualization
    step_start = datetime.now()
    print("\n[Step 10] Baseline models and visualization ...")

    baseline_cv_path = os.path.join(OUTPUT_DIR, 'baselines', 'baselines_cv.pkl')
    if os.path.exists(baseline_cv_path):
        print("  Existing baseline results found, loading ...")
        with open(baseline_cv_path, 'rb') as f:
            data = pickle.load(f)
        baseline_cv_results = data['results']
        baseline_oof_preds = data['oof_preds']
        baseline_oof_ci_results = data.get('oof_ci_results', {})
        if 'baseline_clinical_only' not in baseline_oof_preds:
            print("  Clinical LR baseline cache missing (leak fix); "
                  "recomputing that model only ...")
            _clin_res, _clin_oof, _clin_ci = run_clinical_lr_baseline(
                folds, patch_list_all, clinical_df, categories_all, case_ids,
                os.path.join(OUTPUT_DIR, 'folds'), OUTPUT_DIR)
            baseline_cv_results.update(_clin_res)
            baseline_oof_preds.update(_clin_oof)
            baseline_oof_ci_results.update(_clin_ci)
            with open(baseline_cv_path, 'wb') as _f:
                pickle.dump({'results': baseline_cv_results,
                             'oof_preds': baseline_oof_preds,
                             'oof_ci_results': baseline_oof_ci_results}, _f)
            print("    Clinical LR baseline recomputed and baselines_cv.pkl updated")
        # Threshold logic changed: recompute per-baseline thresholds from
        # cached OOF probabilities and update fold-level metrics accordingly.
        for _k, _bl in baseline_oof_preds.items():
            _thr_new = search_model_threshold(_bl['y_true'], _bl['y_prob'])
            _bl['threshold'] = _thr_new
            if _k in baseline_cv_results:
                _cr = baseline_cv_results[_k]
                _cr['threshold'] = _thr_new
                for _fr in _cr.get('fold_results', []):
                    if 'val_prob_pos' in _fr and 'y_val' in _fr:
                        _yc_fr = prob_to_class(_fr['val_prob_pos'], _thr_new)
                        _yt_fr = _fr['y_val']
                        _tna, _fpa, _fna, _tpa = confusion_matrix(
                            _yt_fr, _yc_fr).ravel()
                        _fr['sensitivity'] = (_tpa / (_tpa + _fna)
                                              if (_tpa + _fna) > 0 else 0.0)
                        _fr['specificity'] = (_tna / (_tna + _fpa)
                                              if (_tna + _fpa) > 0 else 0.0)
                        _fr['val_f1'] = f1_score(_yt_fr, _yc_fr)
        print("    Re-computed each baseline's OOF threshold and "
              "fold-level metrics with the latest threshold logic")
    else:
        baseline_cv_results, baseline_oof_preds, baseline_oof_ci_results = \
            run_baselines_cv(folds, patch_list_all, clinical_df,
                             categories_all, case_ids, OUTPUT_DIR)

    plot_internal_oof_roc(fusion_oof, baseline_oof_preds, OUTPUT_DIR)
    plot_external_figures(ext_state, OUTPUT_DIR, threshold=optimal_threshold)

    print(f"  Elapsed: {datetime.now() - step_start}")

    print("\n[Summary] Generating comprehensive comparison Excel ...")
    _generate_comprehensive_summary(fusion_oof, ext_state, baseline_cv_results,
                                     baseline_oof_preds,
                                     baseline_oof_ci_results=baseline_oof_ci_results,
                                     main_oof_metrics_ci=main_oof_metrics_ci,
                                     output_dir=OUTPUT_DIR)
    print(f"  Excel saved: predictions/all_models_comprehensive_summary.xlsx")

    t_end = datetime.now()
    print(f"\n{'=' * 70}")
    print(f"  Pipeline completed. Total elapsed: {t_end - t_start}")
    print(f"  Results directory: {OUTPUT_DIR}")
    print(f"{'=' * 70}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
