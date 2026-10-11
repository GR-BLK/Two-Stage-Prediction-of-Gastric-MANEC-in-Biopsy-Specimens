#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
WSI + Clinical Features Contrastive Multimodal Fusion Classification Pipeline
(binary B vs C, with class A excluded).

This module implements a complete multimodal classification pipeline that
combines whole-slide image (WSI) patch-level features with tabular clinical
features for binary classification. Multiple fusion architectures are
supported (contrastive fusion with modality dropout, gated fusion,
confidence-guided fusion), and the pipeline performs rigorous internal
cross-validation plus external validation with comprehensive reporting
(BCa bootstrap CIs, Wilson CIs, decision curve analysis, Brier score).

Class label mapping
-------------------
    A  -> excluded class (formerly ADC), filtered out at load time
    B  -> positive class (formerly MANEC), binary label 1
    C  -> negative class (formerly NEC),  binary label 0

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
import glob
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
from scipy.stats import norm
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
    SKLEARN_N_JOBS = N_CPUS
    USE_MULTI_GPU = (N_GPUS > 1)
else:
    print(f"\n  [Hardware] No GPU detected; running on CPU")
    print(f"  [Hardware] Detected {N_CPUS} CPU threads (fully utilized)")
    DEVICE = torch.device('cpu')
    torch.backends.cudnn.benchmark = False
    SKLEARN_N_JOBS = N_CPUS
    USE_MULTI_GPU = False


# ============================================================================
# Global configuration
# ============================================================================
SEED = 2026
N_FOLDS = 10
N_CLASSES = 2

# Binary class labels
#   B  -> positive class (formerly MANEC), binary label 1
#   C  -> negative class (formerly NEC),  binary label 0
#   A  -> excluded (formerly ADC)
CLASSES = ['C', 'B']
CLASSES_ORIGINAL = ['A', 'B', 'C']
POSITIVE_CLASS = 'B'
NEGATIVE_CLASS = 'C'
EXCLUDED_CLASS = 'A'

# Patch-level features (raw patch features, CLAM attention pooling)
PATCH_FEAT_DIM = 1536
CLAM_HIDDEN_DIM = 128
CLAM_DROPOUT = 0.2

# Fusion and training hyperparameters
FUSION_HIDDEN_DIM = 128
COMMON_EMBED_DIM = 64
BATCH_SIZE = 8
MAX_EPOCHS = 200
PATIENCE = 30
LR = 1e-4
WEIGHT_DECAY = 1e-3
DROPOUT_RATE = 0.3
ATTENTION_L1_LAMBDA = 1e-4

# LASSO clinical feature selection (LogisticRegression L1)
ENABLE_LASSO_SELECTION = True
LASSO_CV_MAX_ITER = 5000
MIN_LASSO_FEATURES = 5

# LR model: inner 5-fold CV grid inside the training fold
LR_C_GRID = [0.01, 0.1, 1.0, 2.0, 3.0]

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
DROPOUT_PROB = 0.2
CONTRAST_LOSS_WEIGHT = 0.1

# NN model: inner split fraction (80% train / 20% inner val for early stop)
NN_INNER_VAL_FRAC = 0.2

# Bootstrap confidence intervals
N_BOOTSTRAP = 10000

# Post-processing: per-model threshold optimization on each model's OOF
ENABLE_THRESHOLD_OPTIMIZATION = True
GLOBAL_THRESHOLD = 0.5
MODEL_THRESHOLDS = {}


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
#   C  -> negative class (0)
#   A  -> excluded (filtered at load time)
# ============================================================================
LABEL_MAP_BINARY = {'C': 0, 'B': 1}


def to_binary_labels(categories):
    return np.array([LABEL_MAP_BINARY[c] for c in categories], dtype=np.int64)


# ============================================================================
# Filter out class A (formerly ADC), keeping only B and C
# ============================================================================
def filter_class_a_data(case_ids, patch_list, categories):
    """Remove class A (formerly ADC), keeping only B and C."""
    case_ids = np.array(case_ids)
    mask = np.array([c in ('B', 'C') for c in categories])
    if np.sum(mask) == len(categories):
        print(f"  No class-A samples present; nothing to filter")
        return case_ids, patch_list, categories
    filtered_ids = case_ids[mask]
    filtered_patches = [p for i, p in enumerate(patch_list) if mask[i]]
    filtered_categories = categories[mask]
    n_removed = int(np.sum(~mask))
    print(f"  Excluded {n_removed} class-A cases; "
          f"kept {len(filtered_ids)} (B + C)")
    print(f"  Class distribution after filtering: "
          f"{dict(zip(*np.unique(filtered_categories, return_counts=True)))}")
    return filtered_ids, filtered_patches, filtered_categories


def safe_auc(y_true, y_prob):
    """Defensive wrapper for roc_auc_score: return 0.5 for single-class labels."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_prob)


# ============================================================================
# Stratified bootstrap BCa 95% CI for AUC
# ============================================================================
def auc_bca_ci(y_true, y_prob, n_bootstrap=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    """Compute the stratified bootstrap BCa 95% confidence interval for AUC."""
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    obs_auc = safe_auc(y_true, y_prob)

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
        jack_aucs.append(safe_auc(y_true[jack_idx], y_prob[jack_idx]))
    jack_aucs = np.array(jack_aucs)
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
    """Compute the stratified bootstrap BCa 95% CI for an arbitrary metric."""
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


def prob_to_class(prob, threshold=None):
    """Map probabilities to classes: prob >= threshold -> 1 (B), else 0 (C)."""
    if threshold is None:
        threshold = GLOBAL_THRESHOLD
    return (np.asarray(prob) >= threshold).astype(int)


def round_to_4sf(value):
    """Round a float to 4 significant figures."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float('nan')
    if value == 0:
        return 0.0
    return float(f"{value:.4g}")


def classification_metric_cis(y_true, y_prob, threshold=None,
                              n_bootstrap=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    """Compute BCa bootstrap 95% CIs for sensitivity, specificity, PPV, NPV,
    and accuracy at the given threshold."""
    if threshold is None:
        threshold = GLOBAL_THRESHOLD

    def _make(kind):
        def metric_fn(yt, yp):
            tn, fp, fn, tp = confusion_matrix(
                yt, prob_to_class(yp, threshold), labels=[0, 1]).ravel()
            if kind == 'sensitivity':
                return tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if kind == 'specificity':
                return tn / (tn + fp) if (tn + fp) > 0 else 0.0
            if kind == 'ppv':
                return tp / (tp + fp) if (tp + fp) > 0 else 0.0
            if kind == 'npv':
                return tn / (tn + fn) if (tn + fn) > 0 else 0.0
            if kind == 'accuracy':
                return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            raise ValueError(kind)
        return metric_fn

    return {
        kind + '_ci': bca_ci_general(y_true, y_prob, _make(kind),
                                     n_bootstrap, alpha, seed)
        for kind in ['sensitivity', 'specificity', 'ppv', 'npv', 'accuracy']
    }


def compute_oof_metrics_with_ci(y_true, y_prob, threshold=None,
                                n_bootstrap=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    """Compute all OOF metrics with BCa bootstrap CIs (internal validation)."""
    if threshold is None:
        threshold = GLOBAL_THRESHOLD
    results = {}
    results['brier'] = compute_brier_score(y_true, y_prob)
    results['brier_ci'] = bca_ci_general(y_true, y_prob,
        lambda yt, yp: compute_brier_score(yt, yp), n_bootstrap, alpha, seed)
    results['auc_ci'] = bca_ci_general(y_true, y_prob,
        lambda yt, yp: safe_auc(yt, yp), n_bootstrap, alpha, seed)
    results['f1_ci'] = bca_ci_general(y_true, y_prob,
        lambda yt, yp: f1_score(yt, prob_to_class(yp, threshold)),
        n_bootstrap, alpha, seed)
    results.update(classification_metric_cis(
        y_true, y_prob, threshold=threshold,
        n_bootstrap=n_bootstrap, alpha=alpha, seed=seed))
    return results


# ============================================================================
# Data loading
# ============================================================================
def load_patch_features(patch_dir):
    """Load patch-level raw features from ``*_features.npz`` files.

    Class directory names must be one of ``A`` / ``B`` / ``C``. Class ``A``
    is filtered out at the pipeline level (see ``filter_class_a_data``).

    Parameters
    ----------
    patch_dir : str
        Root directory containing category sub-directories (A / B / C), each
        holding ``*_features.npz`` files.

    Returns
    -------
    case_ids : list of str
    patch_list : list of np.ndarray, each of shape (n_patches_i, PATCH_FEAT_DIM)
    categories : np.ndarray of str
    """
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
        category = dir_name  # 'A', 'B', or 'C'

        data = np.load(npz_path)
        patch_features = data['features'].astype(np.float32)

        case_ids.append(case_id)
        patch_list.append(patch_features)
        categories.append(category)

    categories = np.array(categories)
    n_patches = [p.shape[0] for p in patch_list]
    print(f"  Loaded patch features: {len(case_ids)} cases, "
          f"{PATCH_FEAT_DIM} dims / patch")
    print(f"  Patches per case: min={min(n_patches)}, max={max(n_patches)}, "
          f"median={np.median(n_patches):.0f}")
    print(f"  Class distribution: "
          f"{dict(zip(*np.unique(categories, return_counts=True)))}")
    return case_ids, patch_list, categories


def compute_mean_pooled_features(patch_list):
    """Mean-pool raw patches into fixed-length feature vectors."""
    features = np.array([np.mean(p, axis=0) for p in patch_list], dtype=np.float32)
    return features


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
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(case_ids, categories)):
        train_cases = case_ids[train_idx].tolist()
        val_cases = case_ids[val_idx].tolist()
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
# CLAM attention pooling
# ============================================================================
class CLAMAttentionPooling(nn.Module):
    """CLAM-style attention pooling over variable-length patch sequences."""

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


# ============================================================================
# Mean-pooled feature preprocessor (for the linear LR model)
# ============================================================================
class BranchA_StatPreprocessor:
    """Standardize mean-pooled features (1536-dim) for the linear LR model."""

    def __init__(self, seed=SEED):
        self.seed = seed
        self.scaler = None
        self.output_dim = None

    def fit(self, X_train_raw, y=None):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train_raw)
        self.output_dim = X_train_raw.shape[1]
        print(f"    BranchA: mean-pooled features standardized, "
              f"dim {self.output_dim}")
        return self

    def transform(self, X_raw):
        return self.scaler.transform(X_raw).astype(np.float32)

    def save(self, save_dir, fold_idx):
        with open(os.path.join(save_dir, f'stat_scaler_fold_{fold_idx}.pkl'),
                  'wb') as f:
            pickle.dump({'scaler': self.scaler, 'output_dim': self.output_dim}, f)

    def load(self, load_dir, fold_idx):
        with open(os.path.join(load_dir, f'stat_scaler_fold_{fold_idx}.pkl'),
                  'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.output_dim = data['output_dim']
        return self

    def save_final(self, save_dir, prefix='final'):
        with open(os.path.join(save_dir, f'{prefix}_stat_scaler.pkl'), 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'output_dim': self.output_dim}, f)


# ============================================================================
# Clinical feature preprocessor
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
# Contrastive multimodal fusion with improved modality dropout (MICCAI 2025)
# ============================================================================
class ImprovedModalityDropout(nn.Module):
    """Contrastive multimodal fusion with improved modality dropout.

    Reference
    ---------
    MICCAI 2025: Learning Contrastive Multimodal Fusion with Improved
    Modality Dropout. Official code:
    https://github.com/omron-sinicx/medical-modality-dropout

    Core mechanism
    --------------
    1. During training, randomly drop an entire modality with probability p.
    2. Single-modality encoders plus a multimodal fusion encoder.
    3. Contrastive learning aligns single-modality and multimodal
       representations.
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
                 fusion_hidden=128, dropout_prob=0.2, seed=2026):
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
            nn.Linear(64, 1)
        )

    def forward(self, patches, clinical_features, masks=None):
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
# Baseline models
# ============================================================================
class Baseline4_ImageBranchAOnly(nn.Module):
    """Image patch features only, with CLAM attention pooling."""

    def __init__(self, clinical_input_dim=None, fusion_hidden=FUSION_HIDDEN_DIM,
                 n_classes=N_CLASSES, seed=SEED, **kwargs):
        super().__init__()
        set_seed(seed)
        self.clam_pooling = CLAMAttentionPooling()
        self.net = nn.Sequential(
            nn.Linear(PATCH_FEAT_DIM, fusion_hidden), nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(fusion_hidden, fusion_hidden // 2), nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(fusion_hidden // 2, 1)
        )

    def forward(self, patches, clinical_features=None, masks=None,
                mixup_lam=None, mixup_perm=None):
        if isinstance(patches, list):
            pooled, attention = self.clam_pooling(patches)
        else:
            pooled, attention = self.clam_pooling(patches, masks)
        if mixup_lam is not None and mixup_perm is not None and self.training:
            pooled = mixup_lam * pooled + (1 - mixup_lam) * pooled[mixup_perm]
        return (self.net(pooled), attention, None, None, None,
                torch.tensor(0.0, device=pooled.device))


class Baseline_ClinicalOnlyMLP(nn.Module):
    """Clinical-only MLP, for a fair NN comparison with the image-only MLP."""

    def __init__(self, stat_dim=None, patch_dim=None,
                 clinical_input_dim=None, n_classes=N_CLASSES, seed=SEED, **kwargs):
        super().__init__()
        set_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(clinical_input_dim, 128), nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, 1)
        )

    def forward(self, patches=None, clinical_features=None, masks=None, **kwargs):
        return (self.net(clinical_features), [], None, None, None,
                torch.tensor(0.0, device=clinical_features.device))


# ============================================================================
# MedPatch: confidence-guided multi-stage fusion
# ============================================================================
class ConfidenceGuidedFusion(nn.Module):
    """MedPatch: Confidence-Guided Multi-Stage Fusion.

    Reference
    ---------
    MedPatch: Confidence-Guided Multi-Stage Fusion for Multimodal Clinical
    Data. Official code: https://github.com/nyuad-cai/MedPatch
    """
    def __init__(self, feat_dim: int, n_classes: int = 2):
        super().__init__()

        self.joint_fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(feat_dim, feat_dim)
        )
        self.joint_classifier = nn.Linear(feat_dim, 1)

        self.img_classifier = nn.Linear(feat_dim, 1)
        self.cli_classifier = nn.Linear(feat_dim, 1)

        self.img_confidence = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.cli_confidence = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

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
                 fusion_hidden=128, seed=2026):
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

    def forward(self, patches, clinical_features, masks=None):
        if isinstance(patches, list):
            pooled, _ = self.clam_pooling(patches)
        else:
            pooled, _ = self.clam_pooling(patches, masks)

        img_feat = self.img_encoder(pooled)
        cli_feat = self.cli_encoder(clinical_features)

        logits, confidence = self.medpatch_fusion(img_feat, cli_feat)

        return (logits, None, img_feat, cli_feat, confidence,
                torch.tensor(0.0, device=img_feat.device))


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
            nn.Linear(feat_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.LayerNorm(feat_dim)
        )

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
                 fusion_hidden=128, seed=2026):
        super().__init__()
        torch.manual_seed(seed)

        self.clam_pooling = CLAMAttentionPooling()

        self.img_encoder = nn.Sequential(
            nn.Linear(PATCH_FEAT_DIM, fusion_hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(fusion_hidden, common_dim), nn.ReLU(), nn.Dropout(0.2)
        )

        self.cli_encoder = nn.Sequential(
            nn.Linear(clinical_input_dim, fusion_hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(fusion_hidden, common_dim), nn.ReLU(), nn.Dropout(0.2)
        )

        self.gff = GatedFeatureFusion(feat_dim=common_dim)

        self.classifier = nn.Sequential(
            nn.Linear(common_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, patches, clinical_features, masks=None):
        if isinstance(patches, list):
            pooled, _ = self.clam_pooling(patches)
        else:
            pooled, _ = self.clam_pooling(patches, masks)

        img_feat = self.img_encoder(pooled)
        cli_feat = self.cli_encoder(clinical_features)

        fused, gate = self.gff(img_feat, cli_feat)

        logits = self.classifier(fused)

        return (logits, None, img_feat, cli_feat, gate,
                torch.tensor(0.0, device=img_feat.device))


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

    def forward(self, patches, masks, clinical_features, **kwargs):
        logits, attentions, img_code, cli_code, gate, gate_reg = self.base_model(
            patches=patches, masks=masks, clinical_features=clinical_features
        )
        return logits, attentions, img_code, cli_code, gate, gate_reg


# Padded collate is used unconditionally to keep the numerical path identical
# to previous runs. DataParallel is only applied when multiple GPUs are present.
COLLATE_FN = padded_collate_fn


def wrap_model_for_multi_gpu(model):
    """Wrap the model for DataParallel when multiple GPUs are available."""
    if USE_MULTI_GPU:
        model = MultiGPUModelWrapper(model)
        model = nn.DataParallel(model)
    return model


def build_criterion(device=None):
    """Unified standard BCE with logits, no class weights, no label smoothing.

    Consistent with the LR objective LogisticRegression(penalty='l1',
    class_weight=None).
    """
    return torch.nn.BCEWithLogitsLoss(pos_weight=None)


def compute_metrics(y_true, y_pred_prob, y_pred_class):
    metrics = {}
    prob_pos = y_pred_prob[:, 1] if y_pred_prob.ndim > 1 else y_pred_prob
    metrics['accuracy'] = accuracy_score(y_true, y_pred_class)
    metrics['auc'] = safe_auc(y_true, prob_pos)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class, labels=[0, 1]).ravel()
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics['f1'] = f1_score(y_true, y_pred_class)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred_class,
                                                    labels=[0, 1])
    return metrics


def _is_padded_batch(batch):
    return 'patches_padded' in batch


def _get_model_inputs(batch, device):
    """Unify input handling for single-GPU and multi-GPU modes."""
    clinical = batch['clinical'].to(device)
    labels = batch['label'].to(device)
    if _is_padded_batch(batch):
        return (clinical, labels,
                batch['patches_padded'].to(device), batch['masks'].to(device))
    patches = [p.to(device) for p in batch['patches']]
    return clinical, labels, patches, None


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in dataloader:
        clinical, labels, patches_or_padded, masks = _get_model_inputs(batch, device)
        optimizer.zero_grad()

        logits, attentions, img_code, cli_code, gate, gate_reg = model(
            patches=patches_or_padded, masks=masks, clinical_features=clinical
        )
        loss_cls = criterion(logits, labels.float().unsqueeze(1))

        if img_code is not None and cli_code is not None:
            cos_sim = F.cosine_similarity(img_code, cli_code, dim=1)
            loss_align = torch.mean(1 - cos_sim) * MODAL_ALIGN_LAMBDA
        else:
            loss_align = torch.tensor(0.0, device=device)

        loss_contrast = (gate_reg if isinstance(gate_reg, torch.Tensor)
                         else torch.tensor(0.0, device=device))
        if loss_contrast.dim() > 0:
            loss_contrast = loss_contrast.mean()
        loss_contrast = loss_contrast * CONTRAST_LOSS_WEIGHT

        loss = loss_cls + loss_align + loss_contrast
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
        logits, _, img_code, cli_code, gate, gate_reg = model(
            patches=patches_or_padded, masks=masks, clinical_features=clinical
        )
        loss = criterion(logits, labels.float().unsqueeze(1))
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
    logits_flat = all_preds.reshape(-1)
    prob_pos = torch.sigmoid(torch.from_numpy(logits_flat)).numpy()
    pred_prob = np.column_stack([1 - prob_pos, prob_pos])
    pred_class = prob_to_class(prob_pos, GLOBAL_THRESHOLD)
    gate_vals = np.concatenate(all_gates).flatten() if all_gates else np.array([])

    metrics = compute_metrics(all_labels, pred_prob, pred_class)
    metrics['loss'] = total_loss / len(dataloader)
    metrics['mean_gate'] = float(np.mean(gate_vals)) if len(gate_vals) > 0 else 0.0
    return metrics, pred_prob, pred_class, gate_vals


def train_model(model, train_loader, val_loader, device,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR,
                weight_decay=WEIGHT_DECAY, fold_idx=None):
    criterion = build_criterion(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_auc = -np.inf
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
# Single-fold training pipeline
# ============================================================================
def process_single_fold(fold_info, case_ids, patch_list_all,
                        clinical_df, categories_all, fold_idx, output_dir,
                        device=None, multi_gpu=False):
    if device is None:
        device = DEVICE

    print(f"\n{'=' * 60}")
    print(f"  Processing fold {fold_idx} [GPU: {device}]")
    print(f"{'=' * 60}")

    train_cases = fold_info['train_cases']
    val_cases = fold_info['val_cases']
    case_to_idx = {cid: i for i, cid in enumerate(case_ids)}
    train_indices = [case_to_idx[c] for c in train_cases if c in case_to_idx]
    val_indices = [case_to_idx[c] for c in val_cases if c in case_to_idx]

    y_train_bin = to_binary_labels(categories_all[train_indices])
    y_val_bin = to_binary_labels(categories_all[val_indices])
    folds_dir = os.path.join(output_dir, 'folds')

    train_patches = [patch_list_all[i] for i in train_indices]
    val_patches = [patch_list_all[i] for i in val_indices]

    train_clin_df = clinical_df[clinical_df['case_id'].isin(train_cases)].reset_index(drop=True)
    val_clin_df = clinical_df[clinical_df['case_id'].isin(val_cases)].reset_index(drop=True)
    clin_pre = ClinicalPreprocessor()
    clin_pre.fit(train_clin_df)
    clin_pre.save(folds_dir, fold_idx)
    train_clin_map = dict(zip(train_clin_df['case_id'], clin_pre.transform(train_clin_df)))
    val_clin_map = dict(zip(val_clin_df['case_id'], clin_pre.transform(val_clin_df)))
    X_train_clin = np.array([train_clin_map[c] for c in train_cases if c in train_clin_map])
    X_val_clin = np.array([val_clin_map[c] for c in val_cases if c in val_clin_map])
    clinical_dim = clin_pre.output_dim

    lasso_sel_fold = LassoClinicalSelector(enable=ENABLE_LASSO_SELECTION)
    X_train_clin = lasso_sel_fold.fit_transform(X_train_clin, y_train_bin)
    X_val_clin = lasso_sel_fold.transform(X_val_clin)
    lasso_sel_fold.save(folds_dir, fold_idx)
    clinical_dim = lasso_sel_fold.selected_dim

    _collate_fn = padded_collate_fn if multi_gpu else balanced_collate_fn

    inner_train_cases, inner_val_cases = split_inner_train_val(train_cases, y_train_bin)
    inner_train_indices = [case_to_idx[c] for c in inner_train_cases if c in case_to_idx]
    inner_val_indices = [case_to_idx[c] for c in inner_val_cases if c in case_to_idx]
    y_inner_train_bin = to_binary_labels(categories_all[inner_train_indices])
    y_inner_val_bin = to_binary_labels(categories_all[inner_val_indices])

    X_train_clin_by_case = {c: X_train_clin[i] for i, c in enumerate(train_cases)
                            if c in train_clin_map}
    inner_train_patches = [patch_list_all[i] for i in inner_train_indices]
    inner_val_patches = [patch_list_all[i] for i in inner_val_indices]
    X_inner_train_clin = np.array([X_train_clin_by_case[c] for c in inner_train_cases
                                   if c in X_train_clin_by_case])
    X_inner_val_clin = np.array([X_train_clin_by_case[c] for c in inner_val_cases
                                 if c in X_train_clin_by_case])

    inner_train_dataset = CaseDataset(inner_train_cases, inner_train_patches,
                                      X_inner_train_clin, y_inner_train_bin)
    inner_val_dataset = CaseDataset(inner_val_cases, inner_val_patches,
                                    X_inner_val_clin, y_inner_val_bin)
    inner_train_loader = DataLoader(inner_train_dataset, batch_size=BATCH_SIZE,
                                    shuffle=True, collate_fn=_collate_fn,
                                    generator=torch.Generator().manual_seed(SEED))
    inner_val_loader = DataLoader(inner_val_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False, collate_fn=_collate_fn)
    val_dataset = CaseDataset(val_cases, val_patches, X_val_clin, y_val_bin)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=_collate_fn)

    print(f"\n  [Contrastive fusion: ContrastiveFusionModel] training ...")
    model = ContrastiveFusionModel(
        clinical_input_dim=clinical_dim,
        common_dim=COMMON_EMBED_DIM,
        n_classes=N_CLASSES,
        fusion_hidden=FUSION_HIDDEN_DIM,
        dropout_prob=DROPOUT_PROB,
        seed=SEED
    ).to(device)
    if multi_gpu:
        model = wrap_model_for_multi_gpu(model)
    model, best_epoch, best_val_auc = train_model(
        model, inner_train_loader, inner_val_loader, device, fold_idx=fold_idx
    )
    model_state = (model.module.state_dict()
                   if isinstance(model, nn.DataParallel) else model.state_dict())
    torch.save(model_state,
               os.path.join(folds_dir, f'fusion_model_fold_{fold_idx}.pth'))

    criterion = build_criterion(device)
    train_metrics, _, _, _ = evaluate(model, inner_train_loader, criterion, device)
    train_auc = train_metrics['auc']
    auc_gap = train_auc - best_val_auc

    model.eval()
    val_metrics, val_prob, val_pred, val_gate = evaluate(model, val_loader, criterion, device)
    val_gate_dict = ({cid: float(val_gate[i]) for i, cid in enumerate(val_cases)}
                     if len(val_gate) > 0 else {})

    pred_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    pd.DataFrame({
        'case_id': val_cases,
        'true_label': [CLASSES[l] for l in y_val_bin],
        'pred_label': [CLASSES[p] for p in val_pred],
        'prob_C': val_prob[:, 0],
        'prob_B': val_prob[:, 1],
        'contrast_loss': [val_gate_dict.get(c, 0.0) for c in val_cases],
    }).to_csv(os.path.join(pred_dir,
                            f'fusion_val_prediction_fold_{fold_idx}.csv'),
              index=False)

    fold_summary = {
        'fold': fold_idx,
        'best_epoch': best_epoch,
        'train_auc': float(train_auc),
        'val_auc': float(val_metrics['auc']),
        'auc_gap': float(auc_gap),
        'patch_dim': PATCH_FEAT_DIM,
        'clinical_dim': clinical_dim,
        'mean_gate': float(np.mean(val_gate)) if len(val_gate) > 0 else 0.0,
        'accuracy': float(val_metrics['accuracy']),
        'auc': float(val_metrics['auc']),
        'sensitivity': float(val_metrics['sensitivity']),
        'specificity': float(val_metrics['specificity']),
        'f1': float(val_metrics['f1']),
        'confusion_matrix': val_metrics['confusion_matrix'].tolist(),
    }

    print(f"\n  Fold {fold_idx} result:")
    print(f"    Val AUC: {val_metrics['auc']:.4f} | "
          f"Sens: {val_metrics['sensitivity']:.4f} | "
          f"Spec: {val_metrics['specificity']:.4f}")
    print(f"    AUC gap: {auc_gap:.4f}")

    with open(os.path.join(folds_dir, f'fusion_fold_summary_{fold_idx}.pkl'),
              'wb') as f:
        pickle.dump(fold_summary, f)
    return fold_summary


# ============================================================================
# Post-processing: threshold optimization
# ============================================================================
def search_optimal_threshold(y_true, y_prob_pos):
    """Youden-index-maximizing threshold search on the model's own OOF."""
    thresholds = np.arange(0.01, 0.99, 0.001)
    best_thr, best_youden, best_sens, best_spec = 0.5, -1.0, 0.0, 0.0
    for thr in thresholds:
        y_pred = (y_prob_pos >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden = sens + spec - 1.0
        if youden > best_youden:
            best_youden = youden
            best_thr = thr
            best_sens, best_spec = sens, spec
    return {'threshold': best_thr, 'sensitivity': best_sens,
            'specificity': best_spec, 'youden': best_youden}


def optimize_model_threshold(model_key, oof_y_true, oof_y_prob):
    """Per-model threshold optimization on the model's own OOF predictions."""
    res = search_optimal_threshold(np.asarray(oof_y_true), np.asarray(oof_y_prob))
    MODEL_THRESHOLDS[model_key] = res
    return res


def get_model_threshold(model_key):
    """Fetch the per-model threshold; fall back to GLOBAL_THRESHOLD."""
    cfg = MODEL_THRESHOLDS.get(model_key)
    return cfg['threshold'] if cfg is not None else GLOBAL_THRESHOLD


def split_inner_train_val(train_cases, y_train_bin, frac=NN_INNER_VAL_FRAC,
                          seed=SEED):
    """Split the training fold into an inner 80/20 for NN early stopping."""
    try:
        inner_train, inner_val = train_test_split(
            train_cases, test_size=frac, random_state=seed, stratify=y_train_bin
        )
    except ValueError:
        inner_train, inner_val = train_test_split(
            train_cases, test_size=frac, random_state=seed
        )
    return list(inner_train), list(inner_val)


def select_lr_best_C(X_train, y_train, cv=5, C_grid=None, seed=SEED):
    """Inner 5-fold CV inside the training fold to select LR regularization C."""
    if C_grid is None:
        C_grid = LR_C_GRID
    n_samples = X_train.shape[0]
    if n_samples < 2 * cv or len(np.unique(y_train)) < 2:
        return 1.0
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    best_c, best_auc = C_grid[0], -1.0
    for C in C_grid:
        inner_aucs = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            model = LogisticRegression(
                penalty='l1', C=C, solver='saga',
                max_iter=LASSO_CV_MAX_ITER, random_state=seed,
                class_weight=None
            )
            model.fit(X_train[tr_idx], y_train[tr_idx])
            inner_aucs.append(safe_auc(
                y_train[va_idx],
                model.predict_proba(X_train[va_idx])[:, 1]))
        mean_auc = float(np.mean(inner_aucs))
        if mean_auc > best_auc:
            best_auc, best_c = mean_auc, C
    return best_c


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
# External test inference (main model: image-only linear LR)
# ============================================================================
def process_external_test(final_results_summary, threshold_config=None,
                          output_dir=None, main_threshold=None,
                          sl_dir=None, fy_dir=None):
    print(f"\n{'=' * 60}")
    print(f"  External test inference (main model: image-only linear LR)")
    print(f"{'=' * 60}")
    if main_threshold is None:
        main_threshold = get_model_threshold('main')

    sl_patch_dir = os.path.join(sl_dir, 'SL-patch_features')
    case_ids_sl, patch_list_sl, categories_sl = load_patch_features(sl_patch_dir)

    print("\n  [Class filter] Training set (SL): excluding class A ...")
    case_ids_sl, patch_list_sl, categories_sl = filter_class_a_data(
        case_ids_sl, patch_list_sl, categories_sl)

    y_sl_bin = to_binary_labels(categories_sl)

    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    stat_features_sl = compute_mean_pooled_features(patch_list_sl)
    final_stat_pre = BranchA_StatPreprocessor(seed=SEED)
    final_stat_pre.fit(stat_features_sl)
    final_stat_pre.save_final(models_dir, 'final')
    X_sl_stat = final_stat_pre.transform(stat_features_sl)

    print("\n  Training image-only linear LR (L1 LogisticRegression) ...")
    best_C = select_lr_best_C(X_sl_stat, y_sl_bin)
    print(f"    Inner CV selected C = {best_C:.4g}")
    lr = LogisticRegression(penalty='l1', C=best_C, solver='saga',
                            max_iter=LASSO_CV_MAX_ITER, random_state=SEED,
                            class_weight=None)
    lr.fit(X_sl_stat, y_sl_bin)
    with open(os.path.join(models_dir, 'final_linear_lr.pkl'), 'wb') as f:
        pickle.dump(lr, f)
    print(f"  Training complete. Non-zero features: {np.sum(lr.coef_ != 0)}")

    fy_patch_dir = os.path.join(fy_dir, 'FY-patch_features')
    fy_case_ids, fy_patch_list, fy_categories = load_patch_features(fy_patch_dir)

    print("\n  [Class filter] External test set (FY): excluding class A ...")
    fy_case_ids, fy_patch_list, fy_categories = filter_class_a_data(
        fy_case_ids, fy_patch_list, fy_categories)
    fy_y_bin = to_binary_labels(fy_categories)

    fy_stat_features = compute_mean_pooled_features(fy_patch_list)
    fy_stat_reduced = final_stat_pre.transform(fy_stat_features)

    print("\n  External test inference ...")
    all_prob = lr.predict_proba(fy_stat_reduced)
    all_pred = prob_to_class(all_prob[:, 1], main_threshold)
    all_true = fy_y_bin

    metrics = compute_metrics(all_true, all_prob, all_pred)

    auc_mean, auc_lower, auc_upper = auc_bca_ci(all_true, all_prob[:, 1])

    ext_prob = all_prob[:, 1]
    ext_brier = compute_brier_score(all_true, ext_prob)
    ext_brier_ci = bca_ci_general(all_true, ext_prob,
        lambda yt, yp: compute_brier_score(yt, yp), N_BOOTSTRAP, 0.05, SEED)

    ext_cat_ci = classification_metric_cis(
        all_true, ext_prob, threshold=main_threshold,
        n_bootstrap=N_BOOTSTRAP, alpha=0.05, seed=SEED)
    ext_f1_ci = bca_ci_general(all_true, ext_prob,
        lambda yt, yp: f1_score(yt, prob_to_class(yp, main_threshold)),
        N_BOOTSTRAP, 0.05, SEED)

    print(f"\n  External test results (main model OOF threshold = "
          f"{main_threshold:.4f}):")
    print(f"    AUC: {auc_mean:.4f} (95% CI: {auc_lower:.4f} - {auc_upper:.4f})")
    print(f"    Accuracy: {metrics['accuracy']:.4f} | "
          f"Sensitivity: {metrics['sensitivity']:.4f} | "
          f"Specificity: {metrics['specificity']:.4f}")
    print(f"    Brier: {ext_brier:.4g} "
          f"(95% CI: {ext_brier_ci[1]:.4g}-{ext_brier_ci[2]:.4g})")

    pred_dir = os.path.join(output_dir, 'predictions')
    pd.DataFrame({
        'case_id': fy_case_ids,
        'true_label': [CLASSES[l] for l in all_true],
        f'pred_label_{main_threshold:.2f}': [CLASSES[p] for p in all_pred],
        'prob_C': all_prob[:, 0],
        'prob_B': all_prob[:, 1],
    }).to_csv(os.path.join(pred_dir, 'main_external_predictions.csv'), index=False)

    full_state = {
        'ext_metrics': metrics,
        'ext_auc_ci': (auc_mean, auc_lower, auc_upper),
        'ext_brier_ci': ext_brier_ci,
        'ext_cat_ci': ext_cat_ci,
        'ext_f1_ci': ext_f1_ci,
        'final_stat_pre': final_stat_pre,
        'stat_dim': final_stat_pre.output_dim,
        'fy_case_ids': fy_case_ids,
        'fy_stat_reduced': fy_stat_reduced,
        'fy_y_bin': fy_y_bin,
        'fusion_preds': {'prob': all_prob, 'class': all_pred, 'true': all_true},
        'threshold': main_threshold,
    }
    with open(os.path.join(pred_dir, 'external_test_state.pkl'), 'wb') as f:
        pickle.dump(full_state, f)
    return full_state


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
        'baseline_clinical_mlp': (
            Baseline_ClinicalOnlyMLP,
            'Clinical only (MLP)'
        ),
        'baseline_direct_concat': (
            MedPatchFusionModel,
            'Clinical + patch MedPatch confidence fusion'
        ),
        'baseline_thyfusionnet': (
            ThyFusionNetFusionModel,
            'Clinical + patch ThyFusionNet gated fusion'
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
        fold_probs, fold_trues, fold_ids_list = [], [], []

        for fold_info in tqdm(folds, desc=f"    {display_name}", unit="fold"):
            fold_idx = fold_info['fold']
            train_cases = fold_info['train_cases']
            val_cases = fold_info['val_cases']
            train_indices = [case_to_idx[c] for c in train_cases if c in case_to_idx]
            val_indices = [case_to_idx[c] for c in val_cases if c in case_to_idx]
            y_train_bin = to_binary_labels(categories_all[train_indices])
            y_val_bin = to_binary_labels(categories_all[val_indices])

            clin_pre = ClinicalPreprocessor().load(folds_dir, fold_idx)
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

            inner_train_cases, inner_val_cases = split_inner_train_val(
                train_cases, y_train_bin)
            inner_train_indices = [case_to_idx[c] for c in inner_train_cases
                                   if c in case_to_idx]
            inner_val_indices = [case_to_idx[c] for c in inner_val_cases
                                 if c in case_to_idx]
            y_inner_train_bin = to_binary_labels(categories_all[inner_train_indices])
            y_inner_val_bin = to_binary_labels(categories_all[inner_val_indices])

            train_clin_cases = [c for c in train_cases if c in train_clin_map]
            X_train_clin_by_case = dict(zip(train_clin_cases, X_train_clin))
            inner_train_patches = [patch_list_all[i] for i in inner_train_indices]
            inner_val_patches = [patch_list_all[i] for i in inner_val_indices]
            X_inner_train_clin = np.array(
                [X_train_clin_by_case[c] for c in inner_train_cases
                 if c in X_train_clin_by_case])
            X_inner_val_clin = np.array(
                [X_train_clin_by_case[c] for c in inner_val_cases
                 if c in X_train_clin_by_case])

            inner_train_dataset = CaseDataset(inner_train_cases,
                                              inner_train_patches,
                                              X_inner_train_clin,
                                              y_inner_train_bin)
            inner_val_dataset = CaseDataset(inner_val_cases,
                                            inner_val_patches,
                                            X_inner_val_clin,
                                            y_inner_val_bin)
            inner_train_loader = DataLoader(
                inner_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                collate_fn=COLLATE_FN,
                generator=torch.Generator().manual_seed(SEED))
            inner_val_loader = DataLoader(
                inner_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                collate_fn=COLLATE_FN)
            val_dataset = CaseDataset(val_cases, val_patches, X_val_clin, y_val_bin)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                    shuffle=False, collate_fn=COLLATE_FN)

            model = ModelClass(clinical_input_dim=clinical_dim, seed=SEED).to(DEVICE)
            if USE_MULTI_GPU:
                model = wrap_model_for_multi_gpu(model)
            model, best_epoch, best_auc = train_model(
                model, inner_train_loader, inner_val_loader, DEVICE,
                fold_idx=fold_idx)
            val_metrics, val_prob, val_class, _ = evaluate(
                model, val_loader, build_criterion(), DEVICE)

            fold_results.append({
                'fold': fold_idx, 'val_auc': val_metrics['auc'],
                'val_f1': val_metrics['f1'],
                'sensitivity': val_metrics['sensitivity'],
                'specificity': val_metrics['specificity']
            })
            oof_y_true.extend(y_val_bin)
            oof_y_prob.extend(val_prob[:, 1])
            oof_case_ids.extend(val_cases)
            fold_probs.append(np.asarray(val_prob[:, 1]))
            fold_trues.append(np.asarray(y_val_bin))
            fold_ids_list.append(list(val_cases))

        oof_yt = np.array(oof_y_true)
        oof_yp = np.array(oof_y_prob)
        mean_auc = np.mean([r['val_auc'] for r in fold_results])
        print(f"    Mean Val AUC: {mean_auc:.4f}")

        thr_res = optimize_model_threshold(key, oof_yt, oof_yp)
        model_thr = get_model_threshold(key)
        print(f"    [Threshold] {display_name}: inner-OOF optimal threshold = "
              f"{model_thr:.4f} (Youden={thr_res['youden']:.4f}, "
              f"Sens={thr_res['sensitivity']:.4f}, "
              f"Spec={thr_res['specificity']:.4f})")

        for i, r in enumerate(fold_results):
            yt_i, pr_i = fold_trues[i], fold_probs[i]
            pred_i = prob_to_class(pr_i, model_thr)
            tn, fp, fn, tp = confusion_matrix(yt_i, pred_i, labels=[0, 1]).ravel()
            r['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            r['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            r['val_f1'] = f1_score(yt_i, pred_i)
            r['threshold'] = model_thr

        all_results[key] = {
            'display_name': display_name,
            'fold_results': fold_results,
            'mean_auc': mean_auc,
            'threshold': model_thr
        }
        oof_preds[key] = {
            'case_ids': oof_case_ids,
            'y_true': oof_yt,
            'y_prob': oof_yp,
            'display_name': display_name,
            'threshold': model_thr,
        }
        oof_ci_results[key] = compute_oof_metrics_with_ci(
            oof_yt, oof_yp, threshold=model_thr)

    # Clinical-only baseline: sklearn LASSO LogisticRegression
    print(f"\n  --- Clinical only (LASSO LogisticRegression) ---")
    fold_results_clin = []
    oof_y_true_clin = []
    oof_y_prob_clin = []
    oof_case_ids_clin = []
    fold_probs_clin, fold_trues_clin = [], []

    for fold_info in tqdm(folds, desc="    Clinical only (LASSO LR)", unit="fold"):
        fold_idx = fold_info['fold']
        train_cases = fold_info['train_cases']
        val_cases = fold_info['val_cases']
        train_indices = [case_to_idx[c] for c in train_cases if c in case_to_idx]
        val_indices = [case_to_idx[c] for c in val_cases if c in case_to_idx]
        y_train_bin = to_binary_labels(categories_all[train_indices])
        y_val_bin = to_binary_labels(categories_all[val_indices])

        clin_pre = ClinicalPreprocessor().load(folds_dir, fold_idx)
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

        best_C = select_lr_best_C(X_train_clin_sel, y_train_bin)
        lr_model = LogisticRegression(
            penalty='l1', C=best_C, solver='saga',
            max_iter=LASSO_CV_MAX_ITER, random_state=SEED, class_weight=None
        )
        lr_model.fit(X_train_clin_sel, y_train_bin)
        y_val_prob = lr_model.predict_proba(X_val_clin_sel)
        val_auc = safe_auc(y_val_bin, y_val_prob[:, 1])

        fold_results_clin.append({'fold': fold_idx, 'val_auc': val_auc})
        fold_probs_clin.append(np.asarray(y_val_prob[:, 1]))
        fold_trues_clin.append(np.asarray(y_val_bin))
        oof_y_true_clin.extend(y_val_bin)
        oof_y_prob_clin.extend(y_val_prob[:, 1])
        oof_case_ids_clin.extend(val_cases)

    mean_auc_clin = np.mean([r['val_auc'] for r in fold_results_clin])
    print(f"    Mean Val AUC: {mean_auc_clin:.4f}")

    thr_res_clin = optimize_model_threshold('baseline_clinical_only',
                                            np.array(oof_y_true_clin),
                                            np.array(oof_y_prob_clin))
    model_thr_clin = get_model_threshold('baseline_clinical_only')
    print(f"    [Threshold] Clinical only (LASSO LR): "
          f"inner-OOF optimal threshold = {model_thr_clin:.4f} "
          f"(Youden={thr_res_clin['youden']:.4f}, "
          f"Sens={thr_res_clin['sensitivity']:.4f}, "
          f"Spec={thr_res_clin['specificity']:.4f})")

    for i, r in enumerate(fold_results_clin):
        yt_i, pr_i = fold_trues_clin[i], fold_probs_clin[i]
        pred_i = prob_to_class(pr_i, model_thr_clin)
        tn, fp, fn, tp = confusion_matrix(yt_i, pred_i, labels=[0, 1]).ravel()
        r['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        r['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        r['val_f1'] = f1_score(yt_i, pred_i)
        r['threshold'] = model_thr_clin

    all_results['baseline_clinical_only'] = {
        'display_name': 'Clinical only (LASSO LR)',
        'fold_results': fold_results_clin,
        'mean_auc': mean_auc_clin,
        'threshold': model_thr_clin
    }
    oof_preds['baseline_clinical_only'] = {
        'case_ids': oof_case_ids_clin,
        'y_true': np.array(oof_y_true_clin),
        'y_prob': np.array(oof_y_prob_clin),
        'display_name': 'Clinical only (LASSO LR)',
        'threshold': model_thr_clin,
    }
    oof_ci_results['baseline_clinical_only'] = compute_oof_metrics_with_ci(
        np.array(oof_y_true_clin), np.array(oof_y_prob_clin),
        threshold=model_thr_clin)

    # Contrastive fusion (downgraded to a baseline): reuse its OOF predictions
    print(f"\n  --- Clinical + patch contrastive fusion (baseline) ---")
    fold_results_fusion = []
    oof_y_true_fusion = []
    oof_y_prob_fusion = []
    oof_case_ids_fusion = []
    fold_probs_fusion, fold_trues_fusion = [], []
    pred_dir = os.path.join(output_dir, 'predictions')
    for fold_idx in range(N_FOLDS):
        df = pd.read_csv(os.path.join(
            pred_dir, f'fusion_val_prediction_fold_{fold_idx}.csv'))
        oof_y_true_fusion.extend((df['true_label'] == POSITIVE_CLASS).astype(int).tolist())
        oof_y_prob_fusion.extend(df['prob_B'].tolist())
        oof_case_ids_fusion.extend(df['case_id'].tolist())
        fold_probs_fusion.append(np.asarray(df['prob_B'].tolist(), dtype=float))
        fold_trues_fusion.append((df['true_label'] == POSITIVE_CLASS).astype(int).values)
        sum_path = os.path.join(folds_dir, f'fusion_fold_summary_{fold_idx}.pkl')
        if os.path.exists(sum_path):
            with open(sum_path, 'rb') as f:
                fr = pickle.load(f)
            fold_results_fusion.append({
                'fold': fold_idx,
                'val_auc': fr.get('val_auc', float('nan')),
            })

    mean_auc_fusion = (np.nanmean([r['val_auc'] for r in fold_results_fusion])
                       if fold_results_fusion else float('nan'))
    print(f"    Mean Val AUC: {mean_auc_fusion:.4f}")

    thr_res_fusion = optimize_model_threshold('baseline_fusion',
                                              np.array(oof_y_true_fusion),
                                              np.array(oof_y_prob_fusion))
    model_thr_fusion = get_model_threshold('baseline_fusion')
    print(f"    [Threshold] Clinical + patch contrastive fusion: "
          f"inner-OOF optimal threshold = {model_thr_fusion:.4f} "
          f"(Youden={thr_res_fusion['youden']:.4f}, "
          f"Sens={thr_res_fusion['sensitivity']:.4f}, "
          f"Spec={thr_res_fusion['specificity']:.4f})")

    for i, r in enumerate(fold_results_fusion):
        yt_i, pr_i = fold_trues_fusion[i], fold_probs_fusion[i]
        pred_i = prob_to_class(pr_i, model_thr_fusion)
        tn, fp, fn, tp = confusion_matrix(yt_i, pred_i, labels=[0, 1]).ravel()
        r['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        r['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        r['val_f1'] = f1_score(yt_i, pred_i)
        r['threshold'] = model_thr_fusion

    all_results['baseline_fusion'] = {
        'display_name': 'Clinical + patch contrastive fusion',
        'fold_results': fold_results_fusion,
        'mean_auc': mean_auc_fusion,
        'threshold': model_thr_fusion
    }
    oof_preds['baseline_fusion'] = {
        'case_ids': oof_case_ids_fusion,
        'y_true': np.array(oof_y_true_fusion),
        'y_prob': np.array(oof_y_prob_fusion),
        'display_name': 'Clinical + patch contrastive fusion',
        'threshold': model_thr_fusion,
    }
    oof_ci_results['baseline_fusion'] = compute_oof_metrics_with_ci(
        np.array(oof_y_true_fusion), np.array(oof_y_prob_fusion),
        threshold=model_thr_fusion)

    with open(os.path.join(output_dir, 'baselines', 'baselines_cv.pkl'),
              'wb') as f:
        pickle.dump({'results': all_results, 'oof_preds': oof_preds,
                     'oof_ci_results': oof_ci_results}, f)
    return all_results, oof_preds, oof_ci_results


# ============================================================================
# Comprehensive Excel summary
# ============================================================================
def _generate_comprehensive_summary(main_oof, ext_state, baseline_cv_results,
                                     baseline_oof_preds,
                                     baseline_oof_ci_results=None,
                                     main_oof_metrics_ci=None,
                                     output_dir=None):
    """Generate the comprehensive comparison Excel (internal CV + external test)."""
    import openpyxl

    def _fmt_ci(ci_tuple):
        if ci_tuple is None or np.isnan(ci_tuple[1]):
            return ''
        return f'{round_to_4sf(ci_tuple[1])}-{round_to_4sf(ci_tuple[2])}'

    def _build_oof_row(name, oof_dict, oof_metrics_ci=None):
        thr = oof_dict.get('threshold')
        if thr is None:
            thr = GLOBAL_THRESHOLD
        oof_m = compute_metrics(
            oof_dict['y_true'],
            np.column_stack([1 - oof_dict['y_prob'], oof_dict['y_prob']]),
            prob_to_class(oof_dict['y_prob'], thr)
        )
        row = {
            'Model': name, 'Source': 'Internal OOF',
            'AUC': round_to_4sf(safe_auc(oof_dict['y_true'], oof_dict['y_prob'])),
            'AUC_95CI': '',
            'Sensitivity': round_to_4sf(oof_m['sensitivity']),
            'Specificity': round_to_4sf(oof_m['specificity']),
            'F1': round_to_4sf(oof_m['f1']),
            'Accuracy': round_to_4sf(oof_m['accuracy']),
            'PPV': round_to_4sf(oof_m['ppv']),
            'NPV': round_to_4sf(oof_m['npv']),
        }
        if oof_metrics_ci is None:
            oof_metrics_ci = compute_oof_metrics_with_ci(
                oof_dict['y_true'], oof_dict['y_prob'], threshold=thr)
        row['AUC_95CI'] = _fmt_ci(oof_metrics_ci.get('auc_ci'))
        row['Brier'] = round_to_4sf(oof_metrics_ci.get('brier', float('nan')))
        row['Brier_95CI'] = _fmt_ci(oof_metrics_ci.get('brier_ci'))
        row['Sensitivity_95CI'] = _fmt_ci(oof_metrics_ci.get('sensitivity_ci'))
        row['Specificity_95CI'] = _fmt_ci(oof_metrics_ci.get('specificity_ci'))
        row['Accuracy_95CI'] = _fmt_ci(oof_metrics_ci.get('accuracy_ci'))
        row['F1_95CI'] = _fmt_ci(oof_metrics_ci.get('f1_ci'))
        row['PPV_95CI'] = _fmt_ci(oof_metrics_ci.get('ppv_ci'))
        row['NPV_95CI'] = _fmt_ci(oof_metrics_ci.get('npv_ci'))
        return row

    rows_internal = []
    rows_external = []

    # --- Internal validation ---
    cv_csv_path = os.path.join(output_dir, 'predictions', '10fold_cv_summary.csv')
    if os.path.exists(cv_csv_path):
        cv_table = pd.read_csv(cv_csv_path)
        numeric = cv_table.iloc[:-1]
        aucs = numeric['Val_AUC'].astype(float)
        sens = numeric['Sensitivity'].astype(float)
        specs = numeric['Specificity'].astype(float)
        f1s = numeric['F1'].astype(float)
        accs = numeric['Accuracy'].astype(float)
        rows_internal.append({
            'Model': 'Image only linear LR (main, per-fold mean)',
            'Source': 'Internal 10-fold CV',
            'AUC': round_to_4sf(aucs.mean()),
            'Sensitivity': round_to_4sf(sens.mean()),
            'Specificity': round_to_4sf(specs.mean()),
            'F1': round_to_4sf(f1s.mean()),
            'Accuracy': round_to_4sf(accs.mean()),
        })

    if main_oof_metrics_ci is None:
        main_oof_metrics_ci = compute_oof_metrics_with_ci(
            main_oof['y_true'], main_oof['y_prob'])
    rows_internal.append(_build_oof_row(
        'Image only linear LR (main, OOF)', main_oof, main_oof_metrics_ci))

    baseline_order = [
        ('baseline_fusion', 'Clinical + patch contrastive fusion (baseline)'),
        ('baseline_clinical_only', 'Clinical only LASSO LR (baseline)'),
        ('baseline_stat_only', 'Image patch features only (baseline)'),
        ('baseline_clinical_mlp', 'Clinical only MLP (baseline)'),
        ('baseline_direct_concat', 'Clinical + patch MedPatch confidence fusion (baseline)'),
        ('baseline_thyfusionnet', 'Clinical + patch ThyFusionNet gated fusion (baseline)'),
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
            'Sensitivity': (round_to_4sf(np.nanmean(fold_sens))
                            if not all(np.isnan(fold_sens)) else float('nan')),
            'Specificity': (round_to_4sf(np.nanmean(fold_specs))
                            if not all(np.isnan(fold_specs)) else float('nan')),
            'F1': (round_to_4sf(np.nanmean(fold_f1s))
                   if not all(np.isnan(fold_f1s)) else float('nan')),
        })

    for key, name in baseline_order:
        if key in baseline_oof_preds:
            ci = baseline_oof_ci_results.get(key) if baseline_oof_ci_results else None
            rows_internal.append(_build_oof_row(name, baseline_oof_preds[key], ci))

    # --- External test ---
    def _build_ext_row(name, ext_metrics, auc_ci, brier_ci=None,
                       cat_ci=None, f1_ci=None):
        row = {
            'Model': name, 'Source': 'External test',
            'AUC': round_to_4sf(auc_ci[0]),
            'Sensitivity': round_to_4sf(ext_metrics['sensitivity']),
            'Specificity': round_to_4sf(ext_metrics['specificity']),
            'F1': round_to_4sf(ext_metrics['f1']),
            'Accuracy': round_to_4sf(ext_metrics['accuracy']),
            'PPV': round_to_4sf(ext_metrics.get('ppv', float('nan'))),
            'NPV': round_to_4sf(ext_metrics.get('npv', float('nan'))),
            'AUC_95CI': _fmt_ci(auc_ci),
        }
        if brier_ci is not None:
            row['Brier'] = round_to_4sf(brier_ci[0])
            row['Brier_95CI'] = _fmt_ci(brier_ci)
        if f1_ci is not None:
            row['F1_95CI'] = _fmt_ci(f1_ci)
        if cat_ci is not None:
            row['Sensitivity_95CI'] = _fmt_ci(cat_ci.get('sensitivity_ci'))
            row['Specificity_95CI'] = _fmt_ci(cat_ci.get('specificity_ci'))
            row['PPV_95CI'] = _fmt_ci(cat_ci.get('ppv_ci'))
            row['NPV_95CI'] = _fmt_ci(cat_ci.get('npv_ci'))
            row['Accuracy_95CI'] = _fmt_ci(cat_ci.get('accuracy_ci'))
        return row

    ext_metrics = ext_state['ext_metrics']
    ext_auc_ci = ext_state['ext_auc_ci']
    ext_brier_ci = ext_state.get('ext_brier_ci')
    ext_cat_ci = ext_state.get('ext_cat_ci')
    ext_f1_ci = ext_state.get('ext_f1_ci')
    rows_external.append(_build_ext_row(
        'Image only linear LR (main)', ext_metrics, ext_auc_ci,
        ext_brier_ci, ext_cat_ci, ext_f1_ci))

    pred_dir = os.path.join(output_dir, 'predictions')
    xlsx_path = os.path.join(pred_dir, 'all_models_comprehensive_summary.xlsx')

    cols_int = ['Model', 'Source', 'AUC', 'AUC_95CI',
                'Sensitivity', 'Sensitivity_95CI',
                'Specificity', 'Specificity_95CI',
                'F1', 'F1_95CI',
                'Accuracy', 'Accuracy_95CI',
                'PPV', 'PPV_95CI', 'NPV', 'NPV_95CI',
                'Brier', 'Brier_95CI']
    cols_ext = list(cols_int)

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        if rows_internal:
            df_int = pd.DataFrame(rows_internal)
            df_int[[c for c in cols_int if c in df_int.columns]].to_excel(
                writer, sheet_name='Internal', index=False)
        if rows_external:
            df_ext = pd.DataFrame(rows_external)
            df_ext[[c for c in cols_ext if c in df_ext.columns]].to_excel(
                writer, sheet_name='External', index=False)

    wb = openpyxl.load_workbook(xlsx_path)
    for ws in wb.worksheets:
        for col_letter in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                           'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB', 'AC', 'AD',
                           'AE', 'AF']:
            ws.column_dimensions[col_letter].width = 20
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row,
                                min_col=4, max_col=ws.max_column):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = '0.0000'
    wb.save(xlsx_path)


def plot_external_figures(ext_state, output_dir):
    """External test set: ROC / confusion matrix / decision curve analysis."""
    plt.rcParams['axes.unicode_minus'] = False
    figures_dir = os.path.join(output_dir, 'figures')

    main_name = 'Image only linear LR (main)'
    main_prob = ext_state['fusion_preds']['prob'][:, 1]
    y_true = ext_state['fusion_preds']['true']
    main_auc, main_ci_l, main_ci_u = ext_state['ext_auc_ci']

    model_data = {main_name: main_prob}

    # === ROC curve ===
    fig, ax = plt.subplots(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_true, main_prob)
    ax.plot(fpr, tpr, color='#d7191c', linestyle='-', lw=2,
            label=f'{main_name} (AUC={main_auc:.3f}, '
                  f'95%CI {main_ci_l:.3f}-{main_ci_u:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve on External Test Set '
                 f'(class {POSITIVE_CLASS})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, frameon=True, edgecolor='gray')
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
        y_pred = prob_to_class(prob, ext_state['threshold'])
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=vmax)
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
                 f'(threshold={ext_state["threshold"]:.3f})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir,
                              'external_confusion_matrix_comparison.tiff'),
                dpi=1000, bbox_inches='tight')
    plt.close()
    print("  External confusion matrix figure saved")

    # === Decision curve analysis ===
    fig, ax = plt.subplots(figsize=(10, 8))
    thresholds = np.arange(0.01, 1.00, 0.01)
    prevalence = y_true.mean()
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    treat_none = np.zeros_like(thresholds)
    ax.plot(thresholds, treat_all, 'k--', lw=1.8, label='Treat all')
    ax.plot(thresholds, treat_none, 'k:', lw=1.8, label='Treat none')

    _, nb = calculate_dca(y_true, main_prob, thresholds)
    ax.plot(thresholds, nb, color='#d7191c', linestyle='-', lw=2,
            label=main_name)

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


def plot_internal_oof_roc(main_oof, main_oof_metrics_ci, baseline_oof_preds,
                          baseline_oof_ci_results=None, output_dir=None):
    """Internal OOF summary ROC (main + baselines), matching external format."""
    plt.rcParams['axes.unicode_minus'] = False
    figures_dir = os.path.join(output_dir, 'figures')

    def _model_threshold(oof_dict):
        thr = oof_dict.get('threshold')
        return thr if thr is not None else GLOBAL_THRESHOLD

    def _operating_point(yt, yp, thr):
        y_pred = prob_to_class(yp, thr)
        tn, fp, fn, tp = confusion_matrix(yt, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return sens, spec

    oof_data = []
    main_thr = _model_threshold(main_oof)
    main_sens, main_spec = _operating_point(
        np.asarray(main_oof['y_true']), np.asarray(main_oof['y_prob']), main_thr)
    oof_data.append((
        'Image only linear LR (main)',
        np.asarray(main_oof['y_true']),
        np.asarray(main_oof['y_prob']),
        main_oof_metrics_ci.get('auc_ci'),
        main_thr, main_sens, main_spec,
    ))

    baseline_order = [
        ('baseline_fusion', 'Clinical + patch contrastive fusion (baseline)'),
        ('baseline_clinical_only', 'Clinical only LASSO LR (baseline)'),
        ('baseline_stat_only', 'Image patch features only (baseline)'),
        ('baseline_clinical_mlp', 'Clinical only MLP (baseline)'),
        ('baseline_direct_concat', 'Clinical + patch MedPatch confidence fusion (baseline)'),
        ('baseline_thyfusionnet', 'Clinical + patch ThyFusionNet gated fusion (baseline)'),
    ]
    for key, name in baseline_order:
        if key not in baseline_oof_preds:
            continue
        oof = baseline_oof_preds[key]
        ci = None
        if baseline_oof_ci_results and key in baseline_oof_ci_results:
            ci = baseline_oof_ci_results[key].get('auc_ci')
        yt = np.asarray(oof['y_true'])
        yp = np.asarray(oof['y_prob'])
        thr = _model_threshold(oof)
        sens, spec = _operating_point(yt, yp, thr)
        oof_data.append((name, yt, yp, ci, thr, sens, spec))

    colors = ['#d7191c', '#2c7bb6', '#1a9641', '#f9913a', '#6a3d9a',
              '#d01c8b', '#fdae61', '#39568c', '#dfc27d', '#000000']

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (name, yt, yp, auc_ci, thr, sens, spec) in enumerate(oof_data):
        fpr, tpr, _ = roc_curve(yt, yp)
        color = colors[i % len(colors)]
        if auc_ci is not None:
            label = (f'{name} (AUC={auc_ci[0]:.3f}, '
                     f'95%CI {auc_ci[1]:.3f}-{auc_ci[2]:.3f})\n'
                     f'Thr={thr:.3f}, Sens={sens:.3f}, Spec={spec:.3f}')
        else:
            label = (f'{name} (AUC={safe_auc(yt, yp):.3f})\n'
                     f'Thr={thr:.3f}, Sens={sens:.3f}, Spec={spec:.3f}')
        ax.plot(fpr, tpr, color=color, linestyle='-', lw=2, label=label)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve on Internal OOF Summary '
                 f'(per-model OOF thresholds, class {POSITIVE_CLASS})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, frameon=True, edgecolor='gray')
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'internal_roc_comparison.tiff'),
                dpi=1000, bbox_inches='tight')
    plt.close()
    print("  Internal OOF ROC figure saved (TIFF 1000 dpi)")


# ============================================================================
# Multiprocessing helpers for fold-level parallelism
# ============================================================================
def _fold_worker_from_disk(args):
    """Worker entry: load data from disk and process one fold on a specific GPU."""
    fold_info, fold_idx, gpu_id, output_dir, sl_dir, sl_clinical_file = args

    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    print(f"\n[Worker] Fold {fold_idx} -> GPU {gpu_id} "
          f"({torch.cuda.get_device_name(gpu_id)})")

    sl_patch_dir = os.path.join(sl_dir, 'SL-patch_features')
    case_ids, patch_list_all, categories_all = load_patch_features(sl_patch_dir)
    case_ids, patch_list_all, categories_all = filter_class_a_data(
        case_ids, patch_list_all, categories_all)
    clinical_df = load_clinical_features(sl_clinical_file)

    return process_single_fold(
        fold_info, case_ids, patch_list_all,
        clinical_df, categories_all, fold_idx,
        output_dir=output_dir, device=device, multi_gpu=False
    )


def _run_folds_parallel(tasks, n_workers):
    """Parallel fold training using multiprocessing."""
    import multiprocessing as _mp
    try:
        _mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"  Starting {n_workers} GPU workers for {len(tasks)} folds ...")
    print(f"  Each worker loads data independently (disk I/O); "
          f"this may take a while ...")

    with _mp.Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_fold_worker_from_disk, tasks),
            total=len(tasks), desc="  10-fold training (parallel)", unit="fold"
        ))

    return results


# ============================================================================
# Command-line interface
# ============================================================================
def build_parser():
    parser = argparse.ArgumentParser(
        prog='contrastive_fusion_binary',
        description='WSI + clinical features contrastive multimodal fusion '
                    'classification pipeline (binary B vs C, A excluded).',
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

    parser.add_argument('--n-folds', type=int, default=N_FOLDS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--n-bootstrap', type=int, default=N_BOOTSTRAP)
    parser.add_argument('--no-multi-gpu', action='store_true',
                        help='Disable DataParallel even if multiple GPUs are available.')
    parser.add_argument('--no-threshold-optimization', action='store_true',
                        help='Disable per-model threshold optimization (use 0.5).')
    parser.add_argument('--no-lasso', action='store_true',
                        help='Disable LASSO clinical feature selection.')
    return parser


def main():
    global GLOBAL_THRESHOLD, N_FOLDS, SEED, BATCH_SIZE, MAX_EPOCHS, PATIENCE
    global N_BOOTSTRAP, ENABLE_LASSO_SELECTION, ENABLE_THRESHOLD_OPTIMIZATION
    global USE_MULTI_GPU

    args = build_parser().parse_args()

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
    print("  WSI + clinical features contrastive multimodal fusion pipeline")
    print(f"  Global seed: {SEED} | Started: "
          f"{t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  GPU mode: {'fold-level parallel' if N_GPUS > 1 else 'single GPU / CPU'} "
          f"| Available GPUs: {N_GPUS}")
    print("=" * 70)

    # Step 1: Load data
    step_start = datetime.now()
    print("\n[Step 1] Loading training patch features ...")
    sl_patch_dir = os.path.join(SL_DIR, 'SL-patch_features')
    case_ids, patch_list_all, categories_all = load_patch_features(sl_patch_dir)
    clinical_df = load_clinical_features(SL_CLINICAL_FILE)

    print("\n  [Class filter] Training set: excluding class A ...")
    case_ids, patch_list_all, categories_all = filter_class_a_data(
        case_ids, patch_list_all, categories_all)
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Step 2: 10-fold split
    step_start = datetime.now()
    print("\n[Step 2] Generating stratified 10-fold split ...")
    folds = load_or_create_folds(case_ids, categories_all, OUTPUT_DIR)
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Steps 3-8: Contrastive fusion (downgraded to a baseline)
    step_start = datetime.now()
    print("\n[Steps 3-8] Contrastive fusion (baseline) 10-fold CV ...")
    all_fusion_fold_results = []
    skipped, pending = [], []
    for fold_info in folds:
        fold_idx = fold_info['fold']
        summary_path = os.path.join(OUTPUT_DIR, 'folds',
                                    f'fusion_fold_summary_{fold_idx}.pkl')
        pred_csv_path = os.path.join(OUTPUT_DIR, 'predictions',
                                     f'fusion_val_prediction_fold_{fold_idx}.csv')
        if os.path.exists(summary_path) and os.path.exists(pred_csv_path):
            skipped.append(fold_idx)
        else:
            pending.append(fold_info)

    if skipped:
        print(f"  [Resume] Completed folds: {sorted(skipped)}")
        for fold_idx in sorted(skipped):
            with open(os.path.join(OUTPUT_DIR, 'folds',
                                   f'fusion_fold_summary_{fold_idx}.pkl'),
                      'rb') as f:
                all_fusion_fold_results.append(pickle.load(f))

    if pending:
        if N_GPUS > 1:
            n_workers = min(N_GPUS, len(pending))
            tasks = [(fold_info, fold_info['fold'], i % N_GPUS, OUTPUT_DIR,
                      SL_DIR, SL_CLINICAL_FILE)
                     for i, fold_info in enumerate(pending)]
            results = _run_folds_parallel(tasks, n_workers)
            all_fusion_fold_results.extend(results)
        else:
            for fold_info in tqdm(pending, desc="  10-fold training", unit="fold"):
                fold_idx = fold_info['fold']
                fold_result = process_single_fold(
                    fold_info, case_ids, patch_list_all,
                    clinical_df, categories_all, fold_idx, OUTPUT_DIR
                )
                all_fusion_fold_results.append(fold_result)

    all_fusion_fold_results.sort(key=lambda x: x['fold'])

    fusion_oof_y_true = []
    fusion_oof_y_prob = []
    fusion_oof_case_ids = []
    for fold_idx in range(N_FOLDS):
        df = pd.read_csv(os.path.join(
            OUTPUT_DIR, 'predictions',
            f'fusion_val_prediction_fold_{fold_idx}.csv'))
        fusion_oof_case_ids.extend(df['case_id'].tolist())
        fusion_oof_y_true.extend(
            (df['true_label'] == POSITIVE_CLASS).astype(int).tolist())
        fusion_oof_y_prob.extend(df['prob_B'].tolist())
    fusion_oof = {
        'case_ids': fusion_oof_case_ids,
        'y_true': np.array(fusion_oof_y_true),
        'y_prob': np.array(fusion_oof_y_prob),
        'display_name': 'Clinical + patch contrastive fusion (baseline)'
    }

    fusion_oof_ci = compute_oof_metrics_with_ci(fusion_oof['y_true'],
                                                fusion_oof['y_prob'])
    print("\n  [OOF metrics] Contrastive fusion (baseline) "
          "Brier Score / 95% CI ...")
    print(f"    Brier: {fusion_oof_ci['brier']:.4g} "
          f"(95% CI: {fusion_oof_ci['brier_ci'][1]:.4g}-"
          f"{fusion_oof_ci['brier_ci'][2]:.4g})")

    # Main model: image-only linear LR (mean pooling + StandardScaler + L1 LR)
    print("\n[Steps 3-8] Main model (image-only linear LR) 10-fold CV ...")
    main_fold_results = []
    main_oof_y_true = []
    main_oof_y_prob = []
    main_oof_case_ids = []
    case_to_idx = {cid: i for i, cid in enumerate(case_ids)}
    for fold_info in tqdm(folds, desc="  Main model (image-only linear LR)",
                          unit="fold"):
        fold_idx = fold_info['fold']
        train_cases = fold_info['train_cases']
        val_cases = fold_info['val_cases']
        train_indices = [case_to_idx[c] for c in train_cases if c in case_to_idx]
        val_indices = [case_to_idx[c] for c in val_cases if c in case_to_idx]
        y_train_bin = to_binary_labels(categories_all[train_indices])
        y_val_bin = to_binary_labels(categories_all[val_indices])

        train_patches = [patch_list_all[i] for i in train_indices]
        val_patches = [patch_list_all[i] for i in val_indices]
        X_train_mean = compute_mean_pooled_features(train_patches)
        X_val_mean = compute_mean_pooled_features(val_patches)
        scaler = StandardScaler()
        X_train_stat = scaler.fit_transform(X_train_mean)
        X_val_stat = scaler.transform(X_val_mean)

        best_C = select_lr_best_C(X_train_stat, y_train_bin)
        lr_main = LogisticRegression(penalty='l1', C=best_C, solver='saga',
                                     max_iter=LASSO_CV_MAX_ITER,
                                     random_state=SEED, class_weight=None)
        lr_main.fit(X_train_stat, y_train_bin)
        y_val_prob = lr_main.predict_proba(X_val_stat)
        y_val_class = prob_to_class(y_val_prob[:, 1], GLOBAL_THRESHOLD)
        val_auc = safe_auc(y_val_bin, y_val_prob[:, 1])

        pred_dir = os.path.join(OUTPUT_DIR, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        pd.DataFrame({
            'case_id': val_cases,
            'true_label': [CLASSES[l] for l in y_val_bin],
            'pred_label': [CLASSES[p] for p in y_val_class],
            'prob_C': 1.0 - y_val_prob[:, 1],
            'prob_B': y_val_prob[:, 1],
        }).to_csv(os.path.join(pred_dir, f'val_prediction_fold_{fold_idx}.csv'),
                  index=False)

        main_fold_results.append({
            'fold': fold_idx, 'val_auc': val_auc, 'best_C': best_C,
            'sensitivity': float(np.mean(y_val_class[y_val_bin == 1] == 1))
                if np.sum(y_val_bin == 1) > 0 else 0.0,
            'specificity': float(np.mean(y_val_class[y_val_bin == 0] == 0))
                if np.sum(y_val_bin == 0) > 0 else 0.0,
            'val_f1': f1_score(y_val_bin, y_val_class),
        })
        main_oof_y_true.extend(y_val_bin)
        main_oof_y_prob.extend(y_val_prob[:, 1])
        main_oof_case_ids.extend(val_cases)

    main_oof = {
        'case_ids': main_oof_case_ids,
        'y_true': np.array(main_oof_y_true),
        'y_prob': np.array(main_oof_y_prob),
        'display_name': 'Image only linear LR (main)'
    }
    main_oof_ci = compute_oof_metrics_with_ci(main_oof['y_true'], main_oof['y_prob'])

    print(f"\n  Main model 10-fold CV summary:")
    auc_mean, ci_l, ci_u = auc_bca_ci(main_oof['y_true'], main_oof['y_prob'])
    print(f"    OOF AUC: {auc_mean:.4f} (95% BCa CI: {ci_l:.4f} - {ci_u:.4f})")
    print(f"    10-fold summary saved: predictions/10fold_cv_summary.csv")
    print(f"  Elapsed: {datetime.now() - step_start}")

    # Threshold optimization on the main model's own OOF
    threshold_config = None
    if ENABLE_THRESHOLD_OPTIMIZATION:
        step_start = datetime.now()
        print("\n[Post-processing] Main model inner-OOF threshold search "
              "(Youden maximization) ...")
        oof_prob = main_oof['y_prob']
        oof_true = main_oof['y_true']

        thr_result = optimize_model_threshold('main', oof_true, oof_prob)
        GLOBAL_THRESHOLD = thr_result['threshold']
        main_oof['threshold'] = GLOBAL_THRESHOLD
        threshold_config = {
            'optimal_threshold': GLOBAL_THRESHOLD,
            'youden_index': thr_result['youden'],
            'sensitivity': thr_result['sensitivity'],
            'specificity': thr_result['specificity'],
        }
        print(f"    Youden maximization: main model inner-OOF optimal "
              f"threshold = {GLOBAL_THRESHOLD:.4f}, "
              f"sensitivity = {thr_result['sensitivity']:.4f}, "
              f"specificity = {thr_result['specificity']:.4f}, "
              f"Youden = {thr_result['youden']:.4f}")
        with open(os.path.join(OUTPUT_DIR, 'models', 'threshold_config.pkl'),
                  'wb') as f:
            pickle.dump(threshold_config, f)
        print(f"  Elapsed: {datetime.now() - step_start}")
    else:
        main_oof['threshold'] = GLOBAL_THRESHOLD
        print(f"  [Post-processing] Threshold optimization disabled; "
              f"main model uses default threshold {GLOBAL_THRESHOLD:.4f}")

    main_oof_ci = compute_oof_metrics_with_ci(
        main_oof['y_true'], main_oof['y_prob'], threshold=main_oof['threshold'])

    # Export 10-fold CV summary
    fold_idx_to_res = {r['fold']: r for r in main_fold_results}
    cv_rows = []
    for fold_idx in range(N_FOLDS):
        df = pd.read_csv(os.path.join(OUTPUT_DIR, 'predictions',
                                       f'val_prediction_fold_{fold_idx}.csv'))
        yt = (df['true_label'] == POSITIVE_CLASS).astype(int).values
        prob = df['prob_B'].values
        pred = prob_to_class(prob, GLOBAL_THRESHOLD)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        fr = fold_idx_to_res.get(fold_idx, {})
        cv_rows.append({
            'Fold': fold_idx,
            'Best_C': fr.get('best_C', ''),
            'Val_AUC': safe_auc(yt, prob),
            'Accuracy': accuracy_score(yt, pred),
            'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'F1': f1_score(yt, pred),
        })
    cv_summary_df = pd.DataFrame(cv_rows)
    cv_summary_df.loc['Mean+-Std'] = [
        'Mean+-Std', '',
        f"{cv_summary_df['Val_AUC'].mean():.4f}+-{cv_summary_df['Val_AUC'].std():.4f}",
        f"{cv_summary_df['Accuracy'].mean():.4f}+-{cv_summary_df['Accuracy'].std():.4f}",
        f"{cv_summary_df['Sensitivity'].mean():.4f}+-{cv_summary_df['Sensitivity'].std():.4f}",
        f"{cv_summary_df['Specificity'].mean():.4f}+-{cv_summary_df['Specificity'].std():.4f}",
        f"{cv_summary_df['F1'].mean():.4f}+-{cv_summary_df['F1'].std():.4f}",
    ]
    cv_summary_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions',
                                       '10fold_cv_summary.csv'), index=False)
    print(f"    10-fold summary saved: predictions/10fold_cv_summary.csv "
          f"(threshold = {GLOBAL_THRESHOLD:.4f})")

    # Step 9: External test
    step_start = datetime.now()
    print("\n[Step 9] External test inference (main model) ...")
    ext_state_path = os.path.join(OUTPUT_DIR, 'predictions',
                                   'external_test_state.pkl')
    if os.path.exists(ext_state_path):
        print("  Existing external test results found, loading ...")
        with open(ext_state_path, 'rb') as f:
            ext_state = pickle.load(f)
    else:
        ext_state = process_external_test(
            all_fusion_fold_results, threshold_config=threshold_config,
            output_dir=OUTPUT_DIR, sl_dir=SL_DIR, fy_dir=FY_DIR)
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
        baseline_oof_ci_results = {}
        for key, oof in baseline_oof_preds.items():
            thr = oof.get('threshold')
            if thr is None:
                thr = GLOBAL_THRESHOLD
            baseline_oof_ci_results[key] = compute_oof_metrics_with_ci(
                oof['y_true'], oof['y_prob'], threshold=thr)
        print(f"    Re-computed baseline OOF CIs using each model's own "
              f"OOF threshold")
    else:
        baseline_cv_results, baseline_oof_preds, baseline_oof_ci_results = \
            run_baselines_cv(folds, patch_list_all, clinical_df,
                             categories_all, case_ids, OUTPUT_DIR)

    plot_external_figures(ext_state, OUTPUT_DIR)
    plot_internal_oof_roc(main_oof, main_oof_ci, baseline_oof_preds,
                          baseline_oof_ci_results=baseline_oof_ci_results,
                          output_dir=OUTPUT_DIR)

    print(f"  Elapsed: {datetime.now() - step_start}")

    print("\n[Summary] Generating comprehensive comparison Excel ...")
    _generate_comprehensive_summary(main_oof, ext_state, baseline_cv_results,
                                     baseline_oof_preds,
                                     baseline_oof_ci_results=baseline_oof_ci_results,
                                     main_oof_metrics_ci=main_oof_ci,
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
