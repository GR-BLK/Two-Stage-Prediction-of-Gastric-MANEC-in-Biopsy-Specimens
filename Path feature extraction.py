#!/usr/bin/env python3
"""UNI2 patch-level feature extraction.

Given a directory where each sub-directory corresponds to a case
(patient / slide) and contains 224x224 PNG patches, this script runs the
UNI2 vision transformer over every patch and writes:

    <output-dir>/<case>/<case>_features.npz   # features + file_paths
    <output-dir>/<case>/<case>_labels.csv     # per-patch labels
    <output-dir>/case_summary.csv             # per-case summary
    <output-dir>/all_patches_features.npy     # aggregated features
    <output-dir>/all_patches_labels.csv       # aggregated labels

This repository does NOT ship UNI2 weights or any patient data. Users must
obtain the weights separately and comply with their license and any
applicable medical-data regulations.

License: <SPDX-License-Identifier>  (see LICENSE in the repo root)
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Must be set before importing torch so MPS fallback actually takes effect.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

from PIL import Image
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


UNI_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
UNI_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PNG_SUFFIX = '.png'
FEATURES_SUFFIX = '_features.npz'
LABELS_SUFFIX = '_labels.csv'

logger = logging.getLogger('extract_features')


# ---------------------------------------------------------------------------
# Model definition and loading
# ---------------------------------------------------------------------------

class SwiGLUMlp(nn.Module):
    """SwiGLU MLP used by UNI2's ViT blocks."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.SiLU, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 4)
        self.fc1 = nn.Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x, gate = self.fc1(x).chunk(2, dim=-1)
        x = x * self.act(gate)
        x = self.fc2(x)
        return x


def resolve_device(preference: str) -> torch.device:
    """Return the torch device to use, honoring an explicit preference."""
    if preference == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    if preference == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    if preference == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but not available')
    return torch.device(preference)


def load_uni2_model(model_path: str, device: torch.device) -> nn.Module:
    """Instantiate the UNI2 ViT and load the given state dict."""
    model = VisionTransformer(
        img_size=224,
        patch_size=14,
        embed_dim=1536,
        depth=24,
        num_heads=16,
        mlp_layer=SwiGLUMlp,
        mlp_ratio=8 / 3,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
        reg_tokens=8,
        act_layer=nn.SiLU,
        no_embed_class=True,
    )

    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

    # The UNI2 weights use a no_embed_class layout: pos_embed only covers the
    # patch tokens, while cls/reg tokens have their own parameters. If the
    # loaded pos_embed shape differs, interpolate only the patch-token part.
    if 'pos_embed' in state_dict and 'pos_embed' in model.state_dict():
        model_pos_embed = model.state_dict()['pos_embed']
        loaded_pos_embed = state_dict['pos_embed']
        if loaded_pos_embed.shape != model_pos_embed.shape:
            state_dict['pos_embed'] = torch.nn.functional.interpolate(
                loaded_pos_embed.permute(0, 2, 1),
                size=model_pos_embed.shape[1],
                mode='linear',
            ).permute(0, 2, 1)

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        logger.warning('  Missing keys (first 5): %s', load_result.missing_keys[:5])
    if load_result.unexpected_keys:
        logger.warning('  Unexpected keys (first 5): %s', load_result.unexpected_keys[:5])
    if not load_result.missing_keys and not load_result.unexpected_keys:
        logger.info('  Weight loading: fully matched (no missing / unexpected keys)')

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def filter_zero_features(features, file_paths):
    """Drop patches whose per-dim std is <= 1e-6 (near-constant features).

    If every patch would be dropped, keep the first one to avoid producing an
    empty feature file.
    """
    if len(features) <= 1:
        return features, list(file_paths)
    non_zero_mask = np.std(features, axis=1) > 1e-6
    if not np.any(non_zero_mask):
        return features[[0]], [file_paths[0]]
    kept_paths = [p for p, keep in zip(file_paths, non_zero_mask) if keep]
    return features[non_zero_mask], kept_paths


def parse_case_id(wsi_name: str) -> str:
    """Extract case_id from a WSI/slide name: substring before the first '_'."""
    return wsi_name.split('_')[0] if wsi_name else ''


# ---------------------------------------------------------------------------
# Per-case extraction
# ---------------------------------------------------------------------------

def list_case_folders(input_dir: Path) -> list[str]:
    return sorted(d.name for d in input_dir.iterdir() if d.is_dir())


def list_png_files(case_dir: Path) -> list[str]:
    return sorted(
        str(p) for p in case_dir.iterdir()
        if p.is_file() and p.name.endswith(PNG_SUFFIX) and not p.name.startswith('.')
    )


def case_is_complete(out_case_dir: Path, case_name: str) -> bool:
    """Check that both the features npz and the labels csv exist and are non-empty."""
    npz_path = out_case_dir / f'{case_name}{FEATURES_SUFFIX}'
    csv_path = out_case_dir / f'{case_name}{LABELS_SUFFIX}'
    if not npz_path.is_file() or not csv_path.is_file():
        return False
    try:
        with np.load(npz_path, allow_pickle=True) as f:
            if 'features' not in f or f['features'].shape[0] == 0:
                return False
    except Exception:
        return False
    return True


def extract_case_features(
    case_name: str,
    png_files: list[str],
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    failed_files: list[tuple[str, str]],
) -> tuple[np.ndarray | None, list[str]]:
    """Run inference over all patches of a single case.

    Returns (features, kept_file_paths), or (None, []) if no patch was processed.
    """
    n_patches = len(png_files)
    current_features: list[np.ndarray] = []
    current_file_paths: list[str] = []

    last_update_time = time.time()
    progress_done = 0

    for b_start in range(0, n_patches, batch_size):
        batch_paths = png_files[b_start:b_start + batch_size]
        batch_tensors = []
        batch_valid_paths = []

        for patch_path in batch_paths:
            try:
                with Image.open(patch_path) as im:
                    img = im.convert('RGB')
            except Exception as exc:
                failed_files.append((patch_path, repr(exc)))
                continue
            if img.size != (224, 224):
                img = img.resize((224, 224), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - UNI_MEAN) / UNI_STD
            batch_tensors.append(
                torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)
            )
            batch_valid_paths.append(patch_path)

        if batch_tensors:
            batch = torch.stack(batch_tensors).to(device)
            with torch.inference_mode():
                feats = model(batch).cpu().numpy()
            current_features.append(feats)
            current_file_paths.extend(batch_valid_paths)

        progress_done = min(b_start + batch_size, n_patches)
        now = time.time()
        if progress_done == n_patches or now - last_update_time >= 3:
            last_update_time = now
            pct = (progress_done / n_patches) * 100
            bar_len = 30
            filled = int(bar_len * progress_done / n_patches)
            bar = '█' * filled + '░' * (bar_len - filled)
            logger.info('    [%s] %5d/%d (%5.1f%%)',
                        bar, progress_done, n_patches, pct)

    if not current_features:
        return None, []

    features = np.vstack(current_features)
    features, kept_paths = filter_zero_features(features, current_file_paths)
    if len(features) == 0:
        return None, []
    return features, kept_paths


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_all_patches(output_dir: Path) -> None:
    """Concatenate per-case patch features into one document, streaming to disk.

    Per-case outputs are still preserved; this only produces a single flat
    ``all_patches_features.npy`` + ``all_patches_labels.csv`` for downstream use.
    """
    case_dirs = sorted(
        d for d in output_dir.iterdir()
        if d.is_dir() and any(
            f.is_file() and f.name.endswith(FEATURES_SUFFIX) for f in d.iterdir()
        )
    )

    entries: list[tuple[str, str, int]] = []  # (case_name, npz_name, count)
    total_patches = 0
    feature_dim: int | None = None

    for case_dir in case_dirs:
        for f in sorted(case_dir.iterdir()):
            if not (f.is_file() and f.name.endswith(FEATURES_SUFFIX)):
                continue
            with np.load(f, allow_pickle=True) as data:
                features = data['features']
                if feature_dim is None:
                    feature_dim = features.shape[1]
                total_patches += features.shape[0]
                entries.append((case_dir.name, f.name, features.shape[0]))

    if not entries or total_patches == 0 or feature_dim is None:
        logger.info('  [SKIP] aggregate_all_patches: no feature files found.')
        return

    feat_path = output_dir / 'all_patches_features.npy'
    label_path = output_dir / 'all_patches_labels.csv'

    mm = np.lib.format.open_memmap(
        str(feat_path), mode='w+', dtype=np.float32,
        shape=(total_patches, feature_dim),
    )

    rows = 0
    label_rows: list[dict] = []
    for case_name, npz_name, count in entries:
        npz_path = output_dir / case_name / npz_name
        with np.load(npz_path, allow_pickle=True) as data:
            features = data['features']
            mm[rows:rows + count] = features.astype(np.float32)
        case_id = parse_case_id(case_name)
        label_rows.extend(
            {'slide_id': case_name, 'case_id': case_id, 'category': ''}
            for _ in range(count)
        )
        rows += count

    mm.flush()
    del mm

    pd.DataFrame(label_rows).to_csv(label_path, index=False)

    logger.info('  ✓ all_patches_features.npy (%s patches x %d dims)',
                f'{total_patches:,}', feature_dim)
    logger.info('  ✓ all_patches_labels.csv (%s rows)', f'{len(label_rows):,}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='extract_features',
        description='Extract UNI2 patch-level features for WSI cases.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--input-dir', required=True, type=Path,
        help='Directory containing one sub-directory per case, each with *.png patches.',
    )
    parser.add_argument(
        '--output-dir', required=True, type=Path,
        help='Directory to write per-case features, labels, and aggregated outputs.',
    )
    parser.add_argument(
        '--model-path', required=True, type=Path,
        help='Path to the UNI2 state dict (e.g. pytorch_model.bin).',
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Number of patches per forward pass.',
    )
    parser.add_argument(
        '--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto',
        help='Device to run inference on.',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Re-extract a case even if its output already exists.',
    )
    parser.add_argument(
        '--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
        help='Logging verbosity.',
    )
    return parser


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(message)s',
        stream=os.sys.stdout,
    )


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.log_level)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    model_path: Path = args.model_path

    if not input_dir.is_dir():
        logger.error('Input directory not found: %s', input_dir)
        return 1
    if not model_path.is_file():
        logger.error('Model weights not found: %s', model_path)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('=' * 80)
    logger.info('UNI2 patch feature extraction')
    logger.info('=' * 80)
    logger.info('Input:      %s', input_dir)
    logger.info('Output:     %s', output_dir)
    logger.info('Model:      %s', model_path)
    logger.info('Batch size: %d', args.batch_size)

    logger.info('')
    logger.info('[1/4] Loading UNI2 model ...')
    device = resolve_device(args.device)
    logger.info('  Device: %s', device)
    model = load_uni2_model(str(model_path), device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info('  Model parameters: %.2fB', param_count / 1e9)

    logger.info('')
    logger.info('[2/4] Scanning cases ...')
    case_folders = list_case_folders(input_dir)
    total_cases = len(case_folders)
    logger.info('  Found %d cases', total_cases)
    if total_cases == 0:
        logger.warning('No case sub-directories found under %s', input_dir)
        return 0

    logger.info('  Counting patches ...')
    total_patches_all = 0
    for name in case_folders:
        total_patches_all += len(list_png_files(input_dir / name))
    logger.info('  Total patches to process: %d', total_patches_all)

    logger.info('')
    logger.info('[3/4] Extracting features ...')
    logger.info('-' * 80)

    start_time = time.time()
    case_summaries: list[dict] = []
    failed_files: list[tuple[str, str]] = []

    for case_idx, case_name in enumerate(case_folders):
        case_dir = input_dir / case_name
        out_case_dir = output_dir / case_name
        out_case_dir.mkdir(parents=True, exist_ok=True)

        png_files = list_png_files(case_dir)
        n_patches = len(png_files)
        case_out_path = out_case_dir / f'{case_name}{FEATURES_SUFFIX}'

        logger.info('')
        logger.info('  === [%d/%d] %s (%d patches) ===',
                    case_idx + 1, total_cases, case_name, n_patches)

        # Resume support: skip if outputs already exist and are non-empty.
        if not args.overwrite and case_is_complete(out_case_dir, case_name):
            with np.load(case_out_path, allow_pickle=True) as f:
                resume_count = int(f['features'].shape[0])
            case_summaries.append({
                'case_id': parse_case_id(case_name),
                'slide_id': case_name,
                'category': '',
                'valid_patch_count': resume_count,
            })
            logger.info('  [SKIP] %s (already completed, %d patches)',
                        case_name, resume_count)
            continue

        features, kept_paths = extract_case_features(
            case_name=case_name,
            png_files=png_files,
            model=model,
            device=device,
            batch_size=args.batch_size,
            failed_files=failed_files,
        )

        if features is not None and len(features) > 0:
            np.savez_compressed(case_out_path, features=features, file_paths=kept_paths)
            case_id = parse_case_id(case_name)
            labels = pd.DataFrame({
                'patch_path': kept_paths,
                'slide_id': [case_name] * len(kept_paths),
                'case_id': [case_id] * len(kept_paths),
            })
            labels.to_csv(out_case_dir / f'{case_name}{LABELS_SUFFIX}', index=False)

            case_summaries.append({
                'case_id': case_id,
                'slide_id': case_name,
                'category': '',
                'valid_patch_count': len(features),
            })
            logger.info('  ✓ Completed: %d valid patches', len(features))
        else:
            case_summaries.append({
                'case_id': parse_case_id(case_name),
                'slide_id': case_name,
                'category': '',
                'valid_patch_count': 0,
            })
            logger.error('  ✗ Failed: no valid patches for %s', case_name)

    elapsed_total = time.time() - start_time
    logger.info('')
    logger.info('  Feature extraction elapsed: %02d:%02d:%02d',
                int(elapsed_total // 3600),
                int((elapsed_total % 3600) // 60),
                int(elapsed_total % 60))

    logger.info('')
    logger.info('=' * 80)
    logger.info('Saving per-case summary ...')
    logger.info('=' * 80)
    if case_summaries:
        pd.DataFrame(case_summaries).to_csv(
            output_dir / 'case_summary.csv', index=False)
        logger.info('  ✓ case_summary.csv')

    if failed_files:
        with open(output_dir / 'failed_files.log', 'w', encoding='utf-8') as fh:
            for path, err in failed_files:
                fh.write(f'{path}\t{err}\n')
        logger.warning('  ! failed_files.log (%d entries)', len(failed_files))

    logger.info('')
    logger.info('  Aggregating all cases into one document ...')
    aggregate_all_patches(output_dir)

    total_valid = sum(c['valid_patch_count'] for c in case_summaries)
    logger.info('')
    logger.info('=' * 80)
    logger.info('Feature extraction completed!')
    logger.info('  Total cases: %d', total_cases)
    logger.info('  Total valid patches: %d', total_valid)
    logger.info('  Output: %s', output_dir)
    logger.info('=' * 80)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())