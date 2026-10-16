#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""WSI image processing pipeline: patch extraction -> filtering -> stain normalization.

This script implements a three-stage pipeline for whole-slide images (WSI):

  1. Patch extraction    : read SVS + GeoJSON polygons, cut 224x224 patches
                           that fall inside annotated regions.
  2. Patch filtering     : drop background / hemorrhage / necrosis / blurry
                           patches using color and Laplacian-variance heuristics.
  3. Stain normalization : Macenko-style normalization using a reference
                           template built from a set of reference patches.

Both CPU (numpy / OpenCV) and GPU (PyTorch / CUDA) implementations are
provided. The GPU path uses all visible CUDA devices and is numerically
equivalent to the CPU path.

This repository ships no data and no reference templates. Users must provide
their own SVS, GeoJSON, and reference patch directories, and must ensure that
any patient data is handled in accordance with applicable ethics and privacy
regulations.

License: MIT (see LICENSE in the repo root).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: F401  (kept for downstream compatibility)
import openslide
from PIL import Image
from numpy.linalg import svd, norm, pinv

try:
    import cv2
except ImportError:
    cv2 = None  # CPU filtering needs OpenCV; GPU path uses torch instead.

# Fixed seeds for reproducible reference-template construction and OD sampling.
random.seed(2026)
np.random.seed(2026)

logger = logging.getLogger('wsi_image_processing')


# =====================================================================
# Logging helpers
# =====================================================================

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(message)s',
        stream=sys.stdout,
    )


def progress_write(msg: str) -> None:
    """In-place progress line (uses carriage return, no trailing newline)."""
    sys.stdout.write('\r' + msg)
    sys.stdout.flush()


# =====================================================================
# Stage 1: Patch extraction (CPU)
# =====================================================================

def read_geojson(geojson_path):
    with open(geojson_path, 'r') as f:
        return json.load(f)


def get_polygon_coordinates(geojson_data):
    features = geojson_data.get('features', [])
    polygons = []
    for feature in features:
        geometry = feature.get('geometry', {})
        geom_type = geometry.get('type')
        coords = geometry.get('coordinates', [])
        if geom_type == 'Polygon' and coords:
            polygon = [(int(round(p[0])), int(round(p[1]))) for p in coords[0]]
            polygons.append(polygon)
        elif geom_type == 'MultiPolygon' and coords:
            for poly in coords:
                if poly:
                    polygon = [(int(round(p[0])), int(round(p[1]))) for p in poly[0]]
                    polygons.append(polygon)
    return polygons


def point_in_polygon(x, y, polygon):
    n = len(polygon)
    inside = False
    for i in range(n):
        j = (i + 1) % n
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
    return inside


def get_polygon_bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def extract_patches(svs_path, geojson_path, output_base_dir, patch_size=224, level=0,
                    output_basename=None, skip_existing=True):
    slide = openslide.OpenSlide(svs_path)
    geojson_data = read_geojson(geojson_path)
    polygons = get_polygon_coordinates(geojson_data)

    svs_basename = os.path.splitext(os.path.basename(svs_path))[0]
    if output_basename is None:
        output_basename = svs_basename
    output_dir = os.path.join(output_base_dir, output_basename)
    os.makedirs(output_dir, exist_ok=True)

    if not polygons:
        slide.close()
        return 0

    existing_patches = set()
    if skip_existing and os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith('.png'):
                existing_patches.add(f)

    total_patches = len(existing_patches)

    for poly_idx, polygon in enumerate(polygons):
        bbox = get_polygon_bbox(polygon)
        min_x, min_y, max_x, max_y = bbox

        # GeoJSON / read_region use level-0 coordinates.
        slide_width, slide_height = slide.level_dimensions[0]

        start_x = max(0, min_x)
        start_y = max(0, min_y)
        end_x = min(slide_width, max_x)
        end_y = min(slide_height, max_y)

        step_x = patch_size
        step_y = patch_size

        for y in range(start_y, end_y - patch_size + 1, step_y):
            for x in range(start_x, end_x - patch_size + 1, step_x):
                center_x = x + patch_size // 2
                center_y = y + patch_size // 2

                if point_in_polygon(center_x, center_y, polygon):
                    patch_filename = f'{svs_basename}_poly{poly_idx}_{x}_{y}.png'

                    if skip_existing and patch_filename in existing_patches:
                        continue

                    region = slide.read_region((x, y), level, (patch_size, patch_size))
                    region_rgb = region.convert('RGB')

                    patch_path = os.path.join(output_dir, patch_filename)
                    region_rgb.save(patch_path)

                    total_patches += 1

    slide.close()
    return total_patches


def batch_extract_category(category_dir, output_category_dir, category_name,
                           patch_size=224, level=0):
    svs_files = sorted([f for f in os.listdir(category_dir)
                        if f.endswith('.svs') and not f.startswith('._')])
    geojson_files = sorted([f for f in os.listdir(category_dir)
                            if f.endswith('.geojson') and not f.startswith('._')])

    svs_basenames = {os.path.splitext(f)[0] for f in svs_files}
    geojson_basenames = {os.path.splitext(f)[0] for f in geojson_files}
    paired_basenames = sorted(svs_basenames & geojson_basenames)

    total_files = len(paired_basenames)
    logger.info('  Found %d paired SVS-GeoJSON files in %s', total_files, category_name)

    total_patches_all = 0

    for i, basename in enumerate(paired_basenames):
        svs_path = os.path.join(category_dir, f'{basename}.svs')
        geojson_path = os.path.join(category_dir, f'{basename}.geojson')

        try:
            patch_count = extract_patches(svs_path, geojson_path,
                                          output_category_dir, patch_size, level)
            total_patches_all += patch_count
        except Exception as e:
            logger.error('    Error processing %s: %s', basename, e)

        progress = (i + 1) / total_files
        bar_length = 30
        filled = int(progress * bar_length)
        bar = '#' * filled + '-' * (bar_length - filled)
        progress_write(f'    {i+1}/{total_files} [{bar}] {progress*100:.1f}%')

    sys.stdout.write('\n')
    logger.info('    Total patches extracted: %d', total_patches_all)
    return total_patches_all


def batch_extract(source_base_dir, output_base_dir, categories, patch_size=224, level=0):
    total_patches_all = 0
    start_time = time.time()

    logger.info('Batch Patch Extraction')
    logger.info('Source: %s', source_base_dir)
    logger.info('Output: %s', output_base_dir)
    logger.info('=' * 60)

    for category in categories:
        category_dir = os.path.join(source_base_dir, category)
        output_category_dir = os.path.join(output_base_dir, category)

        if not os.path.exists(category_dir):
            logger.info('  Category %s not found, skipping', category)
            continue

        os.makedirs(output_category_dir, exist_ok=True)

        logger.info('')
        logger.info('Processing category: %s', category)
        category_patches = batch_extract_category(category_dir, output_category_dir,
                                                  category, patch_size, level)
        total_patches_all += category_patches

    elapsed_time = time.time() - start_time
    logger.info('')
    logger.info('=' * 60)
    logger.info('Total patches extracted: %d', total_patches_all)
    logger.info('Total time: %.1fs', elapsed_time)
    logger.info('=' * 60)
    return total_patches_all


# =====================================================================
# Stage 2: Patch filtering (CPU)
# =====================================================================

def is_background_pixel(r, g, b):
    return ((r > 200) & (g > 180) & (b > 150) &
            (np.abs(r - g) < 40) & (np.abs(r - b) < 60) & (np.abs(g - b) < 50))


def filter_background(img_array):
    h, w = img_array.shape[:2]
    total_pixels = h * w
    bg_count = np.sum(is_background_pixel(img_array[:, :, 0],
                                          img_array[:, :, 1],
                                          img_array[:, :, 2]))
    bg_ratio = bg_count / total_pixels
    return bg_ratio > 0.9, bg_ratio


def _require_cv2():
    if cv2 is None:
        raise RuntimeError('Filtering requires opencv-python. '
                           'Install with: pip install opencv-python')


def calculate_varlap(img_array):
    _require_cv2()
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    laplacian = cv2.filter2D(gray, -1, laplacian_kernel)
    return np.var(laplacian)


def filter_hemorrhage(img_array):
    _require_cv2()
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    fresh_hemorrhage = ((h >= 0) & (h <= 10)) | ((h >= 170) & (h <= 180))
    fresh_hemorrhage = fresh_hemorrhage & (s > 70) & (v > 100)

    old_hemorrhage = (h >= 10) & (h <= 30) & (s > 50) & (v > 40) & (v < 130)

    nhem = np.sum(fresh_hemorrhage | old_hemorrhage)
    ratiohem = nhem / (img_array.shape[0] * img_array.shape[1])
    return ratiohem > 0.3, ratiohem


def filter_necrosis(img_array):
    _require_cv2()
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    area = img_array.shape[0] * img_array.shape[1]

    eosinophilic = (r > g) & (g > b)
    ratioeo = np.sum(eosinophilic) / area

    nuclear = (b > r * 0.9) & (b > g * 0.9)
    rationuc = np.sum(nuclear) / area

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    vargray = np.var(gray)

    is_necrosis = (ratioeo > 0.6) & (rationuc < 0.05) & (vargray < 120)
    return is_necrosis, ratioeo, rationuc, vargray


def delete_file(filepath):
    try:
        os.remove(filepath)
    except OSError:
        pass


def process_single_folder(patch_dir, output_dir=None):
    png_files = [f for f in os.listdir(patch_dir)
                 if f.endswith('.png') and not f.startswith('._')]
    if not png_files:
        return None

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    stats = {
        'total': len(png_files),
        'background_filtered': 0,
        'hemorrhage_filtered': 0,
        'necrosis_filtered': 0,
        'blur_filtered': 0,
        'survived': 0,
    }

    varlap_values = []
    surviving_files = []

    for filename in png_files:
        patch_path = os.path.join(patch_dir, filename)

        try:
            img = Image.open(patch_path).convert('RGB')
            img_array = np.array(img)
        except Exception:
            continue

        bg_filtered, _ = filter_background(img_array)
        if bg_filtered:
            if output_dir is None:
                delete_file(patch_path)
            stats['background_filtered'] += 1
            continue

        hemorrhage_filtered, _ = filter_hemorrhage(img_array)
        if hemorrhage_filtered:
            if output_dir is None:
                delete_file(patch_path)
            stats['hemorrhage_filtered'] += 1
            continue

        necrosis_filtered, _, _, _ = filter_necrosis(img_array)
        if necrosis_filtered:
            if output_dir is None:
                delete_file(patch_path)
            stats['necrosis_filtered'] += 1
            continue

        varlap = calculate_varlap(img_array)
        varlap_values.append(varlap)
        surviving_files.append((patch_path, varlap, filename))

    if varlap_values:
        varlap_array = np.array(varlap_values)
        median = np.median(varlap_array)
        q1 = np.percentile(varlap_array, 25)
        q3 = np.percentile(varlap_array, 75)
        iqr = q3 - q1
        threshold_low = max(10, median * 0.35)
    else:
        threshold_low = 10
        median = 0
        iqr = 0

    for patch_path, varlap, filename in surviving_files:
        if varlap < threshold_low:
            if output_dir is None:
                delete_file(patch_path)
            stats['blur_filtered'] += 1
        else:
            if output_dir is not None:
                img = Image.open(patch_path).convert('RGB')
                img.save(os.path.join(output_dir, filename))
            stats['survived'] += 1

    stats['varlap_median'] = median
    stats['varlap_iqr'] = iqr
    stats['threshold_low'] = threshold_low

    return stats


def batch_filter(base_dir, output_base_dir, categories):
    total_stats = {
        'total': 0, 'background_filtered': 0, 'hemorrhage_filtered': 0,
        'necrosis_filtered': 0, 'blur_filtered': 0, 'survived': 0,
        'folders_processed': 0,
    }

    all_folders = []
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        if os.path.exists(cat_dir):
            for folder in os.listdir(cat_dir):
                folder_path = os.path.join(cat_dir, folder)
                if os.path.isdir(folder_path):
                    all_folders.append((category, folder, folder_path))

    if output_base_dir is not None:
        os.makedirs(output_base_dir, exist_ok=True)

    total_folders = len(all_folders)
    start_time = time.time()

    logger.info('Batch Patch Filtering')
    logger.info('Source: %s', base_dir)
    if output_base_dir:
        logger.info('Output: %s', output_base_dir)
    logger.info('=' * 60)

    for i, (category, folder_name, folder_path) in enumerate(all_folders, 1):
        if output_base_dir is not None:
            output_folder_path = os.path.join(output_base_dir, category, folder_name)
        else:
            output_folder_path = None

        stats = process_single_folder(folder_path, output_folder_path)
        if stats is None:
            continue

        total_stats['folders_processed'] += 1
        total_stats['total'] += stats['total']
        total_stats['background_filtered'] += stats['background_filtered']
        total_stats['hemorrhage_filtered'] += stats['hemorrhage_filtered']
        total_stats['necrosis_filtered'] += stats['necrosis_filtered']
        total_stats['blur_filtered'] += stats['blur_filtered']
        total_stats['survived'] += stats['survived']

        elapsed = time.time() - start_time
        remaining = (elapsed / i) * (total_folders - i)
        progress = i / total_folders
        bar_length = 40
        filled = int(progress * bar_length)
        bar = '#' * filled + '-' * (bar_length - filled)

        progress_write(
            f'{i}/{total_folders} [{bar}] {progress*100:.1f}% | '
            f'{category}/{folder_name} | survived: {stats["survived"]}/{stats["total"]} | '
            f'remaining: {remaining:.0f}s'
        )

    sys.stdout.write('\n')
    logger.info('')
    logger.info('=' * 60)
    logger.info('Batch filtering statistics')
    logger.info('=' * 60)
    _total = total_stats['total'] or 1
    logger.info('Folders processed: %d', total_stats['folders_processed'])
    logger.info('Total patches: %d', total_stats['total'])
    logger.info('Background filtered: %d (%.1f%%)', total_stats['background_filtered'],
                total_stats['background_filtered'] / _total * 100)
    logger.info('Hemorrhage filtered: %d (%.1f%%)', total_stats['hemorrhage_filtered'],
                total_stats['hemorrhage_filtered'] / _total * 100)
    logger.info('Necrosis filtered: %d (%.1f%%)', total_stats['necrosis_filtered'],
                total_stats['necrosis_filtered'] / _total * 100)
    logger.info('Blur filtered: %d (%.1f%%)', total_stats['blur_filtered'],
                total_stats['blur_filtered'] / _total * 100)
    logger.info('Survived: %d (%.1f%%)', total_stats['survived'],
                total_stats['survived'] / _total * 100)
    logger.info('Total time: %.1fs', time.time() - start_time)
    logger.info('=' * 60)


# =====================================================================
# Stage 3: Stain normalization (CPU)
# =====================================================================

def rgb_to_od(patch: np.ndarray) -> np.ndarray:
    patch = np.maximum(patch, 1)
    return -np.log(patch / 255.0)


def rgb_from_od(od: np.ndarray) -> np.ndarray:
    rgb = 255.0 * np.exp(-od)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def get_concentrations(od: np.ndarray, vH: np.ndarray, vE: np.ndarray) -> np.ndarray:
    stain_matrix = np.array([vH, vE])
    return od @ pinv(stain_matrix)


def get_stain_vectors(od: np.ndarray, beta: float = 0.15, alpha: float = 1.0) -> tuple:
    od_norm = np.linalg.norm(od, axis=1)
    mask = od_norm >= beta
    od_valid = od[mask]

    if od_valid.shape[0] < 100:
        vH_std = np.array([0.650, 0.704, 0.287])
        vE_std = np.array([0.072, 0.990, 0.105])
        return vH_std, vE_std

    od_mean = np.mean(od_valid, axis=0)
    od_centered = od_valid - od_mean
    U, s, Vt = svd(od_centered, full_matrices=False)
    v1 = Vt[0, :]
    v2 = Vt[1, :]

    if np.sum(v1) < 0:
        v1 = -v1
    if np.sum(v2) < 0:
        v2 = -v2

    proj = od_valid @ np.column_stack([v1, v2])
    angles = np.arctan2(proj[:, 1], proj[:, 0])

    angle_min = np.percentile(angles, alpha)
    angle_max = np.percentile(angles, 100 - alpha)

    vec1 = v1 * np.cos(angle_min) + v2 * np.sin(angle_min)
    vec2 = v1 * np.cos(angle_max) + v2 * np.sin(angle_max)

    vH_std = np.array([0.650, 0.704, 0.287])
    vE_std = np.array([0.072, 0.990, 0.105])

    sim1_h = np.dot(vec1, vH_std) / (norm(vec1) * norm(vH_std))
    sim1_e = np.dot(vec1, vE_std) / (norm(vec1) * norm(vE_std))
    sim2_h = np.dot(vec2, vH_std) / (norm(vec2) * norm(vH_std))
    sim2_e = np.dot(vec2, vE_std) / (norm(vec2) * norm(vE_std))

    if sim1_h + sim2_e > sim1_e + sim2_h:
        vH, vE = vec1, vec2
    else:
        vH, vE = vec2, vec1

    vH = vH / (norm(vH) + 1e-8)
    vE = vE / (norm(vE) + 1e-8)

    return vH, vE


def build_reference_template(patch_list, beta: float = 0.15,
                             max_pixels_per_patch: int = 2000) -> dict:
    """Build a reference template in chunks.

    Iterates over all patches in batches. For each patch, valid OD pixels are
    extracted and randomly subsampled to ``max_pixels_per_patch`` pixels, so
    every patch contributes to the template while peak memory stays bounded.
    Accepts either path strings or pre-loaded numpy arrays; when paths are
    given, patches are read lazily to avoid loading everything into memory.
    """
    all_od_valid = []
    total_valid_pixels = 0
    chunk_size = 200  # number of patches per batch

    def _collect_od(np_patch):
        od = rgb_to_od(np_patch).reshape(-1, 3)
        od_norm = np.linalg.norm(od, axis=1)
        mask = od_norm >= beta
        od_valid = od[mask]
        if od_valid.shape[0] < 100:
            return None
        if od_valid.shape[0] > max_pixels_per_patch:
            idx = np.random.choice(od_valid.shape[0], max_pixels_per_patch, replace=False)
            od_valid = od_valid[idx]
        return od_valid

    for start in range(0, len(patch_list), chunk_size):
        chunk = patch_list[start:start + chunk_size]
        chunk_od = []
        for patch in chunk:
            if isinstance(patch, str):
                try:
                    img = Image.open(patch).convert('RGB')
                    patch = np.array(img)
                except Exception:
                    continue
            od_valid = _collect_od(patch)
            if od_valid is not None:
                chunk_od.append(od_valid)
        if chunk_od:
            chunk_concat = np.vstack(chunk_od)
            all_od_valid.append(chunk_concat)
            total_valid_pixels += chunk_concat.shape[0]
        del chunk_od
        logger.info('    Processed %d/%d patches, cumulative valid pixels %d',
                    min(start + chunk_size, len(patch_list)),
                    len(patch_list), total_valid_pixels)

    if len(all_od_valid) == 0:
        raise ValueError('No valid pixels to build reference template.')

    all_od_concat = np.vstack(all_od_valid)
    vH_ref, vE_ref = get_stain_vectors(all_od_concat, beta=beta)

    C_all = get_concentrations(all_od_concat, vH_ref, vE_ref)
    qH_low, qH_high = np.percentile(C_all[:, 0], [1, 99])
    qE_low, qE_high = np.percentile(C_all[:, 1], [1, 99])

    return {
        'vH': vH_ref,
        'vE': vE_ref,
        'qH_low': qH_low,
        'qH_high': qH_high,
        'qE_low': qE_low,
        'qE_high': qE_high,
    }


def collect_folder_od(folder_path, beta: float = 0.15, max_samples: int = 30):
    png_files = [f for f in os.listdir(folder_path)
                 if f.endswith('.png') and not f.startswith('._')]
    if len(png_files) > max_samples:
        png_files = random.sample(png_files, max_samples)

    all_od_valid = []

    for filename in png_files:
        try:
            img = Image.open(os.path.join(folder_path, filename)).convert('RGB')
            patch = np.array(img)
            od = rgb_to_od(patch).reshape(-1, 3)
            od_norm = np.linalg.norm(od, axis=1)
            mask = od_norm >= beta
            if np.sum(mask) >= 100:
                all_od_valid.append(od[mask])
        except Exception:
            continue

    if len(all_od_valid) == 0:
        return None

    return np.vstack(all_od_valid)


def normalize_patch(patch: np.ndarray, template: dict, vH_src: np.ndarray,
                    vE_src: np.ndarray, beta: float = 0.15) -> np.ndarray:
    od = rgb_to_od(patch).reshape(-1, 3)
    od_norm = np.linalg.norm(od, axis=1)
    mask = od_norm >= beta

    if np.sum(mask) < 100:
        return patch

    od_valid = od[mask]
    od_invalid = od[~mask]

    C_src = get_concentrations(od_valid, vH_src, vE_src)

    C_src[:, 0] = np.clip(C_src[:, 0], template['qH_low'], template['qH_high'])
    C_src[:, 1] = np.clip(C_src[:, 1], template['qE_low'], template['qE_high'])

    stain_matrix_ref = np.array([template['vH'], template['vE']])
    od_normalized_valid = C_src @ stain_matrix_ref

    od_normalized = np.zeros_like(od)
    od_normalized[mask] = od_normalized_valid
    od_normalized[~mask] = od_invalid

    return rgb_from_od(od_normalized.reshape(patch.shape))


def load_random_patches(folder_path, count=None):
    png_files = [f for f in os.listdir(folder_path)
                 if f.endswith('.png') and not f.startswith('._')]

    if count is not None:
        if len(png_files) < count:
            count = len(png_files)
        selected = random.sample(png_files, count)
    else:
        selected = png_files

    patches = []
    for filename in selected:
        try:
            img = Image.open(os.path.join(folder_path, filename)).convert('RGB')
            patches.append(np.array(img))
        except Exception:
            continue

    return patches


def build_reference_template_from_folders(reference_dir, folder_names=None,
                                          beta: float = 0.15):
    """Build a reference template.

    If ``folder_names`` is None, every sub-directory under ``reference_dir``
    is used as a reference folder.
    """
    if folder_names is None:
        folder_names = sorted(
            d for d in os.listdir(reference_dir)
            if os.path.isdir(os.path.join(reference_dir, d)) and not d.startswith('.')
        )
    reference_folders = [os.path.join(reference_dir, name) for name in folder_names]

    all_reference_paths = []
    total = len(reference_folders)
    for i, folder in enumerate(reference_folders, 1):
        if os.path.exists(folder):
            paths = [os.path.join(folder, f) for f in os.listdir(folder)
                     if f.endswith('.png') and not f.startswith('._')]
            all_reference_paths.extend(paths)
            logger.info('  %d/%d: %s - %d patches',
                        i, total, os.path.basename(folder), len(paths))
        else:
            logger.warning('  %d/%d: %s - folder not found',
                           i, total, os.path.basename(folder))

    logger.info('Collected %d reference patches in total', len(all_reference_paths))
    template = build_reference_template(all_reference_paths, beta=beta)
    logger.info('  Reference hematoxylin vector vH: [%.4f, %.4f, %.4f]',
                template['vH'][0], template['vH'][1], template['vH'][2])
    logger.info('  Reference eosin vector vE: [%.4f, %.4f, %.4f]',
                template['vE'][0], template['vE'][1], template['vE'][2])
    logger.info('  Hematoxylin concentration range: [%.4f, %.4f]',
                template['qH_low'], template['qH_high'])
    logger.info('  Eosin concentration range: [%.4f, %.4f]',
                template['qE_low'], template['qE_high'])
    return template


def run_normalization(source_base_dir, output_base_dir, categories,
                      reference_dir, reference_folders=None):
    """Batch normalization: read filtered patches from source_base_dir, write to output_base_dir."""
    logger.info('=' * 70)
    logger.info('Batch stain normalization')
    logger.info('=' * 70)

    logger.info('')
    logger.info('[1/2] Building reference template...')
    template = build_reference_template_from_folders(reference_dir, reference_folders)

    logger.info('')
    logger.info('[2/2] Batch normalization...')

    all_folders = []
    for category in categories:
        cat_dir = os.path.join(source_base_dir, category)
        if os.path.exists(cat_dir):
            for folder in os.listdir(cat_dir):
                folder_path = os.path.join(cat_dir, folder)
                if os.path.isdir(folder_path):
                    all_folders.append((category, folder, folder_path))

    total_folders = len(all_folders)
    start_time = time.time()

    for i, (category, folder_name, folder_path) in enumerate(all_folders, 1):
        output_dir = os.path.join(output_base_dir, category, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        png_files = [f for f in os.listdir(folder_path)
                     if f.endswith('.png') and not f.startswith('._')]
        total_patches = len(png_files)

        elapsed = time.time() - start_time
        avg_time = elapsed / max(i, 1)
        remaining = avg_time * (total_folders - i)
        progress = i / total_folders
        bar_length = 40
        filled = int(progress * bar_length)
        bar = '#' * filled + '-' * (bar_length - filled)

        if total_patches == 0:
            progress_write(f'{i}/{total_folders} [{bar}] {progress*100:.1f}% | '
                           f'{category}/{folder_name} | 0 patches | remaining: {remaining:.0f}s')
            continue

        existing_files = [f for f in os.listdir(output_dir)
                          if f.endswith('.png') and not f.startswith('._')]
        if len(existing_files) >= total_patches:
            progress_write(f'{i}/{total_folders} [{bar}] {progress*100:.1f}% | '
                           f'{category}/{folder_name} | skipped (already done) | '
                           f'remaining: {remaining:.0f}s')
            continue

        od_all = collect_folder_od(folder_path)
        if od_all is None:
            vH_src = np.array([0.650, 0.704, 0.287])
            vE_src = np.array([0.072, 0.990, 0.105])
        else:
            vH_src, vE_src = get_stain_vectors(od_all)

        for filename in png_files:
            output_path = os.path.join(output_dir, filename)
            if os.path.exists(output_path):
                continue
            try:
                img = Image.open(os.path.join(folder_path, filename)).convert('RGB')
                patch = np.array(img)
                normalized = normalize_patch(patch, template, vH_src, vE_src)
                Image.fromarray(normalized).save(output_path)
            except Exception:
                continue

        progress_write(f'{i}/{total_folders} [{bar}] {progress*100:.1f}% | '
                       f'{category}/{folder_name} | {total_patches} patches | '
                       f'remaining: {remaining:.0f}s')

    sys.stdout.write('\n')
    logger.info('')
    logger.info('=' * 70)
    logger.info('Batch normalization completed')
    logger.info('=' * 70)

    total_time = time.time() - start_time
    logger.info('Total time: %.1fs', total_time)

    total_patches_count = 0
    for category in categories:
        cat_dir = os.path.join(output_base_dir, category)
        if os.path.exists(cat_dir):
            count = 0
            for folder in os.listdir(cat_dir):
                folder_path = os.path.join(cat_dir, folder)
                if os.path.isdir(folder_path):
                    count += len([f for f in os.listdir(folder_path)
                                  if f.endswith('.png') and not f.startswith('._')])
            logger.info('%s: %d normalized patches', category, count)
            total_patches_count += count

    logger.info('Total: %d normalized patches', total_patches_count)
    logger.info('Output directory: %s', output_base_dir)


def cleanup_intermediate(output_root, basename):
    """Keep only the final normalized result; remove cut and filtered directories."""
    for suffix in ('_cut', '_filtered'):
        d = os.path.join(output_root, basename + suffix)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            logger.info('  Removed intermediate directory: %s', d)


def process_single_file(svs_path, geojson_path, output_root,
                        reference_dir, reference_folders=None):
    """Run cut -> filter -> normalize for a single SVS file."""
    if not os.path.exists(svs_path):
        logger.error('SVS not found: %s', svs_path)
        return
    if not os.path.exists(geojson_path):
        logger.error('GeoJSON not found: %s', geojson_path)
        return

    basename = os.path.splitext(os.path.basename(svs_path))[0]
    os.makedirs(output_root, exist_ok=True)

    logger.info('=' * 70)
    logger.info('Single-file processing: %s', basename)
    logger.info('=' * 70)

    logger.info('')
    logger.info('[0/3] Building reference template...')
    template = build_reference_template_from_folders(reference_dir, reference_folders)

    cut_dir = os.path.join(output_root, basename + '_cut')
    logger.info('')
    logger.info('[1/3] Cutting...')
    t0 = time.time()
    n_patches = extract_patches(svs_path, geojson_path, cut_dir)
    patches_dir = os.path.join(cut_dir, basename)
    if not os.path.isdir(patches_dir):
        patches_dir = cut_dir
    logger.info('  Extracted %d patches, elapsed %.1fs', n_patches, time.time() - t0)

    filtered_dir = os.path.join(output_root, basename + '_filtered')
    logger.info('')
    logger.info('[2/3] Filtering...')
    t0 = time.time()
    stats = process_single_folder(patches_dir, filtered_dir)
    if stats:
        logger.info('  Total %d, background %d, hemorrhage %d, necrosis %d, '
                    'blur %d, survived %d, elapsed %.1fs',
                    stats['total'], stats['background_filtered'],
                    stats['hemorrhage_filtered'], stats['necrosis_filtered'],
                    stats['blur_filtered'], stats['survived'], time.time() - t0)
    else:
        logger.info('  No patches, elapsed %.1fs', time.time() - t0)

    normalized_dir = os.path.join(output_root, basename + '_normalized')
    logger.info('')
    logger.info('[3/3] Normalizing...')
    if not os.path.isdir(filtered_dir):
        logger.info('  No patches after filtering, skipping normalization')
    else:
        png_files = [f for f in os.listdir(filtered_dir)
                     if f.endswith('.png') and not f.startswith('._')]
        total_patches = len(png_files)
        if total_patches == 0:
            logger.info('  No patches after filtering, skipping normalization')
        else:
            od_all = collect_folder_od(filtered_dir)
            if od_all is None:
                vH_src = np.array([0.650, 0.704, 0.287])
                vE_src = np.array([0.072, 0.990, 0.105])
            else:
                vH_src, vE_src = get_stain_vectors(od_all)
            os.makedirs(normalized_dir, exist_ok=True)
            t0 = time.time()
            for filename in png_files:
                out_path = os.path.join(normalized_dir, filename)
                if os.path.exists(out_path):
                    continue
                try:
                    img = Image.open(os.path.join(filtered_dir, filename)).convert('RGB')
                    patch = np.array(img)
                    normalized = normalize_patch(patch, template, vH_src, vE_src)
                    Image.fromarray(normalized).save(out_path)
                except Exception:
                    continue
            logger.info('  Normalized %d patches, elapsed %.1fs',
                        total_patches, time.time() - t0)

    cleanup_intermediate(output_root, basename)
    logger.info('')
    logger.info('All done. Only normalized results are kept. Output directory: %s',
                output_root)


# =====================================================================
# GPU-accelerated module (PyTorch, uses all CUDA devices)
# =====================================================================
try:
    import torch
    import torch.nn.functional as F
    import multiprocessing as _mp
    from concurrent.futures import ThreadPoolExecutor

    torch.manual_seed(2026)
    TORCH_AVAILABLE = torch.cuda.is_available()
    GPU_DEVICES = list(range(torch.cuda.device_count())) if TORCH_AVAILABLE else []
except Exception as _gpu_e:
    logger.warning('[GPU module failed to load: %s]', _gpu_e)
    TORCH_AVAILABLE = False
    GPU_DEVICES = []


def print_gpu_info():
    if not TORCH_AVAILABLE:
        logger.info('  CUDA not available')
        return
    for g in GPU_DEVICES:
        try:
            name = torch.cuda.get_device_name(g)
            total = torch.cuda.get_device_properties(g).total_memory
            used = total - torch.cuda.mem_get_info(g)[0]
            logger.info('  GPU%d: %s | memory %.1f/%.0f GB',
                        g, name, used / 2**30, total / 2**30)
        except Exception:
            pass


def gpu_mem_str():
    if not TORCH_AVAILABLE:
        return 'CUDA not available'
    parts = []
    for g in GPU_DEVICES:
        try:
            used = torch.cuda.mem_get_info(g)[1] - torch.cuda.mem_get_info(g)[0]
            parts.append(f'GPU{g} {used/2**20:.0f}MB')
        except Exception:
            pass
    return ' | '.join(parts)


def pip_batch_gpu(px, py, polygon):
    """Vectorized ray-casting point-in-polygon test on GPU (many points at once)."""
    inside = torch.zeros_like(px, dtype=torch.bool)
    n = len(polygon)
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[(i + 1) % n]
        mask = (yi > py) != (yj > py)
        safe = torch.where(mask, (yj - yi), torch.ones_like(py))
        cond = mask & (px < (xj - xi) * (py - yi) / safe + xi)
        inside ^= cond
    return inside


def _cut_worker(svs_path, positions, output_dir, patch_size, level, svs_basename, skip_existing):
    slide = openslide.OpenSlide(svs_path)
    count = 0
    for poly_idx, x, y in positions:
        fn = f'{svs_basename}_poly{poly_idx}_{x}_{y}.png'
        p = os.path.join(output_dir, fn)
        if skip_existing and os.path.exists(p):
            count += 1
            continue
        region = slide.read_region((x, y), level, (patch_size, patch_size)).convert('RGB')
        region.save(p)
        count += 1
    slide.close()
    return count


def extract_patches_gpu(svs_path, geojson_path, output_base_dir, patch_size=224, level=0,
                        output_basename=None, skip_existing=True):
    """GPU patch cutting: polygon test runs on all GPUs, patch IO uses multiprocessing."""
    slide = openslide.OpenSlide(svs_path)
    geojson_data = read_geojson(geojson_path)
    polygons = get_polygon_coordinates(geojson_data)
    svs_basename = os.path.splitext(os.path.basename(svs_path))[0]
    if output_basename is None:
        output_basename = svs_basename
    output_dir = os.path.join(output_base_dir, output_basename)
    os.makedirs(output_dir, exist_ok=True)
    if not polygons:
        slide.close()
        return 0

    existing_patches = set()
    if skip_existing and os.path.exists(output_dir):
        existing_patches = {f for f in os.listdir(output_dir) if f.endswith('.png')}

    slide_w, slide_h = slide.level_dimensions[0]
    n_gpu = len(GPU_DEVICES)
    positions = []  # (poly_idx, x, y) top-left corner coordinates
    t0 = time.time()
    for poly_idx, polygon in enumerate(polygons):
        min_x, min_y, max_x, max_y = get_polygon_bbox(polygon)
        start_x = max(0, min_x); start_y = max(0, min_y)
        end_x = min(slide_w, max_x); end_y = min(slide_h, max_y)
        xs = list(range(start_x, end_x - patch_size + 1, patch_size))
        ys = list(range(start_y, end_y - patch_size + 1, patch_size))
        if not xs or not ys:
            continue
        dev0 = f'cuda:{GPU_DEVICES[0]}'
        cx = torch.tensor(xs, dtype=torch.float32, device=dev0) + patch_size // 2
        cy = torch.tensor(ys, dtype=torch.float32, device=dev0) + patch_size // 2
        px, py = torch.meshgrid(cx, cy, indexing='xy')
        px = px.reshape(-1); py = py.reshape(-1)
        n = px.numel()
        chunk = (n + n_gpu - 1) // n_gpu
        masks = []
        for g in range(n_gpu):
            s0 = g * chunk; s1 = min((g + 1) * chunk, n)
            if s0 >= n:
                break
            dev = torch.device(f'cuda:{g}')
            masks.append(pip_batch_gpu(px[s0:s1].to(dev), py[s0:s1].to(dev), polygon).cpu())
        mask = torch.cat(masks)
        idx = mask.nonzero(as_tuple=False).reshape(-1)
        if idx.numel():
            sx = (px[idx].cpu().numpy() - patch_size // 2).astype(int)
            sy = (py[idx].cpu().numpy() - patch_size // 2).astype(int)
            positions.extend((poly_idx, int(x), int(y)) for x, y in zip(sx, sy))
        logger.info('  Polygon %d: candidates %d -> hits %d', poly_idx, n, idx.numel())
    slide.close()
    logger.info('  Polygon test (all GPUs) elapsed %.1fs, hits %d | %s',
                time.time() - t0, len(positions), gpu_mem_str())

    if skip_existing:
        positions = [p for p in positions
                     if f'{svs_basename}_poly{p[0]}_{p[1]}_{p[2]}.png' not in existing_patches]
    if not positions:
        return len(existing_patches)

    t1 = time.time()
    chunks = [positions[i::n_gpu] for i in range(n_gpu)]
    ctx = _mp.get_context('spawn')
    with ctx.Pool(processes=n_gpu) as pool:
        counts = pool.starmap(_cut_worker,
                              [(svs_path, ch, output_dir, patch_size, level,
                                svs_basename, skip_existing) for ch in chunks])
    logger.info('  Cutting (parallel %d processes) elapsed %.1fs, added %d patches',
                n_gpu, time.time() - t1, sum(counts))
    return len(existing_patches) + sum(counts)


def _filter_batch_gpu(device, t):
    """Batch filtering on GPU: background/hemorrhage/necrosis/blur (Laplacian variance)."""
    r = t[..., 0].float(); g = t[..., 1].float(); b = t[..., 2].float()
    bg = (r > 200) & (g > 180) & (b > 150) & ((r - g).abs() < 40) & \
         ((r - b).abs() < 60) & ((g - b).abs() < 50)
    bg_flag = bg.float().mean(dim=(1, 2)) > 0.9

    maxc = torch.maximum(torch.maximum(r, g), b)
    minc = torch.minimum(torch.minimum(r, g), b)
    delta = maxc - minc
    h = torch.zeros_like(maxc)
    m = (delta != 0) & (maxc == r)
    h = torch.where(m, 60.0 * (((g - b) / (delta + 1e-9)) % 6.0), h)
    m = (delta != 0) & (maxc == g)
    h = torch.where(m, 60.0 * (((b - r) / (delta + 1e-9)) + 2.0), h)
    m = (delta != 0) & (maxc == b)
    h = torch.where(m, 60.0 * (((r - g) / (delta + 1e-9)) + 4.0), h)
    h = torch.where(h < 0, h + 360.0, h) / 2.0
    s = torch.where(maxc == 0, torch.zeros_like(maxc), delta / (maxc + 1e-9)) * 255.0
    v = maxc
    fresh = ((h >= 0) & (h <= 10)) | ((h >= 170) & (h <= 180))
    fresh = fresh & (s > 70) & (v > 100)
    old = (h >= 10) & (h <= 30) & (s > 50) & (v > 40) & (v < 130)
    hem_flag = (fresh | old).float().mean(dim=(1, 2)) > 0.3

    eosin = (r > g) & (g > b)
    ratioeo = eosin.float().mean(dim=(1, 2))
    nuclear = (b > r * 0.9) & (b > g * 0.9)
    rationuc = nuclear.float().mean(dim=(1, 2))
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    vargray = gray.var(dim=(1, 2), unbiased=False)
    nec_flag = (ratioeo > 0.6) & (rationuc < 0.05) & (vargray < 120)

    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                     dtype=torch.float32, device=device).view(1, 1, 3, 3)
    lap = F.conv2d(gray[:, None, :, :], k, padding=1)
    varlap = lap.squeeze(1).var(dim=(1, 2), unbiased=False)
    return bg_flag, hem_flag, nec_flag, varlap


def _filter_worker_gpu(device_id, chunk_paths, batch_size=256):
    torch.cuda.set_device(device_id)
    dev = torch.device(f'cuda:{device_id}')
    out = []
    for s in range(0, len(chunk_paths), batch_size):
        paths = chunk_paths[s:s + batch_size]
        arrs = []
        for p in paths:
            try:
                arrs.append(np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8))
            except Exception:
                arrs.append(None)
        inds = [i for i, a in enumerate(arrs) if a is not None]
        if inds:
            t = torch.from_numpy(np.stack([arrs[i] for i in inds])).to(dev)
            bg, hem, nec, var = _filter_batch_gpu(dev, t)
            bg = bg.cpu().numpy(); hem = hem.cpu().numpy()
            nec = nec.cpu().numpy(); var = var.cpu().numpy()
        j = 0
        for i, a in enumerate(arrs):
            if a is None:
                out.append(None)
            else:
                out.append((bool(bg[j]), bool(hem[j]), bool(nec[j]), float(var[j])))
                j += 1
    return out


def process_single_folder_gpu(patch_dir, output_dir=None):
    """GPU filtering: parallel across all GPUs, thresholds identical to CPU path."""
    png_files = [f for f in os.listdir(patch_dir)
                 if f.endswith('.png') and not f.startswith('._')]
    if not png_files:
        return None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    stats = {
        'total': len(png_files),
        'background_filtered': 0,
        'hemorrhage_filtered': 0,
        'necrosis_filtered': 0,
        'blur_filtered': 0,
        'survived': 0,
    }

    n_gpu = len(GPU_DEVICES)
    full_paths = [os.path.join(patch_dir, f) for f in png_files]
    chunks = [full_paths[i::n_gpu] for i in range(n_gpu)]
    ctx = _mp.get_context('spawn')
    with ctx.Pool(processes=n_gpu) as pool:
        res = pool.starmap(_filter_worker_gpu, [(g, chunks[g]) for g in range(n_gpu)])

    results_by_index = [None] * len(png_files)
    for g, r in enumerate(res):
        for k, val in enumerate(r):
            results_by_index[g + k * n_gpu] = val

    varlap_values = []
    surviving = []
    for i, filename in enumerate(png_files):
        r = results_by_index[i]
        if r is None:
            continue
        bg, hem, nec, var = r
        patch_path = os.path.join(patch_dir, filename)
        if bg:
            if output_dir is None:
                delete_file(patch_path)
            stats['background_filtered'] += 1
            continue
        if hem:
            if output_dir is None:
                delete_file(patch_path)
            stats['hemorrhage_filtered'] += 1
            continue
        if nec:
            if output_dir is None:
                delete_file(patch_path)
            stats['necrosis_filtered'] += 1
            continue
        varlap_values.append(var)
        surviving.append((patch_path, var, filename))

    if varlap_values:
        varlap_array = np.array(varlap_values)
        median = np.median(varlap_array)
        q1 = np.percentile(varlap_array, 25)
        q3 = np.percentile(varlap_array, 75)
        iqr = q3 - q1
        threshold_low = max(10, median * 0.35)
    else:
        threshold_low = 10
        median = 0
        iqr = 0

    copy_tasks = []
    for patch_path, var, filename in surviving:
        if var < threshold_low:
            if output_dir is None:
                delete_file(patch_path)
            stats['blur_filtered'] += 1
        else:
            if output_dir is not None:
                copy_tasks.append((patch_path, os.path.join(output_dir, filename)))
            stats['survived'] += 1

    if output_dir is not None and copy_tasks:
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda a: shutil.copyfile(a[0], a[1]), copy_tasks))

    stats['varlap_median'] = median
    stats['varlap_iqr'] = iqr
    stats['threshold_low'] = threshold_low
    return stats


def _collect_od_chunk_gpu(paths, beta, max_px, device, batch_size=64):
    all_od = []
    for s in range(0, len(paths), batch_size):
        batch_paths = paths[s:s + batch_size]
        arrs = []
        for p in batch_paths:
            try:
                arrs.append(np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8))
            except Exception:
                pass
        if not arrs:
            continue
        t = torch.from_numpy(np.stack(arrs)).to(device).float().clamp(min=1)
        N = t.shape[0]
        od = -torch.log(t / 255.0)
        odf = od.reshape(N, -1, 3)
        nrm = torch.linalg.vector_norm(odf, dim=2)
        mask = nrm >= beta
        for i in range(N):
            idx = mask[i].nonzero(as_tuple=False).squeeze(1)
            if idx.numel() >= 100:
                if idx.numel() > max_px:
                    idx = idx[torch.randperm(idx.numel(), device=device)[:max_px]]
                all_od.append(odf[i][idx])
    if not all_od:
        return None
    return torch.cat(all_od)


def build_reference_template_gpu(patch_paths, beta=0.15,
                                 max_pixels_per_patch=2000, chunk_size=200):
    """GPU reference-template construction: batches are dispatched across all GPUs."""
    n_gpu = len(GPU_DEVICES)
    all_od = []
    total = 0
    for start in range(0, len(patch_paths), chunk_size):
        paths = patch_paths[start:start + chunk_size]
        g = (start // chunk_size) % n_gpu
        dev = torch.device(f'cuda:{g}')
        od = _collect_od_chunk_gpu(paths, beta, max_pixels_per_patch, dev)
        if od is not None:
            all_od.append(od.cpu())
            total += od.shape[0]
        logger.info('    Processed %d/%d patches, cumulative valid pixels %d',
                    min(start + chunk_size, len(patch_paths)), len(patch_paths), total)
    if not all_od:
        raise ValueError('No valid pixels to build reference template.')
    od_all = torch.cat(all_od).numpy()
    vH, vE = get_stain_vectors(od_all, beta=beta)
    C_all = get_concentrations(od_all, vH, vE)
    qH_low, qH_high = np.percentile(C_all[:, 0], [1, 99])
    qE_low, qE_high = np.percentile(C_all[:, 1], [1, 99])
    return {
        'vH': vH, 'vE': vE,
        'qH_low': float(qH_low), 'qH_high': float(qH_high),
        'qE_low': float(qE_low), 'qE_high': float(qE_high),
    }


def _collect_od_files_gpu(device_id, paths, beta=0.15):
    torch.cuda.set_device(device_id)
    dev = torch.device(f'cuda:{device_id}')
    all_od = []
    for p in paths:
        try:
            img = Image.open(p).convert('RGB')
            t = torch.from_numpy(np.asarray(img, dtype=np.uint8)).to(dev).float().clamp(min=1)
            od = -torch.log(t / 255.0).reshape(-1, 3)
            nrm = torch.linalg.vector_norm(od, dim=1)
            mask = nrm >= beta
            if int(mask.sum()) >= 100:
                all_od.append(od[mask].cpu().numpy())
        except Exception:
            continue
    if not all_od:
        return None
    return np.vstack(all_od)


def collect_folder_od_gpu(folder_path, beta=0.15, max_samples=30):
    png_files = [f for f in os.listdir(folder_path)
                 if f.endswith('.png') and not f.startswith('._')]
    if len(png_files) > max_samples:
        png_files = random.sample(png_files, max_samples)
    n_gpu = len(GPU_DEVICES)
    chunks = [png_files[i::n_gpu] for i in range(n_gpu)]
    ctx = _mp.get_context('spawn')
    with ctx.Pool(processes=n_gpu) as pool:
        res = pool.starmap(_collect_od_files_gpu,
                           [(g, [os.path.join(folder_path, f) for f in chunks[g]], beta)
                            for g in range(n_gpu)])
    all_od = [r for r in res if r is not None]
    if not all_od:
        return None
    return np.vstack(all_od)


def _normalize_batch_gpu(device, t, vH_src, vE_src, template, beta=0.15):
    """GPU batch stain normalization, equivalent to CPU ``normalize_patch``."""
    N = t.shape[0]
    H, W = t.shape[1], t.shape[2]
    p = t.float().clamp(min=1)
    od = -torch.log(p / 255.0)
    odf = od.reshape(N, -1, 3)
    od_norm = torch.linalg.vector_norm(odf, dim=2)
    mask = od_norm >= beta
    sm_src = torch.stack([vH_src.to(device), vE_src.to(device)])
    pinv_src = torch.linalg.pinv(sm_src)
    sm_ref = torch.stack([template['vH'].to(device), template['vE'].to(device)])
    out = odf.clone()
    for i in range(N):
        m = mask[i]
        if int(m.sum()) < 100:
            continue
        od_valid = odf[i][m]
        od_invalid = odf[i][~m]
        C = od_valid @ pinv_src
        C = C.clone()
        C[:, 0] = torch.clamp(C[:, 0], template['qH_low'], template['qH_high'])
        C[:, 1] = torch.clamp(C[:, 1], template['qE_low'], template['qE_high'])
        od_norm_valid = C @ sm_ref
        out[i][m] = od_norm_valid
        out[i][~m] = od_invalid
    rgb = torch.clamp(255.0 * torch.exp(-out), 0, 255).to(torch.uint8)
    return rgb.reshape(N, H, W, 3)


def _normalize_worker_gpu(device_id, in_dir, out_dir, filenames,
                          template_np, vH_np, vE_np, batch_size=256):
    torch.cuda.set_device(device_id)
    dev = torch.device(f'cuda:{device_id}')
    tpl = {}
    for k in template_np:
        v = template_np[k]
        tpl[k] = torch.from_numpy(v).float().to(dev) if isinstance(v, np.ndarray) else float(v)
    vH = torch.from_numpy(vH_np).float().to(dev)
    vE = torch.from_numpy(vE_np).float().to(dev)
    count = 0
    for s in range(0, len(filenames), batch_size):
        fs = filenames[s:s + batch_size]
        arrs = []
        for f in fs:
            try:
                arrs.append(np.asarray(Image.open(os.path.join(in_dir, f)).convert('RGB'),
                                       dtype=np.uint8))
            except Exception:
                arrs.append(None)
        inds = [i for i, a in enumerate(arrs) if a is not None]
        if not inds:
            continue
        t = torch.from_numpy(np.stack([arrs[i] for i in inds])).to(dev)
        out = _normalize_batch_gpu(dev, t, vH, vE, tpl)
        for j, i in enumerate(inds):
            Image.fromarray(out[j].cpu().numpy()).save(os.path.join(out_dir, fs[i]))
            count += 1
    return count


def normalize_folder_gpu(in_dir, out_dir, template_np, vH_np, vE_np):
    os.makedirs(out_dir, exist_ok=True)
    png_files = [f for f in os.listdir(in_dir)
                 if f.endswith('.png') and not f.startswith('._')]
    todo = [f for f in png_files if not os.path.exists(os.path.join(out_dir, f))]
    n_gpu = len(GPU_DEVICES)
    chunks = [todo[i::n_gpu] for i in range(n_gpu)]
    ctx = _mp.get_context('spawn')
    with ctx.Pool(processes=n_gpu) as pool:
        counts = pool.starmap(_normalize_worker_gpu,
                              [(g, in_dir, out_dir, chunks[g],
                                template_np, vH_np, vE_np) for g in range(n_gpu)])
    return sum(counts)


# =====================================================================
# GPU batch entry points
# =====================================================================

def batch_extract_gpu(source_base_dir, output_base_dir, categories, patch_size=224, level=0):
    """GPU batch cutting, reusing ``extract_patches_gpu``."""
    total_patches_all = 0
    start_time = time.time()

    logger.info('GPU Batch Patch Extraction')
    logger.info('Source: %s', source_base_dir)
    logger.info('Output: %s', output_base_dir)
    logger.info('=' * 60)

    for category in categories:
        category_dir = os.path.join(source_base_dir, category)
        output_category_dir = os.path.join(output_base_dir, category)

        if not os.path.isdir(category_dir):
            continue

        svs_files = sorted([f for f in os.listdir(category_dir)
                            if f.endswith('.svs') and not f.startswith('._')])
        geojson_files = sorted([f for f in os.listdir(category_dir)
                                if f.endswith('.geojson') and not f.startswith('._')])
        svs_basenames = {os.path.splitext(f)[0] for f in svs_files}
        geojson_basenames = {os.path.splitext(f)[0] for f in geojson_files}
        paired_basenames = sorted(svs_basenames & geojson_basenames)

        os.makedirs(output_category_dir, exist_ok=True)

        logger.info('')
        logger.info('Processing category: %s (%d paired files)',
                    category, len(paired_basenames))
        for i, basename in enumerate(paired_basenames, 1):
            svs_path = os.path.join(category_dir, f'{basename}.svs')
            geojson_path = os.path.join(category_dir, f'{basename}.geojson')
            patch_count = 0
            try:
                patch_count = extract_patches_gpu(svs_path, geojson_path,
                                                  output_category_dir, patch_size, level)
                total_patches_all += patch_count
            except Exception as e:
                logger.error('    Error processing %s: %s', basename, e)
            logger.info('    [%d/%d] %s: %d',
                        i, len(paired_basenames), basename, patch_count)

    elapsed_time = time.time() - start_time
    logger.info('')
    logger.info('=' * 60)
    logger.info('Total patches extracted: %d', total_patches_all)
    logger.info('Total time: %.1fs', elapsed_time)
    logger.info('=' * 60)
    return total_patches_all


def batch_filter_gpu(base_dir, output_base_dir, categories):
    """GPU batch filtering, calls ``process_single_folder_gpu`` per folder."""
    total_stats = {
        'total': 0, 'background_filtered': 0, 'hemorrhage_filtered': 0,
        'necrosis_filtered': 0, 'blur_filtered': 0, 'survived': 0,
        'folders_processed': 0,
    }

    all_folders = []
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        if os.path.exists(cat_dir):
            for folder in os.listdir(cat_dir):
                folder_path = os.path.join(cat_dir, folder)
                if os.path.isdir(folder_path):
                    all_folders.append((category, folder, folder_path))

    os.makedirs(output_base_dir, exist_ok=True)
    total_folders = len(all_folders)
    start_time = time.time()

    logger.info('GPU Batch Patch Filtering')
    logger.info('Source: %s', base_dir)
    logger.info('Output: %s', output_base_dir)
    logger.info('=' * 60)

    for i, (category, folder_name, folder_path) in enumerate(all_folders, 1):
        output_folder_path = os.path.join(output_base_dir, category, folder_name)
        stats = process_single_folder_gpu(folder_path, output_folder_path)
        if stats is None:
            continue

        total_stats['folders_processed'] += 1
        total_stats['total'] += stats['total']
        total_stats['background_filtered'] += stats['background_filtered']
        total_stats['hemorrhage_filtered'] += stats['hemorrhage_filtered']
        total_stats['necrosis_filtered'] += stats['necrosis_filtered']
        total_stats['blur_filtered'] += stats['blur_filtered']
        total_stats['survived'] += stats['survived']

        elapsed = time.time() - start_time
        remaining = (elapsed / i) * (total_folders - i)
        progress = i / total_folders
        bar_length = 40
        filled = int(progress * bar_length)
        bar = '#' * filled + '-' * (bar_length - filled)
        progress_write(
            f'{i}/{total_folders} [{bar}] {progress*100:.1f}% | '
            f'{category}/{folder_name} | survived: {stats["survived"]}/{stats["total"]} | '
            f'remaining: {remaining:.0f}s'
        )

    sys.stdout.write('\n')
    logger.info('=' * 60)
    logger.info('GPU batch filtering statistics')
    logger.info('=' * 60)
    _total = total_stats['total'] or 1
    logger.info('Folders processed: %d', total_stats['folders_processed'])
    logger.info('Total patches: %d', total_stats['total'])
    logger.info('Background filtered: %d (%.1f%%)', total_stats['background_filtered'],
                total_stats['background_filtered'] / _total * 100)
    logger.info('Hemorrhage filtered: %d (%.1f%%)', total_stats['hemorrhage_filtered'],
                total_stats['hemorrhage_filtered'] / _total * 100)
    logger.info('Necrosis filtered: %d (%.1f%%)', total_stats['necrosis_filtered'],
                total_stats['necrosis_filtered'] / _total * 100)
    logger.info('Blur filtered: %d (%.1f%%)', total_stats['blur_filtered'],
                total_stats['blur_filtered'] / _total * 100)
    logger.info('Survived: %d (%.1f%%)', total_stats['survived'],
                total_stats['survived'] / _total * 100)
    logger.info('Total time: %.1fs', time.time() - start_time)
    logger.info('=' * 60)


def run_normalization_gpu(source_base_dir, output_base_dir, categories,
                          reference_dir, reference_folders=None):
    """GPU batch normalization: template construction and per-folder normalization on GPU."""
    logger.info('=' * 70)
    logger.info('Batch stain normalization (GPU)')
    logger.info('=' * 70)

    if reference_folders is None:
        reference_folders = sorted(
            d for d in os.listdir(reference_dir)
            if os.path.isdir(os.path.join(reference_dir, d)) and not d.startswith('.')
        )

    logger.info('')
    logger.info('[1/2] Building reference template...')
    all_ref_paths = []
    total_ref = len(reference_folders)
    for i, name in enumerate(reference_folders, 1):
        folder = os.path.join(reference_dir, name)
        if os.path.exists(folder):
            ps = [os.path.join(folder, f) for f in os.listdir(folder)
                  if f.endswith('.png') and not f.startswith('._')]
            all_ref_paths.extend(ps)
            logger.info('  %d/%d: %s - %d patches', i, total_ref, name, len(ps))
        else:
            logger.warning('  %d/%d: %s - folder not found', i, total_ref, name)
    logger.info('Collected %d reference patches in total', len(all_ref_paths))
    template = build_reference_template_gpu(all_ref_paths)
    logger.info('  Reference vH: [%.4f, %.4f, %.4f]',
                template['vH'][0], template['vH'][1], template['vH'][2])
    logger.info('  Reference vE: [%.4f, %.4f, %.4f]',
                template['vE'][0], template['vE'][1], template['vE'][2])

    logger.info('')
    logger.info('[2/2] GPU batch normalization...')
    all_folders = []
    for category in categories:
        cat_dir = os.path.join(source_base_dir, category)
        if os.path.exists(cat_dir):
            for folder in os.listdir(cat_dir):
                folder_path = os.path.join(cat_dir, folder)
                if os.path.isdir(folder_path):
                    all_folders.append((category, folder, folder_path))

    total_folders = len(all_folders)
    start_time = time.time()
    total_norm = 0
    for i, (category, folder_name, folder_path) in enumerate(all_folders, 1):
        output_dir = os.path.join(output_base_dir, category, folder_name)
        png_files = [f for f in os.listdir(folder_path)
                     if f.endswith('.png') and not f.startswith('._')]
        if not png_files:
            continue
        od_all = collect_folder_od_gpu(folder_path)
        if od_all is None:
            vH_src = np.array([0.650, 0.704, 0.287])
            vE_src = np.array([0.072, 0.990, 0.105])
        else:
            vH_src, vE_src = get_stain_vectors(od_all)
        n = normalize_folder_gpu(folder_path, output_dir, template, vH_src, vE_src)
        total_norm += n

        elapsed = time.time() - start_time
        remaining = (elapsed / i) * (total_folders - i)
        progress = i / total_folders
        bar_length = 40
        filled = int(progress * bar_length)
        bar = '#' * filled + '-' * (bar_length - filled)
        progress_write(
            f'{i}/{total_folders} [{bar}] {progress*100:.1f}% | '
            f'{category}/{folder_name} | {n}/{len(png_files)} | remaining: {remaining:.0f}s'
        )

    sys.stdout.write('\n')
    logger.info('=' * 70)
    logger.info('GPU batch normalization completed, %d patches, total time %.1fs',
                total_norm, time.time() - start_time)
    logger.info('Output directory: %s', output_base_dir)
    logger.info('GPU status: %s', gpu_mem_str())


def process_single_file_gpu(svs_path, geojson_path, output_root,
                            reference_dir, reference_folders=None):
    """GPU full pipeline: cut -> filter -> normalize using all visible GPUs."""
    if not TORCH_AVAILABLE or not GPU_DEVICES:
        logger.error('CUDA not available, cannot use GPU pipeline')
        return
    if not os.path.exists(svs_path):
        logger.error('SVS not found: %s', svs_path)
        return
    if not os.path.exists(geojson_path):
        logger.error('GeoJSON not found: %s', geojson_path)
        return

    basename = os.path.splitext(os.path.basename(svs_path))[0]
    os.makedirs(output_root, exist_ok=True)

    logger.info('=' * 70)
    logger.info('Single-file GPU processing: %s  (GPUs: %d)', basename, len(GPU_DEVICES))
    print_gpu_info()
    logger.info('=' * 70)

    if reference_folders is None:
        reference_folders = sorted(
            d for d in os.listdir(reference_dir)
            if os.path.isdir(os.path.join(reference_dir, d)) and not d.startswith('.')
        )

    logger.info('')
    logger.info('[0/3] Building reference template (round-robin across all GPUs)...')
    all_ref_paths = []
    for name in reference_folders:
        folder = os.path.join(reference_dir, name)
        if os.path.exists(folder):
            ps = [os.path.join(folder, f) for f in os.listdir(folder)
                  if f.endswith('.png') and not f.startswith('._')]
            all_ref_paths.extend(ps)
            logger.info('  %s: %d patches', name, len(ps))
        else:
            logger.warning('  %s: folder not found', name)
    logger.info('Collected %d reference patches in total', len(all_ref_paths))
    t0 = time.time()
    template = build_reference_template_gpu(all_ref_paths)
    logger.info('  Reference vH: [%.4f, %.4f, %.4f]',
                template['vH'][0], template['vH'][1], template['vH'][2])
    logger.info('  Reference vE: [%.4f, %.4f, %.4f]',
                template['vE'][0], template['vE'][1], template['vE'][2])
    logger.info('  qH: [%.4f, %.4f]  qE: [%.4f, %.4f]',
                template['qH_low'], template['qH_high'],
                template['qE_low'], template['qE_high'])
    logger.info('  Template build elapsed %.1fs', time.time() - t0)

    cut_dir = os.path.join(output_root, basename + '_cut')
    logger.info('')
    logger.info('[1/3] GPU cutting...')
    t0 = time.time()
    n_patches = extract_patches_gpu(svs_path, geojson_path, cut_dir)
    patches_dir = os.path.join(cut_dir, basename)
    if not os.path.isdir(patches_dir):
        patches_dir = cut_dir
    logger.info('  Extracted %d patches, elapsed %.1fs', n_patches, time.time() - t0)

    filtered_dir = os.path.join(output_root, basename + '_filtered')
    logger.info('')
    logger.info('[2/3] GPU filtering...')
    t0 = time.time()
    stats = process_single_folder_gpu(patches_dir, filtered_dir)
    if stats:
        logger.info('  Total %d, background %d, hemorrhage %d, necrosis %d, '
                    'blur %d, survived %d, elapsed %.1fs',
                    stats['total'], stats['background_filtered'],
                    stats['hemorrhage_filtered'], stats['necrosis_filtered'],
                    stats['blur_filtered'], stats['survived'], time.time() - t0)
    else:
        logger.info('  No patches, elapsed %.1fs', time.time() - t0)

    normalized_dir = os.path.join(output_root, basename + '_normalized')
    logger.info('')
    logger.info('[3/3] GPU normalization...')
    if not os.path.isdir(filtered_dir):
        logger.info('  No patches after filtering, skipping normalization')
    else:
        png_files = [f for f in os.listdir(filtered_dir)
                     if f.endswith('.png') and not f.startswith('._')]
        if not png_files:
            logger.info('  No patches after filtering, skipping normalization')
        else:
            t0 = time.time()
            od_all = collect_folder_od_gpu(filtered_dir)
            if od_all is None:
                vH_src = np.array([0.650, 0.704, 0.287])
                vE_src = np.array([0.072, 0.990, 0.105])
            else:
                vH_src, vE_src = get_stain_vectors(od_all)
            n_norm = normalize_folder_gpu(filtered_dir, normalized_dir,
                                          template, vH_src, vE_src)
            logger.info('  Normalized %d patches, elapsed %.1fs', n_norm, time.time() - t0)

    cleanup_intermediate(output_root, basename)
    logger.info('')
    logger.info('All done. Only normalized results are kept. Output directory: %s',
                output_root)
    logger.info('GPU status: %s', gpu_mem_str())


def _burn_worker(device_id, seconds):
    torch.cuda.set_device(device_id)
    dev = torch.device(f'cuda:{device_id}')
    a = torch.randn(16384, 16384, device=dev)
    b = torch.randn(16384, 16384, device=dev)
    end = time.time() + seconds
    i = 0
    while time.time() < end:
        torch.mm(a, b)
        i += 1
    logger.info('  GPU%d: completed %d 16384x16384 matmuls in %ds',
                device_id, i, seconds)


def gpu_burn_test(seconds=15):
    """Burn-in test: saturate all GPUs concurrently."""
    logger.info('GPU burn-in test for %ds (across %d GPUs)...',
                seconds, len(GPU_DEVICES))
    print_gpu_info()
    t0 = time.time()
    ctx = _mp.get_context('spawn')
    with ctx.Pool(processes=len(GPU_DEVICES)) as pool:
        pool.starmap(_burn_worker, [(g, seconds) for g in GPU_DEVICES])
    logger.info('Burn-in test completed, elapsed %.1fs | %s',
                time.time() - t0, gpu_mem_str())


# =====================================================================
# CLI
# =====================================================================

DEFAULT_CATEGORIES = ['A', 'B', 'C']


def _parse_reference_folders(value):
    if not value:
        return None
    return [x.strip() for x in value.split(',') if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging verbosity')

    parser = argparse.ArgumentParser(
        prog='wsi_image_processing',
        description='WSI processing: cut -> filter -> stain normalization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # CPU batch
    p_batch = sub.add_parser('batch', parents=[common],
                             help='CPU batch processing over a directory layout')
    p_batch.add_argument('source_dir', type=Path,
                         help='Source directory containing A/B/C sub-directories')
    p_batch.add_argument('--reference-dir', type=Path, required=True,
                         help='Reference template directory (sub-directories or '
                              'folders listed via --reference-folders)')
    p_batch.add_argument('--reference-folders',
                         help='Comma-separated reference folder names; '
                              'if omitted, every sub-directory under --reference-dir is used')
    p_batch.add_argument('--categories', default=','.join(DEFAULT_CATEGORIES),
                         help='Comma-separated category sub-directory names')
    p_batch.add_argument('--keep-intermediate', action='store_true',
                         help='Keep the cut / filtered intermediate directories')

    # CPU single
    p_single = sub.add_parser('single', parents=[common], help='CPU single-file processing')
    p_single.add_argument('svs_path', type=Path)
    p_single.add_argument('geojson_path', type=Path)
    p_single.add_argument('output_dir', type=Path)
    p_single.add_argument('--reference-dir', type=Path, required=True)
    p_single.add_argument('--reference-folders')

    # GPU batch
    p_gpu_batch = sub.add_parser('gpu-batch', parents=[common], help='GPU batch processing')
    p_gpu_batch.add_argument('source_dir', type=Path)
    p_gpu_batch.add_argument('--reference-dir', type=Path, required=True)
    p_gpu_batch.add_argument('--reference-folders')
    p_gpu_batch.add_argument('--categories', default=','.join(DEFAULT_CATEGORIES))
    p_gpu_batch.add_argument('--keep-intermediate', action='store_true')

    # GPU single
    p_gpu_single = sub.add_parser('gpu-single', parents=[common],
                                  help='GPU single-file processing')
    p_gpu_single.add_argument('svs_path', type=Path)
    p_gpu_single.add_argument('geojson_path', type=Path)
    p_gpu_single.add_argument('output_dir', type=Path)
    p_gpu_single.add_argument('--reference-dir', type=Path, required=True)
    p_gpu_single.add_argument('--reference-folders')

    # GPU burn test
    p_burn = sub.add_parser('gpu-test', parents=[common], help='GPU burn-in test')
    p_burn.add_argument('--seconds', type=int, default=15)

    return parser


def _require_cuda():
    if not TORCH_AVAILABLE or not GPU_DEVICES:
        logger.error('CUDA not available, cannot use GPU mode')
        return False
    return True


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.log_level)

    categories = [c.strip() for c in args.categories.split(',') if c.strip()] \
        if hasattr(args, 'categories') else DEFAULT_CATEGORIES

    if args.command == 'gpu-test':
        if not _require_cuda():
            return 1
        gpu_burn_test(args.seconds)
        return 0

    if args.command == 'batch':
        source_dir = args.source_dir
        reference_dir = args.reference_dir
        reference_folders = _parse_reference_folders(args.reference_folders)

        if not source_dir.is_dir():
            logger.error('Source directory not found: %s', source_dir)
            return 1
        if not reference_dir.is_dir():
            logger.error('Reference directory not found: %s', reference_dir)
            return 1

        cut_out = str(source_dir) + '-path'
        filtered_out = str(source_dir) + '-path-filtered'
        normalized_out = str(source_dir) + '-path-normalized'

        os.makedirs(cut_out, exist_ok=True)

        logger.info('')
        logger.info('========== Step 1/3: Cutting ==========')
        batch_extract(str(source_dir), cut_out, categories)

        logger.info('')
        logger.info('========== Step 2/3: Filtering ==========')
        batch_filter(cut_out, filtered_out, categories)

        logger.info('')
        logger.info('========== Step 3/3: Normalization ==========')
        run_normalization(filtered_out, normalized_out, categories,
                          str(reference_dir), reference_folders)

        if not args.keep_intermediate:
            for d in (cut_out, filtered_out):
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
                    logger.info('Removed intermediate directory: %s', d)

        logger.info('')
        logger.info('All done. Only normalized results are kept. Output directory: %s',
                    normalized_out)
        return 0

    if args.command == 'single':
        if not args.svs_path.is_file():
            logger.error('SVS not found: %s', args.svs_path)
            return 1
        if not args.geojson_path.is_file():
            logger.error('GeoJSON not found: %s', args.geojson_path)
            return 1
        if not args.reference_dir.is_dir():
            logger.error('Reference directory not found: %s', args.reference_dir)
            return 1

        reference_folders = _parse_reference_folders(args.reference_folders)
        process_single_file(str(args.svs_path), str(args.geojson_path),
                            str(args.output_dir), str(args.reference_dir),
                            reference_folders)
        return 0

    if args.command == 'gpu-batch':
        if not _require_cuda():
            return 1
        source_dir = args.source_dir
        reference_dir = args.reference_dir
        reference_folders = _parse_reference_folders(args.reference_folders)

        if not source_dir.is_dir():
            logger.error('Source directory not found: %s', source_dir)
            return 1
        if not reference_dir.is_dir():
            logger.error('Reference directory not found: %s', reference_dir)
            return 1

        cut_out = str(source_dir) + '-path'
        filtered_out = str(source_dir) + '-path-filtered'
        normalized_out = str(source_dir) + '-path-normalized'
        os.makedirs(cut_out, exist_ok=True)

        logger.info('')
        logger.info('========== Step 1/3: GPU cutting ==========')
        batch_extract_gpu(str(source_dir), cut_out, categories)

        logger.info('')
        logger.info('========== Step 2/3: GPU filtering ==========')
        batch_filter_gpu(cut_out, filtered_out, categories)

        logger.info('')
        logger.info('========== Step 3/3: GPU normalization ==========')
        run_normalization_gpu(filtered_out, normalized_out, categories,
                              str(reference_dir), reference_folders)

        if not args.keep_intermediate:
            for d in (cut_out, filtered_out):
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
                    logger.info('Removed intermediate directory: %s', d)

        logger.info('')
        logger.info('All done. Only normalized results are kept. Output directory: %s',
                    normalized_out)
        return 0

    if args.command == 'gpu-single':
        if not _require_cuda():
            return 1
        if not args.svs_path.is_file():
            logger.error('SVS not found: %s', args.svs_path)
            return 1
        if not args.geojson_path.is_file():
            logger.error('GeoJSON not found: %s', args.geojson_path)
            return 1
        if not args.reference_dir.is_dir():
            logger.error('Reference directory not found: %s', args.reference_dir)
            return 1

        reference_folders = _parse_reference_folders(args.reference_folders)
        process_single_file_gpu(str(args.svs_path), str(args.geojson_path),
                                str(args.output_dir), str(args.reference_dir),
                                reference_folders)
        return 0

    return 0


if __name__ == '__main__':
    raise SystemExit(main())