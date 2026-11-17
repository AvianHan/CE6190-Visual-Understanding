#!/usr/bin/env python3
"""Independently recompute every B1 saved prediction's counts from exported GT.

Never imports model/runner code; never writes evidence. Only the requested audit
JSON is written. Pixel semantics match frozen runner.numpy_binary_counts.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

PREFIX = Path('/scratch/prj0000000262_1/hansy/ce6190-ics-runtime')
CONFIG = 'batch2_cgics_score_pool'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_bytes(path):
    if path.is_symlink():
        raise ValueError('Evidence symlink refused: ' + str(path))
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('Evidence changed during read: ' + str(path))
    return data


def read_json(path):
    return json.loads(checked_bytes(path))


def remap(root, original):
    original = Path(original)
    if '..' in original.parts:
        raise ValueError('Parent traversal refused')
    result = root / 'runtime' / original.relative_to(PREFIX)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError('Evidence path escapes root')
    return result


def binary_counts(prediction, target, ignore_label):
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError('Prediction/GT HxW dimensions differ: %s vs %s' % (prediction.shape, target.shape))
    # The runner validates full prediction masks before computing valid pixels.
    if not np.all((prediction == 0) | (prediction == 1)):
        raise ValueError('Prediction contains nonbinary pixel values')
    valid = np.ones(target.shape, dtype=bool) if ignore_label is None else target != ignore_label
    if not np.all((target[valid] == 0) | (target[valid] == 1)):
        raise ValueError('GT contains nonbinary nonignored pixel values')
    pred, truth = prediction[valid].astype(bool), target[valid].astype(bool)
    return {'tp': int(np.count_nonzero(pred & truth)),
            'fp': int(np.count_nonzero(pred & ~truth)),
            'fn': int(np.count_nonzero(~pred & truth)),
            'tn': int(np.count_nonzero(~pred & ~truth)),
            'ignored': int(np.count_nonzero(~valid))}


def audit_output(root, original, tasks):
    output = remap(root, original)
    manifest_path = remap(root, tasks[0]['manifest'])
    manifest = read_json(manifest_path)
    expected = {e['episode_id']: e for e in manifest['episodes']}
    result = {'output': original, 'dataset': tasks[0].get('dataset'),
              'expected_episodes': len(expected), 'episodes_checked': 0,
              'predictions_checked': 0, 'prediction_sha256_checks': 0,
              'gt_episode_sha256_checks': 0, 'pixel_counts_equal': 0,
              'pixel_iou_equal': 0, 'statuses': {}, 'errors': [],
              'unique_gt_files': [], 'max_iou_abs_error': 0.0,
              'prediction_pixel_total': 0}
    errors, seen, statuses, gt_cache = result['errors'], set(), Counter(), {}
    for row_path in sorted((output / 'episodes').glob('*.json')):
        row = read_json(row_path)
        eid = row.get('episode_id')
        statuses[str(row.get('status'))] += 1
        if eid in seen or eid not in expected:
            errors.append({'episode': eid, 'error': 'Duplicate or unexpected episode ID'})
            continue
        seen.add(eid)
        ep = expected[eid]
        result['episodes_checked'] += 1
        try:
            if row.get('status') != 'ok':
                raise ValueError('Episode not successful')
            query = ep['query']
            gt_path = remap(root, PREFIX / 'data' / query['mask'])
            if gt_path not in gt_cache:
                gt_bytes = checked_bytes(gt_path)
                with Image.open(io.BytesIO(gt_bytes)) as image:
                    if image.format != 'PNG':
                        raise ValueError('GT file is not PNG')
                    gt = np.array(image)
                gt_cache[gt_path] = (digest(gt_bytes), gt)
            gt_hash, gt = gt_cache[gt_path]
            if gt_hash != query['mask_sha256'] or gt_hash != row['query_mask_sha256']:
                raise ValueError('Query GT SHA256 differs from manifest/episode')
            result['gt_episode_sha256_checks'] += 1
            if row['query_image_sha256'] != query['image_sha256']:
                raise ValueError('Query image hash label differs from manifest')
            if (row['dataset'], row['fold'], row['concept_id']) != (ep['dataset'], ep['fold'], ep['concept_id']):
                raise ValueError('Dataset/fold/concept identity differs')
            alternatives = row['alternative_predictions']
            if set(alternatives) != {'ref_only', 'query_only'}:
                raise ValueError('Expected exactly both alternative rules')
            records = {'combined': row, **alternatives}
            for role, record in records.items():
                try:
                    rel = Path(record['prediction_path'])
                    if rel.is_absolute() or '..' in rel.parts or rel.parts[0] != 'predictions':
                        raise ValueError('Unsafe prediction path')
                    prediction_path = output / rel
                    if not prediction_path.resolve().is_relative_to(output.resolve()):
                        raise ValueError('Prediction escaped output')
                    image_bytes = checked_bytes(prediction_path)
                    if digest(image_bytes) != record['prediction_sha256']:
                        raise ValueError('Prediction SHA256 mismatch')
                    result['prediction_sha256_checks'] += 1
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        if image.format != 'PNG':
                            raise ValueError('Prediction file is not PNG')
                        pred = np.array(image)
                    counts = binary_counts(pred, gt, ep['ignore_label'])
                    result['predictions_checked'] += 1
                    result['prediction_pixel_total'] += int(pred.size)
                    saved_counts = record['counts']
                    if any(type(saved_counts.get(k)) is not int or saved_counts[k] != v for k, v in counts.items()):
                        raise ValueError('Pixel confusion counts differ: expected=%s saved=%s' % (counts, saved_counts))
                    result['pixel_counts_equal'] += 1
                    union = counts['tp'] + counts['fp'] + counts['fn']
                    actual_iou = counts['tp']/union if union else None
                    saved_iou = record['iou']
                    if actual_iou is None:
                        if saved_iou is not None:
                            raise ValueError('Empty union must have null IoU')
                    elif (not isinstance(saved_iou, (int, float)) or isinstance(saved_iou, bool)
                          or not math.isfinite(saved_iou) or not math.isclose(actual_iou, saved_iou, rel_tol=0, abs_tol=1e-12)):
                        raise ValueError('Pixel IoU differs: %.17g vs %s' % (actual_iou, saved_iou))
                    elif actual_iou is not None:
                        result['max_iou_abs_error'] = max(result['max_iou_abs_error'], abs(actual_iou-saved_iou))
                    result['pixel_iou_equal'] += 1
                except Exception as exc:
                    errors.append({'episode': eid, 'role': role, 'error': str(exc)})
        except Exception as exc:
            errors.append({'episode': eid, 'error': str(exc)})
    if seen != set(expected):
        errors.append({'error': 'Missing or extra episode IDs', 'missing': sorted(set(expected)-seen), 'extra': sorted(seen-set(expected))})
    result['statuses'] = dict(statuses)
    result['unique_gt_files'] = [str(p.relative_to(root)) for p in sorted(gt_cache)]
    result['status'] = 'PASS' if not errors and result['predictions_checked'] == len(expected)*3 else 'FAIL'
    return result


def audit(root, workers=4):
    root = Path(root).resolve()
    plan = read_json(root / 'runtime/runs/ics-batch2-20261116/scopes/core/plan.json')
    groups = defaultdict(list)
    for task in plan['tasks']:
        if task['config_data']['experiment_id'] == CONFIG:
            groups[task['output']].append(task)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(audit_output, root, path, tasks) for path, tasks in sorted(groups.items())]
        outputs = [future.result() for future in futures]
    counters = ('expected_episodes', 'episodes_checked', 'predictions_checked',
                'prediction_sha256_checks', 'gt_episode_sha256_checks', 'pixel_counts_equal',
                'pixel_iou_equal', 'prediction_pixel_total')
    totals = {key: sum(x[key] for x in outputs) for key in counters}
    gt_paths = sorted({p for x in outputs for p in x['unique_gt_files']})
    frozen = root / 'runtime/runs/ics-batch2-20261116/source/ics/runner.py'
    return {'status': 'PASS' if all(x['status'] == 'PASS' for x in outputs) and len(groups) == 14 else 'FAIL',
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'evidence_root': str(root),
            'elapsed_seconds': time.monotonic()-started, 'output_count': len(groups),
            'totals': totals, 'unique_gt_files_checked': len(gt_paths),
            'max_iou_abs_error': max(x['max_iou_abs_error'] for x in outputs),
            'error_count': sum(len(x['errors']) for x in outputs),
            'frozen_runner_sha256': digest(checked_bytes(frozen)),
            'semantics': 'Binary original-resolution masks, valid pixels exclude manifest ignore_label; tp/fp/fn/tn/ignored recomputed independently, IoU=tp/(tp+fp+fn), null only for empty union.',
            'boundaries': ['No evidence files were modified; this process reads existing files and writes only this audit JSON.',
                           'All saved prediction PNGs and manifest-selected query GT masks were verified; no sampled subset.',
                           'Raw RGB images and reference mask assets were not exported, so original image dimensions, reference pixels and dataset-source conversion cannot be independently reverified here.',
                           'This audit does not rerun models, regenerate candidate concepts, or validate upstream dataset annotations.'],
            'outputs': outputs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    report = audit(args.evidence, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'totals', 'unique_gt_files_checked', 'error_count', 'max_iou_abs_error', 'elapsed_seconds')}, indent=2))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)
