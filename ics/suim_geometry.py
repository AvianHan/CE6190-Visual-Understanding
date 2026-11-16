"""Explicit, pre-sampling SUIM shape-valid subset; no geometry transforms.

Every supported source raster is hashed and inspected before sampling. Only a
confirmed image/mask dimension mismatch makes a source pair ineligible. Decode,
missing-file, EXIF, membership and source-identity errors remain fatal. Original
files are never changed, and a nonempty class cannot silently disappear.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from .data import DatasetIndex, _contained, _image_for_stem, file_sha256

SUIM_REPAIR_IMPLEMENTATION_ID = 'suim-strict-geometry-filter-v1'
CLASSES = ('FV', 'HD', 'PF', 'RI', 'RO', 'SR', 'WR')
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}
INSID3_PIN = '0c165a10cf52ab91f335883d06260de86854adbe'
SUIM_LOADER_SHA256 = 'b59e6cd1d3d2f0d6a02916c9dd8899245d1ea01f132fd2ae742bec4604e2b54a'
POLICY = {
    'id': SUIM_REPAIR_IMPLEMENTATION_ID,
    'scope': 'all source masks in all seven author classes, including empty masks, before query/reference sampling',
    'pair_exclusion': 'confirmed decoded mask dimensions differ from decoded raw-image dimensions',
    'other_errors': 'fatal; never caught and excluded',
    'exif': 'raw raster coordinates; nonidentity image or mask orientation is fatal, no transpose',
    'empty_masks': 'original author nonempty-mask membership retained; separately counted, not a new repair exclusion',
    'geometry_transform': 'none: no resize, crop, rotation, padding or GT editing',
    'class_rule': 'all seven original classes retained with at least two distinct valid nonempty image hashes',
    'query_rule': 'original requested unique-query count remains strict; no repeated-query substitution',
    'interpretation': 'explicit shape-valid subset of author-prepared merged SUIM; not the full author pool or official test-only benchmark',
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _publish_json(path, value):
    from .batch2_data import _publish_json as publish
    publish(path, value)


def _source_evidence(runtime):
    source = Path(runtime) / 'sources/INSID3'
    pin = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                  text=True, timeout=20).strip()
    loader = source / 'datasets/suim.py'
    if pin != INSID3_PIN or loader.is_symlink() or file_sha256(loader) != SUIM_LOADER_SHA256:
        raise ValueError('SUIM source loader differs from the audited INSID3 pin')
    return {'commit': pin, 'loader_path': str(loader), 'loader_sha256': SUIM_LOADER_SHA256,
            'membership': 'seven fixed classes, nonempty masks; image stem matching',
            'repair_is_project_subset': True}


def _regular(root, path):
    path = Path(path)
    resolved = _contained(root, path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('SUIM source must be a regular existing file: ' + str(path))
    return resolved


def _file_record(root, path, *, mask=False):
    path = _regular(root, path)
    before = path.stat()
    with Image.open(path) as image:
        # Decode rather than trust a readable header on a truncated raster.
        image.load()
        record = {'path': path.relative_to(root).as_posix(), 'bytes': before.st_size,
                  'size_wh': list(image.size), 'mode': image.mode,
                  'format': image.format, 'exif_orientation': image.getexif().get(274)}
        if mask:
            raw = np.asarray(image.convert('L'))
            binary = (raw > 0 if raw.max(initial=0) <= 1 else raw >= 128).astype(np.uint8)
            record['foreground_pixels'] = int(binary.sum())
            record['binary_pixels_sha256'] = hashlib.sha256(binary.tobytes()).hexdigest()
    record['sha256'] = file_sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('SUIM source changed while being audited: ' + str(path))
    return record


def _inventory_paths(root):
    base = _contained(root, root / 'SUIM')
    images = _contained(root, base / 'images')
    if not images.is_dir():
        raise FileNotFoundError('SUIM image directory missing')
    image_paths = sorted(p for p in images.iterdir() if p.suffix.lower() in EXTENSIONS)
    mask_paths, issues = [], []
    for cid in CLASSES:
        folder = _contained(root, base / 'masks' / cid)
        if not folder.is_dir():
            issues.append({'class': cid, 'path': str(folder), 'error': 'missing_class_directory'})
            continue
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
        if not files:
            issues.append({'class': cid, 'path': str(folder), 'error': 'no_source_masks'})
        mask_paths.extend((cid, p) for p in files)
    return images, image_paths, mask_paths, issues


def audit_suim_pool(data_root):
    """Return complete audit; failures are listed, never converted to exclusions."""
    root = Path(data_root).resolve()
    image_folder, image_paths, mask_paths, issues = _inventory_paths(root)
    images, masks = {}, []
    for path in image_paths:
        try:
            row = _file_record(root, path)
            images[row['path']] = row
            if row['exif_orientation'] not in (None, 1):
                issues.append({'path': row['path'], 'error': 'nonidentity_image_exif',
                               'orientation': row['exif_orientation']})
        except Exception as exc:
            issues.append({'path': str(path), 'error': type(exc).__name__, 'detail': str(exc)})
    for cid, path in mask_paths:
        try:
            row = _file_record(root, path, mask=True)
            row['class'] = cid
            image = _regular(root, _image_for_stem(image_folder, path.stem))
            image_rel = image.relative_to(root).as_posix()
            row['image_path'] = image_rel
            target = images.get(image_rel)
            if target is None:
                raise ValueError('SUIM matching image did not pass its source audit')
            row['image_sha256'] = target['sha256']
            row['image_size_wh'] = target['size_wh']
            row['shape_matches'] = row['size_wh'] == target['size_wh']
            row['author_nonempty'] = row['foreground_pixels'] > 0
            row['eligible'] = row['shape_matches'] and row['author_nonempty']
            row['ineligible_reason'] = (None if row['eligible'] else
                'mask_image_dimensions_mismatch' if not row['shape_matches'] else 'original_author_empty_mask')
            if row['exif_orientation'] not in (None, 1):
                issues.append({'path': row['path'], 'error': 'nonidentity_mask_exif',
                               'orientation': row['exif_orientation']})
            masks.append(row)
        except Exception as exc:
            issues.append({'path': str(path), 'error': type(exc).__name__, 'detail': str(exc)})
    counts = {}
    for cid in CLASSES:
        rows = [r for r in masks if r['class'] == cid]
        counts[cid] = {'source_pairs': len(rows), 'author_nonempty_pairs': sum(r['author_nonempty'] for r in rows),
                      'dimension_mismatch_pairs': sum(not r['shape_matches'] for r in rows),
                      'excluded_nonempty_pairs': sum(r['author_nonempty'] and not r['shape_matches'] for r in rows),
                      'eligible_nonempty_pairs': sum(r['eligible'] for r in rows),
                      'eligible_distinct_image_hashes': len({r['image_sha256'] for r in rows if r['eligible']})}
        if counts[cid]['eligible_distinct_image_hashes'] < 2:
            issues.append({'class': cid, 'error': 'fewer_than_two_distinct_valid_nonempty_images',
                           'count': counts[cid]['eligible_distinct_image_hashes']})
    exclusions = [{k: row[k] for k in ('path', 'class', 'sha256', 'size_wh', 'image_path',
                   'image_sha256', 'image_size_wh', 'foreground_pixels', 'author_nonempty', 'ineligible_reason')}
                  for row in masks if not row['shape_matches']]
    result = {'schema_version': 1, 'implementation_id': SUIM_REPAIR_IMPLEMENTATION_ID,
              'data_root': str(root), 'policy': POLICY, 'status': 'AUDITED' if not issues else 'BLOCKED',
              'images': list(images.values()), 'masks': masks, 'issues': issues, 'exclusions': exclusions,
              'counts_by_class': counts, 'source_image_count': len(image_paths),
              'source_mask_count': len(mask_paths),
              'original_nonempty_pair_count': sum(r['author_nonempty'] for r in masks),
              'eligible_nonempty_pair_count': sum(r['eligible'] for r in masks),
              'excluded_nonempty_pair_count': sum(r['author_nonempty'] and not r['shape_matches'] for r in masks),
              'dimension_mismatch_pair_count': len(exclusions),
              'dimension_mismatch_image_count': len({r['image_path'] for r in masks if not r['shape_matches']}),
              'eligible_distinct_image_count': len({r['image_sha256'] for r in masks if r['eligible']}),
              'raw_files_modified': False, 'geometry_transforms_performed': False}
    result['audit_sha256'] = _digest(result)
    return result


def prepare_suim_geometry(runtime, output_dir):
    runtime, output = Path(runtime).resolve(), Path(output_dir).resolve()
    path = output / 'suim_geometry_receipt.json'
    if path.exists():
        return verify_suim_geometry(runtime, path)
    source = _source_evidence(runtime)
    audit = audit_suim_pool(runtime / 'data')
    audit_path = output / ('suim_geometry_inputs-' + audit['audit_sha256'] + '.json')
    _publish_json(audit_path, audit)
    if audit['issues']:
        raise ValueError(f'SUIM source audit has {len(audit["issues"])} fatal issue(s); '
                         f'no manifest published; see {audit_path}; first issue: {audit["issues"][0]}')
    receipt = {'schema_version': 1, 'status': 'READY', 'implementation_id': SUIM_REPAIR_IMPLEMENTATION_ID,
               'policy': POLICY, 'source_evidence': source, 'input_audit_path': str(audit_path),
               'input_audit_sha256': file_sha256(audit_path), 'audit': audit,
               'raw_files_modified': False, 'geometry_transforms_performed': False}
    receipt['receipt_sha256'] = _digest(receipt)
    _publish_json(path, receipt)
    return receipt


def verify_suim_geometry(runtime, receipt_path):
    runtime, path = Path(runtime).resolve(), Path(receipt_path)
    if path.is_symlink():
        raise ValueError('SUIM receipt must not be a symlink')
    receipt = json.loads(path.read_text())
    if (receipt.get('status') != 'READY' or receipt.get('implementation_id') != SUIM_REPAIR_IMPLEMENTATION_ID or
            receipt.get('policy') != POLICY or
            _digest({k: v for k, v in receipt.items() if k != 'receipt_sha256'}) != receipt.get('receipt_sha256')):
        raise ValueError('Invalid immutable SUIM geometry receipt')
    if _source_evidence(runtime) != receipt['source_evidence']:
        raise ValueError('SUIM source loader evidence changed')
    audit_path = Path(receipt['input_audit_path'])
    if audit_path.is_symlink() or audit_path.resolve().parent != path.resolve().parent:
        raise ValueError('SUIM input audit is outside its scope')
    if file_sha256(audit_path) != receipt['input_audit_sha256']:
        raise ValueError('SUIM input audit bytes changed')
    saved = json.loads(audit_path.read_text())
    if saved != receipt['audit']:
        raise ValueError('SUIM receipt and complete input audit disagree')
    # Re-audit all raw candidates, including excluded and empty pairs. New
    # files, deleted files and unselected-asset mutations invalidate a resume.
    current = audit_suim_pool(runtime / 'data')
    if current['issues'] or current != saved:
        raise ValueError('SUIM original source pool changed since subset preparation')
    return receipt


def apply_suim_geometry(index, data_root, receipt):
    if index.dataset != 'suim' or receipt.get('implementation_id') != SUIM_REPAIR_IMPLEMENTATION_ID:
        raise ValueError('SUIM subset policy cannot be applied to another dataset')
    root = Path(data_root).resolve()
    audit = receipt['audit']
    by_path = {row['path']: row for row in audit['masks']}
    samples, observed, excluded = {}, set(), set()
    for cid, pool in index.samples.items():
        samples[cid] = []
        for sample in pool:
            rel = _regular(root, sample.mask_source).relative_to(root).as_posix()
            row = by_path[rel]
            if row['class'] != cid or not row['author_nonempty']:
                raise ValueError('SUIM author membership disagrees with full-pool audit')
            if sample.image.resolve().relative_to(root).as_posix() != row['image_path']:
                raise ValueError('SUIM image/mask pairing differs from audited membership')
            if rel in observed:
                raise ValueError('Duplicate SUIM source mask in original membership')
            observed.add(rel)
            if row['eligible']:
                # Keep the same Sample and raw image/mask paths. No transform.
                samples[cid].append(sample)
            else:
                if row['ineligible_reason'] != 'mask_image_dimensions_mismatch':
                    raise ValueError('Unexpected SUIM eligibility exclusion')
                excluded.add(rel)
    expected = {r['path'] for r in audit['masks'] if r['author_nonempty']}
    if (observed != expected or set(samples) != set(CLASSES) or
            sum(map(len, samples.values())) != audit['eligible_nonempty_pair_count'] or
            len(excluded) != audit['excluded_nonempty_pair_count']):
        raise ValueError('SUIM subset lost source membership or changed the seven-class universe')
    provenance = dict(index.provenance)
    provenance['suim_geometry'] = {'implementation_id': SUIM_REPAIR_IMPLEMENTATION_ID,
        'receipt_sha256': receipt['receipt_sha256'], 'input_audit_sha256': receipt['input_audit_sha256'],
        'policy': POLICY, 'counts_by_class': audit['counts_by_class'],
        'original_nonempty_pairs': audit['original_nonempty_pair_count'],
        'eligible_nonempty_pairs': audit['eligible_nonempty_pair_count'],
        'excluded_nonempty_pairs': audit['excluded_nonempty_pair_count'],
        'excluded_pair_count_including_empty': audit['dimension_mismatch_pair_count'],
        'eligible_distinct_images': audit['eligible_distinct_image_count']}
    return DatasetIndex(index.dataset, index.version + '-shape-valid-v1', index.split,
                        index.fold, samples, provenance)


def verify_manifest_pairs(manifest, data_root, receipt):
    """Bind every selected GT/reference mask back to a valid original pair."""
    root = Path(data_root).resolve()
    allowed = {}
    for row in receipt['audit']['masks']:
        if row['eligible']:
            key = (row['class'], row['image_path'], row['image_sha256'])
            allowed.setdefault(key, set()).add((row['binary_pixels_sha256'], tuple(row['size_wh'])))
    pixels = {}
    for episode in manifest['episodes']:
        if episode['dataset'] != 'suim':
            raise ValueError('Non-SUIM episode in SUIM shape-valid scope')
        for asset in [episode['query'], *episode['references']]:
            key = (episode['concept_id'], asset['image'], asset['image_sha256'])
            if key not in allowed:
                raise ValueError('SUIM manifest uses an ineligible image/class pair')
            if asset['mask'] not in pixels:
                mask = _regular(root, root / asset['mask'])
                with Image.open(mask) as image:
                    array = np.asarray(image)
                if array.ndim != 2 or array.dtype != np.uint8 or not np.isin(array, (0, 1)).all():
                    raise ValueError('SUIM materialized mask is not the original binary raster')
                pixels[asset['mask']] = (hashlib.sha256(array.tobytes()).hexdigest(),
                                         tuple(reversed(array.shape)))
            if pixels[asset['mask']] not in allowed[key]:
                raise ValueError('SUIM materialized mask differs from every valid original pair')
