"""Frozen ICS episode construction from author-prepared datasets.

This is a unified sampler, NOT the authors' stochastic dataloader.  It does not
import either model, and it never supplies query annotations to an adapter.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import pickle
import random
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ics_eval import manifest_fingerprint, validate_manifest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _contained(root: Path, path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Asset escapes dataset root: {path}')
    return path


@dataclass
class Sample:
    concept_id: str
    image: Path
    mask_source: Any
    mask_kind: str = 'binary'
    foreground_label: int | None = None
    source_ignore: int | None = None
    group_id: str | None = None


@dataclass
class DatasetIndex:
    dataset: str
    version: str
    split: str
    fold: int
    samples: dict[str, list[Sample]]
    provenance: dict = field(default_factory=dict)


def _pickle(path: Path, trusted: bool):
    if not trusted:
        raise ValueError('Prepared COCO/LVIS metadata uses pickle. Verify its source, then '
                         'supply --trust-prepared-pickle; pickle may execute code.')
    with path.open('rb') as stream:
        return pickle.load(stream)


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError('Expected one of: ' + ', '.join(map(str, paths)))


def _image_for_stem(folder: Path, stem: str) -> Path:
    return _first_existing([folder / (stem + ext) for ext in ('.jpg', '.png', '.jpeg')])


def build_index(dataset: str, data_root: Path, *, fold: int = 0,
                split: str = 'val', split_file: Path | None = None,
                trust_prepared_pickle: bool = False,
                group_map: dict[str, str] | None = None) -> DatasetIndex:
    """Read immutable sample membership; masks are decoded only if selected.

    group_map keys are image paths relative to data_root. For iSAID, documented
    DOTA P#### prefix names can be inferred; otherwise an explicit map is needed.
    """
    root = Path(data_root).resolve()
    aliases = {'coco20i': 'coco', 'lvis92i': 'lvis', 'fss1000': 'fss',
               'pascal5i': 'pascal', 'isic2018': 'isic', 'isaid5i': 'isaid'}
    dataset = aliases.get(dataset, dataset)
    samples: dict[str, list[Sample]] = defaultdict(list)
    provenance: dict = {'membership': 'author_prepared', 'source_files': [],
                        'sampler': 'ce6190_unified_class_balanced_v1',
                        'official_rng_reproduction': False}
    versions = {'coco': 'COCO2014-author-semantic-masks',
                'lvis': 'LVIS-author-lvis_val.pkl', 'fss': 'FSS-1000',
                'pascal': 'VOC2012-SegmentationClassAug',
                'isic': 'ISIC2018-Task1-author-reorganized',
                'suim': 'SUIM-merged-author-foreground-7',
                'isaid': 'iSAID-5i-author-patches'}
    if dataset not in versions:
        raise ValueError(f'Unsupported dataset: {dataset}')

    def source(path: Path):
        provenance['source_files'].append({'path': str(path), 'sha256': file_sha256(path)})

    if dataset == 'coco':
        if split != 'val' or fold not in range(4):
            raise ValueError('COCO adapter currently supports val folds 0..3 only')
        base = _first_existing([root / 'COCO2014', root / 'coco'])
        membership = split_file or base / 'splits' / 'val' / f'fold{fold}.pkl'
        metadata = _pickle(membership, trust_prepared_pickle)
        source(membership)
        for cid in [fold + 4 * i for i in range(20)]:
            if not metadata.get(cid):
                raise ValueError(f'COCO fold {fold} has no samples for required class {cid}')
            for name in sorted(set(metadata[cid])):
                name = str(name)
                samples[str(cid)].append(Sample(str(cid), base / name,
                    base / 'annotations' / str(Path(name).with_suffix('.png')),
                    'semantic', cid + 1))
        provenance['void_policy'] = '255 becomes background, matching anchor COCO loaders'
        provenance['class_fold_rule'] = 'zero-based class id modulo 4'
    elif dataset == 'lvis':
        if split != 'val' or fold not in range(10):
            raise ValueError('LVIS adapter currently supports val folds 0..9 only')
        base = root / 'LVIS'
        membership = split_file or base / 'lvis_val.pkl'
        metadata = _pickle(membership, trust_prepared_pickle)
        source(membership)
        # Keep author insertion order, not lexical sorting: it determines folds.
        eligible = [cid for cid in metadata if len(metadata[cid]) > 1]
        per_fold = len(eligible) // 10
        selected = [eligible[fold + 10 * i] for i in range(per_fold)]
        if not selected:
            raise ValueError('No LVIS classes in selected fold')
        for cid in selected:
            for name, info in sorted(metadata[cid].items()):
                samples[str(cid)].append(Sample(str(cid), base / 'coco' / name,
                    [ann['segmentation'] for ann in info['annotations']], 'lvis'))
        provenance.update({'eligible_categories': len(eligible),
            'selected_category_ids_in_author_order': list(map(str, selected)),
            'category_remainder_excluded_by_author_rule': len(eligible) % 10,
            'class_universe_eligibility': 'more than 1 image; unchanged across n-shot settings',
            'annotation_semantics': 'union of all same-category annotations in image'})
    elif dataset == 'fss':
        if split not in ('test', 'val', 'trn') or fold != 0:
            raise ValueError('FSS supports trn/val/test class splits, fold 0 only')
        base = root / 'FSS-1000'
        membership = split_file or base / 'splits' / f'{split}.txt'
        source(membership)
        categories = sorted(set(x.strip() for x in membership.read_text().splitlines() if x.strip()))
        for cid in categories:
            if '/' in cid or '\\' in cid or cid in ('.', '..'):
                raise ValueError('Unsafe FSS class name')
            for image in sorted((base / 'data' / cid).glob('*.jpg')):
                samples[cid].append(Sample(cid, image, image.with_suffix('.png')))
            if cid not in samples:
                raise ValueError(f'No images for FSS split class {cid}')
        provenance['reference_limit'] = 'distinct images only; normally at most 9 per query'
    elif dataset == 'pascal':
        if split != 'val' or fold not in range(4):
            raise ValueError('PASCAL adapter currently supports val folds 0..3 only')
        base = root / 'VOC2012'
        membership = split_file or root / 'splits' / 'pascal' / 'val' / f'fold{fold}.txt'
        source(membership)
        seen = set()
        for line in membership.read_text().splitlines():
            if not line.strip():
                continue
            name, label = line.strip().rsplit('__', 1)
            cid = int(label) - 1
            if cid not in range(fold * 5, fold * 5 + 5):
                raise ValueError(f'PASCAL class {cid} inconsistent with fold {fold}')
            if (name, cid) not in seen:
                samples[str(cid)].append(Sample(str(cid), base / 'JPEGImages' / (name + '.jpg'),
                    base / 'SegmentationClassAug' / (name + '.png'), 'semantic', cid + 1, 255))
                seen.add((name, cid))
        provenance['void_policy'] = 'query 255 ignored; reference void treated as background'
    elif dataset == 'isic':
        if fold != 0:
            raise ValueError('ISIC directories 1/2/3 are sampled as classes, not held-out folds; use fold0')
        base = root / 'ISIC'
        for cid in ('1', '2', '3'):
            folder = base / 'ISIC2018_Task1-2_Training_Input' / cid
            if not folder.is_dir():
                raise FileNotFoundError(f'Run author DR-Adapter class reorganization first: {folder}')
            for image in sorted(folder.glob('*.jpg')):
                samples[cid].append(Sample(cid, image, base / 'ISIC2018_Task1_Training_GroundTruth' /
                                          (image.stem + '_segmentation.png')))
            if cid not in samples:
                raise ValueError(f'No images for ISIC class directory {cid}')
        split = 'author_reorganized_public_training'
        provenance['warning'] = 'Not ISIC official challenge test; groups are not patient IDs'
    elif dataset == 'suim':
        if fold != 0:
            raise ValueError('SUIM merged protocol uses fold 0 only')
        base = root / 'SUIM'
        for cid in ('FV', 'HD', 'PF', 'RI', 'RO', 'SR', 'WR'):
            for mask in sorted((base / 'masks' / cid).glob('*')):
                if mask.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp'):
                    continue
                if np.any(np.asarray(Image.open(mask).convert('L')) >= 128):
                    samples[cid].append(Sample(cid, _image_for_stem(base / 'images', mask.stem), mask))
        split = 'author_merged'
        provenance['warning'] = 'Merged 1635-image source, not the original SUIM 110-image test split'
    elif dataset == 'isaid':
        if split != 'val' or fold not in range(3):
            raise ValueError('iSAID adapter supports val folds 0..2 only')
        base = root / 'iSAID'
        membership = split_file or base / 'splits' / 'val' / f'fold{fold}.txt'
        source(membership)
        seen = set()
        for line in membership.read_text().splitlines():
            if not line.strip():
                continue
            name, label = line.strip().rsplit('__', 1)
            cid = int(label) - 1
            if cid not in range(fold * 5, fold * 5 + 5):
                raise ValueError(f'iSAID class {cid} inconsistent with fold {fold}')
            if (name, cid) not in seen:
                match = re.match(r'^(P[0-9]+)(?:_|$)', name)
                image = base / 'val' / 'images' / (name + '.png')
                rel = str(image.relative_to(root))
                group = (group_map or {}).get(rel) or (match.group(1) if match else None)
                if group is None:
                    raise ValueError(f'Cannot infer original aerial image for {name}; provide --group-map')
                samples[str(cid)].append(Sample(str(cid), image,
                    base / 'val' / 'semantic_png' / (name + '_instance_color_RGB.png'),
                    'semantic', cid + 1, 255, group))
                seen.add((name, cid))
        provenance['group_policy'] = 'reference and query must come from different original aerial images'
    for cid, pool in samples.items():
        for item in pool:
            item.image = _contained(root, item.image)
            if isinstance(item.mask_source, Path):
                item.mask_source = _contained(root, item.mask_source)
            rel = item.image.relative_to(root).as_posix()
            if group_map and rel in group_map:
                item.group_id = str(group_map[rel])
            elif group_map and item.group_id is None:
                raise ValueError(f'Group map is incomplete for {rel}; cannot certify group-independent pairing')
        if not pool:
            raise ValueError(f'Empty class {cid}')
    if not samples:
        raise ValueError('No indexed samples; check layout and split files')
    if group_map:
        provenance['group_map_sha256'] = hashlib.sha256(json.dumps(group_map, sort_keys=True).encode()).hexdigest()
    return DatasetIndex(dataset, versions[dataset], split, fold, dict(samples), provenance)


def read_binary_mask(sample: Sample, *, reference: bool = False) -> np.ndarray:
    """0/1 plus optional query ignore=255. Reference masks NEVER contain 255."""
    with Image.open(sample.image) as image:
        width, height = image.size
    if sample.mask_kind == 'lvis':
        mask = np.zeros((height, width), dtype=bool)
        for segmentation in sample.mask_source:
            if isinstance(segmentation, np.ndarray):
                one = segmentation
            else:
                try:
                    import pycocotools.mask as mask_util
                except ImportError as exc:
                    raise RuntimeError('LVIS polygon/RLE masks require pycocotools') from exc
                if isinstance(segmentation, list):
                    if not segmentation:
                        continue
                    rle = mask_util.merge(mask_util.frPyObjects(segmentation, height, width))
                elif isinstance(segmentation, dict):
                    rle = segmentation
                    if isinstance(rle.get('counts'), list):
                        rle = mask_util.frPyObjects(rle, height, width)
                else:
                    raise TypeError('Unsupported LVIS annotation')
                one = mask_util.decode(rle)
            if one.shape != mask.shape:
                raise ValueError('LVIS annotation/image dimensions differ')
            mask |= one.astype(bool)
        result = mask.astype(np.uint8)
    else:
        with Image.open(sample.mask_source) as image:
            raw = np.asarray(image if sample.mask_kind == 'semantic' else image.convert('L'))
        if raw.ndim != 2:
            raise ValueError('Expected an indexed semantic mask, not RGB color labels: ' + str(sample.mask_source))
        if sample.mask_kind == 'semantic':
            result = (raw == sample.foreground_label).astype(np.uint8)
            if sample.source_ignore is not None and not reference:
                result[raw == sample.source_ignore] = 255
        elif sample.mask_kind == 'binary':
            # Official FSS/ISIC/SUIM masks are 0/255; accepting already normalized
            # 0/1 masks makes synthetic and materialized source tests unambiguous.
            result = (raw > 0 if raw.max(initial=0) <= 1 else raw >= 128).astype(np.uint8)
        else:
            raise ValueError('Unknown mask kind')
    if result.shape != (height, width):
        raise ValueError('Mask/image dimensions differ: ' + str(sample.image))
    if not np.any(result == 1):
        raise ValueError('Selected source mask is empty: ' + str(sample.image))
    return result


def create_manifest(index: DatasetIndex, data_root: Path, *, queries: int,
                    references_per_query: int = 1, shots: int = 1, seed: int = 0,
                    query_seed: int = 0, insufficient_references: str = 'all',
                    protocol_id: str | None = None) -> tuple[dict, dict]:
    """Deterministic class-balanced draws and distinct support sets.

    Query sampling and reference sampling use independent RNGs, so a main run
    and robustness run can share queries by keeping query_seed/queries fixed.
    """
    root = Path(data_root).resolve()
    if queries < 0 or references_per_query < 1 or shots < 1:
        raise ValueError('queries>=0, references_per_query>=1 and shots>=1 required')
    if insufficient_references not in ('all', 'error'):
        raise ValueError('insufficient_references must be all or error')
    qrng, rrng = random.Random(query_seed), random.Random(seed)
    classes = list(index.samples)
    if not classes:
        raise ValueError('Empty index')
    draws = []
    if queries == 0:
        draws = [(cid, sample) for cid in classes for sample in index.samples[cid]]
    else:
        # Cycle shuffled classes so counts differ by at most one, and cycle a
        # shuffled per-class query pool before reusing a query.
        pools = {}
        for cid in classes:
            pools[cid] = list(index.samples[cid])
            qrng.shuffle(pools[cid])
        order = list(classes)
        qrng.shuffle(order)
        for i in range(queries):
            cid = order[i % len(order)]
            if not pools[cid]:
                pools[cid] = list(index.samples[cid])
                qrng.shuffle(pools[cid])
            draws.append((cid, pools[cid].pop()))
    hash_cache = {}
    asset_cache = {}
    def digest(path: Path):
        key = str(path)
        if key not in hash_cache:
            hash_cache[key] = file_sha256(path)
        return hash_cache[key]

    def asset(sample: Sample, reference: bool):
        key = (str(sample.image), sample.concept_id, reference)
        if key not in asset_cache:
            mask = read_binary_mask(sample, reference=reference)
            content_id = hashlib.sha256(mask.tobytes() + str(mask.shape).encode()).hexdigest()
            role = 'reference' if reference else 'query'
            path = root / 'ics_generated_masks' / index.dataset / role / (content_id + '.png')
            if path.exists():
                # A content-addressed filename is not proof that the bytes still
                # represent the original mask. Never bless an edited/corrupt GT
                # cache simply by hashing its current contents into a new run.
                try:
                    with Image.open(path) as cached_image:
                        cached = np.asarray(cached_image)
                    matches = (cached.dtype == np.uint8 and
                               cached.shape == mask.shape and
                               np.array_equal(cached, mask))
                except (OSError, ValueError):
                    matches = False
                if not matches:
                    raise ValueError(
                        f'Generated mask cache mismatch: {path}. The existing file was '
                        'preserved. Inspect/archive the damaged file, then explicitly '
                        'remove it or use a clean data root before rebuilding the manifest.'
                    )
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(suffix='.png', dir=path.parent, delete=False) as handle:
                    temporary = Path(handle.name)
                try:
                    Image.fromarray(mask).save(temporary)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            asset_cache[key] = {'image': sample.image.relative_to(root).as_posix(),
                'image_sha256': digest(sample.image), 'mask': path.relative_to(root).as_posix(),
                'mask_sha256': digest(path)}
        return dict(asset_cache[key])

    episodes, query_details, query_hashes = [], [], set()
    for draw_idx, (cid, query) in enumerate(draws):
        query_asset = asset(query, False)
        qhash = query_asset['image_sha256']
        query_hashes.add((cid, qhash))
        candidates = list(index.samples[cid])
        rrng.shuffle(candidates)
        chosen, hashes = [], {qhash}
        # Hash all candidates when n-shot combinations are needed. In one-shot
        # runs stop once sufficient distinct candidates have been obtained.
        for candidate in candidates:
            if candidate.image == query.image:
                continue
            if query.group_id is not None and candidate.group_id == query.group_id:
                continue
            chash = digest(candidate.image)
            if chash in hashes:
                continue
            hashes.add(chash)
            chosen.append(candidate)
            if shots == 1 and len(chosen) >= references_per_query:
                break
        if len(chosen) < shots:
            raise ValueError(f'Class {cid}, query {query.image.name}: not enough distinct/group-independent references')
        maximum = math.comb(len(chosen), shots)
        actual = min(references_per_query, maximum)
        if actual != references_per_query and insufficient_references == 'error':
            raise ValueError(f'Class {cid}: requested {references_per_query} support sets; only {maximum} available')
        if shots == 1:
            support_sets = [(x,) for x in chosen[:actual]]
        elif maximum <= 100000:
            support_sets = rrng.sample(list(itertools.combinations(chosen, shots)), actual)
        else:
            seen, support_sets = set(), []
            while len(support_sets) < actual:
                ids = tuple(sorted(rrng.sample(range(len(chosen)), shots)))
                if ids not in seen:
                    seen.add(ids)
                    support_sets.append(tuple(chosen[i] for i in ids))
        qid = hashlib.sha256((cid + ':' + qhash).encode()).hexdigest()[:16]
        query_details.append({'draw_index': draw_idx, 'query_id': qid, 'concept_id': cid,
                              'image_sha256': qhash, 'group_id': query.group_id,
                              'requested_reference_sets': references_per_query,
                              'actual_reference_sets': actual})
        for ridx, references in enumerate(support_sets):
            episodes.append({'episode_id': f'{index.dataset}-f{index.fold}-q{draw_idx:06d}-{qid}-r{ridx:03d}',
                'dataset': index.dataset, 'dataset_version': index.version, 'split': index.split,
                'fold': index.fold, 'track': 'semantic', 'concept_id': str(cid),
                'reference_seed': seed + ridx, 'query': query_asset,
                'references': [asset(ref, True) for ref in references],
                'mask_semantics': 'binary foreground union of prompted class; reference void=background',
                'ignore_label': query.source_ignore})
    manifest = {'schema_version': 1, 'status': 'ready',
                'protocol_id': protocol_id or f'ce6190-unified-v1-{index.dataset}-{index.split}-f{index.fold}',
                'episodes': episodes}
    validate_manifest(manifest)  # hashes are produced from bytes above; full rehash is runner's responsibility.
    sidecar = {'schema_version': 1, 'status': 'prepared_not_model_evaluated',
        'manifest_sha256': manifest_fingerprint(manifest), 'dataset_root': str(root),
        'provenance': index.provenance, 'sampling': {'queries_requested': queries,
            'query_draws': len(draws), 'unique_query_concept_pairs': len(query_hashes),
            'reference_sets_requested_per_query': references_per_query,
            'shots': shots, 'seed': seed, 'query_seed': query_seed,
            'insufficient_references': insufficient_references,
            'query_repeats': len(draws) - len(query_hashes),
            'note': 'Repeated query draws are NOT independent units for uncertainty estimates'},
        'queries': query_details}
    return manifest, sidecar
