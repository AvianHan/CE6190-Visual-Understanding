"""Immutable second-batch cohorts built with the existing dataset semantics.

No model is imported or run here. Old runs are read only. A fresh query must
have both an unseen full-file SHA256 and, for COCO/LVIS, an unseen canonical
COCO image ID. Reference trials are repetitions of one query, not independent
query samples. Cross-domain preparation is separately callable per dataset.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile

import numpy as np
from PIL import Image

from ics.data import (DatasetIndex, _contained, _eligible_samples, build_index,
                      file_sha256, read_binary_mask)
from ics_eval import manifest_fingerprint, validate_manifest


VERSION = 'ce6190-batch2-fresh-v1'
SUIM_REPAIR_IMPLEMENTATION_ID = 'suim-strict-geometry-filter-v1'
DEFAULT_PRIOR_RUNS = ('ics-initial-20261104-spacefix', 'ics-recovery-20261113')
CORE_FOLDS = (('coco', 4), ('lvis', 10))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def coco_image_identity(image, dataset):
    """Normalize only the two audited COCO source filename conventions."""
    if dataset not in ('coco', 'lvis'):
        return None
    name = Path(image).name
    match = re.fullmatch(r'COCO_(?:train|val)2014_(\d{12})\.jpg', name)
    if match is None:
        match = re.fullmatch(r'(\d{12})\.jpg', name)
    if match is None:
        raise ValueError(f'Cannot establish canonical COCO image identity: {image}')
    return 'coco-image:' + str(int(match.group(1)))


def _identities(asset, dataset):
    sha = asset.get('image_sha256', '')
    if not re.fullmatch(r'[0-9a-fA-F]{64}', sha):
        raise ValueError('A query requires a full image SHA256')
    result = {'sha256:' + sha.lower()}
    source = coco_image_identity(asset['image'], dataset)
    if source is not None:
        result.add(source)
    return result


def collect_prior_queries(prior_runs, runtime):
    """Bind every formal prior manifest to actual original output run.json.

    Parent and recovery together cover cohorts whose parent jobs never reached
    inference. A missing binding cannot be silently replaced by the plan path.
    Duplicate references, configuration repetitions and shards do not multiply
    the exclusion set. No prior image/mask data or result is modified.
    """
    runtime = Path(runtime).resolve()
    runs = [Path(p).resolve() for p in prior_runs]
    if len(runs) < 2 or len(set(runs)) != len(runs):
        raise ValueError('Provide distinct parent and recovery prior runs')
    plans, tasks_by_manifest = [], defaultdict(list)
    for run in runs:
        if not run.is_relative_to(runtime / 'runs'):
            raise ValueError('Prior run must be contained in runtime/runs')
        plan_path = run / 'plan.json'
        plan = _read(plan_path)
        if Path(plan['run_dir']).resolve() != run:
            raise ValueError(f'Prior plan/run identity mismatch: {run}')
        tasks = plan.get('tasks', [])
        if not tasks:
            raise ValueError(f'Prior plan has no formal tasks: {run}')
        plans.append({'path': str(plan_path), 'file_sha256': file_sha256(plan_path),
                      'plan_fingerprint': plan.get('fingerprint'),
                      'formal_tasks': len(tasks)})
        for task in tasks:
            if task.get('phase', 'experiment') != 'experiment':
                raise ValueError('Unexpected non-formal task in prior tasks table')
            manifest = Path(task['manifest']).resolve()
            output = Path(task['output']).resolve()
            if not manifest.is_relative_to(runtime) or not output.is_relative_to(run):
                raise ValueError('Prior manifest/output escapes its recorded runtime/run')
            if Path(task['dataset_root']).resolve() != runtime / 'data':
                raise ValueError('Prior run uses a different dataset root')
            tasks_by_manifest[str(manifest)].append(task)
    all_sha, source_ids, manifest_records = set(), set(), []
    for filename, tasks in sorted(tasks_by_manifest.items()):
        path = Path(filename)
        manifest = _read(path)
        validate_manifest(manifest)
        fingerprint = manifest_fingerprint(manifest)
        ids = {ep['episode_id'] for ep in manifest['episodes']}
        bindings = []
        for output in sorted({t['output'] for t in tasks}):
            binding_path = Path(output) / 'run.json'
            if not binding_path.is_file():
                continue
            record = _read(binding_path)
            binding = record.get('binding', {})
            if (binding.get('manifest_sha256') != fingerprint or
                    binding.get('protocol_id') != manifest['protocol_id']):
                raise ValueError(f'Prior manifest disagrees with original binding: {binding_path}')
            selected = binding.get('selected_episode_ids')
            if (not isinstance(selected, list) or not selected or
                    len(set(selected)) != len(selected) or not set(selected) <= ids):
                raise ValueError(f'Invalid prior selected-ID binding: {binding_path}')
            bindings.append({'path': str(binding_path), 'file_sha256': file_sha256(binding_path),
                             'selected_episode_count': len(selected)})
        if not bindings:
            raise ValueError(f'No original output binding for prior formal manifest: {path}')
        datasets, hashes, sources = set(), set(), set()
        for episode in manifest['episodes']:
            identities = _identities(episode['query'], episode['dataset'])
            hashes |= {x[7:] for x in identities if x.startswith('sha256:')}
            sources |= {x for x in identities if x.startswith('coco-image:')}
            datasets.add(episode['dataset'])
        all_sha.update(hashes)
        source_ids.update(sources)
        manifest_records.append({'path': str(path), 'file_sha256': file_sha256(path),
            'manifest_sha256': fingerprint, 'protocol_id': manifest['protocol_id'],
            'datasets': sorted(datasets), 'episodes': len(manifest['episodes']),
            'unique_query_image_sha256_count': len(hashes),
            'unique_canonical_source_image_count': len(sources), 'bindings': bindings})
    result = {'schema_version': 1, 'plans': plans, 'manifests': manifest_records,
              'query_image_sha256': sorted(all_sha), 'canonical_source_image_ids': sorted(source_ids),
              'query_image_sha256_count': len(all_sha), 'canonical_source_image_count': len(source_ids),
              'scope': 'All queries in all formal parent and recovery plan manifests, not only successful tasks',
              'identity_boundary': 'Exact image bytes plus canonical COCO source IDs; not a near-duplicate or patient-ID audit'}
    result['exclusion_sha256'] = _digest(result)
    return result


def _verify_history(history):
    expected = history['exclusion_sha256']
    raw = {k: v for k, v in history.items() if k != 'exclusion_sha256'}
    if _digest(raw) != expected:
        raise ValueError('Prior exclusion receipt digest mismatch')
    for record in history['plans'] + history['manifests']:
        if file_sha256(Path(record['path'])) != record['file_sha256']:
            raise ValueError(f'Prior source changed: {record["path"]}')
    # At least one independently retained original binding per manifest is
    # sufficient on a resume; validate every one recorded for conflict safety.
    for manifest in history['manifests']:
        for binding in manifest['bindings']:
            if file_sha256(Path(binding['path'])) != binding['file_sha256']:
                raise ValueError(f'Prior original binding changed: {binding["path"]}')


def _publish_json(path, content):
    """Create-only, identical-content idempotence; a race never overwrites."""
    path = Path(path)
    data = (json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise ValueError(f'Existing immutable output differs; preserved: {path}')
        return
    fd, name = tempfile.mkstemp(prefix='.batch2-writing-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != data:
                raise ValueError(f'Concurrent immutable output differs; preserved: {path}')
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def _lock(output_dir):
    import fcntl
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / '.prepare.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another preparation owns this scope; no concurrent publication') from exc
        yield


def _asset_factory(root, dataset, digest_cache):
    asset_cache = {}

    def digest(path):
        path = _contained(root, Path(path))
        key = str(path)
        if key not in digest_cache:
            digest_cache[key] = file_sha256(path)
        return digest_cache[key]

    def asset(sample, reference):
        key = (str(sample.image), sample.concept_id, reference)
        if key not in asset_cache:
            mask = read_binary_mask(sample, reference=reference)
            content_id = hashlib.sha256(mask.tobytes() + str(mask.shape).encode()).hexdigest()
            path = _contained(root, root / 'ics_generated_masks' / dataset /
                              ('reference' if reference else 'query') / (content_id + '.png'))
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                fd, name = tempfile.mkstemp(suffix='.png', prefix='.batch2-mask-', dir=path.parent)
                os.close(fd)
                try:
                    Image.fromarray(mask).save(name, format='PNG')
                    try:
                        os.link(name, path)
                    except FileExistsError:
                        pass
                finally:
                    Path(name).unlink(missing_ok=True)
            with Image.open(path) as image:
                current = np.asarray(image)
            if current.dtype != np.uint8 or not np.array_equal(current, mask):
                raise ValueError(f'Generated mask cache mismatch; preserved unchanged: {path}')
            asset_cache[key] = {'image': sample.image.relative_to(root).as_posix(),
                'image_sha256': digest(sample.image), 'mask': path.relative_to(root).as_posix(),
                'mask_sha256': digest(path)}
        return dict(asset_cache[key])
    return asset, digest


def create_fresh_manifest(index, data_root, *, queries, references_per_query=3,
                          seed=20261116, excluded_sha256=(), excluded_source_ids=(),
                          used_query_identities=None, digest_cache=None, cohort='fresh',
                          prior_exclusion_sha256=None):
    """Sample unique new query images; preserve the author's class universe.

    Class-balanced round-robin sampling skips exhausted fresh classes and
    records that coverage loss explicitly. Insufficient total fresh queries is
    fatal, never quietly replaced by repeated or old images. References may
    reuse old images (normal one-shot evaluation), but never the query itself,
    a re-encoding of it, or a duplicate within its reference trials.
    """
    if type(queries) is not int or queries < 1 or type(references_per_query) is not int or references_per_query < 1:
        raise ValueError('Positive integer query and reference counts required')
    if type(seed) is not int or seed < 0:
        raise ValueError('Nonnegative integer seed required')
    excluded_sha256, excluded_source_ids = tuple(excluded_sha256), tuple(excluded_source_ids)
    root = Path(data_root).resolve()
    used = used_query_identities if used_query_identities is not None else set()
    excluded = {'sha256:' + x.lower() for x in excluded_sha256} | set(excluded_source_ids)
    digest_cache = {} if digest_cache is None else digest_cache
    policy = 'exclude' if index.dataset == 'lvis' else 'error'
    samples, eligibility = _eligible_samples(index, root, policy)
    asset, digest = _asset_factory(root, index.dataset, digest_cache)
    classes = list(samples)
    qrng = random.Random(f'{VERSION}:{seed}:{index.dataset}:{index.fold}:query')
    rrng = random.Random(f'{VERSION}:{seed}:{index.dataset}:{index.fold}:reference')
    queues = {cid: list(pool) for cid, pool in samples.items()}
    for pool in queues.values():
        qrng.shuffle(pool)
    order = list(classes)
    qrng.shuffle(order)
    original_used = set(used)
    episodes, query_details = [], []
    rejected = Counter()
    class_draws = Counter()
    attempts_by_class = Counter()

    def identities(sample):
        return _identities({'image': sample.image.relative_to(root).as_posix(),
                            'image_sha256': digest(sample.image)}, index.dataset)

    while len(query_details) < queries:
        progress = False
        for cid in order:
            while queues[cid]:
                query = queues[cid].pop()
                attempts_by_class[cid] += 1
                qids = identities(query)
                if qids & excluded:
                    rejected['prior_query_identity'] += 1
                    continue
                if qids & used:
                    rejected['already_selected_query_identity'] += 1
                    continue
                candidates = list(samples[cid])
                rrng.shuffle(candidates)
                refs, ref_ids = [], set(qids)
                for candidate in candidates:
                    if query.group_id is not None and query.group_id == candidate.group_id:
                        continue
                    ids = identities(candidate)
                    if ids & ref_ids:
                        continue
                    # Both source identity and bytes must differ. Masks remain
                    # the existing original-resolution class/union semantics.
                    ref_ids.update(ids)
                    refs.append(candidate)
                    if len(refs) == references_per_query:
                        break
                if not refs:
                    rejected['no_distinct_reference'] += 1
                    continue
                qasset = asset(query, False)
                ref_assets = [asset(ref, True) for ref in refs]
                draw = len(query_details)
                qid = hashlib.sha256((str(cid) + ':' + qasset['image_sha256']).encode()).hexdigest()[:16]
                query_details.append({'draw_index': draw, 'query_id': qid,
                    'concept_id': str(cid), 'image_sha256': qasset['image_sha256'],
                    'canonical_source_image_id': coco_image_identity(qasset['image'], index.dataset),
                    'group_id': query.group_id, 'requested_reference_sets': references_per_query,
                    'actual_reference_sets': len(ref_assets)})
                for ridx, ref_asset in enumerate(ref_assets):
                    episodes.append({'episode_id': f'b2-{index.dataset}-f{index.fold}-q{draw:06d}-{qid}-r{ridx:03d}',
                        'dataset': index.dataset, 'dataset_version': index.version, 'split': index.split,
                        'fold': index.fold, 'track': 'semantic', 'concept_id': str(cid),
                        'reference_seed': seed + ridx, 'query': dict(qasset), 'references': [ref_asset],
                        'mask_semantics': 'binary foreground union of prompted class; reference void=background',
                        'ignore_label': query.source_ignore})
                used.update(qids)
                class_draws[cid] += 1
                progress = True
                break
            if len(query_details) == queries:
                break
        if not progress:
            used.clear()
            used.update(original_used)
            raise ValueError(f'{index.dataset} fold {index.fold}: requested {queries} unique fresh queries, '
                             f'only {len(query_details)} eligible; no repetitions or old queries substituted. '
                             f'Class counts: {dict(class_draws)}; rejections: {dict(rejected)}')
    manifest = {'schema_version': 1, 'status': 'ready',
                'protocol_id': f'{VERSION}-{index.dataset}-f{index.fold}-{cohort}', 'episodes': episodes}
    validate_manifest(manifest)
    query_hashes = {q['image_sha256'] for q in query_details}
    assert len(query_hashes) == queries
    sidecar = {'schema_version': 1, 'status': 'prepared_not_model_evaluated',
        'manifest_sha256': manifest_fingerprint(manifest), 'dataset_root': str(root),
        'protocol_id': manifest['protocol_id'], 'provenance': dict(index.provenance),
        'sample_eligibility': eligibility,
        'sampling': {'queries_requested': queries, 'query_draws': queries,
            'unique_query_concept_pairs': queries, 'unique_query_images': queries,
            'query_repeats': 0, 'reference_sets_requested_per_query': references_per_query,
            'reference_shortfall_queries': sum(q['actual_reference_sets'] < references_per_query for q in query_details),
            'shots': 1, 'seed': seed, 'query_seed': seed, 'insufficient_references': 'all_explicitly_reported',
            'empty_mask_policy': policy,
            'note': 'Reference trials share their query image; inference never receives query GT or class.'},
        'batch2': {'version': VERSION, 'cohort': cohort,
            'studies': ['score_ablation', 'fresh_validation', 'agreement_gate'] if cohort == 'fresh' else ['transfer'],
            'prior_exclusion_sha256': prior_exclusion_sha256,
            'old_query_image_sha256_count': len(set(excluded_sha256)),
            'old_canonical_source_image_count': len(set(excluded_source_ids)),
            'class_universe': list(map(str, classes)), 'actual_query_counts_by_class': dict(class_draws),
            'unrepresented_fresh_concepts': [str(c) for c in classes if class_draws[c] == 0],
            'sampling_attempts_by_class': dict(attempts_by_class), 'rejections': dict(rejected),
            'identity_boundary': 'Full image SHA256 plus canonical COCO IDs; no near-duplicate or patient-disjoint claim',
            'model_setting_selection': 'Predeclared primary and sensitivity settings; no test-GT parameter selection'},
        'queries': query_details}
    return manifest, sidecar


def _request(runtime, scope, prior_runs, queries_per_fold, transfer_queries,
             references_per_query, seed, trust_prepared_pickle):
    if scope not in ('core', 'suim', 'isic'):
        raise ValueError('scope must be core, suim or isic')
    if any(type(x) is not int or x < 1 for x in (queries_per_fold, transfer_queries, references_per_query)):
        raise ValueError('Requested counts must be positive integers')
    if type(seed) is not int or seed < 0:
        raise ValueError('Nonnegative integer seed required')
    layout = [(ds, f, queries_per_fold, 'fresh') for ds, folds in CORE_FOLDS for f in range(folds)] if scope == 'core' else [(scope, 0, transfer_queries, 'transfer')]
    request = {'version': VERSION, 'runtime': str(runtime), 'scope': scope,
        'prior_runs': [str(x) for x in prior_runs], 'queries_per_fold': queries_per_fold,
        'transfer_queries': transfer_queries, 'references_per_query': references_per_query,
        'seed': seed, 'trust_prepared_pickle': bool(trust_prepared_pickle),
        'cohort_layout': [{'dataset': ds, 'fold': fold, 'queries': n, 'cohort': cohort,
                          'manifest_id': f'batch2_{ds}_fold{fold}_{cohort}'}
                         for ds, fold, n, cohort in layout],
        'cohort_policy': 'Unique query bytes/source IDs; original class universe, explicit unavailable class and reference counts'}
    if scope == 'suim':
        request['geometry_policy'] = SUIM_REPAIR_IMPLEMENTATION_ID
    return request


def verify_scope(output_dir, *, runtime=None, prior_runs=None, expected_scope=None):
    """Verify receipt, originals and selected asset hashes without resampling."""
    output_dir = Path(output_dir).resolve()
    receipt = _read(output_dir / 'data_receipt.json')
    raw = {k: v for k, v in receipt.items() if k != 'receipt_sha256'}
    if receipt.get('status') != 'READY' or _digest(raw) != receipt.get('receipt_sha256'):
        raise ValueError('Invalid immutable batch2 data receipt')
    request = receipt['request']
    if _digest(request) != receipt['request_sha256']:
        raise ValueError('Batch2 preparation request digest mismatch')
    if runtime is not None and str(Path(runtime).resolve()) != request['runtime']:
        raise ValueError('Batch2 runtime does not match receipt')
    if prior_runs is not None and [str(Path(x).resolve()) for x in prior_runs] != request['prior_runs']:
        raise ValueError('Batch2 prior runs do not match receipt')
    if expected_scope is not None and expected_scope != request['scope']:
        raise ValueError('Batch2 scope does not match receipt')
    _verify_history(receipt['history'])
    root = Path(request['runtime']) / 'data'
    geometry = None
    if request['scope'] == 'suim':
        from .suim_geometry import verify_suim_geometry
        if request.get('geometry_policy') != SUIM_REPAIR_IMPLEMENTATION_ID:
            raise ValueError('SUIM manifest does not declare the shape-valid subset policy')
        record = receipt['geometry_eligibility']
        geometry_path = Path(record['path'])
        if (geometry_path.is_symlink() or geometry_path.resolve().parent != output_dir or
                file_sha256(geometry_path) != record['file_sha256']):
            raise ValueError('SUIM geometry receipt path/hash mismatch')
        geometry = verify_suim_geometry(Path(request['runtime']), geometry_path)
        if geometry['receipt_sha256'] != record['receipt_sha256']:
            raise ValueError('SUIM geometry canonical receipt mismatch')
    excluded = {'sha256:' + x for x in receipt['history']['query_image_sha256']} | set(receipt['history']['canonical_source_image_ids'])
    expected_layout = {x['manifest_id']: x for x in request['cohort_layout']}
    if set(receipt['manifests']) != set(expected_layout):
        raise ValueError('Incomplete batch2 manifest set for frozen cohort layout')
    seen = set()
    for mid, entry in receipt['manifests'].items():
        expected = expected_layout[mid]
        if any(entry[k] != expected[k] for k in ('dataset', 'fold', 'cohort')):
            raise ValueError('Manifest scope identity differs from frozen layout')
        path, protocol = Path(entry['path']), Path(entry['protocol_path'])
        if (path.is_symlink() or protocol.is_symlink() or
                not path.resolve().is_relative_to(output_dir) or not protocol.resolve().is_relative_to(output_dir)):
            raise ValueError('Receipt manifest path escapes scope output')
        if file_sha256(path) != entry['file_sha256'] or file_sha256(protocol) != entry['protocol_sha256']:
            raise ValueError('Immutable manifest/protocol bytes changed')
        manifest, side = _read(path), _read(protocol)
        validate_manifest(manifest, root=root)
        if manifest_fingerprint(manifest) != entry['manifest_sha256'] or side['manifest_sha256'] != entry['manifest_sha256']:
            raise ValueError('Manifest canonical fingerprint mismatch')
        if (side['protocol_id'] != manifest['protocol_id'] or
                side['batch2']['cohort'] != entry['cohort'] or
                Path(side['batch2']['exclusion_receipt_path']).resolve() != output_dir / 'data_receipt.json' or
                side['batch2']['prior_exclusion_sha256'] != receipt['history']['exclusion_sha256']):
            raise ValueError('Manifest protocol/exclusion binding mismatch')
        if geometry is not None:
            from .suim_geometry import verify_manifest_pairs
            declared = side['provenance']['suim_geometry']
            if (declared['implementation_id'] != SUIM_REPAIR_IMPLEMENTATION_ID or
                    declared['receipt_sha256'] != geometry['receipt_sha256'] or
                    Path(declared['receipt_path']).resolve() != geometry_path.resolve() or
                    not manifest['protocol_id'].endswith('-shape-valid-v1')):
                raise ValueError('SUIM manifest geometry policy/provenance mismatch')
            verify_manifest_pairs(manifest, root, geometry)
        for source in side['provenance'].get('source_files', []):
            source_path = _contained(root, Path(source['path']))
            if file_sha256(source_path) != source['sha256']:
                raise ValueError(f'Dataset membership source changed: {source_path}')
        local, reference_ids = {}, defaultdict(set)
        for episode in manifest['episodes']:
            if (episode['dataset'], episode['fold']) != (entry['dataset'], entry['fold']):
                raise ValueError('Episode dataset/fold differs from cohort receipt')
            qids = _identities(episode['query'], episode['dataset'])
            if qids & excluded:
                raise ValueError('Prior query leaked into second-batch cohort')
            # Multiple references for one query are intentional; another target
            # on the same image is not an independent new query in this cohort.
            key = episode['query']['image_sha256']
            identity = (episode['concept_id'], tuple(sorted(qids)))
            if key in local and local[key] != identity:
                raise ValueError('Multiple targets claimed as distinct unique query images')
            local[key] = identity
            for ref in episode['references']:
                rids = _identities(ref, episode['dataset'])
                if rids & qids:
                    raise ValueError('Reference/query source identity overlap')
                if rids & reference_ids[key]:
                    raise ValueError('Repeated reference identity within query trials')
                reference_ids[key].update(rids)
        current = set().union(*(set(value[1]) for value in local.values()))
        if current & seen:
            raise ValueError('Query identity overlap across second-batch manifests')
        if (len(local) != entry['unique_query_images'] or len(local) != expected['queries'] or
                len(local) != entry['requested_queries'] or len(manifest['episodes']) != entry['episodes']):
            raise ValueError('Batch2 count receipt mismatch')
        seen.update(current)
    if (receipt['query_images'] != sum(x['unique_query_images'] for x in receipt['manifests'].values()) or
            receipt['episodes'] != sum(x['episodes'] for x in receipt['manifests'].values())):
        raise ValueError('Scope total count mismatch')
    return receipt


def prepare_scope(runtime, output_dir, *, scope='core', prior_runs=None,
                  queries_per_fold=100, transfer_queries=1000,
                  references_per_query=3, seed=20261116, min_free_gib=32,
                  trust_prepared_pickle=True, allow_download=False):
    """Prepare one independently gated scope; return READY only on full audit."""
    runtime, output_dir = Path(runtime).resolve(), Path(output_dir).resolve()
    root = runtime / 'data'
    if not root.is_dir():
        raise FileNotFoundError(f'Existing runtime data root required: {root}')
    prior_runs = [Path(x).resolve() for x in (prior_runs or [runtime / 'runs' / n for n in DEFAULT_PRIOR_RUNS])]
    request = _request(runtime, scope, prior_runs, queries_per_fold, transfer_queries,
                       references_per_query, seed, trust_prepared_pickle)
    if type(min_free_gib) not in (int, float) or not math.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError('min_free_gib must be finite and nonnegative')
    with _lock(output_dir):
        receipt_path = output_dir / 'data_receipt.json'
        if receipt_path.exists():
            existing = _read(receipt_path)
            if existing.get('request') != request:
                raise ValueError('Existing second-batch data scope uses a different request; preserved')
            return verify_scope(output_dir, runtime=runtime, prior_runs=prior_runs, expected_scope=scope)
        if shutil.disk_usage(root).free < min_free_gib * 2**30:
            raise RuntimeError(f'Free disk below {min_free_gib} GiB preparation guard')
        history = collect_prior_queries(prior_runs, runtime)
        source_preparation = None
        if scope != 'core':
            from .batch2_sources import ensure_transfer_sources
            source_preparation = ensure_transfer_sources(runtime, scope, allow_download=allow_download,
                                                         min_free_gib=min_free_gib)
        geometry = None
        if scope == 'suim':
            from .suim_geometry import prepare_suim_geometry
            geometry = prepare_suim_geometry(runtime, output_dir)
        requests = [(x['dataset'], x['fold'], x['queries'], x['cohort']) for x in request['cohort_layout']]
        manifests, used, digest_cache = {}, set(), {}
        for dataset, fold, queries, cohort in requests:
            index = build_index(dataset, root, fold=fold, split='val',
                                trust_prepared_pickle=trust_prepared_pickle)
            if dataset == 'suim':
                from .suim_geometry import apply_suim_geometry
                index = apply_suim_geometry(index, root, geometry)
            manifest, sidecar = create_fresh_manifest(index, root, queries=queries,
                references_per_query=references_per_query, seed=seed,
                excluded_sha256=history['query_image_sha256'],
                excluded_source_ids=history['canonical_source_image_ids'],
                used_query_identities=used, digest_cache=digest_cache, cohort=cohort,
                prior_exclusion_sha256=history['exclusion_sha256'])
            if dataset == 'suim':
                manifest['protocol_id'] += '-shape-valid-v1'
                sidecar['protocol_id'] = manifest['protocol_id']
                sidecar['manifest_sha256'] = manifest_fingerprint(manifest)
                sidecar['provenance']['suim_geometry']['receipt_path'] = str(output_dir / 'suim_geometry_receipt.json')
            mid = f'batch2_{dataset}_fold{fold}_{cohort}'
            sidecar['batch2']['exclusion_receipt_path'] = str(receipt_path)
            path = output_dir / 'manifests' / (mid + '.json')
            protocol = path.with_suffix('.protocol.json')
            _publish_json(protocol, sidecar)
            _publish_json(path, manifest)
            manifests[mid] = {'path': str(path), 'file_sha256': file_sha256(path),
                'manifest_sha256': manifest_fingerprint(manifest),
                'protocol_path': str(protocol), 'protocol_sha256': file_sha256(protocol),
                'dataset': dataset, 'fold': fold, 'cohort': cohort,
                'episodes': len(manifest['episodes']), 'unique_query_images': sidecar['sampling']['unique_query_images'],
                'requested_queries': queries, 'reference_shortfall_queries': sidecar['sampling']['reference_shortfall_queries'],
                'represented_classes': len(sidecar['batch2']['actual_query_counts_by_class']),
                'class_universe_count': len(sidecar['batch2']['class_universe'])}
            print(f'Prepared {mid}: {queries} unique query images, {len(manifest["episodes"])} episodes', flush=True)
        receipt = {'schema_version': 1, 'status': 'READY', 'request': request,
            'request_sha256': _digest(request), 'history': history,
            'source_preparation': source_preparation, 'manifests': manifests,
            'query_images': sum(x['unique_query_images'] for x in manifests.values()),
            'episodes': sum(x['episodes'] for x in manifests.values()),
            'model_work_performed': False, 'existing_results_modified': False}
        if geometry is not None:
            geometry_path = output_dir / 'suim_geometry_receipt.json'
            receipt['geometry_eligibility'] = {'implementation_id': SUIM_REPAIR_IMPLEMENTATION_ID,
                'path': str(geometry_path), 'file_sha256': file_sha256(geometry_path),
                'receipt_sha256': geometry['receipt_sha256']}
        receipt['receipt_sha256'] = _digest(receipt)
        _publish_json(receipt_path, receipt)
        return verify_scope(output_dir, runtime=runtime, prior_runs=prior_runs, expected_scope=scope)
