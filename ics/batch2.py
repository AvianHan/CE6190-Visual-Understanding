"""Second-batch orchestration using the existing quota-aware queue.

Only allocated workers prepare data or run models. One detached parent ticks
independent core/SUIM/ISIC queues sequentially and reserves invisible submissions
across them. An absent transfer source cannot block the core study.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from .orchestration import (build_plan, collect, fingerprint, load_json,
                            resource_profile, run_preflight, run_task, write_plan)
from . import rolling_queue as rq

SCOPES = ('core', 'suim', 'isic')
SCOPE_DATASETS = {'core': ('coco', 'lvis'), 'suim': ('suim',), 'isic': ('isic',)}
DEFAULT_PRIOR_RUNS = ('ics-initial-20261104-spacefix', 'ics-recovery-20261113')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_json(path, value):
    """Create once; a matching resume reuses bytes, a changed request is refused."""
    path = Path(path)
    payload = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError('Immutable batch2 file differs: ' + str(path))
        return
    with os.fdopen(fd, 'wb') as out:
        out.write(payload); out.flush(); os.fsync(out.fileno())


def selection(runtime, root, prior_runs=None):
    runtime, root = Path(runtime).resolve(), Path(root).resolve()
    priors = [Path(p).resolve() for p in (prior_runs or
              [runtime / 'runs' / name for name in DEFAULT_PRIOR_RUNS])]
    if len(priors) != len(set(priors)):
        raise ValueError('Duplicate first-batch prior run')
    if root in priors or any(root == p or root.is_relative_to(p) for p in priors):
        raise ValueError('Second batch must not be inside a first-batch run')
    return {'runtime': str(runtime), 'run_dir': str(root),
            'prior_runs': [str(p) for p in priors], 'scopes': list(SCOPES),
            'queries_per_fold': 100, 'transfer_queries': 1000,
            'references_per_query': 3, 'seed': 20261116,
            'max_inflight': 60, 'shared_limit': 60, 'min_free_gib': 32,
            'gate_policy': 'primary_tau_0.7_predeclared_sensitivities_0.5_0.9_no_test_selection',
            'download_policy': 'core_existing_only_transfer_missing_only_verified_catalog'}


def build_scope_plan(source, parent, request, scope):
    """Build immutable plans before data exist; data execute as CPU Slurm tasks."""
    if scope not in SCOPES:
        raise ValueError('Unknown batch2 scope')
    source, parent = Path(source).resolve(), Path(parent).resolve()
    os.environ['ICS_RUNTIME'] = request['runtime']
    os.environ['ICS_BATCH2_ROOT'] = str(parent)
    run = parent / 'scopes' / scope
    suite = source / 'configs' / ('suite.batch2.' + scope + '.json')
    cluster = source / 'configs/cluster.orion.batch2.json'
    plan = build_plan(suite, cluster, run)
    if Path(plan['project_root']).resolve() != source:
        raise ValueError('Build plans from the frozen batch2 source process')
    plan['batch2'] = {**copy.deepcopy(request), 'scope': scope,
                      'parent_run': str(parent), 'data_output': str(run / 'data')}
    # One real GPU smoke per configuration AND dataset. Passing COCO is not
    # accepted as evidence that the same model/config works on LVIS.
    smoke = []
    for original in plan['smoke']:
        cid = original['config_data']['experiment_id']
        mids = [t['manifest_id'] for t in plan['tasks']
                if t['config_data']['experiment_id'] == cid]
        chosen = {}
        for mid in mids:
            chosen.setdefault(plan['manifests'][mid]['dataset'], mid)
        for dataset, mid in chosen.items():
            item = copy.deepcopy(original)
            item.update(id='smoke-' + cid + '-' + dataset, manifest_id=mid,
                        manifest=plan['manifests'][mid]['path'], dataset=dataset,
                        output=str(run / 'smoke' / cid / dataset))
            smoke.append(item)
    plan['smoke'] = smoke
    for task in plan['tasks']:
        task['dataset'] = plan['manifests'][task['manifest_id']]['dataset']
    # Interleave configs/folds rather than filling the whole window with one
    # configuration; each resource cap remains the inherited 60-job limit.
    plan['tasks'].sort(key=lambda t: (t['shard_index'], t['manifest_id'],
                                      t['config_data']['experiment_id']))
    plan['fingerprint'] = fingerprint({k: v for k, v in plan.items() if k != 'fingerprint'})
    return plan


def prepare(source, parent, request):
    parent, source = Path(parent).resolve(), Path(source).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    metadata = parent / 'batch2.json'
    expected = {'schema_version': 1, 'selection': request, 'source': str(source),
                'source_snapshot_sha256': digest(source / 'snapshot.json'),
                'scope_plans': {}}
    for scope in SCOPES:
        plan = build_scope_plan(source, parent, request, scope)
        path = write_plan(plan)
        rq.initialize_queue(path, max_inflight=60, shared_limit=60,
                            reserved_slots=0, datasets=SCOPE_DATASETS[scope])
        expected['scope_plans'][scope] = {'path': str(path), 'fingerprint': plan['fingerprint']}
    expected['fingerprint'] = fingerprint(expected)
    immutable_json(metadata, expected)
    return status(parent)


def load_batch(parent):
    parent = Path(parent).resolve()
    data = load_json(parent / 'batch2.json')
    if fingerprint({k: v for k, v in data.items() if k != 'fingerprint'}) != data.get('fingerprint'):
        raise ValueError('Batch2 parent metadata checksum mismatch')
    if data['selection']['run_dir'] != str(parent) or Path(data['source']) != parent / 'source':
        raise ValueError('Batch2 parent/source identity mismatch')
    if digest(Path(data['source']) / 'snapshot.json') != data['source_snapshot_sha256']:
        raise ValueError('Frozen batch2 source identity changed')
    for scope, record in data['scope_plans'].items():
        plan = rq._plan(Path(record['path']))
        if plan['fingerprint'] != record['fingerprint'] or plan['batch2']['scope'] != scope:
            raise ValueError('Batch2 child plan identity mismatch')
    return data


def status(parent):
    parent = Path(parent).resolve()
    meta = load_batch(parent)
    scopes = {scope: rq.queue_status(Path(row['path']))
              for scope, row in meta['scope_plans'].items()}
    counts = Counter()
    for item in scopes.values(): counts.update(item['counts'])
    terminal = all(item['terminal'] for item in scopes.values())
    active = sum(item['active'] for item in scopes.values())
    state = (('COMPLETE_WITH_FAILURES' if counts['FAILED'] or counts['BLOCKED'] else 'COMPLETE')
             if terminal else ('RUNNING' if active else 'READY'))
    if not terminal and any(item['status'] == 'PAUSED' for item in scopes.values()):
        state = 'PARTIALLY_PAUSED'
    return {'batch2_status': state, 'terminal': terminal, 'counts': dict(counts),
            'active': active,
            'scope_status': scopes, 'selection': meta['selection'],
            'data_receipts': {s: str(parent / 'scopes' / s / 'data/data_receipt.json') for s in SCOPES},
            'status_note': 'Saved queue observations; this status command does not query Slurm. Transfer failures remain visible and do not block core.',
            'manager_metadata': load_json(parent / 'manager.json') if (parent / 'manager.json').exists() else None}


def tick_batch(parent, stop_requested=None):
    """Exactly one scheduler mutator at a time; account for every child intent."""
    meta = load_batch(parent)
    cursor_path = Path(parent) / 'scope_poll_cursor.json'
    cursor = load_json(cursor_path)['next_scope_index'] if cursor_path.exists() else 0
    if type(cursor) is not int or not 0 <= cursor < len(SCOPES):
        raise ValueError('Invalid scope poll cursor')
    order = SCOPES[cursor:] + SCOPES[:cursor]
    for scope in order:
        record = meta['scope_plans'][scope]
        if stop_requested and stop_requested(): break
        plan_path = Path(record['path'])
        if rq.queue_status(plan_path)['terminal']: continue
        external = []
        for other_scope, other in meta['scope_plans'].items():
            if other_scope == scope: continue
            other_plan = rq._plan(Path(other['path']))
            for task in rq._read_state(other_plan)['tasks'].values():
                if task['status'] in rq.ACTIVE:
                    external.append(task.get('job_id'))
        rq.tick(plan_path, stop_requested=stop_requested, external_active=external)
    rq._atomic_json(cursor_path, {'next_scope_index': (cursor + 1) % len(SCOPES)})
    return status(parent)


def make_worker_launch(plan_path, plan, task):
    """Seal task identity BEFORE sbatch. Worker never reads mutable queue state."""
    run = Path(plan['run_dir'])
    attempt = int(task['attempts']) + 1
    path = run / 'batch2_intents' / task['key'] / ('attempt-%05d.json' % attempt)
    source = Path(plan['project_root'])
    intent = {'schema_version': 1, 'plan_path': str(Path(plan_path).resolve()),
              'plan_fingerprint': plan['fingerprint'], 'worker_source': str(source),
              'source_snapshot_sha256': digest(source / 'snapshot.json'),
              'attempt': attempt, 'task': {k: copy.deepcopy(task[k]) for k in
                  ('key', 'kind', 'resource', 'index', 'dependencies', 'job_name')}}
    immutable_json(path, intent)
    task['batch2_intent'] = str(path)
    task['batch2_intent_sha256'] = digest(path)
    return [plan['cluster']['launcher_python'], '-I', '-B', '-u',
            str(source / 'scripts/batch2_ics.py'), 'worker',
            '--intent', str(path), '--intent-sha256', digest(path)]


def reconcile_intent_identity(plan, task, queue, history):
    """Use sealed worker claims/exact scheduler names to correct ack aliases.

    Never infer no submission from absence, never blindly resubmit uncertainty,
    and never accept a task's marker alone as Slurm success.
    """
    path = task.get('batch2_intent')
    if not path: return
    path = Path(path)
    if digest(path) != task.get('batch2_intent_sha256'):
        raise ValueError('Submission intent bytes changed')
    intent = load_json(path)
    if intent['plan_fingerprint'] != plan['fingerprint'] or intent['task']['job_name'] != task['job_name']:
        raise ValueError('Submission intent identity mismatch')
    records = {**history, **queue}
    names = {jid for jid, row in records.items() if row['job_name'] == task['job_name']}
    claim_path = path.with_suffix('.claim.json')
    claim = load_json(claim_path) if claim_path.exists() else None
    if claim:
        if (claim.get('intent_sha256') != task['batch2_intent_sha256'] or
                claim.get('job_name') != task['job_name']):
            raise ValueError('Worker claim identity mismatch')
        actual = claim.get('slurm_job_id')
        if not re.fullmatch(r'\d+', actual or ''):
            raise ValueError('Malformed worker claim job id')
        if names and names != {actual}:
            return 'Claim and scheduler identities conflict'
        names.add(actual)
    if len(names) > 1:
        return 'Multiple scheduler jobs match sealed task identity'
    if len(names) == 1:
        actual = next(iter(names))
        if task.get('job_id') != actual:
            task.setdefault('acknowledgement_corrections', []).append({
                'previous_job_id': task.get('job_id'), 'actual_job_id': actual,
                'evidence': 'sealed_worker_claim' if claim else 'unique_exact_scheduler_job_name'})
            task['job_id'] = actual


def _visible_read(path):
    # Retry metadata visibility only. A changed hash or malformed JSON is fatal.
    import errno
    for attempt in range(8):
        try: return Path(path).read_bytes()
        except OSError as exc:
            if exc.errno not in {errno.ENOENT, errno.ESTALE} or attempt == 7: raise
            time.sleep(.25)


def execute_intent(intent_path, expected_sha):
    raw = _visible_read(intent_path)
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Sealed batch2 intent SHA mismatch')
    intent = json.loads(raw)
    plan_path = Path(intent['plan_path']).resolve()
    run = plan_path.parent
    intent_path = Path(intent_path).resolve()
    if intent_path.parents[1] != run / 'batch2_intents':
        raise ValueError('Intent outside its batch2 scope')
    plan = rq._plan(plan_path)
    source = Path(intent['worker_source']).resolve()
    if (plan['fingerprint'] != intent['plan_fingerprint'] or
            plan['project_root'] != str(source) or source != Path(plan['batch2']['parent_run']) / 'source'):
        raise ValueError('Intent/plan/source mismatch')
    snapshot_bytes = _visible_read(source / 'snapshot.json')
    if hashlib.sha256(snapshot_bytes).hexdigest() != intent['source_snapshot_sha256']:
        raise ValueError('Frozen worker source changed')
    for name, sha in json.loads(snapshot_bytes)['files'].items():
        p = source / name
        if p.is_symlink() or hashlib.sha256(_visible_read(p)).hexdigest() != sha:
            raise ValueError('Frozen worker file changed: ' + name)
    task = intent['task']
    jid = os.environ.get('SLURM_JOB_ID', '')
    if not re.fullmatch(r'\d+', jid) or os.environ.get('SLURM_JOB_NAME') != task['job_name']:
        raise RuntimeError('A matching Slurm allocation is required; no login-node execution')
    if task['kind'] in {'smoke', 'experiment'}:
        group = [t for t in plan['smoke' if task['kind'] == 'smoke' else 'tasks']
                 if t['resource'] == task['resource']]
        original = group[task['index']]
        if original['id'] != task['key']: raise ValueError('Sealed task index mismatch')
    elif task['kind'] not in {'data', 'preflight', 'collect'}:
        raise ValueError('Unknown sealed task kind')
    # Exclusive execution claim prevents overlapping invocations from touching
    # any completion marker or model output. Claims are never silently removed.
    claim = intent_path.with_suffix('.claim.json')
    fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as out:
        json.dump({'slurm_job_id': jid, 'job_name': task['job_name'],
                   'intent_sha256': expected_sha, 'claimed_at': time.time()}, out)
        out.flush(); os.fsync(out.fileno())
    marker_path = rq._marker_path(plan, task['key'])
    if marker_path.exists(): raise RuntimeError('Task marker already exists; refusing duplicate execution')
    marker = {'schema_version': 1, 'key': task['key'], 'kind': task['kind'],
              'plan_fingerprint': plan['fingerprint'], 'slurm_job_id': jid,
              'job_name': task['job_name'], 'batch2_intent_sha256': expected_sha,
              'status': 'RUNNING', 'started_at': time.time()}
    rq._atomic_json(marker_path, marker)
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame): raise KeyboardInterrupt('Termination signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        settings = plan['batch2']
        if task['kind'] == 'data':
            from .batch2_data import prepare_scope
            prepare_scope(Path(settings['runtime']), Path(settings['data_output']),
                scope=settings['scope'], prior_runs=[Path(p) for p in settings['prior_runs']],
                queries_per_fold=settings['queries_per_fold'], transfer_queries=settings['transfer_queries'],
                references_per_query=settings['references_per_query'], seed=settings['seed'],
                min_free_gib=settings['min_free_gib'], trust_prepared_pickle=True,
                allow_download=settings['scope'] != 'core')
        elif task['kind'] == 'preflight':
            from .batch2_data import verify_scope
            verify_scope(Path(settings['data_output']), runtime=Path(settings['runtime']),
                         prior_runs=[Path(p) for p in settings['prior_runs']], expected_scope=settings['scope'])
            run_preflight(plan)
        elif task['kind'] == 'collect':
            result = collect(plan)
            if result.get('analysis_returncode', 0): raise RuntimeError('Existing collector analysis failed')
            python = resource_profile(plan['cluster'], 'preflight')['python']
            proc = subprocess.run([python, '-I', '-B', str(source / 'scripts/analyze_batch2.py'),
                                   '--run-dir', str(run), '--output', str(run / 'batch2_analysis')],
                                  capture_output=True, text=True, cwd=source)
            (run / 'batch2_analysis.log').write_text(proc.stdout + proc.stderr)
            if proc.returncode: raise RuntimeError('Batch2 scientific analysis failed; see batch2_analysis.log')
        else:
            run_task(plan, original)
        marker.update(status='PASS', ended_at=time.time())
        rq._atomic_json(marker_path, marker)
    except BaseException as exc:
        marker.update(status='FAIL', ended_at=time.time(), error_type=type(exc).__name__, error=str(exc))
        rq._atomic_json(marker_path, marker)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
