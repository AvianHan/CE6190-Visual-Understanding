"""Selective, immutable recovery of a terminal ICS queue (no model imports)."""
from __future__ import annotations

from collections import Counter
import copy
import fcntl
import hashlib
import json
from pathlib import Path

from .orchestration import fingerprint, load_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(root: Path) -> None:
    metadata = load_json(root / 'snapshot.json')
    for name, expected in metadata['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or sha256(path) != expected:
            raise ValueError(f'Frozen source changed: {name}')


def parent_evidence(parent: Path, *, expected_fingerprint: str) -> tuple[dict, dict, dict]:
    """Read old evidence, never reset or write its source/state/markers."""
    from .rolling_queue import _plan, TERMINAL
    plan = _plan(parent / 'plan.json')
    if plan['fingerprint'] != expected_fingerprint:
        raise ValueError('Parent plan fingerprint differs from the audited run')
    verify_source(parent / 'source')
    state = load_json(parent / 'rolling_state.json')
    if state['plan_fingerprint'] != plan['fingerprint']:
        raise ValueError('Parent state/plan mismatch')
    expected_keys = {'data', 'preflight', 'collect'} | {t['id'] for t in plan['smoke'] + plan['tasks']}
    if set(state['tasks']) != expected_keys or any(t['status'] not in TERMINAL for t in state['tasks'].values()):
        raise ValueError('Parent queue is not completely terminal')
    lifecycle = load_json(parent / 'manager.json')
    if lifecycle.get('exit', {}).get('reason') != 'terminal':
        raise ValueError('Parent manager did not record terminal completion')
    paths = [parent / name for name in ('plan.json', 'rolling_state.json', 'manager.json',
                                       'preflight.complete.json', 'source/snapshot.json')]
    for key, task in state['tasks'].items():
        if task['status'] == 'BLOCKED':
            if task.get('job_id') or task.get('attempts'):
                raise ValueError(f'Blocked task has submission evidence: {key}')
            continue
        marker_path = parent / 'rolling_markers' / (key + '.json')
        marker = load_json(marker_path)
        expected = {'key': key, 'status': 'PASS' if task['status'] == 'PASS' else 'FAIL',
                    'plan_fingerprint': plan['fingerprint'], 'slurm_job_id': task.get('job_id')}
        if not task.get('job_id') or any(marker.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Parent task marker mismatch: {key}')
        paths.append(marker_path)
        if task['status'] == 'PASS' and task['kind'] in {'smoke', 'experiment'}:
            complete = parent / 'jobs' / key / 'complete.json'
            value = load_json(complete)
            if any(value.get(k) != v for k, v in {
                'status': 'PASS', 'plan_fingerprint': plan['fingerprint'],
                'task_id': key, 'slurm_job_id': task['job_id']}.items()):
                raise ValueError(f'Parent model completion mismatch: {key}')
            paths.append(complete)
    if any(state['tasks'][key]['status'] != 'PASS' for key in ('data', 'preflight')):
        raise ValueError('Recovery requires successful original data and preflight gates')
    preflight = load_json(parent / 'preflight.complete.json')
    if preflight.get('status') != 'PASS' or preflight.get('plan_fingerprint') != plan['fingerprint']:
        raise ValueError('Parent preflight completion mismatch')
    evidence = {str(p.relative_to(parent)): sha256(p) for p in paths}
    return plan, state, evidence


def assert_parent_inactive(parent: Path, plan: dict, state: dict) -> None:
    """Fail closed if a live query cannot prove old jobs are absent/terminal."""
    from .rolling_queue import _scheduler_snapshot, SLURM_TERMINAL
    lock_path = parent / 'manager.lock'
    if lock_path.exists():
        with lock_path.open('r') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Parent manager is still active') from exc
    queue, history = _scheduler_snapshot(state)
    names = {task['job_name'] for task in state['tasks'].values()}
    jobs = {task['job_id'] for task in state['tasks'].values() if task.get('job_id')}
    if any(jid in jobs or row['job_name'] in names for jid, row in queue.items()):
        raise RuntimeError('Parent jobs still appear in the live scheduler queue')
    for task in state['tasks'].values():
        if not task.get('job_id'):
            continue
        row = history.get(task['job_id'])
        if row is None or row['job_name'] != task['job_name'] or row['state'] not in SLURM_TERMINAL:
            raise RuntimeError(f'Cannot verify terminal Slurm allocation: {task["key"]}')
        if task['status'] == 'PASS' and (row['state'] != 'COMPLETED' or row['exit_code'] != '0:0'):
            raise RuntimeError(f'Slurm contradicts parent PASS: {task["key"]}')


def manifest_evidence(parent: Path, plan: dict, state: dict) -> tuple[dict, str, str]:
    """Old preflight emits exactly one JSON validation per ordered manifest."""
    job_id = state['tasks']['preflight']['job_id']
    log = parent / 'logs' / f'preflight-{job_id}.log'
    content = log.read_text()
    decoder, rows = json.JSONDecoder(), []
    while content.strip():
        content = content.lstrip()
        value, end = decoder.raw_decode(content)
        rows.append(value)
        content = content[end:]
    validations, index = {}, 0
    for mid, manifest in plan['manifests'].items():
        prepared = None
        if index < len(rows) and 'manifest' in rows[index]:
            prepared = rows[index]
            if prepared['manifest'] != manifest['path']:
                raise ValueError('Preflight preparation order/path mismatch')
            index += 1
        if index >= len(rows):
            raise ValueError('Original preflight log omits a manifest validation')
        row = rows[index]
        index += 1
        if (row.get('valid') is not True or row.get('status') != 'ready' or
                not isinstance(row.get('episodes'), int) or row['episodes'] <= 0 or
                not isinstance(row.get('manifest_sha256'), str) or len(row['manifest_sha256']) != 64):
            raise ValueError('Original preflight log cannot bind every manifest unambiguously')
        if prepared and any(prepared.get(k) != row[k] for k in ('manifest_sha256', 'episodes')):
            raise ValueError('Preflight creation/validation fingerprints disagree')
        validations[mid] = {'sha256': row['manifest_sha256'], 'episodes': row['episodes']}
    if index != len(rows):
        raise ValueError('Original preflight log has unexpected trailing records')
    return validations, str(log.relative_to(parent)), sha256(log)


def build_recovery_plan(candidate: dict, old: dict, state: dict, parent: Path,
                        evidence: dict, manifests: dict, *, expected_missing: int,
                        expected_preserved: int) -> dict:
    """Select only old non-PASS tasks; scientific task definitions must match."""
    old_tasks = {task['id']: task for task in old['tasks']}
    new_tasks = {task['id']: task for task in candidate['tasks']}
    if set(old_tasks) != set(new_tasks) or old['manifests'] != candidate['manifests']:
        raise ValueError('Recovery cannot change the scientific task/manifest matrix')
    for key, old_task in old_tasks.items():
        strip = lambda task: {k: v for k, v in task.items() if k not in {'config', 'output'}}
        if strip(old_task) != strip(new_tasks[key]):
            raise ValueError(f'Scientific task changed: {key}')
    selected = [t for t in candidate['tasks'] if state['tasks'][t['id']]['status'] in {'FAILED', 'BLOCKED'}]
    preserved = [key for key in old_tasks if state['tasks'][key]['status'] == 'PASS']
    if len(selected) != expected_missing or len(preserved) != expected_preserved:
        raise ValueError(f'Audited counts changed: missing={len(selected)}, preserved={len(preserved)}')
    if not selected:
        raise ValueError('No missing experiments to recover')
    plan = copy.deepcopy(candidate)
    selected.sort(key=lambda task: 0 if task['config_data']['experiment_id'] in {'insid3_full', 'cgics_full'} and candidate['manifests'][task['manifest_id']].get('cohort') == 'standard' else 1)
    plan['tasks'] = selected
    configs = {t['config_data']['experiment_id'] for t in selected}
    # Both datasets must pass on repaired code; keep these outputs distinct from formal data.
    smoke = []
    for base in candidate['smoke']:
        cid = base['config_data']['experiment_id']
        if cid not in configs:
            continue
        for dataset in ('coco', 'lvis'):
            eligible = [t for t in selected if t['config_data']['experiment_id'] == cid
                        and plan['manifests'][t['manifest_id']]['prepare']['dataset'] == dataset]
            # Partial configurations may have completed all tasks of one dataset.
            # Validate that dataset using an original task, without repeating formal results.
            if not eligible:
                eligible = [t for t in candidate['tasks'] if t['config_data']['experiment_id'] == cid
                            and plan['manifests'][t['manifest_id']]['prepare']['dataset'] == dataset]
            if not eligible:
                continue
            source = eligible[0]
            task = copy.deepcopy(base)
            task.update(id=f'smoke-{cid}-{dataset}', manifest_id=source['manifest_id'],
                        manifest=source['manifest'], dataset_root=source['dataset_root'],
                        output=str(Path(plan['run_dir']) / 'smoke' / cid / dataset))
            smoke.append(task)
    smoke.sort(key=lambda task: 0 if task['config_data']['experiment_id'] in {'insid3_full', 'cgics_full'} else 1)
    plan['smoke'] = smoke
    # Old recipes are retained for protocol checks, but creation is forbidden.
    plan['prepare_missing_manifests'] = False
    plan['recovery'] = {'schema_version': 1, 'parent_run': str(parent.resolve()),
                        'parent_plan_fingerprint': old['fingerprint'], 'parent_evidence': evidence,
                        'manifest_evidence': manifests, 'selected_task_ids': [t['id'] for t in selected],
                        'preserved_task_ids': preserved,
                        'original_status_counts': dict(Counter(state['tasks'][t['id']]['status'] for t in selected)),
                        'data_policy': 'verify_existing_only',
                        'note': 'Parent results remain in parent_run; this plan collects only recovered results.'}
    plan['fingerprint'] = fingerprint({k: v for k, v in plan.items() if k != 'fingerprint'})
    return plan


def verify_recovery_parent(plan: dict) -> None:
    metadata = plan['recovery']
    root = Path(metadata['parent_run'])
    for name, expected in metadata['parent_evidence'].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or sha256(path) != expected:
            raise ValueError(f'Parent evidence changed since recovery preparation: {name}')
    verify_source(root / 'source')


def verify_recovery_assets(plan: dict) -> None:
    """Read-only data gate: hashes original manifests; no downloads/generation."""
    from ics_eval import manifest_fingerprint
    verify_recovery_parent(plan)
    expected = plan['recovery']['manifest_evidence']
    if set(expected) != set(plan['manifests']):
        raise ValueError('Recovery manifest evidence is incomplete')
    for mid, item in plan['manifests'].items():
        manifest_path = Path(item['path'])
        manifest = load_json(manifest_path)
        if manifest_fingerprint(manifest) != expected[mid]['sha256'] or len(manifest['episodes']) != expected[mid]['episodes']:
            raise ValueError(f'Original manifest changed: {mid}')
        if not manifest_path.with_suffix('.protocol.json').is_file() or not Path(item['dataset_root']).is_dir():
            raise FileNotFoundError(f'Existing data/sidecar unavailable: {mid}')
    # Full image/mask hashes and protocol constraints are checked by the subsequent CPU preflight.


def assert_no_partial_success(parent: Path, old: dict, selected: list[str]) -> None:
    """A partially successful failed shard needs an explicit episode-level resume design."""
    tasks = {t['id']: t for t in old['tasks']}
    for key in selected:
        task = tasks[key]
        relative = Path(task['output']).relative_to(Path(old['run_dir']))
        for path in (parent / relative / 'episodes').glob('*.json'):
            row = load_json(path)
            if row.get('status') == 'ok' and row.get('shard_index') == task['shard_index']:
                raise ValueError(f'Selected task has partial successful episodes; preserve/review before recovery: {key}')
