"""Audited in-place routing of accepted batch2 jobs; never submit or retry work.

This companion reads the original queue and frozen intents. It owns only its
policy, lifecycle state and update journals. Canary jobs are existing formal
tasks, so they keep their job IDs, model commands, outputs and scientific plan.
"""
from __future__ import annotations

from collections import Counter
import copy
import datetime as dt
import getpass
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from . import batch2, rolling_queue as rq
from .orchestration import fingerprint, load_json
from .scheduling import _job_fields

TAG = 'acceleration-20261117'
ROUTES = {
    'l40s_reboot': {'partition': 'l40sp', 'node': 'A484652X3B10601', 'gpu': 'l40s'},
    'testqueue_bfs': {'partition': 'testqueue', 'node': 'BFS9B74', 'gpu': 'h200'},
}
FAMILIES = ('insid3', 'cgics_mllm')
KNOWN_CONFIGS = frozenset({
    'batch2_insid3_full', 'batch2_cgics_full', 'batch2_cgics_single',
    'batch2_cgics_bon1', 'batch2_cgics_score_pool', 'batch2_cgics_gate_t05',
    'batch2_cgics_gate_t07', 'batch2_cgics_gate_t09', 'batch2_cgics_random_p25',
    'batch2_cgics_random_p50', 'batch2_cgics_random_p75', 'batch2_cgics_score_gate',
    'batch2_cgics_staged_search',
})
REQUEST_FIELDS = ('Partition', 'ReqNodeList', 'ExcNodeList', 'TimeLimit')
IDENTITY_FIELDS = ('JobId', 'JobName', 'UserId', 'QOS', 'Account', 'Command')
RESOURCE_FIELDS = ('NumCPUs', 'NumNodes', 'NumTasks', 'CPUs/Task',
                   'MinMemoryNode', 'TresPerNode', 'ReqTRES')
DONE_PHASES = {'RELEASED', 'ROLLED_BACK', 'NO_CHANGE', 'SKIPPED_RUNNING'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _command(args):
    """Only scheduler reads plus exact hold/update/release are permitted."""
    require(args and args[0] in {'scontrol', 'sacct'}, 'Unsupported companion command')
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or 'Scheduler command failed')[-2000:])
    return result.stdout


def _show_job(jid):
    require(re.fullmatch(r'\d+', str(jid)), 'Invalid job ID')
    lines = _command(['scontrol', '--oneliner', 'show', 'job', str(jid)]).splitlines()
    rows = [_job_fields(line) for line in lines if line.strip()]
    matches = [r for r in rows if r.get('JobId') == str(jid)]
    require(len(matches) == 1, 'Scheduler did not return exactly this job')
    return matches[0]


def _show_node(node):
    rows = [_job_fields(line) for line in _command(
        ['scontrol', '--oneliner', 'show', 'node', node]).splitlines() if line.strip()]
    rows = [r for r in rows if r.get('NodeName') == node]
    require(len(rows) == 1, 'Cannot establish exact node identity: ' + node)
    return rows[0]


def _show_partition(part):
    rows = [_job_fields(line) for line in _command(
        ['scontrol', '--oneliner', 'show', 'partition', part]).splitlines() if line.strip()]
    rows = [r for r in rows if r.get('PartitionName') == part]
    require(len(rows) == 1, 'Cannot establish exact partition identity: ' + part)
    return rows[0]


def _expand_nodes(value):
    if not value or value in {'(null)', 'None', 'N/A'}:
        return set()
    require(re.fullmatch(r'[A-Za-z0-9_,.\[\]-]+', value), 'Unsafe node expression')
    if '[' not in value:
        return set(value.split(','))
    rows = _command(['scontrol', 'show', 'hostnames', value]).splitlines()
    require(all(re.fullmatch(r'[A-Za-z0-9_.-]+', r.strip()) for r in rows),
            'Invalid expanded node list')
    return {r.strip() for r in rows if r.strip()}


def _history(job_ids, start):
    if not job_ids:
        return {}
    ids = sorted({str(jid) for jid in job_ids})
    require(all(re.fullmatch(r'\d+', jid) for jid in ids), 'Invalid accounting job ID')
    text = _command(['sacct', '--allocations', '--noheader', '--parsable2',
                     '--starttime', dt.datetime.fromtimestamp(start - 120).strftime('%Y-%m-%dT%H:%M:%S'),
                     '--jobs', ','.join(ids), '--format',
                     'JobIDRaw,JobName%100,State,ExitCode,QOS,NodeList,Partition'])
    rows = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split('|')
        if fields[-1] == '':
            fields.pop()
        require(len(fields) == 7, 'Malformed companion accounting record')
        jid, name, state, code, qos, node, partition = [x.strip() for x in fields]
        require(jid in ids and jid not in rows, 'Unexpected or duplicate accounting job')
        rows[jid] = {'job_id': jid, 'job_name': name, 'state': state.split()[0].rstrip('+'),
                     'exit_code': code, 'qos': qos, 'node': node, 'partition': partition}
    return rows


def _seconds(value):
    require(isinstance(value, str) and value != 'UNLIMITED', 'Unknown walltime')
    days, clock = (value.split('-', 1) if '-' in value else ('0', value))
    parts = [int(x) for x in clock.split(':')]
    require(len(parts) in {2, 3}, 'Unsupported scheduler walltime')
    if len(parts) == 2:
        parts.insert(0, 0)
    return int(days) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


class _RouteEvidenceDeferred(RuntimeError):
    """A scheduler query did not establish whether a route is eligible."""


def _route_read(reader, *args):
    try:
        return reader(*args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        raise _RouteEvidenceDeferred(str(exc)) from exc


def _route_evidence(route, account, qos):
        part = _route_read(_show_partition, route['partition'])
        require(part.get('State') == 'UP', 'Candidate partition is not UP')
        require(part.get('RootOnly', 'NO') == 'NO' and part.get('ReqResv', 'NO') == 'NO',
                'Candidate partition requires privileged access or a reservation')
        for key, denied, value in (('AllowAccounts', 'DenyAccounts', account), ('AllowQos', 'DenyQos', qos)):
            allowed = part.get(key, '').split(',')
            require(('ALL' in allowed or value in allowed) and value not in part.get(denied, '').split(','),
                    'Candidate partition does not allow current account/QoS')
        require(_seconds(part['MaxTime']) >= 2 * 3600, 'Candidate partition cannot accept the 2h canary limit')
        nodes = _route_read(_expand_nodes, part['Nodes'])
        require(route['node'] in nodes, 'Candidate node is not a member of its partition')
        node = _route_read(_show_node, route['node'])
        require(node.get('State', '').split('+')[0] in {'IDLE', 'MIXED', 'ALLOCATED'}
                and not any(flag in node.get('State', '') for flag in ('DRAIN', 'FAIL', 'DOWN', 'NOT_RESPOND')),
                'Candidate node is unavailable')
        require('gpu:' + route['gpu'] + ':' in node.get('Gres', ''), 'Candidate GPU type differs')
        require(node.get('BootTime') not in {None, '', 'None', 'Unknown'}, 'Candidate boot identity missing')
        return {'available': True, 'node': node, 'partition': part, 'partition_nodes': sorted(nodes),
                'boot_time': node['BootTime'], 'slurmd_start_time': node.get('SlurmdStartTime')}


def _routes_evidence(account, qos):
    result = {}
    for name, route in ROUTES.items():
        try:
            result[name] = _route_evidence(route, account, qos)
        except _RouteEvidenceDeferred as exc:
            result[name] = {'available': False, 'deferred': True, 'error': str(exc)}
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            result[name] = {'available': False, 'error': str(exc)}
    return result


def _initial_state(policy):
    return {'schema_version': 1, 'policy_fingerprint': policy['fingerprint'],
        'created_at': policy['created_at'], 'updated_at': time.time(), 'routes': {
            name: {family: ({'status': 'WAITING_CANARY'} if policy['route_evidence'][name].get('available') else
                           {'status': 'BLOCKED', 'error': policy['route_evidence'][name].get('error')})
                   for family in FAMILIES} for name in ROUTES},
        'jobs': {}, 'warnings': [], 'last_updates': []}


def _load(run, *, check_snapshot=True):
    run = Path(run).resolve()
    meta = batch2.load_batch(run)
    policy = load_json(run / TAG / 'policy.json')
    require(policy.get('fingerprint') == fingerprint({k: v for k, v in policy.items() if k != 'fingerprint'}),
            'Acceleration policy fingerprint mismatch')
    require(policy['parent_run'] == str(run) and policy['parent_fingerprint'] == meta['fingerprint'],
            'Acceleration belongs to another original run')
    require(policy['source_snapshot_sha256'] == meta['source_snapshot_sha256'], 'Scientific source changed')
    source = Path(policy['controller_source'])
    require(source == run / TAG / 'source', 'Unexpected companion source location')
    require(batch2.digest(source / 'snapshot.json') == policy['controller_snapshot_sha256'],
            'Companion snapshot identity changed')
    if check_snapshot:
        from scripts.queue_ics import verify_snapshot
        verify_snapshot(source)
        verify_snapshot(Path(meta['source']))
    state = load_json(run / TAG / 'state.json')
    require(state['policy_fingerprint'] == policy['fingerprint'], 'Companion state/policy mismatch')
    return meta, policy, state


def prepare(run, controller_source):
    run, source = Path(run).resolve(), Path(controller_source).resolve()
    meta = batch2.load_batch(run)
    require(source == run / TAG / 'source', 'Freeze companion under the original run first')
    from scripts.queue_ics import verify_snapshot
    verify_snapshot(source)
    verify_snapshot(Path(meta['source']))
    path = run / TAG / 'policy.json'
    if path.exists():
        if not (run / TAG / 'state.json').exists():
            # Complete a crash between immutable policy creation and the first
            # state write only when no routing transaction could have started.
            policy = load_json(path)
            require(policy.get('fingerprint') == fingerprint({k:v for k,v in policy.items() if k != 'fingerprint'})
                    and policy.get('parent_fingerprint') == meta['fingerprint']
                    and policy.get('parent_run') == str(run)
                    and policy.get('controller_source') == str(source)
                    and policy.get('controller_snapshot_sha256') == batch2.digest(source / 'snapshot.json'),
                    'Interrupted preparation identity differs')
            require(not any((run / TAG / 'transactions').glob('*.json')), 'State missing after routing began')
            rq._atomic_json(run / TAG / 'state.json', _initial_state(policy))
        _, policy, _ = _load(run)
        return {'acceleration': 'ALREADY_PREPARED', 'policy_fingerprint': policy['fingerprint']}
    users, accounts, qoses = set(), set(), set()
    for record in meta['scope_plans'].values():
        plan = rq._plan(Path(record['path']))
        state = rq._read_state(plan)
        require(state['policy']['shared_limit'] == 60, 'Unexpected original shared limit')
        users.add(state['policy']['user']); qoses.add(state['policy']['qos'])
        accounts.update(rq._profile(plan, t)['account'] for t in state['tasks'].values())
    require(users == {getpass.getuser()} and len(accounts) == 1 and qoses == {'highpriority'},
            'Original user/account/QoS identity is not eligible')
    account, qos = next(iter(accounts)), next(iter(qoses))
    require(account == 'prj0000000262', 'Unexpected original account')
    evidence = _routes_evidence(account, qos)
    require(not any(row.get('deferred') for row in evidence.values()),
            'Candidate route query incomplete; retry prepare after scheduler queries recover')
    require(any(row.get('available') for row in evidence.values()), 'No candidate route is currently eligible')
    policy = {'schema_version': 1, 'created_at': time.time(), 'parent_run': str(run),
              'parent_fingerprint': meta['fingerprint'], 'source_snapshot_sha256': meta['source_snapshot_sha256'],
              'controller_source': str(source), 'controller_snapshot_sha256': batch2.digest(source / 'snapshot.json'),
              'user': next(iter(users)), 'account': account, 'qos': qos,
              'routes': copy.deepcopy(ROUTES), 'route_evidence': evidence,
              'families': list(FAMILIES), 'time_limits': {'insid3': '00:30:00', 'cgics_mllm': '02:00:00'},
              'canary_time_limits': {'insid3': '01:00:00', 'cgics_mllm': '02:00:00'},
              'known_configs': sorted(KNOWN_CONFIGS), 'max_updates_per_tick': 8,
              'canary_policy': 'existing_unstarted_known_formal_task_original_worker_1h_insid3_2h_cg_no_new_submission',
              'failure_policy': 'block_route_family_without_retry_or_queue_state_mutation'}
    policy['fingerprint'] = fingerprint(policy)
    # An interrupted initial state write is completed above from this immutable
    # policy, provided that no transaction journal exists.
    batch2.immutable_json(path, policy)
    state_path = run / TAG / 'state.json'
    require(not state_path.exists(), 'Unmatched companion state already exists')
    rq._atomic_json(state_path, _initial_state(policy))
    return {'acceleration': 'PREPARED_NOT_RUNNING_PROOF', 'policy_fingerprint': policy['fingerprint'],
            'routes': copy.deepcopy(ROUTES), 'original_tasks_reset': 0, 'new_jobs_submitted': 0}


def _save(run, state):
    state['updated_at'] = time.time()
    rq._atomic_json(Path(run) / TAG / 'state.json', state)


def _tasks(meta):
    rows = []
    for scope, record in meta['scope_plans'].items():
        plan = rq._plan(Path(record['path'])); state = rq._read_state(plan)
        for task in state['tasks'].values():
            if task['kind'] == 'experiment':
                rows.append((scope, plan, state, task))
    return rows


def _checked_task(plan, state, task, *, unclaimed=True):
    """Match accepted worker command and immutable intent, not mutable ID alone."""
    require(task['kind'] == 'experiment' and task['resource'] in FAMILIES, 'Unsupported task family')
    run = Path(plan['run_dir']).resolve(); source = Path(plan['project_root']).resolve()
    jid = str(task.get('job_id') or '')
    require(re.fullmatch(r'\d+', jid), 'Task lacks an accepted numeric job ID')
    intent_path = Path(task['batch2_intent'])
    require(not intent_path.is_symlink() and intent_path.resolve() == intent_path and
            intent_path == run / 'batch2_intents' / task['key'] / ('attempt-%05d.json' % task['attempts']),
            'Sealed intent path/attempt differs')
    require(batch2.digest(intent_path) == task['batch2_intent_sha256'], 'Sealed intent hash differs')
    intent = load_json(intent_path)
    require(intent['plan_fingerprint'] == plan['fingerprint'] and intent['plan_path'] == str(run / 'plan.json')
            and intent['worker_source'] == str(source)
            and intent['source_snapshot_sha256'] == batch2.digest(source / 'snapshot.json')
            and intent['attempt'] == task['attempts'], 'Intent plan/source identity differs')
    require(all(intent['task'].get(k) == task.get(k) for k in
                ('key', 'kind', 'resource', 'index', 'dependencies', 'job_name')), 'Intent task identity differs')
    group = [t for t in plan['tasks'] if t['resource'] == task['resource']]
    original = group[task['index']]
    require(original['id'] == task['key'], 'Frozen formal task index differs')
    require(task['dependencies'] and all(state['tasks'][key]['kind'] == 'smoke'
            and state['tasks'][key]['status'] == 'PASS' for key in task['dependencies']),
            'Original same-configuration smoke is not PASS')
    if unclaimed:
        require(not intent_path.with_suffix('.claim.json').exists() and
                not rq._marker_path(plan, task['key']).exists(), 'Original worker has already started')
    script = run / 'rolling_slurm' / (task['key'] + '.sh')
    require(not script.is_symlink(), 'Worker script is a symlink')
    lines = script.read_text().splitlines()
    expected = ['exec', plan['cluster']['launcher_python'], '-I', '-B', '-u',
                str(source / 'scripts/batch2_ics.py'), 'worker', '--intent', str(intent_path),
                '--intent-sha256', task['batch2_intent_sha256']]
    require(lines and shlex.split(lines[-1]) == expected, 'Accepted script differs from frozen worker')
    header = ("#!/usr/bin/env bash\nset -euo pipefail\n"
              "unset HF_TOKEN HUGGING_FACE_HUB_TOKEN ICS_DINOV3_URL PYTHONPATH PYTHONHOME\n"
              "export PYTHONNOUSERSITE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 HF_DEBUG=0\n"
              "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n")
    require(script.read_text() == header + 'exec ' + shlex.join(expected[1:]) + '\n',
            'Accepted worker script header or commands differ')
    return {'scope_run': str(run), 'plan_fingerprint': plan['fingerprint'], 'key': task['key'],
            'job_id': jid, 'job_name': task['job_name'], 'intent_path': str(intent_path),
            'intent_sha256': task['batch2_intent_sha256'], 'script': str(script),
            'script_sha256': batch2.digest(script), 'resource': task['resource'],
            'experiment_id': original['config_data']['experiment_id']}


def _identity(row, binding, policy):
    require(row.get('JobId') == binding['job_id'] and row.get('JobName') == binding['job_name']
            and row.get('UserId', '').split('(')[0] == policy['user']
            and row.get('Account') == policy['account'] and row.get('QOS') == policy['qos']
            and row.get('Command') == binding['script'], 'Live scheduler identity differs')
    require(batch2.digest(Path(binding['intent_path'])) == binding['intent_sha256']
            and batch2.digest(Path(binding['script'])) == binding['script_sha256'],
            'Accepted intent/script changed during routing')


def _empty(value):
    return '' if value in {None, '', '(null)', 'None', 'N/A'} else value


def _request(row):
    return {key: _empty(row.get(key)) for key in REQUEST_FIELDS}


def _held(row):
    return row.get('Priority') == '0' and row.get('Reason') == 'JobHeldUser'


def _update(jid, request):
    return _command(['scontrol', 'update', 'JobId=' + jid] +
                    [key + '=' + request[key] for key in REQUEST_FIELDS])


def _journal(path, record, phase, **extra):
    record.update(phase=phase, updated_at=time.time(), **extra)
    rq._atomic_json(path, record)


def _release(run, policy, path, record, *, rollback):
    """Release only our durable transaction's hold, also after a start race."""
    binding = record['binding']; jid = binding['job_id']
    row = _show_job(jid); _identity(row, binding, policy)
    if row.get('JobState') in rq.SLURM_TERMINAL:
        _journal(path, record, 'SKIPPED_RUNNING', note='Job already terminal; no resource update')
        return
    if row.get('JobState') != 'PENDING':
        if _held(row):
            _command(['scontrol', 'release', jid])
        _journal(path, record, 'SKIPPED_RUNNING', note='Start race: original running resources untouched')
        return
    if not _held(row):
        # An acknowledged release may precede a crash. Never seize a new hold.
        if _request(row) == record['before_request'] or _request(row) == record['desired']:
            _journal(path, record, 'NO_CHANGE', note='No owned hold remains')
            return
        raise ValueError('Pending request changed outside this held transaction')
    if rollback and _request(row) != record['before_request']:
        _journal(path, record, 'ROLLBACK_INTENT')
        _update(jid, record['before_request'])
        row = _show_job(jid); _identity(row, binding, policy)
        require(row.get('JobState') == 'PENDING' and _held(row)
                and _request(row) == record['before_request'], 'Rollback not verified while held')
    _journal(path, record, 'RELEASE_INTENT', rollback=rollback)
    _command(['scontrol', 'release', jid])
    after = _show_job(jid); _identity(after, binding, policy)
    require(not _held(after), 'User hold remains after release')
    _journal(path, record, 'ROLLED_BACK' if rollback else 'RELEASED', after=after)


def _retarget(run, policy, plan, state, task, desired, purpose):
    """Hold/recheck/update/release; never mutate resources of running work."""
    require(task['status'] == 'SUBMITTED', 'Only accepted unstarted tasks may be retargeted')
    binding = _checked_task(plan, state, task)
    before = _show_job(binding['job_id']); _identity(before, binding, policy)
    if before.get('JobState') != 'PENDING':
        return {'status': 'SKIPPED_RUNNING', 'job_id': binding['job_id']}
    require(before.get('Priority') != '0' and not before.get('Reason', '').startswith('JobHeld'),
            'Existing user/admin hold is not ours')
    original_request = _request(before)
    require(set(desired) == set(REQUEST_FIELDS), 'Incomplete scheduling update')
    if desired == original_request:
        return {'status': 'NO_CHANGE', 'job_id': binding['job_id']}
    require(all(before.get(k) is not None for k in ('TimeLimit', 'Partition', 'NumCPUs', 'NumNodes')),
            'Incomplete scheduler request evidence')
    folder = Path(run) / TAG / 'transactions'
    path = folder / (binding['job_id'] + '-' + str(time.time_ns()) + '.json')
    record = {'schema_version': 1, 'policy_fingerprint': policy['fingerprint'], 'created_at': time.time(),
              'binding': binding, 'before': before, 'before_request': original_request,
              'desired': desired, 'purpose': purpose}
    _journal(path, record, 'HOLD_INTENT')
    try:
        _command(['scontrol', 'hold', binding['job_id']])
        held = _show_job(binding['job_id']); _identity(held, binding, policy)
        if held.get('JobState') != 'PENDING':
            _release(run, policy, path, record, rollback=True)
            return {'status': 'SKIPPED_RUNNING', 'job_id': binding['job_id'], 'journal': str(path)}
        require(_held(held), 'Hold was not confirmed as our user hold')
        require(not Path(binding['intent_path']).with_suffix('.claim.json').exists(),
                'Worker started while waiting for hold')
        require(_request(held) == original_request, 'Request changed before our update')
        _journal(path, record, 'UPDATE_INTENT', held=held)
        _update(binding['job_id'], desired)
        after = _show_job(binding['job_id']); _identity(after, binding, policy)
        require(after.get('JobState') == 'PENDING' and _held(after) and _request(after) == desired,
                'Requested routing was not confirmed while held')
        require(all(after.get(k) == before.get(k) for k in RESOURCE_FIELDS),
                'Non-routing CPU/GPU/memory/task allocation request changed')
        _journal(path, record, 'UPDATED', after=after)
        _release(run, policy, path, record, rollback=False)
        return {'status': 'UPDATED', 'job_id': binding['job_id'], 'journal': str(path),
                'before': original_request, 'desired': desired}
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        _journal(path, record, record.get('phase', 'ERROR'), error=str(exc))
        try:
            _release(run, policy, path, record, rollback=True)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as recovery_error:
            _journal(path, record, 'RECOVERY_REQUIRED', recovery_error=str(recovery_error))
        return {'status': 'DEFERRED', 'job_id': binding['job_id'], 'journal': str(path), 'error': str(exc),
                'recovery_required': record['phase'] == 'RECOVERY_REQUIRED'}


def recover_transactions(run):
    _, policy, _ = _load(run)
    results = []
    for path in sorted((Path(run) / TAG / 'transactions').glob('*.json')):
        require(not path.is_symlink(), 'Transaction is a symlink')
        record = load_json(path)
        require(record['policy_fingerprint'] == policy['fingerprint'], 'Foreign routing transaction')
        if record['phase'] in DONE_PHASES:
            continue
        try:
            # A crash never grants a route. Restore its original pending request.
            _release(run, policy, path, record, rollback=True)
            results.append({'job_id': record['binding']['job_id'], 'status': record['phase']})
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            _journal(path, record, 'RECOVERY_REQUIRED', recovery_error=str(exc))
            results.append({'job_id': record['binding']['job_id'], 'status': 'RECOVERY_REQUIRED', 'error': str(exc)})
    return {'transactions': results, 'recovery_required': any(r['status'] == 'RECOVERY_REQUIRED' for r in results)}


def _canary_outcome(policy, plan, state, task, item, history):
    binding = _checked_task(plan, state, task, unclaimed=False)
    require(binding['job_id'] == item['job_id'] and binding['intent_sha256'] == item['intent_sha256'],
            'Canary accepted attempt changed')
    row = history.get(item['job_id'])
    if not row:
        return 'WAITING', 'Waiting for exact accounting evidence'
    require(row['job_name'] == item['job_name'] and row['qos'] == policy['qos'],
            'Canary accounting identity differs')
    if row['state'] not in rq.SLURM_TERMINAL:
        return 'WAITING', None
    if row['state'] != 'COMPLETED' or row['exit_code'] != '0:0':
        return 'BLOCKED', 'Canary ended ' + row['state'] + '/' + row['exit_code']
    require(row['node'] == item['node'] and row['partition'] == item['partition'],
            'Canary did not execute on its exact target route')
    claim = load_json(Path(binding['intent_path']).with_suffix('.claim.json'))
    require(all(claim.get(k) == v for k, v in {'slurm_job_id': item['job_id'],
        'job_name': item['job_name'], 'intent_sha256': item['intent_sha256']}.items()), 'Canary claim differs')
    problem = rq._marker_problem(plan, task)
    if problem:
        return 'WAITING', 'Awaiting exact original model markers: ' + problem
    marker = load_json(rq._marker_path(plan, task['key']))
    require(marker.get('batch2_intent_sha256') == item['intent_sha256']
            and marker.get('job_name') == item['job_name'], 'Canary marker intent/name differs')
    return 'VERIFIED', None


def _desired(policy, plan, task, before, routes, *, canary=None):
    request = _request(before)
    # Preserve every original and any additional live exclusion. Only the
    # explicitly proven/retested target is ever removed.
    excluded = (_expand_nodes(rq._profile(plan, task).get('exclude', '')) |
                _expand_nodes(before.get('ExcNodeList', '')))
    if canary:
        route = ROUTES[canary]
        group = [t for t in plan['tasks'] if t['resource'] == task['resource']]
        require(group[task['index']]['config_data']['experiment_id'] in KNOWN_CONFIGS,
                'Unknown configuration is not eligible as a hardware canary')
        limit = policy['canary_time_limits'][task['resource']]
        if _seconds(request['TimeLimit']) > _seconds(limit):
            request['TimeLimit'] = limit
        excluded.discard(route['node'])
        request.update(Partition=route['partition'], ReqNodeList=route['node'],
                       ExcNodeList=','.join(sorted(excluded)))
        return request
    require(not _empty(before.get('ReqNodeList')), 'Production job already has a required node')
    allowed = set(request['Partition'].split(',')) - {'testqueue'}
    require(set(request['Partition'].split(',')) <= {'h200p', 'l40sp', 'testqueue'},
            'Unexpected original job partition')
    family = task['resource']
    if routes['l40s_reboot'][family]['status'] == 'VERIFIED':
        allowed.add('l40sp')
        excluded.discard(ROUTES['l40s_reboot']['node'])
    if routes['testqueue_bfs'][family]['status'] == 'VERIFIED':
        allowed.add('testqueue')
        excluded.update(set(policy['route_evidence']['testqueue_bfs']['partition_nodes']) -
                        {ROUTES['testqueue_bfs']['node']})
    request.update(Partition=','.join(p for p in ('h200p', 'l40sp', 'testqueue') if p in allowed),
                   ReqNodeList='', ExcNodeList=','.join(sorted(excluded)))
    group = [t for t in plan['tasks'] if t['resource'] == family]
    original = group[task['index']]
    if original['config_data']['experiment_id'] in KNOWN_CONFIGS:
        target = policy['time_limits'][family]
        if _seconds(request['TimeLimit']) > _seconds(target):
            request['TimeLimit'] = target
    return request


def _relink_canary_journals(run, policy, state, by_id):
    """Recover the state link if a released canary preceded the state write."""
    folder = run / TAG / 'transactions'
    for route, families in state['routes'].items():
        for family, item in families.items():
            if not item.get('job_id') or item.get('journal'):
                continue
            current = by_id.get(item['job_id'])
            if not current:
                continue
            try:
                _, plan, queue, task = current
                binding = _checked_task(plan, queue, task, unclaimed=False)
                require(binding['job_id'] == item['job_id']
                        and binding['job_name'] == item['job_name']
                        and binding['intent_sha256'] == item['intent_sha256'],
                        'Canary state binding changed before journal recovery')
                matches = []
                for path in folder.glob(item['job_id'] + '-*.json'):
                    require(not path.is_symlink(), 'Canary transaction is a symlink')
                    record = load_json(path)
                    if (record.get('policy_fingerprint') == policy['fingerprint']
                            and record.get('purpose') == 'canary:' + route + ':' + family
                            and record.get('binding') == binding
                            and record.get('desired', {}).get('ReqNodeList') == item['node']
                            and record.get('desired', {}).get('Partition') == item['partition']):
                        matches.append((path, record))
                require(len(matches) <= 1, 'Multiple canary routing transactions match this attempt')
                if matches:
                    path, record = matches[0]
                    item['journal'] = str(path)
                    if item['status'] == 'CANARY_SELECTED' and record['phase'] == 'RELEASED':
                        item['status'] = 'CANARY_SUBMITTED'
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                state['warnings'].append('Canary journal recovery deferred: ' + str(exc))


def tick(run, stop_requested=None):
    run = Path(run).resolve()
    meta, policy, state = _load(run)
    recovery = recover_transactions(run)
    state['warnings'] = []; state['last_updates'] = []
    if recovery['recovery_required']:
        state['warnings'].append('Routing recovery required; no new pending jobs modified')
        _save(run, state); return status(run)
    stopped = stop_requested or (lambda: False)
    rows = _tasks(meta)
    by_id = {str(task.get('job_id')): (scope, plan, queue, task) for scope, plan, queue, task in rows if task.get('job_id')}
    _relink_canary_journals(run, policy, state, by_id)
    try:
        fresh = _routes_evidence(policy['account'], policy['qos'])
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        state['warnings'].append('Fresh route evidence unavailable: ' + str(exc))
        _save(run, state); return status(run)
    for route in ROUTES:
        if fresh[route].get('deferred'):
            state['warnings'].append('Route query deferred; prior evidence retained: ' + route + ': '
                                     + fresh[route].get('error', 'Unknown scheduler response'))
            continue
        initial = policy['route_evidence'][route]
        if (not initial.get('available') or not fresh[route].get('available') or
                fresh[route].get('boot_time') != initial.get('boot_time') or
                fresh[route].get('slurmd_start_time') != initial.get('slurmd_start_time') or
                fresh[route].get('partition_nodes') != initial.get('partition_nodes')):
            for family in FAMILIES:
                state['routes'][route][family].update(status='BLOCKED',
                    error=fresh[route].get('error', 'Node boot or partition membership changed'))
    # A node may become unavailable while its original accepted canary is still
    # pending. Restore that one job's prior routing instead of leaving it pinned
    # indefinitely; running/completed jobs are never moved.
    for route, families in state['routes'].items():
        if fresh[route].get('deferred'):
            continue
        for item in families.values():
            if item['status'] != 'BLOCKED' or not item.get('journal') or item.get('withdrawn'):
                continue
            current = by_id.get(item.get('job_id'))
            if not current or current[3]['status'] != 'SUBMITTED':
                continue
            _, plan, queue, task = current
            try:
                journal = Path(item['journal'])
                require(journal.parent == run / TAG / 'transactions' and not journal.is_symlink(),
                        'Canary withdrawal journal is outside this run')
                original = load_json(journal)
                require(original['policy_fingerprint'] == policy['fingerprint']
                        and original['binding']['job_id'] == item['job_id'], 'Canary withdrawal identity differs')
                event = _retarget(run, policy, plan, queue, task, original['before_request'], 'withdraw-blocked-canary')
                state['last_updates'].append(event)
                if event['status'] in {'UPDATED', 'NO_CHANGE', 'SKIPPED_RUNNING'}:
                    item['withdrawn'] = True
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                state['warnings'].append('Canary withdrawal deferred: ' + str(exc))
    canaries = [item for families in state['routes'].values() for item in families.values() if item.get('job_id')]
    try:
        history = _history([c['job_id'] for c in canaries], policy['created_at'])
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        history = {}; state['warnings'].append('Canary accounting unavailable: ' + str(exc))
    for route, families in state['routes'].items():
        if fresh[route].get('deferred'):
            continue
        for family, item in families.items():
            if item['status'] not in {'CANARY_SELECTED', 'CANARY_SUBMITTED'}:
                continue
            if item['job_id'] not in by_id:
                item.update(status='BLOCKED', error='Canary not present in original queue'); continue
            scope, plan, queue, task = by_id[item['job_id']]
            try:
                result, error = _canary_outcome(policy, plan, queue, task, item, history)
                if result != 'WAITING':
                    item.update(status=result, checked_at=time.time(), error=error)
                elif error:
                    item['note'] = error
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                item.update(status='BLOCKED', error=str(exc))
    _save(run, state)
    used = {c['job_id'] for c in canaries}
    pending = sorted((r for r in rows if r[3]['status'] == 'SUBMITTED' and r[3]['resource'] in FAMILIES),
                     key=lambda row: (row[0] != 'core', row[3]['resource'], row[3]['key']))
    for route, families in state['routes'].items():
        if fresh[route].get('deferred'):
            continue
        for family, item in families.items():
            if stopped(): break
            if item['status'] != 'WAITING_CANARY': continue
            for scope, plan, queue, task in pending:
                jid = str(task.get('job_id') or '')
                if task['resource'] != family or jid in used: continue
                try:
                    binding = _checked_task(plan, queue, task)
                    before = _show_job(jid); _identity(before, binding, policy)
                    if before.get('JobState') != 'PENDING' or before.get('Priority') == '0': continue
                    desired = _desired(policy, plan, task, before, state['routes'], canary=route)
                    item.update(status='CANARY_SELECTED', scope=scope, key=task['key'], job_id=jid,
                                job_name=task['job_name'], intent_sha256=binding['intent_sha256'],
                                node=ROUTES[route]['node'], partition=ROUTES[route]['partition'],
                                node_boot=fresh[route]['boot_time'])
                    used.add(jid); _save(run, state)
                    event = _retarget(run, policy, plan, queue, task, desired, 'canary:' + route + ':' + family)
                    state['last_updates'].append(event)
                    if event['status'] == 'UPDATED':
                        item.update(status='CANARY_SUBMITTED', journal=event['journal'])
                    elif event['status'] == 'SKIPPED_RUNNING':
                        prior = dict(item)
                        item.clear(); item.update(status='WAITING_CANARY', previous_selection=prior,
                                                  note='Original job started before routing; choose another unstarted task next poll')
                    else: item.update(status='BLOCKED', error='Canary routing deferred: ' + event.get('error', event['status']))
                    _save(run, state); break
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                    state['warnings'].append('Skipped canary candidate ' + jid + ': ' + str(exc))
    # Routing failures on any migrated job stop expansion for that family/node.
    for jid, event in state['jobs'].items():
        current = by_id.get(jid)
        if current and current[3]['status'] == 'FAILED' and not event.get('failure_checked'):
            try:
                outcome = _history([jid], policy['created_at']).get(jid)
                if outcome and outcome['job_name'] == current[3]['job_name'] and outcome['qos'] == policy['qos']:
                    for route in ROUTES:
                        if outcome['node'] == ROUTES[route]['node']:
                            state['routes'][route][current[3]['resource']].update(
                                status='BLOCKED', error='Migrated formal task failed: ' + jid)
                    event['failure_checked'] = True
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                state['warnings'].append('Failure accounting unavailable: ' + str(exc))
    for scope, plan, queue, task in pending:
        # Preserve current proven routing during a transient query outage.
        # Both expansion and withdrawal need fresh evidence; retry next poll.
        if any(evidence.get('deferred') for evidence in fresh.values()): break
        if stopped() or len(state['last_updates']) >= policy['max_updates_per_tick']: break
        jid = str(task.get('job_id') or '')
        if jid in used: continue
        try:
            binding = _checked_task(plan, queue, task)
            before = _show_job(jid); _identity(before, binding, policy)
            if before.get('JobState') != 'PENDING' or before.get('Priority') == '0': continue
            desired = _desired(policy, plan, task, before, state['routes'])
            if desired == _request(before): continue
            event = _retarget(run, policy, plan, queue, task, desired, 'production')
            state['last_updates'].append(event)
            if event['status'] == 'UPDATED':
                state['jobs'][jid] = {'scope': scope, 'key': task['key'], 'job_name': task['job_name'],
                                      'resource': task['resource'], 'desired': desired, 'journal': event['journal']}
            _save(run, state)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            state['warnings'].append('Skipped pending task ' + jid + ': ' + str(exc))
    state['warnings'] = state['warnings'][:20]
    _save(run, state)
    return status(run)


def status(run):
    meta, policy, state = _load(run, check_snapshot=False)
    counts = Counter(task['status'] for _, _, _, task in _tasks(meta))
    return {'acceleration': 'PREPARED' if not state.get('last_updates') and not state['jobs'] else 'ACTIVE_POLICY',
            'original_formal_counts': dict(counts), 'original_formal_total': sum(counts.values()),
            'terminal': all(k in rq.TERMINAL for k in counts),
            'routes': state['routes'], 'migrated_jobs': len(state['jobs']),
            'last_updates': state['last_updates'], 'warnings': state['warnings'],
            'updated_at': state['updated_at'], 'policy_fingerprint': policy['fingerprint'],
            'new_jobs_submitted': 0, 'original_queue_modified': False,
            'status_note': 'Saved companion observation; original manager and all scientific workers remain unchanged.'}
