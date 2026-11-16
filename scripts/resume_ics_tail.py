#!/usr/bin/env python3
"""Audited same-run tail recovery; never changes scientific source or successful results."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

PLAN = '4d7678557ef476de3bc2fd605ee7384745ba991599bc8b91bba5855f298c1a05'
REPAIR = '403253e7e852cfac2b571d4c47d8513ca852264e237ec25691092c6ee42c782c'
TAG = 'tail-resume-20261116'
PINS = {
    'source/snapshot.json': '954ec4729afe36c42dfe8c65264a83fb1ff5888fa4b70a5aa5edb7aba9c4a339',
    'repair/source/snapshot.json': '99cfcae39068b5d0adb6e650de04377648825491147b54a83ba3f70a47a33b7c',
    'scheduling/policy.json': '348b262afef33f989efa8e287f6b0d4b94100afc6578b1b3d5c521889d99b97d',
    'repair/policy.json': '96c8b09dbd1798d39445fe69d1795e1a4bcfaa43ef1cf249b9d6f7aedca6e59a',
}
UNCERTAIN = (
    'insid3_full--lvis_fold6_standard--3', 'insid3_full--lvis_fold7_standard--0',
    'insid3_full--lvis_fold7_standard--1', 'insid3_full--lvis_fold7_standard--3',
    'insid3_full--lvis_fold8_standard--0', 'insid3_full--lvis_fold8_standard--1',
    'insid3_full--lvis_fold8_standard--2', 'insid3_full--lvis_fold8_standard--3',
    'smoke-insid3_no_aggregation-lvis',
)
INTERRUPTED = 'insid3_full--coco_fold1_standard--3'
SELECTED = (INTERRUPTED,) + UNCERTAIN
ALIASES = {
    '201529': ('insid3_full--lvis_fold7_standard--3', 'insid3_full--lvis_fold9_standard--0'),
    '201530': ('insid3_full--lvis_fold8_standard--0', 'insid3_full--lvis_fold9_standard--1'),
}
SCIENTIFIC_FAILURES = {
    'cgics_full--lvis_fold4_reference--5': ('203920', 'd29b60ec730dcb80ea7484b1acfd765c512fdecb091dbd3b2669c7bc0da0c316', '68e95aa81a1fdf0aea7356a623776131fb01dc9963cc7995e47c1002c5af1025'),
    'cgics_full--lvis_fold4_reference--17': ('203967', '5147efd7e2e8afa865fa6dea4eb1bb8ce3d4d14d0f3e9b900804005c84e7bca9', '70ea16901b4fb4ea4f52b06e70a773c1713e24d6ed88b788e2f2464e8d2ed9b3'),
    'cgics_full--lvis_fold4_reference--27': ('204000', 'ee2cb85d7586ea8d21723bb912f2aa57f5a3916b15ff2640bad3e6a7ea1c6b46', '76eb7764804aa65a326350c9f263692ae590abb74feeead581a7a0e2237dd98c'),
    'cgics_full--lvis_fold8_reference--20': ('204471', '366003cd3ae29221d055a0a8f5a0d1e73a9664b7dcf5938e58cfaafae392b71e', '994b702733510e6b9e5d8c8a21849abf5e7357b990c40e921507f086833cf1dd'),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def safe_path(root, path):
    root, path = Path(root).absolute(), Path(path).absolute()
    relative = path.relative_to(root)
    require('..' not in relative.parts, 'Parent traversal refused')
    require(not root.is_symlink(), 'Symlink root refused')
    current = root
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), 'Symlink evidence refused: ' + str(current))
    return path


def read_bytes(path, limit=64 * 2**20):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), 'Missing/nonregular evidence: ' + str(path))
    require(path.stat().st_size <= limit, 'Evidence file too large: ' + str(path))
    return path.read_bytes()


def load(path):
    return json.loads(read_bytes(path))


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_json(path, value):
    atomic_bytes(path, encoded(value))


@contextmanager
def lock(path):
    handle = Path(path).open('a+')
    try:
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError('Lock busy: ' + str(path)) from exc
        yield handle
    finally:
        handle.close()


def verify_static(run):
    run = Path(run).absolute()
    require(run.resolve() == run, 'Run path must not contain symlinks')
    plan = load(safe_path(run, run / 'plan.json'))
    require(plan.get('fingerprint') == PLAN and fingerprint({k:v for k,v in plan.items() if k != 'fingerprint'}) == PLAN,
            'Unexpected or changed immutable plan')
    require(plan['run_dir'] == str(run) and plan['project_root'] == str(run / 'source'), 'Plan/source path mismatch')
    for name, digest in PINS.items():
        require(sha(read_bytes(safe_path(run, run / name))) == digest, 'Pinned evidence changed: ' + name)
    for prefix in ('source', 'repair/source'):
        snapshot = load(run / prefix / 'snapshot.json')
        for name, digest in snapshot['files'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe source snapshot name')
            require(sha(read_bytes(safe_path(run, run / prefix / name))) == digest, 'Frozen source changed: ' + prefix + '/' + name)
    policy = load(run / 'repair/policy.json')
    require(policy['fingerprint'] == REPAIR and policy['controller_source'] == str(run / 'repair/source'), 'Repair controller mismatch')
    require((run / 'repair/applied.json').is_file(), 'Original repair was not applied')
    return plan


def command(argv, timeout=30):
    env = dict(os.environ)
    env['PATH'] = '/cm/shared/apps/slurm/current/bin:' + env.get('PATH', '')
    env['SLURM_CONF'] = '/cm/shared/apps/slurm/var/etc/slurm/slurm.conf'
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            timeout=timeout, env=env)
    require(result.returncode == 0, 'Command failed: ' + ' '.join(argv[:2]) + ': ' + result.stderr[-2000:])
    return result.stdout


def scheduler_evidence(state):
    user = state['policy']['user']
    queue = {}
    for line in command(['squeue', '--noheader', '--all', '--array', '--user', user,
                         '--format', '%i|%j|%T|%q']).splitlines():
        if not line.strip(): continue
        row = [x.strip() for x in line.split('|')]
        require(len(row) == 4 and row[0] not in queue, 'Malformed/duplicate squeue row')
        queue[row[0]] = dict(zip(('job_id', 'job_name', 'state', 'qos'), row))
    history = {}
    start = dt.datetime.fromtimestamp(state['created_at'] - 120).strftime('%Y-%m-%dT%H:%M:%S')
    fields = ('job_id', 'job_name', 'state', 'exit_code', 'qos', 'node', 'submit', 'start', 'end')
    for line in command(['sacct', '--noheader', '--parsable2', '--allocations', '--user', user,
                         '--starttime', start, '--format', 'JobIDRaw,JobName%128,State%40,ExitCode,QOS,NodeList,Submit,Start,End']).splitlines():
        if not line.strip(): continue
        row = [x.strip() for x in line.split('|')]
        if len(row) == 10 and not row[-1]: row.pop()
        require(len(row) == 9 and re.fullmatch(r'\d+(?:_\d+)?', row[0]), 'Malformed sacct allocation row')
        require(row[0] not in history, 'Duplicate sacct allocation identity')
        row[2] = row[2].split()[0].rstrip('+')
        history[row[0]] = dict(zip(fields, row))
    return queue, history


def marker_checks(run, task, status='PASS'):
    key = task['key']; jid = str(task['job_id'])
    marker = load(safe_path(run, run / 'rolling_markers' / (key + '.json')))
    wanted = {'key':key, 'status':status, 'plan_fingerprint':PLAN, 'slurm_job_id':jid}
    require(all(marker.get(k) == v for k,v in wanted.items()), 'Task marker identity mismatch: ' + key)
    if status == 'PASS' and task['kind'] in ('experiment', 'smoke'):
        complete = load(safe_path(run, run / 'jobs' / key / 'complete.json'))
        wanted = {'task_id':key, 'status':'PASS', 'plan_fingerprint':PLAN, 'slurm_job_id':jid}
        require(all(complete.get(k) == v for k,v in wanted.items()), 'Model complete identity mismatch: ' + key)
    return marker


def audit_state(plan, state, queue, history, run):
    run = Path(run); tasks = state['tasks']
    require(state.get('plan_fingerprint') == PLAN, 'State plan identity mismatch')
    require(not state.get('repair_circuit_breaker') and not state.get('last_error'), 'Unreviewed controller error/breaker')
    require(not queue, 'User still has queued/running jobs; no state mutation permitted')
    require(Counter(t['status'] for t in tasks.values()) == Counter({'PASS':1426, 'FAILED':5, 'UNCERTAIN':9, 'PENDING':15}),
            'Task counts changed; this recovery selection is no longer current')
    require({k for k,t in tasks.items() if t['status'] == 'UNCERTAIN'} == set(UNCERTAIN), 'UNCERTAIN task selection changed')
    require({k for k,t in tasks.items() if t['status'] == 'FAILED'} == set(SCIENTIFIC_FAILURES) | {INTERRUPTED}, 'FAILED task selection changed')
    require(Counter(t['kind'] for t in tasks.values() if t['status'] == 'PASS') == Counter({'experiment':1365, 'smoke':59, 'data':1, 'preflight':1}),
            'Successful stage counts changed')
    definitions = {t['id']:t for t in plan['tasks'] + plan['smoke']}
    pending = [k for k,t in tasks.items() if t['status'] == 'PENDING' and t['kind'] == 'experiment']
    require(len(pending) == 14, 'Expected fourteen dependent formal tasks')
    for key in pending:
        task = tasks[key]
        require(definitions[key]['config_data']['experiment_id'] == 'insid3_no_aggregation' and task['attempts'] == 0 and not task.get('job_id'),
                'Pending task no longer matches audited no-aggregation gate')
        require([d for d in task['dependencies'] if tasks[d]['status'] != 'PASS'] == ['smoke-insid3_no_aggregation-lvis'],
                'Pending task has another unmet dependency')
    by_id = defaultdict(list)
    for key, task in tasks.items():
        if task.get('job_id'): by_id[str(task['job_id'])].append(key)
        if task['status'] in ('PASS', 'FAILED'):
            row = history.get(str(task.get('job_id')))
            require(row is not None and row['job_name'] == task['job_name'] and row['qos'] == state['policy']['qos'],
                    'Known allocation identity changed: ' + key)
            expected = ('COMPLETED','0:0') if task['status'] == 'PASS' else ('FAILED','1:0')
            require((row['state'],row['exit_code']) == expected, 'Known allocation outcome changed: ' + key)
            if task['status'] == 'PASS': marker_checks(run, task)
    duplicates = {jid:set(keys) for jid,keys in by_id.items() if len(keys)>1}
    require(duplicates == {jid:set(pair) for jid,pair in ALIASES.items()}, 'Unexpected duplicate job IDs')
    for jid, (wrong, correct) in ALIASES.items():
        require(tasks[wrong].get('job_id') == jid and tasks[correct].get('job_id') == jid and tasks[correct]['status']=='PASS',
                'Audited job ID alias changed')
        marker_checks(run, tasks[correct])
    selected_claims={}
    for key in SELECTED:
        task = tasks[key]; attempt = task.get('repair_attempt', {})
        definitions_for_resource = [t for t in (plan['smoke'] if task['kind']=='smoke' else plan['tasks'])
                                    if t['resource']==task['resource']]
        index=task.get('index')
        require(type(index) is int and 0<=index<len(definitions_for_resource) and definitions_for_resource[index]['id']==key,
                'Task resource/index does not select its immutable plan definition: '+key)
        intent_path = safe_path(run, Path(attempt.get('intent_path', '')))
        require(intent_path == run / 'repair/intents' / key / ('attempt-' + str(task['attempts']) + '.json'), 'Intent path mismatch: ' + key)
        raw = read_bytes(intent_path); intent = json.loads(raw)
        require(sha(raw) == attempt.get('intent_sha256'), 'Intent SHA mismatch: ' + key)
        require(intent['plan_fingerprint']==PLAN and intent['repair_policy_fingerprint']==REPAIR and intent['worker_source']==str(run/'source'),
                'Intent provenance mismatch: ' + key)
        require(intent.get('worker_snapshot_sha256')==PINS['source/snapshot.json'] and intent.get('queue_policy')==state['policy'],
                'Intent worker snapshot/queue policy mismatch: '+key)
        for field in ('key','kind','resource','index','dependencies','job_name','attempts'):
            require(intent['task'].get(field) == task.get(field), 'Sealed task mismatch: ' + key + '/' + field)
        next_intent = intent_path.with_name('attempt-' + str(task['attempts']+1) + '.json')
        require(not next_intent.exists() and not next_intent.with_suffix('.claim.json').exists(), 'Next attempt already exists: ' + key)
        claim_path=intent_path.with_suffix('.claim.json')
        selected_claims[key]=sha(read_bytes(claim_path)) if claim_path.exists() else None
        if key in UNCERTAIN:
            require(not any(row['job_name']==task['job_name'] for row in history.values()), 'Uncertain task now has an allocation: ' + key)
            for path in (intent_path.with_suffix('.claim.json'), run/'rolling_markers'/(key+'.json'), run/'jobs'/key/'complete.json'):
                require(not path.exists() and not path.is_symlink(), 'Uncertain task has execution evidence: ' + str(path))
        else:
            require(task['job_id']=='201500', 'Interrupted task allocation changed')
            claim = load(intent_path.with_suffix('.claim.json'))
            require(str(claim.get('slurm_job_id')) == '201500' and claim.get('intent_sha256') == sha(raw), 'Interrupted claim changed')
            marker = marker_checks(run, task, 'FAIL')
            require(marker.get('error_type')=='KeyboardInterrupt' and marker.get('error')=='Termination signal 15', 'Interrupted task failure cause changed')
    scientific = []
    for key, (jid, name, digest) in SCIENTIFIC_FAILURES.items():
        require(str(tasks[key]['job_id']) == jid, 'Scientific failure allocation changed')
        path = safe_path(run, Path(definitions[key]['output'])/'episodes'/(name+'.json'))
        raw = read_bytes(path); row = json.loads(raw)
        require(sha(raw)==digest and row.get('status')=='error' and row.get('error')=='Reference mask is empty at the CG-ICS working resolution',
                'Scientific failure record changed: ' + key)
        scientific.append({'task':key, 'path':str(path), 'sha256':digest, 'action':'PRESERVE_FAILED'})
    return {'selected':list(SELECTED), 'preserved_pass':1426, 'pending_gate_tasks':pending,
            'preserved_scientific_failures':scientific, 'duplicate_ids_resolved_by_new_attempts':ALIASES,
            'selected_claims':selected_claims, 'history':history, 'queried_at':time.time()}


def inspect_selected_episodes(plan, state, run):
    run = Path(run); definitions = {t['id']:t for t in plan['tasks']+plan['smoke']}
    manifests = {}; rows = []
    for key in SELECTED:
        task = definitions[key]; output = safe_path(run, Path(task['output']))
        manifest_path = Path(task['manifest'])
        if str(manifest_path) not in manifests:
            manifests[str(manifest_path)] = load(manifest_path)
        episodes = manifests[str(manifest_path)]['episodes'][:task.get('max_episodes')]
        selected = [ep for i,ep in enumerate(episodes) if i % task['num_shards'] == task['shard_index']]
        require(len({ep['episode_id'] for ep in selected})==len(selected), 'Duplicate selected episode IDs')
        counts = Counter(); evidence = []
        binding = load(output/'run.json')['binding'] if (output/'run.json').exists() else None
        for ep in selected:
            path = safe_path(run, output/'episodes'/(sha(ep['episode_id'].encode())+'.json'))
            if not path.exists(): counts['missing']+=1; continue
            raw = read_bytes(path); row = json.loads(raw)
            require(row.get('episode_id')==ep['episode_id'] and row.get('status')=='ok', 'Existing error/foreign episode in selected scope: '+str(path))
            require(binding and row.get('run_id')==binding.get('run_id') and row.get('experiment_id')==task['config_data']['experiment_id'], 'Episode output binding mismatch')
            prediction = safe_path(output, output/row['prediction_path'])
            require(sha(read_bytes(prediction))==row.get('prediction_sha256'), 'Successful prediction changed: '+str(prediction))
            counts['ok']+=1; evidence.append({'path':str(path), 'sha256':sha(raw), 'prediction_path':str(prediction), 'prediction_sha256':row['prediction_sha256']})
        if key == INTERRUPTED:
            require(counts == Counter({'ok':155,'missing':95}), 'Interrupted shard differs from audited 155 ok / 95 missing')
        else:
            require(counts.get('ok',0)==0 and counts.get('missing',0)==len(selected) and len(selected)>0,
                    'Uncertain shard has unexpected existing results: '+key)
        rows.append({'task':key, 'counts':dict(counts), 'selected':len(selected), 'preserved_successes':evidence})
    return rows


def build_after_state(state, transaction_id):
    after = copy.deepcopy(state)
    for key in SELECTED:
        task = after['tasks'][key]
        prior = {k:copy.deepcopy(v) for k,v in task.items() if k!='attempt_history'}
        prior['tail_resume_reason'] = 'submission_identity_not_executed' if key in UNCERTAIN else 'terminated_with_partial_results'
        task.setdefault('attempt_history', []).append(prior)
        for field in ('job_id','submitted_at','attempt_started_at','ended_at','scheduler_state','exit_code','error','terminal_seen_at','retry_after','repair_attempt','recovered_at'):
            task.pop(field, None)
        task['status']='PENDING'
    after.update(last_error=None, capacity=None, updated_at=time.time(),
                 tail_resume={'transaction_id':transaction_id,'selected':list(SELECTED),'preserved_scientific_failures':list(SCIENTIFIC_FAILURES)})
    require(Counter(t['status'] for t in after['tasks'].values()) == Counter({'PASS':1426,'FAILED':4,'PENDING':25}), 'Internal after-state count mismatch')
    return after


def tree_fingerprint(path):
    path = Path(path)
    if not path.exists():
        require(not path.is_symlink(), 'Dangling symlink in archive path')
        return None
    require(not path.is_symlink(), 'Archive symlink refused')
    if path.is_file(): return {'file':sha(read_bytes(path))}
    require(path.is_dir(), 'Unexpected archive path type')
    files={}
    for p in sorted(path.rglob('*')):
        require(not p.is_symlink(), 'Archive contains symlink')
        if p.is_file(): files[str(p.relative_to(path))]=sha(read_bytes(p))
        else: require(p.is_dir(), 'Archive contains nonregular entry')
    return {'directory':files}


def archive_selection(run, state):
    rows=[]
    for key in SELECTED:
        for label, relative in (('job',Path('jobs')/key),('marker',Path('rolling_markers')/(key+'.json')),('script',Path('rolling_slurm')/(key+'.sh'))):
            source=safe_path(run,run/relative)
            target=run/TAG/'archive'/key/('attempt-'+str(state['tasks'][key]['attempts']))/label
            require(not target.exists() and not target.is_symlink(), 'Archive target already exists before transaction')
            rows.append({'source':str(source),'archive':str(target),'fingerprint':tree_fingerprint(source)})
    return rows


def check_transaction_evidence(run,txn):
    run=Path(run)
    claims=txn['proof']['selected_claims']
    require(set(claims)==set(SELECTED), 'Transaction claim guard selection changed')
    for key in SELECTED:
        attempt=txn['before_state']['tasks'][key]['repair_attempt']
        intent=safe_path(run,Path(attempt['intent_path']))
        require(sha(read_bytes(intent))==attempt['intent_sha256'], 'Sealed intent changed after preparation: '+key)
        claim=safe_path(run,intent.with_suffix('.claim.json'))
        expected=claims[key]
        if expected is None:
            require(not claim.exists() and not claim.is_symlink(), 'Claim appeared after preparation: '+key)
        else:
            require(sha(read_bytes(claim))==expected, 'Claim changed after preparation: '+key)
    for row in txn['proof']['preserved_scientific_failures']:
        path=safe_path(run,Path(row['path']))
        require(sha(read_bytes(path))==row['sha256'], 'Scientific failure changed after preparation: '+row['task'])
    for item in txn['episode_inspection']:
        for row in item['preserved_successes']:
            path=safe_path(run,Path(row['path']))
            prediction=safe_path(run,Path(row['prediction_path']))
            require(sha(read_bytes(path))==row['sha256'] and sha(read_bytes(prediction))==row['prediction_sha256'],
                    'Preserved successful episode changed after preparation: '+str(path))


def apply_transaction(run, txn, fault=None):
    """Roll forward atomic evidence moves and the state swap; never recreate a started attempt."""
    run=Path(run); validate_transaction(run,txn)
    check_transaction_evidence(run,txn)
    current=sha(read_bytes(run/'rolling_state.json'))
    require(current in {txn['before_sha256'],txn['after_sha256']}, 'State changed outside this transaction')
    for index,row in enumerate(txn['archives']):
        source=Path(row['source']); destination=Path(row['archive']); expected=row['fingerprint']
        require(source.is_relative_to(run) and destination.is_relative_to(run/TAG/'archive'), 'Transaction archive path escaped run')
        a,b=tree_fingerprint(source),tree_fingerprint(destination)
        if expected is None:
            require(a is None and b is None, 'Previously absent evidence appeared during transaction')
        elif a is not None:
            require(a==expected and b is None, 'Evidence changed or both archive/current exist')
            destination.parent.mkdir(parents=True,exist_ok=True)
            os.replace(source,destination)
        else:
            require(b==expected, 'Archived evidence missing or changed')
        if fault: fault('archive',index)
    if current!=txn['after_sha256']:
        atomic_bytes(run/'rolling_state.json',encoded(txn['after_state']))
    if fault: fault('state',None)
    atomic_json(run/TAG/'applied.json', {'transaction_id':txn['transaction_id'], 'applied_at':time.time(),
                'before_sha256':txn['before_sha256'],'after_sha256':txn['after_sha256'],
                'selected':list(SELECTED),'scientific_failures_preserved':list(SCIENTIFIC_FAILURES)})


def validate_transaction(run,txn):
    run=Path(run)
    require(txn.get('integrity_sha256')==fingerprint({k:v for k,v in txn.items() if k!='integrity_sha256'}),
            'Transaction integrity mismatch')
    require(txn['plan_fingerprint']==PLAN and txn['selected']==list(SELECTED), 'Transaction plan/selection mismatch')
    require(sha(encoded(txn['after_state']))==txn['after_sha256'], 'Transaction after-state digest mismatch')
    before_raw=read_bytes(run/TAG/'before_state.json')
    require(sha(before_raw)==txn['before_sha256'] and json.loads(before_raw)==txn['before_state'], 'Transaction before-state mismatch')
    expected_id=sha((PLAN+txn['before_sha256']+TAG).encode())
    require(txn['transaction_id']==expected_id, 'Transaction identity mismatch')
    expected_tasks=build_after_state(txn['before_state'],expected_id)['tasks']
    require(txn['after_state']['tasks']==expected_tasks, 'Transaction modifies unexpected tasks')
    for key,value in txn['before_state'].items():
        if key not in {'tasks','updated_at','last_error','capacity','tail_resume'}:
            require(txn['after_state'].get(key)==value, 'Transaction changes a frozen queue field: '+key)
    require(set(txn['after_state'])<=set(txn['before_state'])|{'tail_resume'}, 'Unexpected queue fields in transaction')
    expected_paths=[]
    for key in SELECTED:
        for label,relative in (('job',Path('jobs')/key),('marker',Path('rolling_markers')/(key+'.json')),('script',Path('rolling_slurm')/(key+'.sh'))):
            source=run/relative
            target=run/TAG/'archive'/key/('attempt-'+str(txn['before_state']['tasks'][key]['attempts']))/label
            expected_paths.append((str(source),str(target)))
    require([(r['source'],r['archive']) for r in txn['archives']]==expected_paths, 'Unexpected archive selection/path')


def publish_transaction(run,txn,before_raw,fault=None):
    """Publish a complete preparation atomically; interrupted staging is never reused."""
    run=Path(run); folder=run/TAG
    require(not folder.exists() and not folder.is_symlink(), 'Tail preparation already exists; inspect rather than overwrite')
    staging=Path(tempfile.mkdtemp(prefix=TAG+'-building-',dir=run))
    atomic_bytes(staging/'before_state.json',before_raw)
    if fault: fault('before_state',None)
    txn['integrity_sha256']=fingerprint({k:v for k,v in txn.items() if k!='integrity_sha256'})
    atomic_json(staging/'transaction.json',txn)
    if fault: fault('transaction',None)
    os.rename(staging,folder)
    directory=os.open(run,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)


def preserved_guard(before, current):
    for key,task in before['tasks'].items():
        if task['status']=='PASS' or key in SCIENTIFIC_FAILURES:
            require(current['tasks'].get(key)==task, 'Preserved task changed: '+key)


def controller(run, action):
    argv=[sys.executable,'-I','-B','-u',str(run/'repair/source/scripts/repair_ics.py'),action,'--run-dir',str(run)]
    result=subprocess.run(argv,check=False,timeout=180)
    require(result.returncode==0,'Existing repair controller '+action+' failed; no blind retry')


def progress(message):
    print('[tail recovery] '+message,file=sys.stderr,flush=True)


def resume(run, audit_only=False):
    progress('核验冻结源码、计划及策略。')
    run=Path(run).absolute(); plan=verify_static(run); folder=run/TAG
    applied=folder/'applied.json'; txn_path=folder/'transaction.json'
    if audit_only:
        progress('读取实时 Slurm 队列及完整作业历史。')
        state=load(run/'rolling_state.json'); queue,history=scheduler_evidence(state)
        progress('核对 1426 个成功任务标记及 10 个选中任务身份。')
        proof=audit_state(plan,state,queue,history,run)
        progress('核对 10 个任务的 episode 范围及 155 个已有预测 hash。')
        proof['episodes']=inspect_selected_episodes(plan,state,run)
        print(json.dumps({k:v for k,v in proof.items() if k!='history'},indent=2));return
    with lock(run/(TAG+'.lock')):
        if applied.exists():
            txn=load(txn_path); saved=load(applied)
            validate_transaction(run,txn)
            require(saved['transaction_id']==txn['transaction_id'] and txn['selected']==list(SELECTED), 'Applied transaction identity mismatch')
            preserved_guard(txn['before_state'],load(run/'rolling_state.json'))
            print('Tail recovery already applied; task states will not be reset.',flush=True)
            progress('恢复原管理器；已应用事务不会再次重置任务。')
            controller(run,'activate');return
        if not txn_path.exists():
            # Do all expensive read-only checks before requesting any pause.
            progress('读取实时 Slurm 队列及完整作业历史。')
            state=load(run/'rolling_state.json'); queue,history=scheduler_evidence(state)
            progress('核对 1426 个成功任务标记及 10 个选中任务身份。')
            proof=audit_state(plan,state,queue,history,run)
            progress('核对 10 个任务的 episode 范围及 155 个已有预测 hash。')
            episodes=inspect_selected_episodes(plan,state,run)
            print(json.dumps({'review':'READY','retry_tasks':list(SELECTED),'dependent_formal_tasks':14,
                              'preserved_pass':1426,'preserved_scientific_errors':4,
                              'selected_episode_counts':{r['task']:r['counts'] for r in episodes}},indent=2),flush=True)
        progress('请求原管理器安全暂停并获取队列锁；现有作业不被取消。')
        controller(run,'pause')
        with lock(run/'manager.lock'),lock(run/'rolling.lock'):
            progress('锁内复核最新 Slurm、状态与结果证据。')
            plan=verify_static(run)
            if txn_path.exists():
                txn=load(txn_path)
                validate_transaction(run,txn)
                require(txn['selected']==list(SELECTED) and txn['plan_fingerprint']==PLAN, 'Transaction selection/plan mismatch')
                current=load(run/'rolling_state.json'); preserved_guard(txn['before_state'],current)
                queue,history=scheduler_evidence(txn['before_state'])
                require(not queue, 'New queued/running jobs appeared during transaction')
                for jid,row in txn['history'].items():
                    require(history.get(jid)==row, 'Accounting changed during transaction: '+jid)
                for key in UNCERTAIN:
                    require(not any(r['job_name']==txn['before_state']['tasks'][key]['job_name'] for r in history.values()), 'Uncertain attempt appeared during transaction')
                require(inspect_selected_episodes(plan,txn['before_state'],run)==txn['episode_inspection'],
                        'Selected episode evidence changed after preparation')
                check_transaction_evidence(run,txn)
            else:
                before_raw=read_bytes(run/'rolling_state.json'); state=json.loads(before_raw)
                queue,history=scheduler_evidence(state)
                proof=audit_state(plan,state,queue,history,run)
                episodes=inspect_selected_episodes(plan,state,run)
                transaction_id=sha((PLAN+sha(before_raw)+TAG).encode())
                after=build_after_state(state,transaction_id)
                txn={'schema_version':1,'plan_fingerprint':PLAN,'transaction_id':transaction_id,'selected':list(SELECTED),
                     'created_at':time.time(),'before_sha256':sha(before_raw),'after_sha256':sha(encoded(after)),
                     'before_state':state,'after_state':after,'history':history,'episode_inspection':episodes,
                     'archives':archive_selection(run,state),'proof':{k:v for k,v in proof.items() if k!='history'}}
                publish_transaction(run,txn,before_raw)
            progress('归档选中任务的旧执行元数据并原子更新队列。')
            apply_transaction(run,txn)
            print('Applied: 10 audited tasks reset for new attempts; 4 scientific failures retained.',flush=True)
        progress('恢复原管理器，沿用 60 个在途任务窗口。')
        controller(run,'activate')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('audit','resume'))
    parser.add_argument('--run-dir',required=True,type=Path)
    args=parser.parse_args()
    try:resume(args.run_dir,audit_only=args.command=='audit')
    except (OSError,ValueError,RuntimeError,KeyError,subprocess.SubprocessError) as exc:
        print('Tail recovery refused: '+str(exc),file=sys.stderr)
        print('No successful result is removed. Keep all evidence and return this error; do not run old repair prepare/apply.',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
