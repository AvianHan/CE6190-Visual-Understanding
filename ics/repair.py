"""Audited same-run infrastructure repair without changing scientific worker sources."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

from .orchestration import fingerprint, load_json
from .rolling_queue import (_atomic_json, _command, _lock, _plan, _read_state,
                            _reconcile, SLURM_TERMINAL)

BAD_NODE = 'A484652X3B10601'
TAG = 'cuda-statefix-20261113'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_repair_policy(plan):
    run = Path(plan['run_dir']).resolve()
    policy = load_json(run / 'repair/policy.json')
    if fingerprint({k:v for k,v in policy.items() if k!='fingerprint'}) != policy.get('fingerprint'):
        raise ValueError('Repair policy fingerprint mismatch')
    if policy.get('plan_fingerprint') != plan['fingerprint'] or policy.get('worker_source') != plan['project_root']:
        raise ValueError('Repair policy changed the immutable plan/worker identity')
    if Path(policy['controller_source']).resolve() != run / 'repair/source':
        raise ValueError('Unexpected repair controller source')
    if policy.get('exclude_nodes') != [BAD_NODE]:
        raise ValueError('Unexpected node-exclusion scope')
    if digest(run / 'source/snapshot.json') != policy['worker_snapshot_sha256']:
        raise ValueError('Original worker snapshot identity changed')
    return policy


def effective_exclusions(profile, policy):
    result = dict(profile)
    if result.get('gpus', 0):
        prior = [v for v in str(result.get('exclude', '')).split(',') if v]
        result['exclude'] = ','.join(dict.fromkeys(prior + policy['exclude_nodes']))
    return result


def immutable_json(path, value):
    """A sealed record is never changed; a repeated identical write is idempotent."""
    path = Path(path)
    data = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2)+'\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        try:
            os.link(temporary,path)
            directory=os.open(path.parent,os.O_RDONLY)
            try:os.fsync(directory)
            finally:os.close(directory)
        except FileExistsError:
            if path.read_bytes()!=data:raise ValueError('Immutable record differs: '+str(path))
    finally:
        os.unlink(temporary)
    return hashlib.sha256(data).hexdigest()


def make_worker_launch(plan_path, plan, state, task):
    policy = load_repair_policy(plan)
    attempt = int(task['attempts']) + 1
    key = task['key']
    token = hashlib.sha256(key.encode()).hexdigest()[:16]
    name = f"icsq-{plan['fingerprint'][:16]}-{token}-r{policy['fingerprint'][:6]}a{attempt}"
    sealed_task = {k:copy.deepcopy(task[k]) for k in ('key','kind','resource','index','dependencies')}
    sealed_task.update(status='SUBMITTING', attempts=attempt, job_name=name)
    intent = {'schema_version':1, 'plan_path':str(Path(plan_path).resolve()),
        'plan_fingerprint':plan['fingerprint'], 'repair_policy_fingerprint':policy['fingerprint'],
        'worker_source':plan['project_root'], 'worker_snapshot_sha256':policy['worker_snapshot_sha256'],
        'task':sealed_task, 'queue_policy':state['policy']}
    path = Path(plan['run_dir'])/'repair/intents'/key/f'attempt-{attempt}.json'
    sha = immutable_json(path,intent)
    task['job_name'] = name
    task['repair_attempt'] = {'policy_fingerprint':policy['fingerprint'], 'intent_path':str(path),
                              'intent_sha256':sha, 'attempt_number':attempt}
    return [plan['cluster'].get('launcher_python','python3'),
            str(Path(policy['controller_source'])/'scripts/repair_worker.py'),
            '--intent',str(path),'--intent-sha256',sha]


def scheduler_evidence(state):
    """Read current queue and exact known allocations, retaining node identity."""
    queue = {}
    text = _command(['squeue','--noheader','--all','--array','--user',state['policy']['user'],
                     '--format','%i|%j|%T|%q'])
    for line in text.splitlines():
        if not line.strip(): continue
        fields = [x.strip() for x in line.split('|')]
        if len(fields)!=4 or not re.fullmatch(r'\d+(?:_\d+)?(?:\+\d+)?',fields[0]):
            raise ValueError('Malformed or unexpanded squeue record')
        jid,name,status,qos=fields
        if jid in queue:raise ValueError('Duplicate queue allocation record')
        queue[jid]={'job_id':jid,'job_name':name,'state':status,'qos':qos}
    ids = sorted({str(t['job_id']) for t in state['tasks'].values() if t.get('job_id')},key=int)
    start=dt.datetime.fromtimestamp(state['created_at']-120).strftime('%Y-%m-%dT%H:%M:%S')
    history={}
    for offset in range(0,len(ids),200):
        batch=ids[offset:offset+200]
        output=_command(['sacct','--noheader','--parsable2','--allocations','--jobs',','.join(batch),
                         '--starttime',start,'--format','JobIDRaw,JobName%128,State%40,ExitCode,QOS,NodeList'])
        for line in output.splitlines():
            if not line.strip():continue
            fields=[x.strip() for x in line.split('|')]
            if len(fields)==7 and fields[-1]=='':fields.pop()
            if len(fields)!=6 or fields[0] not in batch:raise ValueError('Unexpected exact-job accounting record')
            jid,name,status,code,qos,node=fields
            status=status.split()[0].rstrip('+')
            row={'job_id':jid,'job_name':name,'state':status,'exit_code':code,'qos':qos,'node':node}
            if jid in history:raise ValueError('Duplicate allocation accounting')
            history[jid]=row
    return queue,history


def failure_category(run,task,node):
    key,jid=task['key'],str(task['job_id'])
    folder=run/'jobs'/key
    evidence={}
    for label,path in [('runner',folder/'runner.log'),('vllm',folder/'vllm.log'),
                       ('slurm',run/'logs'/f'{key}-{jid}.log')]:
        try:
            with path.open('rb') as f:
                f.seek(0,2);size=f.tell();f.seek(max(0,size-2*2**20))
                evidence[label]=f.read().decode(errors='replace')
        except FileNotFoundError: evidence[label]=''
    if ('CUDA is unavailable; run inside an allocated GPU job' in evidence['runner']
            or 'No CUDA GPUs are available' in evidence['vllm']):
        if node != BAD_NODE:
            raise ValueError(f'CUDA failure on another node requires review: {key} {node}')
        return 'CUDA_UNAVAILABLE_ON_EXCLUDED_NODE'
    if ('[Errno 2] No such file or directory:' in evidence['slurm']
            and str(run/'rolling_state.json') in evidence['slurm']):
        return 'MUTABLE_STATE_FILE_NOT_VISIBLE'
    raise ValueError('Failed task has an unreviewed cause: '+key)


def audit_run(plan,state):
    run=Path(plan['run_dir'])
    queue,history=scheduler_evidence(state)
    known={str(t['job_id']) for t in state['tasks'].values() if t.get('job_id')}
    prefix='icsq-'+plan['fingerprint'][:16]+'-'
    if any(j in known or r['job_name'].startswith(prefix) for j,r in queue.items()):
        raise RuntimeError('This run still has queued/running jobs; no repair mutation allowed')
    reconciled=copy.deepcopy(state)
    for task in reconciled['tasks'].values():
        if task.get('job_id'):
            jid=str(task['job_id'])
            if jid not in history or history[jid]['state'] not in SLURM_TERMINAL:
                raise RuntimeError('Missing/nonterminal exact allocation evidence: '+jid)
            task['status']='SUBMITTED'  # Recheck even previously saved PASS against exact accounting+markers.
    _reconcile(plan,reconciled,queue,history,time.time())
    # Validation must not rewrite timestamps/history on previously confirmed PASS tasks.
    for key,original in state['tasks'].items():
        if original['status']=='PASS' and reconciled['tasks'][key]['status']=='PASS':
            reconciled['tasks'][key]=copy.deepcopy(original)
    failed=[];blocked=[];preserved=[];categories={}
    for key,task in reconciled['tasks'].items():
        if task['status']=='PASS':preserved.append(key)
        elif task['status']=='FAILED':
            if task['kind'] not in {'smoke','experiment'}:raise ValueError('Unexpected infrastructure stage failure')
            categories[key]=failure_category(run,task,history[str(task['job_id'])]['node']);failed.append(key)
        elif task['status']=='BLOCKED':
            if task.get('job_id'):raise ValueError('Blocked task unexpectedly has an allocation')
            blocked.append(key)
        elif task['status'] not in {'PENDING','WAITING_QUOTA'}:
            raise ValueError('Unresolved task state prevents repair: '+key+' '+task['status'])
    return reconciled,{'history':history,'failed':failed,'blocked':blocked,'preserved':preserved,
                       'failure_categories':categories,'queried_at':time.time()}


def inspect_episode_errors(plan,keys,seconds=600):
    """Inspect existing direct episode directories only, never masks or whole result trees."""
    definitions={t['id']:t for t in plan['smoke']+plan['tasks']}
    outputs=sorted({definitions[k]['output'] for k in keys})
    deadline=time.monotonic()+seconds
    rows=[];errors=[]
    for output in outputs:
        path=Path(output)/'episodes'
        print('Checking existing episode records: '+str(path),flush=True)
        count=0
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    if time.monotonic()>deadline:raise RuntimeError('Episode inspection incomplete; repair remains unapplied')
                    if not entry.name.endswith('.json'):continue
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):raise ValueError('Unexpected episode path')
                    row=load_json(Path(entry.path));count+=1
                    if row.get('status') not in {'ok','error'}:raise ValueError('Invalid episode status: '+entry.path)
                    if row['status']=='error':errors.append({'path':entry.path,'episode_id':row.get('episode_id'),'error':row.get('error')})
        except FileNotFoundError:
            if path.exists():raise
        rows.append({'output':output,'existing_records':count})
    return {'outputs':rows,'error_records':errors,'inspected_at':time.time()}


def reset_state(reconciled,proof,policy):
    state=copy.deepcopy(reconciled)
    for key in proof['failed']:
        old=state['tasks'][key]
        history=old.setdefault('attempt_history',[])
        history.append({k:copy.deepcopy(v) for k,v in old.items() if k!='attempt_history'})
        for k in ('job_id','submitted_at','attempt_started_at','ended_at','scheduler_state','exit_code',
                  'error','terminal_seen_at','retry_after','repair_attempt'):
            old.pop(k,None)
        old['status']='PENDING'
    for key in proof['blocked']:
        state['tasks'][key]['status']='PENDING';state['tasks'][key].pop('error',None)
    state.pop('repair_circuit_breaker',None)
    state.update(last_error=None,capacity=None,updated_at=time.time(),
                 repair={'policy_fingerprint':policy['fingerprint'],'applied_at':time.time(),
                         'preserved':len(proof['preserved']),'retried':len(proof['failed']),
                         'unblocked':len(proof['blocked'])})
    return state


def archive_failed(run,proof):
    """Atomic same-filesystem moves of failed task evidence; results are never moved."""
    run=Path(run)
    records=[]
    for key in proof['failed']:
        jid=proof['history_job_ids'][key]
        base=run/'repair/archives'/key/('job-'+str(jid));base.mkdir(parents=True,exist_ok=True)
        for label,source in [('job',run/'jobs'/key),('marker',run/'rolling_markers'/(key+'.json')),
                             ('script',run/'rolling_slurm'/(key+'.sh'))]:
            dest=base/label
            if source.exists() or source.is_symlink():
                if source.is_symlink():raise ValueError('Refusing to archive a symlink: '+str(source))
                if dest.exists():raise ValueError('Both current and archived evidence exist: '+str(source))
                os.replace(source,dest)
                records.append({'source':str(source),'archive':str(dest),'status':'MOVED'})
            elif dest.exists():records.append({'source':str(source),'archive':str(dest),'status':'ALREADY_ARCHIVED'})
            else:records.append({'source':str(source),'archive':str(dest),'status':'ABSENT'})
    return records
