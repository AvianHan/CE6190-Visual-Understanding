"""Audited same-run SUIM data recovery; original model workers remain frozen.

Reuses the queue, sealed intents, snapshot checks, tail-recovery evidence moves,
and detached batch2 watcher. Only the failed SUIM data attempt and its never-run
descendants are reset; the old empty-scope collector is archived and reissued.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import time

from . import batch2, rolling_queue as rq
from .orchestration import fingerprint, load_json
from scripts.queue_ics import acquire_lock, verify_snapshot
from scripts.resume_ics_tail import tree_fingerprint, read_bytes, safe_path, atomic_bytes

TAG = 'suim-repair-20261116'
EXPECTED_PLAN = '6f391f99a754b3d02f14b5f6f8efb4bd265ad809805984eb6258b59f4187c7ce'
EXPECTED_JOB = '206900'


def require(condition, message):
    if not condition: raise ValueError(message)


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def checked(run, expected_plan=EXPECTED_PLAN):
    run = Path(run).resolve()
    meta = batch2.load_batch(run)
    verify_snapshot(run/'source')
    plan = rq._plan(run/'scopes/suim/plan.json')
    require(plan['fingerprint'] == expected_plan, 'Unexpected SUIM plan fingerprint')
    require(plan['batch2']['scope']=='suim' and Path(plan['project_root'])==run/'source', 'Unexpected SUIM source binding')
    return meta, plan


def load_policy(path):
    path=Path(path).resolve(); policy=load_json(path)
    require(policy.get('fingerprint') == fingerprint({k:v for k,v in policy.items() if k!='fingerprint'}), 'Recovery policy integrity mismatch')
    run=Path(policy['parent_run'])
    require(path==run/TAG/'policy.json', 'Unexpected recovery policy path')
    require(Path(policy['controller_source'])==run/TAG/'source', 'Unexpected recovery controller source')
    _, plan=checked(run,policy['plan_fingerprint'])
    require(batch2.digest(run/'source/snapshot.json')==policy['original_source_snapshot_sha256'], 'Original scientific source identity changed')
    source=Path(policy['controller_source']); verify_snapshot(source)
    require(batch2.digest(source/'snapshot.json')==policy['controller_snapshot_sha256'], 'Recovery controller identity changed')
    return policy,plan


def audit_failure(run, plan, expected_job=EXPECTED_JOB):
    """Allow other scopes' live work, but require SUIM itself fully quiescent."""
    run=Path(run); scope=Path(plan['run_dir']); state=rq._read_state(plan); tasks=state['tasks']
    data=tasks['data']; collect=tasks['collect']
    require(state['plan_fingerprint']==plan['fingerprint'], 'Queue/plan mismatch')
    require(len(tasks)==172 and len(plan['tasks'])==156 and len(plan['smoke'])==13, 'SUIM matrix changed')
    require(data['status']=='FAILED' and str(data.get('job_id'))==str(expected_job) and data['attempts']==1, 'Failed SUIM data attempt changed')
    require(collect['status']=='PASS' and collect.get('job_id'), 'Expected the completed empty-scope collector')
    blocked=[k for k in tasks if k not in {'data','collect'}]
    for key in blocked:
        item=tasks[key]
        require(item['status']=='BLOCKED' and item['attempts']==0 and not item.get('job_id'), 'A SUIM model/preflight was already attempted: '+key)
        require(not (scope/'jobs'/key).exists() and not rq._marker_path(plan,key).exists(), 'Unexpected previous worker evidence: '+key)
    for directory in ('results','smoke'):
        path=scope/directory
        require(not path.is_symlink(), 'Refusing symlink model results')
        require(not path.exists() or not any(p.is_file() or p.is_symlink() for p in path.rglob('*')), 'SUIM model results exist; refusing recovery: '+directory)
    sealed_evidence={}
    for key in ('data','collect'):
        task=tasks[key]
        intent_path=scope/'batch2_intents'/key/('attempt-%05d.json' % task['attempts'])
        require(task.get('batch2_intent')==str(intent_path),'Sealed intent path mismatch: '+key)
        raw=read_bytes(safe_path(scope,intent_path))
        require(sha(raw)==task.get('batch2_intent_sha256'),'Sealed intent SHA mismatch: '+key)
        intent=json.loads(raw)
        require(intent.get('attempt')==task['attempts'] and intent.get('plan_fingerprint')==plan['fingerprint']
                and intent.get('plan_path')==str(scope/'plan.json') and intent.get('worker_source')==plan['project_root']
                and intent.get('source_snapshot_sha256')==batch2.digest(Path(plan['project_root'])/'snapshot.json'),
                'Sealed intent plan/source/attempt mismatch: '+key)
        require(all(intent.get('task',{}).get(field)==task.get(field) for field in
                    ('key','kind','resource','index','dependencies','job_name')),'Sealed task identity mismatch: '+key)
        claim_path=intent_path.with_suffix('.claim.json')
        claim_raw=read_bytes(safe_path(scope,claim_path));claim=json.loads(claim_raw)
        require(all(claim.get(field)==value for field,value in {'slurm_job_id':str(task['job_id']),
                    'job_name':task['job_name'],'intent_sha256':task['batch2_intent_sha256']}.items()),
                'Original worker claim identity mismatch: '+key)
        sealed_evidence[str(intent_path)]=sha(raw);sealed_evidence[str(claim_path)]=sha(claim_raw)
    marker=load_json(rq._marker_path(plan,'data'))
    require(all(marker.get(k)==v for k,v in {'status':'FAIL','key':'data','plan_fingerprint':plan['fingerprint'],
                'slurm_job_id':str(expected_job),'error_type':'ValueError'}.items()), 'Failed data marker identity changed')
    require('f_r_1289_.jpg' in marker.get('error','') and 'dimension' in marker.get('error','').lower(), 'Failure is not the audited SUIM geometry error')
    require(rq._marker_problem(plan,collect) is None, 'Old collector completion marker is invalid')
    require(not (scope/'data/data_receipt.json').exists(), 'A finalized SUIM data receipt already exists')
    queue,history=rq._scheduler_snapshot(state)
    prefix='icsq-'+plan['fingerprint'][:16]+'-'
    require(not any(row['job_name'].startswith(prefix) for row in queue.values()), 'SUIM still has queued/running jobs')
    expected={str(data['job_id']):('FAILED','1:0'),str(collect['job_id']):('COMPLETED','0:0')}
    require(len(expected)==2,'Data and collector cannot share a job ID')
    evidence={}
    for item in (data,collect):
        jid=str(item['job_id']); row=history.get(jid)
        require(row and row['job_name']==item['job_name'] and row['qos']==state['policy']['qos'], 'Slurm identity mismatch: '+jid)
        require((row['state'],row['exit_code'])==expected[jid], 'Slurm outcome changed: '+jid)
        require({other for other,record in history.items() if record['job_name']==item['job_name']}=={jid},
                'Duplicate historical allocation for sealed SUIM task: '+item['key'])
        evidence[jid]=row
    known_names={tasks['data']['job_name'],tasks['collect']['job_name']}
    require(not any(row['job_name'].startswith(prefix) and row['job_name'] not in known_names for row in history.values()), 'Unrecorded SUIM allocations need review')
    return state,{'history':evidence,'sealed_evidence':sealed_evidence,'failed_marker':marker,'unblocked':blocked,'queried_at':time.time()}


def prepare_policy(run, source, expected_plan=EXPECTED_PLAN, expected_job=EXPECTED_JOB):
    run,source=Path(run).resolve(),Path(source).resolve()
    _,plan=checked(run,expected_plan)
    from .batch2_data import SUIM_REPAIR_IMPLEMENTATION_ID
    require(SUIM_REPAIR_IMPLEMENTATION_ID=='suim-strict-geometry-filter-v1', 'Unreviewed SUIM data repair implementation')
    require(source==run/TAG/'source','Recovery source is not isolated under the original run')
    verify_snapshot(source)
    path=run/TAG/'policy.json'
    if path.exists():
        policy,_=load_policy(path)
        require(policy['failed_job']==str(expected_job),'Recovery failed-job selection changed')
        return policy
    _,proof=audit_failure(run,plan,expected_job)
    value={'schema_version':1,'parent_run':str(run),'scope_run':plan['run_dir'],
        'plan_fingerprint':plan['fingerprint'],'failed_job':str(expected_job),
        'controller_source':str(source),'controller_snapshot_sha256':batch2.digest(source/'snapshot.json'),
        'original_source_snapshot_sha256':batch2.digest(run/'source/snapshot.json'),
        'data_repair_implementation':SUIM_REPAIR_IMPLEMENTATION_ID,
        'worker_policy':'corrected_SUIM_data_and_preflight_only_original_model_and_collector_source',
        'selected':['data','collect'],'unblock_count':len(proof['unblocked']),
        'retry_policy':'one_audited_attempt_no_model_retries'}
    value['fingerprint']=fingerprint(value)
    batch2.immutable_json(path,value)
    return value


def after_state(before,policy):
    after=copy.deepcopy(before)
    for key,item in after['tasks'].items():
        item.setdefault('attempt_history',[]).append({k:copy.deepcopy(v) for k,v in item.items() if k!='attempt_history'})
        for field in ('job_id','submitted_at','attempt_started_at','ended_at','scheduler_state','exit_code',
                      'error','terminal_seen_at','retry_after','recovered_at','batch2_intent','batch2_intent_sha256'):
            item.pop(field,None)
        item['status']='PENDING'
        if key in ('data','collect'):
            item['job_name'] += '-u'+policy['fingerprint'][:8]+'a'+str(item['attempts']+1)
        if key in ('data','preflight'):
            item['batch2_data_repair']={'policy_path':str(Path(policy['parent_run'])/TAG/'policy.json'),
                                       'policy_fingerprint':policy['fingerprint']}
    after.update(updated_at=before['updated_at'],last_error=None,capacity=None,
                 batch2_suim_recovery={'policy_fingerprint':policy['fingerprint'],
                     'reset_failed_data':1,'reset_empty_collector':1,'unblocked':170,
                     'successful_models_reset':0})
    return after


def archive_selection(scope,folder):
    relatives=[Path('rolling_markers')/(key+'.json') for key in ('data','collect')]
    relatives += [Path('rolling_slurm')/(key+'.sh') for key in ('data','collect')]
    relatives += [Path('jobs')/key for key in ('data','collect')]
    relatives += [Path(p) for p in ('data','execution_summary.json','analysis','analysis.log','batch2_analysis','batch2_analysis.log')]
    records=[]
    for relative in relatives:
        source=safe_path(scope,scope/relative)
        records.append({'source':str(source),'archive':str(folder/'archive'/relative),
                        'fingerprint':tree_fingerprint(source)})
    return records


def validate_transaction(run,txn,policy):
    require(txn.get('integrity_sha256')==fingerprint({k:v for k,v in txn.items() if k!='integrity_sha256'}),'Recovery transaction integrity mismatch')
    require(txn['policy_fingerprint']==policy['fingerprint'],'Recovery transaction policy mismatch')
    for path,wanted in txn['proof']['sealed_evidence'].items():
        require(sha(read_bytes(safe_path(Path(policy['scope_run']),Path(path))))==wanted,'Original sealed evidence changed')
    require(sha(encoded(txn['before_state']))==txn['before_sha256'],'Before-state checksum mismatch')
    require(txn['after_state']==after_state(txn['before_state'],policy),'Transaction changes unexpected queue fields')
    require(sha(encoded(txn['after_state']))==txn['after_sha256'],'After-state checksum mismatch')
    scope=Path(policy['scope_run']); folder=Path(run)/TAG
    expected=archive_selection_paths(scope,folder)
    require([(x['source'],x['archive']) for x in txn['archives']]==expected,'Unexpected recovery archive paths')


def archive_selection_paths(scope,folder):
    # Compute paths without reading evidence that may already have moved.
    relatives=[Path('rolling_markers')/(k+'.json') for k in ('data','collect')]
    relatives += [Path('rolling_slurm')/(k+'.sh') for k in ('data','collect')]
    relatives += [Path('jobs')/k for k in ('data','collect')]
    relatives += [Path(p) for p in ('data','execution_summary.json','analysis','analysis.log','batch2_analysis','batch2_analysis.log')]
    return [(str(scope/p),str(folder/'archive'/p)) for p in relatives]


def apply_transaction(run,policy,fault=None):
    """Roll forward a prepared transaction; caller owns parent and child locks."""
    run=Path(run); folder=run/TAG; scope=Path(policy['scope_run'])
    path=folder/'transaction.json'; applied=folder/'applied.json'
    if applied.exists():
        record=load_json(applied);txn=load_json(path)
        validate_transaction(run,txn,policy)
        require(all(record.get(field)==value for field,value in {
            'policy_fingerprint':policy['fingerprint'],'before_sha256':txn['before_sha256'],
            'after_sha256':txn['after_sha256']}.items()),'Applied recovery receipt identity changed')
        for row in txn['archives']:
            target=safe_path(run,Path(row['archive']))
            require(tree_fingerprint(target)==row['fingerprint'],'Applied archive evidence changed')
        return record
    if path.exists(): txn=load_json(path)
    else:
        plan=rq._plan(scope/'plan.json');before,proof=audit_failure(run,plan,policy['failed_job'])
        after=after_state(before,policy)
        txn={'schema_version':1,'policy_fingerprint':policy['fingerprint'],
            'before_state':before,'after_state':after,'before_sha256':sha(encoded(before)),
            'after_sha256':sha(encoded(after)),'proof':proof,
            'archives':archive_selection(scope,folder),
            'preserved_other_scope_states':{s:batch2.digest(run/'scopes'/s/'rolling_state.json') for s in ('core','isic')}}
        txn['integrity_sha256']=fingerprint(txn)
        rq._atomic_json(path,txn)
        if fault: fault('prepared',None)
    validate_transaction(run,txn,policy)
    current=load_json(scope/rq.STATE_NAME)
    require(current==txn['before_state'] or current==txn['after_state'],'SUIM queue changed outside the interrupted transaction')
    for other,wanted in txn['preserved_other_scope_states'].items():
        require(batch2.digest(run/'scopes'/other/rq.STATE_NAME)==wanted,'Other scope queue changed during transaction: '+other)
    if current==txn['before_state']:
        # Job success/failure cannot be inferred from disappearance; recheck known
        # outcomes before each new/resumed mutation sequence.
        queue,history=rq._scheduler_snapshot(txn['before_state'])
        prefix='icsq-'+policy['plan_fingerprint'][:16]+'-'
        require(not any(r['job_name'].startswith(prefix) for r in queue.values()),'SUIM allocation appeared during repair')
        for jid,row in txn['proof']['history'].items():
            require(history.get(jid)==row,'Historical Slurm identity changed during repair')
    for index,row in enumerate(txn['archives']):
        source=safe_path(scope,Path(row['source']));target=safe_path(run,Path(row['archive']));wanted=row['fingerprint']
        a,b=tree_fingerprint(source),tree_fingerprint(target)
        if wanted is None: require(a is None and b is None,'An absent archive target appeared')
        elif a is not None:
            require(a==wanted and b is None,'Archive source changed or duplicate archive exists')
            target.parent.mkdir(parents=True,exist_ok=True); os.replace(source,target)
        else: require(b==wanted,'Archived evidence missing or changed')
        if fault: fault('archive',index)
    if current!=txn['after_state']: atomic_bytes(scope/rq.STATE_NAME,encoded(txn['after_state']))
    if fault: fault('state',None)
    receipt={'policy_fingerprint':policy['fingerprint'],'applied_at':time.time(),
             'before_sha256':txn['before_sha256'],'after_sha256':txn['after_sha256'],
             'successful_models_reset':0,'old_claims_and_logs':'preserved_in_place'}
    rq._atomic_json(applied,receipt)
    return receipt


def make_data_worker_launch(plan_path,plan,task):
    require(plan.get('batch2',{}).get('scope')=='suim' and task['kind'] in ('data','preflight'), 'Repair dispatch is restricted to SUIM data/preflight')
    selection=task['batch2_data_repair']; policy,_=load_policy(selection['policy_path'])
    require(selection['policy_fingerprint']==policy['fingerprint'],'Task recovery selection mismatch')
    command=batch2.make_worker_launch(plan_path,plan,task,_allow_data_repair=False)
    command[command.index('worker')-1]=str(Path(policy['controller_source'])/'scripts/recover_batch2_suim.py')
    command.extend(['--policy',selection['policy_path'],'--policy-sha256',batch2.digest(selection['policy_path'])])
    return command


def execute_data_worker(intent,expected_sha,policy_path,expected_policy_sha):
    require(batch2.digest(policy_path)==expected_policy_sha,'Recovery worker policy SHA mismatch')
    policy,plan=load_policy(policy_path)
    require(Path(__file__).resolve().parents[1]==Path(policy['controller_source']),'Recovery worker must use its isolated frozen source')
    sealed=load_json(intent)
    require(sealed['plan_fingerprint']==plan['fingerprint'] and sealed['task']['kind'] in ('data','preflight'), 'Unexpected recovery worker scope/kind')
    require(Path(sealed['plan_path'])==Path(policy['scope_run'])/'plan.json','Recovery worker plan path mismatch')
    from .batch2_data import prepare_scope,verify_scope,SUIM_REPAIR_IMPLEMENTATION_ID
    require(SUIM_REPAIR_IMPLEMENTATION_ID==policy['data_repair_implementation'],'Recovery data implementation differs')
    batch2.execute_intent(intent,expected_sha,data_preparer=prepare_scope,data_verifier=verify_scope)


def verified_manager_pid(run,record):
    """Same ownership/cmdline/start-time/fd proof as scale_ics, for batch2 watchers."""
    run=Path(run).resolve()
    require(record.get('hostname')==socket.gethostname(),'Run recovery on the manager host')
    pid=record.get('pid'); require(type(pid) is int and pid>1,'Invalid manager PID')
    proc=Path('/proc')/str(pid)
    require(proc.stat().st_uid==os.getuid(),'Manager process has another owner')
    argv=[v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
    allowed={str(run/'source/scripts/batch2_ics.py'),str(run/TAG/'source/scripts/recover_batch2_suim.py')}
    require(len([v for v in argv if v in allowed])==1 and 'watch' in argv and '--run-dir' in argv,'Process is not the verified batch2 watcher')
    index=argv.index('--run-dir'); require(index+1<len(argv) and Path(argv[index+1]).resolve()==run,'Manager run argument differs')
    fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    boot=next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
    started=boot+int(fields[19])/os.sysconf('SC_CLK_TCK')
    require(abs(started-float(record['started_at']))<=30,'Manager PID start-time differs')
    expected=(run/'manager.lock').stat(); held=False
    for fd in (proc/'fd').iterdir():
        try:
            item=fd.stat();held=held or (item.st_dev,item.st_ino)==(expected.st_dev,expected.st_ino)
        except FileNotFoundError: pass
    require(held,'Manager does not hold this run lock descriptor')
    return pid


def manager_handover_lock(run):
    """Pause verified submitter only; existing Slurm jobs continue untouched."""
    run=Path(run); lock=acquire_lock(run/'manager.lock')
    if lock is not None:return lock
    record=load_json(run/'manager.json');pid=verified_manager_pid(run,record)
    require(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'pidfd unavailable; no unsafe PID signal sent')
    fd=os.pidfd_open(pid)
    try:
        verified_manager_pid(run,record);signal.pidfd_send_signal(fd,signal.SIGTERM)
    finally:os.close(fd)
    print('Verified manager pause requested; existing Slurm jobs are preserved.',flush=True)
    deadline=time.monotonic()+90;notice=time.monotonic()+10
    while lock is None and time.monotonic()<deadline:
        time.sleep(.1);lock=acquire_lock(run/'manager.lock')
        if time.monotonic()>=notice:
            print('Waiting for current submission acknowledgement and manager lock release...',flush=True);notice+=10
    if lock is None:raise RuntimeError('Manager has not released its lock; repeat the same recovery command after inspection')
    return lock
