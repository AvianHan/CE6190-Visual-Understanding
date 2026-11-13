#!/usr/bin/env python3
"""Prepare, apply and start an audited repair of an existing immutable ICS run."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from queue_ics import (acquire_lock,clean_environment,freeze_source,manager_status,
                       print_json,saved_status,verify_snapshot,watcher)
from ics.orchestration import fingerprint,load_json
from ics.rolling_queue import _atomic_json,_lock,_plan,_read_state,_marker_problem
from ics.repair import (BAD_NODE,archive_failed,audit_run,digest,inspect_episode_errors,
                        load_repair_policy,reset_state,scheduler_evidence)
from ics.scheduling import load_policy,partition_evidence


def checked_plan(run):
    plan=_plan(run/'plan.json')
    if Path(plan['run_dir']).resolve()!=run or Path(plan['project_root']).resolve()!=run/'source':
        raise ValueError('Expected the existing run and original frozen source')
    verify_snapshot(run/'source')
    if (run/'scheduling/source').exists():verify_snapshot(run/'scheduling/source')
    return plan


def expected_counts(args,state,proof):
    successful=sum(state['tasks'][k]['kind']=='experiment' for k in proof['preserved'])
    got=(successful,len(proof['failed']),len(proof['blocked']))
    wanted=(args.expected_successful_experiments,args.expected_failed,args.expected_blocked)
    if got!=wanted:raise ValueError(f'Repair coverage changed: observed {got}, expected {wanted}')


def prepare(args):
    run=args.run_dir.resolve();plan=checked_plan(run)
    if plan['fingerprint']!=args.expected_plan_fingerprint:raise ValueError('Unexpected plan fingerprint')
    manager=acquire_lock(run/'manager.lock')
    if manager is None:raise RuntimeError('Stop the existing manager before preparing repair')
    with manager,_lock(run):
        folder=run/'repair';folder.mkdir(exist_ok=True)
        if (folder/'policy.json').exists():
            policy=load_repair_policy(plan);verify_snapshot(Path(policy['controller_source']))
            print_json({'repair':'ALREADY_PREPARED','policy':policy});return 0
        raw_state=load_json(run/'rolling_state.json');before_sha=digest(run/'rolling_state.json')
        reconciled,proof=audit_run(plan,raw_state)
        expected_counts(args,reconciled,proof)
        episodes=inspect_episode_errors(plan,proof['failed'])
        report={'schema_version':1,'plan_fingerprint':plan['fingerprint'],'before_state_sha256':before_sha,
                'proof':proof,'episode_inspection':episodes,'ready':not episodes['error_records']}
        _atomic_json(folder/'prepare.json',report)
        if episodes['error_records']:
            raise ValueError('Existing error episodes need review; --resume would skip them. Repair not prepared.')
        selection=load_json(run/'source/snapshot.json')['selection']
        source=freeze_source(folder,ROOT/selection['suite'],ROOT/selection['cluster'],Path(selection['runtime']))
        scheduling=load_policy(plan)
        policy={'schema_version':1,'created_at':time.time(),'plan_fingerprint':plan['fingerprint'],
                'worker_source':plan['project_root'],'worker_snapshot_sha256':digest(run/'source/snapshot.json'),
                'controller_source':str(source),'exclude_nodes':[BAD_NODE],
                'prepared_state_sha256':before_sha,'scheduling_policy_fingerprint':scheduling['fingerprint'],
                'expected_counts':{'successful_experiments':args.expected_successful_experiments,
                                   'failed':args.expected_failed,'blocked':args.expected_blocked},
                'retry_policy':'missing_episode_only_resume_original_scientific_source',
                'prepared_report_sha256':digest(folder/'prepare.json')}
        policy['fingerprint']=fingerprint(policy)
        _atomic_json(folder/'policy.json',policy)
        print_json({'repair':'PREPARED_NOT_APPLIED','retry_tasks':len(proof['failed']),
                    'preserved_tasks':len(proof['preserved']),'unblock_tasks':len(proof['blocked']),
                    'existing_error_episodes':0,'policy':policy})
    return 0


def same_history_guard(state,expected):
    queue,history=scheduler_evidence(state)
    prefix='icsq-'+state['plan_fingerprint'][:16]+'-'
    if any(row['job_name'].startswith(prefix) for row in queue.values()):raise RuntimeError('Run has live jobs')
    for jid,row in expected.items():
        if history.get(jid)!=row:raise ValueError('Allocation evidence changed during repair transaction: '+jid)


def json_sha(value):
    return hashlib.sha256((json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode()).hexdigest()


def apply(args):
    run=args.run_dir.resolve();plan=checked_plan(run);policy=load_repair_policy(plan)
    manager=acquire_lock(run/'manager.lock')
    if manager is None:raise RuntimeError('Manager must be stopped before repair apply')
    with manager,_lock(run):
        folder=run/'repair';applied=folder/'applied.json';transaction=folder/'transaction.json'
        if applied.exists():
            evidence=load_json(applied)
            if evidence['policy_fingerprint']!=policy['fingerprint']:raise ValueError('Applied repair identity mismatch')
            print_json({'repair':'ALREADY_APPLIED',**saved_status(run)});return 0
        if transaction.exists():
            txn=load_json(transaction)
            if txn['policy_fingerprint']!=policy['fingerprint']:raise ValueError('Transaction policy mismatch')
            if digest(run/'rolling_state.json') not in {txn['before_state_sha256'],txn['after_state_sha256']}:
                raise ValueError('Queue state changed outside the in-progress repair')
            same_history_guard(txn['before_state'],txn['proof']['history'])
            for key in txn['proof']['preserved']:
                if _marker_problem(plan,txn['before_state']['tasks'][key]):
                    raise ValueError('Preserved task marker changed during repair: '+key)
        else:
            if digest(run/'rolling_state.json')!=policy['prepared_state_sha256']:
                raise ValueError('Queue changed after preparation; inspect before applying')
            prepared=load_json(folder/'prepare.json')
            if digest(folder/'prepare.json')!=policy['prepared_report_sha256'] or not prepared['ready']:
                raise ValueError('Prepared review evidence changed or contains error episodes')
            before=_read_state(plan);reconciled,proof=audit_run(plan,before)
            expected=policy['expected_counts']
            actual={'successful_experiments':sum(reconciled['tasks'][k]['kind']=='experiment' for k in proof['preserved']),
                    'failed':len(proof['failed']),'blocked':len(proof['blocked'])}
            if actual!=expected:raise ValueError('Repair coverage changed since prepare')
            proof['history_job_ids']={key:str(reconciled['tasks'][key]['job_id']) for key in proof['failed']}
            after=reset_state(reconciled,proof,policy)
            txn={'schema_version':1,'policy_fingerprint':policy['fingerprint'],'phase':'APPLYING',
                 'before_state_sha256':digest(run/'rolling_state.json'),'after_state_sha256':json_sha(after),
                 'before_state':before,'after_state':after,'proof':proof,'started_at':time.time()}
            _atomic_json(folder/'before_state.json',before)
            _atomic_json(transaction,txn)
        records=archive_failed(run,txn['proof'])
        _atomic_json(folder/'archive_inventory.json',records)
        _atomic_json(run/'rolling_state.json',txn['after_state'])
        _atomic_json(applied,{'policy_fingerprint':policy['fingerprint'],'applied_at':time.time(),
                             'before_state_sha256':txn['before_state_sha256'],'after_state_sha256':txn['after_state_sha256'],
                             'archive_inventory_sha256':digest(folder/'archive_inventory.json')})
        txn['phase']='APPLIED';_atomic_json(transaction,txn)
        print_json({'repair':'APPLIED_NOT_STARTED',**saved_status(run)})
    return 0


def activate(args):
    run=args.run_dir.resolve();plan=checked_plan(run);policy=load_repair_policy(plan)
    if not (run/'repair/applied.json').exists():raise ValueError('Repair must be applied before activation')
    state=_read_state(plan)
    if state.get('repair_circuit_breaker'):raise RuntimeError('Infrastructure circuit breaker requires review; no restart')
    controller=Path(policy['controller_source']);verify_snapshot(controller)
    lock=acquire_lock(run/'manager.lock')
    if lock is None:
        record=load_json(run/'manager.json')
        if record.get('source')!=str(controller):raise RuntimeError('Another manager is active')
        print_json({'manager':'ALREADY_RUNNING_REPAIR',**saved_status(run)});return 0
    with lock:
        if load_policy(plan)['fingerprint']!=policy['scheduling_policy_fingerprint']:
            raise ValueError('Inherited scheduling policy changed')
        partition_evidence(plan,state)
        runtime=Path(load_json(run/'source/snapshot.json')['selection']['runtime'])
        log_path=run/'manager.log'
        with log_path.open('a') as log:
            child=subprocess.Popen([sys.executable,'-I','-B','-u',str(controller/'scripts/repair_ics.py'),
                'watch','--run-dir',str(run),'--interval',str(args.interval),'--lock-fd',str(lock.fileno())],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=clean_environment(runtime),
                start_new_session=True,pass_fds=(lock.fileno(),))
        deadline=time.monotonic()+3;ready=False
        while time.monotonic()<deadline:
            if child.poll() is not None:raise RuntimeError('Repair manager exited during startup; inspect '+str(log_path))
            try:ready=load_json(run/'manager.json').get('pid')==child.pid
            except (OSError,ValueError):pass
            if ready:break
            time.sleep(.05)
        print_json({'manager_start':'STARTED_REPAIR' if ready else 'STARTING_REPAIR',
                    'manager_pid':child.pid,**saved_status(run)})
    return 0


def pause(args):
    run=args.run_dir.resolve();plan=checked_plan(run);policy=load_repair_policy(plan)
    if manager_status(run)['manager']!='ACTIVE':
        print_json({'manager_pause':'ALREADY_STOPPED',**saved_status(run)});return 0
    record=load_json(run/'manager.json')
    def verified():
        if record.get('hostname')!=socket.gethostname() or record.get('source')!=policy['controller_source']:
            raise ValueError('Manager host/source does not match this repair controller')
        pid=record.get('pid')
        if type(pid) is not int or pid<=1:raise ValueError('Invalid manager PID')
        proc=Path('/proc')/str(pid)
        if proc.stat().st_uid!=os.getuid():raise ValueError('Manager has another owner')
        argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        script=str(Path(policy['controller_source'])/'scripts/repair_ics.py')
        if argv.count(script)!=1 or 'watch' not in argv or '--run-dir' not in argv:
            raise ValueError('Manager command is not the prepared repair watcher')
        index=argv.index('--run-dir')
        if index+1>=len(argv) or Path(argv[index+1]).resolve()!=run:raise ValueError('Manager run mismatch')
        fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        boot=next(int(s.split()[1]) for s in Path('/proc/stat').read_text().splitlines() if s.startswith('btime '))
        started=boot+int(fields[19])/os.sysconf('SC_CLK_TCK')
        if abs(started-float(record['started_at']))>30:raise ValueError('Manager PID was reused')
        lock=(run/'manager.lock').stat();held=False
        for descriptor in (proc/'fd').iterdir():
            try:
                info=descriptor.stat()
                if (info.st_dev,info.st_ino)==(lock.st_dev,lock.st_ino):held=True
            except FileNotFoundError:pass
        if not held:raise ValueError('Manager does not hold this run lock')
        return pid
    pid=verified();fd=os.pidfd_open(pid)
    try:verified();signal.pidfd_send_signal(fd,signal.SIGTERM)
    finally:os.close(fd)
    deadline=time.monotonic()+45
    while manager_status(run)['manager']=='ACTIVE' and time.monotonic()<deadline:time.sleep(.5)
    status=manager_status(run)
    print_json({'manager_pause':'STOPPED' if status['manager']!='ACTIVE' else 'STOP_REQUESTED',
                'slurm_jobs':'PRESERVED',**status})
    return 0 if status['manager']!='ACTIVE' else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--expected-plan-fingerprint',required=True)
    p.add_argument('--expected-successful-experiments',type=int,required=True)
    p.add_argument('--expected-failed',type=int,required=True);p.add_argument('--expected-blocked',type=int,required=True)
    for name in ('apply','activate','pause','status','watch'):
        p=sub.add_parser(name);p.add_argument('--run-dir',required=True,type=Path)
        if name in {'activate','watch'}:p.add_argument('--interval',type=float,default=60)
        if name=='watch':p.add_argument('--lock-fd',type=int);p.add_argument('--once',action='store_true')
    args=parser.parse_args(argv)
    if hasattr(args,'interval') and (not math.isfinite(args.interval) or args.interval<10):parser.error('interval must be >=10 seconds')
    if args.command=='prepare':return prepare(args)
    run=args.run_dir.resolve()
    if args.command=='status':
        print_json(saved_status(run));return 0
    plan=checked_plan(run);policy=load_repair_policy(plan);controller=Path(policy['controller_source'])
    if ROOT!=controller:
        command=[sys.executable,'-I','-B','-u',str(controller/'scripts/repair_ics.py'),args.command,'--run-dir',str(run)]
        if hasattr(args,'interval'):command+=['--interval',str(args.interval)]
        if args.command=='watch':raise ValueError('Watch must execute the immutable repair controller')
        return subprocess.run(command,check=False).returncode
    verify_snapshot(ROOT)
    if args.command=='apply':return apply(args)
    if args.command=='activate':return activate(args)
    if args.command=='pause':return pause(args)
    if not (run/'repair/applied.json').exists():raise ValueError('Unapplied repair cannot watch')
    if _read_state(plan).get('repair_circuit_breaker'):raise ValueError('Infrastructure circuit breaker requires review')
    return watcher(run,interval=args.interval,once=args.once,lock_fd=args.lock_fd,repair=True)


if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        print('Repair error: '+str(exc),file=sys.stderr)
        raise SystemExit(1)
