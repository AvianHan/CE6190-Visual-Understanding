#!/usr/bin/env python3
"""Resume the audited failed SUIM data stage inside the existing batch2 run."""
from __future__ import annotations
import argparse
import functools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ics import batch2, batch2_recovery as recovery, rolling_queue as rq
from ics.orchestration import load_json
from scripts.queue_ics import acquire_lock, clean_environment, freeze_source, manager_status, print_json, verify_snapshot
from scripts.batch2_ics import current_status, watch as batch_watch


def freeze_and_reexec(args):
    run=args.run_dir.resolve();meta,_=recovery.checked(run,args.expected_plan_fingerprint)
    folder=run/recovery.TAG;folder.mkdir(parents=True,exist_ok=True)
    runtime=Path(meta['selection']['runtime'])
    source=freeze_source(folder,ROOT/'configs/suite.batch2.core.json',
                         ROOT/'configs/cluster.orion.batch2.json',runtime)
    command=[sys.executable,'-I','-B','-u',str(source/'scripts/recover_batch2_suim.py'),'activate',
        '--run-dir',str(run),'--expected-plan-fingerprint',args.expected_plan_fingerprint,
        '--expected-failed-job',args.expected_failed_job,'--interval',str(args.interval)]
    return subprocess.run(command,env=clean_environment(runtime),check=False).returncode


def launch_manager(run,policy,lock,interval):
    log_path=run/'manager.log'
    runtime=Path(batch2.load_batch(run)['selection']['runtime'])
    with log_path.open('a') as stream:
        child=subprocess.Popen([sys.executable,'-I','-B','-u',
            str(ROOT/'scripts/recover_batch2_suim.py'),'watch','--run-dir',str(run),
            '--interval',str(interval),'--lock-fd',str(lock.fileno())],
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            env=clean_environment(runtime),start_new_session=True,pass_fds=(lock.fileno(),))
    ready=False;end=time.monotonic()+3
    while time.monotonic()<end:
        if child.poll() is not None:raise RuntimeError('Recovery manager exited at startup; inspect '+str(log_path))
        try:ready=load_json(run/'manager.json').get('pid')==child.pid
        except (OSError,ValueError):pass
        if ready:break
        time.sleep(.05)
    print_json({'recovery':'APPLIED','manager_start':'STARTED_RECOVERY' if ready else 'STARTING_RECOVERY',
                'manager_pid':child.pid,'manager_log':str(log_path),**current_status(run)})


def activate(args):
    run=args.run_dir.resolve();source=run/recovery.TAG/'source'
    if ROOT!=source:return freeze_and_reexec(args)
    verify_snapshot(ROOT)
    policy=recovery.prepare_policy(run,ROOT,args.expected_plan_fingerprint,args.expected_failed_job)
    # Validate current allocation rules BEFORE pausing the healthy manager. These
    # policies are separate from original immutable plans and worker snapshots.
    from ics.batch2_scheduling import prepare_policies
    prepare_policies(run,ROOT)
    outer=acquire_lock(run/recovery.TAG/'handover.lock')
    if outer is None:raise RuntimeError('Another SUIM recovery handover is active')
    with outer:
        applied=run/recovery.TAG/'applied.json'
        if applied.exists():recovery.apply_transaction(run,policy)  # Read-only applied receipt verification.
        if applied.exists() and manager_status(run)['manager']=='ACTIVE':
            record=load_json(run/'manager.json')
            if record.get('source')==str(ROOT):
                recovery.verified_manager_pid(run,record)
                print_json({'recovery':'ALREADY_APPLIED','manager_start':'ALREADY_RUNNING_RECOVERY',**current_status(run)})
                return 0
        lock=recovery.manager_handover_lock(run)
        with lock:
            recovery.load_policy(run/recovery.TAG/'policy.json')
            with rq._lock(Path(policy['scope_run'])):
                receipt=recovery.apply_transaction(run,policy)
            summary=current_status(run)
            if summary['terminal']:
                print_json({'recovery':'ALREADY_APPLIED','manager_start':'NOT_NEEDED',**summary})
                return 0 if summary['batch2_status']=='COMPLETE' else 2
            launch_manager(run,policy,lock,args.interval)
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('audit','activate'):
        p=sub.add_parser(name);p.add_argument('--run-dir',type=Path,required=True)
        p.add_argument('--expected-plan-fingerprint',default=recovery.EXPECTED_PLAN)
        p.add_argument('--expected-failed-job',default=recovery.EXPECTED_JOB)
        p.add_argument('--interval',type=float,default=30)
    p=sub.add_parser('status');p.add_argument('--run-dir',type=Path,required=True)
    p=sub.add_parser('watch',help=argparse.SUPPRESS)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--interval',type=float,default=30)
    p.add_argument('--lock-fd',type=int);p.add_argument('--once',action='store_true')
    p=sub.add_parser('worker',help=argparse.SUPPRESS)
    p.add_argument('--intent',type=Path,required=True);p.add_argument('--intent-sha256',required=True)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--policy-sha256',required=True)
    args=parser.parse_args(argv)
    if hasattr(args,'interval') and (not math.isfinite(args.interval) or args.interval<10):parser.error('--interval must be finite and >=10')
    if args.command=='worker':
        recovery.execute_data_worker(args.intent,args.intent_sha256,args.policy,args.policy_sha256);return 0
    run=args.run_dir.resolve()
    if args.command=='status':print_json(current_status(run));return 0
    if args.command=='audit':
        _,plan=recovery.checked(run,args.expected_plan_fingerprint)
        _,proof=recovery.audit_failure(run,plan,args.expected_failed_job)
        print_json({'audit':'READY_FOR_REVIEW','failed_job':args.expected_failed_job,
                    'unblocked':len(proof['unblocked']),'successful_models_reset':0,
                    'slurm_evidence':proof['history'],'data_error':proof['failed_marker']['error']});return 0
    if args.command=='activate':return activate(args)
    policy,_=recovery.load_policy(run/recovery.TAG/'policy.json')
    if ROOT!=Path(policy['controller_source']):raise ValueError('Watcher must execute isolated frozen recovery source')
    if not (run/recovery.TAG/'applied.json').is_file():raise ValueError('Recovery transaction has not completed')
    recovery.apply_transaction(run,policy)  # Verify immutable applied transaction without resetting progress.
    env=clean_environment(Path(batch2.load_batch(run)['selection']['runtime']))
    os.environ.clear();os.environ.update(env)
    return batch_watch(run,args.interval,args.once,args.lock_fd,
                       tick_fn=functools.partial(batch2.tick_batch,scheduling=True))


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,OSError,KeyError,ImportError) as exc:
        print('SUIM recovery error: '+str(exc),file=sys.stderr,flush=True)
        print('Existing Slurm jobs were not cancelled. If handover paused the manager, new submissions stay stopped until this transaction completes. Repeat the same activate command to resume; do not edit queue state.',file=sys.stderr,flush=True)
        raise SystemExit(1)
