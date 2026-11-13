#!/usr/bin/env python3
"""Sealed repair attempt launcher; scientific modules come only from original run/source."""
from __future__ import annotations
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time


def visible_read(path):
    """Bounded retry for publication visibility only; malformed/changed bytes fail closed."""
    for attempt in range(8):
        try:return Path(path).read_bytes()
        except OSError as exc:
            if exc.errno not in {errno.ENOENT,errno.ESTALE} or attempt==7:raise
            time.sleep(.25)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intent',required=True,type=Path)
    parser.add_argument('--intent-sha256',required=True)
    args=parser.parse_args(argv)
    raw=visible_read(args.intent)
    if hashlib.sha256(raw).hexdigest()!=args.intent_sha256:
        raise ValueError('Sealed submission intent hash mismatch')
    intent=json.loads(raw)
    plan_path=Path(intent['plan_path']).resolve();run=plan_path.parent
    if args.intent.resolve().parents[1] != run/'repair/intents':
        raise ValueError('Intent is outside this run repair namespace')
    source=Path(intent['worker_source']).resolve()
    if source != run/'source':raise ValueError('Unexpected scientific source path')
    snapshot_path=source/'snapshot.json'
    snapshot_bytes=visible_read(snapshot_path)
    if hashlib.sha256(snapshot_bytes).hexdigest()!=intent['worker_snapshot_sha256']:
        raise ValueError('Original worker snapshot changed')
    snapshot=json.loads(snapshot_bytes)
    for name,expected in snapshot['files'].items():
        path=source/name
        if path.is_symlink() or hashlib.sha256(visible_read(path)).hexdigest()!=expected:
            raise ValueError('Frozen scientific source changed: '+name)
    policy=json.loads(visible_read(run/'repair/policy.json'))
    if policy.get('fingerprint')!=intent['repair_policy_fingerprint']:
        raise ValueError('Intent belongs to another repair policy')
    # No operational controller package has been imported in this fresh process.
    sys.path.insert(0,str(source))
    from ics.rolling_queue import _plan, _atomic_json, _marker_path
    from ics.orchestration import run_task, collect, fingerprint
    if fingerprint({k:v for k,v in policy.items() if k!='fingerprint'})!=policy.get('fingerprint'):
        raise ValueError('Repair policy checksum mismatch')
    plan=_plan(plan_path)
    if plan['fingerprint']!=intent['plan_fingerprint'] or plan['project_root']!=str(source):
        raise ValueError('Sealed intent changed plan/source identity')
    task=intent['task'];key=task['key']
    jid=os.environ.get('SLURM_JOB_ID','')
    if not re.fullmatch(r'\d+',jid):raise RuntimeError('Slurm allocation required')
    if os.environ.get('SLURM_JOB_NAME')!=task['job_name']:
        raise RuntimeError('Slurm job name differs from sealed attempt identity')
    if task.get('kind') not in {'smoke','experiment','collect'}:
        raise ValueError('Repair cannot rerun data/preflight stages')
    if task['kind']!='collect':
        group=[t for t in plan['smoke' if task['kind']=='smoke' else 'tasks'] if t['resource']==task['resource']]
        original=group[task['index']]
        if original['id']!=key:raise ValueError('Sealed worker index differs from immutable task')
    # One immutable attempt may execute only once, even if the same launcher is
    # invoked concurrently. The losing invocation must never overwrite its marker.
    claim=args.intent.with_suffix('.claim.json')
    fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as handle:
        json.dump({'slurm_job_id':jid,'job_name':task['job_name'],
                   'intent_sha256':args.intent_sha256,'claimed_at':time.time()},handle)
        handle.flush();os.fsync(handle.fileno())
    path=_marker_path(plan,key)
    if path.exists():raise RuntimeError('Attempt marker already exists; refusing duplicate/overlapping execution')
    marker={'schema_version':1,'key':key,'kind':task['kind'],'plan_fingerprint':plan['fingerprint'],
            'slurm_job_id':jid,'status':'RUNNING','started_at':time.time(),
            'repair_intent_sha256':args.intent_sha256,'repair_policy_fingerprint':policy['fingerprint']}
    _atomic_json(path,marker)
    previous=signal.getsignal(signal.SIGTERM)
    def interrupted(signum,frame):raise KeyboardInterrupt('Termination signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    try:
        if task['kind']=='collect':
            result=collect(plan)
            if result.get('analysis_returncode',0):raise RuntimeError('Result analysis failed')
        else:
            run_task(plan,original)
        marker.update(status='PASS',ended_at=time.time());_atomic_json(path,marker)
    except BaseException as exc:
        marker.update(status='FAIL',ended_at=time.time(),error_type=type(exc).__name__,error=str(exc))
        _atomic_json(path,marker)
        raise
    finally:signal.signal(signal.SIGTERM,previous)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        print('Repair worker error: '+str(exc),file=sys.stderr)
        raise SystemExit(1)
