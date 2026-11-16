"""Opt-in batch2 reuse of the validated H200/L40S scheduling policy.

No scientific plan, accepted job ID, worker source or resource request changes.
Only exact PENDING smoke jobs may receive an in-place partition update. New
formal work gains both partitions only after its exact dataset/config smoke
passes on L40S, using the existing scheduling.effective_profile contract.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from . import rolling_queue as rq
from .orchestration import fingerprint
from .scheduling import load_policy, partition_evidence, qos_limit, _job_fields


def prepare_policies(parent, controller_source):
    """Prepare immutable policies; read Slurm limits/partitions, submit nothing."""
    from .batch2 import load_batch, immutable_json
    parent, controller_source = Path(parent).resolve(), Path(controller_source).resolve()
    meta = load_batch(parent)
    if not controller_source.is_relative_to(parent) or not (controller_source / 'snapshot.json').is_file():
        raise ValueError('Scheduling controller must be a frozen source inside the parent run')
    plans = {scope: rq._plan(Path(row['path'])) for scope, row in meta['scope_plans'].items()}
    prepared, fresh = {}, {}
    # Validate all scopes before any new policy is written. A policy cannot
    # accidentally expand a cap when fresh QoS evidence advertises >60 slots.
    for scope, plan in plans.items():
        state = rq._read_state(plan)
        path = Path(plan['run_dir']) / 'scheduling/policy.json'
        if path.exists():
            policy = load_policy(plan)
            if Path(policy['controller_source']).resolve() != controller_source:
                raise ValueError('Existing scheduling controller differs')
            current_qos = qos_limit(state['policy']['qos'], state['policy']['shared_limit'])
            if policy['shared_limit'] > current_qos['effective_limit']:
                raise ValueError('Current verified QoS limit is lower than immutable scheduling policy')
            prepared[scope] = policy
        else:
            inherited = state['policy']['shared_limit']
            evidence = qos_limit(state['policy']['qos'], inherited)
            cap = min(inherited, evidence['effective_limit'], 60)
            if cap <= state['policy']['reserved_slots']:
                raise ValueError('Verified shared quota leaves no usable slots')
            policy = {'schema_version': 1, 'created_at': time.time(),
                      'plan_fingerprint': plan['fingerprint'], 'worker_source': plan['project_root'],
                      'controller_source': str(controller_source), 'batch2_parent': str(parent),
                      'inherited_shared_limit': inherited, 'verified_bound': evidence['effective_limit'],
                      'shared_limit': cap, 'smoke_partition': 'l40sp', 'formal_partitions': 'h200p,l40sp',
                      'qos_evidence': evidence,
                      'note': 'Existing H200/L40S profiles only; original jobs/workers/plans/PASS preserved; pending smoke partition only.'}
            fresh[scope] = policy
        # Refresh eligibility even when reusing an immutable policy.
        current = partition_evidence(plan, state)
        if scope in fresh:
            fresh[scope]['partition_evidence'] = current
            fresh[scope]['fingerprint'] = fingerprint(fresh[scope])
    caps = {p['shared_limit'] for p in [*prepared.values(), *fresh.values()]}
    if len(caps) != 1:
        raise ValueError('Child policies disagree on the global shared limit')
    for scope, policy in fresh.items():
        immutable_json(Path(plans[scope]['run_dir']) / 'scheduling/policy.json', policy)
        prepared[scope] = load_policy(plans[scope])
    return {'status': 'SCHEDULING_PREPARED_NOT_RUNNING_PROOF',
            'shared_limit': next(iter(caps)),
            'policies': {scope: {'fingerprint': p['fingerprint'], 'smoke_partition': p['smoke_partition'],
                                'formal_partitions': p['formal_partitions']} for scope, p in prepared.items()}}


def _checked_command(plan, task):
    """Validate sealed pre-sbatch identity without trusting mutable job ID alone."""
    from .batch2 import digest
    run = Path(plan['run_dir']).resolve()
    source = Path(plan['project_root']).resolve()
    intent_path = Path(task['batch2_intent']).resolve()
    if intent_path.parents[1] != run / 'batch2_intents' or intent_path.parent.name != task['key']:
        raise ValueError('Pending smoke intent path belongs to another task')
    if digest(intent_path) != task['batch2_intent_sha256']:
        raise ValueError('Pending smoke sealed intent hash mismatch')
    intent = json.loads(intent_path.read_text())
    if (intent['plan_fingerprint'] != plan['fingerprint'] or
            Path(intent['plan_path']).resolve() != run / 'plan.json' or
            Path(intent['worker_source']).resolve() != source or
            digest(source / 'snapshot.json') != intent['source_snapshot_sha256']):
        raise ValueError('Pending smoke intent differs from frozen plan/source')
    keys = ('key', 'kind', 'resource', 'index', 'dependencies', 'job_name')
    if any(intent['task'].get(k) != task.get(k) for k in keys):
        raise ValueError('Pending smoke task identity differs from sealed intent')
    if task['kind'] != 'smoke' or intent['attempt'] != task['attempts']:
        raise ValueError('Pending smoke attempt identity differs')
    if intent_path.with_suffix('.claim.json').exists():
        raise ValueError('Worker already claimed execution; do not retarget')
    return [plan['cluster']['launcher_python'], '-I', '-B', '-u',
            str(source / 'scripts/batch2_ics.py'), 'worker', '--intent', str(intent_path),
            '--intent-sha256', task['batch2_intent_sha256']]


def retarget_pending_smokes(plan, state, policy, queue, stop_requested=None):
    """Change only the partition of an exact, accepted PENDING smoke job.

    Slurm's pending-only partition update handles a pending→running race. No
    cancellation, replacement, time-limit decrease, retry or resubmission occurs.
    Failures are recorded and the original accepted job remains authoritative.
    """
    events = []
    run = Path(plan['run_dir']).resolve()
    for task in state['tasks'].values():
        if stop_requested and stop_requested(): break
        jid = str(task.get('job_id', ''))
        if (task['kind'] != 'smoke' or task['status'] != 'SUBMITTED' or not re.fullmatch(r'\d+', jid) or
                jid not in queue or queue[jid]['state'] != 'PENDING'):
            continue
        event_path = None
        try:
            matches = {key for key, value in queue.items() if value.get('job_name') == task['job_name']
                       and value.get('qos') == state['policy']['qos']}
            if matches != {jid}:
                raise ValueError('Pending smoke job name is not unique in the live queue')
            if queue[jid].get('job_name') != task['job_name'] or queue[jid].get('qos') != state['policy']['qos']:
                raise ValueError('Queue identity differs from this accepted smoke')
            command = _checked_command(plan, task)
            script = run / 'rolling_slurm' / (task['key'] + '.sh')
            if script.is_symlink(): raise ValueError('Pending script is a symlink')
            lines = script.read_text().splitlines()
            if not lines or shlex.split(lines[-1]) != ['exec', *command]:
                raise ValueError('Pending script differs from the sealed original worker')
            before = _job_fields(rq._command(['scontrol', '--oneliner', 'show', 'job', jid]))
            profile = rq._profile(plan, task)
            if (before.get('JobId') != jid or before.get('JobName') != task['job_name'] or
                    before.get('UserId', '').split('(')[0] != state['policy']['user'] or
                    before.get('QOS') != state['policy']['qos'] or before.get('Account') != profile['account'] or
                    before.get('Command') != str(script)):
                raise ValueError('Scheduler job identity differs from this frozen run')
            if before.get('JobState') != 'PENDING' or before.get('Partition') == policy['smoke_partition']:
                continue
            event = {'at': time.time(), 'key': task['key'], 'job_id': jid,
                     'plan_fingerprint': plan['fingerprint'], 'intent_sha256': task['batch2_intent_sha256'],
                     'before': before, 'requested_partition': policy['smoke_partition'], 'status': 'INTENT'}
            event_path = run / 'scheduling/updates' / (jid + '-' + str(time.time_ns()) + '.json')
            rq._atomic_json(event_path, event)
            result = subprocess.run(['scontrol', 'update', 'JobId=' + jid,
                                     'Partition=' + policy['smoke_partition']],
                                    capture_output=True, text=True, timeout=30)
            event.update(returncode=result.returncode, stderr=result.stderr[-1500:])
            after = _job_fields(rq._command(['scontrol', '--oneliner', 'show', 'job', jid]))
            same = all(after.get(k) == before.get(k) for k in ('JobId', 'JobName', 'UserId', 'QOS', 'Account', 'Command',
                            'NumCPUs', 'NumNodes', 'MinMemoryNode', 'TresPerNode', 'ReqTRES', 'TimeLimit', 'ExcNodeList'))
            event.update(after=after, status='UPDATED' if result.returncode == 0 and same
                         and after.get('Partition') == policy['smoke_partition'] else 'DEFERRED')
            rq._atomic_json(event_path, event)
            events.append({'key': task['key'], 'job_id': jid, 'status': event['status']})
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            if event_path is not None:
                event.update(status='DEFERRED', error=str(exc))
                rq._atomic_json(event_path, event)
            events.append({'key': task['key'], 'job_id': jid, 'status': 'DEFERRED', 'error': str(exc)})
    return events
