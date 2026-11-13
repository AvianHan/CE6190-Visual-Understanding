"""Plan-bound scheduling changes; scientific workers and accepted jobs stay intact."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from .orchestration import fingerprint
from .rolling_queue import _atomic_json, _command, _profile


def load_policy(plan):
    path = Path(plan['run_dir']) / 'scheduling' / 'policy.json'
    policy = json.loads(path.read_text())
    body = {k: v for k, v in policy.items() if k != 'fingerprint'}
    if fingerprint(body) != policy.get('fingerprint'):
        raise ValueError('Scheduling policy fingerprint mismatch')
    if policy['plan_fingerprint'] != plan['fingerprint']:
        raise ValueError('Scheduling policy belongs to another plan')
    if policy['worker_source'] != plan['project_root']:
        raise ValueError('Scheduling policy changed the frozen worker source')
    if type(policy['shared_limit']) is not int or not 1 <= policy['shared_limit'] <= policy['verified_bound']:
        raise ValueError('Scheduling policy limit is outside the verified inherited bound')
    if policy['smoke_partition'] != 'l40sp' or policy['formal_partitions'] != 'h200p,l40sp':
        raise ValueError('Unsupported hardware policy')
    evidence = policy['qos_evidence']
    bound = evidence['actual_limit'] if evidence['status'] == 'VERIFIED' else policy['inherited_shared_limit']
    if policy['verified_bound'] != bound:
        raise ValueError('Scheduling bound does not match its QoS evidence')
    return policy


def qos_limit(qos, inherited):
    """Use a verified current limit; retain the historical bound when the query is unavailable."""
    evidence = {'queried_at': time.time(), 'qos': qos, 'inherited_limit': inherited}
    try:
        output = _command(['sacctmgr', '--noheader', '--parsable2', 'show', 'qos',
                           'where', 'name=' + qos, 'format=Name,MaxSubmitJobsPU'])
        rows = [line.strip().split('|') for line in output.splitlines() if line.strip()]
        match = [row for row in rows if row[0] == qos]
        if len(match) != 1 or len(match[0]) < 2:
            raise ValueError('Expected exactly one matching QoS row')
        raw = match[0][1].strip()
        if raw and raw.lower() not in {'unlimited', 'none', 'n/a'}:
            actual = int(raw)
            if actual < 1:
                raise ValueError('QoS limit is not positive')
            evidence.update(status='VERIFIED', actual_limit=actual, effective_limit=actual)
        else:
            evidence.update(status='UNBOUNDED_FIELD', effective_limit=inherited)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        evidence.update(status='INHERITED_BOUND_QUERY_FAILED', error=str(exc), effective_limit=inherited)
    return evidence


def hardware_evidence(state):
    ids = [str(t['job_id']) for t in state['tasks'].values()
           if t['kind'] == 'smoke' and t.get('job_id') and t['status'] == 'PASS']
    if not ids:
        return {}, None
    try:
        output = _command(['sacct', '--noheader', '--parsable2', '--allocations',
                           '--jobs', ','.join(ids), '--starttime',
                           dt.datetime.fromtimestamp(state['created_at'] - 120).strftime('%Y-%m-%dT%H:%M:%S'), '--format', 'JobIDRaw,Partition,State,ExitCode'])
        found = {}
        for line in output.splitlines():
            if not line.strip():
                continue
            fields = [part.strip() for part in line.split('|')]
            if len(fields) == 5 and not fields[-1]:
                fields.pop()
            if len(fields) != 4 or not fields[0].isdigit():
                raise ValueError('Malformed smoke hardware accounting')
            jid, partition, status, exit_code = fields
            if jid in found:
                raise ValueError('Duplicate smoke hardware accounting')
            found[jid] = {'partition': partition, 'state': status, 'exit_code': exit_code}
        return found, None
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        # Missing hardware evidence restricts new formal work to the original H200 profile.
        return {}, str(exc)


def effective_profile(plan, state, task, policy, hardware):
    profile = dict(_profile(plan, task))
    if task['kind'] == 'smoke':
        profile.update(partition=policy['smoke_partition'], time='01:00:00')
    elif task['kind'] == 'experiment':
        gates = [state['tasks'][key] for key in task['dependencies']]
        verified = bool(gates) and all(
            gate['kind'] == 'smoke' and gate['status'] == 'PASS'
            and hardware.get(str(gate.get('job_id')), {}) == {
                'partition': 'l40sp', 'state': 'COMPLETED', 'exit_code': '0:0'}
            for gate in gates)
        if verified:
            profile['partition'] = policy['formal_partitions']
    return profile


def fair_tasks(plan, state):
    """Keep recovery tiers, round-robin resource families inside each tier."""
    definitions = {t['id']: t for t in plan['smoke'] + plan['tasks']}
    buckets = defaultdict(lambda: defaultdict(deque))
    for task in state['tasks'].values():
        kind = task['kind']
        if kind in {'data', 'preflight', 'collect'}:
            tier = {'data': 0, 'preflight': 1, 'collect': 6}[kind]
        else:
            definition = definitions[task['key']]
            main = definition['config_data']['experiment_id'] in {'insid3_full', 'cgics_full'}
            if kind == 'smoke':
                tier = 2 if main else 4
            else:
                standard = plan['manifests'][definition['manifest_id']].get('cohort') == 'standard'
                tier = 3 if main and standard else 5
        buckets[tier][task['resource']].append(task)
    active = Counter(t['resource'] for t in state['tasks'].values() if t['status'] in {'SUBMITTING', 'SUBMITTED', 'RUNNING', 'UNCERTAIN'})
    submitted = Counter(t['resource'] for t in state['tasks'].values() if t.get('job_id'))
    ordered = []
    for tier in sorted(buckets):
        families = dict(sorted(buckets[tier].items(), key=lambda item: (active[item[0]], submitted[item[0]])))
        while any(families.values()):
            for family in families.values():
                if family:
                    ordered.append(family.popleft())
    return ordered


def _job_fields(output):
    # scontrol --oneliner fields have no spaces in the checked identity values.
    return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)', output))


def partition_evidence(plan, state):
    evidence = {}
    accounts = {profile['account'] for profile in plan['resources'].values()}
    qos = state['policy']['qos']
    for name in ('h200p', 'l40sp'):
        output = _command(['scontrol', '--oneliner', 'show', 'partition', name])
        rows = [_job_fields(line) for line in output.splitlines() if line.strip()]
        matching = [row for row in rows if row.get('PartitionName') == name]
        if len(matching) != 1:
            raise ValueError('Cannot establish partition identity: ' + name)
        row = matching[0]
        if row.get('State') != 'UP':
            raise ValueError('Partition is not UP: ' + name)
        allowed_qos = row.get('AllowQos', '').split(',')
        allowed_accounts = row.get('AllowAccounts', '').split(',')
        if ('ALL' not in allowed_qos and qos not in allowed_qos
                or qos in row.get('DenyQos', '').split(',')):
            raise ValueError('Partition does not explicitly allow this QoS: ' + name)
        if ('ALL' not in allowed_accounts and not accounts.issubset(allowed_accounts)
                or accounts.intersection(row.get('DenyAccounts', '').split(','))):
            raise ValueError('Partition does not explicitly allow this account: ' + name)
        evidence[name] = row
    return evidence


def retarget_pending_smokes(plan, state, policy, queue, stop_requested=None):
    """Update accepted smoke job partitions, never replace/re-submit a job."""
    run = Path(plan['run_dir'])
    events = []
    for task in state['tasks'].values():
        if stop_requested and stop_requested():
            break
        jid = str(task.get('job_id', ''))
        if task['kind'] != 'smoke' or jid not in queue or queue[jid]['state'] != 'PENDING':
            continue
        try:
            before = _job_fields(_command(['scontrol', '--oneliner', 'show', 'job', jid]))
            expected_script = str(run / 'rolling_slurm' / (task['key'] + '.sh'))
            expected_account = _profile(plan, task)['account']
            command = [plan['cluster'].get('launcher_python', 'python3'),
                       str(Path(plan['project_root']) / 'scripts' / 'queue_ics.py'), 'worker',
                       '--plan', str(run / 'plan.json'), '--key', task['key']]
            script_lines = Path(expected_script).read_text().splitlines()
            if not script_lines or shlex.split(script_lines[-1]) != ['exec', *command]:
                raise ValueError('Accepted job script does not execute the original frozen worker')
            if (before.get('JobId') != jid or before.get('JobName') != task['job_name']
                    or before.get('UserId', '').split('(')[0] != state['policy']['user']
                    or before.get('QOS') != state['policy']['qos']
                    or before.get('Account') != expected_account
                    or before.get('Command') != expected_script):
                raise ValueError('Pending job identity does not match this run')
            if before.get('JobState') != 'PENDING' or before.get('Partition') == policy['smoke_partition']:
                continue
            # Partition changes require PENDING in Slurm. Do not combine a time
            # reduction: a PENDING->RUNNING race must not shorten a live job.
            event = {'at': time.time(), 'key': task['key'], 'job_id': jid,
                     'before': before, 'requested_partition': policy['smoke_partition'],
                     'status': 'INTENT'}
            event_path = run / 'scheduling' / 'updates' / (jid + '.json')
            _atomic_json(event_path, event)
            result = subprocess.run(['scontrol', 'update', 'JobId=' + jid,
                                     'Partition=' + policy['smoke_partition']],
                                    capture_output=True, text=True, timeout=30)
            event.update(returncode=result.returncode, stderr=result.stderr[-1500:])
            after = _job_fields(_command(['scontrol', '--oneliner', 'show', 'job', jid]))
            event.update(after=after, status='UPDATED' if result.returncode == 0
                         and after.get('Partition') == policy['smoke_partition'] else 'DEFERRED')
            _atomic_json(event_path, event)
            events.append({'key': task['key'], 'job_id': jid, 'status': event['status']})
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            events.append({'key': task['key'], 'job_id': jid, 'status': 'DEFERRED', 'error': str(exc)})
    return events
