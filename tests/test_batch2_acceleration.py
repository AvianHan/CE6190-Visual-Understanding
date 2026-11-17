"""Real sealed batch2 fixtures with a stateful, allocated-job Slurm simulator.

These tests exercise existing accepted jobs only. They never contact Slurm or
run a model; unexpected submission/cancellation is a test failure.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ics import batch2, orchestration, rolling_queue as rq
from scripts.queue_ics import freeze_source

ROOT = Path(__file__).resolve().parents[1]


class FakeSlurm:
    def __init__(self):
        self.jobs = {}
        self.calls = []
        self.fail_update = False
        self.hold_race = False
        self.fail_queries = False
        self.history = {}
        self.crash_after_hold = False
        self.crash_after_update = False
        self.race_after_successful_hold = False
        self.boot_times = {}

    def show_job(self, job_id):
        if self.fail_queries:
            raise RuntimeError('scheduler query failed')
        return copy.deepcopy(self.jobs[str(job_id)])

    def show_node(self, node):
        return dict(NodeName=node, State='IDLE', BootTime=self.boot_times.get(node, '2026-11-16T15:51:58'),
                    SlurmdStartTime='2026-11-16T16:05:45',
                    CfgTRES='cpu=112,mem=2063918M,gres/gpu=8',
                    AllocTRES='', CPUEfctv='112', CPUAlloc='0',
                    Gres='gpu:l40s:8' if node.startswith('A484') else 'gpu:h200:8')

    def show_partition(self, part):
        return dict(PartitionName=part, State='UP', AllowQos='ALL',
                    AllowAccounts='ALL', DenyAccounts='', DenyQos='',
                    MaxTime='18:00:00', RootOnly='NO', ReqResv='NO',
                    Nodes='BFS9B74,C64Y574' if part == 'testqueue' else 'A484652X3B00163,A484652X3B10601')

    def expand_nodes(self, nodes):
        if nodes in ('', '(null)', 'None', None):
            return set()
        return set(nodes.split(','))

    def command(self, args):
        args = list(map(str, args)); self.calls.append(args)
        if args[0] in ('sbatch', 'scancel'):
            raise AssertionError('Existing-job accelerator must never submit/cancel')
        if args[:2] == ['scontrol', 'hold']:
            row = self.jobs[args[-1]]
            if self.hold_race:
                self.hold_race = False
                row['JobState'] = 'RUNNING'; row['Reason'] = 'None'
                raise RuntimeError('job started before hold')
            if row['JobState'] != 'PENDING':
                raise RuntimeError('cannot hold non-pending job')
            row.update(Priority='0', Reason='JobHeldUser')
            if self.race_after_successful_hold:
                self.race_after_successful_hold = False
                row['JobState'] = 'RUNNING'
            if self.crash_after_hold:
                self.crash_after_hold = False
                raise KeyboardInterrupt('simulated process death after hold acknowledgement')
            return ''
        if args[:2] == ['scontrol', 'release']:
            row = self.jobs[args[-1]]
            row.update(Priority='100', Reason='Priority')
            return ''
        if args[:2] == ['scontrol', 'update']:
            updates = dict(part.split('=', 1) for part in args[2:])
            jid = updates.pop('JobId', updates.pop('JobID', None))
            row = self.jobs[jid]
            if self.fail_update:
                self.fail_update = False
                raise RuntimeError('simulated update refusal')
            if row['JobState'] != 'PENDING' or row['Priority'] != '0':
                raise AssertionError('Resource updates require an owned pending hold')
            for key, value in updates.items():
                if key in ('NodeList', 'ReqNodeList'):
                    row['ReqNodeList'] = value
                elif key in ('ExcNodeList', 'Exclude'):
                    row['ExcNodeList'] = value
                else:
                    row[key] = value
            if self.crash_after_update:
                self.crash_after_update = False
                raise KeyboardInterrupt('simulated process death after request update')
            return ''
        if args[0] == 'sacctmgr':
            return 'highpriority|60|\n'
        if args[:3] == ['scontrol', 'show', 'hostnames']:
            return '\n'.join(sorted(self.expand_nodes(args[-1]))) + '\n'
        raise AssertionError('Unexpected simulated Slurm call: ' + repr(args))


class AccelerationTests(unittest.TestCase):
    def setUp(self):
        from ics import batch2_acceleration as acceleration
        self.a = acceleration
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.top = Path(self.tmp.name)
        self.runtime = self.top / 'runtime'; self.runtime.mkdir()
        self.run = self.top / 'run'; self.run.mkdir()
        env = patch.dict(os.environ, ICS_RUNTIME=str(self.runtime), ICS_BATCH2_ROOT=str(self.run))
        env.start(); self.addCleanup(env.stop)
        self.source = freeze_source(self.run, ROOT/'configs/suite.batch2.core.json',
                                    ROOT/'configs/cluster.orion.batch2.json', self.runtime)
        with patch.object(orchestration, 'PROJECT_ROOT', self.source):
            batch2.prepare(self.source, self.run, batch2.selection(self.runtime, self.run))
        self.controller = freeze_source(self.run/'acceleration-20261117',
                                        ROOT/'configs/suite.batch2.core.json',
                                        ROOT/'configs/cluster.orion.batch2.json', self.runtime)
        self.plan = rq._plan(self.run/'scopes/core/plan.json')
        self.path = Path(self.plan['run_dir'])/'plan.json'
        self.state = rq._read_state(self.plan)
        for task in self.state['tasks'].values():
            if task['kind'] in ('data', 'preflight', 'smoke'):
                task['status'] = 'PASS'
        self.fake = FakeSlurm()
        self.next_job = 301000
        self.jobs = []
        for family in ('insid3', 'cgics_mllm'):
            choices = [t for t in self.state['tasks'].values()
                       if t['kind'] == 'experiment' and t['resource'] == family]
            for task in choices[:4]:
                self.accept(task)
        self.save()
        self.protected = self.protected_bytes()
        for name, func in [('_command', self.fake.command), ('_show_job', self.fake.show_job),
                           ('_show_node', self.fake.show_node), ('_show_partition', self.fake.show_partition),
                           ('_expand_nodes', self.fake.expand_nodes),
                           ('_history', lambda *a, **k: copy.deepcopy(self.fake.history))]:
            mock = patch.object(self.a, name, side_effect=func)
            mock.start(); self.addCleanup(mock.stop)

    def accept(self, task):
        self.next_job += 1; jid = str(self.next_job)
        with patch.object(rq.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, jid+'\n', '')):
            rq._submit(self.path, self.plan, self.state, task, 10)
        profile = rq._profile(self.plan, task)
        cpus = str(profile.get('cpus', 4)); memory = profile.get('memory', '32G'); gpus = profile['gpus']
        self.fake.jobs[jid] = dict(JobId=jid, JobName=task['job_name'],
            UserId=self.state['policy']['user']+'(1234)', QOS='highpriority',
            Account=profile['account'], JobState='PENDING', Partition='h200p,l40sp',
            Command=str(Path(self.plan['run_dir'])/'rolling_slurm'/(task['key']+'.sh')),
            NumCPUs=cpus, NumNodes='1', MinMemoryNode=memory, TresPerNode='gres/gpu:'+str(gpus),
            ReqTRES=f'cpu={cpus},mem={memory},node=1,gres/gpu={gpus}', TimeLimit='12:00:00',
            ExcNodeList=profile.get('exclude', '(null)'), ReqNodeList='(null)', NodeList='(null)',
            Priority='100', Reason='Priority', Comment='(null)')
        self.jobs.append(task)
        return task

    def save(self):
        rq._atomic_json(self.path.parent/rq.STATE_NAME, self.state)

    def protected_bytes(self):
        paths = [self.run/'batch2.json', self.source/'snapshot.json']
        for scope in batch2.SCOPES:
            paths += [self.run/'scopes'/scope/'plan.json', self.run/'scopes'/scope/rq.STATE_NAME]
        return {str(p): p.read_bytes() for p in paths}

    def assert_original_unchanged(self):
        self.assertEqual(self.protected_bytes(), self.protected)
        self.assertFalse(any(c[0] in ('sbatch', 'scancel') for c in self.fake.calls))

    def prepare(self):
        return self.a.prepare(self.run, self.controller)

    def tick(self):
        return self.a.tick(self.run)

    def only_candidate(self, chosen):
        for task in self.jobs:
            if task is not chosen:
                task['status'] = 'PASS'
                self.fake.jobs[str(task['job_id'])]['JobState'] = 'COMPLETED'
        self.save(); self.protected = self.protected_bytes()

    def mutations_for(self, job_id):
        value = str(job_id)
        return [cmd for cmd in self.fake.calls
                if cmd[:2] in (['scontrol','hold'], ['scontrol','release'], ['scontrol','update'])
                and (value in cmd or 'JobId='+value in cmd or 'JobID='+value in cmd)]

    def own_state(self):
        return json.loads((self.run/'acceleration-20261117/state.json').read_text())

    def complete_canary(self, route='l40s_reboot', family='cgics_mllm', *,
                        node=None, partition=None, claim=True, marker=True, failed=False):
        item = self.own_state()['routes'][route][family]
        task = self.state['tasks'][item['key']]
        jid = task['job_id']
        self.fake.history[jid] = dict(job_id=jid, job_name=task['job_name'],
            state='FAILED' if failed else 'COMPLETED', exit_code='1:0' if failed else '0:0',
            qos='highpriority', node=node or item['node'], partition=partition or item['partition'])
        self.fake.jobs[jid].update(JobState='FAILED' if failed else 'COMPLETED',
                                   NodeList=node or item['node'])
        task['status'] = 'FAILED' if failed else 'PASS'
        if claim:
            rq._atomic_json(Path(task['batch2_intent']).with_suffix('.claim.json'),
                            dict(slurm_job_id=jid, job_name=task['job_name'],
                                 intent_sha256=task['batch2_intent_sha256']))
        if marker:
            rq._atomic_json(rq._marker_path(self.plan, task['key']),
                            dict(status='PASS', plan_fingerprint=self.plan['fingerprint'],
                                 key=task['key'], slurm_job_id=jid, job_name=task['job_name'],
                                 batch2_intent_sha256=task['batch2_intent_sha256']))
            rq._atomic_json(self.path.parent/'jobs'/task['key']/'complete.json',
                            dict(status='PASS', plan_fingerprint=self.plan['fingerprint'],
                                 task_id=task['key'], slurm_job_id=jid))
        self.save(); self.protected = self.protected_bytes()
        return task

    def request_for_canary(self, task, route='l40s_reboot'):
        policy = json.loads((self.run/'acceleration-20261117/policy.json').read_text())
        before = self.fake.jobs[task['job_id']]
        desired = self.a._desired(policy, self.plan, task, before, self.own_state()['routes'], canary=route)
        return policy, desired

    def test_prepare_is_immutable_and_does_not_touch_original_run(self):
        self.prepare()
        policy = self.run/'acceleration-20261117/policy.json'
        first = policy.read_bytes()
        self.prepare()
        self.assertEqual(policy.read_bytes(), first)
        self.assert_original_unchanged()
        self.assertFalse(self.fake.calls)

    def test_prepare_recovers_initial_policy_without_state_when_no_jobs_changed(self):
        self.prepare()
        policy = self.run/'acceleration-20261117/policy.json'
        before = policy.read_bytes()
        (self.run/'acceleration-20261117/state.json').unlink()
        self.prepare()
        self.assertEqual(policy.read_bytes(), before)
        self.assertTrue((self.run/'acceleration-20261117/state.json').is_file())
        self.assert_original_unchanged()

    def test_unavailable_one_route_does_not_block_other_route_canaries(self):
        def partition(part):
            value = self.fake.show_partition(part)
            if part == 'testqueue': value['State'] = 'DOWN'
            return value
        with patch.object(self.a, '_show_partition', side_effect=partition):
            self.prepare(); self.tick()
        routes = self.own_state()['routes']
        self.assertTrue(all(v['status']=='BLOCKED' for v in routes['testqueue_bfs'].values()))
        self.assertTrue(all(v['status']=='CANARY_SUBMITTED' for v in routes['l40s_reboot'].values()))
        self.assert_original_unchanged()

    def test_transient_node_query_error_defers_route_then_retries_successfully(self):
        self.prepare()
        def node(name):
            if name == 'A484652X3B10601':
                raise RuntimeError('transient Slurm controller timeout')
            return self.fake.show_node(name)
        with patch.object(self.a, '_show_node', side_effect=node):
            self.tick()
        routes = self.own_state()['routes']
        self.assertTrue(all(item['status']=='WAITING_CANARY' for item in routes['l40s_reboot'].values()))
        self.assertTrue(all(item['status']=='CANARY_SUBMITTED' for item in routes['testqueue_bfs'].values()))
        self.tick()
        routes = self.own_state()['routes']
        self.assertTrue(all(item['status']=='CANARY_SUBMITTED' for item in routes['l40s_reboot'].values()))
        self.assert_original_unchanged()

    def test_pending_production_canaries_require_no_extra_submission_slots(self):
        # A full user window must not lead to sbatch/test-only/cancel operations.
        for i in range(52):
            self.fake.jobs[str(900000+i)] = dict(self.fake.jobs[self.jobs[0]['job_id']],
                JobId=str(900000+i), JobName='unrelated-'+str(i))
        self.assertEqual(len(self.fake.jobs), 60)
        self.prepare(); self.tick()
        self.assertTrue(any(c[:2] == ['scontrol','hold'] for c in self.fake.calls))
        self.assertTrue(any(c[:2] == ['scontrol','update'] for c in self.fake.calls))
        self.assert_original_unchanged()

    def test_pass_task_is_never_replayed_or_resource_modified(self):
        chosen = self.jobs[0]; self.only_candidate(chosen)
        chosen['status'] = 'PASS'; self.save(); self.protected = self.protected_bytes()
        self.prepare(); self.tick()
        self.assertEqual(self.mutations_for(chosen['job_id']), [])
        self.assert_original_unchanged()

    def test_actual_running_job_is_not_modified_even_when_saved_submitted(self):
        chosen = self.jobs[0]; self.only_candidate(chosen)
        self.fake.jobs[chosen['job_id']].update(JobState='RUNNING', Reason='None')
        before = copy.deepcopy(self.fake.jobs[chosen['job_id']])
        self.prepare(); self.tick()
        self.assertEqual(self.fake.jobs[chosen['job_id']], before)
        self.assertEqual(self.mutations_for(chosen['job_id']), [])
        self.assert_original_unchanged()

    def test_preexisting_user_hold_is_neither_changed_nor_released(self):
        chosen = self.jobs[0]; self.only_candidate(chosen)
        self.fake.jobs[chosen['job_id']].update(Priority='0', Reason='JobHeldUser')
        before = copy.deepcopy(self.fake.jobs[chosen['job_id']])
        self.prepare(); self.tick()
        self.assertEqual(self.fake.jobs[chosen['job_id']], before)
        self.assertEqual(self.mutations_for(chosen['job_id']), [])
        self.assert_original_unchanged()

    def test_wrong_owner_is_not_held_or_modified(self):
        chosen = self.jobs[0]; self.only_candidate(chosen)
        self.fake.jobs[chosen['job_id']]['UserId'] = 'someoneelse(9876)'
        self.prepare(); self.tick()
        self.assertEqual(self.mutations_for(chosen['job_id']), [])
        self.assert_original_unchanged()

    def test_wrong_scheduler_command_is_not_held_or_modified(self):
        chosen = self.jobs[0]; self.only_candidate(chosen)
        self.fake.jobs[chosen['job_id']]['Command'] = '/tmp/unrelated-worker.sh'
        self.prepare(); self.tick()
        self.assertEqual(self.mutations_for(chosen['job_id']), [])
        self.assert_original_unchanged()

    def test_changed_sealed_intent_is_not_held_or_modified(self):
        chosen = self.jobs[0]; self.only_candidate(chosen)
        self.prepare()
        p = Path(chosen['batch2_intent']); p.write_text(p.read_text()+' ')
        self.tick()
        self.assertEqual(self.mutations_for(chosen['job_id']), [])
        self.assert_original_unchanged()

    def test_changed_worker_script_is_not_held_or_modified(self):
        chosen = self.jobs[0]; self.only_candidate(chosen)
        self.prepare()
        p = Path(self.fake.jobs[chosen['job_id']]['Command'])
        p.write_text(p.read_text().replace('exec ', 'echo '))
        self.tick()
        self.assertEqual(self.mutations_for(chosen['job_id']), [])
        self.assert_original_unchanged()

    def test_changed_script_header_with_valid_exec_is_rejected(self):
        chosen = self.jobs[0]; self.only_candidate(chosen)
        self.prepare()
        p = Path(self.fake.jobs[chosen['job_id']]['Command'])
        p.write_text(p.read_text().replace('set -euo pipefail\n', 'set -euo pipefail\nprintf tampered\n'))
        self.tick()
        self.assertEqual(self.mutations_for(chosen['job_id']), [])
        self.assert_original_unchanged()

    def test_existing_worker_claim_prevents_retarget(self):
        chosen = self.jobs[0]; self.only_candidate(chosen)
        self.prepare()
        Path(chosen['batch2_intent']).with_suffix('.claim.json').write_text('{}')
        self.tick()
        self.assertEqual(self.mutations_for(chosen['job_id']), [])
        self.assert_original_unchanged()

    def test_scheduler_query_failure_is_fail_closed(self):
        self.prepare(); self.fake.fail_queries = True
        try:
            self.tick()
        except RuntimeError:
            pass
        self.assertFalse(any(c[:2] in (['scontrol','hold'], ['scontrol','release'], ['scontrol','update'])
                             for c in self.fake.calls))
        self.assert_original_unchanged()

    def test_pending_to_running_hold_race_never_updates_or_cancels(self):
        chosen = self.jobs[0]; self.only_candidate(chosen)
        self.prepare(); self.fake.hold_race = True
        self.tick()
        calls = self.mutations_for(chosen['job_id'])
        self.assertTrue(any(c[:2] == ['scontrol','hold'] for c in calls))
        self.assertFalse(any(c[:2] == ['scontrol','update'] for c in calls))
        self.assertEqual(self.fake.jobs[chosen['job_id']]['JobState'], 'RUNNING')
        self.assert_original_unchanged()

    def test_failed_update_restores_request_and_releases_own_hold(self):
        chosen = self.jobs[0]; self.only_candidate(chosen)
        self.prepare()
        before = copy.deepcopy(self.fake.jobs[chosen['job_id']])
        self.fake.fail_update = True
        self.tick()
        after = self.fake.jobs[chosen['job_id']]
        for key in ('Partition','TimeLimit','ReqNodeList','ExcNodeList','Priority','Reason'):
            self.assertEqual(after[key], before[key], key)
        self.assertTrue(any(c[:2] == ['scontrol','release'] for c in self.mutations_for(chosen['job_id'])))
        self.assert_original_unchanged()

    def test_start_race_after_acknowledged_hold_releases_hold_without_resources_change(self):
        chosen = self.jobs[0]; self.only_candidate(chosen)
        self.prepare(); self.fake.race_after_successful_hold = True
        before = copy.deepcopy(self.fake.jobs[chosen['job_id']])
        self.tick()
        after = self.fake.jobs[chosen['job_id']]
        for key in ('Partition','TimeLimit','ReqNodeList','ExcNodeList'):
            self.assertEqual(after[key], before[key], key)
        self.assertEqual(after['JobState'], 'RUNNING')
        self.assertEqual(after['Priority'], '100')
        self.assertTrue(any(c[:2] == ['scontrol','release'] for c in self.mutations_for(chosen['job_id'])))
        self.assert_original_unchanged()

    def test_crash_after_hold_is_recovered_without_replaying_or_leaving_user_hold(self):
        chosen = self.jobs[0]; self.only_candidate(chosen); self.prepare()
        policy, desired = self.request_for_canary(chosen)
        self.fake.crash_after_hold = True
        with self.assertRaises(KeyboardInterrupt):
            self.a._retarget(self.run, policy, self.plan, self.state, chosen, desired, 'canary:l40s_reboot:insid3')
        self.assertEqual(self.fake.jobs[chosen['job_id']]['Priority'], '0')
        result = self.a.recover_transactions(self.run)
        self.assertFalse(result['recovery_required'])
        self.assertEqual(self.fake.jobs[chosen['job_id']]['Priority'], '100')
        self.assertEqual(self.fake.jobs[chosen['job_id']]['ReqNodeList'], '(null)')
        self.assert_original_unchanged()

    def test_crash_after_update_rolls_back_original_request_then_releases(self):
        chosen = self.jobs[0]; self.only_candidate(chosen); self.prepare()
        policy, desired = self.request_for_canary(chosen)
        before = self.a._request(self.fake.jobs[chosen['job_id']])
        self.fake.crash_after_update = True
        with self.assertRaises(KeyboardInterrupt):
            self.a._retarget(self.run, policy, self.plan, self.state, chosen, desired, 'canary:l40s_reboot:insid3')
        self.assertEqual(self.fake.jobs[chosen['job_id']]['ReqNodeList'], 'A484652X3B10601')
        self.assertEqual(self.fake.jobs[chosen['job_id']]['Priority'], '0')
        result = self.a.recover_transactions(self.run)
        self.assertFalse(result['recovery_required'])
        self.assertEqual(self.a._request(self.fake.jobs[chosen['job_id']]), before)
        self.assertEqual(self.fake.jobs[chosen['job_id']]['Priority'], '100')
        self.assert_original_unchanged()

    def test_known_canary_uses_bounded_time_and_pending_production_is_retimed(self):
        self.prepare(); self.tick()
        own = self.own_state(); canary_ids = set()
        for route, families in own['routes'].items():
            for family, item in families.items():
                self.assertEqual(item['status'], 'CANARY_SUBMITTED')
                self.assertNotIn(item['job_id'], canary_ids); canary_ids.add(item['job_id'])
                record = self.fake.jobs[item['job_id']]
                self.assertEqual(record['ReqNodeList'], self.a.ROUTES[route]['node'])
                self.assertEqual(record['Partition'], self.a.ROUTES[route]['partition'])
                self.assertEqual(record['TimeLimit'], '01:00:00' if family=='insid3' else '02:00:00')
                self.assertNotEqual(record['Priority'], '0')
        for task in self.jobs:
            if task['job_id'] not in canary_ids:
                row = self.fake.jobs[task['job_id']]
                self.assertEqual(row['TimeLimit'], '00:30:00' if task['resource']=='insid3' else '02:00:00')
                self.assertIn('A484652X3B10601', row['ExcNodeList'])
                self.assertNotIn('testqueue', row['Partition'])
        self.assert_original_unchanged()

    def test_unknown_configuration_is_not_selected_for_canary(self):
        # Future frozen plans may contain additional IDs; do not infer their
        # runtime from today's thirteen measured configurations.
        chosen = self.jobs[0]; self.only_candidate(chosen); self.prepare()
        known = self.a.KNOWN_CONFIGS
        group = [t for t in self.plan['tasks'] if t['resource'] == chosen['resource']]
        identifier = group[chosen['index']]['config_data']['experiment_id']
        with patch.object(self.a, 'KNOWN_CONFIGS', frozenset(known - {identifier})):
            self.tick()
        own = self.own_state()
        for families in own['routes'].values():
            self.assertNotIn('job_id', families['insid3'])
        self.assertEqual(self.fake.jobs[chosen['job_id']]['TimeLimit'], '12:00:00')
        self.assert_original_unchanged()

    def test_full_canary_proof_opens_only_matching_route_and_family(self):
        self.prepare(); self.tick()
        canary = self.complete_canary()
        self.fake.calls.clear(); self.tick(); own = self.own_state()
        self.assertEqual(own['routes']['l40s_reboot']['cgics_mllm']['status'], 'VERIFIED')
        self.assertNotEqual(own['routes']['l40s_reboot']['insid3']['status'], 'VERIFIED')
        self.assertNotEqual(own['routes']['testqueue_bfs']['cgics_mllm']['status'], 'VERIFIED')
        canary_ids = {item['job_id'] for families in own['routes'].values() for item in families.values()}
        for task in self.jobs:
            if task['job_id'] in canary_ids: continue
            row = self.fake.jobs[task['job_id']]
            if task['resource'] == 'cgics_mllm':
                self.assertNotIn('A484652X3B10601', row['ExcNodeList'])
                self.assertIn('2FS9B74', row['ExcNodeList'])
            else:
                self.assertIn('A484652X3B10601', row['ExcNodeList'])
        self.assertEqual(self.mutations_for(canary['job_id']), [])
        self.assert_original_unchanged()

    def test_testqueue_proof_excludes_untested_testqueue_nodes(self):
        self.prepare(); self.tick(); self.complete_canary(route='testqueue_bfs')
        self.tick(); own = self.own_state()
        self.assertEqual(own['routes']['testqueue_bfs']['cgics_mllm']['status'], 'VERIFIED')
        canary_ids = {item['job_id'] for families in own['routes'].values() for item in families.values()}
        for task in self.jobs:
            if task['resource']=='cgics_mllm' and task['job_id'] not in canary_ids:
                row = self.fake.jobs[task['job_id']]
                self.assertIn('testqueue', row['Partition'])
                self.assertIn('C64Y574', row['ExcNodeList'])
                self.assertNotIn('BFS9B74', row['ExcNodeList'])
                self.assertIn('A484652X3B10601', row['ExcNodeList'])
        self.assert_original_unchanged()

    def test_accounting_completion_on_wrong_node_does_not_open_route(self):
        self.prepare(); self.tick(); self.complete_canary(node='A484652X3B00163')
        self.tick()
        self.assertEqual(self.own_state()['routes']['l40s_reboot']['cgics_mllm']['status'], 'BLOCKED')
        self.assert_original_unchanged()

    def test_accounting_completion_without_exact_claim_does_not_open_route(self):
        self.prepare(); self.tick(); self.complete_canary(claim=False)
        self.tick()
        self.assertNotEqual(self.own_state()['routes']['l40s_reboot']['cgics_mllm']['status'], 'VERIFIED')
        self.assert_original_unchanged()

    def test_accounting_completion_without_worker_markers_waits_for_evidence(self):
        self.prepare(); self.tick(); self.complete_canary(marker=False)
        self.tick()
        self.assertEqual(self.own_state()['routes']['l40s_reboot']['cgics_mllm']['status'], 'CANARY_SUBMITTED')
        self.assert_original_unchanged()

    def test_canary_failure_blocks_only_its_route_and_family(self):
        self.prepare(); self.tick(); self.complete_canary(failed=True, marker=False)
        self.tick(); routes = self.own_state()['routes']
        self.assertEqual(routes['l40s_reboot']['cgics_mllm']['status'], 'BLOCKED')
        self.assertEqual(routes['l40s_reboot']['insid3']['status'], 'CANARY_SUBMITTED')
        self.assertEqual(routes['testqueue_bfs']['cgics_mllm']['status'], 'CANARY_SUBMITTED')
        self.assertEqual(routes['testqueue_bfs']['insid3']['status'], 'CANARY_SUBMITTED')
        self.assert_original_unchanged()

    def test_reboot_invalidates_verified_route_and_restores_pending_exclusion(self):
        self.prepare(); self.tick(); self.complete_canary(); self.tick()
        own = self.own_state()
        self.assertEqual(own['routes']['l40s_reboot']['cgics_mllm']['status'], 'VERIFIED')
        canary_ids = {item['job_id'] for families in own['routes'].values() for item in families.values()}
        pending = [t for t in self.jobs if t['resource']=='cgics_mllm' and t['job_id'] not in canary_ids]
        self.assertTrue(pending)
        self.assertTrue(all('A484652X3B10601' not in self.fake.jobs[t['job_id']]['ExcNodeList'] for t in pending))
        self.fake.boot_times['A484652X3B10601'] = '2026-11-17T14:00:00'
        self.tick(); own = self.own_state()
        self.assertEqual(own['routes']['l40s_reboot']['cgics_mllm']['status'], 'BLOCKED')
        self.assertTrue(all('A484652X3B10601' in self.fake.jobs[t['job_id']]['ExcNodeList'] for t in pending))
        self.assert_original_unchanged()

    def test_reboot_withdraws_unstarted_pinned_canaries_to_original_route(self):
        self.prepare(); self.tick()
        canaries = self.own_state()['routes']['l40s_reboot']
        self.fake.boot_times['A484652X3B10601'] = '2026-11-17T14:00:00'
        self.tick()
        for item in canaries.values():
            record = self.fake.jobs[item['job_id']]
            self.assertEqual(record['Partition'], 'h200p,l40sp')
            self.assertIn(record['ReqNodeList'], ('', '(null)'))
            self.assertIn('A484652X3B10601', record['ExcNodeList'])
            self.assertEqual(record['Priority'], '100')
        self.assert_original_unchanged()

    def test_release_before_canary_state_save_crash_relinks_journal_for_withdrawal(self):
        self.prepare()
        save = self.a._save
        crashed = []
        def interrupted_save(run, state):
            for route, families in state['routes'].items():
                for family, item in families.items():
                    if not crashed and item['status']=='CANARY_SUBMITTED':
                        journal = json.loads(Path(item['journal']).read_text())
                        self.assertEqual(journal['phase'], 'RELEASED')
                        crashed.append((route, family, copy.deepcopy(item), journal['before_request']))
                        raise KeyboardInterrupt('process died before canary journal link was persisted')
            return save(run, state)
        with patch.object(self.a, '_save', side_effect=interrupted_save):
            with self.assertRaises(KeyboardInterrupt):
                self.tick()
        route, family, item, original = crashed[0]
        saved = self.own_state()['routes'][route][family]
        self.assertEqual(saved['status'], 'CANARY_SELECTED')
        self.assertNotIn('journal', saved)
        self.assertEqual(self.fake.jobs[item['job_id']]['ReqNodeList'], self.a.ROUTES[route]['node'])
        self.fake.boot_times[self.a.ROUTES[route]['node']] = '2026-11-17T14:00:00'
        self.tick()
        self.assertEqual(self.a._request(self.fake.jobs[item['job_id']]), original)
        self.assertEqual(self.fake.jobs[item['job_id']]['Priority'], '100')
        self.assertTrue(self.own_state()['routes'][route][family]['withdrawn'])
        self.assert_original_unchanged()

    def test_changed_original_frozen_source_fails_before_scheduler_mutation(self):
        self.prepare()
        target = self.source/'ics/adapters/insid3.py'
        target.write_text(target.read_text()+'\n# unexpected source change\n')
        with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
            self.tick()
        self.assertEqual(self.fake.calls, [])

    def test_missing_configuration_smoke_pass_refuses_pending_retarget(self):
        chosen = self.jobs[0]; self.only_candidate(chosen)
        self.state['tasks'][chosen['dependencies'][0]]['status'] = 'FAILED'
        self.save(); self.protected = self.protected_bytes()
        self.prepare(); self.tick()
        self.assertEqual(self.mutations_for(chosen['job_id']), [])
        self.assert_original_unchanged()


if __name__ == '__main__':
    unittest.main()
