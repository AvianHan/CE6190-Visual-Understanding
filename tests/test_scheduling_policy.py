"""Scheduling changes preserve scientific gates, accepted IDs and shared limits."""
import copy
import json
from pathlib import Path
import pwd
import os
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ics.orchestration import fingerprint
from ics.scheduling import (effective_profile, fair_tasks, hardware_evidence, load_policy,
                            partition_evidence, qos_limit, retarget_pending_smokes)


class SchedulingPolicyTests(unittest.TestCase):
    def fixture(self, root):
        profiles = {name: {'account': 'project', 'partition': 'h200p', 'qos': 'highpriority',
                          'time': '12:00:00', 'gpus': gpus, 'exclude': 'badnode'}
                    for name, gpus in [('insid3', 1), ('cgics_mllm', 2)]}
        plan = {'run_dir': str(root), 'project_root': str(root / 'source'), 'fingerprint': 'bound',
                'cluster': {'launcher_python': 'python3'}, 'resources': profiles,
                'manifests': {'coco': {'cohort': 'standard'}, 'lvis': {'cohort': 'standard'}},
                'smoke': [], 'tasks': []}
        state = {'created_at': 1000, 'policy': {'user': pwd.getpwuid(os.getuid()).pw_name,
                    'qos': 'highpriority', 'shared_limit': 60}, 'tasks': {}}
        for cid, resource in [('cgics_full', 'cgics_mllm'), ('insid3_full', 'insid3')]:
            deps = []
            for dataset in ['coco', 'lvis']:
                key = 'smoke-' + cid + '-' + dataset
                deps.append(key)
                plan['smoke'].append({'id': key, 'resource': resource, 'manifest_id': dataset,
                                      'config_data': {'experiment_id': cid}})
                state['tasks'][key] = {'key': key, 'job_name': 'job-' + key, 'status': 'PASS',
                    'kind': 'smoke', 'resource': resource, 'dependencies': [], 'job_id': str(len(state['tasks']) + 100)}
            for index in range(3):
                key = cid + '-' + str(index)
                plan['tasks'].append({'id': key, 'resource': resource, 'manifest_id': 'coco',
                                     'config_data': {'experiment_id': cid}})
                state['tasks'][key] = {'key': key, 'job_name': 'job-' + key, 'status': 'PENDING',
                    'kind': 'experiment', 'resource': resource, 'dependencies': deps[:]}
        policy = {'plan_fingerprint': 'bound', 'worker_source': plan['project_root'],
                  'shared_limit': 60, 'inherited_shared_limit': 60, 'verified_bound': 60,
                  'qos_evidence': {'status': 'VERIFIED', 'actual_limit': 60},
                  'smoke_partition': 'l40sp', 'formal_partitions': 'h200p,l40sp'}
        return plan, state, policy

    def test_policy_binding_and_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, state, policy = self.fixture(Path(temporary))
            path = Path(temporary) / 'scheduling' / 'policy.json'
            path.parent.mkdir()
            def save(value):
                value = dict(value)
                value['fingerprint'] = fingerprint(value)
                path.write_text(json.dumps(value))
            save(policy)
            self.assertEqual(load_policy(plan)['shared_limit'], 60)
            altered = dict(policy, plan_fingerprint='other')
            save(altered)
            with self.assertRaisesRegex(ValueError, 'another plan'): load_policy(plan)
            altered = dict(policy, worker_source='/another/source')
            save(altered)
            with self.assertRaisesRegex(ValueError, 'worker source'): load_policy(plan)
            save(policy)
            data = json.loads(path.read_text()); data['shared_limit'] = 80
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'fingerprint'): load_policy(plan)
            save(dict(policy, shared_limit=80))
            with self.assertRaisesRegex(ValueError, 'bound'): load_policy(plan)

    def test_live_qos_bound_can_increase_or_decrease(self):
        for actual in (20, 60, 120):
            with self.subTest(actual=actual), patch('ics.scheduling._command', return_value=f'highpriority|{actual}|\n'):
                evidence = qos_limit('highpriority', 60)
                self.assertEqual(evidence['status'], 'VERIFIED')
                self.assertEqual(evidence['effective_limit'], actual)

    def test_qos_unknown_retains_historical_bound_with_error(self):
        with patch('ics.scheduling._command', side_effect=RuntimeError('accounting unavailable')):
            evidence = qos_limit('highpriority', 60)
        self.assertEqual(evidence['effective_limit'], 60)
        self.assertEqual(evidence['status'], 'INHERITED_BOUND_QUERY_FAILED')
        self.assertIn('accounting unavailable', evidence['error'])
        with patch('ics.scheduling._command', return_value='highpriority||\n'):
            self.assertEqual(qos_limit('highpriority', 60)['status'], 'UNBOUNDED_FIELD')

    def test_exact_two_hardware_gates_and_unchanged_formal_resources(self):
        plan, state, policy = self.fixture(Path('/tmp/run'))
        task = state['tasks']['cgics_full-0']
        hardware = {state['tasks'][key]['job_id']: {'partition': 'l40sp', 'state': 'COMPLETED', 'exit_code': '0:0'}
                    for key in task['dependencies']}
        original = copy.deepcopy(plan)
        profile = effective_profile(plan, state, task, policy, hardware)
        self.assertEqual(profile['partition'], 'h200p,l40sp')
        self.assertEqual(profile['time'], '12:00:00')
        self.assertEqual(profile['gpus'], 2)
        self.assertEqual(profile['exclude'], 'badnode')
        del hardware[next(iter(hardware))]
        self.assertEqual(effective_profile(plan, state, task, policy, hardware)['partition'], 'h200p')
        self.assertEqual(plan, original)
        smoke = state['tasks'][task['dependencies'][0]]
        new_profile = effective_profile(plan, state, smoke, policy, {})
        self.assertEqual((new_profile['partition'], new_profile['time']), ('l40sp', '01:00:00'))

    def test_failed_hardware_accounting_restricts_to_original_partition(self):
        plan, state, policy = self.fixture(Path('/tmp/run'))
        with patch('ics.scheduling._command', side_effect=RuntimeError('timeout')):
            hardware, error = hardware_evidence(state)
        self.assertEqual(hardware, {})
        self.assertIn('timeout', error)
        self.assertEqual(effective_profile(plan, state, state['tasks']['insid3_full-0'], policy, hardware)['partition'], 'h200p')

    def test_cross_day_hardware_query_has_explicit_start(self):
        plan, state, policy = self.fixture(Path('/tmp/run'))
        with patch('ics.scheduling._command', return_value='100|l40sp|COMPLETED|0:0\n') as command:
            hardware, error = hardware_evidence(state)
        self.assertIsNone(error)
        self.assertIn('--starttime', command.call_args.args[0])
        self.assertEqual(hardware['100']['partition'], 'l40sp')

    def test_single_freed_slot_alternates_main_families(self):
        plan, state, policy = self.fixture(Path('/tmp/run'))
        def next_ready():
            return next(t for t in fair_tasks(plan, state) if t['kind'] == 'experiment' and t['status'] == 'PENDING')
        first = next_ready()
        first.update(status='SUBMITTED', job_id='200')
        second = next_ready()
        self.assertNotEqual(first['resource'], second['resource'])
        # Even if the first job finishes before the next window, historical
        # submissions break equal-active ties and prevent repeated same-family wins.
        first['status'] = 'PASS'
        self.assertNotEqual(first['resource'], next_ready()['resource'])

    def test_pending_update_preserves_id_and_does_not_change_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, state, policy = self.fixture(root)
            task = next(t for t in state['tasks'].values() if t['kind'] == 'smoke')
            task['status'] = 'SUBMITTED'
            jid = task['job_id']
            path = root / 'rolling_slurm' / (task['key'] + '.sh')
            path.parent.mkdir()
            path.write_text('exec ' + shlex.join(['python3', str(root / 'source/scripts/queue_ics.py'),
                'worker', '--plan', str(root / 'plan.json'), '--key', task['key']]) + '\n')
            def row(status='PENDING', partition='h200p', account='project'):
                return (f'JobId={jid} JobName={task["job_name"]} UserId={state["policy"]["user"]}(1) '
                        f'QOS=highpriority Account={account} Command={path} JobState={status} Partition={partition}')
            queue = {jid: {'state': 'PENDING'}}
            with patch('ics.scheduling._command', side_effect=[row(), row(partition='l40sp')]), patch(
                    'ics.scheduling.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '', '')) as update:
                events = retarget_pending_smokes(plan, state, policy, queue)
            self.assertEqual(events[0]['status'], 'UPDATED')
            self.assertEqual(task['job_id'], jid)
            self.assertEqual(update.call_args.args[0], ['scontrol', 'update', 'JobId=' + jid, 'Partition=l40sp'])
            with patch('ics.scheduling._command', return_value=row(status='RUNNING')), patch('ics.scheduling.subprocess.run') as update:
                self.assertEqual(retarget_pending_smokes(plan, state, policy, queue), [])
                update.assert_not_called()
            with patch('ics.scheduling._command', return_value=row(account='other')), patch('ics.scheduling.subprocess.run') as update:
                self.assertEqual(retarget_pending_smokes(plan, state, policy, queue)[0]['status'], 'DEFERRED')
                update.assert_not_called()
            # Race after validation: Slurm rejects the partition change; it is not
            # resubmitted and no time limit mutation is sent.
            with patch('ics.scheduling._command', side_effect=[row(), row(status='RUNNING')]), patch(
                    'ics.scheduling.subprocess.run', return_value=subprocess.CompletedProcess([], 1, '', 'no longer pending')):
                self.assertEqual(retarget_pending_smokes(plan, state, policy, queue)[0]['status'], 'DEFERRED')
            self.assertEqual(task['job_id'], jid)

    def test_partition_unavailable_rejected_before_handover(self):
        plan, state, policy = self.fixture(Path('/tmp/run'))
        output = 'PartitionName=h200p State=UP AllowQos=highpriority AllowAccounts=ALL\n'
        with patch('ics.scheduling._command', side_effect=[output, 'PartitionName=l40sp State=DOWN AllowQos=ALL AllowAccounts=ALL']):
            with self.assertRaisesRegex(ValueError, 'not UP'): partition_evidence(plan, state)


if __name__ == '__main__':
    unittest.main()
