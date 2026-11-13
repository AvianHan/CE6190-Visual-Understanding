"""Recovery tests use archived real plan if ICS_RECOVERY_AUDIT_RUN is provided."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from ics.orchestration import atomic_json, fingerprint, write_plan
from ics.recovery import (build_recovery_plan, manifest_evidence, parent_evidence,
                          assert_parent_inactive, verify_recovery_assets, assert_no_partial_success)
from ics.rolling_queue import _task_table, initialize_queue, execute_queue_task, queue_status
from ics_eval import manifest_fingerprint


def fixture(run):
    cfg = {'experiment_id': 'insid3_full', 'adapter': {'name': 'dummy'}}
    manifests = {f'{d}_fold0_standard': {'id': f'{d}_fold0_standard', 'path': str(run.parent / f'{d}.json'),
        'dataset_root': str(run.parent / 'data'), 'cohort': 'standard', 'prepare': {'dataset': d}}
        for d in ('coco', 'lvis')}
    profile = {'python': sys.executable, 'account': 'a', 'partition': 'p', 'time': '01:00:00',
               'qos': 'q', 'gpus': 1, 'max_parallel': 1}
    tasks = [dict(id=f'insid3_full--{mid}--0', resource='insid3', config=str(run / 'configs/full.json'),
              config_data=cfg, manifest_id=mid, manifest=m['path'], dataset_root=m['dataset_root'],
              output=str(run / 'results' / mid), shard_index=0, num_shards=1, max_episodes=None,
              phase='experiment') for mid, m in manifests.items()]
    smoke = dict(tasks[0], id='smoke-insid3_full', phase='smoke', max_episodes=2,
                 output=str(run / 'smoke/full'))
    cluster = {'schema_version': 1, 'resources': {'preflight': dict(profile, gpus=0), 'insid3': profile}}
    plan = {'schema_version': 1, 'run_dir': str(run), 'project_root': str(ROOT),
        'cluster': cluster, 'resources': {'insid3': profile}, 'manifests': manifests,
        'smoke': [smoke], 'tasks': tasks, 'prepare_missing_manifests': True}
    plan['fingerprint'] = fingerprint(plan)
    state = {'tasks': {tasks[0]['id']: {'status': 'FAILED'}, tasks[1]['id']: {'status': 'PASS'}}}
    return plan, state


def candidate(old, run):
    plan = copy.deepcopy(old)
    original = Path(old['run_dir'])
    plan['run_dir'] = str(run)
    for task in plan['tasks'] + plan['smoke']:
        for key in ('config', 'output'):
            task[key] = str(run / Path(task[key]).relative_to(original))
    plan['fingerprint'] = fingerprint({k: v for k, v in plan.items() if k != 'fingerprint'})
    return plan


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old, self.state = fixture(self.root / 'old')
        self.new = candidate(self.old, self.root / 'new')

    def build(self):
        return build_recovery_plan(self.new, self.old, self.state, self.root / 'old', {}, {},
                                   expected_missing=1, expected_preserved=1)

    def test_preserves_success_and_two_dataset_gates(self):
        before = copy.deepcopy(self.old)
        plan = self.build()
        self.assertEqual(self.old, before)
        self.assertEqual([t['id'] for t in plan['tasks']], [self.old['tasks'][0]['id']])
        self.assertEqual(len(plan['smoke']), 2)
        table = _task_table(plan, 'test')
        self.assertEqual(table[plan['tasks'][0]['id']]['dependencies'],
                         ['smoke-insid3_full-coco', 'smoke-insid3_full-lvis'])
        self.assertFalse(plan['prepare_missing_manifests'])

    def test_rejects_scientific_drift_and_count_drift(self):
        self.new['tasks'][0]['config_data']['adapter']['changed'] = True
        with self.assertRaisesRegex(ValueError, 'Scientific task changed'):
            self.build()
        self.new = candidate(self.old, self.root / 'new')
        self.state['tasks'][self.old['tasks'][1]['id']]['status'] = 'BLOCKED'
        with self.assertRaisesRegex(ValueError, 'Audited counts changed'):
            self.build()

    def test_idempotent_queue_preserves_pass_on_resume(self):
        plan = self.build()
        path = write_plan(plan)
        state = initialize_queue(path, max_inflight=2, shared_limit=10)
        state['tasks']['data']['status'] = 'PASS'
        atomic_json(Path(plan['run_dir']) / 'rolling_state.json', state)
        repeated = initialize_queue(path, max_inflight=2, shared_limit=10)
        self.assertEqual(repeated['tasks']['data']['status'], 'PASS')
        self.assertNotIn(self.old['tasks'][1]['id'], repeated['tasks'])
        self.assertEqual(len(repeated['tasks']), 6)

    def test_recovery_data_worker_never_downloads(self):
        plan = self.build()
        path = write_plan(plan)
        state = initialize_queue(path, max_inflight=2, shared_limit=10)
        task = state['tasks']['data']
        task.update(status='RUNNING', job_id='42')
        atomic_json(Path(plan['run_dir']) / 'rolling_state.json', state)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '42', 'SLURM_JOB_NAME': task['job_name']}), \
             patch('ics.recovery.verify_recovery_assets') as verify, \
             patch('ics.rolling_queue.subprocess.run') as command:
            execute_queue_task(path, 'data')
        verify.assert_called_once()
        command.assert_not_called()
        marker = json.loads((Path(plan['run_dir']) / 'rolling_markers/data.json').read_text())
        self.assertEqual(marker['status'], 'PASS')

    def test_failed_asset_check_fails_gate(self):
        plan = self.build()
        path = write_plan(plan)
        state = initialize_queue(path, max_inflight=2, shared_limit=10)
        task = state['tasks']['data']; task.update(status='RUNNING', job_id='42')
        atomic_json(Path(plan['run_dir']) / 'rolling_state.json', state)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '42', 'SLURM_JOB_NAME': task['job_name']}), \
             patch('ics.recovery.verify_recovery_assets', side_effect=ValueError('changed')):
            with self.assertRaisesRegex(ValueError, 'changed'):
                execute_queue_task(path, 'data')
        marker = json.loads((Path(plan['run_dir']) / 'rolling_markers/data.json').read_text())
        self.assertEqual(marker['status'], 'FAIL')

    def test_manifest_mutation_rejected_without_download(self):
        plan = self.build()
        (self.root / 'data').mkdir()
        evidence = {}
        for mid, item in plan['manifests'].items():
            manifest = {'schema_version': 1, 'status': 'ready', 'protocol_id': 'p', 'episodes': [{'id': 1}]}
            p = Path(item['path']); atomic_json(p, manifest)
            atomic_json(p.with_suffix('.protocol.json'), {})
            evidence[mid] = {'sha256': manifest_fingerprint(manifest), 'episodes': 1}
        plan['recovery']['manifest_evidence'] = evidence
        with patch('ics.recovery.verify_recovery_parent'):
            verify_recovery_assets(plan)
            p = Path(next(iter(plan['manifests'].values()))['path'])
            data = json.loads(p.read_text()); data['protocol_id'] = 'changed'; atomic_json(p, data)
            with self.assertRaisesRegex(ValueError, 'Original manifest changed'):
                verify_recovery_assets(plan)

    def test_live_or_unknown_parent_job_blocks(self):
        state = {'tasks': {'x': {'key': 'x', 'job_name': 'x', 'job_id': '42', 'status': 'PASS'}}}
        with patch('ics.rolling_queue._scheduler_snapshot', return_value=({'42': {'job_name': 'x'}}, {})):
            with self.assertRaisesRegex(RuntimeError, 'live scheduler'):
                assert_parent_inactive(self.root, self.old, state)
        with patch('ics.rolling_queue._scheduler_snapshot', return_value=({}, {})):
            with self.assertRaisesRegex(RuntimeError, 'Cannot verify terminal'):
                assert_parent_inactive(self.root, self.old, state)
        with patch('ics.rolling_queue._scheduler_snapshot', return_value=({}, {
            '42': {'job_name': 'x', 'state': 'COMPLETED', 'exit_code': '0:0'}})):
            assert_parent_inactive(self.root, self.old, state)

    def test_plain_start_rejects_recovery_before_rebuild(self):
        import queue_ics
        from argparse import Namespace
        plan = self.build(); write_plan(plan)
        with patch('queue_ics.build_plan') as build:
            with self.assertRaisesRegex(ValueError, 'Selective recovery'):
                queue_ics.start(Namespace(run_dir=Path(plan['run_dir']), runtime=self.root))
        build.assert_not_called()

    def test_another_recovery_rejected(self):
        import recover_ics
        plan = self.build()
        path = self.root / 'other' / 'plan.json'
        atomic_json(path, plan)
        with self.assertRaisesRegex(ValueError, 'Another recovery already'):
            recover_ics.assert_unique_recovery(self.root / 'old', self.root / 'new', self.old['fingerprint'])

    def test_prepare_publishes_and_repeats_without_submission_or_reset(self):
        import recover_ics
        from argparse import Namespace
        parent, run, runtime = self.root / 'old', self.root / 'new', self.root
        atomic_json(parent / 'source/snapshot.json', {'selection': {'runtime': str(runtime)}})
        atomic_json(run / 'source/snapshot.json', {'selection': {'runtime': str(runtime),
                    'suite': 'suite.json', 'cluster': 'cluster.json'}})
        self.state['policy'] = {'max_inflight': 2, 'shared_limit': 10, 'reserved_slots': 0, 'data_workers': 2}
        args = Namespace(from_run=parent, run_dir=run, runtime=runtime, suite=Path('suite.json'),
                         cluster=Path('cluster.json'), expected_parent_fingerprint=self.old['fingerprint'],
                         expected_missing=1, expected_preserved=1)
        with patch.dict(os.environ), \
             patch.object(recover_ics, 'ROOT', run / 'source'), \
             patch.object(recover_ics, 'parent_evidence', return_value=(self.old, self.state, {})), \
             patch.object(recover_ics, 'assert_parent_inactive'), \
             patch.object(recover_ics, 'verify_snapshot'), \
             patch.object(recover_ics, 'manifest_evidence', return_value=({}, 'preflight.log', 'digest')), \
             patch.object(recover_ics, 'build_plan', return_value=self.new), \
             patch.object(recover_ics, 'print_json'), \
             patch.object(recover_ics.subprocess, 'Popen') as popen:
            self.assertEqual(recover_ics.prepare(args), 0)
            path = run / 'rolling_state.json'
            state = json.loads(path.read_text()); state['tasks']['data']['status'] = 'PASS'; atomic_json(path, state)
            plan_before = (run / 'plan.json').read_bytes()
            self.assertEqual(recover_ics.prepare(args), 0)
            self.assertEqual(json.loads(path.read_text())['tasks']['data']['status'], 'PASS')
            self.assertEqual((run / 'plan.json').read_bytes(), plan_before)
        popen.assert_not_called()

    def test_frozen_prepare_rejects_configuration_outside_snapshot(self):
        import recover_ics
        from argparse import Namespace
        parent, run, runtime = self.root / 'old', self.root / 'new', self.root
        atomic_json(parent / 'source/snapshot.json', {'selection': {'runtime': str(runtime)}})
        atomic_json(run / 'source/snapshot.json', {'selection': {'runtime': str(runtime),
                    'suite': 'suite.json', 'cluster': 'cluster.json'}})
        args = Namespace(from_run=parent, run_dir=run, runtime=runtime,
                         suite=self.root / 'external_suite.json', cluster=Path('cluster.json'),
                         expected_parent_fingerprint=self.old['fingerprint'],
                         expected_missing=1, expected_preserved=1)
        with patch.dict(os.environ), \
             patch.object(recover_ics, 'ROOT', run / 'source'), \
             patch.object(recover_ics, 'parent_evidence', return_value=(self.old, self.state, {})), \
             patch.object(recover_ics, 'assert_parent_inactive'), \
             patch.object(recover_ics, 'verify_snapshot'), \
             patch.object(recover_ics, 'build_plan') as build:
            with self.assertRaisesRegex(ValueError, 'Frozen recovery selection'):
                recover_ics.prepare(args)
        build.assert_not_called()
        self.assertFalse((run / 'plan.json').exists())

    def test_scheduler_waits_for_both_dataset_smokes_and_excludes_parent_pass(self):
        from ics.rolling_queue import tick
        plan = self.build(); path = write_plan(plan)
        state = initialize_queue(path, max_inflight=2, shared_limit=10)
        for key in ('data', 'preflight', 'smoke-insid3_full-coco'):
            state['tasks'][key]['status'] = 'PASS'
        atomic_json(Path(plan['run_dir']) / 'rolling_state.json', state)
        submissions = []
        def submit(plan_path, current_plan, current_state, task, now):
            submissions.append(task['key']); task.update(status='SUBMITTED', job_id=str(100 + len(submissions)))
        with patch('ics.rolling_queue._scheduler_snapshot', return_value=({}, {})), \
             patch('ics.rolling_queue._submit', side_effect=submit):
            tick(path)
            self.assertEqual(submissions, ['smoke-insid3_full-lvis'])
            saved = json.loads((Path(plan['run_dir']) / 'rolling_state.json').read_text())
            saved['tasks']['smoke-insid3_full-lvis']['status'] = 'PASS'
            atomic_json(Path(plan['run_dir']) / 'rolling_state.json', saved)
            tick(path)
        self.assertEqual(submissions, ['smoke-insid3_full-lvis', self.old['tasks'][0]['id']])
        self.assertNotIn(self.old['tasks'][1]['id'], submissions)


@unittest.skipUnless(os.environ.get('ICS_RECOVERY_AUDIT_RUN'), 'real evidence path not supplied')
class ArchivedEvidenceTests(unittest.TestCase):
    def test_real_1392_missing_92_preserved_60_smokes(self):
        parent = Path(os.environ['ICS_RECOVERY_AUDIT_RUN']).resolve()
        fp = 'd726d16a8a6b8a5f78eda76f7b90ff0c10c0c8a327a1bda63b8151aa1b3b4e84'
        old, state, evidence = parent_evidence(parent, expected_fingerprint=fp)
        manifests, log_name, log_hash = manifest_evidence(parent, old, state)
        evidence[log_name] = log_hash
        before = {name: hashlib.sha256((parent / name).read_bytes()).hexdigest() for name in evidence}
        with tempfile.TemporaryDirectory() as tmp:
            new = candidate(old, Path(tmp) / 'recovery')
            plan = build_recovery_plan(new, old, state, parent, evidence, manifests,
                                       expected_missing=1392, expected_preserved=92)
            self.assertEqual(len(plan['smoke']), 60)
            self.assertEqual(plan['recovery']['original_status_counts'], {'BLOCKED': 1372, 'FAILED': 20})
            assert_no_partial_success(parent, old, plan['recovery']['selected_task_ids'])
            table = _task_table(plan, 'test')
            self.assertEqual(len(table), 1455)
            experiment_keys = [k for k, t in table.items() if t['kind'] == 'experiment']
            main_standard = [t['id'] for t in plan['tasks'] if t['config_data']['experiment_id'] in
                             {'insid3_full', 'cgics_full'} and plan['manifests'][t['manifest_id']]['cohort'] == 'standard']
            self.assertEqual(experiment_keys[:len(main_standard)], main_standard)
            positions = {k: i for i, k in enumerate(table)}
            other_smokes = [t['id'] for t in plan['smoke'] if t['config_data']['experiment_id'] not in
                            {'insid3_full', 'cgics_full'}]
            self.assertLess(max(positions[k] for k in main_standard), min(positions[k] for k in other_smokes))
        after = {name: hashlib.sha256((parent / name).read_bytes()).hexdigest() for name in evidence}
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
