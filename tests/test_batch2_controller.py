"""Synthetic scheduler + immutable worker contracts; never contacts Slurm."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from ics import batch2, orchestration, rolling_queue as rq
from scripts import queue_ics

ROOT = Path(__file__).resolve().parents[1]


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.top = Path(self.temp.name)
        self.runtime = self.top / 'runtime'; self.runtime.mkdir()
        self.run = self.top / 'run'; self.run.mkdir()
        self.env = mock.patch.dict(os.environ, {'ICS_RUNTIME': str(self.runtime), 'ICS_BATCH2_ROOT': str(self.run)})
        self.env.start(); self.addCleanup(self.env.stop)
        self.source = queue_ics.freeze_source(self.run, ROOT / 'configs/suite.batch2.core.json',
                                              ROOT / 'configs/cluster.orion.batch2.json', self.runtime)
        self.request = batch2.selection(self.runtime, self.run)
        with mock.patch.object(orchestration, 'PROJECT_ROOT', self.source):
            batch2.prepare(self.source, self.run, self.request)
        self.path = self.run / 'scopes/core/plan.json'
        self.plan = rq._plan(self.path)

    def state(self, scope='core'):
        plan = rq._plan(self.run / 'scopes' / scope / 'plan.json')
        return rq._read_state(plan)

    def test_complete_workload_and_same_dataset_smoke_gates(self):
        value = batch2.status(self.run)
        self.assertEqual(sum(s['total'] for s in value['scope_status'].values()), 1101)
        self.assertEqual(len(self.plan['tasks']), 728)
        self.assertEqual(len(self.plan['smoke']), 26)
        state = self.state()
        for item in self.plan['tasks']:
            deps = state['tasks'][item['id']]['dependencies']
            self.assertEqual(deps, ['smoke-' + item['config_data']['experiment_id'] + '-' + item['dataset']])
        for resource in self.plan['resources'].values(): self.assertEqual(resource['max_parallel'], 60)
        self.assertEqual(set(self.plan['resources']['cgics_mllm']['exclude'].split(',')), {'2FS9B74', 'A484652X3B10601'})
        for kind in ('insid3', 'cgics_visual'):
            self.assertEqual(self.plan['cluster']['resources'][kind]['exclude'], 'A484652X3B10601')

    def test_prepare_is_idempotent_and_refuses_changed_request(self):
        original = (self.run / 'batch2.json').read_bytes()
        with mock.patch.object(orchestration, 'PROJECT_ROOT', self.source):
            batch2.prepare(self.source, self.run, self.request)
            changed = {**self.request, 'seed': 1}
            with self.assertRaises(ValueError): batch2.prepare(self.source, self.run, changed)
        self.assertEqual((self.run / 'batch2.json').read_bytes(), original)

    def test_failed_transfer_does_not_block_core(self):
        p = rq._plan(self.run / 'scopes/suim/plan.json'); s = rq._read_state(p)
        s['tasks']['data'].update(status='FAILED', error='Source unavailable')
        rq._propagate_blocks(s); rq._atomic_json(Path(p['run_dir']) / rq.STATE_NAME, s)
        self.assertEqual(self.state()['tasks']['data']['status'], 'PENDING')
        self.assertEqual(batch2.status(self.run)['scope_status']['suim']['counts']['FAILED'], 1)

    def _intent(self):
        task = copy.deepcopy(self.state()['tasks']['data'])
        command = batch2.make_worker_launch(self.path, self.plan, task)
        return task, Path(command[command.index('--intent') + 1])

    def test_worker_executes_before_ack_without_mutable_state(self):
        task, path = self._intent()
        # The queue intentionally still says PENDING, with no job_id. A submitted
        # worker must rely on the sealed pre-sbatch intent, not this mutable file.
        data = mock.Mock()
        with mock.patch.dict(os.environ, {'SLURM_JOB_ID': '900', 'SLURM_JOB_NAME': task['job_name']}), \
             mock.patch('ics.batch2_data.prepare_scope', data):
            batch2.execute_intent(path, task['batch2_intent_sha256'])
        data.assert_called_once()
        self.assertFalse(data.call_args.kwargs['allow_download'])
        marker = json.loads(rq._marker_path(self.plan, task['key']).read_text())
        self.assertEqual(marker['status'], 'PASS'); self.assertEqual(marker['slurm_job_id'], '900')
        before = rq._marker_path(self.plan, task['key']).read_bytes()
        with mock.patch.dict(os.environ, {'SLURM_JOB_ID': '900', 'SLURM_JOB_NAME': task['job_name']}), \
             mock.patch('ics.batch2_data.prepare_scope', data), self.assertRaises(FileExistsError):
            batch2.execute_intent(path, task['batch2_intent_sha256'])
        self.assertEqual(rq._marker_path(self.plan, task['key']).read_bytes(), before)

    def test_worker_fails_wrong_jobname_and_changed_source_before_data(self):
        task, path = self._intent()
        with mock.patch.dict(os.environ, {'SLURM_JOB_ID': '900', 'SLURM_JOB_NAME': 'wrong'}), \
             mock.patch('ics.batch2_data.prepare_scope') as data, self.assertRaises(RuntimeError):
            batch2.execute_intent(path, task['batch2_intent_sha256'])
        data.assert_not_called()
        candidate = self.source / 'ics_eval.py'; candidate.write_text(candidate.read_text() + '\n# changed\n')
        with mock.patch.dict(os.environ, {'SLURM_JOB_ID': '900', 'SLURM_JOB_NAME': task['job_name']}), \
             mock.patch('ics.batch2_data.prepare_scope') as data, self.assertRaises(ValueError):
            batch2.execute_intent(path, task['batch2_intent_sha256'])
        data.assert_not_called()

    def test_reconcile_ack_alias_from_exact_name_and_keep_evidence(self):
        task, path = self._intent()
        task.update(status='SUBMITTED', job_id='111')
        history = {'111': {'job_name': 'unrelated'}, '222': {'job_name': task['job_name']}}
        batch2.reconcile_intent_identity(self.plan, task, {}, history)
        self.assertEqual(task['job_id'], '222')
        self.assertEqual(task['acknowledgement_corrections'][0]['previous_job_id'], '111')

    def test_duplicate_exact_scheduler_identity_is_uncertain_even_with_ack(self):
        task, path = self._intent()
        task.update(status='SUBMITTED', job_id='111')
        rows = {'111': {'job_name': task['job_name']}, '222': {'job_name': task['job_name']}}
        problem = batch2.reconcile_intent_identity(self.plan, task, {}, rows)
        self.assertIn('Multiple scheduler jobs', problem)
        self.assertEqual(task['job_id'], '111')

    def test_missing_ack_and_no_evidence_never_resubmits_or_infers_success(self):
        task, path = self._intent()
        task.update(status='UNCERTAIN')
        before = copy.deepcopy(task)
        batch2.reconcile_intent_identity(self.plan, task, {}, {})
        self.assertEqual(task, before)

    def test_claim_scheduler_conflict_is_not_overwritten(self):
        task, path = self._intent()
        path.with_suffix('.claim.json').write_text(json.dumps({'slurm_job_id': '222',
            'job_name': task['job_name'], 'intent_sha256': task['batch2_intent_sha256']}))
        error = batch2.reconcile_intent_identity(self.plan, task, {}, {'333': {'job_name': task['job_name']}})
        self.assertIn('conflict', error)
        self.assertNotIn('job_id', task)

    def test_external_scope_unseen_intents_reserve_all_shared_slots(self):
        with mock.patch.object(rq, '_scheduler_snapshot', return_value=({}, {})), \
             mock.patch.object(rq, '_submit') as submit:
            value = rq.tick(self.path, external_active=[None] * 60)
        submit.assert_not_called()
        self.assertEqual(value['capacity']['available_before_submission'], 0)
        self.assertEqual(value['capacity']['unseen_other_batch2_scope_reservations'], 60)

    def test_parent_tick_passes_other_scope_reservations_and_is_sequential(self):
        p = rq._plan(self.run / 'scopes/suim/plan.json'); s = rq._read_state(p)
        s['tasks']['data'].update(status='SUBMITTED', job_id='123')
        rq._atomic_json(Path(p['run_dir']) / rq.STATE_NAME, s)
        with mock.patch.object(rq, 'tick') as tick:
            batch2.tick_batch(self.run)
        self.assertEqual(tick.call_count, 3)
        self.assertEqual(tick.call_args_list[0].kwargs['external_active'], ['123'])
        self.assertEqual(tick.call_args_list[1].kwargs['external_active'], [])
        with mock.patch.object(rq, 'tick') as tick:
            batch2.tick_batch(self.run)
        self.assertEqual(Path(tick.call_args_list[0].args[0]).parent.name, 'suim')

    def test_first_batch_path_cannot_be_reused(self):
        with self.assertRaises(ValueError):
            batch2.selection(self.runtime, self.runtime / 'runs/ics-recovery-20261113')

    def test_real_cli_frozen_prepare_and_status_no_scheduler(self):
        other = self.top / 'cli-run'
        proc = subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'scripts/batch2_ics.py'),
            'prepare', '--runtime', str(self.runtime), '--run-dir', str(other)],
            text=True, capture_output=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        first = json.loads(proc.stdout)
        self.assertEqual(first['manager'], 'PREPARED_NOT_STARTED')
        proc = subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'scripts/batch2_ics.py'),
            'status', '--run-dir', str(other)], text=True, capture_output=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(json.loads(proc.stdout)['terminal'])


# Only this temporary repository replaces data preparation. The scheduler runs
# real generated worker scripts synchronously BEFORE returning an acknowledgement.
# No external Slurm, download, model, or GPU is involved.
SUBPROCESS_DATA_FIXTURE = r'''
import json
from pathlib import Path

def prepare_scope(runtime, output_dir, *, scope, **kwargs):
    state = json.loads((output_dir.parent / 'rolling_state.json').read_text())
    task = state['tasks']['data']
    assert task['status'] == 'SUBMITTING'
    assert not task.get('job_id'), 'worker must not depend on post-sbatch ACK'
    destination = Path(runtime) / ('observed-' + scope + '.json')
    destination.write_text(json.dumps({'scope': scope, 'queue_status': task['status'],
                                     'queue_job_id': task.get('job_id')}))
    if scope != 'core':
        raise FileNotFoundError('Explicit synthetic missing transfer asset: ' + scope)
'''

SUBPROCESS_SCHEDULER_FIXTURE = r'''
import json, os, subprocess, sys
from pathlib import Path
root = Path(os.environ['ICS_TEST_SCHEDULER'])
files = sorted(root.glob('job-*.json'))
rows = [json.loads(p.read_text()) for p in files]
name = Path(sys.argv[0]).name
if name == 'squeue':
    for row in rows:
        if row['state'] == 'PENDING':
            print('|'.join(row[k] for k in ('job_id','job_name','state','qos')))
elif name == 'sacct':
    for row in rows:
        print('|'.join(row[k] for k in ('job_id','job_name','state','exit_code','qos')))
elif name == 'sbatch':
    args = sys.argv[1:]
    def flag(key): return args[args.index(key) + 1]
    script = Path(args[-1]); jid = str(1001 + len(rows))
    row = {'job_id': jid, 'job_name': flag('--job-name'), 'qos': flag('--qos'),
           'state': 'PENDING', 'exit_code': '0:0', 'script': str(script)}
    path = root / ('job-' + jid + '.json')
    path.write_text(json.dumps(row))
    if script.stem == 'data':
        env = dict(os.environ, SLURM_JOB_ID=jid, SLURM_JOB_NAME=row['job_name'])
        output = subprocess.run(['/bin/bash', str(script)], env=env,
                                text=True, capture_output=True, timeout=30)
        (root / ('worker-' + jid + '.log')).write_text(output.stdout + output.stderr)
        row.update(state='COMPLETED' if output.returncode == 0 else 'FAILED',
                   exit_code=str(output.returncode) + ':0')
        path.write_text(json.dumps(row))
    # Lost core ACK is deliberately unparseable. The real worker claim and
    # scheduler identity must repair it on the following poll without resubmit.
    print('accepted-without-id' if '/core/' in str(script) and script.stem == 'data' else jid)
else:
    raise RuntimeError('Unexpected test scheduler executable')
'''


class SubprocessControllerTests(unittest.TestCase):
    def test_real_worker_before_ack_and_failed_transfer_core_progress(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            top = Path(directory); fixture = top / 'fixture'; fixture.mkdir()
            for name in ('ics', 'scripts', 'configs'):
                shutil.copytree(ROOT / name, fixture / name,
                                ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copyfile(ROOT / 'ics_eval.py', fixture / 'ics_eval.py')
            (fixture / 'ics/batch2_data.py').write_text(SUBPROCESS_DATA_FIXTURE)
            runtime = top / 'runtime'; runtime.mkdir()
            python = runtime / 'envs/insid3/bin/python'; python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            commands = top / 'bin'; commands.mkdir()
            scheduler = top / 'scheduler'; scheduler.mkdir()
            for name in ('squeue', 'sacct', 'sbatch'):
                path = commands / name
                path.write_text('#!' + sys.executable + '\n' + SUBPROCESS_SCHEDULER_FIXTURE)
                path.chmod(0o700)
            env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ['PATH'],
                       ICS_TEST_SCHEDULER=str(scheduler))
            run = top / 'run'
            def call(script, *args):
                result = subprocess.run([sys.executable, '-I', '-B', str(script), *args],
                                        text=True, capture_output=True, env=env, timeout=45)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout[-2000:])
                return json.loads(result.stdout)
            call(fixture / 'scripts/batch2_ics.py', 'prepare', '--runtime', str(runtime),
                 '--run-dir', str(run))
            script = run / 'source/scripts/batch2_ics.py'
            call(script, 'watch', '--run-dir', str(run), '--once')
            for scope in batch2.SCOPES:
                observation = json.loads((runtime / ('observed-' + scope + '.json')).read_text())
                self.assertEqual(observation['queue_status'], 'SUBMITTING')
                self.assertIsNone(observation['queue_job_id'])
            before = json.loads((run / 'scopes/core/rolling_state.json').read_text())
            self.assertEqual(before['tasks']['data']['status'], 'UNCERTAIN')
            call(script, 'watch', '--run-dir', str(run), '--once')
            core = json.loads((run / 'scopes/core/rolling_state.json').read_text())
            self.assertEqual(core['tasks']['data']['status'], 'PASS')
            self.assertEqual(core['tasks']['data']['job_id'], '1001')
            self.assertEqual(core['tasks']['preflight']['status'], 'SUBMITTED')
            for scope in ('suim', 'isic'):
                state = json.loads((run / 'scopes' / scope / 'rolling_state.json').read_text())
                self.assertEqual(state['tasks']['data']['status'], 'FAILED')
                self.assertEqual(state['tasks']['preflight']['status'], 'BLOCKED')
            jobs = [json.loads(p.read_text()) for p in scheduler.glob('job-*.json')]
            self.assertEqual(sum(Path(row['script']).stem == 'data' for row in jobs), 3)


if __name__ == '__main__': unittest.main()
