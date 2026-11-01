"""Source freezing, credential hygiene and manager lifecycle without Slurm."""
import importlib.util
import argparse
import contextlib
import io
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('queue_cli', Path(__file__).resolve().parents[1] / 'scripts/queue_ics.py')
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class QueueCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'repo'
        self.root.mkdir()
        (self.root / 'configs').mkdir()
        (self.root / 'scripts').mkdir()
        (self.root / 'ics').mkdir()
        (self.root / 'ics_eval.py').write_text('# evaluator\n')
        (self.root / 'scripts' / 'queue_ics.py').write_text('# worker\n')
        self.suite = self.root / 'configs' / 'suite.json'
        self.cluster = self.root / 'configs' / 'cluster.local.json'
        self.suite.write_text('{}')
        self.cluster.write_text('{}')
        self.run = self.base / 'run'
        self.runtime = self.base / 'runtime'

    def freeze(self):
        with mock.patch.object(cli, 'ROOT', self.root):
            return cli.freeze_source(self.run, self.suite, self.cluster, self.runtime)

    def test_freeze_reuses_old_sources_when_clone_changes(self):
        snapshot = self.freeze()
        self.suite.write_text('{"changed": true}')
        (self.root / 'scripts' / 'queue_ics.py').write_text('# changed\n')
        self.assertEqual(self.freeze(), snapshot)
        self.assertEqual((snapshot / 'configs/suite.json').read_text(), '{}')
        cli.verify_snapshot(snapshot)

    def test_changed_frozen_code_is_detected(self):
        snapshot = self.freeze()
        (snapshot / 'ics_eval.py').write_text('# modified\n')
        with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
            self.freeze()

    def test_new_runtime_cannot_reuse_run(self):
        self.freeze()
        self.runtime = self.base / 'other'
        with self.assertRaisesRegex(ValueError, 'another runtime'):
            self.freeze()

    def test_symlink_not_copied(self):
        (self.root / 'ics' / 'unsafe.py').symlink_to(self.root / 'ics_eval.py')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.freeze()
        self.assertFalse((self.run / 'source').exists())

    def test_external_suite_rejected(self):
        self.suite = self.base / 'external.json'
        self.suite.write_text('{}')
        with self.assertRaises(ValueError):
            self.freeze()

    def test_manager_environment_drops_credentials_without_changing_parent(self):
        with mock.patch.dict(os.environ, {'HF_TOKEN': 'secret', 'ICS_DINOV3_URL': 'signed',
                                         'PYTHONPATH': '/wrong', 'SLURM_JOB_ID': '123'}):
            env = cli.clean_environment(self.runtime)
            self.assertNotIn('HF_TOKEN', env)
            self.assertNotIn('ICS_DINOV3_URL', env)
            self.assertNotIn('PYTHONPATH', env)
            self.assertEqual(env['HF_HUB_DISABLE_IMPLICIT_TOKEN'], '1')
            self.assertEqual(env['SLURM_JOB_ID'], '123')
            self.assertEqual(os.environ['HF_TOKEN'], 'secret')

    def test_flock_is_exclusive_and_reusable(self):
        path = self.run / 'manager.lock'
        with cli.acquire_lock(path):
            self.assertIsNone(cli.acquire_lock(path))
            self.assertEqual(cli.manager_status(self.run)['manager'], 'ACTIVE')
        with cli.acquire_lock(path) as handle:
            self.assertIsNotNone(handle)
        self.assertEqual(cli.manager_status(self.run)['manager'], 'STOPPED')

    def test_nonfinite_poll_interval_rejected(self):
        for value in ('nan', 'inf', '-inf', '0'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                cli.main(['watch', '--run-dir', str(self.run), '--interval=' + value])
            self.assertEqual(raised.exception.code, 2)

    def test_concurrent_start_before_state_publication_is_idempotent(self):
        snapshot = self.freeze()
        args = argparse.Namespace(run_dir=self.run, runtime=self.runtime,
                                  suite=Path('configs/suite.json'), cluster=Path('configs/cluster.local.json'))
        with mock.patch.object(cli, 'ROOT', snapshot), mock.patch.dict(os.environ), \
                cli.acquire_lock(self.run / 'manager.lock'), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.start(args), 0)
        self.assertEqual(json.loads(out.getvalue())['manager'], 'ALREADY_STARTING')
        self.assertEqual(args.suite, snapshot / 'configs/suite.json')

    def test_direct_frozen_start_rejects_other_runtime(self):
        snapshot = self.freeze()
        args = argparse.Namespace(run_dir=self.run, runtime=self.base / 'other',
                                  suite=Path('configs/suite.json'), cluster=Path('configs/cluster.local.json'))
        with mock.patch.object(cli, 'ROOT', snapshot), mock.patch.dict(os.environ), \
                self.assertRaisesRegex(ValueError, 'Frozen run selection'):
            cli.start(args)

    def test_failed_background_start_is_reported(self):
        snapshot = self.freeze()
        args = argparse.Namespace(run_dir=self.run, runtime=self.runtime,
            suite=Path('configs/suite.json'), cluster=Path('configs/cluster.local.json'),
            max_inflight=8, shared_limit=60, reserved_slots=0, data_workers=2, interval=60)
        plan = {'manifests': {'m': {'prepare': {'dataset': 'coco'},
                                    'dataset_root': str(self.runtime / 'data')}}}
        with mock.patch.object(cli, 'ROOT', snapshot), mock.patch.dict(os.environ), \
                mock.patch.object(cli, 'build_plan', return_value=plan) as build, \
                mock.patch.object(cli, 'write_plan', return_value=self.run / 'plan.json'), \
                mock.patch('ics.rolling_queue.initialize_queue'), \
                mock.patch('ics.rolling_queue.queue_status', return_value={'terminal': False}), \
                mock.patch.object(cli.shutil, 'which', return_value='/fake/slurm'), \
                mock.patch.object(cli.subprocess, 'Popen') as popen:
            popen.return_value.poll.return_value = 1
            popen.return_value.returncode = 1
            with self.assertRaisesRegex(RuntimeError, 'Manager exited during startup'):
                cli.start(args)
            self.assertEqual(build.call_args.args[0], snapshot / 'configs/suite.json')
        self.assertEqual(cli.manager_status(self.run)['manager'], 'STOPPED')

    def test_inherited_flock_survives_parent_close(self):
        path = self.run / 'manager.lock'
        handle = cli.acquire_lock(path)
        proc = subprocess.Popen([sys.executable, '-c',
            'import sys; print("ready",flush=True); sys.stdin.readline()'],
            pass_fds=(handle.fileno(),), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(), 'ready')
            handle.close()
            self.assertIsNone(cli.acquire_lock(path))
        finally:
            proc.communicate('\n', timeout=5)
        with cli.acquire_lock(path) as next_lock:
            self.assertIsNotNone(next_lock)

    def test_real_background_start_resume_and_shared_full_queue(self):
        """Actual CLI processes, fake Slurm binaries: zero submissions at 60."""
        fakebin = self.base / 'bin'
        fakebin.mkdir()
        for name, body in {
            'squeue': 'for i in range(60): print(f"{1000+i}|generation|RUNNING|highpriority")',
            'sacct': 'pass',
            'sbatch': 'raise SystemExit("Unexpected submission at shared capacity")',
        }.items():
            path = fakebin / name
            path.write_text('#!' + sys.executable + '\n' + body + '\n')
            path.chmod(0o755)
        env = dict(os.environ, PATH=str(fakebin) + os.pathsep + os.environ['PATH'])
        command = [sys.executable, str(cli.ROOT / 'scripts/queue_ics.py'), 'start',
                   '--runtime', str(self.runtime), '--run-dir', str(self.run)]
        manager = None
        try:
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            startup = json.loads(result.stdout)
            self.assertEqual(startup['manager'], 'STARTED')
            manager = startup['manager_pid']
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = json.loads((self.run / 'rolling_state.json').read_text())
                if state.get('capacity'):
                    break
                time.sleep(.02)
            self.assertEqual(state['capacity']['shared_observed'], 60)
            self.assertEqual(len(state['tasks']), 1523)
            self.assertTrue(all(t['attempts'] == 0 for t in state['tasks'].values()))
            second = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(json.loads(second.stdout)['manager'], 'ALREADY_RUNNING')
            self.assertEqual(cli.manager_status(self.run)['manager'], 'ACTIVE')
        finally:
            if manager:
                try:
                    os.kill(manager, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and cli.manager_status(self.run)['manager'] == 'ACTIVE':
                    time.sleep(.02)
                self.assertEqual(cli.manager_status(self.run)['manager'], 'STOPPED')

    def test_phase2_and_unmatched_data_root_fail_before_submitting(self):
        plan = {'manifests': {'m': {'prepare': {'dataset': 'fss'}, 'dataset_root': str(self.runtime / 'data')}}}
        with self.assertRaisesRegex(ValueError, 'phase2'):
            cli.required_datasets(plan, self.runtime)
        plan['manifests']['m']['prepare']['dataset'] = 'coco'
        self.assertEqual(cli.required_datasets(plan, self.runtime), ('coco',))
        plan['manifests']['m']['dataset_root'] = str(self.base / 'other')
        with self.assertRaisesRegex(ValueError, 'dataset_root'):
            cli.required_datasets(plan, self.runtime)


if __name__ == '__main__':
    unittest.main()
