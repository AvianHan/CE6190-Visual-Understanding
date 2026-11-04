"""Manager lifecycle and recovery tests; fake Slurm, no downloads or model work."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('manager_lifecycle_cli', PROJECT/'scripts/queue_ics.py')
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class ManagerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root/'run'
        self.source = self.run/'source'
        self.source.mkdir(parents=True)
        (self.source/'snapshot.json').write_text(json.dumps({'files': {}}))
        self.old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        self.addCleanup(self.restore_signals)
        self.summary = {'terminal': False, 'status': 'RUNNING', 'updated_at': 1234}

    def restore_signals(self):
        for sig, handler in self.old_handlers.items():
            signal.signal(sig, handler)

    def record(self):
        return json.loads((self.run/'manager.json').read_text())

    def watch(self, **kwargs):
        with mock.patch.object(cli, 'ROOT', self.source), contextlib.redirect_stdout(io.StringIO()):
            return cli.watcher(self.run, **kwargs)

    def test_once_records_poll_and_exit_without_changing_queue_state(self):
        state = self.run/'rolling_state.json'
        state.write_text('{"saved": true}')
        original = state.read_bytes(), state.stat().st_mtime_ns
        with mock.patch('ics.rolling_queue.tick', return_value=self.summary) as tick:
            self.assertEqual(self.watch(once=True), 0)
        record = self.record()
        self.assertEqual(record['exit']['reason'], 'once')
        self.assertEqual(record['phase'], 'stopped')
        self.assertLessEqual(record['last_poll_started_at'], record['last_poll_completed_at'])
        self.assertLessEqual(record['last_poll_completed_at'], record['last_heartbeat_at'])
        self.assertTrue(callable(tick.call_args.kwargs['stop_requested']))
        self.assertEqual(cli.manager_status(self.run)['manager'], 'STOPPED')
        self.assertEqual((state.read_bytes(), state.stat().st_mtime_ns), original)

    def test_terminal_failure_records_terminal_exit_code(self):
        with mock.patch('ics.rolling_queue.tick', return_value={**self.summary, 'terminal': True,
                                                               'status': 'COMPLETE_WITH_FAILURES'}):
            self.assertEqual(self.watch(), 2)
        record = self.record()
        self.assertEqual(record['exit']['reason'], 'terminal')
        self.assertEqual(record['exit']['queue_status'], 'COMPLETE_WITH_FAILURES')
        self.assertEqual(record['exit']['exit_code'], 2)

    def test_signal_is_forwarded_as_cooperative_stop_and_recorded(self):
        def tick(_path, *, stop_requested=None):
            self.assertTrue(callable(stop_requested))
            self.assertFalse(stop_requested())
            os.kill(os.getpid(), signal.SIGTERM)
            self.assertTrue(stop_requested())
            return self.summary
        with mock.patch('ics.rolling_queue.tick', side_effect=tick) as run_tick:
            self.assertEqual(self.watch(), 0)
        self.assertEqual(run_tick.call_count, 1)
        self.assertEqual(self.record()['exit']['reason'], 'signal')
        self.assertEqual(self.record()['exit']['signal'], 'SIGTERM')

    def test_fatal_tick_exception_preserved_and_exit_recorded(self):
        with mock.patch('ics.rolling_queue.tick', side_effect=RuntimeError('original poll failure')):
            with self.assertRaisesRegex(RuntimeError, 'original poll failure'):
                self.watch()
        record = self.record()
        self.assertEqual(record['exit']['reason'], 'fatal')
        self.assertEqual(record['exit']['exception_type'], 'RuntimeError')
        self.assertEqual(record['exit']['message'], 'original poll failure')
        self.assertEqual(record['exit']['phase'], 'polling')
        self.assertIsNone(record['last_poll_completed_at'])
        self.assertEqual(cli.manager_status(self.run)['manager'], 'STOPPED')

    def test_snapshot_failure_records_fatal_before_first_poll(self):
        (self.source/'snapshot.json').write_text(json.dumps({'files': {'absent.py': 'hash'}}))
        with mock.patch('ics.rolling_queue.tick') as tick:
            with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
                self.watch()
        tick.assert_not_called()
        self.assertEqual(self.record()['exit']['reason'], 'fatal')
        self.assertEqual(self.record()['exit']['phase'], 'verifying_snapshot')
        self.assertIsNone(self.record()['last_poll_started_at'])

    def test_log_failure_records_fatal_after_successful_poll(self):
        with mock.patch('ics.rolling_queue.tick', return_value=self.summary), \
                mock.patch.object(cli, 'print_json', side_effect=OSError('log unavailable')):
            with self.assertRaisesRegex(OSError, 'log unavailable'):
                self.watch()
        record = self.record()
        self.assertEqual(record['exit']['reason'], 'fatal')
        self.assertEqual(record['exit']['exception_type'], 'OSError')
        self.assertEqual(record['exit']['phase'], 'logging')
        self.assertIsNotNone(record['last_poll_completed_at'])

    def test_lifecycle_write_failure_does_not_mask_original_exception(self):
        with mock.patch('ics.rolling_queue.tick', side_effect=ValueError('original fatal')) as tick, \
                mock.patch.object(cli, 'atomic_json', side_effect=OSError('record unavailable')), \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaisesRegex(ValueError, 'original fatal'):
                self.watch()
        tick.assert_called_once()
        self.assertIn('record unavailable', errors.getvalue())
        self.assertEqual(cli.manager_status(self.run)['manager'], 'STOPPED')

    def status(self, saved=None):
        with cli.acquire_lock(self.run/'manager.lock'):
            pass
        with mock.patch('ics.rolling_queue.queue_status', return_value=saved or self.summary), \
                mock.patch.object(cli.subprocess, 'run') as command, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['status', '--run-dir', str(self.run)]), 0)
        command.assert_not_called()
        return json.loads(output.getvalue())

    def test_stopped_status_marks_saved_state_stale_without_claiming_exit_cause(self):
        # A live/recycled PID in old metadata must not override the unlocked flock.
        meta = self.run/'manager.json'
        meta.write_text(json.dumps({'pid': os.getpid(), 'started_at': 1000}))
        before = meta.read_bytes(), meta.stat().st_mtime_ns
        with mock.patch.object(cli.time, 'time', return_value=1334):
            summary = self.status()
        self.assertEqual(summary['manager'], 'STOPPED')
        self.assertEqual(summary['status'], 'MANAGER_STOPPED')
        self.assertEqual(summary['saved_queue_status'], 'RUNNING')
        self.assertEqual(summary['state_age_seconds'], 100)
        self.assertFalse(summary['state_is_live'])
        self.assertFalse(summary['scheduler_queried'])
        self.assertEqual(summary['manager_exit_reason'], 'unknown')
        self.assertIn('resume', summary)
        self.assertEqual((meta.read_bytes(), meta.stat().st_mtime_ns), before)

    def test_fatal_status_requires_diagnosis_before_resume(self):
        (self.run/'manager.json').write_text(json.dumps({'exit': {'reason': 'fatal',
            'ended_at': 2000, 'exception_type': 'ValueError', 'message': 'Frozen source changed'}}))
        summary = self.status()
        self.assertEqual(summary['status'], 'MANAGER_STOPPED')
        self.assertEqual(summary['manager_exit_reason'], 'fatal')
        self.assertNotIn('resume', summary)
        self.assertIn('recovery', summary)

    def test_old_or_unverifiable_exit_record_cannot_explain_current_manager_stop(self):
        # A resumed manager may update queue state while lifecycle writes fail;
        # a previous session's fatal exit must not be attributed to the new stop.
        for ended, updated in ((100, 200), (None, 200), ('300', 200), (True, 200),
                               (float('nan'), 200), (float('inf'), 200), (float('-inf'), 200),
                               (300, None), (300, float('nan')), (300, float('inf')),
                               (300, True)):
            with self.subTest(ended=ended, updated=updated):
                metadata = self.run/'manager.json'
                metadata.write_text(json.dumps({'pid': 99999, 'exit': {
                    'reason': 'fatal', 'ended_at': ended, 'message': 'old session failure'}}))
                before = metadata.read_bytes(), metadata.stat().st_mtime_ns
                saved = {**self.summary, 'updated_at': updated}
                report = self.status(saved)
                self.assertEqual(report['manager_exit_reason'], 'unknown')
                self.assertEqual(report['status'], 'MANAGER_STOPPED')
                self.assertIn('resume', report)
                self.assertNotIn('recovery', report)
                self.assertIn('manager_lifecycle_note', report)
                self.assertEqual(report['manager_lifecycle']['exit']['message'], 'old session failure')
                self.assertEqual((metadata.read_bytes(), metadata.stat().st_mtime_ns), before)

    def test_unreadable_lifecycle_metadata_does_not_break_readonly_status(self):
        (self.run/'manager.json').write_text('{not-json')
        summary = self.status()
        self.assertEqual(summary['status'], 'MANAGER_STOPPED')
        self.assertEqual(summary['manager_exit_reason'], 'unknown')
        self.assertIn('manager_lifecycle_error', summary)

    def test_terminal_queue_keeps_its_completion_status(self):
        summary = self.status({**self.summary, 'terminal': True, 'status': 'COMPLETE_WITH_FAILURES'})
        self.assertEqual(summary['status'], 'COMPLETE_WITH_FAILURES')
        self.assertNotIn('resume', summary)


class ManagerProcessRecoveryTests(unittest.TestCase):
    def test_real_background_term_and_kill_resume_existing_job_without_resubmission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, runtime, fakebin = root/'run', root/'runtime', root/'bin'
            fakebin.mkdir()
            scheduler = root/'scheduler.json'
            scheduler.write_text(json.dumps({'submissions': []}))
            common = ('import json,os,sys\nfrom pathlib import Path\n'
                      'p=Path(os.environ["ICS_TEST_SCHEDULER"])\n'
                      's=json.loads(p.read_text())\n')
            bodies = {
                'squeue': 'for row in s["submissions"]: print(str(row["id"])+"|"+row["name"]+"|RUNNING|highpriority")\n',
                'sacct': 'pass\n',
                'sbatch': 'name=sys.argv[sys.argv.index("--job-name")+1]\n'
                          'jid=1000+len(s["submissions"])\n'
                          's["submissions"].append({"id":jid,"name":name})\n'
                          'p.write_text(json.dumps(s))\nprint(jid)\n',
            }
            for name, body in bodies.items():
                path = fakebin/name
                path.write_text('#!'+sys.executable+'\n'+common+body)
                path.chmod(0o755)
            env = dict(os.environ, PATH=str(fakebin)+os.pathsep+os.environ['PATH'],
                       ICS_TEST_SCHEDULER=str(scheduler))
            command = [sys.executable, str(PROJECT/'scripts/queue_ics.py'), 'start',
                       '--runtime', str(runtime), '--run-dir', str(run), '--interval', '10']
            manager = None

            def wait_until(predicate):
                deadline = time.monotonic()+8
                while time.monotonic() < deadline:
                    if predicate():
                        return
                    time.sleep(.02)
                self.fail('Timed out waiting for local fake-Slurm manager state')

            def start():
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(result.stdout)['manager_pid']

            try:
                manager = start()
                wait_until(lambda: json.loads((run/'rolling_state.json').read_text())['tasks']['data']['status'] == 'SUBMITTED')
                self.assertEqual(len(json.loads(scheduler.read_text())['submissions']), 1)
                first_job = json.loads((run/'rolling_state.json').read_text())['tasks']['data']['job_id']
                for stop_signal in (signal.SIGTERM, signal.SIGKILL):
                    previous_update = json.loads((run/'rolling_state.json').read_text())['updated_at']
                    os.kill(manager, stop_signal)
                    wait_until(lambda: cli.manager_status(run)['manager'] == 'STOPPED')
                    status = subprocess.run([sys.executable, str(PROJECT/'scripts/queue_ics.py'),
                        'status', '--run-dir', str(run)], env=env, capture_output=True, text=True, check=True)
                    report = json.loads(status.stdout)
                    self.assertEqual(report['status'], 'MANAGER_STOPPED')
                    self.assertEqual(report['manager_exit_reason'],
                                     'signal' if stop_signal == signal.SIGTERM else 'unknown')
                    manager = start()
                    def resumed_poll_finished():
                        lifecycle = json.loads((run/'manager.json').read_text())
                        return (lifecycle['pid'] == manager and lifecycle['last_poll_completed_at'] is not None
                                and json.loads((run/'rolling_state.json').read_text())['updated_at'] > previous_update)
                    wait_until(resumed_poll_finished)
                    state = json.loads((run/'rolling_state.json').read_text())
                    self.assertEqual(state['tasks']['data']['job_id'], first_job)
                    self.assertEqual(state['tasks']['data']['attempts'], 1)
                    self.assertEqual(state['tasks']['data']['status'], 'RUNNING')
                    self.assertEqual(len(json.loads(scheduler.read_text())['submissions']), 1)
            finally:
                if manager and cli.manager_status(run)['manager'] == 'ACTIVE':
                    os.kill(manager, signal.SIGTERM)
                    wait_until(lambda: cli.manager_status(run)['manager'] == 'STOPPED')


if __name__ == '__main__':
    unittest.main()
