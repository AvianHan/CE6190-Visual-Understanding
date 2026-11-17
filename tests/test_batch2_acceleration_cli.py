"""Controller lifecycle checks including a real detached child and SIGTERM."""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import accelerate_batch2 as cli
from scripts.queue_ics import acquire_lock


class AcceleratorLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ics-accelerator-cli-')
        self.run = Path(self.temp.name) / 'batch2'
        self.folder = self.run / cli.acceleration.TAG
        self.folder.mkdir(parents=True)
        self.original = self.run / 'original-marker.json'
        self.original.write_text('{"scientific_output":"unchanged"}\n')
        self.before = self.original.read_bytes()

    def tearDown(self):
        self.assertEqual(self.original.read_bytes(), self.before)
        self.temp.cleanup()

    def manager(self, path):
        return {'manager': 'ACTIVE' if Path(path) == self.run else 'STOPPED'}

    def test_once_lifecycle_and_recovery_without_parent_mutation(self):
        with patch.object(cli, 'verify_snapshot'), patch.object(cli, 'manager_status', self.manager), \
             patch.object(cli.batch2, 'status', return_value={'terminal': False}), \
             patch.object(cli.acceleration, 'recover_transactions') as recover:
            self.assertEqual(cli.watch(self.run, once=True, tick_fn=lambda *a, **k: {'status': 'OK'}), 0)
        record = json.loads((self.folder / 'manager.json').read_text())
        self.assertEqual(record['phase'], 'stopped')
        self.assertEqual(record['exit']['reason'], 'once')
        self.assertEqual(recover.call_count, 2)
        lock = acquire_lock(self.folder / 'manager.lock'); self.assertIsNotNone(lock); lock.close()

    def test_duplicate_watcher_refused(self):
        with acquire_lock(self.folder / 'manager.lock'):
            with self.assertRaisesRegex(RuntimeError, 'already active'):
                cli.watch(self.run, once=True)

    def test_query_failure_is_deferred_and_recorded(self):
        def failed(*a, **k): raise RuntimeError('scheduler unavailable')
        with patch.object(cli, 'verify_snapshot'), patch.object(cli, 'manager_status', self.manager), \
             patch.object(cli.batch2, 'status', return_value={'terminal': False}), \
             patch.object(cli.acceleration, 'recover_transactions'):
            cli.watch(self.run, once=True, tick_fn=failed)
        record = json.loads((self.folder / 'manager.json').read_text())
        self.assertIn('scheduler unavailable', record['last_poll_error']['message'])

    def test_parent_stopped_prevents_mutations(self):
        with patch.object(cli, 'verify_snapshot'), \
             patch.object(cli, 'manager_status', return_value={'manager': 'STOPPED'}), \
             patch.object(cli.acceleration, 'recover_transactions'), \
             patch.object(cli.acceleration, 'tick') as tick:
            cli.watch(self.run, once=True)
        tick.assert_not_called()

    def test_stop_request_recovers_without_new_tick(self):
        (self.folder / 'stop.request.json').write_text('{}')
        with patch.object(cli, 'verify_snapshot'), \
             patch.object(cli.acceleration, 'recover_transactions') as recover, \
             patch.object(cli.acceleration, 'tick') as tick:
            cli.watch(self.run, once=True)
        tick.assert_not_called(); self.assertEqual(recover.call_count, 2)
        self.assertEqual(json.loads((self.folder / 'manager.json').read_text())['exit']['reason'], 'requested')

    def test_cleanup_failure_exits_nonzero_and_is_visible(self):
        with patch.object(cli, 'verify_snapshot'), \
             patch.object(cli.acceleration, 'recover_transactions', side_effect=RuntimeError('held transaction unresolved')):
            with self.assertRaisesRegex(RuntimeError, 'Outstanding hold'):
                cli.watch(self.run, once=True)
        record = json.loads((self.folder / 'manager.json').read_text())
        self.assertEqual(record['exit']['exit_code'], 1)
        self.assertIn('held transaction', record['recovery_required'])

    def test_interval_invalid_values(self):
        for value in ('0', '-1', 'nan', 'inf', '9'):
            with self.subTest(value=value), self.assertRaises(SystemExit) as outcome:
                cli.main(['activate', '--run-dir', str(self.run), '--interval', value])
            self.assertEqual(outcome.exception.code, 2)

    def test_status_does_not_activate_or_tick(self):
        with patch.object(cli.acceleration, 'status', return_value={'state': 'WAITING'}), \
             patch.object(cli, 'manager_status', self.manager), \
             patch.object(cli.acceleration, 'tick') as tick, patch.object(cli.acceleration, 'prepare') as prepare:
            result = cli.current_status(self.run)
        self.assertEqual(result['original_manager'], 'ACTIVE')
        self.assertEqual(result['accelerator'], 'STOPPED')
        tick.assert_not_called(); prepare.assert_not_called()

    @unittest.skipUnless(hasattr(os, 'setsid'), 'POSIX process integration')
    def test_real_detached_process_survives_launcher_and_handles_sigterm(self):
        # Start through a short-lived launcher, just as an SSH terminal exits.
        # Only the engine boundary is replaced; lifecycle/flock/signals are real.
        code = '''import sys,time,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from scripts import accelerate_batch2 as c
run=Path(sys.argv[2])
c.verify_snapshot=lambda root:None
c.manager_status=lambda path:{'manager':'ACTIVE'}
c.batch2.status=lambda run:{'terminal':False}
def recover(run):
 p=run/c.acceleration.TAG/'recovered.txt'
 with p.open('a') as f:f.write('recover\\n')
c.acceleration.recover_transactions=recover
def tick(run,**kw):
 (run/c.acceleration.TAG/'tick.json').write_text(json.dumps({'pid':__import__('os').getpid()}))
 return {'status':'POLL_COMPLETE'}
c.watch(run,interval=10,tick_fn=tick)
'''
        launcher = '''import subprocess,sys,json
p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
print(p.pid)
'''
        launched = subprocess.run([sys.executable, '-c', launcher, code, str(cli.ROOT), str(self.run)],
                                 text=True, capture_output=True, check=True, timeout=10)
        pid = int(launched.stdout.strip())
        try:
            end = time.monotonic() + 8
            while time.monotonic() < end and not (self.folder / 'tick.json').exists(): time.sleep(.03)
            self.assertTrue((self.folder / 'tick.json').exists())
            os.kill(pid, 0)  # The launcher has exited, the watcher remains alive.
            os.kill(pid, signal.SIGTERM)
            end = time.monotonic() + 8
            while time.monotonic() < end:
                record = json.loads((self.folder / 'manager.json').read_text())
                if record['phase'] == 'stopped': break
                time.sleep(.03)
            self.assertEqual(record['exit']['reason'], 'signal')
            self.assertEqual(record['stop_signal'], 'SIGTERM')
            self.assertGreaterEqual((self.folder / 'recovered.txt').read_text().count('recover'), 2)
            lock = acquire_lock(self.folder / 'manager.lock'); self.assertIsNotNone(lock); lock.close()
        finally:
            try: os.kill(pid, signal.SIGTERM)
            except ProcessLookupError: pass


if __name__ == '__main__': unittest.main()
