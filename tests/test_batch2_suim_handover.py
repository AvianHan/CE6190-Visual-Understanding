"""Independent detached handover regression; fake Slurm and no GPU execution.

Exercises actual process ownership, pidfd stop, inherited flock, detached restart
and idempotency. Synthetic accepted jobs remain PENDING; no pretrained model or
Orion service is contacted.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest

from ics import batch2, batch2_recovery as recovery, rolling_queue as rq
from tests import test_batch2_suim_recovery as recovery_fixtures


FAKE_SLURM = r'''
import fcntl,json,os,sys
from pathlib import Path
p=Path(os.environ['ICS_HANDOVER_DATABASE'])
with p.with_suffix('.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 rows=json.loads(p.read_text())
 name=Path(sys.argv[0]).name
 args=sys.argv[1:]
 if name=='squeue':
  for r in rows:
   if r['state']=='PENDING':print('|'.join(r[k] for k in ('job_id','job_name','state','qos')))
 elif name=='sacct':
  for r in rows:print('|'.join(r[k] for k in ('job_id','job_name','state','exit_code','qos')))
 elif name=='sbatch':
  def flag(key):return args[args.index(key)+1]
  jid=str(300000+len(rows))
  rows.append(dict(job_id=jid,job_name=flag('--job-name'),state='PENDING',exit_code='0:0',
                   qos=flag('--qos'),script=args[-1],partition=flag('--partition')))
  p.write_text(json.dumps(rows));print(jid)
 elif name=='sacctmgr':print('highpriority|60|')
 elif name=='scontrol' and 'partition' in args:
  part=args[-1]
  print('PartitionName='+part+' State=UP AllowQos=highpriority AllowAccounts=prj0000000262')
 else:raise RuntimeError('Unexpected fake scheduler command: '+repr(sys.argv))
'''


@unittest.skipUnless(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'),
                     'The production handover requires Linux pidfd support')
class DetachedSuimHandoverTests(unittest.TestCase):
    def test_live_manager_handover_detached_idempotent_resume_preserves_work(self):
        fixture = recovery_fixtures.RecoveryTests()
        fixture.setUp()
        children = []
        old_process = None
        try:
            run, scope = fixture.run, fixture.scope
            # A genuine successful task in a different scope must survive the
            # repair, while its accepted preflight and ISIC data jobs stay live.
            core_path = run / 'scopes/core/plan.json'
            core_plan = rq._plan(core_path)
            core = rq._read_state(core_plan)
            core['tasks']['data'].update(status='PASS', attempts=1, job_id='206899',
                                          preserved_sentinel='previous-core-data')
            rq._atomic_json(Path(core_plan['run_dir']) / rq.STATE_NAME, core)
            core_pass = dict(core['tasks']['data'])
            core_marker = rq._marker_path(core_plan, 'data')
            rq._atomic_json(core_marker, {'status': 'PASS', 'key': 'data',
                                         'plan_fingerprint': core_plan['fingerprint'],
                                         'slurm_job_id': '206899'})
            marker_before = core_marker.read_bytes()
            original_source = (fixture.source / 'snapshot.json').read_bytes()
            original_claim = (scope / 'batch2_intents/data/attempt-00001.claim.json').read_bytes()

            # The data implementation is synthetic, injected before sealing the
            # isolated test snapshot; scientific/model workers remain original.
            data_file = fixture.overlay / 'ics/batch2_data.py'
            data_file.write_text(
                "SUIM_REPAIR_IMPLEMENTATION_ID='suim-strict-geometry-filter-v1'\n"
                "def prepare_scope(*a,**k): raise RuntimeError('No test worker may execute')\n"
                "def verify_scope(*a,**k): raise RuntimeError('No test worker may execute')\n")
            snapshot = json.loads((fixture.overlay / 'snapshot.json').read_text())
            snapshot['files']['ics/batch2_data.py'] = batch2.digest(data_file)
            rq._atomic_json(fixture.overlay / 'snapshot.json', snapshot)

            binaries = fixture.top / 'bin'
            binaries.mkdir()
            database = fixture.top / 'scheduler.json'
            rows = list(fixture.history.values())
            rows.append(dict(job_id='206899', job_name=core_pass['job_name'],
                             state='COMPLETED', exit_code='0:0', qos='highpriority'))
            database.write_text(json.dumps(rows))
            for name in ('squeue', 'sacct', 'sbatch', 'scontrol', 'sacctmgr'):
                path = binaries / name
                path.write_text('#!' + sys.executable + '\n' + FAKE_SLURM)
                path.chmod(0o700)
            env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ['PATH'],
                       ICS_HANDOVER_DATABASE=str(database))
            metadata = run / 'manager.json'

            def wait_for(condition, description, seconds=20):
                end = time.monotonic() + seconds
                while time.monotonic() < end:
                    if condition():
                        return
                    time.sleep(.05)
                log = (run / 'manager.log').read_text()[-5000:] if (run / 'manager.log').exists() else ''
                self.fail(description + '\n' + log)

            def manager_record():
                try:
                    return json.loads(metadata.read_text())
                except (FileNotFoundError, ValueError):
                    return {}

            def call(command):
                result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=45)
                self.assertEqual(result.returncode, 0, result.stdout[-5000:] + result.stderr)
                # Handover emits a progress line before the structured result.
                pos = result.stdout.find('{')
                self.assertGreaterEqual(pos, 0, result.stdout)
                return json.loads(result.stdout[pos:])

            old_command = [sys.executable, '-I', '-B', '-u',
                           str(fixture.source / 'scripts/batch2_ics.py'),
                           'watch', '--run-dir', str(run), '--interval', '10']
            with (run / 'manager.log').open('a') as log:
                old_process = subprocess.Popen(old_command, env=env, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
            children.append(old_process.pid)
            wait_for(lambda: manager_record().get('phase') == 'waiting'
                     and len(json.loads(database.read_text())) == 5,
                     'Original detached manager did not submit exactly two independent-scope jobs')
            self.assertEqual(manager_record()['pid'], old_process.pid)
            accepted_before = {
                name: rq._read_state(rq._plan(run / 'scopes' / name / 'plan.json'))['tasks'][key]['job_id']
                for name, key in (('core', 'preflight'), ('isic', 'data'))}

            activate = [sys.executable, '-I', '-B', '-u',
                        str(fixture.overlay / 'scripts/recover_batch2_suim.py'),
                        'activate', '--run-dir', str(run),
                        '--expected-plan-fingerprint', fixture.plan['fingerprint'],
                        '--expected-failed-job', '206900', '--interval', '10']
            first = call(activate)
            new_pid = first['manager_pid']
            children.append(new_pid)
            self.assertNotEqual(new_pid, old_process.pid)
            self.assertEqual(old_process.wait(timeout=10), 0)
            wait_for(lambda: manager_record().get('pid') == new_pid
                     and manager_record().get('phase') == 'waiting'
                     and len(json.loads(database.read_text())) == 6,
                     'Recovery manager did not detach and submit exactly one new SUIM data attempt')

            # The activate process has exited and all of its pipe handles have
            # closed. A subsequent poll proves no terminal/session dependency.
            first_poll = manager_record()['last_poll_completed_at']
            wait_for(lambda: manager_record().get('last_poll_completed_at', 0) > first_poll,
                     'Detached recovery manager did not poll after launcher exit', seconds=16)
            accepted_rows = json.loads(database.read_text())
            self.assertEqual(len(accepted_rows), 6)
            second = call(activate)
            self.assertEqual(second['manager_start'], 'ALREADY_RUNNING_RECOVERY')
            self.assertEqual(manager_record()['pid'], new_pid)
            self.assertEqual(json.loads(database.read_text()), accepted_rows)

            # Explicit duplicate watch must fail to acquire the same manager lock.
            duplicate = subprocess.run([sys.executable, '-I', '-B',
                str(fixture.overlay / 'scripts/recover_batch2_suim.py'), 'watch',
                '--run-dir', str(run), '--interval', '10', '--once'],
                env=env, text=True, capture_output=True, timeout=20)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn('already holds the lock', duplicate.stderr)
            self.assertEqual(json.loads(database.read_text()), accepted_rows)

            os.kill(new_pid, signal.SIGTERM)
            wait_for(lambda: manager_record().get('exit') is not None,
                     'Recovery manager did not exit cleanly')
            third = call(activate)
            restarted_pid = third['manager_pid']
            children.append(restarted_pid)
            wait_for(lambda: manager_record().get('pid') == restarted_pid
                     and manager_record().get('phase') == 'waiting',
                     'Recovery manager did not resume after a clean stop')
            self.assertEqual(json.loads(database.read_text()), accepted_rows)
            self.assertEqual(rq._read_state(core_plan)['tasks']['data'], core_pass)
            self.assertEqual(core_marker.read_bytes(), marker_before)
            for name, key in (('core', 'preflight'), ('isic', 'data')):
                state = rq._read_state(rq._plan(run / 'scopes' / name / 'plan.json'))
                self.assertEqual(state['tasks'][key]['job_id'], accepted_before[name])
            self.assertEqual((fixture.source / 'snapshot.json').read_bytes(), original_source)
            self.assertEqual((scope / 'batch2_intents/data/attempt-00001.claim.json').read_bytes(), original_claim)
            resumed = rq._read_state(fixture.plan)
            self.assertEqual(resumed['tasks']['data']['attempts'], 2)
            self.assertEqual(resumed['tasks']['collect']['attempt_history'][-1]['status'], 'PASS')
            self.assertTrue((run / recovery.TAG / 'archive/rolling_markers/collect.json').is_file())
            os.kill(restarted_pid, signal.SIGTERM)
            wait_for(lambda: manager_record().get('exit') is not None, 'Final manager stop timed out')

            # Simulate the final terminal ledger after all accepted test work
            # has completed. A repeated activation must report NOT_NEEDED,
            # rather than treating a normally exiting watcher as startup failure.
            for name in batch2.SCOPES:
                plan = rq._plan(run / 'scopes' / name / 'plan.json')
                state = rq._read_state(plan)
                for task in state['tasks'].values():
                    task['status'] = 'PASS'
                rq._atomic_json(Path(plan['run_dir']) / rq.STATE_NAME, state)
            final_rows = json.loads(database.read_text())
            for row in final_rows:
                if row['state'] == 'PENDING':
                    row.update(state='COMPLETED', exit_code='0:0')
            database.write_text(json.dumps(final_rows))
            finished_manager = manager_record()
            terminal = call(activate)
            self.assertEqual(terminal['manager_start'], 'NOT_NEEDED')
            self.assertTrue(terminal['terminal'])
            self.assertEqual(terminal['batch2_status'], 'COMPLETE')
            self.assertEqual(manager_record(), finished_manager)
            self.assertEqual(json.loads(database.read_text()), final_rows)
        finally:
            # Only the exact temporary-run watcher can be signalled, never an
            # unrelated PID that has been recycled after a test failure.
            for pid in children:
                try:
                    command = Path(f'/proc/{pid}/cmdline').read_bytes()
                    if str(fixture.run).encode() in command and b'watch' in command:
                        os.kill(pid, signal.SIGTERM)
                except (FileNotFoundError, ProcessLookupError):
                    pass
            if old_process is not None:
                try:
                    old_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            fixture.doCleanups()


if __name__ == '__main__':
    unittest.main()
