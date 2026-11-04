import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('diagnose_queue', ROOT / 'scripts/diagnose_queue.py')
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class DiagnoseQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.run = self.base / 'run'
        self.run.mkdir()
        (self.run / 'logs').mkdir()
        (self.run / 'rolling_markers').mkdir()
        (self.run / 'plan.json').write_text(json.dumps({'run_dir': str(self.run)}))
        self.tasks = {'data': {'key': 'data', 'status': 'PASS', 'job_id': '123'},
                      'preflight': {'key': 'preflight', 'status': 'RUNNING', 'job_id': '124_0'}}
        self.save_state()
        (self.run / 'manager.log').write_text('old\nmanager evidence\n')
        for key, task in self.tasks.items():
            (self.run / 'logs' / f'{key}-{task["job_id"]}.log').write_text(key + ' evidence\n')
            (self.run / 'rolling_markers' / (key + '.json')).write_text('{"status": "RUNNING"}')

    def save_state(self):
        (self.run / 'rolling_state.json').write_text(json.dumps({'tasks': self.tasks}))

    def invoke(self, responses=None, lines='80'):
        output = io.StringIO()
        default = subprocess.CompletedProcess([], 0, 'status evidence\n', '')
        with patch.object(diagnostics.subprocess, 'run', side_effect=responses,
                          return_value=default) as mocked, contextlib.redirect_stdout(output):
            self.assertEqual(diagnostics.main(['--run-dir', str(self.run), '--tail-lines', lines]), 0)
        return output.getvalue(), mocked

    def test_only_current_clone_status_and_readonly_accounting_are_called(self):
        text, mocked = self.invoke(lines='1')
        calls = mocked.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args[0], [sys.executable, '-I', str(ROOT / 'scripts/queue_ics.py'),
                                          'status', '--run-dir', str(self.run)])
        self.assertEqual(calls[1].args[0], ['sacct', '-n', '-P', '-X', '-j', '123,124_0',
                                         '--format=JobIDRaw,JobName,State,ExitCode,Start,End,Elapsed'])
        for call in calls:
            self.assertEqual(call.kwargs['timeout'], 30)
            self.assertNotIn('shell', call.kwargs)
        self.assertIn('manager evidence', text)
        self.assertNotIn('\nold\n', text)
        self.assertIn('preflight evidence', text)

    def test_failed_status_bad_state_and_missing_log_preserve_other_evidence(self):
        (self.run / 'rolling_state.json').write_text('not json')
        (self.run / 'rolling_markers/data.json').unlink()
        failed = subprocess.CompletedProcess([], 1, '', 'status failed')
        text, mocked = self.invoke([failed])
        self.assertEqual(mocked.call_count, 1)
        self.assertIn('status 返回码: 1', text)
        self.assertIn('manager evidence', text)
        self.assertIn('读取已保存任务失败', text)
        self.assertIn('rolling_markers/preflight.json', text)
        self.assertIn('RUNNING', text)
        self.assertIn('没有可查询', text)

    def test_secrets_are_redacted_and_sections_bounded_even_on_timeout(self):
        secret = 'hf_abc123XYZ'
        signed = 'https://example.test/weight.pth?Signature=privatevalue&Expires=123'
        (self.run / 'manager.log').write_text(secret + '\n' + signed + '\nauthorization: Bearer privatesecret\n')
        self.tasks['preflight']['error'] = 'access_token="secretword"'
        self.save_state()
        status = subprocess.CompletedProcess([], 0, secret + '\n' + 'x' * 50000, '')
        timeout = subprocess.TimeoutExpired(['sacct'], 30, output=secret.encode())
        text, mocked = self.invoke([status, timeout])
        self.assertEqual(mocked.call_count, 2)
        for value in (secret, 'privatevalue', 'privatesecret', 'secretword'):
            self.assertNotIn(value, text)
        self.assertIn('[REDACTED_HF_TOKEN]', text)
        self.assertIn('[QUERY_REDACTED]', text)
        self.assertIn('输出已截断', text)
        self.assertIn('sacct 查询超时', text)
        self.assertLess(len(text), 20000)

    def test_bad_job_ids_and_symlink_evidence_never_read_outside_run(self):
        outside = self.base / 'credential.txt'
        outside.write_text('CREDENTIAL_MUST_NOT_BE_READ')
        (self.run / 'manager.log').unlink()
        (self.run / 'manager.log').symlink_to(outside)
        (self.run / 'rolling_markers/data.json').unlink()
        (self.run / 'rolling_markers/data.json').symlink_to(outside)
        self.tasks['data']['key'] = '../../credential.txt'
        self.tasks['data']['job_id'] = '../../credential.txt'
        self.tasks['preflight']['job_id'] = '--help'
        self.save_state()
        text, mocked = self.invoke()
        self.assertEqual(mocked.call_count, 1)
        self.assertNotIn('CREDENTIAL_MUST_NOT_BE_READ', text)
        self.assertIn('拒绝路径越界或符号链接', text)
        self.assertIn('job ID 格式无效', text)
        self.assertIn('rolling_markers/preflight.json', text)

    def test_plan_cannot_redirect_status_to_another_run(self):
        (self.run / 'plan.json').write_text(json.dumps({'run_dir': str(self.base)}))
        text, mocked = self.invoke()
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(mocked.call_args.args[0][0], 'sacct')
        self.assertIn('plan.run_dir 与选定 run 不一致', text)
        self.assertIn('manager evidence', text)

    def test_tail_lines_bounds_reject_before_any_evidence_or_command(self):
        with patch.object(diagnostics.subprocess, 'run') as mocked:
            for value in ('0', '201'):
                with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as exc:
                        diagnostics.main(['--run-dir', str(self.run), '--tail-lines', value])
                    self.assertEqual(exc.exception.code, 2)
            mocked.assert_not_called()

    def test_actual_isolated_cli_is_readonly_and_does_not_print_environment(self):
        self.tasks = {}
        self.save_state()
        before = {str(p.relative_to(self.run)): p.read_bytes()
                  for p in self.run.rglob('*') if p.is_file()}
        result = subprocess.run([
            sys.executable, '-I', str(ROOT / 'scripts/diagnose_queue.py'),
            '--run-dir', str(self.run)
        ], cwd=self.base, env=dict(os.environ, HF_TOKEN='ENV_CREDENTIAL_MUST_STAY_PRIVATE'),
            capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('status 返回码: 1', result.stdout)  # Minimal plan lacks a fingerprint.
        self.assertIn('manager evidence', result.stdout)
        self.assertNotIn('ENV_CREDENTIAL_MUST_STAY_PRIVATE', result.stdout + result.stderr)
        after = {str(p.relative_to(self.run)): p.read_bytes()
                 for p in self.run.rglob('*') if p.is_file()}
        self.assertEqual(after, before)


if __name__ == '__main__':
    unittest.main()
