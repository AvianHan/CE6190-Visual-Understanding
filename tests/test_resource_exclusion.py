"""Real Slurm command composition, with subprocess boundary mocked only."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SOURCE = Path(os.environ.get('ICS_TEST_SOURCE', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(SOURCE))
from ics.orchestration import resource_profile, sbatch_args
from ics.rolling_queue import _submit


class ResourceExclusionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {'ICS_RUNTIME': str(self.root / 'runtime')})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.cluster = json.loads((SOURCE / 'configs/cluster.orion.recovery.json').read_text())

    def command(self, resource):
        return sbatch_args(resource_profile(self.cluster, resource), name='test-exclusion', output=self.root / 'log')

    def test_recovery_change_is_only_mllm_node_exclusion(self):
        old = json.loads((SOURCE / 'configs/cluster.orion.json').read_text())
        current = copy.deepcopy(self.cluster)
        self.assertEqual(current['resources']['cgics_mllm'].pop('exclude'), '2FS9B74')
        old.pop('description')
        current.pop('description')
        self.assertEqual(current, old)

    def test_exclusion_passed_as_single_argv_value_and_not_required_nodes(self):
        cmd = self.command('cgics_mllm')
        self.assertEqual(cmd.count('--exclude'), 1)
        self.assertEqual(cmd[cmd.index('--exclude') + 1], '2FS9B74')
        self.assertNotIn('--nodelist', cmd)
        self.assertEqual(cmd[cmd.index('--gres') + 1], 'gpu:2')

    def test_cpu_insid3_visual_jobs_keep_all_eligible_nodes(self):
        for name in ('preflight', 'insid3', 'cgics_visual'):
            with self.subTest(resource=name):
                self.assertNotIn('--exclude', self.command(name))

    def test_profile_override_preserves_default_inheritance(self):
        self.cluster['defaults']['exclude'] = 'example-node'
        inherited = self.command('insid3')
        overridden = self.command('cgics_mllm')
        self.assertEqual(inherited[inherited.index('--exclude') + 1], 'example-node')
        self.assertEqual(overridden[overridden.index('--exclude') + 1], '2FS9B74')

    def test_exclusion_rejects_newlines(self):
        profile = resource_profile(self.cluster, 'cgics_mllm')
        for value in ('2FS9B74\nother', '2FS9B74\rother'):
            profile['exclude'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Invalid Slurm exclude'):
                sbatch_args(profile, name='test', output=self.root / 'log')

    def test_rolling_submission_uses_exclusion_in_actual_sbatch_call(self):
        run = self.root / 'run'
        run.mkdir()
        task = {'key': 'smoke-example', 'kind': 'smoke', 'resource': 'cgics_mllm',
                'job_name': 'icsq-example', 'attempts': 0, 'status': 'PENDING'}
        state = {'tasks': {task['key']: task}}
        plan = {'run_dir': str(run), 'cluster': self.cluster,
                'resources': {'cgics_mllm': resource_profile(self.cluster, 'cgics_mllm')}}
        result = mock.Mock(returncode=0, stdout='12345\n', stderr='')
        with mock.patch('ics.rolling_queue.subprocess.run', return_value=result) as launch:
            _submit(run / 'plan.json', plan, state, task, 100.0)
        cmd = launch.call_args.args[0]
        self.assertEqual(cmd[cmd.index('--exclude') + 1], '2FS9B74')
        self.assertEqual(task['status'], 'SUBMITTED')
        self.assertEqual(task['job_id'], '12345')
        self.assertTrue(Path(cmd[-1]).is_file())
        self.assertNotIn('shell', launch.call_args.kwargs)


if __name__ == '__main__':
    unittest.main()
