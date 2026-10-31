"""Synthetic scheduler tests: no sbatch, GPUs, data downloads, or model inference."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ics.orchestration import (build_plan, write_plan, plan_summary, sbatch_args,
                               split_devices, submit_slurm, DevicePool, expand, safe_id, run_preflight)


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = self.root / 'experiment.json'
        self.cfg.write_text(json.dumps({'experiment_id':'example', 'method':'insid3', 'variant':'full',
                                       'seed':0, 'adapter':{'variant':'full'}}))
        self.suite = self.root / 'suite.json'
        self.suite.write_text(json.dumps({'schema_version':1, 'smoke_episodes':2,
            'manifests':[{'id':'coco_f0', 'path':str(self.root/'manifest.json'), 'dataset_root':str(self.root/'data'),
                          'prepare':{'queries':10,'references_per_query':3}}],
            'experiments':[{'config':str(self.cfg),'resource':'insid3','manifest_ids':['coco_f0'],'shards':2}]}))
        self.cluster = self.root/'cluster.json'
        self.cluster.write_text(json.dumps({'schema_version':1,'max_parallel':2,
            'defaults':{'account':'example-account','partition':'example-partition','time':'00:30:00','cpus':2},
            'resources':{'preflight':{'python':'/usr/bin/python3','gpus':0},
                         'insid3':{'python':'/usr/bin/python3','gpus':1}}}))

    def plan(self):
        return build_plan(self.suite, self.cluster, self.root/'run')

    def test_plan_only_creates_no_submission(self):
        p = self.plan()
        self.assertEqual(len(p['tasks']),2)
        self.assertEqual(len(p['smoke']),1)
        self.assertEqual(plan_summary(p)['resources']['insid3']['requested_episode_estimate'],30)
        path = write_plan(p)
        self.assertTrue(path.exists())
        self.assertFalse((path.parent/'submission.json').exists())
        self.assertEqual(p['tasks'][0]['output'],p['tasks'][1]['output'])

    def test_changed_plan_cannot_reuse_run_directory(self):
        p=self.plan();write_plan(p)
        c=json.loads(self.cfg.read_text());c['seed']=2;self.cfg.write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError,'different plan'):
            write_plan(self.plan())

    def test_actual_slurm_resources_required(self):
        with self.assertRaisesRegex(ValueError,'actual account'):
            sbatch_args({'gpus':1},name='example',output=self.root/'log')
        cmd=sbatch_args({'account':'a','partition':'p','time':'01:00:00','gpus':2},
                        name='example',output=self.root/'log',array='0-9%2',dependency='afterok:42')
        self.assertIn('gpu:2',cmd)
        self.assertIn('afterok:42',cmd)
        self.assertIn('--kill-on-invalid-dep=yes',cmd)

    def test_submission_idempotent_and_dependencies(self):
        path=write_plan(self.plan())
        n=[100]
        def sbatch(cmd,**kwargs):
            n[0]+=1
            return mock.Mock(stdout=str(n[0])+';cluster\n')
        with mock.patch('ics.orchestration.slurm_array_limit', return_value=1000), mock.patch('ics.orchestration.shutil.which',return_value='/usr/bin/sbatch'), mock.patch('ics.orchestration.subprocess.run',side_effect=sbatch) as launch:
            state=submit_slurm(path)
            self.assertEqual(launch.call_count,4)
            again=submit_slurm(path)
            self.assertEqual(launch.call_count,4)
        self.assertEqual(state['jobs'],again['jobs'])
        self.assertIn('afterok:101',state['jobs']['smoke-insid3']['command'])
        self.assertIn('afterany:102',state['jobs']['experiments-insid3']['command'])
        self.assertIn('afterany:103',state['jobs']['collect']['command'])

    def test_array_chunk_offsets_and_bounded_resource_dependency(self):
        path=write_plan(self.plan())
        number=[100]
        def sbatch(cmd,**kwargs):
            number[0]+=1
            return mock.Mock(stdout=str(number[0]))
        with mock.patch('ics.orchestration.slurm_array_limit',return_value=1), mock.patch('ics.orchestration.shutil.which',return_value='/usr/bin/sbatch'), mock.patch('ics.orchestration.subprocess.run',side_effect=sbatch):
            state=submit_slurm(path)
        self.assertEqual(len(state['jobs']),5)
        second=state['jobs']['experiments-insid3-part001']
        self.assertIn('afterany:102:103',second['command'])
        self.assertIn('--offset 1',(path.parent/'slurm'/'experiments-insid3-part001.sh').read_text())

    def test_reused_manifest_recipe_mismatch_is_not_silently_accepted(self):
        plan=self.plan()
        path=Path(plan['manifests']['coco_f0']['path'])
        path.write_text(json.dumps({'episodes':[]}))
        path.with_suffix('.protocol.json').write_text(json.dumps({'sampling':{'queries_requested':9}}))
        with mock.patch('ics.orchestration.subprocess.run') as call:
            with self.assertRaisesRegex(ValueError,'different queries'):
                run_preflight(plan)
            call.assert_not_called()

    def test_missing_concurrency_does_not_submit_preflight(self):
        c=json.loads(self.cluster.read_text());c['max_parallel']=None;self.cluster.write_text(json.dumps(c))
        path=write_plan(self.plan())
        with mock.patch('ics.orchestration.shutil.which',return_value='/usr/bin/sbatch'), mock.patch('ics.orchestration.subprocess.run') as launch:
            with self.assertRaisesRegex(ValueError,'max_parallel'):
                submit_slurm(path)
            launch.assert_not_called()

    def test_pending_task_budget_rejected_before_any_submission(self):
        c=json.loads(self.cluster.read_text());c['max_submit_tasks']=4;self.cluster.write_text(json.dumps(c))
        path=write_plan(self.plan())  # 1 smoke +2 production elements +preflight +collect =5
        with mock.patch('ics.orchestration.shutil.which',return_value='/usr/bin/sbatch'), mock.patch('ics.orchestration.subprocess.run') as launch:
            with self.assertRaisesRegex(ValueError,'max_submit_tasks=4'):
                submit_slurm(path)
            launch.assert_not_called()

    def test_device_assignment_is_explicit_and_disjoint(self):
        self.assertEqual(split_devices(['GPU-a','GPU-b','GPU-c'],3),(['GPU-a'],['GPU-b','GPU-c']))
        with self.assertRaises(RuntimeError):
            split_devices(['0'],2)
        pool=DevicePool(['0','1','2']);a=pool.acquire(2);b=pool.acquire(1)
        self.assertFalse(set(a)&set(b));pool.release(a);pool.release(b)
        with self.assertRaises(ValueError):
            DevicePool(['0','0'])

    def test_unresolved_env_and_unsafe_identifiers_rejected(self):
        with mock.patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'Unresolved'):
                expand('${ICS_RUNTIME}/data')
        with self.assertRaises(ValueError):
            safe_id('../escape')

    def test_owned_mllm_requires_separate_gpus(self):
        c=json.loads(self.cluster.read_text());c['resources']['cgics_mllm']={'gpus':1,'python':'/usr/bin/python3'}
        self.cluster.write_text(json.dumps(c))
        s=json.loads(self.suite.read_text());s['experiments'][0]['resource']='cgics_mllm';self.suite.write_text(json.dumps(s))
        with self.assertRaisesRegex(ValueError,'separate allocated GPUs'):
            self.plan()

if __name__=='__main__':
    unittest.main()
