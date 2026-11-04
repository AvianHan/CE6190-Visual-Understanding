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

    def cached_manifest(self, *, dataset='coco', policy=None, protocol='legacy-protocol', audit=None):
        """Isolate preflight reuse checks; the external asset evaluator is mocked."""
        from ics_eval import manifest_fingerprint
        plan = self.plan()
        recipe = plan['manifests']['coco_f0']['prepare']
        recipe.update(dataset=dataset, fold=0)
        path = Path(plan['manifests']['coco_f0']['path'])
        manifest = {'schema_version': 1, 'status': 'ready', 'protocol_id': protocol,
                    'episodes': [{'fold': 0, 'dataset': dataset,
                                  'split': 'test' if dataset == 'fss' else 'val'}]}
        sidecar = {'manifest_sha256': manifest_fingerprint(manifest),
                   'sampling': {'queries_requested': 10, 'reference_sets_requested_per_query': 3}}
        if policy is not None:
            sidecar['sampling']['empty_mask_policy'] = policy
        if audit is not None:
            sidecar['sample_eligibility'] = audit
        path.write_text(json.dumps(manifest))
        path.with_suffix('.protocol.json').write_text(json.dumps(sidecar))
        return plan, path, sidecar

    def test_legacy_coco_manifest_reused_without_preparing_or_rewriting(self):
        plan, path, _ = self.cached_manifest()
        before = path.read_bytes(), path.with_suffix('.protocol.json').read_bytes()
        with mock.patch('ics.orchestration.subprocess.run') as call:
            run_preflight(plan)
        self.assertEqual(call.call_count, 1)
        self.assertTrue(call.call_args.args[0][1].endswith('ics_eval.py'))
        self.assertEqual(before, (path.read_bytes(), path.with_suffix('.protocol.json').read_bytes()))
        self.assertEqual(json.loads((Path(plan['run_dir'])/'preflight.complete.json').read_text())['status'], 'PASS')

    def test_reused_manifest_dataset_and_split_must_match_recipe(self):
        for field, requested in (('dataset', 'lvis'), ('split', 'train')):
            with self.subTest(field=field):
                plan, path, _ = self.cached_manifest()
                plan['manifests']['coco_f0']['prepare'][field] = requested
                before = path.read_bytes(), path.with_suffix('.protocol.json').read_bytes()
                with mock.patch('ics.orchestration.subprocess.run') as call:
                    with self.assertRaisesRegex(ValueError, 'different ' + field):
                        run_preflight(plan)
                call.assert_not_called()
                self.assertEqual(before, (path.read_bytes(), path.with_suffix('.protocol.json').read_bytes()))
                self.assertFalse((Path(plan['run_dir'])/'preflight.complete.json').exists())

    def test_reused_manifest_checks_every_episode_dataset_and_split(self):
        from ics_eval import manifest_fingerprint
        for field, changed in (('dataset', 'lvis'), ('split', 'train')):
            with self.subTest(field=field):
                plan, path, sidecar = self.cached_manifest()
                manifest = json.loads(path.read_text())
                manifest['episodes'].append({**manifest['episodes'][0], field: changed})
                sidecar['manifest_sha256'] = manifest_fingerprint(manifest)
                path.write_text(json.dumps(manifest))
                path.with_suffix('.protocol.json').write_text(json.dumps(sidecar))
                with mock.patch('ics.orchestration.subprocess.run') as call:
                    with self.assertRaisesRegex(ValueError, 'different ' + field):
                        run_preflight(plan)
                call.assert_not_called()

    def test_reused_manifest_reference_shortfall_policy_must_match_recipe(self):
        for stored, requested in ((None, 'error'), ('all', 'error'), ('error', None)):
            with self.subTest(stored=stored, requested=requested):
                plan, path, sidecar = self.cached_manifest()
                if stored is not None:
                    sidecar['sampling']['insufficient_references'] = stored
                    path.with_suffix('.protocol.json').write_text(json.dumps(sidecar))
                if requested is not None:
                    plan['manifests']['coco_f0']['prepare']['insufficient_references'] = requested
                with mock.patch('ics.orchestration.subprocess.run') as call:
                    with self.assertRaisesRegex(ValueError, 'different insufficient_references'):
                        run_preflight(plan)
                call.assert_not_called()

    def test_legacy_default_dataset_split_and_shortfall_policies_remain_reusable(self):
        for dataset in ('coco', 'lvis', 'fss'):
            with self.subTest(dataset=dataset):
                plan, path, _ = self.cached_manifest(dataset=dataset)
                before = path.read_bytes(), path.with_suffix('.protocol.json').read_bytes()
                with mock.patch('ics.orchestration.subprocess.run') as call:
                    run_preflight(plan)
                self.assertEqual(call.call_count, 1)
                self.assertEqual(before, (path.read_bytes(), path.with_suffix('.protocol.json').read_bytes()))

    def test_real_prepared_manifest_cannot_satisfy_different_dataset_recipe(self):
        """Real prepared PNG assets/manifest: no mocked evaluator or preparation."""
        import sys
        import numpy as np
        from PIL import Image
        from ics.data import DatasetIndex, Sample, create_manifest
        plan = self.plan()
        entry = plan['manifests']['coco_f0']
        data = Path(entry['dataset_root'])
        data.mkdir()
        samples = []
        for number in range(3):
            image, mask = data/f'image{number}.png', data/f'mask{number}.png'
            Image.fromarray(np.full((4, 4, 3), number*50, dtype=np.uint8)).save(image)
            Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(mask)
            samples.append(Sample('one', image, mask))
        index = DatasetIndex('coco', 'fixture', 'val', 0, {'one': samples})
        manifest, sidecar = create_manifest(index, data, queries=1)
        path = Path(entry['path'])
        path.write_text(json.dumps(manifest))
        path.with_suffix('.protocol.json').write_text(json.dumps(sidecar))
        entry['prepare'] = {'dataset': 'coco', 'fold': 0, 'queries': 1}
        plan['cluster']['resources']['preflight']['python'] = sys.executable
        run_preflight(plan)
        (Path(plan['run_dir'])/'preflight.complete.json').unlink()
        entry['prepare']['dataset'] = 'lvis'
        with self.assertRaisesRegex(ValueError, 'different dataset'):
            run_preflight(plan)
        self.assertFalse((Path(plan['run_dir'])/'preflight.complete.json').exists())

    def test_exclusion_policy_cannot_reuse_legacy_strict_manifest(self):
        plan, path, _ = self.cached_manifest(dataset='lvis')
        plan['manifests']['coco_f0']['prepare']['empty_mask_policy'] = 'exclude'
        with mock.patch('ics.orchestration.subprocess.run') as call:
            with self.assertRaisesRegex(ValueError, 'different empty_mask_policy'):
                run_preflight(plan)
        call.assert_not_called()
        self.assertTrue(path.exists())
        self.assertFalse((Path(plan['run_dir'])/'preflight.complete.json').exists())

    def test_exclusion_policy_requires_matching_versioned_audit(self):
        for audit in (None, {'policy': 'exclude', 'version': 'old'},
                      {'policy': 'error', 'version': 'lvis-nonempty-v2'}):
            with self.subTest(audit=audit):
                plan, _, _ = self.cached_manifest(dataset='lvis', policy='exclude', audit=audit)
                plan['manifests']['coco_f0']['prepare']['empty_mask_policy'] = 'exclude'
                with mock.patch('ics.orchestration.subprocess.run') as call:
                    with self.assertRaisesRegex(ValueError, 'eligibility audit'):
                        run_preflight(plan)
                call.assert_not_called()

    def test_explicit_protocol_identity_checked_before_reuse(self):
        plan, path, sidecar = self.cached_manifest()
        plan['manifests']['coco_f0']['prepare']['protocol_id'] = 'new-protocol'
        with mock.patch('ics.orchestration.subprocess.run') as call:
            with self.assertRaisesRegex(ValueError, 'different protocol_id'):
                run_preflight(plan)
            call.assert_not_called()
        plan['manifests']['coco_f0']['prepare']['protocol_id'] = 'legacy-protocol'
        sidecar['protocol_id'] = 'inconsistent-sidecar-protocol'
        path.with_suffix('.protocol.json').write_text(json.dumps(sidecar))
        with mock.patch('ics.orchestration.subprocess.run') as call:
            with self.assertRaisesRegex(ValueError, 'protocol_id mismatch'):
                run_preflight(plan)
            call.assert_not_called()

    def test_missing_manifest_forwards_policy_and_accepts_audited_result(self):
        audit = {'policy': 'exclude', 'version': 'lvis-nonempty-v2'}
        plan, path, sidecar = self.cached_manifest(dataset='lvis', policy='exclude',
                                                  protocol='lvis-nonempty-f0', audit=audit)
        manifest_bytes = path.read_bytes()
        path.unlink()
        path.with_suffix('.protocol.json').unlink()
        plan['prepare_missing_manifests'] = True
        plan['manifests']['coco_f0']['prepare'].update(
            empty_mask_policy='exclude', protocol_id='lvis-nonempty-f0')
        def execute(command, **kwargs):
            if command[1].endswith('prepare_episodes.py'):
                self.assertEqual(command[command.index('--empty-mask-policy')+1], 'exclude')
                self.assertEqual(command[command.index('--protocol-id')+1], 'lvis-nonempty-f0')
                path.write_bytes(manifest_bytes)
                path.with_suffix('.protocol.json').write_text(json.dumps(sidecar))
        with mock.patch('ics.orchestration.subprocess.run', side_effect=execute) as call:
            run_preflight(plan)
        self.assertEqual(call.call_count, 2)

    def test_invalid_empty_policy_rejected_before_preparation(self):
        for dataset, policy in (('lvis', 'silently-ignore'), ('coco', 'exclude')):
            with self.subTest(dataset=dataset, policy=policy):
                plan = self.plan()
                plan['prepare_missing_manifests'] = True
                plan['manifests']['coco_f0']['prepare'].update(dataset=dataset, empty_mask_policy=policy)
                with mock.patch('ics.orchestration.subprocess.run') as call:
                    with self.assertRaisesRegex(ValueError, 'empty_mask_policy'):
                        run_preflight(plan)
                call.assert_not_called()

    def test_lvis_recovery_suite_preserves_workload_and_coco_manifests(self):
        project = Path(__file__).resolve().parents[1]
        original = json.loads((project/'configs/suite.initial.json').read_text())
        recovery_path = project/'configs/suite.initial.lvisfix.json'
        recovery = json.loads(recovery_path.read_text())
        self.assertEqual(original['experiments'], recovery['experiments'])
        self.assertEqual(len(recovery['manifests']), 42)
        coco_count = lvis_count = 0
        for old, new in zip(original['manifests'], recovery['manifests']):
            if old['prepare']['dataset'] == 'coco':
                coco_count += 1
                self.assertEqual(old, new)
            else:
                lvis_count += 1
                self.assertEqual(new['path'], '${ICS_RUNTIME}/manifests/lvis-nonempty-v2/'+Path(old['path']).name)
                expected = copy.deepcopy(old)
                expected['path'] = new['path']
                expected['prepare']['empty_mask_policy'] = 'exclude'
                expected['prepare']['protocol_id'] = 'ce6190-unified-v2-nonempty-'+old['id']
                self.assertEqual(expected, new)
        self.assertEqual((coco_count, lvis_count), (12, 30))
        with mock.patch.dict(os.environ, {'ICS_RUNTIME': str(self.root/'runtime')}):
            plan = build_plan(recovery_path, project/'configs/cluster.orion.json', self.root/'recovery-run')
        self.assertEqual((len(plan['smoke']), len(plan['tasks'])), (36, 1484))
        self.assertEqual(len(plan['smoke'])+len(plan['tasks'])+3, 1523)

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
