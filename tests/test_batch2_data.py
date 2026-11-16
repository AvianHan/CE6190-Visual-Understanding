"""Synthetic evidence contracts, not claims of real dataset/model success."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from ics.batch2_data import (coco_image_identity, collect_prior_queries,
    create_fresh_manifest, prepare_scope, verify_scope, _publish_json)
from ics.data import DatasetIndex, Sample
from ics_eval import inference_view, manifest_fingerprint, validate_manifest


class Batch2DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = Path(self.temp.name) / 'runtime'
        self.root = self.runtime / 'data'
        self.root.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def sample(self, number, cid='dog', *, source_id=None, lvis=False,
               empty=False, group=None, coco_layout=None):
        if coco_layout == '2014':
            relative = f'COCO2014/train2014/COCO_train2014_{(source_id or number):012d}.jpg'
        elif coco_layout == '2017':
            relative = f'LVIS/coco/train2017/{(source_id or number):012d}.jpg'
        else:
            relative = f'images/{cid}_{number}.png'
        image = self.root / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (3, 2), (number % 250, number // 250, 17)).save(image)
        mask = self.root / 'masks' / f'{cid}_{number}.png'
        mask.parent.mkdir(exist_ok=True)
        array = np.zeros((2, 3), dtype=np.uint8)
        if not empty:
            array[0, 0] = 1
        Image.fromarray(array).save(mask)
        return Sample(cid, image, [array] if lvis else mask,
                      'lvis' if lvis else 'binary', group_id=group)

    def index(self, samples, dataset='synthetic', fold=0):
        return DatasetIndex(dataset, 'fixture-original-union', 'test', fold, samples,
                            {'source_files': [], 'class_universe': list(samples)})

    def old_runs(self, manifests):
        runs = []
        for index in range(2):
            run = self.runtime / 'runs' / f'prior{index}'
            run.mkdir(parents=True)
            tasks = []
            for j, manifest in enumerate(manifests):
                path = self.runtime / 'manifests' / f'old-{j}.json'
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(manifest))
                output = run / 'results' / f'output-{j}'
                output.mkdir(parents=True)
                # Parent may not have model bindings; recovery evidence covers
                # them, as in the actual failed first run.
                if index == 1:
                    (output / 'run.json').write_text(json.dumps({'binding': {
                        'manifest_sha256': manifest_fingerprint(manifest),
                        'protocol_id': manifest['protocol_id'],
                        'selected_episode_ids': [e['episode_id'] for e in manifest['episodes']]}}))
                tasks.append({'id': f'job-{j}', 'manifest': str(path), 'output': str(output),
                              'phase': 'experiment', 'dataset_root': str(self.root)})
            (run / 'plan.json').write_text(json.dumps({'run_dir': str(run), 'tasks': tasks,
                                                      'fingerprint': f'fixture-{index}'}))
            runs.append(run)
        return runs

    def old_fixture(self):
        items = [self.sample(i + 1) for i in range(3)]
        manifest, _ = create_fresh_manifest(self.index({'dog': items}), self.root,
                                           queries=1, references_per_query=1)
        return manifest

    def test_canonical_source_identity_catches_different_encoding(self):
        a = self.sample(11, source_id=500, coco_layout='2014')
        b = self.sample(99, source_id=500, coco_layout='2017')
        self.assertNotEqual(hashlib.sha256(a.image.read_bytes()).hexdigest(),
                            hashlib.sha256(b.image.read_bytes()).hexdigest())
        self.assertEqual(coco_image_identity(a.image, 'coco'), coco_image_identity(b.image, 'lvis'))
        with self.assertRaisesRegex(ValueError, 'canonical COCO'):
            coco_image_identity('some_unverified_image.jpg', 'coco')
        self.assertIsNone(coco_image_identity('anything.jpg', 'suim'))

    def test_deterministic_fresh_unique_and_gt_isolated(self):
        pool = [self.sample(n + 1) for n in range(8)]
        old_sha = hashlib.sha256(pool[0].image.read_bytes()).hexdigest()
        kwargs = dict(queries=5, references_per_query=3, seed=71, excluded_sha256=[old_sha])
        a, sa = create_fresh_manifest(self.index({'dog': pool}), self.root, **kwargs)
        b, sb = create_fresh_manifest(self.index({'dog': pool}), self.root, **kwargs)
        self.assertEqual((a, sa), (b, sb))
        self.assertEqual(len(a['episodes']), 15)
        self.assertEqual(sa['sampling']['unique_query_images'], 5)
        self.assertNotIn(old_sha, {ep['query']['image_sha256'] for ep in a['episodes']})
        validate_manifest(a, root=self.root)
        inputs = inference_view(a['episodes'][0])
        self.assertEqual(set(inputs), {'query_image', 'references'})
        self.assertNotIn('concept_id', inputs)
        self.assertNotIn(a['episodes'][0]['query']['mask'], json.dumps(inputs))

    def test_cross_dataset_source_ids_excluded_and_shared_unique(self):
        c = [self.sample(100 + i, source_id=1000 + i, coco_layout='2014') for i in range(5)]
        l = [self.sample(200 + i, source_id=1000 + i, coco_layout='2017', lvis=True) for i in range(5)]
        used = set()
        mc, _ = create_fresh_manifest(self.index({'0': c}, 'coco'), self.root,
                                     queries=2, references_per_query=1, used_query_identities=used)
        selected_ids = {coco_image_identity(e['query']['image'], 'coco') for e in mc['episodes']}
        ml, _ = create_fresh_manifest(self.index({'0': l}, 'lvis'), self.root,
                                     queries=2, references_per_query=1, used_query_identities=used)
        lvis_ids = {coco_image_identity(e['query']['image'], 'lvis') for e in ml['episodes']}
        self.assertFalse(selected_ids & lvis_ids)
        blocked_id = coco_image_identity(l[0].image, 'lvis')
        ml2, _ = create_fresh_manifest(self.index({'0': l}, 'lvis'), self.root,
                                      queries=3, references_per_query=1,
                                      excluded_source_ids=[blocked_id])
        self.assertNotIn(blocked_id, {coco_image_identity(e['query']['image'], 'lvis') for e in ml2['episodes']})

    def test_query_identity_uniqueness_across_concepts(self):
        dog = [self.sample(10 + i) for i in range(4)]
        bird = [Sample('bird', x.image, x.mask_source) for x in dog]
        manifest, side = create_fresh_manifest(self.index({'dog': dog, 'bird': bird}), self.root,
                                               queries=4, references_per_query=1)
        self.assertEqual(len({ep['query']['image_sha256'] for ep in manifest['episodes']}), 4)
        self.assertEqual(side['sampling']['query_repeats'], 0)

    def test_insufficient_fresh_queries_fail_without_repetition_or_used_mutation(self):
        pool = [self.sample(n + 1) for n in range(3)]
        used = {'sha256:' + 'a' * 64}
        original = set(used)
        with self.assertRaisesRegex(ValueError, 'only 3 eligible; no repetitions'):
            create_fresh_manifest(self.index({'dog': pool}), self.root, queries=4,
                                  used_query_identities=used)
        self.assertEqual(used, original)

    def test_reference_shortfall_is_explicit_and_unique(self):
        pool = [self.sample(n + 1) for n in range(3)]
        manifest, side = create_fresh_manifest(self.index({'dog': pool}), self.root,
                                               queries=1, references_per_query=5)
        self.assertEqual(len(manifest['episodes']), 2)
        self.assertEqual(side['sampling']['reference_shortfall_queries'], 1)
        self.assertEqual(side['queries'][0]['actual_reference_sets'], 2)
        refs = {ep['references'][0]['image_sha256'] for ep in manifest['episodes']}
        self.assertEqual(len(refs), 2)

    def test_reference_reencoding_of_query_is_not_eligible(self):
        a = self.sample(11, source_id=400, coco_layout='2014')
        b = self.sample(55, source_id=400, coco_layout='2017')
        c = self.sample(66, source_id=401, coco_layout='2017')
        m, _ = create_fresh_manifest(self.index({'0': [a, b, c]}, 'coco'), self.root,
                                     queries=2, references_per_query=8)
        for ep in m['episodes']:
            self.assertNotEqual(coco_image_identity(ep['query']['image'], 'coco'),
                                coco_image_identity(ep['references'][0]['image'], 'coco'))

    def test_unrepresented_fresh_classes_are_disclosed(self):
        dog = [self.sample(1 + i, 'dog') for i in range(3)]
        cat = [self.sample(20 + i, 'cat') for i in range(4)]
        old = [hashlib.sha256(x.image.read_bytes()).hexdigest() for x in dog]
        _, side = create_fresh_manifest(self.index({'dog': dog, 'cat': cat}), self.root,
                                       queries=3, references_per_query=1, excluded_sha256=old)
        self.assertEqual(side['batch2']['class_universe'], ['dog', 'cat'])
        self.assertEqual(side['batch2']['unrepresented_fresh_concepts'], ['dog'])
        self.assertEqual(side['batch2']['actual_query_counts_by_class'], {'cat': 3})

    def test_lvis_empty_union_policy_preserves_original_class_universe(self):
        pool = [self.sample(30 + i, '0', lvis=True, empty=i == 0, coco_layout='2017') for i in range(4)]
        m, side = create_fresh_manifest(self.index({'0': pool}, 'lvis', fold=7), self.root,
                                       queries=2, references_per_query=3)
        self.assertEqual(side['sample_eligibility']['excluded_count'], 1)
        self.assertEqual(side['sample_eligibility']['scope'], 'all_candidates_before_sampling')
        self.assertEqual({ep['fold'] for ep in m['episodes']}, {7})
        self.assertEqual(side['batch2']['class_universe'], ['0'])
        pool[1].mask_source = [np.zeros((2, 3), np.uint8)]
        pool[2].mask_source = [np.zeros((2, 3), np.uint8)]
        with self.assertRaisesRegex(ValueError, 'no class was dropped'):
            create_fresh_manifest(self.index({'0': pool}, 'lvis'), self.root, queries=1)

    def test_lvis_malformed_dimensions_remain_fatal(self):
        pool = [self.sample(40 + i, '0', lvis=True, coco_layout='2017') for i in range(3)]
        pool[0].mask_source = [np.zeros((3, 3), np.uint8)]
        with self.assertRaisesRegex(ValueError, 'dimensions differ'):
            create_fresh_manifest(self.index({'0': pool}, 'lvis'), self.root, queries=1)

    def test_tampered_generated_mask_is_preserved(self):
        index = self.index({'dog': [self.sample(1 + i) for i in range(3)]})
        m, _ = create_fresh_manifest(index, self.root, queries=1)
        path = self.root / m['episodes'][0]['query']['mask']
        path.write_bytes(b'broken')
        with self.assertRaises(OSError):
            create_fresh_manifest(index, self.root, queries=1)
        self.assertEqual(path.read_bytes(), b'broken')

    def test_parent_plus_recovery_all_formal_manifests_bound(self):
        old = self.old_fixture()
        old2 = copy.deepcopy(old)
        old2['protocol_id'] += '-reference'
        old2['episodes'][0]['episode_id'] += '-reference'
        runs = self.old_runs([old, old2])
        ledger = collect_prior_queries(runs, self.runtime)
        self.assertEqual(len(ledger['plans']), 2)
        self.assertEqual(len(ledger['manifests']), 2)
        self.assertEqual(ledger['query_image_sha256_count'], 1)
        self.assertTrue(all(x['bindings'] for x in ledger['manifests']))

    def test_old_manifest_binding_mismatch_is_fatal(self):
        runs = self.old_runs([self.old_fixture()])
        file = runs[1] / 'results/output-0/run.json'
        record = json.loads(file.read_text())
        record['binding']['manifest_sha256'] = '0' * 64
        file.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'disagrees with original binding'):
            collect_prior_queries(runs, self.runtime)

    def test_missing_old_bindings_are_not_silently_accepted(self):
        runs = self.old_runs([self.old_fixture()])
        (runs[1] / 'results/output-0/run.json').unlink()
        with self.assertRaisesRegex(ValueError, 'No original output binding'):
            collect_prior_queries(runs, self.runtime)

    def test_immutable_json_idempotence_and_conflict_preserves_bytes(self):
        path = self.runtime / 'publish.json'
        _publish_json(path, {'x': 1})
        original = path.read_bytes()
        _publish_json(path, {'x': 1})
        with self.assertRaisesRegex(ValueError, 'differs; preserved'):
            _publish_json(path, {'x': 2})
        self.assertEqual(path.read_bytes(), original)

    def test_prepare_resume_verify_and_changed_request_rejected(self):
        old = self.old_fixture()
        runs = self.old_runs([old])
        new = self.index({'dog': [self.sample(100 + i) for i in range(6)]})
        output = self.runtime / 'batch2/core'
        with mock.patch('ics.batch2_data.CORE_FOLDS', (('synthetic', 1),)), \
             mock.patch('ics.batch2_data.build_index', return_value=new):
            receipt = prepare_scope(self.runtime, output, prior_runs=runs,
                                    queries_per_fold=3, min_free_gib=0)
            repeated = prepare_scope(self.runtime, output, prior_runs=runs,
                                     queries_per_fold=3, min_free_gib=0)
            self.assertEqual(receipt, repeated)
            self.assertEqual(receipt['query_images'], 3)
            self.assertEqual(receipt['status'], 'READY')
            with self.assertRaisesRegex(ValueError, 'different request'):
                prepare_scope(self.runtime, output, prior_runs=runs,
                              queries_per_fold=4, min_free_gib=0)
        verify_scope(output, runtime=self.runtime, prior_runs=runs, expected_scope='core')
        path = Path(next(iter(receipt['manifests'].values()))['path'])
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            verify_scope(output)

    def test_changed_prior_evidence_blocks_resume(self):
        runs = self.old_runs([self.old_fixture()])
        new = self.index({'dog': [self.sample(100 + i) for i in range(6)]})
        output = self.runtime / 'batch2/core'
        with mock.patch('ics.batch2_data.CORE_FOLDS', (('synthetic', 1),)), \
             mock.patch('ics.batch2_data.build_index', return_value=new):
            prepare_scope(self.runtime, output, prior_runs=runs, queries_per_fold=2, min_free_gib=0)
        (runs[0] / 'plan.json').write_text((runs[0] / 'plan.json').read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'Prior source changed'):
            verify_scope(output)

    def test_selected_asset_hash_change_blocks_resume(self):
        runs = self.old_runs([self.old_fixture()])
        new = self.index({'dog': [self.sample(100 + i) for i in range(6)]})
        output = self.runtime / 'batch2/core'
        with mock.patch('ics.batch2_data.CORE_FOLDS', (('synthetic', 1),)), \
             mock.patch('ics.batch2_data.build_index', return_value=new):
            receipt = prepare_scope(self.runtime, output, prior_runs=runs,
                                    queries_per_fold=2, min_free_gib=0)
        m = json.loads(Path(next(iter(receipt['manifests'].values()))['path']).read_text())
        (self.root / m['episodes'][0]['query']['image']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            verify_scope(output)

    def test_space_guard_runs_before_preparation(self):
        runs = self.old_runs([self.old_fixture()])
        with self.assertRaisesRegex(RuntimeError, 'Free disk below'):
            prepare_scope(self.runtime, self.runtime / 'blocked', prior_runs=runs,
                          min_free_gib=10**12)

    def test_nonfinite_guard_and_invalid_seed_rejected(self):
        for guard in (float('nan'), float('inf'), True, -1):
            with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                prepare_scope(self.runtime, self.runtime / 'blocked', min_free_gib=guard)
        for seed in (True, -1, 1.5):
            with self.assertRaisesRegex(ValueError, 'integer seed'):
                prepare_scope(self.runtime, self.runtime / 'blocked', seed=seed)

    def test_exclusion_generators_are_counted_and_enforced(self):
        pool = [self.sample(100 + i) for i in range(5)]
        sha = hashlib.sha256(pool[0].image.read_bytes()).hexdigest()
        manifest, side = create_fresh_manifest(self.index({'dog': pool}), self.root,
            queries=3, excluded_sha256=(x for x in [sha]))
        self.assertEqual(side['batch2']['old_query_image_sha256_count'], 1)
        self.assertNotIn(sha, {e['query']['image_sha256'] for e in manifest['episodes']})

    def test_generated_mask_symlink_escape_is_not_written(self):
        pool = [self.sample(100 + i) for i in range(3)]
        external = Path(self.temp.name) / 'external'
        external.mkdir()
        (self.root / 'ics_generated_masks').symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escapes dataset root'):
            create_fresh_manifest(self.index({'dog': pool}), self.root, queries=1)
        self.assertEqual(list(external.iterdir()), [])

    def test_membership_source_change_blocks_resume(self):
        runs = self.old_runs([self.old_fixture()])
        new = self.index({'dog': [self.sample(100 + i) for i in range(6)]})
        membership = self.root / 'membership.fixture'
        membership.write_bytes(b'original cohort membership')
        new.provenance['source_files'] = [{'path': str(membership),
            'sha256': hashlib.sha256(membership.read_bytes()).hexdigest()}]
        output = self.runtime / 'batch2/core'
        with mock.patch('ics.batch2_data.CORE_FOLDS', (('synthetic', 1),)), \
             mock.patch('ics.batch2_data.build_index', return_value=new):
            prepare_scope(self.runtime, output, prior_runs=runs, queries_per_fold=2, min_free_gib=0)
        membership.write_bytes(b'changed membership')
        with self.assertRaisesRegex(ValueError, 'Dataset membership source changed'):
            verify_scope(output)

    def test_transfer_acquisition_is_scope_local_and_not_repeated_on_resume(self):
        runs = self.old_runs([self.old_fixture()])
        new = self.index({'dog': [self.sample(100 + i) for i in range(6)]}, dataset='isic')
        output = self.runtime / 'batch2/isic'
        with mock.patch('ics.batch2_sources.ensure_transfer_sources', return_value={'status': 'VERIFIED'}) as acquire, \
             mock.patch('ics.batch2_data.build_index', return_value=new):
            receipt = prepare_scope(self.runtime, output, scope='isic', prior_runs=runs,
                transfer_queries=2, min_free_gib=0, allow_download=True)
            acquire.assert_called_once_with(self.runtime, 'isic', allow_download=True, min_free_gib=0)
            self.assertEqual(receipt['source_preparation'], {'status': 'VERIFIED'})
            repeated = prepare_scope(self.runtime, output, scope='isic', prior_runs=runs,
                transfer_queries=2, min_free_gib=0, allow_download=True)
            self.assertEqual(repeated, receipt)
            self.assertEqual(acquire.call_count, 1)

    def test_failed_transfer_does_not_block_core(self):
        runs = self.old_runs([self.old_fixture()])
        with mock.patch('ics.batch2_sources.ensure_transfer_sources', side_effect=FileNotFoundError('transfer missing')):
            with self.assertRaisesRegex(FileNotFoundError, 'transfer missing'):
                prepare_scope(self.runtime, self.runtime / 'batch2/isic', scope='isic',
                              prior_runs=runs, min_free_gib=0)
        new = self.index({'dog': [self.sample(100 + i) for i in range(6)]})
        with mock.patch('ics.batch2_data.CORE_FOLDS', (('synthetic', 1),)), \
             mock.patch('ics.batch2_data.build_index', return_value=new), \
             mock.patch('ics.batch2_sources.ensure_transfer_sources') as acquire:
            result = prepare_scope(self.runtime, self.runtime / 'batch2/core', prior_runs=runs,
                                   queries_per_fold=2, min_free_gib=0)
            self.assertEqual(result['status'], 'READY')
            acquire.assert_not_called()


if __name__ == '__main__':
    unittest.main()
