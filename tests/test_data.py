"""Synthetic contract tests; these are NOT segmentation model results."""
from __future__ import annotations
import importlib.util
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
from PIL import Image

from ics.data import DatasetIndex, Sample, build_index, create_manifest, read_binary_mask
from ics_eval import inference_view, validate_manifest


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def sample(self, cid, n, *, group=None, semantic=False):
        image = self.root / f'{cid}_{n}.png'
        Image.new('RGB', (3, 2), (int(n) * 20, 10, 15)).save(image)
        mask = self.root / f'{cid}_{n}_mask.png'
        if semantic:
            values = np.array([[1, 255, 0], [0, 1, 0]], dtype=np.uint8)
        else:
            values = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        Image.fromarray(values).save(mask)
        return Sample(str(cid), image, mask, 'semantic' if semantic else 'binary',
                      1 if semantic else None, 255 if semantic else None, group)

    def index(self, samples):
        return DatasetIndex('synthetic', 'fixture', 'test', 0, samples)

    def test_deterministic_and_schema_and_gt_boundary(self):
        index = self.index({'dog': [self.sample('dog', n) for n in range(4)]})
        a, side = create_manifest(index, self.root, queries=3, references_per_query=2, seed=17)
        b, _ = create_manifest(index, self.root, queries=3, references_per_query=2, seed=17)
        self.assertEqual(a, b)
        validate_manifest(a, root=self.root)
        model_inputs = inference_view(a['episodes'][0])
        self.assertEqual(set(model_inputs), {'query_image', 'references'})
        self.assertNotIn('concept_id', model_inputs)
        self.assertNotIn(a['episodes'][0]['query']['mask'], json.dumps(model_inputs))
        self.assertEqual(side['sampling']['unique_query_concept_pairs'], 3)

    def test_query_draws_independent_of_reference_budget(self):
        index = self.index({'dog': [self.sample('dog', n) for n in range(5)]})
        a, sa = create_manifest(index, self.root, queries=4, references_per_query=1)
        b, sb = create_manifest(index, self.root, queries=4, references_per_query=4)
        self.assertEqual([x['query_id'] for x in sa['queries']], [x['query_id'] for x in sb['queries']])
        self.assertEqual(len(a['episodes']), 4)
        self.assertEqual(len(b['episodes']), 16)

    def test_modified_mask_cache_is_rejected_not_blessed(self):
        index = self.index({'dog': [self.sample('dog', n) for n in range(3)]})
        manifest, _ = create_manifest(index, self.root, queries=1)
        cached_path = self.root / manifest['episodes'][0]['query']['mask']
        wrong = np.zeros((2, 3), dtype=np.uint8)
        Image.fromarray(wrong).save(cached_path)
        tampered_bytes = cached_path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Generated mask cache mismatch'):
            create_manifest(index, self.root, queries=1)
        self.assertEqual(cached_path.read_bytes(), tampered_bytes)

    def test_truncated_http_is_not_promoted_to_complete_archive(self):
        path = Path(__file__).resolve().parents[1] / 'scripts' / 'download_data.py'
        spec = importlib.util.spec_from_file_location('ics_download_http_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        response = io.BytesIO(b'partial')
        response.status = 200
        response.headers = {'Content-Length': '100'}
        target = self.root / 'download.zip'
        with mock.patch.object(module, 'urlopen', return_value=response):
            with self.assertRaisesRegex(OSError, 'Incomplete HTTP download'):
                module.download_http('https://example.invalid/data.zip', target)
        self.assertFalse(target.exists())
        self.assertEqual(target.with_suffix('.zip.part').read_bytes(), b'partial')

    def test_available_reference_cap_and_no_duplicates(self):
        items = [self.sample('dog', n) for n in range(3)]
        index = self.index({'dog': items})
        manifest, side = create_manifest(index, self.root, queries=1, references_per_query=50)
        self.assertEqual(len(manifest['episodes']), 2)
        self.assertEqual(side['queries'][0]['actual_reference_sets'], 2)
        refs = [e['references'][0]['image_sha256'] for e in manifest['episodes']]
        self.assertEqual(len(set(refs)), 2)
        with self.assertRaises(ValueError):
            create_manifest(index, self.root, queries=1, references_per_query=50, insufficient_references='error')

    def test_content_duplicates_are_excluded(self):
        first, second, third = [self.sample('dog', n) for n in range(3)]
        second.image.write_bytes(first.image.read_bytes())
        manifest, _ = create_manifest(self.index({'dog': [first, second, third]}), self.root,
                                      queries=1, references_per_query=50)
        self.assertEqual(len(manifest['episodes']), 1)
        self.assertNotEqual(manifest['episodes'][0]['query']['image_sha256'],
                            manifest['episodes'][0]['references'][0]['image_sha256'])

    def test_group_exclusion(self):
        items = [self.sample('dog', n, group='sceneA' if n < 2 else 'sceneB') for n in range(4)]
        manifest, _ = create_manifest(self.index({'dog': items}), self.root, queries=4, references_per_query=5)
        groups = {s.image.name: s.group_id for s in items}
        for ep in manifest['episodes']:
            qgroup = groups[Path(ep['query']['image']).name]
            self.assertNotEqual(qgroup, groups[Path(ep['references'][0]['image']).name])

    def test_ignore_not_foreground_in_reference(self):
        item = self.sample('dog', 1, semantic=True)
        q = read_binary_mask(item)
        r = read_binary_mask(item, reference=True)
        self.assertEqual(q[0, 1], 255)
        self.assertEqual(r[0, 1], 0)
        self.assertEqual(set(np.unique(r)), {0, 1})

    def test_lvis_instance_union(self):
        item = self.sample('dog', 1)
        first = np.array([[1, 0, 0], [0, 0, 0]], dtype=bool)
        second = np.array([[0, 0, 0], [0, 0, 1]], dtype=bool)
        item.mask_kind = 'lvis'
        item.mask_source = [first, second]
        np.testing.assert_array_equal(read_binary_mask(item), (first | second).astype(np.uint8))

    def test_lvis_fold_uses_author_order_and_pickle_guard(self):
        base = self.root / 'LVIS'
        base.mkdir()
        metadata = {cid: {'val2017/a.jpg': {'annotations': []}, 'val2017/b.jpg': {'annotations': []}}
                    for cid in [200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]}
        with (base / 'lvis_val.pkl').open('wb') as stream:
            pickle.dump(metadata, stream)
        with self.assertRaises(ValueError):
            build_index('lvis', self.root)
        index = build_index('lvis', self.root, fold=0, trust_prepared_pickle=True)
        self.assertEqual(list(index.samples), ['200'])
        self.assertEqual(index.provenance['category_remainder_excluded_by_author_rule'], 1)

    def test_nshot_support_sets_distinct(self):
        items = [self.sample('dog', n) for n in range(4)]
        manifest, side = create_manifest(self.index({'dog': items}), self.root,
            queries=1, references_per_query=10, shots=2)
        self.assertEqual(len(manifest['episodes']), 3)
        combinations = {tuple(sorted(r['image_sha256'] for r in ep['references'])) for ep in manifest['episodes']}
        self.assertEqual(len(combinations), 3)
        self.assertEqual(side['queries'][0]['actual_reference_sets'], 3)

    def test_class_balance_and_explicit_repeats(self):
        index = self.index({'dog': [self.sample('dog', n) for n in range(3)],
                            'bird': [self.sample('bird', n + 3) for n in range(3)]})
        manifest, side = create_manifest(index, self.root, queries=10)
        self.assertEqual(sum(ep['concept_id'] == 'dog' for ep in manifest['episodes']), 5)
        self.assertEqual(side['sampling']['query_repeats'], 4)

    def test_zip_traversal_and_overwrite_rejected(self):
        path = Path(__file__).resolve().parents[1] / 'scripts' / 'download_data.py'
        spec = importlib.util.spec_from_file_location('ics_download_data', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        bad = self.root / 'bad.zip'
        with zipfile.ZipFile(bad, 'w') as z:
            z.writestr('../escape', b'bad')
        with self.assertRaises(ValueError):
            module.extract_zip(bad, self.root / 'extract')
        okay = self.root / 'okay.zip'
        with zipfile.ZipFile(okay, 'w') as z:
            z.writestr('inside/a.txt', b'one')
        module.extract_zip(okay, self.root / 'extract', 'inside')
        module.extract_zip(okay, self.root / 'extract', 'inside')
        (self.root / 'extract' / 'a.txt').write_text('different')
        with self.assertRaises(FileExistsError):
            module.extract_zip(okay, self.root / 'extract', 'inside')


if __name__ == '__main__':
    unittest.main()
