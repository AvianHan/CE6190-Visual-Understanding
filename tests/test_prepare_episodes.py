"""CLI policy and audit tests; no dataset downloads or model inference."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('prepare_episodes_cli', PROJECT_ROOT/'scripts/prepare_episodes.py')
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class PrepareEpisodesCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root/'manifests'/'lvis.json'
        self.argv = ['prepare_episodes.py', '--dataset', 'lvis', '--data-root', str(self.root/'data'),
                     '--output', str(self.output), '--queries', '10']
        self.manifest = {'protocol_id': 'requested-nonempty-v2', 'episodes': [{'fold': 0}]}
        self.metadata = {'manifest_sha256': 'test-fingerprint',
                         'sampling': {'queries_requested': 10, 'empty_mask_policy': 'exclude'},
                         'sample_eligibility': {
                             'policy': 'exclude', 'version': 'lvis-nonempty-v2',
                             'scope': 'all_fold_candidates', 'input_count': 3,
                             'eligible_count': 2, 'excluded_count': 1,
                             'pool_fingerprint': 'test-pool-fingerprint',
                             'pool_fingerprint_scope': 'eligible_image_concept_pairs',
                             'counts_by_concept': {'7': {'input': 3, 'eligible': 2, 'excluded': 1}},
                             'excluded_samples': [{'image': 'LVIS/coco/train2017/empty.jpg',
                                                   'concept_id': 7,
                                                   'reason': 'empty_decoded_foreground_union',
                                                   'annotation_count': 1}]}}

    def test_explicit_exclusion_writes_full_audit_and_prints_bounded_summary(self):
        index = object()
        stream = io.StringIO()
        args = self.argv+['--empty-mask-policy', 'exclude', '--protocol-id', 'requested-nonempty-v2']
        with mock.patch('sys.argv', args), mock.patch.object(CLI, 'build_index', return_value=index), \
                mock.patch.object(CLI, 'create_manifest', return_value=(self.manifest, self.metadata)) as create, \
                contextlib.redirect_stdout(stream):
            CLI.main()
        self.assertIs(create.call_args.args[0], index)
        self.assertEqual(create.call_args.kwargs['empty_mask_policy'], 'exclude')
        self.assertEqual(create.call_args.kwargs['protocol_id'], 'requested-nonempty-v2')
        sidecar_path = self.output.with_suffix('.protocol.json')
        sidecar = json.loads(sidecar_path.read_text())
        self.assertEqual(sidecar['protocol_id'], self.manifest['protocol_id'])
        self.assertEqual(sidecar['sample_eligibility']['excluded_samples'],
                         self.metadata['sample_eligibility']['excluded_samples'])
        self.assertEqual(json.loads(self.output.read_text()), self.manifest)
        summary = json.loads(stream.getvalue())
        self.assertEqual(summary['sample_eligibility']['excluded_count'], 1)
        self.assertEqual(summary['sample_eligibility']['eligible_count'], 2)
        self.assertEqual(summary['eligibility_audit'], str(sidecar_path))
        self.assertNotIn('excluded_samples', summary['sample_eligibility'])
        self.assertNotIn('empty.jpg', stream.getvalue())
        self.assertFalse(list(self.output.parent.glob('*.tmp')))

    def test_default_policy_remains_strict(self):
        metadata = copy.deepcopy(self.metadata)
        metadata['sampling']['empty_mask_policy'] = 'error'
        metadata['sample_eligibility'].update(policy='error', version='strict-v1')
        with mock.patch('sys.argv', self.argv), mock.patch.object(CLI, 'build_index'), \
                mock.patch.object(CLI, 'create_manifest', return_value=(self.manifest, metadata)) as create, \
                contextlib.redirect_stdout(io.StringIO()):
            CLI.main()
        self.assertEqual(create.call_args.kwargs['empty_mask_policy'], 'error')

    def test_invalid_or_non_lvis_exclusion_rejected_before_reading_data(self):
        for dataset, policy in (('coco', 'exclude'), ('lvis', 'ignore')):
            with self.subTest(dataset=dataset, policy=policy):
                argv = list(self.argv)
                argv[argv.index('--dataset')+1] = dataset
                argv += ['--empty-mask-policy', policy]
                with mock.patch('sys.argv', argv), mock.patch.object(CLI, 'build_index') as build, \
                        mock.patch.object(CLI, 'create_manifest') as create, \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    CLI.main()
                self.assertEqual(exc.exception.code, 2)
                build.assert_not_called()
                create.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_existing_sidecar_still_requires_explicit_overwrite(self):
        sidecar = self.output.with_suffix('.protocol.json')
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text('existing audit')
        with mock.patch('sys.argv', self.argv+['--empty-mask-policy', 'exclude']), \
                mock.patch.object(CLI, 'build_index') as build, contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            CLI.main()
        build.assert_not_called()
        self.assertEqual(sidecar.read_text(), 'existing audit')

    def invoke_fixture_cli(self):
        with mock.patch('sys.argv', self.argv), mock.patch.object(CLI, 'build_index'), \
                mock.patch.object(CLI, 'create_manifest', return_value=(self.manifest, copy.deepcopy(self.metadata))), \
                contextlib.redirect_stdout(io.StringIO()):
            CLI.main()

    def test_interrupted_manifest_publication_resumes_matching_sidecar_without_rewriting(self):
        original_replace, original_link = Path.replace, os.link
        def fail_manifest_replace(source, destination):
            if Path(destination) == self.output:
                raise OSError('injected failure between sidecar and manifest publication')
            return original_replace(source, destination)
        def fail_manifest_link(source, destination, **kwargs):
            if Path(destination) == self.output:
                raise OSError('injected failure between sidecar and manifest publication')
            return original_link(source, destination, **kwargs)
        with mock.patch.object(Path, 'replace', fail_manifest_replace), \
                mock.patch('os.link', side_effect=fail_manifest_link), \
                self.assertRaisesRegex(OSError, 'injected failure'):
            self.invoke_fixture_cli()
        sidecar = self.output.with_suffix('.protocol.json')
        self.assertFalse(self.output.exists())
        self.assertTrue(sidecar.exists())
        before = sidecar.read_bytes(), sidecar.stat().st_mtime_ns
        self.invoke_fixture_cli()
        self.assertEqual(json.loads(self.output.read_text()), self.manifest)
        self.assertEqual(before, (sidecar.read_bytes(), sidecar.stat().st_mtime_ns))
        self.assertFalse(list(self.output.parent.glob('*.tmp')))

    def test_mismatched_orphan_sidecar_is_preserved_and_not_published(self):
        sidecar = self.output.with_suffix('.protocol.json')
        sidecar.parent.mkdir(parents=True)
        for field, changed in (('protocol_id', 'other-protocol'),
                               ('manifest_sha256', 'other-content'),
                               ('provenance', {'source': 'different'})):
            with self.subTest(field=field):
                metadata = {**copy.deepcopy(self.metadata), 'protocol_id': self.manifest['protocol_id']}
                metadata[field] = changed
                sidecar.write_text(json.dumps(metadata))
                before = sidecar.read_bytes(), sidecar.stat().st_mtime_ns
                with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
                    self.invoke_fixture_cli()
                self.assertIn('does not match', errors.getvalue())
                self.assertFalse(self.output.exists())
                self.assertEqual(before, (sidecar.read_bytes(), sidecar.stat().st_mtime_ns))

    def test_existing_complete_manifest_still_refuses_implicit_overwrite(self):
        self.invoke_fixture_cli()
        before = self.output.read_bytes(), self.output.stat().st_mtime_ns
        with mock.patch('sys.argv', self.argv), mock.patch.object(CLI, 'build_index') as build, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            CLI.main()
        build.assert_not_called()
        self.assertEqual(before, (self.output.read_bytes(), self.output.stat().st_mtime_ns))

    def test_manifest_created_by_another_writer_during_publication_is_preserved(self):
        original_link = os.link
        def competing_writer(source, destination, **kwargs):
            if Path(destination) == self.output:
                self.output.write_text('other writer owns this manifest')
            return original_link(source, destination, **kwargs)
        with mock.patch('os.link', side_effect=competing_writer), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke_fixture_cli()
        self.assertEqual(self.output.read_text(), 'other writer owns this manifest')
        self.assertFalse(list(self.output.parent.glob('*.tmp')))

    def test_incomplete_json_encoding_does_not_publish_sidecar(self):
        self.manifest['unserializable'] = {1, 2}
        with self.assertRaises(TypeError):
            self.invoke_fixture_cli()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix('.protocol.json').exists())
        self.assertFalse(list(self.output.parent.glob('*.tmp')))

    def test_real_cli_excludes_empty_lvis_candidate_and_writes_valid_manifest(self):
        """Exercise the index -> sampler -> files contract with trusted synthetic metadata."""
        from ics_eval import validate_manifest
        root = self.root/'data'
        folder = root/'LVIS'/'coco'/'train2017'
        folder.mkdir(parents=True)
        annotations = {}
        for number in range(3):
            name = f'train2017/{number:012d}.jpg'
            Image.new('RGB', (3, 2), (30*number, 20, 50)).save(root/'LVIS'/'coco'/name)
            mask = np.zeros((2, 3), dtype=np.uint8)
            if number:
                mask[0, 1] = 1
            annotations[name] = {'annotations': [{'segmentation': mask}]}
        # Ten author-order categories are required to retain one fixed fold class.
        metadata = {category: copy.deepcopy(annotations) for category in range(10)}
        with (root/'LVIS'/'lvis_val.pkl').open('wb') as stream:
            pickle.dump(metadata, stream)
        command = [sys.executable, '-I', str(PROJECT_ROOT/'scripts/prepare_episodes.py'),
                   '--dataset', 'lvis', '--data-root', str(root), '--output', str(self.output),
                   '--queries', '0', '--trust-prepared-pickle',
                   '--empty-mask-policy', 'exclude', '--protocol-id', 'integration-lvis-nonempty-v2']
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        summary = json.loads(result.stdout)
        self.assertEqual(summary['sample_eligibility']['excluded_count'], 1)
        self.assertEqual(summary['sample_eligibility']['scope'], 'all_candidates_before_sampling')
        manifest = json.loads(self.output.read_text())
        self.assertEqual(manifest['protocol_id'], 'integration-lvis-nonempty-v2')
        self.assertEqual(len(manifest['episodes']), 2)
        validate_manifest(manifest, root=root)
        sidecar = json.loads(self.output.with_suffix('.protocol.json').read_text())
        self.assertEqual(sidecar['protocol_id'], manifest['protocol_id'])
        self.assertEqual(sidecar['sampling']['empty_mask_policy'], 'exclude')
        self.assertEqual(sidecar['sample_eligibility']['version'], 'lvis-nonempty-v2')
        self.assertEqual(sidecar['sample_eligibility']['excluded_samples'][0]['image'],
                         'LVIS/coco/train2017/000000000000.jpg')
        # Simulate the published sidecar surviving without its final manifest;
        # a real second CLI process must recompute the same content and recover.
        sidecar_path = self.output.with_suffix('.protocol.json')
        before = sidecar_path.read_bytes(), sidecar_path.stat().st_mtime_ns
        expected_manifest = self.output.read_bytes()
        self.output.unlink()
        subprocess.run(command, text=True, capture_output=True, check=True)
        self.assertEqual(self.output.read_bytes(), expected_manifest)
        self.assertEqual(before, (sidecar_path.read_bytes(), sidecar_path.stat().st_mtime_ns))


if __name__ == '__main__':
    unittest.main()
