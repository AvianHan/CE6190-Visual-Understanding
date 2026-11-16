"""SUIM full-pool eligibility audit: only proven shape mismatches are excluded."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from ics import batch2_data as data
from ics.data import DatasetIndex, Sample, build_index, file_sha256, read_binary_mask
from ics.suim_geometry import (CLASSES, SUIM_REPAIR_IMPLEMENTATION_ID, _source_evidence,
    audit_suim_pool, prepare_suim_geometry, verify_suim_geometry, apply_suim_geometry,
    verify_manifest_pairs)
from ics_eval import inference_view, manifest_fingerprint


class SuimGeometryTests(unittest.TestCase):
    BASE_COMMIT = 'caf7be44042440a43d77876adf28a77158459f61'

    def original_module(self):
        original=subprocess.check_output(['git','show',self.BASE_COMMIT+':ics/batch2_data.py'],text=True)
        namespace={'__name__':'ics._original_batch2_data','__package__':'ics'}
        exec(compile(original,'original_batch2_data.py','exec'),namespace)
        return namespace
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runtime = Path(self.tmp.name) / 'runtime'
        self.root = self.runtime / 'data'
        self.base = self.root / 'SUIM'
        (self.base / 'images').mkdir(parents=True)
        for cid in CLASSES:
            (self.base / 'masks' / cid).mkdir(parents=True)
        self.source = mock.patch('ics.suim_geometry._source_evidence', return_value={'source': 'synthetic-test-only'})
        self.source.start()
        for k, cid in enumerate(CLASSES):
            for j in range(3):
                self.pair(cid, f'{cid}_{j}', color=10 + 20*k + j)
        self.pair('FV', 'mismatch', image_wh=(4, 3), mask_wh=(4, 5), color=201)
        self.pair('HD', 'empty', empty=True, color=202)
        self.output = self.runtime / 'batch2/suim/data'

    def tearDown(self):
        self.source.stop()
        self.tmp.cleanup()

    def pair(self, cid, name, *, image_wh=(4, 3), mask_wh=None, color=11, empty=False,
             image_orientation=None, mask_orientation=None):
        image = self.base / 'images' / (name + '.png')
        kwargs = {}
        if image_orientation is not None:
            exif = Image.Exif(); exif[274] = image_orientation; kwargs['exif'] = exif
        Image.new('RGB', image_wh, (color, 45, 89)).save(image, **kwargs)
        mask = self.base / 'masks' / cid / (name + '.png')
        width, height = mask_wh or image_wh
        arr = np.zeros((height, width), np.uint8)
        if not empty: arr[0:2, 1:3] = 255
        kwargs = {}
        if mask_orientation is not None:
            exif = Image.Exif(); exif[274] = mask_orientation; kwargs['exif'] = exif
        Image.fromarray(arr).save(mask, **kwargs)
        return image, mask

    def originals(self):
        return {str(p): file_sha256(p) for p in self.base.rglob('*') if p.is_file()}

    def priors(self):
        image = self.root / 'old.png'; mask = self.root / 'oldmask.png'
        ref = self.root / 'oldref.png'
        Image.new('RGB', (4, 3), (244, 22, 99)).save(image)
        Image.new('RGB', (4, 3), (245, 22, 99)).save(ref)
        Image.fromarray(np.ones((3, 4), np.uint8)).save(mask)
        index = DatasetIndex('synthetic', 'fixture', 'test', 0,
            {'old': [Sample('old', image, mask), Sample('old', ref, mask)]})
        manifest, _ = data.create_fresh_manifest(index, self.root, queries=1, references_per_query=1)
        path = self.runtime / 'manifests/old.json'; path.parent.mkdir(); path.write_text(json.dumps(manifest))
        runs=[]
        for n in range(2):
            run=self.runtime / 'runs' / f'old{n}'; out=run/'results/old'; out.mkdir(parents=True)
            (out/'run.json').write_text(json.dumps({'binding': {'manifest_sha256': manifest_fingerprint(manifest),
                'protocol_id': manifest['protocol_id'], 'selected_episode_ids': [manifest['episodes'][0]['episode_id']]}}))
            (run/'plan.json').write_text(json.dumps({'run_dir':str(run),'tasks':[{'id':'old','manifest':str(path),
                'output':str(out),'dataset_root':str(self.root),'phase':'experiment'}]}));runs.append(run)
        return runs

    def test_full_pool_exclusions_are_exact_and_raw_bytes_unchanged(self):
        before = self.originals()
        receipt = prepare_suim_geometry(self.runtime, self.output)
        audit = receipt['audit']
        self.assertEqual(audit['source_mask_count'], 23)
        self.assertEqual(audit['original_nonempty_pair_count'], 22)
        self.assertEqual(audit['eligible_nonempty_pair_count'], 21)
        self.assertEqual(audit['excluded_nonempty_pair_count'], 1)
        self.assertEqual(audit['dimension_mismatch_pair_count'], 1)
        self.assertEqual(audit['exclusions'][0]['image_size_wh'], [4, 3])
        self.assertEqual(audit['exclusions'][0]['size_wh'], [4, 5])
        self.assertEqual(audit['exclusions'][0]['ineligible_reason'], 'mask_image_dimensions_mismatch')
        self.assertEqual(before, self.originals())
        self.assertFalse((self.root / 'ics_generated_masks').exists())
        self.assertEqual(verify_suim_geometry(self.runtime, self.output/'suim_geometry_receipt.json'), receipt)
        self.assertEqual(prepare_suim_geometry(self.runtime, self.output), receipt)

    def test_only_mismatch_pair_removed_valid_mask_pixels_and_paths_preserved(self):
        receipt = prepare_suim_geometry(self.runtime, self.output)
        old = build_index('suim', self.root)
        new = apply_suim_geometry(old, self.root, receipt)
        self.assertEqual(set(new.samples), set(CLASSES))
        self.assertEqual(sum(map(len, old.samples.values())), 22)
        self.assertEqual(sum(map(len, new.samples.values())), 21)
        originals={str(s.mask_source):s for pool in old.samples.values() for s in pool}
        for pool in new.samples.values():
            for sample in pool:
                self.assertIs(sample, originals[str(sample.mask_source)])
                self.assertTrue(np.array_equal(read_binary_mask(sample), read_binary_mask(originals[str(sample.mask_source)])))
        self.assertIn('shape-valid-v1', new.version)
        self.assertEqual(new.provenance['suim_geometry']['excluded_nonempty_pairs'], 1)

    def test_nonidentity_image_exif_is_fatal_even_for_dimension_mismatch(self):
        self.pair('FV', 'orientation', mask_wh=(4, 5), image_orientation=6, color=203)
        audit=audit_suim_pool(self.root)
        self.assertEqual(audit['status'], 'BLOCKED')
        self.assertTrue(any(x['error']=='nonidentity_image_exif' for x in audit['issues']))
        with self.assertRaisesRegex(ValueError,'fatal issue'):
            prepare_suim_geometry(self.runtime,self.output)
        self.assertFalse((self.output/'suim_geometry_receipt.json').exists())
        self.assertEqual(len(list(self.output.glob('suim_geometry_inputs-*.json'))),1)

    def test_nonidentity_mask_exif_is_fatal(self):
        self.pair('FV','maskorientation',mask_orientation=2,color=203)
        audit=audit_suim_pool(self.root)
        self.assertTrue(any(x['error']=='nonidentity_mask_exif' for x in audit['issues']))
        self.assertEqual(audit['status'],'BLOCKED')

    def test_corruption_missing_image_and_class_loss_are_not_silent_exclusions(self):
        (self.base/'masks/FV/FV_0.png').write_bytes(b'corrupt')
        (self.base/'images/HD_0.png').unlink()
        for p in (self.base/'masks/PF').glob('*'): p.unlink()
        audit=audit_suim_pool(self.root)
        self.assertEqual(audit['status'],'BLOCKED')
        self.assertTrue(any(x['error']=='UnidentifiedImageError' for x in audit['issues']))
        self.assertTrue(any(x['error']=='FileNotFoundError' for x in audit['issues']))
        self.assertTrue(any(x.get('class')=='PF' for x in audit['issues']))
        self.assertEqual(audit['dimension_mismatch_pair_count'],1)
        with self.assertRaisesRegex(ValueError,'fatal issue'):
            prepare_suim_geometry(self.runtime,self.output)

    def test_class_with_only_one_distinct_valid_image_blocks_before_sampling(self):
        source=(self.base/'images/WR_0.png').read_bytes()
        for p in (self.base/'images').glob('WR_*.png'):p.write_bytes(source)
        audit=audit_suim_pool(self.root)
        self.assertEqual(audit['counts_by_class']['WR']['eligible_distinct_image_hashes'],1)
        self.assertTrue(any(x.get('class')=='WR' for x in audit['issues']))
        with self.assertRaisesRegex(ValueError,'fatal issue'):
            prepare_suim_geometry(self.runtime,self.output)

    def test_changed_excluded_asset_or_new_unselected_file_blocks_resume(self):
        prepare_suim_geometry(self.runtime,self.output)
        self.pair('FV','mismatch',image_wh=(4,3),mask_wh=(4,6),color=201)
        with self.assertRaisesRegex(ValueError,'source pool changed'):
            verify_suim_geometry(self.runtime,self.output/'suim_geometry_receipt.json')

    def test_added_unselected_asset_blocks_resume(self):
        prepare_suim_geometry(self.runtime,self.output)
        self.pair('FV','newfile',color=204)
        with self.assertRaisesRegex(ValueError,'source pool changed'):
            verify_suim_geometry(self.runtime,self.output/'suim_geometry_receipt.json')

    def test_unknown_pair_or_lost_membership_is_rejected(self):
        receipt=prepare_suim_geometry(self.runtime,self.output)
        index=build_index('suim',self.root);index.samples['FV'].pop()
        with self.assertRaisesRegex(ValueError,'lost source membership'):
            apply_suim_geometry(index,self.root,receipt)
        index.dataset='isic'
        with self.assertRaisesRegex(ValueError,'another dataset'):
            apply_suim_geometry(index,self.root,receipt)

    def test_end_to_end_scope_has_explicit_protocol_no_bad_query_or_reference(self):
        runs=self.priors();before=self.originals()
        with mock.patch('ics.batch2_sources.ensure_transfer_sources',return_value={'status':'fixture'}) as source:
            receipt=data.prepare_scope(self.runtime,self.output,scope='suim',prior_runs=runs,
                transfer_queries=10,min_free_gib=0)
            self.assertEqual(receipt['request']['geometry_policy'],SUIM_REPAIR_IMPLEMENTATION_ID)
            self.assertEqual(receipt['query_images'],10)
            self.assertEqual(data.verify_scope(self.output),receipt)
            # Existing model/collector workers deliberately retain old code.
            self.assertEqual(self.original_module()['verify_scope'](self.output),receipt)
            self.assertEqual(data.prepare_scope(self.runtime,self.output,scope='suim',prior_runs=runs,
                transfer_queries=10,min_free_gib=0),receipt)
            self.assertEqual(source.call_count,1)
        entry=receipt['manifests']['batch2_suim_fold0_transfer']
        manifest=json.loads(Path(entry['path']).read_text())
        self.assertTrue(manifest['protocol_id'].endswith('-shape-valid-v1'))
        for ep in manifest['episodes']:
            self.assertNotIn('mismatch',ep['query']['image'])
            self.assertTrue(all('mismatch' not in r['image'] for r in ep['references']))
            self.assertEqual(set(inference_view(ep)),{'query_image','references'})
        self.assertEqual(before,self.originals())
        self.assertFalse((self.root/'ics_generated_masks/suim/author_nearest_v1').exists())
        geometry=json.loads((self.output/'suim_geometry_receipt.json').read_text())
        bad=json.loads(json.dumps(manifest))
        bad['episodes'][0]['query']['image']='SUIM/images/mismatch.png'
        bad['episodes'][0]['query']['image_sha256']=file_sha256(self.base/'images/mismatch.png')
        with self.assertRaisesRegex(ValueError,'ineligible image/class pair'):
            verify_manifest_pairs(bad,self.root,geometry)

    def test_strict_requested_query_count_is_not_lowered(self):
        runs=self.priors()
        with mock.patch('ics.batch2_sources.ensure_transfer_sources',return_value={'status':'fixture'}):
            with self.assertRaisesRegex(ValueError,'requested 1000 unique fresh queries, only 21 eligible'):
                data.prepare_scope(self.runtime,self.output,scope='suim',prior_runs=runs,min_free_gib=0)
        self.assertFalse((self.output/'data_receipt.json').exists())
        self.assertTrue((self.output/'suim_geometry_receipt.json').exists())

    def test_core_and_isic_request_bytes_equal_original_implementation(self):
        # Compare the unchanged scopes with the exact pre-repair implementation,
        # rather than a second manually copied expected request dictionary.
        namespace=self.original_module()
        for scope in ('core','isic'):
            args=(self.runtime,scope,[self.runtime/'runs/a',self.runtime/'runs/b'],100,1000,3,20261116,True)
            self.assertEqual(data._canonical(data._request(*args)),
                             data._canonical(namespace['_request'](*args)))
        self.assertEqual(data.SUIM_REPAIR_IMPLEMENTATION_ID,SUIM_REPAIR_IMPLEMENTATION_ID)

    def test_real_source_gate_rejects_unknown_pin_or_changed_loader(self):
        loader=self.runtime/'sources/INSID3/datasets/suim.py'
        loader.parent.mkdir(parents=True);loader.write_text('fixture pinned loader')
        with mock.patch('ics.suim_geometry.subprocess.check_output',return_value='unexpected'):
            with self.assertRaisesRegex(ValueError,'audited INSID3 pin'):_source_evidence(self.runtime)
        with mock.patch('ics.suim_geometry.subprocess.check_output',return_value='0c165a10cf52ab91f335883d06260de86854adbe'):
            with self.assertRaisesRegex(ValueError,'audited INSID3 pin'):_source_evidence(self.runtime)
            with mock.patch('ics.suim_geometry.SUIM_LOADER_SHA256',file_sha256(loader)):
                self.assertEqual(_source_evidence(self.runtime)['loader_path'],str(loader))

    def test_escaped_source_symlink_is_fatal_and_does_not_become_exclusion(self):
        target=Path(self.tmp.name)/'outside.png';Image.new('L',(4,3),255).save(target)
        path=self.base/'masks/FV/FV_0.png';path.unlink();path.symlink_to(target)
        audit=audit_suim_pool(self.root)
        self.assertEqual(audit['status'],'BLOCKED')
        self.assertTrue(any('escapes dataset root' in x.get('detail','') for x in audit['issues']))
        self.assertEqual(audit['dimension_mismatch_pair_count'],1)


if __name__ == '__main__':unittest.main()
