import errno
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ics import batch2_sources as sources


class SourcePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = Path(self.temp.name)
        self.root = self.runtime / 'data'
        self.images = self.root / 'ISIC/ISIC2018_Task1-2_Training_Input'
        self.masks = self.root / 'ISIC/ISIC2018_Task1_Training_GroundTruth'
        self.mapping = {'ISIC_0001': '1', 'ISIC_0002': '2', 'ISIC_0003': '3'}

    def seed_isic(self):
        self.images.mkdir(parents=True)
        self.masks.mkdir(parents=True)
        for name in self.mapping:
            (self.images / (name + '.jpg')).write_bytes(('image-' + name).encode())
            (self.masks / (name + '_segmentation.png')).write_bytes(('mask-' + name).encode())

    def mapping_patch(self):
        return patch.object(sources, '_official_mapping', return_value=(self.mapping, {'fixture': True}))

    def test_missing_sources_never_download_without_flag(self):
        with self.mapping_patch(), patch.object(sources, '_downloader') as dl:
            with self.assertRaisesRegex(FileNotFoundError, 'Missing transfer assets'):
                sources.ensure_transfer_sources(self.runtime, 'isic', min_free_gib=0)
            dl.return_value.acquire.assert_not_called()

    def test_existing_flat_inputs_reused_originals_preserved_and_idempotent(self):
        self.seed_isic()
        before = {p.name: sources._sha(p) for p in self.images.glob('*.jpg')}
        downloader = sources._downloader()
        with self.mapping_patch(), patch.object(sources, '_downloader', return_value=downloader), patch.object(downloader, 'acquire') as acquire:
            first = sources.ensure_transfer_sources(self.runtime, 'isic', allow_download=True, min_free_gib=0)
            second = sources.ensure_transfer_sources(self.runtime, 'isic', min_free_gib=0)
            acquire.assert_not_called()
        self.assertEqual(first['organization_actions'], {'hardlink': 3})
        self.assertEqual(second['organization_actions'], {'existing_identical': 3})
        self.assertEqual(before, {p.name: sources._sha(p) for p in self.images.glob('*.jpg')})
        self.assertTrue((self.root / 'batch2_source_receipts/isic.json').is_file())

    def test_missing_masks_only_acquires_mask_asset(self):
        self.seed_isic()
        missing = self.masks / 'ISIC_0002_segmentation.png'
        missing.unlink()
        downloader = sources._downloader()
        def acquire(asset, root, archives, guard):
            self.assertEqual(asset['id'], 'isic2018_masks')
            missing.write_bytes(b'restored-mask')
            return {'id': asset['id'], 'status': 'fixture'}
        with self.mapping_patch(), patch.object(sources, '_downloader', return_value=downloader), patch.object(downloader, 'acquire', side_effect=acquire) as acq:
            report = sources.ensure_transfer_sources(self.runtime, 'isic', allow_download=True, min_free_gib=0)
        self.assertEqual(acq.call_count, 1)
        self.assertEqual(report['assets'][0]['id'], 'isic2018_masks')

    def test_wrong_class_and_differing_file_preserved(self):
        self.seed_isic()
        wrong = self.images / '2/ISIC_0001.jpg'
        wrong.parent.mkdir()
        wrong.write_bytes(b'user-image')
        with self.mapping_patch(), self.assertRaisesRegex(ValueError, 'wrong author class'):
            sources.ensure_transfer_sources(self.runtime, 'isic', min_free_gib=0)
        self.assertEqual(wrong.read_bytes(), b'user-image')
        wrong.unlink()
        collision = self.images / '1/ISIC_0001.jpg'
        collision.parent.mkdir(exist_ok=True)
        collision.write_bytes(b'other-user-image')
        with self.mapping_patch(), self.assertRaisesRegex(ValueError, 'differs'):
            sources.ensure_transfer_sources(self.runtime, 'isic', min_free_gib=0)
        self.assertEqual(collision.read_bytes(), b'other-user-image')

    def test_copy_fallback_preserves_source_and_target_bytes(self):
        self.seed_isic()
        src = self.images / 'ISIC_0001.jpg'
        dst = self.images / '1/ISIC_0001.jpg'
        guard = sources._downloader().CapacityGuard()
        with patch.object(sources.os, 'link', side_effect=OSError(errno.EOPNOTSUPP, 'fixture')):
            self.assertEqual(sources._publish_same_bytes(src, dst, guard), 'copy')
        self.assertEqual(src.read_bytes(), dst.read_bytes())

    def test_symlink_escape_rejected(self):
        self.root.mkdir()
        (self.root / 'ISIC').symlink_to(self.runtime / 'outside', target_is_directory=True)
        with self.mapping_patch(), self.assertRaisesRegex(ValueError, 'within data root'):
            sources.ensure_transfer_sources(self.runtime, 'isic', min_free_gib=0)

    def test_suim_existing_layout_no_download(self):
        (self.root / 'SUIM/images').mkdir(parents=True)
        (self.root / 'SUIM/images/a.jpg').write_bytes(b'fixture')
        for name in sources.SUIM_CLASSES:
            p = self.root / 'SUIM/masks' / name
            p.mkdir(parents=True)
            (p / 'a.png').write_bytes(b'fixture')
        downloader = sources._downloader()
        with patch.object(sources, '_downloader', return_value=downloader), patch.object(downloader, 'acquire') as acquire:
            result = sources.ensure_transfer_sources(self.runtime, 'suim', allow_download=True, min_free_gib=0)
            acquire.assert_not_called()
        self.assertEqual(result['class_file_counts'], dict.fromkeys(sources.SUIM_CLASSES, 1))
        self.assertIn('not official test-only', result['protocol'])

    def test_wrong_pin_and_changed_mapping_fail_before_mutation(self):
        with patch.object(sources.subprocess, 'check_output', return_value='wrong'):
            with self.assertRaisesRegex(ValueError, 'pinned'):
                sources._official_mapping(self.runtime)
        table = self.runtime / 'sources/CG-ICS/datasets/isic/class_id.csv'
        table.parent.mkdir(parents=True)
        table.write_text('ID,Class\nISIC_0001,nevus\n')
        with patch.object(sources.subprocess, 'check_output', return_value=sources.CG_PIN):
            with self.assertRaisesRegex(ValueError, 'changed'):
                sources._official_mapping(self.runtime)


if __name__ == '__main__':
    unittest.main()
