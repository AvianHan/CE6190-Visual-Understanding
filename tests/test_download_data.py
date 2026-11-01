"""Offline download/restart tests using small synthetic archives, not datasets."""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'download_data.py'
SPEC = importlib.util.spec_from_file_location('ics_download_data_test', SCRIPT)
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


class DownloadDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archives = self.root / 'archives'
        self.archives.mkdir()
        self.asset = {'id': 'fixture', 'dataset': 'coco', 'destination': 'images',
                      'url': 'https://example.invalid/images.zip', 'sha256': None}
        self.archive = self.archives / 'fixture.zip'
        self.receipt = self.archives / 'fixture.receipt.json'

    def tearDown(self):
        self.temp.cleanup()

    def make_zip(self, members=None):
        with zipfile.ZipFile(self.archive, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in (members or {'train/a.jpg': b'original image'}).items():
                archive.writestr(name, data)

    def acquire(self, **kwargs):
        return downloader.acquire(self.asset, self.root, self.archives, **kwargs)

    @staticmethod
    def response(data, status=200, headers=None):
        result = io.BytesIO(data)
        result.status = status
        result.headers = headers or {}
        return result

    def test_restart_reuses_verified_archive_and_members_without_rereading_bytes(self):
        self.make_zip()
        self.acquire()
        with mock.patch.object(downloader, 'sha256', side_effect=AssertionError('rehash')):
            with mock.patch.object(downloader, 'crc32', side_effect=AssertionError('reread')):
                result = self.acquire()
        self.assertEqual(result['status'], 'reused_extraction_receipt')
        self.assertEqual(result['verified_files'], 1)

    def test_missing_extracted_file_is_repaired_without_download(self):
        self.make_zip({'train/a.jpg': b'a', 'train/b.jpg': b'b'})
        self.acquire()
        deleted = self.root / 'images/train/b.jpg'
        deleted.unlink()
        with mock.patch.object(downloader, 'download_http', side_effect=AssertionError('download')):
            result = self.acquire()
        self.assertEqual(result['status'], 'extracted')
        self.assertEqual(deleted.read_bytes(), b'b')
        self.assertEqual(result['verified_files'], 2)

    def test_same_size_modified_extraction_is_not_overwritten(self):
        self.make_zip({'train/a.jpg': b'good'})
        self.acquire()
        target = self.root / 'images/train/a.jpg'
        original_stat = target.stat()
        target.write_bytes(b'evil')
        os.utime(target, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1000000))
        with self.assertRaisesRegex(FileExistsError, 'differing file'):
            self.acquire()
        self.assertEqual(target.read_bytes(), b'evil')

    def test_touched_matching_archive_is_hashed_once_and_receipt_refreshed(self):
        self.make_zip()
        self.acquire()
        old = self.archive.stat()
        os.utime(self.archive, ns=(old.st_atime_ns, old.st_mtime_ns + 1000000))
        with mock.patch.object(downloader, 'sha256', wraps=downloader.sha256) as digest:
            self.acquire()
            self.acquire()
        self.assertEqual(digest.call_count, 1)

    def test_changed_archive_is_rejected_before_extraction(self):
        self.make_zip()
        self.acquire()
        with self.archive.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'Archive changed'):
            self.acquire()

    def test_old_receipt_is_upgraded_and_existing_member_not_extracted_again(self):
        self.make_zip()
        self.acquire()
        prior = json.loads(self.receipt.read_text())
        del prior['files']
        del prior['archive_stat']
        self.receipt.write_text(json.dumps(prior))
        with mock.patch.object(zipfile.ZipFile, 'open', side_effect=AssertionError('re-extract')):
            self.acquire()
        self.assertEqual(len(json.loads(self.receipt.read_text())['files']), 1)

    def test_valid_receipt_survives_removed_archive(self):
        self.make_zip()
        self.acquire()
        self.archive.unlink()
        with mock.patch.object(downloader, 'download_http', side_effect=AssertionError('download')):
            self.assertEqual(self.acquire()['status'], 'reused_extraction_receipt')

    def test_strip_prefix_layout_and_path_traversal_rejection(self):
        self.asset['strip_prefix'] = 'wrapper'
        self.make_zip({'wrapper/train/a.jpg': b'a'})
        self.acquire()
        self.assertTrue((self.root / 'images/train/a.jpg').exists())
        self.make_zip({'wrapper/../outside.txt': b'bad'})
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            downloader.extract_zip(self.archive, self.root / 'fresh', 'wrapper')
        self.assertFalse((self.root / 'outside.txt').exists())

    def test_zip_symlink_rejected(self):
        with zipfile.ZipFile(self.archive, 'w') as archive:
            info = zipfile.ZipInfo('unsafe')
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, '../outside')
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            self.acquire()
        self.assertFalse(self.receipt.exists())

    def test_extraction_budget_uses_known_uncompressed_size(self):
        self.make_zip({'image.jpg': b'x' * 100})
        usage = shutil._ntuple_diskusage(1000, 900, 100)
        with mock.patch.object(downloader.shutil, 'disk_usage', return_value=usage):
            with self.assertRaisesRegex(OSError, '100 additional bytes plus 20'):
                self.acquire(guard=downloader.CapacityGuard(20))
        self.assertFalse((self.root / 'images/image.jpg').exists())
        self.assertFalse(self.receipt.exists())

    def test_known_http_length_is_checked_before_copy(self):
        response = self.response(b'x' * 100, headers={'Content-Length': '100'})
        target = self.root / 'download.zip'
        usage = shutil._ntuple_diskusage(1000, 900, 100)
        with mock.patch.object(downloader, 'urlopen', return_value=response):
            with mock.patch.object(downloader.shutil, 'disk_usage', return_value=usage):
                with self.assertRaisesRegex(OSError, 'Insufficient free space'):
                    downloader.download_http(self.asset['url'], target, downloader.CapacityGuard(20))
        self.assertFalse(target.exists())
        self.assertFalse(target.with_suffix('.zip.part').exists())

    def test_space_drop_during_http_keeps_partial_without_promoting(self):
        target = self.root / 'download.zip'
        response = self.response(b'x' * (2 * 1024 * 1024))
        guard = mock.Mock()
        guard.check.side_effect = [None, None, OSError('reserve reached')]
        with mock.patch.object(downloader, 'urlopen', return_value=response):
            with self.assertRaisesRegex(OSError, 'reserve reached'):
                downloader.download_http(self.asset['url'], target, guard)
        self.assertFalse(target.exists())
        self.assertEqual(target.with_suffix('.zip.part').stat().st_size, 1024 * 1024)

    def test_http_range_resumes_at_requested_offset(self):
        target = self.root / 'download.zip'
        target.with_suffix('.zip.part').write_bytes(b'ab')
        response = self.response(b'cdef', 206, {'Content-Range': 'bytes 2-5/6', 'Content-Length': '4'})
        with mock.patch.object(downloader, 'urlopen', return_value=response) as request:
            downloader.download_http(self.asset['url'], target)
        self.assertEqual(request.call_args.args[0].get_header('Range'), 'bytes=2-')
        self.assertEqual(target.read_bytes(), b'abcdef')

    def test_truncated_range_without_content_length_is_not_promoted(self):
        target = self.root / 'download.zip'
        target.with_suffix('.zip.part').write_bytes(b'ab')
        response = self.response(b'cd', 206, {'Content-Range': 'bytes 2-5/6'})
        with mock.patch.object(downloader, 'urlopen', return_value=response):
            with self.assertRaisesRegex(OSError, 'Incomplete HTTP range'):
                downloader.download_http(self.asset['url'], target)
        self.assertFalse(target.exists())
        self.assertEqual(target.with_suffix('.zip.part').read_bytes(), b'abcd')

    def test_drive_process_terminated_when_free_space_falls(self):
        guard = mock.Mock()
        guard.check.side_effect = [None, OSError('reserve reached')]
        process = mock.MagicMock()
        process.__enter__.return_value = process
        process.wait.side_effect = [subprocess.TimeoutExpired('gdown', 1), 0]
        with mock.patch.object(downloader.subprocess, 'Popen', return_value=process):
            with self.assertRaisesRegex(OSError, 'reserve reached'):
                downloader.download_drive('https://drive.google.com/fixture', self.archive, guard)
        process.terminate.assert_called_once()
        self.assertFalse(self.archive.exists())


if __name__ == '__main__':
    unittest.main()
