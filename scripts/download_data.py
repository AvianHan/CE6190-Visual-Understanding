#!/usr/bin/env python3
"""Plan/download only requested dataset archives; never downloads model weights.

Default is a dry plan. --execute downloads/extracts assets, records observed
checksums, resumes HTTP downloads and reuses complete extraction receipts.
Optional free-space guards are best-effort shared-filesystem checks, not quotas.
Archives are retained; no existing differing dataset file is overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import zipfile
import zlib


class CapacityGuard:
    """Best-effort shared-filesystem reserve, not a quota or reservation."""

    def __init__(self, minimum_bytes=0):
        self.minimum_bytes = minimum_bytes

    def check(self, path: Path, needed_bytes=0):
        if not self.minimum_bytes:
            return
        path = Path(path)
        while not path.exists():
            path = path.parent
        free = shutil.disk_usage(path).free
        if free < self.minimum_bytes + needed_bytes:
            raise OSError(f'Insufficient free space at {path}: {free} bytes available; '
                          f'{needed_bytes} additional bytes plus {self.minimum_bytes} '
                          'reserve required. Shared free space is not a storage quota.')


def copy_stream(source, target, guard: CapacityGuard, path: Path):
    for chunk in iter(lambda: source.read(1024 * 1024), b''):
        guard.check(path, len(chunk))
        target.write(chunk)


def file_stat(path):
    info = Path(path).stat()
    return {'size': info.st_size, 'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}


def write_receipt(path: Path, record: dict):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def crc32(path):
    value = 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value = zlib.crc32(chunk, value)
    return value


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def contained(root: Path, relative: str) -> Path:
    part = PurePosixPath(relative)
    if not relative or part.is_absolute() or '..' in part.parts or '\\' in relative:
        raise ValueError(f'Unsafe archive/destination path: {relative!r}')
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Extraction escapes root through a symlink')
    return target


def verify_inventory(destination: Path, files: list[dict]):
    """Validate receipt members, repairing only missing files in the next stage."""
    verified = {}
    complete = True
    for record in files:
        target = contained(destination, record['path'])
        if not target.exists():
            complete = False
            continue
        if not target.is_file():
            raise FileExistsError(f'Expected an extracted file: {target}')
        current = file_stat(target)
        # Unchanged local metadata avoids reading every COCO image on each poll.
        # A changed file must still match the ZIP CRC; never bless changed data.
        if current != record['stat']:
            if current['size'] != record['stat']['size'] or crc32(target) != record['crc32']:
                raise FileExistsError(f'Refusing to overwrite differing file: {target}')
        verified[record['path']] = dict(record, stat=current)
    return complete, verified


def extract_zip(path: Path, destination: Path, strip_prefix: str | None = None,
                guard: CapacityGuard | None = None, verified: dict | None = None):
    """Validate every ZIP path and reuse matching files without extracting twice."""
    guard = guard or CapacityGuard()
    verified = verified or {}
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        members = []
        for member in archive.infolist():
            name = member.filename
            # Validate before stripping, so e.g. prefix/../escape never passes.
            contained(destination, name.rstrip('/'))
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError('Archive symlinks are not allowed')
            if strip_prefix:
                parts = PurePosixPath(name).parts
                if parts and parts[0] == strip_prefix:
                    name = '/'.join(parts[1:])
                    if not name:
                        continue
            target = contained(destination, name.rstrip('/'))
            members.append((member, target))
        missing_bytes = 0
        records = []
        pending = []
        for member, target in members:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            relative = target.relative_to(destination.resolve()).as_posix()
            if target.exists():
                current = file_stat(target)
                previous = verified.get(relative)
                cached_match = (previous and previous['stat'] == current
                                and previous['crc32'] == member.CRC)
                if (not target.is_file() or current['size'] != member.file_size or
                        (not cached_match and crc32(target) != member.CRC)):
                    raise FileExistsError(f'Refusing to overwrite differing file: {target}')
                records.append({'path': relative, 'crc32': member.CRC, 'stat': current})
                continue
            missing_bytes += member.file_size
            pending.append((member, target, relative))
        # This is known uncompressed size, not an invented whole-dataset estimate.
        # Concurrent jobs can consume shared space after this check, hence checks
        # also happen while copying. No filesystem space is reserved atomically.
        guard.check(destination, missing_bytes)
        for member, target, relative in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + '.ics-extract.part')
            try:
                with archive.open(member) as src, tmp.open('wb') as dst:
                    copy_stream(src, dst, guard, destination)
                if target.exists():
                    if sha256(tmp) != sha256(target):
                        raise FileExistsError(f'Refusing to overwrite differing file: {target}')
                    tmp.unlink()
                else:
                    tmp.replace(target)
                records.append({'path': relative, 'crc32': member.CRC, 'stat': file_stat(target)})
            finally:
                if tmp.exists():
                    tmp.unlink()
        return records


def download_http(url: str, target: Path, guard: CapacityGuard | None = None):
    guard = guard or CapacityGuard()
    partial = target.with_suffix(target.suffix + '.part')
    offset = partial.stat().st_size if partial.exists() else 0
    request = Request(url, headers={'User-Agent': 'CE6190-ICS-data-preparation/1'})
    if offset:
        request.add_header('Range', f'bytes={offset}-')
    try:
        response = urlopen(request, timeout=60)
    except HTTPError as exc:
        if exc.code != 416 or not offset:
            raise
        # Do not assume a 416 partial file is complete. Restart cleanly.
        response = urlopen(Request(url, headers={'User-Agent': 'CE6190-ICS-data-preparation/1'}), timeout=60)
        offset = 0
    with response:
        resumed = offset and response.status == 206
        content_range = response.headers.get('Content-Range', '')
        range_match = re.fullmatch(r'bytes ([0-9]+)-([0-9]+)/([0-9]+)', content_range)
        if resumed and (not range_match or int(range_match.group(1)) != offset):
            raise ValueError('Server returned an unexpected resume range')
        expected_header = response.headers.get('Content-Length')
        expected_bytes = int(expected_header) if expected_header is not None else None
        starting_size = offset if resumed else 0
        guard.check(target.parent, expected_bytes or 0)
        with partial.open('ab' if resumed else 'wb') as stream:
            copy_stream(response, stream, guard, target.parent)
        received_bytes = partial.stat().st_size - starting_size
        if expected_bytes is not None and received_bytes != expected_bytes:
            raise OSError(f'Incomplete HTTP download: expected {expected_bytes} response '
                          f'bytes, received {received_bytes}; partial retained at {partial}')
        if resumed and received_bytes != int(range_match.group(2)) + 1 - offset:
            raise OSError(f'Incomplete HTTP range; partial retained at {partial}')
        if resumed and int(range_match.group(2)) + 1 != int(range_match.group(3)):
            raise OSError(f'HTTP range ended before the complete asset; partial retained at {partial}')
    partial.replace(target)


def download_drive(url: str, target: Path, guard: CapacityGuard):
    partial = target.with_suffix('.zip.part')
    command = [sys.executable, '-m', 'gdown', '--fuzzy', '--continue', url, '-O', str(partial)]
    guard.check(target.parent)
    # Drive does not expose a reliable size here. Monitor its child process so
    # an unknown-size transfer still respects the reserve on a best-effort basis.
    with subprocess.Popen(command) as process:
        try:
            while True:
                try:
                    code = process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    guard.check(target.parent)
            if code:
                raise subprocess.CalledProcessError(code, command)
            guard.check(target.parent)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    partial.replace(target)


def acquire(asset: dict, root: Path, archives: Path, guard: CapacityGuard | None = None):
    guard = guard or CapacityGuard()
    archive = archives / (asset['id'] + '.zip')
    receipt = archives / (asset['id'] + '.receipt.json')
    destination = contained(root, asset['destination'])
    prior = None
    verified = {}
    observed = None
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if prior.get('source_url') != asset['url'] or prior.get('destination') != str(destination):
            raise ValueError(f'Existing receipt conflicts with catalog entry {asset["id"]}')
        if 'strip_prefix' in prior and prior['strip_prefix'] != asset.get('strip_prefix'):
            raise ValueError(f'Existing receipt has a different archive layout: {asset["id"]}')
        if archive.exists():
            observed = (prior['observed_sha256'] if prior.get('archive_stat') == file_stat(archive)
                        else sha256(archive))
            if observed != prior['observed_sha256']:
                raise ValueError(f'Archive changed after extraction: {archive}')
        if asset.get('sha256') and prior['observed_sha256'] != asset['sha256']:
            raise ValueError('Expected SHA256 mismatch: ' + asset['id'])
        if 'files' in prior:
            complete, verified = verify_inventory(destination, prior['files'])
            if complete:
                updated = dict(prior, files=list(verified.values()))
                if archive.exists():
                    updated['archive_stat'] = file_stat(archive)
                if updated != prior:
                    write_receipt(receipt, updated)
                return {'id': asset['id'], 'status': 'reused_extraction_receipt', 'receipt': str(receipt),
                        'verified_files': len(verified)}
    archives.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        guard.check(archives)
        if 'drive.google.com/' in asset['url']:
            # gdown 5.2.0 is the version recommended by INSID3 data.md.
            download_drive(asset['url'], archive, guard)
        else:
            download_http(asset['url'], archive, guard)
    observed = observed or sha256(archive)
    if asset.get('sha256') and observed != asset['sha256']:
        raise ValueError('Expected SHA256 mismatch: ' + asset['id'])
    if not zipfile.is_zipfile(archive):
        raise ValueError(f'Download is not a ZIP archive (possible login/quota response): {archive}')
    files = extract_zip(archive, destination, asset.get('strip_prefix'), guard, verified)
    record = {'id': asset['id'], 'status': 'extracted', 'source_url': asset['url'],
        'destination': str(destination), 'archive': str(archive),
        'strip_prefix': asset.get('strip_prefix'), 'archive_stat': file_stat(archive),
        'files': files,
        'observed_sha256': observed, 'expected_sha256': asset.get('sha256'),
        'upstream_checksum_verified': bool(asset.get('sha256')),
        'completed_utc': datetime.now(timezone.utc).isoformat()}
    write_receipt(receipt, record)
    return {key: value for key, value in record.items() if key != 'files'} | {'verified_files': len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', required=True, choices=['coco','lvis','fss','pascal','isic','suim','isaid'])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--archive-dir', type=Path)
    parser.add_argument('--catalog', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/data_sources.json')
    parser.add_argument('--workers', type=int, default=2)
    capacity = parser.add_mutually_exclusive_group()
    capacity.add_argument('--min-free-gb', type=float, default=0,
                          help='Best-effort free-space reserve in decimal GB (default: disabled)')
    capacity.add_argument('--min-free-gib', type=float, default=0,
                          help='Best-effort free-space reserve in binary GiB (default: disabled)')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    minimum = args.min_free_gb * 10**9 + args.min_free_gib * 2**30
    if not math.isfinite(minimum) or minimum < 0:
        parser.error('--min-free-gb/--min-free-gib must be finite and nonnegative')
    guard = CapacityGuard(math.ceil(minimum))
    catalog = json.loads(args.catalog.read_text())
    assets = [a for a in catalog['assets'] if a['dataset'] in args.datasets]
    notes = {k: v for k, v in catalog['additional_preparation'].items() if k in args.datasets}
    root = args.data_root.resolve()
    archives = (args.archive_dir or root / 'download_archives').resolve()
    if not args.execute:
        print(json.dumps({'status': 'plan_only', 'assets': assets, 'additional_preparation': notes,
                          'data_root': str(root), 'archive_dir': str(archives),
                          'minimum_free_bytes': guard.minimum_bytes}, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    # Separate archive assets target disjoint files. Receipts are per asset.
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(acquire, asset, root, archives, guard): asset['id'] for asset in assets}
        for future in as_completed(futures):
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as exc:
                failure = {'id': futures[future], 'status': 'failed', 'error': str(exc)}
                failures.append(failure)
                print(json.dumps(failure), flush=True)
    if notes:
        print(json.dumps({'additional_preparation': notes}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
