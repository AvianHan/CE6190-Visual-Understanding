#!/usr/bin/env python3
"""Plan/download only requested dataset archives; never downloads model weights.

Default is a dry plan. --execute downloads/extracts assets, records observed
checksums, resumes HTTP downloads and reuses complete extraction receipts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import zipfile


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


def extract_zip(path: Path, destination: Path, strip_prefix: str | None = None):
    """Reject path traversal and symlinks, and never replace differing local files."""
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
        for member, target in members:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + '.ics-extract.part')
            try:
                with archive.open(member) as src, tmp.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                if target.exists():
                    if sha256(tmp) != sha256(target):
                        raise FileExistsError(f'Refusing to overwrite differing file: {target}')
                    tmp.unlink()
                else:
                    tmp.replace(target)
            finally:
                if tmp.exists():
                    tmp.unlink()


def download_http(url: str, target: Path):
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
        with partial.open('ab' if resumed else 'wb') as stream:
            shutil.copyfileobj(response, stream, 1024 * 1024)
        received_bytes = partial.stat().st_size - starting_size
        if expected_bytes is not None and received_bytes != expected_bytes:
            raise OSError(f'Incomplete HTTP download: expected {expected_bytes} response '
                          f'bytes, received {received_bytes}; partial retained at {partial}')
        if resumed and int(range_match.group(2)) + 1 != int(range_match.group(3)):
            raise OSError(f'HTTP range ended before the complete asset; partial retained at {partial}')
    partial.replace(target)


def acquire(asset: dict, root: Path, archives: Path):
    archive = archives / (asset['id'] + '.zip')
    receipt = archives / (asset['id'] + '.receipt.json')
    destination = contained(root, asset['destination'])
    if receipt.exists() and destination.is_dir():
        prior = json.loads(receipt.read_text())
        if prior.get('source_url') != asset['url'] or prior.get('destination') != str(destination):
            raise ValueError(f'Existing receipt conflicts with catalog entry {asset["id"]}')
        if archive.exists() and sha256(archive) != prior['observed_sha256']:
            raise ValueError(f'Archive changed after extraction: {archive}')
        return {'id': asset['id'], 'status': 'reused_extraction_receipt', 'receipt': str(receipt)}
    archives.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        if 'drive.google.com/' in asset['url']:
            # gdown 5.2.0 is the version recommended by INSID3 data.md.
            partial = archive.with_suffix('.zip.part')
            subprocess.run([sys.executable, '-m', 'gdown', '--fuzzy', '--continue',
                            asset['url'], '-O', str(partial)], check=True)
            partial.replace(archive)
        else:
            download_http(asset['url'], archive)
    observed = sha256(archive)
    if asset.get('sha256') and observed != asset['sha256']:
        raise ValueError('Expected SHA256 mismatch: ' + asset['id'])
    if not zipfile.is_zipfile(archive):
        raise ValueError(f'Download is not a ZIP archive (possible login/quota response): {archive}')
    extract_zip(archive, destination, asset.get('strip_prefix'))
    record = {'id': asset['id'], 'status': 'extracted', 'source_url': asset['url'],
        'destination': str(destination), 'archive': str(archive),
        'observed_sha256': observed, 'expected_sha256': asset.get('sha256'),
        'upstream_checksum_verified': bool(asset.get('sha256')),
        'completed_utc': datetime.now(timezone.utc).isoformat()}
    temporary = receipt.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(receipt)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', required=True, choices=['coco','lvis','fss','pascal','isic','suim','isaid'])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--archive-dir', type=Path)
    parser.add_argument('--catalog', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/data_sources.json')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    catalog = json.loads(args.catalog.read_text())
    assets = [a for a in catalog['assets'] if a['dataset'] in args.datasets]
    notes = {k: v for k, v in catalog['additional_preparation'].items() if k in args.datasets}
    root = args.data_root.resolve()
    archives = (args.archive_dir or root / 'download_archives').resolve()
    if not args.execute:
        print(json.dumps({'status': 'plan_only', 'assets': assets, 'additional_preparation': notes,
                          'data_root': str(root), 'archive_dir': str(archives)}, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    # Separate archive assets target disjoint files. Receipts are per asset.
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(acquire, asset, root, archives): asset['id'] for asset in assets}
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
