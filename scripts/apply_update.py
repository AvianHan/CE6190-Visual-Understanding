#!/usr/bin/env python3
"""Apply this downloaded bundle to an existing clone, preserving Git and backups."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def apply(source: Path, target: Path, execute=False):
    source, target = source.resolve(), target.expanduser().resolve()
    if source == target or target.is_relative_to(source):
        raise ValueError('Run the updater from the extracted download, targeting the existing Git clone')
    if not (target / '.git').exists():
        raise ValueError('Target must be your existing Git clone; no repository will be created')
    remote = subprocess.run(['git', '-C', str(target), 'remote', 'get-url', 'origin'],
                            capture_output=True, text=True)
    if remote.returncode or 'avianhan/ce6190-visual-understanding' not in remote.stdout.lower():
        raise ValueError('Target origin is not AvianHan/CE6190-Visual-Understanding; check --target')
    manifest = json.loads((source / 'UPDATE_MANIFEST.json').read_text())
    changes = []
    for name, expected in manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '.git' in relative.parts:
            raise ValueError('Unsafe update path')
        src, dst = source / name, target / name
        if not src.is_file() or src.is_symlink() or not src.resolve().is_relative_to(source):
            raise ValueError('Invalid source file: ' + name)
        if sha(src) != expected:
            raise ValueError('Downloaded bundle file hash mismatch: ' + name)
        if not dst.resolve().is_relative_to(target) or dst.is_symlink():
            raise ValueError('Target symlink would escape/redirect update: ' + name)
        if dst.exists() and not dst.is_file():
            raise ValueError('Target is not a regular file: ' + name)
        if dst.exists() and sha(dst) == expected:
            continue
        changes.append((name, src, dst, dst.exists()))
    result = {'status': 'PLANNED', 'target': str(target),
              'new_files': [n for n,_,_,old in changes if not old],
              'replaced_files': [n for n,_,_,old in changes if old],
              'git_actions': 'none', 'unlisted_existing_files': 'preserved'}
    if not execute or not changes:
        result['status'] = 'UP_TO_DATE' if not changes else 'PLANNED'
        return result
    backup_root = target.parent / (target.name + '-update-backups')
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    backup = Path(tempfile.mkdtemp(prefix=stamp + '-', dir=backup_root))
    # All replaced files are backed up before any replacement is made.
    for name, _, dst, old in changes:
        if old:
            original = backup / name
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, original)
    for name, src, dst, _ in changes:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + '.ics-update-tmp')
        shutil.copy2(src, tmp)
        tmp.replace(dst)
    result.update(status='APPLIED', backup=str(backup))
    (backup/'update_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--apply', action='store_true', help='Apply verified files, with backup of every replacement')
    a = p.parse_args()
    print(json.dumps(apply(a.source, a.target, a.apply), indent=2, ensure_ascii=False))
