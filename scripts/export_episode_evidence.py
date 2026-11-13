#!/usr/bin/env python3
"""Read-only, episode-first evidence export for the existing ICS recovery run.

Only direct plan output/episodes directories are enumerated. Predictions, locks,
datasets, weights and environments are never traversed. Original episode bytes
are preserved in JSONL envelopes with SHA256; no experiment or collector runs.
"""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tarfile
import time

RUN = Path('/scratch/prj0000000262_1/hansy/ce6190-ics-runtime/runs/ics-recovery-20261113')
EXPECTED = '4d7678557ef476de3bc2fd605ee7384745ba991599bc8b91bba5855f298c1a05'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def safe_tag(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value):
        raise ValueError('Tag must contain 1–64 ASCII letters, digits, underscores or hyphens')
    return value


def safe_path(root, path):
    """Lexical containment and no symlink in any path component; no traversal."""
    root, path = Path(root).absolute(), Path(path).absolute()
    relative = path.relative_to(root)
    if '..' in relative.parts:
        raise ValueError('Parent traversal is not allowed')
    check = root
    if root.is_symlink():
        raise ValueError('Symlink run root is not allowed')
    for part in relative.parts:
        check = check / part
        if check.is_symlink():
            raise ValueError('Symlink path not copied: ' + str(check))
    return relative


def read_regular(path, limit):
    """Read one regular file without following its final symlink; retain exact bytes."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('Not a regular file')
        if before.st_size > limit:
            raise ValueError('File exceeds per-file size cap')
        with os.fdopen(fd, 'rb', closefd=False) as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            raise ValueError('File grew beyond per-file size cap')
        after = os.fstat(fd)
        return data, {'size_before': before.st_size, 'mtime_ns_before': before.st_mtime_ns,
                      'included_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                      'changed_during_read': (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)}
    finally:
        os.close(fd)


def probe(command):
    started = time.time()
    try:
        value = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {'command': command, 'started_at': started, 'ended_at': time.time(),
                'exit': value.returncode, 'stdout': value.stdout, 'stderr': value.stderr}
    except (OSError, subprocess.SubprocessError) as exc:
        return {'command': command, 'started_at': started, 'ended_at': time.time(), 'error': str(exc)}


def output_groups(plan, state, accounting=None):
    accounting = accounting or {}
    groups = {}
    for phase, definitions in (('experiment', plan.get('tasks', [])), ('smoke', plan.get('smoke', []))):
        for task in definitions:
            path = str(task['output'])
            group = groups.setdefault(path, {'output': path, 'phase': phase,
                'config': task['config_data']['experiment_id'], 'manifest_id': task['manifest_id'],
                'output_id': hashlib.sha256(path.encode()).hexdigest()[:16], 'tasks': []})
            current = state.get('tasks', {}).get(task['id'], {})
            group['tasks'].append({'task_id': task['id'], 'shard_index': task['shard_index'],
                'num_shards': task['num_shards'], 'max_episodes': task.get('max_episodes'),
                'manifest': task['manifest'], 'saved_status': current.get('status', 'UNKNOWN'),
                'job_id': current.get('job_id'),
                'accounting_state': accounting.get(str(current.get('job_id')))})
    def priority(group):
        statuses = {task['saved_status'] for task in group['tasks']}
        submitted = any(task['job_id'] for task in group['tasks'])
        completed = any(task['accounting_state'] == 'COMPLETED' for task in group['tasks'])
        return (group['phase'] != 'experiment', not completed, 'PASS' not in statuses, not submitted,
                group['config'], group['manifest_id'])
    return sorted(groups.values(), key=priority)


class EpisodeExport:
    def __init__(self, run, out, seconds=600, raw_budget=200*2**20, max_entries=200000):
        self.run, self.out = Path(run), Path(out)
        self.deadline = time.monotonic() + seconds
        self.raw_budget, self.max_entries = raw_budget, max_entries
        self.bytes = 0
        self.entries = 0
        self.included = 0
        self.omissions = []
        self.inventory = []
        self.progress_at = time.monotonic()

    def expired(self):
        return time.monotonic() >= self.deadline or self.entries >= self.max_entries

    def episodes(self, group, records, index):
        folder = Path(group['output']) / 'episodes'
        report = {'output_id': group['output_id'], 'config': group['config'], 'phase': group['phase'],
                  'manifest_id': group['manifest_id'], 'path': str(folder), 'status': 'PENDING',
                  'entries_seen': 0, 'json_seen': 0, 'included': 0, 'raw_bytes': 0, 'record_status_counts': {}}
        counts = Counter()
        if self.expired():
            report['status'] = 'UNSCANNED_TIME_OR_ENTRY_BUDGET'
            self.omissions.append({'path': str(folder), 'reason': report['status']})
            return report
        try:
            safe_path(self.run, folder)
            # Deliberately enumerate ONLY this directory. No rglob/walk/JSONL pre-pass.
            with os.scandir(folder) as entries:
                report['status'] = 'SCANNED'
                for entry in entries:
                    if self.expired():
                        report['status'] = 'PARTIAL_TIME_OR_ENTRY_BUDGET'
                        self.omissions.append({'path': str(folder), 'reason': report['status'],
                                               'remaining_entries': 'not_enumerated'})
                        break
                    self.entries += 1; report['entries_seen'] += 1
                    if not entry.name.endswith('.json'):
                        continue
                    report['json_seen'] += 1
                    row = {'path': str(Path(entry.path).relative_to(self.run)), 'output_id': group['output_id']}
                    self.inventory.append(row)
                    try:
                        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                            raise ValueError('Non-regular/symlink episode not copied')
                        remaining = self.raw_budget - self.bytes
                        if remaining <= 0:
                            raise ValueError('Raw episode byte budget exhausted')
                        data, meta = read_regular(entry.path, min(2*2**20, remaining))
                        row.update(meta)
                        envelope = {'output_id': group['output_id'], 'path': row['path'], **meta}
                        record = None
                        try:
                            text = data.decode('utf-8')
                            envelope['raw_utf8'] = text
                            try:
                                record = json.loads(text)
                                if not isinstance(record, dict):
                                    raise ValueError('Episode JSON is not an object')
                            except (ValueError, TypeError) as exc:
                                envelope['parse_error'] = str(exc)
                        except UnicodeDecodeError:
                            envelope['raw_base64'] = base64.b64encode(data).decode('ascii')
                            envelope['parse_error'] = 'Episode is not UTF-8'
                        records.write(json.dumps(envelope, ensure_ascii=False) + '\n')
                        record = record if isinstance(record, dict) else {}
                        status = record.get('status', 'UNPARSEABLE')
                        counts[str(status)] += 1
                        index.writerow({'output_id': group['output_id'], 'config': group['config'],
                            'phase': group['phase'], 'manifest_id': group['manifest_id'],
                            'path': row['path'], 'sha256': meta['sha256'], 'bytes': len(data),
                            'changed_during_read': meta['changed_during_read'], 'episode_id': record.get('episode_id'),
                            'status': status, 'iou': record.get('iou'), 'inference_seconds': record.get('inference_seconds'),
                            'episode_wall_seconds': record.get('episode_wall_seconds'),
                            'query_image_sha256': record.get('query_image_sha256'),
                            'shard_index': record.get('shard_index'), 'num_shards': record.get('num_shards')})
                        self.bytes += len(data); self.included += 1
                        report['included'] += 1; report['raw_bytes'] += len(data)
                    except (OSError, ValueError) as exc:
                        row['error'] = str(exc)
                        self.omissions.append({**row, 'reason': 'EPISODE_NOT_COPIED'})
                        report['status'] = 'SCANNED_WITH_OMISSIONS' if report['status'] == 'SCANNED' else report['status']
                    if time.monotonic() - self.progress_at >= 15:
                        records.flush()
                        print('Episode progress: %s / %s included=%d total=%d' %
                              (group['config'], group['manifest_id'], report['included'], self.included), flush=True)
                        self.progress_at = time.monotonic()
        except FileNotFoundError:
            report['status'] = 'DIRECTORY_ABSENT'
        except (OSError, ValueError) as exc:
            report['status'] = 'DIRECTORY_UNREADABLE'
            report['error'] = str(exc)
            self.omissions.append({'path': str(folder), 'reason': report['status'], 'error': str(exc)})
        report['record_status_counts'] = dict(counts)
        return report


def collect(run, out, home, with_scheduler=True, seconds=600, raw_budget=200*2**20,
            expected_fingerprint=EXPECTED, metadata_seconds=120):
    run, out, home = Path(run).absolute(), Path(out).absolute(), Path(home).absolute()
    if run.is_symlink() or not run.is_dir():
        raise ValueError('Run must be an existing real directory')
    if out.exists() or out == run or run in out.parents:
        raise ValueError('Use a new export directory outside the run')
    # Validate identity before creating any output or scanning result directories.
    plan_bytes, _ = read_regular(run / 'plan.json', 12*2**20)
    state_bytes, _ = read_regular(run / 'rolling_state.json', 4*2**20)
    plan, state = json.loads(plan_bytes), json.loads(state_bytes)
    if plan.get('fingerprint') != expected_fingerprint or state.get('plan_fingerprint') != expected_fingerprint:
        raise ValueError('Unexpected plan/state fingerprint')
    out.mkdir(parents=True)
    (out / 'plan.json').write_bytes(plan_bytes)
    (out / 'state_before.json').write_bytes(state_bytes)
    started = time.time()
    probes = []
    accounting = {}
    if with_scheduler:
        ids = sorted({str(task['job_id']) for task in state.get('tasks', {}).values() if task.get('job_id')})
        if ids:
            answer = probe(['sacct','--allocations','--noheader','--parsable2','-S','2026-11-13',
                            '-j',','.join(ids),'--format=JobIDRaw,State'])
            probes.append(answer)
            if answer.get('exit') == 0:
                for line in answer['stdout'].splitlines():
                    fields = line.split('|')
                    if len(fields) >= 2 and fields[0].isdigit():
                        accounting[fields[0]] = fields[1]
    groups = output_groups(plan, state, accounting)
    dump(out / 'output_task_mapping.json', groups)
    export = EpisodeExport(run, out, seconds, raw_budget)
    fields = ['output_id','config','phase','manifest_id','path','sha256','bytes','changed_during_read',
              'episode_id','status','iou','inference_seconds','episode_wall_seconds','query_image_sha256',
              'shard_index','num_shards']
    reports = []
    print('Episode-first export: %d unique plan outputs; no recursive result scan.' % len(groups), flush=True)
    with (out / 'episodes.jsonl').open('w') as records, (out / 'episode_index.csv').open('w', newline='') as index_file:
        index = csv.DictWriter(index_file, fieldnames=fields); index.writeheader()
        for number, group in enumerate(groups, 1):
            report = export.episodes(group, records, index)
            reports.append(report)
            print('[%d/%d] %s %s %s: %d records (%s)' %
                  (number,len(groups),group['phase'],group['config'],group['manifest_id'],report['included'],report['status']), flush=True)
    # Source/metadata have a fresh bounded budget; they can never starve episode collection.
    print('Episode collection finished: %d files / %.1f MiB. Capturing exact source manifest.' %
          (export.included, export.bytes / 2**20), flush=True)
    metadata_deadline = time.monotonic() + metadata_seconds
    metadata_used = Counter()
    metadata_inventory = []
    def copy_relative(path, destination=None, group='metadata', force=False, per_file=8*2**20):
        row = {'path': str(path), 'group': group}
        metadata_inventory.append(row)
        try:
            if not force and time.monotonic() >= metadata_deadline:
                raise ValueError('Metadata time budget exhausted')
            relative = safe_path(run, path)
            cap = 8*2**20 if group == 'source' else 40*2**20
            remaining = cap - metadata_used[group]
            if remaining <= 0:
                raise ValueError('Metadata group byte budget exhausted')
            data, meta = read_regular(path, min(remaining, per_file))
            target = out / (destination or Path('run') / relative)
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
            metadata_used[group] += len(data); row.update(meta, destination=str(target.relative_to(out)))
            return data
        except (OSError, ValueError) as exc:
            row['error'] = str(exc)
            return None
    for source in ('source', 'scheduling/source', 'repair/source'):
        raw = copy_relative(run / source / 'snapshot.json', per_file=2*2**20)
        if not raw:
            continue
        try:
            snapshot = json.loads(raw)
            for name, digest in snapshot['files'].items():
                relative = Path(name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Unsafe snapshot file name')
                copied = copy_relative(run / source / relative, group='source')
                if copied is not None:
                    metadata_inventory[-1]['expected_sha256'] = digest
                    metadata_inventory[-1]['snapshot_hash_matches'] = hashlib.sha256(copied).hexdigest() == digest
        except (ValueError, KeyError, TypeError) as exc:
            export.omissions.append({'path':source+'/snapshot.json','reason':'INVALID_SOURCE_MANIFEST','error':str(exc)})
    # Exact output costs plus one-level shard summaries. Never enter predictions/locks/attempts.
    included_outputs = {report['output_id'] for report in reports if report['included']}
    for group in groups:
        if group['output_id'] not in included_outputs:
            continue
        for filename in ('run.json','adapter.json'):
            copy_relative(Path(group['output']) / filename)
        folder = Path(group['output']) / 'shards'
        try:
            safe_path(run, folder)
            if time.monotonic() >= metadata_deadline:
                raise ValueError('Shard metadata time budget exhausted before enumeration')
            with os.scandir(folder) as entries:
                for entry in entries:
                    if time.monotonic() >= metadata_deadline:
                        raise ValueError('Shard metadata time budget exhausted; remaining entries unenumerated')
                    if entry.name.endswith('.json'):
                        copy_relative(Path(entry.path), per_file=2*2**20)
        except (OSError, ValueError) as exc:
            metadata_inventory.append({'path':str(folder),'group':'metadata','error':str(exc)})
    # Current and historical job costs are addressed using known task/job IDs, never a recursive archive scan.
    for key, current in state.get('tasks', {}).items():
        if not current.get('job_id') and not current.get('attempt_history'):
            continue
        locations = [run/'jobs'/key]
        for attempt in current.get('attempt_history', []):
            jid = str(attempt.get('job_id', ''))
            if jid.isdigit():
                locations.append(run/'repair/archives'/key/('job-'+jid)/'job')
        for location in dict.fromkeys(locations):
            for name in ('status.json','complete.json','failure.json','gpu_memory.json','config.json'):
                copy_relative(location/name, per_file=2*2**20)
    for name in ('policy.json','prepare.json','applied.json','archive_inventory.json','transaction.json'):
        copy_relative(run/'repair'/name, force=True)
    # Enumerate external path values from the plan, but only lstat their metadata, never their content.
    external_values = {}
    def path_values(value, label):
        if isinstance(value, dict):
            for key, child in value.items(): path_values(child, label+'/'+str(key))
        elif isinstance(value, list):
            for i, child in enumerate(value): path_values(child, label+'/'+str(i))
        elif isinstance(value, str) and value.startswith('/'):
            external_values.setdefault(value, []).append(label)
    path_values(plan.get('resources', {}), 'resources')
    path_values(plan.get('manifests', {}), 'manifests')
    for task in plan.get('tasks', []) + plan.get('smoke', []):
        path_values(task.get('config_data', {}).get('adapter', {}), 'adapter/'+task['config_data']['experiment_id'])
    external = []
    for name, labels in sorted(external_values.items()):
        row = {'path':name,'plan_fields':sorted(set(labels)),'content_copied':False}
        try:
            if time.monotonic() >= metadata_deadline:
                raise ValueError('Metadata time budget exhausted; external path not inspected')
            info = Path(name).lstat()
            row.update(size=info.st_size,mtime_ns=info.st_mtime_ns,symlink=stat.S_ISLNK(info.st_mode))
        except (OSError, ValueError) as exc: row['error'] = str(exc)
        external.append(row)
    dump(out/'external_paths.json', external)
    for name in ('manager.json','scheduling/policy.json'):
        copy_relative(run / name, force=True)
    after = copy_relative(run / 'rolling_state.json', Path('state_after.json'), group='final_state', force=True, per_file=4*2**20)
    if with_scheduler:
        probes.append(probe(['squeue','-r','-u',state['policy']['user'],'-o','%i|%P|%T|%M|%b|%j|%R']))
    dump(out / 'probe_results.json', probes)
    detail = {'path': str(home / 'CE6190-home-detail-20261113.txt'), 'copied': False}
    try:
        data, info = read_regular(Path(detail['path']), 2*2**20)
        (out / 'home_detail.txt').write_bytes(data); detail.update(info, copied=True)
    except (OSError, ValueError) as exc:
        detail['error'] = str(exc)
    dump(out / 'directory_coverage.json', reports)
    dump(out / 'file_inventory.json', export.inventory)
    dump(out / 'metadata_inventory.json', metadata_inventory)
    dump(out / 'omissions.json', export.omissions)
    complete = all(row['status'] in {'SCANNED','DIRECTORY_ABSENT'} for row in reports)
    report = {'schema_version':1,'run':str(run),'started_at':started,'ended_at':time.time(),
        'non_atomic':True,'plan_fingerprint':plan['fingerprint'],'episode_files':export.included,
        'raw_episode_bytes':export.bytes,'episode_budget_bytes':raw_budget,'episode_seconds':seconds,
        'metadata_seconds':metadata_seconds,'metadata_bytes':dict(metadata_used),
        'all_plan_episode_directories_scanned':complete,'state_after_captured':after is not None,
        'directory_status_counts':dict(Counter(row['status'] for row in reports)),
        'home_detail':detail,'record_format':'episodes.jsonl raw_utf8 (or raw_base64) reconstructs exact original bytes; sha256 covers original bytes',
        'scope':'Direct plan output/episodes/*.json; exact snapshot sources; output run/adapter/shard cost JSON; known current/archive job cost JSON; exact repair metadata; external path metadata only.',
        'omitted_by_design':['predictions/masks','locks','attempts history','datasets','weights','environments','external manifest content'],
        'interpretation':'Directory absent is not zero expected episodes or experiment success. Saved task state may be stale after manager stop. No metric completeness claim is made.'}
    dump(out / 'EXPORT_REPORT.json', report)
    archive = out / 'CE6190_ICS_Episode_Evidence.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for path in sorted(out.rglob('*')):  # Only the newly written local export, never server experiment trees.
            if path.is_file() and path != archive:
                tar.add(path, arcname=str(path.relative_to(out)), recursive=False)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / 'SHA256SUMS.txt').write_text(digest+'  '+archive.name+'\n')
    print('EPISODES=%d\nCOVERAGE_COMPLETE=%s\nARCHIVE=%s\nSHA256=%s' %
          (export.included, complete, archive, digest), flush=True)
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True,type=safe_tag)
    args = parser.parse_args()
    collect(RUN, Path.home()/('CE6190-episode-evidence-'+args.tag), Path.home())
