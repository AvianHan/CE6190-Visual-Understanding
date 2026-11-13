#!/usr/bin/env python3
"""Prepare/restart a selective recovery queue without altering the original run."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ics.orchestration import build_plan, load_json, write_plan
from ics.recovery import (parent_evidence, assert_parent_inactive, manifest_evidence,
                          build_recovery_plan, assert_no_partial_success,
                          verify_recovery_parent)
from ics.rolling_queue import initialize_queue, queue_status, _plan
from scripts.queue_ics import (acquire_lock, clean_environment, freeze_source, verify_snapshot,
                       required_datasets, saved_status, print_json)


def assert_unique_recovery(parent: Path, run: Path, expected_fingerprint: str):
    for path in parent.parent.glob('*/plan.json'):
        if path.parent.resolve() == run or path.parent.resolve() == parent:
            continue
        value = load_json(path)
        if value.get('recovery', {}).get('parent_plan_fingerprint') == expected_fingerprint:
            raise ValueError(f'Another recovery already references this parent: {path.parent}; resume it instead')


def prepare(args):
    parent, run, runtime = args.from_run.resolve(), args.run_dir.resolve(), args.runtime.resolve()
    if parent == run or parent.is_relative_to(run) or run.is_relative_to(parent):
        raise ValueError('Recovery output must be a separate sibling run, outside the parent')
    old, state, evidence = parent_evidence(parent, expected_fingerprint=args.expected_parent_fingerprint)
    if Path(load_json(parent / 'source/snapshot.json')['selection']['runtime']).resolve() != runtime:
        raise ValueError('Recovery must use the original runtime')
    assert_parent_inactive(parent, old, state)
    assert_unique_recovery(parent, run, old["fingerprint"])
    env = clean_environment(runtime)
    os.environ.clear()
    os.environ.update(env)
    if ROOT != run / 'source':
        suite = args.suite if args.suite.is_absolute() else ROOT / args.suite
        cluster = args.cluster if args.cluster.is_absolute() else ROOT / args.cluster
        frozen = freeze_source(run, suite, cluster, runtime)
        selection = load_json(frozen / 'snapshot.json')['selection']
        cmd = [sys.executable, '-u', str(frozen / 'scripts/recover_ics.py'), 'prepare',
               '--from-run', str(parent), '--run-dir', str(run), '--runtime', str(runtime),
               '--suite', str(frozen / selection['suite']), '--cluster', str(frozen / selection['cluster']),
               '--expected-parent-fingerprint', args.expected_parent_fingerprint,
               '--expected-missing', str(args.expected_missing), '--expected-preserved', str(args.expected_preserved)]
        return subprocess.run(cmd, env=env, check=False).returncode
    verify_snapshot(ROOT)
    selection = load_json(ROOT / 'snapshot.json')['selection']
    suite = args.suite if args.suite.is_absolute() else ROOT / args.suite
    cluster = args.cluster if args.cluster.is_absolute() else ROOT / args.cluster
    if (suite.resolve() != ROOT / selection['suite'] or
            cluster.resolve() != ROOT / selection['cluster'] or
            str(runtime) != selection['runtime']):
        raise ValueError('Frozen recovery selection cannot be changed; use its original prepare command')
    args.suite, args.cluster = suite, cluster
    lock = acquire_lock(run / 'manager.lock')
    if lock is None:
        raise RuntimeError('Recovery manager is active; use status instead of preparation')
    registry = acquire_lock(parent.parent / ("recovery-" + old["fingerprint"] + ".lock"))
    if registry is None:
        lock.close()
        raise RuntimeError("Another recovery preparation for this parent is active")
    with lock, registry:
        assert_unique_recovery(parent, run, old["fingerprint"])
        manifests, log_name, log_hash = manifest_evidence(parent, old, state)
        evidence[log_name] = log_hash
        candidate = build_plan(args.suite, args.cluster, run)
        plan = build_recovery_plan(candidate, old, state, parent, evidence, manifests,
                                   expected_missing=args.expected_missing,
                                   expected_preserved=args.expected_preserved)
        assert_no_partial_success(parent, old, plan['recovery']['selected_task_ids'])
        path = write_plan(plan)
        policy = state['policy']
        initialize_queue(path, max_inflight=policy['max_inflight'], shared_limit=policy['shared_limit'],
                         reserved_slots=policy['reserved_slots'], datasets=required_datasets(plan, runtime),
                         data_workers=policy['data_workers'])
        print_json({'recovery': 'PREPARED_NOT_STARTED', 'run_dir': str(run),
                    'selected_experiments': len(plan['tasks']),
                    'preserved_parent_experiments': len(plan['recovery']['preserved_task_ids']),
                    'smoke_checks': len(plan['smoke']), 'data_policy': 'verify_existing_only',
                    **queue_status(path)})
    return 0


def start(args):
    run = args.run_dir.resolve()
    plan = _plan(run / 'plan.json')
    if not plan.get('recovery'):
        raise ValueError('This is not a prepared selective recovery plan')
    snapshot = run / 'source'
    verify_snapshot(snapshot)
    metadata = load_json(snapshot / 'snapshot.json')
    runtime = Path(metadata['selection']['runtime'])
    env = clean_environment(runtime)
    if ROOT != snapshot:
        return subprocess.run([sys.executable, '-u', str(snapshot / 'scripts/recover_ics.py'),
                               'start', '--run-dir', str(run), '--interval', str(args.interval)],
                              env=env, check=False).returncode
    os.environ.clear()
    os.environ.update(env)
    lock = acquire_lock(run / 'manager.lock')
    if lock is None:
        print_json({'manager': 'ALREADY_RUNNING', **queue_status(run / 'plan.json')})
        return 0
    with lock:
        verify_recovery_parent(plan)
        parent = Path(plan['recovery']['parent_run'])
        old, state, _ = parent_evidence(parent, expected_fingerprint=plan['recovery']['parent_plan_fingerprint'])
        assert_parent_inactive(parent, old, state)
        summary = queue_status(run / 'plan.json')
        if summary['terminal']:
            print_json({'manager': 'NOT_NEEDED', **summary})
            return 0 if summary['status'] == 'COMPLETE' else 2
        log_path = run / 'manager.log'
        with log_path.open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts/queue_ics.py'),
                'watch', '--run-dir', str(run), '--interval', str(args.interval),
                '--lock-fd', str(lock.fileno())], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
                pass_fds=(lock.fileno(),))
        deadline = time.monotonic() + 3
        ready = False
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f'Recovery manager exited with {child.returncode}; inspect {log_path}')
            try:
                ready = load_json(run / 'manager.json').get('pid') == child.pid
            except (OSError, ValueError):
                pass
            if ready:
                break
            time.sleep(0.05)
        print_json({'manager': 'STARTED' if ready else 'STARTING', 'manager_pid': child.pid,
                    'manager_log': str(log_path), **summary})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Verify parent and freeze filtered plan; never submit')
    prep.add_argument('--from-run', type=Path, required=True)
    prep.add_argument('--run-dir', type=Path, required=True)
    prep.add_argument('--runtime', type=Path, required=True)
    prep.add_argument('--suite', type=Path, default=Path('configs/suite.initial.lvisfix.json'))
    prep.add_argument('--cluster', type=Path, default=Path('configs/cluster.orion.recovery.json'))
    prep.add_argument('--expected-parent-fingerprint', required=True)
    prep.add_argument('--expected-missing', type=int, required=True)
    prep.add_argument('--expected-preserved', type=int, required=True)
    launch = commands.add_parser('start', help='Start/resume only a prepared immutable recovery plan')
    launch.add_argument('--run-dir', type=Path, required=True)
    launch.add_argument('--interval', type=float, default=60)
    status = commands.add_parser('status', help='Read saved status without scheduler mutation')
    status.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        if args.expected_missing < 1 or args.expected_preserved < 0:
            parser.error('Expected counts must be nonnegative and missing must be positive')
        return prepare(args)
    if args.command == 'start':
        if not math.isfinite(args.interval) or args.interval < 10:
            parser.error('interval must be finite and at least 10 seconds')
        return start(args)
    print_json(saved_status(args.run_dir.resolve()))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.TimeoutExpired) as exc:
        print(f'Recovery error: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
