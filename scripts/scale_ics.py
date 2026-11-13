#!/usr/bin/env python3
"""Prepare and activate a separate scheduling controller for the existing ICS run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from queue_ics import (acquire_lock, clean_environment, freeze_source, manager_status,
                       print_json, saved_status, verify_snapshot, watcher)
from ics.orchestration import fingerprint, load_json
from ics.rolling_queue import _atomic_json, _plan, _read_state, queue_status
from ics.scheduling import load_policy, qos_limit, partition_evidence


def checked_plan(run):
    plan = _plan(run / 'plan.json')
    if Path(plan['run_dir']).resolve() != run or Path(plan['project_root']).resolve() != run / 'source':
        raise ValueError('Expected this existing run with its original frozen worker source')
    verify_snapshot(run / 'source')
    _read_state(plan)
    return plan


def prepare(args):
    run = args.run_dir.resolve()
    plan = checked_plan(run)
    if plan['fingerprint'] != args.expected_plan_fingerprint:
        raise ValueError('Unexpected plan fingerprint; refusing scheduling change')
    directory = run / 'scheduling'
    directory.mkdir(exist_ok=True)
    lock = acquire_lock(directory / 'prepare.lock')
    if lock is None:
        raise RuntimeError('Another scheduling preparation is active')
    with lock:
        if (directory / 'policy.json').exists():
            policy = load_policy(plan)
            verify_snapshot(directory / 'source')
            if args.shared_limit is not None and args.shared_limit != policy['shared_limit']:
                raise ValueError('Existing immutable policy has a different shared limit')
            print_json({'scheduling': 'ALREADY_PREPARED', 'policy': policy})
            return 0
        state = _read_state(plan)
        inherited = state['policy']['shared_limit']
        evidence = qos_limit(state['policy']['qos'], inherited)
        if args.shared_limit is not None and not 1 <= args.shared_limit <= evidence['effective_limit']:
            raise ValueError('--shared-limit may only lower the current verified submission bound')
        cap = args.shared_limit or evidence['effective_limit']
        if cap <= state['policy']['reserved_slots']:
            raise ValueError('Shared limit leaves no usable submission slots')
        partitions = partition_evidence(plan, state)
        selection = load_json(run / 'source' / 'snapshot.json')['selection']
        source = freeze_source(directory, ROOT / selection['suite'], ROOT / selection['cluster'],
                               Path(selection['runtime']))
        policy = {'schema_version': 1, 'created_at': time.time(),
                  'plan_fingerprint': plan['fingerprint'], 'worker_source': plan['project_root'],
                  'controller_source': str(source), 'inherited_shared_limit': inherited,
                  'verified_bound': evidence['effective_limit'], 'shared_limit': cap, 'smoke_partition': 'l40sp',
                  'formal_partitions': 'h200p,l40sp', 'qos_evidence': evidence,
                  'partition_evidence': partitions,
                  'note': 'Existing workers, plan, task IDs, accepted jobs and PASS results are preserved.'}
        policy['fingerprint'] = fingerprint(policy)
        _atomic_json(directory / 'policy.json', policy)
        print_json({'scheduling': 'PREPARED_NOT_ACTIVATED', 'policy': policy})
    return 0


def verified_manager_pid(run, record):
    """Check local process ownership, exact watcher arguments and start time before SIGTERM."""
    if record.get('hostname') != socket.gethostname():
        raise RuntimeError('Manager is on another host; run activate on its recorded hostname')
    pid = record.get('pid')
    if type(pid) is not int or pid <= 1:
        raise RuntimeError('Manager PID is invalid')
    proc = Path('/proc') / str(pid)
    if proc.stat().st_uid != os.getuid():
        raise RuntimeError('Manager process has another owner')
    argv = [v.decode() for v in (proc / 'cmdline').read_bytes().split(b'\0') if v]
    allowed = {str(run / 'source' / 'scripts' / 'queue_ics.py'),
               str(run / 'scheduling' / 'source' / 'scripts' / 'scale_ics.py')}
    scripts = [value for value in argv if value in allowed]
    if len(scripts) != 1 or 'watch' not in argv or '--run-dir' not in argv:
        raise RuntimeError('Manager command does not match an authorized watcher')
    pos = argv.index('--run-dir')
    if pos + 1 >= len(argv) or Path(argv[pos + 1]).resolve() != run:
        raise RuntimeError('Manager command points at another run')
    stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                if line.startswith('btime '))
    started = boot + int(stat[19]) / os.sysconf('SC_CLK_TCK')
    if abs(started - float(record['started_at'])) > 30:
        raise RuntimeError('Manager PID start time does not match lifecycle evidence')
    expected_lock = (run / 'manager.lock').stat()
    held = False
    for fd in (proc / 'fd').iterdir():
        try:
            target = fd.stat()
            if (target.st_dev, target.st_ino) == (expected_lock.st_dev, expected_lock.st_ino):
                held = True
        except FileNotFoundError:
            pass
    if not held:
        raise RuntimeError('Manager process does not hold this run lock descriptor')
    return pid


def activate(args):
    run = args.run_dir.resolve()
    plan = checked_plan(run)
    policy = load_policy(plan)
    controller = run / 'scheduling' / 'source'
    if Path(policy['controller_source']).resolve() != controller:
        raise ValueError('Unexpected scheduling controller path')
    verify_snapshot(controller)  # Check BEFORE stopping a healthy legacy manager.
    summary = queue_status(run / 'plan.json')
    if summary['terminal']:
        print_json({'manager_start': 'NOT_NEEDED', **saved_status(run)})
        return 0 if summary['status'] == 'COMPLETE' else 2
    partitions = partition_evidence(plan, _read_state(plan))
    if ROOT != controller:
        return subprocess.run([sys.executable, '-I', '-B', '-u',
            str(controller / 'scripts' / 'scale_ics.py'), 'activate', '--run-dir', str(run),
            '--interval', str(args.interval)], check=False).returncode
    outer = acquire_lock(run / 'scheduling' / 'handover.lock')
    if outer is None:
        raise RuntimeError('Another manager handover is active')
    with outer:
        event_path = run / 'scheduling' / 'handovers' / (str(time.time_ns()) + '.json')
        event = {'at': time.time(), 'plan_fingerprint': plan['fingerprint'],
                 'policy_fingerprint': policy['fingerprint'], 'controller_source': str(controller),
                 'partition_evidence': partitions}
        lock = acquire_lock(run / 'manager.lock')
        if lock is None:
            previous = load_json(run / 'manager.json')
            pid = verified_manager_pid(run, previous)
            if previous.get('source') == str(controller):
                print_json({'manager_start': 'ALREADY_RUNNING_SCALED', **saved_status(run)})
                return 0
            event.update(previous_manager=previous, action='SIGTERM_MANAGER_ONLY')
            _atomic_json(event_path, event)
            # pidfd avoids signaling a recycled PID on supported Linux/Python.
            if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
                fd = os.pidfd_open(pid)
                try:
                    verified_manager_pid(run, previous)
                    signal.pidfd_send_signal(fd, signal.SIGTERM)
                finally:
                    os.close(fd)
            else:
                verified_manager_pid(run, previous)
                os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 90
            next_notice = time.monotonic() + 10
            while lock is None and time.monotonic() < deadline:
                time.sleep(0.1)
                lock = acquire_lock(run / 'manager.lock')
                if time.monotonic() >= next_notice:
                    print('Waiting for the previous manager to persist its current submission and release its lock...', flush=True)
                    next_notice += 10
            if lock is None:
                event.update(status='HANDOVER_WAIT_TIMEOUT')
                _atomic_json(event_path, event)
                raise RuntimeError('Previous manager has not released its lock; no second manager started. Inspect and repeat activate.')
        with lock:
            # The original manager has now finished its current sbatch acknowledgement.
            checked_plan(run)
            verify_snapshot(controller)
            state = _read_state(plan)
            backup = event_path.with_suffix('.state.json')
            _atomic_json(backup, state)
            event.update(state_backup=str(backup),
                         state_sha256=hashlib.sha256((run / 'rolling_state.json').read_bytes()).hexdigest())
            _atomic_json(event_path, event)
            selection = load_json(controller / 'snapshot.json')['selection']
            env = clean_environment(Path(selection['runtime']))
            log_path = run / 'manager.log'
            with log_path.open('a') as log:
                child = subprocess.Popen([sys.executable, '-I', '-B', '-u',
                    str(controller / 'scripts' / 'scale_ics.py'), 'watch', '--run-dir', str(run),
                    '--interval', str(args.interval), '--lock-fd', str(lock.fileno())],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=env, start_new_session=True, pass_fds=(lock.fileno(),))
            event.update(status='CONTROLLER_LAUNCHED', manager_pid=child.pid)
            _atomic_json(event_path, event)
            ready = False
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(f'Controller exited during startup ({child.returncode}); inspect {log_path}')
                try:
                    ready = load_json(run / 'manager.json').get('pid') == child.pid
                except (OSError, ValueError):
                    pass
                if ready:
                    break
                time.sleep(0.05)
            print_json({'manager_start': 'STARTED_SCALED' if ready else 'STARTING_SCALED',
                        'manager_pid': child.pid, 'handover_record': str(event_path),
                        **saved_status(run)})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--expected-plan-fingerprint', required=True)
    p.add_argument('--shared-limit', type=int)
    p = sub.add_parser('activate')
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--interval', type=float, default=60)
    p = sub.add_parser('status')
    p.add_argument('--run-dir', required=True, type=Path)
    p = sub.add_parser('watch', help=argparse.SUPPRESS)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--interval', type=float, default=60)
    p.add_argument('--lock-fd', type=int)
    p.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    if hasattr(args, 'interval') and (not math.isfinite(args.interval) or args.interval < 10):
        parser.error('--interval must be finite and at least 10 seconds')
    if args.command == 'prepare':
        return prepare(args)
    if args.command == 'activate':
        return activate(args)
    run = args.run_dir.resolve()
    if args.command == 'status':
        result = saved_status(run)
        if (run / 'scheduling' / 'policy.json').exists():
            result['scheduling_policy'] = load_policy(checked_plan(run))
        print_json(result)
        return 0
    plan = checked_plan(run)
    policy = load_policy(plan)
    if ROOT != Path(policy['controller_source']).resolve():
        raise ValueError('Watch must execute the prepared immutable controller')
    verify_snapshot(ROOT)
    return watcher(run, interval=args.interval, once=args.once, lock_fd=args.lock_fd, scheduling=True)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f'Scheduling error: {exc}', file=sys.stderr)
        raise SystemExit(1)
