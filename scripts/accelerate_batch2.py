#!/usr/bin/env python3
"""Run the additive pending-job accelerator beside the existing batch2 manager."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ics import batch2, batch2_acceleration as acceleration
from ics.orchestration import atomic_json, load_json
from scripts.queue_ics import (acquire_lock, clean_environment, freeze_source,
                               manager_status, print_json, verify_snapshot)


def recover_owned_holds(run):
    value = acceleration.recover_transactions(run)
    if isinstance(value, dict) and value.get('recovery_required'):
        raise RuntimeError('One or more owned hold transactions still require recovery')
    return value


def current_status(run):
    run = Path(run).resolve()
    folder = run / acceleration.TAG
    result = acceleration.status(run)
    result['original_manager'] = manager_status(run)['manager']
    result['accelerator'] = manager_status(folder)['manager']
    if (folder / 'manager.json').exists():
        record = load_json(folder / 'manager.json')
        result['accelerator_lifecycle'] = record
        result['heartbeat_age_seconds'] = round(time.time() - record.get('last_heartbeat_at', 0), 1)
    return result


def watch(run, interval=20, once=False, lock_fd=None, tick_fn=None):
    run = Path(run).resolve()
    folder = run / acceleration.TAG
    lock = os.fdopen(lock_fd, 'a+') if lock_fd is not None else acquire_lock(folder / 'manager.lock')
    if lock is None:
        raise RuntimeError('The accelerator is already active for this run')
    stop = threading.Event()
    previous = {}
    record = {'schema_version': 1, 'pid': os.getpid(), 'hostname': socket.gethostname(),
              'source': str(ROOT), 'parent_run': str(run), 'started_at': time.time(),
              'interval': interval, 'phase': 'initializing', 'exit': None}
    def publish(phase):
        record.update(phase=phase, last_heartbeat_at=time.time())
        atomic_json(folder / 'manager.json', record)
    def interrupted(signum, frame):
        record['stop_signal'] = signal.Signals(signum).name
        stop.set()
    reason, code = 'signal', 0
    with lock:
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(sig, interrupted)
            publish('initializing')
            recover_owned_holds(run)
            while not stop.is_set():
                verify_snapshot(ROOT)
                if (folder / 'stop.request.json').exists():
                    reason = 'requested'; break
                if manager_status(run)['manager'] != 'ACTIVE':
                    # Restore our outstanding holds and remain available while
                    # the user inspects/restarts the original manager. Never
                    # take over its submission or queue-state responsibilities.
                    recover_owned_holds(run)
                    publish('waiting_for_original_manager')
                    if once: reason = 'once'; break
                    stop.wait(interval); continue
                publish('polling')
                record['last_poll_started_at'] = time.time()
                try:
                    value = (tick_fn or acceleration.tick)(run, stop_requested=stop.is_set)
                    record.pop('last_poll_error', None)
                    record['last_poll_completed_at'] = time.time()
                    print_json(value)
                except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
                    # Network/Slurm query failure must not stop the original
                    # healthy queue or authorize an unproven hardware route.
                    record['last_poll_error'] = {'type': type(exc).__name__, 'message': str(exc)[:2000]}
                    print_json({'accelerator_poll': 'DEFERRED', 'error': str(exc)})
                    recover_owned_holds(run)
                if batch2.status(run)['terminal']:
                    reason = 'original_run_terminal'; break
                if once: reason = 'once'; break
                publish('waiting')
                stop.wait(interval)
            return 0
        except BaseException as exc:
            reason, code = 'fatal', 1
            record['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:2000]}
            raise
        finally:
            cleanup_error = None
            try:
                recover_owned_holds(run)
            except BaseException as exc:
                cleanup_error = exc
                record['recovery_required'] = str(exc)[:2000]
                reason, code = 'transaction_recovery_required', 1
            record['exit'] = {'reason': reason, 'ended_at': time.time(), 'exit_code': code}
            publish('stopped')
            for sig, handler in previous.items(): signal.signal(sig, handler)
            if cleanup_error is not None:
                raise RuntimeError('Outstanding hold recovery requires attention: ' + str(cleanup_error)) from cleanup_error


def activate(args):
    run = args.run_dir.resolve()
    meta = batch2.load_batch(run)
    runtime = Path(meta['selection']['runtime'])
    folder = run / acceleration.TAG
    folder.mkdir(parents=True, exist_ok=True)
    if ROOT != folder / 'source':
        source = freeze_source(folder, ROOT / 'configs/suite.batch2.core.json',
                               ROOT / 'configs/cluster.orion.batch2.json', runtime)
        return subprocess.run([sys.executable, '-I', '-B', '-u',
            str(source / 'scripts/accelerate_batch2.py'), 'activate',
            '--run-dir', str(run), '--interval', str(args.interval)],
            env=clean_environment(runtime), check=False).returncode
    verify_snapshot(ROOT)
    guard = acquire_lock(folder / 'launch.lock')
    if guard is None: raise RuntimeError('Another accelerator activation is in progress')
    with guard:
        if manager_status(folder)['manager'] == 'ACTIVE':
            print_json({'activation': 'ALREADY_RUNNING', **current_status(run)}); return 0
        if batch2.status(run)['terminal']:
            print_json({'activation': 'NOT_NEEDED_RUN_TERMINAL', **batch2.status(run)}); return 0
        if manager_status(run)['manager'] != 'ACTIVE':
            raise RuntimeError('Original batch2 manager is not active; no jobs changed')
        acceleration.prepare(run, ROOT)
        lock = acquire_lock(folder / 'manager.lock')
        if lock is None: raise RuntimeError('Accelerator lock was acquired concurrently')
        with lock:
            stop_path = folder / 'stop.request.json'
            if stop_path.exists():
                stop_path.rename(folder / ('stop.previous-' + str(time.time_ns()) + '.json'))
            logfile = folder / 'controller.log'
            with logfile.open('a') as stream:
                child = subprocess.Popen([sys.executable, '-I', '-B', '-u',
                    str(ROOT / 'scripts/accelerate_batch2.py'), 'watch', '--run-dir', str(run),
                    '--interval', str(args.interval), '--lock-fd', str(lock.fileno())],
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    env=clean_environment(runtime), start_new_session=True, pass_fds=(lock.fileno(),))
            ready = False; deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError('Accelerator exited at startup; inspect ' + str(logfile))
                try: ready = load_json(folder / 'manager.json').get('pid') == child.pid
                except (OSError, ValueError): pass
                if ready: break
                time.sleep(.05)
            print_json({'activation': 'STARTED' if ready else 'STARTING', 'pid': child.pid,
                        'log': str(logfile), 'note': 'Startup is not proof that a new node passed.',
                        **current_status(run)})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('activate', 'watch'):
        p = sub.add_parser(name)
        p.add_argument('--run-dir', type=Path, required=True)
        p.add_argument('--interval', type=float, default=20)
        if name == 'watch':
            p.add_argument('--lock-fd', type=int)
            p.add_argument('--once', action='store_true')
    for name in ('status', 'stop'):
        p = sub.add_parser(name); p.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if hasattr(args, 'interval') and (not math.isfinite(args.interval) or args.interval < 10):
        parser.error('--interval must be finite and at least 10 seconds')
    if args.command == 'activate': return activate(args)
    if args.command == 'status': print_json(current_status(args.run_dir)); return 0
    if args.command == 'stop':
        batch2.load_batch(args.run_dir)
        folder = args.run_dir.resolve() / acceleration.TAG
        if not (folder / 'policy.json').is_file(): raise ValueError('No accelerator policy exists')
        atomic_json(folder / 'stop.request.json', {'requested_at': time.time(), 'by_uid': os.getuid()})
        print_json({'accelerator': 'STOP_REQUESTED', 'original_manager': 'UNCHANGED',
                    'note': 'Pending modifications already released remain valid; existing jobs continue.'})
        return 0
    if ROOT != args.run_dir.resolve() / acceleration.TAG / 'source':
        raise ValueError('The watcher must run from its own frozen acceleration source')
    verify_snapshot(ROOT)
    return watch(args.run_dir, args.interval, args.once, args.lock_fd)


if __name__ == '__main__':
    try: raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
        print('Acceleration error: ' + str(exc), file=sys.stderr, flush=True)
        print('The original manager and experiment outputs are preserved.', file=sys.stderr, flush=True)
        raise SystemExit(1)
