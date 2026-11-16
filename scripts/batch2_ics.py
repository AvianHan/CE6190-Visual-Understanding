#!/usr/bin/env python3
"""Prepare/start/status for the unattended, independently gated second batch."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ics import batch2
from ics.orchestration import atomic_json, load_json
from scripts.queue_ics import (acquire_lock, clean_environment, diagnostic,
                               freeze_source, manager_status, print_json,
                               verify_snapshot)


def current_status(run):
    value = {**batch2.status(run), **manager_status(run)}
    if value['manager'] == 'STOPPED' and not value['terminal']:
        value['batch2_status'] = 'MANAGER_STOPPED'
    return value


def watch(run, interval=30, once=False, lock_fd=None, tick_fn=None):
    run = Path(run).resolve()
    lock = os.fdopen(lock_fd, 'a+') if lock_fd is not None else acquire_lock(run / 'manager.lock')
    if lock is None: raise RuntimeError('A batch2 manager already holds the lock')
    stop = threading.Event()
    old = {}
    lifecycle = {'schema_version': 1, 'pid': os.getpid(), 'hostname': socket.gethostname(),
                 'source': str(ROOT), 'started_at': time.time(), 'interval': interval,
                 'phase': 'initializing', 'exit': None}
    def publish(phase):
        lifecycle.update(phase=phase, last_heartbeat_at=time.time())
        atomic_json(run / 'manager.json', lifecycle)
    def interrupted(signum, frame):
        lifecycle['stop_signal'] = signal.Signals(signum).name
        stop.set()
    reason, code = 'signal', 0
    with lock:
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                old[sig] = signal.signal(sig, interrupted)
            publish('initializing')
            while not stop.is_set():
                verify_snapshot(ROOT)
                publish('polling')
                lifecycle['last_poll_started_at'] = time.time()
                state = (tick_fn or batch2.tick_batch)(run, stop_requested=stop.is_set)
                lifecycle.update(last_poll_completed_at=time.time(),
                                 last_queue_status=state['batch2_status'])
                print_json(state)
                if state['terminal']:
                    reason = 'terminal'
                    code = 0 if state['batch2_status'] == 'COMPLETE' else 2
                    return code
                if once:
                    reason = 'once'; return 0
                publish('waiting')
                stop.wait(interval)
            return 0
        except BaseException as exc:
            reason, code = 'fatal', 1
            lifecycle['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:2000]}
            raise
        finally:
            lifecycle['exit'] = {'reason': reason, 'ended_at': time.time(), 'exit_code': code}
            publish('stopped')
            for sig, handler in old.items(): signal.signal(sig, handler)


def prepare_or_start(args):
    run, runtime = args.run_dir.resolve(), args.runtime.resolve()
    request = batch2.selection(runtime, run, args.prior_run)
    run.mkdir(parents=True, exist_ok=True)
    env = clean_environment(runtime)
    env['ICS_BATCH2_ROOT'] = str(run)
    os.environ.clear(); os.environ.update(env)
    if ROOT != run / 'source':
        source = freeze_source(run, ROOT / 'configs/suite.batch2.core.json',
                               ROOT / 'configs/cluster.orion.batch2.json', runtime)
        cmd = [sys.executable, '-I', '-B', '-u', str(source / 'scripts/batch2_ics.py'), args.command,
               '--runtime', str(runtime), '--run-dir', str(run), '--interval', str(args.interval)]
        for prior in request['prior_runs']: cmd.extend(['--prior-run', prior])
        return subprocess.run(cmd, env=env, check=False).returncode
    verify_snapshot(ROOT)
    lock = acquire_lock(run / 'manager.lock')
    if lock is None:
        print_json({'manager': 'ALREADY_RUNNING', **batch2.status(run)})
        return 0
    with lock:
        result = batch2.prepare(ROOT, run, request)
        if args.command == 'prepare':
            print_json({'manager': 'PREPARED_NOT_STARTED', **result}); return 0
        if result['terminal']:
            print_json({'manager': 'NOT_NEEDED', **result})
            return 0 if result['batch2_status'] == 'COMPLETE' else 2
        for binary in ('squeue', 'sacct', 'sbatch'):
            if not shutil.which(binary): raise RuntimeError(binary + ' is unavailable; start on Orion login node')
        log_path = run / 'manager.log'
        with log_path.open('a') as log:
            proc = subprocess.Popen([sys.executable, '-I', '-B', '-u',
                str(ROOT / 'scripts/batch2_ics.py'), 'watch', '--run-dir', str(run),
                '--interval', str(args.interval), '--lock-fd', str(lock.fileno())],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=env, start_new_session=True, pass_fds=(lock.fileno(),))
        ready = False
        end = time.monotonic() + 3
        while time.monotonic() < end:
            if proc.poll() is not None:
                raise RuntimeError('Batch2 manager exited at startup; inspect ' + str(log_path))
            try: item = load_json(run / 'manager.json')
            except (OSError, ValueError): item = {}
            if item.get('pid') == proc.pid: ready = True; break
            time.sleep(.05)
        print_json({'manager': 'STARTED' if ready else 'STARTING', 'manager_pid': proc.pid,
                    'manager_log': str(log_path), **result})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'start'):
        p = sub.add_parser(command)
        p.add_argument('--run-dir', type=Path, required=True)
        p.add_argument('--runtime', type=Path, required=True)
        p.add_argument('--prior-run', type=Path, action='append')
        p.add_argument('--interval', type=float, default=30)
    p = sub.add_parser('status'); p.add_argument('--run-dir', type=Path, required=True)
    p = sub.add_parser('watch'); p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--interval', type=float, default=30); p.add_argument('--once', action='store_true')
    p.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    p = sub.add_parser('worker', help=argparse.SUPPRESS)
    p.add_argument('--intent', type=Path, required=True); p.add_argument('--intent-sha256', required=True)
    args = parser.parse_args(argv)
    if hasattr(args, 'interval') and (not math.isfinite(args.interval) or args.interval < 10):
        parser.error('--interval must be finite and at least 10 seconds')
    if args.command in ('prepare', 'start'): return prepare_or_start(args)
    if args.command == 'worker':
        batch2.execute_intent(args.intent, args.intent_sha256); return 0
    run = args.run_dir.resolve()
    if args.command == 'status': print_json(current_status(run)); return 0
    metadata = batch2.load_batch(run)
    source = Path(metadata['source'])
    if ROOT != source:
        if args.lock_fd is not None: raise ValueError('Lock descriptor requires direct frozen-source watcher')
        command = [sys.executable, '-I', '-B', '-u', str(source / 'scripts/batch2_ics.py'),
                   'watch', '--run-dir', str(run), '--interval', str(args.interval)]
        if args.once: command.append('--once')
        return subprocess.run(command, check=False).returncode
    env = clean_environment(Path(metadata['selection']['runtime']))
    env['ICS_BATCH2_ROOT'] = str(run)
    os.environ.clear(); os.environ.update(env)
    return watch(run, args.interval, args.once, args.lock_fd)


if __name__ == '__main__':
    try: raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        diagnostic('Batch2 error: ' + str(exc)); raise SystemExit(1)
