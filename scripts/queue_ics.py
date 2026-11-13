#!/usr/bin/env python3
"""Start/resume the bounded ICS Slurm queue; the manager does no model work."""
from __future__ import annotations

import argparse
import fcntl
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
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ics.orchestration import atomic_json, build_plan, load_json, write_plan


def clean_environment(runtime: Path) -> dict:
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                "ICS_DINOV3_URL"):
        env.pop(key, None)
    env.update(ICS_RUNTIME=str(runtime), PYTHONNOUSERSITE="1",
               HF_HOME=str(runtime / "cache" / "huggingface"),
               HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1",
               HF_DEBUG="0", HF_HUB_VERBOSITY="warning")
    return env


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def freeze_source(run_dir: Path, suite: Path, cluster: Path, runtime: Path) -> Path:
    """Publish one immutable operational source copy; resumes reuse that copy."""
    selection = {"suite": str(suite.resolve().relative_to(ROOT)),
                 "cluster": str(cluster.resolve().relative_to(ROOT)),
                 "runtime": str(runtime)}
    lock = acquire_lock(run_dir / "source.lock")
    if lock is None:
        raise RuntimeError("Another start is preparing this run; repeat the same start command")
    with lock:
        target = run_dir / "source"
        if target.exists():
            metadata = load_json(target / "snapshot.json")
            if metadata["selection"] != selection:
                raise ValueError("Run directory uses another runtime/suite/cluster; choose a new --run-dir")
            verify_snapshot(target)
            return target
        if (run_dir / "plan.json").exists():
            raise ValueError("Existing non-snapshot plan; choose a new --run-dir")
        temporary = Path(tempfile.mkdtemp(prefix="source-building-", dir=run_dir))
        try:
            paths = [ROOT / "ics_eval.py", suite.resolve(), cluster.resolve()]
            for directory in ("ics", "scripts", "configs"):
                paths.extend(p for p in (ROOT / directory).rglob("*")
                             if p.suffix in {".py", ".json"} and p.is_file()
                             and "__pycache__" not in p.parts
                             and not p.name.endswith(".local.json"))
            hashes = {}
            for path in sorted(set(paths)):
                if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                    raise ValueError(f"Cannot freeze a symlink: {path}")
                relative = path.relative_to(ROOT)
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
            for value in (selection["suite"], selection["cluster"]):
                if value not in hashes:
                    raise ValueError("Suite/cluster must be JSON files inside the project")
            atomic_json(temporary / "snapshot.json", {
                "schema_version": 1, "selection": selection, "created_at": time.time(),
                "files": hashes,
                "note": "Project code/config frozen; runtime environments, upstream sources and weights are external."})
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return target


def verify_snapshot(root: Path) -> None:
    for name, expected in load_json(root / "snapshot.json")["files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {name}; restore it before resuming")


def print_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def manager_status(run_dir: Path) -> dict:
    """Probe the existing flock, not a potentially recycled process ID."""
    path = run_dir / "manager.lock"
    if not path.exists():
        return {"manager": "NOT_STARTED"}
    with path.open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"manager": "ACTIVE"}
    return {"manager": "STOPPED"}


def diagnostic(message: str) -> None:
    """A failed diagnostic destination must never replace the original failure."""
    try:
        print(message, file=sys.stderr, flush=True)
    except Exception:
        pass


def saved_status(run_dir: Path) -> dict:
    """Describe saved queue state and current lock ownership without querying Slurm."""
    from ics.rolling_queue import queue_status
    summary = {**queue_status(run_dir / "plan.json"), **manager_status(run_dir)}
    now = time.time()
    updated = summary.get("updated_at")
    def valid_time(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    age = max(0.0, now - updated) if valid_time(updated) else None
    lifecycle = None
    metadata = run_dir / "manager.json"
    try:
        lifecycle = load_json(metadata)
        if not isinstance(lifecycle, dict):
            raise ValueError("manager.json is not an object")
    except FileNotFoundError:
        pass  # Older or abruptly stopped managers may have no lifecycle record.
    except (OSError, ValueError) as exc:
        summary["manager_lifecycle_error"] = f"{type(exc).__name__}: {exc}"
        lifecycle = None
    exit_record = lifecycle.get("exit") if lifecycle else None
    reason = None
    if isinstance(exit_record, dict):
        ended = exit_record.get("ended_at")
        known_reason = exit_record.get("reason") in ("signal", "terminal", "once", "fatal")
        comparable = valid_time(updated) and valid_time(ended)
        stale = comparable and ended < updated
        summary["manager_lifecycle_stale"] = stale
        if comparable and not stale and known_reason:
            reason = exit_record["reason"]
        else:
            summary["manager_lifecycle_note"] = (
                "The recorded exit predates the saved queue observation; retained as historical diagnostics, "
                "not the current manager's exit cause." if stale else
                "The recorded exit cannot be matched to the saved queue observation with valid timestamps; "
                "the current manager's exit cause is unknown.")
    summary.update(saved_queue_status=summary["status"], state_age_seconds=age,
                   state_is_live=False, scheduler_queried=False, status_read_at=now,
                   manager_lifecycle=lifecycle,
                   manager_exit_reason=(reason or "unknown") if summary["manager"] == "STOPPED" else None,
                   status_note="Task states are the last persisted queue observation; this command does not query Slurm.")
    if summary["manager"] == "STOPPED" and summary.get("terminal") is False:
        summary["status"] = "MANAGER_STOPPED"
        summary["status_note"] += " The manager is stopped; saved task states do not establish current Slurm job states."
        if reason == "fatal":
            summary["recovery"] = (
                "Inspect manager_lifecycle.exit and manager.log, and fix the recorded cause before resuming this run; "
                "do not reset task state or create a duplicate run.")
        else:
            summary["resume"] = "Repeat the same start command"
            summary["resume_note"] = (
                "Resume this same run so its first poll reconciles existing Slurm jobs before submitting more work; "
                "do not reset task state. An unrecorded manager exit cause remains unknown.")
    elif summary.get("terminal") and summary.get("status") == "COMPLETE_WITH_FAILURES":
        summary["recovery"] = (
            "Inspect failed tasks, fix the cause, and start a new --run-dir; "
            "repeating start does not retry terminal tasks.")
    return summary


def required_datasets(plan: dict, runtime: Path) -> tuple[str, ...]:
    datasets = set()
    for manifest in plan["manifests"].values():
        dataset = manifest.get("prepare", {}).get("dataset")
        if dataset not in {"coco", "lvis"}:
            raise ValueError("Rolling data preparation currently supports COCO/LVIS recipes only; "
                             "phase2 requires its documented manual data preparation")
        if Path(manifest["dataset_root"]).resolve() != runtime / "data":
            raise ValueError("Rolling data preparation requires dataset_root=${ICS_RUNTIME}/data")
        datasets.add(dataset)
    return tuple(sorted(datasets))


def watcher(run_dir: Path, *, interval: float = 60, once: bool = False,
            lock_fd: int | None = None, scheduling: bool = False) -> int:
    from ics.rolling_queue import tick
    lock = os.fdopen(lock_fd, "a+") if lock_fd is not None else acquire_lock(run_dir / "manager.lock")
    if lock is None:
        raise RuntimeError("A manager already holds this run's lock")
    stop = threading.Event()
    stop_signal = None
    old_handlers = {}
    def interrupted(signum, _frame):
        nonlocal stop_signal
        if stop_signal is None:
            stop_signal = signal.Signals(signum).name
        stop.set()

    started = time.time()
    lifecycle = {"schema_version": 2, "pid": os.getpid(), "hostname": socket.gethostname(),
                 "started_at": started, "source": str(ROOT), "interval": interval,
                 "phase": "initializing", "last_heartbeat_at": started,
                 "last_poll_started_at": None, "last_poll_completed_at": None, "exit": None}
    if scheduling:
        lifecycle['worker_source'] = str(run_dir / 'source')
        lifecycle['scheduling_policy'] = str(run_dir / 'scheduling' / 'policy.json')
    def publish(phase):
        lifecycle.update(phase=phase, last_heartbeat_at=time.time())
        try:
            atomic_json(run_dir / "manager.json", lifecycle)
        except Exception as exc:
            # This record is diagnostic, not queue state. For example, a full
            # filesystem must not mask the original poll/verification exception.
            diagnostic(f"Manager lifecycle record unavailable ({type(exc).__name__}): {exc}")

    reason, exit_code, summary, failure = "signal", 0, None, None
    with lock:
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.signal(sig, interrupted)
            publish("initializing")
            while not stop.is_set():
                publish("verifying_snapshot")
                verify_snapshot(ROOT)
                if stop.is_set():
                    break
                lifecycle["last_poll_started_at"] = time.time()
                publish("polling")
                if scheduling:
                    verify_snapshot(run_dir / 'source')
                    summary = tick(run_dir / "plan.json", stop_requested=stop.is_set, scheduling=True)
                else:
                    summary = tick(run_dir / "plan.json", stop_requested=stop.is_set)
                lifecycle["last_poll_completed_at"] = time.time()
                lifecycle["last_queue_status"] = summary["status"]
                publish("logging")
                print_json(summary)
                if summary["terminal"]:
                    reason = "terminal"
                    exit_code = 0 if summary["status"] == "COMPLETE" else 2
                    return exit_code
                if stop.is_set():
                    break
                if once:
                    reason = "once"
                    return 0
                publish("waiting")
                stop.wait(interval)
            return 0
        except BaseException as exc:
            reason, exit_code, failure = "fatal", 1, exc
            raise
        finally:
            lifecycle["exit"] = {"reason": reason, "ended_at": time.time(), "exit_code": exit_code,
                                 "phase": lifecycle["phase"]}
            if stop_signal:
                lifecycle["exit"]["signal"] = stop_signal
            if isinstance(summary, dict) and "status" in summary:
                lifecycle["exit"]["queue_status"] = summary["status"]
            if failure is not None:
                lifecycle["exit"].update(exception_type=type(failure).__name__, message=str(failure)[:2000])
            publish("stopped")
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def start(args) -> int:
    from ics.rolling_queue import initialize_queue, queue_status
    run_dir, runtime = args.run_dir.resolve(), args.runtime.resolve()
    if (run_dir / "plan.json").exists() and load_json(run_dir / "plan.json").get("recovery"):
        raise ValueError("Selective recovery plan: use scripts/recover_ics.py start --run-dir instead")
    env = clean_environment(runtime)
    os.environ.clear()
    os.environ.update(env)
    if ROOT != run_dir / "source":
        suite = args.suite if args.suite.is_absolute() else ROOT / args.suite
        cluster = args.cluster if args.cluster.is_absolute() else ROOT / args.cluster
        snapshot = freeze_source(run_dir, suite, cluster, runtime)
        selection = load_json(snapshot / "snapshot.json")["selection"]
        cmd = [sys.executable, "-u", str(snapshot / "scripts" / "queue_ics.py"), "start",
               "--runtime", str(runtime), "--run-dir", str(run_dir),
               "--suite", str(snapshot / selection["suite"]),
               "--cluster", str(snapshot / selection["cluster"]),
               "--max-inflight", str(args.max_inflight), "--shared-limit", str(args.shared_limit),
               "--reserved-slots", str(args.reserved_slots), "--data-workers", str(args.data_workers),
               "--interval", str(args.interval)]
        return subprocess.run(cmd, env=env, check=False).returncode
    verify_snapshot(ROOT)
    args.suite = args.suite if args.suite.is_absolute() else ROOT / args.suite
    args.cluster = args.cluster if args.cluster.is_absolute() else ROOT / args.cluster
    selection = load_json(ROOT / "snapshot.json")["selection"]
    if (args.suite.resolve() != ROOT / selection["suite"] or
            args.cluster.resolve() != ROOT / selection["cluster"] or
            str(runtime) != selection["runtime"]):
        raise ValueError("Frozen run selection cannot be changed; use its original start command")
    lock = acquire_lock(run_dir / "manager.lock")
    if lock is None:
        if (run_dir / "plan.json").exists() and (run_dir / "rolling_state.json").exists():
            print_json({"manager": "ALREADY_RUNNING", **queue_status(run_dir / "plan.json")})
        else:
            print_json({"manager": "ALREADY_STARTING", "run_dir": str(run_dir)})
        return 0
    with lock:
        plan = build_plan(args.suite, args.cluster, run_dir)
        datasets = required_datasets(plan, runtime)
        # Runtime is explicit in the command and retained for credential-free restarts.
        plan_path = write_plan(plan)
        initialize_queue(plan_path, max_inflight=args.max_inflight,
                         shared_limit=args.shared_limit, reserved_slots=args.reserved_slots,
                         datasets=datasets, data_workers=args.data_workers)
        summary = queue_status(plan_path)
        if summary["terminal"]:
            print_json({"manager": "NOT_NEEDED", **summary})
            return 0 if summary["status"] == "COMPLETE" else 2
        for executable in ("squeue", "sacct", "sbatch"):
            if not shutil.which(executable):
                raise RuntimeError(f"{executable} is unavailable; run start on the Slurm login node")
        log_path = run_dir / "manager.log"
        with log_path.open("a") as log:
            proc = subprocess.Popen([
                sys.executable, "-u", str(ROOT / "scripts" / "queue_ics.py"), "watch",
                "--run-dir", str(run_dir), "--interval", str(args.interval),
                "--lock-fd", str(lock.fileno())],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=env, start_new_session=True, pass_fds=(lock.fileno(),))
        ready = False
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Manager exited during startup (exit={proc.returncode}); see {log_path}")
            try:
                record = load_json(run_dir / "manager.json")
            except (OSError, ValueError):
                record = None  # Old/bad diagnostic metadata is not a child failure.
            if isinstance(record, dict) and record.get("pid") == proc.pid:
                ready = True
                break
            time.sleep(0.05)
        print_json({"manager": "STARTED" if ready else "STARTING", "manager_pid": proc.pid,
                    "manager_log": str(log_path), **summary})
    # Closing the parent's descriptor does not release the child's inherited flock.
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("start", help="Freeze sources, prepare the full queue, and start/resume the manager")
    p.add_argument("--runtime", type=Path, default=Path(os.environ["ICS_RUNTIME"]) if os.environ.get("ICS_RUNTIME") else None)
    p.add_argument("--suite", type=Path, default=Path("configs/suite.initial.json"))
    p.add_argument("--cluster", type=Path, default=Path("configs/cluster.orion.json"))
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--max-inflight", type=int, default=8)
    p.add_argument("--shared-limit", type=int, default=60)
    p.add_argument("--reserved-slots", type=int, default=0)
    p.add_argument("--data-workers", type=int, default=2)
    p.add_argument("--interval", type=float, default=60)
    p = sub.add_parser("watch", help="Run manager in foreground, or one poll with --once")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--once", action="store_true")
    p.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    p = sub.add_parser("status", help="Read saved state; does not query or submit to Slurm")
    p.add_argument("--run-dir", type=Path, required=True)
    p = sub.add_parser("worker", help=argparse.SUPPRESS)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--key", required=True)
    args = parser.parse_args(argv)
    if hasattr(args, "interval") and (not math.isfinite(args.interval) or args.interval < 10):
        parser.error("--interval must be finite and at least 10 seconds")
    if args.command == "start" and args.runtime is None:
        parser.error("start requires --runtime or ICS_RUNTIME")
    if args.command == "start":
        return start(args)
    if args.command == "status":
        run_dir = args.run_dir.resolve()
        print_json(saved_status(run_dir))
        return 0
    if args.command == "watch":
        run_dir = args.run_dir.resolve()
        snapshot = run_dir / "source"
        metadata = load_json(snapshot / "snapshot.json")
        env = clean_environment(Path(metadata["selection"]["runtime"]))
        if ROOT != snapshot:
            cmd = [sys.executable, "-u", str(snapshot / "scripts" / "queue_ics.py"),
                   "watch", "--run-dir", str(run_dir), "--interval", str(args.interval)]
            if args.once:
                cmd.append("--once")
            return subprocess.run(cmd, env=env, check=False).returncode
        os.environ.clear()
        os.environ.update(env)
        return watcher(run_dir, interval=args.interval, once=args.once, lock_fd=args.lock_fd)
    from ics.rolling_queue import execute_queue_task
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Termination signal {signum}; closing owned processes")
    signal.signal(signal.SIGTERM, interrupted)
    if (ROOT / "snapshot.json").exists():
        verify_snapshot(ROOT)
    execute_queue_task(args.plan.resolve(), args.key)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        diagnostic(f"Queue error: {exc}")
        raise SystemExit(1)
