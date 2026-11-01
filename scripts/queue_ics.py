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
    return {"manager": "STOPPED", "resume": "Repeat the same start command"}


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
            lock_fd: int | None = None) -> int:
    from ics.rolling_queue import tick
    lock = os.fdopen(lock_fd, "a+") if lock_fd is not None else acquire_lock(run_dir / "manager.lock")
    if lock is None:
        raise RuntimeError("A manager already holds this run's lock")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    with lock:
        verify_snapshot(ROOT)
        atomic_json(run_dir / "manager.json", {"pid": os.getpid(), "started_at": time.time(),
                                               "source": str(ROOT), "interval": interval})
        while not stop.is_set():
            verify_snapshot(ROOT)
            summary = tick(run_dir / "plan.json")
            print_json(summary)
            if summary["terminal"]:
                return 0 if summary["status"] == "COMPLETE" else 2
            if once:
                return 0
            stop.wait(interval)
    return 0


def start(args) -> int:
    from ics.rolling_queue import initialize_queue, queue_status
    run_dir, runtime = args.run_dir.resolve(), args.runtime.resolve()
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
            if (run_dir / "manager.json").exists():
                if load_json(run_dir / "manager.json").get("pid") == proc.pid:
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
        from ics.rolling_queue import queue_status
        run_dir = args.run_dir.resolve()
        print_json({**queue_status(run_dir / "plan.json"), **manager_status(run_dir)})
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
        print(f"Queue error: {exc}", file=sys.stderr)
        raise SystemExit(1)
