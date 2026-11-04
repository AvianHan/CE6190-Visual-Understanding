"""Restartable, quota-aware Slurm submission; no computation runs in the manager.

All data preparation, inference and collection execute in separately allocated
Slurm jobs. Unknown submission acknowledgements are never blindly retried.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import signal
import subprocess
import tempfile
import time

from .orchestration import (PROJECT_ROOT, collect, fingerprint, load_json,
                            resource_profile, run_preflight, run_task,
                            safe_id, sbatch_args)

TERMINAL = {"PASS", "FAILED", "BLOCKED"}
ACTIVE = {"SUBMITTING", "SUBMITTED", "RUNNING", "UNCERTAIN"}
SLURM_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "PREEMPTED",
                  "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "REVOKED",
                  "SPECIAL_EXIT"}
STATE_NAME = "rolling_state.json"
MARKER_GRACE_SECONDS = 60


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, indent=2, ensure_ascii=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _plan(path: Path) -> dict:
    plan = load_json(Path(path))
    expected = plan.get("fingerprint")
    body = {k: v for k, v in plan.items() if k != "fingerprint"}
    if not expected or fingerprint(body) != expected:
        raise ValueError("Plan fingerprint does not match its contents")
    return plan


@contextmanager
def _lock(run: Path):
    run.mkdir(parents=True, exist_ok=True)
    with (run / "rolling.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another rolling-queue operation is active") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _positive(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _profile(plan: dict, task: dict) -> dict:
    return (resource_profile(plan["cluster"], "preflight") if task["resource"] == "preflight"
            else plan["resources"][task["resource"]])


def _task_table(plan: dict, queue_id: str) -> dict:
    tasks = {}

    def add(key, kind, resource, dependencies, index=None):
        safe_id(key)
        if key in tasks:
            raise ValueError(f"Duplicate rolling task key: {key}")
        token = hashlib.sha256(key.encode()).hexdigest()[:16]
        tasks[key] = {"key": key, "kind": kind, "resource": resource,
                      "dependencies": dependencies, "index": index,
                      "job_name": f"icsq-{queue_id}-{token}", "status": "PENDING",
                      "attempts": 0}

    add("data", "data", "preflight", [])
    add("preflight", "preflight", "preflight", ["data"])
    smoke_by_config = {}
    indices = Counter()
    for task in plan["smoke"]:
        resource = task["resource"]
        add(task["id"], "smoke", resource, ["preflight"], indices[resource])
        indices[resource] += 1
        smoke_by_config[task["config_data"]["experiment_id"]] = task["id"]
    indices = Counter()
    for task in plan["tasks"]:
        resource = task["resource"]
        cid = task["config_data"]["experiment_id"]
        if cid not in smoke_by_config:
            raise ValueError(f"No smoke task for {cid}")
        add(task["id"], "experiment", resource, [smoke_by_config[cid]], indices[resource])
        indices[resource] += 1
    add("collect", "collect", "preflight", list(tasks))
    return tasks


def initialize_queue(plan_path: Path, *, max_inflight=8, shared_limit=60,
                     reserved_slots=0, datasets=("coco", "lvis"), data_workers=2) -> dict:
    """Create/reopen an identical queue. This never submits or executes a job."""
    plan_path = Path(plan_path).resolve()
    plan = _plan(plan_path)
    run = Path(plan["run_dir"])
    _positive(max_inflight, "max_inflight")
    _positive(shared_limit, "shared_limit")
    _positive(data_workers, "data_workers")
    if type(reserved_slots) is not int or not 0 <= reserved_slots < shared_limit:
        raise ValueError("reserved_slots must be between zero and shared_limit - 1")
    if max_inflight > shared_limit - reserved_slots:
        raise ValueError("max_inflight exceeds the usable shared submission limit")
    allowed = {"coco", "lvis", "fss", "pascal", "isic", "suim", "isaid"}
    datasets = list(dict.fromkeys(datasets))
    if not datasets or any(x not in allowed for x in datasets):
        raise ValueError("Pass explicit supported datasets")
    roots = {m["dataset_root"] for m in plan["manifests"].values()}
    if len(roots) != 1:
        raise ValueError("Rolling data preparation requires one common dataset_root")
    needed = {m.get("prepare", {}).get("dataset") for m in plan["manifests"].values()}
    if needed - {None} - set(datasets):
        raise ValueError("Download datasets do not cover all manifest recipes")
    profiles = [resource_profile(plan["cluster"], "preflight"), *plan["resources"].values()]
    qos = {p.get("qos") for p in profiles}
    if len(qos) != 1 or not next(iter(qos)):
        raise ValueError("All rolling profiles need the same explicit QOS")
    safe_id(next(iter(qos)))
    for profile in profiles:
        sbatch_args(profile, name="ics-validation", output=run / "logs" / "validation.log")
    for name, profile in plan["resources"].items():
        _positive(profile.get("max_parallel", plan["cluster"].get("max_parallel")),
                  f"max_parallel for {name}")
    policy = {"max_inflight": max_inflight, "shared_limit": shared_limit,
              "reserved_slots": reserved_slots, "datasets": datasets,
              "data_workers": data_workers, "min_free_gib": 32,
              "data_root": next(iter(roots)), "qos": next(iter(qos)),
              "user": pwd.getpwuid(os.getuid()).pw_name}
    with _lock(run):
        if (run / "submission.json").exists() or (run / "submission.lock").exists():
            raise ValueError("Existing static submission found; use a new run directory")
        path = run / STATE_NAME
        if path.exists():
            state = load_json(path)
            if state["plan_fingerprint"] != plan["fingerprint"] or state["policy"] != policy:
                raise ValueError("Existing rolling queue has a different plan or policy")
            return state
        if any((run / name).exists() for name in ("rolling_slurm", "rolling_markers")):
            raise ValueError("Previous rolling work exists but its state is missing; restore state, do not resubmit")
        now = time.time()
        queue_id = plan["fingerprint"][:16]
        state = {"schema_version": 1, "plan_path": str(plan_path),
                 "plan_fingerprint": plan["fingerprint"], "queue_id": queue_id,
                 "created_at": now, "updated_at": now, "policy": policy,
                 "tasks": _task_table(plan, queue_id), "last_error": None,
                 "capacity": None}
        _atomic_json(path, state)
        return state


def _read_state(plan: dict) -> dict:
    path = Path(plan["run_dir"]) / STATE_NAME
    if not path.exists():
        raise FileNotFoundError("Initialize the rolling queue first")
    state = load_json(path)
    if state.get("plan_fingerprint") != plan["fingerprint"]:
        raise ValueError("Rolling state belongs to another plan")
    return state


def _command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"{args[0]} query failed (exit={result.returncode}): {result.stderr.strip()[:800]}")
    return result.stdout


def _scheduler_snapshot(state: dict) -> tuple[dict, dict]:
    """Expanded queue is shared across accounts; accounting excludes step rows."""
    user = state["policy"]["user"]
    queued = _command(["squeue", "--noheader", "--all", "--array", "--user", user,
                       "--format", "%i|%j|%T|%q"])
    start = dt.datetime.fromtimestamp(state["created_at"] - 120).strftime("%Y-%m-%dT%H:%M:%S")
    accounted = _command(["sacct", "--noheader", "--parsable2", "--allocations",
                          "--user", user, "--starttime", start,
                          "--format", "JobIDRaw,JobName%128,State%40,ExitCode,QOS"])
    queue, history = {}, {}
    for line in queued.splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split("|")]
        if len(parts) != 4 or not re.fullmatch(r"\d+(?:_\d+)?(?:\+\d+)?", parts[0]) or not all(parts):
            raise ValueError("Malformed or unexpanded squeue output; refusing submissions")
        jid, name, status, qos = parts
        if jid in queue:
            raise ValueError("Duplicate squeue task id")
        queue[jid] = {"job_id": jid, "job_name": name, "state": status, "qos": qos}
    for line in accounted.splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split("|")]
        if len(parts) == 6 and parts[-1] == "":
            parts.pop()
        if len(parts) != 5:
            raise ValueError("Malformed sacct output; refusing submissions")
        jid, name, status, exit_code, qos = parts
        if "." in jid:  # Defensive: never use batch/extern step success.
            continue
        if not re.fullmatch(r"\d+(?:_\d+)?(?:\+\d+)?", jid) or not name or not status:
            raise ValueError("Malformed sacct allocation record")
        row = {"job_id": jid, "job_name": name, "state": status.split()[0].rstrip("+"),
               "exit_code": exit_code, "qos": qos}
        if jid in history and history[jid] != row:
            raise ValueError("Conflicting accounting records; refusing submissions")
        history[jid] = row
    return queue, history


def _marker_path(plan: dict, key: str) -> Path:
    return Path(plan["run_dir"]) / "rolling_markers" / (safe_id(key) + ".json")


def _marker_problem(plan: dict, task: dict) -> str | None:
    path = _marker_path(plan, task["key"])
    if not path.exists():
        return "missing"
    try:
        marker = load_json(path)
    except (OSError, ValueError):
        return "Unreadable completion marker"
    expected = {"status": "PASS", "plan_fingerprint": plan["fingerprint"],
                "key": task["key"], "slurm_job_id": task["job_id"]}
    if any(marker.get(k) != v for k, v in expected.items()):
        return "Completion marker does not match this task, plan and Slurm job"
    if task["kind"] in {"smoke", "experiment"}:
        path = Path(plan["run_dir"]) / "jobs" / task["key"] / "complete.json"
        try:
            original = load_json(path)
        except (OSError, ValueError):
            return "Missing or unreadable model-worker completion marker"
        if any(original.get(k) != v for k, v in {
            "status": "PASS", "plan_fingerprint": plan["fingerprint"],
            "task_id": task["key"], "slurm_job_id": task["job_id"]}.items()):
            return "Model-worker completion marker mismatch"
    return None


def _reconcile(plan: dict, state: dict, queue: dict, history: dict, now: float):
    records = dict(history)
    records.update(queue)
    for task in state["tasks"].values():
        if task["status"] not in ACTIVE:
            continue
        if not task.get("job_id"):
            matches = {jid for jid, r in records.items()
                       if r["job_name"] == task["job_name"] and r["qos"] == state["policy"]["qos"]}
            if len(matches) == 1:
                task["job_id"] = matches.pop()
                task["recovered_at"] = now
            else:
                task["status"] = "UNCERTAIN"
                task["error"] = ("Multiple jobs have this submission name; operator review required"
                                 if matches else "Submission acknowledgement unknown; waiting for scheduler evidence")
                continue
        jid = task["job_id"]
        if jid in queue:
            record = queue[jid]
            if record["job_name"] != task["job_name"] or record["qos"] != state["policy"]["qos"]:
                task.update(status="UNCERTAIN", error="Scheduler identity mismatch")
                continue
            task["scheduler_state"] = record["state"]
            task["status"] = "RUNNING" if record["state"] == "RUNNING" else "SUBMITTED"
            task.pop("error", None)
            continue
        record = history.get(jid)
        if not record:
            task.update(status="UNCERTAIN", error="Job absent from queue; awaiting accounting record")
            continue
        if record["job_name"] != task["job_name"] or record["qos"] != state["policy"]["qos"]:
            task.update(status="UNCERTAIN", error="Accounting identity mismatch")
            continue
        task["scheduler_state"] = record["state"]
        task["exit_code"] = record["exit_code"]
        if record["state"] not in SLURM_TERMINAL:
            task["status"] = "SUBMITTED"
            continue
        if record["state"] != "COMPLETED" or record["exit_code"] != "0:0":
            task.update(status="FAILED", ended_at=now,
                        error=f"Slurm {record['state']} exit {record['exit_code']}")
            continue
        problem = _marker_problem(plan, task)
        if problem == "missing":
            first = task.setdefault("terminal_seen_at", now)
            if now - first < MARKER_GRACE_SECONDS:
                task.update(status="SUBMITTED", error="Awaiting completion marker visibility")
                continue
        if problem:
            task.update(status="FAILED", ended_at=now, error=problem)
        else:
            task.update(status="PASS", ended_at=now)
            task.pop("error", None)


def _propagate_blocks(state: dict):
    changed = True
    while changed:
        changed = False
        for task in state["tasks"].values():
            if task["kind"] == "collect" or task["status"] not in {"PENDING", "WAITING_QUOTA"}:
                continue
            bad = [key for key in task["dependencies"]
                   if state["tasks"][key]["status"] in {"FAILED", "BLOCKED"}]
            if bad:
                task.update(status="BLOCKED", error="Failed prerequisites: " + ", ".join(bad))
                changed = True


def _ready(state, task, now):
    if task["status"] not in {"PENDING", "WAITING_QUOTA"} or task.get("retry_after", 0) > now:
        return False
    statuses = [state["tasks"][key]["status"] for key in task["dependencies"]]
    return (all(s in TERMINAL for s in statuses) if task["kind"] == "collect"
            else all(s == "PASS" for s in statuses))


def _submission_error(stderr: str) -> str:
    text = stderr.lower()
    if any(x in text for x in ("qosmaxsubmitjobperuserlimit", "assocmaxsubmitjoblimit",
                              "maximum number of submitted jobs", "maxsubmitjob",
                              "batch job submission failed: job violates accounting/qos policy")):
        return "quota"
    if any(x in text for x in ("invalid account", "invalid partition", "invalid qos",
                              "invalid time limit", "requested node configuration is not available",
                              "memory specification can not be satisfied", "access denied",
                              "permission denied")):
        return "rejected"
    return "uncertain"


def _submit(plan_path: Path, plan: dict, state: dict, task: dict, now: float):
    run = Path(plan["run_dir"])
    script = run / "rolling_slurm" / (task["key"] + ".sh")
    script.parent.mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)
    launcher = plan["cluster"].get("launcher_python", "python3")
    command = [launcher, str(PROJECT_ROOT / "scripts" / "queue_ics.py"), "worker",
               "--plan", str(plan_path), "--key", task["key"]]
    header = ("#!/usr/bin/env bash\nset -euo pipefail\n"
              "unset HF_TOKEN HUGGING_FACE_HUB_TOKEN ICS_DINOV3_URL PYTHONPATH PYTHONHOME\n"
              "export PYTHONNOUSERSITE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 HF_DEBUG=0\n")
    if task["kind"] in {"smoke", "experiment"}:
        header += "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
    script.write_text(header + "exec " + shlex.join(command) + "\n")
    sbatch = sbatch_args(_profile(plan, task), name=task["job_name"],
                         output=run / "logs" / (task["key"] + "-%j.log")) + [str(script)]
    # Durable intent must precede the side effect, including on a process crash.
    task.update(status="SUBMITTING", attempt_started_at=now, attempts=task["attempts"] + 1)
    task.pop("error", None)
    state["updated_at"] = now
    _atomic_json(run / STATE_NAME, state)
    try:
        result = subprocess.run(sbatch, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        task.update(status="FAILED", error=f"Submission executable unavailable: {exc.filename}")
    except (OSError, subprocess.SubprocessError) as exc:
        task.update(status="UNCERTAIN", error=f"Submission acknowledgement unknown ({type(exc).__name__})")
    else:
        ack = result.stdout.strip().split(";")[0]
        if result.returncode == 0 and re.fullmatch(r"\d+", ack):
            task.update(status="SUBMITTED", job_id=ack, submitted_at=now)
        elif result.returncode == 0:
            task.update(status="UNCERTAIN", error="sbatch succeeded but acknowledgement could not be parsed")
        else:
            kind = _submission_error(result.stderr)
            message = result.stderr.strip()[:1500] or f"sbatch exit {result.returncode}"
            if kind == "quota":
                delay = min(300, 15 * 2 ** min(task["attempts"], 5))
                task.update(status="WAITING_QUOTA", error=message, retry_after=now + delay)
            elif kind == "rejected":
                task.update(status="FAILED", error=message, ended_at=now)
            else:
                task.update(status="UNCERTAIN", error="Uncertain submission: " + message)
    _atomic_json(run / STATE_NAME, state)


def tick(plan_path: Path) -> dict:
    """Reconcile and submit one bounded window; return without sleeping."""
    plan_path = Path(plan_path).resolve()
    plan = _plan(plan_path)
    run = Path(plan["run_dir"])
    with _lock(run):
        state = _read_state(plan)
        now = time.time()
        if (run / "submission.json").exists() or (run / "submission.lock").exists():
            state.update(last_error="Static submission appeared in this run; refusing mixed launchers", updated_at=now)
            _atomic_json(run / STATE_NAME, state)
            return _summary(plan, state)
        try:
            queue, history = _scheduler_snapshot(state)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            state.update(last_error=str(exc), updated_at=now)
            _atomic_json(run / STATE_NAME, state)
            return _summary(plan, state)
        state["last_error"] = None
        _reconcile(plan, state, queue, history, now)
        _propagate_blocks(state)
        policy = state["policy"]
        shared = sum(r["qos"] == policy["qos"] for r in queue.values())
        active = [t for t in state["tasks"].values() if t["status"] in ACTIVE]
        # Reserve accepted/intended work not yet visible in this scheduler snapshot.
        invisible = sum(not t.get("job_id") or t["job_id"] not in queue for t in active)
        remaining = max(0, min(policy["max_inflight"] - len(active),
                               policy["shared_limit"] - policy["reserved_slots"] - shared - invisible))
        state["capacity"] = {"shared_observed": shared, "unseen_own_reservations": invisible,
                             "own_active": len(active), "available_before_submission": remaining,
                             "observed_at": now}
        by_resource = Counter(t["resource"] for t in active)
        for task in state["tasks"].values():
            if remaining <= 0:
                break
            if not _ready(state, task, now):
                continue
            resource = task["resource"]
            if resource != "preflight":
                cap = plan["resources"][resource].get("max_parallel", plan["cluster"].get("max_parallel"))
                if by_resource[resource] >= cap:
                    continue
            _submit(plan_path, plan, state, task, now)
            if task["status"] in ACTIVE:
                remaining -= 1
                by_resource[resource] += 1
            elif task["status"] == "WAITING_QUOTA":
                break  # A competing submitter consumed a slot; back off globally.
            _propagate_blocks(state)
        state["updated_at"] = time.time()
        _atomic_json(run / STATE_NAME, state)
        return _summary(plan, state)


def _summary(plan: dict, state: dict) -> dict:
    tasks = list(state["tasks"].values())
    counts = dict(Counter(t["status"] for t in tasks))
    stage_counts = {kind: Counter() for kind in
                    ("data", "preflight", "smoke", "experiment", "collect")}
    for task in tasks:
        stage_counts.setdefault(task["kind"], Counter())[task["status"]] += 1
    attention_priority = {"FAILED": 0, "UNCERTAIN": 1, "WAITING_QUOTA": 2, "BLOCKED": 3}
    attention = sorted((t for t in tasks if t["status"] in attention_priority),
                       key=lambda t: attention_priority[t["status"]])
    # Summarize the root failures first without dropping any persistent task state.
    attention_preview = [{key: t.get(key) for key in
                          ("key", "kind", "resource", "job_id", "status", "error")}
                         for t in attention[:10]]
    terminal = all(t["status"] in TERMINAL for t in tasks)
    active = sum(counts.get(s, 0) for s in ACTIVE)
    if terminal:
        status = "COMPLETE_WITH_FAILURES" if counts.get("FAILED") or counts.get("BLOCKED") else "COMPLETE"
    elif state.get("last_error") or counts.get("UNCERTAIN"):
        status = "PAUSED"
    elif active:
        status = "RUNNING"
    elif counts.get("WAITING_QUOTA"):
        status = "WAITING_CAPACITY"
    elif (state.get("capacity") or {}).get("available_before_submission") == 0:
        status = "WAITING_CAPACITY"
    else:
        status = "READY"
    return {"status": status, "terminal": terminal, "total": len(state["tasks"]),
            "active": active, "counts": counts,
            "stage_counts": {kind: dict(values) for kind, values in stage_counts.items()},
            "submitted_ever": sum(bool(t.get("job_id")) for t in tasks),
            "plan_path": state["plan_path"], "plan_fingerprint": state["plan_fingerprint"],
            "queue_state_path": str(Path(plan["run_dir"]) / STATE_NAME),
            "policy": state["policy"], "capacity": state.get("capacity"),
            "last_error": state.get("last_error"), "updated_at": state["updated_at"],
            "attention": attention_preview, "attention_total": len(attention),
            "attention_truncated": len(attention) > len(attention_preview)}


def queue_status(plan_path: Path) -> dict:
    """Read only. Does not contact Slurm or alter task state."""
    plan = _plan(Path(plan_path))
    return _summary(plan, _read_state(plan))


def execute_queue_task(plan_path: Path, key: str) -> None:
    """Allocated worker entrypoint; writes job-bound evidence after cleanup."""
    plan = _plan(Path(plan_path))
    state = _read_state(plan)
    task = state["tasks"][key]
    if task["status"] not in ACTIVE:
        raise RuntimeError("Worker does not have an active submission intent")
    jid = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"\d+", jid):
        raise RuntimeError("Rolling workers require a Slurm allocation; refusing login-node execution")
    if task.get("job_id") and task["job_id"] != jid:
        raise RuntimeError("Worker Slurm job id differs from recorded submission")
    if os.environ.get("SLURM_JOB_NAME") != task["job_name"]:
        raise RuntimeError("Worker Slurm job name differs from recorded submission")
    marker = {"schema_version": 1, "key": key, "kind": task["kind"],
              "plan_fingerprint": plan["fingerprint"], "slurm_job_id": jid,
              "status": "RUNNING", "started_at": time.time()}
    path = _marker_path(plan, key)
    _atomic_json(path, marker)
    previous_signal = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Termination signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if task["kind"] == "data":
            policy = state["policy"]
            python = resource_profile(plan["cluster"], "preflight")["python"]
            subprocess.run([python, str(PROJECT_ROOT / "scripts" / "download_data.py"),
                            "--datasets", *policy["datasets"], "--data-root", policy["data_root"],
                            "--workers", str(policy["data_workers"]), "--min-free-gib",
                            str(policy["min_free_gib"]), "--execute"], check=True, cwd=PROJECT_ROOT)
        elif task["kind"] == "preflight":
            run_preflight(plan)
        elif task["kind"] == "collect":
            result = collect(plan)
            if result.get("analysis_returncode", 0) != 0:
                raise RuntimeError(f"Result analysis failed (exit={result['analysis_returncode']})")
        else:
            group = [t for t in plan["smoke" if task["kind"] == "smoke" else "tasks"]
                     if t["resource"] == task["resource"]]
            original = group[task["index"]]
            if original["id"] != key:
                raise ValueError("Worker index does not match immutable plan task")
            run_task(plan, original)
        marker.update(status="PASS", ended_at=time.time())
        _atomic_json(path, marker)
    except BaseException as exc:
        marker.update(status="FAIL", ended_at=time.time(), error_type=type(exc).__name__, error=str(exc))
        _atomic_json(path, marker)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_signal)
