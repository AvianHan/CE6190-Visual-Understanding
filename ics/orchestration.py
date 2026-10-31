"""Dependency-aware Slurm/local orchestration. Planning never submits jobs.

The scheduler only uses GPUs allocated by Slurm or explicitly supplied locally.
A separate local vLLM process is owned by each concept-guided job by default.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_NAMES = {"insid3", "cgics_visual", "cgics_mllm"}


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def fingerprint(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expand(value):
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, str):
        value = os.path.expanduser(os.path.expandvars(value))
        if re.search(r"\$\{[^}]+\}|\$[A-Za-z_]", value):
            raise ValueError(f"Unresolved environment variable in configuration: {value}")
    return value


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Unsafe identifier: {value!r}")
    return value


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_path(value: str, relative_to: Path = PROJECT_ROOT) -> str:
    p = Path(expand(value))
    return str(p.resolve() if p.is_absolute() else (relative_to / p).resolve())


def resource_profile(cluster: dict, name: str) -> dict:
    p = dict(cluster.get("defaults", {}))
    p.update(cluster.get("resources", {}).get(name, {}))
    if "python" not in p:
        raise ValueError(f"Resource profile {name} needs an explicit python executable")
    p["python"] = resolve_path(p["python"])
    p["gpus"] = int(p.get("gpus", 0))
    if p["gpus"] < 0:
        raise ValueError("GPU allocation cannot be negative")
    if name in RESOURCE_NAMES and p["gpus"] < 1:
        raise ValueError(f"GPU experiment profile {name} must request GPUs")
    if name == "cgics_mllm":
        mode = p.get("mllm_mode", "owned")
        if mode not in {"owned", "external"}:
            raise ValueError("mllm_mode must be owned or external")
        if mode == "owned":
            if p["gpus"] < 2:
                raise ValueError("Owned SAM3 + vLLM profile requires separate allocated GPUs (>=2)")
            for field in ("mllm_python", "mllm_model_path"):
                if not p.get(field):
                    raise ValueError(f"Owned MLLM profile requires {field}")
                p[field] = resolve_path(p[field])
        else:
            endpoint = p.get("mllm_endpoint", "")
            if not endpoint.startswith(("http://127.0.0.1:", "http://localhost:")):
                if not p.get("allow_remote_mllm", False):
                    raise ValueError("Remote MLLM requires explicit allow_remote_mllm: true; images leave this worker")
            if not endpoint:
                raise ValueError("External MLLM requires mllm_endpoint")
    return p


def build_plan(suite_path: Path, cluster_path: Path, run_dir: Path) -> dict:
    suite, cluster = expand(load_json(suite_path)), expand(load_json(cluster_path))
    if suite.get("schema_version") != 1 or cluster.get("schema_version") != 1:
        raise ValueError("Expected suite and cluster schema_version=1")
    manifests = {}
    disabled_manifests = {m["id"] for m in suite["manifests"] if not m.get("enabled", True)}
    for m in suite["manifests"]:
        if not m.get("enabled", True):
            continue
        mid = safe_id(m["id"])
        if mid in manifests:
            raise ValueError("Duplicate manifest id")
        manifests[mid] = {**m, "path": resolve_path(m["path"]), "dataset_root": resolve_path(m["dataset_root"])}
    run_dir = run_dir.resolve()
    tasks, smoke = [], []
    config_ids = set()
    resources = {}
    for item in suite["experiments"]:
        if not item.get("enabled", True):
            continue
        cfg_path = Path(resolve_path(item["config"]))
        cfg = expand(load_json(cfg_path))
        cid = safe_id(cfg["experiment_id"])
        if cid in config_ids:
            raise ValueError(f"Duplicate experiment id {cid}")
        config_ids.add(cid)
        rname = item["resource"]
        if rname not in RESOURCE_NAMES:
            raise ValueError(f"Unknown GPU resource {rname}")
        if rname not in resources:
            resources[rname] = resource_profile(cluster, rname)
        selected = [mid for mid in item.get("manifest_ids", []) if mid not in disabled_manifests]
        if not selected:
            raise ValueError(f"No manifests selected for {cid}")
        snap_cfg = str(run_dir / "configs" / f"{cid}.json")
        first = manifests[selected[0]]
        smoke.append({"id": f"smoke-{cid}", "resource": rname, "config": snap_cfg,
                      "config_data": cfg, "manifest_id": selected[0], "manifest": first["path"],
                      "dataset_root": first["dataset_root"], "output": str(run_dir / "smoke" / cid),
                      "shard_index": 0, "num_shards": 1,
                      "max_episodes": int(suite.get("smoke_episodes", 2)), "phase": "smoke"})
        for mid in selected:
            if mid not in manifests:
                raise ValueError(f"Unknown manifest {mid}")
            m = manifests[mid]
            shards = int(m.get("shards", item.get("shards", suite.get("shards_by_cohort", {}).get(m.get("cohort"), suite.get("shards", 1)))))
            if shards < 1:
                raise ValueError("shards must be positive")
            for shard in range(shards):
                tasks.append({"id": f"{cid}--{mid}--{shard}", "resource": rname,
                              "config": snap_cfg, "config_data": cfg, "manifest_id": mid,
                              "manifest": m["path"], "dataset_root": m["dataset_root"],
                              "output": str(run_dir / "results" / cid / mid),
                              "shard_index": shard, "num_shards": shards,
                              "max_episodes": None, "phase": "experiment"})
    if not tasks:
        raise ValueError("Suite contains no enabled experiments")
    used = {x["manifest_id"] for x in tasks}
    manifests = {k: v for k, v in manifests.items() if k in used}
    for task in smoke:
        if task["max_episodes"] < 1:
            raise ValueError("Smoke validation cannot be disabled")
    plan = {"schema_version": 1, "project_root": str(PROJECT_ROOT), "run_dir": str(run_dir),
            "suite_path": str(suite_path.resolve()), "cluster_path": str(cluster_path.resolve()),
            "workload_description": suite.get("description", ""), "cluster": cluster,
            "manifests": manifests, "resources": resources, "smoke": smoke, "tasks": tasks,
            "prepare_missing_manifests": bool(suite.get("prepare_missing_manifests", False))}
    plan["fingerprint"] = fingerprint(plan)
    return plan


def write_plan(plan: dict) -> Path:
    run = Path(plan["run_dir"])
    path = run / "plan.json"
    if path.exists() and load_json(path).get("fingerprint") != plan["fingerprint"]:
        raise ValueError("Run directory has a different plan; choose a new --run-dir")
    for task in plan["smoke"]:
        atomic_json(Path(task["config"]), task["config_data"])
    for phase in ("smoke", "tasks"):
        for rname in plan["resources"]:
            group = [t for t in plan[phase] if t["resource"] == rname]
            atomic_json(run / "tasks" / f"{phase}-{rname}.json", group)
    atomic_json(path, plan)
    return path


def plan_summary(plan: dict) -> dict:
    by_resource = {}
    for name, profile in plan["resources"].items():
        tasks = [t for t in plan["tasks"] if t["resource"] == name]
        episodes = 0
        for t in tasks:
            m = plan["manifests"][t["manifest_id"]]
            recipe = m.get("prepare", {})
            estimate = recipe.get("queries", 0) * recipe.get("references_per_query", 1)
            episodes += estimate / t["num_shards"]
        by_resource[name] = {"smoke_jobs": sum(t["resource"] == name for t in plan["smoke"]),
                             "experiment_jobs": len(tasks), "gpus_per_job": profile["gpus"],
                             "requested_episode_estimate": int(episodes),
                             "max_parallel": profile.get("max_parallel", plan["cluster"].get("max_parallel"))}
    return {"status": "PLANNED_NOT_SUBMITTED", "plan_fingerprint": plan["fingerprint"],
            "manifests": len(plan["manifests"]), "resources": by_resource,
            "note": "Episode counts are requested upper estimates; available distinct references may be fewer. No model result exists yet."}


def _safe_sbatch_value(value, name):
    value = str(value)
    if not value or "\n" in value or "\r" in value:
        raise ValueError(f"Invalid Slurm {name}")
    return value


def sbatch_args(profile: dict, *, name: str, output: Path, array: str | None = None,
                dependency: str | None = None) -> list[str]:
    cmd = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes", "--job-name", name, "--output", str(output)]
    for key, flag in (("account", "--account"), ("partition", "--partition"), ("time", "--time")):
        if not profile.get(key):
            raise ValueError(f"Slurm profile needs actual {key}; probe the cluster then fill config")
        cmd.extend([flag, _safe_sbatch_value(profile[key], key)])
    cpus = int(profile.get("cpus", 1))
    if cpus < 1:
        raise ValueError("cpus must be positive")
    cmd.extend(["--cpus-per-task", str(cpus)])
    if profile.get("memory"):
        cmd.extend(["--mem", _safe_sbatch_value(profile["memory"], "memory")])
    if int(profile.get("gpus", 0)):
        gres = profile.get("gres") or f"gpu:{profile['gpus']}"
        cmd.extend(["--gres", _safe_sbatch_value(gres, "gres")])
    for key, flag in (("qos", "--qos"), ("constraint", "--constraint"), ("reservation", "--reservation")):
        if profile.get(key):
            cmd.extend([flag, _safe_sbatch_value(profile[key], key)])
    if array:
        cmd.extend(["--array", array])
    if dependency:
        cmd.extend(["--dependency", dependency])
    return cmd


def worker_script(plan_path: Path, phase: str, resource: str | None = None, offset: int = 0) -> str:
    plan = load_json(plan_path)
    # The orchestrator is stdlib-only and uses the explicitly configured launcher Python.
    python = plan["cluster"].get("launcher_python", "python3")
    cmd = [python, str(PROJECT_ROOT / "scripts" / "launch_suite.py"), "worker",
           "--plan", str(plan_path), "--phase", phase]
    if resource:
        cmd += ["--resource", resource, "--offset", str(offset)]
    return "#!/usr/bin/env bash\nset -euo pipefail\n" + "exec " + shlex.join(cmd) + "\n"


def slurm_array_limit(cluster: dict) -> int:
    configured = int(cluster.get("max_array_size", 1000))
    if configured < 1:
        raise ValueError("max_array_size must be positive")
    if shutil.which("scontrol"):
        try:
            output = subprocess.check_output(["scontrol", "show", "config"], text=True, timeout=15, stderr=subprocess.DEVNULL)
            match = re.search(r"(?m)^\s*MaxArraySize\s*=\s*(\d+)", output)
            if match:
                actual = int(match.group(1))
                if actual < 1:
                    raise ValueError("Cluster MaxArraySize disables arrays")
                return min(configured, actual)
        except (OSError, subprocess.SubprocessError):
            pass
    return configured


def submit_slurm(plan_path: Path) -> dict:
    if not shutil.which("sbatch"):
        raise RuntimeError("sbatch is unavailable. Planning is still possible; submit from the actual cluster.")
    plan = load_json(plan_path)
    run = Path(plan["run_dir"])
    state_path, lock_path = run / "submission.json", run / "submission.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError("Submission lock exists; another launcher may be active. Inspect it before removing.")
    os.close(fd)
    try:
        state = load_json(state_path) if state_path.exists() else {"plan_fingerprint": plan["fingerprint"], "jobs": {}}
        if state["plan_fingerprint"] != plan["fingerprint"]:
            raise ValueError("Submission state belongs to another plan")
        (run / "logs").mkdir(exist_ok=True)
        (run / "slurm").mkdir(exist_ok=True)
        cpu = resource_profile(plan["cluster"], "preflight")
        # Validate EVERY profile before submitting even the first job.
        sbatch_args(cpu, name="ics-preflight", output=run / "logs" / "preflight-%j.log")
        for rname, p in plan["resources"].items():
            sbatch_args(p, name="ics-" + rname, output=run / "logs" / "x-%j.log")
            cap = p.get("max_parallel", plan["cluster"].get("max_parallel"))
            if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
                raise ValueError(f"Set max_parallel for {rname} from actual allocation/account limits")
        max_submit = plan["cluster"].get("max_submit_tasks")
        planned_elements = len(plan["smoke"]) + len(plan["tasks"]) + 2
        if max_submit is not None:
            if type(max_submit) is not int or max_submit < 1:
                raise ValueError("max_submit_tasks must be a positive available task budget or null")
            if planned_elements > max_submit:
                raise ValueError(f"Plan requests {planned_elements} submitted task elements, above configured available max_submit_tasks={max_submit}. No jobs submitted. Adjust shards/cohorts/suite or use a sufficient actual allocation; max_parallel does not reduce pending task count.")
        array_limit = slurm_array_limit(plan["cluster"])
        def submit(key, profile, phase, resource=None, dependency=None, n=None, offset=0):
            if key in state["jobs"]:
                return state["jobs"][key]["job_id"]
            script = run / "slurm" / (key + ".sh")
            script.write_text(worker_script(plan_path, phase, resource, offset))
            array = None
            if n is not None:
                cap = profile.get("max_parallel", plan["cluster"].get("max_parallel"))
                array = f"0-{n-1}%{cap}"
            command = sbatch_args(profile, name="ics-" + key, output=run / "logs" / (key + "-%A_%a.log"),
                                  array=array, dependency=dependency) + [str(script)]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split(";")[0]
            if not re.fullmatch(r"\d+", job_id):
                raise RuntimeError(f"Cannot parse sbatch job id: {result.stdout!r}; inspect submission before retry")
            state["jobs"][key] = {"job_id": job_id, "command": command, "submitted_at": time.time()}
            atomic_json(state_path, state)
            return job_id
        pre = submit("preflight", cpu, "preflight")
        production_ids = []
        for rname, profile in plan["resources"].items():
            sm = [t for t in plan["smoke"] if t["resource"] == rname]
            jobs = [t for t in plan["tasks"] if t["resource"] == rname]
            smoke_ids = []
            for offset in range(0, len(sm), array_limit):
                key = "smoke-" + rname + (f"-part{offset // array_limit:03d}" if len(sm) > array_limit else "")
                dependency = "afterok:" + pre
                if smoke_ids:
                    dependency += ",afterany:" + smoke_ids[-1]
                smoke_ids.append(submit(key, profile, "smoke", rname, dependency,
                                        min(array_limit, len(sm)-offset), offset))
            previous = None
            for offset in range(0, len(jobs), array_limit):
                key = "experiments-" + rname + (f"-part{offset // array_limit:03d}" if len(jobs) > array_limit else "")
                dependencies = smoke_ids + ([previous] if previous else [])
                previous = submit(key, profile, "experiment", rname, "afterany:" + ":".join(dependencies),
                                  min(array_limit, len(jobs)-offset), offset)
                production_ids.append(previous)
        submit("collect", cpu, "collect", dependency="afterany:" + ":".join(production_ids))
        return {"status": "SUBMITTED", **state,
                "note": "Existing job ids are reused. Failed jobs are not silently resubmitted; inspect squeue/sacct and use a new run or explicit resumed worker."}
    finally:
        lock_path.unlink(missing_ok=True)


def run_preflight(plan: dict) -> None:
    # Model assets are validated per configuration in GPU smoke jobs, so an optional
    # missing checkpoint never blocks unrelated model families at the common data gate.
    python = resource_profile(plan["cluster"], "preflight")["python"]
    for mid, m in plan["manifests"].items():
        target = Path(m["path"])
        if not target.exists():
            if not plan["prepare_missing_manifests"] or not m.get("prepare"):
                raise FileNotFoundError(f"Missing manifest {mid}: {target}")
            r = m["prepare"]
            cmd = [python, str(PROJECT_ROOT / "scripts" / "prepare_episodes.py"), "--dataset", r["dataset"],
                   "--data-root", m["dataset_root"], "--output", str(target)]
            for key in ("fold", "queries", "references_per_query", "shots", "seed", "query_seed"):
                if key in r:
                    cmd.extend(["--" + key.replace("_", "-"), str(r[key])])
            for key in ("split", "split_file", "group_map", "protocol_id", "insufficient_references"):
                if r.get(key) is not None:
                    cmd.extend(["--" + key.replace("_", "-"), str(r[key])])
            if r.get("trust_prepared_pickle", False):
                cmd.append("--trust-prepared-pickle")
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        if m.get("prepare"):
            sidecar_path = target.with_suffix(".protocol.json")
            if not sidecar_path.exists():
                raise ValueError(f"Existing manifest needs its protocol sidecar: {sidecar_path}")
            sidecar = load_json(sidecar_path)
            sampling = sidecar["sampling"]
            mapping = {"queries": "queries_requested", "references_per_query": "reference_sets_requested_per_query",
                       "shots": "shots", "seed": "seed", "query_seed": "query_seed"}
            for requested, stored in mapping.items():
                if requested in m["prepare"] and m["prepare"][requested] != sampling.get(stored):
                    raise ValueError(f"Existing manifest {mid} has a different {requested}; use a new manifest path, never silently overwrite")
            from ics_eval import manifest_fingerprint
            manifest = load_json(target)
            if sidecar.get("manifest_sha256") != manifest_fingerprint(manifest):
                raise ValueError(f"Manifest/sidecar fingerprint mismatch: {mid}")
            if any(e["fold"] != m["prepare"]["fold"] for e in manifest["episodes"]):
                raise ValueError(f"Existing manifest fold differs from recipe: {mid}")
        subprocess.run([python, str(PROJECT_ROOT / "ics_eval.py"), str(target),
                        "--dataset-root", m["dataset_root"]], check=True, cwd=PROJECT_ROOT)
    atomic_json(Path(plan["run_dir"]) / "preflight.complete.json",
                {"status": "PASS", "plan_fingerprint": plan["fingerprint"], "time": time.time(),
                 "note": "File/dependency validation only. GPU validity is established by model smoke jobs."})


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def split_devices(devices: list[str], count: int) -> tuple[list[str], list[str]]:
    if len(devices) < count:
        raise RuntimeError(f"Need {count} explicitly allocated GPUs, got {len(devices)}")
    if count < 2:
        return devices[:1], []
    return devices[:1], devices[1:count]


def _allocated_devices(profile: dict, explicit: list[str] | None) -> list[str]:
    if explicit is not None:
        devices = explicit
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not visible:
            raise RuntimeError("CUDA_VISIBLE_DEVICES is unset/empty. Refusing to infer GPUs from nvidia-smi; use allocation or --gpus.")
        devices = [s.strip() for s in visible.split(",") if s.strip()]
    if len(devices) != len(set(devices)) or any(x in {"-1", "NoDevFiles"} for x in devices):
        raise RuntimeError("Invalid or duplicated GPU device assignment")
    if len(devices) < profile["gpus"]:
        raise RuntimeError("Allocated GPU list is smaller than resource profile request")
    return devices[:profile["gpus"]]


def _wait_health(proc, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Owned vLLM exited before becoming healthy (exit={proc.returncode}); see job vllm log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1)
    raise TimeoutError("Owned vLLM did not become healthy before configured startup timeout")



def terminate_owned_process(proc):
    """Terminate the session/group we own, including children after parent exit."""
    if proc is None:
        return
    def group_alive():
        try:
            os.killpg(proc.pid, 0)
            return True
        except ProcessLookupError:
            return False
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 20
    while group_alive() and time.monotonic() < deadline:
        proc.poll()  # reap exited parent; process group may still have live children
        time.sleep(.1)
    if group_alive():
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def run_task(plan: dict, task: dict, *, devices: list[str] | None = None) -> None:
    if not (Path(plan["run_dir"]) / "preflight.complete.json").exists():
        raise RuntimeError("Preflight did not pass")
    profile = plan["resources"][task["resource"]]
    devs = _allocated_devices(profile, devices)
    job_dir = Path(plan["run_dir"]) / "jobs" / task["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    lock = job_dir / "running.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f"Task lock exists: {lock}. Inspect stale/running job before retry")
    os.close(fd)
    cfg = json.loads(json.dumps(task["config_data"]))
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(profile.get("cpus", 1))
    env["TOKENIZERS_PARALLELISM"] = "false"
    server = None
    runner = None
    server_log = None
    from ics.resource_monitor import Monitor
    monitor = Monitor(output_path=job_dir / "gpu_memory.json", interval=1.0)
    monitor.start()
    meta = {"task_id": task["id"], "resource": task["resource"], "phase": task["phase"],
            "plan_fingerprint": plan["fingerprint"], "start": time.time(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "status": "RUNNING",
            "gpu_assignment": devs, "gpu_memory_scope": "Runner allocator peak plus sampled memory attributed to owned runner/MLLM process trees. No unrelated process is counted as owned."}
    try:
        for field in ("upstream_dir", "checkpoint", "sam3_checkpoint", "dinov3_source", "dinov2_source", "sam3_source"):
            value = cfg["adapter"].get(field)
            if value and not Path(value).exists():
                raise FileNotFoundError(f"Missing model asset {field}: {value}")
        if task["phase"] == "experiment":
            smokes = [s for s in plan["smoke"] if s["config_data"]["experiment_id"] == task["config_data"]["experiment_id"]]
            bad = [s["id"] for s in smokes if not (Path(plan["run_dir"]) / "jobs" / s["id"] / "complete.json").exists()]
            if bad:
                raise RuntimeError("Resource family smoke validation has not passed: " + ", ".join(bad))
        if task["resource"] == "cgics_mllm" and profile.get("mllm_mode", "owned") == "owned":
            sam_devices, lm_devices = split_devices(devs, profile["gpus"])
            env["CUDA_VISIBLE_DEVICES"] = ",".join(sam_devices)
            lm_env = dict(env)
            lm_env["CUDA_VISIBLE_DEVICES"] = ",".join(lm_devices)
            port = _free_port()
            cfg["adapter"]["endpoint"] = f"http://127.0.0.1:{port}/v1"
            cfg.setdefault("runtime", {})["managed_local_mllm"] = True
            cfg["runtime"]["mllm_model_path"] = profile["mllm_model_path"]
            served = cfg["adapter"].get("mllm_model", "qwen3-vl-4b")
            command = [profile["mllm_python"], "-m", "vllm.entrypoints.openai.api_server",
                       "--model", profile["mllm_model_path"], "--served-model-name", served,
                       "--host", "127.0.0.1", "--port", str(port),
                       "--tensor-parallel-size", str(len(lm_devices)), "--disable-log-requests"]
            if profile.get("mllm_gpu_memory_utilization") is not None:
                v = float(profile["mllm_gpu_memory_utilization"])
                if not 0 < v <= 1:
                    raise ValueError("mllm_gpu_memory_utilization must be in (0,1]")
                command += ["--gpu-memory-utilization", str(v)]
            if profile.get("mllm_max_model_len"):
                command += ["--max-model-len", str(int(profile["mllm_max_model_len"]))]
            if profile.get("mllm_max_num_seqs") is not None:
                command += ["--max-num-seqs", str(int(profile["mllm_max_num_seqs"]))]
            if profile.get("mllm_processor_cache_gb") is not None:
                command += ["--mm-processor-cache-gb", str(float(profile["mllm_processor_cache_gb"]))]
            meta["mllm_scope"] = "owned_dedicated_allocated_gpus"
            meta["mllm_server_start"] = time.time()
            server_log = (job_dir / "vllm.log").open("a")
            server = subprocess.Popen(command, env=lm_env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            monitor.add_process(server.pid, role="mllm")
            _wait_health(server, port, float(profile.get("mllm_startup_timeout", 900)))
            meta["mllm_server_ready"] = time.time()
        else:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(devs[:1])
            if task["resource"] == "cgics_mllm":
                cfg["adapter"]["endpoint"] = profile["mllm_endpoint"]
                cfg["adapter"]["allow_remote_endpoint"] = bool(profile.get("allow_remote_mllm", False))
                meta["mllm_scope"] = "external_server_memory_and_contention_not_measured_by_worker"
        cfg["adapter"]["device"] = "cuda"
        runtime_config = job_dir / "config.json"
        atomic_json(runtime_config, cfg)
        atomic_json(job_dir / "status.json", meta)
        cmd = [profile["python"], str(PROJECT_ROOT / "scripts" / "run_experiment.py"),
               "--config", str(runtime_config), "--manifest", task["manifest"],
               "--dataset-root", task["dataset_root"], "--output", task["output"],
               "--shard-index", str(task["shard_index"]), "--num-shards", str(task["num_shards"]), "--resume"]
        if task["max_episodes"] is not None:
            cmd += ["--max-episodes", str(task["max_episodes"])]
        meta["runner_start"] = time.time()
        with (job_dir / "runner.log").open("a") as out:
            runner = subprocess.Popen(cmd, env=env, cwd=PROJECT_ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            monitor.add_process(runner.pid, role="runner")
            returncode = runner.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, cmd)
        meta.update(status="PASS", end=time.time())
        atomic_json(job_dir / "complete.json", meta)
    except BaseException as e:
        meta.update(status="FAIL", end=time.time(), error_type=type(e).__name__, error=str(e))
        atomic_json(job_dir / "failure.json", meta)
        raise
    finally:
        terminate_owned_process(runner)
        terminate_owned_process(server)
        if server_log:
            server_log.close()
        meta["gpu_sampling"] = monitor.stop()
        if meta["status"] == "PASS":
            atomic_json(job_dir / "complete.json", meta)
        atomic_json(job_dir / "status.json", meta)
        lock.unlink(missing_ok=True)


def collect(plan: dict) -> dict:
    run = Path(plan["run_dir"])
    rows = []
    for t in plan["smoke"] + plan["tasks"]:
        p = run / "jobs" / t["id"] / "status.json"
        rows.append(load_json(p) if p.exists() else {"task_id": t["id"], "status": "NOT_RUN_OR_DEPENDENCY_BLOCKED"})
    summary = {"plan_fingerprint": plan["fingerprint"], "tasks": rows,
               "counts": {status: sum(r["status"] == status for r in rows)
                          for status in sorted({r["status"] for r in rows})}}
    atomic_json(run / "execution_summary.json", summary)
    if any((run / "results").rglob("*.json")) if (run / "results").exists() else False:
        p = resource_profile(plan["cluster"], "preflight")["python"]
        result = subprocess.run([p, str(PROJECT_ROOT / "scripts" / "analyze_results.py"),
                                 "--root", str(run / "results"), "--output", str(run / "analysis")],
                                cwd=PROJECT_ROOT, capture_output=True, text=True)
        (run / "analysis.log").write_text(result.stdout + result.stderr)
        summary["analysis_returncode"] = result.returncode
        atomic_json(run / "execution_summary.json", summary)
    return summary


class DevicePool:
    def __init__(self, devices: list[str]):
        if not devices or len(devices) != len(set(devices)):
            raise ValueError("Pass a nonempty, unique explicit local --gpus list")
        self.available = list(devices)
        self.total = len(devices)
        self.cond = threading.Condition()

    def acquire(self, n):
        if n > self.total:
            raise ValueError("Task GPU request exceeds explicit local GPU pool")
        with self.cond:
            self.cond.wait_for(lambda: len(self.available) >= n)
            result = self.available[:n]
            del self.available[:n]
            return result

    def release(self, devices):
        with self.cond:
            self.available += devices
            self.cond.notify_all()


def run_local(plan_path: Path, devices: list[str]) -> dict:
    plan = load_json(plan_path)
    pool = DevicePool(devices)
    for p in plan["resources"].values():
        if p["gpus"] > pool.total:
            raise ValueError("Local explicit GPU list cannot satisfy a resource profile")
    run_preflight(plan)
    errors = []
    error_lock = threading.Lock()
    def one(task):
        allocated = pool.acquire(plan["resources"][task["resource"]]["gpus"])
        try:
            run_task(plan, task, devices=allocated)
        finally:
            pool.release(allocated)
    def family(name):
        cap = plan["resources"][name].get("max_parallel", plan["cluster"].get("max_parallel"))
        if not isinstance(cap, int) or cap < 1:
            raise ValueError("Local execution also requires explicit positive max_parallel")
        with concurrent.futures.ThreadPoolExecutor(max_workers=cap) as ex:
            futures = [ex.submit(one, t) for t in plan["smoke"] if t["resource"] == name]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    with error_lock:
                        errors.append(str(e))  # only its own config is blocked; other smoke-passed configs continue
            futures = [ex.submit(one, t) for t in plan["tasks"] if t["resource"] == name]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    with error_lock:
                        errors.append(str(e))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(plan["resources"])) as ex:
        futures = [ex.submit(family, n) for n in plan["resources"]]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                errors.append(str(e))
    result = collect(plan)
    result["errors"] = errors
    return result
