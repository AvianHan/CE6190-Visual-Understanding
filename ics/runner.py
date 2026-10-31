"""Resume-safe, sharded ICS inference. Query annotations never enter adapters.

Per-episode JSON files are the source of truth; JSONL is only an exported view.
No network calls or model downloads are performed by this module.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from PIL import Image

from ics_eval import inference_view, manifest_fingerprint, validate_manifest


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expand_config(value):
    """Expand environment variables strictly; an unset variable is an error."""
    if isinstance(value, dict):
        return {k: expand_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_config(v) for v in value]
    if isinstance(value, str):
        variables = re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", value)
        missing = [a or b for a, b in variables if (a or b) not in os.environ]
        if missing:
            raise ValueError("Unset environment variables in config: " + ", ".join(sorted(set(missing))))
        return os.path.expanduser(os.path.expandvars(value))
    return value


def atomic_json(path, value, *, create_only=False):
    """Publish a complete JSON file; optional race-safe first writer wins."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if create_only:
            try:
                os.link(name, path)
                return True
            except FileExistsError:
                return False
        os.replace(name, path)
        return True
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_png(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".writing-", suffix=".png", dir=path.parent)
    os.close(fd)
    try:
        Image.fromarray(array.astype(np.uint8)).save(name, format="PNG")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def episode_lock(path):
    """OS releases flock after worker death; stale lock files do not block resume."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def git_info(path):
    def command(*args):
        p = subprocess.run(["git", "-C", str(path), *args], text=True,
                           capture_output=True, timeout=15)
        return p.stdout.strip() if p.returncode == 0 else None
    try:
        diff = command("diff", "--binary", "HEAD")
        return {"commit": command("rev-parse", "HEAD"),
                "tracked_changes": command("status", "--porcelain", "--untracked-files=no"),
                "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff is not None else None}
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "tracked_changes": None}


def project_code_digest():
    root = Path(__file__).resolve().parents[1]
    files = [root / "ics_eval.py"]
    files.extend(sorted((root / "ics").rglob("*.py")))
    files.extend(sorted((root / "scripts").glob("*.py")))
    return canonical_sha256({p.relative_to(root).as_posix(): file_sha256(p)
                             for p in files if p.is_file()})


def configured_provenance(config):
    """Hash configured local weights; capture revisions of local source repos.

    Deliberately reads full weight bytes once per worker. No stat-only digest is
    trusted for resume. Setup may pass already pinned local paths; remote model
    identifiers are recorded as unresolved, never called a checkpoint hash.
    """
    out = {}
    def walk(obj, prefix=""):
        if not isinstance(obj, dict):
            return
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                walk(value, name)
            elif isinstance(value, str) and value:
                lower = key.lower()
                p = Path(value)
                is_weight = any(s in lower for s in ("checkpoint", "weight", "model_path", "mllm_path"))
                is_source = lower in {"upstream_dir", "dinov3_source", "dinov2_source", "sam3_source"}
                if is_source:
                    out[name] = {"path": str(p), **git_info(p)}
                elif is_weight and p.is_file():
                    out[name] = {"path": str(p), "sha256": file_sha256(p), "bytes": p.stat().st_size}
                elif is_weight and p.is_dir():
                    suffixes = {".pt", ".pth", ".bin", ".safetensors", ".json", ".model", ".txt", ".tiktoken"}
                    hashes = {f.relative_to(p).as_posix(): file_sha256(f)
                              for f in sorted(p.rglob("*")) if f.is_file() and f.suffix in suffixes}
                    out[name] = {"path": str(p), "files": hashes, "sha256": canonical_sha256(hashes)}
                elif is_weight:
                    out[name] = {"identifier": value, "sha256": None, "local_file_resolved": False}
    walk(config)
    return out


def scientific_config(config, *, validate_model_path=True):
    """Only normalize a managed local server's ephemeral TCP port.

    Its local model directory is hashed independently. External endpoint identity
    remains part of the fingerprint because a server may serve another model.
    """
    config = json.loads(json.dumps(config))
    runtime = config.get("runtime", {})
    if runtime.get("managed_local_mllm"):
        model = Path(runtime.get("mllm_model_path", ""))
        if not runtime.get("mllm_model_path") or (validate_model_path and not model.is_dir()):
            raise ValueError("Managed local MLLM requires existing runtime.mllm_model_path")
        endpoint = config["adapter"].get("endpoint")
        if endpoint:
            parts = urlsplit(endpoint)
            if parts.hostname not in {"localhost", "127.0.0.1", "::1"} or parts.username or parts.password:
                raise ValueError("Managed MLLM endpoint must be unauthenticated loopback; use API key environment variable")
            config["adapter"]["endpoint"] = urlunsplit((parts.scheme, "managed-local-server", parts.path, parts.query, parts.fragment))
    return config


def _torch():
    # Adapters import torch as needed. Analysis/unit tests need no torch install.
    return sys.modules.get("torch")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch = _torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def cuda_sync():
    torch = _torch()
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return []
    devices = list(range(torch.cuda.device_count()))
    for device in devices:
        torch.cuda.synchronize(device)
    return devices


def reset_cuda_peaks():
    devices = cuda_sync()
    torch = _torch()
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    return devices


def cuda_peaks():
    devices = cuda_sync()
    torch = _torch()
    return {str(i): {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(i)),
                     "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(i)),
                     "device_name": torch.cuda.get_device_name(i)} for i in devices}


def numpy_binary_counts(prediction, target, *, ignore_label=None):
    """Vectorized equivalent of the original ics_eval.binary_counts contract."""
    p, t = np.asarray(prediction), np.asarray(target)
    if p.shape != t.shape or p.ndim != 2:
        raise ValueError(f"Expected matching HxW masks, got {p.shape} and {t.shape}")
    valid = np.ones(t.shape, dtype=bool) if ignore_label is None else t != ignore_label
    if not np.all(np.isin(p[valid], [0, 1])) or not np.all(np.isin(t[valid], [0, 1])):
        raise ValueError("Expected mapped binary masks; target may only also contain ignore_label")
    p, t = p[valid].astype(bool), t[valid].astype(bool)
    return {"tp": int(np.count_nonzero(p & t)), "fp": int(np.count_nonzero(p & ~t)),
            "fn": int(np.count_nonzero(~p & t)), "tn": int(np.count_nonzero(~p & ~t)),
            "ignored": int(np.count_nonzero(~valid))}


def _validate_assets_once(manifest, root):
    # Preserve the original structural validator, avoid rehashing the same image
    # fifty times in a reference-robustness manifest.
    validate_manifest(manifest)
    checked = {}
    for ep in manifest["episodes"]:
        for asset in [ep["query"], *ep["references"]]:
            for kind in ("image", "mask"):
                rel, expected = asset[kind], asset[kind + "_sha256"].lower()
                p = (root / rel).resolve()
                if not p.is_relative_to(root):
                    raise ValueError("Symlink escapes dataset root")
                if rel not in checked:
                    checked[rel] = file_sha256(p)
                if checked[rel] != expected:
                    raise ValueError(f"SHA256 mismatch: {rel}")


def _row_key(episode_id):
    return hashlib.sha256(episode_id.encode()).hexdigest()


def _row_base(ep, binding, config):
    return {"result_schema_version": 2, "run_id": binding["run_id"],
            "experiment_id": config["experiment_id"], "method": config["method"],
            "variant": config["variant"], "protocol_id": binding["protocol_id"],
            "evaluation_protocol": binding["evaluation_protocol"],
            "config_sha256": binding["config_sha256"], "manifest_sha256": binding["manifest_sha256"],
            "episode_id": ep["episode_id"], "dataset": ep["dataset"],
            "dataset_version": ep["dataset_version"], "split": ep["split"], "fold": ep["fold"],
            "track": ep["track"], "concept_id": ep["concept_id"],
            "reference_seed": ep["reference_seed"], "nshot": len(ep["references"]),
            "query_image_sha256": ep["query"]["image_sha256"],
            "query_mask_sha256": ep["query"]["mask_sha256"],
            "reference_image_sha256": [r["image_sha256"] for r in ep["references"]],
            "mask_semantics": ep["mask_semantics"], "base_seed": config["seed"]}


def run_experiment(config, *, manifest_path, dataset_root, output_dir,
                   shard_index=0, num_shards=1, resume=False, max_episodes=None,
                   retry_failed=False, adapter_factory=None):
    """Run one shard; injectable factory is for tests, never selected from CLI."""
    config = expand_config(config)
    for field in ("experiment_id", "method", "variant", "seed", "adapter"):
        if field not in config:
            raise ValueError(f"Missing experiment config field: {field}")
    if not isinstance(config["adapter"], dict) or type(config["seed"]) is not int:
        raise ValueError("adapter must be an object and seed must be an integer")
    if type(num_shards) is not int or num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard index/count")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    if retry_failed and not resume:
        raise ValueError("--retry-failed requires --resume")
    root = Path(expand_config(str(dataset_root))).resolve()
    output = Path(expand_config(str(output_dir))).resolve()
    manifest_path = Path(expand_config(str(manifest_path))).resolve()
    manifest = json.loads(manifest_path.read_text())
    _validate_assets_once(manifest, root)
    episodes = manifest["episodes"][:max_episodes]
    adapter_config = dict(config["adapter"])
    adapter_config.setdefault("name", config["method"])
    adapter_config.setdefault("variant", config["variant"])
    adapter_config.setdefault("seed", config["seed"])
    if adapter_config["variant"] != config["variant"]:
        raise ValueError("Top-level and adapter variant differ")
    config["adapter"] = adapter_config
    binding = {"config_sha256": canonical_sha256(scientific_config(config)),
               "manifest_sha256": manifest_fingerprint(manifest),
               "protocol_id": manifest["protocol_id"],
               "evaluation_protocol": config.get("evaluation_protocol", "unified_original_resolution"),
               "code_sha256": project_code_digest(),
               "configured_provenance": configured_provenance({"adapter": adapter_config, "runtime": config.get("runtime", {})}),
               "selected_episode_ids": [ep["episode_id"] for ep in episodes],
               "dataset_root": str(root)}
    binding["run_id"] = config["experiment_id"] + "-" + canonical_sha256(binding)[:16]
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("episodes", "predictions", "locks", "shards", "attempts"):
        (output / folder).mkdir(exist_ok=True)
    meta = {"schema_version": 2, "binding": binding, "config": config,
            "created_at": utc_now(), "manifest_path": str(manifest_path),
            "expected_episodes": len(episodes), "git": git_info(Path(__file__).resolve().parents[1]),
            "expected_episode_scopes": [{"episode_id": ep["episode_id"],
                                         "dataset": ep["dataset"], "split": ep["split"],
                                         "fold": ep["fold"], "track": ep["track"],
                                         "nshot": len(ep["references"])} for ep in episodes],
            "note": "No test GT is available to adapters. Counts use original query resolution."
                    " This differs from upstream INSID3's resized-GT evaluator."}
    if not atomic_json(output / "run.json", meta, create_only=True):
        prior = json.loads((output / "run.json").read_text())
        if prior["binding"] != binding:
            raise ValueError("Run directory binding differs (config, code, weights, data, or manifest); use a new output directory")
    selected = [ep for i, ep in enumerate(episodes) if i % num_shards == shard_index]
    existing = [(output / "episodes" / (_row_key(ep["episode_id"]) + ".json")) for ep in selected]
    if not resume and any(p.exists() for p in existing):
        raise FileExistsError("This shard already has results; use --resume or a new output directory")
    for path in existing:
        if path.exists():
            row = json.loads(path.read_text())
            if row["run_id"] != binding["run_id"]:
                raise ValueError("Existing episode belongs to a different run")
            if row["status"] == "ok":
                pred = output / row["prediction_path"]
                if file_sha256(pred) != row["prediction_sha256"]:
                    raise ValueError("Saved prediction has changed; do not silently reuse it")
    todo = [ep for ep, path in zip(selected, existing)
            if not path.exists() or (retry_failed and json.loads(path.read_text())["status"] != "ok")]
    if adapter_factory is None:
        from ics.adapters import create_adapter
        adapter_factory = create_adapter
    process = {"hostname": socket.gethostname(), "pid": os.getpid(),
               "python": platform.python_version(), "platform": platform.platform(),
               "shard_index": shard_index, "num_shards": num_shards,
               "started_at": utc_now(), "memory_scope": "current Python process; excludes external MLLM server/other processes",
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    process["transport_endpoint"] = adapter_config.get("endpoint")
    try:
        process["packages"] = {p: importlib.metadata.version(p) for p in ("numpy", "Pillow")}
    except importlib.metadata.PackageNotFoundError:
        process["packages"] = {}
    adapter = None
    completed = failed = busy = 0
    try:
        if todo:
            seed_everything(config["seed"])
            load_start = time.perf_counter()
            adapter = adapter_factory(adapter_config)
            cuda_sync()
            process["model_load_seconds"] = time.perf_counter() - load_start
            process["adapter_metadata"] = adapter.metadata()
            # Metadata may expose inferred layers or checkpoint revisions not in config.
            metadata_path = output / "adapter.json"
            if not atomic_json(metadata_path, process["adapter_metadata"], create_only=True):
                if json.loads(metadata_path.read_text()) != process["adapter_metadata"]:
                    raise ValueError("Adapter metadata differs from prior workers; cannot merge run")
            for ep in todo:
                key = _row_key(ep["episode_id"])
                destination = output / "episodes" / (key + ".json")
                with episode_lock(output / "locks" / (key + ".lock")) as acquired:
                    if not acquired:
                        busy += 1
                        continue
                    if destination.exists():
                        prior = json.loads(destination.read_text())
                        if prior["status"] == "ok" or not retry_failed:
                            continue
                        # Preserve the previous failed attempt before doing any
                        # new work. Content-addressed create is idempotent even
                        # if a worker dies after archive but before replacement.
                        archive = output / "attempts" / key / (canonical_sha256(prior) + ".json")
                        atomic_json(archive, prior, create_only=True)
                    attempt_number = 1 + len(list((output / "attempts" / key).glob("*.json")))
                    # Same query receives the same RNG seed across reference choices.
                    digest = canonical_sha256([config["seed"], ep["query"]["image_sha256"]])
                    episode_seed = int(digest[:8], 16)
                    seed_everything(episode_seed)
                    row = _row_base(ep, binding, config)
                    row.update({"episode_seed": episode_seed, "started_at": utc_now(),
                                "shard_index": shard_index, "num_shards": num_shards})
                    started = time.perf_counter()
                    infer_start = None
                    try:
                        view = inference_view(ep)
                        inputs = {"query_image": str(root / view["query_image"]),
                                  "references": [{k: str(root / r[k]) for k in ("image", "mask")}
                                                 for r in view["references"]]}
                        with Image.open(inputs["query_image"]) as image:
                            width, height = image.size
                        reset_cuda_peaks()
                        infer_start = time.perf_counter()
                        prediction = adapter.predict(inputs, seed=episode_seed)
                        cuda_sync()
                        infer_seconds = time.perf_counter() - infer_start
                        peaks = cuda_peaks()
                        mask = np.asarray(prediction.mask)
                        if mask.shape != (height, width) or not np.all(np.isin(mask, [0, 1])):
                            raise ValueError(f"Adapter must return binary original-resolution mask {(height, width)}, got {mask.shape}")
                        # Query GT is decoded only AFTER the adapter has returned.
                        with Image.open(root / ep["query"]["mask"]) as image:
                            truth = np.asarray(image).copy()
                        counts = numpy_binary_counts(mask, truth, ignore_label=ep["ignore_label"])
                        union = counts["tp"] + counts["fp"] + counts["fn"]
                        valid_pixels = counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"]
                        relative_pred = "predictions/" + key + ".png"
                        atomic_png(output / relative_pred, mask)
                        row.update({"status": "ok", "counts": counts,
                                    "iou": counts["tp"] / union if union else None,
                                    "target_area_fraction": (counts["tp"] + counts["fn"]) / valid_pixels if valid_pixels else None,
                                    "prediction_path": relative_pred,
                                    "prediction_sha256": file_sha256(output / relative_pred),
                                    "inference_seconds": infer_seconds, "cuda_memory": peaks,
                                    "memory_scope": process["memory_scope"],
                                    "telemetry": prediction.telemetry, "artifacts": prediction.artifacts})
                        # Reject non-JSON/non-finite telemetry before marking complete.
                        json.dumps(row, allow_nan=False)
                        completed += 1
                    except Exception as exc:
                        # Failed episodes remain in denominator via coverage/failure rate.
                        # A failed prediction is not fabricated as an all-background mask.
                        row = _row_base(ep, binding, config) | {
                            "episode_seed": episode_seed, "status": "error", "counts": None,
                            "iou": None, "error_type": type(exc).__name__, "error": str(exc),
                            "traceback": traceback.format_exc(limit=10)}
                        if infer_start is not None:
                            row["failed_inference_wall_seconds"] = time.perf_counter() - infer_start
                        # Error strata may use GT only after the failed prediction;
                        # no fabricated predicted mask/counts are constructed.
                        try:
                            with Image.open(root / ep["query"]["mask"]) as image:
                                error_truth = np.asarray(image)
                            valid = np.ones(error_truth.shape, dtype=bool) if ep["ignore_label"] is None else error_truth != ep["ignore_label"]
                            if np.all(np.isin(error_truth[valid], [0, 1])):
                                denominator = int(np.count_nonzero(valid))
                                row["target_area_fraction"] = int(np.count_nonzero((error_truth == 1) & valid)) / denominator if denominator else None
                        except Exception as annotation_error:
                            row["annotation_analysis_error"] = str(annotation_error)
                        if hasattr(adapter, "failure_diagnostics"):
                            try:
                                diagnostics = adapter.failure_diagnostics()
                                json.dumps(diagnostics, allow_nan=False)
                                row["failure_diagnostics"] = diagnostics
                            except Exception as diagnostic_error:
                                row["diagnostics_error"] = str(diagnostic_error)
                        try:
                            row["cuda_memory"] = cuda_peaks()
                        except Exception as memory_error:
                            row["memory_measurement_error"] = str(memory_error)
                        failed += 1
                    row["episode_wall_seconds"] = time.perf_counter() - started
                    row["attempt_number"] = attempt_number
                    row["finished_at"] = utc_now()
                    atomic_json(destination, row)
    except Exception as exc:
        process["fatal_error"] = {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc(limit=15)}
        raise
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception as exc:
                process["close_error"] = {"type": type(exc).__name__, "message": str(exc)}
        process.update({"finished_at": utc_now(), "new_ok": completed,
                        "new_errors": failed, "busy_skipped": busy,
                        "already_present": len(selected) - len(todo)})
        attempt = f"{shard_index:05d}-{os.getpid()}-{time.time_ns()}"
        atomic_json(output / "shards" / (attempt + ".json"), process)
    final_rows = [json.loads(p.read_text()) for p in existing if p.exists()]
    return {"run_id": binding["run_id"], "output": str(output),
            "new_ok": completed, "new_errors": failed, "busy_skipped": busy,
            "already_present": len(selected) - len(todo), "expected_global": len(episodes),
            "errors_in_shard": sum(r["status"] == "error" for r in final_rows),
            "pending_in_shard": len(selected) - len(final_rows)}
