"""Audit and compare batch-2 outputs without turning repeated references into n.

The ordinary runner row remains the formal observation. Fixed-pool alternatives
are paired outcomes nested in that row, not additional jobs or inference runs.
No selection of a winning threshold or parameter is performed here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np
from PIL import Image

from ics.analysis import load_run
from ics.runner import (atomic_json, canonical_sha256, scientific_config,
                        _verify_saved_predictions)
from ics_eval import manifest_fingerprint
from ics.batch2_data import verify_scope


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def distribution(values):
    values = [float(x) for x in values if _finite(x)]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "sum": sum(values) if values else None}


def verify_counts(record):
    """Arithmetic validation only; hash-checking PNGs is not a pixel recompute."""
    if record.get("status", "ok") == "error":
        if record.get("counts") is not None or record.get("iou") is not None:
            raise ValueError("Error result must retain null counts/IoU")
        return
    counts = record.get("counts")
    if not isinstance(counts, dict) or any(type(counts.get(k)) is not int or counts[k] < 0
                                           for k in ("tp", "fp", "fn", "tn", "ignored")):
        raise ValueError("Invalid binary confusion counts")
    union = counts["tp"] + counts["fp"] + counts["fn"]
    expected = counts["tp"] / union if union else None
    if expected is None:
        if record.get("iou") is not None:
            raise ValueError("Empty union requires null IoU")
    elif not _finite(record.get("iou")) or not math.isclose(record["iou"], expected, abs_tol=1e-12):
        raise ValueError("Saved IoU disagrees with counts")


def verify_score_pool(row):
    audit = row.get("artifacts", {}).get("batch2_score_audit")
    if not isinstance(audit, dict):
        raise ValueError("Successful fixed-pool output lacks score audit")
    payload = audit.get("fingerprint_payload")
    if not isinstance(payload, dict) or canonical_sha256(payload) != audit.get("pool_fingerprint"):
        raise ValueError("Fixed candidate pool fingerprint mismatch")
    candidates = audit.get("candidates")
    if not candidates or candidates != payload.get("candidates"):
        raise ValueError("Candidate pool differs from fingerprint payload")
    rules = audit.get("rules", {})
    if set(rules) != {"combined", "ref_only", "query_only"}:
        raise ValueError("Fixed-pool three-rule coverage incomplete")
    for i, candidate in enumerate(candidates):
        if candidate.get("candidate_index") != i or not isinstance(candidate.get("prompt"), str):
            raise ValueError("Invalid candidate order or prompt")
        for name in ("ref_iou", "query_presence", "combined_score"):
            if not _finite(candidate.get(name)) or not 0 <= candidate[name] <= 1:
                raise ValueError("Invalid candidate score")
        if not math.isclose(candidate["combined_score"], candidate["ref_iou"] * candidate["query_presence"], abs_tol=1e-12):
            raise ValueError("Combined score does not match frozen product rule")
    common = audit.get("common_fallback", {})
    maximum = max(c["combined_score"] for c in candidates)
    fallback = maximum < 1e-6
    if (common.get("score_column") != "combined_score" or common.get("threshold") != 1e-6 or
            common.get("max_combined_score") != maximum or common.get("active") is not fallback):
        raise ValueError("Fixed-pool common fallback differs from the frozen shared rule")
    for name, score_name in (("combined", "combined_score"), ("ref_only", "ref_iou"), ("query_only", "query_presence")):
        index = max(range(len(candidates)), key=lambda i: candidates[i][score_name])
        selected = candidates[index]
        rule = rules[name]
        if (rule.get("selected_candidate_index") != index or
                rule.get("fallback_to_visual") is not fallback or
                rule.get("selected_concept") != ("visual" if fallback else selected["prompt"]) or
                not _finite(rule.get("selected_score")) or
                not math.isclose(rule["selected_score"], selected[score_name], abs_tol=1e-12)):
            raise ValueError(f"Selection differs from frozen {name} rule")
    return audit


def _gate_trigger(gate):
    value = gate.get("search_triggered", gate.get("triggered_search"))
    if "search_triggered" in gate and "triggered_search" in gate and gate["search_triggered"] != gate["triggered_search"]:
        raise ValueError("Gate trigger aliases disagree")
    return value if type(value) is bool else None


def verify_gate(row, config):
    variant = config["variant"]
    if variant not in {"agreement_gate", "random_gate", "initial_score_gate"}:
        return
    gate = row.get("artifacts", {}).get("batch2_gate")
    if not isinstance(gate, dict) or gate.get("policy") != variant or gate.get("query_gt_used") is not False:
        raise ValueError("Missing or invalid query-GT-free gate audit")
    triggered = _gate_trigger(gate)
    if triggered is None:
        raise ValueError("Missing gate search decision")
    settings = config["adapter"]
    if variant == "initial_score_gate":
        score = gate["initial_score"]
        product = score["ref_iou"] * score["query_score"]
        threshold = settings.get("score_gate_threshold", .8)
        if not math.isclose(product, score["combined_score"], abs_tol=1e-12) or gate.get("threshold") != threshold:
            raise ValueError("Initial-score gate evidence disagrees with frozen configuration")
        expected = product < threshold
    else:
        agreement = gate.get("agreement", {})
        names = ("intersection", "union", "text_area", "visual_area")
        if any(type(agreement.get(k)) is not int or agreement[k] < 0 for k in names):
            raise ValueError("Invalid gate overlap counts")
        inter, union, text, visual = (agreement[k] for k in names)
        if inter > min(text, visual) or union != text + visual - inter:
            raise ValueError("Gate overlap counts disagree")
        both = bool(text and visual)
        iou = inter/union if union else None
        if agreement.get("both_nonempty") is not both or agreement.get("query_gt_used") is not False:
            raise ValueError("Invalid empty-mask or query-GT gate evidence")
        if (iou is None and agreement.get("iou") is not None) or (iou is not None and
                (not _finite(agreement.get("iou")) or not math.isclose(iou, agreement["iou"], abs_tol=1e-12))):
            raise ValueError("Gate IoU differs from prediction overlap counts")
        if variant == "agreement_gate":
            threshold = settings.get("gate_threshold", .7)
            if gate.get("threshold") != threshold:
                raise ValueError("Agreement threshold differs from frozen configuration")
            expected = not (both and iou is not None and iou >= threshold)
        else:
            decision = gate.get("random_decision", {})
            probability = settings.get("random_search_probability", .5)
            if decision.get("probability") != probability or not _finite(decision.get("uniform_draw")) or not 0 <= decision["uniform_draw"] < 1:
                raise ValueError("Invalid random-control decision")
            identity_rows = [{"role": "query_image", "sha256": row["query_image_sha256"]}]
            for index, (image_hash, mask_hash) in enumerate(zip(row["reference_image_sha256"], row["_reference_mask_sha256"])):
                identity_rows += [{"role": f"reference_{index}_image", "sha256": image_hash},
                                  {"role": f"reference_{index}_mask", "sha256": mask_hash}]
            identity = canonical_sha256({"schema_version": 1, "inputs": identity_rows})
            from ics.cgics_batch2 import random_search_decision
            reconstructed = random_search_decision(identity, row["episode_seed"], probability)
            if decision != reconstructed:
                raise ValueError("Random gate identity/draw differs from frozen input and seed")
            expected = decision["uniform_draw"] < probability
            if decision.get("requires_search") is not expected:
                raise ValueError("Random-control probability and decision disagree")
    if triggered is not expected:
        raise ValueError("Recorded gate search decision violates its predeclared rule")
    if not triggered and gate.get("final_concept") != gate.get("initial_concept"):
        raise ValueError("Nonsearch route unexpectedly changed concept")


def _verify_gate_probe_masks(output, row, variant):
    if variant not in {"agreement_gate", "random_gate"}:
        return
    from ics.cgics_batch2 import prediction_agreement
    with Image.open(output / row["prediction_path"]) as im:
        primary_shape = np.asarray(im).shape
    masks = []
    for key in ("gate_text", "gate_visual"):
        with Image.open(output / row["alternative_predictions"][key]["prediction_path"]) as im:
            mask = np.asarray(im)
        if mask.shape != primary_shape or not np.isin(mask, [0, 1]).all():
            raise ValueError("Saved gate probe is not a binary aligned original-resolution mask")
        masks.append(mask > 0)
    actual = prediction_agreement(*masks)
    if actual != row["artifacts"]["batch2_gate"].get("agreement"):
        raise ValueError("Gate decision evidence differs from saved prediction probes")


def _episode_identity(row):
    return (row["dataset"], row["split"], row["fold"], row["track"], row["concept_id"],
            row["query_image_sha256"], row["query_mask_sha256"],
            tuple(sorted(zip(row["reference_image_sha256"], row.get("_reference_mask_sha256", [])))),
            row["base_seed"])


def _target_identity(row):
    return (row["query_image_sha256"], row["query_mask_sha256"], row["concept_id"], row["base_seed"])


def _runner_gpu(row):
    names = sorted({v.get("device_name") for v in row.get("cuda_memory", {}).values()
                    if isinstance(v, dict) and v.get("device_name")})
    return "+".join(names) if names else None


def _planned_config(task):
    """Reproduce only the worker's documented owned-server injection/defaults."""
    cfg = json.loads(json.dumps(task["config_data"]))
    adapter = cfg["adapter"]
    for field, value in (("name", cfg["method"]), ("variant", cfg["variant"]), ("seed", cfg["seed"])):
        adapter.setdefault(field, value)
    profile = task.get("_resource_profile", {})
    if task.get("resource") == "cgics_mllm" and profile and profile.get("mllm_mode", "owned") == "owned":
        adapter["endpoint"] = "http://127.0.0.1:1/v1"
        cfg.setdefault("runtime", {}).update(managed_local_mllm=True, mllm_model_path=profile["mllm_model_path"])
    return scientific_config(cfg, validate_model_path=False)


def _manifest_context(path, receipt_cache):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    sidecar_path = path.with_suffix(".protocol.json")
    if not sidecar_path.is_file():
        raise ValueError("Batch2 output lacks frozen protocol/exclusion provenance")
    sidecar = json.loads(sidecar_path.read_text())
    batch2 = sidecar.get("batch2", {})
    if not batch2.get("exclusion_receipt_path") or not batch2.get("prior_exclusion_sha256"):
        raise ValueError("Batch2 protocol lacks prior-query exclusion receipt")
    receipt_path = Path(batch2["exclusion_receipt_path"]).resolve()
    if receipt_path.name != "data_receipt.json":
        raise ValueError("Unexpected exclusion receipt name")
    if receipt_path not in receipt_cache:
        # This checks original binding/manifest hashes, old query byte hashes AND
        # canonical source IDs, all new assets and scope membership exactly once.
        receipt_cache[receipt_path] = verify_scope(receipt_path.parent)
    receipt = receipt_cache[receipt_path]
    entries = [entry for entry in receipt["manifests"].values() if Path(entry["path"]).resolve() == path]
    if len(entries) != 1 or entries[0]["cohort"] != batch2.get("cohort"):
        raise ValueError("Manifest is not uniquely registered in its cohort receipt")
    if (batch2["prior_exclusion_sha256"] != receipt["history"]["exclusion_sha256"] or
            manifest_fingerprint(manifest) != entries[0]["manifest_sha256"]):
        raise ValueError("Manifest prior-exclusion/receipt binding mismatch")
    return manifest, sidecar_path, batch2


def _read_output(path, tasks, receipt_cache):
    path = Path(path).resolve()
    run = load_run(path)
    meta, binding = run["meta"], run["meta"]["binding"]
    if canonical_sha256(scientific_config(meta["config"], validate_model_path=False)) != binding["config_sha256"]:
        raise ValueError(f"Configuration binding mismatch: {path}")
    identity_binding = dict(binding)
    identity_binding.pop("run_id")
    if binding["run_id"] != meta["config"]["experiment_id"] + "-" + canonical_sha256(identity_binding)[:16]:
        raise ValueError("Run ID does not match its complete binding")
    for task in tasks:
        if _planned_config(task) != scientific_config(meta["config"], validate_model_path=False):
            raise ValueError("Output configuration differs from complete frozen task configuration")
        if Path(task["manifest"]).resolve() != Path(meta["manifest_path"]).resolve():
            raise ValueError("Output manifest path differs from frozen task")
    ids = binding["selected_episode_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in original run binding")
    manifest_path = Path(meta["manifest_path"])
    if tasks and manifest_path.resolve() != Path(tasks[0]["manifest"]).resolve():
        raise ValueError("Output manifest path differs from frozen task")
    manifest, sidecar_path, batch2 = _manifest_context(manifest_path, receipt_cache)
    if manifest_fingerprint(manifest) != binding["manifest_sha256"]:
        raise ValueError("Manifest bytes differ from output binding")
    if ids != [ep["episode_id"] for ep in manifest["episodes"]]:
        raise ValueError("Formal output does not cover the complete selected manifest")
    episodes = {ep["episode_id"]: ep for ep in manifest["episodes"]}
    indices = {eid: index for index, eid in enumerate(ids)}
    shard_counts = {t["num_shards"] for t in tasks}
    if len(shard_counts) != 1:
        raise ValueError("Conflicting or absent shard assignment in plan")
    num_shards = next(iter(shard_counts))
    if {t["shard_index"] for t in tasks} != set(range(num_shards)):
        raise ValueError("Frozen plan does not contain every unique formal shard")
    if len(tasks) != num_shards:
        raise ValueError("Duplicate formal shard in plan")
    cohort = batch2["cohort"]
    pools = 0
    for row in run["rows"]:
        ep = episodes[row["episode_id"]]
        if row.get("config_sha256") != binding["config_sha256"]:
            raise ValueError("Episode config binding mismatch")
        expected_identity = {"dataset": ep["dataset"], "split": ep["split"], "fold": ep["fold"],
                             "track": ep["track"], "concept_id": ep["concept_id"],
                             "query_image_sha256": ep["query"]["image_sha256"],
                             "query_mask_sha256": ep["query"]["mask_sha256"],
                             "reference_image_sha256": [r["image_sha256"] for r in ep["references"]]}
        if any(row.get(k) != v for k, v in expected_identity.items()):
            raise ValueError("Episode identity differs from frozen manifest")
        if row.get("num_shards") != num_shards or row.get("shard_index") != indices[row["episode_id"]] % num_shards:
            raise ValueError("Episode modulo shard assignment mismatch")
        if row.get("base_seed") != meta["config"]["seed"]:
            raise ValueError("Episode seed differs from bound configuration")
        row["_reference_mask_sha256"] = [r["mask_sha256"] for r in ep["references"]]
        expected_seed = int(canonical_sha256([meta["config"]["seed"], ep["query"]["image_sha256"]])[:8], 16)
        if row.get("episode_seed") != expected_seed:
            raise ValueError("Episode model seed differs from frozen query seed rule")
        verify_counts(row)
        if row["status"] == "ok":
            _verify_saved_predictions(path, row, meta["config"]["variant"])
            for alternative in row.get("alternative_predictions", {}).values():
                verify_counts(alternative)
            if meta["config"]["variant"] == "fixed_pool_scores":
                verify_score_pool(row)
                pools += 1
            verify_gate(row, meta["config"])
            _verify_gate_probe_masks(path, row, meta["config"]["variant"])
        row["_reference_mask_sha256"] = [r["mask_sha256"] for r in ep["references"]]
        row["_cohort"] = cohort
    seen = {row["episode_id"] for row in run["rows"]}
    run.update(cohort=cohort, protocol_sidecar=str(sidecar_path), batch2_provenance=batch2,
               expected=len(ids), missing_episode_ids=sorted(set(ids) - seen), verified_pools=pools)
    return run


def _job_costs(root, plan):
    """Current worker attempts only; preserve missing measurements as unknown."""
    rows = []
    for task in plan["tasks"]:
        path = root / "jobs" / task["id"] / "status.json"
        if not path.is_file():
            rows.append({"task_id": task["id"], "status": "NO_STATUS", "worker_wall_seconds": None,
                         "allocated_gpu_seconds_proxy": None})
            continue
        value = json.loads(path.read_text())
        if value.get("task_id") != task["id"] or value.get("plan_fingerprint") != plan["fingerprint"]:
            raise ValueError("Worker cost record belongs to a different task/plan")
        start, end = value.get("start"), value.get("end")
        wall = end-start if _finite(start) and _finite(end) and end >= start else None
        assignment = value.get("gpu_assignment")
        count = len(assignment) if isinstance(assignment, list) and assignment else None
        mstart, mready = value.get("mllm_server_start"), value.get("mllm_server_ready")
        startup = mready-mstart if _finite(mstart) and _finite(mready) and mready >= mstart else None
        sample = value.get("gpu_sampling", {})
        rows.append({"task_id": task["id"], "resource": task["resource"], "status": value.get("status"),
                     "slurm_job_id": value.get("slurm_job_id"), "worker_wall_seconds": wall,
                     "allocated_gpus": count, "allocated_gpu_seconds_proxy": wall*count if wall is not None and count is not None else None,
                     "mllm_server_startup_seconds": startup,
                     "sampled_owned_total_peak_mib": sample.get("owned_simultaneous_total_peak_mib"),
                     "sampling_status": sample.get("status"), "memory_samples": sample.get("samples"),
                     "memory_failed_samples": sample.get("failed_samples"), "memory_scope": value.get("gpu_memory_scope"),
                     "source": str(path)})
    return {"run_dir": str(root), "expected_formal_tasks": len(plan["tasks"]), "tasks": rows,
            "worker_wall_seconds_observed": distribution([r.get("worker_wall_seconds") for r in rows]),
            "allocated_gpu_seconds_proxy_observed": distribution([r.get("allocated_gpu_seconds_proxy") for r in rows]),
            "scope": "Current task worker records only; proxy is allocation time, not utilization, energy or monetary bill. Missing values are not zero. Do not add this overlapping total to episode costs."}


def clustered_mean_ci(target_values, *, replicates=1000, seed=6190):
    """Equal target mean; resample image clusters with all targets kept together."""
    if type(replicates) is not int or replicates < 1:
        raise ValueError("Bootstrap replicates must be a positive integer")
    groups = defaultdict(list)
    for target_key, value in target_values.items():
        if not _finite(value):
            raise ValueError("Nonfinite target statistic")
        groups[target_key[0]].append(value)
    n = sum(map(len, groups.values()))
    point = sum(sum(v) for v in groups.values()) / n if n else None
    result = {"mean": point, "query_targets": n, "query_image_clusters": len(groups),
              "ci95": None, "bootstrap_replicates": replicates,
              "resampling_unit": "query image SHA; all references and concepts kept together"}
    if len(groups) < 2:
        result["uncertainty_note"] = "Fewer than two image clusters; interval not estimable"
        return result
    sums = np.array([sum(x) for x in groups.values()])
    counts = np.array([len(x) for x in groups.values()])
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, replicates, 128):
        index = rng.integers(0, len(groups), size=(min(128, replicates-start), len(groups)))
        draws.extend((sums[index].sum(1) / counts[index].sum(1)).tolist())
    result["ci95"] = np.percentile(draws, [2.5, 97.5]).tolist()
    result["interval_scope"] = "Exploratory, no multiplicity adjustment; interval containing zero is not equivalence"
    return result


def _inputs(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "ok" and row.get("iou") is not None:
            groups[_episode_identity(row)].append(row)
    return groups


def _target_values(rows):
    targets = defaultdict(list)
    for copies in _inputs(rows).values():
        targets[_target_identity(copies[0])].append(statistics.mean(r["iou"] for r in copies))
    return {target: statistics.mean(values) for target, values in targets.items()}


def paired_statistics(left, right, *, replicates=1000, seed=6190, predicate=None):
    """Compare common successful exact inputs, then average references per target."""
    li, ri = _inputs(left), _inputs(right)
    deltas, matched_times, gate_groups = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    common = sorted(set(li) & set(ri), key=str)
    accepted = 0
    for key in common:
        aa, bb = li[key], ri[key]
        if predicate and not predicate(aa[0], bb[0]):
            continue
        accepted += 1
        delta = statistics.mean(r["iou"] for r in bb) - statistics.mean(r["iou"] for r in aa)
        target = _target_identity(aa[0])
        deltas[target].append(delta * 100)
        ag, bg = {_runner_gpu(r) for r in aa}, {_runner_gpu(r) for r in bb}
        if len(ag) == len(bg) == 1 and ag == bg and None not in ag:
            av = [r.get("inference_seconds") for r in aa]
            bv = [r.get("inference_seconds") for r in bb]
            if all(_finite(x) and x >= 0 for x in av + bv):
                matched_times[next(iter(ag))].append((statistics.mean(av), statistics.mean(bv)))
        gate = bb[0].get("artifacts", {}).get("batch2_gate", {})
        trigger = _gate_trigger(gate)
        if trigger is not None:
            gate_groups[str(trigger).lower()][target].append(delta * 100)
    estimates = {k: statistics.mean(v) for k, v in deltas.items()}
    timing = []
    for gpu, pairs in sorted(matched_times.items()):
        a, b = distribution([x[0] for x in pairs]), distribution([x[1] for x in pairs])
        timing.append({"runner_gpu_model": gpu, "exact_input_pairs": len(pairs), "left_seconds": a,
                       "right_seconds": b, "ratio_of_means_right_over_left": b["mean"] / a["mean"] if a["mean"] else None,
                       "scope": "Same recorded runner GPU model; MLLM GPU, contention and startup not controlled by this stratum"})
    return {"common_successful_unique_inputs": accepted,
            "left_successful_unique_inputs": len(li), "right_successful_unique_inputs": len(ri),
            "paired_query_delta_pp": clustered_mean_ci(estimates, replicates=replicates, seed=seed),
            "matched_runner_latency": timing,
            "right_policy_trigger_strata": {k: clustered_mean_ci({t: statistics.mean(v) for t, v in d.items()},
                                                                 replicates=replicates, seed=seed)
                                             for k, d in gate_groups.items()},
            "estimand": "Common successful exact inputs; repeated identical inputs averaged, then references per query-target averaged"}


def _class_pooled(rows):
    classes = defaultdict(lambda: Counter())
    for row in rows:
        if row["status"] == "ok":
            classes[(row["fold"], row["concept_id"])].update(row["counts"])
    folds = defaultdict(list)
    for (fold, _), c in classes.items():
        union = c["tp"] + c["fp"] + c["fn"]
        if union:
            folds[fold].append(c["tp"] / union)
    return statistics.mean(statistics.mean(v) for v in folds.values()) if folds else None


def analyze_batch2(run_dirs, output, *, bootstrap_replicates=1000):
    """Accept queue run roots with frozen plan.json; partial outputs stay partial."""
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    outputs, plans, missing_outputs, worker_costs = {}, [], [], []
    receipt_cache = {}
    for root in sorted({Path(p).resolve() for p in run_dirs}):
        plan = json.loads((root / "plan.json").read_text())
        plans.append({"run_dir": str(root), "fingerprint": plan["fingerprint"], "formal_tasks": len(plan["tasks"])})
        worker_costs.append(_job_costs(root, plan))
        for original in plan["tasks"]:
            task = dict(original)
            task["_resource_profile"] = plan.get("resources", {}).get(task["resource"], {})
            task["_scope"] = plan.get("batch2", {}).get("scope")
            path = Path(task["output"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Formal output escapes its queue run")
            outputs.setdefault(path, []).append(task)
    runs, bindings = [], set()
    series, expected, contracts, rows_by_output = defaultdict(list), Counter(), {}, []
    unknown_expected = set()
    for path, tasks in sorted(outputs.items(), key=lambda x: str(x[0])):
        cfg = _planned_config(tasks[0])
        if any(_planned_config(task) != cfg for task in tasks):
            raise ValueError("Conflicting frozen configurations share an output directory")
        manifest_path = Path(tasks[0]["manifest"])
        if any(Path(t["manifest"]).resolve() != manifest_path.resolve() for t in tasks):
            raise ValueError("Conflicting frozen manifests share an output directory")
        if manifest_path.is_file():
            manifest, _, batch2 = _manifest_context(manifest_path, receipt_cache)
            datasets = Counter(ep["dataset"] for ep in manifest["episodes"])
            cohort, expected_count = batch2["cohort"], len(manifest["episodes"])
        else:
            datasets = {tasks[0].get("dataset", "UNKNOWN"): 0}
            cohort = "fresh" if tasks[0].get("_scope") == "core" else "transfer" if tasks[0].get("_scope") in {"suim", "isic"} else "UNKNOWN"
            expected_count = None
        roles = ["combined", "ref_only", "query_only"] if cfg["variant"] == "fixed_pool_scores" else [None]
        for role in roles:
            label = cfg["experiment_id"] + (":" + role if role else "")
            signature = canonical_sha256(cfg)
            if label in contracts and contracts[label]["config_sha256"] != signature:
                raise ValueError("One experiment label has multiple scientific configurations")
            contracts[label] = {"config_sha256": signature, "variant": cfg["variant"], "adapter": cfg["adapter"],
                                "base_seed": cfg["seed"], "nested_rule": role,
                                "cost_eligible": cfg["variant"] != "fixed_pool_scores"}
            for dataset, number in datasets.items():
                key = (label, dataset, cohort)
                expected[key] += number
                if expected_count is None:
                    unknown_expected.add(key)
        if not (path / "run.json").is_file():
            missing_outputs.append({"output": str(path), "experiment": cfg["experiment_id"], "datasets": list(datasets),
                                    "cohort": cohort, "formal_shards": len(tasks), "expected_episodes": expected_count,
                                    "status": "NO_OUTPUT_BINDING", "counts_as_success": False})
            continue
        run = _read_output(path, tasks, receipt_cache)
        identity = canonical_sha256(run["meta"]["binding"])
        if identity in bindings:
            raise ValueError("Duplicate output binding in different directories; refuse double counting")
        bindings.add(identity)
        runs.append(run)
        for role in roles:
            label = cfg["experiment_id"] + (":" + role if role else "")
            for source in run["rows"]:
                row = dict(source)
                if role and role != "combined" and row["status"] == "ok":
                    row.update(source["alternative_predictions"][role])
                if role:
                    row.pop("inference_seconds", None)  # Never assign joint cost to one rule.
                    row.pop("cuda_memory", None)
                series[(label, row["dataset"], run["cohort"])].append(row)
        rows_by_output.append({"path": str(run["path"]), "run_id": run["meta"]["binding"]["run_id"],
                               "expected": run["expected"], "observed": len(run["rows"]),
                               "ok": sum(r["status"] == "ok" for r in run["rows"]),
                               "errors": sum(r["status"] == "error" for r in run["rows"]),
                               "missing_episode_ids": run["missing_episode_ids"], "verified_fixed_pools": run["verified_pools"],
                               "protocol_sidecar": run["protocol_sidecar"], "batch2_provenance": run["batch2_provenance"],
                               "joint_inference_seconds": distribution([r.get("inference_seconds") for r in run["rows"]]),
                               "runner_model_load_seconds_observed": distribution([
                                   json.loads(p.read_text()).get("model_load_seconds")
                                   for p in (run["path"] / "shards").glob("*.json")]),
                               "archived_failed_attempts": len(run["prior_attempts"]),
                               "failed_attempt_wall_seconds": distribution([r.get("failed_inference_wall_seconds")
                                                                           for r in run["rows"] + run["prior_attempts"] if r["status"] == "error"])})
    summaries, comparisons, cost_frontier = [], [], []
    for key in sorted(expected):
        label, dataset, cohort = key
        rows = series[key]
        good = [r for r in rows if r["status"] == "ok"]
        failed = sum(r["status"] == "error" for r in rows)
        contract = contracts[label]
        target_values = _target_values(rows)
        gate_values = [r.get("artifacts", {}).get("batch2_gate", {}) for r in good]
        triggers = [_gate_trigger(g) for g in gate_values if _gate_trigger(g) is not None]
        summary = {"experiment": label, "dataset": dataset, "cohort": cohort, "contract": contract,
                   "expected": None if key in unknown_expected else expected[key], "observed": len(rows), "ok": len(good), "errors": failed,
                   "missing": None if key in unknown_expected else expected[key] - len(rows), "status": "COMPLETE" if key not in unknown_expected and len(rows) == expected[key] and not failed else "CONDITIONAL_OR_INCOMPLETE",
                   "episode_mean_iou": distribution([r.get("iou") for r in good])["mean"],
                   "mean_fold_class_pooled_iou": _class_pooled(rows),
                   "equal_query_target_iou": clustered_mean_ci(target_values, replicates=bootstrap_replicates),
                   "unique_successful_inputs": len(_inputs(rows)),
                   "trigger_rate_successful_observed": sum(triggers)/len(triggers) if triggers else None,
                   "gate_observations": len(triggers), "agreement_iou": distribution([g.get("agreement_iou") for g in gate_values]),
                   "inference_seconds": distribution([r.get("inference_seconds") for r in good]) if contract["cost_eligible"] else None,
                   "cost_scope": "All probe/fallback work included in actual adapter wall; startup/evaluation/output excluded" if contract["cost_eligible"] else "Shared 3-rule audit cost only; no per-rule end-to-end latency claim"}
        summaries.append(summary)
        if contract["cost_eligible"]:
            by_gpu = defaultdict(list)
            for row in good:
                by_gpu[_runner_gpu(row) or "UNRECORDED"].append(row)
            for gpu, subset in sorted(by_gpu.items()):
                cost_frontier.append({"experiment": label, "dataset": dataset, "cohort": cohort, "runner_gpu_model": gpu,
                                      "episodes": len(subset), "episode_iou": distribution([r.get("iou") for r in subset]),
                                      "inference_seconds": distribution([r.get("inference_seconds") for r in subset]),
                                      "mllm_requests": distribution([r.get("telemetry", {}).get("mllm_requests") for r in subset]),
                                      "sam_calls": distribution([r.get("telemetry", {}).get("sam_calls") for r in subset])})
    for a, b in itertools.combinations(sorted(expected), 2):
        if a[1:] != b[1:] or contracts[a[0]]["base_seed"] != contracts[b[0]]["base_seed"]:
            continue
        gate_variants = {"agreement_gate", "random_gate", "initial_score_gate"}
        if contracts[a[0]]["variant"] in gate_variants and contracts[b[0]]["variant"] not in gate_variants:
            a, b = b, a  # Baseline left, gate right: gate−baseline and its trigger strata.
        result = paired_statistics(series[a], series[b], replicates=bootstrap_replicates)
        if result["common_successful_unique_inputs"]:
            comparisons.append({"left": a[0], "right": b[0], "dataset": a[1], "cohort": a[2], **result})
    missing_rows = sum(len(r["missing_episode_ids"]) for r in runs)
    total_rows = sum(len(r["rows"]) for r in runs)
    errors = sum(r["status"] == "error" for run in runs for r in run["rows"])
    report = {"schema_version": 1, "analysis_kind": "BATCH2_AUDIT_AND_PAIRED_ANALYSIS", "plans": plans,
              "status": "INCOMPLETE" if missing_outputs or missing_rows else "COMPLETE_WITH_FAILURES" if errors else "COMPLETE",
              "status_scope": "Episode output coverage only; does not assert manager or Slurm terminal state",
              "formal_observed_records": total_rows, "formal_ok": total_rows-errors, "formal_errors": errors,
              "formal_missing_records_in_bound_outputs": missing_rows, "outputs_without_binding": missing_outputs,
              "formal_output_coverage": rows_by_output, "configuration_results": summaries,
              "paired_comparisons": comparisons, "observed_cost_points": cost_frontier,
              "current_worker_costs": worker_costs,
              "predeclared_policy": {"primary_agreement_threshold": .7, "sensitivity_thresholds": [.5, .9],
                                     "random_control_probabilities": [.25, .5, .75],
                                     "selection": "All registered configurations reported. No test-driven winner or threshold selection.",
                                     "random_scope": "Fixed probability controls; not falsely labeled exact matched-trigger-rate controls"},
              "limitations": ["SHA checks and count arithmetic do not re-evaluate every prediction PNG against GT; gate prediction overlap is recomputed from saved probes.",
                              "Inference excludes model/server setup, GT evaluation and output writes; startup/job memory remain separate metadata.",
                              "Same runner GPU names do not establish identical MLLM GPUs or contention.",
                              "Errors are null, never zeros; comparisons condition on common successful inputs.",
                              "Intervals cluster query images, not reference trials; exploratory intervals have no multiplicity correction.",
                              "A transfer benchmark is a new domain, not proof of patient-disjointness or complete author-protocol replication."]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "batch2_analysis.json", report)
    lines = ["# CE6190 batch-2 results", "", f"Status: **{report['status']}**. Formal records: {total_rows}; errors: {errors}; missing bound rows: {missing_rows}; unbound outputs: {len(missing_outputs)}.",
             "", "Nested scoring alternatives are paired controls, not extra formal episodes. No test-based configuration selection.",
             "", "| Configuration | Dataset / cohort | OK / expected | Errors | Episode IoU | Equal-query IoU | Mean inference s |", "|---|---|---:|---:|---:|---:|---:|"]
    def fmt(v, scale=1):
        return "—" if v is None else f"{v*scale:.3f}"
    for s in summaries:
        lines.append(f"| {s['experiment']} | {s['dataset']} / {s['cohort']} | {s['ok']} / {s['expected']} | {s['errors']} | {fmt(s['episode_mean_iou'],100)} | {fmt(s['equal_query_target_iou']['mean'],100)} | {fmt(s['inference_seconds']['mean']) if s['inference_seconds'] else 'shared audit'} |")
    lines += ["", "## Interpretation limits", ""] + ["- " + x for x in report["limitations"]]
    lines += ["", "Complete paired differences, image-cluster intervals, trigger strata, actual cost distributions and exact output coverage are in batch2_analysis.json.", ""]
    (output / "BATCH2_RESULTS.md").write_text("\n".join(lines))
    return report
