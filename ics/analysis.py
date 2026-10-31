"""Analyze actual runner outputs without merging incompatible protocols.

Successful-episode metrics are labeled conditional when errors/pending work
exist. No failed prediction is substituted with a fabricated mask or IoU.
"""
from __future__ import annotations

import csv
import html
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ics_eval import summarize_counts
from ics.runner import atomic_json, canonical_sha256, file_sha256, scientific_config


SCOPE = ("dataset", "split", "fold", "track", "nshot")


def scope_key(row):
    return tuple(row[k] for k in SCOPE)


def mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(values) if values else None


def load_run(path):
    path = Path(path).resolve()
    meta = json.loads((path / "run.json").read_text())
    expected = set(meta["binding"]["selected_episode_ids"])
    rows, seen = [], set()
    for result in sorted((path / "episodes").glob("*.json")):
        row = json.loads(result.read_text())
        if row["run_id"] != meta["binding"]["run_id"]:
            raise ValueError(f"Foreign run result in {result}")
        if row["episode_id"] not in expected or row["episode_id"] in seen:
            raise ValueError(f"Unexpected/duplicate episode in {result}")
        if row["manifest_sha256"] != meta["binding"]["manifest_sha256"]:
            raise ValueError("Manifest fingerprint differs within run")
        if row["status"] not in {"ok", "error"}:
            raise ValueError("Unknown episode status")
        if row["status"] == "ok":
            pred = (path / row["prediction_path"]).resolve()
            if not pred.is_relative_to(path):
                raise ValueError("Prediction path escapes run directory")
            if file_sha256(pred) != row["prediction_sha256"]:
                raise ValueError(f"Prediction digest mismatch: {pred}")
        seen.add(row["episode_id"])
        rows.append(row)
    # A worker may die after archival but before replacing the current error.
    # Deduplicate by full row digest so that attempt is never counted twice.
    digests = {canonical_sha256(row) for row in rows}
    prior_attempts = []
    for archive in sorted((path / "attempts").glob("*/*.json")):
        row = json.loads(archive.read_text())
        if row["run_id"] != meta["binding"]["run_id"] or row["episode_id"] not in expected:
            raise ValueError(f"Foreign retry attempt in {archive}")
        if row["status"] != "error":
            raise ValueError("Only failed attempts may be retried/archived")
        digest = canonical_sha256(row)
        if archive.stem != digest:
            raise ValueError(f"Archived retry attempt was modified: {archive}")
        if digest not in digests:
            prior_attempts.append(row)
            digests.add(digest)
    return {"path": path, "meta": meta, "rows": rows, "prior_attempts": prior_attempts}


def reference_statistics(rows):
    """Per query+concept spread, never STD over unrelated query images."""
    groups = defaultdict(list)
    for row in rows:
        # base_seed differentiates repeated stochastic runs; fixed configuration
        # here means changing references is the only episode-level intervention.
        key = (row["dataset"], row["split"], row["fold"], row["track"], row["nshot"],
               row["concept_id"], row["query_image_sha256"], row["query_mask_sha256"])
        groups[key].append(row)
    out = []
    for key, samples in sorted(groups.items(), key=lambda x: str(x[0])):
        observed = [r for r in samples if r["status"] == "ok" and r["iou"] is not None]
        references = [tuple(sorted(r["reference_image_sha256"])) for r in samples]
        distinct = len(set(references))
        # Repeated references are not independent reference observations.
        unique_good = {}
        for row in observed:
            ref_key = tuple(sorted(row["reference_image_sha256"]))
            unique_good.setdefault(ref_key, []).append(row["iou"])
        means = [statistics.mean(v) for v in unique_good.values()]
        average = mean(means)
        std = statistics.pstdev(means) if len(means) >= 2 else None
        out.append({"dataset": key[0], "split": key[1], "fold": key[2],
                    "track": key[3], "nshot": key[4], "concept_id": key[5],
                    "query_image_sha256": key[6], "query_mask_sha256": key[7],
                    "observed_episodes": len(samples), "distinct_reference_sets": distinct,
                    "successful_reference_sets": len(means),
                    "error_episodes": sum(r["status"] == "error" for r in samples),
                    "mean_reference_iou": average, "population_std_across_references": std,
                    "cv_across_references": std / average if std is not None and average else None,
                    "worst_observed_reference_iou": min(means) if means else None,
                    "best_observed_reference_iou": max(means) if means else None,
                    "note": "Observed distinct reference sets only; check run coverage before interpreting robustness."})
    return out


def _numeric_leaves(obj, prefix=""):
    result = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            result.update(_numeric_leaves(value, f"{prefix}.{key}" if prefix else key))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and math.isfinite(obj):
        result[prefix] = obj
    return result


def summarize_run(run):
    meta, rows = run["meta"], run["rows"]
    prior_attempts = run.get("prior_attempts", [])
    groups = defaultdict(list)
    expected_groups = defaultdict(int)
    for ep in meta.get("expected_episode_scopes", []):
        expected_groups[scope_key(ep)] += 1
    for row in rows:
        groups[scope_key(row)].append(row)
    summaries = []
    for key in sorted(set(expected_groups) | set(groups), key=str):
        group = groups[key]
        good = [r for r in group if r["status"] == "ok"]
        failed = sum(r["status"] == "error" for r in group)
        expected = expected_groups.get(key, len(group))
        pending = expected - len(group)
        metrics = summarize_counts(good) if good else None
        costs = defaultdict(list)
        for row in good:
            for metric, value in _numeric_leaves(row.get("telemetry", {})).items():
                costs[metric].append(value)
        all_attempts = group + [r for r in prior_attempts if scope_key(r) == key]
        all_costs = defaultdict(list)
        for row in all_attempts:
            telemetry = row.get("telemetry", {}) if row["status"] == "ok" else row.get("failure_diagnostics", {}).get("telemetry", {})
            for metric, value in _numeric_leaves(telemetry).items():
                all_costs[metric].append(value)
        summaries.append({"scope": dict(zip(SCOPE, key)), "expected_episodes": expected,
                          "ok": len(good), "errors": failed, "pending": pending,
                          "coverage": len(good) / expected if expected else None,
                          "error_rate_among_attempted": failed / len(group) if group else None,
                          "metrics_status": "complete" if not failed and not pending else "conditional_on_successful_episodes",
                          "metrics": metrics,
                          "inference_seconds_mean": mean([r.get("inference_seconds") for r in good]),
                          "episode_wall_seconds_mean_success": mean([r.get("episode_wall_seconds") for r in good]),
                          "committed_attempts_observed": len(all_attempts),
                          "failed_attempts_observed": sum(r["status"] == "error" for r in all_attempts),
                          "episode_wall_seconds_total_latest_attempts": sum(r.get("episode_wall_seconds", 0) for r in group),
                          "episode_wall_seconds_total_all_committed_attempts": sum(r.get("episode_wall_seconds", 0) for r in all_attempts),
                          "cost_scope": "Latest-result accuracy; cumulative cost includes immutable archived failed attempts. Interrupted attempts without a committed row are not measurable here.",
                          "telemetry_means": {k: mean(v) for k, v in costs.items()},
                          "telemetry_observed_counts": {k: len(v) for k, v in costs.items()},
                          "all_attempt_numeric_telemetry_totals": {k: sum(v) for k, v in all_costs.items()},
                          "all_attempt_numeric_telemetry_observed_counts": {k: len(v) for k, v in all_costs.items()}})
    return {"run_id": meta["binding"]["run_id"], "experiment_id": meta["config"]["experiment_id"],
            "method": meta["config"]["method"], "variant": meta["config"]["variant"],
            "base_seed": meta["config"]["seed"], "manifest_sha256": meta["binding"]["manifest_sha256"],
            "expected": meta["expected_episodes"], "observed": len(rows),
            "ok": sum(r["status"] == "ok" for r in rows),
            "errors": sum(r["status"] == "error" for r in rows),
            "pending": meta["expected_episodes"] - len(rows), "groups": summaries,
            "archived_failed_attempts": len(prior_attempts),
            "total_committed_attempts": len(rows) + len(prior_attempts),
            "fold_aggregation": "Fold results kept separate. No automatic cross-dataset or unlabeled fold average.",
            "memory_scope": "Each row reports current Python-process CUDA allocator peaks; external MLLM memory excluded."}


def paired_comparisons(runs):
    out = []
    for left, right in itertools.combinations(runs, 2):
        lm, rm = left["meta"], right["meta"]
        lb, rb = lm["binding"], rm["binding"]
        if lm["config"]["seed"] != rm["config"]["seed"]:
            continue  # Stochastic repeats are reported separately below.
        if (lb["manifest_sha256"], lb["protocol_id"], lb["evaluation_protocol"]) != (
                rb["manifest_sha256"], rb["protocol_id"], rb["evaluation_protocol"]):
            continue
        lr = {r["episode_id"]: r for r in left["rows"]}
        rr = {r["episode_id"]: r for r in right["rows"]}
        expected_ids = set(lb["selected_episode_ids"]) & set(rb["selected_episode_ids"])
        paired = defaultdict(list)
        for eid in sorted(expected_ids & set(lr) & set(rr)):
            a, b = lr[eid], rr[eid]
            if scope_key(a) != scope_key(b) or a["query_image_sha256"] != b["query_image_sha256"]:
                raise ValueError("Same episode id has inconsistent identity")
            if a["reference_image_sha256"] != b["reference_image_sha256"]:
                raise ValueError("Same episode id has different references")
            if a["status"] == b["status"] == "ok" and a["iou"] is not None and b["iou"] is not None:
                paired[scope_key(a)].append((a, b))
        scope_expected = defaultdict(int)
        for scope in lm.get("expected_episode_scopes", []):
            if scope["episode_id"] in expected_ids:
                scope_expected[scope_key(scope)] += 1
        for scope in sorted(set(paired) | set(scope_expected), key=str):
            pairs = paired[scope]
            query_deltas = defaultdict(list)
            for a, b in pairs:
                query_deltas[(a["concept_id"], a["query_image_sha256"], a["query_mask_sha256"])].append(b["iou"] - a["iou"])
            out.append({"left_run": lb["run_id"], "right_run": rb["run_id"],
                        "scope": dict(zip(SCOPE, scope)), "expected_common_episodes": scope_expected.get(scope),
                        "valid_paired_episodes": len(pairs), "valid_query_clusters": len(query_deltas),
                        "mean_episode_delta_right_minus_left": mean([b["iou"] - a["iou"] for a, b in pairs]),
                        "mean_query_delta_right_minus_left": mean([mean(v) for v in query_deltas.values()]),
                        "right_wins": sum(b["iou"] > a["iou"] for a, b in pairs),
                        "ties": sum(b["iou"] == a["iou"] for a, b in pairs),
                        "left_wins": sum(b["iou"] < a["iou"] for a, b in pairs),
                        "interpretation": "Paired observed successes; unequal/missing failures require coverage review. Not a significance test."})
    return out


def repeated_input_statistics(runs):
    """Keep stochastic repeat variation distinct from reference variation."""
    groups = defaultdict(list)
    for run in runs:
        # Local server ports differ between workers. The runner already hashed
        # the served model; analyzing downloaded logs need not have model bytes.
        cfg = scientific_config(run["meta"]["config"], validate_model_path=False)
        cfg.pop("seed", None)
        cfg.pop("experiment_id", None)
        cfg.pop("output", None)
        cfg["adapter"] = dict(cfg["adapter"])
        cfg["adapter"].pop("seed", None)
        signature = canonical_sha256(cfg)
        for row in run["rows"]:
            if row["status"] == "ok" and row["iou"] is not None:
                key = (signature, row["manifest_sha256"], row["episode_id"])
                groups[key].append(row)
    result = []
    for key, rows in groups.items():
        seeds = {r["base_seed"] for r in rows}
        if len(seeds) >= 2:
            result.append({"config_without_seed_sha256": key[0], "manifest_sha256": key[1],
                           "episode_id": key[2], "seeds": sorted(seeds),
                           "mean_iou": mean([r["iou"] for r in rows]),
                           "population_std_repeated_input": statistics.pstdev(r["iou"] for r in rows)})
    return result


def area_statistics(rows, thresholds=(.01, .1)):
    """Foreground area / valid query pixels; thresholds fixed before analysis.

    These bins describe the union foreground mask, not COCO instance-area AP
    categories. Runtime failures are counted in their bin whenever GT is valid.
    """
    thresholds = tuple(float(t) for t in thresholds)
    if any(not math.isfinite(t) or not 0 < t < 1 for t in thresholds) or tuple(sorted(set(thresholds))) != thresholds:
        raise ValueError("Area thresholds must be strictly increasing fractions inside (0,1)")
    boundaries = (0., *thresholds, 1.)
    groups = defaultdict(list)
    for row in rows:
        area = row.get("target_area_fraction")
        if area is None or not math.isfinite(area) or not 0 <= area <= 1:
            index = None
        else:
            index = next((i for i, upper in enumerate(boundaries[1:]) if area < upper), len(boundaries)-2)
        groups[(scope_key(row), index)].append(row)
    result = []
    for (scope, index), group in sorted(groups.items(), key=lambda item: str(item[0])):
        good = [r for r in group if r["status"] == "ok" and r["iou"] is not None]
        failed = sum(r["status"] == "error" for r in group)
        lower, upper = (None, None) if index is None else (boundaries[index], boundaries[index+1])
        result.append({"scope": dict(zip(SCOPE, scope)), "bin_index": index,
                       "area_lower_inclusive": lower, "area_upper": upper,
                       "upper_inclusive": index == len(boundaries)-2 if index is not None else None,
                       "observed_episodes": len(group), "valid_iou_episodes": len(good), "errors": failed,
                       "mean_episode_iou": mean([r["iou"] for r in good]),
                       "mean_inference_seconds_success": mean([r.get("inference_seconds") for r in good]),
                       "note": "Union foreground fraction, original query valid pixels. Missing/pending cases require run coverage review; accuracy conditional on successful predictions."})
    return result


def write_csv(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def _overlay(image, mask, color):
    image = np.asarray(image.convert("RGB")).copy()
    selected = np.asarray(mask).astype(bool)
    image[selected] = (image[selected] * .5 + np.asarray(color) * .5).astype(np.uint8)
    return Image.fromarray(image)


def make_case_images(run, output, per_extreme=3):
    """Deterministic best/worst observed examples, clearly selected by IoU."""
    manifest_path = Path(run["meta"]["manifest_path"])
    if not manifest_path.is_file():
        return [], ["Manifest unavailable; case images not generated."]
    manifest = json.loads(manifest_path.read_text())
    episodes = {ep["episode_id"]: ep for ep in manifest["episodes"]}
    data_root = Path(run["meta"]["binding"]["dataset_root"])
    good = sorted([r for r in run["rows"] if r["status"] == "ok" and r["iou"] is not None],
                  key=lambda r: (r["iou"], r["episode_id"]))
    choices = [("lowest_observed_iou", r) for r in good[:per_extreme]]
    choices += [("highest_observed_iou", r) for r in reversed(good[-per_extreme:])]
    out, warnings, seen = [], [], set()
    for category, row in choices:
        if row["episode_id"] in seen:
            continue
        seen.add(row["episode_id"])
        ep = episodes[row["episode_id"]]
        try:
            query = Image.open(data_root / ep["query"]["image"]).convert("RGB")
            truth = np.asarray(Image.open(data_root / ep["query"]["mask"])) == 1
            pred = np.asarray(Image.open(run["path"] / row["prediction_path"])) == 1
            reference = Image.open(data_root / ep["references"][0]["image"]).convert("RGB")
            reference_mask = np.asarray(Image.open(data_root / ep["references"][0]["mask"])) == 1
            panels = [(_overlay(reference, reference_mask, (0, 220, 0)), "Reference + mask"),
                      (query, "Query"), (_overlay(query, truth, (0, 220, 0)), "Query GT (evaluation only)"),
                      (_overlay(query, pred, (255, 90, 0)), "Prediction")]
            canvas = Image.new("RGB", (4 * 320, 360), "white")
            draw = ImageDraw.Draw(canvas)
            for i, (panel, label) in enumerate(panels):
                panel.thumbnail((310, 310))
                canvas.paste(panel, (i * 320 + (320-panel.width)//2, 30 + (310-panel.height)//2))
                draw.text((i * 320 + 8, 8), label, fill="black")
            name = canonical_sha256([run["meta"]["binding"]["run_id"], row["episode_id"]])[:24] + ".png"
            destination = output / "cases" / name
            destination.parent.mkdir(exist_ok=True)
            canvas.save(destination)
            out.append({"run_id": run["meta"]["binding"]["run_id"], "episode_id": row["episode_id"],
                        "selection": category, "iou": row["iou"], "image": "cases/" + name})
        except (OSError, ValueError) as exc:
            warnings.append(f"Case {row['episode_id']} unavailable: {exc}")
    return out, warnings


def analyze_runs(paths, output, *, case_images=3, area_thresholds=(.01, .1)):
    paths = sorted(set(Path(p).resolve() for p in paths), key=str)
    if not paths:
        raise ValueError("No run.json files found")
    runs = [load_run(p) for p in paths]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries = [summarize_run(run) for run in runs]
    refs = [dict(run_id=run["meta"]["binding"]["run_id"], **r)
            for run in runs for r in reference_statistics(run["rows"])]
    pairs = paired_comparisons(runs)
    repeats = repeated_input_statistics(runs)
    areas = [dict(run_id=run["meta"]["binding"]["run_id"], **row)
             for run in runs for row in area_statistics(run["rows"], area_thresholds)]
    report = {"status": "analysis_of_observed_runs", "runs": summaries,
              "paired_comparisons": pairs, "reference_statistics": refs,
              "repeated_input_statistics": repeats,
              "target_area_thresholds": list(area_thresholds), "target_area_statistics": areas,
              "cautions": ["Metrics on successes are conditional if errors/pending episodes exist.",
                           "Accuracy uses the latest committed attempt per episode; cumulative cost includes archived failed attempts.",
                           "Reference STD and stochastic repeat STD are separate analyses.",
                           "No fold/dataset average or hypothesis significance is assumed.",
                           "Different process GPU peaks are not summed into simultaneous system memory."]}
    cases, warnings = [], []
    if case_images:
        for run in runs:
            new_cases, new_warnings = make_case_images(run, output, case_images)
            cases += new_cases
            warnings += new_warnings
    report["cases"] = cases
    report["warnings"] = warnings
    atomic_json(output / "analysis.json", report)
    flat = []
    for summary in summaries:
        for group in summary["groups"]:
            metric = group["metrics"] or {}
            flat.append({"run_id": summary["run_id"], "experiment_id": summary["experiment_id"], "method": summary["method"],
                         "variant": summary["variant"], **group["scope"],
                         **{k: group[k] for k in ("expected_episodes", "ok", "errors", "pending", "coverage", "metrics_status", "inference_seconds_mean")},
                         **{k: metric.get(k) for k in ("mean_episode_iou", "class_mean_episode_iou", "macro_class_pooled_iou")}})
    write_csv(output / "metrics.csv", flat)
    write_csv(output / "reference_stability.csv", refs)
    write_csv(output / "paired_comparisons.csv", pairs)
    write_csv(output / "repeated_inputs.csv", repeats)
    write_csv(output / "target_area.csv", areas)
    errors = [dict(run_id=run["meta"]["binding"]["run_id"], episode_id=r["episode_id"],
                   error_type=r.get("error_type"), error=r.get("error"))
              for run in runs for r in run["rows"] if r["status"] == "error"]
    write_csv(output / "failures.csv", errors)
    all_failed_attempts = [dict(run_id=run["meta"]["binding"]["run_id"],
                               episode_id=row["episode_id"], attempt_number=row.get("attempt_number"),
                               episode_wall_seconds=row.get("episode_wall_seconds"),
                               error_type=row.get("error_type"), error=row.get("error"),
                               failure_diagnostics=row.get("failure_diagnostics", {}))
                           for run in runs for row in [*run["rows"], *run.get("prior_attempts", [])]
                           if row["status"] == "error"]
    write_csv(output / "failed_attempts.csv", all_failed_attempts)
    for run in runs:
        export = output / ("episodes-" + canonical_sha256(run["meta"]["binding"]["run_id"])[:16] + ".jsonl")
        export.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in run["rows"]))
        archive_export = output / ("prior-attempts-" + canonical_sha256(run["meta"]["binding"]["run_id"])[:16] + ".jsonl")
        archive_export.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in run.get("prior_attempts", [])))
    headers = ["experiment_id", "method", "variant", "dataset", "fold", "ok", "errors", "pending", "mean_episode_iou", "macro_class_pooled_iou", "inference_seconds_mean"]
    table = "<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead><tbody>"
    for row in flat:
        table += "<tr>" + "".join(f"<td>{html.escape(str(row.get(h)))}</td>" for h in headers) + "</tr>"
    table += "</tbody></table>"
    case_html = "".join(f'<figure><img src="{html.escape(c["image"])}"><figcaption>{html.escape(c["run_id"])} | {html.escape(c["episode_id"])} | {c["selection"]} | IoU={c["iou"]:.4f}</figcaption></figure>' for c in cases)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>ICS experiment analysis</title><style>body{font:15px system-ui;margin:28px;color:#18232c}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #cbd5df}th{background:#e7eef4}img{max-width:100%}figure{margin:24px 0}p{max-width:1000px}</style><h1>ICS experiment analysis</h1><p>Observed results only. Accuracy is conditional on successful episodes when errors or pending work exist. Folds and task tracks remain separate. Best/worst cases below are deliberately selected by observed IoU, not a random qualitative sample.</p>'
    page += table + case_html + '<p>Detailed metrics, reference stability, paired deltas, runtime telemetry and failures are available in analysis.json and the adjacent CSV files.</p></html>'
    (output / "index.html").write_text(page)
    return {"output": str(output), "runs": len(runs), "rows": sum(len(r["rows"]) for r in runs),
            "errors": len(errors), "pending": sum(s["pending"] for s in summaries)}
