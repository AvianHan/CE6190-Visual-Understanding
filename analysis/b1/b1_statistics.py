#!/usr/bin/env python3
"""B1-only descriptive and paired statistics from immutable exported evidence.

This script does not replace the separate scientific provenance/pixel audit.
Run: python3 b1_statistics.py --evidence ../b1_evidence --output .
Dependencies: Python 3.9+ and NumPy. No cluster connection or mutations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

RULES = ("combined", "ref_only", "query_only")
PAIRS = (("combined", "ref_only"), ("combined", "query_only"), ("query_only", "ref_only"))
SCORE_COLUMNS = {"combined": "combined_score", "ref_only": "ref_iou", "query_only": "query_presence"}


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def population_std(values):
    return float(np.std(values))


def median(values):
    return float(np.median(values))


def alt(row, rule):
    return row if rule == "combined" else row["alternative_predictions"][rule]


def target_key(row):
    return (row["dataset"], row["split"], row["fold"], row["track"],
            row["query_image_sha256"], row["query_mask_sha256"], row["concept_id"], row["base_seed"])


def ci(values, clusters, replicates, seed, strata=None):
    """Resample query-image clusters, keeping every target/reference paired.

    Dataset bootstrap matches the frozen batch2_analysis.py estimator. Optional
    fixed-fold stratification is separately labeled sensitivity analysis.
    """
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[cluster].append(float(value))
    keys = sorted(grouped)
    sums = np.array([sum(grouped[k]) for k in keys])
    counts = np.array([len(grouped[k]) for k in keys])
    result = {"mean": float(sums.sum() / counts.sum()), "query_targets": len(values),
              "query_image_clusters": len(keys), "ci95": None}
    if len(keys) < 2:
        return result
    rng = np.random.default_rng(seed)
    draws = []
    if strata is None:
        for start in range(0, replicates, 128):
            idx = rng.integers(0, len(keys), size=(min(128, replicates-start), len(keys)))
            draws.extend((sums[idx].sum(1) / counts[idx].sum(1)).tolist())
    else:
        by_stratum = defaultdict(set)
        for cluster, stratum in zip(clusters, strata):
            by_stratum[stratum].add(cluster)
        key_to_index = {key: i for i, key in enumerate(keys)}
        groups = [[key_to_index[k] for k in sorted(v)] for _, v in sorted(by_stratum.items())]
        for start in range(0, replicates, 128):
            size = min(128, replicates-start)
            fold_draws = []
            for group in groups:
                idx = np.asarray(group)[rng.integers(0, len(group), size=(size, len(group)))]
                fold_draws.append(sums[idx].sum(1) / counts[idx].sum(1))
            draws.extend(np.mean(fold_draws, axis=0).tolist())
    result["ci95"] = np.percentile(draws, [2.5, 97.5]).tolist()
    return result


def class_pooled(rows, rule):
    classes = defaultdict(Counter)
    for row in rows:
        classes[(row["fold"], row["concept_id"])].update(alt(row, rule)["counts"])
    folds = defaultdict(list)
    for (fold, _), counts in classes.items():
        union = counts["tp"] + counts["fp"] + counts["fn"]
        if union:
            folds[fold].append(counts["tp"] / union)
    return mean(mean(v) for v in folds.values())


def diagnostic(rows):
    fallback = sum(r["artifacts"]["batch2_score_audit"]["common_fallback"]["active"] for r in rows)
    ties = {}
    for rule, col in SCORE_COLUMNS.items():
        n = 0
        nonfallback_n = 0
        for row in rows:
            audit = row["artifacts"]["batch2_score_audit"]
            vals = [c[col] for c in audit["candidates"]]
            tied = sum(v == max(vals) for v in vals) > 1
            n += tied
            nonfallback_n += tied and not audit["common_fallback"]["active"]
        ties[rule] = {"exact_max_score_tie_episodes": n, "nonfallback_exact_tie_episodes": nonfallback_n}
    pairs = {}
    for a, b in PAIRS:
        same_concept = same_index = same_png = same_iou = 0
        same_concept_different_png = 0
        for row in rows:
            rules = row["artifacts"]["batch2_score_audit"]["rules"]
            sc = rules[a]["selected_concept"] == rules[b]["selected_concept"]
            sp = alt(row, a)["prediction_sha256"] == alt(row, b)["prediction_sha256"]
            same_concept += sc
            same_index += rules[a]["selected_candidate_index"] == rules[b]["selected_candidate_index"]
            same_png += sp
            same_iou += alt(row, a)["iou"] == alt(row, b)["iou"]
            same_concept_different_png += sc and not sp
        pairs[f"{a}_vs_{b}"] = {"episodes": len(rows), "same_selected_prompt": same_concept,
                               "same_selected_index": same_index, "identical_png_sha256": same_png,
                               "same_iou": same_iou, "same_prompt_different_png": same_concept_different_png,
                               "different_selected_prompt": len(rows)-same_concept,
                               "different_png_sha256": len(rows)-same_png}
    return {"episodes": len(rows), "common_visual_fallback_episodes": fallback,
            "candidate_pool_sizes": dict(Counter(len(r["artifacts"]["batch2_score_audit"]["candidates"]) for r in rows)),
            "max_score_ties": ties, "pairwise_agreement": pairs}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(evidence, output, replicates=10000, seed=6190):
    root = evidence / "runtime/runs/ics-batch2-20261116/scopes/core/results/batch2_cgics_score_pool"
    files = sorted(root.glob("*/episodes/*.json"))
    rows = [json.loads(p.read_text()) for p in files]
    if len(rows) != 4200 or any(r["status"] != "ok" for r in rows):
        raise ValueError("Expected 4200 B1 successful canonical episodes")
    ids = [(r["run_id"], r["episode_id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate canonical B1 episode")
    grouped = defaultdict(list)
    for row in rows:
        if any(alt(row, rule).get("status") != "ok" or alt(row, rule).get("iou") is None for rule in RULES):
            raise ValueError("Missing paired successful score rule")
        grouped[target_key(row)].append(row)
    queries = []
    for key, refs in sorted(grouped.items()):
        distinct_refs = {tuple(r["reference_image_sha256"]) for r in refs}
        if len(refs) != 3 or len(distinct_refs) != 3:
            raise ValueError("Expected three distinct reference images per query-target")
        query = dict(zip(("dataset", "split", "fold", "track", "query_image_sha256", "query_mask_sha256", "concept_id", "base_seed"), key))
        query["references"] = len(refs)
        for rule in RULES:
            vals = [alt(r, rule)["iou"] for r in refs]
            query[f"{rule}_iou"] = mean(vals)
            query[f"{rule}_reference_population_std"] = population_std(vals)
        for a, b in PAIRS:
            query[f"{a}_minus_{b}_pp"] = 100*(query[f"{a}_iou"]-query[f"{b}_iou"])
        queries.append(query)
    datasets = {}
    summary_csv, paired_csv, fold_csv, diagnostic_csv = [], [], [], []
    for dataset in sorted({r["dataset"] for r in rows}):
        rr = [r for r in rows if r["dataset"] == dataset]
        qq = [q for q in queries if q["dataset"] == dataset]
        clusters = [q["query_image_sha256"] for q in qq]
        folds = [q["fold"] for q in qq]
        if len(set(clusters)) != len(qq):
            raise ValueError("This export expected one target per unique query image; revise fold sensitivity if shared")
        result = {"episodes": len(rr), "query_targets": len(qq), "unique_query_images": len(set(clusters)),
                  "folds": len(set(folds)), "references_per_query_target": 3,
                  "rules": {}, "paired_comparisons": {}, "fold_results": [], "scoring_diagnostics": diagnostic(rr)}
        for rule in RULES:
            vals = [q[f"{rule}_iou"] for q in qq]
            est = ci(vals, clusters, replicates, seed)
            est["fold_stratified_ci95_sensitivity"] = ci(vals, clusters, replicates, seed, folds)["ci95"]
            est["episode_mean_iou"] = mean(alt(r, rule)["iou"] for r in rr)
            est["mean_fold_class_pooled_iou"] = class_pooled(rr, rule)
            est["mean_reference_population_std"] = mean(q[f"{rule}_reference_population_std"] for q in qq)
            result["rules"][rule] = est
            summary_csv.append({"dataset": dataset, "rule": rule, "episodes": len(rr), "unique_queries": len(qq),
                                "equal_query_mean_iou_pct": 100*est["mean"], "ci95_low_pct": 100*est["ci95"][0], "ci95_high_pct": 100*est["ci95"][1],
                                "mean_fold_class_pooled_iou_pct": 100*est["mean_fold_class_pooled_iou"],
                                "mean_reference_population_std_pp": 100*est["mean_reference_population_std"]})
        for a, b in PAIRS:
            vals = [q[f"{a}_minus_{b}_pp"] for q in qq]
            est = ci(vals, clusters, replicates, seed)
            est["fold_stratified_ci95_sensitivity"] = ci(vals, clusters, replicates, seed, folds)["ci95"]
            est.update({"direction": f"{a} minus {b}", "unit": "percentage points",
                        "queries_positive_delta": sum(v > 1e-12 for v in vals),
                        "queries_negative_delta": sum(v < -1e-12 for v in vals),
                        "queries_zero_delta": sum(abs(v) <= 1e-12 for v in vals),
                        "median_query_delta_pp": median(vals)})
            changed = [r for r in rr if alt(r, a)["prediction_sha256"] != alt(r, b)["prediction_sha256"]]
            est["descriptive_mean_delta_changed_prediction_episodes_pp"] = mean(100*(alt(r,a)["iou"]-alt(r,b)["iou"]) for r in changed) if changed else None
            result["paired_comparisons"][f"{a}_minus_{b}"] = est
            paired_csv.append({"dataset": dataset, "comparison": f"{a}_minus_{b}", "mean_delta_pp": est["mean"],
                               "ci95_low_pp": est["ci95"][0], "ci95_high_pp": est["ci95"][1],
                               "unique_queries": len(qq), "positive_delta_queries": est["queries_positive_delta"],
                               "negative_delta_queries": est["queries_negative_delta"], "zero_delta_queries": est["queries_zero_delta"]})
        for fold in sorted(set(folds)):
            rrr = [r for r in rr if r["fold"] == fold]
            qqq = [q for q in qq if q["fold"] == fold]
            entry = {"dataset": dataset, "fold": fold, "episodes": len(rrr), "unique_queries": len(qqq),
                     "observed_classes": len({r["concept_id"] for r in rrr})}
            for rule in RULES:
                entry[f"{rule}_equal_query_iou_pct"] = 100*mean(q[f"{rule}_iou"] for q in qqq)
                entry[f"{rule}_class_pooled_iou_pct"] = 100*class_pooled(rrr, rule)
            for a,b in PAIRS:
                entry[f"{a}_minus_{b}_pp"] = mean(q[f"{a}_minus_{b}_pp"] for q in qqq)
            result["fold_results"].append(entry)
            fold_csv.append(entry)
        datasets[dataset] = result
        for key, values in result["scoring_diagnostics"]["pairwise_agreement"].items():
            diagnostic_csv.append({"dataset": dataset, "pair": key, **values})
    episode_csv = []
    for row in rows:
        audit = row["artifacts"]["batch2_score_audit"]
        item = {key: row[key] for key in ("dataset", "fold", "episode_id", "concept_id", "query_image_sha256", "query_mask_sha256")}
        item["reference_image_sha256"] = row["reference_image_sha256"][0]
        item["common_visual_fallback"] = audit["common_fallback"]["active"]
        item["candidate_count"] = len(audit["candidates"])
        for rule in RULES:
            item[f"{rule}_iou"] = alt(row, rule)["iou"]
            item[f"{rule}_selected_index"] = audit["rules"][rule]["selected_candidate_index"]
            item[f"{rule}_selected_prompt"] = audit["rules"][rule]["selected_concept"]
            item[f"{rule}_prediction_sha256"] = alt(row, rule)["prediction_sha256"]
        episode_csv.append(item)
    report = {"schema_version": 1, "analysis": "B1 fixed candidate pool paired scoring rules",
              "source_episode_count": len(rows), "source_output_count": len({p.parent.parent for p in files}),
              "source_episode_inventory_sha256": hashlib.sha256("".join(str(p.relative_to(evidence))+":"+hashlib.sha256(p.read_bytes()).hexdigest()+"\n" for p in files).encode()).hexdigest(),
              "expected_shards": 56, "rules": list(RULES), "bootstrap_replicates": replicates, "bootstrap_seed": seed,
              "estimand": "Average the three distinct references per query-target, then equal-weight query-targets within each dataset. Equal 100-query folds make this numerically identical to equal-weight fold means.",
              "ci_method": "Percentile bootstrap of query-image SHA clusters; all reference outcomes and three rules paired within each cluster. Dataset bootstrap matches frozen official batch2 analysis; separate fixed-fold-stratified intervals are sensitivity analysis.",
              "secondary_metric": "Pool confusion counts within each fold/class, take class mean within fold, then equal fold mean; matches official mean_fold_class_pooled_iou.",
              "limitations": ["Three rules are nested paired controls, not independent experiments; 12600 predictions are not 12600 independent samples.",
                              "One seed and one fixed initial candidate pool per episode; no general claim about full CG-ICS expansion/restart policy.",
                              "Intervals are exploratory, no multiplicity adjustment. Query resampling conditions on the three observed references and recorded candidate pools.",
                              "Joint three-rule runtime cannot establish per-rule end-to-end latency.",
                              "B1-only statistics do not certify B2/B3/B4 completion, or the still-separate scientific provenance and pixel audit."],
              "datasets": datasets}
    output.mkdir(parents=True, exist_ok=True)
    (output / "statistics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    for name, contents in (("b1_summary.csv", summary_csv), ("b1_paired_comparisons.csv", paired_csv),
                           ("b1_fold_results.csv", fold_csv), ("b1_scoring_diagnostics.csv", diagnostic_csv),
                           ("b1_query_results.csv", queries), ("b1_episode_results.csv", episode_csv)):
        write_csv(output / name, contents)
    print(json.dumps({d: {"rules": v["rules"], "comparisons": v["paired_comparisons"], "diagnostics": v["scoring_diagnostics"]} for d,v in datasets.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path(__file__).resolve().parent.parent / "b1_evidence")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=6190)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("replicates must be positive")
    analyze(args.evidence.resolve(), args.output.resolve(), args.replicates, args.seed)
