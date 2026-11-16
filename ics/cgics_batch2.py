"""Pure, query-GT-free rules for the second ICS study.

These functions only choose among predictions or scored concepts. They never
read an episode manifest, a query annotation, or a ground-truth class name.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

BATCH2_VARIANTS = frozenset({
    "fixed_pool_scores", "agreement_gate", "random_gate", "initial_score_gate",
})
RULES = ("combined", "ref_only", "query_only")
SCORE_FALLBACK_THRESHOLD = 1e-6


def finite_probability(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [0,1]")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0,1]")
    return float(value)


def canonical_fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def prediction_agreement(text_mask, visual_mask):
    """Original-resolution binary prediction overlap; empty-empty is NOT reliable."""
    text = np.asarray(text_mask)
    visual = np.asarray(visual_mask)
    if text.ndim != 2 or visual.shape != text.shape:
        raise ValueError("Agreement masks must be two aligned H x W predictions")
    if not np.isin(text, [0, 1]).all() or not np.isin(visual, [0, 1]).all():
        raise ValueError("Agreement masks must be binary")
    text, visual = text.astype(bool), visual.astype(bool)
    text_area, visual_area = int(text.sum()), int(visual.sum())
    intersection, union = int((text & visual).sum()), int((text | visual).sum())
    return {"intersection": intersection, "union": union,
            "text_area": text_area, "visual_area": visual_area,
            "both_nonempty": bool(text_area and visual_area),
            "iou": intersection / union if union else None,
            "resolution": "original_query", "query_gt_used": False}


def agreement_requires_search(agreement, threshold):
    threshold = finite_probability(threshold, "gate_threshold")
    return not (agreement["both_nonempty"] and agreement["iou"] is not None
                and agreement["iou"] >= threshold)


def input_content_fingerprint(inputs):
    """Content identity for independent gate randomization; paths are not hashed."""
    rows = [{"role": "query_image", "sha256": _file_sha(inputs["query_image"])}]
    for index, reference in enumerate(inputs["references"]):
        for role in ("image", "mask"):
            rows.append({"role": f"reference_{index}_{role}", "sha256": _file_sha(reference[role])})
    return canonical_fingerprint({"schema_version": 1, "inputs": rows})


def _file_sha(value):
    h = hashlib.sha256()
    with Path(value).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def random_search_decision(input_fingerprint, seed, probability):
    probability = finite_probability(probability, "random_search_probability")
    # A dedicated hash draw cannot consume model/MLLM/numpy random state. The
    # probability is deliberately absent from this seed so probabilities form
    # nested subsets on the same inputs. This is a budget-control frontier,
    # not a claim of exactly matching a learned gate's realized search rate.
    payload = {"domain": "CE6190-ICS-batch2-random-gate-v1",
               "input_fingerprint": input_fingerprint, "seed": int(seed)}
    # IEEE-754 doubles represent these 53-bit fractions exactly. Using the
    # full 64 bits can round the largest hashes to 1.0, breaking the p=1
    # staged-always-search control.
    bits = int(canonical_fingerprint(payload)[:16], 16) >> 11
    draw = bits / 2**53
    return {"requires_search": draw < probability, "uniform_draw": draw,
            "probability": probability, "input_fingerprint": input_fingerprint,
            "rng": "sha256-independent-of-model-streams"}


def make_score_audit(raw_rows, source_pin, *, requested_candidates, generated_candidates):
    """One 1-shot pool, three scores; no random alpha=beta=0 special case."""
    if not raw_rows:
        raise ValueError("A fixed candidate pool must not be empty")
    candidates, seen = [], set()
    for index, row in enumerate(raw_rows):
        prompt = row["prompt"]
        if not isinstance(prompt, str) or not prompt.strip() or prompt in seen:
            raise ValueError("Fixed-pool candidates must be unique nonempty strings")
        seen.add(prompt)
        ref = finite_probability(float(row["ref_iou"]), "ref_iou")
        query = finite_probability(float(row["query_score"]), "query_presence")
        combined = ref * query
        if not math.isclose(float(row["combined_score"]), combined, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Pinned combined score must be ref_iou * query_presence")
        candidates.append({"candidate_index": index, "prompt": prompt,
                           "ref_iou": ref, "query_presence": query,
                           "combined_score": combined})
    payload = {"schema_version": 1, "source_pin": source_pin, "candidates": candidates}
    rules = {}
    columns = {"combined": "combined_score", "ref_only": "ref_iou", "query_only": "query_presence"}
    # The pinned baseline has a low-score visual fallback. Hold this decision
    # common across all rules: thresholding their differently scaled scores
    # would confound ranking with a second, rule-specific fallback policy.
    maximum_combined = max(row["combined_score"] for row in candidates)
    common_fallback = maximum_combined < SCORE_FALLBACK_THRESHOLD
    for rule, column in columns.items():
        best = max(candidates, key=lambda row: row[column])  # Python max retains first tie.
        score = best[column]
        rules[rule] = {"selected_candidate_index": best["candidate_index"],
                       "selected_concept": "visual" if common_fallback else best["prompt"],
                       "selected_score": score, "fallback_to_visual": common_fallback}
    return {"schema_version": 1, "pool_fingerprint": canonical_fingerprint(payload),
            "fingerprint_payload": payload, "candidates": candidates, "rules": rules,
            "requested_candidates": int(requested_candidates),
            "generated_candidates": int(generated_candidates), "unique_candidates": len(candidates),
            "generation": "one initial MLLM pool; no expansion, restart, or early pruning",
            "tie_policy": "first_candidate_in_generated_order",
            "fallback_policy": "one common max combined score < 1e-6 decision uses literal prompt 'visual' for all rules",
            "common_fallback": {"score_column": "combined_score", "max_combined_score": maximum_combined,
                                "threshold": SCORE_FALLBACK_THRESHOLD, "active": common_fallback},
            "score_definition": {"combined": "ref_iou * query_presence", "ref_only": "ref_iou",
                                 "query_only": "query_presence"},
            "prediction_scope": "identical query and visual prompts; only ranking rule differs",
            "cost_scope": "joint audit workload; shared pool generation/scoring/encoding counted once; not three independent model latencies"}
