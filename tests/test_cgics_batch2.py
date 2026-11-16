"""Batch2 scientific contracts with synthetic masks and the pinned score method.

No weights or servers are loaded. The scorer fixture below is extracted verbatim
from upstream tree_search.py at the recorded commit, not a rewritten algorithm.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ics.adapters.cgics import CGICSAdapter, PIN, _OfficialBackend, component_settings
from ics.cgics_batch2 import (agreement_requires_search, canonical_fingerprint,
                             input_content_fingerprint, make_score_audit,
                             prediction_agreement, random_search_decision)


class FakeBackend:
    def __init__(self, text=None, visual=None):
        self.events = []
        self.text = np.array([[1, 0], [0, 0]], dtype=bool) if text is None else text
        self.visual_prediction = self.text.copy() if visual is None else visual
        self.telemetry, self.artifacts = {}, {}
        self.pool_calls = 0
        self.raw_rows = [
            {"prompt": "dog", "ref_iou": .9, "query_score": .1, "combined_score": .09},
            {"prompt": "animal", "ref_iou": .2, "query_score": .95, "combined_score": .19},
            {"prompt": "canine", "ref_iou": .7, "query_score": .7, "combined_score": .49},
        ]
        self.score = {"prompt": "dog", "ref_iou": .9, "query_score": .9, "combined_score": .81}

    def metadata(self):
        return {"backend": "fake_batch2_contract"}

    def reset(self, seed):
        self.events.append(("reset", seed))
        self.telemetry = {"sam_calls": 0, "sam_prompt_head_calls": 0, "mllm_requests": 0}
        self.artifacts = {}

    def synchronize(self):
        pass

    def prepare(self, inputs):
        self.events.append("prepare")
        return inputs

    def build_state(self, prepared, components):
        self.events.append(("build", dict(components)))

    def select(self, prepared, mode):
        self.events.append(("select", mode))
        self.telemetry["mllm_requests"] += 1
        return "initial dog" if mode == "single" else "searched dog"

    def visual(self, prepared):
        self.events.append("visual")
        self.telemetry["sam_calls"] += 1
        self.telemetry["sam_prompt_head_calls"] += 1
        return "fixed visual instances"

    def final(self, prepared, prompt, visual):
        self.events.append(("final", prompt, visual))
        self.telemetry["sam_calls"] += 1
        self.telemetry["sam_prompt_head_calls"] += 1
        if visual is None:
            return self.text.copy()
        if prompt is None:
            return self.visual_prediction.copy()
        # Distinct, valid masks make mismatched rule/output assignment observable.
        table = {"canine": [[1, 1], [0, 0]], "dog": [[1, 0], [1, 0]],
                 "animal": [[0, 1], [0, 1]], "initial dog": [[1, 0], [0, 0]],
                 "searched dog": [[1, 1], [1, 0]], "visual": [[0, 0], [1, 1]]}
        return np.asarray(table[prompt], dtype=bool)

    def fixed_pool(self, prepared):
        self.pool_calls += 1
        self.events.append("fixed_pool")
        self.telemetry["mllm_requests"] += 1
        self.telemetry["sam_calls"] += 2 * len(self.raw_rows)
        self.telemetry["sam_prompt_head_calls"] += 2 * len(self.raw_rows)
        return make_score_audit(self.raw_rows, PIN, requested_candidates=5,
                                generated_candidates=len(self.raw_rows))

    def initial_score(self, prepared, prompt):
        self.events.append(("initial_score", prompt))
        self.telemetry["sam_calls"] += 2
        return dict(self.score)

    def begin_search(self, prepared):
        self.events.append("begin_search")

    def clear(self):
        self.events.append("clear")

    def close(self):
        self.clear()


class Batch2PolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        for name, data in (("query", b"query-bytes"), ("ref", b"ref-bytes"), ("mask", b"mask-bytes")):
            (root / name).write_bytes(data)
        self.inputs = {"query_image": str(root / "query"),
                       "references": [{"image": str(root / "ref"), "mask": str(root / "mask")}]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_shared_pool_three_rankings_are_real_paired_outputs(self):
        backend = FakeBackend()
        result = CGICSAdapter({"variant": "fixed_pool_scores"}, _backend=backend).predict(self.inputs, seed=17)
        audit = result.artifacts["batch2_score_audit"]
        self.assertEqual(backend.pool_calls, 1)
        self.assertEqual(result.telemetry["mllm_requests"], 1)
        self.assertEqual(result.telemetry["sam_calls"], 10)  # six scoring + visual + three final.
        self.assertEqual(set(result.alternative_masks), {"ref_only", "query_only"})
        self.assertEqual([audit["rules"][k]["selected_concept"] for k in ("combined", "ref_only", "query_only")],
                         ["canine", "dog", "animal"])
        np.testing.assert_array_equal(result.mask, [[1, 1], [0, 0]])
        np.testing.assert_array_equal(result.alternative_masks["ref_only"], [[1, 0], [1, 0]])
        np.testing.assert_array_equal(result.alternative_masks["query_only"], [[0, 1], [0, 1]])
        finals = [x for x in backend.events if isinstance(x, tuple) and x[0] == "final"]
        self.assertTrue(all(x[2] == "fixed visual instances" for x in finals))
        self.assertEqual(audit["pool_fingerprint"], canonical_fingerprint(audit["fingerprint_payload"]))
        for rule in audit["rules"].values():
            self.assertEqual(rule["final_sam_calls"], 1)
            self.assertGreaterEqual(rule["final_inference_seconds"], 0)
        json.dumps(result.artifacts, allow_nan=False)

    def test_agreement_accepts_initial_without_search_or_joint_probe(self):
        backend = FakeBackend()
        result = CGICSAdapter({"variant": "agreement_gate", "gate_threshold": .7}, _backend=backend).predict(self.inputs)
        gate = result.artifacts["batch2_gate"]
        self.assertEqual(gate["agreement_iou"], 1.)
        self.assertFalse(gate["search_triggered"])
        self.assertNotIn("begin_search", backend.events)
        self.assertEqual(result.telemetry["mllm_requests"], 1)
        self.assertEqual(result.telemetry["sam_calls"], 4)  # visual transfer, two probes, final joint.
        self.assertIn(("final", "initial dog", None), backend.events)
        self.assertIn(("final", None, "fixed visual instances"), backend.events)
        self.assertEqual(sum(x == ("final", "initial dog", "fixed visual instances") for x in backend.events), 1)
        self.assertEqual(set(result.alternative_masks), {"gate_text", "gate_visual"})
        self.assertEqual(prediction_agreement(result.alternative_masks["gate_text"],
                                              result.alternative_masks["gate_visual"]), gate["agreement"])

    def test_disagreement_runs_full_after_probes_and_counts_everything(self):
        backend = FakeBackend(visual=np.array([[0, 0], [0, 1]], dtype=bool))
        result = CGICSAdapter({"variant": "agreement_gate"}, _backend=backend).predict(self.inputs)
        gate = result.artifacts["batch2_gate"]
        self.assertTrue(gate["search_triggered"])
        self.assertEqual(gate["final_concept"], "searched dog")
        self.assertEqual(result.telemetry["mllm_requests"], 2)
        self.assertLess(backend.events.index(("final", None, "fixed visual instances")),
                        backend.events.index("begin_search"))
        self.assertIn("fallback_search_s", result.telemetry["stages"])
        np.testing.assert_array_equal(result.mask, [[1, 1], [1, 0]])

    def test_empty_empty_and_one_empty_require_search(self):
        empty = np.zeros((2, 2), dtype=bool)
        for other in (empty, np.ones((2, 2), dtype=bool)):
            backend = FakeBackend(text=empty, visual=other)
            result = CGICSAdapter({"variant": "agreement_gate", "gate_threshold": 0.}, _backend=backend).predict(self.inputs)
            self.assertTrue(result.artifacts["batch2_gate"]["search_triggered"])
        self.assertIsNone(prediction_agreement(empty, empty)["iou"])

    def test_threshold_boundary_is_predeclared_and_inclusive(self):
        text = np.array([[1, 1], [0, 0]], dtype=bool)
        visual = np.array([[1, 0], [0, 0]], dtype=bool)
        a = prediction_agreement(text, visual)
        self.assertFalse(agreement_requires_search(a, .5))
        self.assertTrue(agreement_requires_search(a, .500001))

    def test_random_controls_are_independent_nested_and_content_identified(self):
        identity = input_content_fingerprint(self.inputs)
        state = random.getstate()
        decisions = [random_search_decision(identity, 17, p) for p in (0., .25, .5, .75, 1.)]
        self.assertEqual(random.getstate(), state)
        self.assertEqual(len({d["uniform_draw"] for d in decisions}), 1)
        self.assertEqual([d["requires_search"] for d in decisions], sorted(d["requires_search"] for d in decisions))
        clone = Path(self.tmp.name) / "other-query-name"
        clone.write_bytes(Path(self.inputs["query_image"]).read_bytes())
        self.assertEqual(identity, input_content_fingerprint({**self.inputs, "query_image": str(clone)}))
        clone.write_bytes(b"different image")
        self.assertNotEqual(identity, input_content_fingerprint({**self.inputs, "query_image": str(clone)}))

    def test_random_maximum_hash_preserves_exact_zero_and_one_controls(self):
        with patch("ics.cgics_batch2.canonical_fingerprint", return_value="f" * 64):
            never = random_search_decision("identity", 17, 0.)
            always = random_search_decision("identity", 17, 1.)
        self.assertEqual(always["uniform_draw"], (2**53 - 1) / 2**53)
        self.assertLess(always["uniform_draw"], 1.)
        self.assertFalse(never["requires_search"])
        self.assertTrue(always["requires_search"])

    def test_random_zero_and_one_execute_both_probes_but_ignore_agreement(self):
        for probability, expected in ((0., False), (1., True)):
            backend = FakeBackend()
            result = CGICSAdapter({"variant": "random_gate", "random_search_probability": probability}, _backend=backend).predict(self.inputs, seed=17)
            gate = result.artifacts["batch2_gate"]
            self.assertEqual(gate["search_triggered"], expected)
            self.assertEqual(gate["agreement_iou"], 1.)
            self.assertEqual(result.telemetry["sam_calls"], 4)
            self.assertEqual(result.telemetry["mllm_requests"], 1 + expected)

    def test_initial_score_gate_is_distinct_from_new_agreement_cue(self):
        backend = FakeBackend(text=np.zeros((2, 2), dtype=bool))
        result = CGICSAdapter({"variant": "initial_score_gate", "score_gate_threshold": .8}, _backend=backend).predict(self.inputs)
        gate = result.artifacts["batch2_gate"]
        self.assertFalse(gate["search_triggered"])
        self.assertEqual(gate["initial_score"]["combined_score"], .81)
        self.assertNotIn("agreement", gate)
        self.assertEqual(result.alternative_masks, {})
        self.assertNotIn(("final", None, "fixed visual instances"), backend.events)
        backend.score["combined_score"] = .2
        again = CGICSAdapter({"variant": "initial_score_gate"}, _backend=backend).predict(self.inputs)
        self.assertTrue(again.artifacts["batch2_gate"]["search_triggered"])

    def test_failure_retains_stage_and_always_clears(self):
        backend = FakeBackend()
        original = backend.final
        def fail_on_visual(prepared, prompt, visual):
            if prompt is None:
                raise RuntimeError("probe failure")
            return original(prepared, prompt, visual)
        backend.final = fail_on_visual
        adapter = CGICSAdapter({"variant": "agreement_gate"}, _backend=backend)
        with self.assertRaisesRegex(RuntimeError, "probe failure"):
            adapter.predict(self.inputs)
        self.assertEqual(backend.events[-1], "clear")
        diagnostics = adapter.failure_diagnostics()
        self.assertEqual(diagnostics["last_stage"], "gate_visual_prediction_s")
        self.assertEqual(diagnostics["telemetry"]["sam_calls"], 2)
        self.assertEqual(diagnostics["artifacts"]["batch2_gate"]["initial_concept"], "initial dog")

    def test_metadata_leak_and_multishot_rejected(self):
        for field in ("query_mask", "query_gt", "class_name", "concept_id"):
            backend = FakeBackend()
            with self.assertRaises(ValueError):
                CGICSAdapter({"variant": "agreement_gate"}, _backend=backend).predict({**self.inputs, field: "forbidden"})
            self.assertEqual(backend.events, [])
        backend = FakeBackend()
        with self.assertRaisesRegex(ValueError, "1-shot"):
            CGICSAdapter({"variant": "fixed_pool_scores"}, _backend=backend).predict(
                {**self.inputs, "references": self.inputs["references"] * 2})
        self.assertEqual(backend.events, [])

    def test_invalid_probabilities_and_mislabelled_policy_rejected(self):
        for value in (-.1, 1.1, float("nan"), float("inf"), True):
            for key in ("gate_threshold", "random_search_probability", "score_gate_threshold"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    component_settings({"variant": "agreement_gate", key: value})
        for config in ({"variant": "fixed_pool_scores", "alpha": 0},
                       {"variant": "fixed_pool_scores", "alpha": True},
                       {"variant": "random_gate", "beta": 0},
                       {"variant": "custom", "concept_mode": "gate", "visual_mode": "boxes"}):
            with self.assertRaises(ValueError):
                component_settings(config)

    def test_fixed_pool_ties_zero_fallback_and_nonfinite_scores(self):
        rows = [{"prompt": p, "ref_iou": 0., "query_score": 0., "combined_score": 0.} for p in ("dog", "cat")]
        audit = make_score_audit(rows, PIN, requested_candidates=2, generated_candidates=2)
        for row in audit["rules"].values():
            self.assertEqual(row["selected_candidate_index"], 0)
            self.assertEqual(row["selected_concept"], "visual")
            self.assertTrue(row["fallback_to_visual"])
        self.assertTrue(audit["common_fallback"]["active"])
        # A component-only rule can have a large selected score when all
        # combined scores are zero. The fallback must still be shared.
        rows[0]["ref_iou"] = .9
        rows[1]["query_score"] = .8
        audit = make_score_audit(rows, PIN, requested_candidates=2, generated_candidates=2)
        self.assertEqual(audit["rules"]["ref_only"]["selected_score"], .9)
        self.assertEqual(audit["rules"]["query_only"]["selected_score"], .8)
        self.assertTrue(all(row["fallback_to_visual"] for row in audit["rules"].values()))
        self.assertTrue(all(row["selected_concept"] == "visual" for row in audit["rules"].values()))
        # Once the common fallback is off, every rule is pure stable argmax.
        rows[0]["query_score"], rows[0]["combined_score"] = .2, .18
        audit = make_score_audit(rows, PIN, requested_candidates=2, generated_candidates=2)
        self.assertFalse(audit["common_fallback"]["active"])
        self.assertEqual([audit["rules"][rule]["selected_concept"] for rule in ("combined", "ref_only", "query_only")],
                         ["dog", "dog", "cat"])
        self.assertTrue(all(not row["fallback_to_visual"] for row in audit["rules"].values()))
        broken = copy.deepcopy(rows)
        broken[0]["query_score"] = float("nan")
        with self.assertRaises(ValueError):
            make_score_audit(broken, PIN, requested_candidates=2, generated_candidates=2)
        broken = copy.deepcopy(rows)
        broken[1]["prompt"] = "dog"
        with self.assertRaises(ValueError):
            make_score_audit(broken, PIN, requested_candidates=2, generated_candidates=2)


class BackendSemanticsTests(unittest.TestCase):
    def test_lazy_reference_encoding_and_fallback_seed_offset_preserve_counters(self):
        calls = []
        class Torch:
            cuda = SimpleNamespace(is_available=lambda: False)
            inference_mode = staticmethod(contextlib.nullcontext)
            manual_seed = staticmethod(lambda seed: calls.append(("torch_seed", seed)))
        backend = _OfficialBackend.__new__(_OfficialBackend)
        backend.torch, backend.np, backend.seed = Torch, np, 17
        backend.telemetry = {"mllm_requests": 3, "sam_calls": 12}
        backend.artifacts = {}
        backend.segmenter = SimpleNamespace(ref_states=[], processor=SimpleNamespace(set_image=lambda image: calls.append(("encode", image)) or image))
        backend.begin_search({"refs": ["reference"]})
        backend.ensure_reference_states({"refs": ["reference"]})
        self.assertEqual(calls.count(("encode", "reference")), 1)
        self.assertEqual(backend._sampling_offset, 3)
        self.assertEqual(backend.telemetry, {"mllm_requests": 3, "sam_calls": 12})
        self.assertFalse(backend.artifacts["fallback_sampling_stream"]["counter_reset"])

    def test_mllm_request_seed_stream_resets_without_erasing_cost(self):
        calls = []
        class Client:
            def __init__(self):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            def with_options(self, **kwargs):
                return self
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
                                       choices=[SimpleNamespace(message=SimpleNamespace(content="dog"))])
        class MLLM:
            def __init__(self, **kwargs):
                self.client, self.api_model = Client(), kwargs["model_name"]
            def generate(self, **kwargs):
                return [self._call_api_once([])]
        b = _OfficialBackend.__new__(_OfficialBackend)
        b.config, b.seed = {"llm_seed_from_episode": True}, 17
        b.mllm_module = SimpleNamespace(MLLM=MLLM)
        b.telemetry = {"mllm_requests": 0, "mllm_prompt_tokens": 0, "mllm_completion_tokens": 0,
                       "mllm_usage_missing_requests": 0, "mllm_request_wall_s": 0.}
        b.artifacts = {"mllm_responses": [], "generation_events": []}
        mllm = b._make_mllm()
        mllm.generate(mode="initial")
        b._sampling_offset = b.telemetry["mllm_requests"]
        mllm.generate(mode="initial")
        mllm.generate(mode="child")
        self.assertEqual([x["seed"] for x in calls], [17, 17, 18])
        self.assertEqual(b.telemetry["mllm_requests"], 3)
        self.assertEqual(b.telemetry["mllm_prompt_tokens"], 15)

    def test_fixed_pool_backend_generates_once_and_preserves_first_unique_order(self):
        b = _OfficialBackend.__new__(_OfficialBackend)
        b.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        b.config = {"BON": 5}
        b.split = lambda x: "split mask"
        b.tree_module = SimpleNamespace(make_ref_image_for_llm=lambda *args, **kwargs: ("highlight", None))
        generated, scored = [], []
        b.mllm = SimpleNamespace(generate=lambda **kwargs: generated.append(kwargs) or ["dog", "cat", "dog"])
        def score(ref_index, mask, candidates, alpha, beta):
            scored.append((ref_index, mask, candidates, alpha, beta))
            return [{"prompt": x, "ref_iou": .5, "query_score": .8, "combined_score": .4} for x in candidates]
        b.searcher = SimpleNamespace(_score_single_ref_raw=score)
        result = b.fixed_pool({"refs": ["ref"], "ref_tensors": ["tensor"], "masks": ["mask"]})
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0]["num_candidates"], 5)
        self.assertEqual(scored, [(0, "mask", ["dog", "cat"], 1., 1.)])
        self.assertEqual(result["generated_candidates"], 3)
        self.assertEqual(result["unique_candidates"], 2)

    def test_pinned_score_fixture_agrees_with_three_rules_and_exposes_zero_zero_random(self):
        self.assertEqual(hashlib.sha256(PINNED_SCORE_METHOD.encode()).hexdigest(), PINNED_SCORE_METHOD_SHA256)
        namespace = {"List": list, "Dict": dict, "Any": object, "random": random.Random(18),
                     "calculate_iou": lambda pred, gt: pred.value}
        exec(textwrap.dedent(PINNED_SCORE_METHOD), namespace)
        method = namespace["_score_single_ref_raw"]
        class Tensor:
            ndim = 2
            def __init__(self, value): self.value = value
            def float(self): return self
            def cpu(self): return self
            def __gt__(self, other): return self
        metrics = {"a": (.8, .2), "b": (.3, .9), "c": (.6, .6)}
        def infer(state, prompt):
            r, q = metrics[prompt]
            return {"semantic_mask": Tensor(r), "presence_score": q}
        self_obj = SimpleNamespace(segmenter=SimpleNamespace(ref_states=["ref"], query_state="query", infer=infer))
        combined = method(self_obj, 0, Tensor(1.), list(metrics), 1., 1.)
        audit = make_score_audit(combined, PIN, requested_candidates=3, generated_candidates=3)
        for key, alpha, beta in (("combined", 1., 1.), ("ref_only", 1., 0.), ("query_only", 0., 1.)):
            actual = method(self_obj, 0, Tensor(1.), list(metrics), alpha, beta)
            picked = max(actual, key=lambda x: x["combined_score"])
            self.assertEqual(audit["rules"][key]["selected_concept"], picked["prompt"])
        zero = method(self_obj, 0, Tensor(1.), ["a"], 0., 0.)[0]["combined_score"]
        self.assertGreater(zero, 0.)
        self.assertLess(zero, 1.)


# Verbatim upstream method from commit 8da845743f15c18ec490567fbd86dba58c8a388f
PINNED_SCORE_METHOD = 'def _score_single_ref_raw(\n        self,\n        ref_idx: int,\n        ref_mask,\n        candidates: List[str],\n        alpha: float,\n        beta: float,\n    ) -> List[Dict[str, Any]]:\n        """Internal implementation of single-ref scoring; returns a list of raw per-candidate scoring records."""\n        ref_state = self.segmenter.ref_states[ref_idx]\n\n        # Binarize ref_mask (aligned with the original select_prompt handling)\n        ref_mask_tensor = ref_mask.float()\n        while ref_mask_tensor.ndim > 2:\n            ref_mask_tensor = ref_mask_tensor.squeeze(0)\n        ref_mask_binary = (ref_mask_tensor > 0.5).float().cpu()\n\n        out: List[Dict[str, Any]] = []\n        for prompt in candidates:\n            # Compute IoU on the ref\n            res_ref = self.segmenter.infer(ref_state, prompt=prompt)\n            pred = res_ref[\'semantic_mask\']\n            pred_binary = (pred > 0.5).float().cpu()\n            ref_iou_value = float(calculate_iou(pred_binary, ref_mask_binary))\n\n            # Take presence_score on the query\n            res_q = self.segmenter.infer(self.segmenter.query_state, prompt=prompt)\n            query_score_value = float(res_q[\'presence_score\'])\n\n            # combined (original select_prompt logic)\n            if alpha == 0 and beta == 0:\n                combined_score = random.random()\n            else:\n                combined_score = (ref_iou_value ** alpha) * (query_score_value ** beta)\n\n            out.append({\n                "prompt": prompt,\n                "ref_iou": ref_iou_value,\n                "query_score": query_score_value,\n                "combined_score": float(combined_score),\n            })\n        return out'
PINNED_SCORE_METHOD_SHA256 = 'ea7759a2cb1418506bc45b29b2d845760a90071f9a1a55ad199e606ae1b94086'

if __name__ == "__main__":
    unittest.main()
