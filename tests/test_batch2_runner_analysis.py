"""Synthetic integrity/statistics checks; no GPU or scientific score claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ics.adapters.base import Prediction
from ics.runner import canonical_sha256, file_sha256, run_experiment
from ics.batch2_analysis import (analyze_batch2, clustered_mean_ci, paired_statistics,
                                verify_counts, verify_score_pool, verify_gate)
from ics.cgics_batch2 import (make_score_audit, prediction_agreement,
                              input_content_fingerprint, random_search_decision)
from ics_eval import manifest_fingerprint


class PairedAdapter:
    def __init__(self, *, invalid=None):
        self.invalid = invalid
    def metadata(self):
        return {"type": "synthetic-paired-control-test"}
    def close(self):
        pass
    def predict(self, inputs, *, seed=0):
        if set(inputs) != {"query_image", "references"}:
            raise AssertionError("Query GT leaked to adapter")
        mask = np.array([[1, 0, 0], [0, 0, 0]], dtype=bool)
        ref = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
        alternatives = {"ref_only": ref, "query_only": np.zeros_like(mask)}
        audit = make_score_audit([
            {"prompt": "dog", "ref_iou": .4, "query_score": .8, "combined_score": .32},
            {"prompt": "animal", "ref_iou": .8, "query_score": .3, "combined_score": .24}],
            "synthetic-pin", requested_candidates=2, generated_candidates=2)
        if self.invalid == "shape":
            alternatives["ref_only"] = np.zeros((1, 1))
        elif self.invalid == "binary":
            alternatives["query_only"] = np.full(mask.shape, 2)
        elif self.invalid == "missing":
            alternatives.pop("query_only")
        elif self.invalid == "unsafe":
            alternatives = {"../escape": mask}
        return Prediction(mask, {"sam_calls": 7, "mllm_requests": 1},
                          {"batch2_score_audit": audit}, alternatives)


class GateAdapter(PairedAdapter):
    def predict(self, inputs, *, seed=0):
        result = super().predict(inputs, seed=seed)
        text = result.mask
        visual = result.alternative_masks["ref_only"]
        agreement = prediction_agreement(text, visual)
        result.alternative_masks = {"gate_text": text, "gate_visual": visual}
        result.artifacts = {"batch2_gate": {"policy": "agreement_gate", "query_gt_used": False,
            "agreement": agreement, "agreement_iou": agreement["iou"], "threshold": .7,
            "search_triggered": True, "initial_concept": "dog", "final_concept": "animal"}}
        return result


class RandomGateAdapter(GateAdapter):
    def predict(self, inputs, *, seed=0):
        result = super().predict(inputs, seed=seed)
        decision = random_search_decision(input_content_fingerprint(inputs), seed, .5)
        gate = result.artifacts["batch2_gate"]
        gate.update(policy="random_gate", random_decision=decision,
                    search_triggered=decision["requires_search"])
        gate["final_concept"] = "animal" if decision["requires_search"] else "dog"
        return result


class Batch2RunnerAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        def asset(name, value):
            image, mask = self.data / (name + ".png"), self.data / (name + "-mask.png")
            Image.fromarray(np.full((2, 3, 3), value, dtype=np.uint8)).save(image)
            Image.fromarray(np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)).save(mask)
            return {"image": image.name, "mask": mask.name, "image_sha256": file_sha256(image), "mask_sha256": file_sha256(mask)}
        reference = asset("reference", 200)
        self.manifest = {"schema_version": 1, "status": "ready", "protocol_id": "synthetic-batch2",
                         "episodes": [{"episode_id": f"e{i}", "dataset": "synthetic", "dataset_version": "fixture",
                                       "split": "test", "fold": 0, "track": "semantic", "concept_id": str(i % 2),
                                       "reference_seed": 0, "query": asset(f"q{i}", 20+i), "references": [reference],
                                       "mask_semantics": "binary synthetic", "ignore_label": 255} for i in range(3)]}
        self.manifest_path = self.data / "synthetic_fresh.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.receipt = {"history": {"exclusion_sha256": "synthetic-prior-digest"}, "manifests": {
            "synthetic": {"path": str(self.manifest_path), "cohort": "fresh",
                          "manifest_sha256": manifest_fingerprint(self.manifest)}}}
        self.manifest_path.with_suffix(".protocol.json").write_text(json.dumps({"batch2": {
            "cohort": "fresh", "studies": ["score_ablation"],
            "exclusion_receipt_path": str(self.data / "data_receipt.json"),
            "prior_exclusion_sha256": "synthetic-prior-digest"}}))
        # Real receipt verification has exhaustive asset/source fixtures in
        # test_batch2_data; these runner fixtures isolate its validated API.
        self.receipt_patch = patch("ics.batch2_analysis.verify_scope", return_value=self.receipt)
        self.receipt_patch.start()
        self.scope = self.root / "queue"
        self.output = self.scope / "results/score/synthetic_fresh"
        self.config = {"experiment_id": "score", "method": "test_double", "variant": "fixed_pool_scores", "seed": 0, "adapter": {}}
        self.code_patch = patch("ics.runner.project_code_digest", return_value="synthetic-code")
        self.code_patch.start()
    def tearDown(self):
        self.code_patch.stop()
        self.receipt_patch.stop()
        self.tmp.cleanup()
    def run_fake(self, **kwargs):
        return run_experiment(self.config, manifest_path=self.manifest_path, dataset_root=self.data,
                              output_dir=self.output, **kwargs)
    def rows(self):
        return [json.loads(p.read_text()) for p in sorted((self.output / "episodes").glob("*.json"))]
    def plan(self):
        self.scope.mkdir(exist_ok=True)
        self.plan_data = {"fingerprint": "synthetic-plan", "tasks": [
            {"id": "score--synthetic--0", "output": str(self.output), "manifest": str(self.manifest_path),
             "num_shards": 1, "shard_index": 0, "resource": "cgics_mllm", "config_data": self.config}]}
        (self.scope / "plan.json").write_text(json.dumps(self.plan_data))
    def analyze(self):
        return analyze_batch2([self.scope], self.root / "analysis", bootstrap_replicates=20)
    def test_all_unbound_configs_still_have_coverage_rows(self):
        self.plan()
        result = self.analyze()
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(len(result["configuration_results"]), 3)
        self.assertTrue(all(x["observed"] == 0 and x["expected"] == 3 and x["missing"] == 3
                            for x in result["configuration_results"]))
        self.assertEqual(result["outputs_without_binding"][0]["experiment"], "score")
    def test_missing_cohort_provenance_refused(self):
        self.plan()
        self.manifest_path.with_suffix(".protocol.json").unlink()
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.analyze()
    def test_wrong_exclusion_binding_refused(self):
        self.plan()
        self.receipt["history"]["exclusion_sha256"] = "changed-history"
        with self.assertRaisesRegex(ValueError, "receipt binding"):
            self.analyze()
    def test_shared_fixed_pool_fallback_is_not_rule_specific(self):
        audit = make_score_audit([
            {"prompt": "dog", "ref_iou": .9, "query_score": 0., "combined_score": 0.},
            {"prompt": "animal", "ref_iou": 0., "query_score": .8, "combined_score": 0.}],
            "synthetic-pin", requested_candidates=2, generated_candidates=2)
        verify_score_pool({"artifacts": {"batch2_score_audit": audit}})
        audit["rules"]["ref_only"]["fallback_to_visual"] = False
        with self.assertRaisesRegex(ValueError, "frozen ref_only"):
            verify_score_pool({"artifacts": {"batch2_score_audit": audit}})
    def test_unplanned_extra_adapter_option_is_rejected(self):
        self.run_fake(adapter_factory=lambda _: PairedAdapter())
        self.plan()
        self.plan_data["tasks"][0]["config_data"]["adapter"]["gate_threshold"] = .7
        (self.scope / "plan.json").write_text(json.dumps(self.plan_data))
        with self.assertRaisesRegex(ValueError, "frozen task"):
            self.analyze()
    def test_saved_gate_probe_pixels_and_trigger_are_audited(self):
        self.config.update(variant="agreement_gate", adapter={"gate_threshold": .7})
        self.run_fake(adapter_factory=lambda _: GateAdapter())
        self.plan()
        result = self.analyze()
        self.assertEqual(result["configuration_results"][0]["trigger_rate_successful_observed"], 1)
        p = next((self.output / "episodes").glob("*.json"))
        row = json.loads(p.read_text())
        # Keep internally consistent false overlap counts while the hashed
        # original probes remain unchanged: pixel audit must catch this.
        row["artifacts"]["batch2_gate"]["agreement"] = prediction_agreement(
            np.zeros((2, 3)), np.zeros((2, 3)))
        p.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "saved prediction probes"):
            self.analyze()
    def test_gate_vs_single_reports_gate_trigger_strata_in_correct_direction(self):
        self.config.update(experiment_id="a_gate", variant="agreement_gate", adapter={"gate_threshold": .7})
        self.run_fake(adapter_factory=lambda _: GateAdapter())
        gate_task = {"id": "a_gate--0", "output": str(self.output), "manifest": str(self.manifest_path),
                     "num_shards": 1, "shard_index": 0, "resource": "cgics_mllm", "config_data": copy.deepcopy(self.config)}
        self.output = self.scope / "results/single/synthetic_fresh"
        self.config.update(experiment_id="z_single", variant="single", adapter={})
        self.run_fake(adapter_factory=lambda _: PairedAdapter())
        self.plan()
        self.plan_data["tasks"].append(gate_task)
        (self.scope / "plan.json").write_text(json.dumps(self.plan_data))
        result = self.analyze()
        comparison = result["paired_comparisons"][0]
        self.assertEqual((comparison["left"], comparison["right"]), ("z_single", "a_gate"))
        self.assertEqual(comparison["right_policy_trigger_strata"]["true"]["query_targets"], 3)
    def test_random_gate_draw_bound_to_original_input_hashes_and_query_seed(self):
        self.config.update(variant="random_gate", adapter={"random_search_probability": .5})
        self.run_fake(adapter_factory=lambda _: RandomGateAdapter())
        self.plan()
        self.assertEqual(self.analyze()["formal_ok"], 3)
        p = next((self.output / "episodes").glob("*.json"))
        row = json.loads(p.read_text())
        row["artifacts"]["batch2_gate"]["random_decision"]["input_fingerprint"] = "tampered"
        p.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "identity/draw"):
            self.analyze()
    def test_nested_alternatives_evaluated_after_gt_boundary(self):
        result = self.run_fake(adapter_factory=lambda _: PairedAdapter())
        self.assertEqual(result["new_ok"], 3)
        for row in self.rows():
            self.assertEqual(row["iou"], .5)
            self.assertEqual(row["alternative_predictions"]["ref_only"]["iou"], 1)
            self.assertEqual(row["alternative_predictions"]["query_only"]["iou"], 0)
            self.assertNotIn("inference_seconds", row["alternative_predictions"]["ref_only"])
        self.assertEqual(len(list((self.output / "predictions").glob("*.png"))), 9)
    def test_invalid_alternative_never_becomes_zero_filled_success(self):
        for invalid in ("shape", "binary", "missing"):
            with self.subTest(invalid=invalid):
                self.output = self.scope / ("results-" + invalid)
                result = self.run_fake(adapter_factory=lambda _: PairedAdapter(invalid=invalid))
                self.assertEqual(result["new_errors"], 3)
                for row in self.rows():
                    self.assertIsNone(row["counts"])
                    self.assertIsNone(row["iou"])
                    self.assertNotIn("alternative_predictions", row)
                    self.assertEqual(row["num_shards"], 1)
    def test_unsafe_rule_is_error(self):
        self.config["variant"] = "test_alternatives"
        result = self.run_fake(adapter_factory=lambda _: PairedAdapter(invalid="unsafe"))
        self.assertEqual(result["new_errors"], 3)
        self.assertFalse((self.scope / "escape.png").exists())
    def test_completed_resume_validates_alternatives_without_inference(self):
        self.run_fake(adapter_factory=lambda _: PairedAdapter())
        result = self.run_fake(resume=True, adapter_factory=lambda _: self.fail("unexpected inference"))
        self.assertEqual(result["already_present"], 3)
        alternate = next((self.output / "predictions").glob("*--query_only.png"))
        alternate.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "prediction has changed"):
            self.run_fake(resume=True, adapter_factory=lambda _: self.fail("unexpected inference"))
    def test_missing_alternative_cannot_be_reused(self):
        self.run_fake(adapter_factory=lambda _: PairedAdapter())
        p = next((self.output / "episodes").glob("*.json"))
        row = json.loads(p.read_text())
        row["alternative_predictions"].pop("ref_only")
        p.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "missing its paired"):
            self.run_fake(resume=True, adapter_factory=lambda _: PairedAdapter())
    def test_saved_alternative_path_escape_rejected(self):
        self.run_fake(adapter_factory=lambda _: PairedAdapter())
        p = next((self.output / "episodes").glob("*.json"))
        row = json.loads(p.read_text())
        row["alternative_predictions"]["ref_only"]["prediction_path"] = "../../outside.png"
        p.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.run_fake(resume=True, adapter_factory=lambda _: PairedAdapter())
    def test_full_score_analysis_does_not_triple_formal_count_or_latency(self):
        self.run_fake(adapter_factory=lambda _: PairedAdapter())
        self.plan()
        result = self.analyze()
        self.assertEqual(result["formal_observed_records"], 3)
        self.assertEqual(len(result["configuration_results"]), 3)
        self.assertEqual(len(result["paired_comparisons"]), 3)
        self.assertEqual(result["observed_cost_points"], [])
        self.assertTrue(all(s["inference_seconds"] is None for s in result["configuration_results"]))
        self.assertEqual(result["formal_output_coverage"][0]["verified_fixed_pools"], 3)
    def test_missing_and_error_records_remain_visible(self):
        self.run_fake(adapter_factory=lambda _: PairedAdapter(invalid="shape"))
        self.plan()
        next((self.output / "episodes").glob("*.json")).unlink()
        result = self.analyze()
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["formal_errors"], 2)
        self.assertEqual(result["formal_missing_records_in_bound_outputs"], 1)
        self.assertTrue(all(s["episode_mean_iou"] is None for s in result["configuration_results"]))
    def test_duplicate_result_file_is_not_counted_twice(self):
        self.run_fake(adapter_factory=lambda _: PairedAdapter())
        self.plan()
        p = next((self.output / "episodes").glob("*.json"))
        (p.parent / "duplicate.json").write_bytes(p.read_bytes())
        with self.assertRaisesRegex(ValueError, "Unexpected/duplicate"):
            self.analyze()
    def test_wrong_shard_identity_rejected(self):
        self.run_fake(adapter_factory=lambda _: PairedAdapter())
        self.plan()
        p = next((self.output / "episodes").glob("*.json"))
        row = json.loads(p.read_text())
        row["shard_index"] = 1
        p.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "modulo shard"):
            self.analyze()
    def test_config_identity_rejected(self):
        self.run_fake(adapter_factory=lambda _: PairedAdapter())
        self.plan()
        self.plan_data["tasks"][0]["config_data"]["seed"] = 9
        (self.scope / "plan.json").write_text(json.dumps(self.plan_data))
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            self.analyze()
    def test_candidate_pool_mutation_and_wrong_winner_rejected(self):
        self.run_fake(adapter_factory=lambda _: PairedAdapter())
        row = self.rows()[0]
        verify_score_pool(row)
        changed = copy.deepcopy(row)
        changed["artifacts"]["batch2_score_audit"]["candidates"][0]["prompt"] = "tampered"
        with self.assertRaisesRegex(ValueError, "Candidate pool differs"):
            verify_score_pool(changed)
        changed = copy.deepcopy(row)
        changed["artifacts"]["batch2_score_audit"]["rules"]["ref_only"]["selected_candidate_index"] = 0
        with self.assertRaisesRegex(ValueError, "Selection differs"):
            verify_score_pool(changed)
    def test_null_and_bad_confusion_metrics_rejected(self):
        verify_counts({"status": "error", "counts": None, "iou": None})
        with self.assertRaisesRegex(ValueError, "null"):
            verify_counts({"status": "error", "counts": None, "iou": 0})
        with self.assertRaisesRegex(ValueError, "disagrees"):
            verify_counts({"counts": {"tp": 1, "fp": 0, "fn": 1, "tn": 0, "ignored": 0}, "iou": 1})
    def test_cluster_unit_is_images_not_reference_rows(self):
        # Two concepts from image A remain one resampling cluster, image B one.
        values = {("A", "mask1", "dog", 0): 1, ("A", "mask2", "cat", 0): 1,
                  ("B", "mask3", "bird", 0): 0}
        result = clustered_mean_ci(values, replicates=50)
        self.assertEqual(result["query_targets"], 3)
        self.assertEqual(result["query_image_clusters"], 2)
        self.assertAlmostEqual(result["mean"], 2/3)
        single = clustered_mean_ci({("A", "m", "dog", 0): 1}, replicates=10)
        self.assertIsNone(single["ci95"])
    def test_gate_canonical_artifact_drives_trigger_strata(self):
        self.run_fake(adapter_factory=lambda _: PairedAdapter())
        left, right = self.rows(), self.rows()
        for row in left + right:
            row["_reference_mask_sha256"] = [self.manifest["episodes"][0]["references"][0]["mask_sha256"]]
        for i, row in enumerate(right):
            row["artifacts"]["batch2_gate"] = {"search_triggered": i == 0}
        result = paired_statistics(left, right, replicates=20)
        self.assertEqual(set(result["right_policy_trigger_strata"]), {"true", "false"})
    def test_empty_empty_gate_cannot_be_accepted(self):
        agreement = prediction_agreement(np.zeros((2, 3)), np.zeros((2, 3)))
        row = {"artifacts": {"batch2_gate": {"policy": "agreement_gate", "query_gt_used": False,
                                            "agreement": agreement, "threshold": .7, "search_triggered": True,
                                            "initial_concept": "dog", "final_concept": "animal"}}}
        config = {"variant": "agreement_gate", "adapter": {"gate_threshold": .7}}
        verify_gate(row, config)
        row["artifacts"]["batch2_gate"]["search_triggered"] = False
        with self.assertRaisesRegex(ValueError, "violates"):
            verify_gate(row, config)


if __name__ == "__main__":
    unittest.main()
