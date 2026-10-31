"""Metric/failure accounting tests on synthetic masks only."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ics.analysis import (analyze_runs, area_statistics, load_run,
                          reference_statistics, repeated_input_statistics, summarize_run)
from ics.runner import atomic_json, canonical_sha256, run_experiment
from test_runner import FakeAdapter, fixture


class AnalysisTests(unittest.TestCase):
    def test_retry_preserves_failed_attempt_and_cumulative_cost(self):
        class CountingFake(FakeAdapter):
            def predict(self, inputs, *, seed=0):
                prediction = super().predict(inputs, seed=seed)
                prediction.telemetry["mllm_requests"] = 2
                return prediction
            def failure_diagnostics(self):
                return {"telemetry": {"mllm_requests": 3}, "last_stage": "synthetic_failure"}
        with tempfile.TemporaryDirectory() as tmp, patch("ics.runner.project_code_digest", return_value="test-code"):
            root = Path(tmp)
            manifest_path, _ = fixture(root / "data", count=1)
            config = {"experiment_id": "retry-test", "method": "test_double", "variant": "full", "seed": 0, "adapter": {}}
            kwargs = dict(manifest_path=manifest_path, dataset_root=root / "data", output_dir=root / "run")
            run_experiment(config, **kwargs, adapter_factory=lambda _: CountingFake(fail_name="query-0"))
            old_file = next((root / "run/episodes").glob("*.json"))
            old = json.loads(old_file.read_text())
            # Simulate death immediately after archiving but before replacement.
            archive = root / "run/attempts" / old_file.stem / (canonical_sha256(old) + ".json")
            atomic_json(archive, old, create_only=True)
            self.assertEqual(len(load_run(root / "run")["prior_attempts"]), 0)
            run_experiment(config, **kwargs, resume=True, retry_failed=True, adapter_factory=lambda _: CountingFake())
            run = load_run(root / "run")
            self.assertEqual(len(run["prior_attempts"]), 1)
            self.assertEqual(run["prior_attempts"][0]["failure_diagnostics"]["telemetry"]["mllm_requests"], 3)
            self.assertIn("traceback", run["prior_attempts"][0])
            summary = summarize_run(run)
            self.assertEqual((summary["ok"], summary["errors"], summary["archived_failed_attempts"]), (1, 0, 1))
            group = summary["groups"][0]
            self.assertEqual(group["committed_attempts_observed"], 2)
            self.assertEqual(group["all_attempt_numeric_telemetry_totals"]["mllm_requests"], 5)
            self.assertAlmostEqual(group["episode_wall_seconds_total_all_committed_attempts"],
                                   old["episode_wall_seconds"] + run["rows"][0]["episode_wall_seconds"])
    def test_stochastic_repeats_group_ephemeral_ports(self):
        runs = []
        for seed, port, iou in ((1, 1234, .4), (2, 5678, .8)):
            config = {"experiment_id": f"seed-{seed}", "method": "cgics", "variant": "full", "seed": seed,
                      "adapter": {"name": "cgics", "seed": seed, "endpoint": f"http://127.0.0.1:{port}/v1"},
                      "runtime": {"managed_local_mllm": True, "mllm_model_path": "/unavailable/downloaded-log-model-path"}}
            row = {"status": "ok", "iou": iou, "base_seed": seed, "manifest_sha256": "same", "episode_id": "same"}
            runs.append({"meta": {"config": config}, "rows": [row]})
        result = repeated_input_statistics(runs)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["population_std_repeated_input"], .2)
    def test_area_bins_keep_failed_episodes(self):
        base = {"dataset": "synthetic", "split": "unit", "fold": 0, "track": "semantic", "nshot": 1}
        rows = [dict(base, status="ok", iou=.5, target_area_fraction=.005),
                dict(base, status="error", iou=None, target_area_fraction=.005),
                dict(base, status="ok", iou=.8, target_area_fraction=.1),
                dict(base, status="error", iou=None, target_area_fraction=None)]
        result = {r["bin_index"]: r for r in area_statistics(rows)}
        self.assertEqual(result[0]["observed_episodes"], 2)
        self.assertEqual(result[0]["errors"], 1)
        self.assertEqual(result[2]["mean_episode_iou"], .8)
        self.assertEqual(result[None]["errors"], 1)
        with self.assertRaises(ValueError):
            area_statistics(rows, thresholds=[.1, .01])
    def test_partial_failure_coverage_and_html(self):
        with tempfile.TemporaryDirectory() as tmp, patch("ics.runner.project_code_digest", return_value="test-code"):
            root = Path(tmp)
            manifest_path, _ = fixture(root / "data", count=5)
            config = {"experiment_id": "synthetic", "method": "test_double", "variant": "full", "seed": 2, "adapter": {}}
            run_experiment(config, manifest_path=manifest_path, dataset_root=root / "data",
                           output_dir=root / "run", shard_index=0, num_shards=2,
                           adapter_factory=lambda _: FakeAdapter(fail_name="query-2"))
            summary = summarize_run(load_run(root / "run"))
            self.assertEqual((summary["ok"], summary["errors"], summary["pending"]), (2, 1, 2))
            self.assertEqual(summary["groups"][0]["metrics_status"], "conditional_on_successful_episodes")
            result = analyze_runs([root / "run"], root / "analysis", case_images=1)
            self.assertEqual((result["errors"], result["pending"]), (1, 2))
            self.assertTrue((root / "analysis/index.html").is_file())
            self.assertTrue(list((root / "analysis/cases").glob("*.png")))
    def test_reference_std_is_within_query_and_distinct_references(self):
        def row(query, ref, iou):
            return {"dataset": "test", "split": "unit", "fold": 0, "track": "semantic", "nshot": 1,
                    "concept_id": "dog", "query_image_sha256": query, "query_mask_sha256": "gt-"+query,
                    "reference_image_sha256": [ref], "status": "ok", "iou": iou}
        rows = [row("a", "r1", .2), row("a", "r2", .8), row("b", "r1", .9), row("b", "r2", .9)]
        result = reference_statistics(rows)
        by_query = {r["query_image_sha256"]: r for r in result}
        self.assertAlmostEqual(by_query["a"]["population_std_across_references"], .3)
        self.assertAlmostEqual(by_query["b"]["population_std_across_references"], 0.)
        rows.append(row("a", "r1", .2))
        a = [r for r in reference_statistics(rows) if r["query_image_sha256"] == "a"][0]
        self.assertEqual(a["distinct_reference_sets"], 2)
        self.assertAlmostEqual(a["population_std_across_references"], .3)
    def test_paired_comparisons_require_same_manifest(self):
        with tempfile.TemporaryDirectory() as tmp, patch("ics.runner.project_code_digest", return_value="test-code"):
            root = Path(tmp)
            manifest_path, _ = fixture(root / "data", count=2)
            for name in ("left", "right"):
                config = {"experiment_id": name, "method": "test_double", "variant": name, "seed": 2, "adapter": {}}
                run_experiment(config, manifest_path=manifest_path, dataset_root=root / "data", output_dir=root / name,
                               adapter_factory=lambda _: FakeAdapter())
            analyze_runs([root / "left", root / "right"], root / "analysis", case_images=0)
            report = json.loads((root / "analysis/analysis.json").read_text())
            pair = report["paired_comparisons"][0]
            self.assertEqual(pair["valid_paired_episodes"], 2)
            self.assertEqual(pair["mean_episode_delta_right_minus_left"], 0.)


if __name__ == "__main__":
    unittest.main()
