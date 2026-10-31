"""Synthetic contract tests only; no claim of model/dataset performance."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ics.adapters.base import Prediction
from ics.runner import (canonical_sha256, expand_config, file_sha256,
                        numpy_binary_counts, run_experiment, scientific_config)
from ics_eval import binary_counts


def fixture(root, count=3):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    def asset(name, pixel):
        image = root / (name + ".png")
        mask = root / (name + "_mask.png")
        Image.fromarray(np.full((2, 3, 3), pixel, dtype=np.uint8)).save(image)
        Image.fromarray(np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)).save(mask)
        return {"image": image.name, "mask": mask.name,
                "image_sha256": file_sha256(image), "mask_sha256": file_sha256(mask)}
    ref = asset("reference", 200)
    episodes = []
    for i in range(count):
        episodes.append({"episode_id": f"synthetic-{i}", "dataset": "synthetic_test_fixture",
                         "dataset_version": "test-only", "split": "unit_test", "fold": 0,
                         "track": "semantic", "concept_id": str(i % 2), "reference_seed": 0,
                         "query": asset(f"query-{i}", 30 + i), "references": [ref],
                         "mask_semantics": "test binary foreground", "ignore_label": 255})
    manifest = {"schema_version": 1, "status": "ready", "protocol_id": "test-only-not-scientific",
                "episodes": episodes}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


class FakeAdapter:
    def __init__(self, fail_name=None, wrong_shape=False):
        self.fail_name, self.wrong_shape = fail_name, wrong_shape
        self.seen = []
    def metadata(self):
        return {"kind": "synthetic_test_double"}
    def close(self):
        pass
    def predict(self, inputs, *, seed=0):
        assert set(inputs) == {"query_image", "references"}
        assert all(set(ref) == {"image", "mask"} for ref in inputs["references"])
        assert all(Path(p).is_absolute() for ref in inputs["references"] for p in ref.values())
        self.seen.append((inputs, seed))
        if self.fail_name and self.fail_name in inputs["query_image"]:
            raise RuntimeError("Deliberate synthetic failure")
        mask = np.ones((1, 1), dtype=bool) if self.wrong_shape else np.array([[1, 0, 0], [0, 0, 0]], dtype=bool)
        return Prediction(mask, {"real_in_test_call_count": 1}, {"test": True})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.manifest_path, self.manifest = fixture(self.data)
        self.config = {"experiment_id": "unit-test", "method": "test_double", "variant": "full", "seed": 3, "adapter": {}}
        # Other agents can edit unrelated project modules while tests run.
        self.code_patch = patch("ics.runner.project_code_digest", return_value="test-code")
        self.code_patch.start()
    def tearDown(self):
        self.code_patch.stop()
        self.tmp.cleanup()
    def run_shard(self, **kwargs):
        return run_experiment(self.config, manifest_path=self.manifest_path,
                              dataset_root=self.data, output_dir=self.root / "run", **kwargs)
    def test_shards_resume_and_whitelisted_model_inputs(self):
        adapter = FakeAdapter()
        a = self.run_shard(shard_index=0, num_shards=2, adapter_factory=lambda _: adapter)
        b = self.run_shard(shard_index=1, num_shards=2, adapter_factory=lambda _: adapter)
        self.assertEqual((a["new_ok"], b["new_ok"]), (2, 1))
        self.assertEqual(len(adapter.seen), 3)
        result = self.run_shard(resume=True, adapter_factory=lambda _: self.fail("must not load model on completed resume"))
        self.assertEqual(result["already_present"], 3)
        self.assertEqual(result["errors_in_shard"], 0)
        with self.assertRaises(FileExistsError):
            self.run_shard(adapter_factory=lambda _: FakeAdapter())
    def test_weights_change_invalidates_resume(self):
        weight = self.root / "weights.pt"
        weight.write_bytes(b"synthetic-weight-v1")
        self.config["adapter"]["checkpoint"] = str(weight)
        self.run_shard(adapter_factory=lambda _: FakeAdapter())
        weight.write_bytes(b"synthetic-weight-v2")
        with self.assertRaisesRegex(ValueError, "binding differs"):
            self.run_shard(resume=True, adapter_factory=lambda _: FakeAdapter())
    def test_corrupted_prediction_rejected_on_resume(self):
        self.run_shard(adapter_factory=lambda _: FakeAdapter())
        next((self.root / "run/predictions").glob("*.png")).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "prediction has changed"):
            self.run_shard(resume=True, adapter_factory=lambda _: FakeAdapter())
    def test_error_kept_and_explicit_retry(self):
        result = self.run_shard(adapter_factory=lambda _: FakeAdapter(fail_name="query-1"))
        self.assertEqual(result["new_errors"], 1)
        result = self.run_shard(resume=True, adapter_factory=lambda _: self.fail("no retry without flag"))
        self.assertEqual(result["errors_in_shard"], 1)
        result = self.run_shard(resume=True, retry_failed=True, adapter_factory=lambda _: FakeAdapter())
        self.assertEqual(result["new_ok"], 1)
        self.assertEqual(result["errors_in_shard"], 0)
    def test_wrong_shape_is_failure_not_resized_silently(self):
        result = self.run_shard(adapter_factory=lambda _: FakeAdapter(wrong_shape=True))
        self.assertEqual(result["new_errors"], 3)
        for row in (self.root / "run/episodes").glob("*.json"):
            parsed = json.loads(row.read_text())
            self.assertIsNone(parsed["counts"])
    def test_vectorized_counts_equal_original_contract(self):
        random = np.random.default_rng(4)
        for _ in range(20):
            prediction = random.integers(0, 2, (8, 9))
            target = random.choice([0, 1, 255], size=(8, 9))
            actual = numpy_binary_counts(prediction, target, ignore_label=255)
            expected = binary_counts(prediction.ravel().tolist(), target.ravel().tolist(), ignore_label=255)
            self.assertEqual(actual, expected)
    def test_unset_environment_variable_rejected(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                expand_config({"checkpoint": "${MISSING_TEST_VAR}/file.pt"})
    def test_managed_local_port_only_normalization(self):
        model = self.root / "model"
        model.mkdir()
        config = copy.deepcopy(self.config)
        config["runtime"] = {"managed_local_mllm": True, "mllm_model_path": str(model)}
        config["adapter"]["endpoint"] = "http://127.0.0.1:1234/v1"
        first = canonical_sha256(scientific_config(config))
        config["adapter"]["endpoint"] = "http://127.0.0.1:5678/v1"
        self.assertEqual(first, canonical_sha256(scientific_config(config)))
        config["adapter"]["endpoint"] = "https://example.com/v1"
        with self.assertRaises(ValueError):
            scientific_config(config)


if __name__ == "__main__":
    unittest.main()
