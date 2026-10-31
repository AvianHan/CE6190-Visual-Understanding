import copy
import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from ics_eval import (binary_counts, inference_view, manifest_fingerprint,
                      summarize_counts, validate_manifest)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((Path(__file__).resolve().parents[1] /
                                    "examples/episodes.example.json").read_text())

    def test_query_ground_truth_and_class_names_never_enter_model_input(self):
        episode = copy.deepcopy(self.manifest["episodes"][0])
        episode["secret_extra_annotation"] = "leak"
        view = inference_view(episode)
        self.assertEqual(set(view), {"query_image", "references"})
        serial = json.dumps(view)
        self.assertNotIn(episode["query"]["mask"], serial)
        self.assertNotIn(episode["concept_id"], serial)
        self.assertNotIn("leak", serial)

    def test_duplicate_content_and_ids_rejected(self):
        validate_manifest(self.manifest, allow_example=True)
        changed = copy.deepcopy(self.manifest)
        changed["episodes"][0]["references"][0]["image_sha256"] = changed["episodes"][0]["query"]["image_sha256"]
        with self.assertRaises(ValueError):
            validate_manifest(changed, allow_example=True)
        self.manifest["episodes"].append(copy.deepcopy(self.manifest["episodes"][0]))
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest, allow_example=True)

    def test_reference_change_changes_manifest_identity(self):
        changed = copy.deepcopy(self.manifest)
        changed["episodes"][0]["references"][0]["mask_sha256"] = "e" * 64
        self.assertNotEqual(manifest_fingerprint(changed), manifest_fingerprint(self.manifest))

    def test_synthetic_examples_cannot_pass_as_ready(self):
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest)

    def test_file_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = self.manifest["episodes"][0]
            for asset in [episode["query"], *episode["references"]]:
                for kind in ("image", "mask"):
                    path = root / asset[kind]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # Cross the streaming chunk boundary using real file bytes.
                    data = asset[kind].encode() * 70000
                    path.write_bytes(data)
                    asset[kind + "_sha256"] = hashlib.sha256(data).hexdigest()
            self.manifest["status"] = "ready"
            validate_manifest(self.manifest, root=root)
            (root / episode["references"][0]["mask"]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_manifest(self.manifest, root=root)

    def test_ignore_and_invalid_labels(self):
        self.assertEqual(binary_counts([1, 0, 1, 0, 1], [1, 1, 0, 0, 255], ignore_label=255),
                         {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "ignored": 1})
        with self.assertRaises(ValueError):
            binary_counts([0], [2])

    def test_three_aggregations_differ_for_unequal_sizes_and_classes(self):
        base = dict(run_id="test", method="synthetic", variant="test", protocol_id="unit-test",
                    manifest_sha256="0" * 64, dataset="test", split="test", fold=0,
                    track="semantic", nshot=1)
        rows = [dict(base, episode_id="a", concept_id="c", counts=dict(tp=1, fp=0, fn=0, tn=0, ignored=0)),
                dict(base, episode_id="b", concept_id="c", counts=dict(tp=0, fp=0, fn=9, tn=0, ignored=0)),
                dict(base, episode_id="d", concept_id="d", counts=dict(tp=4, fp=1, fn=0, tn=0, ignored=0))]
        summary = summarize_counts(rows)
        self.assertAlmostEqual(summary["macro_class_pooled_iou"], 0.45)
        self.assertAlmostEqual(summary["class_mean_episode_iou"], 0.65)
        self.assertAlmostEqual(summary["mean_episode_iou"], 0.6)
        rows[1]["track"] = "part"
        with self.assertRaises(ValueError):
            summarize_counts(rows)


if __name__ == "__main__":
    unittest.main()
