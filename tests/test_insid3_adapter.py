"""Synthetic adapter checks. These are NOT model inference or benchmark results."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import os
import subprocess
import sys

import numpy as np
from PIL import Image

from ics.adapters.insid3 import (
    SOURCE_SHA256, _effective_settings, _read_inputs, _verify_source,
)


class INSID3BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Image.fromarray(np.zeros((5, 7, 3), dtype=np.uint8)).save(self.root / "query.png")
        Image.fromarray(np.ones((3, 4, 3), dtype=np.uint8)).save(self.root / "reference.png")
        Image.fromarray(np.array([[0, 0, 0, 0], [0, 255, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)).save(self.root / "mask.png")
        self.inputs = {"query_image": str(self.root / "query.png"),
                       "references": [{"image": str(self.root / "reference.png"),
                                       "mask": str(self.root / "mask.png")}]}

    def tearDown(self):
        self.temp.cleanup()

    def test_query_gt_cannot_cross_adapter_boundary(self):
        for forbidden in ("query_mask", "concept_id", "category_name"):
            contaminated = {**self.inputs, forbidden: "should never reach a model"}
            with self.assertRaises(ValueError):
                _read_inputs(contaminated)

    def test_reference_encoding_and_original_sizes(self):
        query, refs = _read_inputs(self.inputs)
        self.assertEqual(query.size, (7, 5))
        self.assertEqual(refs[0][0].size, (4, 3))
        self.assertEqual(int((np.asarray(refs[0][1]) > 0).sum()), 1)

    def test_empty_multiclass_and_misaligned_reference_masks_fail(self):
        for array in (np.zeros((3, 4), dtype=np.uint8),
                      np.full((3, 4), 7, dtype=np.uint8),
                      np.full((4, 3), 255, dtype=np.uint8)):
            Image.fromarray(array).save(self.root / "mask.png")
            with self.assertRaises(ValueError):
                _read_inputs(self.inputs)

    def test_reference_query_identity_is_rejected(self):
        self.inputs["references"][0]["image"] = self.inputs["query_image"]
        with self.assertRaises(ValueError):
            _read_inputs(self.inputs)

    def test_source_change_rejected_before_importing_torch(self):
        for name in SOURCE_SHA256:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("unexpected changed source")
        with self.assertRaisesRegex(RuntimeError, "differs from audited pin"):
            _verify_source(self.root)

    def test_component_combinations_are_explicit(self):
        settings = _effective_settings({"variant": "no_aggregation", "use_debias": False})
        self.assertFalse(settings["use_debias"])
        self.assertEqual(settings["aggregation"], "seed_only")
        settings = _effective_settings({"variant": "simple_similarity", "use_debias": False})
        self.assertEqual(settings["prediction_mode"], "similarity")
        self.assertFalse(settings["use_debias"])

    def test_patch_size_incompatible_resolution_fails(self):
        with self.assertRaisesRegex(ValueError, "patch size 14"):
            _effective_settings({"encoder": "dinov2", "image_size": 1024})
        for encoder in ("dinov3", "dinov2", "sam3"):
            self.assertEqual(_effective_settings({"encoder": encoder, "image_size": 1008})["image_size"], 1008)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU tensor ablations require torch; no model weights are used")
class INSID3TensorAblationTests(unittest.TestCase):
    """Small constructed tensors distinguish the causal component switches."""
    def setUp(self):
        from ics.adapters.insid3 import _build_model_class
        class Parent:
            def _seed_and_aggregate(self, *args):
                return "upstream-route"
        with mock.patch("ics.adapters.insid3.importlib.import_module") as load:
            load.return_value.downsample_mask = None
            self.make = lambda settings: _build_model_class(Parent, torch, settings)
            self.seed_class = self.make({"aggregation": "seed_only"})
            self.cross_class = self.make({"aggregation": "cross_only"})
            self.full_class = self.make({"aggregation": "self_cross"})
        self.args = (
            torch.tensor([[True, True, True, False]]),
            torch.tensor([[0, 0, 1, 2]]),
            torch.tensor([[1., 0.], [.8, .6], [.6, .8]]), 3,
            torch.tensor([[1.], [0.]]),
            torch.tensor([[[[1., 1., .8, .6]], [[0., 0., .6, .8]]]]),
            torch.tensor([[[[1., 1., .8, .6]], [[0., 0., .6, .8]]]]), 1, 4,
        )

    def test_seed_only_does_not_keep_other_matching_clusters(self):
        instance = self.seed_class()
        output = instance._seed_and_aggregate(*self.args)
        self.assertEqual(output.tolist(), [[True, True, False, False]])

    def test_cross_only_retains_cross_match_but_not_non_candidates(self):
        instance = self.cross_class()
        instance.merge_threshold = .5
        output = instance._seed_and_aggregate(*self.args)
        self.assertEqual(output.tolist(), [[True, True, True, False]])

    def test_full_delegates_to_unmodified_upstream(self):
        instance = self.full_class()
        self.assertEqual(instance._seed_and_aggregate(*self.args), "upstream-route")


@unittest.skipIf(torch is None, "Pinned-source parity requires CPU torch")
class PinnedSourceParityTests(unittest.TestCase):
    def test_full_path_matches_upstream_with_same_synthetic_encoder(self):
        """Optional integration: real upstream processing, toy features, no checkpoint."""
        source = os.environ.get("ICS_TEST_INSID3_SOURCE")
        if not source:
            self.skipTest("Set ICS_TEST_INSID3_SOURCE to the audited local checkout")
        script = r'''
import importlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ics.adapters.insid3 import _verify_source, _add_source, _build_model_class, _effective_settings
source = Path(sys.argv[1]).resolve()
_verify_source(source)
_add_source(source, "utils")
_add_source(source, "models")
Original = importlib.import_module("models.insid3").INSID3
class ToyEncoder(torch.nn.Module):
    def get_intermediate_layers(self, x, n=1, reshape=True):
        rgb = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))
        constant = torch.ones_like(rgb[:, :1]) * .3
        return [torch.cat([rgb, constant], dim=1)]
ref = np.zeros((19, 23, 3), dtype=np.uint8)
ref[:, :] = [20, 30, 230]
ref[:10, :12] = [240, 20, 30]
mask = np.zeros((19, 23), dtype=np.uint8)
mask[:10, :12] = 255
query = np.zeros((17, 29, 3), dtype=np.uint8)
query[:, :] = [20, 30, 230]
query[9:, 15:] = [240, 20, 30]
options = dict(image_size=32, svd_components=1, tau=.6, merge_threshold=.2,
               resize_to_orig_size=True, device="cpu")
settings = _effective_settings(dict(image_size=32, svd_components=1, device="cpu"))
Experimental = _build_model_class(Original, torch, settings)
models = [Original(encoder=ToyEncoder(), **options), Experimental(encoder=ToyEncoder(), **options)]
predictions = []
for model in models:
    model.eval()
    model.set_reference(Image.fromarray(ref), Image.fromarray(mask))
    model.set_target(Image.fromarray(query))
    result = model.segment()
    assert tuple(result.shape) == (17, 29)
    assert result.dtype == torch.bool
    predictions.append(result)
assert torch.equal(*predictions), "Full adapter changed pinned-source mask"
print(json.dumps({"synthetic_upstream_parity": True, "shape": [17, 29]}))
'''
        proc = subprocess.run([sys.executable, "-c", script, source], capture_output=True,
                              text=True, timeout=90)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn('"synthetic_upstream_parity": true', proc.stdout)


if __name__ == "__main__":
    unittest.main()
