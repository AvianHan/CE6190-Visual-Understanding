"""Synthetic contract/branch tests. No checkpoints, API server, or model inference."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from ics.adapters.cgics import CGICSAdapter, _OfficialBackend, component_settings

try:
    import torch
    from torchvision.transforms import functional as TF
except ImportError:
    torch = TF = None


class DummyBackend:
    def __init__(self):
        self.events = []
        self.telemetry = {}
        self.artifacts = {}

    def metadata(self):
        return {"backend": "unit_test_fake"}

    def reset(self, seed):
        self.events.append(("reset", seed))
        self.telemetry = {"mllm_requests": 0}
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
        return "dog"

    def visual(self, prepared):
        self.events.append("visual")
        return "query_boxes_from_visual_masks"

    def direct_mask(self, prepared, masks):
        self.events.append(("direct", masks))
        return np.zeros((4, 6), dtype=bool)

    def final(self, prepared, prompt, masks):
        self.events.append(("final", prompt, masks))
        return np.ones((4, 6), dtype=bool)

    def clear(self):
        self.events.append("clear")

    def close(self):
        self.clear()


def inputs():
    return {"query_image": "/synthetic/query.png", "references": [{"image": "/synthetic/reference.png", "mask": "/synthetic/reference_mask.png"}]}


class ComponentTests(unittest.TestCase):
    def test_routes_remain_distinct(self):
        for variant, concept, visual in (
            ("full", "search", "boxes"), ("text_only", "search", "none"),
            ("visual_only", "none", "boxes"), ("concat", "none", "concat"),
            ("no_search", "candidates", "boxes"), ("single_concept_text", "single", "none"),
        ):
            with self.subTest(variant=variant):
                backend = DummyBackend()
                adapter = CGICSAdapter({"variant": variant}, _backend=backend)
                result = adapter.predict(inputs(), seed=17)
                self.assertEqual(result.mask.shape, (4, 6))
                self.assertEqual("visual" in backend.events, visual != "none")
                self.assertEqual(any(isinstance(e, tuple) and e[0] == "select" for e in backend.events), concept != "none")
                self.assertEqual(any(isinstance(e, tuple) and e[0] == "direct" for e in backend.events), visual == "concat")
                self.assertEqual(any(isinstance(e, tuple) and e[0] == "final" for e in backend.events), visual != "concat")
                self.assertEqual(backend.events[-1], "clear")
                if concept == "none":
                    self.assertEqual(result.telemetry["mllm_requests"], 0)

    def test_visual_only_performs_final_query_inference(self):
        backend = DummyBackend()
        CGICSAdapter({"variant": "visual_only"}, _backend=backend).predict(inputs())
        self.assertIn(("final", None, "query_boxes_from_visual_masks"), backend.events)

    def test_query_gt_or_class_metadata_rejected_before_backend(self):
        for extra in ("query_mask", "concept_id", "class_name", "query_gt"):
            backend = DummyBackend()
            adapter = CGICSAdapter({}, _backend=backend)
            bad = dict(inputs(), **{extra: "forbidden"})
            with self.assertRaises(ValueError):
                adapter.predict(bad)
            self.assertEqual(backend.events, [])

    def test_failure_always_clears_image_states(self):
        backend = DummyBackend()
        def fail(*args):
            raise RuntimeError("synthetic failure")
        backend.select = fail
        adapter = CGICSAdapter({}, _backend=backend)
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            adapter.predict(inputs())
        self.assertEqual(backend.events[-1], "clear")
        diagnostics = adapter.failure_diagnostics()
        self.assertEqual(diagnostics["last_stage"], "concept_reasoning_s")
        self.assertIn("concept_reasoning_s", diagnostics["telemetry"]["stages"])
        with self.assertRaises(ValueError):
            adapter.predict(dict(inputs(), query_gt="forbidden"))
        self.assertEqual(adapter.failure_diagnostics()["telemetry"], {})

    def test_impossible_or_mislabelled_combinations_rejected(self):
        for config in (
            {"variant": "custom", "concept_mode": "none", "visual_mode": "none"},
            {"variant": "custom", "concept_mode": "search", "visual_mode": "concat"},
            {"variant": "full", "concept_mode": "none"},
            {"variant": "unknown"}, {"BON": 0}, {"alpha": float("nan")},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                component_settings(config)

    def test_real_counter_wrapper_counts_actual_invocations(self):
        backend = _OfficialBackend.__new__(_OfficialBackend)
        backend.telemetry = {"sam_prompt_head_calls": 0}
        module = SimpleNamespace(forward=lambda x: x + 1)
        backend._instrument(module, "forward", "sam_prompt_head_calls")
        self.assertEqual(module.forward(3), 4)
        self.assertEqual(module.forward(8), 9)
        self.assertEqual(backend.telemetry["sam_prompt_head_calls"], 2)

    def test_missing_usage_is_unknown_not_zero(self):
        backend = DummyBackend()
        def select(prepared, mode):
            backend.telemetry.update(mllm_usage_missing_requests=1, mllm_prompt_tokens=42, mllm_completion_tokens=3)
            return "dog"
        backend.select = select
        result = CGICSAdapter({}, _backend=backend).predict(inputs())
        self.assertIsNone(result.telemetry["mllm_prompt_tokens"])
        self.assertEqual(result.telemetry["mllm_prompt_tokens_reported_partial"], 42)

    def test_mllm_usage_and_sampling_are_recorded_from_actual_client_result(self):
        calls = []
        class Client:
            def __init__(self):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            def with_options(self, **kwargs):
                self.options = kwargs
                return self

            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=53, completion_tokens=7),
                                       choices=[SimpleNamespace(message=SimpleNamespace(content="dog"))])

        class MLLM:
            def __init__(self, **kwargs):
                self.client = Client()
                self.api_model = kwargs["model_name"]

            def generate(self, **kwargs):
                return [self._call_api_once([], max_tokens=128)]

        backend = _OfficialBackend.__new__(_OfficialBackend)
        backend.config = {"llm_seed_from_episode": True, "llm_temperature": 0.0}
        backend.seed = 42
        backend.mllm_module = SimpleNamespace(MLLM=MLLM)
        backend.telemetry = {"mllm_requests": 0, "mllm_prompt_tokens": 0, "mllm_completion_tokens": 0,
                             "mllm_usage_missing_requests": 0, "mllm_request_wall_s": 0.0}
        backend.artifacts = {"mllm_responses": [], "generation_events": []}
        mllm = backend._make_mllm()
        self.assertEqual(mllm.generate(mode="initial"), ["dog"])
        self.assertEqual(mllm.generate(mode="child", base_prompt="dog"), ["dog"])
        self.assertEqual(backend.telemetry["mllm_requests"], 2)
        self.assertEqual(backend.telemetry["mllm_prompt_tokens"], 106)
        self.assertEqual(backend.telemetry["mllm_completion_tokens"], 14)
        self.assertEqual([c["seed"] for c in calls], [42, 43])
        self.assertEqual(mllm.client.options["max_retries"], 0)


@unittest.skipIf(torch is None, "torch/torchvision unavailable; install model environment for tensor boundary checks")
class TensorBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        Image.new("RGB", (20, 11), (20, 30, 40)).save(root / "query.png")
        Image.new("RGB", (17, 13), (90, 100, 110)).save(root / "reference.png")
        raw = np.zeros((13, 17), dtype=np.uint8)
        raw[3:11, 2:12] = 255
        Image.fromarray(raw).save(root / "mask.png")
        self.inputs = {"query_image": str(root / "query.png"), "references": [{"image": str(root / "reference.png"), "mask": str(root / "mask.png")}]}
        self.backend = _OfficialBackend.__new__(_OfficialBackend)
        b = self.backend
        b.torch, b.np, b.TF = torch, np, TF
        b.device, b.image_size = "cpu", 16
        b.artifacts = {}

    def tearDown(self):
        self.directory.cleanup()

    def test_prepare_preserves_binary_reference_and_square_working_images(self):
        prepared = self.backend.prepare(self.inputs)
        self.assertEqual(prepared["original_size"], (20, 11))
        self.assertEqual(tuple(prepared["masks"][0].shape), (16, 16))
        self.assertTrue(set(prepared["masks"][0].unique().tolist()).issubset({0., 1.}))
        self.assertEqual(prepared["cats"][0].size, (32, 16))
        self.assertEqual(tuple(prepared["query_tensor"].shape), (3, 16, 16))

    def test_visual_route_slices_query_half_and_retains_empty_reference_predictions(self):
        b = self.backend
        prepared = b.prepare(self.inputs)
        instances = torch.zeros((2, 1, 16, 32), dtype=torch.bool)
        instances[0, 0, 2:8, 2:8] = True  # reference-side instance
        instances[1, 0, 5:12, 21:28] = True  # query-side instance
        b.split = lambda mask: mask.unsqueeze(0)
        seen = []
        def infer(state, **kwargs):
            seen.append(kwargs["mask"].clone())
            return {"instances": instances}
        b.segmenter = SimpleNamespace(cat_states=[{}], infer=infer)
        visual = b.visual(prepared)
        self.assertEqual(tuple(visual.shape), (2, 16, 16))
        self.assertFalse(bool(visual[0].any()))
        self.assertEqual(int(visual[1].sum()), 49)
        self.assertFalse(bool(seen[0][:, :, 16:].any()))
        restored = b.direct_mask(prepared, visual)
        self.assertEqual(restored.shape, (11, 20))
        self.assertEqual(restored.dtype, np.bool_)

    def test_zero_instance_concatenation_is_valid_empty_prediction(self):
        prepared = self.backend.prepare(self.inputs)
        output = self.backend.direct_mask(prepared, torch.zeros((0, 16, 16), dtype=torch.bool))
        self.assertEqual(output.shape, (11, 20))
        self.assertFalse(output.any())

    def test_label_map_is_rejected_instead_of_treating_all_classes_as_foreground(self):
        mask_path = self.inputs["references"][0]["mask"]
        Image.fromarray(np.full((13, 17), 2, dtype=np.uint8)).save(mask_path)
        with self.assertRaisesRegex(ValueError, "mapped binary"):
            self.backend.prepare(self.inputs)


if __name__ == "__main__":
    unittest.main()
