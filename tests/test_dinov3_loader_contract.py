"""CPU-only loader contract tests; no PyTorch tensors or real checkpoint load.

Executes the actual pinned DINOv3 factory and _make_dinov3_vit functions.
Only the tensor model, torch.hub/load and network transfer are controlled doubles.
This intentionally does not stand in for a real-model CPU/GPU integration test.
"""
from __future__ import annotations

import ast
from enum import Enum
import hashlib
import importlib
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
from typing import Optional, Union
import unittest

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "tests/fixtures/insid3_before_recovery.py"
PATCHED = ROOT / "ics/adapters/insid3.py"
OFFICIAL = ROOT / "tests/fixtures/dinov3_pinned_backbones.py"
OFFICIAL_SHA256 = "6c9a18aca0b5fcb1fce439213a94a528e072376a6023718e03dbd9fcfaa10deb"


def function_tree(path, names):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name in names]
    assert {n.name for n in selected} == names
    return ast.Module(body=selected, type_ignores=[])


class DummyModule:
    pass


class DummyViT:
    def __init__(self, **kwargs):
        self.architecture = kwargs
        self.initialized = False
        self.loaded = None
        self.strict = None
        self.lifecycle = []

    def init_weights(self):
        self.initialized = True

    def load_state_dict(self, state, strict):
        self.strict = strict
        if strict and set(state) != {"fixture_weight"}:
            raise RuntimeError("fixture state keys mismatch")
        self.loaded = state
        self.lifecycle.append("load")

    def eval(self):
        self.lifecycle.append("eval")
        return self

    def requires_grad_(self, flag):
        self.grad = flag
        self.lifecycle.append("freeze")
        return self

    def to(self, device):
        self.device = device
        self.lifecycle.append("device")
        return self


class Fixture:
    def __init__(self, tmp, payload=None):
        self.tmp = Path(tmp)
        self.source = self.tmp / "pinned_source"
        self.source.mkdir()
        (self.source / "hubconf.py").write_text("# marker only\n")
        self.checkpoint = self.tmp / "dinov3_L.pth"
        self.checkpoint.write_bytes(b"fixture-checkpoint-not-a-model")
        self.payload = payload if payload is not None else {"fixture_weight": 42}
        self.network_calls = []
        self.hub_calls = []
        self.load_calls = []
        package = "_dinov3_loader_contract"
        for name in [package, package + ".models", package + ".hub"]:
            m = ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m
        vi = ModuleType(package + ".models.vision_transformer")
        vi.DinoVisionTransformer = DummyViT
        sys.modules[vi.__name__] = vi

        def block_network(*args, **kwargs):
            self.network_calls.append((args, kwargs))
            raise AssertionError("loader attempted a URL transfer")

        self.official = {
            "__name__": package + ".hub.backbones", "__package__": package + ".hub",
            "Enum": Enum, "Union": Union, "Optional": Optional,
            "_safe_load_state_dict_from_url": block_network,
            "_make_dinov3_vit_model_url": block_network,
            "convert_path_or_url_to_url": block_network,
        }
        names = {"Weights", "_make_dinov3_vit", "dinov3_vits16",
                 "dinov3_vitb16", "dinov3_vitl16"}
        exec(compile(function_tree(OFFICIAL, names), str(OFFICIAL), "exec"), self.official)
        self.torch = SimpleNamespace(hub=SimpleNamespace(load=self.hub_load),
                                     load=self.torch_load,
                                     nn=SimpleNamespace(Module=DummyModule))

    def hub_load(self, source_path, entry, *, source, **kwargs):
        self.hub_calls.append((source_path, entry, source, kwargs))
        if entry.startswith("dinov3_"):
            return self.official[entry](**kwargs)
        self.dinov2_pretrained = kwargs.get("pretrained")
        return DummyViT(dinov2_fixture=True)

    def torch_load(self, checkpoint, **kwargs):
        self.load_calls.append((Path(checkpoint), kwargs))
        return self.payload

    def build(self, path, encoder="dinov3", size="large"):
        ns = {"Path": Path, "time": time, "importlib": importlib,
              "_sha256": lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
              "_git_info": lambda p: {"fixture": True}}
        exec(compile(function_tree(path, {"_build_encoder"}), str(path), "exec"), ns)
        config = {"checkpoint": str(self.checkpoint), encoder + "_source": str(self.source)}
        return ns["_build_encoder"](config, {"encoder": encoder, "model_size": size,
                                             "device": "cpu"}, self.torch)


class LoaderContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(self.temp.name)

    def test_official_pin_bytes_are_fixed(self):
        self.assertEqual(hashlib.sha256(OFFICIAL.read_bytes()).hexdigest(), OFFICIAL_SHA256)

    def test_original_reproduces_reported_renamed_checkpoint_error(self):
        with self.assertRaisesRegex(ValueError, "Unexpected weights specification for the ViT-L backbone"):
            self.f.build(ORIGINAL)
        self.assertEqual(self.f.load_calls, [])

    def test_repaired_renamed_checkpoint_loads_locally_into_default_lvd_architecture(self):
        counted, meta = self.f.build(PATCHED)
        model = counted.underlying
        self.assertEqual(self.f.hub_calls[0][1:], ("dinov3_vitl16", "local", {"pretrained": False}))
        self.assertEqual(self.f.load_calls, [(self.f.checkpoint, {"map_location": "cpu", "weights_only": True})])
        self.assertEqual(model.loaded, self.f.payload)
        self.assertTrue(model.strict)
        self.assertTrue(model.initialized)
        self.assertEqual(model.architecture["embed_dim"], 1024)
        self.assertEqual(model.architecture["depth"], 24)
        self.assertEqual(model.architecture["n_storage_tokens"], 4)
        self.assertFalse(model.architecture["untie_global_and_local_cls_norm"])
        self.assertEqual(model.lifecycle, ["load", "eval", "freeze", "device"])
        self.assertFalse(model.grad)
        self.assertEqual(self.f.network_calls, [])
        self.assertEqual(meta["checkpoint_sha256"], hashlib.sha256(self.f.checkpoint.read_bytes()).hexdigest())

    def test_state_dict_wrapper_is_loaded_strictly(self):
        self.f.payload = {"state_dict": {"fixture_weight": 77}}
        counted, _ = self.f.build(PATCHED)
        self.assertEqual(counted.underlying.loaded, {"fixture_weight": 77})
        self.assertTrue(counted.underlying.strict)

    def test_incompatible_state_is_not_silently_accepted(self):
        self.f.payload = {"wrong_key": 99}
        with self.assertRaisesRegex(RuntimeError, "fixture state keys mismatch"):
            self.f.build(PATCHED)
        self.assertEqual(self.f.network_calls, [])

    def test_missing_checkpoint_fails_before_hub_or_network(self):
        self.f.checkpoint.unlink()
        with self.assertRaises(FileNotFoundError):
            self.f.build(PATCHED)
        self.assertEqual(self.f.hub_calls, [])
        self.assertEqual(self.f.network_calls, [])

    def test_small_and_base_keep_offline_hub_factory_selection(self):
        for size, entry in [("small", "dinov3_vits16"), ("base", "dinov3_vitb16")]:
            with self.subTest(size=size):
                counted, _ = self.f.build(PATCHED, size=size)
                self.assertEqual(self.f.hub_calls[-1][1], entry)
                self.assertTrue(counted.underlying.strict)
        self.assertEqual(self.f.network_calls, [])

    def test_dinov2_existing_offline_contract_is_preserved(self):
        counted, _ = self.f.build(PATCHED, encoder="dinov2")
        self.assertEqual(self.f.hub_calls[0][1:], ("dinov2_vitl14", "local", {"pretrained": False}))
        self.assertEqual(self.f.load_calls[0][1], {"map_location": "cpu", "weights_only": True})
        self.assertTrue(counted.underlying.strict)
        self.assertEqual(self.f.network_calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
