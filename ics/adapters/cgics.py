"""Pinned CG-ICS source adapter and explicitly labelled component ablations.

Heavy model imports happen only when the real backend is constructed. The
adapter does not download weights, start an API server, or read query masks.
See docs/CGICS_ADAPTER.md for paper/code differences and measurement scope.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import importlib
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types
from urllib.parse import urlparse

from .base import Adapter, Prediction
from ..cgics_batch2 import (BATCH2_VARIANTS, RULES, agreement_requires_search,
                            finite_probability, input_content_fingerprint,
                            make_score_audit, prediction_agreement, random_search_decision)

PIN = "8da845743f15c18ec490567fbd86dba58c8a388f"
_PREFIX = "_ce6190_cgics_8da8457"
_VARIANTS = {
    "full": ("search", "boxes"),
    "text_only": ("search", "none"),
    "visual_only": ("none", "boxes"),
    "concat": ("none", "concat"),
    "no_search": ("candidates", "boxes"),
    "single_concept_text": ("single", "none"),
    "single_concept_full": ("single", "boxes"),
    "fixed_pool_scores": ("fixed_pool", "boxes"),
    "agreement_gate": ("gate", "boxes"),
    "random_gate": ("gate", "boxes"),
    "initial_score_gate": ("gate", "boxes"),
}


def component_settings(config):
    """Validate compatible components before loading a checkpoint."""
    variant = config.get("variant", "full")
    if variant == "custom":
        concept = config.get("concept_mode")
        visual = config.get("visual_mode")
    elif variant in _VARIANTS:
        concept, visual = _VARIANTS[variant]
        if (config.get("concept_mode", concept), config.get("visual_mode", visual)) != (concept, visual):
            raise ValueError("Component overrides require variant='custom'; do not relabel them as an original variant")
    else:
        raise ValueError(f"Unknown CG-ICS variant: {variant}")
    if concept not in {"search", "candidates", "single", "none", "fixed_pool", "gate"}:
        raise ValueError("Unknown concept_mode")
    if concept in {"fixed_pool", "gate"} and variant not in BATCH2_VARIANTS:
        raise ValueError("Batch2 policies require their explicitly named variant")
    if visual not in {"boxes", "concat", "none"}:
        raise ValueError("visual_mode must be boxes, concat, or none")
    if visual == "concat" and concept != "none":
        raise ValueError("concat is a direct visual baseline, not joint concept inference")
    if visual == "none" and concept == "none":
        raise ValueError("At least one concept or visual prompt route is required")
    if type(config.get("image_size", 512)) is not int or config.get("image_size", 512) < 16:
        raise ValueError("image_size must be an integer >=16")
    for key, default in (("BON", 5), ("num_expand_per_node", 5)):
        if type(config.get(key, default)) is not int or config.get(key, default) < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key, default in (("num_loops", 3), ("max_restarts", 3)):
        if type(config.get(key, default)) is not int or config.get(key, default) < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    for key in ("alpha", "beta"):
        value = config.get(key, 1.0)
        if not isinstance(value, (int, float)) or not 0 <= value < float("inf"):
            raise ValueError(f"{key} must be finite and non-negative")
    for key, default in (("expansion_threshold", .5), ("early_stop_threshold", .8), ("confidence_threshold", .5)):
        if not 0 <= config.get(key, default) <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if config.get("seg_type", "object") not in {"object", "part"}:
        raise ValueError("seg_type must be object or part")
    if variant in BATCH2_VARIANTS:
        for name, default in (("gate_threshold", .7), ("random_search_probability", .5),
                              ("score_gate_threshold", .8)):
            finite_probability(config.get(name, default), name)
        if any(isinstance(config.get(key, 1.), bool) or config.get(key, 1.) != 1.
               for key in ("alpha", "beta")):
            raise ValueError("Batch2 policies keep original alpha=beta=1; score rules are isolated on a shared pool")
    return {"concept": concept, "visual": visual}


def _validate_inputs(inputs):
    if set(inputs) != {"query_image", "references"}:
        raise ValueError("Adapter accepts only query_image and references; GT/class metadata must stay in evaluator")
    if not isinstance(inputs["references"], list) or not inputs["references"]:
        raise ValueError("At least one reference is required")
    paths = [inputs["query_image"]]
    for ref in inputs["references"]:
        if set(ref) != {"image", "mask"}:
            raise ValueError("A reference contains exactly image and mask")
        paths.extend((ref["image"], ref["mask"]))
    if any(not isinstance(p, str) or not Path(p).is_absolute() for p in paths):
        raise ValueError("Runner must resolve all model input paths to absolute paths")


class CGICSAdapter(Adapter):
    def __init__(self, config: dict, *, _backend=None):
        self.config = dict(config)
        self.components = component_settings(self.config)
        self.backend = _backend if _backend is not None else _OfficialBackend(self.config, self.components)
        self._stages = {}
        self._started = None
        self._last_stage = None
        self._diagnostics_active = False

    def metadata(self):
        return {
            "method": "CG-ICS", "variant": self.config.get("variant", "full"),
            "components": dict(self.components), "upstream_commit": PIN,
            "execution_kind": ("project_policy_on_pinned_components" if self.config.get("variant") in BATCH2_VARIANTS
                               else "pinned_source_on_fixed_manifest"),
            "prediction_resolution": "original query; nearest resize from CG-ICS working resolution",
            "cache_policy": "within-episode image states only; cleared between episodes",
            "mllm_memory_scope": "external server excluded from this process GPU peak",
            **self.backend.metadata(),
        }

    def predict(self, inputs: dict, *, seed: int = 0) -> Prediction:
        self._diagnostics_active = False
        _validate_inputs(inputs)
        if self.config.get("variant") in BATCH2_VARIANTS and len(inputs["references"]) != 1:
            raise ValueError("Batch2 scoring/gating policies are explicitly defined for 1-shot")
        self.backend.reset(seed)
        self._diagnostics_active = True
        stages = {}
        started = time.perf_counter()
        self._stages, self._started = stages, started

        def run_stage(name, fn):
            self.backend.synchronize()
            tick = time.perf_counter()
            self._last_stage = name
            try:
                return fn()
            finally:
                self.backend.synchronize()
                stages[name] = time.perf_counter() - tick

        try:
            prepared = run_stage("input_preprocessing_s", lambda: self.backend.prepare(inputs))
            run_stage("image_encoding_s", lambda: self.backend.build_state(prepared, self.components))
            prompt = None
            alternative_masks = {}
            if self.config.get("variant") in BATCH2_VARIANTS:
                mask, prompt, alternative_masks = self._predict_batch2(prepared, inputs, seed, run_stage, stages)
            else:
                if self.components["concept"] != "none":
                    prompt = run_stage("concept_reasoning_s", lambda: self.backend.select(prepared, self.components["concept"]))
                visual_masks = None
                if self.components["visual"] != "none":
                    visual_masks = run_stage("visual_route_s", lambda: self.backend.visual(prepared))
                if self.components["visual"] == "concat":
                    mask = run_stage("output_restore_s", lambda: self.backend.direct_mask(prepared, visual_masks))
                else:
                    mask = run_stage("final_sam_and_restore_s", lambda: self.backend.final(prepared, prompt, visual_masks))
            telemetry = copy.deepcopy(self.backend.telemetry)
            if telemetry.get("mllm_usage_missing_requests", 0):
                for key in ("mllm_prompt_tokens", "mllm_completion_tokens"):
                    telemetry[key + "_reported_partial"] = telemetry[key]
                    telemetry[key] = None
            telemetry["stages"] = stages
            telemetry["adapter_wall_s"] = time.perf_counter() - started
            artifacts = copy.deepcopy(self.backend.artifacts)
            artifacts["selected_concept"] = prompt
            artifacts["components"] = dict(self.components)
            return Prediction(mask=mask, telemetry=telemetry, artifacts=artifacts,
                              alternative_masks=alternative_masks)
        finally:
            self.backend.clear()

    def _predict_batch2(self, prepared, inputs, seed, run_stage, stages):
        variant = self.config["variant"]
        if variant == "fixed_pool_scores":
            audit = run_stage("fixed_pool_generation_and_scoring_s", lambda: self.backend.fixed_pool(prepared))
            self.backend.artifacts["batch2_score_audit"] = audit
            visual = run_stage("visual_route_s", lambda: self.backend.visual(prepared))
            # Rotate evaluation order independently of model RNG to avoid always
            # measuring the combined rule first. No selection depends on GT.
            shift = int(audit["pool_fingerprint"][:8], 16) % len(RULES)
            order = RULES[shift:] + RULES[:shift]
            audit["final_rule_order"] = list(order)
            predictions = {}
            for rule in order:
                selected = audit["rules"][rule]
                before = dict(self.backend.telemetry)
                stage = f"final_rule_{rule}_s"
                predictions[rule] = run_stage(stage, lambda selected=selected: self.backend.final(
                    prepared, selected["selected_concept"], visual))
                selected["final_inference_seconds"] = stages[stage]
                for source, destination in (("sam_calls", "final_sam_calls"),
                                            ("sam_prompt_head_calls", "final_prompt_head_calls")):
                    selected[destination] = self.backend.telemetry.get(source, 0) - before.get(source, 0)
            return predictions["combined"], audit["rules"]["combined"]["selected_concept"], {
                rule: predictions[rule] for rule in ("ref_only", "query_only")}

        initial = run_stage("initial_concept_s", lambda: self.backend.select(prepared, "single"))
        visual = run_stage("visual_route_s", lambda: self.backend.visual(prepared))
        gate = {"schema_version": 1, "policy": variant, "initial_concept": initial,
                "query_gt_used": False, "threshold_predeclared": True,
                "fallback": "original full tree search with episode-seed sampling stream; all work counted"}
        self.backend.artifacts["batch2_gate"] = gate
        probes = {}
        if variant == "initial_score_gate":
            score = run_stage("initial_original_score_s", lambda: self.backend.initial_score(prepared, initial))
            threshold = self.config.get("score_gate_threshold", .8)
            gate.update(initial_score=score, threshold=threshold,
                        signal="original ref_iou * query_presence; no mask-agreement cue")
            needs_search = score["combined_score"] < threshold
        else:
            text_mask = run_stage("gate_text_prediction_s", lambda: self.backend.final(prepared, initial, None))
            visual_mask = run_stage("gate_visual_prediction_s", lambda: self.backend.final(prepared, None, visual))
            agreement = run_stage("gate_agreement_s", lambda: prediction_agreement(text_mask, visual_mask))
            gate.update(agreement=agreement, agreement_iou=agreement["iou"],
                        text_nonempty=bool(agreement["text_area"]),
                        visual_nonempty=bool(agreement["visual_area"]),
                        signal="text-only versus visual-only prediction IoU")
            if variant == "agreement_gate":
                threshold = self.config.get("gate_threshold", .7)
                needs_search = agreement_requires_search(agreement, threshold)
                gate["threshold"] = threshold
            else:
                identity = run_stage("gate_input_hash_s", lambda: input_content_fingerprint(inputs))
                decision = random_search_decision(identity, seed, self.config.get("random_search_probability", .5))
                gate["random_decision"] = decision
                gate.update(random_probability=decision["probability"], random_draw=decision["uniform_draw"])
                gate["signal"] = "independent hash coin; agreement measured but unused"
                needs_search = decision["requires_search"]
            # Persist these exact predictions for gate auditing and qualitative
            # cases; evaluator-only GT scores cannot affect the decision above.
            probes = {"gate_text": text_mask, "gate_visual": visual_mask}
        gate["search_triggered"] = bool(needs_search)
        gate["triggered_search"] = bool(needs_search)
        prompt = initial
        if needs_search:
            run_stage("fallback_setup_s", lambda: self.backend.begin_search(prepared))
            prompt = run_stage("fallback_search_s", lambda: self.backend.select(prepared, "search"))
        gate["final_concept"] = prompt
        mask = run_stage("final_sam_and_restore_s", lambda: self.backend.final(prepared, prompt, visual))
        return mask, prompt, probes

    def close(self):
        self.backend.close()

    def failure_diagnostics(self):
        """Keep actual partial work on failed episodes; no images/credentials."""
        if not self._diagnostics_active:
            return {"telemetry": {}, "artifacts": {}, "last_stage": "input_validation_or_reset"}
        telemetry = copy.deepcopy(self.backend.telemetry)
        if telemetry.get("mllm_usage_missing_requests", 0):
            for key in ("mllm_prompt_tokens", "mllm_completion_tokens"):
                telemetry[key + "_reported_partial"] = telemetry.get(key)
                telemetry[key] = None
        telemetry["stages"] = dict(self._stages)
        telemetry["adapter_wall_s"] = time.perf_counter() - self._started if self._started is not None else None
        return {"telemetry": telemetry, "artifacts": copy.deepcopy(self.backend.artifacts), "last_stage": self._last_stage}


def _normalize_bbox(bbox, img_w, img_h):
    """Exact tensor branch of upstream normalize_bbox; avoids plotting import side effects."""
    normalized = bbox.clone()
    normalized[..., 0] /= img_w
    normalized[..., 1] /= img_h
    normalized[..., 2] /= img_w
    normalized[..., 3] /= img_h
    return normalized


def _load_upstream(root):
    root = Path(root).resolve()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != PIN:
        raise ValueError(f"CG-ICS source pin mismatch: expected {PIN}, got {actual}")
    if subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", "model", "utils", "sam3/sam3"], check=False).returncode:
        raise ValueError("CG-ICS tracked model source is modified; use the audited pinned checkout")
    if _PREFIX in sys.modules:
        if Path(sys.modules[_PREFIX].__file__).parent != root:
            raise RuntimeError("A different CG-ICS source directory is already loaded; use separate worker processes")
        required = ("model.sam3", "model.mllm", "model.tree_search", "utils.connected")
        if all(f"{_PREFIX}.{n}" in sys.modules for n in required):
            return {n: sys.modules[f"{_PREFIX}.{n}"] for n in required}
        # A failed import must not leave a partially initialized package reusable.
        for name in list(sys.modules):
            if name == _PREFIX or name.startswith(_PREFIX + "."):
                del sys.modules[name]
    sam_root = root / "sam3"
    loaded = sys.modules.get("sam3")
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(sam_root):
        raise RuntimeError("Another SAM3 package is already imported; use the CG-ICS environment in a fresh process")
    sys.path.insert(0, str(sam_root))
    importlib.import_module("sam3")
    if not Path(sys.modules["sam3"].__file__).resolve().is_relative_to(sam_root):
        raise RuntimeError("SAM3 import did not resolve to the pinned CG-ICS vendored source")
    for name, rel in ((_PREFIX, ""), (f"{_PREFIX}.utils", "utils"), (f"{_PREFIX}.model", "model")):
        module = types.ModuleType(name)
        module.__path__ = [str(root / rel)]
        module.__file__ = str(root / rel / "__init__.py")
        sys.modules[name] = module
    for name in ("utils.connected", "utils.new_viz", "utils.metrics", "utils.ref_image_edit", "model.sam3", "model.mllm", "model.tree_search"):
        path = root / (name.replace(".", "/") + ".py")
        source = path.read_text()
        source = source.replace("from utils.", f"from {_PREFIX}.utils.")
        source = source.replace("from sam3.sam3.", "from sam3.")
        source = source.replace("from sam3.visualization_utils import normalize_bbox", "from ics.adapters.cgics import _normalize_bbox as normalize_bbox")
        fullname = f"{_PREFIX}.{name}"
        module = types.ModuleType(fullname)
        module.__file__ = str(path)
        module.__package__ = fullname.rsplit(".", 1)[0]
        sys.modules[fullname] = module
        exec(compile(source, str(path), "exec"), module.__dict__)
    return {n: sys.modules[f"{_PREFIX}.{n}"] for n in ("model.sam3", "model.mllm", "model.tree_search", "utils.connected")}


class _OfficialBackend:
    def __init__(self, config, components):
        import numpy as np
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF

        self.config, self.components = config, components
        self.torch, self.np, self.Image, self.TF = torch, np, Image, TF
        self.device = str(config.get("device", "cuda"))
        self.image_size = config.get("image_size", 512)
        checkpoint = Path(config.get("sam3_checkpoint", config.get("checkpoint", ""))).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("Set sam3_checkpoint to an existing locally prepared sam3.pt; adapter never downloads it")
        self.checkpoint = checkpoint
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.checkpoint_sha256 = digest.hexdigest()
        modules = _load_upstream(config["upstream_dir"])
        sam_module = modules["model.sam3"]
        self.tree_module = modules["model.tree_search"]
        self.split = modules["utils.connected"].split_connected_components
        self.mllm_module = modules["model.mllm"]
        # Only initialization plumbing differs: explicit device and no implicit downloads.
        backend = self

        class ConfiguredSegmenter(sam_module.Sam3Segmenter):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.device = backend.device
                self.model = sam_module.build_sam3_image_model(
                    checkpoint_path=str(checkpoint), device=backend.device,
                    load_from_HF=False, eval_mode=True,
                ).to(backend.device)
                self.model.eval()
                self.processor = sam_module.Sam3Processor(
                    self.model, device=backend.device,
                    confidence_threshold=config.get("confidence_threshold", .5),
                )
                self.ref_states = self.query_state = self.cat_states = None

        self.segmenter = ConfiguredSegmenter()
        self.segmenter.eval()
        self.telemetry, self.artifacts = {}, {}
        self._instrument(self.segmenter, "infer", "sam_calls")
        self._instrument(self.segmenter.model.backbone, "forward_image", "sam_image_encoder_calls")
        self._instrument(self.segmenter.model, "forward_grounding", "sam_prompt_head_calls")
        self.mllm = None
        self.searcher = None
        if components["concept"] != "none":
            self.mllm = self._make_mllm()
            args = types.SimpleNamespace(**{
                key: config.get(key, value) for key, value in {
                    "BON": 5, "num_loops": 3, "num_expand_per_node": 5,
                    "alpha": 1., "beta": 1., "expansion_threshold": .5,
                    "max_restarts": 3, "early_stop_threshold": .8,
                }.items()
            })
            if components["concept"] == "candidates":
                args.num_loops = 0
                args.max_restarts = 0
            self.searcher = self.tree_module.TreeSearcher(self.segmenter, self.mllm, args, verbose=False)
            original_score = self.searcher._score_single_ref_raw

            @functools.wraps(original_score)
            def score(*args, **kwargs):
                rows = original_score(*args, **kwargs)
                self.telemetry["candidate_score_evaluations"] += len(rows)
                self.artifacts["score_events"].extend(copy.deepcopy(rows))
                return rows

            self.searcher._score_single_ref_raw = score

    def _instrument(self, obj, method, counter):
        original = getattr(obj, method)

        @functools.wraps(original)
        def counted(*args, **kwargs):
            self.telemetry[counter] += 1
            return original(*args, **kwargs)

        setattr(obj, method, counted)

    def _make_mllm(self):
        config = self.config
        endpoint = config.get("endpoint", "http://127.0.0.1:22002/v1")
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query:
            raise ValueError("endpoint must be an HTTP(S) base URL without embedded credentials/query")
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"} and not config.get("allow_remote_endpoint", False):
            raise ValueError("Non-local MLLM endpoint requires explicit allow_remote_endpoint=true; default uses your local server")
        key_env = config.get("api_key_env")
        api_key = os.environ[key_env] if key_env else "EMPTY"
        outer = self

        class MeasuredMLLM(self.mllm_module.MLLM):
            def _call_api_once(self, messages, max_tokens=128):
                call_index = outer.telemetry["mllm_requests"]
                sampling_index = call_index - getattr(outer, "_sampling_offset", 0)
                outer.telemetry["mllm_requests"] += 1
                kwargs = {"model": self.api_model, "messages": messages, "max_tokens": max_tokens}
                if config.get("llm_temperature") is not None:
                    kwargs["temperature"] = config["llm_temperature"]
                requested_seed = outer.seed if config.get("llm_seed_from_episode", False) else config.get("llm_seed")
                if requested_seed is not None:
                    kwargs["seed"] = (int(requested_seed) + sampling_index) % (2**31)
                started = time.perf_counter()
                try:
                    response = self.client.chat.completions.create(**kwargs)
                    usage = getattr(response, "usage", None)
                    if usage is None:
                        outer.telemetry["mllm_usage_missing_requests"] += 1
                    else:
                        missing = False
                        for source, dest in (("prompt_tokens", "mllm_prompt_tokens"), ("completion_tokens", "mllm_completion_tokens")):
                            value = getattr(usage, source, None)
                            if value is None:
                                missing = True
                            else:
                                outer.telemetry[dest] += int(value)
                        outer.telemetry["mllm_usage_missing_requests"] += int(missing)
                    text = response.choices[0].message.content
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("MLLM returned no textual candidate response")
                    outer.artifacts["mllm_responses"].append({"text": text.strip(), "request_seed": kwargs.get("seed")})
                    return text.strip()
                finally:
                    outer.telemetry["mllm_request_wall_s"] += time.perf_counter() - started

            def generate(self, *args, **kwargs):
                output = super().generate(*args, **kwargs)
                outer.artifacts["generation_events"].append({
                    "mode": kwargs.get("mode", args[0] if args else None),
                    "parent": kwargs.get("base_prompt"), "candidates": list(output),
                })
                return output

        mllm = MeasuredMLLM(api_key=api_key, base_url=endpoint,
                            model_name=config.get("mllm_model", "qwen3-vl-4b"),
                            seg_type=config.get("seg_type", "object"))
        # Disable invisible SDK retries: each recorded API call is one attempted HTTP request.
        mllm.client = mllm.client.with_options(max_retries=0, timeout=config.get("request_timeout_s", 300.0))
        return mllm

    def metadata(self):
        return {
            "sam3_checkpoint": str(self.checkpoint),
            "sam3_checkpoint_sha256": self.checkpoint_sha256,
            "mllm_model": self.config.get("mllm_model", "qwen3-vl-4b") if self.mllm else None,
            "mllm_requested_revision": self.config.get("mllm_revision", self.config.get("mllm_model_revision")) if self.mllm else None,
            "mllm_revision_verification": "server launch provenance required; OpenAI model name alone cannot verify checkpoint revision" if self.mllm else None,
            "mllm_endpoint_host": urlparse(self.config.get("endpoint", "http://127.0.0.1:22002/v1")).hostname if self.mllm else None,
            "image_size": self.image_size, "sam3_internal_resolution": 1008,
            "device": self.device,
            "llm_seed_from_episode": self.config.get("llm_seed_from_episode", False),
            "llm_seed": self.config.get("llm_seed"),
            "llm_temperature": self.config.get("llm_temperature"),
            "source_adaptations": ["isolated utils/model import namespaces", "sam3.sam3 imports normalized to vendored sam3", "normalize_bbox tensor arithmetic extracted without plotting side effects", "explicit device and disabled implicit downloads", "original-size nearest output restoration"],
            "counter_scope": "segmenter.infer, model.backbone.forward_image, model.forward_grounding, OpenAI completion calls; attempted calls",
        }

    def reset(self, seed):
        self.seed = int(seed)
        self._sampling_offset = 0
        random.seed(seed)
        self.np.random.seed(seed % (2**32))
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        self.telemetry = {k: 0 for k in (
            "sam_calls", "sam_image_encoder_calls", "sam_prompt_head_calls",
            "mllm_requests", "mllm_prompt_tokens", "mllm_completion_tokens",
            "mllm_usage_missing_requests", "candidate_score_evaluations")}
        self.telemetry["mllm_request_wall_s"] = 0.
        self.telemetry["cache_policy"] = "within_episode_only"
        self.telemetry["mllm_server_gpu_memory_bytes"] = None
        self.artifacts = {"generation_events": [], "mllm_responses": [], "score_events": []}

    def synchronize(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def prepare(self, inputs):
        from PIL import Image
        size = (self.image_size, self.image_size)
        with Image.open(inputs["query_image"]) as image:
            original_size = image.size
            query = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        refs, masks = [], []
        for ref in inputs["references"]:
            with Image.open(ref["image"]) as image:
                ref_size = image.size
                refs.append(image.convert("RGB").resize(size, Image.Resampling.BILINEAR))
            with Image.open(ref["mask"]) as image:
                if image.size != ref_size:
                    raise ValueError("Reference mask and image sizes differ")
                raw = self.np.asarray(image)
                if raw.ndim != 2 or not set(self.np.unique(raw)).issubset({0, 1, 255}):
                    raise ValueError("Reference masks must be mapped binary images (0/1 or 0/255), not class label maps")
                binary = Image.fromarray((raw > 0).astype(self.np.uint8))
                # Match upstream F.interpolate(mode='nearest'), not PIL's different nearest coordinates.
                tensor = self.torch.from_numpy(self.np.array(binary, dtype=self.np.float32)).unsqueeze(0).unsqueeze(0)
                mask = self.torch.nn.functional.interpolate(tensor, size=size, mode="nearest")[0, 0]
                if not bool(mask.any()):
                    raise ValueError("Reference mask is empty at the CG-ICS working resolution")
                masks.append(mask)
        ref_tensors = [self.TF.to_tensor(image) for image in refs]
        query_tensor = self.TF.to_tensor(query)
        cats = [self.TF.to_pil_image(self.torch.cat([ref, query_tensor], dim=2)) for ref in ref_tensors]
        return {"refs": refs, "ref_tensors": ref_tensors, "masks": masks,
                "query": query, "query_tensor": query_tensor, "cats": cats,
                "original_size": original_size}

    def build_state(self, prepared, components):
        with self.torch.inference_mode():
            # Full reproduces build_state; ablations omit encodings for removed branches.
            self.segmenter.ref_states = [self.segmenter.processor.set_image(ref) for ref in prepared["refs"]] if components["concept"] in {"search", "candidates", "fixed_pool"} else []
            self.segmenter.query_state = self.segmenter.processor.set_image(prepared["query"]) if components["visual"] != "concat" else None
            self.segmenter.cat_states = [self.segmenter.processor.set_image(cat) for cat in prepared["cats"]] if components["visual"] != "none" else None

    def ensure_reference_states(self, prepared):
        with self.torch.inference_mode():
            if not self.segmenter.ref_states:
                self.segmenter.ref_states = [self.segmenter.processor.set_image(ref) for ref in prepared["refs"]]

    def begin_search(self, prepared):
        # Full-search sampling starts from the same episode seed as the original
        # full baseline, without discarding the initial request/cost diagnostics.
        self.ensure_reference_states(prepared)
        self._sampling_offset = self.telemetry["mllm_requests"]
        random.seed(self.seed)
        self.np.random.seed(self.seed % (2**32))
        self.torch.manual_seed(self.seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(self.seed)
        self.artifacts["fallback_sampling_stream"] = {
            "episode_seed": self.seed, "first_actual_request_index": self._sampling_offset,
            "counter_reset": False,
            "note": "same requested sampling seeds as full baseline; does not guarantee deterministic server execution"}

    def initial_score(self, prepared, prompt):
        self.ensure_reference_states(prepared)
        with self.torch.inference_mode():
            row = self.searcher._score_single_ref_raw(0, prepared["masks"][0], [prompt], 1., 1.)[0]
            # Fail closed on NaN/out-of-range scores rather than accidentally
            # accepting a concept because NaN < threshold is False.
            make_score_audit([row], PIN, requested_candidates=1, generated_candidates=1)
            return row

    def fixed_pool(self, prepared):
        if len(prepared["refs"]) != 1:
            raise ValueError("Fixed-pool score study is defined for 1-shot only")
        with self.torch.inference_mode():
            ref_mask = self.split(prepared["masks"][0])
            highlighted, _ = self.tree_module.make_ref_image_for_llm(
                prepared["ref_tensors"][0], ref_mask, crop=True, alpha=.3)
            generated = self.mllm.generate(mode="initial", ref_image=prepared["refs"][0],
                                           ref_image_masked=highlighted,
                                           num_candidates=self.config.get("BON", 5))
            if not generated or any(not isinstance(x, str) or not x.strip() for x in generated):
                raise ValueError("MLLM returned an invalid fixed candidate pool")
            candidates = list(dict.fromkeys(generated))
            rows = self.searcher._score_single_ref_raw(0, prepared["masks"][0], candidates, 1., 1.)
            if [r["prompt"] for r in rows] != candidates:
                raise ValueError("Pinned scorer changed fixed candidate ordering")
            return make_score_audit(rows, PIN, requested_candidates=self.config.get("BON", 5),
                                    generated_candidates=len(generated))

    def select(self, prepared, mode):
        with self.torch.inference_mode():
            if mode == "single":
                if len(prepared["refs"]) != 1:
                    raise ValueError("single concept ablation is defined for 1-shot only; multi-reference selection needs a separate protocol")
                ref_mask = self.split(prepared["masks"][0])
                highlighted, _ = self.tree_module.make_ref_image_for_llm(prepared["ref_tensors"][0], ref_mask, crop=True, alpha=.3)
                return self.mllm.generate(mode="initial", ref_image=prepared["refs"][0], ref_image_masked=highlighted, num_candidates=1)[0]
            prompt = self.searcher.select(prepared["ref_tensors"], prepared["masks"], prepared["query_tensor"])
            result = self.searcher.last_result
            self.telemetry.update({
                "search_nodes": int(result.num_nodes), "search_restarts": int(result.num_restarts),
                "search_max_depth_reached": int(result.max_depth_reached),
                "search_reported_evaluations_excluding_rerank": int(result.total_evaluations),
            })
            self.artifacts["search_best_score"] = float(result.best_score)
            self.artifacts["search_nodes"] = [{
                "prompt": node.prompt, "score": float(node.score), "ref_iou": float(node.ref_iou),
                "query_presence": float(node.query_score), "depth": node.depth, "mode": node.mode,
                "parent": node.parent.prompt if node.parent is not None else None,
            } for node in result.all_nodes]
            return prompt

    def visual(self, prepared):
        query_masks = []
        with self.torch.inference_mode():
            for idx, ref_mask in enumerate(prepared["masks"]):
                cat_mask = self.torch.cat([ref_mask, self.torch.zeros_like(ref_mask)], dim=1)
                res = self.segmenter.infer(self.segmenter.cat_states[idx], mask=self.split(cat_mask))
                instances = res["instances"]
                query_masks.append(instances[:, :, :, instances.shape[-1] // 2:].squeeze(1).to(self.device))
            combined = self.torch.cat(query_masks, dim=0)
            self.artifacts["visual_query_instance_count_including_empty"] = int(combined.shape[0])
            self.artifacts["visual_nonempty_query_instance_count"] = int(combined.flatten(1).any(1).sum().item()) if combined.shape[0] else 0
            return combined

    def _restore(self, mask, prepared):
        width, height = prepared["original_size"]
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D semantic mask, got {tuple(mask.shape)}")
        output = self.torch.nn.functional.interpolate(mask.float()[None, None], size=(height, width), mode="nearest")[0, 0]
        return (output > .5).cpu().numpy()

    def direct_mask(self, prepared, visual_masks):
        # This unified concatenation baseline uses the union of query-side instances.
        # It is explicitly not claimed to reproduce INSID3 Table 9's unspecified mask-prompt implementation.
        mask = visual_masks.any(dim=0) if visual_masks.shape[0] else self.torch.zeros((self.image_size, self.image_size), device=self.device)
        return self._restore(mask, prepared)

    def final(self, prepared, prompt, visual_masks):
        with self.torch.inference_mode():
            res = self.segmenter.infer(self.segmenter.query_state, prompt=prompt, mask=visual_masks)
            return self._restore(res["semantic_mask"], prepared)

    def clear(self):
        self.segmenter.clear()

    def close(self):
        self.clear()
        if self.mllm is not None:
            self.mllm.client.close()


def create_adapter(config: dict) -> CGICSAdapter:
    return CGICSAdapter(config)
