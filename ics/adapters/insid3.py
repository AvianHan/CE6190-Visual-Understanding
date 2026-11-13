"""Pinned INSID3 plus explicitly labelled, composable feature/processing variants.

Heavy dependencies are imported only when an adapter is constructed. No network
downloads occur here. Query ground truth is never accepted by this interface.
"""
from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path
import subprocess
import sys
import time

from .base import Adapter, Prediction


UPSTREAM_COMMIT = "0c165a10cf52ab91f335883d06260de86854adbe"
SOURCE_SHA256 = {
    "models/insid3.py": "bab5cda135d80532581337c853b4a537b8692aa0dc15ad37f1673f98161652fc",
    "utils/clustering.py": "3742712bd8edc13d09e0fd135a902526c82c26cc5cec373391154e145e311fe8",
    "utils/data.py": "f6482daf4598f05781c02b4b06d026d11f93bd70fbfcca727cef3e033e83e495",
    "utils/refinement.py": "038c68fe04ea5cdc127da50b926e3dd2ef16ffd400873eea356fbffc66fc188c",
    "utils/keypoints.py": "a0307b9591fdcf99548dc938d9cf9ee1f7c7df76f4816750341e840bdc9aeb96",
}

VARIANTS = {
    "full": {},
    "no_debias": {"use_debias": False},
    "no_aggregation": {"aggregation": "seed_only"},
    "cross_only": {"aggregation": "cross_only"},
    "simple_similarity": {"prediction_mode": "similarity"},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_info(path: Path) -> dict:
    def git(*args):
        proc = subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                              text=True, timeout=15)
        return proc.stdout.strip() if proc.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"),
            "tracked_changes": git("diff", "--name-only", "HEAD")}


def _verify_source(path: Path) -> dict:
    observed = {name: _sha256(path / name) for name in SOURCE_SHA256}
    mismatches = [name for name in SOURCE_SHA256 if observed[name] != SOURCE_SHA256[name]]
    if mismatches:
        raise RuntimeError("INSID3 source differs from audited pin: " + ", ".join(mismatches))
    info = _git_info(path)
    if info["commit"] is not None and info["commit"] != UPSTREAM_COMMIT:
        raise RuntimeError(f"Expected INSID3 commit {UPSTREAM_COMMIT}, got {info['commit']}")
    return {"repository": "https://github.com/visinf/INSID3", "audited_commit": UPSTREAM_COMMIT,
            "git": info, "source_sha256": observed}


def _effective_settings(config: dict) -> dict:
    variant = config.get("variant", "full")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown INSID3 variant {variant!r}; choose {list(VARIANTS)}")
    encoder = config.get("encoder", "dinov3")
    if encoder not in {"dinov3", "dinov2", "sam3"}:
        raise ValueError("encoder must be dinov3, dinov2 or sam3")
    settings = {
        "encoder": encoder, "model_size": "large", "device": "cuda",
        "image_size": 1024 if encoder == "dinov3" else 1008,
        "svd_components": 500, "tau": 0.6, "merge_threshold": 0.2,
        "similarity_threshold": 0.55, "mask_refiner": "bilinear", "crf_size": 640,
        "use_debias": True, "aggregation": "self_cross", "prediction_mode": "clusters",
        "seed": 0,
    }
    settings.update(VARIANTS[variant])
    for key in settings:
        if key in config:
            settings[key] = config[key]
    if settings["model_size"] not in {"small", "base", "large"}:
        raise ValueError("model_size must be small, base or large")
    if encoder == "sam3" and settings["model_size"] != "large":
        raise ValueError("SAM3 trunk is the fixed official model; model_size must be large")
    patch = 16 if encoder == "dinov3" else 14
    size = settings["image_size"]
    if type(size) is not int or size < patch or size % patch:
        raise ValueError(f"image_size must be a positive multiple of encoder patch size {patch}")
    if type(settings["svd_components"]) is not int or settings["svd_components"] < 0:
        raise ValueError("svd_components must be a nonnegative integer")
    if type(settings["use_debias"]) is not bool:
        raise ValueError("use_debias must be bool")
    if settings["aggregation"] not in {"self_cross", "cross_only", "seed_only"}:
        raise ValueError("Unknown aggregation")
    if settings["prediction_mode"] not in {"clusters", "similarity"}:
        raise ValueError("Unknown prediction_mode")
    if settings["mask_refiner"] not in {"bilinear", "crf"}:
        raise ValueError("mask_refiner must be bilinear or crf")
    for name in ("tau", "merge_threshold", "similarity_threshold"):
        value = settings[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 < settings["tau"] < 1:
        raise ValueError("tau must be between 0 and 1")
    for name in ("merge_threshold", "similarity_threshold"):
        if not -1 <= settings[name] <= 1:
            raise ValueError(f"{name} must be between -1 and 1")
    settings["variant"] = variant
    return settings


def _add_source(path: Path, namespace: str) -> None:
    """Reject cross-repository module collisions rather than silently import them."""
    if not path.is_dir():
        raise FileNotFoundError(f"Local source directory is missing: {path}")
    module = sys.modules.get(namespace)
    if module is not None:
        filename = getattr(module, "__file__", None)
        roots = list(getattr(module, "__path__", []))
        locations = ([filename] if filename else []) + roots
        if not locations or not all(Path(p).resolve().is_relative_to(path) for p in locations):
            raise RuntimeError(f"Module namespace {namespace!r} already belongs to another source; "
                               "run each model configuration in its own process")
    sys.path.insert(0, str(path))


def _read_inputs(inputs: dict):
    """Validate the whitelist and masks before any model inference."""
    import numpy as np
    from PIL import Image
    if set(inputs) != {"query_image", "references"}:
        raise ValueError("Adapter accepts only query_image and references; query GT/class data forbidden")
    refs = inputs["references"]
    if not isinstance(refs, list) or not refs:
        raise ValueError("At least one reference is required")
    query_path = Path(inputs["query_image"]).resolve()
    with Image.open(query_path) as image:
        query = image.convert("RGB")
    decoded = []
    for ref in refs:
        if set(ref) != {"image", "mask"}:
            raise ValueError("Each reference accepts only image and mask")
        if Path(ref["image"]).resolve() == query_path:
            raise ValueError("Reference and query images must differ")
        with Image.open(ref["image"]) as image:
            rgb = image.convert("RGB")
        with Image.open(ref["mask"]) as image:
            array = np.asarray(image)
        if array.ndim != 2 or array.shape != (rgb.height, rgb.width):
            raise ValueError("Reference mask must be single-channel at its image's original resolution")
        if not np.isin(array, [0, 1, 255]).all():
            raise ValueError("Reference mask must already be mapped to binary values 0/1 or 0/255")
        if not (array > 0).any():
            raise ValueError("Empty reference mask cannot define a foreground concept")
        decoded.append((rgb, Image.fromarray((array > 0).astype("uint8") * 255)))
    return query, decoded


def _build_model_class(upstream_class, torch, settings):
    """Subclass only experiment switches; the full route calls pinned upstream code."""
    F = torch.nn.functional
    downsample_mask = importlib.import_module("utils.data").downsample_mask

    class ExperimentalINSID3(upstream_class):
        def _build_positional_basis(self, device):
            if not settings["use_debias"] or self.svd_components == 0:
                return None
            return super()._build_positional_basis(device)

        def _debias_features(self, fmaps_norm):
            if self.positional_basis is None:
                return fmaps_norm
            return super()._debias_features(fmaps_norm)

        @torch.no_grad()
        def predict_mask(self, ref_images, ref_masks, tgt_image):
            self.experiment_trace = {}
            if settings["prediction_mode"] == "clusters":
                return super().predict_mask(ref_images, ref_masks, tgt_image)
            shots = ref_images.shape[0]
            images = torch.cat([ref_images, tgt_image], dim=0).unsqueeze(0)
            features = F.normalize(self._extract_features(images), p=2, dim=2)
            features = self._debias_features(features)
            _, _, _, height, width = features.shape
            prototypes = []
            for shot in range(shots):
                mask = downsample_mask(ref_masks[shot:shot + 1].unsqueeze(1), height, width)
                foreground = features[0, shot, :, mask]
                if foreground.shape[1]:
                    prototypes.append(foreground.mean(dim=1))
            if not prototypes:
                raise ValueError("No reference foreground survived feature-grid mapping")
            prototype = F.normalize(torch.stack(prototypes).mean(dim=0), p=2, dim=0)
            similarity = torch.einsum("chw,c->hw", features[0, shots], prototype)
            prediction = similarity > settings["similarity_threshold"]
            self.experiment_trace = {"feature_grid": [height, width],
                                     "prediction_mode": "similarity"}
            return self._finalize_mask(prediction, tgt_image)

        def _seed_and_aggregate(self, candidate_mask, cluster_labels, cluster_protos, K,
                                ref_prototype, feat_tgt, feat_tgt_deb, h, w):
            self.experiment_trace = {"cluster_count": K, "feature_grid": [h, w],
                                     "candidate_patch_count": int(candidate_mask.sum().item())}
            if settings["aggregation"] == "self_cross":
                return super()._seed_and_aggregate(candidate_mask, cluster_labels, cluster_protos,
                                                   K, ref_prototype, feat_tgt, feat_tgt_deb, h, w)
            matched_mask = candidate_mask & (cluster_labels >= 0)
            if matched_mask.sum() == 0:
                return candidate_mask
            matched_ids, n_pixels = cluster_labels[matched_mask].unique(return_counts=True)
            similarities = (ref_prototype.T @ cluster_protos[matched_ids].T).squeeze(0)
            seed_id = matched_ids[int(similarities.argmax().item())].item()
            self.experiment_trace["seed_cluster_id"] = seed_id
            if settings["aggregation"] == "seed_only":
                return cluster_labels == seed_id
            # Same cross term and candidate-area weights as upstream; remove ONLY intra_sim.
            all_areas = cluster_labels[cluster_labels >= 0].unique(return_counts=True)[1]
            per_cluster = torch.zeros(K, device=cluster_labels.device)
            per_cluster[matched_ids] = n_pixels.float()
            weights = per_cluster / all_areas
            weights[seed_id] = 1.0
            foreground_sim = torch.einsum("bchw,cd->bhw", feat_tgt_deb, ref_prototype).squeeze(0)
            cross = torch.empty(K, device=foreground_sim.device, dtype=foreground_sim.dtype)
            for k in range(K):
                index = cluster_labels == k
                cross[k] = foreground_sim[index].mean() if index.any() else 0.0
            combined = cross * weights
            output = torch.zeros(h, w, dtype=torch.bool, device=cluster_labels.device)
            valid = cluster_labels >= 0
            output[valid] = combined[cluster_labels[valid]] > self.merge_threshold
            return output

    return ExperimentalINSID3


def _build_encoder(config, settings, torch):
    checkpoint = Path(config["checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Download the authorized model checkpoint first: {checkpoint}")
    kind = settings["encoder"]
    name = {"small": "s", "base": "b", "large": "l"}[settings["model_size"]]
    source_key = {"dinov3": "dinov3_source", "dinov2": "dinov2_source", "sam3": "sam3_source"}[kind]
    if not config.get(source_key):
        raise ValueError(f"{source_key} must identify a local pinned source checkout")
    source = Path(config[source_key]).expanduser().resolve()
    if kind in {"dinov3", "dinov2"}:
        if not (source / "hubconf.py").is_file():
            raise FileNotFoundError(f"Missing hubconf.py in {source}")
        if kind == "dinov3":
            # The pinned ViT-L hub infers LVD/SAT architecture from weights and
            # rejects renamed local checkpoint paths. Build the default LVD
            # architecture offline, then load our authorized local checkpoint.
            encoder = torch.hub.load(str(source), f"dinov3_vit{name}16", source="local",
                                     pretrained=False)
        else:
            encoder = torch.hub.load(str(source), f"dinov2_vit{name}14", source="local",
                                     pretrained=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        encoder.load_state_dict(state, strict=True)
        feature_level = "last_normalized_patch_tokens"
    else:
        _add_source(source, "sam3")
        builder = importlib.import_module("sam3.model_builder")
        # Construct ONLY the image trunk. No language encoder, neck, or decoder is instantiated.
        encoder = builder._create_vit_backbone(compile_mode=None)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        prefix = "detector.backbone.vision_backbone.trunk."
        trunk_state = {key[len(prefix):]: value for key, value in state.items()
                       if key.startswith(prefix)}
        if not trunk_state:
            raise ValueError("SAM3 checkpoint has no official detector.backbone.vision_backbone.trunk weights")
        encoder.load_state_dict(trunk_state, strict=True)
        del state, trunk_state
        feature_level = "sam3_vit_trunk_last_map_before_neck"
    encoder.eval().requires_grad_(False).to(settings["device"])

    class CountedEncoder(torch.nn.Module):
        def __init__(self, underlying):
            super().__init__()
            self.underlying = underlying
            self.reset_counters()

        def reset_counters(self):
            self.calls = 0
            self.images = 0
            self.seconds = 0.0
            self.grid = None

        def get_intermediate_layers(self, x, n=1, reshape=True):
            if n != 1 or not reshape:
                raise ValueError("This adapter exposes the last dense feature map only")
            self.calls += 1
            self.images += int(x.shape[0])
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
            started = time.perf_counter()
            if kind == "sam3":
                # Upstream INSID3 inputs use ImageNet normalization. Undo it and use SAM3's.
                mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
                std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                sam_x = ((x * std + mean) - 0.5) / 0.5
                output = [self.underlying(sam_x)[-1].float()]
            else:
                output = self.underlying.get_intermediate_layers(x, n=1, reshape=True)
                if kind == "dinov2":
                    output = [output[0].float()]
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
            self.seconds += time.perf_counter() - started
            self.grid = list(output[0].shape[-2:])
            return output

    return CountedEncoder(encoder), {
        "encoder": kind, "feature_level": feature_level,
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "encoder_source": str(source), "encoder_source_git": _git_info(source),
        "feature_dtype": "upstream" if kind == "dinov3" else "float32_after_extraction",
        "normalization": "ImageNet" if kind != "sam3" else "SAM3_mean_std_0.5",
        "pretrained_supervision": "self_supervised" if kind != "sam3" else "SAM3_supervised_pretraining",
    }


class INSID3Adapter(Adapter):
    def __init__(self, config: dict):
        self.config = dict(config)
        self.settings = _effective_settings(config)
        upstream = Path(config["upstream_dir"]).expanduser().resolve()
        self.source_metadata = _verify_source(upstream)
        _add_source(upstream, "utils")
        _add_source(upstream, "models")
        import torch
        self.torch = torch
        torch.manual_seed(self.settings["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if self.settings["device"].startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run inside an allocated GPU job")
        upstream_class = importlib.import_module("models.insid3").INSID3
        encoder, self.encoder_metadata = _build_encoder(config, self.settings, torch)
        if (self.settings["mask_refiner"] == "crf" and self.settings["encoder"] == "sam3"):
            # Upstream finalization receives ImageNet-normalized images; its CRF remains valid.
            pass
        model_class = _build_model_class(upstream_class, torch, self.settings)
        self.model = model_class(
            encoder=encoder, image_size=self.settings["image_size"],
            svd_components=self.settings["svd_components"], tau=self.settings["tau"],
            merge_threshold=self.settings["merge_threshold"],
            mask_refiner=self.settings["mask_refiner"], crf_size=self.settings["crf_size"],
            resize_to_orig_size=True, device=self.settings["device"],
        ).eval().to(self.settings["device"])
        self.model.requires_grad_(False)
        self.basis_build_encoder_calls = encoder.calls
        self.effective_svd_rank = 0 if self.model.positional_basis is None else int(self.model.positional_basis.shape[1])

    def metadata(self) -> dict:
        settings = dict(self.settings)
        settings["effective_svd_rank"] = self.effective_svd_rank
        return {
            "adapter": "insid3", "settings": settings, "upstream": self.source_metadata,
            **self.encoder_metadata,
            "output_resolution": "original_query", "protocol": "unified_original_resolution",
            "reference_cache": "none_across_episodes",
            "basis_input": "ImageNet_normalized_black_image_centered_features",
            "basis_build_encoder_calls": self.basis_build_encoder_calls,
            "source_behavior_notes": [
                "Pinned source uses a black image and centered SVD features, unlike paper's Gaussian-noise description.",
                "Pinned source candidates use forward-positive AND backward-nearest-neighbour gates.",
                "Pinned source aggregation uses candidate-area weights; seed is not unconditionally forced into final mask.",
                "Upstream evaluation resizes GT to model output; unified evaluation returns original query resolution.",
                "Non-DINOv3 encoders are experimental adaptations, not author-reported SAM3/DINOv2 exact reproductions.",
            ],
        }

    def predict(self, inputs: dict, *, seed: int = 0) -> Prediction:
        torch = self.torch
        torch.manual_seed(seed)
        encoder = self.model.encoder
        encoder.reset_counters()
        self.model.reset_state()
        start = time.perf_counter()
        query, refs = _read_inputs(inputs)
        with torch.inference_mode():
            for image, mask in refs:
                self.model.set_reference(image, mask)
            self.model.set_target(query)
            if self.settings["device"].startswith("cuda"):
                torch.cuda.synchronize(self.settings["device"])
            preprocessed = time.perf_counter()
            try:
                result = self.model.segment()
                if self.settings["device"].startswith("cuda"):
                    torch.cuda.synchronize(self.settings["device"])
                segmented = time.perf_counter()
                array = result.detach().cpu().numpy().astype(bool)
            finally:
                self.model.reset_state()
        if array.shape != (query.height, query.width):
            raise RuntimeError("Adapter produced a mask at the wrong query resolution")
        return Prediction(
            mask=array,
            telemetry={
                "preprocess_seconds": preprocessed - start,
                "segmentation_seconds": segmented - preprocessed,
                "encoder_seconds": encoder.seconds,
                "encoder_calls": encoder.calls, "encoder_images": encoder.images,
                "feature_grid": encoder.grid,
                "instrumentation": "get_intermediate_layers actual entry; includes image batch, excludes one-time basis construction",
            },
            artifacts={"processing": dict(getattr(self.model, "experiment_trace", {})),
                       "seed": seed},
        )

    def close(self):
        if hasattr(self, "model"):
            self.model.reset_state()
            del self.model


def create_adapter(config: dict) -> INSID3Adapter:
    return INSID3Adapter(config)
