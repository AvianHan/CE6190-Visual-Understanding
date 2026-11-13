# Historical function only: reproduces renamed local DINOv3 checkpoint failure.
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
            encoder = torch.hub.load(str(source), f"dinov3_vit{name}16", source="local",
                                     weights=str(checkpoint))
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
