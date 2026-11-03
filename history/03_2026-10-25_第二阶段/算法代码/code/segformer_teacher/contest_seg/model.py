from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .labels import NUM_CLASSES


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SimpleUNet(nn.Module):
    """Small dependency-free model for pipeline validation, not the final model."""

    def __init__(self, num_classes=NUM_CLASSES, base_channels=32):
        super().__init__()
        self.enc1 = ConvBlock(3, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bridge = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bridge = self.bridge(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bridge), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class SegFormerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.use_gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self.use_gradient_checkpointing = True

    def forward(self, x):
        if self.training and self.use_gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint

            logits = checkpoint(
                lambda values: self.model(pixel_values=values).logits,
                x,
                use_reentrant=False,
            )
        else:
            logits = self.model(pixel_values=x).logits
        return F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )


class DepthwiseSeparableBlock(nn.Module):
    """Small batch-size friendly detail block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.GroupNorm(1, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class EdgeRefineSegFormer(nn.Module):
    """Boundary-gated half-resolution residual refinement for SegFormer.

    The residual prediction layer is zero initialized, so loading a trained base
    SegFormer produces exactly the same logits before refinement training.
    """

    def __init__(self, base, num_classes=NUM_CLASSES, detail_channels=32):
        super().__init__()
        self.base = base
        self.refine_scale = 1.0
        self.refine_gate_threshold = None
        self.detail_encoder = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, 24),
            nn.GELU(),
            DepthwiseSeparableBlock(24, detail_channels),
            DepthwiseSeparableBlock(detail_channels, detail_channels),
        )
        fusion_channels = detail_channels + num_classes
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, 48, kernel_size=1, bias=False),
            nn.GroupNorm(1, 48),
            nn.GELU(),
            DepthwiseSeparableBlock(48, 48),
        )
        self.boundary_head = nn.Conv2d(48, 1, kernel_size=1)
        self.residual_head = nn.Conv2d(48, num_classes, kernel_size=1)
        nn.init.constant_(self.boundary_head.bias, -2.0)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def gradient_checkpointing_enable(self):
        self.base.gradient_checkpointing_enable()

    def forward_with_aux(self, x):
        coarse_logits = self.base(x)
        detail = self.detail_encoder(x)
        coarse_half = F.interpolate(
            coarse_logits,
            size=detail.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = self.fusion(torch.cat([detail, coarse_half], dim=1))
        boundary_logits = self.boundary_head(features)
        residual_half = self.residual_head(features)
        gate = torch.sigmoid(boundary_logits)
        effective_gate = gate
        if self.refine_gate_threshold is not None:
            effective_gate = gate * (gate >= self.refine_gate_threshold)
        correction = F.interpolate(
            effective_gate * residual_half * self.refine_scale,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return {
            "logits": coarse_logits + correction,
            "coarse_logits": coarse_logits,
            "boundary_logits": boundary_logits,
            "residual_half": residual_half,
            "gate": gate
        }

    def forward(self, x):
        return self.forward_with_aux(x)["logits"]


class DinoHRDASegmenter(nn.Module):
    """DINOv2 context encoder fused with a high-resolution detail branch.

    The transformer sees a fixed, relatively small global view while the CNN
    branch keeps the original input resolution.  This is deliberately a single
    model (rather than an ensemble) and keeps DINO's quadratic attention cost
    independent of the training crop size.
    """

    def __init__(
        self,
        backbone,
        num_classes=NUM_CLASSES,
        context_size=448,
        context_channels=160,
        detail_channels=96,
    ):
        super().__init__()
        self.backbone = backbone
        self.context_size = context_size
        hidden_size = backbone.config.hidden_size
        self.feature_layers = (3, 6, 9, 12)
        self.context_projection = nn.Sequential(
            nn.Conv2d(hidden_size * len(self.feature_layers), context_channels, 1, bias=False),
            nn.GroupNorm(8, context_channels),
            nn.GELU(),
            DepthwiseSeparableBlock(context_channels, context_channels),
        )
        self.detail_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            DepthwiseSeparableBlock(32, 48),
            nn.Conv2d(48, detail_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, detail_channels),
            nn.GELU(),
            DepthwiseSeparableBlock(detail_channels, detail_channels),
        )
        fused_channels = context_channels + detail_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels, 192, 1, bias=False),
            nn.GroupNorm(8, 192),
            nn.GELU(),
            DepthwiseSeparableBlock(192, 192),
            nn.Dropout2d(0.1),
            DepthwiseSeparableBlock(192, 128),
        )
        self.head = nn.Conv2d(128, num_classes, 1)

    def gradient_checkpointing_enable(self):
        self.backbone.gradient_checkpointing_enable()

    def set_backbone_trainable(self, last_blocks=0):
        """Freeze DINO, optionally unfreezing only its last N transformer blocks."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if last_blocks > 0:
            for block in self.backbone.encoder.layer[-last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.backbone.layernorm.parameters():
                parameter.requires_grad_(True)

    def _context_features(self, x):
        context_input = F.interpolate(
            x,
            size=(self.context_size, self.context_size),
            mode="bilinear",
            align_corners=False,
        )
        output = self.backbone(
            pixel_values=context_input,
            output_hidden_states=True,
            return_dict=True,
        )
        patch_height = self.context_size // self.backbone.config.patch_size
        patch_width = self.context_size // self.backbone.config.patch_size
        features = []
        for layer_index in self.feature_layers:
            tokens = output.hidden_states[layer_index][:, 1:, :]
            feature = tokens.transpose(1, 2).reshape(
                tokens.shape[0], tokens.shape[2], patch_height, patch_width
            )
            features.append(feature)
        return self.context_projection(torch.cat(features, dim=1))

    def forward(self, x):
        detail = self.detail_encoder(x)
        context = self._context_features(x)
        context = F.interpolate(
            context, size=detail.shape[-2:], mode="bilinear", align_corners=False
        )
        logits = self.head(self.fusion(torch.cat([context, detail], dim=1)))
        return F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )


class TimmDinoHRDASegmenter(nn.Module):
    """DINOv2/v3 timm backbone with a compact high-resolution detail branch."""

    def __init__(
        self,
        backbone,
        num_classes=NUM_CLASSES,
        context_size=448,
        context_channels=160,
        detail_channels=96,
    ):
        super().__init__()
        self.backbone = backbone
        self.context_size = context_size
        self.feature_layers = (2, 5, 8, 11)
        hidden_size = backbone.num_features
        self.context_projection = nn.Sequential(
            nn.Conv2d(hidden_size * len(self.feature_layers), context_channels, 1, bias=False),
            nn.GroupNorm(8, context_channels),
            nn.GELU(),
            DepthwiseSeparableBlock(context_channels, context_channels),
        )
        self.detail_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            DepthwiseSeparableBlock(32, 48),
            nn.Conv2d(48, detail_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, detail_channels),
            nn.GELU(),
            DepthwiseSeparableBlock(detail_channels, detail_channels),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(context_channels + detail_channels, 192, 1, bias=False),
            nn.GroupNorm(8, 192),
            nn.GELU(),
            DepthwiseSeparableBlock(192, 192),
            nn.Dropout2d(0.1),
            DepthwiseSeparableBlock(192, 128),
        )
        self.head = nn.Conv2d(128, num_classes, 1)

    def gradient_checkpointing_enable(self):
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

    def set_backbone_trainable(self, last_blocks=0):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if last_blocks > 0:
            for block in self.backbone.blocks[-last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for module_name in ("norm", "fc_norm"):
                module = getattr(self.backbone, module_name, None)
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)

    def forward(self, x):
        context_input = F.interpolate(
            x,
            size=(self.context_size, self.context_size),
            mode="bilinear",
            align_corners=False,
        )
        features = self.backbone.forward_intermediates(
            context_input,
            indices=self.feature_layers,
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        context = self.context_projection(torch.cat(features, dim=1))
        detail = self.detail_encoder(x)
        context = F.interpolate(
            context, size=detail.shape[-2:], mode="bilinear", align_corners=False
        )
        logits = self.head(self.fusion(torch.cat([context, detail], dim=1)))
        return F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )


def _build_timm_dino(architecture, pretrained):
    import timm

    checkpoint_path = None
    if pretrained and not isinstance(pretrained, bool):
        candidate = Path(pretrained)
        checkpoint_path = candidate / "model.safetensors" if candidate.is_dir() else candidate
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing DINO checkpoint: {checkpoint_path}")
    kwargs = {"pretrained": False, "num_classes": 0, "dynamic_img_size": True}
    backbone = timm.create_model(architecture, **kwargs)
    if checkpoint_path is not None:
        from timm.models import load_checkpoint

        load_checkpoint(backbone, str(checkpoint_path), strict=True)
    elif pretrained:
        backbone = timm.create_model(
            architecture, pretrained=True, num_classes=0, dynamic_img_size=True
        )
    return backbone


def _segformer_config(name, num_classes):
    from transformers import SegformerConfig

    variants = {
        "segformer_b2": ([3, 4, 6, 3], 768),
        "segformer_b3": ([3, 4, 18, 3], 768),
        "segformer_b4": ([3, 8, 27, 3], 768),
        "segformer_b5": ([3, 6, 40, 3], 768)
    }
    depths, decoder_hidden_size = variants[name]
    return SegformerConfig(
        num_labels=num_classes,
        depths=depths,
        hidden_sizes=[64, 128, 320, 512],
        decoder_hidden_size=decoder_hidden_size,
        id2label={index: str(index) for index in range(num_classes)},
        label2id={str(index): index for index in range(num_classes)},
    )


def build_model(name="simple_unet", num_classes=NUM_CLASSES, pretrained=False):
    if name == "simple_unet":
        return SimpleUNet(num_classes=num_classes)
    if name == "deeplabv3_resnet50":
        from torchvision.models.segmentation import deeplabv3_resnet50

        model = deeplabv3_resnet50(
            weights=None, weights_backbone=None, num_classes=num_classes
        )

        class LogitsOnly(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, x):
                output = self.wrapped(x)["out"]
                return F.interpolate(
                    output, size=x.shape[-2:], mode="bilinear", align_corners=False
                )

        return LogitsOnly(model)
    if name == "unetformer_r18":
        from third_party.GeoSeg.geoseg.models.UNetFormer import UNetFormer

        model = UNetFormer(
            decode_channels=64,
            dropout=0.1,
            backbone_name="swsl_resnet18",
            pretrained=isinstance(pretrained, bool) and pretrained,
            window_size=8,
            num_classes=num_classes,
        )
        if pretrained and not isinstance(pretrained, bool):
            from safetensors.torch import load_file

            weights = load_file(str(pretrained), device="cpu")
            incompatible = model.backbone.load_state_dict(weights, strict=False)
            if incompatible.missing_keys or set(incompatible.unexpected_keys) != {
                "fc.bias",
                "fc.weight"
            }:
                raise ValueError(
                    "Unexpected UNetFormer backbone checkpoint keys: "
                    f"missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
        return model
    if name == "dinov2_s_hrda":
        from transformers import Dinov2Config, Dinov2Model

        if pretrained and not isinstance(pretrained, bool):
            backbone = Dinov2Model.from_pretrained(
                str(pretrained), local_files_only=True
            )
        elif pretrained:
            backbone = Dinov2Model.from_pretrained("facebook/dinov2-small")
        else:
            backbone = Dinov2Model(
                Dinov2Config(
                    hidden_size=384,
                    num_hidden_layers=12,
                    num_attention_heads=6,
                    patch_size=14,
                    image_size=518,
                    mlp_ratio=4,
                    qkv_bias=True,
                )
            )
        return DinoHRDASegmenter(backbone, num_classes=num_classes)
    if name == "dinov2_b_hrda":
        backbone = _build_timm_dino(
            "vit_base_patch14_reg4_dinov2.lvd142m", pretrained
        )
        return TimmDinoHRDASegmenter(backbone, num_classes=num_classes)
    if name == "dinov2_s_timm_hrda":
        backbone = _build_timm_dino(
            "vit_small_patch14_reg4_dinov2.lvd142m", pretrained
        )
        return TimmDinoHRDASegmenter(backbone, num_classes=num_classes)
    if name == "dinov3_b_hrda":
        backbone = _build_timm_dino(
            "vit_base_patch16_dinov3.lvd1689m", pretrained
        )
        return TimmDinoHRDASegmenter(backbone, num_classes=num_classes)
    if name == "dinov3_s_hrda":
        backbone = _build_timm_dino(
            "vit_small_patch16_dinov3.lvd1689m", pretrained
        )
        return TimmDinoHRDASegmenter(backbone, num_classes=num_classes)
    edge_refine = name.endswith("_edgerefine")
    base_name = name.removesuffix("_edgerefine") if edge_refine else name
    if base_name in {"segformer_b2", "segformer_b3", "segformer_b4", "segformer_b5"}:
        from transformers import SegformerForSemanticSegmentation

        if pretrained:
            variant = base_name.rsplit("_", 1)[1]
            repository = (
                str(pretrained)
                if not isinstance(pretrained, bool)
                else f"nvidia/segformer-{variant}-finetuned-ade-512-512"
            )
            model = SegformerForSemanticSegmentation.from_pretrained(
                repository,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            model = SegformerForSemanticSegmentation(
                _segformer_config(base_name, num_classes)
            )
        wrapped = SegFormerWrapper(model)
        if edge_refine:
            return EdgeRefineSegFormer(wrapped, num_classes=num_classes)
        return wrapped
    if name == "d2ls_convnext_tiny":
        from third_party.D2LS.network.models.d2ls import DynamicDictionaryLearning

        model = DynamicDictionaryLearning(
            model="convnext_tiny",
            token_length=num_classes,
            num_classes=num_classes,
            l=3,
            embedding_dim=256,
            use_aux=True,
            has_contrastive_loss=True,
            pretrained=isinstance(pretrained, bool) and pretrained,
        )
        if pretrained and not isinstance(pretrained, bool):
            from torchvision.models import convnext_tiny

            checkpoint = torch.load(pretrained, map_location="cpu", weights_only=True)
            pretrained_model = convnext_tiny(weights=None)
            pretrained_model.load_state_dict(checkpoint)
            pretrained_backbone = nn.Sequential(
                *(list(pretrained_model.children())[:-1])
            )
            model.backbone.load_state_dict(pretrained_backbone.state_dict())
        return model
    raise ValueError(f"Unknown model: {name}")
