"""
EfficientNet-B4 Deepfake Detector
===================================
Pretrained ImageNet backbone fine-tuned for binary deepfake detection.

Architecture Overview
---------------------
  Input (380×380 RGB)
       ↓
  EfficientNet-B4 backbone (pretrained, timm)
       ↓  1792-channel spatial feature map
  Adaptive Average Pool → (B, 1792)
       ↓
  Dropout(0.4) → Linear(1792→512) → GELU → Dropout(0.3) → Linear(512→1)
       ↓
  Logit (scalar per image) ── BCEWithLogitsLoss during training
       ↓ sigmoid at inference
  Probability [0, 1]  (≥ 0.5 → FAKE)

Heatmaps
--------
  Grad-CAM on the last MBConv block of EfficientNet-B4.
  See inference.py for usage — model.get_target_layer() returns the hook target.

Requirements
------------
  pip install timm
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:
    raise ImportError(
        "timm is required.  Install with:  pip install timm"
    ) from exc


class SRMConv2d(nn.Module):
    """
    Spatial Rich Model (SRM) High-Pass Filter.
    Extracts high-frequency noise residuals to expose deepfake checkerboard artifacts.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        
        # Laplacian high-pass filter to extract noise/edges
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32) / 8.0
        
        # Shape: (out_channels, in_channels/groups, H, W)
        kernel = kernel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False
        
    def forward(self, x):
        return self.conv(x)


class EfficientNetB4Detector(nn.Module):
    """
    EfficientNet-B4 backbone + custom two-layer classification head.

    Forward signature (identical to the old Xception model for compatibility):
        logits    : (B,)    raw binary logit
        attn_map  : None    (use Grad-CAM externally for heatmaps)
        embedding : (B, 512) pre-classifier feature vector
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        drop_rate: float = 0.4,
        use_srm: bool = True,
    ):
        super().__init__()
        self.use_srm = use_srm
        in_chans = 6 if use_srm else 3

        if use_srm:
            self.srm_layer = SRMConv2d(in_channels=3)
            # ImageNet normalization constants — used to un-normalize before SRM
            self.register_buffer("_img_mean",
                                 torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("_img_std",
                                 torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ── Backbone ────────────────────────────────────────────────────────
        # global_pool='' keeps the spatial (H×W) feature map alive so that
        # Grad-CAM can hook into the last convolutional block.
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool="",
        )
        feat_dim = self.backbone.num_features  # 1792 for EfficientNet-B4

        # ── Global Average Pool ─────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Classification Head ─────────────────────────────────────────────
        # Two-stage with GELU + Dropout — strong regularisation that helps
        # generalise across the 40 manipulation methods in DF40.
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),              # head[0]
            nn.Linear(feat_dim, 512),           # head[1]
            nn.GELU(),                          # head[2]
            nn.Dropout(drop_rate * 0.75),       # head[3]
            nn.Linear(512, num_classes),        # head[4]
        )
        self._init_head()

    # ── Weight initialisation ────────────────────────────────────────────────

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Fine-tuning helpers ──────────────────────────────────────────────────

    def freeze_backbone(self):
        """Freeze backbone; only the classification head trains (warm-up)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze the full network for end-to-end fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_backbone_params(self):
        return list(self.backbone.parameters())

    def get_head_params(self):
        return list(self.gap.parameters()) + list(self.head.parameters())

    # ── Grad-CAM target ──────────────────────────────────────────────────────

    def get_target_layer(self):
        """
        Last MBConv block of EfficientNet-B4.
        Pass this to GradCAM(model, target_layers=[model.get_target_layer()])
        """
        return self.backbone.blocks[-1]

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x):
        if self.use_srm:
            # Un-normalize to [0, 1] so the Laplacian filter sees real pixel
            # intensities where AI upsampling artifacts are actually visible.
            raw_x = x * self._img_std + self._img_mean   # [0, 1] range
            srm_x = self.srm_layer(raw_x)
            x = torch.cat([x, srm_x], dim=1)
            
        # (B, 1792, H', W') — spatial feature map
        features   = self.backbone.forward_features(x)
        # (B, 1792)
        pooled     = self.gap(features).flatten(1)
        # Head: Dropout → Linear → GELU → Dropout → Linear
        drop1      = self.head[0](pooled)
        lin1       = self.head[1](drop1)
        act        = self.head[2](lin1)
        embedding  = act                          # (B, 512) — exposed for analysis
        drop2      = self.head[3](embedding)
        logits     = self.head[4](drop2).squeeze(-1)   # (B,)
        return logits, None, embedding


# Backward-compatibility alias
DeepfakeDetector = EfficientNetB4Detector


# ── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading EfficientNet-B4 pretrained weights (requires internet)…")
    model  = DeepfakeDetector(pretrained=True)
    dummy  = torch.randn(2, 3, 380, 380)
    logits, attn, emb = model(dummy)
    print(f"Logits    : {logits.shape}")   # (2,)
    print(f"Embedding : {emb.shape}")       # (2, 512)
    print(f"AttnMap   : {attn}")           # None — use Grad-CAM instead
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total/1e6:.1f}M")
    print(f"Grad-CAM target: {model.get_target_layer().__class__.__name__}")
