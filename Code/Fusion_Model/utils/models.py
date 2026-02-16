"""
Model architecture for multimodal (image + CBC) anemia classification.

Components
----------
ImageEncoder  — VGG-16 pretrained feature extractor → 512-d embedding
TabularEncoder — small MLP for 10 CBC features → 32-d embedding
FusionModel   — late-fusion: concat(512, 32) → FC → 4-class logits
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """VGG-16 conv features → AdaptiveAvgPool → 512-d vector."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg16(weights=weights)
        self.features = vgg.features          # conv blocks
        self.pool = nn.AdaptiveAvgPool2d(1)   # → (batch, 512, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)            # (batch, 512)
        return x


class TabularEncoder(nn.Module):
    """Two-layer MLP  10 → 32 → 32  with ReLU + Dropout."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                    # (batch, 32)


class FusionModel(nn.Module):
    """
    Late-fusion classifier.

    concat(ImageEncoder(img), TabularEncoder(cbc)) → FC layers → 4-class logits
    """

    def __init__(
        self,
        num_classes: int = 4,
        tab_input_dim: int = 10,
        pretrained_image: bool = True,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained_image)
        self.tabular_encoder = TabularEncoder(input_dim=tab_input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    # ---- Freeze / unfreeze helpers for two-phase training ----
    def freeze_image_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def unfreeze_image_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = True

    def to_channels_last(self):
        """Convert conv layers to channels-last memory format (faster on MPS & CUDA)."""
        self.image_encoder = self.image_encoder.to(memory_format=torch.channels_last)
        return self

    def forward(self, image: torch.Tensor, cbc: torch.Tensor) -> torch.Tensor:
        img_emb = self.image_encoder(image)       # (batch, 512)
        tab_emb = self.tabular_encoder(cbc)       # (batch, 32)
        fused = torch.cat([img_emb, tab_emb], dim=1)  # (batch, 544)
        logits = self.classifier(fused)           # (batch, 4)
        return logits
