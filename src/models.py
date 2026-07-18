from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models


class TripleImageEncoder(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        aggregation: Literal["concat", "mean"] = "concat",
    ):
        super().__init__()
        self.aggregation = aggregation
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.single_embedding_dim = in_features
        self.output_dim = in_features * 3 if aggregation == "concat" else in_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, C, H, W]
        b, n, c, h, w = images.shape
        x = images.view(b * n, c, h, w)
        emb = self.backbone(x)
        emb = emb.view(b, n, -1)
        if self.aggregation == "mean":
            return emb.mean(dim=1)
        if self.aggregation == "concat":
            return emb.reshape(b, -1)
        raise ValueError(f"Unknown aggregation: {self.aggregation}")


class ImageOnlyNet(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True, aggregation="concat", hidden_dim=128, dropout=0.25):
        super().__init__()
        self.encoder = TripleImageEncoder(pretrained=pretrained, freeze_backbone=freeze_backbone, aggregation=aggregation)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(images)
        return self.head(emb).squeeze(1)


class MultimodalNet(nn.Module):
    def __init__(
        self,
        n_tabular_features: int,
        pretrained=True,
        freeze_backbone=True,
        aggregation="concat",
        tab_hidden_dim=32,
        fusion_hidden_dim=128,
        dropout=0.25,
    ):
        super().__init__()
        self.encoder = TripleImageEncoder(pretrained=pretrained, freeze_backbone=freeze_backbone, aggregation=aggregation)
        self.tabular_branch = nn.Sequential(
            nn.Linear(n_tabular_features, tab_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim + tab_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        img_emb = self.encoder(images)
        tab_emb = self.tabular_branch(tabular)
        fused = torch.cat([img_emb, tab_emb], dim=1)
        return self.head(fused).squeeze(1)
