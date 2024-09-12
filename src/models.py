import torch
from torch import nn


class CTModel(nn.Module):
    def __init__(self, with_anchors: bool = False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.subtype_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2)
        )
        self.anchor_heads = AnchorHeads() if with_anchors else None

    def forward_diagnosis(self, features):
        embedding = self.encoder(features)
        return self.subtype_head(embedding), embedding

    def forward(self, features):
        logits, embedding = self.forward_diagnosis(features)
        anchors = self.anchor_heads(embedding) if self.anchor_heads is not None else None
        return logits, anchors, embedding


class WSIModel(nn.Module):
    def __init__(self, with_anchors: bool = True):
        super().__init__()
        self.patch_encoder = nn.Sequential(
            nn.Linear(1792, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
        )
        self.subtype_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2)
        )
        self.anchor_heads = AnchorHeads() if with_anchors else None

    def encode(self, patches):
        return self.patch_encoder(patches).mean(dim=1)

    def forward_diagnosis(self, patches):
        embedding = self.encode(patches)
        return self.subtype_head(embedding), embedding

    def forward(self, patches):
        logits, embedding = self.forward_diagnosis(patches)
        anchors = self.anchor_heads(embedding) if self.anchor_heads is not None else None
        return logits, anchors, embedding


class AnchorHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_size = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.cyfra = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.scc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, embedding):
        return torch.cat(
            [self.t_size(embedding), self.cyfra(embedding), self.scc(embedding)], dim=1
        )

