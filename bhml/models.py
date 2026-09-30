"""Neural network models for the black-hole ML tasks.

Starts with a small MLP for the photon capture classifier (Phase 2).
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """A small fully-connected network that outputs a single logit.

    For binary classification: apply torch.sigmoid to the output to get
    P(capture). Trained with BCEWithLogitsLoss, which expects the raw logit.
    """

    def __init__(self, in_dim: int = 2, hidden: tuple[int, ...] = (32, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]  # final logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # shape (N,) logits
