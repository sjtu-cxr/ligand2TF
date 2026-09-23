"""Candidate-level gated correction with a Top-10-weighted ranking loss."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class CandidateGatedCorrection(torch.nn.Module):
    """Bounded candidate correction around a frozen backbone ranking."""

    def __init__(self, *, feature_count: int, width: int, gamma: float) -> None:
        super().__init__()
        if feature_count <= 0 or width <= 0:
            raise ValueError("feature_count and width must be positive")
        if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be finite and in (0, 1]")
        self.feature_count = int(feature_count)
        self.gamma = float(gamma)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(self.feature_count, int(width)),
            torch.nn.ReLU(),
        )
        self.gate_head = torch.nn.Linear(int(width), 1)
        self.delta_head = torch.nn.Linear(int(width), 1)

    def forward(
        self, features: torch.Tensor, backbone_rank_percentile: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim < 2 or features.shape[-1] != self.feature_count:
            raise ValueError("features must end in the configured feature dimension")
        if backbone_rank_percentile.shape != features.shape[:-1]:
            raise ValueError("backbone rank percentiles must align with features")
        hidden = self.trunk(features)
        gate = torch.sigmoid(self.gate_head(hidden).squeeze(-1))
        delta = torch.tanh(self.delta_head(hidden).squeeze(-1))
        score = backbone_rank_percentile + self.gamma * gate * delta
        return score, gate, delta


def topk_pair_weights(backbone_ranks: torch.Tensor) -> torch.Tensor:
    """Return fixed 4/2/1 negative weights for ranks 1-10/11-50/later."""

    if not torch.is_floating_point(backbone_ranks):
        backbone_ranks = backbone_ranks.to(dtype=torch.float32)
    if backbone_ranks.ndim != 1 or not torch.isfinite(backbone_ranks).all():
        raise ValueError("backbone ranks must be a finite vector")
    if torch.any(backbone_ranks < 1.0):
        raise ValueError("backbone ranks must be one-based")
    return torch.where(
        backbone_ranks <= 10.0,
        torch.full_like(backbone_ranks, 4.0),
        torch.where(
            backbone_ranks <= 50.0,
            torch.full_like(backbone_ranks, 2.0),
            torch.ones_like(backbone_ranks),
        ),
    )


def weighted_pairwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    query_coordinates: torch.Tensor,
    backbone_ranks: torch.Tensor,
) -> torch.Tensor:
    """Query-macro positive-vs-unlabeled logistic loss with Top-k weights."""

    if not (
        scores.ndim == labels.ndim == query_coordinates.ndim == backbone_ranks.ndim == 1
        and scores.shape == labels.shape == query_coordinates.shape == backbone_ranks.shape
    ):
        raise ValueError("pairwise training vectors must align")
    if labels.dtype != torch.bool:
        raise ValueError("labels must be boolean")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    losses = []
    for query in torch.unique(query_coordinates, sorted=True):
        mask = query_coordinates == query
        query_labels = labels[mask]
        if not torch.any(query_labels) or not torch.any(~query_labels):
            continue
        positive = scores[mask][query_labels]
        negative = scores[mask][~query_labels]
        weights = topk_pair_weights(backbone_ranks[mask][~query_labels])
        pair_losses = F.softplus(-(positive[:, None] - negative[None, :]))
        losses.append((pair_losses * weights[None, :]).sum() / (
            positive.numel() * weights.sum()
        ))
    if not losses:
        raise ValueError("sample lacks a positive/unlabeled query contrast")
    return torch.stack(losses).mean()


def gated_correction_regularization(
    gate: torch.Tensor,
    delta: torch.Tensor,
    *,
    lambda_corr: float,
    lambda_gate: float,
) -> torch.Tensor:
    """Penalize score displacement and indiscriminate gate activation."""

    if gate.shape != delta.shape or gate.numel() == 0:
        raise ValueError("gate and correction tensors must align and be nonempty")
    if lambda_corr < 0.0 or lambda_gate < 0.0:
        raise ValueError("regularization weights must be nonnegative")
    if not torch.isfinite(gate).all() or not torch.isfinite(delta).all():
        raise ValueError("gate and correction tensors must be finite")
    return (
        float(lambda_corr) * torch.mean(torch.square(gate * delta))
        + float(lambda_gate) * torch.mean(gate)
    )


__all__ = [
    "CandidateGatedCorrection",
    "gated_correction_regularization",
    "topk_pair_weights",
    "weighted_pairwise_loss",
]
