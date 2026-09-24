"""Deterministic frozen-score features for the Stage A residual ranker."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
from numbers import Integral, Real

import numpy as np
import torch
import torch.nn.functional as F

from src.grouped_validation import grouped_meta_fold
from src.unified_dstar_ensemble import rank_descending


FEATURE_NAMES = (
    "transfer_rank",
    "dstar_rank",
    "transfer_score_robust_z",
    "dstar_score_robust_z",
    "transfer_available",
    "response_supported",
    "transfer_top_margin",
    "dstar_top_margin",
    "dstar_seed_rank_std",
    "top10_overlap",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}

_ROBUST_Z_NUMERATOR = 0.6744897501960817


@dataclass(frozen=True)
class SelectionMetrics:
    """Exact validation hit counts and query-macro reciprocal rank."""

    h10_hits: int
    h50_hits: int
    mrr: float

    def __post_init__(self) -> None:
        for name in ("h10_hits", "h50_hits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        if (
            isinstance(self.mrr, bool)
            or not isinstance(self.mrr, Real)
            or not math.isfinite(float(self.mrr))
            or not 0.0 <= float(self.mrr) <= 1.0
        ):
            raise ValueError("mrr must be finite and in [0, 1]")
        object.__setattr__(self, "mrr", float(self.mrr))


@dataclass(frozen=True)
class CandidateResult:
    """One canonical residual configuration and its meta-OOF result."""

    config_id: str
    metrics: SelectionMetrics
    complexity: int

    def __post_init__(self) -> None:
        if not isinstance(self.config_id, str) or not self.config_id:
            raise ValueError("config_id must be a nonempty string")
        if not isinstance(self.metrics, SelectionMetrics):
            raise TypeError("metrics must be SelectionMetrics")
        if (
            isinstance(self.complexity, bool)
            or not isinstance(self.complexity, Integral)
            or self.complexity < 0
        ):
            raise ValueError("complexity must be a nonnegative integer")
        object.__setattr__(self, "complexity", int(self.complexity))


def select_feasible(
    candidates: Sequence[CandidateResult], baseline: SelectionMetrics
) -> CandidateResult:
    """Apply the frozen noninferiority gates and lexicographic selector."""

    if not isinstance(baseline, SelectionMetrics):
        raise TypeError("baseline must be SelectionMetrics")
    feasible = [
        candidate
        for candidate in candidates
        if candidate.metrics.h10_hits >= baseline.h10_hits
        and candidate.metrics.h50_hits >= baseline.h50_hits
    ]
    if not feasible:
        return CandidateResult("transfer_fallback", baseline, complexity=0)
    return max(
        feasible,
        key=lambda candidate: (
            candidate.metrics.h10_hits,
            candidate.metrics.h50_hits,
            candidate.metrics.mrr,
            -candidate.complexity,
            candidate.config_id,
        ),
    )


def _real_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must contain real numeric values")
    return np.asarray(array, dtype=np.float64)


def _candidate_hashes(value: Sequence[str], candidate_count: int) -> tuple[str, ...]:
    hashes = tuple(value)
    if len(hashes) != candidate_count:
        raise ValueError("candidate hash count must match score columns")
    if any(not isinstance(candidate_hash, str) or not candidate_hash for candidate_hash in hashes):
        raise ValueError("candidate hashes must be nonempty strings")
    if len(set(hashes)) != len(hashes):
        raise ValueError("candidate hashes must be unique")
    return hashes


def _rank_percentiles(
    scores: np.ndarray, candidate_hashes: tuple[str, ...]
) -> np.ndarray:
    ranks = rank_descending(scores, candidate_hashes)
    result = np.zeros(scores.shape, dtype=np.float64)
    for query_index, row_ranks in enumerate(ranks):
        available = ~np.isnan(row_ranks)
        count = int(np.count_nonzero(available))
        if count:
            result[query_index, available] = 1.0 - (
                row_ranks[available] - 1.0
            ) / max(count - 1, 1)
    return result


def _robust_z(scores: np.ndarray) -> np.ndarray:
    result = np.zeros(scores.shape, dtype=np.float64)
    for query_index, row in enumerate(scores):
        available = ~np.isnan(row)
        values = row[available]
        if not values.size:
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad > 0.0:
            result[query_index, available] = (
                _ROBUST_Z_NUMERATOR * (values - median) / mad
            )
        else:
            standard_deviation = float(np.std(values, ddof=0))
            if standard_deviation > 0.0:
                result[query_index, available] = (
                    values - median
                ) / standard_deviation
    return result


def _top_margin(scores: np.ndarray) -> np.ndarray:
    margins = np.zeros(scores.shape[0], dtype=np.float64)
    for query_index, row in enumerate(scores):
        values = row[~np.isnan(row)]
        if values.size >= 2:
            top_two = np.partition(values, values.size - 2)[-2:]
            margins[query_index] = float(top_two[1] - top_two[0])
    return margins


def _top10_overlap(
    transfer: np.ndarray,
    dstar: np.ndarray,
    candidate_hashes: tuple[str, ...],
) -> np.ndarray:
    overlaps = np.zeros(transfer.shape[0], dtype=np.float64)
    transfer_ranks = rank_descending(transfer, candidate_hashes)
    dstar_ranks = rank_descending(dstar, candidate_hashes)
    for query_index in range(transfer.shape[0]):
        available_count = int(np.count_nonzero(~np.isnan(transfer[query_index])))
        top_count = min(10, available_count, transfer.shape[1])
        if not top_count:
            continue
        transfer_top = set(np.flatnonzero(transfer_ranks[query_index] <= top_count))
        dstar_top = set(np.flatnonzero(dstar_ranks[query_index] <= top_count))
        overlaps[query_index] = len(transfer_top & dstar_top) / top_count
    return overlaps


def _support_matrix(
    response_supported: object, *, query_count: int, candidate_count: int
) -> np.ndarray:
    raw_support = np.asarray(response_supported)
    if not (
        np.issubdtype(raw_support.dtype, np.bool_)
        or (
            np.issubdtype(raw_support.dtype, np.number)
            and not np.issubdtype(raw_support.dtype, np.complexfloating)
        )
    ):
        raise ValueError("response support must contain boolean or binary numeric values")
    support = np.asarray(raw_support, dtype=np.float64)
    expected_shape = (query_count, candidate_count)
    if support.shape == (candidate_count,):
        support = np.broadcast_to(support, expected_shape)
    elif support.shape != expected_shape:
        raise ValueError(
            "response support shape must match candidates or the score matrix"
        )
    if not np.all(np.isfinite(support)):
        raise ValueError("response support values must be finite")
    if not np.all((support == 0.0) | (support == 1.0)):
        raise ValueError("response support values must be binary")
    return np.asarray(support, dtype=np.float64)


def calibrated_dstar_fallback(
    dstar_scores: object, calibrated_scores: object
) -> np.ndarray:
    """Replace explicit Dstar gaps with a deterministic score below the row minimum."""

    raw = _real_array(dstar_scores, name="Dstar scores")
    calibrated = _real_array(calibrated_scores, name="calibrated Dstar scores")
    if raw.ndim != 2 or calibrated.shape != raw.shape:
        raise ValueError("Dstar and calibrated scores must have one common 2D shape")
    if np.any(~(np.isnan(raw) | np.isfinite(raw))):
        raise ValueError("nonmissing Dstar scores must be finite")
    if not np.isfinite(calibrated).all():
        raise ValueError("calibrated Dstar scores must be finite")
    result = calibrated.copy()
    for query_index in range(raw.shape[0]):
        available = ~np.isnan(raw[query_index])
        if not np.any(available):
            raise ValueError("each query must have at least one available Dstar score")
        minimum = float(np.min(calibrated[query_index, available]))
        decrement = max(1.0, abs(minimum) * 1e-6)
        result[query_index, ~available] = minimum - decrement
    if not np.isfinite(result).all():
        raise ValueError("Dstar fallback scores must be finite")
    return result


def build_candidate_features(
    transfer: object,
    dstar_seeds: object,
    response_supported: object,
    candidate_hashes: Sequence[str],
) -> np.ndarray:
    """Build the frozen Stage A feature tensor as ``(query, candidate, feature)``.

    ``dstar_seeds`` is ordered as ``(seed, query, candidate)``. A two-dimensional
    matrix is accepted as a single seed. Transfer and Dstar ``NaN`` values denote
    explicit candidate-level unavailability; all other nonfinite values are rejected.
    """

    transfer_scores = _real_array(transfer, name="transfer scores")
    if transfer_scores.ndim != 2 or not all(transfer_scores.shape):
        raise ValueError("transfer scores must be a nonempty two-dimensional matrix")
    if np.any(~(np.isnan(transfer_scores) | np.isfinite(transfer_scores))):
        raise ValueError("nonmissing transfer scores must be finite")

    seed_scores = _real_array(dstar_seeds, name="Dstar seed scores")
    if seed_scores.ndim == 2:
        seed_scores = seed_scores[None, ...]
    if seed_scores.ndim != 3 or seed_scores.shape[0] == 0:
        raise ValueError("Dstar seed scores must have shape (seed, query, candidate)")
    if seed_scores.shape[1:] != transfer_scores.shape:
        raise ValueError("Dstar seed score shape must match transfer scores")
    if np.any(~(np.isnan(seed_scores) | np.isfinite(seed_scores))):
        raise ValueError("nonmissing Dstar seed scores must be finite")
    dstar_available = ~np.isnan(seed_scores[0])
    if any(
        not np.array_equal(~np.isnan(seed_scores[seed]), dstar_available)
        for seed in range(1, seed_scores.shape[0])
    ):
        raise ValueError("Dstar seed availability masks must be identical")
    if not np.all(np.any(dstar_available, axis=1)):
        raise ValueError("each query must have at least one available Dstar score")

    query_count, candidate_count = transfer_scores.shape
    hashes = _candidate_hashes(candidate_hashes, candidate_count)
    support = _support_matrix(
        response_supported,
        query_count=query_count,
        candidate_count=candidate_count,
    )

    available = ~np.isnan(transfer_scores)
    dstar_scores = np.full(transfer_scores.shape, np.nan, dtype=np.float64)
    dstar_scores[dstar_available] = np.mean(
        seed_scores[:, dstar_available], axis=0, dtype=np.float64
    )
    transfer_ranks = _rank_percentiles(transfer_scores, hashes)
    dstar_ranks = _rank_percentiles(dstar_scores, hashes)
    seed_rank_sum = np.zeros(transfer_scores.shape, dtype=np.float64)
    seed_rank_square_sum = np.zeros(transfer_scores.shape, dtype=np.float64)
    for seed in range(seed_scores.shape[0]):
        ranks = _rank_percentiles(seed_scores[seed], hashes)
        seed_rank_sum += ranks
        seed_rank_square_sum += ranks * ranks
    seed_rank_mean = seed_rank_sum / seed_scores.shape[0]
    seed_rank_variance = np.maximum(
        seed_rank_square_sum / seed_scores.shape[0] - seed_rank_mean * seed_rank_mean,
        0.0,
    )

    features = np.zeros(
        (query_count, candidate_count, len(FEATURE_NAMES)), dtype=np.float32
    )
    features[..., FEATURE_INDEX["transfer_available"]] = available
    features[..., FEATURE_INDEX["transfer_rank"]] = transfer_ranks
    features[..., FEATURE_INDEX["dstar_rank"]] = dstar_ranks
    features[..., FEATURE_INDEX["transfer_score_robust_z"]] = _robust_z(
        transfer_scores
    )
    features[..., FEATURE_INDEX["dstar_score_robust_z"]] = _robust_z(dstar_scores)
    features[..., FEATURE_INDEX["response_supported"]] = support
    features[..., FEATURE_INDEX["transfer_top_margin"]] = np.where(
        available, _top_margin(transfer_scores)[:, None], 0.0
    )
    features[..., FEATURE_INDEX["dstar_top_margin"]] = np.where(
        dstar_available, _top_margin(dstar_scores)[:, None], 0.0
    )
    features[..., FEATURE_INDEX["dstar_seed_rank_std"]] = np.sqrt(
        seed_rank_variance
    )
    features[..., FEATURE_INDEX["top10_overlap"]] = np.where(
        available & dstar_available,
        _top10_overlap(transfer_scores, dstar_scores, hashes)[:, None],
        0.0,
    )

    dstar_numeric = (
        "dstar_rank",
        "dstar_score_robust_z",
        "dstar_top_margin",
        "dstar_seed_rank_std",
        "top10_overlap",
    )
    for name in dstar_numeric:
        features[..., FEATURE_INDEX[name]][~dstar_available] = 0.0

    transfer_numeric = (
        "transfer_rank",
        "transfer_score_robust_z",
        "transfer_top_margin",
        "top10_overlap",
    )
    for name in transfer_numeric:
        features[..., FEATURE_INDEX[name]][~available] = 0.0
    if not np.all(np.isfinite(features)):
        raise ValueError("candidate features must be finite")
    return features


class NonnegativeLinearResidual(torch.nn.Module):
    """Linear residual whose effective feature weights cannot become negative."""

    def __init__(self, feature_count: int = len(FEATURE_NAMES)) -> None:
        super().__init__()
        if isinstance(feature_count, bool) or feature_count <= 0:
            raise ValueError("feature count must be a positive integer")
        self.raw_weights = torch.nn.Parameter(torch.zeros(int(feature_count)))
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def weights(self) -> torch.Tensor:
        return F.softplus(self.raw_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights() + self.bias


class MLPResidual(torch.nn.Module):
    """One-hidden-layer residual head with the frozen Stage A width."""

    def __init__(
        self, feature_count: int = len(FEATURE_NAMES), width: int = 32
    ) -> None:
        super().__init__()
        if isinstance(feature_count, bool) or feature_count <= 0:
            raise ValueError("feature count must be a positive integer")
        if isinstance(width, bool) or width <= 0:
            raise ValueError("width must be a positive integer")
        self.network = torch.nn.Sequential(
            torch.nn.Linear(int(feature_count), int(width)),
            torch.nn.GELU(),
            torch.nn.Linear(int(width), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def fuse_residual_scores(
    calibrated_transfer: torch.Tensor,
    dstar_score: torch.Tensor,
    residual: torch.Tensor,
    residual_without_transfer: torch.Tensor,
    transfer_available: torch.Tensor,
    *,
    base_mode: str = "transfer_base",
) -> torch.Tensor:
    """Add a residual to the configured base score."""

    expected_shape = calibrated_transfer.shape
    named_tensors = {
        "Dstar score": dstar_score,
        "residual": residual,
        "transfer-free residual": residual_without_transfer,
        "transfer availability mask": transfer_available,
    }
    for name, value in named_tensors.items():
        if value.shape != expected_shape:
            raise ValueError(f"{name} shape must match calibrated transfer scores")
    if transfer_available.dtype != torch.bool:
        raise ValueError("transfer availability mask must be boolean")
    if base_mode == "dstar_base":
        return dstar_score + residual
    if base_mode != "transfer_base":
        raise ValueError("base mode must be 'transfer_base' or 'dstar_base'")
    return torch.where(
        transfer_available,
        calibrated_transfer + residual,
        dstar_score + residual_without_transfer,
    )


def pairwise_loss(
    positive_scores: torch.Tensor, unlabeled_scores: torch.Tensor
) -> torch.Tensor:
    """Mean logistic loss over every positive-versus-unlabeled score pair."""

    positives = positive_scores.reshape(-1)
    unlabeled = unlabeled_scores.reshape(-1)
    if positives.numel() == 0 or unlabeled.numel() == 0:
        return positives.sum() * 0.0 + unlabeled.sum() * 0.0
    return F.softplus(unlabeled[:, None] - positives[None, :]).mean()


def pairwise_positive_unlabeled_loss(
    fused_scores: torch.Tensor,
    labels: torch.Tensor,
    sampled_unlabeled: torch.Tensor,
) -> torch.Tensor:
    """Average query-local pairwise retrieval loss over sampled unlabeled items."""

    if fused_scores.ndim != 2:
        raise ValueError("fused scores must be a two-dimensional matrix")
    if labels.shape != fused_scores.shape:
        raise ValueError("label shape must match fused scores")
    if sampled_unlabeled.shape != fused_scores.shape:
        raise ValueError("sampled unlabeled mask shape must match fused scores")
    if labels.dtype != torch.bool or sampled_unlabeled.dtype != torch.bool:
        raise ValueError("labels and sampled unlabeled mask must be boolean")

    query_losses = []
    for query_index in range(fused_scores.shape[0]):
        positives = fused_scores[query_index][labels[query_index]]
        unlabeled_mask = sampled_unlabeled[query_index] & ~labels[query_index]
        unlabeled = fused_scores[query_index][unlabeled_mask]
        if positives.numel() and unlabeled.numel():
            query_losses.append(pairwise_loss(positives, unlabeled))
    if not query_losses:
        return fused_scores.sum() * 0.0
    return torch.stack(query_losses).mean()


def masked_transfer_kl(
    transfer_scores: torch.Tensor,
    fused_scores: torch.Tensor,
    temperature: float,
    transfer_available: torch.Tensor,
) -> torch.Tensor:
    """KL over available candidates, skipping rows with fewer than two."""

    if transfer_scores.ndim != 2 or fused_scores.shape != transfer_scores.shape:
        raise ValueError("transfer and fused scores must have the same two-dimensional shape")
    if transfer_available.dtype != torch.bool:
        raise ValueError("transfer availability mask must be boolean")
    if transfer_available.shape == (transfer_scores.shape[0],):
        availability = transfer_available[:, None].expand_as(transfer_scores)
    elif transfer_available.shape == transfer_scores.shape:
        availability = transfer_available
    else:
        raise ValueError(
            "transfer availability mask must match score rows or score candidates"
        )
    availability = availability.to(device=transfer_scores.device)
    if fused_scores.device != transfer_scores.device:
        raise ValueError("transfer and fused scores must be on the same device")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be positive and finite")

    scale = float(temperature)
    query_losses = []
    detached_transfer = transfer_scores.detach()
    for query_index in range(transfer_scores.shape[0]):
        candidate_mask = availability[query_index]
        if int(candidate_mask.sum().item()) < 2:
            continue
        transfer = detached_transfer[query_index][candidate_mask]
        fused = fused_scores[query_index][candidate_mask]
        if not torch.all(torch.isfinite(transfer)) or not torch.all(torch.isfinite(fused)):
            raise ValueError("available transfer and fused scores must be finite")
        query_losses.append(
            F.kl_div(
                F.log_softmax(fused / scale, dim=-1),
                F.softmax(transfer / scale, dim=-1),
                reduction="sum",
            )
            * (scale * scale)
        )
    if not query_losses:
        return fused_scores.reshape(-1)[:0].sum()
    return torch.stack(query_losses).mean()


def residual_square_penalty(residual: torch.Tensor) -> torch.Tensor:
    """Mean squared residual magnitude used by the Stage A L2 term."""

    if residual.numel() == 0:
        return residual.sum() * 0.0
    return residual.square().mean()


def deterministic_candidate_indices(
    labels: torch.Tensor,
    transfer_scores: torch.Tensor,
    dstar_scores: torch.Tensor,
    query_ids: Sequence[str],
    candidate_hashes: Sequence[str],
    *,
    seed: int = 20260726,
    transfer_top_k: int = 100,
    dstar_top_k: int = 100,
    hash_unlabeled_count: int = 256,
) -> tuple[torch.Tensor, ...]:
    """Return deduplicated candidate indices using stable query identities."""

    if labels.ndim != 2 or labels.dtype != torch.bool:
        raise ValueError("labels must be a two-dimensional boolean matrix")
    if transfer_scores.shape != labels.shape or dstar_scores.shape != labels.shape:
        raise ValueError("score shapes must match labels")
    stable_query_ids = tuple(query_ids)
    if len(stable_query_ids) != labels.shape[0]:
        raise ValueError("query ID count must match score rows")
    if any(not isinstance(query_id, str) or not query_id for query_id in stable_query_ids):
        raise ValueError("query IDs must be nonempty strings")
    if len(set(stable_query_ids)) != len(stable_query_ids):
        raise ValueError("query IDs must be unique stable identifiers")
    for name, value in (
        ("transfer top-k", transfer_top_k),
        ("Dstar top-k", dstar_top_k),
        ("hash unlabeled count", hash_unlabeled_count),
    ):
        if isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")

    hashes = _candidate_hashes(candidate_hashes, labels.shape[1])
    transfer = np.asarray(transfer_scores.detach().cpu(), dtype=np.float64)
    dstar = np.asarray(dstar_scores.detach().cpu(), dtype=np.float64)
    if np.any(~(np.isnan(transfer) | np.isfinite(transfer))):
        raise ValueError("nonmissing transfer scores must be finite")
    if np.any(~(np.isnan(dstar) | np.isfinite(dstar))):
        raise ValueError("nonmissing Dstar scores must be finite")
    label_array = np.asarray(labels.detach().cpu(), dtype=bool)
    selected = label_array.copy()

    for query_index, query_id in enumerate(stable_query_ids):
        transfer_order = sorted(
            np.flatnonzero(~np.isnan(transfer[query_index])),
            key=lambda index: (-transfer[query_index, index], hashes[index]),
        )
        dstar_order = sorted(
            range(labels.shape[1]),
            key=lambda index: (
                bool(np.isnan(dstar[query_index, index])),
                (
                    -dstar[query_index, index]
                    if not np.isnan(dstar[query_index, index])
                    else 0.0
                ),
                hashes[index],
            ),
        )
        selected[query_index, transfer_order[: int(transfer_top_k)]] = True
        selected[query_index, dstar_order[: int(dstar_top_k)]] = True

        unlabeled = np.flatnonzero(~label_array[query_index])
        hash_order = sorted(
            unlabeled,
            key=lambda index: (
                sha256(
                    f"{int(seed)}:{query_id}:{hashes[index]}".encode("utf-8")
                ).digest(),
                hashes[index],
            ),
        )
        selected[query_index, hash_order[: int(hash_unlabeled_count)]] = True

    return tuple(
        torch.as_tensor(
            np.flatnonzero(selected[query_index]),
            dtype=torch.long,
            device=labels.device,
        )
        for query_index in range(labels.shape[0])
    )


def sample_candidate_mask(
    labels: torch.Tensor,
    transfer_scores: torch.Tensor,
    dstar_scores: torch.Tensor,
    query_ids: Sequence[str],
    candidate_hashes: Sequence[str],
    *,
    seed: int = 20260726,
    transfer_top_k: int = 100,
    dstar_top_k: int = 100,
    hash_unlabeled_count: int = 256,
) -> torch.Tensor:
    """Convert deterministic per-query candidate indices to a boolean mask."""

    indexes = deterministic_candidate_indices(
        labels,
        transfer_scores,
        dstar_scores,
        query_ids,
        candidate_hashes,
        seed=seed,
        transfer_top_k=transfer_top_k,
        dstar_top_k=dstar_top_k,
        hash_unlabeled_count=hash_unlabeled_count,
    )
    selected = torch.zeros_like(labels)
    for query_index, candidate_indexes in enumerate(indexes):
        selected[query_index, candidate_indexes] = True
    return selected


__all__ = [
    "CandidateResult",
    "FEATURE_INDEX",
    "FEATURE_NAMES",
    "MLPResidual",
    "NonnegativeLinearResidual",
    "SelectionMetrics",
    "build_candidate_features",
    "calibrated_dstar_fallback",
    "deterministic_candidate_indices",
    "fuse_residual_scores",
    "grouped_meta_fold",
    "masked_transfer_kl",
    "pairwise_loss",
    "pairwise_positive_unlabeled_loss",
    "residual_square_penalty",
    "sample_candidate_mask",
    "select_feasible",
]
