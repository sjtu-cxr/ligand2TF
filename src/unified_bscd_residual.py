"""Frozen-score features for an additive residual on the unified B_SCD model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.transfer_distilled_residual import (
    _rank_percentiles,
    _support_matrix,
    _top10_overlap,
    _top_margin,
)


CHANNELS = ("S", "C", "Dstar", "B_SD", "B_CD", "B_SCD")
FEATURE_NAMES = tuple(
    [f"{name}_rank" for name in CHANNELS]
    + [f"{name}_score" for name in CHANNELS]
    + ["S_available", "C_available", "response_supported"]
    + [f"{name}_minus_B_SCD" for name in CHANNELS[:-1]]
    + [f"{name}_top_margin" for name in CHANNELS]
    + ["Dstar_seed_rank_std"]
    + [f"{name}_B_SCD_top10_overlap" for name in CHANNELS[:-1]]
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


def require_finite_matrix(values: object, name: str) -> np.ndarray:
    """Return ``values`` as a nonempty finite real-valued 2D matrix."""

    try:
        raw = np.asarray(values)
        if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
            raw.dtype, np.complexfloating
        ):
            raise ValueError
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a nonempty finite 2D matrix") from error
    if matrix.ndim != 2 or not all(matrix.shape) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a nonempty finite 2D matrix")
    return matrix


def eligible_residual_configs(configs: Sequence[Any]) -> tuple[Any, ...]:
    """Select the frozen residual configurations that use no distillation."""

    selected = tuple(config for config in configs if config.lambda_distill == 0.0)
    if not selected:
        raise ValueError("zero-distillation residual grid is empty")
    return selected


def fuse_bscd_residual(
    backbone: object, residual: object | None
) -> np.ndarray:
    """Add a residual correction, or return an exact copy of the backbone."""

    base = require_finite_matrix(backbone, "B_SCD")
    if residual is None:
        return base.copy()
    correction = require_finite_matrix(residual, "residual")
    if correction.shape != base.shape:
        raise ValueError("residual and B_SCD shapes differ")
    with np.errstate(over="ignore", invalid="ignore"):
        fused = base + correction
    if not np.isfinite(fused).all():
        raise ValueError("fused B_SCD residual scores must be finite")
    return fused


def residual_magnitude_summary(residual: object) -> dict[str, float]:
    """Summarize the magnitude of a finite residual correction matrix."""

    correction = require_finite_matrix(residual, "residual")
    absolute = np.abs(correction)
    maximum = float(np.max(absolute))
    if maximum == 0.0:
        return {
            "mean_abs": 0.0,
            "median_abs": 0.0,
            "p95_abs": 0.0,
            "p99_abs": 0.0,
            "max_abs": 0.0,
            "rms": 0.0,
        }
    scaled = absolute / maximum

    def restore_scale(value: float) -> float:
        return maximum * min(float(value), 1.0)

    return {
        "mean_abs": restore_scale(np.mean(scaled)),
        "median_abs": restore_scale(np.median(scaled)),
        "p95_abs": restore_scale(np.percentile(scaled, 95)),
        "p99_abs": restore_scale(np.percentile(scaled, 99)),
        "max_abs": maximum,
        "rms": restore_scale(np.sqrt(np.mean(np.square(scaled)))),
    }


def _score_matrices(scores: object) -> dict[str, np.ndarray]:
    if not isinstance(scores, Mapping) or set(scores) != set(CHANNELS):
        raise ValueError(f"scores must have exact channel keys {CHANNELS}")
    matrices: dict[str, np.ndarray] = {}
    common_shape: tuple[int, int] | None = None
    for channel in CHANNELS:
        raw = np.asarray(scores[channel])
        if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
            raw.dtype, np.complexfloating
        ):
            raise ValueError("scores must have one common nonempty finite 2D shape")
        matrix = np.asarray(raw, dtype=np.float64)
        if (
            matrix.ndim != 2
            or not all(matrix.shape)
            or not np.isfinite(matrix).all()
            or (common_shape is not None and matrix.shape != common_shape)
        ):
            raise ValueError("scores must have one common nonempty finite 2D shape")
        common_shape = matrix.shape
        matrices[channel] = matrix
    return matrices


def _availability(value: object, shape: tuple[int, int], *, name: str) -> np.ndarray:
    available = np.asarray(value)
    if available.dtype != np.bool_ or available.shape != shape:
        raise ValueError(f"{name} must be a matching boolean matrix")
    return np.asarray(available, dtype=bool)


def _dstar_seeds(value: object, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise ValueError("Dstar seeds must contain real numeric values")
    seeds = np.asarray(raw, dtype=np.float64)
    if seeds.ndim != 3 or seeds.shape[1:] != shape:
        raise ValueError("Dstar seeds must have shape (seed, query, candidate)")
    if seeds.shape[0] != 3:
        raise ValueError("Dstar seed scores must contain exactly three seeds")
    if np.isinf(seeds).any():
        raise ValueError("Dstar seeds must not contain infinity")
    seed_available = np.isfinite(seeds[0])
    if any(
        not np.array_equal(np.isfinite(seed), seed_available) for seed in seeds[1:]
    ):
        raise ValueError("Dstar seed finite masks must match")
    return seeds, seed_available


def _candidate_hashes(value: Sequence[str], count: int) -> tuple[str, ...]:
    try:
        hashes = tuple(value)
    except TypeError as error:
        raise ValueError("candidate hashes must be aligned unique nonempty strings") from error
    if (
        len(hashes) != count
        or any(not isinstance(item, str) or not item for item in hashes)
        or len(set(hashes)) != count
    ):
        raise ValueError("candidate hashes must be aligned unique nonempty strings")
    return hashes


def build_bscd_residual_features(
    scores: object,
    dstar_seed_scores: object,
    s_available: object,
    c_available: object,
    response_supported: object,
    candidate_hashes: Sequence[str],
) -> np.ndarray:
    """Build a finite ``(query, candidate, feature)`` float32 feature tensor."""

    matrices = _score_matrices(scores)
    shape = matrices["B_SCD"].shape
    s_mask = _availability(s_available, shape, name="S availability")
    c_mask = _availability(c_available, shape, name="C availability")
    seeds, seed_available = _dstar_seeds(dstar_seed_scores, shape)
    hashes = _candidate_hashes(candidate_hashes, shape[1])
    support = _support_matrix(
        response_supported, query_count=shape[0], candidate_count=shape[1]
    )

    ranked_scores = dict(matrices)
    ranked_scores["S"] = np.where(s_mask, matrices["S"], np.nan)
    ranked_scores["C"] = np.where(c_mask, matrices["C"], np.nan)

    features = np.zeros(shape + (len(FEATURE_NAMES),), dtype=np.float32)
    for channel in CHANNELS:
        features[..., FEATURE_INDEX[f"{channel}_rank"]] = _rank_percentiles(
            ranked_scores[channel], hashes
        )
        features[..., FEATURE_INDEX[f"{channel}_score"]] = matrices[channel]
        features[..., FEATURE_INDEX[f"{channel}_top_margin"]] = np.broadcast_to(
            _top_margin(ranked_scores[channel])[:, None], shape
        )

    features[..., FEATURE_INDEX["S_available"]] = s_mask
    features[..., FEATURE_INDEX["C_available"]] = c_mask
    features[..., FEATURE_INDEX["response_supported"]] = support

    backbone = matrices["B_SCD"]
    for channel in CHANNELS[:-1]:
        features[..., FEATURE_INDEX[f"{channel}_minus_B_SCD"]] = (
            matrices[channel] - backbone
        )
        features[..., FEATURE_INDEX[f"{channel}_B_SCD_top10_overlap"]] = (
            np.broadcast_to(
                _top10_overlap(ranked_scores[channel], backbone, hashes)[:, None],
                shape,
            )
        )

    seed_rank_sum = np.zeros(shape, dtype=np.float64)
    seed_rank_square_sum = np.zeros(shape, dtype=np.float64)
    for seed in seeds:
        ranks = _rank_percentiles(seed, hashes)
        seed_rank_sum += ranks
        seed_rank_square_sum += ranks * ranks
    seed_rank_mean = seed_rank_sum / seeds.shape[0]
    seed_rank_variance = np.maximum(
        seed_rank_square_sum / seeds.shape[0] - seed_rank_mean * seed_rank_mean,
        0.0,
    )
    features[..., FEATURE_INDEX["Dstar_seed_rank_std"]] = np.sqrt(
        seed_rank_variance
    )
    features[..., FEATURE_INDEX["Dstar_seed_rank_std"]][~seed_available] = 0.0

    for channel, available in (("S", s_mask), ("C", c_mask)):
        dependent = (
            f"{channel}_rank",
            f"{channel}_score",
            f"{channel}_minus_B_SCD",
            f"{channel}_top_margin",
            f"{channel}_B_SCD_top10_overlap",
        )
        for name in dependent:
            features[..., FEATURE_INDEX[name]][~available] = 0.0

    if not np.isfinite(features).all():
        raise ValueError("B_SCD residual features must be finite float32 values")
    return features


__all__ = [
    "CHANNELS",
    "FEATURE_INDEX",
    "FEATURE_NAMES",
    "build_bscd_residual_features",
    "eligible_residual_configs",
    "fuse_bscd_residual",
    "require_finite_matrix",
    "residual_magnitude_summary",
]
