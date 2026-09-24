"""Frozen contracts shared by the full-universe random-edge baselines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


EDGE_COUNT = 874
CANDIDATE_COUNT = 6457
FOLDS = tuple(range(5))
MODEL_SEEDS = (42, 20260717, 20260718)
RANDOM_SEEDS = tuple(range(2026072900, 2026073000))


def _candidate_tuple(candidate_hashes: Sequence[str]) -> tuple[str, ...]:
    values = tuple(candidate_hashes)
    if (
        not values
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError("candidate hashes must be unique nonempty strings")
    return values


def candidate_hash_tie_scores(candidate_hashes: Sequence[str]) -> np.ndarray:
    """Return scores whose descending order is ascending candidate hash."""

    values = _candidate_tuple(candidate_hashes)
    order = sorted(range(len(values)), key=lambda index: values[index])
    scores = np.empty(len(values), dtype=np.float64)
    scores[np.asarray(order, dtype=np.int64)] = np.arange(
        len(values), 0, -1, dtype=np.float64
    )
    return scores


def chemical_availability(canonical_smiles: object) -> bool:
    """Return whether a query has a nonempty molecular-structure key."""

    return isinstance(canonical_smiles, str) and bool(canonical_smiles.strip())


def availability_fallback(
    channel_scores: object,
    fallback_scores: object,
    available: object,
) -> np.ndarray:
    """Use one channel where available and an exact fallback elsewhere."""

    channel = np.asarray(channel_scores, dtype=np.float64)
    fallback = np.asarray(fallback_scores, dtype=np.float64)
    mask = np.asarray(available)
    if (
        channel.ndim != 2
        or channel.shape != fallback.shape
        or mask.shape != (channel.shape[0],)
        or mask.dtype != bool
    ):
        raise ValueError("channel, fallback, and availability shapes are invalid")
    if not np.isfinite(channel).all() or not np.isfinite(fallback).all():
        raise ValueError("channel and fallback scores must be finite")
    output = np.array(fallback, copy=True)
    output[mask] = channel[mask]
    return output
