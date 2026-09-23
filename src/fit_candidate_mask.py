"""Fit-consistent candidate masking and deterministic ranking metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math

import numpy as np


def build_exclusion_mask(
    candidate_hashes: Sequence[str],
    *,
    train_responders: Iterable[str],
    validation_responders: Iterable[str],
    test_responders: Iterable[str],
) -> np.ndarray:
    """Exclude documented fit responders while protecting outer-test positives."""

    candidates = tuple(map(str, candidate_hashes))
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("candidate hashes must be unique and nonempty")
    excluded = set(map(str, train_responders)) | set(
        map(str, validation_responders)
    )
    relevant = set(map(str, test_responders))
    if excluded & relevant:
        raise ValueError("test responder appears in fit exclusion set")
    missing = excluded.difference(candidates)
    if missing:
        raise ValueError(
            f"fit responder is absent from candidate library: {sorted(missing)[:3]}"
        )
    return np.asarray([value in excluded for value in candidates], dtype=bool)


def metric_row(
    scores: Sequence[float] | np.ndarray,
    candidate_hashes: Sequence[str],
    relevant_hashes: Iterable[str],
    exclusion_mask: Sequence[bool] | np.ndarray,
) -> dict[str, float | int]:
    """Calculate manuscript retrieval metrics under one candidate mask."""

    values = np.asarray(scores, dtype=np.float64)
    candidates = tuple(map(str, candidate_hashes))
    excluded = np.asarray(exclusion_mask, dtype=bool)
    if values.shape != (len(candidates),) or excluded.shape != values.shape:
        raise ValueError("scores, candidates, and mask are misaligned")
    relevant = set(map(str, relevant_hashes))
    if not relevant:
        raise ValueError("query lacks a test responder")
    excluded_hashes = {candidates[i] for i in np.flatnonzero(excluded)}
    if relevant & excluded_hashes:
        raise ValueError("test responder is excluded")
    missing = relevant.difference(candidates)
    if missing:
        raise ValueError(f"test responder is absent from candidate library: {sorted(missing)[:3]}")

    sortable = np.nan_to_num(values, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    order = np.lexsort((np.asarray(candidates), -sortable))
    active = order[~excluded[order]]
    positions = np.asarray(
        [
            index + 1
            for index, candidate_index in enumerate(active)
            if candidates[candidate_index] in relevant
        ],
        dtype=int,
    )
    if not positions.size:
        raise ValueError("query has no active test responder")
    n_positive = len(relevant)
    row: dict[str, float | int] = {
        "n_candidates": int(active.size),
        "n_positives": n_positive,
        "best_positive_rank": int(positions.min()),
        "mrr": float(1.0 / positions.min()),
    }
    for k in (1, 10, 50):
        hits = int(np.count_nonzero(positions <= k))
        row[f"hit@{k}"] = float(hits > 0)
        row[f"known_positive_coverage@{k}"] = float(hits / n_positive)
    recall_budget = max(1, math.ceil(0.01 * active.size))
    row["recall@1%"] = float(
        np.count_nonzero(positions <= recall_budget) / n_positive
    )
    top = positions[positions <= 10]
    dcg = float(np.sum(1.0 / np.log2(top + 1)))
    ideal = np.arange(1, min(n_positive, 10) + 1)
    row["ndcg@10"] = float(dcg / np.sum(1.0 / np.log2(ideal + 1)))
    return row
