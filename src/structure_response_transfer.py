"""Structure-similarity response-transfer primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Sequence

import numpy as np

from src.unified_dstar_ensemble import rank_descending


def _validated_identities(
    values: Sequence[str], *, label: str
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be unique nonempty strings")
    try:
        identities = tuple(values)
    except TypeError as error:
        raise ValueError(f"{label} must be unique nonempty strings") from error
    if (
        any(not isinstance(value, str) or not value for value in identities)
        or len(set(identities)) != len(identities)
    ):
        raise ValueError(f"{label} must be unique nonempty strings")
    return identities


@dataclass(frozen=True, eq=False)
class CompactStructureSimilarity:
    """Dense candidate-by-responder structure-similarity cache."""

    candidate_hashes: tuple[str, ...]
    responder_hashes: tuple[str, ...]
    scores: np.ndarray
    _responder_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        candidates = _validated_identities(
            self.candidate_hashes, label="candidate hashes"
        )
        responders = _validated_identities(
            self.responder_hashes, label="responder hashes"
        )
        matrix = np.asarray(self.scores)
        expected = (len(candidates), len(responders))
        if matrix.dtype != np.float32 or matrix.shape != expected:
            raise ValueError(
                "structure scores must be an aligned float32 matrix"
            )
        if not np.isfinite(matrix).all() or np.any(
            (matrix < 0.0) | (matrix > 1.0)
        ):
            raise ValueError("structure scores must be finite in [0, 1]")
        matrix = np.array(matrix, dtype=np.float32, order="C", copy=True)
        matrix.setflags(write=False)

        object.__setattr__(self, "candidate_hashes", candidates)
        object.__setattr__(self, "responder_hashes", responders)
        object.__setattr__(self, "scores", matrix)
        object.__setattr__(
            self,
            "_responder_index",
            {value: index for index, value in enumerate(responders)},
        )

    def nearest(self, responders: Sequence[str]) -> np.ndarray:
        """Return each candidate's maximum score to selected responders."""

        if isinstance(responders, (str, bytes)):
            values = (responders,)
        else:
            try:
                values = tuple(responders)
            except TypeError as error:
                raise ValueError("responders must be a collection") from error
        unknown = [
            value
            for value in values
            if not isinstance(value, str)
            or value not in self._responder_index
        ]
        if unknown:
            raise ValueError(f"unknown responder: {unknown[0]}")
        if not values:
            return np.zeros(len(self.candidate_hashes), dtype=np.float32)
        columns = [self._responder_index[value] for value in values]
        return np.max(self.scores[:, columns], axis=1).astype(
            np.float32, copy=False
        )


def active_rank_percentile(
    scores: Sequence[float] | np.ndarray,
    active_mask: Sequence[bool] | np.ndarray,
    candidate_hashes: Sequence[str],
) -> np.ndarray:
    """Calibrate finite scores to descending percentiles over active items."""

    raw_values = np.asarray(scores)
    active = np.asarray(active_mask)
    hashes = _validated_identities(candidate_hashes, label="candidate hashes")
    if (
        raw_values.ndim != 1
        or active.ndim != 1
        or raw_values.shape != active.shape
        or raw_values.shape != (len(hashes),)
    ):
        raise ValueError("ranking inputs must align as one-dimensional vectors")
    if active.dtype != np.bool_:
        raise ValueError("active mask must be boolean")
    if (
        not np.issubdtype(raw_values.dtype, np.number)
        or np.issubdtype(raw_values.dtype, np.complexfloating)
    ):
        raise ValueError("scores must contain finite real values")
    values = raw_values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("scores must contain finite real values")

    indexes = np.flatnonzero(active)
    if indexes.size == 0:
        raise ValueError("at least one active candidate is required")
    active_hashes = tuple(hashes[index] for index in indexes)
    ranks = rank_descending(values[indexes][None, :], active_hashes)[0]
    result = np.zeros(values.shape, dtype=np.float64)
    result[indexes] = 1.0 - (ranks - 1.0) / max(indexes.size - 1, 1)
    return result


def combine_transfer_scores(
    sequence: Sequence[float] | np.ndarray,
    structure: Sequence[float] | np.ndarray,
    *,
    active_mask: Sequence[bool] | np.ndarray,
    candidate_hashes: Sequence[str],
    structure_weight: float,
) -> np.ndarray:
    """Fuse active-library sequence and structure rank percentiles."""

    if (
        isinstance(structure_weight, bool)
        or not isinstance(structure_weight, Real)
        or float(structure_weight) not in {0.25, 0.5, 0.75, 1.0}
    ):
        raise ValueError("structure weight is outside the frozen grid")
    sequence_percentile = active_rank_percentile(
        sequence, active_mask, candidate_hashes
    )
    structure_percentile = active_rank_percentile(
        structure, active_mask, candidate_hashes
    )
    return np.maximum(
        sequence_percentile,
        float(structure_weight) * structure_percentile,
    )
