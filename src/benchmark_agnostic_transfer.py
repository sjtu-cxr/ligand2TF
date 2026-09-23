"""Benchmark-agnostic availability derivation and transfer-first routing."""

from __future__ import annotations

import numpy as np

from src.unified_transfer_first_routing import transfer_first_scores


def _aligned_2d(values, message):
    arrays = [np.asarray(value) for value in values]
    if any(array.ndim != 2 for array in arrays):
        raise ValueError(message)
    if len({array.shape for array in arrays}) != 1:
        raise ValueError(message)
    return arrays


def _boolean_masks(values):
    masks = _aligned_2d(values, "availability must be aligned boolean matrices")
    if any(mask.dtype != np.bool_ for mask in masks):
        raise ValueError("availability must be aligned boolean matrices")
    return masks


def witness_availability(s_witness, c_witness, s_scores, c_scores):
    """Return channel availability from legal witnesses with finite scores."""
    arrays = _aligned_2d(
        (s_witness, c_witness, s_scores, c_scores),
        "witnesses and scores must be aligned nonempty matrices",
    )
    if arrays[0].size == 0:
        raise ValueError("witnesses and scores must be aligned nonempty matrices")

    s_mask, c_mask, s_values, c_values = arrays
    if s_mask.dtype != np.bool_ or c_mask.dtype != np.bool_:
        raise ValueError("witnesses must be boolean matrices")
    if any(not np.issubdtype(values.dtype, np.number) for values in (s_values, c_values)):
        raise ValueError("scores must be real matrices")
    if any(np.issubdtype(values.dtype, np.complexfloating) for values in (s_values, c_values)):
        raise ValueError("scores must be real matrices")
    if np.isinf(s_values).any() or np.isinf(c_values).any():
        raise ValueError("scores must not contain infinities")

    return s_mask & np.isfinite(s_values), c_mask & np.isfinite(c_values)


def transfer_first_auto(s, c, dstar, s_available, c_available, beta):
    """Route S/C/Dstar scores according to channel availability."""
    return transfer_first_scores(
        s, c, dstar, s_available, c_available, beta=beta
    )


def availability_state(s_available, c_available):
    """Encode availability as ``2 * S + C`` using int8 state codes."""
    s_mask, c_mask = _boolean_masks((s_available, c_available))
    return 2 * s_mask.astype(np.int8) + c_mask.astype(np.int8)


__all__ = ["availability_state", "transfer_first_auto", "witness_availability"]
