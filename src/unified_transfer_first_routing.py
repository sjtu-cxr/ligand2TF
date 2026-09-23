"""Availability-aware S/C fusion followed by hard D* fallback."""

from __future__ import annotations

import numpy as np


BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def transfer_first_scores(s, c, dstar, s_available, c_available, *, beta):
    if float(beta) not in BETAS:
        raise ValueError("beta must belong to the frozen grid")
    matrices = [np.asarray(value, dtype=np.float64) for value in (s, c, dstar)]
    if any(value.ndim != 2 for value in matrices) or len({value.shape for value in matrices}) != 1:
        raise ValueError("S, C and Dstar must be aligned matrices")
    if any(not np.isfinite(value).all() for value in matrices):
        raise ValueError("calibrated scores must be finite")
    masks = [np.asarray(value) for value in (s_available, c_available)]
    if any(value.dtype != np.bool_ or value.shape != matrices[0].shape for value in masks):
        raise ValueError("availability must be aligned boolean matrices")
    s_score, c_score, anchor = matrices
    s_mask, c_mask = masks
    both = s_mask & c_mask
    only_s = s_mask & ~c_mask
    only_c = c_mask & ~s_mask
    fused = anchor.copy()
    fused[only_s] = s_score[only_s]
    fused[only_c] = c_score[only_c]
    fused[both] = float(beta) * s_score[both] + (1.0 - float(beta)) * c_score[both]
    return fused, s_mask | c_mask


__all__ = ["BETAS", "transfer_first_scores"]
