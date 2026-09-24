"""Portable gate training primitives extracted without numerical changes.

This module accepts already prepared fold-local arrays. It does not construct
benchmark evidence or replace the validation/refit protocol.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
import math
import os
import numpy as np
import pandas as pd
import torch
from src.grouped_validation import atomic_split_units, grouped_meta_fold
from src.candidate_gated_top10 import CandidateGatedCorrection, gated_correction_regularization, weighted_pairwise_loss
from src.transfer_distilled_residual import sample_candidate_mask
from src.unified_bscd_residual import FEATURE_NAMES
from src.unified_dstar_ensemble import SUPPORTED_SEEDS, rank_descending

TRAINING_SEEDS = tuple(SUPPORTED_SEEDS)


META_FOLDS = 3


META_SEED = 20260726


TRAINING_EPOCHS = 30


LEARNING_RATE = 1e-2


DETERMINISM_POLICY = {
    "cublas_workspace_config": ":4096:8",
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "deterministic_algorithms": True,
    "dtype": "float32",
    "mixed_precision": False,
}


def configure_torch_determinism() -> None:
    """Apply the frozen deterministic float32 execution policy."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISM_POLICY[
        "cublas_workspace_config"
    ]
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def resolve_training_device(value: str | torch.device) -> torch.device:
    """Resolve auto/cpu/cuda and fail closed when CUDA was explicitly requested."""

    requested = value.type if isinstance(value, torch.device) else value
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _mean_rank_predictions(
    predictions: Sequence[np.ndarray], candidate_hashes: Sequence[str]
) -> np.ndarray:
    if len(predictions) != 3:
        raise ValueError("mean-rank aggregation requires exactly three training seeds")
    arrays = [np.asarray(value, dtype=np.float64) for value in predictions]
    if not arrays or arrays[0].ndim != 2 or any(
        value.shape != arrays[0].shape for value in arrays
    ):
        raise ValueError("seed prediction matrices must have one common 2D shape")
    ranks = [rank_descending(value, candidate_hashes) for value in arrays]
    if any(not np.isfinite(value).all() for value in ranks):
        raise ValueError("OOF predictions must rank every candidate")
    return -np.mean(np.stack(ranks, axis=0), axis=0)


def grouped_oof_predictions(
    *,
    query_ids: Sequence[str],
    split_units: Sequence[str],
    candidate_hashes: Sequence[str],
    training_seeds: Sequence[int],
    train_predict: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate three-fold held-out predictions and a per-query exclusion proof."""

    query_ids = tuple(query_ids)
    split_units = tuple(split_units)
    candidate_hashes = tuple(candidate_hashes)
    seeds = tuple(int(seed) for seed in training_seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("grouped OOF requires exactly three distinct training seeds")
    if len(query_ids) != len(split_units) or not query_ids or not candidate_hashes:
        raise ValueError("query IDs, split units, and candidates must be nonempty and aligned")
    assignments = grouped_meta_fold(split_units, folds=META_FOLDS, seed=META_SEED)
    output = np.empty((len(query_ids), len(candidate_hashes)), dtype=np.float64)
    predicted = np.zeros(len(query_ids), dtype=bool)
    proof: list[dict[str, Any]] = []
    for meta_fold in range(META_FOLDS):
        holdout_indexes = np.flatnonzero(assignments == meta_fold)
        train_indexes = np.flatnonzero(assignments != meta_fold)
        if not len(holdout_indexes) or not len(train_indexes):
            raise ValueError("each grouped meta fold must have train and holdout queries")
        training_units = sorted({split_units[int(index)] for index in train_indexes})
        training_atoms = sorted(
            {
                atom
                for index in train_indexes
                for atom in atomic_split_units(split_units[int(index)])
            }
        )
        seed_predictions = [
            np.asarray(train_predict(train_indexes, holdout_indexes, seed))
            for seed in seeds
        ]
        expected_shape = (len(holdout_indexes), len(candidate_hashes))
        if any(value.shape != expected_shape for value in seed_predictions):
            raise ValueError("train_predict returned an invalid held-out score shape")
        output[holdout_indexes] = _mean_rank_predictions(
            seed_predictions, candidate_hashes
        )
        predicted[holdout_indexes] = True
        for index in holdout_indexes:
            unit = split_units[int(index)]
            holdout_atoms = sorted(atomic_split_units(unit))
            if set(holdout_atoms) & set(training_atoms):
                raise ValueError("grouped meta fold contains train/holdout unit leakage")
            proof.append(
                {
                    "query_id": query_ids[int(index)],
                    "split_unit": unit,
                    "meta_fold": meta_fold,
                    "training_split_units": training_units,
                    "holdout_atomic_split_units": holdout_atoms,
                    "training_atomic_split_units": training_atoms,
                    "training_seeds": list(seeds),
                }
            )
    if not np.all(predicted) or not np.isfinite(output).all():
        raise ValueError("every validation query must receive one finite OOF prediction")
    proof.sort(key=lambda row: query_ids.index(str(row["query_id"])))
    return output, proof


@dataclass(frozen=True)
class CandidateGateConfig:
    gamma: float
    lambda_corr: float
    width: int = 32
    lambda_gate: float = 0.001

    def __post_init__(self) -> None:
        if self.gamma not in {0.005, 0.01}:
            raise ValueError("gamma is outside the frozen grid")
        if self.lambda_corr not in {0.01, 0.1}:
            raise ValueError("lambda_corr is outside the frozen grid")
        if self.width != 32 or self.lambda_gate != 0.001:
            raise ValueError("width and gate penalty are frozen")

    @property
    def config_id(self) -> str:
        return f"gamma={self.gamma:g};lambda_corr={self.lambda_corr:g};lambda_gate={self.lambda_gate:g};width={self.width}"


@dataclass(frozen=True)
class RankingMetrics:
    query_count: int
    h10: float
    h50: float
    mrr: float

    def __post_init__(self) -> None:
        if isinstance(self.query_count, bool) or self.query_count <= 0:
            raise ValueError("query_count must be positive")
        object.__setattr__(self, "query_count", int(self.query_count))
        for name in ("h10", "h50", "mrr"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, value)


def frozen_candidate_gate_grid() -> tuple[CandidateGateConfig, ...]:
    return tuple(
        CandidateGateConfig(gamma, lambda_corr)
        for gamma in (0.005, 0.01)
        for lambda_corr in (0.01, 0.1)
    )


def _model_complexity(config: CandidateGateConfig) -> int:
    return config.width * (32 + 1) + 2 * (config.width + 1)


def select_candidate_gate_config(
    candidates: Mapping[CandidateGateConfig, RankingMetrics],
    backbone: RankingMetrics,
    *,
    mrr_tolerance: float = 0.002,
) -> tuple[CandidateGateConfig | None, RankingMetrics]:
    """Select H@10 first under fixed backbone H@50/MRR constraints."""

    if mrr_tolerance < 0.0:
        raise ValueError("MRR tolerance must be nonnegative")
    feasible = [
        (config, metrics)
        for config, metrics in candidates.items()
        if metrics.query_count == backbone.query_count
        and metrics.h50 + 1e-12 >= backbone.h50
        and metrics.mrr + 1e-12 >= backbone.mrr - mrr_tolerance
    ]
    if not feasible:
        return None, backbone
    return max(
        feasible,
        key=lambda value: (
            value[1].h10,
            value[1].mrr,
            value[1].h50,
            -_model_complexity(value[0]),
            value[0].config_id,
        ),
    )


def active_backbone_rank_data(
    backbone: np.ndarray,
    eligible_mask: np.ndarray,
    candidate_hashes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return active-library rank percentiles and one-based ordinal ranks."""

    scores = np.asarray(backbone, dtype=np.float64)
    eligible = np.asarray(eligible_mask, dtype=bool)
    candidates = tuple(map(str, candidate_hashes))
    if scores.ndim != 2 or scores.shape != eligible.shape:
        raise ValueError("backbone and eligibility matrices must align")
    if len(candidates) != scores.shape[1] or len(set(candidates)) != len(candidates):
        raise ValueError("candidate hashes must align and be unique")
    if not np.isfinite(scores).all() or not np.all(eligible.any(axis=1)):
        raise ValueError("each query needs finite scores and an active candidate")
    percentiles = np.full(scores.shape, -1e9, dtype=np.float64)
    ranks = np.full(scores.shape, scores.shape[1] + 1.0, dtype=np.float64)
    for row in range(scores.shape[0]):
        active = np.flatnonzero(eligible[row])
        active_hashes = tuple(candidates[index] for index in active)
        active_ranks = rank_descending(scores[row, active][None, :], active_hashes)[0]
        ranks[row, active] = active_ranks
        percentiles[row, active] = 1.0 - (active_ranks - 1.0) / max(len(active) - 1, 1)
    return percentiles, ranks


def mean_rank_ensemble(
    predictions: Sequence[np.ndarray],
    candidate_hashes: Sequence[str],
    eligible_mask: np.ndarray,
) -> np.ndarray:
    """Combine seed predictions by active-library mean rank percentile."""

    if not predictions:
        raise ValueError("at least one prediction matrix is required")
    eligible = np.asarray(eligible_mask, dtype=bool)
    total = np.zeros(eligible.shape, dtype=np.float64)
    for prediction in predictions:
        percentiles, _ = active_backbone_rank_data(
            prediction, eligible, candidate_hashes
        )
        total += percentiles
    result = total / len(predictions)
    result[~eligible] = -1e9
    return result


def ranking_metrics(
    scores: np.ndarray,
    *,
    candidate_hashes: Sequence[str],
    relevant_hashes: Sequence[Sequence[str]],
    eligible_mask: np.ndarray,
    query_ids: Sequence[str],
) -> tuple[RankingMetrics, pd.DataFrame]:
    """Compute exact active-library query metrics and per-query ranks."""

    matrix = np.asarray(scores, dtype=np.float64)
    eligible = np.asarray(eligible_mask, dtype=bool)
    candidates = tuple(map(str, candidate_hashes))
    relevant = tuple(tuple(map(str, values)) for values in relevant_hashes)
    queries = tuple(map(str, query_ids))
    if matrix.shape != eligible.shape or matrix.shape != (len(queries), len(candidates)):
        raise ValueError("ranking inputs must align")
    if len(relevant) != len(queries) or not np.isfinite(matrix).all():
        raise ValueError("ranking scores and relevant items must be finite and aligned")
    candidate_index = {value: index for index, value in enumerate(candidates)}
    rows = []
    for row, query in enumerate(queries):
        active = np.flatnonzero(eligible[row])
        active_hashes = tuple(candidates[index] for index in active)
        ranks = rank_descending(matrix[row, active][None, :], active_hashes)[0]
        lookup = dict(zip(active.tolist(), ranks.tolist(), strict=True))
        positives = [
            candidate_index[value]
            for value in relevant[row]
            if value in candidate_index and eligible[row, candidate_index[value]]
        ]
        if not positives:
            raise ValueError("each query must contain an active relevant candidate")
        best = float(min(lookup[index] for index in positives))
        rows.append({
            "query_id": query,
            "best_positive_rank": best,
            "H10": float(best <= 10),
            "H50": float(best <= 50),
            "MRR": 1.0 / best,
        })
    table = pd.DataFrame(rows)
    metrics = RankingMetrics(
        len(table), float(table.H10.mean()), float(table.H50.mean()), float(table.MRR.mean())
    )
    return metrics, table


def _validate_training_arrays(arrays: Mapping[str, np.ndarray]) -> tuple[int, int]:
    features = np.asarray(arrays["features"])
    backbone = np.asarray(arrays["backbone"])
    labels = np.asarray(arrays["labels"])
    eligible = np.asarray(arrays["eligible_mask"])
    if features.ndim != 3 or features.shape[-1] != len(FEATURE_NAMES):
        raise ValueError("features must follow the frozen 32-feature contract")
    if not (
        backbone.shape == labels.shape == eligible.shape == features.shape[:2]
        and labels.dtype == np.bool_
        and eligible.dtype == np.bool_
        and np.isfinite(features).all()
        and np.isfinite(backbone).all()
    ):
        raise ValueError("training arrays must be finite and aligned")
    return backbone.shape


def _validate_prediction_arrays(arrays: Mapping[str, np.ndarray]) -> tuple[int, int]:
    features = np.asarray(arrays["features"])
    backbone = np.asarray(arrays["backbone"])
    eligible = np.asarray(arrays["eligible_mask"])
    if (
        features.ndim != 3
        or features.shape[-1] != len(FEATURE_NAMES)
        or backbone.shape != eligible.shape
        or backbone.shape != features.shape[:2]
        or eligible.dtype != np.bool_
        or not np.isfinite(features).all()
        or not np.isfinite(backbone).all()
    ):
        raise ValueError("prediction arrays must follow the finite 32-feature contract")
    return backbone.shape


def fit_candidate_gate(
    arrays: Mapping[str, np.ndarray],
    *,
    indexes: np.ndarray,
    query_ids: Sequence[str],
    candidate_hashes: Sequence[str],
    config: CandidateGateConfig,
    training_seed: int,
    epochs: int,
    device: torch.device,
) -> CandidateGatedCorrection:
    """Fit one candidate-gated correction on fold-local query indexes."""

    query_count, candidate_count = _validate_training_arrays(arrays)
    selected_indexes = np.asarray(indexes, dtype=np.int64)
    if selected_indexes.ndim != 1 or not len(selected_indexes) or np.any(
        (selected_indexes < 0) | (selected_indexes >= query_count)
    ):
        raise ValueError("training indexes must be a nonempty aligned vector")
    if len(query_ids) != query_count or len(candidate_hashes) != candidate_count:
        raise ValueError("query and candidate identities must align")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    configure_torch_determinism()
    torch.manual_seed(int(training_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(training_seed))
    model = CandidateGatedCorrection(
        feature_count=len(FEATURE_NAMES), width=config.width, gamma=config.gamma
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    labels_cpu = torch.as_tensor(arrays["labels"][selected_indexes], dtype=torch.bool)
    backbone_cpu = torch.as_tensor(
        arrays["backbone"][selected_indexes], dtype=torch.float32
    )
    selected = sample_candidate_mask(
        labels_cpu,
        backbone_cpu,
        backbone_cpu,
        [query_ids[int(index)] for index in selected_indexes],
        candidate_hashes,
        seed=int(training_seed),
    )
    eligible_cpu = torch.as_tensor(
        arrays["eligible_mask"][selected_indexes], dtype=torch.bool
    )
    selected &= eligible_cpu
    if not torch.all(torch.any(selected & labels_cpu, dim=1)):
        raise ValueError("eligible candidate mask removed a training positive")
    row_cpu, column_cpu = selected.nonzero(as_tuple=True)
    global_rows = selected_indexes[np.asarray(row_cpu, dtype=np.int64)]
    columns = np.asarray(column_cpu, dtype=np.int64)
    base_percentile, base_ranks = active_backbone_rank_data(
        arrays["backbone"], arrays["eligible_mask"], candidate_hashes
    )
    sampled_features = torch.as_tensor(
        arrays["features"][global_rows, columns], dtype=torch.float32, device=device
    )
    sampled_base = torch.as_tensor(
        base_percentile[global_rows, columns], dtype=torch.float32, device=device
    )
    sampled_ranks = torch.as_tensor(
        base_ranks[global_rows, columns], dtype=torch.float32, device=device
    )
    sampled_labels = labels_cpu[row_cpu, column_cpu].to(device)
    row = row_cpu.to(device)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        scores, gate, delta = model(sampled_features, sampled_base)
        loss = weighted_pairwise_loss(scores, sampled_labels, row, sampled_ranks)
        loss = loss + gated_correction_regularization(
            gate,
            delta,
            lambda_corr=config.lambda_corr,
            lambda_gate=config.lambda_gate,
        )
        if not torch.isfinite(loss):
            raise ValueError("candidate-gated training produced nonfinite loss")
        loss.backward()
        optimizer.step()
    return model


def predict_candidate_gate(
    model: CandidateGatedCorrection,
    arrays: Mapping[str, np.ndarray],
    *,
    indexes: np.ndarray,
    candidate_hashes: Sequence[str],
    device: torch.device,
) -> np.ndarray:
    """Score complete candidate libraries for the requested query indexes."""

    query_count, _ = _validate_prediction_arrays(arrays)
    selected = np.asarray(indexes, dtype=np.int64)
    if selected.ndim != 1 or np.any((selected < 0) | (selected >= query_count)):
        raise ValueError("prediction indexes must align")
    base, _ = active_backbone_rank_data(
        arrays["backbone"][selected],
        arrays["eligible_mask"][selected],
        candidate_hashes,
    )
    features = torch.as_tensor(
        arrays["features"][selected], dtype=torch.float32, device=device
    )
    base_tensor = torch.as_tensor(base, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        scores, _, _ = model(features, base_tensor)
    result = np.asarray(scores.cpu(), dtype=np.float64)
    result[~np.asarray(arrays["eligible_mask"][selected], dtype=bool)] = -1e9
    if not np.isfinite(result).all():
        raise ValueError("candidate-gated predictions must be finite")
    return result
