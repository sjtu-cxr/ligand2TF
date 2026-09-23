import inspect

import numpy as np
import torch

from src.gate_training import (
    CandidateGateConfig,
    RankingMetrics,
    active_backbone_rank_data,
    fit_candidate_gate,
    frozen_candidate_gate_grid,
    mean_rank_ensemble,
    predict_candidate_gate,
    ranking_metrics,
    select_candidate_gate_config,
)


def test_frozen_candidate_gate_grid_has_exact_four_configs():
    grid = frozen_candidate_gate_grid()

    assert len(grid) == 4
    assert {(value.gamma, value.lambda_corr) for value in grid} == {
        (0.005, 0.01),
        (0.005, 0.1),
        (0.01, 0.01),
        (0.01, 0.1),
    }
    assert {value.width for value in grid} == {32}
    assert {value.lambda_gate for value in grid} == {0.001}


def test_selector_prioritizes_h10_subject_to_h50_and_mrr_floors():
    baseline = RankingMetrics(100, 0.20, 0.40, 0.10)
    high_h10_unsafe = CandidateGateConfig(0.005, 0.01)
    lower_h10_safe = CandidateGateConfig(0.01, 0.1)
    candidates = {
        high_h10_unsafe: RankingMetrics(100, 0.30, 0.39, 0.20),
        lower_h10_safe: RankingMetrics(100, 0.25, 0.40, 0.098),
    }

    selected, metrics = select_candidate_gate_config(candidates, baseline)

    assert selected == lower_h10_safe
    assert metrics == candidates[lower_h10_safe]


def test_selector_uses_h10_before_secondary_metrics():
    baseline = RankingMetrics(100, 0.20, 0.40, 0.10)
    high_mrr = CandidateGateConfig(0.005, 0.01)
    high_h10 = CandidateGateConfig(0.01, 0.1)
    candidates = {
        high_mrr: RankingMetrics(100, 0.24, 0.60, 0.30),
        high_h10: RankingMetrics(100, 0.25, 0.40, 0.098),
    }

    selected, _ = select_candidate_gate_config(candidates, baseline)

    assert selected == high_h10


def test_selector_returns_backbone_when_no_candidate_is_feasible():
    baseline = RankingMetrics(100, 0.20, 0.40, 0.10)
    config = CandidateGateConfig(0.005, 0.01)

    selected, metrics = select_candidate_gate_config(
        {config: RankingMetrics(100, 0.30, 0.39, 0.20)}, baseline
    )

    assert selected is None
    assert metrics == baseline


def test_selection_api_has_no_split_dataset_or_test_label_input():
    parameters = inspect.signature(select_candidate_gate_config).parameters
    assert "split" not in parameters
    assert "dataset" not in parameters
    assert "test_labels" not in parameters


def _synthetic_arrays():
    rng = np.random.default_rng(9)
    backbone = np.array([
        [0.9, 0.8, 0.7, 0.6, 0.5, -1e9],
        [0.5, 0.4, 0.9, 0.8, 0.7, 0.6],
    ])
    eligible = np.ones((2, 6), dtype=bool)
    eligible[0, 5] = False
    labels = np.zeros((2, 6), dtype=bool)
    labels[0, 2] = True
    labels[1, 4] = True
    return {
        "features": rng.normal(size=(2, 6, 32)).astype(np.float32),
        "backbone": backbone,
        "eligible_mask": eligible,
        "labels": labels,
    }


def test_active_backbone_ranks_respect_eligibility_and_hash_ties():
    arrays = _synthetic_arrays()
    percentile, ranks = active_backbone_rank_data(
        arrays["backbone"], arrays["eligible_mask"], tuple(f"p{i}" for i in range(6))
    )

    assert percentile.shape == ranks.shape == (2, 6)
    assert ranks[0, 0] == 1.0 and ranks[0, 4] == 5.0
    assert percentile[0, 0] == 1.0 and percentile[0, 4] == 0.0
    assert percentile[0, 5] < -1e8


def test_training_and_prediction_produce_finite_full_candidate_scores():
    arrays = _synthetic_arrays()
    candidates = tuple(f"p{i}" for i in range(6))
    queries = ("q0", "q1")
    model = fit_candidate_gate(
        arrays,
        indexes=np.array([0, 1]),
        query_ids=queries,
        candidate_hashes=candidates,
        config=CandidateGateConfig(0.005, 0.01),
        training_seed=42,
        epochs=1,
        device=torch.device("cpu"),
    )

    scores = predict_candidate_gate(
        model,
        arrays,
        indexes=np.array([0, 1]),
        candidate_hashes=candidates,
        device=torch.device("cpu"),
    )

    assert scores.shape == (2, 6)
    assert np.isfinite(scores).all()
    assert np.all(scores[~arrays["eligible_mask"]] < -1e8)


def test_metrics_and_ensemble_use_only_active_candidate_library():
    candidates = ("a", "b", "c", "d")
    eligible = np.array([[True, True, False, True]])
    first = np.array([[4.0, 3.0, 100.0, 2.0]])
    second = np.array([[2.0, 4.0, 100.0, 3.0]])

    combined = mean_rank_ensemble([first, second], candidates, eligible)
    metrics, rows = ranking_metrics(
        combined,
        candidate_hashes=candidates,
        relevant_hashes=(("a",),),
        eligible_mask=eligible,
        query_ids=("q",),
    )

    assert combined[0, 2] < -1e8
    assert metrics.query_count == 1
    assert rows.iloc[0].best_positive_rank in {1.0, 2.0}
