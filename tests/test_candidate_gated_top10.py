import inspect

import torch

from src.candidate_gated_top10 import (
    CandidateGatedCorrection,
    gated_correction_regularization,
    topk_pair_weights,
    weighted_pairwise_loss,
)


def test_candidate_gate_bounds_correction_around_backbone():
    torch.manual_seed(3)
    model = CandidateGatedCorrection(feature_count=32, width=32, gamma=0.01)
    features = torch.randn(7, 32)
    backbone = torch.linspace(0.1, 0.9, 7)

    score, gate, delta = model(features, backbone)

    assert score.shape == gate.shape == delta.shape == backbone.shape
    assert torch.all((gate >= 0.0) & (gate <= 1.0))
    assert torch.all((delta >= -1.0) & (delta <= 1.0))
    assert torch.all(torch.abs(score - backbone) <= 0.01 + 1e-7)


def test_model_api_has_no_split_dataset_or_test_label_argument():
    parameters = inspect.signature(CandidateGatedCorrection.forward).parameters
    assert "split" not in parameters
    assert "dataset" not in parameters
    assert "test_labels" not in parameters


def test_topk_pair_weights_emphasize_top10_then_top50():
    ranks = torch.tensor([1.0, 10.0, 11.0, 50.0, 51.0, 500.0])

    weights = topk_pair_weights(ranks)

    torch.testing.assert_close(weights, torch.tensor([4.0, 4.0, 2.0, 2.0, 1.0, 1.0]))


def test_weighted_pairwise_loss_decreases_when_positives_outrank_negatives():
    labels = torch.tensor([True, False, False, True, False])
    queries = torch.tensor([0, 0, 0, 1, 1])
    ranks = torch.tensor([30.0, 5.0, 100.0, 20.0, 7.0])
    bad = torch.tensor([0.0, 1.0, 0.5, 0.0, 1.0])
    good = torch.tensor([2.0, 1.0, 0.5, 2.0, 1.0])

    bad_loss = weighted_pairwise_loss(bad, labels, queries, ranks)
    good_loss = weighted_pairwise_loss(good, labels, queries, ranks)

    assert torch.isfinite(bad_loss) and torch.isfinite(good_loss)
    assert good_loss < bad_loss


def test_regularization_is_finite_nonnegative_and_zero_without_gate_use():
    gate = torch.tensor([0.0, 0.5, 1.0])
    delta = torch.tensor([-1.0, 0.2, 0.4])

    value = gated_correction_regularization(
        gate, delta, lambda_corr=0.1, lambda_gate=0.001
    )
    zero = gated_correction_regularization(
        torch.zeros(3), delta, lambda_corr=0.1, lambda_gate=0.001
    )

    assert torch.isfinite(value) and value >= 0.0
    torch.testing.assert_close(zero, torch.tensor(0.0))
