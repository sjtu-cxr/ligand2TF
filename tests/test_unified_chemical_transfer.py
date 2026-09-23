import numpy as np
import pandas as pd

from src.unified_chemical_transfer import (
    chemical_transfer_scores,
    comparison_deltas,
    screen_variants,
    select_configuration,
)


def _tables():
    return {
        "morgan_r2": {("q", "l1"): 0.8, ("q", "l2"): 0.3},
        "morgan_r3": {("q", "l1"): 0.4, ("q", "l2"): 0.6},
        "maccs": {("q", "l1"): 0.2, ("q", "l2"): 0.4},
        "rdkit": {("q", "l1"): 0.6, ("q", "l2"): 0.2},
    }


def test_direct_memory_takes_per_fingerprint_candidate_history_maximum():
    result = chemical_transfer_scores(
        query_ids=("q",),
        candidate_hashes=("p1", "p2", "p3"),
        fit_edges=(("l1", "p1"), ("l2", "p1"), ("l2", "p2")),
        similarity_tables=_tables(),
    )

    assert np.allclose(result.scores["morgan"], [[0.8, 0.3, 0.0]])
    assert np.allclose(result.scores["multifp"], [[0.6, 0.375, 0.0]])
    assert result.available.tolist() == [[True, True, False]]
    assert result.witness_counts.tolist() == [[2, 1, 0]]


def test_exact_query_ligand_is_not_its_own_transfer_witness():
    tables = {
        name: {("q", "q"): 1.0, ("q", "l1"): 0.4}
        for name in ("morgan_r2", "morgan_r3", "maccs", "rdkit")
    }
    result = chemical_transfer_scores(
        query_ids=("q",),
        candidate_hashes=("p1", "p2"),
        fit_edges=(("q", "p1"), ("l1", "p1"), ("q", "p2")),
        similarity_tables=tables,
    )

    assert np.allclose(result.scores["morgan"], [[0.4, 0.0]])
    assert result.available.tolist() == [[True, False]]
    assert result.witness_counts.tolist() == [[1, 0]]


def test_query_without_fingerprint_is_unavailable_not_zero_evidence():
    result = chemical_transfer_scores(
        query_ids=("missing",),
        candidate_hashes=("p1",),
        fit_edges=(("l1", "p1"),),
        similarity_tables=_tables(),
    )

    assert np.allclose(result.scores["morgan"], [[0.0]])
    assert np.allclose(result.scores["multifp"], [[0.0]])
    assert result.available.tolist() == [[False]]


def test_candidate_history_is_available_when_its_ligand_fingerprint_is_missing():
    result = chemical_transfer_scores(
        query_ids=("q",),
        candidate_hashes=("p1", "p2"),
        fit_edges=(("l1", "p1"), ("ion:unknown", "p2")),
        similarity_tables=_tables(),
    )

    assert result.available.tolist() == [[True, True]]
    assert result.witness_counts.tolist() == [[1, 0]]
    assert np.allclose(result.scores["morgan"], [[0.8, 0.0]])


def test_similarity_scores_preserve_float64_tie_precision():
    tables = {
        name: {("q", "l1"): 0.123456789123}
        for name in ("morgan_r2", "morgan_r3", "maccs", "rdkit")
    }
    result = chemical_transfer_scores(
        query_ids=("q",), candidate_hashes=("p1",),
        fit_edges=(("l1", "p1"),), similarity_tables=tables,
    )

    assert result.scores["morgan"].dtype == np.float64
    assert result.scores["morgan"][0, 0] == 0.123456789123


def test_select_configuration_prioritizes_h10_then_h50_then_mrr():
    rows = [
        {"beta": 0.5, "hit_at_10": 0.4, "hit_at_50": 0.8, "mrr": 0.3},
        {"beta": 0.75, "hit_at_10": 0.5, "hit_at_50": 0.6, "mrr": 0.4},
        {"beta": 1.0, "hit_at_10": 0.5, "hit_at_50": 0.7, "mrr": 0.2},
    ]

    assert select_configuration(rows)["beta"] == 1.0


def test_screen_variants_requires_one_competitive_split_without_other_collapse():
    deltas = [
        {"variant": "morgan", "split": "Random-edge", "delta_hit_at_10": 0.0, "delta_hit_at_50": 0.0},
        {"variant": "morgan", "split": "Ligand-Morgan-0.5", "delta_hit_at_10": -0.009, "delta_hit_at_50": -0.019},
        {"variant": "multifp", "split": "Random-edge", "delta_hit_at_10": 0.01, "delta_hit_at_50": 0.01},
        {"variant": "multifp", "split": "Ligand-Morgan-0.5", "delta_hit_at_10": -0.02, "delta_hit_at_50": 0.0},
    ]

    assert screen_variants(deltas) == ("morgan",)


def test_comparison_deltas_handles_metric_names_with_at_signs():
    summary = pd.DataFrame([
        {"split": "Random-edge", "variant": "current_backbone", "hit@10": 0.4, "hit@50": 0.5, "mrr": 0.2},
        {"split": "Random-edge", "variant": "morgan", "hit@10": 0.42, "hit@50": 0.49, "mrr": 0.21},
    ])

    assert comparison_deltas(summary) == [{
        "split": "Random-edge", "variant": "morgan",
        "delta_hit_at_10": 0.02, "delta_hit_at_50": -0.01,
        "delta_mrr": 0.01,
    }]
