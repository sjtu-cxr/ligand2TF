from __future__ import annotations

import inspect
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import src.unified_dstar_training as training_module
from src.unified_dstar_data import EvaluationQuery, QueryEpisode
from src.unified_dstar_training import (
    CheckpointSelector,
    _stable_content_hash,
    _training_content,
    _validation_content,
    downweighted_multi_positive_listwise_loss,
    evaluate_full_library,
    train_fold,
)


class TinyFeatureStore:
    def __init__(self) -> None:
        self.protein_values = {
            "a": 0.0,
            "b": 0.0,
            "c": 0.0,
            "p1": 1.0,
            "p2": 2.0,
            "p3": 3.0,
            "p4": 4.0,
        }
        self.protein_batches: list[tuple[str, ...]] = []

    def ligand_batch(self, keys):
        count = len(keys)
        ones = torch.ones((count, 1), dtype=torch.float32)
        zeros = torch.zeros((count, 1), dtype=torch.float32)
        return ones, zeros, zeros, torch.zeros(count, dtype=torch.bool)

    def protein_batch(self, hashes):
        normalized = tuple(str(value) for value in hashes)
        self.protein_batches.append(normalized)
        return torch.tensor(
            [[self.protein_values[value]] for value in normalized],
            dtype=torch.float32,
        )

    def audit(self):
        return {
            "cache_paths": {"tiny": "/tmp/tiny.pkl"},
            "missing_ion_policy": "zero",
            "missing_key_counts": {},
        }


class TinyRanker(torch.nn.Module):
    def __init__(self, *, tied: bool = False) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.tied = tied
        self.score_shapes: list[tuple[int, int]] = []

    def encode_ligand(self, molformer, ecfp, ion_features, ion_mask):
        return molformer * self.scale

    def encode_protein(self, esm):
        if self.tied:
            return torch.zeros_like(esm)
        return esm

    def score(self, ligand_z, protein_z):
        self.score_shapes.append((ligand_z.shape[0], protein_z.shape[0]))
        return ligand_z @ protein_z.T


class RankedFeatureStore(TinyFeatureStore):
    def protein_batch(self, hashes):
        normalized = tuple(str(value) for value in hashes)
        self.protein_batches.append(normalized)
        return torch.tensor(
            [[float(value[1:])] for value in normalized], dtype=torch.float32
        )


def validation_metrics(h10: float, *, mrr: float | None = None):
    selected_mrr = h10 if mrr is None else mrr
    return {
        "H@1": h10,
        "H@10": h10,
        "H@50": h10,
        "MRR": selected_mrr,
        "nDCG@10": h10,
        "Recall@1%": h10,
        "Coverage@10": h10,
    }


def one_episode():
    return [
        QueryEpisode(
            ligand_key="ligand",
            positive_hashes=("p3",),
            unlabeled_hashes=("p1", "p2"),
        )
    ]


def one_validation_query():
    return [
        EvaluationQuery(
            ligand_key="ligand",
            train_positive_hashes=("p3",),
            relevant_hashes=("p2",),
            candidate_hashes=("p1", "p2"),
            split_unit="unit",
        )
    ]


def test_unlabeled_weight_changes_denominator_only():
    scores = torch.tensor([[2.0, 1.0, 0.0]])
    pos = torch.tensor([[True, False, False]])
    valid = torch.ones_like(pos)

    weak = downweighted_multi_positive_listwise_loss(
        scores, pos, valid, unlabeled_weight=0.1
    )
    full = downweighted_multi_positive_listwise_loss(
        scores, pos, valid, unlabeled_weight=1.0
    )
    expected = -(
        2.0 - torch.log(torch.exp(torch.tensor(2.0)) + 0.1 * torch.exp(torch.tensor(1.0)) + 0.1)
    )

    assert weak == pytest.approx(expected.item())
    assert weak < full


def test_loss_multi_positive_numerator_contains_every_positive():
    scores = torch.tensor([[2.0, 1.0, 0.0]])
    pos = torch.tensor([[True, True, False]])
    valid = torch.ones_like(pos)

    actual = downweighted_multi_positive_listwise_loss(
        scores, pos, valid, unlabeled_weight=0.1
    )
    numerator = torch.logsumexp(scores[:, :2], dim=1)
    denominator = torch.logsumexp(
        torch.tensor([[2.0, 1.0, math.log(0.1)]]), dim=1
    )

    assert actual == pytest.approx((-(numerator - denominator)).item())


@pytest.mark.parametrize("weight", [-0.01, 1.01, float("nan"), float("inf")])
def test_loss_rejects_invalid_unlabeled_weight(weight):
    scores = torch.zeros((1, 2))
    mask = torch.tensor([[True, False]])
    with pytest.raises(ValueError, match="unlabeled_weight"):
        downweighted_multi_positive_listwise_loss(
            scores, mask, torch.ones_like(mask), unlabeled_weight=weight
        )


def test_loss_ignores_rows_without_valid_positives_and_invalid_scores():
    scores = torch.tensor([[2.0, float("nan")], [4.0, 3.0]], requires_grad=True)
    pos = torch.tensor([[True, False], [False, True]])
    valid = torch.tensor([[True, False], [False, False]])

    loss = downweighted_multi_positive_listwise_loss(scores, pos, valid)

    assert loss.item() == pytest.approx(0.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_loss_rejects_nonfinite_valid_scores_and_bad_masks():
    scores = torch.tensor([[float("inf"), 0.0]])
    pos = torch.tensor([[True, False]])
    valid = torch.ones_like(pos)
    with pytest.raises(ValueError, match="finite"):
        downweighted_multi_positive_listwise_loss(scores, pos, valid)
    with pytest.raises(ValueError, match="shape"):
        downweighted_multi_positive_listwise_loss(
            torch.zeros((1, 2)), pos[:, :1], valid
        )


def test_checkpoint_selector_is_lexicographic_and_keeps_earlier_exact_tie():
    selector = CheckpointSelector()
    assert selector.update(
        epoch=5, h10=0.2, mrr=0.1, ndcg10=0.9, state={"weight": torch.tensor(1.0)}
    )
    assert selector.update(
        epoch=10, h10=0.2, mrr=0.2, ndcg10=0.1, state={"weight": torch.tensor(2.0)}
    )
    assert not selector.update(
        epoch=15, h10=0.2, mrr=0.2, ndcg10=0.1, state={"weight": torch.tensor(3.0)}
    )
    assert selector.update(
        epoch=20, h10=0.3, mrr=0.0, ndcg10=0.0, state={"weight": torch.tensor(4.0)}
    )

    assert selector.best_epoch == 20
    assert selector.best_metrics == {"H@10": 0.3, "MRR": 0.0, "nDCG@10": 0.0}
    assert selector.best_state["weight"].item() == 4.0


def test_checkpoint_selector_exact_tie_uses_earlier_epoch_regardless_of_call_order():
    selector = CheckpointSelector()
    selector.update(epoch=10, h10=0.2, mrr=0.1, ndcg10=0.3)

    assert selector.update(epoch=5, h10=0.2, mrr=0.1, ndcg10=0.3)
    assert selector.best_epoch == 5


def test_checkpoint_selector_uses_ndcg_after_h10_and_mrr_ties():
    selector = CheckpointSelector()
    selector.update(epoch=5, h10=0.2, mrr=0.1, ndcg10=0.3)

    assert selector.update(epoch=10, h10=0.2, mrr=0.1, ndcg10=0.4)
    assert selector.best_epoch == 10
    assert selector.best_metrics["nDCG@10"] == 0.4


def test_checkpoint_selector_snapshots_state_without_aliasing():
    weight = torch.tensor(1.0)
    state = {
        "weight": weight,
        "nested": {"bias": torch.tensor(2.0)},
    }
    selector = CheckpointSelector()
    selector.update(epoch=1, h10=0.2, mrr=0.1, ndcg10=0.3, state=state)

    weight.add_(10.0)
    state["nested"]["bias"].add_(10.0)

    assert selector.best_state["weight"].item() == 1.0
    assert selector.best_state["nested"]["bias"].item() == 2.0


@pytest.mark.parametrize(
    "changed",
    [
        QueryEpisode("other", ("p3",), ("p1", "p2")),
        QueryEpisode("ligand", ("p2",), ("p1", "p2")),
        QueryEpisode("ligand", ("p3",), ("p1", "p4")),
    ],
)
def test_training_content_hash_changes_when_any_episode_field_changes(changed):
    original = QueryEpisode("ligand", ("p3",), ("p1", "p2"))
    original_hash = _stable_content_hash(_training_content([original]))

    assert _stable_content_hash(_training_content([changed])) != original_hash


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ligand_key", "other"),
        ("relevant_hashes", ("p1",)),
        ("candidate_hashes", ("p1", "p2", "p4")),
        ("train_positive_hashes", ("p4",)),
        ("split_unit", "other-unit"),
    ],
)
def test_validation_content_hash_changes_when_any_query_field_changes(field, value):
    original = EvaluationQuery(
        ligand_key="ligand",
        train_positive_hashes=("p3",),
        relevant_hashes=("p2",),
        candidate_hashes=("p1", "p2"),
        split_unit="unit",
    )
    original_hash = _stable_content_hash(_validation_content([original]))

    changed = replace(original, **{field: value})
    assert _stable_content_hash(_validation_content([changed])) != original_hash


def test_content_hashes_are_independent_of_record_and_hash_order():
    episodes = [
        QueryEpisode("b", ("p3", "p2"), ("p4", "p1")),
        QueryEpisode("a", ("p1",), ("p3", "p2")),
    ]
    queries = [
        EvaluationQuery("b", ("p4",), ("p3", "p2"), ("p4", "p3", "p2"), "u2"),
        EvaluationQuery("a", (), ("p1",), ("p2", "p1"), "u1"),
    ]

    reordered_episodes = [
        replace(episodes[1], unlabeled_hashes=("p2", "p3")),
        replace(
            episodes[0],
            positive_hashes=("p2", "p3"),
            unlabeled_hashes=("p1", "p4"),
        ),
    ]
    reordered_queries = [
        replace(queries[1], candidate_hashes=("p1", "p2")),
        replace(
            queries[0],
            relevant_hashes=("p2", "p3"),
            candidate_hashes=("p2", "p3", "p4"),
        ),
    ]

    assert _stable_content_hash(_training_content(episodes)) == _stable_content_hash(
        _training_content(reordered_episodes)
    )
    assert _stable_content_hash(_validation_content(queries)) == _stable_content_hash(
        _validation_content(reordered_queries)
    )


def test_full_library_metrics_use_all_candidates_batches_and_md5_ties():
    store = TinyFeatureStore()
    query = EvaluationQuery(
        ligand_key="ligand",
        train_positive_hashes=(),
        relevant_hashes=("a", "c"),
        candidate_hashes=("c", "b", "a"),
        split_unit="unit",
    )

    aggregate, per_query = evaluate_full_library(
        TinyRanker(tied=True),
        [query],
        store,
        candidate_batch_size=2,
        device="cpu",
    )

    expected_ndcg = (1.0 + 1.0 / math.log2(4.0)) / (1.0 + 1.0 / math.log2(3.0))
    assert aggregate == pytest.approx(
        {
            "H@1": 1.0,
            "H@10": 1.0,
            "H@50": 1.0,
            "MRR": 1.0,
            "nDCG@10": expected_ndcg,
            "Recall@1%": 0.5,
            "Coverage@10": 1.0,
        }
    )
    assert per_query.loc[0, "positive_ranks"] == "1;3"
    assert per_query.loc[0, "n_candidates"] == 3
    assert store.protein_batches == [("c", "b"), ("a",)]


def test_full_library_macro_metrics_and_recall_one_percent_cutoff_for_101():
    candidates = tuple(f"h{index:03d}" for index in range(101))
    queries = [
        EvaluationQuery("q1", (), ("h099",), candidates, "u1"),
        EvaluationQuery("q2", (), ("h050",), candidates, "u2"),
    ]

    aggregate, per_query = evaluate_full_library(
        TinyRanker(),
        queries,
        RankedFeatureStore(),
        candidate_batch_size=17,
        device="cpu",
    )

    assert per_query["n_candidates"].tolist() == [101, 101]
    assert per_query["best_positive_rank"].tolist() == [2, 51]
    assert per_query["Recall@1%"].tolist() == [1.0, 0.0]
    assert aggregate["H@10"] == pytest.approx(0.5)
    assert aggregate["H@50"] == pytest.approx(0.5)
    assert aggregate["Recall@1%"] == pytest.approx(0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_full_library_cuda_device_roundtrip():
    model = TinyRanker().cuda()
    aggregate, _ = evaluate_full_library(
        model,
        one_validation_query(),
        TinyFeatureStore(),
        candidate_batch_size=1,
        device="cuda",
    )

    assert aggregate["H@1"] == 1.0
    assert next(model.parameters()).device.type == "cuda"
    assert model.training


def test_train_fold_has_no_test_input_and_writes_selected_outputs(tmp_path):
    assert "test_queries" not in inspect.signature(train_fold).parameters
    with pytest.raises(TypeError):
        train_fold(
            TinyRanker(),
            [],
            [],
            TinyFeatureStore(),
            tmp_path,
            test_queries=[],
        )

    store = TinyFeatureStore()
    episodes = [
        QueryEpisode(
            ligand_key="ligand-1",
            positive_hashes=("p3",),
            unlabeled_hashes=("p1", "p2"),
        ),
        QueryEpisode(
            ligand_key="ligand-2",
            positive_hashes=("p3", "p2"),
            unlabeled_hashes=("p1", "p4"),
        ),
    ]
    validation = [
        EvaluationQuery(
            ligand_key="ligand",
            train_positive_hashes=("p3",),
            relevant_hashes=("p2",),
            candidate_hashes=("p1", "p2"),
            split_unit="unit",
        )
    ]
    model = TinyRanker()
    result = train_fold(
        model,
        episodes,
        validation,
        store,
        tmp_path,
        device="cpu",
        max_epochs=2,
        validation_interval=1,
        patience=3,
        query_batch_size=2,
        unlabeled_per_query=2,
        candidate_batch_size=1,
        seed=7,
    )

    expected_files = {
        "best_model.pt",
        "train_history.tsv",
        "validation_per_query.tsv",
        "fold_manifest.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_files
    history = pd.read_csv(tmp_path / "train_history.tsv", sep="\t")
    per_query = pd.read_csv(tmp_path / "validation_per_query.tsv", sep="\t")
    manifest = json.loads((tmp_path / "fold_manifest.json").read_text())
    checkpoint = torch.load(tmp_path / "best_model.pt", map_location="cpu")

    assert len(history) == 2
    assert per_query["n_candidates"].tolist() == [2]
    assert result["selected_epoch"] == 1
    assert manifest["selected_epoch"] == 1
    assert checkpoint["epoch"] == 1
    assert manifest["validation_metrics"] == result["validation_metrics"]
    assert set(manifest["validation_metrics"]) == {
        "H@1",
        "H@10",
        "H@50",
        "MRR",
        "nDCG@10",
        "Recall@1%",
        "Coverage@10",
    }
    assert manifest["candidate_count"] == 2
    assert manifest["train_hash_basis"] == "stable_training_episode_content"
    assert manifest["validation_hash_basis"] == "stable_validation_query_content"
    assert manifest["feature_store_audit"]["cache_paths"]["tiny"] == "/tmp/tiny.pkl"
    assert manifest["config"]["optimizer"] == {
        "name": "AdamW",
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
    }
    seen = [value for batch in store.protein_batches for value in batch]
    assert {"p1", "p2"}.issubset(seen)
    assert ("p3", "p1", "p2", "p3", "p2", "p1", "p4") in store.protein_batches
    assert (1, 3) in model.score_shapes
    assert (1, 4) in model.score_shapes
    assert (2, 7) not in model.score_shapes


def test_manifest_converts_path_and_numpy_config_values(tmp_path):
    train_fold(
        TinyRanker(),
        one_episode(),
        one_validation_query(),
        TinyFeatureStore(),
        tmp_path,
        max_epochs=1,
        validation_interval=1,
        unlabeled_per_query=2,
        config={
            "path": Path("/tmp/cache.npy"),
            "integer": np.int64(7),
            "array": np.asarray([1.5, 2.5], dtype=np.float32),
        },
    )

    manifest = json.loads((tmp_path / "fold_manifest.json").read_text())
    run_config = manifest["config"]["run_config"]
    assert run_config == {
        "path": "/tmp/cache.npy",
        "integer": 7,
        "array": [1.5, 2.5],
    }


def test_unserializable_manifest_fails_before_replacing_outputs_or_leaving_temp(
    tmp_path,
):
    original = {}
    for name in (
        "best_model.pt",
        "train_history.tsv",
        "validation_per_query.tsv",
        "fold_manifest.json",
    ):
        payload = f"old:{name}".encode()
        (tmp_path / name).write_bytes(payload)
        original[name] = payload

    with pytest.raises(TypeError, match="JSON-compatible"):
        train_fold(
            TinyRanker(),
            one_episode(),
            one_validation_query(),
            TinyFeatureStore(),
            tmp_path,
            max_epochs=1,
            validation_interval=1,
            unlabeled_per_query=2,
            config={"unsupported": object()},
        )

    assert {path.name for path in tmp_path.iterdir()} == set(original)
    assert {
        name: (tmp_path / name).read_bytes() for name in original
    } == original


def test_early_stopping_occurs_after_exactly_three_non_improvements(
    tmp_path, monkeypatch
):
    validation_calls = []

    def constant_evaluation(model, queries, feature_store, **kwargs):
        validation_calls.append(float(model.scale.detach().cpu()))
        return validation_metrics(0.5), pd.DataFrame({"n_candidates": [2]})

    monkeypatch.setattr(
        training_module, "evaluate_full_library", constant_evaluation
    )
    result = train_fold(
        TinyRanker(),
        one_episode(),
        one_validation_query(),
        TinyFeatureStore(),
        tmp_path,
        max_epochs=10,
        validation_interval=1,
        patience=3,
        unlabeled_per_query=2,
    )

    history = pd.read_csv(tmp_path / "train_history.tsv", sep="\t")
    assert len(validation_calls) == 4
    assert history["epoch"].tolist() == [1, 2, 3, 4]
    assert result["selected_epoch"] == 1


def test_scheduler_advances_each_epoch_and_history_records_learning_rate(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        training_module,
        "evaluate_full_library",
        lambda *args, **kwargs: (
            validation_metrics(0.5),
            pd.DataFrame({"n_candidates": [2]}),
        ),
    )
    train_fold(
        TinyRanker(),
        one_episode(),
        one_validation_query(),
        TinyFeatureStore(),
        tmp_path,
        max_epochs=3,
        validation_interval=3,
        unlabeled_per_query=2,
    )

    history = pd.read_csv(tmp_path / "train_history.tsv", sep="\t")
    expected = [
        3e-4 * (1.0 + math.cos(math.pi * step / 50.0)) / 2.0
        for step in range(3)
    ]
    assert history["learning_rate"].tolist() == pytest.approx(expected)
    assert history["learning_rate"].is_monotonic_decreasing
    assert history["learning_rate"].nunique() == 3


def test_later_validation_improvement_saves_that_epoch_weights(
    tmp_path, monkeypatch
):
    observed_weights = []

    def improving_evaluation(model, queries, feature_store, **kwargs):
        observed_weights.append(model.scale.detach().cpu().clone())
        score = 0.2 if len(observed_weights) == 1 else 0.8
        return validation_metrics(score), pd.DataFrame({"n_candidates": [2]})

    monkeypatch.setattr(
        training_module, "evaluate_full_library", improving_evaluation
    )
    result = train_fold(
        TinyRanker(),
        one_episode(),
        one_validation_query(),
        TinyFeatureStore(),
        tmp_path,
        max_epochs=2,
        validation_interval=1,
        unlabeled_per_query=2,
    )

    checkpoint = torch.load(tmp_path / "best_model.pt", map_location="cpu")
    assert result["selected_epoch"] == 2
    assert checkpoint["epoch"] == 2
    assert checkpoint["model_state_dict"]["scale"] == pytest.approx(
        observed_weights[1].item()
    )
    assert checkpoint["model_state_dict"]["scale"] != pytest.approx(
        observed_weights[0].item()
    )
