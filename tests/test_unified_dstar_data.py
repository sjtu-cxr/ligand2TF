from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.unified_dstar_data import (
    EvaluationQuery,
    FEATURE_ORDER,
    FeatureStore,
    FoldFrames,
    QueryEpisode,
    build_evaluation_queries,
    build_query_candidates,
    build_training_episodes,
    load_candidate_table,
    load_fold_frames,
    normalize_role_frame,
)


REQUIRED_COLUMNS = {
    "edge_id",
    "ligand_key",
    "sequence_md5",
    "split_unit",
    "split_role",
}


def _write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _sequence_md5(sequence):
    return hashlib.md5(sequence.encode()).hexdigest()


def _tf_row(role, edge_id, ligand, protein, unit, fold=0, **extra):
    return {
        "outer_fold": fold,
        "split_role": role,
        "edge_id": edge_id,
        "ligand_key": ligand,
        "sequence_md5": protein,
        "split_unit": unit,
        **extra,
    }


def _morgan_row(role, edge_id, ligand, protein, unit, **extra):
    return {
        "split_role": role,
        "edge_id": edge_id,
        "ligand_key": f"legacy-{ligand}",
        "ligand_key_for_c_light": ligand,
        "sequence_md5": protein,
        "cluster_id": unit,
        **extra,
    }


def test_tf_outer_edges_are_filtered_normalized_and_partitioned(tmp_path):
    _write_tsv(
        tmp_path / "outer_edges.tsv",
        [
            _tf_row(
                "train",
                "e1",
                "l1",
                "p1",
                "tf-a",
                canonical_smiles="CC",
                canonical_ion_label=None,
            ),
            _tf_row("train", "e2", "l2", "p2", "tf-b"),
            _tf_row("val", "e3", "l1", "p3", "tf-c"),
            _tf_row("test", "e4", "l1", "p4", "tf-d"),
            _tf_row("train", "other", "l9", "p9", "tf-z", fold=1),
        ],
    )

    frames = load_fold_frames("TF-50", tmp_path, outer_fold=0)

    assert isinstance(frames, FoldFrames)
    assert (len(frames.train), len(frames.val), len(frames.test)) == (2, 1, 1)
    assert REQUIRED_COLUMNS <= set(frames.train.columns)
    assert {"canonical_smiles", "canonical_ion_label"} <= set(frames.train.columns)
    assert frames.train.loc[0, "canonical_smiles"] == "CC"
    assert frames.train["split_role"].tolist() == ["train", "train"]


def test_morgan_role_files_use_c_light_ligand_key_and_cluster_unit(tmp_path):
    fold = tmp_path / "fold_2"
    _write_tsv(
        fold / "train.tsv",
        [
            _morgan_row(
                "train",
                "e1",
                "l1",
                "p1",
                "cluster-a",
                canonical_smiles="CCC",
            ),
            _morgan_row("train", "e2", "l2", "p2", "cluster-b"),
        ],
    )
    _write_tsv(fold / "val.tsv", [_morgan_row("val", "e3", "l3", "p3", "cluster-c")])
    _write_tsv(
        fold / "test.tsv", [_morgan_row("test", "e4", "l4", "p4", "cluster-d")]
    )

    frames = load_fold_frames("Ligand-Morgan-0.5", tmp_path, outer_fold=2)

    assert (len(frames.train), len(frames.val), len(frames.test)) == (2, 1, 1)
    assert frames.train["ligand_key"].tolist() == ["l1", "l2"]
    assert frames.train["split_unit"].tolist() == ["cluster-a", "cluster-b"]
    assert frames.test["split_role"].tolist() == ["test"]


def test_fold_normalization_uses_explicit_ion_prefix_for_non_smiles_ions(tmp_path):
    _write_tsv(
        tmp_path / "outer_edges.tsv",
        [
            _tf_row(
                "train",
                "e1",
                "legacy-zinc",
                "p1",
                "tf-a",
                canonical_smiles=None,
                canonical_ion_label="Zn2+",
            ),
            _tf_row(
                "train",
                "e2",
                "OO",
                "p2",
                "tf-b",
                canonical_smiles="OO",
                canonical_ion_label="H2O2",
            ),
            _tf_row("val", "e3", "l3", "p3", "tf-c"),
            _tf_row("test", "e4", "l4", "p4", "tf-d"),
        ],
    )

    frames = load_fold_frames("TF-50", tmp_path, outer_fold=0)

    assert frames.train["ligand_key"].tolist() == ["ion:Zn2+", "OO"]
    assert frames.train["canonical_ion_label"].tolist() == ["Zn2+", "H2O2"]


def test_public_role_normalization_supports_complete_refit_duplicates():
    frame = pd.DataFrame([
        _tf_row("train", "e1", "Zn2+", "p1", "all", canonical_smiles="", canonical_ion_label="Zn2+"),
        _tf_row("train", "e2", "Zn2+", "p1", "all", canonical_smiles="", canonical_ion_label="Zn2+"),
        _tf_row("train", "e3", "OO", "p2", "all", canonical_smiles="OO", canonical_ion_label="H2O2"),
    ])
    normalized = normalize_role_frame(
        frame, role="train", ligand_source="ligand_key",
        split_unit_source="split_unit", context="refit",
        enforce_unique_edges=False,
    )
    assert normalized["ligand_key"].tolist() == ["ion:Zn2+", "ion:Zn2+", "OO"]


@pytest.mark.parametrize("split_name", ["TF-50", "Ligand-Morgan-0.5"])
def test_split_unit_overlap_is_rejected(tmp_path, split_name):
    if split_name == "TF-50":
        _write_tsv(
            tmp_path / "outer_edges.tsv",
            [
                _tf_row("train", "e1", "l1", "p1", "shared"),
                _tf_row("val", "e2", "l2", "p2", "shared"),
                _tf_row("test", "e3", "l3", "p3", "test-only"),
            ],
        )
    else:
        fold = tmp_path / "fold_0"
        _write_tsv(
            fold / "train.tsv", [_morgan_row("train", "e1", "l1", "p1", "shared")]
        )
        _write_tsv(
            fold / "val.tsv", [_morgan_row("val", "e2", "l2", "p2", "shared")]
        )
        _write_tsv(
            fold / "test.tsv",
            [_morgan_row("test", "e3", "l3", "p3", "test-only")],
        )

    with pytest.raises(ValueError, match="split_unit.*overlap"):
        load_fold_frames(split_name, tmp_path, outer_fold=0)


def test_duplicate_ligand_sequence_edge_within_role_is_rejected(tmp_path):
    _write_tsv(
        tmp_path / "outer_edges.tsv",
        [
            _tf_row("train", "e1", "l1", "p1", "tf-a"),
            _tf_row("train", "e2", "l1", "p1", "tf-a"),
            _tf_row("val", "e3", "l2", "p2", "tf-b"),
            _tf_row("test", "e4", "l3", "p3", "tf-c"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate.*ligand.*sequence"):
        load_fold_frames("TF-50", tmp_path, outer_fold=0)


def test_duplicate_edge_id_within_role_is_rejected(tmp_path):
    _write_tsv(
        tmp_path / "outer_edges.tsv",
        [
            _tf_row("train", "duplicate", "l1", "p1", "tf-a"),
            _tf_row("train", "duplicate", "l2", "p2", "tf-b"),
            _tf_row("val", "e3", "l3", "p3", "tf-c"),
            _tf_row("test", "e4", "l4", "p4", "tf-d"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate edge_id"):
        load_fold_frames("TF-50", tmp_path, outer_fold=0)


def test_duplicate_edge_id_across_roles_is_rejected(tmp_path):
    _write_tsv(
        tmp_path / "outer_edges.tsv",
        [
            _tf_row("train", "duplicate", "l1", "p1", "tf-a"),
            _tf_row("val", "duplicate", "l2", "p2", "tf-b"),
            _tf_row("test", "e3", "l3", "p3", "tf-c"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate edge_id.*across roles"):
        load_fold_frames("TF-50", tmp_path, outer_fold=0)


def test_duplicate_ligand_sequence_pair_across_roles_is_rejected(tmp_path):
    _write_tsv(
        tmp_path / "outer_edges.tsv",
        [
            _tf_row("train", "e1", "same-ligand", "same-protein", "tf-a"),
            _tf_row("val", "e2", "same-ligand", "same-protein", "tf-b"),
            _tf_row("test", "e3", "l3", "p3", "tf-c"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate ligand-sequence.*across roles"):
        load_fold_frames("TF-50", tmp_path, outer_fold=0)


def test_numeric_and_string_edge_ids_collide_across_roles(tmp_path):
    _write_tsv(
        tmp_path / "outer_edges.tsv",
        [
            _tf_row("train", 1, "l1", "p1", "tf-a"),
            _tf_row("val", "1", "l2", "p2", "tf-b"),
            _tf_row("test", "3", "l3", "p3", "tf-c"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate edge_id.*across roles"):
        load_fold_frames("TF-50", tmp_path, outer_fold=0)


def test_tsv_loader_preserves_leading_zeroes_in_key_columns(tmp_path):
    _write_tsv(
        tmp_path / "outer_edges.tsv",
        [
            _tf_row("train", "001", "0001", "00001", "000001"),
            _tf_row("val", "002", "0002", "00002", "000002"),
            _tf_row("test", "003", "0003", "00003", "000003"),
        ],
    )

    frames = load_fold_frames("TF-50", tmp_path, outer_fold=0)

    key_columns = ["edge_id", "ligand_key", "sequence_md5", "split_unit"]
    assert frames.train.loc[0, key_columns].to_dict() == {
        "edge_id": "001",
        "ligand_key": "0001",
        "sequence_md5": "00001",
        "split_unit": "000001",
    }


def test_tf_rejects_unknown_role_in_selected_fold(tmp_path):
    _write_tsv(
        tmp_path / "outer_edges.tsv",
        [_tf_row("development", "e1", "l1", "p1", "tf-a")],
    )

    with pytest.raises(ValueError, match="split_role"):
        load_fold_frames("TF-50", tmp_path, outer_fold=0)


def test_candidate_table_requires_unique_hashes_and_sequences(tmp_path):
    valid = tmp_path / "valid.tsv"
    sequence_one = "AAAA"
    sequence_two = "BBBB"
    hash_one = _sequence_md5(sequence_one)
    hash_two = _sequence_md5(sequence_two)
    _write_tsv(
        valid,
        [
            {
                "sequence_md5": hash_one,
                "protein_sequence": sequence_one,
                "name": "one",
            },
            {
                "sequence_md5": hash_two,
                "protein_sequence": sequence_two,
                "name": "two",
            },
        ],
    )
    loaded = load_candidate_table(valid)
    assert loaded["sequence_md5"].tolist() == [hash_one, hash_two]

    duplicate = tmp_path / "duplicate.tsv"
    _write_tsv(
        duplicate,
        [
            {"sequence_md5": hash_one, "protein_sequence": sequence_one},
            {"sequence_md5": hash_one, "protein_sequence": sequence_two},
        ],
    )
    with pytest.raises(ValueError, match="unique.*sequence_md5|duplicate"):
        load_candidate_table(duplicate)

    missing_column = tmp_path / "missing.tsv"
    _write_tsv(missing_column, [{"sequence_md5": "p1"}])
    with pytest.raises(ValueError, match="protein_sequence"):
        load_candidate_table(missing_column)


def test_candidate_table_rejects_same_sequence_under_different_hashes(tmp_path):
    duplicate_sequence = tmp_path / "duplicate_sequence.tsv"
    _write_tsv(
        duplicate_sequence,
        [
            {"sequence_md5": "p1", "protein_sequence": "SAMESEQUENCE"},
            {"sequence_md5": "p2", "protein_sequence": "SAMESEQUENCE"},
        ],
    )

    with pytest.raises(ValueError, match="unique protein_sequence|duplicate"):
        load_candidate_table(duplicate_sequence)


def test_candidate_table_rejects_sequence_md5_mismatch(tmp_path):
    mismatch = tmp_path / "mismatch.tsv"
    _write_tsv(
        mismatch,
        [{"sequence_md5": "0" * 32, "protein_sequence": "AAAA"}],
    )

    with pytest.raises(ValueError, match="sequence_md5 mismatch"):
        load_candidate_table(mismatch)


def test_candidate_filter_removes_train_positives_only_and_preserves_order():
    candidates = ["p3", "p1", "held-out", "p2"]

    result = build_query_candidates(candidates, {"p1", "p2"})

    assert result == ["p3", "held-out"]


def test_candidate_filter_rejects_duplicate_candidates_and_unknown_positive():
    with pytest.raises(ValueError, match="duplicate"):
        build_query_candidates(["p1", "p1"], set())
    with pytest.raises(ValueError, match="unknown.*train positive"):
        build_query_candidates(["p1"], {"not-in-library"})


def test_training_episodes_are_deterministic_and_include_all_train_positives():
    train = pd.DataFrame(
        [
            {"ligand_key": "b", "sequence_md5": "p3"},
            {"ligand_key": "a", "sequence_md5": "p2"},
            {"ligand_key": "a", "sequence_md5": "p1"},
        ]
    )
    candidates = ["p1", "p2", "p3", "p4", "p5", "p6"]

    first = build_training_episodes(train, candidates, unknown_per_query=2, seed=17)
    second = build_training_episodes(train, candidates, unknown_per_query=2, seed=17)

    assert first == second
    assert all(isinstance(episode, QueryEpisode) for episode in first)
    assert [episode.ligand_key for episode in first] == ["a", "b"]
    assert first[0].positive_hashes == ("p1", "p2")
    assert set(first[0].unlabeled_hashes).isdisjoint(first[0].positive_hashes)
    assert len(first[0].unlabeled_hashes) == 2


def test_training_episodes_fail_when_unlabeled_pool_is_too_small():
    train = pd.DataFrame([{"ligand_key": "a", "sequence_md5": "p1"}])

    with pytest.raises(ValueError, match="insufficient"):
        build_training_episodes(train, ["p1", "p2"], unknown_per_query=2, seed=1)


def test_training_episodes_reject_non_train_split_roles():
    edges = pd.DataFrame(
        [
            {
                "ligand_key": "a",
                "sequence_md5": "p1",
                "split_role": "train",
            },
            {
                "ligand_key": "a",
                "sequence_md5": "p2",
                "split_role": "val",
            },
        ]
    )

    with pytest.raises(ValueError, match="split_role.*train"):
        build_training_episodes(
            edges,
            ["p1", "p2", "p3"],
            unknown_per_query=1,
            seed=1,
        )


def test_evaluation_query_joins_repeated_tf_units_and_uses_train_only_filter():
    train = pd.DataFrame(
        [
            {"ligand_key": "a", "sequence_md5": "p1"},
            {"ligand_key": "other", "sequence_md5": "p4"},
        ]
    )
    evaluation = pd.DataFrame(
        [
            {"ligand_key": "a", "sequence_md5": "p2", "split_unit": "tf-z"},
            {"ligand_key": "a", "sequence_md5": "p3", "split_unit": "tf-a"},
            {"ligand_key": "a", "sequence_md5": "p2", "split_unit": "tf-z"},
        ]
    )

    queries = build_evaluation_queries(
        train,
        evaluation,
        candidate_hashes=["p1", "p2", "p3", "p4"],
    )

    assert queries == [
        EvaluationQuery(
            ligand_key="a",
            train_positive_hashes=("p1",),
            relevant_hashes=("p2", "p3"),
            candidate_hashes=("p2", "p3", "p4"),
            split_unit="tf-a;tf-z",
        )
    ]


def test_evaluation_relevant_hash_must_survive_candidate_filter():
    train = pd.DataFrame([{"ligand_key": "a", "sequence_md5": "p1"}])
    evaluation = pd.DataFrame(
        [{"ligand_key": "a", "sequence_md5": "p1", "split_unit": "tf-a"}]
    )

    with pytest.raises(ValueError, match="relevant"):
        build_evaluation_queries(train, evaluation, ["p1", "p2"])


def test_evaluation_queries_reject_non_train_rows_in_train_edges():
    train = pd.DataFrame(
        [
            {
                "ligand_key": "a",
                "sequence_md5": "p1",
                "split_role": "val",
            }
        ]
    )
    evaluation = pd.DataFrame(
        [
            {
                "ligand_key": "a",
                "sequence_md5": "p2",
                "split_unit": "unit-a",
                "split_role": "test",
            }
        ]
    )

    with pytest.raises(ValueError, match="train_edges split_role.*train"):
        build_evaluation_queries(train, evaluation, ["p1", "p2"])


def test_evaluation_queries_reject_train_role_as_held_out():
    train = pd.DataFrame(
        [
            {
                "ligand_key": "a",
                "sequence_md5": "p1",
                "split_role": "train",
            }
        ]
    )
    evaluation = pd.DataFrame(
        [
            {
                "ligand_key": "a",
                "sequence_md5": "p2",
                "split_unit": "unit-a",
                "split_role": "train",
            }
        ]
    )

    with pytest.raises(ValueError, match="evaluation_edges split_role.*val.*test"):
        build_evaluation_queries(train, evaluation, ["p1", "p2"])


def _write_pickle(path, value):
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _feature_store(
    tmp_path,
    *,
    molformer=None,
    ecfp=None,
    esm=None,
    missing_ion_policy="error",
):
    paths = {
        "molformer_path": tmp_path / "molformer.pkl",
        "esm_path": tmp_path / "esm.pkl",
        "ecfp_path": tmp_path / "ecfp.pkl",
        "ion_descriptors_path": tmp_path / "ions.json",
    }
    _write_pickle(
        paths["molformer_path"],
        molformer
        if molformer is not None
        else {
            "regular": np.arange(3, dtype=np.float32),
        },
    )
    _write_pickle(
        paths["ecfp_path"],
        ecfp
        if ecfp is not None
        else {
            "regular": np.arange(4, dtype=np.float32),
            "fallback": np.ones(4, dtype=np.float32),
        },
    )
    _write_pickle(
        paths["esm_path"],
        esm if esm is not None else {"p1": np.arange(5, dtype=np.float32)},
    )
    ion_values = {name: index + 0.5 for index, name in enumerate(FEATURE_ORDER)}
    paths["ion_descriptors_path"].write_text(
        json.dumps(
            {
                "feature_order": list(reversed(FEATURE_ORDER)),
                "descriptors": {"Zn2+": ion_values},
            }
        )
    )
    return FeatureStore(
        **paths,
        molformer_dim=3,
        ecfp_dim=4,
        esm_dim=5,
        missing_ion_policy=missing_ion_policy,
    )


def test_feature_store_batches_have_expected_shapes_fallback_and_ion_mask(tmp_path):
    store = _feature_store(tmp_path)

    molformer, ecfp, ion_features, ion_mask = store.ligand_batch(
        ["regular", "fallback", "ion:Zn2+"]
    )
    proteins = store.protein_batch(["p1"])

    assert molformer.shape == (3, 3)
    assert ecfp.shape == (3, 4)
    assert ion_features.shape == (3, 10)
    assert ion_mask.shape == (3,)
    assert proteins.shape == (1, 5)
    assert molformer.dtype == ecfp.dtype == ion_features.dtype == torch.float32
    assert proteins.dtype == torch.float32
    assert ion_mask.dtype == torch.bool
    torch.testing.assert_close(molformer[1], torch.zeros(3))
    torch.testing.assert_close(ecfp[2], torch.zeros(4))
    torch.testing.assert_close(
        ion_features[2],
        torch.tensor([index + 0.5 for index in range(10)], dtype=torch.float32),
    )
    assert ion_mask.tolist() == [False, False, True]

    audit = store.audit()
    assert audit["dimensions"] == {
        "molformer": 3,
        "ecfp": 4,
        "esm": 5,
        "ion": 10,
    }
    assert audit["missing_key_counts"]["molformer"] == 1
    assert audit["cache_paths"]["esm"].endswith("esm.pkl")


def test_feature_store_does_not_infer_ion_identity_from_descriptor_label(tmp_path):
    store = _feature_store(tmp_path)

    with pytest.raises(KeyError, match="regular ligand.*MoLFormer.*ECFP"):
        store.ligand_batch(["Zn2+"])


def test_feature_store_missing_explicit_ion_descriptor_is_fatal_and_audited(
    tmp_path,
):
    store = _feature_store(tmp_path)

    with pytest.raises(KeyError, match="ion descriptor missing.*Cd_unspecified"):
        store.ligand_batch(["ion:Cd_unspecified"])

    audit = store.audit()
    assert audit["missing_ion_policy"] == "error"
    assert audit["missing_key_counts"]["ion_descriptor"] == 1
    assert audit["missing_keys"]["ion_descriptor"] == ["ion:Cd_unspecified"]


def test_feature_store_zero_policy_audits_unknown_ion_fallback(tmp_path):
    store = _feature_store(tmp_path, missing_ion_policy="zero")

    molformer, ecfp, ion_features, ion_mask = store.ligand_batch(
        ["ion:Cd_unspecified"]
    )

    torch.testing.assert_close(molformer, torch.zeros((1, 3)))
    torch.testing.assert_close(ecfp, torch.zeros((1, 4)))
    torch.testing.assert_close(ion_features, torch.zeros((1, 10)))
    assert ion_mask.tolist() == [True]
    audit = store.audit()
    assert audit["missing_ion_policy"] == "zero"
    assert audit["missing_key_counts"]["ion_descriptor"] == 1
    assert audit["missing_keys"]["ion_descriptor"] == ["ion:Cd_unspecified"]


def test_feature_store_rejects_unknown_missing_ion_policy(tmp_path):
    with pytest.raises(ValueError, match="missing_ion_policy"):
        _feature_store(tmp_path, missing_ion_policy="invent")


def test_feature_store_missing_esm_is_fatal(tmp_path):
    store = _feature_store(tmp_path)

    with pytest.raises(KeyError, match="ESM.*missing"):
        store.protein_batch(["unknown-protein"])

    assert store.audit()["missing_key_counts"]["esm"] == 1


def test_feature_store_regular_ligand_missing_all_features_is_fatal(tmp_path):
    store = _feature_store(tmp_path)

    with pytest.raises(KeyError, match="missing.*MoLFormer.*ECFP"):
        store.ligand_batch(["unknown-ligand"])

    assert store.audit()["missing_key_counts"]["all_ligand_features"] == 1


def test_feature_store_rejects_wrong_cache_dimensions_at_load_time(tmp_path):
    with pytest.raises(ValueError, match="MoLFormer.*dimension"):
        _feature_store(
            tmp_path,
            molformer={"regular": np.ones(2, dtype=np.float32)},
        )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_TF_ROOT = (
    PROJECT_ROOT
    / "data/model_training/v66/results/unified_retrieval_v3/splits/TF-50"
)
REAL_MORGAN_ROOT = (
    PROJECT_ROOT
    / "data/model_training/v66/results/unified_retrieval_v3/"
    "morgan05_d_fusion/splits"
)
REAL_CANDIDATE_PATH = (
    PROJECT_ROOT
    / "data/model_training/v66/candidates/"
    "curated_prokaryotic_tf_candidates_v4_with_md5.tsv"
)
REAL_CACHE_PATHS = (
    PROJECT_ROOT
    / "data/model_training/v66/features/molformer_internal_v66_ligands.pkl",
    PROJECT_ROOT
    / "data/model_training/v66/features/"
    "esm2_150M_curated_prokaryotic_tf_v4_regprecise_all.pkl",
    PROJECT_ROOT / "data/processed/modeling/ecfp4_cache.pkl",
    PROJECT_ROOT / "data/processed/modeling/ion_descriptors.json",
)
REAL_MORGAN_FOLD_PATHS = tuple(
    REAL_MORGAN_ROOT / f"fold_{fold}" / f"{role}.tsv"
    for fold in range(5)
    for role in ("train", "val", "test")
)
REAL_TF_SMOKE_PATHS = (
    REAL_TF_ROOT / "outer_edges.tsv",
    REAL_CANDIDATE_PATH,
    *REAL_CACHE_PATHS,
)
REAL_MORGAN_SMOKE_PATHS = (
    *REAL_MORGAN_FOLD_PATHS,
    REAL_CANDIDATE_PATH,
    *REAL_CACHE_PATHS,
)


@pytest.mark.skipif(
    not all(path.exists() for path in REAL_TF_SMOKE_PATHS),
    reason="real TF-50 split, candidate, or cache files are unavailable",
)
def test_real_tf50_all_folds_training_evaluation_and_zero_ion_smoke():
    expected_tf_counts = [
        (465, 203, 206),
        (536, 135, 203),
        (551, 188, 135),
        (544, 142, 188),
        (526, 206, 142),
    ]
    expected_query_counts = [
        (333, 160, 170),
        (362, 115, 160),
        (368, 151, 115),
        (386, 110, 151),
        (381, 170, 110),
    ]
    candidates = load_candidate_table(REAL_CANDIDATE_PATH)
    candidate_hashes = candidates["sequence_md5"].tolist()
    assert len(candidate_hashes) == len(set(candidate_hashes)) == 6457

    tf_keys = set()
    for outer_fold in range(5):
        frames = load_fold_frames("TF-50", REAL_TF_ROOT, outer_fold)
        assert (
            len(frames.train),
            len(frames.val),
            len(frames.test),
        ) == expected_tf_counts[outer_fold]
        episodes = build_training_episodes(
            frames.train, candidate_hashes, unknown_per_query=2, seed=42
        )
        val_queries = build_evaluation_queries(
            frames.train, frames.val, candidate_hashes
        )
        test_queries = build_evaluation_queries(
            frames.train, frames.test, candidate_hashes
        )
        assert (
            len(episodes),
            len(val_queries),
            len(test_queries),
        ) == expected_query_counts[outer_fold]
        tf_keys.update(
            pd.concat([frames.train, frames.val, frames.test])[
                "ligand_key"
            ].astype(str)
        )

    store = FeatureStore(missing_ion_policy="zero")
    assert store.protein_batch(candidate_hashes).shape == (6457, 640)
    tf_keys = sorted(tf_keys)
    molformer, ecfp, ion_features, ion_mask = store.ligand_batch(tf_keys)
    assert molformer.shape == (552, 768)
    assert ecfp.shape == (552, 2048)
    assert ion_features.shape == (552, 10)
    assert ion_mask.shape == (552,)

    expected_fallbacks = {
        "ion:Cd_unspecified",
        "ion:Co_unspecified",
        "ion:Cr_unspecified",
        "ion:Cu_unspecified",
        "ion:Fe_unspecified",
        "ion:Mn_unspecified",
        "ion:Na+",
        "ion:Ni_unspecified",
        "ion:Zn_unspecified",
        "ion:[2Fe-2S]_cluster",
        "ion:molybdate_or_molybdenum_unspecified",
    }
    audit = store.audit()
    assert audit["missing_ion_policy"] == "zero"
    assert audit["missing_key_counts"]["ion_descriptor"] == 11
    assert set(audit["missing_keys"]["ion_descriptor"]) == expected_fallbacks
    assert audit["missing_key_counts"]["all_ligand_features"] == 0


@pytest.mark.skipif(
    not all(path.exists() for path in REAL_MORGAN_SMOKE_PATHS),
    reason="real Morgan folds, candidate, or cache files are unavailable",
)
def test_real_morgan_all_folds_training_evaluation_and_cache_smoke():
    expected_morgan_counts = [
        (473, 157, 158),
        (473, 158, 157),
        (473, 157, 158),
        (473, 158, 157),
        (472, 158, 158),
    ]
    expected_query_counts = [
        (317, 111, 103),
        (311, 109, 111),
        (308, 114, 109),
        (323, 94, 114),
        (334, 103, 94),
    ]
    candidates = load_candidate_table(REAL_CANDIDATE_PATH)
    candidate_hashes = candidates["sequence_md5"].tolist()
    assert len(candidate_hashes) == len(set(candidate_hashes)) == 6457

    morgan_keys = set()
    for outer_fold in range(5):
        frames = load_fold_frames(
            "Ligand-Morgan-0.5", REAL_MORGAN_ROOT, outer_fold
        )
        assert (
            len(frames.train),
            len(frames.val),
            len(frames.test),
        ) == expected_morgan_counts[outer_fold]
        episodes = build_training_episodes(
            frames.train, candidate_hashes, unknown_per_query=2, seed=42
        )
        val_queries = build_evaluation_queries(
            frames.train, frames.val, candidate_hashes
        )
        test_queries = build_evaluation_queries(
            frames.train, frames.test, candidate_hashes
        )
        assert (
            len(episodes),
            len(val_queries),
            len(test_queries),
        ) == expected_query_counts[outer_fold]
        morgan_keys.update(
            pd.concat([frames.train, frames.val, frames.test])[
                "ligand_key"
            ].astype(str)
        )

    store = FeatureStore()
    assert store.protein_batch(candidate_hashes).shape == (6457, 640)
    morgan_keys = sorted(morgan_keys)
    assert len(morgan_keys) == 531
    assert store.ligand_batch(morgan_keys)[0].shape == (531, 768)
    audit = store.audit()
    assert audit["missing_key_counts"]["ion_descriptor"] == 0
    assert audit["missing_key_counts"]["all_ligand_features"] == 0
