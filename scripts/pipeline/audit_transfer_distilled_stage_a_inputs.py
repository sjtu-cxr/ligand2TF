#!/usr/bin/env python3
"""Fail-closed input and leakage audit for Stage A residual ranking."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRUSTED_SOURCE_ROOTS = (
    ROOT,
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unified_dstar_data import (  # noqa: E402
    build_evaluation_queries,
    load_fold_frames,
)
from src.unified_dstar_ensemble import (  # noqa: E402
    CandidateScoreBundle,
    SUPPORTED_FOLDS,
    SUPPORTED_SEEDS,
    SUPPORTED_SPLITS,
    file_sha256,
    load_candidate_score_bundle,
)


VALIDATION_ROOT = (
    ROOT
    / "data/model_training/v66/results/unified_dstar_ensemble/validation_scores"
)
TEST_BASE_ROOT = (
    ROOT
    / "data/model_training/v66/results/unified_dstar_ensemble/locked_test_scores"
)
NESTED_ROOT = (
    ROOT
    / "data/model_training/v66/results/unified_dstar_nested_routing"
)
TEST_DSTAR_ROOT = NESTED_ROOT / "outer_test_scores"
ARCHITECTURE_LOCK_ROOT = NESTED_ROOT / "architecture_locks"
TF_SPLIT_ROOT = (
    ROOT / "data/model_training/v66/results/unified_retrieval_v3/splits/TF-50"
)
LIGAND_SPLIT_ROOT = (
    ROOT
    / "data/model_training/v66/results/unified_retrieval_v3/"
    "morgan05_d_fusion/splits"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data/model_training/v66/results/transfer_distilled_residual_stage_a/"
    "audit/input_audit.json"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_CHANNELS = {
    "TF-50": {
        "validation": frozenset({"D", "Dstar", "S"}),
        "test": frozenset({"D", "Dstar", "Ensemble", "S"}),
    },
    "Ligand-Morgan-0.5": {
        "validation": frozenset({"C", "D", "Dstar", "P"}),
        "test": frozenset({"C", "D", "Dstar", "Ensemble", "P"}),
    },
}
ENSEMBLE_ROOT = ROOT / "data/model_training/v66/results/unified_dstar_ensemble"
INPUT_AUDIT_PATH = ENSEMBLE_ROOT / "input_audit.json"
GLOBAL_ARCHITECTURE_LOCK_PATH = (
    ROOT
    / "data/model_training/v66/results/unified_dstar/"
    "architecture_selection/locked_bundle/architecture_lock.json"
)
FIXED_TEST_PROVENANCE = {
    "ensemble_lock": ENSEMBLE_ROOT / "locked_bundle/ensemble_lock.json",
    "phase1_code_lock": ENSEMBLE_ROOT / "phase1_code_lock.json",
    "validation_bundle_lock": ENSEMBLE_ROOT / "validation_bundle_lock.json",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing {label}: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def atomic_split_units(value: str) -> tuple[str, ...]:
    """Parse one canonical, possibly composite, split-unit value."""

    _require(isinstance(value, str), "split unit must be a string")
    atoms = tuple(part.strip() for part in value.split(";"))
    _require(
        bool(atoms) and all(atoms) and len(set(atoms)) == len(atoms),
        f"invalid atomic split-unit value: {value!r}",
    )
    return atoms


def _atomic_unit_set(values: Sequence[str]) -> set[str]:
    return {atom for value in values for atom in atomic_split_units(value)}


def grouped_meta_fold(
    split_units: Sequence[str], *, folds: int = 3, seed: int = 20260726
) -> np.ndarray:
    """Assign atomically connected split units to deterministic meta folds."""

    _require(isinstance(folds, int) and folds > 1, "meta-fold count must exceed one")
    parsed = [atomic_split_units(value) for value in split_units]
    _require(bool(parsed), "meta-fold assignment requires split units")
    parent = list(range(len(parsed)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    owners: dict[str, int] = {}
    for index, atoms in enumerate(parsed):
        for atom in atoms:
            if atom in owners:
                union(index, owners[atom])
            else:
                owners[atom] = index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(parsed)):
        components[find(index)].append(index)
    ordered = sorted(
        components.values(),
        key=lambda indexes: (
            sha256(
                f"{seed}:".encode("ascii")
                + ";".join(
                    sorted({atom for i in indexes for atom in parsed[i]})
                ).encode("utf-8")
            ).hexdigest(),
            indexes,
        ),
    )
    assignments = np.empty(len(parsed), dtype=np.int64)
    for component_index, indexes in enumerate(ordered):
        target = component_index % folds
        assignments[indexes] = target
    atom_folds: dict[str, set[int]] = defaultdict(set)
    for atoms, fold in zip(parsed, assignments):
        for atom in atoms:
            atom_folds[atom].add(int(fold))
    _require(
        all(len(values) == 1 for values in atom_folds.values()),
        "atomic split unit crosses grouped meta folds",
    )
    return assignments


def _resolve_source_path(value: object, project_root: Path) -> Path:
    _require(isinstance(value, str) and bool(value), "declared source path is invalid")
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    try:
        resolved = path.resolve(strict=True)
        allowed_roots = tuple(
            dict.fromkeys(
                [project_root.resolve(), *(root.resolve() for root in TRUSTED_SOURCE_ROOTS)]
            )
        )
        _require(
            any(resolved.is_relative_to(root) for root in allowed_roots),
            f"declared source path is outside trusted roots: {path}",
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ValueError(f"declared source path is missing or outside trusted roots: {path}") from error
    _require(resolved.is_file(), f"declared source artifact is not a file: {resolved}")
    return resolved


def authenticate_source_tree(
    value: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
    prefix: str = "",
    hash_cache: dict[Path, str] | None = None,
) -> dict[str, dict[str, str]]:
    """Resolve and hash every explicit ``{path, sha256}`` source record."""

    _require(isinstance(value, Mapping), "source provenance tree must be a mapping")
    cache = {} if hash_cache is None else hash_cache
    records: dict[str, dict[str, str]] = {}

    def visit(node: object, name: str) -> None:
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for index, child in enumerate(node):
                visit(child, f"{name}[{index}]")
            return
        if not isinstance(node, Mapping):
            return
        if "path" in node and "sha256" in node:
            _require(set(node) == {"path", "sha256"}, f"source record schema mismatch: {name}")
            expected = _require_sha(node["sha256"], f"declared source hash {name}")
            path = _resolve_source_path(node["path"], project_root)
            if path not in cache:
                cache[path] = file_sha256(path)
            actual = cache[path]
            _require(actual == expected, f"source SHA-256 mismatch for {name}: {path}")
            records[name] = {"path": str(path), "sha256": actual}
            return
        if "path" in node:
            _require(
                name.startswith("prior_materializations[")
                and set(node)
                == {"inventory_path", "inventory_sha256", "ordinal", "path", "reason"},
                f"unhashed source path is not permitted: {name}.path",
            )
            directory = Path(str(node["path"])).resolve(strict=True)
            _require(directory.is_dir(), f"prior materialization is not a directory: {directory}")
            allowed_roots = tuple(root.resolve() for root in TRUSTED_SOURCE_ROOTS)
            _require(
                any(directory.is_relative_to(root) for root in allowed_roots),
                f"prior materialization is outside trusted roots: {directory}",
            )
        consumed: set[str] = set()
        pairs: list[tuple[str, str]] = []
        for path_key in node:
            if not isinstance(path_key, str) or not path_key.endswith("_path"):
                continue
            if name == "run_claim" and path_key == "claim_path":
                historical = Path(str(node[path_key])).resolve(strict=False)
                _require(
                    historical.is_absolute()
                    and any(
                        historical.is_relative_to(root.resolve())
                        for root in TRUSTED_SOURCE_ROOTS
                    ),
                    "historical run-claim path is outside trusted roots",
                )
                consumed.add(path_key)
                continue
            candidates = [f"{path_key[:-5]}_sha256"]
            if path_key == "artifact_path":
                candidates.insert(0, "sha256")
            elif path_key == "source_path":
                candidates.insert(0, "sha256")
            hash_key = next((candidate for candidate in candidates if candidate in node), None)
            if hash_key is not None:
                pairs.append((path_key, hash_key))
            else:
                raise ValueError(f"source path has no paired SHA-256: {name}.{path_key}")
        for path_key, hash_key in dict.fromkeys(pairs):
            if path_key in node or hash_key in node:
                expected = _require_sha(node[hash_key], f"declared source hash {name}.{path_key}")
                path = _resolve_source_path(node[path_key], project_root)
                if path not in cache:
                    cache[path] = file_sha256(path)
                actual = cache[path]
                _require(actual == expected, f"source SHA-256 mismatch for {name}.{path_key}: {path}")
                record_name = f"{name}.{path_key}"
                records[record_name] = {"path": str(path), "sha256": actual}
                consumed.update({path_key, hash_key})
        for key, child in node.items():
            if key in consumed:
                continue
            child_name = f"{name}.{key}" if name else str(key)
            if isinstance(key, str) and key.endswith("_sha256"):
                _require_sha(child, f"unpaired provenance hash {child_name}")
            visit(child, child_name)

    visit(value, prefix)
    _require(bool(records), "source provenance tree contains no authenticatable records")
    return records


def audit_historical_code_lock(
    value: Mapping[str, Any], *, project_root: Path = ROOT
) -> dict[str, Any]:
    """Validate a hash-bound historical code lock without rewriting history.

    The enclosing run claim authenticates the code-lock JSON itself.  Source paths
    may subsequently change in a mutable worktree, so mismatches are recorded as
    reproducibility gaps rather than treated as score-artifact tampering.
    """

    _require(
        value.get("schema") == "unified_dstar_ensemble_scorer_execution_code_lock"
        and value.get("schema_version") == 1,
        "historical execution code lock schema mismatch",
    )
    code_files = value.get("code_files")
    _require(isinstance(code_files, Mapping) and bool(code_files), "code lock files missing")
    records: dict[str, Any] = {}
    for name, record in sorted(code_files.items()):
        _require(
            isinstance(record, Mapping) and set(record) == {"path", "sha256"},
            f"historical code record schema mismatch: {name}",
        )
        expected = _require_sha(record["sha256"], f"historical code hash {name}")
        path = _resolve_source_path(record["path"], project_root)
        current = file_sha256(path)
        records[str(name)] = {
            "path": str(path),
            "historical_sha256": expected,
            "current_sha256": current,
            "current_match": current == expected,
        }
    lineage = value.get("lineage")
    _require(isinstance(lineage, Mapping) and bool(lineage), "code lock lineage missing")
    for key, declared in lineage.items():
        _require_sha(declared, f"historical code-lock lineage {key}")
    return {
        "files": records,
        "current_match_count": sum(item["current_match"] for item in records.values()),
        "historical_file_count": len(records),
        "historical_source_snapshot_available": all(
            item["current_match"] for item in records.values()
        ),
    }


def _input_audit_record(
    audit: Mapping[str, Any], *, split: str, seed: int, fold: int
) -> Mapping[str, Any]:
    _require(audit.get("schema_version") == 2, "upstream input audit schema mismatch")
    bundles = audit.get("bundles")
    _require(isinstance(bundles, list), "upstream input audit bundles missing")
    matches = [
        item
        for item in bundles
        if isinstance(item, Mapping)
        and (item.get("split"), item.get("seed"), item.get("outer_fold"))
        == (split, seed, fold)
    ]
    _require(len(matches) == 1, "upstream input audit identity coverage mismatch")
    return matches[0]


def authenticate_bundle_provenance(
    bundle: CandidateScoreBundle,
    *,
    input_audit_path: Path = INPUT_AUDIT_PATH,
    architecture_lock_path: Path = GLOBAL_ARCHITECTURE_LOCK_PATH,
    project_root: Path = ROOT,
    hash_cache: dict[Path, str] | None = None,
) -> dict[str, Any]:
    """Authenticate every source hash declared by one canonical bundle."""

    cache = {} if hash_cache is None else hash_cache
    audit = _read_json(input_audit_path, "upstream input audit")
    audit_path = _resolve_source_path(str(input_audit_path), project_root)
    if audit_path not in cache:
        cache[audit_path] = file_sha256(audit_path)
    audit_sha = cache[audit_path]
    record = _input_audit_record(
        audit,
        split=bundle.split,
        seed=bundle.seed,
        fold=bundle.outer_fold,
    )
    stage_record = record.get(bundle.stage)
    _require(isinstance(stage_record, Mapping), "upstream input audit stage record missing")
    authenticated = authenticate_source_tree(
        stage_record,
        project_root=project_root,
        prefix=bundle.stage,
        hash_cache=cache,
    )
    architecture_path = _resolve_source_path(str(architecture_lock_path), project_root)
    if architecture_path not in cache:
        cache[architecture_path] = file_sha256(architecture_path)
    architecture_sha = cache[architecture_path]
    _require(
        audit.get("architecture_lock_sha256") == architecture_sha,
        "upstream input audit architecture lock SHA-256 mismatch",
    )
    expected = {
        "input_audit": audit_sha,
        "architecture_lock": architecture_sha,
        **{name: item["sha256"] for name, item in authenticated.items()},
    }
    fixed_paths: dict[str, str] = {}
    if bundle.stage == "test":
        for name, path in FIXED_TEST_PROVENANCE.items():
            resolved = _resolve_source_path(str(path), project_root)
            if resolved not in cache:
                cache[resolved] = file_sha256(resolved)
            expected[name] = cache[resolved]
            fixed_paths[name] = str(resolved)
    _require(
        dict(bundle.source_hashes) == dict(sorted(expected.items())),
        f"{bundle.stage} bundle source provenance differs from authenticated chain",
    )
    return {
        "input_audit_path": str(audit_path),
        "input_audit_sha256": audit_sha,
        "architecture_lock_path": str(architecture_path),
        "architecture_lock_sha256": architecture_sha,
        "authenticated_source_count": len(authenticated) + 2 + len(fixed_paths),
        "authenticated_sources": authenticated,
        "fixed_source_paths": fixed_paths,
    }


def _payload_hashes(directory: Path, names: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in names:
        path = directory / name
        if not path.is_file():
            raise ValueError(f"missing audited source file: {path}")
        hashes[name] = file_sha256(path)
    return hashes


def _validate_identity(
    bundle: CandidateScoreBundle,
    *,
    split: str,
    seed: int,
    fold: int,
    stage: str,
) -> None:
    observed = (bundle.split, bundle.seed, bundle.outer_fold, bundle.stage)
    expected = (split, seed, fold, stage)
    _require(observed == expected, f"{stage} bundle identity mismatch: {observed} != {expected}")
    observed_channels = frozenset(bundle.channels)
    expected_channels = EXPECTED_CHANNELS[split][stage]
    _require(
        observed_channels == expected_channels,
        f"{stage} bundle channel schema mismatch: "
        f"{sorted(observed_channels)} != {sorted(expected_channels)}",
    )
    required_attribute = (
        "selection_inputs_only" if stage == "validation" else "locked_outer_test"
    )
    _require(
        bundle.attributes.get(required_attribute) is True,
        f"{stage} bundle lacks required {required_attribute} provenance",
    )
    for name, value in bundle.source_hashes.items():
        _require_sha(value, f"{stage} bundle source hash {name}")


def validate_stage_pair(
    *,
    validation: CandidateScoreBundle,
    test: CandidateScoreBundle,
    expected_split: str | None = None,
    expected_seed: int | None = None,
    expected_fold: int | None = None,
) -> None:
    """Validate identity, schema, alignment, and validation/test separation."""

    _require(validation.stage == "validation", "validation bundle must have stage='validation'")
    _require(test.stage == "test", "test bundle must have stage='test'")
    split = expected_split or validation.split
    seed = validation.seed if expected_seed is None else expected_seed
    fold = validation.outer_fold if expected_fold is None else expected_fold
    _validate_identity(validation, split=split, seed=seed, fold=fold, stage="validation")
    _validate_identity(test, split=split, seed=seed, fold=fold, stage="test")
    _require(
        validation.candidate_hashes == test.candidate_hashes,
        f"{split} seed {seed} fold {fold} candidate order mismatch between stages",
    )
    _require(
        set(validation.channels) == set(test.channels) - {"Ensemble"},
        f"{split} seed {seed} fold {fold} validation/test channel schema mismatch",
    )
    if split == "TF-50":
        overlap = _atomic_unit_set(validation.split_units) & _atomic_unit_set(
            test.split_units
        )
        _require(
            not overlap,
            "TF-50 atomic split-unit leakage between validation and test: "
            f"{sorted(overlap)[:5]}",
        )
    else:
        overlap = set(validation.query_ids) & set(test.query_ids)
        _require(
            not overlap,
            f"{split} query leakage between validation and test: {sorted(overlap)[:5]}",
        )


def validate_seed_bundles(bundles: Sequence[CandidateScoreBundle]) -> None:
    """Require exact fold-local metadata alignment across training seeds."""

    _require(bool(bundles), "seed bundle collection must not be empty")
    reference = bundles[0]
    fields = (
        "query_ids",
        "candidate_hashes",
        "relevant_hashes",
        "train_positive_hashes",
        "split_units",
        "ligand_supported",
    )
    labels = {
        "query_ids": "query order",
        "candidate_hashes": "candidate order",
        "relevant_hashes": "relevant-label alignment",
        "train_positive_hashes": "train-positive alignment",
        "split_units": "split-unit alignment",
        "ligand_supported": "support alignment",
    }
    for bundle in bundles[1:]:
        _require(bundle.split == reference.split, "seed bundle split mismatch")
        _require(bundle.outer_fold == reference.outer_fold, "seed bundle fold mismatch")
        _require(bundle.stage == reference.stage, "seed bundle stage mismatch")
        _require(set(bundle.channels) == set(reference.channels), "seed bundle channel schema mismatch")
        for field in fields:
            _require(
                getattr(bundle, field) == getattr(reference, field),
                f"seed bundle {labels[field]} mismatch",
            )


def audit_nested_dstar(path: Path, base_test: CandidateScoreBundle) -> dict[str, Any]:
    """Authenticate and align one selected nested outer-test D* matrix."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            _require(
                set(archive.files) == {"candidate_hashes", "ligand_keys", "scores"},
                "nested Dstar archive schema mismatch",
            )
            candidates = tuple(archive["candidate_hashes"].astype(str))
            queries = tuple(archive["ligand_keys"].astype(str))
            scores = np.asarray(archive["scores"], dtype=np.float64)
    except FileNotFoundError as error:
        raise ValueError(f"missing nested Dstar score archive: {path}") from error
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("nested Dstar"):
            raise
        raise ValueError(f"invalid nested Dstar score archive: {path}") from error
    _require(candidates == base_test.candidate_hashes, "nested Dstar candidate order mismatch")
    _require(queries == base_test.query_ids, "nested Dstar query order mismatch")
    _require(
        scores.shape == (len(queries), len(candidates)),
        "nested Dstar score matrix shape mismatch",
    )
    base_dstar = np.asarray(base_test.channels["Dstar"])
    _require(
        np.array_equal(np.isnan(scores), np.isnan(base_dstar)),
        "nested Dstar filtered-candidate mask mismatch",
    )
    _require(
        np.isfinite(scores[~np.isnan(scores)]).all(),
        "nested Dstar nonmissing scores must be finite",
    )
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "query_count": len(queries),
        "candidate_count": len(candidates),
        "finite_score_count": int(np.count_nonzero(np.isfinite(scores))),
        "masked_score_count": int(np.count_nonzero(np.isnan(scores))),
    }


def validate_architecture_provenance(
    *,
    lock_path: Path,
    test_manifest: Mapping[str, Any],
    split: str,
    seed: int,
    fold: int,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    """Validate that one nested test score came from its fold-local validation lock."""

    lock = _read_json(lock_path, "architecture lock")
    _require(
        lock.get("schema") == "unified_dstar_outer_fold_architecture_lock"
        and lock.get("schema_version") == 1,
        "architecture lock schema mismatch",
    )
    _require(
        (lock.get("split"), lock.get("outer_fold")) == (split, fold),
        "architecture lock split/fold identity mismatch",
    )
    rule = lock.get("selection_rule")
    _require(isinstance(rule, Mapping), "architecture lock selection rule missing")
    _require(
        rule.get("score") == "sqrt(mean_validation_H@10 * mean_validation_MRR)"
        and rule.get("aggregation")
        == "three training seeds within one split/outer fold"
        and rule.get("tie_break") == "architecture A"
        and
        rule.get("other_outer_folds_used") is False
        and rule.get("outer_test_results_used") is False,
        "architecture lock selection rule mismatch",
    )
    lock_sha = file_sha256(lock_path)
    _require(
        test_manifest.get("lock_sha256") == lock_sha,
        "nested test manifest architecture lock SHA-256 mismatch",
    )
    _require(
        test_manifest.get("schema") == "unified_dstar_locked_outer_test"
        and test_manifest.get("schema_version") == 1,
        "nested test manifest schema mismatch",
    )
    _require(
        (
            test_manifest.get("split"),
            test_manifest.get("seed"),
            test_manifest.get("fold"),
        )
        == (split, seed, fold),
        "nested test manifest identity mismatch",
    )
    architecture = lock.get("selected_architecture")
    _require(test_manifest.get("architecture") == architecture, "nested test architecture differs from lock")
    _require(
        test_manifest.get("selection_scope") == "split_and_outer_fold_only",
        "nested test selection scope is not fold-local",
    )
    shared_hash = _require_sha(lock.get("shared_config_sha256"), "lock shared config hash")
    _require(test_manifest.get("shared_config_sha256") == shared_hash, "nested test shared config hash differs from lock")
    validated_runs = lock.get("validated_runs")
    _require(isinstance(validated_runs, list), "architecture lock validated runs missing")
    expected_identities = {
        (architecture, split, expected_seed, fold)
        for architecture in ("A", "B")
        for expected_seed in SUPPORTED_SEEDS
    }
    observed_identities = {
        (run.get("architecture"), run.get("split"), run.get("seed"), run.get("fold"))
        for run in validated_runs
        if isinstance(run, Mapping)
    }
    _require(
        len(validated_runs) == 6
        and lock.get("validation_run_count") == 6
        and observed_identities == expected_identities,
        "architecture lock must contain exactly A/B x three seeds",
    )
    hash_cache: dict[Path, str] = {}
    authenticated_runs: list[dict[str, Any]] = []
    for run in validated_runs:
        _require(isinstance(run, Mapping), "architecture validation run schema mismatch")
        artifacts = run.get("artifact_sha256")
        expected_artifacts = {
            "best_model.pt",
            "fold_manifest.json",
            "train_history.tsv",
            "validation_per_query.tsv",
        }
        _require(
            isinstance(artifacts, Mapping) and set(artifacts) == expected_artifacts,
            "architecture validation artifact hash schema mismatch",
        )
        manifest_path = _resolve_source_path(run.get("manifest_path"), project_root)
        _require(manifest_path.name == "fold_manifest.json", "validation manifest filename mismatch")
        actual_hashes: dict[str, str] = {}
        for name in sorted(expected_artifacts):
            path = manifest_path.parent / name
            _require(path.is_file(), f"missing validation artifact: {path}")
            actual = hash_cache.setdefault(path, file_sha256(path))
            expected_hash = _require_sha(artifacts[name], f"validation artifact hash {name}")
            _require(actual == expected_hash, f"validation artifact {name} SHA-256 mismatch: {path}")
            actual_hashes[name] = actual
        manifest = _read_json(manifest_path, "validation fold manifest")
        metrics = manifest.get("validation_metrics")
        _require(isinstance(metrics, Mapping), "validation fold metrics missing")
        h10 = float(run.get("H@10", math.nan))
        mrr = float(run.get("MRR", math.nan))
        _require(
            math.isfinite(h10)
            and math.isfinite(mrr)
            and math.isclose(
                h10, float(metrics.get("H@10", math.nan)), rel_tol=0.0, abs_tol=1e-15
            )
            and math.isclose(
                mrr, float(metrics.get("MRR", math.nan)), rel_tol=0.0, abs_tol=1e-15
            ),
            "architecture lock metrics differ from validation manifest",
        )
        _require(
            run.get("selected_epoch") == manifest.get("selected_epoch"),
            "architecture lock epoch differs from validation manifest",
        )
        authenticated_runs.append(
            {
                "architecture": run["architecture"],
                "seed": run["seed"],
                "manifest_path": str(manifest_path),
                "artifact_sha256": actual_hashes,
                "H@10": h10,
                "MRR": mrr,
            }
        )

    recomputed: dict[str, dict[str, float | int]] = {}
    for name in ("A", "B"):
        rows = [row for row in authenticated_runs if row["architecture"] == name]
        mean_h10 = math.fsum(float(row["H@10"]) for row in rows) / 3.0
        mean_mrr = math.fsum(float(row["MRR"]) for row in rows) / 3.0
        recomputed[name] = {
            "mean_H@10": mean_h10,
            "mean_MRR": mean_mrr,
            "G": math.sqrt(mean_h10 * mean_mrr),
            "n_seeds": 3,
        }
    declared_architectures = lock.get("architectures")
    _require(isinstance(declared_architectures, Mapping), "architecture aggregate metrics missing")
    for name in ("A", "B"):
        declared = declared_architectures.get(name)
        _require(isinstance(declared, Mapping), f"architecture {name} aggregate missing")
        for metric, expected_value in recomputed[name].items():
            _require(
                declared.get(metric) == expected_value,
                f"architecture {name} aggregate {metric} differs from validation runs",
            )
    recomputed_selected = "B" if recomputed["B"]["G"] > recomputed["A"]["G"] else "A"
    _require(
        architecture == recomputed_selected,
        "selected_architecture does not follow declared validation rule",
    )
    summary_name = lock.get("summary_filename")
    _require(summary_name == "fold_architecture_summary.tsv", "architecture summary filename mismatch")
    summary_path = lock_path.parent / summary_name
    _require(summary_path.is_file(), f"missing architecture summary: {summary_path}")
    summary_sha = file_sha256(summary_path)
    _require(
        lock.get("summary_exact_bytes_sha256") == summary_sha,
        "architecture summary SHA-256 mismatch",
    )
    expected_run = [
        run for run in authenticated_runs
        if (split, fold, run.get("seed"), run.get("architecture"))
        == (split, fold, seed, architecture)
    ]
    _require(len(expected_run) == 1, "architecture lock lacks exactly one selected seed run")
    artifacts = expected_run[0].get("artifact_sha256")
    checkpoint_hash = _require_sha(artifacts.get("best_model.pt"), "selected validation checkpoint hash")
    _require(test_manifest.get("checkpoint_sha256") == checkpoint_hash, "nested test checkpoint provenance mismatch")
    checkpoint_path = _resolve_source_path(test_manifest.get("checkpoint_path"), project_root)
    selected_path = Path(expected_run[0]["manifest_path"]).parent / "best_model.pt"
    _require(
        checkpoint_path == selected_path
        and file_sha256(checkpoint_path) == checkpoint_hash,
        "nested test checkpoint path is not the selected validation artifact",
    )
    validation_input = lock.get("validation_input")
    if validation_input is not None:
        _require(isinstance(validation_input, Mapping), "architecture validation input schema mismatch")
        validation_manifest = (
            project_root
            / "data/model_training/v66/results/unified_dstar/validation_inputs/"
            "validation_input_manifest.json"
        )
        _require(
            validation_input.get("manifest_sha256") == file_sha256(validation_manifest),
            "architecture validation input manifest SHA-256 mismatch",
        )
        source_split = validation_input.get("source_split_hash")
        _require(isinstance(source_split, Mapping), "architecture source split provenance missing")
        authenticate_source_tree(source_split, project_root=project_root)
    return {
        "path": str(lock_path.resolve()),
        "sha256": lock_sha,
        "selected_architecture": architecture,
        "shared_config_sha256": shared_hash,
        "checkpoint_sha256": checkpoint_hash,
        "validation_run_count": len(authenticated_runs),
        "summary_sha256": summary_sha,
    }


def audit_fold_support(
    bundle: CandidateScoreBundle,
    *,
    split_root: Path,
    evaluation_role: str,
) -> dict[str, Any]:
    """Rebuild query labels and support solely from the fold-local split files."""

    _require(evaluation_role in {"val", "test"}, "evaluation role must be val or test")
    frames = load_fold_frames(bundle.split, split_root, bundle.outer_fold)
    evaluation = frames.val if evaluation_role == "val" else frames.test
    queries = tuple(build_evaluation_queries(frames.train, evaluation, bundle.candidate_hashes))
    expected = {
        "query_ids": tuple(query.ligand_key for query in queries),
        "relevant_hashes": tuple(query.relevant_hashes for query in queries),
        "train_positive_hashes": tuple(query.train_positive_hashes for query in queries),
        "split_units": tuple(query.split_unit for query in queries),
        "ligand_supported": tuple(bool(query.train_positive_hashes) for query in queries),
    }
    for name, values in expected.items():
        _require(
            getattr(bundle, name) == values,
            f"{bundle.split} fold {bundle.outer_fold} {evaluation_role} "
            f"bundle {name} differs from fold-local split manifests",
        )
    meta_assignments = grouped_meta_fold(bundle.split_units)
    atomic_units = _atomic_unit_set(bundle.split_units)
    if bundle.split == "TF-50":
        source_path = split_root / "outer_edges.tsv"
        train_path = source_path
        evaluation_path = source_path
    else:
        fold_root = split_root / f"fold_{bundle.outer_fold}"
        train_path = fold_root / "train.tsv"
        evaluation_path = fold_root / f"{evaluation_role}.tsv"
    return {
        "split_source_path": str(source_path.resolve()) if bundle.split == "TF-50" else str(split_root.resolve()),
        "train_path": str(train_path.resolve()),
        "train_sha256": file_sha256(train_path),
        "evaluation_path": str(evaluation_path.resolve()),
        "evaluation_sha256": file_sha256(evaluation_path),
        "queries": len(queries),
        "supported_queries": int(sum(expected["ligand_supported"])),
        "atomic_split_units": len(atomic_units),
        "meta_fold_query_counts": [
            int(np.count_nonzero(meta_assignments == fold)) for fold in range(3)
        ],
        "meta_fold_assignment_sha256": sha256(
            meta_assignments.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    }


def discover_bundle_directories(
    root: Path,
    *,
    splits: Sequence[str] = SUPPORTED_SPLITS,
    seeds: Sequence[int] = SUPPORTED_SEEDS,
    folds: Sequence[int] = SUPPORTED_FOLDS,
    expected_root_files: Sequence[str] = (),
    validate_reservations: bool = False,
) -> dict[tuple[str, int, int], Path]:
    """Discover only the frozen split/seed/fold tree; reject stale siblings."""

    expected = {
        (split, int(seed), int(fold))
        for split in splits
        for seed in seeds
        for fold in folds
    }
    observed: dict[tuple[str, int, int], Path] = {}
    if not root.is_dir():
        raise ValueError(f"bundle root is missing: {root}")
    unexpected: list[str] = []
    expected_splits = set(splits)
    allowed_files = set(expected_root_files)
    root_files = {path.name for path in root.iterdir() if path.is_file()}
    _require(
        root_files == allowed_files,
        f"bundle root control files mismatch: {sorted(root_files)} != {sorted(allowed_files)}",
    )
    reservation_root = root / ".reservations"
    if validate_reservations:
        _require(reservation_root.is_dir(), f"reservation root is missing: {reservation_root}")
        expected_reservation_leaves = {
            reservation_root / split / f"seed_{seed}" / f"fold_{fold}"
            for split in splits
            for seed in seeds
            for fold in folds
        }
        observed_reservation_leaves = {
            path
            for path in reservation_root.glob("*/*/*")
            if path.is_dir()
        }
        _require(
            observed_reservation_leaves == expected_reservation_leaves,
            "reservation split/seed/fold coverage mismatch",
        )
        expected_reservation_dirs = (
            {reservation_root}
            | {reservation_root / split for split in splits}
            | {
                reservation_root / split / f"seed_{seed}"
                for split in splits
                for seed in seeds
            }
            | expected_reservation_leaves
        )
        observed_reservation_dirs = {
            path for path in reservation_root.rglob("*") if path.is_dir()
        } | {reservation_root}
        _require(
            observed_reservation_dirs == expected_reservation_dirs,
            "reservation hierarchy contains unexpected directories",
        )
        reservation_files = {
            path for path in reservation_root.rglob("*") if path.is_file()
        }
        expected_reservation_files = {
            leaf / "reservation_status" for leaf in expected_reservation_leaves
        }
        _require(
            reservation_files == expected_reservation_files,
            "reservation status-file coverage mismatch",
        )
        _require(
            not any(path.is_symlink() for path in reservation_root.rglob("*")),
            "reservation hierarchy must not contain symlinks",
        )
        for status_path in sorted(reservation_files):
            fields: dict[str, str] = {}
            for line in status_path.read_text(encoding="ascii").splitlines():
                if not line:
                    continue
                _require(
                    line.count("=") == 1,
                    f"malformed reservation status line: {status_path}",
                )
                key, value = line.split("=", 1)
                _require(
                    key not in fields,
                    f"duplicate reservation status field {key}: {status_path}",
                )
                fields[key] = value
            _require(
                set(fields) == {"status", "job_id", "exit_status"},
                f"reservation status fields mismatch: {status_path}",
            )
            _require(
                fields["status"] == "complete"
                and re.fullmatch(r"[1-9][0-9]*[.][A-Za-z0-9_-]+", fields["job_id"])
                is not None
                and fields["exit_status"] == "0",
                f"reservation did not record a successful completed job: {status_path}",
            )
    else:
        _require(not reservation_root.exists(), f"unexpected reservation root: {reservation_root}")
    for split_path in root.iterdir():
        if split_path == reservation_root and validate_reservations:
            continue
        if split_path.name in allowed_files and split_path.is_file():
            continue
        if not split_path.is_dir() or split_path.name not in expected_splits:
            unexpected.append(str(split_path))
            continue
        for seed_path in split_path.iterdir():
            match = re.fullmatch(r"seed_(\d+)", seed_path.name)
            if not seed_path.is_dir() or match is None:
                unexpected.append(str(seed_path))
                continue
            seed = int(match.group(1))
            if seed_path.name != f"seed_{seed}" or seed not in set(map(int, seeds)):
                unexpected.append(str(seed_path))
                continue
            for fold_path in seed_path.iterdir():
                fold_match = re.fullmatch(r"fold_(\d+)", fold_path.name)
                if not fold_path.is_dir() or fold_match is None:
                    unexpected.append(str(fold_path))
                    continue
                fold = int(fold_match.group(1))
                identity = (split_path.name, seed, fold)
                if fold_path.name != f"fold_{fold}":
                    unexpected.append(str(fold_path))
                    continue
                if identity not in expected:
                    unexpected.append(str(fold_path))
                elif identity in observed:
                    unexpected.append(str(fold_path))
                else:
                    observed[identity] = fold_path
    _require(not unexpected, f"unexpected or superseded bundle directories: {unexpected[:5]}")
    missing = sorted(expected - set(observed))
    _require(not missing, f"missing canonical bundle directories: {missing[:5]}")
    return observed


def validate_nested_manifest_outputs(directory: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    expected_names = {
        "candidate_scores.npz",
        "test_manifest.json",
        "test_per_query.tsv",
        "test_positive_support.tsv",
        "test_query_manifest.tsv",
    }
    observed_names = {path.name for path in directory.iterdir()}
    _require(observed_names == expected_names, f"nested test directory files do not match schema: {directory}")
    output_hashes = manifest.get("output_sha256")
    _require(isinstance(output_hashes, Mapping), "nested test output hashes missing")
    _require(set(output_hashes) == expected_names - {"test_manifest.json"}, "nested test output hash schema mismatch")
    actual = _payload_hashes(directory, sorted(expected_names))
    for name in expected_names - {"test_manifest.json"}:
        _require(output_hashes.get(name) == actual[name], f"nested test output SHA-256 mismatch for {name}")
    return actual


def _discover_lock_paths(root: Path) -> dict[tuple[str, int], Path]:
    expected = {(split, fold) for split in SUPPORTED_SPLITS for fold in SUPPORTED_FOLDS}
    observed: dict[tuple[str, int], Path] = {}
    for split in SUPPORTED_SPLITS:
        split_root = root / split
        if not split_root.is_dir():
            raise ValueError(f"architecture lock split directory missing: {split_root}")
        names = {path.name for path in split_root.iterdir() if path.is_dir()}
        expected_names = {f"fold_{fold}" for fold in SUPPORTED_FOLDS}
        _require(names == expected_names, f"architecture lock fold coverage mismatch for {split}")
        for fold in SUPPORTED_FOLDS:
            bundle_dir = split_root / f"fold_{fold}" / "locked_bundle"
            expected_files = {"architecture_lock.json", "fold_architecture_summary.tsv"}
            _require(
                bundle_dir.is_dir() and {p.name for p in bundle_dir.iterdir()} == expected_files,
                f"architecture lock bundle files do not match schema: {bundle_dir}",
            )
            observed[(split, fold)] = bundle_dir / "architecture_lock.json"
    _require(set(observed) == expected, "architecture lock split/fold coverage mismatch")
    return observed


def authenticate_locked_test_controls(
    root: Path,
    directories: Mapping[tuple[str, int, int], Path],
) -> dict[str, Any]:
    """Authenticate the immutable whole-run claim, inventory, and execution record."""

    inventory_path = root / "locked_test_score_inventory.json"
    execution_path = root / "outer_test_execution.json"
    claim_path = root / "outer_test_run_claim.json"
    inventory = _read_json(inventory_path, "locked test inventory")
    execution = _read_json(execution_path, "locked test execution")
    claim = _read_json(claim_path, "locked test run claim")
    _require(
        inventory.get("schema") == "unified_dstar_ensemble_locked_test_score_inventory"
        and inventory.get("schema_version") == 4,
        "locked test inventory schema mismatch",
    )
    _require(
        execution.get("schema") == "unified_dstar_ensemble_outer_test_execution"
        and execution.get("schema_version") == 4,
        "locked test execution schema mismatch",
    )
    _require(
        claim.get("schema") == "unified_dstar_ensemble_outer_test_run_claim"
        and claim.get("schema_version") == 1,
        "locked test run claim schema mismatch",
    )
    _require(Path(claim.get("output_root", "")).resolve() == root.resolve(), "run claim output root mismatch")
    records = inventory.get("bundles")
    _require(
        isinstance(records, list)
        and inventory.get("bundle_count") == len(directories) == 30,
        "locked test inventory count mismatch",
    )
    indexed: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for record in records:
        _require(isinstance(record, Mapping), "locked test inventory record schema mismatch")
        identity = (record.get("split"), record.get("seed"), record.get("fold"))
        _require(identity in directories and identity not in indexed, "locked test inventory identity mismatch")
        indexed[identity] = record
    _require(set(indexed) == set(directories), "locked test inventory coverage mismatch")
    for identity, directory in directories.items():
        declared = indexed[identity].get("files")
        expected_names = {
            "bundle_manifest.json",
            "candidate_scores.npz",
            "query_manifest.tsv",
            "test_per_query.tsv",
        }
        _require(isinstance(declared, Mapping) and set(declared) == expected_names, "locked test inventory file schema mismatch")
        for name in expected_names:
            expected_hash = _require_sha(declared[name], f"locked test inventory {name}")
            _require(file_sha256(directory / name) == expected_hash, f"locked test inventory SHA-256 mismatch for {directory / name}")
    claim_sha = file_sha256(claim_path)
    inventory_sha = file_sha256(inventory_path)
    for container, label in ((inventory, "inventory"), (execution, "execution")):
        run_claim = container.get("run_claim")
        _require(isinstance(run_claim, Mapping), f"locked test {label} run claim missing")
        _require(
            run_claim.get("artifact_path") == str(claim_path.resolve())
            and run_claim.get("sha256") == claim_sha
            and run_claim.get("payload") == claim,
            f"locked test {label} run claim authentication mismatch",
        )
    output_inventory = execution.get("output_inventory")
    _require(
        isinstance(output_inventory, Mapping)
        and output_inventory.get("path") == str(inventory_path.resolve())
        and output_inventory.get("sha256") == inventory_sha,
        "locked test execution inventory authentication mismatch",
    )
    fixed = {
        "input_audit_sha256": INPUT_AUDIT_PATH,
        "ensemble_lock_sha256": FIXED_TEST_PROVENANCE["ensemble_lock"],
        "phase1_code_lock_sha256": FIXED_TEST_PROVENANCE["phase1_code_lock"],
        "validation_bundle_lock_sha256": FIXED_TEST_PROVENANCE["validation_bundle_lock"],
    }
    lineage = claim.get("lineage")
    _require(isinstance(lineage, Mapping), "locked test run claim lineage missing")
    for key, path in fixed.items():
        actual = file_sha256(path)
        _require(lineage.get(key) == actual, f"locked test run claim {key} mismatch")
        _require(inventory.get(key) == actual, f"locked test inventory {key} mismatch")
    execution_lock_sha = _require_sha(
        lineage.get("execution_code_lock_sha256"),
        "locked test execution code lock hash",
    )
    candidates = [
        path
        for path in ENSEMBLE_ROOT.glob("scorer_execution_code_lock*.json")
        if path.is_file() and file_sha256(path) == execution_lock_sha
    ]
    _require(len(candidates) == 1, "immutable execution code lock artifact is unresolved or ambiguous")
    _require(
        inventory.get("execution_code_lock_sha256") == execution_lock_sha
        and execution.get("execution_code_lock_sha256") == execution_lock_sha,
        "locked test execution code lock lineage mismatch",
    )
    code_lock = _read_json(candidates[0], "execution code lock")
    historical_code_audit = audit_historical_code_lock(code_lock, project_root=ROOT)
    authenticate_source_tree(execution, project_root=ROOT)
    return {
        "inventory_sha256": inventory_sha,
        "execution_sha256": file_sha256(execution_path),
        "run_claim_sha256": claim_sha,
        "execution_code_lock_path": str(candidates[0].resolve()),
        "execution_code_lock_sha256": execution_lock_sha,
        "historical_code_audit": historical_code_audit,
    }


def audit_inputs(
    *,
    validation_root: Path = VALIDATION_ROOT,
    test_base_root: Path = TEST_BASE_ROOT,
    test_dstar_root: Path = TEST_DSTAR_ROOT,
    architecture_lock_root: Path = ARCHITECTURE_LOCK_ROOT,
    tf_split_root: Path = TF_SPLIT_ROOT,
    ligand_split_root: Path = LIGAND_SPLIT_ROOT,
) -> dict[str, Any]:
    """Audit every frozen Stage A input and return a serializable inventory."""

    validation_dirs = discover_bundle_directories(
        validation_root,
        validate_reservations=True,
    )
    test_control_files = (
        "locked_test_score_inventory.json",
        "outer_test_execution.json",
        "outer_test_run_claim.json",
    )
    test_dirs = discover_bundle_directories(
        test_base_root,
        expected_root_files=test_control_files,
    )
    test_controls = authenticate_locked_test_controls(test_base_root, test_dirs)
    nested_dirs = discover_bundle_directories(test_dstar_root)
    lock_paths = _discover_lock_paths(architecture_lock_root)
    rows: list[dict[str, Any]] = []
    provenance_hash_cache: dict[Path, str] = {}
    for split in SUPPORTED_SPLITS:
        split_root = tf_split_root if split == "TF-50" else ligand_split_root
        for fold in SUPPORTED_FOLDS:
            validation_bundles: list[CandidateScoreBundle] = []
            test_bundles: list[CandidateScoreBundle] = []
            for seed in SUPPORTED_SEEDS:
                identity = (split, seed, fold)
                validation_dir = validation_dirs[identity]
                test_dir = test_dirs[identity]
                nested_dir = nested_dirs[identity]
                validation = load_candidate_score_bundle(validation_dir)
                test = load_candidate_score_bundle(test_dir)
                validation_bundles.append(validation)
                test_bundles.append(test)
                validate_stage_pair(
                    validation=validation,
                    test=test,
                    expected_split=split,
                    expected_seed=seed,
                    expected_fold=fold,
                )
                validation_provenance = authenticate_bundle_provenance(
                    validation,
                    hash_cache=provenance_hash_cache,
                )
                test_provenance = authenticate_bundle_provenance(
                    test,
                    hash_cache=provenance_hash_cache,
                )
                validation_support = audit_fold_support(validation, split_root=split_root, evaluation_role="val")
                test_support = audit_fold_support(test, split_root=split_root, evaluation_role="test")
                manifest_path = nested_dir / "test_manifest.json"
                nested_manifest = _read_json(manifest_path, "nested test manifest")
                nested_output_hashes = validate_nested_manifest_outputs(nested_dir, nested_manifest)
                architecture = validate_architecture_provenance(
                    lock_path=lock_paths[(split, fold)],
                    test_manifest=nested_manifest,
                    split=split,
                    seed=seed,
                    fold=fold,
                )
                nested_scores = audit_nested_dstar(nested_dir / "candidate_scores.npz", test)
                _require(nested_manifest.get("query_count") == len(test.query_ids), "nested test query count mismatch")
                _require(nested_manifest.get("candidate_count") == len(test.candidate_hashes), "nested test candidate count mismatch")
                validation_hashes = _payload_hashes(validation_dir, ("bundle_manifest.json", "candidate_scores.npz", "query_manifest.tsv"))
                test_hashes = _payload_hashes(test_dir, ("bundle_manifest.json", "candidate_scores.npz", "query_manifest.tsv", "test_per_query.tsv"))
                rows.append(
                    {
                        "split": split,
                        "seed": seed,
                        "outer_fold": fold,
                        "status": "pass",
                        "validation_query_count": len(validation.query_ids),
                        "test_query_count": len(test.query_ids),
                        "candidate_count": len(test.candidate_hashes),
                        "validation_bundle_path": str(validation_dir.resolve()),
                        "validation_bundle_manifest_sha256": validation_hashes["bundle_manifest.json"],
                        "validation_payload_sha256": validation_hashes,
                        "validation_source_hashes": dict(validation.source_hashes),
                        "validation_provenance": validation_provenance,
                        "test_bundle_path": str(test_dir.resolve()),
                        "test_bundle_manifest_sha256": test_hashes["bundle_manifest.json"],
                        "test_payload_sha256": test_hashes,
                        "test_source_hashes": dict(test.source_hashes),
                        "test_provenance": test_provenance,
                        "nested_test_manifest_sha256": nested_output_hashes["test_manifest.json"],
                        "nested_output_sha256": nested_output_hashes,
                        "nested_dstar": nested_scores,
                        "architecture_lock": architecture,
                        "validation_fold_support": validation_support,
                        "test_fold_support": test_support,
                    }
                )
            validate_seed_bundles(validation_bundles)
            validate_seed_bundles(test_bundles)
    _require(len(rows) == 30, "audit inventory must contain exactly 30 split/seed/fold rows")
    source_hashes = {
        "audit_script": file_sha256(Path(__file__)),
        "tf_split_manifest": file_sha256(tf_split_root / "outer_edges.tsv"),
        "ligand_split_manifest": file_sha256(ligand_split_root / "manifest.json"),
        **{
            f"locked_test_control.{name}": file_sha256(test_base_root / name)
            for name in test_control_files
        },
    }
    return {
        "schema": "transfer_distilled_residual_stage_a_input_audit",
        "schema_version": 1,
        "status": "pass",
        "expected_splits": list(SUPPORTED_SPLITS),
        "expected_seeds": list(SUPPORTED_SEEDS),
        "expected_folds": list(SUPPORTED_FOLDS),
        "counts": {
            "rows": len(rows),
            "validation_bundles": len(rows),
            "test_bundles": len(rows),
            "nested_dstar_scores": len(rows),
            "architecture_locks": len(lock_paths),
        },
        "source_hashes": source_hashes,
        "locked_test_controls": test_controls,
        "inventory": rows,
    }


def _summary_frame(audit: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for item in audit["inventory"]:
        rows.append(
            {
                "split": item["split"],
                "seed": item["seed"],
                "outer_fold": item["outer_fold"],
                "status": item["status"],
                "validation_queries": item["validation_query_count"],
                "test_queries": item["test_query_count"],
                "candidates": item["candidate_count"],
                "selected_architecture": item["architecture_lock"]["selected_architecture"],
                "validation_bundle_manifest_sha256": item["validation_bundle_manifest_sha256"],
                "test_bundle_manifest_sha256": item["test_bundle_manifest_sha256"],
                "nested_dstar_sha256": item["nested_dstar"]["sha256"],
                "architecture_lock_sha256": item["architecture_lock"]["sha256"],
                "train_manifest_sha256": item["test_fold_support"]["train_sha256"],
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "outer_fold", "seed"])


def _write_outputs(audit: Mapping[str, Any], output: Path) -> Path:
    output = output.resolve()
    tsv_path = output.with_suffix(".tsv")
    output.parent.mkdir(parents=True, exist_ok=True)
    json_bytes = (json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii")
    tsv_bytes = _summary_frame(audit).to_csv(sep="\t", index=False).encode("ascii")
    staging = Path(tempfile.mkdtemp(prefix=".stage_a_audit.", dir=output.parent))
    try:
        (staging / output.name).write_bytes(json_bytes)
        (staging / tsv_path.name).write_bytes(tsv_bytes)
        os.replace(staging / output.name, output)
        os.replace(staging / tsv_path.name, tsv_path)
    finally:
        staging.rmdir()
    return tsv_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, default=VALIDATION_ROOT)
    parser.add_argument("--test-base-root", type=Path, default=TEST_BASE_ROOT)
    parser.add_argument("--test-dstar-root", type=Path, default=TEST_DSTAR_ROOT)
    parser.add_argument("--architecture-lock-root", type=Path, default=ARCHITECTURE_LOCK_ROOT)
    parser.add_argument("--tf-split-root", type=Path, default=TF_SPLIT_ROOT)
    parser.add_argument("--ligand-split-root", type=Path, default=LIGAND_SPLIT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        audit = audit_inputs(
            validation_root=args.validation_root,
            test_base_root=args.test_base_root,
            test_dstar_root=args.test_dstar_root,
            architecture_lock_root=args.architecture_lock_root,
            tf_split_root=args.tf_split_root,
            ligand_split_root=args.ligand_split_root,
        )
        tsv_path = _write_outputs(audit, args.output)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Stage A input audit: FAIL: {error}", file=sys.stderr)
        return 1
    counts = audit["counts"]
    print(
        "Stage A input audit: PASS; "
        f"{counts['validation_bundles']} validation bundles, "
        f"{counts['test_bundles']} test bundles, "
        f"{counts['nested_dstar_scores']} nested Dstar scores, "
        f"{counts['architecture_locks']} architecture locks."
    )
    print(f"JSON: {args.output.resolve()}")
    print(f"TSV: {tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
