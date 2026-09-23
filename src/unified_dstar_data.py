"""Strict data and feature-cache contracts for the unified D* baseline."""

from __future__ import annotations

import hashlib
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch


ROLES = ("train", "val", "test")
REQUIRED_EDGE_COLUMNS = (
    "edge_id",
    "ligand_key",
    "sequence_md5",
    "split_unit",
    "split_role",
)
OPTIONAL_LIGAND_COLUMNS = ("canonical_smiles", "canonical_ion_label")
ION_KEY_PREFIX = "ion:"
FEATURE_ORDER = (
    "charge",
    "atomic_number",
    "period",
    "group",
    "transition_metal",
    "redox_active",
    "essential_metal",
    "toxic_heavy_metal",
    "is_monovalent",
    "valence_ambiguous",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOLFORMER_PATH = (
    PROJECT_ROOT
    / "data/model_training/v66/features/molformer_internal_v66_ligands.pkl"
)
DEFAULT_ESM_PATH = (
    PROJECT_ROOT
    / "data/model_training/v66/features/"
    "esm2_150M_curated_prokaryotic_tf_v4_regprecise_all.pkl"
)
DEFAULT_ECFP_PATH = PROJECT_ROOT / "data/processed/modeling/ecfp4_cache.pkl"
DEFAULT_ION_DESCRIPTORS_PATH = (
    PROJECT_ROOT / "data/processed/modeling/ion_descriptors.json"
)


@dataclass(frozen=True)
class FoldFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class QueryEpisode:
    ligand_key: str
    positive_hashes: tuple[str, ...]
    unlabeled_hashes: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationQuery:
    ligand_key: str
    train_positive_hashes: tuple[str, ...]
    relevant_hashes: tuple[str, ...]
    candidate_hashes: tuple[str, ...]
    split_unit: str


def _read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{context} missing required columns: {', '.join(missing)}")


def normalize_role_frame(
    frame: pd.DataFrame,
    *,
    role: str,
    ligand_source: str,
    split_unit_source: str,
    context: str,
    enforce_unique_edges: bool = True,
) -> pd.DataFrame:
    """Normalize a role frame, including the canonical explicit-ion key policy."""

    if not isinstance(enforce_unique_edges, bool):
        raise ValueError("enforce_unique_edges must be boolean")
    source_columns = ("edge_id", ligand_source, "sequence_md5", split_unit_source)
    _require_columns(frame, source_columns, context)

    if "split_role" in frame.columns:
        observed_roles = set(frame["split_role"].dropna().astype(str))
        if observed_roles != {role}:
            raise ValueError(
                f"{context} split_role values must be exactly {role!r}; "
                f"found {sorted(observed_roles)!r}"
            )

    output = pd.DataFrame(
        {
            "edge_id": frame["edge_id"],
            "ligand_key": frame[ligand_source],
            "sequence_md5": frame["sequence_md5"],
            "split_unit": frame[split_unit_source],
            "split_role": role,
        }
    )
    for column in OPTIONAL_LIGAND_COLUMNS:
        if column in frame.columns:
            output[column] = frame[column]

    if "canonical_ion_label" in output.columns:
        ion_labels = output["canonical_ion_label"]
        has_ion_label = ion_labels.notna() & ion_labels.astype(str).str.strip().ne("")
        if "canonical_smiles" in output.columns:
            smiles = output["canonical_smiles"]
            lacks_smiles = smiles.isna() | smiles.astype(str).str.strip().eq("")
        else:
            lacks_smiles = pd.Series(True, index=output.index)
        explicit_ion = has_ion_label & lacks_smiles
        output.loc[explicit_ion, "ligand_key"] = (
            ION_KEY_PREFIX
            + ion_labels.loc[explicit_ion].astype(str).str.strip()
        )

    for column in REQUIRED_EDGE_COLUMNS:
        if output[column].isna().any() or output[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{context} has null or blank values in required column {column}")

    duplicate_edge_ids = output["edge_id"].duplicated(keep=False)
    if duplicate_edge_ids.any():
        examples = output.loc[duplicate_edge_ids, "edge_id"].astype(str).unique()
        raise ValueError(
            f"{context} has duplicate edge_id values within role {role}: "
            f"{examples[:5].tolist()}"
        )

    duplicate = output.duplicated(["ligand_key", "sequence_md5"], keep=False)
    if enforce_unique_edges and duplicate.any():
        examples = output.loc[duplicate, ["ligand_key", "sequence_md5"]].head(3)
        raise ValueError(
            f"{context} has duplicate ligand-sequence edges within role {role}: "
            f"{examples.to_dict('records')}"
        )

    return output.reset_index(drop=True)


# Backward-compatible private alias for callers outside this module.
_normalize_role_frame = normalize_role_frame


def _validate_edge_uniqueness(frames: dict[str, pd.DataFrame]) -> None:
    combined = pd.concat(
        [frames[role].loc[:, REQUIRED_EDGE_COLUMNS] for role in ROLES],
        ignore_index=True,
    )
    duplicate_edge_ids = combined["edge_id"].duplicated(keep=False)
    if duplicate_edge_ids.any():
        examples = combined.loc[duplicate_edge_ids, "edge_id"].astype(str).unique()
        raise ValueError(
            f"duplicate edge_id values across roles: {examples[:5].tolist()}"
        )

    duplicate_pairs = combined.duplicated(
        ["ligand_key", "sequence_md5"], keep=False
    )
    if duplicate_pairs.any():
        examples = combined.loc[
            duplicate_pairs, ["ligand_key", "sequence_md5"]
        ].drop_duplicates()
        raise ValueError(
            "duplicate ligand-sequence pairs across roles: "
            f"{examples.head(5).to_dict('records')}"
        )


def _validate_split_unit_disjoint(frames: dict[str, pd.DataFrame]) -> None:
    units = {role: set(frame["split_unit"].astype(str)) for role, frame in frames.items()}
    for index, left in enumerate(ROLES):
        for right in ROLES[index + 1 :]:
            overlap = units[left] & units[right]
            if overlap:
                examples = sorted(overlap)[:5]
                raise ValueError(
                    f"split_unit overlap between {left} and {right}: {examples}"
                )


def load_fold_frames(
    split_name: str, split_root: str | Path, outer_fold: int
) -> FoldFrames:
    """Load one TF-50 or Ligand-Morgan-0.5 fold into a shared schema."""

    root = Path(split_root)
    normalized: dict[str, pd.DataFrame] = {}

    if split_name == "TF-50":
        source = _read_tsv(root / "outer_edges.tsv")
        _require_columns(source, ("outer_fold", "split_role"), str(root / "outer_edges.tsv"))
        selected = source.loc[source["outer_fold"] == str(outer_fold)].copy()
        observed_roles = set(selected["split_role"].dropna().astype(str))
        unknown_roles = observed_roles - set(ROLES)
        if unknown_roles or selected["split_role"].isna().any():
            raise ValueError(
                f"TF-50 split_role contains invalid values: {sorted(unknown_roles)!r}"
            )
        for role in ROLES:
            role_frame = selected.loc[selected["split_role"].astype(str) == role].copy()
            normalized[role] = normalize_role_frame(
                role_frame,
                role=role,
                ligand_source="ligand_key",
                split_unit_source="split_unit",
                context=f"TF-50 fold {outer_fold} {role}",
            )
    elif split_name == "Ligand-Morgan-0.5":
        fold_root = root / f"fold_{outer_fold}"
        for role in ROLES:
            path = fold_root / f"{role}.tsv"
            role_frame = _read_tsv(path)
            normalized[role] = normalize_role_frame(
                role_frame,
                role=role,
                ligand_source="ligand_key_for_c_light",
                split_unit_source="cluster_id",
                context=str(path),
            )
    else:
        raise ValueError(
            "split_name must be 'TF-50' or 'Ligand-Morgan-0.5', "
            f"got {split_name!r}"
        )

    _validate_edge_uniqueness(normalized)
    _validate_split_unit_disjoint(normalized)
    return FoldFrames(
        train=normalized["train"],
        val=normalized["val"],
        test=normalized["test"],
    )


def load_candidate_table(path: str | Path) -> pd.DataFrame:
    """Load the candidate library and enforce its hash/sequence identity contract."""

    frame = _read_tsv(path)
    _require_columns(frame, ("sequence_md5", "protein_sequence"), str(path))
    for column in ("sequence_md5", "protein_sequence"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"candidate table has null or blank {column} values")
    if frame["sequence_md5"].duplicated().any():
        duplicates = (
            frame.loc[frame["sequence_md5"].duplicated(keep=False), "sequence_md5"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            f"candidate table requires unique sequence_md5 values; duplicates: "
            f"{duplicates[:5]}"
        )
    if frame["protein_sequence"].duplicated().any():
        duplicates = (
            frame.loc[
                frame["protein_sequence"].duplicated(keep=False),
                "protein_sequence",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            "candidate table requires unique protein_sequence values; "
            f"duplicate examples: {duplicates[:5]}"
        )
    computed_hashes = frame["protein_sequence"].map(
        lambda sequence: hashlib.md5(sequence.encode()).hexdigest()
    )
    mismatch = frame["sequence_md5"] != computed_hashes
    if mismatch.any():
        examples = pd.DataFrame(
            {
                "sequence_md5": frame.loc[mismatch, "sequence_md5"],
                "computed_md5": computed_hashes.loc[mismatch],
            }
        )
        raise ValueError(
            "candidate table sequence_md5 mismatch for protein_sequence: "
            f"{examples.head(5).to_dict('records')}"
        )
    return frame


def _validated_candidate_hashes(candidate_hashes: Sequence[str]) -> list[str]:
    candidates = [str(candidate) for candidate in candidate_hashes]
    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate_hashes contains duplicate candidate hashes")
    if any(not candidate.strip() for candidate in candidates):
        raise ValueError("candidate_hashes contains a blank candidate hash")
    return candidates


def build_query_candidates(
    candidate_hashes: Sequence[str], train_positive_hashes: Iterable[str]
) -> list[str]:
    """Remove only same-query training positives from a candidate library."""

    candidates = _validated_candidate_hashes(candidate_hashes)
    positives = {str(value) for value in train_positive_hashes}
    unknown = positives - set(candidates)
    if unknown:
        raise ValueError(
            f"unknown train positive hashes not in candidate library: {sorted(unknown)[:5]}"
        )
    return [candidate for candidate in candidates if candidate not in positives]


def _validate_edge_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    _require_columns(frame, columns, name)
    for column in columns:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} has null or blank values in {column}")


def _validate_split_roles(
    frame: pd.DataFrame, allowed_roles: Sequence[str], name: str
) -> None:
    if "split_role" not in frame.columns:
        return
    roles = frame["split_role"]
    invalid = roles.isna() | ~roles.astype(str).isin(allowed_roles)
    if invalid.any():
        observed = sorted(roles.loc[invalid].dropna().astype(str).unique())
        allowed = ", ".join(repr(role) for role in allowed_roles)
        raise ValueError(
            f"{name} split_role must contain only {allowed}; found {observed!r}"
        )


def build_training_episodes(
    train_edges: pd.DataFrame,
    candidate_hashes: Sequence[str],
    unknown_per_query: int,
    seed: int,
) -> list[QueryEpisode]:
    """Construct deterministic query episodes using training edges only."""

    _validate_edge_columns(
        train_edges, ("ligand_key", "sequence_md5"), "train_edges"
    )
    _validate_split_roles(train_edges, ("train",), "train_edges")
    if not isinstance(unknown_per_query, int) or unknown_per_query < 0:
        raise ValueError("unknown_per_query must be a non-negative integer")

    candidates = _validated_candidate_hashes(candidate_hashes)
    rng = random.Random(seed)
    episodes: list[QueryEpisode] = []
    ligand_keys = sorted(train_edges["ligand_key"].astype(str).unique())
    for ligand_key in ligand_keys:
        group = train_edges.loc[
            train_edges["ligand_key"].astype(str) == ligand_key, "sequence_md5"
        ]
        positives = tuple(sorted(set(group.astype(str))))
        pool = build_query_candidates(candidates, positives)
        if len(pool) < unknown_per_query:
            raise ValueError(
                f"insufficient unlabeled candidates for ligand {ligand_key!r}: "
                f"need {unknown_per_query}, found {len(pool)}"
            )
        episodes.append(
            QueryEpisode(
                ligand_key=ligand_key,
                positive_hashes=positives,
                unlabeled_hashes=tuple(rng.sample(pool, unknown_per_query)),
            )
        )
    return episodes


def build_evaluation_queries(
    train_edges: pd.DataFrame,
    evaluation_edges: pd.DataFrame,
    candidate_hashes: Sequence[str],
) -> list[EvaluationQuery]:
    """Build full-library evaluation queries without consulting held-out labels."""

    _validate_edge_columns(
        train_edges, ("ligand_key", "sequence_md5"), "train_edges"
    )
    _validate_edge_columns(
        evaluation_edges,
        ("ligand_key", "sequence_md5", "split_unit"),
        "evaluation_edges",
    )
    _validate_split_roles(train_edges, ("train",), "train_edges")
    _validate_split_roles(evaluation_edges, ("val", "test"), "evaluation_edges")
    candidates = _validated_candidate_hashes(candidate_hashes)
    train_ligands = train_edges["ligand_key"].astype(str)
    evaluation_ligands = evaluation_edges["ligand_key"].astype(str)
    queries: list[EvaluationQuery] = []
    # Preserve the explicit tuple API while sharing identical full-library filters.
    candidate_cache: dict[tuple[str, ...], tuple[str, ...]] = {}

    for ligand_key in sorted(evaluation_ligands.unique()):
        train_group = train_edges.loc[
            train_ligands == ligand_key, "sequence_md5"
        ].astype(str)
        evaluation_group = evaluation_edges.loc[evaluation_ligands == ligand_key]
        train_positives = tuple(sorted(set(train_group)))
        relevant = tuple(sorted(set(evaluation_group["sequence_md5"].astype(str))))
        query_candidates = candidate_cache.get(train_positives)
        if query_candidates is None:
            query_candidates = tuple(
                build_query_candidates(candidates, train_positives)
            )
            candidate_cache[train_positives] = query_candidates
        missing_relevant = set(relevant) - set(query_candidates)
        if missing_relevant:
            raise ValueError(
                f"relevant hashes for ligand {ligand_key!r} are absent after "
                f"train-positive filtering: {sorted(missing_relevant)}"
            )
        split_unit = ";".join(
            sorted(set(evaluation_group["split_unit"].astype(str)))
        )
        queries.append(
            EvaluationQuery(
                ligand_key=ligand_key,
                train_positive_hashes=train_positives,
                relevant_hashes=relevant,
                candidate_hashes=query_candidates,
                split_unit=split_unit,
            )
        )
    return queries


def _load_pickle_mapping(path: Path, cache_name: str) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{cache_name} cache must contain a dictionary")
    result: dict[str, Any] = {}
    for key, value in raw.items():
        normalized_key = str(key)
        if normalized_key in result:
            raise ValueError(
                f"{cache_name} cache has keys that collide after string conversion: "
                f"{normalized_key!r}"
            )
        result[normalized_key] = value
    return result


def _validated_vector_cache(
    raw: dict[str, Any], *, dimension: int, cache_name: str
) -> dict[str, np.ndarray]:
    validated: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        try:
            vector = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{cache_name} cache entry {key!r} is not a numeric vector"
            ) from error
        if vector.ndim != 1 or vector.shape[0] != dimension:
            raise ValueError(
                f"{cache_name} cache entry {key!r} has dimension "
                f"{tuple(vector.shape)}, expected ({dimension},)"
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"{cache_name} cache entry {key!r} is not finite")
        validated[key] = vector
    return validated


def _load_ion_descriptors(path: Path) -> dict[str, np.ndarray]:
    with path.open() as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("ion descriptor JSON must contain an object")

    descriptor_data = raw.get("descriptors", raw)
    if not isinstance(descriptor_data, dict):
        raise ValueError("ion descriptor JSON 'descriptors' must contain an object")
    if descriptor_data is raw and "feature_order" in descriptor_data:
        descriptor_data = {
            key: value for key, value in descriptor_data.items() if key != "feature_order"
        }

    descriptors: dict[str, np.ndarray] = {}
    for label, value in descriptor_data.items():
        if isinstance(value, dict):
            missing = [feature for feature in FEATURE_ORDER if feature not in value]
            if missing:
                raise ValueError(
                    f"ion descriptor {label!r} missing features: {missing}"
                )
            ordered = [value[feature] for feature in FEATURE_ORDER]
        elif isinstance(value, (list, tuple)):
            source_order = raw.get("feature_order")
            if source_order is None or len(source_order) != len(value):
                raise ValueError(
                    f"ion descriptor {label!r} list requires a matching feature_order"
                )
            by_name = dict(zip(source_order, value))
            missing = [feature for feature in FEATURE_ORDER if feature not in by_name]
            if missing:
                raise ValueError(
                    f"ion descriptor {label!r} missing features: {missing}"
                )
            ordered = [by_name[feature] for feature in FEATURE_ORDER]
        else:
            raise ValueError(
                f"ion descriptor {label!r} must be an object or ordered list"
            )
        vector = np.asarray(ordered, dtype=np.float32)
        if vector.shape != (len(FEATURE_ORDER),) or not np.isfinite(vector).all():
            raise ValueError(f"ion descriptor {label!r} must have 10 finite features")
        descriptors[str(label)] = vector
    return descriptors


class FeatureStore:
    """Validated tensor access to frozen ligand and protein feature caches."""

    def __init__(
        self,
        molformer_path: str | Path = DEFAULT_MOLFORMER_PATH,
        esm_path: str | Path = DEFAULT_ESM_PATH,
        ecfp_path: str | Path = DEFAULT_ECFP_PATH,
        ion_descriptors_path: str | Path = DEFAULT_ION_DESCRIPTORS_PATH,
        *,
        molformer_dim: int = 768,
        ecfp_dim: int = 2048,
        esm_dim: int = 640,
        missing_ion_policy: str = "error",
    ) -> None:
        if missing_ion_policy not in {"error", "zero"}:
            raise ValueError(
                "missing_ion_policy must be 'error' or 'zero', "
                f"got {missing_ion_policy!r}"
            )
        dimensions = {
            "molformer": molformer_dim,
            "ecfp": ecfp_dim,
            "esm": esm_dim,
        }
        for name, dimension in dimensions.items():
            if not isinstance(dimension, int) or dimension <= 0:
                raise ValueError(f"{name}_dim must be a positive integer")

        self.molformer_dim = molformer_dim
        self.ecfp_dim = ecfp_dim
        self.esm_dim = esm_dim
        self.missing_ion_policy = missing_ion_policy
        self._paths = {
            "molformer": Path(molformer_path),
            "esm": Path(esm_path),
            "ecfp": Path(ecfp_path),
            "ion_descriptors": Path(ion_descriptors_path),
        }
        self._molformer = _validated_vector_cache(
            _load_pickle_mapping(self._paths["molformer"], "MoLFormer"),
            dimension=molformer_dim,
            cache_name="MoLFormer",
        )
        self._esm = _validated_vector_cache(
            _load_pickle_mapping(self._paths["esm"], "ESM"),
            dimension=esm_dim,
            cache_name="ESM",
        )
        self._ecfp = _validated_vector_cache(
            _load_pickle_mapping(self._paths["ecfp"], "ECFP"),
            dimension=ecfp_dim,
            cache_name="ECFP",
        )
        self._ion_descriptors = _load_ion_descriptors(
            self._paths["ion_descriptors"]
        )
        self._missing: dict[str, set[str]] = {
            "molformer": set(),
            "ecfp": set(),
            "esm": set(),
            "ion_descriptor": set(),
            "all_ligand_features": set(),
        }
        self._zero_molformer = np.zeros(molformer_dim, dtype=np.float32)
        self._zero_ecfp = np.zeros(ecfp_dim, dtype=np.float32)
        self._zero_ion = np.zeros(len(FEATURE_ORDER), dtype=np.float32)

    @staticmethod
    def _tensor(rows: list[np.ndarray], dimension: int) -> torch.Tensor:
        if not rows:
            return torch.empty((0, dimension), dtype=torch.float32)
        return torch.from_numpy(np.stack(rows).astype(np.float32, copy=False))

    def ligand_batch(
        self, keys: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        molformer_rows: list[np.ndarray] = []
        ecfp_rows: list[np.ndarray] = []
        ion_rows: list[np.ndarray] = []
        ion_mask: list[bool] = []

        for raw_key in keys:
            key = str(raw_key)
            if key.startswith(ION_KEY_PREFIX):
                ion_label = key[len(ION_KEY_PREFIX) :].strip()
                if not ion_label:
                    self._missing["ion_descriptor"].add(key)
                    raise KeyError(f"ion descriptor missing for empty ion key {key!r}")
                descriptor = self._ion_descriptors.get(ion_label)
                if descriptor is None:
                    self._missing["ion_descriptor"].add(key)
                    if self.missing_ion_policy == "error":
                        raise KeyError(
                            f"ion descriptor missing for key {key!r}, "
                            f"label {ion_label!r}"
                        )
                    descriptor = self._zero_ion
                molformer_rows.append(self._zero_molformer)
                ecfp_rows.append(self._zero_ecfp)
                ion_rows.append(descriptor)
                ion_mask.append(True)
                continue

            molformer = self._molformer.get(key)
            ecfp = self._ecfp.get(key)
            if molformer is None:
                self._missing["molformer"].add(key)
            if ecfp is None:
                self._missing["ecfp"].add(key)
            if molformer is None and ecfp is None:
                self._missing["all_ligand_features"].add(key)
                raise KeyError(
                    f"regular ligand {key!r} is missing both MoLFormer and ECFP features"
                )

            molformer_rows.append(
                molformer if molformer is not None else self._zero_molformer
            )
            ecfp_rows.append(ecfp if ecfp is not None else self._zero_ecfp)
            ion_rows.append(self._zero_ion)
            ion_mask.append(False)

        return (
            self._tensor(molformer_rows, self.molformer_dim),
            self._tensor(ecfp_rows, self.ecfp_dim),
            self._tensor(ion_rows, len(FEATURE_ORDER)),
            torch.tensor(ion_mask, dtype=torch.bool),
        )

    def protein_batch(self, hashes: Sequence[str]) -> torch.Tensor:
        rows: list[np.ndarray] = []
        missing: list[str] = []
        for raw_hash in hashes:
            sequence_md5 = str(raw_hash)
            vector = self._esm.get(sequence_md5)
            if vector is None:
                self._missing["esm"].add(sequence_md5)
                missing.append(sequence_md5)
            else:
                rows.append(vector)
        if missing:
            raise KeyError(f"ESM embeddings missing for protein hashes: {missing[:5]}")
        return self._tensor(rows, self.esm_dim)

    def audit(self) -> dict[str, Any]:
        """Return serializable cache provenance and observed missing-key counts."""

        return {
            "missing_ion_policy": self.missing_ion_policy,
            "cache_paths": {
                name: str(path) for name, path in self._paths.items()
            },
            "dimensions": {
                "molformer": self.molformer_dim,
                "ecfp": self.ecfp_dim,
                "esm": self.esm_dim,
                "ion": len(FEATURE_ORDER),
            },
            "cache_entry_counts": {
                "molformer": len(self._molformer),
                "ecfp": len(self._ecfp),
                "esm": len(self._esm),
                "ion": len(self._ion_descriptors),
            },
            "missing_key_counts": {
                name: len(keys) for name, keys in self._missing.items()
            },
            "missing_keys": {
                name: sorted(keys) for name, keys in self._missing.items()
            },
        }
