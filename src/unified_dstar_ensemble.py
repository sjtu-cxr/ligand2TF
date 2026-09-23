from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping
import ctypes
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_EVEN
import errno
from hashlib import sha256
import io
from itertools import product
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
import zipfile

import numpy as np


SUPPORTED_SPLITS = ("TF-50", "Ligand-Morgan-0.5")
SUPPORTED_SEEDS = (42, 20260717, 20260718)
SUPPORTED_FOLDS = tuple(range(5))
BUNDLE_SCHEMA = "unified_dstar_candidate_score_bundle"
BUNDLE_SCHEMA_VERSION = 2
BUNDLE_FILENAMES = (
    "candidate_scores.npz",
    "query_manifest.tsv",
    "bundle_manifest.json",
)
DERIVED_CHANNELS = frozenset({"Ensemble"})
BUNDLE_TRUST_MODEL = MappingProxyType(
    {
        "bundle_manifest": "checksum_and_lineage_record",
        "trust_root": "external_lock_or_audit_hash",
    }
)
_WEIGHT_QUANTUM = Decimal("0.000000000001")
BASELINES = {
    "TF-50": {
        "h10": 0.076487,
        "mrr": 0.031421,
        "unique_ligand_h10": 0.055254,
        "unique_ligand_mrr": 0.022784,
    },
    "Ligand-Morgan-0.5": {
        "h10": 0.250471,
        "mrr": 0.174377,
    },
}


def _finite_acceptance_metric(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and in [0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite and in [0, 1]") from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return number


def acceptance_decision(
    *,
    split: str,
    h10: float,
    mrr: float,
    unique_ligand_h10: float | None = None,
    unique_ligand_mrr: float | None = None,
) -> bool:
    """Apply strict primary and TF unique-ligand robustness thresholds."""

    if split not in BASELINES:
        raise ValueError(f"unsupported split: {split}")
    actual_h10 = _finite_acceptance_metric(h10, f"{split} H@10")
    actual_mrr = _finite_acceptance_metric(mrr, f"{split} MRR")
    baseline = BASELINES[split]
    primary = actual_h10 > baseline["h10"] and actual_mrr > baseline["mrr"]
    if split == "Ligand-Morgan-0.5":
        if unique_ligand_h10 is not None or unique_ligand_mrr is not None:
            raise ValueError("ligand split has no unique-ligand robustness gate")
        return primary
    if unique_ligand_h10 is None or unique_ligand_mrr is None:
        raise ValueError("TF-50 requires both unique-ligand robustness metrics")
    robust_h10 = _finite_acceptance_metric(
        unique_ligand_h10, "TF-50 unique-ligand H@10"
    )
    robust_mrr = _finite_acceptance_metric(
        unique_ligand_mrr, "TF-50 unique-ligand MRR"
    )
    return (
        primary
        and robust_h10 >= baseline["unique_ligand_h10"]
        and robust_mrr >= baseline["unique_ligand_mrr"]
    )


def _canonical_weights(
    weights: Mapping[str, float],
) -> Mapping[str, float]:
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("weights must be a nonempty mapping")
    decimal_weights: dict[str, Decimal] = {}
    for name, weight in weights.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(weight, bool)
            or not isinstance(weight, Real)
        ):
            raise ValueError("weights must map channel names to real values")
        value = float(weight)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("weights must be finite and nonnegative")
        decimal_weights[name] = Decimal(str(value))

    total = sum(decimal_weights.values(), Decimal(0))
    if total <= 0:
        raise ValueError("at least one channel weight must be positive")
    exact = {
        name: decimal_weights[name] / total
        for name in sorted(decimal_weights)
    }
    rounded = {
        name: value.quantize(
            _WEIGHT_QUANTUM,
            rounding=ROUND_HALF_EVEN,
        )
        for name, value in exact.items()
    }
    residual = Decimal(1) - sum(rounded.values(), Decimal(0))
    if residual:
        adjustment_name = min(
            exact,
            key=lambda name: (-exact[name], name),
        )
        rounded[adjustment_name] += residual
    return MappingProxyType(
        {name: float(rounded[name]) for name in sorted(rounded)}
    )


@dataclass(frozen=True)
class FusionConfig:
    k: int | None
    weights: Mapping[str, float]
    is_baseline_fallback: bool = False
    config_id: str = field(init=False)
    nonzero_channel_count: int = field(init=False)
    lexicographic_key: tuple[tuple[str, float], ...] = field(init=False)

    def __post_init__(self) -> None:
        if type(self.is_baseline_fallback) is not bool:
            raise ValueError("is_baseline_fallback must be a boolean")
        if self.is_baseline_fallback:
            if self.k is not None or dict(self.weights):
                raise ValueError(
                    "baseline fallback must have no k or channel weights"
                )
            frozen_weights: Mapping[str, float] = MappingProxyType({})
            lexicographic_key: tuple[tuple[str, float], ...] = ()
            config_id = "baseline"
        else:
            if (
                isinstance(self.k, bool)
                or not isinstance(self.k, Integral)
                or self.k <= 0
            ):
                raise ValueError("fusion k must be a positive integer")
            object.__setattr__(self, "k", int(self.k))
            frozen_weights = _canonical_weights(self.weights)
            lexicographic_key = tuple(frozen_weights.items())
            payload = json.dumps(
                {
                    "k": self.k,
                    "weights": [
                        [name, f"{weight:.12f}"]
                        for name, weight in lexicographic_key
                    ],
                },
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            config_id = (
                f"weighted_rrf:k={self.k}:sha256:"
                f"{sha256(payload).hexdigest()}"
            )

        object.__setattr__(self, "weights", frozen_weights)
        object.__setattr__(
            self,
            "nonzero_channel_count",
            sum(weight > 0.0 for weight in frozen_weights.values()),
        )
        object.__setattr__(
            self,
            "lexicographic_key",
            lexicographic_key,
        )
        object.__setattr__(self, "config_id", config_id)

    @classmethod
    def baseline_fallback(cls) -> FusionConfig:
        return cls(k=None, weights={}, is_baseline_fallback=True)

    def __hash__(self) -> int:
        return hash(
            (
                self.k,
                self.lexicographic_key,
                self.is_baseline_fallback,
            )
        )


BASELINE_FUSION_CONFIG = FusionConfig.baseline_fallback()


def _channel_names(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a collection of channel names")
    names = tuple(values)
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(f"{label} must contain unique nonempty names")
    return tuple(sorted(names))


def generate_fusion_grid(
    channels: tuple[str, ...],
    classical_channels: tuple[str, ...],
    weight_values: tuple[float, ...] = (0, 0.1, 0.25, 0.5, 0.75, 1),
    ks: tuple[int, ...] = (5, 10, 20, 60),
) -> tuple[FusionConfig, ...]:
    """Generate canonical weighted-RRF configurations."""
    channel_names = _channel_names(channels, "channels")
    classical_names = _channel_names(
        classical_channels,
        "classical_channels",
    )
    if not set(classical_names).issubset(channel_names):
        raise ValueError("classical channels must be fusion channels")
    if isinstance(weight_values, (str, bytes)):
        raise ValueError("weight_values must be a nonempty collection")
    try:
        values = tuple(weight_values)
    except TypeError as error:
        raise ValueError(
            "weight_values must be a nonempty collection"
        ) from error
    if not values:
        raise ValueError("weight_values must not be empty")
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(
                "weight_values must contain finite nonnegative real numbers"
            )
    if not any(value > 0 for value in values):
        raise ValueError("weight_values must include a positive value")

    if isinstance(ks, (str, bytes)):
        raise ValueError("ks must be a nonempty collection")
    try:
        k_values = tuple(ks)
    except TypeError as error:
        raise ValueError("ks must be a nonempty collection") from error
    if not k_values:
        raise ValueError("ks must not be empty")
    if any(
        isinstance(k, bool)
        or not isinstance(k, Integral)
        or k <= 0
        for k in k_values
    ):
        raise ValueError("ks must contain positive integers")

    configurations: dict[str, FusionConfig] = {}
    for k in k_values:
        for combination in product(values, repeat=len(channel_names)):
            raw_weights = dict(zip(channel_names, combination))
            if not any(raw_weights[name] > 0 for name in classical_names):
                continue
            config = FusionConfig(k=k, weights=raw_weights)
            if not any(
                config.weights[name] > 0.0 for name in classical_names
            ):
                continue
            configurations.setdefault(config.config_id, config)
    return tuple(
        sorted(
            configurations.values(),
            key=lambda config: (config.k, config.lexicographic_key),
        )
    )


def select_configuration(
    table: object,
    baseline_h10: float,
    baseline_mrr: float,
) -> FusionConfig:
    """Select a dual-metric improvement with deterministic tie breaking."""
    baselines = (baseline_h10, baseline_mrr)
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
        or value <= 0
        for value in baselines
    ):
        raise ValueError("baseline denominators must be positive and finite")
    if not hasattr(table, "columns") or not hasattr(table, "to_dict"):
        raise ValueError("table must provide columns and row records")
    required_columns = {"config", "h10", "mrr"}
    columns = set(table.columns)
    if not required_columns.issubset(columns):
        raise ValueError(
            "selection table columns must include config, h10, and mrr"
        )
    records = table.to_dict("records")

    candidates: list[
        tuple[
            tuple[
                float,
                float,
                int,
                int,
                tuple[tuple[str, float], ...],
            ],
            FusionConfig,
        ]
    ] = []
    observed_ids: set[str] = set()
    for record in records:
        config = record["config"]
        if (
            not isinstance(config, FusionConfig)
            or config.is_baseline_fallback
        ):
            raise ValueError(
                "selection table configs must be active FusionConfig values"
            )
        if config.config_id in observed_ids:
            raise ValueError("selection table contains duplicate configs")
        observed_ids.add(config.config_id)

        metric_values = (record["h10"], record["mrr"])
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(float(value))
            for value in metric_values
        ):
            raise ValueError("selection metrics must be finite real values")
        gain_h10 = (
            float(record["h10"]) - float(baseline_h10)
        ) / float(baseline_h10)
        gain_mrr = (
            float(record["mrr"]) - float(baseline_mrr)
        ) / float(baseline_mrr)
        if gain_h10 <= 0.0 or gain_mrr <= 0.0:
            continue
        minimum_gain = min(gain_h10, gain_mrr)
        mean_gain = gain_h10 / 2.0 + gain_mrr / 2.0
        key = (
            -minimum_gain,
            -mean_gain,
            config.nonzero_channel_count,
            config.k,
            config.lexicographic_key,
        )
        candidates.append((key, config))

    if not candidates:
        return BASELINE_FUSION_CONFIG
    return min(candidates, key=lambda item: item[0])[1]


@dataclass(frozen=True)
class EnsembleBundleId:
    split: str
    seed: int
    outer_fold: int
    stage: str

    def relative_dir(self) -> Path:
        if self.split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split: {self.split}")
        if (
            isinstance(self.seed, bool)
            or self.seed not in SUPPORTED_SEEDS
        ):
            raise ValueError(f"unsupported seed: {self.seed!r}")
        if (
            isinstance(self.outer_fold, bool)
            or self.outer_fold not in SUPPORTED_FOLDS
        ):
            raise ValueError(f"unsupported outer fold: {self.outer_fold!r}")
        if self.stage not in {"validation", "test"}:
            raise ValueError(f"unsupported stage: {self.stage}")
        root = (
            "validation_scores"
            if self.stage == "validation"
            else "locked_test_scores"
        )
        return (
            Path(root)
            / self.split
            / f"seed_{self.seed}"
            / f"fold_{self.outer_fold}"
        )


def file_sha256(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_channels(
    split: str, *, ligand_supported: bool | None = None
) -> tuple[str, ...]:
    if split == "TF-50":
        if ligand_supported is None:
            raise ValueError("TF-50 requires ligand_supported")
        return ("S", "Dstar", "D") if ligand_supported else ("Dstar", "D")
    if split == "Ligand-Morgan-0.5":
        return ("C", "P", "Dstar", "D")
    raise ValueError(f"unsupported split: {split}")


def rank_descending(
    scores: np.ndarray,
    candidate_hashes: tuple[str, ...],
) -> np.ndarray:
    """Rank finite scores per query, resolving ties by candidate hash."""
    matrix = np.asarray(scores)
    if matrix.ndim != 2:
        raise ValueError("scores must be a two-dimensional matrix")
    if (
        not np.issubdtype(matrix.dtype, np.number)
        or np.issubdtype(matrix.dtype, np.complexfloating)
    ):
        raise ValueError("scores must be real numeric values")
    if not isinstance(candidate_hashes, tuple):
        candidate_hashes = tuple(candidate_hashes)
    if matrix.shape[1] != len(candidate_hashes):
        raise ValueError("candidate hash count must match score columns")
    if any(
        not isinstance(candidate_hash, str) or not candidate_hash
        for candidate_hash in candidate_hashes
    ):
        raise ValueError("candidate hashes must be nonempty strings")
    if len(set(candidate_hashes)) != len(candidate_hashes):
        raise ValueError("candidate hashes must be unique")

    if np.any(~(np.isnan(matrix) | np.isfinite(matrix))):
        raise ValueError("nonmissing scores must be finite")

    ranks = np.full(matrix.shape, np.nan, dtype=np.float64)
    for query_index, row in enumerate(matrix):
        available = np.flatnonzero(~np.isnan(row))
        hash_order = sorted(
            available.tolist(),
            key=candidate_hashes.__getitem__,
        )
        order = sorted(
            hash_order,
            key=lambda candidate_index: row[candidate_index],
            reverse=True,
        )
        ranks[query_index, order] = (
            np.arange(len(order), dtype=np.float64) + 1.0
        )
    return ranks


def weighted_rrf(
    ranks_by_channel: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    k: float,
) -> np.ndarray:
    """Fuse ranks while normalizing weights over available evidence."""
    if isinstance(k, bool) or not isinstance(k, Real):
        raise ValueError("k must be a positive finite real number")
    k_value = float(k)
    if not np.isfinite(k_value) or k_value <= 0.0:
        raise ValueError("k must be a positive finite real number")
    if not isinstance(ranks_by_channel, Mapping) or not ranks_by_channel:
        raise ValueError("ranks_by_channel must be a nonempty mapping")
    if not isinstance(weights, Mapping):
        raise ValueError("weights must be a mapping")
    if set(ranks_by_channel) != set(weights):
        raise ValueError("rank and weight channels must match")

    normalized_weights: dict[str, float] = {}
    for name, weight in weights.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(weight, bool)
            or not isinstance(weight, Real)
        ):
            raise ValueError("weights must map channel names to real values")
        value = float(weight)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("weights must be finite and nonnegative")
        normalized_weights[name] = value
    if not any(value > 0.0 for value in normalized_weights.values()):
        raise ValueError("at least one channel weight must be positive")

    shape: tuple[int, int] | None = None
    rank_matrices: dict[str, np.ndarray] = {}
    for name, ranks in ranks_by_channel.items():
        matrix = np.asarray(ranks)
        if (
            matrix.ndim != 2
            or not np.issubdtype(matrix.dtype, np.number)
            or np.issubdtype(matrix.dtype, np.complexfloating)
        ):
            raise ValueError("rank matrices must be two-dimensional and real")
        numeric = matrix.astype(np.float64, copy=False)
        if np.any(
            ~(np.isnan(numeric) | (np.isfinite(numeric) & (numeric > 0.0)))
        ):
            raise ValueError("nonmissing ranks must be positive and finite")
        if shape is None:
            shape = numeric.shape
        elif numeric.shape != shape:
            raise ValueError("rank matrices must have the same shape")
        rank_matrices[name] = numeric

    assert shape is not None
    cell_weight_scale = np.zeros(shape, dtype=np.float64)
    for name in sorted(rank_matrices):
        weight = normalized_weights[name]
        if weight == 0.0:
            continue
        available = ~np.isnan(rank_matrices[name])
        cell_weight_scale[available] = np.maximum(
            cell_weight_scale[available],
            weight,
        )

    weighted_scores = np.zeros(shape, dtype=np.float64)
    available_weight = np.zeros(shape, dtype=np.float64)
    for name in sorted(rank_matrices):
        ranks = rank_matrices[name]
        weight = normalized_weights[name]
        if weight == 0.0:
            continue
        available = ~np.isnan(ranks)
        scaled_weight = weight / cell_weight_scale[available]
        weighted_scores[available] += scaled_weight / (
            k_value + ranks[available]
        )
        available_weight[available] += scaled_weight

    fused = np.full(shape, np.nan, dtype=np.float64)
    available = available_weight > 0.0
    fused[available] = (
        weighted_scores[available] / available_weight[available]
    )
    return fused


def _validate_unique_strings(
    values: tuple[str, ...], label: str
) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{label} must be a nonempty tuple")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique and ordered")


def _validate_hash_groups(
    groups: tuple[tuple[str, ...], ...], label: str, n_queries: int
) -> None:
    if not isinstance(groups, tuple) or len(groups) != n_queries:
        raise ValueError(f"{label} length must match query IDs")
    for group in groups:
        if not isinstance(group, tuple):
            raise ValueError(f"{label} entries must be tuples")
        if any(not isinstance(value, str) or not value for value in group):
            raise ValueError(f"{label} must contain nonempty strings")
        if len(set(group)) != len(group):
            raise ValueError(f"{label} entries must not contain duplicates")


def _validate_source_hashes(source_hashes: Mapping[str, str]) -> None:
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("source_hashes must be a nonempty mapping")
    for name, digest in source_hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("source hash names must be nonempty strings")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"source hash {name!r} must be lowercase SHA-256"
            )


def _validate_attributes(attributes: Mapping[str, object]) -> None:
    if not isinstance(attributes, Mapping):
        raise ValueError("attributes must be a mapping")
    for name, value in attributes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("attributes keys must be nonempty strings")
        if value is None or isinstance(value, (str, bool, int)):
            continue
        if isinstance(value, float) and np.isfinite(value):
            continue
        raise ValueError(
            "attributes values must be finite JSON scalars"
        )


class _ReadOnlyChannelMapping(Mapping[str, np.ndarray]):
    __slots__ = ("__arrays",)

    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.__arrays = dict(arrays)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.__arrays[name].view()

    def __iter__(self) -> Iterator[str]:
        return iter(self.__arrays)

    def __len__(self) -> int:
        return len(self.__arrays)


@dataclass(frozen=True)
class CandidateScoreBundle:
    query_ids: tuple[str, ...]
    candidate_hashes: tuple[str, ...]
    relevant_hashes: tuple[tuple[str, ...], ...]
    train_positive_hashes: tuple[tuple[str, ...], ...]
    split_units: tuple[str, ...]
    ligand_supported: tuple[bool, ...]
    channels: Mapping[str, np.ndarray]
    split: str
    seed: int
    outer_fold: int
    stage: str
    source_hashes: Mapping[str, str]
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.channels, Mapping):
            channels: dict[str, object] = {}
            for name, matrix in self.channels.items():
                if isinstance(matrix, np.ndarray):
                    matrix = np.array(
                        matrix,
                        copy=True,
                        order="C",
                        subok=False,
                    )
                channels[name] = matrix
            object.__setattr__(
                self,
                "channels",
                MappingProxyType(channels),
            )
        if isinstance(self.source_hashes, Mapping):
            object.__setattr__(
                self,
                "source_hashes",
                MappingProxyType(dict(self.source_hashes)),
            )
        if isinstance(self.attributes, Mapping):
            object.__setattr__(
                self,
                "attributes",
                MappingProxyType(dict(self.attributes)),
            )
        self.validate()
        immutable_channels = {}
        for name, matrix in self.channels.items():
            contiguous = np.ascontiguousarray(matrix)
            immutable_channels[name] = np.frombuffer(
                contiguous.tobytes(order="C"),
                dtype=contiguous.dtype,
            ).reshape(contiguous.shape)
        object.__setattr__(
            self,
            "channels",
            _ReadOnlyChannelMapping(immutable_channels),
        )

    def validate(self) -> None:
        _validate_unique_strings(self.query_ids, "query IDs")
        _validate_unique_strings(self.candidate_hashes, "candidate hashes")
        n_queries = len(self.query_ids)
        n_candidates = len(self.candidate_hashes)
        _validate_hash_groups(
            self.relevant_hashes, "relevant hash metadata", n_queries
        )
        _validate_hash_groups(
            self.train_positive_hashes,
            "train-positive hash metadata",
            n_queries,
        )
        if (
            not isinstance(self.split_units, tuple)
            or len(self.split_units) != n_queries
        ):
            raise ValueError("split unit metadata length must match query IDs")
        if any(
            not isinstance(split_unit, str) or not split_unit
            for split_unit in self.split_units
        ):
            raise ValueError("split units must be nonempty strings")
        if (
            not isinstance(self.ligand_supported, tuple)
            or len(self.ligand_supported) != n_queries
        ):
            raise ValueError(
                "ligand-supported metadata length must match query IDs"
            )
        if any(type(value) is not bool for value in self.ligand_supported):
            raise ValueError("ligand_supported values must be booleans")

        EnsembleBundleId(
            split=self.split,
            seed=self.seed,
            outer_fold=self.outer_fold,
            stage=self.stage,
        ).relative_dir()
        _validate_source_hashes(self.source_hashes)
        _validate_attributes(self.attributes)

        if not isinstance(self.channels, Mapping) or not self.channels:
            raise ValueError("channels must be a nonempty mapping")
        derived_channels = set(self.channels) & DERIVED_CHANNELS
        if derived_channels and (
            self.stage != "test" or derived_channels != {"Ensemble"}
        ):
            raise ValueError(
                "derived Ensemble channel is permitted only in test bundles"
            )
        expected_shape = (n_queries, n_candidates)
        for name, matrix in self.channels.items():
            if not isinstance(name, str) or not name:
                raise ValueError("channel names must be nonempty")
            if not isinstance(matrix, np.ndarray):
                raise ValueError(f"channel {name!r} must be a NumPy array")
            if (
                not np.issubdtype(matrix.dtype, np.number)
                or np.issubdtype(matrix.dtype, np.complexfloating)
            ):
                raise ValueError(
                    f"channel {name!r} scores must be real numeric"
                )
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"channel {name!r} shape {matrix.shape} does not match "
                    f"{expected_shape}"
                )
            if np.any(~(np.isnan(matrix) | np.isfinite(matrix))):
                raise ValueError(
                    f"channel {name!r} nonmissing scores must be finite"
                )

        candidate_indexes = {
            candidate_hash: index
            for index, candidate_hash in enumerate(self.candidate_hashes)
        }
        matrices = tuple(self.channels.values())
        for query_index in range(n_queries):
            applicable_channels = required_channels(
                self.split,
                ligand_supported=self.ligand_supported[query_index],
            )
            missing_channels = tuple(
                name
                for name in applicable_channels
                if name not in self.channels
            )
            if missing_channels:
                raise ValueError(
                    f"query {self.query_ids[query_index]!r} is missing "
                    f"required applicable channels: {missing_channels}"
                )
            applicable_matrices = tuple(
                self.channels[name] for name in applicable_channels
            )
            for name in sorted(
                set(self.channels)
                - set(applicable_channels)
                - DERIVED_CHANNELS
            ):
                if np.any(~np.isnan(self.channels[name][query_index])):
                    raise ValueError(
                        f"non-applicable channel {name!r} must be NaN for "
                        f"the entire row of query "
                        f"{self.query_ids[query_index]!r}"
                    )
            relevant = set(self.relevant_hashes[query_index])
            train_positive = set(self.train_positive_hashes[query_index])
            if relevant & train_positive:
                raise ValueError(
                    f"relevant and train-positive hashes overlap for "
                    f"query {self.query_ids[query_index]!r}"
                )
            for candidate_hash in relevant:
                candidate_index = candidate_indexes.get(candidate_hash)
                if candidate_index is None or not any(
                    np.isfinite(matrix[query_index, candidate_index])
                    for matrix in applicable_matrices
                ):
                    raise ValueError(
                        f"relevant hash {candidate_hash!r} is not an "
                        f"eligible finite candidate for query "
                        f"{self.query_ids[query_index]!r}"
                    )
            for candidate_hash in train_positive:
                candidate_index = candidate_indexes.get(candidate_hash)
                if candidate_index is None:
                    raise ValueError(
                        f"train-positive hash {candidate_hash!r} is not in "
                        "the candidate library"
                    )
                if any(
                    not np.isnan(matrix[query_index, candidate_index])
                    for matrix in matrices
                ):
                    raise ValueError(
                        f"train-positive hash {candidate_hash!r} must be NaN "
                        f"in every channel for query "
                        f"{self.query_ids[query_index]!r}"
                    )
            if "Ensemble" in self.channels:
                eligible = np.ones(n_candidates, dtype=bool)
                eligible[
                    [
                        candidate_indexes[candidate_hash]
                        for candidate_hash in train_positive
                    ]
                ] = False
                if not np.isfinite(
                    self.channels["Ensemble"][query_index, eligible]
                ).all():
                    raise ValueError(
                        f"derived Ensemble channel lacks eligible finite "
                        f"scores for query {self.query_ids[query_index]!r}"
                    )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _candidate_order_sha256(candidate_hashes: tuple[str, ...]) -> str:
    return sha256(_canonical_json_bytes(list(candidate_hashes))).hexdigest()


def _write_and_sync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        os.fchmod(handle.fileno(), 0o644)
        handle.flush()
        os.fsync(handle.fileno())


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def _write_deterministic_npz(
    path: Path, channels: Mapping[str, np.ndarray]
) -> None:
    with path.open("xb") as handle:
        with zipfile.ZipFile(
            handle, mode="w", compression=zipfile.ZIP_STORED
        ) as archive:
            for name in sorted(channels):
                entry = zipfile.ZipInfo(
                    filename=f"{name}.npy",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                entry.compress_type = zipfile.ZIP_STORED
                entry.create_system = 3
                entry.external_attr = 0o644 << 16
                archive.writestr(entry, _npy_bytes(channels[name]))
        os.fchmod(handle.fileno(), 0o644)
        handle.flush()
        os.fsync(handle.fileno())


def _query_manifest_bytes(bundle: CandidateScoreBundle) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(
        (
            "query_id",
            "relevant_hashes",
            "train_positive_hashes",
            "split_unit",
            "ligand_supported",
        )
    )
    for index, query_id in enumerate(bundle.query_ids):
        writer.writerow(
            (
                query_id,
                json.dumps(
                    bundle.relevant_hashes[index],
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
                json.dumps(
                    bundle.train_positive_hashes[index],
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
                bundle.split_units[index],
                "1" if bundle.ligand_supported[index] else "0",
            )
        )
    return output.getvalue().encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        _rename_noreplace_with_lock(source, destination)
        return
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            f"refusing to overwrite candidate-score bundle: {destination}",
            destination,
        )
    unsupported = {
        errno.EINVAL,
        errno.ENOSYS,
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        getattr(errno, "ENOTSUP", errno.EINVAL),
    }
    if error_number in unsupported:
        _rename_noreplace_with_lock(source, destination)
        return
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination,
    )


def _rename_noreplace_with_lock(
    source: Path, destination: Path
) -> None:
    lock_path = destination.parent / f".{destination.name}.publish.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as error:
        raise FileExistsError(
            error.errno,
            f"candidate-score publication already in progress: "
            f"{destination}",
            destination,
        ) from error
    try:
        os.close(descriptor)
        try:
            destination.lstat()
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                errno.EEXIST,
                f"refusing to overwrite candidate-score bundle: "
                f"{destination}",
                destination,
            )
        os.rename(source, destination)
    finally:
        lock_path.unlink(missing_ok=True)
        _fsync_directory(destination.parent)


def save_candidate_score_bundle(
    bundle: CandidateScoreBundle, output_dir: str | Path
) -> None:
    if not isinstance(bundle, CandidateScoreBundle):
        raise TypeError("bundle must be a CandidateScoreBundle")
    bundle = CandidateScoreBundle(
        query_ids=bundle.query_ids,
        candidate_hashes=bundle.candidate_hashes,
        relevant_hashes=bundle.relevant_hashes,
        train_positive_hashes=bundle.train_positive_hashes,
        split_units=bundle.split_units,
        ligand_supported=bundle.ligand_supported,
        channels=bundle.channels,
        split=bundle.split,
        seed=bundle.seed,
        outer_fold=bundle.outer_fold,
        stage=bundle.stage,
        source_hashes=bundle.source_hashes,
        attributes=bundle.attributes,
    )
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite candidate-score bundle: {destination}"
        )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging.",
            dir=destination.parent,
        )
    )
    try:
        score_path = staging / "candidate_scores.npz"
        query_path = staging / "query_manifest.tsv"
        _write_deterministic_npz(score_path, bundle.channels)
        _write_and_sync(query_path, _query_manifest_bytes(bundle))
        channels = sorted(bundle.channels)
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "split": bundle.split,
            "seed": bundle.seed,
            "outer_fold": bundle.outer_fold,
            "stage": bundle.stage,
            "source_hashes": dict(sorted(bundle.source_hashes.items())),
            "attributes": dict(sorted(bundle.attributes.items())),
            "trust_model": dict(BUNDLE_TRUST_MODEL),
            "candidate_hashes": list(bundle.candidate_hashes),
            "candidate_order_sha256": _candidate_order_sha256(
                bundle.candidate_hashes
            ),
            "channels": channels,
            "matrix_shape": [
                len(bundle.query_ids),
                len(bundle.candidate_hashes),
            ],
            "matrix_dtypes": {
                name: bundle.channels[name].dtype.str for name in channels
            },
            "file_hashes": {
                "candidate_scores.npz": file_sha256(score_path),
                "query_manifest.tsv": file_sha256(query_path),
            },
        }
        _write_and_sync(
            staging / "bundle_manifest.json",
            _canonical_json_bytes(manifest),
        )
        _fsync_directory(staging)
        _rename_noreplace(staging, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
            _fsync_directory(destination.parent)
        raise


def _read_payload(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read()
    except OSError as error:
        raise ValueError(f"unable to snapshot bundle payload: {path}") from error


def _load_manifest(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid bundle manifest") from error
    if not isinstance(value, dict):
        raise ValueError("bundle manifest must be a JSON object")
    required = {
        "schema",
        "schema_version",
        "split",
        "seed",
        "outer_fold",
        "stage",
        "source_hashes",
        "attributes",
        "trust_model",
        "candidate_hashes",
        "candidate_order_sha256",
        "channels",
        "matrix_shape",
        "matrix_dtypes",
        "file_hashes",
    }
    if set(value) != required:
        raise ValueError("bundle manifest schema fields do not match")
    if (
        value["schema"] != BUNDLE_SCHEMA
        or value["schema_version"] != BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported bundle manifest schema version")
    if value["trust_model"] != dict(BUNDLE_TRUST_MODEL):
        raise ValueError("bundle manifest trust model does not match")
    return value


def _load_query_manifest(
    payload: bytes,
) -> tuple[
    tuple[str, ...],
    tuple[tuple[str, ...], ...],
    tuple[tuple[str, ...], ...],
    tuple[str, ...],
    tuple[bool, ...],
]:
    expected_fields = [
        "query_id",
        "relevant_hashes",
        "train_positive_hashes",
        "split_unit",
        "ligand_supported",
    ]
    try:
        text = payload.decode("utf-8")
        reader = csv.DictReader(
            io.StringIO(text, newline=""),
            delimiter="\t",
        )
        if reader.fieldnames != expected_fields:
            raise ValueError("query manifest columns do not match")
        rows = list(reader)
    except (UnicodeError, csv.Error) as error:
        raise ValueError("invalid query manifest") from error

    query_ids: list[str] = []
    relevant_hashes: list[tuple[str, ...]] = []
    train_positive_hashes: list[tuple[str, ...]] = []
    split_units: list[str] = []
    ligand_supported: list[bool] = []
    for row in rows:
        try:
            relevant = json.loads(row["relevant_hashes"])
            train_positive = json.loads(row["train_positive_hashes"])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("query manifest hash lists are invalid") from error
        if not isinstance(relevant, list) or not isinstance(
            train_positive, list
        ):
            raise ValueError("query manifest hash lists must be JSON arrays")
        support = row["ligand_supported"]
        if support not in {"0", "1"}:
            raise ValueError("query manifest ligand_supported must be 0 or 1")
        query_ids.append(row["query_id"])
        relevant_hashes.append(tuple(relevant))
        train_positive_hashes.append(tuple(train_positive))
        split_units.append(row["split_unit"])
        ligand_supported.append(support == "1")
    return (
        tuple(query_ids),
        tuple(relevant_hashes),
        tuple(train_positive_hashes),
        tuple(split_units),
        tuple(ligand_supported),
    )


def load_candidate_score_bundle(
    output_dir: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> CandidateScoreBundle:
    """Load checksummed data, optionally authenticating its external manifest hash."""
    directory = Path(output_dir)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    names = {path.name for path in directory.iterdir()}
    base_names = set(BUNDLE_FILENAMES)
    allowed_names = (base_names, base_names | {"test_per_query.tsv"})
    if names not in allowed_names:
        raise ValueError("candidate-score bundle files do not match schema")

    payloads = {
        filename: _read_payload(directory / filename)
        for filename in BUNDLE_FILENAMES
    }
    payload_hashes = {
        filename: sha256(payload).hexdigest()
        for filename, payload in payloads.items()
    }
    if expected_manifest_sha256 is not None and (
        not isinstance(expected_manifest_sha256, str)
        or payload_hashes["bundle_manifest.json"]
        != expected_manifest_sha256
    ):
        raise ValueError("external manifest SHA-256 mismatch")
    manifest = _load_manifest(payloads["bundle_manifest.json"])
    if "test_per_query.tsv" in names and manifest["stage"] != "test":
        raise ValueError(
            "test_per_query.tsv is permitted only in test bundles"
        )
    file_hashes = manifest["file_hashes"]
    if not isinstance(file_hashes, dict) or set(file_hashes) != {
        "candidate_scores.npz",
        "query_manifest.tsv",
    }:
        raise ValueError("bundle manifest file hashes do not match schema")
    for filename, expected_hash in file_hashes.items():
        if (
            not isinstance(expected_hash, str)
            or payload_hashes[filename] != expected_hash
        ):
            raise ValueError(f"SHA-256 mismatch for {filename}")

    candidate_values = manifest["candidate_hashes"]
    if not isinstance(candidate_values, list):
        raise ValueError("manifest candidate hashes must be a list")
    candidate_hashes = tuple(candidate_values)
    candidate_order_hash = manifest["candidate_order_sha256"]
    if (
        not isinstance(candidate_order_hash, str)
        or _candidate_order_sha256(candidate_hashes) != candidate_order_hash
    ):
        raise ValueError("candidate order SHA-256 mismatch")

    channel_values = manifest["channels"]
    matrix_dtypes = manifest["matrix_dtypes"]
    if (
        not isinstance(channel_values, list)
        or any(not isinstance(name, str) for name in channel_values)
        or channel_values != sorted(channel_values)
        or len(set(channel_values)) != len(channel_values)
        or not isinstance(matrix_dtypes, dict)
        or set(matrix_dtypes) != set(channel_values)
    ):
        raise ValueError("manifest channel metadata is invalid")
    channels: dict[str, np.ndarray] = {}
    try:
        with np.load(
            io.BytesIO(payloads["candidate_scores.npz"]),
            allow_pickle=False,
        ) as archive:
            if archive.files != channel_values:
                raise ValueError("NPZ channel names do not match manifest")
            for name in channel_values:
                channels[name] = archive[name]
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("invalid candidate score NPZ") from error

    (
        query_ids,
        relevant_hashes,
        train_positive_hashes,
        split_units,
        ligand_supported,
    ) = _load_query_manifest(payloads["query_manifest.tsv"])
    bundle = CandidateScoreBundle(
        query_ids=query_ids,
        candidate_hashes=candidate_hashes,
        relevant_hashes=relevant_hashes,
        train_positive_hashes=train_positive_hashes,
        split_units=split_units,
        ligand_supported=ligand_supported,
        channels=channels,
        split=manifest["split"],
        seed=manifest["seed"],
        outer_fold=manifest["outer_fold"],
        stage=manifest["stage"],
        source_hashes=manifest["source_hashes"],
        attributes=manifest["attributes"],
    )
    expected_shape = [len(query_ids), len(candidate_hashes)]
    if manifest["matrix_shape"] != expected_shape:
        raise ValueError("manifest matrix shape does not match bundle")
    actual_dtypes = {
        name: channels[name].dtype.str for name in sorted(channels)
    }
    if matrix_dtypes != actual_dtypes:
        raise ValueError("manifest matrix dtype does not match NPZ")
    return bundle
