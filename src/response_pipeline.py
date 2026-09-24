"""Frozen response reconstruction; extracted numerical functions.

The sole adapted I/O hook loads an explicit candidate-aligned similarity cache.
No original project imports or implicit data locations are used.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd
from src.benchmark_agnostic_transfer import transfer_first_auto, witness_availability
from src.transfer_distilled_residual import _robust_z, calibrated_dstar_fallback
from src.structure_response_transfer import CompactStructureSimilarity, combine_transfer_scores
from src.unified_bscd_residual import FEATURE_NAMES, build_bscd_residual_features
from src.unified_chemical_transfer import FINGERPRINT_NAMES, chemical_transfer_scores
from src.unified_random_edge_full_baselines import candidate_hash_tie_scores

def _label_matrix(relevant, candidates):
    # Labels are optional for prediction; never used to construct evidence.
    index = {value: i for i, value in enumerate(candidates)}
    labels = np.zeros((len(relevant), len(candidates)), dtype=bool)
    for row, values in enumerate(relevant):
        if any(value not in index for value in values):
            raise ValueError('Relevant candidate absent from candidate library')
        labels[row, [index[value] for value in values]] = True
    return labels

def _load_sequence_similarity(source):
    with np.load(source.sequence_similarity_path, allow_pickle=False) as archive:
        candidates = tuple(archive['candidate_hashes'].astype(str))
        scores = archive['scores'].copy()
    if candidates != source.candidate_hashes:
        raise ValueError('Protein similarity cache candidate order mismatch')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Protein similarities must be finite and in [0, 1]')
    return CompactSequenceSimilarity(candidates, scores)

@dataclass(frozen=True)
class QueryEvidence:
    s_scores: np.ndarray
    c_scores: np.ndarray
    s_available: np.ndarray
    c_available: np.ndarray


@dataclass(frozen=True)
class CompactSequenceSimilarity:
    """Candidate-indexed directional MMseqs identity-coverage scores."""

    candidate_hashes: tuple[str, ...]
    scores: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.scores)
        count = len(self.candidate_hashes)
        if matrix.dtype != np.float32 or matrix.shape != (count, count):
            raise ValueError("compact similarity must be a square float32 matrix")
        object.__setattr__(
            self, "_candidate_index",
            {value: index for index, value in enumerate(self.candidate_hashes)},
        )

    def similarity(self, left: str, right: str) -> float:
        left_index = self._candidate_index.get(str(left))
        right_index = self._candidate_index.get(str(right))
        if left_index is None or right_index is None:
            return 0.0
        return float(max(
            self.scores[left_index, right_index],
            self.scores[right_index, left_index],
        ))

    def nearest(self, responders: Sequence[str]) -> np.ndarray:
        indexes = [self._candidate_index[value] for value in responders]
        if not indexes:
            return np.zeros(len(self.candidate_hashes), dtype=np.float32)
        forward = self.scores[:, indexes]
        reverse = self.scores[indexes, :].T
        return np.max(np.maximum(forward, reverse), axis=1)


@dataclass(frozen=True)
class FoldEvidenceSource:
    split: str
    fold: int
    stage: str
    query_ids: tuple[str, ...]
    split_units: tuple[str, ...]
    candidate_hashes: tuple[str, ...]
    fit_edges: pd.DataFrame
    exclusion_masks: np.ndarray
    relevant_hashes: tuple[tuple[str, ...], ...]
    dstar_seed_scores: np.ndarray
    candidate_path: Path
    sequence_similarity_path: Path


@dataclass(frozen=True)
class DStarSeedOverride:
    """Three raw D* seed rows aligned to each episode's active library."""

    split: str
    fold: int
    stage: str
    query_ids: tuple[str, ...]
    candidate_hashes: tuple[tuple[str, ...], ...]
    scores: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        queries = tuple(map(str, self.query_ids))
        candidates = tuple(tuple(map(str, values)) for values in self.candidate_hashes)
        if not queries or len(queries) != len(set(queries)):
            raise ValueError("override query IDs must be nonempty and unique")
        if len(candidates) != len(queries) or len(self.scores) != len(queries):
            raise ValueError("override episode inventory must align")
        owned = []
        for hashes, raw in zip(candidates, self.scores, strict=True):
            values = np.array(raw, dtype=np.float64, copy=True, order="C")
            if (
                not hashes
                or len(hashes) != len(set(hashes))
                or values.shape != (3, len(hashes))
                or not np.isfinite(values).all()
            ):
                raise ValueError("override scores must be finite three-seed active rows")
            values.setflags(write=False)
            owned.append(values)
        object.__setattr__(self, "query_ids", queries)
        object.__setattr__(self, "candidate_hashes", candidates)
        object.__setattr__(self, "scores", tuple(owned))


def apply_dstar_seed_override(
    source: FoldEvidenceSource, override: DStarSeedOverride
) -> FoldEvidenceSource:
    """Replace raw D* only after strict fold/episode/active-library alignment."""

    if not isinstance(source, FoldEvidenceSource) or not isinstance(
        override, DStarSeedOverride
    ):
        raise TypeError("source and Dstar override have invalid types")
    if (override.split, override.fold, override.stage) != (
        source.split,
        source.fold,
        source.stage,
    ):
        raise ValueError("Dstar override split/fold/stage identity mismatch")
    if override.query_ids != source.query_ids:
        raise ValueError("Dstar override query identity or order mismatch")
    expected_shape = (
        3,
        len(source.query_ids),
        len(source.candidate_hashes),
    )
    raw = np.asarray(source.dstar_seed_scores)
    if raw.shape != expected_shape or not np.issubdtype(raw.dtype, np.number):
        raise ValueError("canonical Dstar seed source is misaligned")
    replaced = np.array(raw, dtype=np.float64, copy=True, order="C")
    for row, (hashes, scores) in enumerate(
        zip(override.candidate_hashes, override.scores, strict=True)
    ):
        active = np.flatnonzero(~source.exclusion_masks[row])
        expected = tuple(source.candidate_hashes[index] for index in active)
        if hashes != expected:
            raise ValueError("Dstar override active candidate identity or order mismatch")
        replaced[:, row, active] = scores
    return replace(source, dstar_seed_scores=replaced)


@dataclass(frozen=True)
class ActiveRandomFrameworkRow:
    candidate_indexes: np.ndarray
    candidate_hashes: tuple[str, ...]
    features: np.ndarray | None
    labels: np.ndarray | None
    eligible_mask: np.ndarray
    backbone: np.ndarray
    S: np.ndarray
    C: np.ndarray
    T: np.ndarray
    Dstar: np.ndarray
    dstar_seed_scores: np.ndarray
    s_available: np.ndarray
    c_available: np.ndarray
    response_supported: np.ndarray


@dataclass(frozen=True)
class RandomFrameworkInputs:
    split: str
    fold: int
    stage: str
    query_ids: tuple[str, ...]
    split_units: tuple[str, ...]
    candidate_hashes: tuple[str, ...]
    relevant_hashes: tuple[tuple[str, ...], ...]
    exclusion_masks: np.ndarray
    features: np.ndarray | None
    labels: np.ndarray
    eligible_mask: np.ndarray
    backbone: np.ndarray
    S: np.ndarray
    C: np.ndarray
    T: np.ndarray
    Dstar: np.ndarray
    dstar_seed_scores: np.ndarray
    s_available: np.ndarray
    c_available: np.ndarray
    response_supported: np.ndarray


def _edge_pairs(fit_edges) -> tuple[tuple[str, str], ...]:
    if isinstance(fit_edges, pd.DataFrame):
        required = {"ligand_key", "sequence_md5"}
        if not required <= set(fit_edges.columns):
            raise ValueError("fit_edges lacks ligand_key or sequence_md5")
        records = fit_edges[["ligand_key", "sequence_md5"]].itertuples(
            index=False, name=None
        )
    else:
        records = fit_edges
    return tuple(sorted({(str(ligand), str(candidate)) for ligand, candidate in records}))


def _similarity_value(similarity: Mapping, left: str, right: str) -> float | None:
    if isinstance(similarity, CompactSequenceSimilarity):
        return similarity.similarity(left, right)
    if left == right:
        return 1.0
    values = []
    for key in ((left, right), (right, left)):
        value = similarity.get(key)
        if isinstance(value, Mapping):
            value = value.get("identity_coverage")
        if value is not None:
            number = float(value)
            if np.isfinite(number) and number > 0.0:
                values.append(number)
    return max(values) if values else None


def _chemical_tables(chemical_similarity: Mapping) -> Mapping[str, Mapping]:
    if set(chemical_similarity) == set(FINGERPRINT_NAMES):
        return chemical_similarity
    return {name: chemical_similarity for name in FINGERPRINT_NAMES}


def build_query_evidence(
    query_id,
    candidate_hashes,
    fit_edges,
    sequence_similarity,
    chemical_similarity,
):
    """Build raw S/C evidence for one query from legal fold-local witnesses."""

    query = str(query_id)
    candidates = tuple(map(str, candidate_hashes))
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("candidate hashes must be unique and nonempty")
    pairs = _edge_pairs(fit_edges)
    responders = tuple(candidate for ligand, candidate in pairs if ligand == query)
    s_scores = np.zeros(len(candidates), dtype=np.float64)
    s_witness = np.zeros(len(candidates), dtype=bool)
    for index, candidate in enumerate(candidates):
        values = [
            value
            for responder in responders
            if (value := _similarity_value(sequence_similarity, candidate, responder))
            is not None
        ]
        if responders:
            # The established MMseqs matrix is dense-by-definition: absence of
            # a reported alignment is a finite zero, provided the query has a
            # legal same-ligand fold-local responder history.
            s_scores[index] = max(values, default=0.0)
            s_witness[index] = True

    chemical = chemical_transfer_scores(
        query_ids=(query,),
        candidate_hashes=candidates,
        fit_edges=pairs,
        similarity_tables=_chemical_tables(chemical_similarity),
    )
    c_scores = np.asarray(chemical.scores["morgan"][0], dtype=np.float64)
    c_witness = np.asarray(chemical.available[0], dtype=bool)
    s_available, c_available = witness_availability(
        s_witness[None, :], c_witness[None, :], s_scores[None, :], c_scores[None, :]
    )
    return QueryEvidence(s_scores, c_scores, s_available[0], c_available[0])


def _vector(value, count: int, name: str, dtype) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != (count,):
        raise ValueError(f"{name} must align with candidates")
    return result


def _boolean_vector(value, count: int, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.bool_ or result.shape != (count,):
        raise ValueError(f"{name} must be a boolean vector aligned with candidates")
    return np.asarray(result, dtype=bool)


def _calibrated_dstar(raw_seeds: np.ndarray) -> np.ndarray:
    mean = np.mean(raw_seeds, axis=0, keepdims=True)
    return calibrated_dstar_fallback(mean, _robust_z(mean))[0]


def build_active_row(
    *,
    candidate_hashes,
    exclusion_mask,
    raw_s,
    raw_c,
    raw_dstar_seeds,
    s_witness,
    c_witness,
    response_supported,
    beta,
    labels=None,
    build_features: bool = True,
):
    """Apply R_fit, then calibrate, route, and build Random-edge features."""

    candidates = tuple(map(str, candidate_hashes))
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("candidate hashes must be unique and nonempty")
    count = len(candidates)
    excluded = _boolean_vector(exclusion_mask, count, "exclusion mask")
    s_raw = _vector(raw_s, count, "raw S", np.float64)
    c_raw = _vector(raw_c, count, "raw C", np.float64)
    s_seen = _boolean_vector(s_witness, count, "S witness")
    c_seen = _boolean_vector(c_witness, count, "C witness")
    support = _boolean_vector(response_supported, count, "response support")
    seeds = np.asarray(raw_dstar_seeds, dtype=np.float64)
    if seeds.shape != (3, count):
        raise ValueError("raw Dstar seeds must have shape (3, candidate)")
    if np.isinf(s_raw).any() or np.isinf(c_raw).any() or np.isinf(seeds).any():
        raise ValueError("raw scores must not contain infinity")
    active = np.flatnonzero(~excluded)
    if not len(active):
        raise ValueError("active candidate library is empty")
    active_candidates = tuple(candidates[index] for index in active)

    s_available, c_available = witness_availability(
        s_seen[active][None, :],
        c_seen[active][None, :],
        s_raw[active][None, :],
        c_raw[active][None, :],
    )
    sequence = _pure_transfer_scores(
        np.where(s_available, s_raw[active][None, :], np.nan), active_candidates
    )
    chemical = _pure_transfer_scores(
        np.where(c_available, c_raw[active][None, :], np.nan), active_candidates
    )
    active_seeds = seeds[:, active]
    dstar = _calibrated_dstar(active_seeds)
    b_sd, _ = transfer_first_auto(
        sequence, chemical, dstar[None, :], s_available,
        np.zeros_like(c_available), beta,
    )
    b_cd, _ = transfer_first_auto(
        sequence, chemical, dstar[None, :], np.zeros_like(s_available),
        c_available, beta,
    )
    transfer, transfer_available = transfer_first_auto(
        sequence, chemical, dstar[None, :], s_available, c_available, beta
    )
    backbone = np.where(transfer_available, transfer, dstar[None, :])
    scores = {
        "S": sequence,
        "C": chemical,
        "Dstar": dstar[None, :],
        "B_SD": b_sd,
        "B_CD": b_cd,
        "B_SCD": backbone,
    }
    features = None
    if build_features:
        features = build_bscd_residual_features(
            scores=scores,
            # The frozen Random-edge feature protocol consumes the three raw Dstar
            # seed matrices; only their ensemble mean is calibrated as a channel.
            dstar_seed_scores=active_seeds[:, None, :],
            s_available=s_available,
            c_available=c_available,
            response_supported=support[active],
            candidate_hashes=active_candidates,
        )[0]
    active_labels = None
    if labels is not None:
        all_labels = _boolean_vector(labels, count, "labels")
        if np.any(all_labels & excluded):
            raise ValueError("R_fit removed an evaluation positive")
        active_labels = all_labels[active]
    return ActiveRandomFrameworkRow(
        active,
        active_candidates,
        None if features is None else np.asarray(features, dtype=np.float32),
        active_labels,
        np.ones(len(active), dtype=bool),
        backbone[0],
        sequence[0],
        chemical[0],
        transfer[0],
        dstar,
        active_seeds,
        s_available[0],
        c_available[0],
        support[active],
    )


def protein_transfer_raw(
    variant,
    responders,
    candidate_hashes,
    active_mask,
    sequence_similarity,
    structure_similarity,
    structure_weight,
):
    """Select raw sequence, structure, or calibrated joint transfer scores."""

    if variant not in {"sequence", "structure", "joint"}:
        raise ValueError(f"unknown protein transfer variant: {variant}")
    if variant == "structure":
        if structure_similarity is None:
            raise ValueError("structure similarity is required for structural transfer")
        return structure_similarity.nearest(responders)
    if variant == "joint":
        _validate_structure_weight(structure_weight)
    if sequence_similarity is None or not callable(
        getattr(sequence_similarity, "nearest", None)
    ):
        raise ValueError("sequence similarity is required for sequence transfer")
    sequence = sequence_similarity.nearest(responders)
    if variant == "sequence":
        return sequence
    if structure_similarity is None:
        raise ValueError("structure similarity is required for structural transfer")
    structural = structure_similarity.nearest(responders)
    if variant == "structure":
        return structural
    return combine_transfer_scores(
        sequence,
        structural,
        active_mask=active_mask,
        candidate_hashes=candidate_hashes,
        structure_weight=structure_weight,
    )


def _validate_structure_weight(structure_weight) -> None:
    if (
        isinstance(structure_weight, bool)
        or not isinstance(structure_weight, Real)
        or float(structure_weight) not in {0.25, 0.5, 0.75, 1.0}
    ):
        raise ValueError("structure weight is outside the frozen grid")


def _align_structure_similarity(
    structure_similarity: CompactStructureSimilarity,
    candidate_hashes: tuple[str, ...],
    fold_responders: tuple[str, ...],
) -> CompactStructureSimilarity:
    cache_candidates = structure_similarity.candidate_hashes
    if not candidate_hashes or len(candidate_hashes) != len(set(candidate_hashes)):
        raise ValueError("structure candidate hashes must be unique and nonempty")
    if (
        len(cache_candidates) != len(candidate_hashes)
        or set(cache_candidates) != set(candidate_hashes)
    ):
        raise ValueError(
            "structure similarity candidate set does not exactly match fold candidates"
        )
    missing = sorted(set(fold_responders).difference(structure_similarity.responder_hashes))
    if missing:
        raise ValueError(f"fold-local responder is absent from structure cache: {missing[0]}")
    if cache_candidates == candidate_hashes:
        return structure_similarity
    cache_index = {value: index for index, value in enumerate(cache_candidates)}
    row_indexes = [cache_index[value] for value in candidate_hashes]
    return CompactStructureSimilarity(
        candidate_hashes,
        structure_similarity.responder_hashes,
        structure_similarity.scores[row_indexes],
    )


def build_random_framework_inputs(
    source: FoldEvidenceSource,
    *,
    beta: float = 0.5,
    build_features: bool = True,
    protein_transfer_variant: str = "sequence",
    structure_similarity: CompactStructureSimilarity | None = None,
    structure_weight: float = 1.0,
    dstar_seed_override: DStarSeedOverride | None = None,
) -> RandomFrameworkInputs:
    """Construct common fold-local evidence and Random-edge residual inputs."""

    if dstar_seed_override is not None:
        source = apply_dstar_seed_override(source, dstar_seed_override)

    queries, candidates = source.query_ids, source.candidate_hashes
    shape = (len(queries), len(candidates))
    if (
        source.exclusion_masks.dtype != np.bool_
        or source.exclusion_masks.shape != shape
        or source.dstar_seed_scores.shape != (3, *shape)
    ):
        raise ValueError("fold source arrays are misaligned")
    if protein_transfer_variant not in {"sequence", "structure", "joint"}:
        raise ValueError(
            f"unknown protein transfer variant: {protein_transfer_variant}"
        )
    if protein_transfer_variant != "sequence" and structure_similarity is None:
        raise ValueError("structure similarity is required for structural transfer")
    if protein_transfer_variant == "joint":
        _validate_structure_weight(structure_weight)
    sequence_similarity = None
    if protein_transfer_variant in {"sequence", "joint"}:
        sequence_similarity = _load_sequence_similarity(source)
        if sequence_similarity is None or not callable(
            getattr(sequence_similarity, "nearest", None)
        ):
            raise ValueError("sequence similarity is required for sequence transfer")
        if tuple(sequence_similarity.candidate_hashes) != candidates:
            raise ValueError("similarity source candidate order is misaligned")
    pairs = _edge_pairs(source.fit_edges)
    responders_by_ligand: dict[str, list[str]] = {}
    for ligand, candidate in pairs:
        responders_by_ligand.setdefault(ligand, []).append(candidate)
    if protein_transfer_variant != "sequence":
        if not isinstance(structure_similarity, CompactStructureSimilarity):
            raise ValueError("structure similarity must be a compact structure cache")
        structure_similarity = _align_structure_similarity(
            structure_similarity,
            candidates,
            tuple(candidate for _, candidate in pairs),
        )
    # Preserve the established Random-edge MMseqs materialization precision so
    # tied values and therefore candidate-hash rank percentiles reproduce it.
    s_raw = np.zeros(shape, dtype=np.float32)
    s_witness = np.zeros(shape, dtype=bool)
    for row, query in enumerate(queries):
        responders = tuple(responders_by_ligand.get(query, ()))
        if responders:
            if protein_transfer_variant == "sequence":
                s_raw[row] = sequence_similarity.nearest(responders)
            else:
                s_raw[row] = protein_transfer_raw(
                    protein_transfer_variant,
                    responders,
                    candidates,
                    ~source.exclusion_masks[row],
                    sequence_similarity,
                    structure_similarity,
                    structure_weight,
                )
            s_witness[row] = True
    chemical = chemical_transfer_scores(
        query_ids=queries, candidate_hashes=candidates, fit_edges=source.fit_edges
    )
    c_raw = np.asarray(chemical.scores["morgan"], dtype=np.float64)
    c_witness = np.asarray(chemical.available, dtype=bool)
    labels = _label_matrix(source.relevant_hashes, candidates)
    eligible = ~source.exclusion_masks
    if not np.all(~labels | eligible):
        raise ValueError("R_fit removed an evaluation positive")
    supported_set = {candidate for _, candidate in pairs}
    support = np.asarray([candidate in supported_set for candidate in candidates], dtype=bool)

    features = (
        np.zeros((*shape, len(FEATURE_NAMES)), dtype=np.float32)
        if build_features
        else None
    )
    fill = np.full(shape, -1e9, dtype=np.float64)
    matrices = {name: fill.copy() for name in ("backbone", "S", "C", "T", "Dstar")}
    calibrated_seeds = np.full((3, *shape), np.nan, dtype=np.float64)
    s_available = np.zeros(shape, dtype=bool)
    c_available = np.zeros(shape, dtype=bool)
    for row in range(len(queries)):
        active = build_active_row(
            candidate_hashes=candidates,
            exclusion_mask=source.exclusion_masks[row],
            raw_s=s_raw[row], raw_c=c_raw[row],
            raw_dstar_seeds=source.dstar_seed_scores[:, row],
            s_witness=s_witness[row], c_witness=c_witness[row],
            response_supported=support, labels=labels[row], beta=beta,
            build_features=build_features,
        )
        indexes = active.candidate_indexes
        if features is not None:
            if active.features is None:
                raise AssertionError("feature build unexpectedly returned no features")
            features[row, indexes] = active.features
        for name in matrices:
            matrices[name][row, indexes] = getattr(active, name)
        calibrated_seeds[:, row, indexes] = active.dstar_seed_scores
        s_available[row, indexes] = active.s_available
        c_available[row, indexes] = active.c_available
    return RandomFrameworkInputs(
        source.split, source.fold, source.stage, queries, source.split_units,
        candidates, source.relevant_hashes, source.exclusion_masks, features,
        labels, eligible, matrices["backbone"], matrices["S"], matrices["C"],
        matrices["T"], matrices["Dstar"], calibrated_seeds, s_available,
        c_available, support,
    )


def _pure_transfer_scores(raw: np.ndarray, candidate_hashes: Sequence[str]) -> np.ndarray:
    """Calibrate transfer while turning a wholly missing query into a hash tie."""

    values = np.asarray(raw, dtype=np.float64).copy()
    tie = candidate_hash_tie_scores(candidate_hashes)
    for row in range(values.shape[0]):
        available = np.isfinite(values[row])
        if not np.any(available):
            values[row] = tie
    from src.transfer_distilled_residual import _robust_z

    calibrated = _robust_z(values)
    for row in range(values.shape[0]):
        missing = ~np.isfinite(values[row])
        if np.any(missing):
            finite = calibrated[row, ~missing]
            floor = float(finite.min(initial=0.0)) - 1.0
            calibrated[row, missing] = floor
    return calibrated
