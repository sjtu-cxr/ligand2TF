"""Canonical fold-local chemical response-transfer definitions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys


FINGERPRINT_NAMES = ("morgan_r2", "morgan_r3", "maccs", "rdkit")


@dataclass(frozen=True)
class ChemicalTransferResult:
    scores: Mapping[str, np.ndarray]
    available: np.ndarray
    witness_counts: np.ndarray


def select_configuration(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Select a summarized validation configuration by retrieval priorities."""

    if not rows:
        raise ValueError("configuration rows must not be empty")
    required = ("hit_at_10", "hit_at_50", "mrr")
    normalized = []
    for row in rows:
        record = dict(row)
        values = tuple(float(record[name]) for name in required)
        if not np.isfinite(values).all():
            raise ValueError("configuration metrics must be finite")
        normalized.append(record)
    return max(
        normalized,
        key=lambda row: tuple(float(row[name]) for name in required),
    )


def screen_variants(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """Apply the frozen Stage 1 competitiveness and no-collapse rule."""

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    retained = []
    for variant, records in sorted(grouped.items()):
        if len({str(record["split"]) for record in records}) != 2:
            raise ValueError("each variant requires both evaluated splits")
        competitive = any(
            float(record["delta_hit_at_10"]) >= 0.0
            and float(record["delta_hit_at_50"]) >= 0.0
            for record in records
        )
        no_collapse = all(
            float(record["delta_hit_at_10"]) >= -0.01
            and float(record["delta_hit_at_50"]) >= -0.02
            for record in records
        )
        if competitive and no_collapse:
            retained.append(variant)
    return tuple(retained)


def comparison_deltas(summary: pd.DataFrame) -> list[dict[str, object]]:
    """Compare proposed summary rows with each split's current backbone."""

    required = {"split", "variant", "hit@10", "hit@50", "mrr"}
    if not required <= set(summary.columns):
        raise ValueError("summary lacks comparison columns")
    current = summary.loc[summary["variant"].eq("current_backbone")]
    if current["split"].duplicated().any():
        raise ValueError("current backbone must be unique within each split")
    current_by_split = current.set_index("split")
    rows = []
    for _, record in summary.loc[~summary["variant"].eq("current_backbone")].iterrows():
        split = str(record["split"])
        if split not in current_by_split.index:
            raise ValueError(f"current backbone missing split {split}")
        baseline = current_by_split.loc[split]
        rows.append({
            "split": split,
            "variant": str(record["variant"]),
            "delta_hit_at_10": round(float(record["hit@10"] - baseline["hit@10"]), 12),
            "delta_hit_at_50": round(float(record["hit@50"] - baseline["hit@50"]), 12),
            "delta_mrr": round(float(record["mrr"] - baseline["mrr"]), 12),
        })
    return rows


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
    pairs = tuple(sorted({(str(ligand), str(candidate)) for ligand, candidate in records}))
    if any(not ligand or not candidate for ligand, candidate in pairs):
        raise ValueError("fit edges require nonempty ligand and candidate identities")
    return pairs


def _fingerprints(ligands: Sequence[str]):
    output: dict[str, dict[str, object | None]] = {
        name: {} for name in FINGERPRINT_NAMES
    }
    for ligand in ligands:
        molecule = None if str(ligand).startswith("ion:") else Chem.MolFromSmiles(str(ligand))
        output["morgan_r2"][ligand] = (
            None if molecule is None else AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
        )
        output["morgan_r3"][ligand] = (
            None if molecule is None else AllChem.GetMorganFingerprintAsBitVect(molecule, 3, nBits=2048)
        )
        output["maccs"][ligand] = None if molecule is None else MACCSkeys.GenMACCSKeys(molecule)
        output["rdkit"][ligand] = None if molecule is None else Chem.RDKFingerprint(molecule)
    return output


def _rdkit_similarity_tables(ligands: Sequence[str]):
    fingerprints = _fingerprints(ligands)
    tables: dict[str, dict[tuple[str, str], float]] = {
        name: {} for name in FINGERPRINT_NAMES
    }
    for name, cache in fingerprints.items():
        for left in ligands:
            left_fp = cache[left]
            if left_fp is None:
                continue
            for right in ligands:
                right_fp = cache[right]
                if right_fp is not None:
                    tables[name][(left, right)] = float(
                        DataStructs.TanimotoSimilarity(left_fp, right_fp)
                    )
    return tables


def chemical_transfer_scores(
    *,
    query_ids: Sequence[str],
    candidate_hashes: Sequence[str],
    fit_edges: pd.DataFrame | Sequence[tuple[str, str]],
    similarity_tables: Mapping[str, Mapping[tuple[str, str], float]] | None = None,
) -> ChemicalTransferResult:
    """Score direct candidate response memory with Morgan or four fingerprints."""

    queries = tuple(map(str, query_ids))
    candidates = tuple(map(str, candidate_hashes))
    if not queries or not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("queries and unique candidates must be nonempty")
    candidate_index = {value: index for index, value in enumerate(candidates)}
    pairs = _edge_pairs(fit_edges)
    unknown = {candidate for _, candidate in pairs}.difference(candidates)
    if unknown:
        raise ValueError(f"fit responder absent from candidates: {sorted(unknown)[:3]}")
    histories: dict[str, list[str]] = defaultdict(list)
    for ligand, candidate in pairs:
        histories[candidate].append(ligand)
    if similarity_tables is None:
        universe = tuple(sorted(set(queries) | {ligand for ligand, _ in pairs}))
        tables = _rdkit_similarity_tables(universe)
    else:
        if set(similarity_tables) != set(FINGERPRINT_NAMES):
            raise ValueError(f"similarity tables must be {FINGERPRINT_NAMES}")
        tables = {name: dict(similarity_tables[name]) for name in FINGERPRINT_NAMES}

    per_fp = {
        name: np.zeros((len(queries), len(candidates)), dtype=np.float64)
        for name in FINGERPRINT_NAMES
    }
    available = np.zeros((len(queries), len(candidates)), dtype=bool)
    counts = np.zeros((len(queries), len(candidates)), dtype=np.int32)
    for query_index, query in enumerate(queries):
        query_known = any(any(left == query for left, _ in table) for table in tables.values())
        if not query_known:
            continue
        for candidate, support_ligands in histories.items():
            nonself = [ligand for ligand in support_ligands if ligand != query]
            if not nonself:
                continue
            column = candidate_index[candidate]
            available[query_index, column] = True
            usable = [
                ligand
                for ligand in nonself
                if any((query, ligand) in tables[name] for name in FINGERPRINT_NAMES)
            ]
            if not usable:
                continue
            counts[query_index, column] = len(usable)
            for name in FINGERPRINT_NAMES:
                values = [
                    float(tables[name][(query, ligand)])
                    for ligand in usable
                    if (query, ligand) in tables[name]
                ]
                if values:
                    per_fp[name][query_index, column] = max(values)
    multifp = np.mean(np.stack([per_fp[name] for name in FINGERPRINT_NAMES]), axis=0)
    return ChemicalTransferResult(
        scores={"morgan": per_fp["morgan_r2"], "multifp": multifp},
        available=available,
        witness_counts=counts,
    )
