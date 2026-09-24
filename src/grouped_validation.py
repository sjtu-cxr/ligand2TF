"""Frozen atomic-group validation partitioning."""
from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from hashlib import sha256
import numpy as np

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def atomic_split_units(value: str) -> tuple[str, ...]:
    """Parse one canonical, possibly composite, split-unit value."""

    _require(isinstance(value, str), "split unit must be a string")
    atoms = tuple(part.strip() for part in value.split(";"))
    _require(
        bool(atoms) and all(atoms) and len(set(atoms)) == len(atoms),
        f"invalid atomic split-unit value: {value!r}",
    )
    return atoms


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
