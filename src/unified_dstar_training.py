"""Shared training and full-library validation for unified D* models."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from src.unified_dstar_data import EvaluationQuery, FeatureStore, QueryEpisode


METRIC_COLUMNS = (
    "H@1",
    "H@10",
    "H@50",
    "MRR",
    "nDCG@10",
    "Recall@1%",
    "Coverage@10",
)


def downweighted_multi_positive_listwise_loss(
    scores: torch.Tensor,
    pos_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    unlabeled_weight: float = 0.1,
) -> torch.Tensor:
    """Compute a multi-positive listwise loss with denominator-only weighting."""

    if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
        raise ValueError("scores must be a rank-2 tensor")
    if not scores.is_floating_point():
        raise ValueError("scores must have a floating-point dtype")
    for name, mask in (("pos_mask", pos_mask), ("valid_mask", valid_mask)):
        if not isinstance(mask, torch.Tensor) or mask.shape != scores.shape:
            raise ValueError(f"{name} must have the same shape as scores")
        if mask.dtype != torch.bool:
            raise ValueError(f"{name} must have dtype bool")

    if isinstance(unlabeled_weight, bool):
        raise ValueError("unlabeled_weight must be finite and in [0, 1]")
    try:
        weight = float(unlabeled_weight)
    except (TypeError, ValueError) as error:
        raise ValueError("unlabeled_weight must be finite and in [0, 1]") from error
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("unlabeled_weight must be finite and in [0, 1]")
    if not bool(torch.isfinite(scores.masked_select(valid_mask)).all()):
        raise ValueError("scores at valid positions must be finite")

    effective_pos = pos_mask & valid_mask
    active_rows = effective_pos.any(dim=1)
    if not bool(active_rows.any()):
        return scores.masked_fill(~valid_mask, 0.0).sum() * 0.0

    active_scores = scores[active_rows]
    active_valid = valid_mask[active_rows]
    active_pos = effective_pos[active_rows]
    active_scores = active_scores.masked_fill(~active_valid, 0.0)
    negative_infinity = torch.tensor(
        float("-inf"), dtype=scores.dtype, device=scores.device
    )

    numerator = torch.logsumexp(
        active_scores.masked_fill(~active_pos, negative_infinity), dim=1
    )
    log_weights = torch.full_like(active_scores, negative_infinity)
    log_weights = torch.where(
        active_pos,
        torch.zeros_like(active_scores),
        log_weights,
    )
    if weight > 0.0:
        unlabeled = active_valid & ~active_pos
        log_weights = torch.where(
            unlabeled,
            torch.full_like(active_scores, math.log(weight)),
            log_weights,
        )
    denominator = torch.logsumexp(active_scores + log_weights, dim=1)
    return -(numerator - denominator).mean()


def _snapshot_state(state: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if state is None:
        return None
    snapshot: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            snapshot[str(key)] = value.detach().cpu().clone()
        else:
            snapshot[str(key)] = copy.deepcopy(value)
    return snapshot


class CheckpointSelector:
    """Select H@10, MRR, nDCG@10 lexicographically, preserving early ties."""

    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_metrics: dict[str, float] | None = None
        self.best_state: dict[str, Any] | None = None
        self._best_objective: tuple[float, float, float] | None = None

    def update(
        self,
        *,
        epoch: int,
        h10: float,
        mrr: float,
        ndcg10: float,
        state: Mapping[str, Any] | None = None,
    ) -> bool:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        values = (float(h10), float(mrr), float(ndcg10))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("checkpoint metrics must be finite")
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("checkpoint metrics must be in [0, 1]")
        if self._best_objective is not None:
            if values < self._best_objective:
                return False
            if (
                values == self._best_objective
                and self.best_epoch is not None
                and epoch >= self.best_epoch
            ):
                return False

        self._best_objective = values
        self.best_epoch = epoch
        self.best_metrics = {
            "H@10": values[0],
            "MRR": values[1],
            "nDCG@10": values[2],
        }
        self.best_state = _snapshot_state(state)
        return True


def _as_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {resolved}")
    return resolved


def _move_ligand_batch(
    feature_store: FeatureStore, keys: Sequence[str], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        tensor.to(device) for tensor in feature_store.ligand_batch(keys)
    )  # type: ignore[return-value]


def _query_metric_row(
    query: EvaluationQuery,
    candidate_hashes: list[str],
    scores: list[float],
) -> dict[str, Any]:
    if len(candidate_hashes) != len(scores):
        raise ValueError("candidate hashes and scores have different lengths")
    if not candidate_hashes:
        raise ValueError(f"evaluation query {query.ligand_key!r} has no candidates")
    if len(candidate_hashes) != len(set(candidate_hashes)):
        raise ValueError(
            f"evaluation query {query.ligand_key!r} has duplicate candidates"
        )
    relevant = set(map(str, query.relevant_hashes))
    if not relevant:
        raise ValueError(
            f"evaluation query {query.ligand_key!r} has no positive labels"
        )
    missing = relevant - set(candidate_hashes)
    if missing:
        raise ValueError(
            f"evaluation query {query.ligand_key!r} positives absent from candidates: "
            f"{sorted(missing)[:5]}"
        )
    if not all(math.isfinite(score) for score in scores):
        raise ValueError(
            f"evaluation query {query.ligand_key!r} produced non-finite scores"
        )

    ranked = sorted(
        zip(scores, candidate_hashes),
        key=lambda item: (-item[0], item[1]),
    )
    positive_ranks = [
        rank
        for rank, (_, sequence_md5) in enumerate(ranked, start=1)
        if sequence_md5 in relevant
    ]
    n_candidates = len(candidate_hashes)
    n_positives = len(relevant)
    best_rank = positive_ranks[0]

    def hits_at(k: int) -> int:
        return sum(rank <= k for rank in positive_ranks)

    dcg10 = sum(
        1.0 / math.log2(rank + 1.0) for rank in positive_ranks if rank <= 10
    )
    ideal_dcg10 = sum(
        1.0 / math.log2(rank + 1.0)
        for rank in range(1, min(n_positives, 10) + 1)
    )
    recall_cutoff = max(1, math.ceil(0.01 * n_candidates))
    return {
        "ligand_key": str(query.ligand_key),
        "split_unit": str(query.split_unit),
        "n_candidates": n_candidates,
        "n_positives": n_positives,
        "best_positive_rank": best_rank,
        "positive_ranks": ";".join(map(str, positive_ranks)),
        "H@1": float(best_rank <= 1),
        "H@10": float(best_rank <= 10),
        "H@50": float(best_rank <= 50),
        "MRR": 1.0 / best_rank,
        "nDCG@10": dcg10 / ideal_dcg10,
        "Recall@1%": hits_at(recall_cutoff) / n_positives,
        "Coverage@10": hits_at(10) / n_positives,
    }


def evaluate_full_library(
    model: torch.nn.Module,
    queries: Sequence[EvaluationQuery],
    feature_store: FeatureStore,
    candidate_batch_size: int = 1024,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Score every query-specific candidate and compute positive-only metrics."""

    if (
        not isinstance(candidate_batch_size, int)
        or isinstance(candidate_batch_size, bool)
        or candidate_batch_size <= 0
    ):
        raise ValueError("candidate_batch_size must be a positive integer")
    if not queries:
        raise ValueError("queries must contain at least one evaluation query")

    resolved_device = _as_device(device)
    model.to(resolved_device)
    was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for query in queries:
                candidate_hashes = list(map(str, query.candidate_hashes))
                ligand_inputs = _move_ligand_batch(
                    feature_store, [str(query.ligand_key)], resolved_device
                )
                ligand_z = model.encode_ligand(*ligand_inputs)
                query_scores: list[float] = []
                for start in range(0, len(candidate_hashes), candidate_batch_size):
                    batch_hashes = candidate_hashes[
                        start : start + candidate_batch_size
                    ]
                    protein_features = feature_store.protein_batch(batch_hashes).to(
                        resolved_device
                    )
                    protein_z = model.encode_protein(protein_features)
                    batch_scores = model.score(ligand_z, protein_z)
                    if batch_scores.shape != (1, len(batch_hashes)):
                        raise ValueError(
                            "model.score must return one score per query-candidate pair"
                        )
                    query_scores.extend(
                        batch_scores[0].detach().cpu().to(torch.float64).tolist()
                    )
                rows.append(
                    _query_metric_row(query, candidate_hashes, query_scores)
                )
    finally:
        model.train(was_training)

    per_query = pd.DataFrame.from_records(rows)
    aggregate = {
        metric: float(per_query[metric].mean()) for metric in METRIC_COLUMNS
    }
    return aggregate, per_query


def _stable_content_hash(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _training_content(episodes: Sequence[QueryEpisode]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "ligand_key": str(episode.ligand_key),
                "positive_hashes": sorted(map(str, episode.positive_hashes)),
                "unlabeled_hashes": sorted(map(str, episode.unlabeled_hashes)),
            }
            for episode in episodes
        ),
        key=lambda row: (
            row["ligand_key"],
            row["positive_hashes"],
            row["unlabeled_hashes"],
        ),
    )


def _validation_content(
    queries: Sequence[EvaluationQuery],
) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "ligand_key": str(query.ligand_key),
                "positive_hashes": sorted(map(str, query.relevant_hashes)),
                "candidate_hashes": sorted(map(str, query.candidate_hashes)),
                "train_positive_hashes": sorted(
                    map(str, query.train_positive_hashes)
                ),
                "split_unit": str(query.split_unit),
            }
            for query in queries
        ),
        key=lambda row: (
            row["ligand_key"],
            row["positive_hashes"],
            row["candidate_hashes"],
            row["train_positive_hashes"],
            row["split_unit"],
        ),
    )


def _atomic_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, sep="\t", index=False)
    os.replace(temporary, path)


def _json_compatible(value: Any, *, location: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(
                f"{location} must contain finite JSON-compatible numbers"
            )
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_compatible(value.item(), location=location)
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist(), location=location)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                converted_key = key
            elif isinstance(key, Path):
                converted_key = str(key)
            else:
                raise TypeError(
                    f"{location} has non-string JSON-compatible mapping key "
                    f"of type {type(key).__name__}"
                )
            if converted_key in converted:
                raise TypeError(
                    f"{location} has mapping keys that collide after conversion: "
                    f"{converted_key!r}"
                )
            converted[converted_key] = _json_compatible(
                item, location=f"{location}.{converted_key}"
            )
        return converted
    if isinstance(value, (list, tuple)):
        return [
            _json_compatible(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{location} contains unsupported non-JSON-compatible type "
        f"{type(value).__name__}"
    )


def _stable_json_text(value: Mapping[str, Any]) -> str:
    compatible = _json_compatible(value)
    try:
        return (
            json.dumps(
                compatible,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "manifest could not be serialized as stable JSON-compatible content"
        ) from error


def _atomic_json_text(serialized: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_training_inputs(
    episodes: Sequence[QueryEpisode],
    validation_queries: Sequence[EvaluationQuery],
    unlabeled_per_query: int,
) -> None:
    if not episodes:
        raise ValueError("training_episodes must not be empty")
    if not validation_queries:
        raise ValueError("validation_queries must not be empty")
    for episode in episodes:
        positives = tuple(map(str, episode.positive_hashes))
        unlabeled = tuple(map(str, episode.unlabeled_hashes))
        if not positives:
            raise ValueError(
                f"training episode {episode.ligand_key!r} has no positives"
            )
        if len(unlabeled) != unlabeled_per_query:
            raise ValueError(
                f"training episode {episode.ligand_key!r} must have exactly "
                f"{unlabeled_per_query} unlabeled candidates"
            )
        if len(set(positives)) != len(positives) or len(set(unlabeled)) != len(
            unlabeled
        ):
            raise ValueError(
                f"training episode {episode.ligand_key!r} has duplicate candidates"
            )
        if set(positives) & set(unlabeled):
            raise ValueError(
                f"training episode {episode.ligand_key!r} overlaps positive "
                "and unlabeled candidates"
            )


def _train_query_batch(
    model: torch.nn.Module,
    episodes: Sequence[QueryEpisode],
    feature_store: FeatureStore,
    device: torch.device,
    unlabeled_weight: float,
) -> torch.Tensor:
    ligand_inputs = _move_ligand_batch(
        feature_store,
        [str(episode.ligand_key) for episode in episodes],
        device,
    )
    ligand_z = model.encode_ligand(*ligand_inputs)
    candidate_lists = [
        list(map(str, episode.positive_hashes))
        + list(map(str, episode.unlabeled_hashes))
        for episode in episodes
    ]
    flattened = [value for candidates in candidate_lists for value in candidates]
    protein_z = model.encode_protein(
        feature_store.protein_batch(flattened).to(device)
    )
    offset = 0
    score_rows: list[torch.Tensor] = []
    positive_rows: list[torch.Tensor] = []
    for row_index, (episode, candidates) in enumerate(zip(episodes, candidate_lists)):
        width = len(candidates)
        query_scores = model.score(
            ligand_z[row_index : row_index + 1],
            protein_z[offset : offset + width],
        )
        if query_scores.shape != (1, width):
            raise ValueError(
                "model.score must return one score per query-candidate pair"
            )
        score_rows.append(query_scores[0])
        positive_rows.append(
            torch.arange(width, device=device) < len(episode.positive_hashes)
        )
        offset += width
    padded_scores = pad_sequence(score_rows, batch_first=True)
    pos_mask = pad_sequence(positive_rows, batch_first=True)
    lengths = torch.tensor([len(row) for row in score_rows], device=device)
    valid_mask = (
        torch.arange(padded_scores.shape[1], device=device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    return downweighted_multi_positive_listwise_loss(
        padded_scores,
        pos_mask,
        valid_mask,
        unlabeled_weight=unlabeled_weight,
    )


def _run_training_epoch(
    model: torch.nn.Module,
    training_episodes: Sequence[QueryEpisode],
    feature_store: FeatureStore,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    query_batch_size: int,
    unlabeled_weight: float,
    random_generator: random.Random,
    epoch: int,
) -> float:
    """Run every episode once, sharing the epoch body between fit modes."""

    model.train()
    indices = list(range(len(training_episodes)))
    random_generator.shuffle(indices)
    losses: list[float] = []
    for start in range(0, len(indices), query_batch_size):
        batch = [
            training_episodes[index]
            for index in indices[start : start + query_batch_size]
        ]
        optimizer.zero_grad(set_to_none=True)
        loss = _train_query_batch(
            model, batch, feature_store, device, unlabeled_weight
        )
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"non-finite training loss at epoch {epoch}")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError(f"non-finite gradient for parameter {name!r} at epoch {epoch}")
        try:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
        except RuntimeError as error:
            raise ValueError(f"non-finite gradient norm at epoch {epoch}") from error
        optimizer.step()
        for name, parameter in model.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise ValueError(f"non-finite model parameter {name!r} after optimizer step at epoch {epoch}")
        _assert_finite_tensor_tree(optimizer.state_dict(), "optimizer state")
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def _assert_finite_tensor_tree(value: Any, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite tensor in {label}")
    elif isinstance(value, Mapping):
        for child in value.values():
            _assert_finite_tensor_tree(child, label)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_finite_tensor_tree(child, label)


def _validate_fixed_training_inputs(
    episodes: Sequence[QueryEpisode], unlabeled_per_query: int
) -> None:
    if not episodes:
        raise ValueError("training_episodes must not be empty")
    for episode in episodes:
        positives = tuple(map(str, episode.positive_hashes))
        unlabeled = tuple(map(str, episode.unlabeled_hashes))
        if not positives:
            raise ValueError(f"training episode {episode.ligand_key!r} has no positives")
        if len(unlabeled) != unlabeled_per_query:
            raise ValueError(
                f"training episode {episode.ligand_key!r} must have exactly "
                f"{unlabeled_per_query} unlabeled candidates"
            )
        if len(positives) != len(set(positives)) or len(unlabeled) != len(set(unlabeled)):
            raise ValueError(f"training episode {episode.ligand_key!r} has duplicate candidates")
        if set(positives) & set(unlabeled):
            raise ValueError(
                f"training episode {episode.ligand_key!r} overlaps positive and unlabeled candidates"
            )


def train_fixed_epochs(
    model: torch.nn.Module,
    training_episodes: Sequence[QueryEpisode],
    feature_store: FeatureStore,
    output_dir: str | Path,
    *,
    epochs: int,
    device: str | torch.device = "cpu",
    query_batch_size: int = 16,
    unlabeled_per_query: int = 128,
    unlabeled_weight: float = 0.1,
    seed: int = 42,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Refit on all supplied episodes for exactly ``epochs`` without validation."""

    integers = {
        "epochs": epochs,
        "query_batch_size": query_batch_size,
        "unlabeled_per_query": unlabeled_per_query,
    }
    for name, value in integers.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if isinstance(unlabeled_weight, bool) or not math.isfinite(float(unlabeled_weight)):
        raise ValueError("unlabeled_weight must be finite")
    _validate_fixed_training_inputs(training_episodes, unlabeled_per_query)

    resolved_device = _as_device(device)
    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.parent / f".{output_path.name}.train.lock"
    try:
        lock_path.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"training output is locked: {output_path}") from error
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.staging.", dir=output_path.parent))
    try:
        random_generator = random.Random(seed)
        torch.manual_seed(seed)
        if resolved_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model.to(resolved_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        history_rows: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            learning_rate = float(optimizer.param_groups[0]["lr"])
            loss = _run_training_epoch(
                model, training_episodes, feature_store, optimizer,
                device=resolved_device, query_batch_size=query_batch_size,
                unlabeled_weight=float(unlabeled_weight),
                random_generator=random_generator, epoch=epoch,
            )
            scheduler.step()
            _assert_finite_tensor_tree(scheduler.state_dict(), "scheduler state")
            history_rows.append({
                "epoch": epoch, "train_loss": loss, "learning_rate": learning_rate
            })

        training_config: dict[str, Any] = {
            "model_class": model.__class__.__name__,
            "architecture": getattr(model, "architecture", None),
            "seed": seed,
            "device": str(resolved_device),
            "epochs": epochs,
            "query_batch_size": query_batch_size,
            "unlabeled_per_query": unlabeled_per_query,
            "unlabeled_weight": float(unlabeled_weight),
            "gradient_clip_norm": 1.0,
            "optimizer": {"name": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4},
            "scheduler": {"name": "CosineAnnealingLR", "T_max": 50},
            "run_config": dict(config or {}),
        }
        content = _training_content(training_episodes)
        manifest: dict[str, Any] = {
            "schema": "unified_dstar_fixed_epoch_refit",
            "schema_version": 1,
            "status": "complete",
            "validation_used": False,
            "case_inputs_used": [],
            "target_identity_used": False,
            "requested_epochs": epochs,
            "actual_epochs": len(history_rows),
            "train_content_sha256": _stable_content_hash(content),
            "train_hash_basis": "stable_complete_training_episode_content",
            "query_count": len(training_episodes),
            "positive_count": sum(len(row["positive_hashes"]) for row in content),
            "feature_store_audit": feature_store.audit(),
            "config": training_config,
        }
        state = _snapshot_state(model.state_dict())
        _assert_finite_tensor_tree(state, "final checkpoint state")
        checkpoint = {
            "epoch": epochs,
            "model_state_dict": state,
            "config": training_config,
            "validation_used": False,
        }
        _atomic_torch_save(checkpoint, staging / "last_model.pt")
        _atomic_dataframe(pd.DataFrame.from_records(history_rows), staging / "train_history.tsv")
        _atomic_json_text(_stable_json_text(manifest), staging / "refit_manifest.json")
        for item in staging.iterdir():
            with item.open("rb") as handle:
                os.fsync(handle.fileno())
        directory_fd = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite output directory: {output_path}")
        os.rename(staging, output_path)
        parent_fd = os.open(output_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return {"actual_epochs": epochs, "output_dir": str(output_path)}
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if lock_path.exists():
            lock_path.rmdir()


def train_fold(
    model: torch.nn.Module,
    training_episodes: Sequence[QueryEpisode],
    validation_queries: Sequence[EvaluationQuery],
    feature_store: FeatureStore,
    output_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    max_epochs: int = 50,
    validation_interval: int = 5,
    patience: int = 3,
    query_batch_size: int = 16,
    unlabeled_per_query: int = 128,
    unlabeled_weight: float = 0.1,
    candidate_batch_size: int = 1024,
    seed: int = 42,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one fold using train episodes and validation queries only."""

    integer_options = {
        "max_epochs": max_epochs,
        "validation_interval": validation_interval,
        "patience": patience,
        "query_batch_size": query_batch_size,
        "unlabeled_per_query": unlabeled_per_query,
        "candidate_batch_size": candidate_batch_size,
    }
    for name, value in integer_options.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    _validate_training_inputs(
        training_episodes, validation_queries, unlabeled_per_query
    )

    resolved_device = _as_device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    random_generator = random.Random(seed)
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model.to(resolved_device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50
    )
    selector = CheckpointSelector()
    history_rows: list[dict[str, Any]] = []
    best_per_query: pd.DataFrame | None = None
    best_validation_metrics: dict[str, float] | None = None
    validations_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        epoch_loss = _run_training_epoch(
            model,
            training_episodes,
            feature_store,
            optimizer,
            device=resolved_device,
            query_batch_size=query_batch_size,
            unlabeled_weight=unlabeled_weight,
            random_generator=random_generator,
            epoch=epoch,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "learning_rate": current_lr,
            "validated": False,
        }
        should_validate = (
            epoch % validation_interval == 0 or epoch == max_epochs
        )
        if should_validate:
            metrics, per_query = evaluate_full_library(
                model,
                validation_queries,
                feature_store,
                candidate_batch_size=candidate_batch_size,
                device=resolved_device,
            )
            improved = selector.update(
                epoch=epoch,
                h10=metrics["H@10"],
                mrr=metrics["MRR"],
                ndcg10=metrics["nDCG@10"],
                state=model.state_dict(),
            )
            row.update({"validated": True, **metrics})
            if improved:
                best_per_query = per_query.copy()
                best_validation_metrics = dict(metrics)
                validations_without_improvement = 0
            else:
                validations_without_improvement += 1
        history_rows.append(row)
        if should_validate and validations_without_improvement >= patience:
            break

    if (
        selector.best_epoch is None
        or selector.best_metrics is None
        or selector.best_state is None
        or best_per_query is None
        or best_validation_metrics is None
    ):
        raise RuntimeError("training completed without a validation checkpoint")
    model.load_state_dict(selector.best_state)

    training_config: dict[str, Any] = {
        "model_class": model.__class__.__name__,
        "architecture": getattr(model, "architecture", None),
        "seed": seed,
        "device": str(resolved_device),
        "max_epochs": max_epochs,
        "validation_interval": validation_interval,
        "patience": patience,
        "query_batch_size": query_batch_size,
        "unlabeled_per_query": unlabeled_per_query,
        "unlabeled_weight": float(unlabeled_weight),
        "candidate_batch_size": candidate_batch_size,
        "gradient_clip_norm": 1.0,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        },
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 50},
        "run_config": dict(config or {}),
    }
    candidate_counts = {
        str(query.ligand_key): len(query.candidate_hashes)
        for query in validation_queries
    }
    unique_candidate_counts = sorted(set(candidate_counts.values()))
    candidate_count: int | None = (
        unique_candidate_counts[0] if len(unique_candidate_counts) == 1 else None
    )
    manifest: dict[str, Any] = {
        "train_content_sha256": _stable_content_hash(
            _training_content(training_episodes)
        ),
        "train_hash_basis": "stable_training_episode_content",
        "validation_content_sha256": _stable_content_hash(
            _validation_content(validation_queries)
        ),
        "validation_hash_basis": "stable_validation_query_content",
        "config": training_config,
        "feature_store_audit": feature_store.audit(),
        "candidate_count": candidate_count,
        "candidate_counts_by_query": candidate_counts,
        "selected_epoch": selector.best_epoch,
        "validation_metrics": best_validation_metrics,
    }
    checkpoint = {
        "epoch": selector.best_epoch,
        "metrics": best_validation_metrics,
        "selection_metrics": selector.best_metrics,
        "model_state_dict": selector.best_state,
        "config": training_config,
    }
    history = pd.DataFrame.from_records(history_rows)
    serialized_manifest = _stable_json_text(manifest)

    _atomic_torch_save(checkpoint, output_path / "best_model.pt")
    _atomic_dataframe(history, output_path / "train_history.tsv")
    _atomic_dataframe(
        best_per_query, output_path / "validation_per_query.tsv"
    )
    _atomic_json_text(serialized_manifest, output_path / "fold_manifest.json")
    return {
        "selected_epoch": selector.best_epoch,
        "validation_metrics": best_validation_metrics,
        "candidate_count": candidate_count,
        "output_dir": str(output_path),
    }
