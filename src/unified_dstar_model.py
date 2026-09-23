"""Common dual-encoder API for the two D* candidate architectures."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Projection(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class UnifiedDStar(nn.Module):
    """Dual encoder supporting the prespecified Architecture A and B variants."""

    def __init__(
        self,
        architecture: str,
        molformer_dim: int = 768,
        ecfp_dim: int = 2048,
        ion_dim: int = 10,
        esm_dim: int = 640,
        hidden_dim: int = 256,
        contrast_dim: int = 128,
        dropout: float = 0.1,
        temperature_init: float = 0.07,
        residual_gate_init: float = -3.0,
    ) -> None:
        super().__init__()
        if architecture not in {"A", "B"}:
            raise ValueError("architecture must be 'A' or 'B'")
        dimensions = {
            "molformer_dim": molformer_dim,
            "ecfp_dim": ecfp_dim,
            "ion_dim": ion_dim,
            "esm_dim": esm_dim,
            "hidden_dim": hidden_dim,
            "contrast_dim": contrast_dim,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        temperature = float(temperature_init)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature_init must be finite and positive")
        residual_gate = float(residual_gate_init)
        if not math.isfinite(residual_gate):
            raise ValueError("residual_gate_init must be finite")

        self.architecture = architecture
        self.molformer_dim = molformer_dim
        self.ecfp_dim = ecfp_dim
        self.ion_dim = ion_dim
        self.esm_dim = esm_dim
        self.contrast_dim = contrast_dim

        projection_args = (hidden_dim, contrast_dim, float(dropout))
        self.molformer_projection = _Projection(molformer_dim, *projection_args)
        self.ecfp_projection = _Projection(ecfp_dim, *projection_args)
        self.ion_projection = _Projection(ion_dim, *projection_args)
        self.protein_projection = _Projection(esm_dim, *projection_args)

        if architecture == "B":
            self.ligand_residual_gate = nn.Parameter(
                torch.tensor(residual_gate, dtype=torch.float32)
            )
            self.protein_residual_gate = nn.Parameter(
                torch.tensor(residual_gate, dtype=torch.float32)
            )
            self.protein_residual_adapter = nn.Sequential(
                nn.LayerNorm(esm_dim),
                nn.Linear(esm_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, esm_dim),
            )

        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature), dtype=torch.float32)
        )

    @staticmethod
    def _validate_matrix(
        value: torch.Tensor,
        name: str,
        rows: int | None,
        columns: int,
    ) -> None:
        expected_rows = "B" if rows is None else str(rows)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.shape[1] != columns
            or (rows is not None and value.shape[0] != rows)
        ):
            raise ValueError(
                f"{name} must have shape [{expected_rows}, {columns}], "
                f"got {tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__}"
            )

    def _validate_ligand_inputs(
        self,
        molformer: torch.Tensor,
        ecfp: torch.Tensor,
        ion_features: torch.Tensor,
        ion_mask: torch.Tensor,
    ) -> None:
        self._validate_matrix(molformer, "molformer", None, self.molformer_dim)
        batch_size = molformer.shape[0]
        self._validate_matrix(ecfp, "ecfp", batch_size, self.ecfp_dim)
        self._validate_matrix(ion_features, "ion_features", batch_size, self.ion_dim)
        if (
            not isinstance(ion_mask, torch.Tensor)
            or ion_mask.ndim != 1
            or ion_mask.shape[0] != batch_size
        ):
            raise ValueError(
                f"ion_mask must have shape [{batch_size}], "
                f"got {tuple(ion_mask.shape) if isinstance(ion_mask, torch.Tensor) else type(ion_mask).__name__}"
            )
        if ion_mask.dtype != torch.bool:
            raise ValueError(f"ion_mask must have dtype bool, got {ion_mask.dtype}")

    @staticmethod
    def _finite_or_zero(value: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

    def encode_ligand(
        self,
        molformer: torch.Tensor,
        ecfp: torch.Tensor,
        ion_features: torch.Tensor,
        ion_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_ligand_inputs(molformer, ecfp, ion_features, ion_mask)
        finite_molformer = self._finite_or_zero(molformer)
        finite_ecfp = self._finite_or_zero(ecfp)
        finite_ion_features = self._finite_or_zero(ion_features)

        if self.architecture == "A":
            regular = self.ecfp_projection(finite_ecfp)
            valid_molformer = torch.isfinite(molformer).all(dim=1) & (molformer != 0).any(dim=1)
            if bool(valid_molformer.any()):
                regular = regular.clone()
                regular[valid_molformer] = self.molformer_projection(
                    finite_molformer[valid_molformer]
                )
        else:
            regular = self.molformer_projection(finite_molformer)
            regular = regular + torch.sigmoid(self.ligand_residual_gate) * self.ecfp_projection(
                finite_ecfp
            )

        ion = self.ion_projection(finite_ion_features)
        combined = torch.where(ion_mask.unsqueeze(1), ion, regular)
        return F.normalize(combined, dim=-1)

    def encode_protein(self, esm: torch.Tensor) -> torch.Tensor:
        self._validate_matrix(esm, "esm", None, self.esm_dim)
        esm = self._finite_or_zero(esm)
        if self.architecture == "B":
            esm = esm + torch.sigmoid(self.protein_residual_gate) * self.protein_residual_adapter(
                esm
            )
        return F.normalize(self.protein_projection(esm), dim=-1)

    def score(self, ligand_z: torch.Tensor, protein_z: torch.Tensor) -> torch.Tensor:
        self._validate_matrix(ligand_z, "ligand_z", None, self.contrast_dim)
        self._validate_matrix(protein_z, "protein_z", None, self.contrast_dim)
        scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        return scale * (ligand_z @ protein_z.T)
