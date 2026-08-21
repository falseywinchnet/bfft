"""Spectral response heads for chart allocation.

The affine geometry remains the responsibility of self-context/CFF.  This
module changes only the scalar response assigned to each candidate chart:

    features -> symmetric affine matrix pencil -> selected eigenvalue.

The lower-triangular coordinates are embedded isometrically, so ordinary
optimizer steps do not accidentally privilege diagonal or off-diagonal matrix
entries.  Initialization follows the gap-preserving construction in the
spectral-neuron paper, including the small diagonal jitter needed to prevent
the feature matrices from commuting with the randomly rotated base matrix.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SymmetricPackedEmbedding(nn.Module):
    """Isometrically unpack lower-triangular coordinates into symmetric matrices."""

    def __init__(self, matrix_dim: int):
        super().__init__()
        if matrix_dim < 2:
            raise ValueError("matrix_dim must be at least two")
        self.matrix_dim = int(matrix_dim)
        row, column = torch.tril_indices(matrix_dim, matrix_dim)
        packed_index = torch.empty(matrix_dim, matrix_dim, dtype=torch.long)
        coordinates = torch.arange(len(row))
        packed_index[row, column] = coordinates
        packed_index[column, row] = coordinates
        self.register_buffer("packed_index", packed_index.flatten(), persistent=False)
        self.register_buffer("diagonal", row == column, persistent=False)

    @property
    def packed_dim(self) -> int:
        return self.matrix_dim * (self.matrix_dim + 1) // 2

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.shape[-1] != self.packed_dim:
            raise ValueError(
                f"expected {self.packed_dim} packed coordinates, got {packed.shape[-1]}"
            )
        scaled = torch.where(
            self.diagonal,
            packed,
            packed / math.sqrt(2.0),
        )
        return scaled[..., self.packed_index].view(
            *packed.shape[:-1], self.matrix_dim, self.matrix_dim
        )

    def pack(self, matrix: torch.Tensor) -> torch.Tensor:
        """Inverse embedding for a symmetric matrix."""
        row, column = torch.tril_indices(
            self.matrix_dim, self.matrix_dim, device=matrix.device
        )
        packed = matrix[..., row, column]
        return torch.where(self.diagonal, packed, packed * math.sqrt(2.0))


class SpectralResponse(nn.Module):
    """Return one ordered eigenvalue of a learned symmetric affine pencil."""

    def __init__(
        self,
        feature_count: int,
        *,
        matrix_dim: int = 5,
        eigen_index: int | None = None,
        jitter: bool = True,
    ):
        super().__init__()
        if feature_count < 1:
            raise ValueError("feature_count must be positive")
        self.feature_count = int(feature_count)
        self.matrix_dim = int(matrix_dim)
        self.eigen_index = (
            matrix_dim // 2 if eigen_index is None else int(eigen_index)
        )
        if not 0 <= self.eigen_index < matrix_dim:
            raise ValueError("eigen_index is outside the spectrum")
        self.embedding = SymmetricPackedEmbedding(matrix_dim)
        self.pencil = nn.Linear(feature_count, self.embedding.packed_dim)
        self.jitter = bool(jitter)
        self.last_gap: torch.Tensor | None = None
        self.last_value: torch.Tensor | None = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        device = self.pencil.weight.device
        dtype = self.pencil.weight.dtype

        random_matrix = torch.randn(
            self.matrix_dim, self.matrix_dim, device=device, dtype=dtype
        )
        orthogonal, triangular = torch.linalg.qr(random_matrix)
        signs = torch.where(triangular.diagonal() < 0, -1.0, 1.0)
        orthogonal = orthogonal * signs
        spectrum = (
            torch.arange(self.matrix_dim, device=device) - self.eigen_index
        ).sign().to(dtype)
        base = (orthogonal * spectrum) @ orthogonal.mT
        self.pencil.bias.copy_(self.embedding.pack(base))

        # Each feature initially has a familiar affine slope alpha_i I.  The
        # small diagonal jitter makes the matrices noncommuting without
        # sacrificing the unit eigengap around the selected eigenvalue.
        bound = self.feature_count**-0.5
        alpha = torch.empty(
            self.feature_count, device=device, dtype=dtype
        ).uniform_(-bound, bound)
        matrices = alpha[:, None, None] * torch.eye(
            self.matrix_dim, device=device, dtype=dtype
        )
        if self.jitter:
            epsilon = torch.empty(
                self.feature_count, self.matrix_dim, device=device, dtype=dtype
            ).uniform_(
                -1.0 / (20.0 * self.feature_count),
                1.0 / (20.0 * self.feature_count),
            )
            matrices = matrices + torch.diag_embed(epsilon)
        self.pencil.weight.copy_(self.embedding.pack(matrices).mT)

    def coefficient_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return A0 and feature matrices Ai in the affine pencil."""
        return (
            self.embedding(self.pencil.bias),
            self.embedding(self.pencil.weight.mT),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        matrix = self.embedding(self.pencil(features))
        eigenvalues = torch.linalg.eigvalsh(matrix)
        value = eigenvalues[..., self.eigen_index]

        if self.eigen_index == 0:
            gap = eigenvalues[..., 1] - eigenvalues[..., 0]
        elif self.eigen_index == self.matrix_dim - 1:
            gap = eigenvalues[..., -1] - eigenvalues[..., -2]
        else:
            lower = value - eigenvalues[..., self.eigen_index - 1]
            upper = eigenvalues[..., self.eigen_index + 1] - value
            gap = torch.minimum(lower, upper)
        self.last_gap = gap.detach()
        self.last_value = value.detach()
        return value.unsqueeze(-1)


def spectral_response_summary(model: nn.Module) -> dict[str, float | None]:
    """Summarize the most recent eigengaps across all spectral heads."""
    gaps = [
        module.last_gap.flatten()
        for module in model.modules()
        if isinstance(module, SpectralResponse) and module.last_gap is not None
    ]
    if not gaps:
        return {
            "spectral_gap_mean": None,
            "spectral_gap_p10": None,
            "spectral_gap_min": None,
        }
    gap = torch.cat(gaps).float()
    return {
        "spectral_gap_mean": float(gap.mean()),
        "spectral_gap_p10": float(torch.quantile(gap, 0.1)),
        "spectral_gap_min": float(gap.min()),
    }
