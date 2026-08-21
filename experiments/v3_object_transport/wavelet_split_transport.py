"""Typed split transport between proposal topology and leader content.

Proposal incidence and wavelet-leader regularity answer different questions:
the former supplies candidate structural paths, while the latter supplies
nonlocal content correspondence. Concatenating them into one descriptor
erases that distinction. This module instead treats each as an independent
off-diagonal generator and composes their analytical heat flows.

For normalized proposal connection ``P`` and spectrally normalized content
connection ``C``, the symmetric half-flow is the Strang product

    H_1/2 = exp(P/4) exp(C/2) exp(P/4).

Thus ``H_1/2 B H_1/2`` transports a base region kernel through paths which can
alternate content correspondence and proposal structure. Subtracting the
identity from either generator changes only a global scalar, removed by the
final kernel normalization, so it is omitted numerically. There is no merge
threshold, seed, class count, affinity bandwidth, or iterative stopping rule.
"""

from __future__ import annotations

import numpy as np

from experiments.v3_object_transport.participation_algebra import (
    normalized_linear_kernel,
)


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    """Symmetrize and normalize a positive-semidefinite transported kernel."""
    value = 0.5 * (
        np.asarray(kernel, dtype=np.float64)
        + np.asarray(kernel, dtype=np.float64).T)
    diagonal = np.maximum(np.diag(value), 0.0)
    denominator = np.sqrt(diagonal[:, None] * diagonal[None, :])
    return np.divide(
        value, denominator, out=np.zeros_like(value),
        where=denominator > 1e-30)


def content_connection(region_embedding: np.ndarray) -> np.ndarray:
    """Return the unit-spectral, off-diagonal signed content connection."""
    connection = normalized_linear_kernel(region_embedding).astype(np.float64)
    np.fill_diagonal(connection, 0.0)
    connection = 0.5 * (connection + connection.T)
    if not len(connection):
        return connection
    spectral_radius = float(np.max(np.abs(np.linalg.eigvalsh(connection))))
    if spectral_radius > 1e-30:
        connection /= spectral_radius
    return connection


def gated_content_connection(
    region_embedding: np.ndarray,
    context_kernel: np.ndarray,
) -> np.ndarray:
    """Conjoin content and boundary-role context by an exact Schur product."""
    content = normalized_linear_kernel(region_embedding).astype(np.float64)
    context = np.asarray(context_kernel, dtype=np.float64)
    if context.shape != content.shape:
        raise ValueError("context kernel must match the content regions")
    connection = content * (0.5 * (context + context.T))
    np.fill_diagonal(connection, 0.0)
    connection = 0.5 * (connection + connection.T)
    if not len(connection):
        return connection
    spectral_radius = float(np.max(np.abs(np.linalg.eigvalsh(connection))))
    if spectral_radius > 1e-30:
        connection /= spectral_radius
    return connection


def _symmetric_exponential(
    generator: np.ndarray,
    time: float,
) -> np.ndarray:
    value = 0.5 * (
        np.asarray(generator, dtype=np.float64)
        + np.asarray(generator, dtype=np.float64).T)
    eigenvalue, eigenvector = np.linalg.eigh(value)
    return (
        eigenvector * np.exp(time * eigenvalue)[None, :]
    ) @ eigenvector.T


def analytical_split_transport(
    proposal_connection: np.ndarray,
    region_embedding: np.ndarray,
    base_kernel: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compose proposal and content heat, then transport ``base_kernel``."""
    proposal = 0.5 * (
        np.asarray(proposal_connection, dtype=np.float64)
        + np.asarray(proposal_connection, dtype=np.float64).T)
    base = np.asarray(base_kernel, dtype=np.float64)
    if proposal.ndim != 2 or proposal.shape[0] != proposal.shape[1]:
        raise ValueError("proposal connection must be square")
    if base.shape != proposal.shape:
        raise ValueError("base kernel must match proposal connection")
    if np.asarray(region_embedding).shape[0] != proposal.shape[0]:
        raise ValueError("region embedding must match proposal connection")
    content = content_connection(region_embedding)
    return analytical_connection_split_transport(proposal, content, base)


def analytical_connection_split_transport(
    proposal_connection: np.ndarray,
    content: np.ndarray,
    base_kernel: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compose proposal heat with an already typed content connection."""
    proposal = 0.5 * (
        np.asarray(proposal_connection, dtype=np.float64)
        + np.asarray(proposal_connection, dtype=np.float64).T)
    base = np.asarray(base_kernel, dtype=np.float64)
    content = 0.5 * (
        np.asarray(content, dtype=np.float64)
        + np.asarray(content, dtype=np.float64).T)
    if proposal.ndim != 2 or proposal.shape[0] != proposal.shape[1]:
        raise ValueError("proposal connection must be square")
    if base.shape != proposal.shape or content.shape != proposal.shape:
        raise ValueError("content and base kernels must match proposal")
    proposal_quarter = _symmetric_exponential(proposal, 0.25)
    content_half = _symmetric_exponential(content, 0.5)
    half_flow = proposal_quarter @ content_half @ proposal_quarter
    return {
        "content_connection": content,
        "split_heat_kernel": normalize_kernel(half_flow @ half_flow),
        "transported_base_kernel": normalize_kernel(
            half_flow @ base @ half_flow),
    }
