"""Observer-space extraction through transported analysis and adjoint synthesis.

Every terminal projective chart induces an explicit target-free affine
operator ``A_d`` on the observed raster.  Structure is defined relationally:
it is what remains coherent under transport to these observer charts.  The
analysis operators

    B_d = sqrt(pi_d) (I - A_d)

send transport-inconsistent content into observer residual space.  The unique
screened least-squares reconstruction

    argmin_x ||x-y||^2 + sum_d ||B_d x||^2

is obtained by the positive normal equation ``(I + B*B)x = y``.  Thus the
left-behind observer residual is recomposed by the exact adjoint, without a
blur radius, noise class, iteration time, or fitted regularization strength.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .causal_information_lineage_2d import (
    causal_information_phase_integrated_law_2d,
)
from .cross_chart_transport_closure_2d import _projective_chart_index
from .witnessed_characteristic_transport_2d import _validate


def observer_chart_operators_2d(
    law: dict[str, np.ndarray],
) -> tuple[tuple[sparse.csr_matrix, ...], np.ndarray, dict[str, float | int]]:
    """Materialize terminal target-free transport charts and ownership."""
    signal = np.asarray(law["signal"], dtype=np.float64)
    mass = np.asarray(law["hj_simplex_collision_mass"], dtype=np.float64)
    tangent = np.asarray(law["tangent"], dtype=np.float64)
    identity = np.asarray(law["source_identity"], dtype=np.int64)
    coefficient = np.asarray(law["source_coefficient"], dtype=np.float64)
    if signal.ndim != 3 or mass.shape != signal.shape:
        raise ValueError("terminal signal law must have shape HxWxK")
    if identity.shape != coefficient.shape or identity.shape[:-1] != signal.shape:
        raise ValueError("terminal source graph must have shape HxWxKxS")
    _representatives, chart_index = _projective_chart_index(tangent)
    chart_count = int(np.max(chart_index)) + 1
    height, width, branch_count = signal.shape
    pixels = height * width
    rows = np.repeat(np.arange(pixels, dtype=np.int64), identity.shape[-1])
    charts = []
    chart_mass = np.empty((pixels, chart_count), dtype=np.float64)
    flat_mass = mass.reshape(pixels, branch_count)
    flat_identity = identity.reshape(pixels, branch_count, -1)
    flat_coefficient = coefficient.reshape(pixels, branch_count, -1)
    tiny = np.finfo(float).tiny
    for chart in range(chart_count):
        members = np.flatnonzero(chart_index == chart)
        owned = np.sum(flat_mass[:, members], axis=1)
        chart_mass[:, chart] = owned
        conditional = np.divide(
            flat_mass[:, members],
            owned[:, None],
            out=np.zeros((pixels, members.size), dtype=np.float64),
            where=owned[:, None] > tiny,
        )
        chart_rows = np.tile(rows, members.size)
        chart_columns = flat_identity[:, members, :].transpose(
            1, 0, 2).reshape(-1)
        chart_values = (
            conditional[:, :, None] * flat_coefficient[:, members, :]
        ).transpose(1, 0, 2).reshape(-1)
        operator = sparse.coo_matrix(
            (chart_values, (chart_rows, chart_columns)),
            shape=(pixels, pixels),
        ).tocsr()
        operator.eliminate_zeros()
        charts.append(operator)
    ownership = chart_mass / np.sum(chart_mass, axis=1, keepdims=True)
    row_mass_error = max(float(np.max(np.abs(
        np.asarray(operator.sum(axis=1)).reshape(-1)[
            chart_mass[:, chart] > tiny] - 1.0)))
        for chart, operator in enumerate(charts))
    maximum_target_self = max(float(np.max(np.abs(operator.diagonal())))
                              for operator in charts)
    return tuple(charts), ownership.reshape(
        (height, width, chart_count)), {
        "chart_count": chart_count,
        "maximum_chart_row_mass_error": row_mass_error,
        "maximum_target_self_coefficient": maximum_target_self,
        "total_chart_nonzeros": int(sum(
            operator.nnz for operator in charts)),
    }


def observer_transport_extraction_readout_2d(
    observation: np.ndarray,
    law: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, float | int | str]]:
    """Extract coherent structure and adjoint-recompose it in scene space."""
    image = _validate(observation)
    charts, ownership, operator_diagnostic = observer_chart_operators_2d(law)
    if ownership.shape[:2] != image.shape:
        raise ValueError("transport chart ownership must align with observation")
    pixels = image.size
    identity = sparse.identity(pixels, dtype=np.float64, format="csr")
    analysis_blocks = []
    flat_ownership = ownership.reshape(pixels, ownership.shape[-1])
    for chart, operator in enumerate(charts):
        screen = sparse.diags(np.sqrt(flat_ownership[:, chart]), format="csr")
        analysis_blocks.append(screen @ (identity - operator))
    analysis = sparse.vstack(analysis_blocks, format="csr")
    augmented = sparse.vstack((identity, analysis), format="csr")
    flat_observation = image.reshape(-1)
    right_hand_side = np.concatenate((
        flat_observation,
        np.zeros(analysis.shape[0], dtype=np.float64),
    ))
    numerical = np.sqrt(np.finfo(float).eps)
    solved = sparse_linalg.lsmr(
        augmented,
        right_hand_side,
        atol=numerical,
        btol=numerical,
        maxiter=4 * pixels,
    )
    structure = solved[0]
    observer_residual = analysis @ structure
    scene_residual = flat_observation - structure
    adjoint_recomposition = analysis.T @ observer_residual
    normal_error = scene_residual - adjoint_recomposition
    initial_observer_residual = analysis @ flat_observation
    objective_before = float(np.dot(
        initial_observer_residual, initial_observer_residual))
    objective_after = float(
        np.dot(scene_residual, scene_residual)
        + np.dot(observer_residual, observer_residual))
    readouts = {
        "observer_transport_structure": structure.reshape(image.shape),
        "observer_scene_residual": scene_residual.reshape(image.shape),
        "observer_analysis_residual": observer_residual.reshape(
            (len(charts),) + image.shape),
        "observer_adjoint_recomposition": adjoint_recomposition.reshape(
            image.shape),
        "observer_chart_ownership": ownership,
    }
    diagnostic: dict[str, float | int | str] = {
        "status": (
            "screened positive observer transport analysis with exact adjoint "
            "scene recomposition"
        ),
        **operator_diagnostic,
        "analysis_nonzeros": int(analysis.nnz),
        "least_squares_stop_code": int(solved[1]),
        "least_squares_iterations": int(solved[2]),
        "least_squares_residual_norm": float(solved[3]),
        "normal_equation_maximum_error": float(np.max(np.abs(normal_error))),
        "objective_before": objective_before,
        "objective_after": objective_after,
        "observer_residual_contraction": float(
            np.linalg.norm(observer_residual)
            / max(np.linalg.norm(initial_observer_residual),
                  np.finfo(float).tiny)),
        "scene_residual_rms": float(np.sqrt(np.mean(scene_residual ** 2))),
        "structure_minimum": float(np.min(structure)),
        "structure_maximum": float(np.max(structure)),
    }
    return readouts, diagnostic


def denoise_observer_transport_extraction_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 4,
    quantile_count: int = 16,
    phase_count: int = 1,
    memory_ceiling_bytes: int | None = None,
    complete_residual_moment: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Infer transport charts, extract their common observer-space structure."""
    image = _validate(observation)
    law, transport = causal_information_phase_integrated_law_2d(
        image,
        angular_count=angular_count,
        quantile_count=quantile_count,
        phase_count=phase_count,
        memory_ceiling_bytes=memory_ceiling_bytes,
        complete_residual_moment=complete_residual_moment,
    )
    readouts, extraction = observer_transport_extraction_readout_2d(image, law)
    return readouts["observer_transport_structure"], {
        "status": "inferred observer transport followed by adjoint extraction",
        "transport": transport,
        "extraction": extraction,
        "readouts": readouts,
    }
