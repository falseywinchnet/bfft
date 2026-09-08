"""Canonical companion coordinate and exact two-current transport law.

The signed Eikonal distance ``phi`` is the normal Fermi coordinate of an
owner.  Its companion ``psi`` is arclength on the transverse zero-line,
carried unchanged along the normal geodesic flow.  Before the cut locus this
gives an orthogonal owner chart

    X* g = dphi^2 + h(phi, psi)^2 dpsi^2.

The inverse ray-tube width ``a = 1/h`` obeys the first-order conservation law

    div_g(a grad_g phi) = 0,       a|_{phi=0} = 1.

For a scalar potential F, its coordinate current is

    dF = j_phi dphi + j_psi dpsi.

Only one component and one transverse trace are free.  Given j_phi and
b(psi)=F(0,psi), exactness uniquely forces

    F(phi,psi) = b(psi) + integral_0^phi j_phi(s,psi) ds,
    j_psi(phi,psi) = b'(psi)
        + integral_0^phi partial_psi j_phi(s,psi) ds.

The discrete construction below is the commuting-difference form of that
identity.  It uses no Poisson solve or iterative projection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


Array = np.ndarray


def integrable_two_current(
    boundary_trace: Array,
    normal_increment: Array,
) -> tuple[Array, Array, Array]:
    """Integrate a normal current and derive its unique companion current.

    ``boundary_trace`` has shape ``(Q, ...)`` and stores F(0, psi).  The
    normal increments have shape ``(P-1, Q, ...)``.  The returned potential
    has shape ``(P, Q, ...)``.  The returned currents are the forward
    differences of that one potential, so their cell curl is algebraically
    zero.
    """

    boundary = np.asarray(boundary_trace)
    normal = np.asarray(normal_increment)
    if boundary.ndim < 1 or normal.ndim != boundary.ndim + 1:
        raise ValueError("normal current must add one leading phi axis")
    if normal.shape[1:] != boundary.shape:
        raise ValueError("normal-current rays must match the boundary trace")
    dtype = np.result_type(boundary.dtype, normal.dtype)
    potential = np.empty(
        (normal.shape[0] + 1,) + boundary.shape, dtype=dtype
    )
    potential[0] = boundary
    potential[1:] = boundary[None] + np.cumsum(normal, axis=0, dtype=dtype)
    admitted_normal = np.diff(potential, axis=0)
    forced_tangent = np.diff(potential, axis=1)
    return potential, admitted_normal, forced_tangent


def discrete_closure_defect(normal: Array, tangent: Array) -> Array:
    """Return delta_psi(normal)-delta_phi(tangent) on chart cells."""

    normal_current = np.asarray(normal)
    tangent_current = np.asarray(tangent)
    if normal_current.ndim < 2 or tangent_current.ndim != normal_current.ndim:
        raise ValueError("both current arrays need phi and psi axes")
    if (
        tangent_current.shape[0] != normal_current.shape[0] + 1
        or tangent_current.shape[1] + 1 != normal_current.shape[1]
        or tangent_current.shape[2:] != normal_current.shape[2:]
    ):
        raise ValueError("normal and tangent currents do not bound one grid")
    return np.diff(normal_current, axis=1) - np.diff(tangent_current, axis=0)


def symbolic_certificate() -> dict[str, object]:
    """Return exact SymPy identities for a curved Fermi chart and currents."""

    import sympy as sp

    phi, psi, radius = sp.symbols("phi psi R", positive=True)

    # Euclidean circle: the zero-line is r=R and psi=R*theta is its
    # arclength.  X(phi,psi) is the normal exponential map.
    angle = psi / radius
    x = (radius + phi) * sp.cos(angle)
    y = (radius + phi) * sp.sin(angle)
    d_phi = sp.Matrix((sp.diff(x, phi), sp.diff(y, phi)))
    d_psi = sp.Matrix((sp.diff(x, psi), sp.diff(y, psi)))
    metric = sp.simplify(sp.Matrix((
        (d_phi.dot(d_phi), d_phi.dot(d_psi)),
        (d_psi.dot(d_phi), d_psi.dot(d_psi)),
    )))
    h = sp.simplify(sp.sqrt(sp.factor(metric.det())))
    inverse_width = sp.simplify(1 / h)
    laplace_phi = sp.simplify(sp.diff(h, phi) / h)
    density_transport = sp.simplify(
        sp.diff(inverse_width, phi) + inverse_width * laplace_phi
    )

    # A nonseparable polynomial current makes the closure check substantive.
    q, s = sp.symbols("q s")
    boundary = 2 + 3 * q + 5 * q**2
    normal = 7 + 11 * s + 13 * q + 17 * s * q + 19 * s**2 * q
    potential = sp.expand(boundary + sp.integrate(normal, (s, 0, s)))
    tangent = sp.diff(potential, q)
    closure = sp.simplify(sp.diff(normal, q) - sp.diff(tangent, s))

    # Pullback by arbitrary one-dimensional monotone reparameterizations.
    u = s + s**3
    v = q + q**3
    source = 2 * u**2 + 3 * u * v + 5 * v**2 + 7 * u**2 * v
    transported_phi = sp.diff(source, s)
    transported_psi = sp.diff(source, q)
    pullback_closure = sp.simplify(
        sp.diff(transported_phi, q) - sp.diff(transported_psi, s)
    )

    return {
        "circle_map": {"x": str(x), "y": str(y)},
        "pullback_metric": [[str(sp.simplify(item)) for item in row]
                            for row in metric.tolist()],
        "ray_tube_width_h": str(h),
        "inverse_width_a": str(inverse_width),
        "eikonal_identity": str(sp.simplify(metric.inv()[0, 0])),
        "orthogonality_identity": str(sp.simplify(metric[0, 1])),
        "laplace_beltrami_phi": str(laplace_phi),
        "density_transport_residual": str(density_transport),
        "current_closure_residual": str(closure),
        "pullback_closure_residual": str(pullback_closure),
        "all_exact": bool(
            sp.simplify(metric.inv()[0, 0] - 1) == 0
            and sp.simplify(metric[0, 1]) == 0
            and density_transport == 0
            and closure == 0
            and pullback_closure == 0
        ),
    }


def run(output: Path) -> dict[str, object]:
    certificate = symbolic_certificate()

    # Integer arithmetic makes the discrete commuting-square audit exact,
    # without a floating-point tolerance.
    boundary = np.array((2, 5, 11, 17, 23), dtype=np.int64)
    normal = np.array((
        (1, 2, 3, 4, 5),
        (-2, -1, 0, 1, 2),
        (7, 5, 3, 1, -1),
        (0, 4, 0, -4, 0),
    ), dtype=np.int64)
    potential, recovered_normal, tangent = integrable_two_current(
        boundary, normal
    )
    defect = discrete_closure_defect(recovered_normal, tangent)
    if not np.array_equal(recovered_normal, normal):
        raise AssertionError("normal current was not reproduced exactly")
    if not np.array_equal(defect, np.zeros_like(defect)):
        raise AssertionError("derived companion current is not exactly closed")

    perturbed_tangent = tangent.copy()
    perturbed_tangent[2, 2] += 1
    independent_defect = discrete_closure_defect(
        recovered_normal, perturbed_tangent
    )
    result = {
        "symbolic": certificate,
        "discrete": {
            "arithmetic": "signed 64-bit integers",
            "potential": potential.tolist(),
            "normal_current_reproduced_exactly": True,
            "maximum_absolute_closure_defect": int(np.max(np.abs(defect))),
            "independent_tangent_edit_closure_defect": independent_defect.tolist(),
            "independent_tangent_edit_maximum_defect": int(np.max(
                np.abs(independent_defect)
            )),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "output/support_geometry/eikonal_two_current/certificate.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.out), indent=2))


if __name__ == "__main__":
    main()
