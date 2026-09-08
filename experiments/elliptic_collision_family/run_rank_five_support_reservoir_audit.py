"""Exact Jacobian audit of every low-discriminant rank-five support wall.

The finite integral three-split classification supplies every wall in the
current conductor range.  For each exact rank-five wall

    E: y^2 = x^3 - m*x + a6,

this module attaches the horizontal-support reservoir

    delta^2 = 4*m^3 - 27*(h^2-a6)^2.

Its binary-quartic Jacobian measures the available directions in which a
new rational-height support current can appear.  PARI proves every rank
exactly.  This is a finite classification of the already-proved wall list,
not a coefficient or height search.
"""

from __future__ import annotations

from dataclasses import dataclass
from shutil import which
import subprocess

try:
    from .run_three_current_height_conic import binary_quartic_invariants
    from .run_three_split_exact_audit import pari_audit, classify_three_split_curves
except ImportError:  # Direct execution from this directory.
    from run_three_current_height_conic import binary_quartic_invariants
    from run_three_split_exact_audit import pari_audit, classify_three_split_curves


@dataclass(frozen=True)
class ReservoirAudit:
    m: int
    a6: int
    base_conductor: int
    support_count: int
    reservoir_model: tuple[int, int, int, int, int]
    reservoir_conductor: int
    root_number: int
    rank_lower: int
    rank_upper: int


def reservoir_raw_model(m: int, a6: int) -> tuple[int, int, int, int, int]:
    coefficients = (-27, 0, 54 * a6, 0, 4 * m**3 - 27 * a6**2)
    invariant_i, invariant_j = binary_quartic_invariants(coefficients)
    return 0, 0, 0, -27 * invariant_i, -27 * invariant_j


def exact_rank_five_walls():
    """Return the complete exact rank-five subset of the classified layer."""

    return tuple(
        audit
        for audit in pari_audit(classify_three_split_curves())
        if audit.rank_lower == audit.rank_upper == 5
    )


def audit_reservoirs() -> tuple[ReservoirAudit, ...]:
    gp = which("gp")
    if gp is None:
        raise RuntimeError("PARI/GP is required for the reservoir audit")
    walls = exact_rank_five_walls()
    models = [reservoir_raw_model(audit.curve.m, audit.curve.a6) for audit in walls]
    program = (
        "default(parisizemax,4000000000);\n"
        + "models=["
        + ",".join("[" + ",".join(map(str, model)) + "]" for model in models)
        + "];"
        + "for(i=1,#models,E=ellinit(models[i]);M=ellminimalmodel(E,&v);"
        + "r=ellrank(M);print(i,\",\",M.a1,\",\",M.a2,\",\",M.a3,\",\","
        + "M.a4,\",\",M.a6,\",\",ellglobalred(M)[1],\",\",ellrootno(M),"
        + "\",\",r[1],\",\",r[2]));quit;\n"
    )
    completed = subprocess.run(
        [gp, "-q", "-f"],
        input=program,
        text=True,
        capture_output=True,
        check=True,
    )
    rows = {}
    for line in completed.stdout.splitlines():
        values = tuple(map(int, line.split(",")))
        rows[values[0] - 1] = values[1:]
    if len(rows) != len(walls):
        raise ArithmeticError(completed.stderr or "incomplete reservoir PARI audit")

    result = []
    for index, wall in enumerate(walls):
        a1, a2, a3, a4, a6, conductor, root, lower, upper = rows[index]
        result.append(
            ReservoirAudit(
                wall.curve.m,
                wall.curve.a6,
                wall.conductor,
                len(wall.curve.currents),
                (a1, a2, a3, a4, a6),
                conductor,
                root,
                lower,
                upper,
            )
        )
    return tuple(sorted(result, key=lambda audit: audit.base_conductor))


def main() -> None:
    for audit in audit_reservoirs():
        print(
            f"m={audit.m} a6={audit.a6} base_N={audit.base_conductor} "
            f"supports={audit.support_count} reservoir={audit.reservoir_model} "
            f"reservoir_N={audit.reservoir_conductor} root={audit.root_number} "
            f"rank=[{audit.rank_lower},{audit.rank_upper}]"
        )


if __name__ == "__main__":
    main()
