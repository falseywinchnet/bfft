#!/usr/bin/env python3
"""Exponent ledger for candidate midpoint-Hessian recovery theorems.

The ledger separates three claims that are easy to conflate:

* the paper's importance-sampling cost;
* the normalized-correlation cost of applying Goldreich--Levin to a sparse
  empirical histogram; and
* the cost that would remain if a direct affine-coset transport theorem
  supplied target-width samples without endpoint importance weights.

Only the last line is a conjectural implication.  All displayed exponents
before it are elementary consequences of the paper's definitions.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


T0 = 0.23147
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "out" / "walsh_recovery_theorem_ledger.json"


def g(width: float, *, t0: float = T0) -> float:
    return 0.5 * math.log2(width / t0)


def iota(r: float, R: float) -> float:
    if not (0.0 < r <= R):
        raise ValueError("expected 0 < r <= R")
    return 0.5 * math.log2(R * R / (r * (2.0 * R - r)))


def kappa(r: float, *, t0: float = T0) -> float:
    gr = g(r, t0=t0)
    return 2.0 * r - min(gr, 2.0 * gr)


def attenuation_exponent(r: float, R: float) -> float:
    """a where the target coefficient w*mu is 2^(-a n), up to poly(n)."""
    return r + 0.5 * math.log2(R / r)


def sample_exponent(r: float, R: float) -> float:
    return iota(r, R) + 2.0 * r


def sparse_histogram_gl_exponent(
    r: float,
    R: float,
    domain_exponent: float,
) -> float:
    """Optimistic GL time after the sparse-histogram normalization loss.

    An M-sparse empirical histogram on a domain of size 2^(ell*n) has a
    normalized target correlation no larger than

        2^(-(a + max(ell-m, 0))*n),

    up to polynomial and occupancy factors.  A gamma^-2 heavy-correlation
    search therefore has the exponent returned here.  This is optimistic:
    a nonuniform source distribution can only worsen the L-infinity
    normalization needed by a bounded-function Goldreich--Levin theorem.
    """
    m = sample_exponent(r, R)
    a = attenuation_exponent(r, R)
    return 2.0 * (a + max(domain_exponent - m, 0.0))


@dataclass(frozen=True)
class Candidate:
    name: str
    r: float
    R: float
    chi: float
    domain_exponent: float
    sample_exponent: float
    attenuation_exponent: float
    population_energy_exponent: float
    gl_exponent_with_occupancy: float
    isolation_exponent: float
    overall_exponent: float
    conjectural: bool


def make_candidate(
    name: str,
    *,
    r: float,
    R: float,
    chi: float,
    direct_target_coset_transport: bool,
) -> Candidate:
    ell = 1.0 - chi
    m = 2.0 * r if direct_target_coset_transport else sample_exponent(r, R)
    a = r if direct_target_coset_transport else attenuation_exponent(r, R)
    gr = g(r)
    isolation = chi - 1.0 - min(gr, 2.0 * gr) + 2.0 * r
    gl_exp = sparse_histogram_gl_exponent(r, R, ell)
    if direct_target_coset_transport:
        # No sparse recovery is needed: run the paper's complete ell-bit WHT.
        overall = max(0.5, m, ell)
    else:
        overall = max(0.5, m, gl_exp)
    return Candidate(
        name=name,
        r=r,
        R=R,
        chi=chi,
        domain_exponent=ell,
        sample_exponent=m,
        attenuation_exponent=a,
        population_energy_exponent=kappa(r),
        gl_exponent_with_occupancy=gl_exp,
        isolation_exponent=isolation,
        overall_exponent=overall,
        conjectural=direct_target_coset_transport,
    )


def build_report(epsilon: float = 1e-3) -> dict[str, object]:
    paper = make_candidate(
        "paper_parameters_with_ordinary_GL",
        r=0.2222355,
        R=0.400613,
        chi=0.3961331,
        direct_target_coset_transport=False,
    )
    h0 = make_candidate(
        "h0_sparse_histogram_with_ordinary_GL",
        r=0.22407354148,
        R=T0 * (1.0 + 1e-6),
        chi=0.0,
        direct_target_coset_transport=False,
    )
    transported = make_candidate(
        "direct_random_half_coset_transport",
        r=T0 + epsilon,
        R=T0 + epsilon,
        chi=0.5 - epsilon,
        direct_target_coset_transport=True,
    )
    return {
        "experiment": "walsh_recovery_theorem_exponent_ledger",
        "t0": T0,
        "epsilon": epsilon,
        "candidates": [asdict(x) for x in (paper, h0, transported)],
        "interpretation": {
            "ordinary_gl": (
                "Fails to lower the exponent after normalized Fourier and "
                "histogram-occupancy factors are included."
            ),
            "transport_implication": (
                "Conditional on target-width sampling or an equivalent "
                "uniform Walsh-moment estimator in one random half coset, "
                "the existing full WHT has exponent 1/2+epsilon."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.epsilon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
