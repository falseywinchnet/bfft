"""
Hourglass-FFT middle-conversion sparsity probe.

For N = A*B, the Chebyshev identity T_N = T_A(T_B(x)) = T_B(T_A(x)) gives two
"half-done" residue representations of a degree-<N polynomial f:

  Rep1 (outer T_A): { f mod (T_B(x) - beta_i) }_{i<A}   -> A residues, size B
  Rep2 (outer T_B): { f mod (T_A(x) - alpha_j) }_{j<B}  -> B residues, size A

The hourglass wants to run the top half of the tree in one composition order,
switch to the other with a CHEAP O(N) local map, and finish in bin order. The
switch is the linear map M with Rep2 = M @ Rep1.

  Win : M is permutation / sign / scale / bounded-block (max nnz/row bounded).
  Fail: max nnz/row grows with N (the switch hides a second transform).

NOTE: with A == B the two compositions are IDENTICAL (T_A = T_B), so the swap is
trivially the identity permutation -- a degenerate test. The meaningful case is
A != B. This probe sweeps genuinely asymmetric splits and checks whether M is a
true permutation (one nonzero per row AND column), reporting the distinct
nonzero magnitudes (sign/scale structure).
"""

import numpy as np


def cheb_T(n, x):
    return np.cos(n * np.arccos(np.clip(x, -1.0, 1.0)))


def build_reduce_map(N, num_res, mod_deg):
    """c (N Chebyshev coeffs of f) -> stacked residues mod (T_mod_deg(x)-gamma_i),
    gamma_i = roots of T_num_res. Each residue: mod_deg Chebyshev coeffs."""
    U = np.zeros((N, N))
    row = 0
    for i in range(num_res):
        phi = (2 * i + 1) * np.pi / (2 * num_res)
        xs = np.cos((phi + 2 * np.pi * np.arange(mod_deg)) / mod_deg)
        Eval = np.array([[cheb_T(j, xm) for j in range(N)] for xm in xs])
        Vmat = np.array([[cheb_T(b, xm) for b in range(mod_deg)] for xm in xs])
        U[row:row + mod_deg, :] = np.linalg.solve(Vmat, Eval)
        row += mod_deg
    return U


def verify_reduction(N, num_res, mod_deg, U):
    """Reduce a random f, then check residue i agrees with f at a root of
    T_mod_deg(x) = gamma_i (the defining property of the residue)."""
    rng = np.random.default_rng(0)
    c = rng.standard_normal(N)
    coeffs = U @ c
    worst = 0.0
    for i in range(num_res):
        phi = (2 * i + 1) * np.pi / (2 * num_res)
        x = np.cos((phi + 2 * np.pi * 1) / mod_deg)      # one root of T_mod_deg = gamma_i
        f_x = sum(c[j] * cheb_T(j, x) for j in range(N))
        r = coeffs[i * mod_deg:(i + 1) * mod_deg]
        r_x = sum(r[b] * cheb_T(b, x) for b in range(mod_deg))
        worst = max(worst, abs(f_x - r_x))
    return worst


def classify(M, rel_tol=1e-9):
    scale = np.max(np.abs(M))
    mask = np.abs(M) > rel_tol * scale
    nnz_row = mask.sum(axis=1)
    nnz_col = mask.sum(axis=0)
    is_perm = nnz_row.max() == 1 and nnz_col.max() == 1
    mags = np.sort(np.unique(np.round(np.abs(M[mask]) / scale, 6)))[::-1]
    return int(nnz_row.max()), float(nnz_row.mean()), is_perm, mags[:6]


def analyze(N, A, B):
    U1 = build_reduce_map(N, A, B)   # Rep1: A residues, size B
    U2 = build_reduce_map(N, B, A)   # Rep2: B residues, size A
    e1 = verify_reduction(N, A, B, U1)
    e2 = verify_reduction(N, B, A, U2)
    M = U2 @ np.linalg.inv(U1)
    mx, mean, is_perm, mags = classify(M)
    tag = "PERMUTATION" if is_perm else ("bounded" if mx <= 4 else "DENSE/growing")
    print(f"  N={N:4d} A={A:3d} B={B:3d}: max nnz/row={mx:4d} mean={mean:6.1f}  "
          f"{tag:13s}  reduce_err={max(e1, e2):.0e}  |vals|/max={list(mags)}")
    return mx


def main():
    print("Composition swap  Rep1{T_A(T_B)} -> Rep2{T_B(T_A)},  M = U2 @ inv(U1)\n")

    print("Degenerate balanced split A=B (two compositions coincide):")
    for N, A, B in [(16, 4, 4), (64, 8, 8), (256, 16, 16)]:
        analyze(N, A, B)

    print("\nGenuinely asymmetric A != B, fixed A=4, growing B=N/4:")
    res = []
    for N, A, B in [(64, 4, 16), (256, 4, 64), (1024, 4, 256)]:
        res.append((N, B, analyze(N, A, B)))

    print("\nAsymmetric A != B, fixed B=4, growing A=N/4:")
    for N, A, B in [(64, 16, 4), (256, 64, 4), (1024, 256, 4)]:
        analyze(N, A, B)

    print("\nScaling (fixed A=4): does max nnz/row track the growing factor B?")
    for N, B, mx in res:
        print(f"  N={N:5d}: max nnz/row={mx:4d}  B={B:4d}  ratio={mx / B:.3f}")


if __name__ == "__main__":
    main()
