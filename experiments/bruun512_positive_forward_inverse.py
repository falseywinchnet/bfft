"""
bruun512_positive_fft_resonator.py

Standalone 512-point real FFT and inverse real FFT implemented as positive
phase-sheet transport over a Bruun residue ordering.

The internal march uses nonnegative real sheet values. Negative signs are
represented as routing to the opposite sheet. Complex/quadrature structure
appears only in residue packing and unpacking.

Dependency: numpy

Primary functions:
    positive_cone_rfft(x, ...)
    positive_cone_irfft(X, ...)
    run_self_tests()
"""

from __future__ import annotations

import numpy as np

N = 512
L = 9


def graydecode_int(g: int) -> int:
    x = int(g)
    while g:
        g >>= 1
        x ^= g
    return x


def bitrev_int(x: int, bits: int) -> int:
    y = 0
    for _ in range(bits):
        y = (y << 1) | (x & 1)
        x >>= 1
    return y


def bruun_idx_int(m: int, l: int = L) -> int:
    if m <= 0:
        return 0
    t = int(m).bit_length() - 1
    r = int(m) ^ (1 << t)
    return (2 * graydecode_int(bitrev_int(r, t)) + 1) << ((l - 2) - t)


def make_index_table() -> np.ndarray:
    idx = np.zeros(N // 2, dtype=np.int64)
    for m in range(1, N // 2):
        idx[m] = bruun_idx_int(m, L)
    return idx


def make_twiddle_table() -> tuple[np.ndarray, list[float]]:
    c = np.zeros(256, dtype=np.float64)
    s_special = [0.0, np.sqrt(0.5)]
    c[1] = np.sqrt(0.5)
    for m in range(1, 128):
        cm = c[m]
        sm = s_special[1] if m == 1 else c[m ^ 1]
        ce = np.sqrt(0.5 * (1.0 + cm))
        se = sm / (2.0 * ce)
        c[2 * m] = ce
        c[2 * m + 1] = se
    return c, s_special


IDX = make_index_table()
C, S_SPECIAL = make_twiddle_table()


def split_pm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0.0, x, 0.0), np.where(x < 0.0, -x, 0.0)


def project_pm(xp: np.ndarray, xn: np.ndarray) -> np.ndarray:
    return xp - xn


def add_routed(dstp: np.ndarray, dstn: np.ndarray, srcp: np.ndarray, srcn: np.ndarray, weight: float, flip: bool = False) -> None:
    if flip:
        dstp += weight * srcn
        dstn += weight * srcp
    else:
        dstp += weight * srcp
        dstn += weight * srcn


def annihilate_common_mode(xp: np.ndarray, xn: np.ndarray, where=slice(None)) -> None:
    common = np.minimum(xp[where], xn[where])
    xp[where] -= common
    xn[where] -= common


def sheet_stats(xp: np.ndarray, xn: np.ndarray) -> dict[str, float]:
    projected = xp - xn
    sheet_mass = float(np.sum(xp) + np.sum(xn))
    projected_l1 = float(np.sum(np.abs(projected)))
    if projected_l1 == 0.0:
        ratio = 0.0 if sheet_mass == 0.0 else float("inf")
    else:
        ratio = sheet_mass / projected_l1
    return {
        "sheet_mass": sheet_mass,
        "projected_l1": projected_l1,
        "cancellation_ratio": ratio,
    }


def pack_residue_to_rfft(work: np.ndarray) -> np.ndarray:
    work = np.asarray(work, dtype=np.float64)
    if work.shape != (N,):
        raise ValueError("expected residue vector of length 512")
    x = np.empty(N // 2 + 1, dtype=np.complex128)
    x[0] = work[0] + work[1]
    x[N // 2] = work[0] - work[1]
    for m in range(1, N // 2):
        k = IDX[m]
        x[k] = work[2 * m] - 1j * work[2 * m + 1]
    return x


def unpack_rfft_to_residue(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    if x.shape != (N // 2 + 1,):
        raise ValueError("expected rfft vector of length 257")
    work = np.empty(N, dtype=np.float64)
    work[0] = 0.5 * (x[0].real + x[N // 2].real)
    work[1] = 0.5 * (x[0].real - x[N // 2].real)
    for m in range(1, N // 2):
        k = IDX[m]
        work[2 * m] = x[k].real
        work[2 * m + 1] = -x[k].imag
    return work


def positive_binomial_transport_inplace(wp: np.ndarray, wn: np.ndarray, left: slice, right: slice, out_sum: slice, out_diff: slice) -> None:
    ap = wp[left].copy()
    an = wn[left].copy()
    bp = wp[right].copy()
    bn = wn[right].copy()
    up = np.zeros_like(ap)
    un = np.zeros_like(an)
    vp = np.zeros_like(ap)
    vn = np.zeros_like(an)
    add_routed(up, un, ap, an, 1.0, False)
    add_routed(up, un, bp, bn, 1.0, False)
    add_routed(vp, vn, ap, an, 1.0, False)
    add_routed(vp, vn, bp, bn, 1.0, True)
    wp[out_sum] = up
    wn[out_sum] = un
    wp[out_diff] = vp
    wn[out_diff] = vn


def positive_bruun_node_inplace(wp: np.ndarray, wn: np.ndarray, p: int, q: int, c: float, s: float) -> None:
    A0p = wp[p:p + q].copy()
    A0n = wn[p:p + q].copy()
    B0p = wp[p + q:p + 2 * q].copy()
    B0n = wn[p + q:p + 2 * q].copy()
    A1p = wp[p + 2 * q:p + 3 * q].copy()
    A1n = wn[p + 2 * q:p + 3 * q].copy()
    B1p = wp[p + 3 * q:p + 4 * q].copy()
    B1n = wn[p + 3 * q:p + 4 * q].copy()

    Rp = np.zeros(q, dtype=np.float64)
    Rn = np.zeros(q, dtype=np.float64)
    Ip = np.zeros(q, dtype=np.float64)
    In = np.zeros(q, dtype=np.float64)

    add_routed(Rp, Rn, B0p, B0n, c, False)
    add_routed(Rp, Rn, B1p, B1n, s, True)
    add_routed(Ip, In, B0p, B0n, s, False)
    add_routed(Ip, In, B1p, B1n, c, False)

    Y0p = np.zeros(q, dtype=np.float64)
    Y0n = np.zeros(q, dtype=np.float64)
    Y1p = np.zeros(q, dtype=np.float64)
    Y1n = np.zeros(q, dtype=np.float64)
    Y2p = np.zeros(q, dtype=np.float64)
    Y2n = np.zeros(q, dtype=np.float64)
    Y3p = np.zeros(q, dtype=np.float64)
    Y3n = np.zeros(q, dtype=np.float64)

    add_routed(Y0p, Y0n, A0p, A0n, 1.0, False)
    add_routed(Y0p, Y0n, Rp, Rn, 1.0, False)
    add_routed(Y1p, Y1n, A1p, A1n, 1.0, False)
    add_routed(Y1p, Y1n, Ip, In, 1.0, False)
    add_routed(Y2p, Y2n, A0p, A0n, 1.0, False)
    add_routed(Y2p, Y2n, Rp, Rn, 1.0, True)
    add_routed(Y3p, Y3n, Ip, In, 1.0, False)
    add_routed(Y3p, Y3n, A1p, A1n, 1.0, True)

    wp[p:p + q] = Y0p
    wn[p:p + q] = Y0n
    wp[p + q:p + 2 * q] = Y1p
    wn[p + q:p + 2 * q] = Y1n
    wp[p + 2 * q:p + 3 * q] = Y2p
    wn[p + 2 * q:p + 3 * q] = Y2n
    wp[p + 3 * q:p + 4 * q] = Y3p
    wn[p + 3 * q:p + 4 * q] = Y3n


def positive_cone_rfft(input_data: np.ndarray, stage_scale: float = 0.5, annihilate_stage: bool = True, return_trace: bool = False, return_sheets: bool = False):
    x = np.asarray(input_data, dtype=np.float64)
    if x.shape != (N,):
        raise ValueError("expected input vector of length 512")

    xp, xn = split_pm(x)
    wp = np.zeros(N, dtype=np.float64)
    wn = np.zeros(N, dtype=np.float64)
    scale_exp = 0
    trace = []

    def close_stage(tag: str) -> None:
        nonlocal scale_exp
        if annihilate_stage:
            annihilate_common_mode(wp, wn)
        wp[:] *= stage_scale
        wn[:] *= stage_scale
        scale_exp += 1
        if return_trace:
            row = {"tag": tag, "scale_exp": float(scale_exp)}
            row.update(sheet_stats(wp, wn))
            trace.append(row)

    wp[:N // 2] = xp[:N // 2] + xp[N // 2:]
    wn[:N // 2] = xn[:N // 2] + xn[N // 2:]
    wp[N // 2:] = xp[:N // 2] + xn[N // 2:]
    wn[N // 2:] = xn[:N // 2] + xp[N // 2:]
    close_stage("S0")

    for stage in range(1, 8):
        prefix = N >> stage
        half = prefix >> 1
        positive_binomial_transport_inplace(wp, wn, slice(0, half), slice(half, prefix), slice(0, half), slice(half, prefix))
        q = (N // 4) >> stage
        for m in range(1, 2 ** stage):
            p = 4 * q * m
            c = C[m]
            s = S_SPECIAL[1] if m == 1 else C[m ^ 1]
            positive_bruun_node_inplace(wp, wn, p, q, c, s)
        close_stage(f"S{stage}")

    compensation = stage_scale ** (-scale_exp)
    work = (wp - wn) * compensation
    X = pack_residue_to_rfft(work)

    if return_trace and return_sheets:
        return X, work, wp, wn, compensation, trace
    if return_sheets:
        return X, work, wp, wn, compensation
    if return_trace:
        return X, trace
    return X


def positive_inverse_binomial_inplace(wp: np.ndarray, wn: np.ndarray, left: slice, right: slice) -> None:
    up = wp[left].copy()
    un = wn[left].copy()
    vp = wp[right].copy()
    vn = wn[right].copy()
    ap = np.zeros_like(up)
    an = np.zeros_like(un)
    bp = np.zeros_like(up)
    bn = np.zeros_like(un)
    add_routed(ap, an, up, un, 0.5, False)
    add_routed(ap, an, vp, vn, 0.5, False)
    add_routed(bp, bn, up, un, 0.5, False)
    add_routed(bp, bn, vp, vn, 0.5, True)
    wp[left] = ap
    wn[left] = an
    wp[right] = bp
    wn[right] = bn


def positive_inverse_bruun_node_inplace(wp: np.ndarray, wn: np.ndarray, p: int, q: int, c: float, s: float) -> None:
    Y0p = wp[p:p + q].copy()
    Y0n = wn[p:p + q].copy()
    Y1p = wp[p + q:p + 2 * q].copy()
    Y1n = wn[p + q:p + 2 * q].copy()
    Y2p = wp[p + 2 * q:p + 3 * q].copy()
    Y2n = wn[p + 2 * q:p + 3 * q].copy()
    Y3p = wp[p + 3 * q:p + 4 * q].copy()
    Y3n = wn[p + 3 * q:p + 4 * q].copy()

    A0p = np.zeros(q, dtype=np.float64)
    A0n = np.zeros(q, dtype=np.float64)
    Rp = np.zeros(q, dtype=np.float64)
    Rn = np.zeros(q, dtype=np.float64)
    Ip = np.zeros(q, dtype=np.float64)
    In = np.zeros(q, dtype=np.float64)
    A1p = np.zeros(q, dtype=np.float64)
    A1n = np.zeros(q, dtype=np.float64)

    add_routed(A0p, A0n, Y0p, Y0n, 0.5, False)
    add_routed(A0p, A0n, Y2p, Y2n, 0.5, False)
    add_routed(Rp, Rn, Y0p, Y0n, 0.5, False)
    add_routed(Rp, Rn, Y2p, Y2n, 0.5, True)
    add_routed(Ip, In, Y1p, Y1n, 0.5, False)
    add_routed(Ip, In, Y3p, Y3n, 0.5, False)
    add_routed(A1p, A1n, Y1p, Y1n, 0.5, False)
    add_routed(A1p, A1n, Y3p, Y3n, 0.5, True)

    B0p = np.zeros(q, dtype=np.float64)
    B0n = np.zeros(q, dtype=np.float64)
    B1p = np.zeros(q, dtype=np.float64)
    B1n = np.zeros(q, dtype=np.float64)
    add_routed(B0p, B0n, Rp, Rn, c, False)
    add_routed(B0p, B0n, Ip, In, s, False)
    add_routed(B1p, B1n, Rp, Rn, s, True)
    add_routed(B1p, B1n, Ip, In, c, False)

    wp[p:p + q] = A0p
    wn[p:p + q] = A0n
    wp[p + q:p + 2 * q] = B0p
    wn[p + q:p + 2 * q] = B0n
    wp[p + 2 * q:p + 3 * q] = A1p
    wn[p + 2 * q:p + 3 * q] = A1n
    wp[p + 3 * q:p + 4 * q] = B1p
    wn[p + 3 * q:p + 4 * q] = B1n


def positive_inverse_residue_to_real(work: np.ndarray, annihilate_stage: bool = True, return_trace: bool = False, return_sheets: bool = False):
    work = np.asarray(work, dtype=np.float64)
    if work.shape != (N,):
        raise ValueError("expected residue vector of length 512")

    wp, wn = split_pm(work)
    trace = []

    def close_reverse_stage(tag: str) -> None:
        if annihilate_stage:
            annihilate_common_mode(wp, wn)
        if return_trace:
            row = {"tag": tag}
            row.update(sheet_stats(wp, wn))
            trace.append(row)

    close_reverse_stage("residue")

    for stage in range(7, 0, -1):
        q = (N // 4) >> stage
        for m in range(1, 2 ** stage):
            p = 4 * q * m
            c = C[m]
            s = S_SPECIAL[1] if m == 1 else C[m ^ 1]
            positive_inverse_bruun_node_inplace(wp, wn, p, q, c, s)
        prefix = N >> stage
        half = prefix >> 1
        positive_inverse_binomial_inplace(wp, wn, slice(0, half), slice(half, prefix))
        close_reverse_stage(f"undo S{stage}")

    positive_inverse_binomial_inplace(wp, wn, slice(0, N // 2), slice(N // 2, N))
    close_reverse_stage("undo S0")

    x = wp - wn
    if return_trace and return_sheets:
        return x, wp, wn, trace
    if return_sheets:
        return x, wp, wn
    if return_trace:
        return x, trace
    return x


def positive_cone_irfft(X: np.ndarray, annihilate_stage: bool = True, return_trace: bool = False, return_sheets: bool = False):
    work = unpack_rfft_to_residue(X)
    return positive_inverse_residue_to_real(work, annihilate_stage=annihilate_stage, return_trace=return_trace, return_sheets=return_sheets)


def signed_bruun_rfft(input_data: np.ndarray) -> np.ndarray:
    x = np.asarray(input_data, dtype=np.float64)
    if x.shape != (N,):
        raise ValueError("expected input vector of length 512")
    work = np.empty(N, dtype=np.float64)
    work[:N // 2] = x[:N // 2] + x[N // 2:]
    work[N // 2:] = x[:N // 2] - x[N // 2:]
    for stage in range(1, 8):
        prefix = N >> stage
        half = prefix >> 1
        a = work[:half].copy()
        b = work[half:prefix].copy()
        work[:half] = a + b
        work[half:prefix] = a - b
        q = (N // 4) >> stage
        for m in range(1, 2 ** stage):
            p = 4 * q * m
            c = C[m]
            s = S_SPECIAL[1] if m == 1 else C[m ^ 1]
            A0 = work[p:p + q].copy()
            B0 = work[p + q:p + 2 * q].copy()
            A1 = work[p + 2 * q:p + 3 * q].copy()
            B1 = work[p + 3 * q:p + 4 * q].copy()
            R = c * B0 - s * B1
            I = s * B0 + c * B1
            work[p:p + q] = A0 + R
            work[p + q:p + 2 * q] = A1 + I
            work[p + 2 * q:p + 3 * q] = A0 - R
            work[p + 3 * q:p + 4 * q] = I - A1
    return pack_residue_to_rfft(work)


def signed_inverse_residue_to_real(work: np.ndarray) -> np.ndarray:
    work = np.asarray(work, dtype=np.float64).copy()
    if work.shape != (N,):
        raise ValueError("expected residue vector of length 512")
    for stage in range(7, 0, -1):
        q = (N // 4) >> stage
        for m in range(1, 2 ** stage):
            p = 4 * q * m
            c = C[m]
            s = S_SPECIAL[1] if m == 1 else C[m ^ 1]
            Y0 = work[p:p + q].copy()
            Y1 = work[p + q:p + 2 * q].copy()
            Y2 = work[p + 2 * q:p + 3 * q].copy()
            Y3 = work[p + 3 * q:p + 4 * q].copy()
            A0 = 0.5 * (Y0 + Y2)
            R = 0.5 * (Y0 - Y2)
            I = 0.5 * (Y1 + Y3)
            A1 = 0.5 * (Y1 - Y3)
            B0 = c * R + s * I
            B1 = -s * R + c * I
            work[p:p + q] = A0
            work[p + q:p + 2 * q] = B0
            work[p + 2 * q:p + 3 * q] = A1
            work[p + 3 * q:p + 4 * q] = B1
        prefix = N >> stage
        half = prefix >> 1
        u = work[:half].copy()
        v = work[half:prefix].copy()
        work[:half] = 0.5 * (u + v)
        work[half:prefix] = 0.5 * (u - v)
    u = work[:N // 2].copy()
    v = work[N // 2:].copy()
    work[:N // 2] = 0.5 * (u + v)
    work[N // 2:] = 0.5 * (u - v)
    return work


def signed_bruun_irfft(X: np.ndarray) -> np.ndarray:
    return signed_inverse_residue_to_real(unpack_rfft_to_residue(X))


def default_test_cases() -> dict[str, np.ndarray]:
    return {
        "random_positive_42": np.random.default_rng(42).random(N),
        "normal_43": np.random.default_rng(43).normal(size=N),
        "normal_44": np.random.default_rng(44).normal(size=N),
        "ramp": np.arange(N, dtype=np.float64),
        "alternating": np.where(np.arange(N) % 2 == 0, 1.0, -1.0),
        "impulse_0": np.eye(N)[0],
        "ones": np.ones(N),
        "zeros": np.zeros(N),
    }


def run_self_tests() -> list[dict[str, float | str]]:
    rows = []
    for name, x in default_test_cases().items():
        X = positive_cone_rfft(x)
        xr = positive_cone_irfft(X)
        X_np = np.fft.rfft(x)
        xr_np = np.fft.irfft(X, n=N)
        X_signed = signed_bruun_rfft(x)
        xr_signed = signed_bruun_irfft(X)
        _, work, wp, wn, compensation = positive_cone_rfft(x, return_sheets=True)
        _, iwp, iwn = positive_cone_irfft(X, return_sheets=True)
        rows.append({
            "case": name,
            "forward_vs_numpy_max_abs": float(np.max(np.abs(X - X_np))),
            "forward_vs_signed_max_abs": float(np.max(np.abs(X - X_signed))),
            "roundtrip_x_to_X_to_x_max_abs": float(np.max(np.abs(xr - x))),
            "roundtrip_x_to_X_to_x_l2": float(np.linalg.norm(xr - x)),
            "irfft_vs_numpy_irfft_max_abs": float(np.max(np.abs(xr - xr_np))),
            "irfft_vs_signed_irfft_max_abs": float(np.max(np.abs(xr - xr_signed))),
            "forward_final_sheet_mass_internal": float(np.sum(wp) + np.sum(wn)),
            "forward_compensated_sheet_mass": float((np.sum(wp) + np.sum(wn)) * compensation),
            "forward_final_cancellation_ratio": sheet_stats(wp, wn)["cancellation_ratio"],
            "inverse_final_sheet_mass": float(np.sum(iwp) + np.sum(iwn)),
            "inverse_final_cancellation_ratio": sheet_stats(iwp, iwn)["cancellation_ratio"],
        })
    return rows


def print_self_tests() -> None:
    rows = run_self_tests()
    headers = [
        "case",
        "forward_vs_numpy_max_abs",
        "forward_vs_signed_max_abs",
        "roundtrip_x_to_X_to_x_max_abs",
        "irfft_vs_numpy_irfft_max_abs",
        "forward_final_cancellation_ratio",
        "inverse_final_cancellation_ratio",
    ]
    widths = {}
    for h in headers:
        widths[h] = max(len(h), max(len(f"{row[h]:.6e}") if isinstance(row[h], float) else len(str(row[h])) for row in rows))
    print(" ".join(h.ljust(widths[h]) for h in headers))
    for row in rows:
        parts = []
        for h in headers:
            value = row[h]
            if isinstance(value, float):
                parts.append(f"{value:.6e}".ljust(widths[h]))
            else:
                parts.append(str(value).ljust(widths[h]))
        print(" ".join(parts))


if __name__ == "__main__":
    print_self_tests()
