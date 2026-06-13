
import numpy as np
from bruun512_phase_cone_transport import (
    N, C, S_SPECIAL, IDX,
    annihilate_common_mode, binomial_transport_inplace,
    bruun_transport_node_inplace, pack_residue_to_rfft
)


def split_pm(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0.0, x, 0.0), np.where(x < 0.0, np.abs(x), 0.0)


def add_routed(dstp, dstn, srcp, srcn, weight, flip=False):
    if flip:
        dstp += weight * srcn
        dstn += weight * srcp
    else:
        dstp += weight * srcp
        dstn += weight * srcn


def unpack_rfft_to_residue(X):
    X = np.asarray(X, dtype=np.complex128)
    if X.shape[0] != 257:
        raise ValueError("expected rfft length 257")

    work = np.empty(512, dtype=np.float64)
    work[0] = 0.5 * (X[0].real + X[256].real)
    work[1] = 0.5 * (X[0].real - X[256].real)

    for m in range(1, 256):
        k = IDX[m]
        work[2*m] = X[k].real
        work[2*m + 1] = -X[k].imag

    return work


def positive_inverse_binomial_inplace(wp, wn, left, right):
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


def positive_inverse_bruun_node_inplace(wp, wn, p, q, c, s):
    Y0p = wp[p:p + q].copy()
    Y0n = wn[p:p + q].copy()
    Y1p = wp[p + q:p + 2*q].copy()
    Y1n = wn[p + q:p + 2*q].copy()
    Y2p = wp[p + 2*q:p + 3*q].copy()
    Y2n = wn[p + 2*q:p + 3*q].copy()
    Y3p = wp[p + 3*q:p + 4*q].copy()
    Y3n = wn[p + 3*q:p + 4*q].copy()

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
    wp[p + q:p + 2*q] = B0p
    wn[p + q:p + 2*q] = B0n
    wp[p + 2*q:p + 3*q] = A1p
    wn[p + 2*q:p + 3*q] = A1n
    wp[p + 3*q:p + 4*q] = B1p
    wn[p + 3*q:p + 4*q] = B1n


def positive_inverse_residue_to_real(work, annihilate_stage=True, return_trace=False):
    work = np.asarray(work, dtype=np.float64)
    if work.shape[0] != 512:
        raise ValueError("expected residue length 512")

    wp, wn = split_pm(work)
    trace = []

    def canonicalize(tag):
        if annihilate_stage:
            annihilate_common_mode(wp, wn)
        if return_trace:
            projected = wp - wn
            sheet_mass = float(np.sum(wp) + np.sum(wn))
            projected_l1 = float(np.sum(np.abs(projected)))
            trace.append({
                "tag": tag,
                "sheet_mass": sheet_mass,
                "projected_l1": projected_l1,
                "cancellation_ratio": sheet_mass / max(projected_l1, 1e-300),
            })

    canonicalize("residue")

    for stage in range(7, 0, -1):
        q = (N // 4) >> stage

        for m in range(1, 2**stage):
            p = 4 * q * m
            c = C[m]
            s = S_SPECIAL[1] if m == 1 else C[m ^ 1]
            positive_inverse_bruun_node_inplace(wp, wn, p, q, c, s)

        prefix = N >> stage
        half = prefix >> 1
        positive_inverse_binomial_inplace(wp, wn, slice(0, half), slice(half, prefix))

        canonicalize(f"undo S{stage}")

    positive_inverse_binomial_inplace(wp, wn, slice(0, 256), slice(256, 512))
    canonicalize("undo S0")

    x = wp - wn
    if return_trace:
        return x, wp, wn, trace

    return x


def positive_irfft(X, annihilate_stage=True, return_trace=False):
    work = unpack_rfft_to_residue(X)
    return positive_inverse_residue_to_real(
        work,
        annihilate_stage=annihilate_stage,
        return_trace=return_trace,
    )


def phase_cone_stage_derived_rfft(input_data, stage_scale=0.5):
    x = np.asarray(input_data, dtype=np.float64)
    if x.shape[0] != 512:
        raise ValueError("expected length 512")

    xp, xn = split_pm(x)
    wp = np.zeros(512, dtype=np.float64)
    wn = np.zeros(512, dtype=np.float64)
    scale_exp = 0

    wp[:256], wn[:256] = xp[:256] + xp[256:], xn[:256] + xn[256:]
    wp[256:], wn[256:] = xp[:256] + xn[256:], xn[:256] + xp[256:]

    annihilate_common_mode(wp, wn)
    wp *= stage_scale
    wn *= stage_scale
    scale_exp += 1

    for stage in range(1, 8):
        prefix = N >> stage
        half = prefix >> 1

        binomial_transport_inplace(
            wp,
            wn,
            slice(0, half),
            slice(half, prefix),
            slice(0, half),
            slice(half, prefix),
        )

        q = (N // 4) >> stage
        for m in range(1, 2**stage):
            p = 4 * q * m
            c = C[m]
            s = S_SPECIAL[1] if m == 1 else C[m ^ 1]
            bruun_transport_node_inplace(wp, wn, p, q, c, s)

        annihilate_common_mode(wp, wn)
        wp *= stage_scale
        wn *= stage_scale
        scale_exp += 1

    work = (wp - wn) * (stage_scale ** (-scale_exp))
    return pack_residue_to_rfft(work)


def run_roundtrip_tests():
    cases = {
        "random_positive_42": np.random.default_rng(42).random(512),
        "normal_43": np.random.default_rng(43).normal(size=512),
        "normal_44": np.random.default_rng(44).normal(size=512),
        "ramp": np.arange(512, dtype=np.float64),
        "alternating": np.where(np.arange(512) % 2 == 0, 1.0, -1.0),
        "impulse_0": np.eye(512)[0],
        "ones": np.ones(512),
        "zeros": np.zeros(512),
    }

    rows = []
    for name, x in cases.items():
        X = phase_cone_stage_derived_rfft(x)
        xr, wp, wn, trace = positive_irfft(X, annihilate_stage=True, return_trace=True)
        rows.append({
            "case": name,
            "positive_forward_inverse_max_abs": float(np.max(np.abs(xr - x))),
            "positive_forward_inverse_l2": float(np.linalg.norm(xr - x)),
            "final_sheet_mass": float(np.sum(wp) + np.sum(wn)),
            "final_cancellation_ratio": float((np.sum(wp) + np.sum(wn)) / max(np.sum(np.abs(wp - wn)), 1e-300)),
        })

    return rows


if __name__ == "__main__":
    for row in run_roundtrip_tests():
        print(row)
