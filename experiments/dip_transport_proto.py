"""Transport model + transport-optimal reform of the DIP real kernel.

model_current      : exact replica of src/detail/bruun_dip_kernel.hpp today
                     (R2HC reflected packing; each cell scatters to 4 rows
                     {d, 2e-d, e-d, e+d} -> 4 diverging write streams).
model_interleaved  : transport-optimal diagonal-major packing (dc/ny on own
                     rows; middle bins stored as ADJACENT (a,b) packets in
                     diagonal order -> each cell writes 2 contiguous 2-wide
                     slots {d, e-d} and reads 1 contiguous 2-wide bin).
"""
import numpy as np


def pair_reduce_cs(ea, eb, oa, ob, c, s):
    r = c * oa - s * ob
    i = s * oa + c * ob
    return ea + r, eb + i, ea - r, i - eb


def phase_index(d, e, n):
    return (d * n) // (2 * e)


# ---- MODEL A: current C++ kernel (R2HC reflected rows) ----
def seed4_current(x, n):
    q = n >> 2
    dst = np.empty(n)
    for i in range(q):
        a, b, c, d = x[i], x[q + i], x[2 * q + i], x[3 * q + i]
        dst[i] = a + b + c + d
        dst[q + i] = a - c
        dst[2 * q + i] = a - b + c - d
        dst[3 * q + i] = b - d
    return dst


def forward_stage_current(src, dst, e, q, n):
    q2 = q >> 1
    for i in range(q2):
        lo = src[i]; hi = src[q2 + i]
        dst[i] = lo + hi
        dst[e * q2 + i] = lo - hi
    if e >= 2:
        base_ny = (e // 2) * q
        for i in range(q2):
            dst[(e // 2) * q2 + i] = src[base_ny + i]
            dst[(2 * e - e // 2) * q2 + i] = src[base_ny + q2 + i]
    for d in range(1, e // 2):
        k = phase_index(d, e, n)
        c = np.cos(2 * np.pi * k / n); s = np.sin(2 * np.pi * k / n)
        ia = d * q; ib = (e - d) * q
        for i in range(q2):
            ea, eb = src[ia + i], src[ib + i]
            oa, ob = src[ia + q2 + i], src[ib + q2 + i]
            la, lb, ha, hb = pair_reduce_cs(ea, eb, oa, ob, c, s)
            dst[d * q2 + i] = la
            dst[(2 * e - d) * q2 + i] = lb
            dst[(e - d) * q2 + i] = ha
            dst[(e + d) * q2 + i] = hb


def dip_current(x):
    n = len(x); half = n >> 1
    src = seed4_current(x, n); dst = np.empty(n)
    e, q = 4, n >> 2
    while e < n:
        forward_stage_current(src, dst, e, q, n)
        src, dst = dst, src; e <<= 1; q >>= 1
    out = np.zeros(half + 1, np.complex128)
    out[0] = src[0]; out[half] = src[half]
    for d in range(1, half):
        out[d] = src[d] - 1j * src[n - d]
    return out


# ---- MODEL B: interleaved diagonal-major packing ----
# level e: row0=dc, row1=ny, rows(2d,2d+1)=(a[d],b[d]) for d in [1,e/2)
def seed4_interleaved(x, n):
    q = n >> 2
    dst = np.empty(n)
    r0, r1, r2, r3 = dst[0:q], dst[q:2*q], dst[2*q:3*q], dst[3*q:4*q]
    for i in range(q):
        a, b, c, d = x[i], x[q + i], x[2 * q + i], x[3 * q + i]
        r0[i] = a + b + c + d   # dc
        r1[i] = a - b + c - d   # ny
        r2[i] = a - c           # a[1]
        r3[i] = b - d           # b[1]
    return dst


def forward_stage_interleaved(src, dst, e, q, n):
    q2 = q >> 1
    dc = src[0:q]; ny = src[q:2 * q]
    dst[0:q2] = dc[0:q2] + dc[q2:q]     # new dc
    dst[q2:q] = dc[0:q2] - dc[q2:q]     # new ny
    if e >= 2:
        a_row = e * q2; b_row = (e + 1) * q2   # new diagonal e/2 -> rows e,e+1
        dst[a_row:a_row + q2] = ny[0:q2]
        dst[b_row:b_row + q2] = ny[q2:q]
    for d in range(1, e // 2):
        k = phase_index(d, e, n)
        c = np.cos(2 * np.pi * k / n); s = np.sin(2 * np.pi * k / n)
        ia = (2 * d) * q; ib = (2 * d + 1) * q
        ea = src[ia:ia + q2]; eb = src[ib:ib + q2]
        oa = src[ia + q2:ia + q]; ob = src[ib + q2:ib + q]
        la, lb, ha, hb = pair_reduce_cs(ea, eb, oa, ob, c, s)
        od = (2 * d) * q2                # new bin d -> rows 2d,2d+1
        dst[od:od + q2] = la; dst[od + q2:od + 2 * q2] = lb
        oe = (2 * (e - d)) * q2          # new bin e-d -> rows 2(e-d),+1
        dst[oe:oe + q2] = ha; dst[oe + q2:oe + 2 * q2] = hb


def dip_interleaved(x):
    n = len(x); half = n >> 1
    src = seed4_interleaved(x, n); dst = np.empty(n)
    e, q = 4, n >> 2
    while e < n:
        forward_stage_interleaved(src, dst, e, q, n)
        src, dst = dst, src; e <<= 1; q >>= 1
    out = np.zeros(half + 1, np.complex128)
    out[0] = src[0]; out[half] = src[1]
    for d in range(1, half):
        out[d] = src[2 * d] - 1j * src[2 * d + 1]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    print("== current vs interleaved vs numpy.rfft ==")
    for p in range(2, 14):
        n = 1 << p
        x = rng.standard_normal(n)
        ref = np.fft.rfft(x)
        e_cur = np.max(np.abs(dip_current(x) - ref))
        e_int = np.max(np.abs(dip_interleaved(x) - ref))
        print(f"  N={n:6d}: current={e_cur:.2e}  interleaved={e_int:.2e}")


# ---- inverse of MODEL B (interleaved), plus seed8/seed16 interleaved ----
def pair_expand_cs(la, lb, ha, hb, c, s):
    r = 0.5 * (la - ha)
    i = 0.5 * (lb + hb)
    ea = 0.5 * (la + ha)
    eb = 0.5 * (lb - hb)
    oa = c * r + s * i
    ob = c * i - s * r
    return ea, eb, oa, ob


def inverse_stage_interleaved(src, dst, e, q, n):
    """level 2e -> level e. src interleaved(level 2e), dst interleaved(level e)."""
    q2 = q >> 1
    dc = src[0:q2]; ny = src[q2:q]     # level-2e rows 0,1 (length q2 each)
    dst[0:q2] = 0.5 * (dc + ny)         # dc_E
    dst[q2:q] = 0.5 * (dc - ny)         # dc_O
    if e >= 2:
        a_row = e * q2; b_row = (e + 1) * q2
        out_ny = src[q:2 * q]  # placeholder; real read below
        # new bin e/2 lives at level-2e rows e, e+1
        dst[q:q + q2] = src[a_row:a_row + q2]        # ny_E
        dst[q + q2:2 * q] = src[b_row:b_row + q2]    # ny_O
    for d in range(1, e // 2):
        k = phase_index(d, e, n)
        c = np.cos(2 * np.pi * k / n); s = np.sin(2 * np.pi * k / n)
        od = (2 * d) * q2; oe = (2 * (e - d)) * q2
        la = src[od:od + q2]; lb = src[od + q2:od + 2 * q2]
        ha = src[oe:oe + q2]; hb = src[oe + q2:oe + 2 * q2]
        ea, eb, oa, ob = pair_expand_cs(la, lb, ha, hb, c, s)
        ia = (2 * d) * q; ib = (2 * d + 1) * q
        dst[ia:ia + q2] = ea; dst[ia + q2:ia + q] = oa
        dst[ib:ib + q2] = eb; dst[ib + q2:ib + q] = ob


def iseed4_interleaved(src, out, n):
    """final inverse from level-4 interleaved back to time samples."""
    q = n >> 2
    r0 = src[0:q]; r1 = src[q:2*q]; r2 = src[2*q:3*q]; r3 = src[3*q:4*q]
    for i in range(q):
        dc, ny, a1, b1 = r0[i], r1[i], r2[i], r3[i]
        ac = 0.5 * (dc + ny); bd = 0.5 * (dc - ny)
        out[i] = 0.5 * (ac + a1)
        out[2 * q + i] = 0.5 * (ac - a1)
        out[q + i] = 0.5 * (bd + b1)
        out[3 * q + i] = 0.5 * (bd - b1)


def idip_interleaved(X, n):
    half = n >> 1
    src = np.empty(n); dst = np.empty(n)
    src[0] = X[0].real; src[1] = X[half].real
    for d in range(1, half):
        src[2 * d] = X[d].real
        src[2 * d + 1] = -X[d].imag
    e, q = n, 1
    while e > 4:
        inverse_stage_interleaved(src, dst, e >> 1, q << 1, n)
        src, dst = dst, src; e >>= 1; q <<= 1
    out = np.empty(n)
    iseed4_interleaved(src, out, n)
    return out


# interleaved seed8 / seed16 (permutation of current seed outputs)
def seed8_vals(a, b, c, d, e, f, g, h):
    rt = 0.7071067811865475244
    ae = a - e; bf = b - f; cg = c - g; dh = d - h
    rot_r = rt * (bf - dh); rot_i = rt * (bf + dh)
    r0 = a + b + c + d + e + f + g + h
    r1 = ae + rot_r; r2 = a - c + e - g; r3 = ae - rot_r
    r4 = a - b + c - d + e - f + g - h
    r5 = rot_i - cg; r6 = b - d + f - h; r7 = cg + rot_i
    return r0, r1, r2, r3, r4, r5, r6, r7


def seed8_interleaved(x, n):
    q = n >> 3
    dst = np.empty(n)
    R = [dst[k*q:(k+1)*q] for k in range(8)]
    for i in range(q):
        v = seed8_vals(*[x[j*q + i] for j in range(8)])
        # current R2HC rows: 0=dc,1=a1,2=a2,3=a3,4=ny,5=b3,6=b2,7=b1
        dc, a1, a2, a3, ny, b3, b2, b1 = v
        # interleaved rows: 0=dc,1=ny,2=a1,3=b1,4=a2,5=b2,6=a3,7=b3
        R[0][i]=dc; R[1][i]=ny; R[2][i]=a1; R[3][i]=b1
        R[4][i]=a2; R[5][i]=b2; R[6][i]=a3; R[7][i]=b3
    return dst


def dip_interleaved_seed8(x):
    n = len(x); half = n >> 1
    src = seed8_interleaved(x, n); dst = np.empty(n)
    e, q = 8, n >> 3
    while e < n:
        forward_stage_interleaved(src, dst, e, q, n)
        src, dst = dst, src; e <<= 1; q >>= 1
    out = np.zeros(half + 1, np.complex128)
    out[0] = src[0]; out[half] = src[1]
    for d in range(1, half):
        out[d] = src[2 * d] - 1j * src[2 * d + 1]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    print("\n== interleaved forward(seed8) + inverse roundtrip ==")
    for p in range(3, 14):
        n = 1 << p
        x = rng.standard_normal(n)
        ref = np.fft.rfft(x)
        f8 = dip_interleaved_seed8(x)
        ef8 = np.max(np.abs(f8 - ref))
        X = dip_interleaved(x)
        rt = idip_interleaved(X, n)
        ert = np.max(np.abs(rt - x))
        print(f"  N={n:6d}: seed8_fwd={ef8:.2e}  roundtrip={ert:.2e}")


def seed16_interleaved(x, n):
    q = n >> 4
    dst = np.empty(n)
    R = [dst[k*q:(k+1)*q] for k in range(16)]
    c1,s1 = 0.9238795325112867, 0.3826834323650898
    c2,s2 = 0.7071067811865476, 0.7071067811865476
    c3,s3 = 0.3826834323650898, 0.9238795325112867
    for i in range(q):
        le = seed8_vals(*[x[(2*j)*q + i] for j in range(8)])
        ro = seed8_vals(*[x[(2*j+1)*q + i] for j in range(8)])
        dc = le[0]+ro[0]; ny = le[0]-ro[0]; a4 = le[4]; b4 = ro[4]
        a1,b1,a7,b7 = pair_reduce_cs(le[1],le[7],ro[1],ro[7],c1,s1)
        a2,b2,a6,b6 = pair_reduce_cs(le[2],le[6],ro[2],ro[6],c2,s2)
        a3,b3,a5,b5 = pair_reduce_cs(le[3],le[5],ro[3],ro[5],c3,s3)
        vals = [dc,ny,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6,a7,b7]
        for k in range(16):
            R[k][i] = vals[k]
    return dst


def dip_interleaved_seed16(x):
    n = len(x); half = n >> 1
    src = seed16_interleaved(x, n); dst = np.empty(n)
    e, q = 16, n >> 4
    while e < n:
        forward_stage_interleaved(src, dst, e, q, n)
        src, dst = dst, src; e <<= 1; q >>= 1
    out = np.zeros(half + 1, np.complex128)
    out[0] = src[0]; out[half] = src[1]
    for d in range(1, half):
        out[d] = src[2 * d] - 1j * src[2 * d + 1]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(13)
    print("\n== interleaved seed16 forward ==")
    for p in range(4, 15):
        n = 1 << p
        x = rng.standard_normal(n)
        ref = np.fft.rfft(x)
        e16 = np.max(np.abs(dip_interleaved_seed16(x) - ref))
        print(f"  N={n:6d}: seed16_fwd={e16:.2e}")


def itail8_interleaved(src, out, q, n):
    """level-8 interleaved [dc,ny,a1,b1,a2,b2,a3,b3] -> 8 time samples/col."""
    inv_2rt = 0.7071067811865475244
    R = [src[k*q:(k+1)*q] for k in range(8)]
    # interleaved rows: 0=dc,1=ny,2=a1,3=b1,4=a2,5=b2,6=a3,7=b3
    dc,ny,a1,b1,a2,b2,a3,b3 = R
    O = [out[k*q:(k+1)*q] for k in range(8)]
    for i in range(q):
        # map to the current tail8 r0..r7 = dc,a1,a2,a3,ny,b3,b2,b1
        r0,r1,r2,r3,r4,r5,r6,r7 = dc[i],a1[i],a2[i],a3[i],ny[i],b3[i],b2[i],b1[i]
        even_sum = 0.5 * (r0 + r4); odd_sum = 0.5 * (r0 - r4)
        ae = 0.5 * (r1 + r3); cg = 0.5 * (r7 - r5)
        bf_m = inv_2rt * (r1 - r3); bf_p = inv_2rt * (r5 + r7)
        bf = 0.5 * (bf_m + bf_p); dh = 0.5 * (bf_p - bf_m)
        ae_sum = 0.5 * (even_sum + r2); cg_sum = 0.5 * (even_sum - r2)
        bf_sum = 0.5 * (odd_sum + r6); dh_sum = 0.5 * (odd_sum - r6)
        O[0][i] = 0.5 * (ae_sum + ae); O[4][i] = 0.5 * (ae_sum - ae)
        O[2][i] = 0.5 * (cg_sum + cg); O[6][i] = 0.5 * (cg_sum - cg)
        O[1][i] = 0.5 * (bf_sum + bf); O[5][i] = 0.5 * (bf_sum - bf)
        O[3][i] = 0.5 * (dh_sum + dh); O[7][i] = 0.5 * (dh_sum - dh)


def idip_interleaved_prod(X, n):
    half = n >> 1
    src = np.empty(n); dst = np.empty(n)
    src[0] = X[0].real; src[1] = X[half].real
    for d in range(1, half):
        src[2 * d] = X[d].real
        src[2 * d + 1] = -X[d].imag
    e, q = n, 1
    while e > 8:
        inverse_stage_interleaved(src, dst, e >> 1, q << 1, n)
        src, dst = dst, src; e >>= 1; q <<= 1
    out = np.empty(n)
    itail8_interleaved(src, out, q, n)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    print("\n== production-shaped inverse (loop + itail8) roundtrip ==")
    for p in range(3, 15):
        n = 1 << p
        x = rng.standard_normal(n)
        X = dip_interleaved_seed16(x) if n >= 16 else dip_interleaved_seed8(x)
        rt = idip_interleaved_prod(X, n)
        print(f"  N={n:6d}: roundtrip={np.max(np.abs(rt - x)):.2e}")


def itail4_interleaved(src, out, q, n):
    R = [src[k*q:(k+1)*q] for k in range(4)]
    dc,ny,a1,b1 = R
    O = [out[k*q:(k+1)*q] for k in range(4)]
    for i in range(q):
        r0,r1,r2,r3 = dc[i],a1[i],ny[i],b1[i]
        ac = 0.5*(r0+r2); bd = 0.5*(r0-r2)
        O[0][i]=0.5*(ac+r1); O[2][i]=0.5*(ac-r1)
        O[1][i]=0.5*(bd+r3); O[3][i]=0.5*(bd-r3)


if __name__ == "__main__":
    n = 4
    rng = np.random.default_rng(3)
    x = rng.standard_normal(n)
    src = seed4_interleaved(x, n)
    out = np.empty(n)
    itail4_interleaved(src, out, n>>2, n)
    print(f"\n== itail4 n=4 roundtrip: {np.max(np.abs(out - x)):.2e}")


def idip_interleaved_full(X, n):
    """pure inverse_stage loop to e=1 (mirrors the blocked path leaf)."""
    half = n >> 1
    src = np.empty(n); dst = np.empty(n)
    src[0] = X[0].real; src[1] = X[half].real
    for d in range(1, half):
        src[2 * d] = X[d].real
        src[2 * d + 1] = -X[d].imag
    e, q = n, 1
    while e > 1:
        inverse_stage_interleaved(src, dst, e >> 1, q << 1, n)
        src, dst = dst, src; e >>= 1; q <<= 1
    return src[:n]


if __name__ == "__main__":
    rng = np.random.default_rng(19)
    print("\n== pure inverse_stage loop to e=1 (blocked-leaf) roundtrip ==")
    for p in range(2, 13):
        n = 1 << p
        x = rng.standard_normal(n)
        X = np.fft.rfft(x)
        rt = idip_interleaved_full(X, n)
        print(f"  N={n:6d}: roundtrip={np.max(np.abs(rt - x)):.2e}")
