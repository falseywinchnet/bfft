#ifndef BFFT_CHEBYSHEV_DCT_RADIX4_HPP
#define BFFT_CHEBYSHEV_DCT_RADIX4_HPP

// Production-shaped radix-4 Chebyshev DCT-III plan (ROADMAP Phase 1 hot path).
//
// This is the convention-faithful realization of the DCT-III forward/inverse
// that chebyshev_dct_kernel.hpp prototyped with per-node std::vector allocations
// and per-node node_const() sqrt recomputation. It mirrors the discipline of the
// shipped real-FFT kernel (src/detail/bruun_kernel.hpp, RFFT):
//
//   * a plan object precomputes everything size-dependent ONCE in init():
//       - the per-node Chebyshev constants {u, v, u2, v2, 1/u, 1/v, 1/w} into a
//         64-byte-aligned table indexed by the tree's pre-order node id, so the
//         transform never calls sqrt/divide (mirrors RFFT's C_ twiddle table);
//       - the leaf-order -> DCT-bin permutation, built in O(N) from the closed
//         form of the T_n nodes (not the O(N^2) nearest-match the prototype used
//         only as a reference);
//       - reusable 64-byte-aligned scratch owned by the plan.
//   * forward()/inverse() run IN PLACE on a caller-provided work buffer with
//     ZERO per-call heap allocation, exactly like RFFT::forward_residues.
//   * the tree body is the all-FMA even/odd node (the throughput optimum from
//     capstone.md), depth-first, reading constants from the table; RESTRICT and
//     the [a0|a1|a2|a3] block layout match the FFT kernel.
//
// Pipeline (forward DCT-III, n a power of four):
//   X[0..n-1]  --pack-->  flat Chebyshev coeffs c (c0=X0, cj=2Xj)
//   c          --flat_to_tower_ip (in place, guarded back-substitution)-->  tower
//   tower      --eval_fwd (depth-first all-FMA radix-4, in place)-->  leaf values
//   leaves     --scatter via perm_-->  Y (DCT bin order)
// Inverse runs the schedule backwards: gather, inverse node tree, tower_to_flat,
// unpack. The two are exact structural inverses.
//
// Phase 2 (DCT-IV / DST-IV). A numerical search shows DCT-IV is NOT a diagonal
// (or bidiagonal) step on the single cosine tree of the same length. The clean
// route is the capstone's C/S split with a per-k half-sample rotation:
//   Ck = sum_j X_j cos(j th_k)              (cosine / T tree)
//   Sk = sin(th_k) sum_m X_{m+1} U_m(x_k)   (sine projection; U-series -> T-series
//                                            by an O(N) parity suffix-sum, then
//                                            the SAME tree)
//   [DCT-IV_k; DST-IV_k] = 2 R(th_k/2) [Ck; Sk],  R = rotation matrix.
// So DCT-IV/DST-IV cost two cosine-tree evaluations plus an O(N) diagonal — the
// "two small real schedules" the capstone anticipated. Both are self-inverse up
// to 2n. th_k = pi(2k+1)/(2n), x_k = cos th_k (the T_n roots, same nodes as DCT-III).

#include <cmath>
#include <cstddef>
#include <cstdlib>

#include "chebyshev_composed_radix4_kernel.hpp"  // node_const, is_pow4, leaf_x, monomial_node_fma

#if defined(__GNUC__) || defined(__clang__)
#define BFFT_DCT_RESTRICT __restrict__
#else
#define BFFT_DCT_RESTRICT
#endif

namespace cheb_dct_r4 {

constexpr double kPi = 3.141592653589793238462643383279502884;

// ---------------------------------------------------------------------------
// 64-byte aligned double allocation (mirrors RFFT's aligned heap_array). One
// allocation per plan buffer, freed in the destructor; nothing in the hot path.
// ---------------------------------------------------------------------------
inline double* aligned_alloc_doubles(std::size_t count) {
    if (count == 0) count = 1;
    const std::size_t bytes = count * sizeof(double);
    void* p = nullptr;
#if defined(_WIN32)
    p = _aligned_malloc(bytes, 64);
#else
    if (posix_memalign(&p, 64, bytes) != 0) p = nullptr;
#endif
    return static_cast<double*>(p);
}
inline void aligned_free_doubles(double* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

// Per-node constants. u,v,u2,v2 drive the forward node; the reciprocals drive
// the inverse node (so it is multiply-only, no per-element divide). w = u2 - v2.
struct NodeConst {
    double u, v, u2, v2;
    double invu, invv, invw;
};

// Internal-node count of a radix-4 tree on m elements: (m-1)/3.
inline int internal_nodes(int m) { return (m - 1) / 3; }

// ---------------------------------------------------------------------------
// Flat-Chebyshev <-> nested power-of-T_q tower repack, in place / arena-based.
//
// Identical algebra to chebyshev_dct_kernel.hpp, but with no per-node heap
// allocation. add_Tpow takes a write limit `lim` so the in-place forward
// back-substitution can subtract a block's cross-band aliasing WITHOUT touching
// the band that already holds the recovered block.
// ---------------------------------------------------------------------------
inline void add_T_single(double* BFFT_DCT_RESTRICT f, const double* BFFT_DCT_RESTRICT g,
                         int q, int a, double s, int lim) {
    // Callers always pass a in {q, 2q, 3q} with g of degree < q, so a - r > 0
    // for every r and no fold reaches a negative index (no std::abs needed).
    for (int r = 0; r < q; ++r) {
        const double v = 0.5 * s * g[r];
        const int i1 = a + r;
        if (i1 < lim) f[i1] += v;
        const int i2 = a - r;
        if (i2 < lim) f[i2] += v;
    }
}

// f += s * T_q^p * g, writing only indices < lim. g has degree < q.
inline void add_Tpow(double* BFFT_DCT_RESTRICT f, const double* BFFT_DCT_RESTRICT g,
                     int q, int p, double s, int lim) {
    if (p == 1) {
        add_T_single(f, g, q, q, s, lim);
    } else if (p == 2) {
        add_T_single(f, g, q, 2 * q, 0.5 * s, lim);   // (T_{2q})/2
        for (int r = 0; r < q && r < lim; ++r) f[r] += 0.5 * s * g[r];  // (T_0)/2
    } else {  // p == 3
        add_T_single(f, g, q, 3 * q, 0.25 * s, lim);  // (T_{3q})/4
        add_T_single(f, g, q, q, 0.75 * s, lim);      // (3 T_q)/4
    }
}

// Flat Chebyshev coeffs -> nested tower, in place on p[off, off+m).
inline void flat_to_tower_ip(double* p, int off, int m) {
    if (m == 1) return;
    const int q = m / 4;
    double* c = p + off;

    // g3 occupies [3q,4q): recover in place, then strip its alias from [0,3q).
    c[3 * q] *= 4.0;
    for (int r = 1; r < q; ++r) c[3 * q + r] *= 8.0;
    add_Tpow(c, c + 3 * q, q, 3, -1.0, 3 * q);

    // g2 occupies [2q,3q).
    c[2 * q] *= 2.0;
    for (int r = 1; r < q; ++r) c[2 * q + r] *= 4.0;
    add_Tpow(c, c + 2 * q, q, 2, -1.0, 2 * q);

    // g1 occupies [q,2q). (c[q] keeps factor 1.)
    for (int r = 1; r < q; ++r) c[q + r] *= 2.0;
    add_Tpow(c, c + q, q, 1, -1.0, q);

    // g0 = remaining [0,q). Recurse into all four blocks.
    flat_to_tower_ip(p, off, q);
    flat_to_tower_ip(p, off + q, q);
    flat_to_tower_ip(p, off + 2 * q, q);
    flat_to_tower_ip(p, off + 3 * q, q);
}

// Nested tower (src) -> flat Chebyshev coeffs (out), out != src, using arena
// (size >= m). Out-of-place avoids the read/write aliasing of in-place synthesis.
inline void tower_to_flat_oop(const double* BFFT_DCT_RESTRICT src, int off, int m,
                              double* BFFT_DCT_RESTRICT out, double* BFFT_DCT_RESTRICT arena) {
    if (m == 1) { out[off] = src[off]; return; }
    const int q = m / 4;
    tower_to_flat_oop(src, off, q, out, arena);
    tower_to_flat_oop(src, off + q, q, out, arena);
    tower_to_flat_oop(src, off + 2 * q, q, out, arena);
    tower_to_flat_oop(src, off + 3 * q, q, out, arena);
    // out[off, off+m) = [flat g0 | flat g1 | flat g2 | flat g3]; combine.
    for (int i = 0; i < m; ++i) arena[i] = 0.0;
    for (int r = 0; r < q; ++r) arena[r] += out[off + r];
    add_Tpow(arena, out + off + q, q, 1, 1.0, m);
    add_Tpow(arena, out + off + 2 * q, q, 2, 1.0, m);
    add_Tpow(arena, out + off + 3 * q, q, 3, 1.0, m);
    for (int i = 0; i < m; ++i) out[off + i] = arena[i];
}

// ---------------------------------------------------------------------------
// The plan.
// ---------------------------------------------------------------------------
class ChebDCT {
public:
    ChebDCT() = default;
    ~ChebDCT() {
        aligned_free_doubles(reinterpret_cast<double*>(nc_));
        aligned_free_doubles(scratch_a_);
        aligned_free_doubles(scratch_b_);
        aligned_free_doubles(tw_cos2_);
        aligned_free_doubles(tw_sin2_);
        aligned_free_doubles(tw_sin_);
        aligned_free_doubles(inv_sin_);
        std::free(perm_);
        std::free(sched_off_);
        std::free(sched_q_);
    }
    ChebDCT(const ChebDCT&) = delete;
    ChebDCT& operator=(const ChebDCT&) = delete;

    // n must be a power of four (the radix-4 tree's natural size). Returns false
    // otherwise. All allocation and constant precompute happens here, once.
    bool init(int n) {
        if (!cheb_composed_r4::is_pow4(n) || n < 4) return false;
        n_ = n;
        const int nodes = internal_nodes(n);
        nc_ = reinterpret_cast<NodeConst*>(aligned_alloc_doubles(
            (nodes * sizeof(NodeConst) + sizeof(double) - 1) / sizeof(double)));
        scratch_a_ = aligned_alloc_doubles(n);
        scratch_b_ = aligned_alloc_doubles(n);
        tw_cos2_ = aligned_alloc_doubles(n);
        tw_sin2_ = aligned_alloc_doubles(n);
        tw_sin_ = aligned_alloc_doubles(n);
        inv_sin_ = aligned_alloc_doubles(n);
        perm_ = static_cast<int*>(std::malloc(sizeof(int) * n));
        nodes_ = nodes;
        sched_off_ = static_cast<int*>(std::malloc(sizeof(int) * (nodes ? nodes : 1)));
        sched_q_ = static_cast<int*>(std::malloc(sizeof(int) * (nodes ? nodes : 1)));
        if (!nc_ || !scratch_a_ || !scratch_b_ || !tw_cos2_ || !tw_sin2_ ||
            !tw_sin_ || !inv_sin_ || !perm_ || !sched_off_ || !sched_q_)
            return false;

        fill_constants(0, n, 0.0, 0);
        build_permutation(n);
        // Half-sample twiddles for the DCT-IV/DST-IV rotation and the DST sine
        // weight (with its reciprocal for the DST-III inverse), by bin k.
        for (int k = 0; k < n; ++k) {
            const double thk = kPi * (2.0 * k + 1.0) / (2.0 * n);
            tw_cos2_[k] = std::cos(0.5 * thk);
            tw_sin2_[k] = std::sin(0.5 * thk);
            tw_sin_[k] = std::sin(thk);
            inv_sin_[k] = 1.0 / tw_sin_[k];  // sin(th_k) > 0 for all k in [0,n)
        }
        return true;
    }

    int size() const { return n_; }

    // Forward DCT-III. input/output have n_ doubles; work has n_ doubles and may
    // alias neither. No heap allocation. output must differ from work.
    void dct3_forward(const double* BFFT_DCT_RESTRICT input,
                      double* BFFT_DCT_RESTRICT output,
                      double* BFFT_DCT_RESTRICT work) const {
        // Pack samples into flat Chebyshev coefficients: c0 = X0, cj = 2 Xj.
        work[0] = input[0];
        for (int j = 1; j < n_; ++j) work[j] = 2.0 * input[j];
        tree_eval_inplace(work, output);
    }

    // Evaluate a flat Chebyshev T-series sum_j a[j] T_j at the T_n nodes, in DCT
    // bin order. dct3 is this with a = {X0, 2X1, ...}; DCT-IV/DST-IV use it twice.
    // a is not modified; work has n_ doubles and must differ from out and a.
    void eval_T(const double* BFFT_DCT_RESTRICT a,
                double* BFFT_DCT_RESTRICT out,
                double* BFFT_DCT_RESTRICT work) const {
        for (int i = 0; i < n_; ++i) work[i] = a[i];
        tree_eval_inplace(work, out);
    }

    // Forward DCT-IV (FFTW REDFT11). Self-inverse up to 2n. No per-call heap
    // allocation; uses plan scratch. work has n_ doubles, distinct from output.
    //
    // DCT-IV = capstone C/S split with a per-k half-sample rotation:
    //   Ck = sum_j X_j cos(j th_k)            (cosine / T tree)
    //   Sk = sin(th_k) sum_m X_{m+1} U_m(x_k) (sine projection, U->T then tree)
    //   Y_k = 2[ cos(th_k/2) Ck - sin(th_k/2) Sk ]
    // (DCT-IV is NOT a diagonal-only step on one cosine tree; it genuinely needs
    //  the sine companion. See the note above the class.)
    void dct4_forward(const double* BFFT_DCT_RESTRICT input,
                      double* BFFT_DCT_RESTRICT output,
                      double* BFFT_DCT_RESTRICT work) const {
        cs_projections(input, work);  // scratch_a_ = Ck, work = qk (pre-sin)
        for (int k = 0; k < n_; ++k) {
            const double Ck = scratch_a_[k];
            const double Sk = tw_sin_[k] * work[k];
            output[k] = 2.0 * (tw_cos2_[k] * Ck - tw_sin2_[k] * Sk);
        }
    }

    // Forward DST-IV (FFTW RODFT11). Self-inverse up to 2n. Same projections as
    // DCT-IV, rotated by +th_k/2 instead of -th_k/2.
    void dst4_forward(const double* BFFT_DCT_RESTRICT input,
                      double* BFFT_DCT_RESTRICT output,
                      double* BFFT_DCT_RESTRICT work) const {
        cs_projections(input, work);
        for (int k = 0; k < n_; ++k) {
            const double Ck = scratch_a_[k];
            const double Sk = tw_sin_[k] * work[k];
            output[k] = 2.0 * (tw_sin2_[k] * Ck + tw_cos2_[k] * Sk);
        }
    }

    // Inverses: DCT-IV and DST-IV are self-inverse up to 2n.
    void dct4_inverse(const double* BFFT_DCT_RESTRICT input,
                      double* BFFT_DCT_RESTRICT output,
                      double* BFFT_DCT_RESTRICT work) const {
        dct4_forward(input, output, work);
        const double s = 1.0 / (2.0 * n_);
        for (int k = 0; k < n_; ++k) output[k] *= s;
    }
    void dst4_inverse(const double* BFFT_DCT_RESTRICT input,
                      double* BFFT_DCT_RESTRICT output,
                      double* BFFT_DCT_RESTRICT work) const {
        dst4_forward(input, output, work);
        const double s = 1.0 / (2.0 * n_);
        for (int k = 0; k < n_; ++k) output[k] *= s;
    }

    // Inverse of dct3_forward (equals FFTW REDFT10 / 2n applied to the bins).
    // input has the DCT-III bins, output recovers the samples X. Uses plan
    // scratch; no per-call heap allocation. work has n_ doubles.
    void dct3_inverse(const double* BFFT_DCT_RESTRICT input,
                      double* BFFT_DCT_RESTRICT output,
                      double* BFFT_DCT_RESTRICT work) const {
        tree_eval_inverse(input, scratch_a_, work);  // scratch_a_ = flat T-coeffs c
        output[0] = scratch_a_[0];
        for (int j = 1; j < n_; ++j) output[j] = 0.5 * scratch_a_[j];
    }

    // Forward DST-III (FFTW RODFT01). The sine sibling of DCT-III: evaluation of
    // a U-series at the SAME T_n nodes, scaled by sin(th_k).
    //   Y_k = sin(th_k) * sum_p d_p U_p(x_k),  d_p = 2 X_p (p<n-1), d_{n-1}=X_{n-1}
    // The U-series is converted to a T-series (u_to_t) and run through the same
    // tree. Inverse is DST-II / 2n. No per-call heap allocation.
    void dst3_forward(const double* BFFT_DCT_RESTRICT input,
                      double* BFFT_DCT_RESTRICT output,
                      double* BFFT_DCT_RESTRICT work) const {
        // U-series coeffs d (DC-and-bulk doubled, top coeff single) into work.
        for (int p = 0; p < n_ - 1; ++p) work[p] = 2.0 * input[p];
        work[n_ - 1] = input[n_ - 1];
        u_to_t(work, work);                 // -> flat T-coeffs b, in place
        eval_T(work, output, scratch_a_);   // -> qk in bin order
        for (int k = 0; k < n_; ++k) output[k] *= tw_sin_[k];  // * sin(th_k)
    }

    // Inverse of dst3_forward (true inverse; equals FFTW RODFT10 / 2n on bins).
    void dst3_inverse(const double* BFFT_DCT_RESTRICT input,
                      double* BFFT_DCT_RESTRICT output,
                      double* BFFT_DCT_RESTRICT work) const {
        for (int k = 0; k < n_; ++k) work[k] = input[k] * inv_sin_[k];  // / sin(th_k)
        // Invert the tree evaluation -> flat T-coeffs b (uses output as gather scratch).
        tree_eval_inverse(work, scratch_a_, output);
        t_to_u(scratch_a_, scratch_a_);     // T-coeffs -> U-coeffs d, in place
        for (int p = 0; p < n_ - 1; ++p) output[p] = 0.5 * scratch_a_[p];
        output[n_ - 1] = scratch_a_[n_ - 1];
    }

    // Forward DST-II (FFTW RODFT10) = 2n * (DST-III)^{-1}; its grid differs from
    // the tree's, so it is realized through the DST-III inverse schedule.
    void dst2_forward(const double* BFFT_DCT_RESTRICT input,
                      double* BFFT_DCT_RESTRICT output,
                      double* BFFT_DCT_RESTRICT work) const {
        dst3_inverse(input, output, work);
        const double s = 2.0 * n_;
        for (int k = 0; k < n_; ++k) output[k] *= s;
    }
    // Inverse DST-II = DST-III / 2n.
    void dst2_inverse(const double* BFFT_DCT_RESTRICT input,
                      double* BFFT_DCT_RESTRICT output,
                      double* BFFT_DCT_RESTRICT work) const {
        dst3_forward(input, output, work);
        const double s = 1.0 / (2.0 * n_);
        for (int k = 0; k < n_; ++k) output[k] *= s;
    }

private:
    int n_ = 0;
    int nodes_ = 0;                  // internal node count = (n-1)/3
    NodeConst* nc_ = nullptr;        // pre-order node-constant table
    double* scratch_a_ = nullptr;    // n_ doubles
    double* scratch_b_ = nullptr;    // n_ doubles (arena)
    double* tw_cos2_ = nullptr;      // cos(theta_k / 2), by bin k
    double* tw_sin2_ = nullptr;      // sin(theta_k / 2), by bin k
    double* tw_sin_ = nullptr;       // sin(theta_k), by bin k
    double* inv_sin_ = nullptr;      // 1 / sin(theta_k), by bin k
    int* perm_ = nullptr;            // leaf index -> DCT bin index
    int* sched_off_ = nullptr;       // block offset of pre-order node id
    int* sched_q_ = nullptr;         // block quarter-size q of pre-order node id

    // flat coeffs already in `work` -> evaluate at the T_n nodes -> scatter to
    // out in DCT bin order. Shared by dct3/eval_T. work is consumed in place.
    // Single fused pass: at each node, the local flat->tower split and the node
    // combine run on the same block while it is hot in cache, instead of a
    // separate full flat_to_tower traversal followed by a full eval traversal.
    void tree_eval_inplace(double* BFFT_DCT_RESTRICT work,
                           double* BFFT_DCT_RESTRICT out) const {
        // Flat iterative schedule, pre-order (parent id < child ids), so each
        // node is processed before its children consume its output. No recursion.
        for (int id = 0; id < nodes_; ++id) {
            double* c = work + sched_off_[id];
            const int q = sched_q_[id];
            local_split(c, q);
            forward_node(c, q, nc_[id].u, nc_[id].v);
        }
        for (int i = 0; i < n_; ++i) out[perm_[i]] = work[i];
    }

    // Inverse of tree_eval_inplace: node values in bin order (vals) -> flat
    // T-coeffs (coeffs_out). gather_work (consumed) is the bin->leaf gather
    // buffer; coeffs_out, gather_work, vals and scratch_b_ must be distinct.
    void tree_eval_inverse(const double* BFFT_DCT_RESTRICT vals,
                           double* BFFT_DCT_RESTRICT coeffs_out,
                           double* BFFT_DCT_RESTRICT gather_work) const {
        for (int i = 0; i < n_; ++i) gather_work[i] = vals[perm_[i]];  // gather
        // Flat schedule, post-order (children before parent) = descending id.
        for (int id = nodes_ - 1; id >= 0; --id)
            inverse_node(gather_work + sched_off_[id], sched_q_[id], nc_[id]);
        tower_to_flat_oop(gather_work, 0, n_, coeffs_out, scratch_b_); // -> flat
    }

    // U-series coeffs d (length n) -> flat T-series coeffs b (length n), via
    //   U_m = 2(T_m + T_{m-2} + ...), T_0 term weighted 1.
    // A parity suffix-sum; safe in place (b may alias d).
    void u_to_t(const double* BFFT_DCT_RESTRICT d, double* BFFT_DCT_RESTRICT b) const {
        for (int t = n_ - 1; t >= 0; --t)
            b[t] = d[t] + (t + 2 < n_ ? b[t + 2] : 0.0);
        for (int t = 1; t < n_; ++t) b[t] *= 2.0;
    }

    // Inverse of u_to_t: flat T-series coeffs b -> U-series coeffs d.
    //   s_t = b_t / (t==0?1:2);  d_p = s_p - s_{p+2}.  Safe in place.
    void t_to_u(const double* BFFT_DCT_RESTRICT b, double* BFFT_DCT_RESTRICT d) const {
        d[0] = b[0];
        for (int t = 1; t < n_; ++t) d[t] = 0.5 * b[t];            // d now holds s
        for (int p = 0; p < n_; ++p) d[p] -= (p + 2 < n_ ? d[p + 2] : 0.0);
    }

    // Compute the cosine projection Ck into scratch_a_ and the (pre-sin) sine
    // projection qk into qk_out, both in bin order. Shared by DCT-IV/DST-IV.
    //   Ck = sum_j X_j cos(j th_k)        -> T-series, a_j = X_j
    //   qk = sum_m X_{m+1} U_m(x_k)       -> U-series, converted to T-series
    // (Sk = sin(th_k) * qk is applied by the caller.) Buffers: scratch_a_ = Ck,
    // scratch_b_ = internal, qk_out = qk; all must be distinct.
    void cs_projections(const double* BFFT_DCT_RESTRICT X,
                        double* BFFT_DCT_RESTRICT qk_out) const {
        // Cosine projection: a_j = X_j (no DC doubling) -> scratch_a_ = Ck.
        eval_T(X, scratch_a_, scratch_b_);

        // Sine projection: U-series coeffs d_m = X_{m+1}, m = 0..n-2, to T coeffs
        // b by the parity suffix-sum of U_m = 2(T_m + T_{m-2} + ...), T_0 weight 1.
        double* b = scratch_b_;
        for (int t = n_ - 1; t >= 0; --t) {
            const double dt = (t <= n_ - 2) ? X[t + 1] : 0.0;     // d_t = X_{t+1}
            const double prev = (t + 2 <= n_ - 1) ? b[t + 2] : 0.0;
            b[t] = dt + prev;                                     // s[t]
        }
        for (int t = 1; t < n_; ++t) b[t] *= 2.0;                 // weight (T_0 stays 1)

        tree_eval_inplace(b, qk_out);  // eval T-series b at the nodes -> qk
    }

    // Fill nc_[id] and the iterative schedule (sched_off_/sched_q_) for the node
    // at block offset `off` spanning m elements with anchor value a. Pre-order
    // ids: children at id+1, id+1+S, id+1+2S, id+1+3S, S = child internal count.
    void fill_constants(int off, int m, double a, int id) {
        if (m == 1) return;
        const int q = m / 4;
        const cheb_composed_r4::NodeConst k = cheb_composed_r4::node_const(a);
        NodeConst& e = nc_[id];
        e.u = k.u; e.v = k.v;
        e.u2 = k.u * k.u; e.v2 = k.v * k.v;
        e.invu = 1.0 / k.u; e.invv = 1.0 / k.v;
        e.invw = 1.0 / (e.u2 - e.v2);  // u2 - v2 == w
        sched_off_[id] = off;
        sched_q_[id] = q;
        const int S = internal_nodes(q);
        fill_constants(off,           q,  k.u, id + 1);
        fill_constants(off + q,       q, -k.u, id + 1 + S);
        fill_constants(off + 2 * q,   q,  k.v, id + 1 + 2 * S);
        fill_constants(off + 3 * q,   q, -k.v, id + 1 + 3 * S);
    }

    // All-FMA even/odd forward node with RESTRICT so the q-lane loop vectorizes
    // (2-wide fmla.2d on NEON / vfmadd on AVX). Same algebra as monomial_node_fma
    // in the composed kernel, but the composed version omits restrict and stays
    // scalar; the plan owns this copy to get the vector form on its hot path.
    // p = [a0|a1|a2|a3] each q -> [f(+u)|f(-u)|f(+v)|f(-v)].
    static void forward_node(double* BFFT_DCT_RESTRICT p, int q, double u, double v) {
        const double u2 = u * u, v2 = v * v;
        double* BFFT_DCT_RESTRICT a0 = p;
        double* BFFT_DCT_RESTRICT a1 = p + q;
        double* BFFT_DCT_RESTRICT a2 = p + 2 * q;
        double* BFFT_DCT_RESTRICT a3 = p + 3 * q;
        for (int n = 0; n < q; ++n) {
            const double A0 = a0[n], A1 = a1[n], A2 = a2[n], A3 = a3[n];
            const double eu = A0 + A2 * u2;
            const double ou = A1 + A3 * u2;
            const double ev = A0 + A2 * v2;
            const double ov = A1 + A3 * v2;
            a0[n] = eu + u * ou;   // fma
            a1[n] = eu - u * ou;   // fnma (product duplicated, not shared)
            a2[n] = ev + v * ov;
            a3[n] = ev - v * ov;
        }
    }

    // One level of the flat-Chebyshev -> nested-tower split, in place on c[0,m),
    // m = 4q. Recovers g3,g2,g1,g0 into the four blocks by back-substitution; the
    // recursion is owned by the iterative schedule, so this is non-recursive. The
    // fused single pass (local split + node combine per block, one traversal) is
    // equivalent to flat_to_tower (full) then eval (full): towering acts within a
    // block, the node combine acts across the four blocks per lane, and the two
    // commute (see the commute proof in the ROADMAP).
    // Branch-free, with the index ranges of each Chebyshev fold worked out in
    // closed form (the add_Tpow bounds checks become loop limits, so the affine
    // loops can vectorize). Back-substitution order g3 -> g2 -> g1 -> g0.
    static void local_split(double* BFFT_DCT_RESTRICT c, int q) {
        double* BFFT_DCT_RESTRICT g3 = c + 3 * q;
        double* BFFT_DCT_RESTRICT g2 = c + 2 * q;
        double* BFFT_DCT_RESTRICT g1 = c + q;

        // Recover g3 (top quarter), then strip its fold from [0,3q).
        g3[0] *= 4.0;
        for (int r = 1; r < q; ++r) g3[r] *= 8.0;
        for (int r = 1; r < q; ++r) c[3 * q - r] += -0.125 * g3[r];   // T_{3q-r}
        c[q] += -0.75 * g3[0];                                        // 3 T_q, r=0
        for (int r = 1; r < q; ++r) {                                 // 3 T_{q+-r}
            const double v = -0.375 * g3[r];
            c[q + r] += v;
            c[q - r] += v;
        }

        // Recover g2 ([2q,3q) after g3 removed), then strip its fold from [0,2q).
        g2[0] *= 2.0;
        for (int r = 1; r < q; ++r) g2[r] *= 4.0;
        for (int r = 1; r < q; ++r) c[2 * q - r] += -0.25 * g2[r];    // T_{2q-r}
        for (int r = 0; r < q; ++r) c[r] += -0.5 * g2[r];             // T_0 part

        // Recover g1 ([q,2q) after g2 removed), then strip its fold from [0,q).
        for (int r = 1; r < q; ++r) g1[r] *= 2.0;                     // g1[0] weight 1
        for (int r = 1; r < q; ++r) c[q - r] += -0.5 * g1[r];         // T_{q-r}
        // g0 = c[0,q) remains.
    }

    // Inverse of forward_node: [f(+u)|f(-u)|f(+v)|f(-v)] -> [a0|a1|a2|a3].
    // Multiply-only using precomputed reciprocals.
    static void inverse_node(double* BFFT_DCT_RESTRICT p, int q, const NodeConst& k) {
        double* BFFT_DCT_RESTRICT P0 = p;
        double* BFFT_DCT_RESTRICT P1 = p + q;
        double* BFFT_DCT_RESTRICT P2 = p + 2 * q;
        double* BFFT_DCT_RESTRICT P3 = p + 3 * q;
        for (int n = 0; n < q; ++n) {
            const double fpu = P0[n], fmu = P1[n], fpv = P2[n], fmv = P3[n];
            const double eu = 0.5 * (fpu + fmu);
            const double ou = (0.5 * (fpu - fmu)) * k.invu;  // ou = u*ou / u
            const double ev = 0.5 * (fpv + fmv);
            const double ov = (0.5 * (fpv - fmv)) * k.invv;
            const double a2 = (eu - ev) * k.invw;
            const double a3 = (ou - ov) * k.invw;
            P0[n] = eu - a2 * k.u2;   // a0
            P1[n] = ou - a3 * k.u2;   // a1
            P2[n] = a2;
            P3[n] = a3;
        }
    }

    // Leaf-order -> DCT bin index, in O(N) from the closed form of the T_n nodes.
    void build_permutation(int n) {
        // leaf_x emits the node x-coordinates in the tree's leaf order.
        double* xleaf = scratch_a_;  // borrow scratch (init time only)
        cheb_composed_r4::leaf_x(0, n, 0.0, xleaf);
        for (int i = 0; i < n; ++i) {
            double x = xleaf[i];
            if (x > 1.0) x = 1.0;
            if (x < -1.0) x = -1.0;
            // node k satisfies acos(x) = pi(2k+1)/(2n)  =>  k = (acos(x)*2n/pi - 1)/2
            const double kf = (std::acos(x) * 2.0 * n / kPi - 1.0) * 0.5;
            int k = static_cast<int>(std::lround(kf));
            if (k < 0) k = 0;
            if (k >= n) k = n - 1;
            perm_[i] = k;
        }
    }
};

}  // namespace cheb_dct_r4

#endif  // BFFT_CHEBYSHEV_DCT_RADIX4_HPP
