#pragma once

// Internal FCT (fast correlated transform) kernel.
//
// FORWARD-ONLY, by construction.  For each standard bin k of a real frame of
// length n, the FCT emits the leading-edge correlation
//
//     C(k, tau_k) = sum_{t=0}^{tau_k-1} x[t] * exp(-2*pi*i*k*t/n)
//
// at the slice tau_k in [1, n] where the bin maximally correlates under the
// score |C|^2 / tau (energy-normalized coherent mass; the emitted phase is
// the arctan(A/B) of the sine/cosine correlation pair at that slice).  Bins
// with no coherent leading edge default to tau = n, i.e. the plain FFT bin.
// The underlying fixed-diagonal object (THLE) has |det| = 1 but exponentially
// growing condition number, and the adaptive selection is nonlinear -- there
// is no inverse; see notes/thle_transform.md.
//
// THE WALK (validated in experiments/thle_discovery.py, 2026-07-09):
//   1. MANIFOLD: the dyadic pyramid of contiguous-block spectra is an EXACT
//      sampling of the leading-edge manifold on its lattice:
//          C(g, m*L) = sum_{b<m} DFT_L(block_b)[f],   g = f*n/L,
//      because on-grid waves have an integer number of cycles per block (the
//      inter-block twiddle is identically 1; no Dirichlet leak on-grid).
//   2. LANDSCAPES (row-shared): per level, running block sums give the score
//      per (row, boundary); each row keeps its best boundary.  All n/L bins
//      that proxy to a row share the work.
//   3. LEVEL CHOICE with a coherence stop: a bin trusts a level only while
//      its proxy row keeps rel * (its best score over levels).  When the
//      level's frequency grid outruns the component bandwidth the proxy
//      dies; the detected collapse is the stopping rule.  The finest trusted
//      level localizes the boundary to +-L.
//   4. ACTIVITY GATE: bins under act * mean|x|^2 have no coherent leading
//      edge and read the plain full transform.
//   5. REFINEMENT (anchor-shared, per-bin exact): one truncated transform
//      per distinct window start anchors C(:, lo); each bin then walks its
//      own window with the endpoint recurrence
//          C(k, tau+1) = C(k, tau) + x[tau] * exp(-2*pi*i*k*tau/n),
//      extending the window geometrically while the argmax sits on an edge.
//   6. EGRESS: refinement already holds the exact slice values; the full
//      transform (the pyramid's top level) serves the defaults.
//
// The block/prefix transforms run through the library's own real-FFT plans
// (bfft_plan per level size), exactly as the STFT kernel does -- standard
// spectrum order, no external FFT.
//
// Cost: pyramid n log^2 n cells (the (DFT, ODFT) tower closure of
// notes/thle_transform.md T5b is the production lever to halve this), plus
// signal-adaptive refinement (measured 12-30x under the per-bin DDC bank).
//
// Precision: double only.  CONCURRENCY: a plan owns its scratch, so one plan
// serves one thread at a time; create one plan per thread.

#include <bfft/bfft.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fct {

#ifndef M_PI
static constexpr double kPi = 3.141592653589793238462643383279502884;
#else
static constexpr double kPi = M_PI;
#endif

struct cplx {
    double re = 0.0;
    double im = 0.0;
};

inline bool is_power2_i(int v) { return v > 0 && (v & (v - 1)) == 0; }

inline int ilog2_i(int v) {
    int t = 0;
    while ((1 << t) < v) ++t;
    return t;
}

// One dyadic level: a real-FFT plan for block length L plus its scratch and
// the row-shared landscape results (best boundary + best score per row).
struct level_state {
    int t = 0;                       // block length L = 2^t
    bfft_plan* plan = nullptr;
    std::vector<double> work;
    std::vector<bfft_complex> scratch;
    std::vector<bfft_complex> spec;      // one block spectrum, L/2 + 1 rows
    std::vector<cplx> cum;               // running block sum per row
    std::vector<int> best_m;             // per row: best boundary block index
    std::vector<double> best_s;          // per row: best score
};

class plan {
public:
    explicit plan(int n,
                  int t_min = 4,
                  double rel = 0.5,
                  double act = 1.5)
        : n_(n), rel_(rel), act_(act) {
        T_ = ilog2_i(n_);
        t_min_ = t_min;
        if (t_min_ > T_) t_min_ = T_;
        if (t_min_ < 2) t_min_ = 2;
        nbins_ = n_ / 2 + 1;

        levels_.resize(static_cast<size_t>(T_ - t_min_ + 1));
        for (int t = t_min_; t <= T_; ++t) {
            level_state& lv = levels_[static_cast<size_t>(t - t_min_)];
            lv.t = t;
            const size_t L = static_cast<size_t>(1) << t;
            if (bfft_plan_create(L, &lv.plan) != BFFT_OK) {
                destroy();
                ok_ = false;
                return;
            }
            lv.work.resize(bfft_plan_work_size(lv.plan));
            lv.scratch.resize(bfft_plan_native_scratch_size(lv.plan));
            lv.spec.resize(L / 2 + 1);
            lv.cum.resize(L / 2 + 1);
            lv.best_m.resize(L / 2 + 1);
            lv.best_s.resize(L / 2 + 1);
        }
        prefix_.resize(static_cast<size_t>(n_));
        anchor_.resize(static_cast<size_t>(nbins_));
        full_.resize(static_cast<size_t>(nbins_));
        sc_.resize(levels_.size() * static_cast<size_t>(nbins_));
        bd_.resize(levels_.size() * static_cast<size_t>(nbins_));
        tau_c_.resize(static_cast<size_t>(nbins_));
        win_.resize(static_cast<size_t>(nbins_));
        order_.resize(static_cast<size_t>(nbins_));
        ok_ = true;
    }

    ~plan() { destroy(); }

    plan(const plan&) = delete;
    plan& operator=(const plan&) = delete;

    bool valid() const { return ok_; }
    int size() const { return n_; }
    int bins() const { return nbins_; }

    // Forward FCT.  x has n real samples; out has n/2 + 1 packed standard
    // bins C(k, tau_k); tau (optional, may be null) receives the selected
    // slice per bin, in [1, n].  Mutates internal scratch (one thread per
    // plan).
    bool forward(const double* x, cplx* out, int64_t* tau) {
        if (!ok_) return false;
        const int n = n_;
        const int nb = nbins_;

        // ---- activity floor ----
        double mean_sq = 0.0;
        for (int i = 0; i < n; ++i) mean_sq += x[i] * x[i];
        mean_sq /= static_cast<double>(n);
        const double floor_score = act_ * mean_sq;

        // ---- 1+2. pyramid levels with row-shared landscapes ----
        for (level_state& lv : levels_) {
            const int L = 1 << lv.t;
            const int B = n >> lv.t;
            const int rows = L / 2 + 1;
            for (int f = 0; f < rows; ++f) {
                lv.cum[static_cast<size_t>(f)] = cplx{};
                lv.best_m[static_cast<size_t>(f)] = 0;
                lv.best_s[static_cast<size_t>(f)] = -1.0;
            }
            for (int b = 0; b < B; ++b) {
                if (bfft_forward(lv.plan, x + static_cast<size_t>(b) * L,
                                 lv.spec.data(), lv.work.data(),
                                 lv.scratch.data()) != BFFT_OK) {
                    return false;
                }
                const double tau_here =
                    static_cast<double>(b + 1) * static_cast<double>(L);
                for (int f = 0; f < rows; ++f) {
                    cplx& c = lv.cum[static_cast<size_t>(f)];
                    c.re += lv.spec[static_cast<size_t>(f)].re;
                    c.im += lv.spec[static_cast<size_t>(f)].im;
                    const double s = (c.re * c.re + c.im * c.im) / tau_here;
                    if (s > lv.best_s[static_cast<size_t>(f)]) {
                        lv.best_s[static_cast<size_t>(f)] = s;
                        lv.best_m[static_cast<size_t>(f)] = b;
                    }
                }
            }
            if (lv.t == T_) {
                // top level block spectrum IS the full transform
                for (int f = 0; f < rows; ++f) {
                    full_[static_cast<size_t>(f)].re = lv.cum[static_cast<size_t>(f)].re;
                    full_[static_cast<size_t>(f)].im = lv.cum[static_cast<size_t>(f)].im;
                }
            }
        }

        // ---- 3. per-bin level choice with the coherence stop ----
        const size_t nlev = levels_.size();
        for (size_t li = 0; li < nlev; ++li) {
            const level_state& lv = levels_[li];
            const int L = 1 << lv.t;
            for (int k = 0; k < nb; ++k) {
                int f = static_cast<int>(
                    std::llround(static_cast<double>(k) * L / n));
                if (f > L / 2) f = L / 2;
                sc_[li * nb + static_cast<size_t>(k)] =
                    lv.best_s[static_cast<size_t>(f)];
                bd_[li * nb + static_cast<size_t>(k)] =
                    (lv.best_m[static_cast<size_t>(f)] + 1) * L;
            }
        }
        for (int k = 0; k < nb; ++k) {
            double smax = 0.0;
            for (size_t li = 0; li < nlev; ++li) {
                const double s = sc_[li * nb + static_cast<size_t>(k)];
                if (s > smax) smax = s;
            }
            const bool active =
                smax > floor_score && k >= 1 && k <= n / 2;
            if (!active) {
                tau_c_[static_cast<size_t>(k)] = n;
                win_[static_cast<size_t>(k)] = 0;
                continue;
            }
            // finest trusted level = smallest block length holding rel*smax
            size_t chosen = nlev - 1;
            for (size_t li = 0; li < nlev; ++li) {
                if (sc_[li * nb + static_cast<size_t>(k)] >= rel_ * smax) {
                    chosen = li;
                    break;
                }
            }
            tau_c_[static_cast<size_t>(k)] = bd_[chosen * nb + static_cast<size_t>(k)];
            win_[static_cast<size_t>(k)] = 1 << levels_[chosen].t;
        }

        // ---- 4+5. anchor-shared per-bin refinement ----
        // Sort active bins by their window start so bins sharing an anchor
        // run consecutively against one truncated transform.
        int n_active = 0;
        for (int k = 0; k < nb; ++k) {
            if (win_[static_cast<size_t>(k)] > 0) {
                order_[static_cast<size_t>(n_active++)] = k;
            }
        }
        auto lo_of = [&](int k) {
            const int wide =
                std::max(2 * win_[static_cast<size_t>(k)], n / 8);
            return std::max(tau_c_[static_cast<size_t>(k)] - wide, 1);
        };
        std::sort(order_.begin(), order_.begin() + n_active,
                  [&](int a, int b) { return lo_of(a) < lo_of(b); });

        // defaults first; refinement overwrites active bins
        for (int k = 0; k < nb; ++k) {
            out[k] = full_[static_cast<size_t>(k)];
            if (tau) tau[k] = n;
        }

        int cur_anchor = -1;
        for (int i = 0; i < n_active; ++i) {
            const int k = order_[static_cast<size_t>(i)];
            const int wide = std::max(2 * win_[static_cast<size_t>(k)], n / 8);
            int lo = std::max(tau_c_[static_cast<size_t>(k)] - wide, 1);
            int hi = std::min(tau_c_[static_cast<size_t>(k)] + wide, n);
            if (lo != cur_anchor) {
                // one truncated transform anchors C(:, lo) for this group
                for (int t = 0; t < lo; ++t) prefix_[static_cast<size_t>(t)] = x[t];
                for (int t = lo; t < n; ++t) prefix_[static_cast<size_t>(t)] = 0.0;
                level_state& top = levels_.back();
                if (bfft_forward(top.plan, prefix_.data(), anchor_.data(),
                                 top.work.data(), top.scratch.data()) != BFFT_OK) {
                    return false;
                }
                cur_anchor = lo;
            }
            cplx c_lo{anchor_[static_cast<size_t>(k)].re,
                      anchor_[static_cast<size_t>(k)].im};
            const double wk = 2.0 * kPi * static_cast<double>(k) / n;
            int best_tau = 0;
            cplx best_c{};
            for (;;) {
                // walk [lo, hi) with the endpoint recurrence
                double rr = std::cos(wk * lo);
                double ri = -std::sin(wk * lo);
                const double sr = std::cos(wk);
                const double si = -std::sin(wk);
                cplx c = c_lo;
                double best_s = (c.re * c.re + c.im * c.im) /
                                static_cast<double>(lo);
                best_tau = lo;
                best_c = c;
                for (int t = lo; t < hi; ++t) {
                    c.re += x[t] * rr;
                    c.im += x[t] * ri;
                    const double s = (c.re * c.re + c.im * c.im) /
                                     static_cast<double>(t + 1);
                    if (s > best_s) {
                        best_s = s;
                        best_tau = t + 1;
                        best_c = c;
                    }
                    const double nr = rr * sr - ri * si;
                    ri = rr * si + ri * sr;
                    rr = nr;
                }
                const int span = hi - lo;
                if (best_tau >= hi - 3 && hi < n) {
                    hi = std::min(n, hi + span);
                    continue;
                }
                if (best_tau <= lo + 2 && lo > 1) {
                    lo = std::max(1, lo - span);
                    // re-anchor by direct prefix sum (rare; O(lo))
                    double pr = 0.0, pi_ = 0.0;
                    double qr = 1.0, qi = 0.0;
                    for (int t = 0; t < lo; ++t) {
                        pr += x[t] * qr;
                        pi_ += x[t] * qi;
                        const double nq = qr * sr - qi * si;
                        qi = qr * si + qi * sr;
                        qr = nq;
                    }
                    c_lo = cplx{pr, pi_};
                    continue;
                }
                break;
            }
            out[k] = best_c;
            if (tau) tau[k] = best_tau;
        }
        return true;
    }

private:
    void destroy() {
        for (level_state& lv : levels_) {
            bfft_plan_destroy(lv.plan);
            lv.plan = nullptr;
        }
    }

    int n_ = 0;
    int T_ = 0;
    int t_min_ = 4;
    int nbins_ = 0;
    double rel_ = 0.5;
    double act_ = 1.5;
    bool ok_ = false;

    std::vector<level_state> levels_;
    std::vector<double> prefix_;
    std::vector<bfft_complex> anchor_;
    std::vector<cplx> full_;
    std::vector<double> sc_;     // [level][bin] proxy score
    std::vector<int> bd_;        // [level][bin] proxy boundary
    std::vector<int> tau_c_;
    std::vector<int> win_;
    std::vector<int> order_;
};

} // namespace fct
