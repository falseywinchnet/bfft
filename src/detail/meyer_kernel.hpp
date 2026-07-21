#pragma once

// Internal Meyer G-norm decomposer kernel (transport geometry fusion
// descent).  See include/bfft/meyer.h for the contract,
// notes/meyer_bregman_ladder.md for the algorithm measurements, and
// notes/meyer_accel_theory.md for the reduced-composite theory (the
// alternation is ISTA at step 1/lambda on a static convex composite).
//
// TRANSFORM ECONOMY.  One Split Bregman sweep of ROF(g, c) is
//
//     u = F^-1[ (c*g_hat - eta*div(d-b)_hat) / (c - eta*lap_hat) ]
//     t = grad(u) + b;  coef = max(|t| - 1/eta, 0)/|t|
//     d = coef*t;       b <- b + grad(u) - d
//
// Only div(d-b) ever needs a forward transform: g_hat is maintained
// spectrally.  In the outer alternation g_u = f - v = u + w and
// g_v = f - u, so keeping the spectra of f, u and w makes every sweep cost
// exactly one forward + one inverse 2-D transform.  The texture image v is
// never materialized until the end (v = f - u - w, one fused pass).
//
// SHRINK ALGEBRA.  With t = grad(u) + b the update collapses to
//
//     b_new  = (1 - coef) * t
//     d - b_new = (2*coef - 1) * t      (the only combination ever used)
//
// so the fused spatial pass reads u's forward neighbors and b, and writes
// b and db = d - b_new: two planes in, four planes out, no temporaries.
//
// 2-D TRANSFORMS over the library's 1-D real plans, k-major layout.
// Forward: rows are transformed in panels of eight into a small complex
// stage, then panel-transposed into column-major Re/Im planes
// (reT/imT[k*H + i]); the column stage is ZERO-COPY -- each column rfft
// reads a contiguous H-run and writes its half-spectrum directly into the
// spectrum array, whose layout is complex (k, m) = a[2*(k*HB + m)].  The
// screened-Poisson symbol is real and even in both indices, so scaling the
// two planes entrywise IS the 2-D spectral solve, and linear combinations
// of spectra are exact; symbol tables are pre-expanded to interleaved
// (re, im) stride so every spectral loop is a pure unit-stride stream.
// Inverse reverses the two stages.
//
// THREADING.  Every stage is embarrassingly parallel: row panels, column
// transforms, shrink rows, and solve ranges partition disjointly.  A
// persistent pool of T-1 workers plus the caller executes each stage as
// one barrier region (~6 regions per sweep); each lane owns its own row
// and column plans, work buffers, stage panel, and line buffer, so no
// library state is ever shared between lanes.  The rung tolerance
// reduction stays SERIAL by design: outputs are bit-identical for every
// thread count.
//
// Everything is double; the library's inverse is numpy-normalized, so no
// scale factors appear anywhere.

#include <bfft/bfft.h>

#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace meyer {

constexpr std::size_t PANEL = 8;   // rows per panel transpose; H,W >= 8

// ---- persistent worker pool: run(f) executes f(tid) on all T lanes ------

class pool {
public:
    void start(int threads) {
        T = threads < 1 ? 1 : threads;
        for (int t = 1; t < T; ++t)
            workers.emplace_back([this, t] { worker(t); });
    }
    ~pool() {
        if (!workers.empty()) {
            {
                std::lock_guard<std::mutex> lk(m);
                stopping = true;
                ++epoch;
            }
            cv.notify_all();
            for (auto& w : workers) w.join();
        }
    }
    int lanes() const { return T; }

    void run(const std::function<void(int)>& f) {
        if (T <= 1) {
            f(0);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m);
            job = &f;
            done = 0;
            ++epoch;
        }
        cv.notify_all();
        f(0);
        std::unique_lock<std::mutex> lk(m);
        cv_done.wait(lk, [this] { return done == T - 1; });
        job = nullptr;
    }

private:
    void worker(int tid) {
        long seen = 0;
        for (;;) {
            const std::function<void(int)>* f = nullptr;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return epoch != seen; });
                seen = epoch;
                if (stopping) return;
                f = job;
            }
            (*f)(tid);
            {
                std::lock_guard<std::mutex> lk(m);
                ++done;
            }
            cv_done.notify_one();
        }
    }

    std::vector<std::thread> workers;
    std::mutex m;
    std::condition_variable cv, cv_done;
    const std::function<void(int)>* job = nullptr;
    long epoch = 0;
    int done = 0, T = 1;
    bool stopping = false;
};

struct fft1 {
    bfft_plan* plan = nullptr;
    std::size_t n = 0, bins = 0;
    std::vector<double> work;
    std::vector<bfft_complex> scratch;

    bfft_status init(std::size_t size) {
        n = size;
        bfft_status st = bfft_plan_create(size, &plan);
        if (st != BFFT_OK) return st;
        bins = bfft_plan_bins(plan);
        work.resize(bfft_plan_work_size(plan));
        scratch.resize(bfft_plan_native_scratch_size(plan));
        return BFFT_OK;
    }
    ~fft1() { bfft_plan_destroy(plan); }

    void fwd(const double* in, bfft_complex* out) {
        bfft_forward(plan, in, out, work.data(), scratch.data());
    }
    void inv(const bfft_complex* in, double* out) {
        bfft_inverse(plan, in, out);
    }
};

// Spectrum of a real H x W image, k-major: plane a = column spectra of the
// row-rfft Re plane, plane b likewise for Im; a[2*(k*HB + m)] is the real
// part of column-frequency m at row-frequency k.
struct spectrum {
    std::vector<double> a, b;   // each 2*WB*HB doubles
    void alloc(std::size_t hb, std::size_t wb) {
        a.assign(2 * wb * hb, 0.0);
        b.assign(2 * wb * hb, 0.0);
    }
    void zero() {
        std::memset(a.data(), 0, a.size() * sizeof(double));
        std::memset(b.data(), 0, b.size() * sizeof(double));
    }
};

// per-lane transform state: own plans, work buffers, stage, line
struct lane {
    fft1 row, col;
    std::vector<bfft_complex> stage;   // PANEL * WB
    std::vector<double> line;          // W

    bfft_status init(std::size_t H, std::size_t W, std::size_t WB) {
        bfft_status st = row.init(W);
        if (st != BFFT_OK) return st;
        st = col.init(H);
        if (st != BFFT_OK) return st;
        stage.assign(PANEL * WB, bfft_complex{0.0, 0.0});
        line.assign(W, 0.0);
        return BFFT_OK;
    }
};

struct engine {
    std::size_t H = 0, W = 0, HB = 0, WB = 0;
    double lam = 0.05, mu = 40.0, rung_tol = 1e-5;
    int passes = 64, rung_sweeps = 600;

    pool P;
    std::vector<std::unique_ptr<lane>> lanes;

    // symbol tables 1/(c - eta*lap_hat), expanded to interleaved (re, im)
    // stride: s[2*(k*HB+m)] == s[2*(k*HB+m)+1], one per (c, eta) pair
    std::vector<double> s_u, s_v, s_r0, s_r1, s_r2;

    // spatial planes, H*W
    std::vector<double> u, w, xit;             // xit = generic ROF iterate
    std::vector<double> bux, buy, dbux, dbuy;  // u-solver Bregman state
    std::vector<double> bvx, bvy, dbvx, dbvy;  // v-solver Bregman state
    std::vector<double> rbx, rby, rdbx, rdby;  // rung solver (reused)
    std::vector<double> vplane, prev;

    // column-major stage planes for the 2-D transforms, WB*H each
    std::vector<double> reT, imT;

    spectrum f_spec, u_spec, w_spec, d_spec, v_spec;

    bfft_status init(std::size_t h, std::size_t wdt, double lam_, double mu_,
                     int passes_, int rung_sweeps_, double rung_tol_,
                     int threads) {
        H = h; W = wdt; HB = H / 2 + 1; WB = W / 2 + 1;
        lam = lam_; mu = mu_; passes = passes_;
        rung_sweeps = rung_sweeps_; rung_tol = rung_tol_;

        if (threads < 1) {
            const unsigned hw = std::thread::hardware_concurrency();
            threads = hw >= 4 ? 4 : (hw ? int(hw) : 1);
        }
        // no more lanes than row panels
        const std::size_t max_lanes = H / PANEL;
        if (std::size_t(threads) > max_lanes) threads = int(max_lanes);
        P.start(threads);
        lanes.clear();
        for (int t = 0; t < P.lanes(); ++t) {
            lanes.emplace_back(new lane());
            bfft_status st = lanes.back()->init(H, W, WB);
            if (st != BFFT_OK) return st;
        }

        symbol(s_u, lam, 2.0 * lam);
        symbol(s_v, 1.0 / mu, 10.0 / mu);
        const double m0 = mu, m1 = mu / 4.0, m2 = mu / 16.0;
        symbol(s_r0, 1.0 / m0, 10.0 / m0);
        symbol(s_r1, 1.0 / m1, 10.0 / m1);
        symbol(s_r2, 1.0 / m2, 10.0 / m2);

        const std::size_t n = H * W;
        for (auto* p : {&u, &w, &xit, &bux, &buy, &dbux, &dbuy, &bvx, &bvy,
                        &dbvx, &dbvy, &rbx, &rby, &rdbx, &rdby, &vplane,
                        &prev})
            p->assign(n, 0.0);
        reT.assign(WB * H, 0.0);
        imT.assign(WB * H, 0.0);
        for (auto* s : {&f_spec, &u_spec, &w_spec, &d_spec, &v_spec})
            s->alloc(HB, WB);
        return BFFT_OK;
    }

    void symbol(std::vector<double>& s, double c, double eta) {
        s.resize(2 * WB * HB);
        const double tau_h = 2.0 * M_PI / double(H);
        const double tau_w = 2.0 * M_PI / double(W);
        for (std::size_t k = 0; k < WB; ++k) {
            const double lx = 2.0 * std::cos(tau_w * double(k)) - 2.0;
            double* srow = s.data() + 2 * k * HB;
            for (std::size_t m = 0; m < HB; ++m) {
                const double ly = 2.0 * std::cos(tau_h * double(m)) - 2.0;
                const double val = 1.0 / (c - eta * (ly + lx));
                srow[2 * m] = val;
                srow[2 * m + 1] = val;
            }
        }
    }

    // ---- 2-D transforms, panel row stage + zero-copy column stage -------

    // stage panel (PANEL rows of row-spectra) -> column-major planes
    void panel_scatter(const lane& L, std::size_t i0) {
        const bfft_complex* __restrict st = L.stage.data();
        double* __restrict re = reT.data();
        double* __restrict im = imT.data();
        for (std::size_t k = 0; k < WB; ++k) {
            double* __restrict rk = re + k * H + i0;
            double* __restrict ik = im + k * H + i0;
            for (std::size_t r = 0; r < PANEL; ++r) {
                rk[r] = st[r * WB + k].re;
                ik[r] = st[r * WB + k].im;
            }
        }
    }

    // column-major planes -> stage panel of complex rows
    void panel_gather(lane& L, std::size_t i0) {
        bfft_complex* __restrict st = L.stage.data();
        const double* __restrict re = reT.data();
        const double* __restrict im = imT.data();
        for (std::size_t k = 0; k < WB; ++k) {
            const double* __restrict rk = re + k * H + i0;
            const double* __restrict ik = im + k * H + i0;
            for (std::size_t r = 0; r < PANEL; ++r) {
                st[r * WB + k].re = rk[r];
                st[r * WB + k].im = ik[r];
            }
        }
    }

    void cols_fwd(spectrum& spec) {
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            for (std::size_t k = tid; k < WB; k += P.lanes()) {
                L.col.fwd(reT.data() + k * H,
                          reinterpret_cast<bfft_complex*>(
                              spec.a.data() + 2 * k * HB));
                L.col.fwd(imT.data() + k * H,
                          reinterpret_cast<bfft_complex*>(
                              spec.b.data() + 2 * k * HB));
            }
        });
    }

    void cols_inv(const spectrum& spec) {
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            for (std::size_t k = tid; k < WB; k += P.lanes()) {
                L.col.inv(reinterpret_cast<const bfft_complex*>(
                              spec.a.data() + 2 * k * HB),
                          reT.data() + k * H);
                L.col.inv(reinterpret_cast<const bfft_complex*>(
                              spec.b.data() + 2 * k * HB),
                          imT.data() + k * H);
            }
        });
    }

    void fwd2d(const double* x, spectrum& spec) {
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            for (std::size_t i0 = tid * PANEL; i0 < H;
                 i0 += P.lanes() * PANEL) {
                for (std::size_t r = 0; r < PANEL; ++r)
                    L.row.fwd(x + (i0 + r) * W, L.stage.data() + r * WB);
                panel_scatter(L, i0);
            }
        });
        cols_fwd(spec);
    }

    void inv2d(const spectrum& spec, double* x) {
        cols_inv(spec);
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            for (std::size_t i0 = tid * PANEL; i0 < H;
                 i0 += P.lanes() * PANEL) {
                panel_gather(L, i0);
                for (std::size_t r = 0; r < PANEL; ++r)
                    L.row.inv(L.stage.data() + r * WB, x + (i0 + r) * W);
            }
        });
    }

    // rhs = div(db), computed row-by-row into the lane's line buffer and
    // row-transformed: no full-plane rhs buffer exists.
    void fwd2d_div(const std::vector<double>& dbx,
                   const std::vector<double>& dby, spectrum& spec) {
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            double* __restrict ln = L.line.data();
            for (std::size_t i0 = tid * PANEL; i0 < H;
                 i0 += P.lanes() * PANEL) {
                for (std::size_t r = 0; r < PANEL; ++r) {
                    const std::size_t i = i0 + r;
                    const double* __restrict px = dbx.data() + i * W;
                    const double* __restrict py = dby.data() + i * W;
                    const double* __restrict pyn =
                        dby.data() + ((i == 0 ? H : i) - 1) * W;
                    ln[0] = px[0] - px[W - 1] + py[0] - pyn[0];
                    for (std::size_t j = 1; j < W; ++j)
                        ln[j] = px[j] - px[j - 1] + py[j] - pyn[j];
                    L.row.fwd(ln, L.stage.data() + r * WB);
                }
                panel_scatter(L, i0);
            }
        });
        cols_fwd(spec);
    }

    // ---- fused spatial shrink (rows partition across lanes) --------------
    // t = grad(x) + b;  coef = max(|t|-1/eta,0)/max(|t|,eps);
    // b <- (1-coef)*t;  db <- (2*coef-1)*t.   Wrap column peeled.
    void shrink(const std::vector<double>& x, std::vector<double>& bx,
                std::vector<double>& by, std::vector<double>& dbx,
                std::vector<double>& dby, double eta) {
        const double thr = 1.0 / eta;
        P.run([&](int tid) {
            for (std::size_t i = tid; i < H; i += P.lanes()) {
                const double* __restrict xi = x.data() + i * W;
                const double* __restrict xn =
                    x.data() + ((i + 1 == H) ? 0 : i + 1) * W;
                double* __restrict pbx = bx.data() + i * W;
                double* __restrict pby = by.data() + i * W;
                double* __restrict pdx = dbx.data() + i * W;
                double* __restrict pdy = dby.data() + i * W;
                for (std::size_t j = 0; j < W - 1; ++j) {
                    const double tx = xi[j + 1] - xi[j] + pbx[j];
                    const double ty = xn[j] - xi[j] + pby[j];
                    const double r = std::sqrt(tx * tx + ty * ty);
                    const double coef =
                        std::fmax(r - thr, 0.0) / std::fmax(r, 1e-12);
                    pbx[j] = (1.0 - coef) * tx;
                    pby[j] = (1.0 - coef) * ty;
                    pdx[j] = (2.0 * coef - 1.0) * tx;
                    pdy[j] = (2.0 * coef - 1.0) * ty;
                }
                {
                    const std::size_t j = W - 1;
                    const double tx = xi[0] - xi[j] + pbx[j];
                    const double ty = xn[j] - xi[j] + pby[j];
                    const double r = std::sqrt(tx * tx + ty * ty);
                    const double coef =
                        std::fmax(r - thr, 0.0) / std::fmax(r, 1e-12);
                    pbx[j] = (1.0 - coef) * tx;
                    pby[j] = (1.0 - coef) * ty;
                    pdx[j] = (2.0 * coef - 1.0) * tx;
                    pdy[j] = (2.0 * coef - 1.0) * ty;
                }
            }
        });
    }

    // ---- spectral solves: concrete unit-stride streams, range-split ------

    std::size_t n2() const { return 2 * WB * HB; }

    void split(int tid, std::size_t n, std::size_t& lo, std::size_t& hi) {
        const std::size_t T = std::size_t(P.lanes());
        const std::size_t chunk = ((n / 2) / T) * 2;   // even split
        lo = tid * chunk;
        hi = (tid == int(T) - 1) ? n : lo + chunk;
    }

    // out = c * src * s          (first sweep: Bregman state is zero)
    void solve_scale(const double* srcA, const double* srcB, double c,
                     const double* s, double* outA, double* outB) {
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict pa = srcA;
            const double* __restrict pb = srcB;
            const double* __restrict ps = s;
            double* __restrict oa = outA;
            double* __restrict ob = outB;
            for (std::size_t r = lo; r < hi; ++r) oa[r] = c * pa[r] * ps[r];
            for (std::size_t r = lo; r < hi; ++r) ob[r] = c * pb[r] * ps[r];
        });
    }

    // io = (c * (io + q) - eta * d) * s     (u-step: g = u + w, in place)
    void solve_sum_inplace(double* ioA, double* ioB, const double* qA,
                           const double* qB, const double* dA,
                           const double* dB, double c, double eta,
                           const double* s) {
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict qa = qA;
            const double* __restrict qb = qB;
            const double* __restrict da = dA;
            const double* __restrict db = dB;
            const double* __restrict ps = s;
            double* ia = ioA;
            double* ib = ioB;
            for (std::size_t r = lo; r < hi; ++r)
                ia[r] = (c * (ia[r] + qa[r]) - eta * da[r]) * ps[r];
            for (std::size_t r = lo; r < hi; ++r)
                ib[r] = (c * (ib[r] + qb[r]) - eta * db[r]) * ps[r];
        });
    }

    // out = c * (p - q) * s      (v-step first sweep: g = f - u)
    void solve_diff_scale(const double* pA, const double* pB,
                          const double* qA, const double* qB, double c,
                          const double* s, double* outA, double* outB) {
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict pa = pA;
            const double* __restrict pb = pB;
            const double* __restrict qa = qA;
            const double* __restrict qb = qB;
            const double* __restrict ps = s;
            double* __restrict oa = outA;
            double* __restrict ob = outB;
            for (std::size_t r = lo; r < hi; ++r)
                oa[r] = c * (pa[r] - qa[r]) * ps[r];
            for (std::size_t r = lo; r < hi; ++r)
                ob[r] = c * (pb[r] - qb[r]) * ps[r];
        });
    }

    // out = (c * (p - q) - eta * d) * s      (v-step: g = f - u)
    void solve_diff(const double* pA, const double* pB, const double* qA,
                    const double* qB, const double* dA, const double* dB,
                    double c, double eta, const double* s, double* outA,
                    double* outB) {
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict pa = pA;
            const double* __restrict pb = pB;
            const double* __restrict qa = qA;
            const double* __restrict qb = qB;
            const double* __restrict da = dA;
            const double* __restrict db = dB;
            const double* __restrict ps = s;
            double* __restrict oa = outA;
            double* __restrict ob = outB;
            for (std::size_t r = lo; r < hi; ++r)
                oa[r] = (c * (pa[r] - qa[r]) - eta * da[r]) * ps[r];
            for (std::size_t r = lo; r < hi; ++r)
                ob[r] = (c * (pb[r] - qb[r]) - eta * db[r]) * ps[r];
        });
    }

    // out = (c * g - eta * d) * s            (rung sweeps: g fixed)
    void solve_g(const double* gA, const double* gB, const double* dA,
                 const double* dB, double c, double eta, const double* s,
                 double* outA, double* outB) {
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict ga = gA;
            const double* __restrict gb = gB;
            const double* __restrict da = dA;
            const double* __restrict db = dB;
            const double* __restrict ps = s;
            double* __restrict oa = outA;
            double* __restrict ob = outB;
            for (std::size_t r = lo; r < hi; ++r)
                oa[r] = (c * ga[r] - eta * da[r]) * ps[r];
            for (std::size_t r = lo; r < hi; ++r)
                ob[r] = (c * gb[r] - eta * db[r]) * ps[r];
        });
    }

    // ---- the alternation (shared by split and decompose) -----------------
    //
    // Runs the TGFD passes and leaves u (cartoon layer), w (texture-side
    // ROF survivor), vplane = f - u - w (texture layer), and the spectra
    // of f, u, w in place.

    void run_passes(const double* image) {
        const std::size_t n = H * W;
        for (auto* p : {&u, &w, &bux, &buy, &dbux, &dbuy, &bvx, &bvy, &dbvx,
                        &dbvy})
            std::memset(p->data(), 0, n * sizeof(double));
        u_spec.zero();
        w_spec.zero();

        fwd2d(image, f_spec);

        const double c_u = lam, eta_u = 2.0 * lam;
        const double c_v = 1.0 / mu, eta_v = 10.0 / mu;

        for (int p = 0; p < passes; ++p) {
            // u-step: g = f - v = u + w (both spectra maintained)
            if (p == 0) {
                solve_scale(f_spec.a.data(), f_spec.b.data(), c_u,
                            s_u.data(), u_spec.a.data(), u_spec.b.data());
            } else {
                fwd2d_div(dbux, dbuy, d_spec);
                solve_sum_inplace(u_spec.a.data(), u_spec.b.data(),
                                  w_spec.a.data(), w_spec.b.data(),
                                  d_spec.a.data(), d_spec.b.data(), c_u,
                                  eta_u, s_u.data());
            }
            inv2d(u_spec, u.data());
            shrink(u, bux, buy, dbux, dbuy, eta_u);

            // v-step: g = f - u
            if (p == 0) {
                solve_diff_scale(f_spec.a.data(), f_spec.b.data(),
                                 u_spec.a.data(), u_spec.b.data(), c_v,
                                 s_v.data(), w_spec.a.data(),
                                 w_spec.b.data());
            } else {
                fwd2d_div(dbvx, dbvy, d_spec);
                solve_diff(f_spec.a.data(), f_spec.b.data(),
                           u_spec.a.data(), u_spec.b.data(),
                           d_spec.a.data(), d_spec.b.data(), c_v, eta_v,
                           s_v.data(), w_spec.a.data(), w_spec.b.data());
            }
            inv2d(w_spec, w.data());
            shrink(w, bvx, bvy, dbvx, dbvy, eta_v);
        }

        // v = f - u - w, single fused pass
        {
            double* __restrict vp = vplane.data();
            const double* __restrict fp = image;
            const double* __restrict up = u.data();
            const double* __restrict wp = w.data();
            for (std::size_t i = 0; i < n; ++i)
                vp[i] = fp[i] - up[i] - wp[i];
        }
    }

    // ---- split: the model decomposition alone, no ladder ----------------
    //
    // cartoon = u, texture = v: exactly the pair Gilles' Algorithm 3
    // produces.  (The five-output decompose() instead reports
    // cartoon = u + s0, folding the ladder's coarsest survivor back into
    // the cartoon so that cartoon + bands = u + v.)

    void split(const double* image, double* cartoon, double* texture) {
        run_passes(image);
        const std::size_t n = H * W;
        std::memcpy(cartoon, u.data(), n * sizeof(double));
        std::memcpy(texture, vplane.data(), n * sizeof(double));
    }

    // ---- the full decomposition -----------------------------------------

    void decompose(const double* image, double* cartoon, double* texture,
                   double* band_coarse, double* band_mid, double* band_fine) {
        const std::size_t n = H * W;
        run_passes(image);

        // g for every rung is v: spectrum by linear combination, no
        // transform
        {
            double* __restrict va = v_spec.a.data();
            double* __restrict vb = v_spec.b.data();
            const double* __restrict fa = f_spec.a.data();
            const double* __restrict fb = f_spec.b.data();
            const double* __restrict ua = u_spec.a.data();
            const double* __restrict ub = u_spec.b.data();
            const double* __restrict wa = w_spec.a.data();
            const double* __restrict wb = w_spec.b.data();
            const std::size_t m = n2();
            for (std::size_t r = 0; r < m; ++r) {
                va[r] = fa[r] - ua[r] - wa[r];
                vb[r] = fb[r] - ub[r] - wb[r];
            }
        }

        // ladder rungs: independent solves, fresh states, coarse -> fine
        // (u_spec is reused as spectral scratch; u spatial stays intact)
        const double rung_mu[3] = {mu, mu / 4.0, mu / 16.0};
        const std::vector<double>* rung_s[3] = {&s_r0, &s_r1, &s_r2};
        double* rung_out[3] = {cartoon, band_mid, band_fine};  // staging
        for (int rr = 0; rr < 3; ++rr) {
            const double c = 1.0 / rung_mu[rr], eta = 10.0 / rung_mu[rr];
            std::memset(rbx.data(), 0, n * sizeof(double));
            std::memset(rby.data(), 0, n * sizeof(double));
            std::memcpy(prev.data(), vplane.data(), n * sizeof(double));
            bool done = false;
            for (int sweep = 0; sweep < rung_sweeps && !done; ++sweep) {
                if (sweep == 0) {
                    solve_scale(v_spec.a.data(), v_spec.b.data(), c,
                                rung_s[rr]->data(), u_spec.a.data(),
                                u_spec.b.data());
                } else {
                    fwd2d_div(rdbx, rdby, d_spec);
                    solve_g(v_spec.a.data(), v_spec.b.data(),
                            d_spec.a.data(), d_spec.b.data(), c, eta,
                            rung_s[rr]->data(), u_spec.a.data(),
                            u_spec.b.data());
                }
                inv2d(u_spec, xit.data());
                shrink(xit, rbx, rby, rdbx, rdby, eta);
                if (rung_tol > 0.0) {
                    // SERIAL by design: bit-identical for all thread counts
                    const double* __restrict xi = xit.data();
                    double* __restrict pv = prev.data();
                    double d0 = 0, d1 = 0, x0 = 0, x1 = 0;
                    for (std::size_t i = 0; i < n; i += 2) {
                        const double a0 = xi[i] - pv[i];
                        const double a1 = xi[i + 1] - pv[i + 1];
                        d0 += a0 * a0;
                        d1 += a1 * a1;
                        x0 += xi[i] * xi[i];
                        x1 += xi[i + 1] * xi[i + 1];
                        pv[i] = xi[i];
                        pv[i + 1] = xi[i + 1];
                    }
                    done = (d0 + d1) <= rung_tol * rung_tol * (x0 + x1);
                }
            }
            std::memcpy(rung_out[rr], xit.data(), n * sizeof(double));
        }

        // assemble outputs: cartoon/band_mid/band_fine currently hold
        // s0/s1/s2 and are rewritten in place (per-index reads complete
        // before the overwrites of the same slots), so those pointers
        // deliberately carry no restrict qualifier
        {
            const double* __restrict vp = vplane.data();
            const double* __restrict up = u.data();
            for (std::size_t i = 0; i < n; ++i) {
                const double a0 = cartoon[i], a1 = band_mid[i],
                             a2 = band_fine[i];
                texture[i] = vp[i];
                cartoon[i] = up[i] + a0;
                band_mid[i] = a2 - a1;
                band_fine[i] = vp[i] - a2;
                band_coarse[i] = a1 - a0;
            }
        }
    }
};

}  // namespace meyer
