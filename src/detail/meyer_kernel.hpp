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
// SHRINK ALGEBRA.  The full ladder path uses the fused update
//
//     b_new  = (1 - coef) * t
//     d - b_new = (2*coef - 1) * t      (the only combination ever used)
//
// The split-only API closes the same recursion on t itself:
//
//     p = t - 2*proj(t);  u <- solve(c*g - eta*div(p))
//     t <- grad(u) + proj(t)
//
// It carries two planes per subproblem instead of b plus materialized db.
// The scalar divergence is streamed into the soon-to-be-overwritten iterate
// plane, so the reflected vector field never lands in memory.
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

#include <algorithm>
#include <cmath>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace meyer {

using trace_visitor = void (*)(int pass, const double* cartoon,
                               const double* texture, std::size_t count,
                               void* user);

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

    template <typename Function>
    void run(Function&& f) {
        if (T <= 1) {
            f(0);
            return;
        }
        using function_type = std::remove_reference_t<Function>;
        {
            std::lock_guard<std::mutex> lk(m);
            job_context = std::addressof(f);
            job_invoke = &invoke<function_type>;
            done = 0;
            ++epoch;
        }
        cv.notify_all();
        f(0);
        std::unique_lock<std::mutex> lk(m);
        cv_done.wait(lk, [this] { return done == T - 1; });
        job_context = nullptr;
        job_invoke = nullptr;
    }

private:
    template <typename Function>
    static void invoke(void* context, int tid) {
        (*static_cast<Function*>(context))(tid);
    }

    void worker(int tid) {
        long seen = 0;
        for (;;) {
            void* context = nullptr;
            void (*call)(void*, int) = nullptr;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return epoch != seen; });
                seen = epoch;
                if (stopping) return;
                context = job_context;
                call = job_invoke;
            }
            call(context, tid);
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
    void* job_context = nullptr;
    void (*job_invoke)(void*, int) = nullptr;
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

// One-axis real spectra for FACR.  The swept coordinate is unit-stride
// within each transformed-axis bin, independently for real and imaginary
// parts.  This makes both Thomas passes streaming.
struct facr_spectrum {
    std::vector<double> a, b;  // each bins * swept
    void alloc(std::size_t swept, std::size_t bins) {
        a.assign(swept * bins, 0.0);
        b.assign(swept * bins, 0.0);
    }
    void zero() {
        std::memset(a.data(), 0, a.size() * sizeof(double));
        std::memset(b.data(), 0, b.size() * sizeof(double));
    }
};

struct tri_factors {
    double c = 0.0, eta = 0.0;
    std::vector<double> pivot;  // [k * swept + s]
    std::vector<double> diagonal;  // unmodified periodic diagonal per bin
};

// per-lane transform state: own plans, work buffers, stage, line
struct lane {
    fft1 row, col;
    std::vector<bfft_complex> stage;   // PANEL * WB
    std::vector<double> line;          // max(H, W)
    std::vector<double> reflect_x, reflect_y;
    std::vector<double> correction;    // swept-axis Sherman-Morrison vector

    bfft_status init(std::size_t H, std::size_t W, std::size_t WB) {
        bfft_status st = BFFT_OK;
        if (W >= 8 && (W & (W - 1)) == 0) {
            st = row.init(W);
            if (st != BFFT_OK) return st;
        }
        if (H >= 8 && (H & (H - 1)) == 0) {
            st = col.init(H);
            if (st != BFFT_OK) return st;
        }
        const std::size_t max_bins = std::max(H / 2 + 1, WB);
        stage.assign(PANEL * max_bins, bfft_complex{0.0, 0.0});
        line.assign(std::max(H, W), 0.0);
        reflect_x.assign(std::max(H, W), 0.0);
        reflect_y.assign(std::max(H, W), 0.0);
        correction.assign(std::max(H, W), 0.0);
        return BFFT_OK;
    }
};

struct engine {
    std::size_t H = 0, W = 0, HB = 0, WB = 0;
    double lam = 0.05, mu = 40.0, rung_tol = 1e-5;
    int passes = 64, rung_sweeps = 600;
    int solver = 0;
    bool facr_active = false;
    bool sweep_height = true;
    std::size_t FS = 0, FT = 0, FB = 0;

    pool P;
    std::vector<std::unique_ptr<lane>> lanes;

    // symbol tables 1/(c - eta*lap_hat), expanded to interleaved (re, im)
    // stride: s[2*(k*HB+m)] == s[2*(k*HB+m)+1], one per (c, eta) pair
    std::vector<double> s_u, s_v, s_r0, s_r1, s_r2;
    // table for the general ROF entry point, rebuilt only when (c, eta)
    // changes
    std::vector<double> s_gen;
    double gen_c = 0.0, gen_eta = 0.0;

    tri_factors t_u, t_v, t_r0, t_r1, t_r2, t_gen;

    // spatial planes, H*W
    std::vector<double> u, w, xit;             // xit = generic ROF iterate
    // bu*/bv* are the reduced t fields for split, and are reused as the
    // conventional b fields if the lazy ladder path is requested.
    std::vector<double> bux, buy, dbux, dbuy;
    std::vector<double> bvx, bvy, dbvx, dbvy;
    std::vector<double> rbx, rby, rdbx, rdby;  // rung solver (reused)
    std::vector<double> vplane, prev;

    // column-major stage planes for the 2-D transforms, WB*H each
    std::vector<double> reT, imT;

    spectrum f_spec, u_spec, w_spec, d_spec, v_spec;
    facr_spectrum ff_spec, fu_spec, fw_spec, fd_spec, fv_spec;

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
        // no more lanes than transform-line panels
        const std::size_t max_lanes =
            std::max<std::size_t>(1, std::max(H, W) / PANEL);
        if (std::size_t(threads) > max_lanes) threads = int(max_lanes);
        P.start(threads);
        lanes.clear();
        for (int t = 0; t < P.lanes(); ++t) {
            lanes.emplace_back(new lane());
            bfft_status st = lanes.back()->init(H, W, WB);
            if (st != BFFT_OK) return st;
        }

        const std::size_t n = H * W;
        // The split-only hot path owns just its two iterates and two
        // R^2-valued reflected-dual fields.  Ladder/ROF state is lazy.
        for (auto* p : {&u, &w, &bux, &buy, &bvx, &bvy})
            p->assign(n, 0.0);
        if (pow2_ge8(H) && pow2_ge8(W)) ensure_spectral_storage();
        return BFFT_OK;
    }

    static bool pow2_ge8(std::size_t n) {
        return n >= 8 && (n & (n - 1)) == 0;
    }

    void ensure_spectral_storage() {
        if (!s_u.empty()) return;
        symbol(s_u, lam, 2.0 * lam);
        symbol(s_v, 1.0 / mu, 10.0 / mu);
        reT.assign(WB * H, 0.0);
        imT.assign(WB * H, 0.0);
        for (auto* s : {&f_spec, &u_spec, &w_spec, &d_spec})
            s->alloc(HB, WB);
    }

    void ensure_decompose_storage() {
        const std::size_t n = H * W;
        for (auto* p : {&dbux, &dbuy, &dbvx, &dbvy, &vplane})
            if (p->empty()) p->assign(n, 0.0);
        ensure_rof_storage();
        if (facr_active) {
            if (fv_spec.a.empty()) fv_spec.alloc(FS, FB);
            if (t_r0.pivot.empty()) {
                const double m0 = mu, m1 = mu / 4.0, m2 = mu / 16.0;
                build_factors(t_r0, 1.0 / m0, 10.0 / m0);
                build_factors(t_r1, 1.0 / m1, 10.0 / m1);
                build_factors(t_r2, 1.0 / m2, 10.0 / m2);
            }
        } else {
            if (v_spec.a.empty()) v_spec.alloc(HB, WB);
            if (s_r0.empty()) {
                const double m0 = mu, m1 = mu / 4.0, m2 = mu / 16.0;
                symbol(s_r0, 1.0 / m0, 10.0 / m0);
                symbol(s_r1, 1.0 / m1, 10.0 / m1);
                symbol(s_r2, 1.0 / m2, 10.0 / m2);
            }
        }
    }

    void ensure_rof_storage() {
        const std::size_t n = H * W;
        for (auto* p : {&xit, &rbx, &rby, &rdbx, &rdby, &prev})
            if (p->empty()) p->assign(n, 0.0);
    }

    void ensure_visit_storage() {
        if (vplane.empty()) vplane.assign(H * W, 0.0);
    }

    void clear_spectral_storage() {
        for (auto* v : {&s_u, &s_v, &s_r0, &s_r1, &s_r2, &s_gen,
                        &reT, &imT})
            std::vector<double>().swap(*v);
        for (auto* s : {&f_spec, &u_spec, &w_spec, &d_spec, &v_spec}) {
            std::vector<double>().swap(s->a);
            std::vector<double>().swap(s->b);
        }
    }

    void clear_facr_storage() {
        for (auto* s : {&ff_spec, &fu_spec, &fw_spec, &fd_spec, &fv_spec}) {
            std::vector<double>().swap(s->a);
            std::vector<double>().swap(s->b);
        }
        for (auto* t : {&t_u, &t_v, &t_r0, &t_r1, &t_r2, &t_gen}) {
            std::vector<double>().swap(t->pivot);
            std::vector<double>().swap(t->diagonal);
        }
    }

    bool set_solver(int mode) {
        if (mode < 0 || mode > 2) return false;
        if (mode != 0 && !pow2_ge8(H) && !pow2_ge8(W)) return false;
        if (mode == 0 && (!pow2_ge8(H) || !pow2_ge8(W))) return false;
        solver = mode;
        facr_active = mode != 0 && (mode == 2 ||
            !pow2_ge8(H) || !pow2_ge8(W));
        if (!facr_active) {
            clear_facr_storage();
            ensure_spectral_storage();
            return true;
        }
        clear_spectral_storage();

        // Sweep the axis whose removal saves the most padding.  If only one
        // dimension is transformable, the choice is forced.
        if (!pow2_ge8(H)) {
            sweep_height = true;
        } else if (!pow2_ge8(W)) {
            sweep_height = false;
        } else {
            sweep_height = true;  // equal transformed area: deterministic tie
        }
        FS = sweep_height ? H : W;
        FT = sweep_height ? W : H;
        FB = FT / 2 + 1;

        for (auto* s : {&ff_spec, &fu_spec, &fw_spec, &fd_spec})
            s->alloc(FS, FB);
        build_factors(t_u, lam, 2.0 * lam);
        build_factors(t_v, 1.0 / mu, 10.0 / mu);
        t_gen.pivot.clear();
        return true;
    }

    void build_factors(tri_factors& t, double c, double eta) {
        t.c = c;
        t.eta = eta;
        t.pivot.resize(FB * FS);
        t.diagonal.resize(FB);
        const double tau = 2.0 * M_PI / double(FT);
        for (std::size_t k = 0; k < FB; ++k) {
            const double lt = 2.0 * std::cos(tau * double(k)) - 2.0;
            const double base = c - eta * lt;
            const double d = base + 2.0 * eta;
            t.diagonal[k] = d;
            assert(d > 2.0 * eta);
            double* p = t.pivot.data() + k * FS;
            if (solver == 2) {
                p[0] = 1.0 / (base + eta);
                for (std::size_t s = 1; s < FS; ++s) {
                    const double diag =
                        (s + 1 == FS) ? base + eta : d;
                    p[s] = 1.0 / (diag - eta * eta * p[s - 1]);
                }
            } else {
                // Cyclic Thomas via one Sherman-Morrison correction.
                const double gamma = -d;
                p[0] = 1.0 / (d - gamma);
                for (std::size_t s = 1; s < FS; ++s) {
                    const double diag = (s + 1 == FS)
                        ? d - eta * eta / gamma : d;
                    p[s] = 1.0 / (diag - eta * eta * p[s - 1]);
                }
            }
        }
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

    static void reflected(double tx, double ty, double threshold,
                          double& px, double& py) {
        const double radius = std::sqrt(tx * tx + ty * ty);
        const double shrink =
            std::fmax(radius - threshold, 0.0) /
            std::fmax(radius, 1e-12);
        const double scale = 2.0 * shrink - 1.0;
        px = scale * tx;
        py = scale * ty;
    }

    // Transform div(t - 2*proj(t)) without materializing the reflected
    // vector field.  This closes Split Bregman on one R^2 state.
    void fwd2d_reflection(const std::vector<double>& tx,
                          const std::vector<double>& ty, double eta,
                          spectrum& spec) {
        const double threshold = 1.0 / eta;
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            for (std::size_t i0 = std::size_t(tid) * PANEL; i0 < H;
                 i0 += std::size_t(P.lanes()) * PANEL) {
                for (std::size_t r = 0; r < PANEL; ++r) {
                    const std::size_t y = i0 + r;
                    const std::size_t yp = (y == 0 ? H : y) - 1;
                    for (std::size_t x = 0; x < W; ++x)
                        reflected(tx[y * W + x], ty[y * W + x], threshold,
                                  L.reflect_x[x], L.reflect_y[x]);
                    for (std::size_t x = 0; x < W; ++x) {
                        double unused, prev_y;
                        reflected(tx[yp * W + x], ty[yp * W + x],
                                  threshold, unused, prev_y);
                        const std::size_t xp = (x == 0 ? W : x) - 1;
                        L.line[x] = L.reflect_x[x] - L.reflect_x[xp] +
                            L.reflect_y[x] - prev_y;
                    }
                    L.row.fwd(L.line.data(), L.stage.data() + r * WB);
                }
                panel_scatter(L, i0);
            }
        });
        cols_fwd(spec);
    }

    // ---- one-axis transforms + FACR sweeps ------------------------------

    fft1& facr_fft(lane& L) { return sweep_height ? L.row : L.col; }

    template <bool Divergence>
    void facr_fwd_impl(const double* x, const std::vector<double>* dbx,
                       const std::vector<double>* dby, facr_spectrum& spec) {
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            fft1& F = facr_fft(L);
            for (std::size_t s0 = std::size_t(tid) * PANEL; s0 < FS;
                s0 += std::size_t(P.lanes()) * PANEL) {
                const std::size_t nr = std::min(PANEL, FS - s0);
                for (std::size_t r = 0; r < nr; ++r) {
                    const std::size_t s = s0 + r;
                    double* line = L.line.data();
                    const double* input = nullptr;
                    if constexpr (!Divergence) {
                        if (sweep_height) {
                            input = x + s * W;
                        } else {
                            for (std::size_t y = 0; y < H; ++y)
                                line[y] = x[y * W + s];
                            input = line;
                        }
                    } else {
                        const double* px = dbx->data();
                        const double* py = dby->data();
                        if (sweep_height) {
                            const std::size_t y = s;
                            const std::size_t yp = (y == 0 ? H : y) - 1;
                            const double* xr = px + y * W;
                            const double* yr = py + y * W;
                            const double* yprev = py + yp * W;
                            line[0] = xr[0] - xr[W - 1] + yr[0] -
                                (solver == 2 && y == 0 ? 0.0 : yprev[0]);
                            for (std::size_t j = 1; j < W; ++j)
                                line[j] = xr[j] - xr[j - 1] + yr[j] -
                                    (solver == 2 && y == 0
                                         ? 0.0 : yprev[j]);
                        } else {
                            const std::size_t col = s;
                            const std::size_t xp =
                                (col == 0 ? W : col) - 1;
                            for (std::size_t y = 0; y < H; ++y) {
                                const std::size_t yp =
                                    (y == 0 ? H : y) - 1;
                                const std::size_t i = y * W + col;
                                line[y] = px[i] - px[y * W + xp] + py[i] -
                                    py[yp * W + col];
                                if (solver == 2 && col == 0)
                                    line[y] += px[y * W + xp];
                            }
                        }
                        input = line;
                    }
                    F.fwd(input, L.stage.data() + r * FB);
                }
                for (std::size_t k = 0; k < FB; ++k) {
                    double* ar = spec.a.data() + k * FS + s0;
                    double* br = spec.b.data() + k * FS + s0;
                    for (std::size_t r = 0; r < nr; ++r) {
                        const bfft_complex z = L.stage[r * FB + k];
                        ar[r] = z.re;
                        br[r] = z.im;
                    }
                }
            }
        });
    }

    void facr_fwd(const double* x, facr_spectrum& spec) {
        facr_fwd_impl<false>(x, nullptr, nullptr, spec);
    }

    void facr_fwd_div(const std::vector<double>& dbx,
                      const std::vector<double>& dby, facr_spectrum& spec) {
        facr_fwd_impl<true>(nullptr, &dbx, &dby, spec);
    }

    void facr_fwd_reflection(const std::vector<double>& tx,
                             const std::vector<double>& ty, double eta,
                             facr_spectrum& spec) {
        const double threshold = 1.0 / eta;
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            fft1& F = facr_fft(L);
            for (std::size_t s0 = std::size_t(tid) * PANEL; s0 < FS;
                 s0 += std::size_t(P.lanes()) * PANEL) {
                const std::size_t nr = std::min(PANEL, FS - s0);
                for (std::size_t r = 0; r < nr; ++r) {
                    const std::size_t s = s0 + r;
                    if (sweep_height) {
                        const std::size_t y = s;
                        const std::size_t yp = (y == 0 ? H : y) - 1;
                        for (std::size_t x = 0; x < W; ++x)
                            reflected(tx[y * W + x], ty[y * W + x],
                                      threshold, L.reflect_x[x],
                                      L.reflect_y[x]);
                        for (std::size_t x = 0; x < W; ++x) {
                            double unused, prev_y = 0.0;
                            if (!(solver == 2 && y == 0))
                                reflected(tx[yp * W + x], ty[yp * W + x],
                                          threshold, unused, prev_y);
                            const std::size_t xp = (x == 0 ? W : x) - 1;
                            L.line[x] =
                                L.reflect_x[x] - L.reflect_x[xp] +
                                L.reflect_y[x] - prev_y;
                        }
                    } else {
                        const std::size_t x = s;
                        const std::size_t xp = (x == 0 ? W : x) - 1;
                        for (std::size_t y = 0; y < H; ++y)
                            reflected(tx[y * W + x], ty[y * W + x],
                                      threshold, L.reflect_x[y],
                                      L.reflect_y[y]);
                        for (std::size_t y = 0; y < H; ++y) {
                            const std::size_t yp = (y == 0 ? H : y) - 1;
                            double prev_x = 0.0, unused;
                            if (!(solver == 2 && x == 0))
                                reflected(tx[y * W + xp], ty[y * W + xp],
                                          threshold, prev_x, unused);
                            double prev_y;
                            reflected(tx[yp * W + x], ty[yp * W + x],
                                      threshold, unused, prev_y);
                            L.line[y] =
                                L.reflect_x[y] - prev_x +
                                L.reflect_y[y] - prev_y;
                        }
                    }
                    F.fwd(L.line.data(), L.stage.data() + r * FB);
                }
                for (std::size_t k = 0; k < FB; ++k) {
                    double* ar = spec.a.data() + k * FS + s0;
                    double* br = spec.b.data() + k * FS + s0;
                    for (std::size_t r = 0; r < nr; ++r) {
                        const bfft_complex z = L.stage[r * FB + k];
                        ar[r] = z.re;
                        br[r] = z.im;
                    }
                }
            }
        });
    }

    void facr_inv(const facr_spectrum& spec, double* x) {
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            fft1& F = facr_fft(L);
            for (std::size_t s0 = std::size_t(tid) * PANEL; s0 < FS;
                 s0 += std::size_t(P.lanes()) * PANEL) {
                const std::size_t nr = std::min(PANEL, FS - s0);
                for (std::size_t k = 0; k < FB; ++k) {
                    const double* ar = spec.a.data() + k * FS + s0;
                    const double* br = spec.b.data() + k * FS + s0;
                    for (std::size_t r = 0; r < nr; ++r) {
                        L.stage[r * FB + k].re = ar[r];
                        L.stage[r * FB + k].im = br[r];
                    }
                }
                for (std::size_t r = 0; r < nr; ++r) {
                    const std::size_t s = s0 + r;
                    if (sweep_height) {
                        F.inv(L.stage.data() + r * FB, x + s * W);
                    } else {
                        F.inv(L.stage.data() + r * FB, L.line.data());
                        for (std::size_t y = 0; y < H; ++y)
                            x[y * W + s] = L.line[y];
                    }
                }
            }
        });
    }

    template <typename RhsA, typename RhsB>
    void facr_solve_pair(lane& L, std::size_t k, facr_spectrum& out,
                         const tri_factors& t, RhsA&& ra, RhsB&& rb) {
                double* va = out.a.data() + k * FS;
                double* vb = out.b.data() + k * FS;
                const double* p = t.pivot.data() + k * FS;
                if (solver == 1 && FS == 2) {
                    // The two periodic neighbors coincide at length two,
                    // so their couplings add instead of forming distinct
                    // tridiagonal and corner entries.
                    const double d = t.diagonal[k];
                    const double off = 2.0 * t.eta;
                    const double inv_det = 1.0 / (d * d - off * off);
                    const std::size_t i = k * FS;
                    const double ar0 = ra(i), ar1 = ra(i + 1);
                    const double br0 = rb(i), br1 = rb(i + 1);
                    va[0] = (d * ar0 + off * ar1) * inv_det;
                    va[1] = (off * ar0 + d * ar1) * inv_det;
                    vb[0] = (d * br0 + off * br1) * inv_det;
                    vb[1] = (off * br0 + d * br1) * inv_det;
                    return;
                }
                va[0] = ra(k * FS) * p[0];
                vb[0] = rb(k * FS) * p[0];
                for (std::size_t s = 1; s < FS; ++s) {
                    const double ps = p[s];
                    va[s] = (ra(k * FS + s) + t.eta * va[s - 1]) * ps;
                    vb[s] = (rb(k * FS + s) + t.eta * vb[s - 1]) * ps;
                }
                for (std::size_t s = FS - 1; s-- > 0;) {
                    const double ep = t.eta * p[s];
                    va[s] += ep * va[s + 1];
                    vb[s] += ep * vb[s + 1];
                }

                if (solver == 1) {
                    // Recompute the fixed correction vector in lane-local
                    // storage.  This avoids an image-sized table per
                    // (c,eta) while keeping its hot working set in cache.
                    double* z = L.correction.data();
                    const double d = t.diagonal[k];
                    const double gamma = -d;
                    z[0] = gamma * p[0];
                    for (std::size_t s = 1; s < FS; ++s) {
                        const double urhs =
                            (s + 1 == FS) ? -t.eta : 0.0;
                        z[s] = (urhs + t.eta * z[s - 1]) * p[s];
                    }
                    for (std::size_t s = FS - 1; s-- > 0;)
                        z[s] += t.eta * p[s] * z[s + 1];
                    const double q = t.eta / d;
                    const double inv_denom =
                        1.0 / (1.0 + z[0] + q * z[FS - 1]);
                    const double scale_a =
                        (va[0] + q * va[FS - 1]) * inv_denom;
                    const double scale_b =
                        (vb[0] + q * vb[FS - 1]) * inv_denom;
                    for (std::size_t s = 0; s < FS; ++s) {
                        va[s] -= scale_a * z[s];
                        vb[s] -= scale_b * z[s];
                    }
                }
    }

    template <typename RhsA, typename RhsB>
    void facr_solve(facr_spectrum& out, const tri_factors& t,
                    RhsA&& ra, RhsB&& rb) {
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            for (std::size_t k = std::size_t(tid); k < FB;
                 k += std::size_t(P.lanes())) {
                facr_solve_pair(L, k, out, t, ra, rb);
            }
        });
    }

    void facr_scale(const facr_spectrum& src, double c,
                    const tri_factors& t, facr_spectrum& out) {
        facr_solve(out, t,
            [&](std::size_t i) { return c * src.a[i]; },
            [&](std::size_t i) { return c * src.b[i]; });
    }

    void facr_sum_inplace(facr_spectrum& io, const facr_spectrum& q,
                          const facr_spectrum& d, double c, double eta,
                          const tri_factors& t) {
        facr_solve(io, t,
            [&](std::size_t i) {
                return c * (io.a[i] + q.a[i]) - eta * d.a[i];
            },
            [&](std::size_t i) {
                return c * (io.b[i] + q.b[i]) - eta * d.b[i];
            });
    }

    void facr_diff(const facr_spectrum& p, const facr_spectrum& q,
                   const facr_spectrum* d, double c, double eta,
                   const tri_factors& t, facr_spectrum& out) {
        facr_solve(out, t,
            [&](std::size_t i) {
                return c * (p.a[i] - q.a[i]) -
                    (d == nullptr ? 0.0 : eta * d->a[i]);
            },
            [&](std::size_t i) {
                return c * (p.b[i] - q.b[i]) -
                    (d == nullptr ? 0.0 : eta * d->b[i]);
            });
    }

    void facr_g(const facr_spectrum& g, const facr_spectrum& d,
                double c, double eta, const tri_factors& t,
                facr_spectrum& out) {
        facr_solve(out, t,
            [&](std::size_t i) { return c * g.a[i] - eta * d.a[i]; },
            [&](std::size_t i) { return c * g.b[i] - eta * d.b[i]; });
    }

    // t_next = grad(x) + proj(t, 1/eta), in place.  Together with
    // fwd*_reflection this is exactly the reduced Split-Bregman recursion.
    void update_reflected_dual(const std::vector<double>& x,
                               std::vector<double>& tx,
                               std::vector<double>& ty, double eta) {
        const double threshold = 1.0 / eta;
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const double* xi = x.data() + y * W;
                const double* xn =
                    x.data() + ((y + 1 == H) ? 0 : y + 1) * W;
                double* px = tx.data() + y * W;
                double* py = ty.data() + y * W;
                const bool neumann_y =
                    solver == 2 && sweep_height && y + 1 == H;
                for (std::size_t j = 0; j < W; ++j) {
                    const double old_x = px[j], old_y = py[j];
                    const double radius =
                        std::sqrt(old_x * old_x + old_y * old_y);
                    const double shrink =
                        std::fmax(radius - threshold, 0.0) /
                        std::fmax(radius, 1e-12);
                    const double project = 1.0 - shrink;
                    const bool neumann_x =
                        solver == 2 && !sweep_height && j + 1 == W;
                    const double gx = neumann_x ? 0.0 :
                        xi[j + 1 == W ? 0 : j + 1] - xi[j];
                    const double gy = neumann_y ? 0.0 : xn[j] - xi[j];
                    px[j] = neumann_x ? 0.0 : gx + project * old_x;
                    py[j] = neumann_y ? 0.0 : gy + project * old_y;
                }
            }
        });
    }

    // Row-streamed divergence for the transpose-oriented FACR path.  The
    // current reflected row and one predecessor row are enough, so the
    // iterate plane can serve as scalar scratch without restoring either
    // eliminated vector-field plane.
    void reflection_divergence_plane(const std::vector<double>& tx,
                                     const std::vector<double>& ty,
                                     double eta, double* out) {
        const double threshold = 1.0 / eta;
        P.run([&](int tid) {
            const std::size_t lanes_n = std::size_t(P.lanes());
            const std::size_t lo = H * std::size_t(tid) / lanes_n;
            const std::size_t hi = H * (std::size_t(tid) + 1) / lanes_n;
            if (lo == hi) return;
            lane& L = *lanes[tid];
            const std::size_t yp0 = (lo == 0 ? H : lo) - 1;
            for (std::size_t x = 0; x < W; ++x) {
                double unused;
                if (solver == 2 && sweep_height && lo == 0) {
                    L.line[x] = 0.0;
                } else {
                    reflected(tx[yp0 * W + x], ty[yp0 * W + x], threshold,
                              unused, L.line[x]);
                }
            }
            for (std::size_t y = lo; y < hi; ++y) {
                for (std::size_t x = 0; x < W; ++x)
                    reflected(tx[y * W + x], ty[y * W + x], threshold,
                              L.reflect_x[x], L.reflect_y[x]);
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xp = (x == 0 ? W : x) - 1;
                    const double prev_x =
                        (solver == 2 && x == 0) ? 0.0 : L.reflect_x[xp];
                    out[y * W + x] =
                        L.reflect_x[x] - prev_x +
                        L.reflect_y[x] - L.line[x];
                }
                std::memcpy(L.line.data(), L.reflect_y.data(),
                            W * sizeof(double));
            }
        });
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
                const bool neumann_y =
                    solver == 2 && sweep_height && i + 1 == H;
                for (std::size_t j = 0; j < W - 1; ++j) {
                    const double tx = xi[j + 1] - xi[j] + pbx[j];
                    const double ty = neumann_y
                        ? 0.0 : xn[j] - xi[j] + pby[j];
                    const double r = std::sqrt(tx * tx + ty * ty);
                    const double coef =
                        std::fmax(r - thr, 0.0) / std::fmax(r, 1e-12);
                    pbx[j] = (1.0 - coef) * tx;
                    pdx[j] = (2.0 * coef - 1.0) * tx;
                    if (neumann_y) {
                        pby[j] = 0.0;
                        pdy[j] = 0.0;
                    } else {
                        pby[j] = (1.0 - coef) * ty;
                        pdy[j] = (2.0 * coef - 1.0) * ty;
                    }
                }
                {
                    const std::size_t j = W - 1;
                    const bool neumann_x = solver == 2 && !sweep_height;
                    const double tx = neumann_x
                        ? 0.0 : xi[0] - xi[j] + pbx[j];
                    const double ty = neumann_y
                        ? 0.0 : xn[j] - xi[j] + pby[j];
                    const double r = std::sqrt(tx * tx + ty * ty);
                    const double coef =
                        std::fmax(r - thr, 0.0) / std::fmax(r, 1e-12);
                    if (neumann_x) {
                        pbx[j] = 0.0;
                        pdx[j] = 0.0;
                    } else {
                        pbx[j] = (1.0 - coef) * tx;
                        pdx[j] = (2.0 * coef - 1.0) * tx;
                    }
                    if (neumann_y) {
                        pby[j] = 0.0;
                        pdy[j] = 0.0;
                    } else {
                        pby[j] = (1.0 - coef) * ty;
                        pdy[j] = (2.0 * coef - 1.0) * ty;
                    }
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

    void emit_split_state(const double* image, int pass,
                          double* cartoon_trace, double* texture_trace,
                          trace_visitor visitor, void* user) {
        if ((cartoon_trace == nullptr || texture_trace == nullptr) &&
            visitor == nullptr)
            return;
        const std::size_t n = H * W;
        double* cp = cartoon_trace == nullptr ? nullptr :
            cartoon_trace + std::size_t(pass) * n;
        double* tp = texture_trace == nullptr ? vplane.data() :
            texture_trace + std::size_t(pass) * n;
        for (std::size_t i = 0; i < n; ++i) {
            if (cp != nullptr) cp[i] = u[i];
            tp[i] = image[i] - u[i] - w[i];
        }
        if (visitor != nullptr)
            visitor(pass + 1, u.data(), tp, n, user);
    }

    void finish_split_texture(const double* image, double* texture) {
        if (texture == nullptr) return;
        const std::size_t n = H * W;
        for (std::size_t i = 0; i < n; ++i)
            texture[i] = image[i] - u[i] - w[i];
    }

    void run_split_reduced_spectral(
            const double* image, double* texture_out = nullptr,
            double* cartoon_trace = nullptr, double* texture_trace = nullptr,
            trace_visitor visitor = nullptr, void* user = nullptr) {
        const std::size_t n = H * W;
        for (auto* p : {&u, &w, &bux, &buy, &bvx, &bvy})
            std::memset(p->data(), 0, n * sizeof(double));
        u_spec.zero();
        w_spec.zero();
        fwd2d(image, f_spec);
        const double c_u = lam, eta_u = 2.0 * lam;
        const double c_v = 1.0 / mu, eta_v = 10.0 / mu;
        for (int pass = 0; pass < passes; ++pass) {
            if (pass == 0) {
                solve_scale(f_spec.a.data(), f_spec.b.data(), c_u,
                            s_u.data(), u_spec.a.data(), u_spec.b.data());
            } else {
                reflection_divergence_plane(bux, buy, eta_u, u.data());
                fwd2d(u.data(), d_spec);
                solve_sum_inplace(u_spec.a.data(), u_spec.b.data(),
                                  w_spec.a.data(), w_spec.b.data(),
                                  d_spec.a.data(), d_spec.b.data(), c_u,
                                  eta_u, s_u.data());
            }
            inv2d(u_spec, u.data());
            update_reflected_dual(u, bux, buy, eta_u);

            if (pass == 0) {
                solve_diff_scale(f_spec.a.data(), f_spec.b.data(),
                                 u_spec.a.data(), u_spec.b.data(), c_v,
                                 s_v.data(), w_spec.a.data(),
                                 w_spec.b.data());
            } else {
                reflection_divergence_plane(bvx, bvy, eta_v, w.data());
                fwd2d(w.data(), d_spec);
                solve_diff(f_spec.a.data(), f_spec.b.data(),
                           u_spec.a.data(), u_spec.b.data(),
                           d_spec.a.data(), d_spec.b.data(), c_v, eta_v,
                           s_v.data(), w_spec.a.data(), w_spec.b.data());
            }
            inv2d(w_spec, w.data());
            update_reflected_dual(w, bvx, bvy, eta_v);
            emit_split_state(image, pass, cartoon_trace, texture_trace,
                             visitor, user);
        }
        finish_split_texture(image, texture_out);
    }

    void run_split_reduced_facr(
            const double* image, double* texture_out = nullptr,
            double* cartoon_trace = nullptr, double* texture_trace = nullptr,
            trace_visitor visitor = nullptr, void* user = nullptr) {
        const std::size_t n = H * W;
        for (auto* p : {&u, &w, &bux, &buy, &bvx, &bvy})
            std::memset(p->data(), 0, n * sizeof(double));
        fu_spec.zero();
        fw_spec.zero();
        facr_fwd(image, ff_spec);
        const double c_u = lam, eta_u = 2.0 * lam;
        const double c_v = 1.0 / mu, eta_v = 10.0 / mu;
        for (int pass = 0; pass < passes; ++pass) {
            if (pass == 0) {
                facr_scale(ff_spec, c_u, t_u, fu_spec);
            } else {
                reflection_divergence_plane(bux, buy, eta_u, u.data());
                facr_fwd(u.data(), fd_spec);
                facr_sum_inplace(fu_spec, fw_spec, fd_spec, c_u, eta_u,
                                  t_u);
            }
            facr_inv(fu_spec, u.data());
            update_reflected_dual(u, bux, buy, eta_u);

            if (pass == 0) {
                facr_diff(ff_spec, fu_spec, nullptr, c_v, eta_v, t_v,
                          fw_spec);
            } else {
                reflection_divergence_plane(bvx, bvy, eta_v, w.data());
                facr_fwd(w.data(), fd_spec);
                facr_diff(ff_spec, fu_spec, &fd_spec, c_v, eta_v, t_v,
                          fw_spec);
            }
            facr_inv(fw_spec, w.data());
            update_reflected_dual(w, bvx, bvy, eta_v);
            emit_split_state(image, pass, cartoon_trace, texture_trace,
                             visitor, user);
        }
        finish_split_texture(image, texture_out);
    }

    void run_split_reduced(
            const double* image, double* texture_out = nullptr,
            double* cartoon_trace = nullptr, double* texture_trace = nullptr,
            trace_visitor visitor = nullptr, void* user = nullptr) {
        if (facr_active)
            run_split_reduced_facr(image, texture_out, cartoon_trace,
                                   texture_trace, visitor, user);
        else
            run_split_reduced_spectral(image, texture_out, cartoon_trace,
                                       texture_trace, visitor, user);
    }

    // ---- the alternation (shared by split and decompose) -----------------
    //
    // Runs the TGFD passes and leaves u (cartoon layer), w (texture-side
    // ROF survivor), vplane = f - u - w (texture layer), and the spectra
    // of f, u, w in place.

    void run_passes_spectral(const double* image,
                             double* cartoon_trace = nullptr,
                             double* texture_trace = nullptr,
                             trace_visitor visitor = nullptr,
                             void* user = nullptr) {
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

            if ((cartoon_trace != nullptr && texture_trace != nullptr) ||
                visitor != nullptr) {
                double* __restrict cp = cartoon_trace == nullptr
                    ? nullptr
                    : cartoon_trace + static_cast<std::size_t>(p) * n;
                double* __restrict tp = texture_trace == nullptr
                    ? vplane.data()
                    : texture_trace + static_cast<std::size_t>(p) * n;
                const double* __restrict fp = image;
                const double* __restrict up = u.data();
                const double* __restrict wp = w.data();
                for (std::size_t i = 0; i < n; ++i) {
                    if (cp != nullptr) cp[i] = up[i];
                    tp[i] = fp[i] - up[i] - wp[i];
                }
                if (visitor != nullptr)
                    visitor(p + 1, u.data(), tp, n, user);
            }
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

    void run_passes_facr(const double* image,
                         double* cartoon_trace = nullptr,
                         double* texture_trace = nullptr,
                         trace_visitor visitor = nullptr,
                         void* user = nullptr) {
        const std::size_t n = H * W;
        for (auto* p : {&u, &w, &bux, &buy, &dbux, &dbuy, &bvx, &bvy,
                        &dbvx, &dbvy})
            std::memset(p->data(), 0, n * sizeof(double));
        fu_spec.zero();
        fw_spec.zero();
        facr_fwd(image, ff_spec);

        const double c_u = lam, eta_u = 2.0 * lam;
        const double c_v = 1.0 / mu, eta_v = 10.0 / mu;
        for (int pass = 0; pass < passes; ++pass) {
            if (pass == 0) {
                facr_scale(ff_spec, c_u, t_u, fu_spec);
            } else {
                facr_fwd_div(dbux, dbuy, fd_spec);
                facr_sum_inplace(fu_spec, fw_spec, fd_spec, c_u, eta_u,
                                  t_u);
            }
            facr_inv(fu_spec, u.data());
            shrink(u, bux, buy, dbux, dbuy, eta_u);

            if (pass == 0) {
                facr_diff(ff_spec, fu_spec, nullptr, c_v, eta_v, t_v,
                          fw_spec);
            } else {
                facr_fwd_div(dbvx, dbvy, fd_spec);
                facr_diff(ff_spec, fu_spec, &fd_spec, c_v, eta_v, t_v,
                          fw_spec);
            }
            facr_inv(fw_spec, w.data());
            shrink(w, bvx, bvy, dbvx, dbvy, eta_v);

            if ((cartoon_trace != nullptr && texture_trace != nullptr) ||
                visitor != nullptr) {
                double* cp = cartoon_trace == nullptr ? nullptr :
                    cartoon_trace + std::size_t(pass) * n;
                double* tp = texture_trace == nullptr ? vplane.data() :
                    texture_trace + std::size_t(pass) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    if (cp != nullptr) cp[i] = u[i];
                    tp[i] = image[i] - u[i] - w[i];
                }
                if (visitor != nullptr)
                    visitor(pass + 1, u.data(), tp, n, user);
            }
        }
        for (std::size_t i = 0; i < n; ++i)
            vplane[i] = image[i] - u[i] - w[i];
    }

    void run_passes(const double* image, double* cartoon_trace = nullptr,
                    double* texture_trace = nullptr,
                    trace_visitor visitor = nullptr, void* user = nullptr) {
        if (!facr_active)
            run_passes_spectral(image, cartoon_trace, texture_trace,
                                visitor, user);
        else
            run_passes_facr(image, cartoon_trace, texture_trace,
                            visitor, user);
    }

    // ---- a plain ROF solve from a spectrum already in hand ---------------
    //
    // out <- argmin_x TV(x) + (c/2)|x - g|^2 by Split Bregman sweeps from a
    // FRESH state (Bregman states are c- and eta-scaled and must never be
    // carried between different fidelity constants).  gs is the spectrum of
    // g and gplane its samples (needed only to seed the change test); s is
    // the symbol table for this (c, eta) pair.  Sweeps stop early once the
    // relative iterate change falls below tol (tol <= 0 disables the test).
    // u_spec is used as spectral scratch: the u plane is left intact, so
    // decompose() may call this after run_passes().

    void rof_from_spec(const spectrum& gs, const double* gplane, double c,
                       double eta, const std::vector<double>& s, int sweeps,
                       double tol, double* out) {
        const std::size_t n = H * W;
        std::memset(rbx.data(), 0, n * sizeof(double));
        std::memset(rby.data(), 0, n * sizeof(double));
        std::memcpy(prev.data(), gplane, n * sizeof(double));
        bool done = false;
        for (int sweep = 0; sweep < sweeps && !done; ++sweep) {
            if (sweep == 0) {
                solve_scale(gs.a.data(), gs.b.data(), c, s.data(),
                            u_spec.a.data(), u_spec.b.data());
            } else {
                fwd2d_div(rdbx, rdby, d_spec);
                solve_g(gs.a.data(), gs.b.data(), d_spec.a.data(),
                        d_spec.b.data(), c, eta, s.data(), u_spec.a.data(),
                        u_spec.b.data());
            }
            inv2d(u_spec, xit.data());
            shrink(xit, rbx, rby, rdbx, rdby, eta);
            if (tol > 0.0) {
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
                done = (d0 + d1) <= tol * tol * (x0 + x1);
            }
        }
        std::memcpy(out, xit.data(), n * sizeof(double));
    }

    void rof_from_facr(const facr_spectrum& gs, const double* gplane,
                       double c, double eta, const tri_factors& t,
                       int sweeps, double tol, double* out) {
        const std::size_t n = H * W;
        std::memset(rbx.data(), 0, n * sizeof(double));
        std::memset(rby.data(), 0, n * sizeof(double));
        std::memset(rdbx.data(), 0, n * sizeof(double));
        std::memset(rdby.data(), 0, n * sizeof(double));
        std::memcpy(prev.data(), gplane, n * sizeof(double));
        bool done = false;
        for (int sweep = 0; sweep < sweeps && !done; ++sweep) {
            if (sweep == 0) {
                facr_scale(gs, c, t, fu_spec);
            } else {
                facr_fwd_div(rdbx, rdby, fd_spec);
                facr_g(gs, fd_spec, c, eta, t, fu_spec);
            }
            facr_inv(fu_spec, xit.data());
            shrink(xit, rbx, rby, rdbx, rdby, eta);
            if (tol > 0.0) {
                double d0 = 0.0, d1 = 0.0, x0 = 0.0, x1 = 0.0;
                std::size_t i = 0;
                for (; i + 1 < n; i += 2) {
                    const double a0 = xit[i] - prev[i];
                    const double a1 = xit[i + 1] - prev[i + 1];
                    d0 += a0 * a0; d1 += a1 * a1;
                    x0 += xit[i] * xit[i]; x1 += xit[i + 1] * xit[i + 1];
                    prev[i] = xit[i]; prev[i + 1] = xit[i + 1];
                }
                if (i < n) {
                    const double a = xit[i] - prev[i];
                    d0 += a * a; x0 += xit[i] * xit[i]; prev[i] = xit[i];
                }
                done = (d0 + d1) <= tol * tol * (x0 + x1);
            }
        }
        std::memcpy(out, xit.data(), n * sizeof(double));
    }

    // ---- public ROF: transform the image, then solve --------------------
    //
    // The symbol table depends on (c, eta) and costs WB*HB cosine pairs to
    // build, so the last one is cached: effect pipelines call this
    // repeatedly with fixed constants.

    void rof(const double* image, double* smooth, double c, double eta,
             int sweeps, double tol) {
        ensure_rof_storage();
        if (facr_active) {
            if (t_gen.pivot.empty() || c != t_gen.c || eta != t_gen.eta)
                build_factors(t_gen, c, eta);
            facr_fwd(image, ff_spec);
            rof_from_facr(ff_spec, image, c, eta, t_gen, sweeps, tol,
                           smooth);
            return;
        }
        if (s_gen.empty() || c != gen_c || eta != gen_eta) {
            symbol(s_gen, c, eta);
            gen_c = c;
            gen_eta = eta;
        }
        fwd2d(image, f_spec);
        rof_from_spec(f_spec, image, c, eta, s_gen, sweeps, tol, smooth);
    }

    // ---- split: the model decomposition alone, no ladder ----------------
    //
    // cartoon = u, texture = v: exactly the pair Gilles' Algorithm 3
    // produces.  (The five-output decompose() instead reports
    // cartoon = u + s0, folding the ladder's coarsest survivor back into
    // the cartoon so that cartoon + bands = u + v.)

    void split(const double* image, double* cartoon, double* texture) {
        run_split_reduced(image, texture);
        const std::size_t n = H * W;
        std::memcpy(cartoon, u.data(), n * sizeof(double));
    }

    // All intermediate model states from one run.  Outputs are
    // passes*height*width doubles in pass-major order.
    void split_trace(const double* image, double* cartoon_trace,
                     double* texture_trace) {
        run_split_reduced(image, nullptr, cartoon_trace, texture_trace);
    }

    void split_visit(const double* image, trace_visitor visitor, void* user) {
        ensure_visit_storage();
        run_split_reduced(image, nullptr, nullptr, nullptr, visitor, user);
    }

    // ---- the full decomposition -----------------------------------------

    void decompose(const double* image, double* cartoon, double* texture,
                   double* band_coarse, double* band_mid, double* band_fine) {
        ensure_decompose_storage();
        if (facr_active) {
            decompose_facr(image, cartoon, texture, band_coarse, band_mid,
                           band_fine);
            return;
        }
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
        const std::vector<double>* rung_s[3] = {&s_r0, &s_r1, &s_r2};
        const double rung_mu[3] = {mu, mu / 4.0, mu / 16.0};
        double* rung_out[3] = {cartoon, band_mid, band_fine};  // staging
        for (int rr = 0; rr < 3; ++rr)
            rof_from_spec(v_spec, vplane.data(), 1.0 / rung_mu[rr],
                          10.0 / rung_mu[rr], *rung_s[rr], rung_sweeps,
                          rung_tol, rung_out[rr]);

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

    void decompose_facr(const double* image, double* cartoon,
                        double* texture, double* band_coarse,
                        double* band_mid, double* band_fine) {
        const std::size_t n = H * W;
        run_passes_facr(image);
        const std::size_t m = FS * FB;
        for (std::size_t i = 0; i < m; ++i) {
            fv_spec.a[i] = ff_spec.a[i] - fu_spec.a[i] - fw_spec.a[i];
            fv_spec.b[i] = ff_spec.b[i] - fu_spec.b[i] - fw_spec.b[i];
        }
        const tri_factors* rung_t[3] = {&t_r0, &t_r1, &t_r2};
        const double rung_mu[3] = {mu, mu / 4.0, mu / 16.0};
        double* rung_out[3] = {cartoon, band_mid, band_fine};
        for (int rr = 0; rr < 3; ++rr)
            rof_from_facr(fv_spec, vplane.data(), 1.0 / rung_mu[rr],
                           10.0 / rung_mu[rr], *rung_t[rr], rung_sweeps,
                           rung_tol, rung_out[rr]);
        for (std::size_t i = 0; i < n; ++i) {
            const double a0 = cartoon[i], a1 = band_mid[i],
                         a2 = band_fine[i];
            texture[i] = vplane[i];
            cartoon[i] = u[i] + a0;
            band_mid[i] = a2 - a1;
            band_fine[i] = vplane[i] - a2;
            band_coarse[i] = a1 - a0;
        }
    }
};

}  // namespace meyer
