// dip_algo.cpp
//
// C++ port of active_delta_center5_fast1 (the DIP / finite-Zak walk + in-house
// PGHI seed + record OLA baseline).  Faithful translation of
// active_delta_center5_cpp_reference.py, validated operator-by-operator against
// that reference.
//
// Geometry (fixed, matches the reference):
//   long aperture : n=1024, hop=128, level=5  -> state [Ib,32,32]
//   short aperture: n=128,  hop=32,  level=2  -> attaches inside the long state
//   active short offsets (deltas): central five, 12..16
//
// Compiled into the same libiqwaterfall library as the viewer core.
//
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <complex>
#include <vector>
#include <queue>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <string>

#include <bfft/bfft.h>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#if defined(_WIN32)
#define IQW_API __declspec(dllexport)
#else
#define IQW_API __attribute__((visibility("default")))
#endif

namespace {

using cd = std::complex<double>;
constexpr double PI = 3.14159265358979323846;

// Fixed geometry constants.
// Shared column count q* = 32; QB = QS = HS = q* are fixed (the attach identity
// r0 = delta/q* pins the short hop to q*).  The aperture sizes NB/NS and their
// derived EB/TB/ES/TS/HB/NDELTA are per-solver runtime values (see DipBase).
constexpr int QSTAR = 32, QB = 32, QS = 32, HS = 32;

#if defined(__APPLE__)
// Thread-local Accelerate provider for the full frame FFTs. Paused-state
// transforms are only 2..32 points and stay in the scalar kernel below; at
// frame lengths >=128, vDSP's vectorized provider amortizes split/copy cost.
struct VdspFftProvider {
    vDSP_DFT_SetupD forward[18]{};
    vDSP_DFT_SetupD inverse[18]{};
    std::vector<double> in_re, in_im, out_re, out_im;

    ~VdspFftProvider() {
        for (int i = 0; i < 18; ++i) {
            if (forward[i]) vDSP_DFT_DestroySetupD(forward[i]);
            if (inverse[i]) vDSP_DFT_DestroySetupD(inverse[i]);
        }
    }

    bool execute(cd* a, int n, int sign) {
        if (n < 128 || n > (1 << 17) || (n & (n - 1))) return false;
        int lg = (int)std::lround(std::log2((double)n));
        vDSP_DFT_SetupD& setup = sign < 0 ? forward[lg] : inverse[lg];
        if (!setup) {
            setup = vDSP_DFT_zop_CreateSetupD(
                nullptr, (vDSP_Length)n,
                sign < 0 ? vDSP_DFT_FORWARD : vDSP_DFT_INVERSE);
        }
        if (!setup) return false;
        in_re.resize(n); in_im.resize(n); out_re.resize(n); out_im.resize(n);
        for (int i = 0; i < n; ++i) {
            in_re[i] = a[i].real();
            in_im[i] = a[i].imag();
        }
        vDSP_DFT_ExecuteD(setup, in_re.data(), in_im.data(),
                         out_re.data(), out_im.data());
        for (int i = 0; i < n; ++i) a[i] = cd(out_re[i], out_im[i]);
        return true;
    }
};

thread_local VdspFftProvider g_vdsp_fft;
#endif

// Portable complex FFT adapter built from two SIMD-native BFFT real plans.
// Forward recombines the two Hermitian half spectra. Inverse splits an
// arbitrary complex spectrum into the Hermitian spectra of its real and
// imaginary time-domain components, then performs two real inverse transforms.
struct BfftComplexProvider {
    bfft_plan* plans[18]{};
    std::vector<double> re, im, work;
    std::vector<bfft_complex> A, B, scratch;

    ~BfftComplexProvider() {
        for (bfft_plan* p : plans) if (p) bfft_plan_destroy(p);
    }

    bool execute(cd* a, int n, int sign) {
        if (n < 128 || n > (1 << 17) || (n & (n - 1))) return false;
        int lg = (int)std::lround(std::log2((double)n));
        bfft_plan*& plan = plans[lg];
        if (!plan && bfft_plan_create((size_t)n, &plan) != BFFT_OK) return false;
        size_t bins = bfft_plan_bins(plan);
        re.resize(n); im.resize(n); A.resize(bins); B.resize(bins);
        work.resize(bfft_plan_work_size(plan));
        scratch.resize(bfft_plan_native_scratch_size(plan) + 1);
        if (sign < 0) {
            for (int k = 0; k < n; ++k) { re[k] = a[k].real(); im[k] = a[k].imag(); }
            if (bfft_forward(plan, re.data(), A.data(), work.data(), scratch.data()) != BFFT_OK ||
                bfft_forward(plan, im.data(), B.data(), work.data(), scratch.data()) != BFFT_OK)
                return false;
            int half = n / 2;
            for (int k = 0; k < n; ++k) {
                int m = k <= half ? k : n - k;
                double ar = A[m].re, ai = k <= half ? A[m].im : -A[m].im;
                double br = B[m].re, bi = k <= half ? B[m].im : -B[m].im;
                a[k] = cd(ar - bi, ai + br);
            }
        } else {
            int half = n / 2;
            for (int k = 0; k <= half; ++k) {
                cd x = a[k];
                cd y = std::conj(a[(n - k) % n]);
                cd r = 0.5 * (x + y);
                cd d = x - y;
                cd q(0.5 * d.imag(), -0.5 * d.real());
                A[k].re = r.real(); A[k].im = r.imag();
                B[k].re = q.real(); B[k].im = q.imag();
            }
            if (bfft_inverse(plan, A.data(), re.data()) != BFFT_OK ||
                bfft_inverse(plan, B.data(), im.data()) != BFFT_OK)
                return false;
            // bfft_inverse is normalized; fft_pow2's internal contract is an
            // unnormalized inverse (its callers apply 1/N themselves).
            for (int k = 0; k < n; ++k) a[k] = (double)n * cd(re[k], im[k]);
        }
        return true;
    }
};

thread_local BfftComplexProvider g_bfft_fft;

enum class FullFftProvider { scalar, bfft, accelerate };
FullFftProvider full_fft_provider() {
    static FullFftProvider value = [] {
        const char* env = std::getenv("IQW_FFT_PROVIDER");
        if (env && std::string(env) == "scalar") return FullFftProvider::scalar;
        if (env && std::string(env) == "bfft") return FullFftProvider::bfft;
#if defined(__APPLE__)
        if (env && std::string(env) == "accelerate") return FullFftProvider::accelerate;
#endif
        return FullFftProvider::bfft;
    }();
    return value;
}

// ---------------------------------------------------------------------------
// In-place iterative radix-2 FFT.  sign=-1 forward (numpy fft, unnormalized),
// sign=+1 inverse without the 1/n scale (caller scales).
// ---------------------------------------------------------------------------
void fft_pow2(cd* a, int n, int sign) {
    FullFftProvider provider = full_fft_provider();
    if (provider == FullFftProvider::bfft && g_bfft_fft.execute(a, n, sign)) return;
#if defined(__APPLE__)
    if (provider == FullFftProvider::accelerate && g_vdsp_fft.execute(a, n, sign)) return;
#endif
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = sign * 2.0 * PI / len;
        cd wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1.0, 0.0);
            for (int k = 0; k < len / 2; ++k) {
                cd u = a[i + k];
                cd v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

using Ten = std::vector<cd>;   // flat tensor; caller tracks dims

// prefix_to_level: frames [I,n] -> Z [I,e,q], e=2^level, q=n/e.  FFT along the
// e-axis (axis=1 of the [I,e,q] reshape).
Ten prefix_to_level(const std::vector<double>& frames, int I, int n, int level) {
    int e = 1 << level, q = n >> level;
    Ten Z((size_t)I * e * q);
    std::vector<cd> col(e);
    for (int i = 0; i < I; ++i)
        for (int b = 0; b < q; ++b) {
            for (int a = 0; a < e; ++a) col[a] = cd(frames[(size_t)i * n + a * q + b], 0.0);
            fft_pow2(col.data(), e, -1);
            for (int a = 0; a < e; ++a) Z[((size_t)i * e + a) * q + b] = col[a];
        }
    return Z;
}

// inverse_prefix: Z [I,e,q] -> frames [I,e*q] (complex), ifft along e-axis.
Ten inverse_prefix(const Ten& Z, int I, int e, int q) {
    int n = e * q;
    Ten out((size_t)I * n);
    std::vector<cd> col(e);
    double inv = 1.0 / e;
    for (int i = 0; i < I; ++i)
        for (int b = 0; b < q; ++b) {
            for (int a = 0; a < e; ++a) col[a] = Z[((size_t)i * e + a) * q + b];
            fft_pow2(col.data(), e, +1);
            for (int a = 0; a < e; ++a) out[(size_t)i * n + a * q + b] = col[a] * inv;
        }
    return out;
}

// One DIF suffix butterfly: B [I,e,q] -> C [I,2e,q/2].
Ten stage_fwd(const Ten& B, int I, int e, int q) {
    int q2 = q / 2, e2 = 2 * e;
    Ten C((size_t)I * e2 * q2);
    for (int a = 0; a < e; ++a) {
        cd w(std::cos(-PI * a / e), std::sin(-PI * a / e));
        for (int i = 0; i < I; ++i)
            for (int b = 0; b < q2; ++b) {
                cd A = B[((size_t)i * e + a) * q + b];
                cd D = B[((size_t)i * e + a) * q + (q2 + b)];
                cd wd = w * D;
                C[((size_t)i * e2 + a) * q2 + b] = A + wd;
                C[((size_t)i * e2 + (e + a)) * q2 + b] = A - wd;
            }
    }
    return C;
}

// Inverse suffix butterfly: C [I,2e,q2] -> B [I,e,2*q2].
Ten stage_bwd(const Ten& C, int I, int e2, int q2) {
    int e = e2 / 2, q = 2 * q2;
    Ten B((size_t)I * e * q);
    for (int a = 0; a < e; ++a) {
        cd w(std::cos(-PI * a / e), std::sin(-PI * a / e));
        cd cw = std::conj(w);
        for (int i = 0; i < I; ++i)
            for (int b = 0; b < q2; ++b) {
                cd P = C[((size_t)i * e2 + a) * q2 + b];
                cd Q = C[((size_t)i * e2 + (e + a)) * q2 + b];
                B[((size_t)i * e + a) * q + b] = 0.5 * (P + Q);
                B[((size_t)i * e + a) * q + (q2 + b)] = 0.5 * cw * (P - Q);
            }
    }
    return B;
}

// finish_from_level: paused Z [I,e,q] -> endpoint spectrum [I,n].
Ten finish_from_level(Ten C, int I, int e, int q) {
    int n = e * q;
    int m = (int)std::lround(std::log2((double)n));
    int lev = (int)std::lround(std::log2((double)e));
    int cur_e = e, cur_q = q;
    for (int l = lev; l < m; ++l) {
        C = stage_fwd(C, I, cur_e, cur_q);
        cur_e *= 2; cur_q /= 2;
    }
    // C is [I,n,1]; take [:,:,0].
    Ten Y((size_t)I * n);
    for (int i = 0; i < I; ++i)
        for (int a = 0; a < n; ++a) Y[(size_t)i * n + a] = C[(size_t)i * n + a];
    return Y;
}

// back_to_level: endpoint spectrum Y [I,n] -> paused [I,2^level,n/2^level].
Ten back_to_level(const Ten& Y, int I, int n, int level) {
    int m = (int)std::lround(std::log2((double)n));
    Ten C = Y;               // as [I,n,1]
    int cur_e2 = n, cur_q2 = 1;
    for (int l = m - 1; l >= level; --l) {
        C = stage_bwd(C, I, cur_e2, cur_q2);
        cur_e2 /= 2; cur_q2 *= 2;
    }
    return C;
}

// tap3: periodic-Hann analysis tap in paused state.  ph indexed by column b.
// out[i,a,b] = 0.5 Z - 0.25 ph[b] Z[a-1] - 0.25 conj(ph[b]) Z[a+1].
Ten tap3(const Ten& Z, int I, int e, int q, const std::vector<cd>& ph) {
    Ten out((size_t)I * e * q);
    for (int i = 0; i < I; ++i)
        for (int a = 0; a < e; ++a) {
            int am = (a - 1 + e) % e, ap = (a + 1) % e;
            for (int b = 0; b < q; ++b) {
                cd z0 = Z[((size_t)i * e + a) * q + b];
                cd zm = Z[((size_t)i * e + am) * q + b];
                cd zp = Z[((size_t)i * e + ap) * q + b];
                out[((size_t)i * e + a) * q + b] =
                    0.5 * z0 - 0.25 * ph[b] * zm - 0.25 * std::conj(ph[b]) * zp;
            }
        }
    return out;
}

// tap3_adj: adjoint of tap3.
// out[i,a,b] = 0.5 G - 0.25 conj(ph[b]) G[a+1] - 0.25 ph[b] G[a-1].
Ten tap3_adj(const Ten& G, int I, int e, int q, const std::vector<cd>& ph) {
    Ten out((size_t)I * e * q);
    for (int i = 0; i < I; ++i)
        for (int a = 0; a < e; ++a) {
            int am = (a - 1 + e) % e, ap = (a + 1) % e;
            for (int b = 0; b < q; ++b) {
                cd g0 = G[((size_t)i * e + a) * q + b];
                cd gm = G[((size_t)i * e + am) * q + b];
                cd gp = G[((size_t)i * e + ap) * q + b];
                out[((size_t)i * e + a) * q + b] =
                    0.5 * g0 - 0.25 * std::conj(ph[b]) * gp - 0.25 * ph[b] * gm;
            }
        }
    return out;
}

std::vector<double> periodic_hann(int n) {
    std::vector<double> w(n);
    for (int k = 0; k < n; ++k) w[k] = 0.5 - 0.5 * std::cos(2.0 * PI * k / n);
    return w;
}

std::vector<int> frame_offsets(int L, int n, int hop) {
    std::vector<int> offs;
    for (int o = 0; o + n <= L; o += hop) offs.push_back(o);
    return offs;
}

// Selected attachment matrices M[d] = F_es @ crop_d @ F_eb^{-1}. Fast1 uses
// five central deltas, so constructing every possible matrix was pure setup
// waste (97 matrices at NB=4096/NS=1024). The k sum is geometric:
//   sum_k exp(i*2pi*k*(b/eb-a/es)).
std::vector<Ten> attachment_mats_selected(
        int eb, int es, const std::vector<int>& deltas) {
    std::vector<Ten> mats(deltas.size(), Ten((size_t)es * eb));
    for (size_t di = 0; di < deltas.size(); ++di) {
        int d = deltas[di];
        for (int a = 0; a < es; ++a)
            for (int b = 0; b < eb; ++b) {
                double half = PI * ((double)b / eb - (double)a / es);
                double den = std::sin(half);
                cd series;
                if (std::abs(den) < 1e-14) {
                    series = cd((double)es, 0.0);
                } else {
                    double amp = std::sin(es * half) / den;
                    double angle = (es - 1) * half;
                    series = amp * cd(std::cos(angle), std::sin(angle));
                }
                double shift = 2.0 * PI * d * b / eb;
                cd phase(std::cos(shift), std::sin(shift));
                mats[di][a * eb + b] = phase * series / (double)eb;
            }
    }
    return mats;
}

// ---------------------------------------------------------------------------
// PGHI phase field.  mag is [I,nfft] full-spectrum magnitude; only 0..nfft/2
// drive propagation, negative half is filled by Hermitian antisymmetry.
// Writes phase_full [I,nfft].
// ---------------------------------------------------------------------------
void pghi_phases(const double* mag, int I, int nfft, int n, int hop,
                 double* phase_full) {
    const int F = nfft / 2 + 1;
    const double rel_tol = 1e-9, lam_factor = 0.25645;
    std::vector<double> A((size_t)F * I), s((size_t)F * I);
    double amax = 0.0;
    for (int f = 0; f < F; ++f)
        for (int i = 0; i < I; ++i) {
            double v = mag[(size_t)i * nfft + f];
            A[(size_t)f * I + i] = v;
            s[(size_t)f * I + i] = std::log(v + 1e-30);
            amax = std::max(amax, v);
        }
    double lam = lam_factor * (double)n * (double)n;

    std::vector<double> phi_t((size_t)F * I), phi_w((size_t)F * I);
    for (int f = 0; f < F; ++f) {
        double wf = (double)f / nfft;
        for (int i = 0; i < I; ++i) {
            double dsdw = 0.0, dsdt = 0.0;
            if (f > 0 && f < F - 1)
                dsdw = 0.5 * nfft * (s[(size_t)(f + 1) * I + i] - s[(size_t)(f - 1) * I + i]);
            if (i > 0 && i < I - 1)
                dsdt = (s[(size_t)f * I + (i + 1)] - s[(size_t)f * I + (i - 1)]) / (2.0 * hop);
            phi_t[(size_t)f * I + i] = 2.0 * PI * wf + dsdw / lam;
            phi_w[(size_t)f * I + i] = -lam * dsdt;
        }
    }

    std::vector<double> phase((size_t)F * I, 0.0);
    std::vector<char> active((size_t)F * I), done((size_t)F * I);
    double thr = rel_tol * (amax + 1e-30);
    for (size_t k = 0; k < A.size(); ++k) {
        active[k] = A[k] > thr;
        done[k] = !active[k];
    }

    // Descending-magnitude order, ties broken by larger flat index first
    // (matches numpy argsort(...)[::-1]).
    std::vector<int> order(A.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int x, int y) {
        if (A[x] != A[y]) return A[x] > A[y];
        return x > y;
    });

    // Max-heap on magnitude; ties -> smaller f then smaller i (matches heapq on
    // (-a, m, i)).
    struct Node { double a; int f, i; };
    auto cmp = [](const Node& x, const Node& y) {
        if (x.a != y.a) return x.a < y.a;     // larger a = higher priority
        if (x.f != y.f) return x.f > y.f;     // smaller f = higher priority
        return x.i > y.i;                     // smaller i = higher priority
    };
    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> heap(cmp);

    auto AT = [&](int f, int i) -> double { return A[(size_t)f * I + i]; };
    auto PH = [&](int f, int i) -> double& { return phase[(size_t)f * I + i]; };
    auto DN = [&](int f, int i) -> char& { return done[(size_t)f * I + i]; };
    auto PT = [&](int f, int i) -> double { return phi_t[(size_t)f * I + i]; };
    auto PW = [&](int f, int i) -> double { return phi_w[(size_t)f * I + i]; };

    for (int flat : order) {
        int f = flat / I, i = flat % I;
        if (!active[flat]) break;
        if (done[flat]) continue;
        done[flat] = 1;
        while (!heap.empty()) heap.pop();
        heap.push({AT(f, i), f, i});
        while (!heap.empty()) {
            Node cur = heap.top(); heap.pop();
            int f0 = cur.f, i0 = cur.i;
            if (i0 + 1 < I && !DN(f0, i0 + 1)) {
                PH(f0, i0 + 1) = PH(f0, i0) + 0.5 * hop * (PT(f0, i0) + PT(f0, i0 + 1));
                DN(f0, i0 + 1) = 1; heap.push({AT(f0, i0 + 1), f0, i0 + 1});
            }
            if (i0 - 1 >= 0 && !DN(f0, i0 - 1)) {
                PH(f0, i0 - 1) = PH(f0, i0) - 0.5 * hop * (PT(f0, i0) + PT(f0, i0 - 1));
                DN(f0, i0 - 1) = 1; heap.push({AT(f0, i0 - 1), f0, i0 - 1});
            }
            if (f0 + 1 < F && !DN(f0 + 1, i0)) {
                PH(f0 + 1, i0) = PH(f0, i0) + 0.5 / nfft * (PW(f0, i0) + PW(f0 + 1, i0));
                DN(f0 + 1, i0) = 1; heap.push({AT(f0 + 1, i0), f0 + 1, i0});
            }
            if (f0 - 1 >= 0 && !DN(f0 - 1, i0)) {
                PH(f0 - 1, i0) = PH(f0, i0) - 0.5 / nfft * (PW(f0, i0) + PW(f0 - 1, i0));
                DN(f0 - 1, i0) = 1; heap.push({AT(f0 - 1, i0), f0 - 1, i0});
            }
        }
    }

    double c0 = (n - 1) / 2.0;
    for (int i = 0; i < I; ++i) {
        for (int f = 0; f < F; ++f)
            phase_full[(size_t)i * nfft + f] =
                PH(f, i) - 2.0 * PI * f * c0 / nfft;
        for (int pos = 1; pos < nfft / 2; ++pos)
            phase_full[(size_t)i * nfft + (nfft - pos)] =
                -phase_full[(size_t)i * nfft + pos];
    }
}

// ---------------------------------------------------------------------------
// Magnitudes yb, ys: |fft| of windowed frames (real input).
// ---------------------------------------------------------------------------
void windowed_frame_mag(const std::vector<double>& x, int L, int n, int hop,
                        std::vector<double>& mag_out, int& I_out) {
    auto offs = frame_offsets(L, n, hop);
    int I = (int)offs.size();
    I_out = I;
    auto win = periodic_hann(n);
    mag_out.assign((size_t)I * n, 0.0);
    std::vector<cd> frame(n);
    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < n; ++j) frame[j] = cd(x[offs[i] + j] * win[j], 0.0);
        fft_pow2(frame.data(), n, -1);
        for (int j = 0; j < n; ++j) mag_out[(size_t)i * n + j] = std::abs(frame[j]);
    }
}

// ifft of endpoint spectra -> real frames (ifft_dip then real()).
std::vector<double> ifft_real_rows(const Ten& Y, int I, int n) {
    // ifft over the full length: back_to_level(Y,0)[:,0,:] == full ifft.
    std::vector<double> out((size_t)I * n);
    std::vector<cd> row(n);
    double inv = 1.0 / n;
    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < n; ++j) row[j] = Y[(size_t)i * n + j];
        fft_pow2(row.data(), n, +1);
        for (int j = 0; j < n; ++j) out[(size_t)i * n + j] = (row[j] * inv).real();
    }
    return out;
}

// Complex-input variant of prefix_to_level (FFT along the e-axis).
Ten prefix_to_level_c(const Ten& frames, int I, int n, int level) {
    int e = 1 << level, q = n >> level;
    Ten Z((size_t)I * e * q);
    std::vector<cd> col(e);
    for (int i = 0; i < I; ++i)
        for (int b = 0; b < q; ++b) {
            for (int a = 0; a < e; ++a) col[a] = frames[(size_t)i * n + a * q + b];
            fft_pow2(col.data(), e, -1);
            for (int a = 0; a < e; ++a) Z[((size_t)i * e + a) * q + b] = col[a];
        }
    return Z;
}

// Complex spectra and magnitudes of windowed IQ frames [I,n].  The measured
// spectrum is retained for waterfall seeding: unlike the older magnitude-only
// reconstruction experiment, an IQ viewer already owns phase and must not run
// PGHI merely to invent it again.
void windowed_frame_spectrum_c(const std::vector<cd>& z, int L, int n, int hop,
                               Ten& spectrum_out,
                               std::vector<double>& mag_out, int& I_out) {
    auto offs = frame_offsets(L, n, hop);
    int I = (int)offs.size();
    I_out = I;
    auto win = periodic_hann(n);
    spectrum_out.assign((size_t)I * n, cd(0, 0));
    mag_out.assign((size_t)I * n, 0.0);
    std::vector<cd> frame(n);
    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < n; ++j) frame[j] = z[offs[i] + j] * win[j];
        fft_pow2(frame.data(), n, -1);
        for (int j = 0; j < n; ++j) {
            spectrum_out[(size_t)i * n + j] = frame[j];
            mag_out[(size_t)i * n + j] = std::abs(frame[j]);
        }
    }
}

// ifft of endpoint spectra keeping complex output [I,n].
Ten ifft_complex_rows(const Ten& Y, int I, int n) {
    Ten out((size_t)I * n);
    std::vector<cd> row(n);
    double inv = 1.0 / n;
    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < n; ++j) row[j] = Y[(size_t)i * n + j];
        fft_pow2(row.data(), n, +1);
        for (int j = 0; j < n; ++j) out[(size_t)i * n + j] = row[j] * inv;
    }
    return out;
}

// One independent symmetric-Hann STFT magnitude family.  This is the direct
// C++ translation of viewer/two_lattice.py::MagnitudeFamily: endpoint radial
// projection followed by windowed overlap/add into one latent waveform.
class MagnitudeProjectorC {
public:
    int L, n, hop, I;
    std::vector<int> offs;
    std::vector<double> win, target, den;
    std::vector<char> supported;

    MagnitudeProjectorC(const Ten& observed, int length, int nfft, int frame_hop)
        : L(length), n(nfft), hop(frame_hop) {
        offs = frame_offsets(L, n, hop);
        I = (int)offs.size();
        if (I <= 0) return;
        win.resize(n);
        for (int j = 0; j < n; ++j)
            win[j] = 0.5 - 0.5 * std::cos(2.0 * PI * j / (n - 1));
        target.resize((size_t)I * n);
        den.assign(L, 0.0);
        std::vector<cd> row(n);
        for (int i = 0; i < I; ++i) {
            int o = offs[i];
            for (int j = 0; j < n; ++j) {
                row[j] = observed[o + j] * win[j];
                den[o + j] += win[j] * win[j];
            }
            fft_pow2(row.data(), n, -1);
            for (int j = 0; j < n; ++j)
                target[(size_t)i * n + j] = std::abs(row[j]);
        }
        std::vector<double> positive;
        positive.reserve(L);
        for (double v : den) if (v > 0.0) positive.push_back(v);
        double median = 1.0;
        if (!positive.empty()) {
            size_t mid = positive.size() / 2;
            std::nth_element(positive.begin(), positive.begin() + mid,
                             positive.end());
            median = positive[mid];
        }
        supported.resize(L);
        for (int t = 0; t < L; ++t) supported[t] = den[t] > 1e-2 * median;
    }

    Ten project(const Ten& latent) const {
        Ten acc(L, cd(0.0, 0.0));
        std::vector<cd> row(n);
        const double inv = 1.0 / n;
        for (int i = 0; i < I; ++i) {
            int o = offs[i];
            for (int j = 0; j < n; ++j) row[j] = latent[o + j] * win[j];
            fft_pow2(row.data(), n, -1);
            for (int j = 0; j < n; ++j) {
                double a = std::abs(row[j]);
                row[j] *= target[(size_t)i * n + j] / (a + 1e-12);
            }
            fft_pow2(row.data(), n, +1);
            for (int j = 0; j < n; ++j)
                acc[o + j] += row[j] * (inv * win[j]);
        }
        Ten out(L);
        for (int t = 0; t < L; ++t)
            out[t] = supported[t] ? acc[t] / std::max(den[t], 1e-8)
                                  : latent[t];
        return out;
    }
};

// Full-spectrum PGHI (complex signals): propagate over all nfft bins, no
// Hermitian fold.  Same heap/gradient structure as pghi_phases.
// Warm-startable core. If n_warm>0, the first n_warm frames (columns i) are
// pre-seeded from warm_internal (internal [F, n_warm] phase from a previous
// tile) and flooded first, giving phase CONTINUITY across tiles. phase_internal
// (optional [F, I]) receives the pre-c0 phase for chaining the next tile.
void pghi_core(const double* mag, int I, int nfft, int n, int hop,
               int n_warm, const double* warm_internal,
               double* phase_full, double* phase_internal) {
    const int F = nfft;                       // all bins
    const double rel_tol = 1e-9, lam_factor = 0.25645;
    std::vector<double> A((size_t)F * I), s((size_t)F * I);
    double amax = 0.0;
    for (int f = 0; f < F; ++f)
        for (int i = 0; i < I; ++i) {
            double v = mag[(size_t)i * nfft + f];
            A[(size_t)f * I + i] = v;
            s[(size_t)f * I + i] = std::log(v + 1e-30);
            amax = std::max(amax, v);
        }
    double lam = lam_factor * (double)n * (double)n;
    std::vector<double> phi_t((size_t)F * I), phi_w((size_t)F * I);
    for (int f = 0; f < F; ++f) {
        double wf = (double)f / nfft;
        for (int i = 0; i < I; ++i) {
            double dsdw = 0.0, dsdt = 0.0;
            if (f > 0 && f < F - 1)
                dsdw = 0.5 * nfft * (s[(size_t)(f + 1) * I + i] - s[(size_t)(f - 1) * I + i]);
            if (i > 0 && i < I - 1)
                dsdt = (s[(size_t)f * I + (i + 1)] - s[(size_t)f * I + (i - 1)]) / (2.0 * hop);
            phi_t[(size_t)f * I + i] = 2.0 * PI * wf + dsdw / lam;
            phi_w[(size_t)f * I + i] = -lam * dsdt;
        }
    }
    std::vector<double> phase((size_t)F * I, 0.0);
    std::vector<char> active((size_t)F * I), done((size_t)F * I);
    double thr = rel_tol * (amax + 1e-30);
    for (size_t k = 0; k < A.size(); ++k) { active[k] = A[k] > thr; done[k] = !active[k]; }
    struct Node { double a; int f, i; };
    auto cmp = [](const Node& x, const Node& y) {
        if (x.a != y.a) return x.a < y.a;
        if (x.f != y.f) return x.f > y.f;
        return x.i > y.i;
    };
    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> heap(cmp);
    auto AT = [&](int f, int i) -> double { return A[(size_t)f * I + i]; };
    auto PH = [&](int f, int i) -> double& { return phase[(size_t)f * I + i]; };
    auto DN = [&](int f, int i) -> char& { return done[(size_t)f * I + i]; };
    auto PT = [&](int f, int i) -> double { return phi_t[(size_t)f * I + i]; };
    auto PW = [&](int f, int i) -> double { return phi_w[(size_t)f * I + i]; };
    auto flood = [&]() {
        while (!heap.empty()) {
            Node cur = heap.top(); heap.pop();
            int f0 = cur.f, i0 = cur.i;
            if (i0 + 1 < I && !DN(f0, i0 + 1)) {
                PH(f0, i0 + 1) = PH(f0, i0) + 0.5 * hop * (PT(f0, i0) + PT(f0, i0 + 1));
                DN(f0, i0 + 1) = 1; heap.push({AT(f0, i0 + 1), f0, i0 + 1});
            }
            if (i0 - 1 >= 0 && !DN(f0, i0 - 1)) {
                PH(f0, i0 - 1) = PH(f0, i0) - 0.5 * hop * (PT(f0, i0) + PT(f0, i0 - 1));
                DN(f0, i0 - 1) = 1; heap.push({AT(f0, i0 - 1), f0, i0 - 1});
            }
            if (f0 + 1 < F && !DN(f0 + 1, i0)) {
                PH(f0 + 1, i0) = PH(f0, i0) + 0.5 / nfft * (PW(f0, i0) + PW(f0 + 1, i0));
                DN(f0 + 1, i0) = 1; heap.push({AT(f0 + 1, i0), f0 + 1, i0});
            }
            if (f0 - 1 >= 0 && !DN(f0 - 1, i0)) {
                PH(f0 - 1, i0) = PH(f0, i0) - 0.5 / nfft * (PW(f0, i0) + PW(f0 - 1, i0));
                DN(f0 - 1, i0) = 1; heap.push({AT(f0 - 1, i0), f0 - 1, i0});
            }
        }
    };

    // Warm start: seed the overlap frames from the previous tile and flood.
    if (n_warm > 0 && warm_internal) {
        for (int i = 0; i < n_warm && i < I; ++i)
            for (int f = 0; f < F; ++f) {
                size_t k = (size_t)f * I + i;
                if (active[k]) {
                    phase[k] = warm_internal[(size_t)f * n_warm + i];
                    done[k] = 1;
                    heap.push({A[k], f, i});
                }
            }
        flood();
    }

    // Seed each remaining connected component at its exact maximum. The old
    // code sorted every cell merely to obtain these component maxima. Typical
    // STFT activity is one connected component, so an exact max scan avoids an
    // O(FI log(FI)) global sort without changing PGHI traversal or phase.
    while (true) {
        int flat = -1;
        for (int k = 0; k < (int)A.size(); ++k) {
            if (!active[k] || done[k]) continue;
            if (flat < 0 || A[k] > A[flat] ||
                (A[k] == A[flat] && k > flat)) flat = k;
        }
        if (flat < 0) break;
        int f = flat / I, i = flat % I;
        done[flat] = 1;
        heap.push({AT(f, i), f, i});
        flood();
    }

    double c0 = (n - 1) / 2.0;
    for (int i = 0; i < I; ++i)
        for (int f = 0; f < F; ++f)
            phase_full[(size_t)i * nfft + f] = PH(f, i) - 2.0 * PI * f * c0 / nfft;
    if (phase_internal)
        for (int f = 0; f < F; ++f)
            for (int i = 0; i < I; ++i)
                phase_internal[(size_t)f * I + i] = PH(f, i);
}

void pghi_phases_full(const double* mag, int I, int nfft, int n, int hop,
                      double* phase_full) {
    pghi_core(mag, I, nfft, n, hop, 0, nullptr, phase_full, nullptr);
}

// ---------------------------------------------------------------------------
// Shared state + the magnitude-domain forward/adjoint gradient.  Reused by both
// the real solver (Solver) and the complex solver (SolverC); the DIP walk and
// tap operators are complex-valued, so only seeding/consensus differ.
// ---------------------------------------------------------------------------
struct DipBase {
    int L = 0, Ib = 0, Is = 0, Dsel = 0;
    // Runtime aperture geometry (QB = QS = HS = q* = 32 are compile-time).
    int NB = 1024, HB = 128, TB = 5, EB = 32;   // long
    int NS = 128,  TS = 2, ES = 4;              // short (HS = 32)
    int NDELTA = 29;
    std::vector<int> ob, os, dsel;
    std::vector<double> win_b, win_s;
    std::vector<double> yb, ys;                 // magnitudes [Ib,NB],[Is,NS]
    std::vector<Ten> M;                         // selected [Dsel][ES,EB]
    std::vector<cd> phb, phs;                   // [QB],[QS]
    std::vector<std::vector<int>> short_index;  // [Ib][Dsel]
    std::vector<double> w_short_occ;            // [Ib*Dsel]
    std::vector<double> cons_den;               // [L]
    std::vector<char> covered;                  // [L]

    // Everything except yb/ys (which are real- vs complex-specific).  nb/ns are
    // the aperture window sizes (multiples of q*=32, ns < nb, eb-es >= 4).  Pass
    // nds=0 for the default central-5 deltas centered at (EB-ES)/2.
    void setup_common(int Lin, const int* dselin, int nds, bool renorm,
                      int nb, int ns) {
        L = Lin;
        NB = nb; NS = ns;
        EB = NB / QSTAR; ES = NS / QSTAR;
        // Snap ES to the largest power of two with ES <= EB-4 (ES = 2^TS, and
        // the central-5 deltas need EB-ES >= 4). Guards against bad apertures.
        {
            int want = std::min(ES, EB - 4);
            int e = 2;
            while (e * 2 <= want) e *= 2;
            ES = std::max(2, e);
            NS = ES * QSTAR;
        }
        TB = (int)std::lround(std::log2((double)EB));
        TS = (int)std::lround(std::log2((double)ES));
        HB = NB / 8;
        NDELTA = EB - ES + 1;

        if (nds > 0) {
            dsel.assign(dselin, dselin + nds);
        } else {
            int center = (EB - ES) / 2;             // central-5 default
            for (int c = center - 2; c <= center + 2; ++c)
                dsel.push_back(std::min(std::max(c, 0), EB - ES));
        }
        ob = frame_offsets(L, NB, HB);
        os = frame_offsets(L, NS, HS);
        Ib = (int)ob.size(); Is = (int)os.size(); Dsel = (int)dsel.size();
        win_b = periodic_hann(NB); win_s = periodic_hann(NS);

        M = attachment_mats_selected(EB, ES, dsel);
        phb.resize(QB); for (int k = 0; k < QB; ++k)
            phb[k] = cd(std::cos(2 * PI * k / NB), std::sin(2 * PI * k / NB));
        phs.resize(QS); for (int k = 0; k < QS; ++k)
            phs[k] = cd(std::cos(2 * PI * k / NS), std::sin(2 * PI * k / NS));

        short_index.assign(Ib, std::vector<int>(Dsel));
        std::vector<double> occ(Is, 0.0);
        for (int i = 0; i < Ib; ++i)
            for (int c = 0; c < Dsel; ++c) {
                int si = i * (HB / HS) + dsel[c];
                short_index[i][c] = si; occ[si] += 1.0;
            }
        for (double& o : occ) o = std::max(o, 1.0);
        w_short_occ.assign((size_t)Ib * Dsel, 0.0);
        double ren_fac = renorm ? ((double)NDELTA / Dsel) : 1.0;
        for (int i = 0; i < Ib; ++i)
            for (int c = 0; c < Dsel; ++c)
                w_short_occ[(size_t)i * Dsel + c] =
                    (((double)NB / NS) / occ[short_index[i][c]]) * ren_fac;

        cons_den.assign(L, 0.0);
        for (int i = 0; i < Ib; ++i)
            for (int j = 0; j < NB; ++j) cons_den[ob[i] + j] += 1.0;
        covered.assign(L, 0);
        for (int t = 0; t < L; ++t) {
            covered[t] = cons_den[t] > 0.5;
            if (cons_den[t] < 1e-8) cons_den[t] = 1e-8;
        }
    }

    Ten forward_long(const Ten& Z) const {
        Ten H = tap3(Z, Ib, EB, QB, phb);
        return finish_from_level(H, Ib, EB, QB);
    }

    Ten forward_short_occ(const Ten& Z, Ten& Zs_out) const {
        Zs_out.assign((size_t)Ib * Dsel * ES * QS, cd(0,0));
        for (int i = 0; i < Ib; ++i)
            for (int d = 0; d < Dsel; ++d)
                for (int a = 0; a < ES; ++a)
                    for (int qq = 0; qq < QS; ++qq) {
                        cd acc(0,0);
                        for (int b = 0; b < EB; ++b)
                            acc += M[d][a * EB + b] * Z[((size_t)i * EB + b) * QB + qq];
                        Zs_out[(((size_t)i * Dsel + d) * ES + a) * QS + qq] = acc;
                    }
        Ten Hs = tap3(Zs_out, Ib * Dsel, ES, QS, phs);
        return finish_from_level(Hs, Ib * Dsel, ES, QS);
    }

    double loss_grad(const Ten& Z, Ten& G) const {
        G.assign((size_t)Ib * EB * QB, cd(0,0));
        double loss = 0.0;

        Ten Yb = forward_long(Z);
        Ten Rb((size_t)Ib * NB);
        for (int i = 0; i < Ib; ++i)
            for (int j = 0; j < NB; ++j) {
                cd yv = Yb[(size_t)i * NB + j];
                double mb = std::abs(yv);
                double tgt = yb[(size_t)i * NB + j];
                double diff = mb - tgt;
                loss += diff * diff;
                Rb[(size_t)i * NB + j] = (1.0 - tgt / (mb + 1e-12)) * yv;
            }
        Ten Gb = back_to_level(Rb, Ib, NB, TB);
        double gsc = ((double)NB / EB) * 2.0;
        for (auto& v : Gb) v *= gsc;
        Ten GbA = tap3_adj(Gb, Ib, EB, QB, phb);
        for (size_t k = 0; k < G.size(); ++k) G[k] += GbA[k];

        Ten Zs;
        Ten Ys = forward_short_occ(Z, Zs);
        Ten Rs((size_t)Ib * Dsel * NS);
        for (int i = 0; i < Ib; ++i)
            for (int d = 0; d < Dsel; ++d) {
                double W = w_short_occ[(size_t)i * Dsel + d];
                int si = short_index[i][d];
                for (int j = 0; j < NS; ++j) {
                    cd yv = Ys[((size_t)i * Dsel + d) * NS + j];
                    double ms = std::abs(yv);
                    double tgt = ys[(size_t)si * NS + j];
                    double diff = ms - tgt;
                    loss += W * diff * diff;
                    Rs[((size_t)i * Dsel + d) * NS + j] = (2.0 * W) * (1.0 - tgt / (ms + 1e-12)) * yv;
                }
            }
        Ten Gs = back_to_level(Rs, Ib * Dsel, NS, TS);
        double gss = (double)NS / ES;
        for (auto& v : Gs) v *= gss;
        Ten GsA = tap3_adj(Gs, Ib * Dsel, ES, QS, phs);
        for (int i = 0; i < Ib; ++i)
            for (int b = 0; b < EB; ++b)
                for (int qq = 0; qq < QS; ++qq) {
                    cd acc(0,0);
                    for (int d = 0; d < Dsel; ++d)
                        for (int a = 0; a < ES; ++a)
                            acc += std::conj(M[d][a * EB + b]) *
                                   GsA[(((size_t)i * Dsel + d) * ES + a) * QS + qq];
                    G[((size_t)i * EB + b) * QB + qq] += acc;
                }
        return loss;
    }
};

// ---------------------------------------------------------------------------
// Real solver (validated against active_delta_center5_cpp_reference.py).
// ---------------------------------------------------------------------------
struct Solver : DipBase {
    std::vector<double> x;

    Solver(const double* xin, int Lin, const int* ds, int nds, bool ren)
        : x(xin, xin + Lin) {
        setup_common(Lin, ds, nds, ren, 1024, 128);   // reference geometry
        int t;
        windowed_frame_mag(x, L, NB, HB, yb, t);
        windowed_frame_mag(x, L, NS, HS, ys, t);
    }

    Ten prefix_rect(const std::vector<double>& u) const {
        std::vector<double> fr((size_t)Ib * NB);
        for (int i = 0; i < Ib; ++i)
            for (int j = 0; j < NB; ++j) fr[(size_t)i * NB + j] = u[ob[i] + j];
        return prefix_to_level(fr, Ib, NB, TB);
    }

    std::vector<double> rect_consensus(const Ten& Z) const {
        Ten fr = inverse_prefix(Z, Ib, EB, QB);
        std::vector<double> acc(L, 0.0);
        for (int i = 0; i < Ib; ++i)
            for (int j = 0; j < NB; ++j) acc[ob[i] + j] += fr[(size_t)i * NB + j].real();
        std::vector<double> u(L);
        for (int t = 0; t < L; ++t) u[t] = covered[t] ? acc[t] / cons_den[t] : 0.0;
        return u;
    }

    // PGHI record-OLA seed for one aperture (n,hop) given its magnitudes.
    std::vector<double> record_seed(const std::vector<double>& mag, int Imag,
                                    int n, int hop, const std::vector<int>& offs,
                                    const std::vector<double>& win) const {
        std::vector<double> phase((size_t)Imag * n);
        pghi_phases(mag.data(), Imag, n, n, hop, phase.data());
        Ten Y((size_t)Imag * n);
        for (size_t k = 0; k < Y.size(); ++k)
            Y[k] = mag[k] * cd(std::cos(phase[k]), std::sin(phase[k]));
        std::vector<double> frames = ifft_real_rows(Y, Imag, n);
        std::vector<double> wsq(L, 0.0);
        for (int i = 0; i < Imag; ++i)
            for (int j = 0; j < n; ++j) wsq[offs[i] + j] += win[j] * win[j];
        std::vector<double> pos;
        for (double v : wsq) if (v > 0) pos.push_back(v);
        std::sort(pos.begin(), pos.end());
        double med = pos.empty() ? 1.0 :
            (pos.size() % 2 ? pos[pos.size()/2] : 0.5*(pos[pos.size()/2-1]+pos[pos.size()/2]));
        std::vector<double> acc(L, 0.0);
        for (int i = 0; i < Imag; ++i)
            for (int j = 0; j < n; ++j)
                acc[offs[i] + j] += frames[(size_t)i * n + j] * win[j];
        std::vector<double> u0(L);
        for (int t = 0; t < L; ++t) {
            double den = std::max(wsq[t], 1e-8);
            u0[t] = (wsq[t] > 1e-2 * med) ? acc[t] / den : 0.0;
        }
        return u0;
    }

    std::vector<double> seed_u0(bool fuse) const {
        std::vector<double> uL = record_seed(yb, Ib, NB, HB, ob, win_b);
        if (!fuse) return uL;
        // Fuse the short-aperture record seed (dense frames -> phase-true basin)
        // after resolving its global sign against the long seed.
        std::vector<double> uS = record_seed(ys, Is, NS, HS, os, win_s);
        double d = 0.0;
        for (int t = 0; t < L; ++t) d += uS[t] * uL[t];
        double s = (d < 0.0) ? -1.0 : 1.0;
        for (int t = 0; t < L; ++t) uL[t] = 0.5 * (uL[t] + s * uS[t]);
        return uL;
    }

    std::vector<double> run(double steepest_scale, int n_steps, bool fuse,
                            double& loss0_out) {
        std::vector<double> u = seed_u0(fuse);
        Ten Z = prefix_rect(u);
        int steps = std::max(1, n_steps);
        for (int s = 0; s < steps; ++s) {
            Ten G;
            double loss = loss_grad(Z, G);
            if (s == 0) loss0_out = loss;
            for (size_t k = 0; k < Z.size(); ++k) Z[k] -= steepest_scale * G[k];
            u = rect_consensus(Z);
            Z = prefix_rect(u);
        }
        return u;
    }
};

// ---------------------------------------------------------------------------
// Complex solver: reconstructs a complex baseband signal from its complex
// STFT magnitudes.  Full-spectrum PGHI (no Hermitian fold) and complex
// consensus mean the magnitude waterfall is invariant to the residual global
// phase -- no per-frame mirror leakage.
// ---------------------------------------------------------------------------
struct SolverC : DipBase {
    std::vector<cd> z;
    Ten Yb_measured, Ys_measured;

    SolverC(const cd* zin, int Lin, const int* ds, int nds, bool ren,
            int nb, int ns)
        : z(zin, zin + Lin) {
        setup_common(Lin, ds, nds, ren, nb, ns);
        int t;
        windowed_frame_spectrum_c(z, L, NB, HB, Yb_measured, yb, t);
        windowed_frame_spectrum_c(z, L, NS, HS, Ys_measured, ys, t);
    }

    Ten prefix_rect(const Ten& u) const {
        Ten fr((size_t)Ib * NB);
        for (int i = 0; i < Ib; ++i)
            for (int j = 0; j < NB; ++j) fr[(size_t)i * NB + j] = u[ob[i] + j];
        return prefix_to_level_c(fr, Ib, NB, TB);
    }

    Ten rect_consensus(const Ten& Z) const {
        Ten fr = inverse_prefix(Z, Ib, EB, QB);
        Ten acc(L, cd(0,0));
        for (int i = 0; i < Ib; ++i)
            for (int j = 0; j < NB; ++j) acc[ob[i] + j] += fr[(size_t)i * NB + j];
        Ten u(L, cd(0,0));
        for (int t = 0; t < L; ++t) u[t] = covered[t] ? acc[t] / cons_den[t] : cd(0,0);
        return u;
    }

    Ten record_seed(const std::vector<double>& mag, int Imag, int n, int hop,
                    const std::vector<int>& offs, const std::vector<double>& win,
                    int n_warm = 0, const double* warm_internal = nullptr,
                    double* internal_out = nullptr) const {
        std::vector<double> phase((size_t)Imag * n);
        pghi_core(mag.data(), Imag, n, n, hop, n_warm, warm_internal,
                  phase.data(), internal_out);
        Ten Y((size_t)Imag * n);
        for (size_t k = 0; k < Y.size(); ++k)
            Y[k] = mag[k] * cd(std::cos(phase[k]), std::sin(phase[k]));
        Ten frames = ifft_complex_rows(Y, Imag, n);
        std::vector<double> wsq(L, 0.0);
        for (int i = 0; i < Imag; ++i)
            for (int j = 0; j < n; ++j) wsq[offs[i] + j] += win[j] * win[j];
        std::vector<double> pos;
        for (double v : wsq) if (v > 0) pos.push_back(v);
        std::sort(pos.begin(), pos.end());
        double med = pos.empty() ? 1.0 :
            (pos.size() % 2 ? pos[pos.size()/2] : 0.5*(pos[pos.size()/2-1]+pos[pos.size()/2]));
        Ten acc(L, cd(0,0));
        for (int i = 0; i < Imag; ++i)
            for (int j = 0; j < n; ++j)
                acc[offs[i] + j] += frames[(size_t)i * n + j] * win[j];
        Ten u0(L, cd(0,0));
        for (int t = 0; t < L; ++t) {
            double den = std::max(wsq[t], 1e-8);
            u0[t] = (wsq[t] > 1e-2 * med) ? acc[t] / den : cd(0,0);
        }
        return u0;
    }

    Ten seed_u0(bool fuse, int n_warm = 0,
                const double* warm_internal = nullptr,
                double* internal_out = nullptr) const {
        // Long aperture carries the warm phase field (continuity across tiles).
        Ten uL = record_seed(yb, Ib, NB, HB, ob, win_b, n_warm, warm_internal,
                             internal_out);
        if (!fuse) return uL;
        Ten uS = record_seed(ys, Is, NS, HS, os, win_s);
        cd dot(0,0);
        for (int t = 0; t < L; ++t) dot += std::conj(uS[t]) * uL[t];
        cd rot = std::abs(dot) > 1e-30 ? dot / std::abs(dot) : cd(1,0);
        for (int t = 0; t < L; ++t) uL[t] = 0.5 * (uL[t] + rot * uS[t]);
        return uL;
    }

    // Warm-started solve. warm_internal is the previous tile's long-aperture
    // internal PGHI phase for its first n_warm frames (aligned to this tile's
    // frames 0..n_warm-1); internal_out receives this tile's [NB, Ib] internal
    // phase for chaining. Pass n_warm=0 for a cold solve.
    Ten run_warm(double steepest_scale, int n_steps, bool fuse, int n_warm,
                 const double* warm_internal, double* internal_out,
                 double& loss0_out) {
        Ten u = seed_u0(fuse, n_warm, warm_internal, internal_out);
        Ten Z = prefix_rect(u);
        int steps = std::max(1, n_steps);
        for (int s = 0; s < steps; ++s) {
            Ten G;
            double loss = loss_grad(Z, G);
            if (s == 0) loss0_out = loss;
            for (size_t k = 0; k < Z.size(); ++k) Z[k] -= steepest_scale * G[k];
            u = rect_consensus(Z);
            Z = prefix_rect(u);
        }
        return u;
    }

    int n_frames_long() const { return Ib; }

    Ten run(double steepest_scale, int n_steps, bool fuse, double& loss0_out) {
        Ten u = seed_u0(fuse);
        Ten Z = prefix_rect(u);
        int steps = std::max(1, n_steps);
        for (int s = 0; s < steps; ++s) {
            Ten G;
            double loss = loss_grad(Z, G);
            if (s == 0) loss0_out = loss;
            for (size_t k = 0; k < Z.size(); ++k) Z[k] -= steepest_scale * G[k];
            u = rect_consensus(Z);
            Z = prefix_rect(u);
        }
        return u;
    }

    // Super-resolution WATERFALL readout: solve the active-delta state (record
    // OLA seed + up to n_steps fixed L5 gradient steps, per-frame, NO record
    // consensus) and read the long-aperture endpoint magnitude DIRECTLY off the
    // refined paused state: |suffix_from_level(tap3(Zb))|.  The short-window
    // constraints (attached inside each long state via M_delta) refine Zb so the
    // long readout carries fine-time detail -- the multi-aperture fusion.  Never
    // synthesizes a record.  Returns [Ib, NB] magnitudes.
    std::vector<double> waterfall(double steepest_scale, int n_steps, bool fuse) {
        Ten Z = prefix_rect(seed_u0(fuse));
        int steps = std::max(0, n_steps);
        for (int s = 0; s < steps; ++s) {
            Ten G;
            loss_grad(Z, G);
            for (size_t k = 0; k < Z.size(); ++k) Z[k] -= steepest_scale * G[k];
        }
        Ten Y = forward_long(Z);              // [Ib, NB] endpoint spectrum
        std::vector<double> mag((size_t)Ib * NB);
        for (size_t k = 0; k < mag.size(); ++k) mag[k] = std::abs(Y[k]);
        return mag;
    }

    // Direct measured seed: the paused-DIP state OF THE RECORD ITSELF.
    // The former tap3_adj pullback of Yb_measured was NOT spectrum-consistent:
    // tap3 is pointwise periodic-Hann in frame time, so the readout round-trip
    // applied tap3 . tap3_adj = Hann^2, widening the mainlobe from 4 to 6 bins
    // and erasing tone pairs the plain Hann STFT resolves (measured: a 1.5 kHz
    // pair at NB=1024/Fs=456k lost -17.5 dB -> -4 dB of valley).  The adjoint
    // is not the inverse.  prefix_rect(z) is exact: forward_long reproduces
    // the measured Hann spectra at zero steps, phase-true, and the short
    // endpoints attach in-state through M_delta as before.
    Ten direct_seed() const {
        return prefix_rect(z);
    }

    // Separable projection pullbacks with the radial residual R = y*Y/|Y| - Y,
    // pulled back through the suffix adjoint (q* * back_to_level), tap3 adjoint,
    // and (short) M_delta adjoint.  These are ADDED to Zb (projection style).
    void proj_pullback(const Ten& Z, Ten& Glong, Ten& Gshort) const {
        // Long.
        Ten Yb = forward_long(Z);
        Ten Rb((size_t)Ib * NB);
        for (int i = 0; i < Ib; ++i)
            for (int j = 0; j < NB; ++j) {
                cd yv = Yb[(size_t)i * NB + j];
                double m = std::abs(yv);
                double y = yb[(size_t)i * NB + j];
                Rb[(size_t)i * NB + j] = y * yv / (m + 1e-12) - yv;
            }
        // Minimum-norm projection: since P P^H = q* I, the correction is
        // P^{-1} R = back_to_level(R) with NO q* factor (the q* belongs to the
        // squared-loss gradient path in loss_grad, kept separate).
        Ten Gb = back_to_level(Rb, Ib, NB, TB);
        Glong = tap3_adj(Gb, Ib, EB, QB, phb);

        // Short (attached via M_delta), weighted by w_short_occ.
        Ten Zs;
        Ten Ys = forward_short_occ(Z, Zs);
        Ten Rs((size_t)Ib * Dsel * NS);
        for (int i = 0; i < Ib; ++i)
            for (int d = 0; d < Dsel; ++d) {
                double W = w_short_occ[(size_t)i * Dsel + d];
                int si = short_index[i][d];
                for (int j = 0; j < NS; ++j) {
                    cd yv = Ys[((size_t)i * Dsel + d) * NS + j];
                    double m = std::abs(yv);
                    double y = ys[(size_t)si * NS + j];
                    Rs[((size_t)i * Dsel + d) * NS + j] = W * (y * yv / (m + 1e-12) - yv);
                }
            }
        Ten Gs = back_to_level(Rs, Ib * Dsel, NS, TS);   // no q* (projection)
        Gs = tap3_adj(Gs, Ib * Dsel, ES, QS, phs);
        Gshort.assign((size_t)Ib * EB * QB, cd(0, 0));
        for (int i = 0; i < Ib; ++i)
            for (int b = 0; b < EB; ++b)
                for (int q = 0; q < QS; ++q) {
                    cd acc(0, 0);
                    for (int d = 0; d < Dsel; ++d)
                        for (int a = 0; a < ES; ++a)
                            acc += std::conj(M[d][a * EB + b]) *
                                   Gs[(((size_t)i * Dsel + d) * ES + a) * QS + q];
                    Gshort[((size_t)i * EB + b) * QB + q] = acc;
                }
    }

    // Projection-style waterfall. mode: 0 seed only, 1 short-only, 2 long-only,
    // 3 long+short. direct: use the direct tap-adj seed (else record OLA).
    // Per-frame averaged update Zb += (alpha / (1 + sum_d w)) * pullback.
    // Seed + up to n_steps projection updates -> refined paused state Zb.
    Ten refine_state(int mode, double alpha, int n_steps, bool direct) {
        Ten Z;
        if (direct) Z = direct_seed();
        else Z = prefix_rect(seed_u0(false));
        std::vector<double> norm(Ib, 1.0);
        for (int i = 0; i < Ib; ++i) {
            double sw = 0.0;
            for (int d = 0; d < Dsel; ++d) sw += w_short_occ[(size_t)i * Dsel + d];
            norm[i] = 1.0 + sw;
        }
        int steps = std::max(0, n_steps);
        for (int s = 0; s < steps; ++s) {
            Ten Gl, Gs;
            proj_pullback(Z, Gl, Gs);
            for (int i = 0; i < Ib; ++i) {
                double a = alpha / norm[i];
                for (int b = 0; b < EB; ++b)
                    for (int q = 0; q < QB; ++q) {
                        size_t k = ((size_t)i * EB + b) * QB + q;
                        cd upd(0, 0);
                        if (mode == 1) upd = Gs[k];
                        else if (mode == 2) upd = Gl[k];
                        else if (mode == 3) upd = Gl[k] + Gs[k];
                        Z[k] += a * upd;
                    }
            }
        }
        return Z;
    }

    std::vector<double> waterfall_proj(int mode, double alpha, int n_steps,
                                       bool direct) {
        Ten Y = forward_long(refine_state(mode, alpha, n_steps, direct));
        std::vector<double> mag((size_t)Ib * NB);
        for (size_t k = 0; k < mag.size(); ++k) mag[k] = std::abs(Y[k]);
        return mag;
    }

    // F1xT2 fusion readout: off the SAME refined state, read the long endpoint
    // (fine frequency, coarse time) [Ib,NB] and the short endpoints at the
    // CONSTRAINED deltas only (fine time, coarse frequency) [Ib,Dsel,NS] -- so
    // every emitted fine-time gate is a measured, fitted short, not a
    // model-implied one. Dsel = the active delta set.
    void fusion_readout(int mode, double alpha, int n_steps, bool direct,
                        std::vector<double>& long_mag,
                        std::vector<double>& short_mag) {
        Ten Z = refine_state(mode, alpha, n_steps, direct);
        Ten Yl = forward_long(Z);
        long_mag.resize((size_t)Ib * NB);
        for (size_t k = 0; k < long_mag.size(); ++k) long_mag[k] = std::abs(Yl[k]);

        Ten Zs;
        Ten Ys = forward_short_occ(Z, Zs);               // [Ib, Dsel, NS]
        short_mag.assign((size_t)Ib * Dsel * NS, 0.0);
        for (size_t k = 0; k < short_mag.size(); ++k) short_mag[k] = std::abs(Ys[k]);
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------
extern "C" {

// Full one-shot solve.  x has L doubles; u_out has L doubles; loss0 optional.
// dsel/ndsel select active deltas (pass null for the default center-5).
IQW_API void iqw_dip_run(const double* x, int L, const int* dsel, int ndsel,
                         int renorm, double steepest_scale, int n_steps,
                         int fuse_seed, double* u_out, double* loss0_out) {
    static const int CENTER5[5] = {12, 13, 14, 15, 16};
    const int* ds = dsel ? dsel : CENTER5;
    int nds = dsel ? ndsel : 5;
    Solver S(x, L, ds, nds, renorm != 0);
    double loss0 = 0.0;
    std::vector<double> u = S.run(steepest_scale, n_steps, fuse_seed != 0, loss0);
    std::copy(u.begin(), u.end(), u_out);
    if (loss0_out) *loss0_out = loss0;
}

// Complex-generalized one-shot solve. z_in / u_out are interleaved complex
// (re,im) arrays of length L (i.e. 2L doubles each). Reconstructs a complex
// baseband signal from its complex STFT magnitudes; the magnitude waterfall is
// invariant to the residual global phase (no mirror ambiguity).
IQW_API void iqw_dip_run_complex(const double* z_in, int L, const int* dsel,
                                 int ndsel, int renorm, double steepest_scale,
                                 int n_steps, int fuse_seed, int nb, int ns,
                                 double* u_out, double* loss0_out) {
    int nds = dsel ? ndsel : 0;                    // 0 => geometry-default deltas
    const cd* z = reinterpret_cast<const cd*>(z_in);
    SolverC S(z, L, dsel, nds, renorm != 0, nb, ns);
    double loss0 = 0.0;
    Ten u = S.run(steepest_scale, n_steps, fuse_seed != 0, loss0);
    cd* uo = reinterpret_cast<cd*>(u_out);
    std::copy(u.begin(), u.end(), uo);
    if (loss0_out) *loss0_out = loss0;
}

// Shared-seeded unified two-lattice solve.  The shared seed is exactly the
// active_delta_center5_fast1 complex path above.  Each unified step is the
// original Python P_short(P_long(momentum)) waveform projection, with
// independent symmetric-Hann frame lattices.  h_short is normally ns/2.
IQW_API void iqw_dip_unified(const double* z_in, int L, const int* dsel,
                             int ndsel, int renorm, double steepest_scale,
                             int shared_steps, int nb, int ns, int h_short,
                             int unified_steps, double beta, double* u_out,
                             double* loss0_out) {
    int nds = dsel ? ndsel : 0;
    const cd* zp = reinterpret_cast<const cd*>(z_in);
    Ten observed(zp, zp + L);
    SolverC S(zp, L, dsel, nds, renorm != 0, nb, ns);
    double loss0 = 0.0;
    Ten latent = S.run(steepest_scale, shared_steps, false, loss0);
    MagnitudeProjectorC long_family(observed, L, nb, nb / 8);
    MagnitudeProjectorC short_family(observed, L, ns, h_short);
    Ten previous = latent;
    int steps = std::max(0, unified_steps);
    for (int step = 0; step < steps; ++step) {
        Ten trial(L);
        for (int t = 0; t < L; ++t)
            trial[t] = latent[t] + beta * (latent[t] - previous[t]);
        previous = latent;
        latent = short_family.project(long_family.project(trial));
    }
    cd cross(0.0, 0.0);
    for (int t = 0; t < L; ++t) cross += std::conj(latent[t]) * observed[t];
    if (std::abs(cross) > 1e-30) {
        cd rot = cross / std::abs(cross);
        for (cd& v : latent) v *= rot;
    }
    cd* out = reinterpret_cast<cd*>(u_out);
    std::copy(latent.begin(), latent.end(), out);
    if (loss0_out) *loss0_out = loss0;
}

// Number of long-aperture frames for record length L and window nb (hb=nb/8).
IQW_API int iqw_dip_frames_long(int L, int nb) {
    int hb = nb / 8;
    return (L - nb) / hb + 1;
}

// Super-resolution waterfall readout for a complex record. z_in is interleaved
// complex (2L doubles). out_mag receives [Ib, NB] float64 magnitudes read
// directly off the refined active-delta state (long-aperture endpoint), where
// Ib = iqw_dip_frames_long(L), NB = 1024. n_steps=0 gives the pure seed readout.
IQW_API void iqw_dip_waterfall_complex(const double* z_in, int L,
                                       const int* dsel, int ndsel, int renorm,
                                       double steepest_scale, int n_steps,
                                       int fuse_seed, int nb, int ns,
                                       double* out_mag) {
    int nds = dsel ? ndsel : 0;
    const cd* z = reinterpret_cast<const cd*>(z_in);
    SolverC S(z, L, dsel, nds, renorm != 0, nb, ns);
    std::vector<double> m = S.waterfall(steepest_scale, n_steps, fuse_seed != 0);
    std::copy(m.begin(), m.end(), out_mag);
}

// Projection-style super-res / diagnostic readout. mode: 0 seed, 1 short-only,
// 2 long-only, 3 long+short. direct_seed uses the direct tap-adjoint seed.
IQW_API void iqw_dip_waterfall_proj(const double* z_in, int L, const int* dsel,
                                    int ndsel, int renorm, int mode, double alpha,
                                    int n_steps, int direct_seed, int nb, int ns,
                                    double* out_mag) {
    int nds = dsel ? ndsel : 0;
    const cd* z = reinterpret_cast<const cd*>(z_in);
    SolverC S(z, L, dsel, nds, renorm != 0, nb, ns);
    std::vector<double> m = S.waterfall_proj(mode, alpha, n_steps, direct_seed != 0);
    std::copy(m.begin(), m.end(), out_mag);
}

// F1xT2 fusion readout. long_out receives [Ib, nb] (fine freq); short_out
// receives [Ib, NDELTA, ns] (fine time), NDELTA = nb/32 - ns/32 + 1, both off
// the same refined state. The caller fuses long-freq with short-time.
IQW_API void iqw_dip_fusion(const double* z_in, int L, const int* dsel,
                            int ndsel, int renorm, int mode, double alpha,
                            int n_steps, int direct_seed, int nb, int ns,
                            double* long_out, double* short_out) {
    int nds = dsel ? ndsel : 0;
    const cd* z = reinterpret_cast<const cd*>(z_in);
    SolverC S(z, L, dsel, nds, renorm != 0, nb, ns);
    std::vector<double> lm, sm;
    S.fusion_readout(mode, alpha, n_steps, direct_seed != 0, lm, sm);
    std::copy(lm.begin(), lm.end(), long_out);
    std::copy(sm.begin(), sm.end(), short_out);
}

// Warm-started complex solve. warm_internal (may be null) is the previous
// tile's long-aperture internal PGHI phase [NB, n_warm] for the overlap frames;
// internal_out (may be null) receives this tile's [NB, Ib] internal phase for
// chaining the next tile. z_in/u_out are interleaved complex (2L doubles).
IQW_API void iqw_dip_run_complex_warm(const double* z_in, int L, const int* dsel,
                                      int ndsel, int renorm, double steepest_scale,
                                      int n_steps, int fuse_seed, int nb, int ns,
                                      int n_warm, const double* warm_internal,
                                      double* internal_out, double* u_out,
                                      double* loss0_out) {
    int nds = dsel ? ndsel : 0;
    const cd* z = reinterpret_cast<const cd*>(z_in);
    SolverC S(z, L, dsel, nds, renorm != 0, nb, ns);
    double loss0 = 0.0;
    Ten u = S.run_warm(steepest_scale, n_steps, fuse_seed != 0, n_warm,
                       warm_internal, internal_out, loss0);
    cd* uo = reinterpret_cast<cd*>(u_out);
    std::copy(u.begin(), u.end(), uo);
    if (loss0_out) *loss0_out = loss0;
}

// Just the PGHI record seed u0 (for bisecting validation).
IQW_API void iqw_dip_seed(const double* x, int L, double* u0_out) {
    static const int CENTER5[5] = {12, 13, 14, 15, 16};
    Solver S(x, L, CENTER5, 5, true);
    std::vector<double> u0 = S.seed_u0(false);
    std::copy(u0.begin(), u0.end(), u0_out);
}

// Windowed-frame magnitudes yb [Ib*NB] and ys [Is*NS] (for validation).
IQW_API void iqw_dip_magnitudes(const double* x, int L, double* yb_out,
                                double* ys_out) {
    std::vector<double> xv(x, x + L), yb, ys;
    int Ib, Is;
    windowed_frame_mag(xv, L, 1024, 128, yb, Ib);   // reference geometry
    windowed_frame_mag(xv, L, 128, 32, ys, Is);
    std::copy(yb.begin(), yb.end(), yb_out);
    std::copy(ys.begin(), ys.end(), ys_out);
}

// loss0 at the seed (for validation).
IQW_API double iqw_dip_loss0(const double* x, int L) {
    static const int CENTER5[5] = {12, 13, 14, 15, 16};
    Solver S(x, L, CENTER5, 5, true);
    Ten Z0 = S.prefix_rect(S.seed_u0(false));
    Ten G;
    return S.loss_grad(Z0, G);
}

}  // extern "C"
