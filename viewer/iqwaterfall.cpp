// iqwaterfall.cpp
//
// Monolithic high-performance backend for an IQ file reader / zoomable
// waterfall viewer.  Exposes a flat C ABI (loadable via ctypes) built on top
// of the project's BFFT real-FFT kernel.
//
// Contents
// --------
//   1. Sample-format conversion (port of the C# WaveFile `convert_to_float`
//      reference, covering u8/s8/s16/s24/s32/f32).
//   2. IQ source: WAV-with-header auto-detect, or raw interleaved IQ with a
//      caller-specified sample format and rate.  Random-access windowed reads.
//   3. Waterfall engine: per-frame windowed complex FFT (via two BFFT real
//      FFTs), fftshift, magnitude -> dB.  This is the streaming display core.
//
// The DIP / PGHI reconstruction algorithm (active_delta_center5) is ported in
// a companion translation unit; this file is the viewer/data-path core.
//
// Build (macOS):
//   c++ -O3 -std=c++17 -fPIC -shared -I../include iqwaterfall.cpp \
//       ../build/libbfft.a -o libiqwaterfall.dylib
//
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

#include <bfft/bfft.h>
#include <bfft/fct.h>

#if defined(_WIN32)
#define IQW_API __declspec(dllexport)
#else
#define IQW_API __attribute__((visibility("default")))
#endif

// ---------------------------------------------------------------------------
// Sample formats.  IQW_FMT_AUTO parses a WAV/RIFF header; the rest are raw.
// ---------------------------------------------------------------------------
enum iqw_fmt {
    IQW_FMT_AUTO = 0,
    IQW_FMT_U8   = 1,   // unsigned 8-bit (WAV PCM u8, rtl-sdr style)
    IQW_FMT_S8   = 2,   // signed 8-bit
    IQW_FMT_S16  = 3,   // signed 16-bit LE
    IQW_FMT_S24  = 4,   // signed 24-bit LE
    IQW_FMT_S32  = 5,   // signed 32-bit LE
    IQW_FMT_F32  = 6,   // 32-bit float
};

static int bytes_per_sample(int fmt) {
    switch (fmt) {
        case IQW_FMT_U8:  return 1;
        case IQW_FMT_S8:  return 1;
        case IQW_FMT_S16: return 2;
        case IQW_FMT_S24: return 3;
        case IQW_FMT_S32: return 4;
        case IQW_FMT_F32: return 4;
        default:          return 0;
    }
}

// ---------------------------------------------------------------------------
// 8-bit LUT + adaptive DC offset.  Direct port of the reference convert path.
// ---------------------------------------------------------------------------
static float g_lut8[256];
static bool  g_lut8_ready = false;
static void init_lut8() {
    if (g_lut8_ready) return;
    for (int i = 0; i < 256; ++i)
        g_lut8[i] = (static_cast<float>(i) - 127.5f) * (1.0f / 127.5f);
    g_lut8_ready = true;
}

// Convert `length` scalar samples (channels flattened) from `in` of format
// `fmt` into `out` floats.  `qoff` carries the adaptive 8-bit offset across
// calls (may be null for the other formats).
static void convert_to_float(const void* in, float* out, int64_t length,
                             int fmt, float* qoff) {
    switch (fmt) {
        case IQW_FMT_F32: {
            std::memcpy(out, in, static_cast<size_t>(length) * sizeof(float));
            break;
        }
        case IQW_FMT_S32: {
            const float scale = 1.0f / 2147483648.0f;      // 1/2^31
            const int32_t* p = static_cast<const int32_t*>(in);
            for (int64_t i = 0; i < length; ++i) out[i] = p[i] * scale;
            break;
        }
        case IQW_FMT_S24: {
            const float scale = 1.0f / 8388608.0f;          // 1/2^23
            const uint8_t* p = static_cast<const uint8_t*>(in);
            for (int64_t i = 0; i < length; ++i) {
                int32_t v = (int32_t)p[3*i] | ((int32_t)p[3*i+1] << 8) |
                            ((int32_t)p[3*i+2] << 16);
                if (v & 0x800000) v |= ~0xFFFFFF;           // sign extend
                out[i] = v * scale;
            }
            break;
        }
        case IQW_FMT_S16: {
            const float scale = 1.0f / 32768.0f;            // 1/2^15
            const int16_t* p = static_cast<const int16_t*>(in);
            for (int64_t i = 0; i < length; ++i) out[i] = p[i] * scale;
            break;
        }
        case IQW_FMT_S8: {
            const float scale = 1.0f / 128.0f;
            const int8_t* p = static_cast<const int8_t*>(in);
            for (int64_t i = 0; i < length; ++i) out[i] = p[i] * scale;
            break;
        }
        case IQW_FMT_U8: {
            init_lut8();
            const uint8_t* p = static_cast<const uint8_t*>(in);
            float offset = qoff ? *qoff : 0.0f;
            for (int64_t i = 0; i < length; ++i) {
                float s = g_lut8[p[i]];
                offset += 0.0001f * (s - offset);
                s -= offset;
                s *= 1.0f - offset;
                out[i] = s;
            }
            if (qoff) *qoff = offset;
            break;
        }
        default: {
            for (int64_t i = 0; i < length; ++i) out[i] = 0.0f;
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// IQ source
// ---------------------------------------------------------------------------
struct iqw_source {
    FILE*   fp        = nullptr;
    int     fmt       = IQW_FMT_S16;
    int     channels  = 2;          // 2 => complex IQ, 1 => real
    int     bits      = 16;
    double  rate      = 1.0;
    int64_t data_pos  = 0;          // byte offset of first sample frame
    int64_t data_len  = 0;          // number of sample frames (complex samples)
    int     block     = 4;          // bytes per sample frame = channels*bps
    float   qoff      = 0.0f;       // adaptive 8-bit DC offset
    // reusable IO scratch
    std::vector<uint8_t> raw;
    std::vector<float>   flat;
};

static int fmt_from_bits(int bits, bool is_float) {
    if (is_float) return IQW_FMT_F32;
    switch (bits) {
        case 8:  return IQW_FMT_U8;   // WAV PCM 8-bit is unsigned
        case 16: return IQW_FMT_S16;
        case 24: return IQW_FMT_S24;
        case 32: return IQW_FMT_S32;
        default: return IQW_FMT_S16;
    }
}

static uint32_t rd_u32(FILE* f) {
    uint8_t b[4]; if (fread(b, 1, 4, f) != 4) return 0;
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}
static uint16_t rd_u16(FILE* f) {
    uint8_t b[2]; if (fread(b, 1, 2, f) != 2) return 0;
    return (uint16_t)b[0] | ((uint16_t)b[1] << 8);
}
static uint64_t rd_u64(FILE* f) {
    uint8_t b[8]; if (fread(b, 1, 8, f) != 8) return 0;
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(b[i]) << (8 * i);
    return v;
}

// Parse WAV/RIFF plus RF64/BW64 (EBU large-file WAV).  RF64 stores the real
// data size in its ds64 chunk while the ordinary 32-bit data length is all 1s.
static bool parse_wav(iqw_source* s) {
    FILE* f = s->fp;
    fseek(f, 0, SEEK_SET);
    char tag[4];
    if (fread(tag, 1, 4, f) != 4) return false;
    const bool is_rf64 = memcmp(tag, "RF64", 4) == 0 ||
                         memcmp(tag, "BW64", 4) == 0;
    if (!is_rf64 && memcmp(tag, "RIFF", 4) != 0) return false;
    rd_u32(f);  // riff size
    if (fread(tag, 1, 4, f) != 4 || memcmp(tag, "WAVE", 4) != 0) return false;

    int16_t  fmt_tag = 1;
    uint16_t channels = 0, bits = 0;
    uint32_t rate = 0;
    bool have_fmt = false;
    uint64_t data_size64 = 0;
    uint64_t sample_count64 = 0;

    // Walk chunks until we find 'fmt ' then 'data'.
    while (true) {
        if (fread(tag, 1, 4, f) != 4) return false;
        uint32_t len = rd_u32(f);
        if (memcmp(tag, "ds64", 4) == 0) {
            if (len < 28) return false;
            const long after = ftell(f) + static_cast<long>(len);
            (void)rd_u64(f);                 // 64-bit RIFF size
            data_size64 = rd_u64(f);
            sample_count64 = rd_u64(f);
            (void)rd_u32(f);                 // optional table entry count
            fseek(f, after + (len & 1), SEEK_SET);
        } else if (memcmp(tag, "fmt ", 4) == 0) {
            long after = ftell(f) + (long)len;
            fmt_tag  = (int16_t)rd_u16(f);
            channels = rd_u16(f);
            rate     = rd_u32(f);
            rd_u32(f);            // avg bytes/sec
            rd_u16(f);            // block align
            bits     = rd_u16(f);
            have_fmt = true;
            fseek(f, after, SEEK_SET);   // skip any extension bytes
        } else if (memcmp(tag, "data", 4) == 0) {
            if (!have_fmt) return false;
            s->data_pos = ftell(f);
            int block = (int)channels * ((int)bits / 8);
            if (block <= 0) return false;
            s->block    = block;
            s->channels = channels;
            s->bits     = bits;
            s->rate     = (double)rate;
            s->fmt      = fmt_from_bits(bits, fmt_tag == 3 /*IEEE float*/);
            const uint64_t data_bytes =
                (is_rf64 && len == 0xffffffffu && data_size64 > 0)
                    ? data_size64 : static_cast<uint64_t>(len);
            s->data_len = sample_count64 > 0
                ? static_cast<int64_t>(sample_count64)
                : static_cast<int64_t>(data_bytes / block);
            return true;
        } else {
            // skip unknown chunk (pad to even per RIFF spec)
            long skip = (long)len + (long)(len & 1);
            if (fseek(f, skip, SEEK_CUR) != 0) return false;
        }
    }
}

extern "C" {

IQW_API iqw_source* iqw_open(const char* path, int fmt, double sample_rate,
                             int is_complex) {
    FILE* f = fopen(path, "rb");
    if (!f) return nullptr;
    iqw_source* s = new iqw_source();
    s->fp = f;

    if (fmt == IQW_FMT_AUTO) {
        if (!parse_wav(s)) { fclose(f); delete s; return nullptr; }
    } else {
        int bps = bytes_per_sample(fmt);
        if (bps == 0) { fclose(f); delete s; return nullptr; }
        s->fmt      = fmt;
        s->channels = is_complex ? 2 : 1;
        s->bits     = bps * 8;
        s->rate     = sample_rate > 0 ? sample_rate : 1.0;
        s->block    = bps * s->channels;
        s->data_pos = 0;
        fseek(f, 0, SEEK_END);
        long end = ftell(f);
        s->data_len = (int64_t)end / s->block;
    }
    return s;
}

IQW_API void iqw_close(iqw_source* s) {
    if (!s) return;
    if (s->fp) fclose(s->fp);
    delete s;
}

IQW_API int64_t iqw_num_samples(const iqw_source* s) { return s ? s->data_len : 0; }
IQW_API double  iqw_sample_rate(const iqw_source* s) { return s ? s->rate : 0.0; }
IQW_API int     iqw_is_complex(const iqw_source* s)  { return s ? (s->channels >= 2) : 0; }
IQW_API int     iqw_bits(const iqw_source* s)        { return s ? s->bits : 0; }
IQW_API int     iqw_format(const iqw_source* s)      { return s ? s->fmt : 0; }

// Read `count` complex samples starting at complex-sample index `start` into
// `out_iq` (count*2 interleaved floats: I,Q).  Out-of-range regions are zero.
// Returns the number of in-range samples read.
IQW_API int64_t iqw_read(iqw_source* s, int64_t start, int64_t count,
                         float* out_iq) {
    if (!s || !out_iq || count <= 0) return 0;
    for (int64_t i = 0; i < count * 2; ++i) out_iq[i] = 0.0f;
    if (start >= s->data_len || start + count <= 0) return 0;

    int64_t s0 = std::max<int64_t>(start, 0);
    int64_t s1 = std::min<int64_t>(start + count, s->data_len);
    int64_t n  = s1 - s0;
    if (n <= 0) return 0;

    int ch = s->channels;
    int scalars = (int)ch;                       // scalars per frame
    size_t nbytes = (size_t)n * s->block;
    if (s->raw.size() < nbytes) s->raw.resize(nbytes);
    if ((int64_t)s->flat.size() < n * scalars) s->flat.resize(n * scalars);

    if (fseek(s->fp, (long)(s->data_pos + s0 * s->block), SEEK_SET) != 0) return 0;
    size_t got = fread(s->raw.data(), 1, nbytes, s->fp);
    int64_t frames_got = (int64_t)(got / s->block);
    if (frames_got <= 0) return 0;

    convert_to_float(s->raw.data(), s->flat.data(),
                     frames_got * scalars, s->fmt, &s->qoff);

    int64_t dst0 = s0 - start;   // offset into out where valid data begins
    if (ch >= 2) {
        for (int64_t i = 0; i < frames_got; ++i) {
            out_iq[2 * (dst0 + i) + 0] = s->flat[i * scalars + 0];
            out_iq[2 * (dst0 + i) + 1] = s->flat[i * scalars + 1];
        }
    } else {
        for (int64_t i = 0; i < frames_got; ++i) {
            out_iq[2 * (dst0 + i) + 0] = s->flat[i];
            out_iq[2 * (dst0 + i) + 1] = 0.0f;
        }
    }
    return frames_got;
}

} // extern "C"

// ---------------------------------------------------------------------------
// Waterfall engine
// ---------------------------------------------------------------------------
enum iqw_window {
    IQW_WIN_HANN = 0,
    IQW_WIN_HAMMING = 1,
    IQW_WIN_BLACKMAN = 2,
    IQW_WIN_BLACKMAN_HARRIS = 3,
    IQW_WIN_RECT = 4,
};

struct iqw_wf {
    int n_fft = 1024;
    bfft_plan* plan = nullptr;
    size_t bins = 0;          // n_fft/2 + 1
    std::vector<double> window;
    std::vector<double> wi, wq;        // windowed real inputs
    std::vector<double> work;
    std::vector<bfft_complex> scratch; // native scratch for bfft_forward
    std::vector<bfft_complex> A, B;    // rfft outputs
    std::vector<float> iq;             // interleaved read buffer for one frame
};

static void fill_window(std::vector<double>& w, int n, int type) {
    w.resize(n);
    for (int k = 0; k < n; ++k) {
        // Symmetric analysis windows: matches np.hanning/np.hamming and the
        // paused-suffix STFT reference. Rows are center-timed by the app.
        double x = n > 1 ? (double)k / (double)(n - 1) : 0.0;
        double v;
        switch (type) {
            case IQW_WIN_HAMMING:
                v = 0.54 - 0.46 * std::cos(2 * M_PI * x); break;
            case IQW_WIN_BLACKMAN:
                v = 0.42 - 0.5 * std::cos(2 * M_PI * x)
                        + 0.08 * std::cos(4 * M_PI * x); break;
            case IQW_WIN_BLACKMAN_HARRIS:
                v = 0.35875 - 0.48829 * std::cos(2 * M_PI * x)
                        + 0.14128 * std::cos(4 * M_PI * x)
                        - 0.01168 * std::cos(6 * M_PI * x); break;
            case IQW_WIN_RECT:
                v = 1.0; break;
            case IQW_WIN_HANN:
            default:
                v = 0.5 - 0.5 * std::cos(2 * M_PI * x); break;
        }
        w[k] = v;
    }
}

// Turn the N interleaved IQ samples already sitting in e->iq into one
// fftshifted dB row (via two BFFT real FFTs + Hermitian recombination).
static void wf_frame_to_row(iqw_wf* e, float* row, int remove_dc) {
    const int N = e->n_fft;
    const int half = N / 2;
    const double eps = 1e-12;

    if (remove_dc) {
        double mi = 0, mq = 0;
        for (int k = 0; k < N; ++k) { mi += e->iq[2*k]; mq += e->iq[2*k+1]; }
        mi /= N; mq /= N;
        for (int k = 0; k < N; ++k) { e->iq[2*k] -= (float)mi; e->iq[2*k+1] -= (float)mq; }
    }

    const double* w = e->window.data();
    for (int k = 0; k < N; ++k) {
        e->wi[k] = (double)e->iq[2*k]     * w[k];
        e->wq[k] = (double)e->iq[2*k + 1] * w[k];
    }
    bfft_forward(e->plan, e->wi.data(), e->A.data(), e->work.data(), e->scratch.data());
    bfft_forward(e->plan, e->wq.data(), e->B.data(), e->work.data(), e->scratch.data());

    for (int c = 0; c < N; ++c) {
        int kk = c + half;
        if (kk >= N) kk -= N;
        double xr, xi;
        if (kk <= half) {
            xr = e->A[kk].re - e->B[kk].im;
            xi = e->A[kk].im + e->B[kk].re;
        } else {
            int m = N - kk;
            double ar = e->A[m].re, ai = -e->A[m].im;
            double br = e->B[m].re, bi = -e->B[m].im;
            xr = ar - bi;
            xi = ai + br;
        }
        double mag = std::sqrt(xr * xr + xi * xi);
        row[c] = (float)(20.0 * std::log10(mag + eps));
    }
}

extern "C" {

IQW_API iqw_wf* iqw_wf_create(int n_fft, int window_type) {
    if (n_fft < 4 || (n_fft & (n_fft - 1)) != 0) return nullptr;
    iqw_wf* e = new iqw_wf();
    e->n_fft = n_fft;
    if (bfft_plan_create((size_t)n_fft, &e->plan) != BFFT_OK) {
        delete e; return nullptr;
    }
    e->bins = bfft_plan_bins(e->plan);
    fill_window(e->window, n_fft, window_type);
    e->wi.resize(n_fft); e->wq.resize(n_fft);
    e->work.resize(bfft_plan_work_size(e->plan));
    e->scratch.resize(bfft_plan_native_scratch_size(e->plan) + 1);
    e->A.resize(e->bins); e->B.resize(e->bins);
    e->iq.resize((size_t)n_fft * 2);
    return e;
}

IQW_API void iqw_wf_destroy(iqw_wf* e) {
    if (!e) return;
    if (e->plan) bfft_plan_destroy(e->plan);
    delete e;
}

IQW_API int iqw_wf_nfft(const iqw_wf* e) { return e ? e->n_fft : 0; }

// Render `n_rows` waterfall rows starting at complex-sample `start` with frame
// spacing `hop`.  out_db is n_rows * n_fft floats, row-major.  Each row is one
// windowed FFT frame, fftshifted (column 0 = -Fs/2, column n_fft-1 ~ +Fs/2),
// stored as 20*log10(|X|).  If `remove_dc`, the frame DC is subtracted first.
IQW_API void iqw_wf_render(iqw_wf* e, iqw_source* src, int64_t start,
                           int64_t hop, int n_rows, float* out_db,
                           int remove_dc) {
    if (!e || !src || !out_db || n_rows <= 0) return;
    const int N = e->n_fft;
    for (int r = 0; r < n_rows; ++r) {
        int64_t fstart = start + (int64_t)r * hop;
        iqw_read(src, fstart, N, e->iq.data());
        wf_frame_to_row(e, out_db + (size_t)r * N, remove_dc);
    }
}

// Render a waterfall directly from an in-memory interleaved-IQ buffer (I,Q
// pairs), `nsamples` complex samples long.  Frames starting outside the buffer
// are zero-padded.  Used to display reconstructed (DIP) signal buffers.
IQW_API void iqw_wf_render_mem(iqw_wf* e, const float* iq, int64_t nsamples,
                               int64_t start, int64_t hop, int n_rows,
                               float* out_db, int remove_dc) {
    if (!e || !iq || !out_db || n_rows <= 0) return;
    const int N = e->n_fft;
    for (int r = 0; r < n_rows; ++r) {
        int64_t fstart = start + (int64_t)r * hop;
        for (int k = 0; k < N; ++k) {
            int64_t s = fstart + k;
            if (s >= 0 && s < nsamples) {
                e->iq[2*k]   = iq[2*s];
                e->iq[2*k+1] = iq[2*s+1];
            } else {
                e->iq[2*k] = 0.0f; e->iq[2*k+1] = 0.0f;
            }
        }
        wf_frame_to_row(e, out_db + (size_t)r * N, remove_dc);
    }
}

} // extern "C"

// ---------------------------------------------------------------------------
// Adaptive leading-edge FCT waterfall, complex/IQ.
//
// Unlike a windowed STFT, a row is a family of prefixes beginning at the row
// origin.  The caller receives tau so it can place each bin at its selected
// endpoint in waterfall time.  Intensity is the transform's objective amplitude
// |C(tau)|/sqrt(tau), whose square is the normalized correlation being maximized.
// Keeping that canonical scale (rather than multiplying it by sqrt(N)) prevents
// short null prefixes from saturating a conventional waterfall dB range.  A
// rectangular aperture is intentional; a symmetric Hann would bias the prefix
// selector.
// ---------------------------------------------------------------------------
struct iqw_fct {
    int n = 0;
    fct_plan* plan = nullptr;
    std::vector<bfft_complex> input;
    std::vector<bfft_complex> output;
    std::vector<int64_t> tau;
    std::vector<float> iq;
};

static void fct_frame(iqw_fct* e, float* db, float* support, int remove_dc) {
    const int n = e->n;
    double mr = 0.0, mi = 0.0;
    if (remove_dc) {
        for (int t = 0; t < n; ++t) {
            mr += e->iq[2 * t];
            mi += e->iq[2 * t + 1];
        }
        mr /= n;
        mi /= n;
    }
    for (int t = 0; t < n; ++t) {
        e->input[t].re = e->iq[2 * t] - mr;
        e->input[t].im = e->iq[2 * t + 1] - mi;
    }
    if (fct_forward_complex(e->plan, e->input.data(), e->output.data(),
                            e->tau.data()) != BFFT_OK) {
        std::fill(db, db + n, -240.0f);
        std::fill(support, support + n, 1.0f);
        return;
    }
    const double eps = 1e-15;
    for (int c = 0; c < n; ++c) {
        int k = c + n / 2;
        if (k >= n) k -= n;
        const double tr = std::max<int64_t>(1, e->tau[k]);
        const double re = e->output[k].re, im = e->output[k].im;
        const double mag = std::sqrt(re * re + im * im) / std::sqrt(tr);
        db[c] = static_cast<float>(20.0 * std::log10(mag + eps));
        support[c] = static_cast<float>(tr / n);
    }
}

extern "C" {

IQW_API iqw_fct* iqw_fct_create_ex(int n_fft, int min_support,
                                    double activity) {
    if (n_fft < 16 || (n_fft & (n_fft - 1)) != 0) return nullptr;
    if (min_support < 1 || min_support > n_fft || activity < 0.0)
        return nullptr;
    iqw_fct* e = new iqw_fct();
    e->n = n_fft;
    // FCT now means the intrinsic exact support transform.  The minimum is a
    // declared feasible-domain boundary, not a proxy selection heuristic.
    if (fct_plan_create_ex(static_cast<size_t>(n_fft), min_support, 0.5,
                           activity, &e->plan) != BFFT_OK) {
        delete e;
        return nullptr;
    }
    e->input.resize(n_fft);
    e->output.resize(n_fft);
    e->tau.resize(n_fft);
    e->iq.resize(static_cast<size_t>(2) * n_fft);
    return e;
}

IQW_API iqw_fct* iqw_fct_create(int n_fft) {
    return iqw_fct_create_ex(n_fft, 1, 0.0);
}

IQW_API void iqw_fct_destroy(iqw_fct* e) {
    if (!e) return;
    fct_plan_destroy(e->plan);
    delete e;
}

IQW_API void iqw_fct_render(iqw_fct* e, iqw_source* src, int64_t start,
                            int64_t hop, int n_rows, float* out_db,
                            float* out_support, int remove_dc) {
    if (!e || !src || !out_db || !out_support || n_rows <= 0) return;
    for (int r = 0; r < n_rows; ++r) {
        iqw_read(src, start + static_cast<int64_t>(r) * hop, e->n,
                 e->iq.data());
        fct_frame(e, out_db + static_cast<size_t>(r) * e->n,
                  out_support + static_cast<size_t>(r) * e->n, remove_dc);
    }
}

IQW_API void iqw_fct_render_mem(iqw_fct* e, const float* iq, int64_t nsamples,
                                int64_t start, int64_t hop, int n_rows,
                                float* out_db, float* out_support,
                                int remove_dc) {
    if (!e || !iq || !out_db || !out_support || n_rows <= 0) return;
    for (int r = 0; r < n_rows; ++r) {
        const int64_t base = start + static_cast<int64_t>(r) * hop;
        for (int t = 0; t < e->n; ++t) {
            const int64_t s = base + t;
            if (s >= 0 && s < nsamples) {
                e->iq[2 * t] = iq[2 * s];
                e->iq[2 * t + 1] = iq[2 * s + 1];
            } else {
                e->iq[2 * t] = e->iq[2 * t + 1] = 0.0f;
            }
        }
        fct_frame(e, out_db + static_cast<size_t>(r) * e->n,
                  out_support + static_cast<size_t>(r) * e->n, remove_dc);
    }
}

} // extern "C"

// ---------------------------------------------------------------------------
// Reassigned spectrogram (super-resolution readout), complex/IQ, via bfft.
// A long window supplies fine frequency; reassignment relocates each energy
// point to its instantaneous frequency and group-delay-corrected time.
// ---------------------------------------------------------------------------
struct iqw_ra {
    int n = 1024;
    bfft_plan* plan = nullptr;
    size_t bins = 0;
    std::vector<double> g, dg, tg;               // analysis / derivative / time
    std::vector<double> wr, wi;                   // windowed real / imag segment
    std::vector<double> work;
    std::vector<bfft_complex> scratch;
    std::vector<bfft_complex> A, B;               // rfft outputs
    std::vector<double> Yr, Yi, Ytr, Yti, Ydr, Ydi;   // full-n complex spectra
    std::vector<double> P;                         // reassigned power accumulator
};

// Full-n complex FFT of (re + j im) from two BFFT real FFTs.
static void ra_cfft(iqw_ra* e, const double* re, const double* im,
                    double* outr, double* outi) {
    bfft_forward(e->plan, re, e->A.data(), e->work.data(), e->scratch.data());
    bfft_forward(e->plan, im, e->B.data(), e->work.data(), e->scratch.data());
    const int n = e->n, half = n / 2;
    for (int k = 0; k < n; ++k) {
        double Rr, Ri, Ir, Ii;
        if (k <= half) { Rr = e->A[k].re; Ri = e->A[k].im; Ir = e->B[k].re; Ii = e->B[k].im; }
        else { int m = n - k; Rr = e->A[m].re; Ri = -e->A[m].im; Ir = e->B[m].re; Ii = -e->B[m].im; }
        outr[k] = Rr - Ii;      // (Rr + jRi) + j(Ir + jIi)
        outi[k] = Ri + Ir;
    }
}

extern "C" {

IQW_API iqw_ra* iqw_ra_create(int n_fft) {
    if (n_fft < 4 || (n_fft & (n_fft - 1)) != 0) return nullptr;
    iqw_ra* e = new iqw_ra();
    e->n = n_fft;
    if (bfft_plan_create((size_t)n_fft, &e->plan) != BFFT_OK) { delete e; return nullptr; }
    e->bins = bfft_plan_bins(e->plan);
    e->g.resize(n_fft); e->dg.resize(n_fft); e->tg.resize(n_fft);
    for (int k = 0; k < n_fft; ++k)
        e->g[k] = 0.5 - 0.5 * std::cos(2.0 * M_PI * k / (n_fft - 1));   // np.hanning
    for (int k = 0; k < n_fft; ++k) {                                    // np.gradient
        if (k == 0) e->dg[k] = e->g[1] - e->g[0];
        else if (k == n_fft - 1) e->dg[k] = e->g[k] - e->g[k - 1];
        else e->dg[k] = 0.5 * (e->g[k + 1] - e->g[k - 1]);
        e->tg[k] = (k - (n_fft - 1) / 2.0) * e->g[k];
    }
    e->wr.resize(n_fft); e->wi.resize(n_fft);
    e->work.resize(bfft_plan_work_size(e->plan));
    e->scratch.resize(bfft_plan_native_scratch_size(e->plan) + 1);
    e->A.resize(e->bins); e->B.resize(e->bins);
    e->Yr.resize(n_fft); e->Yi.resize(n_fft);
    e->Ytr.resize(n_fft); e->Yti.resize(n_fft);
    e->Ydr.resize(n_fft); e->Ydi.resize(n_fft);
    return e;
}

IQW_API void iqw_ra_destroy(iqw_ra* e) {
    if (!e) return;
    if (e->plan) bfft_plan_destroy(e->plan);
    delete e;
}

IQW_API int iqw_ra_nfft(const iqw_ra* e) { return e ? e->n : 0; }

// Reassigned spectrogram from an in-memory interleaved-IQ buffer.
// out_db is n_rows * n_fft floats, row-major, fftshifted (col 0 = -Fs/2),
// 10*log10(reassigned power). Time centers are start + r*hop (buffer indices).
IQW_API void iqw_ra_render_mem(iqw_ra* e, const float* iq, int64_t nsamples,
                               int64_t start, int64_t hop, int n_rows,
                               float* out_db) {
    if (!e || !iq || !out_db || n_rows <= 0) return;
    const int n = e->n, half = n / 2;
    const double eps = 1e-12, twopi = 2.0 * M_PI;
    e->P.assign((size_t)n_rows * n, 0.0);

    for (int r = 0; r < n_rows; ++r) {
        int64_t c = (int64_t)r * hop;    // center, buffer-local
        const double* w;
        // three windowed segments -> three full-n complex spectra
        auto load = [&](const double* win, double* outr, double* outi) {
            for (int k = 0; k < n; ++k) {
                int64_t s = start + c - half + k;
                double sr = 0.0, si = 0.0;
                if (s >= 0 && s < nsamples) { sr = iq[2 * s]; si = iq[2 * s + 1]; }
                e->wr[k] = win[k] * sr; e->wi[k] = win[k] * si;
            }
            ra_cfft(e, e->wr.data(), e->wi.data(), outr, outi);
        };
        (void)w;
        load(e->g.data(), e->Yr.data(), e->Yi.data());
        load(e->tg.data(), e->Ytr.data(), e->Yti.data());
        load(e->dg.data(), e->Ydr.data(), e->Ydi.data());

        double emax = 0.0;
        for (int k = 0; k < n; ++k) {
            double E = e->Yr[k] * e->Yr[k] + e->Yi[k] * e->Yi[k];
            if (E > emax) emax = E;
        }
        double thr = 1e-8 * (emax + 1e-30);   // Python reference threshold
        for (int k = 0; k < n; ++k) {
            double E = e->Yr[k] * e->Yr[k] + e->Yi[k] * e->Yi[k];
            if (E <= thr) continue;
            double inv = 1.0 / (E + 1e-30);
            double reax = e->Ytr[k] * e->Yr[k] + e->Yti[k] * e->Yi[k];   // Re(Yt conj Y)
            double imdc = e->Ydi[k] * e->Yr[k] - e->Ydr[k] * e->Yi[k];   // Im(Yd conj Y)
            double that = (double)c + reax * inv;
            double khat = k - imdc * inv * n / twopi;
            long col = (long)std::lround(that / (double)hop);
            if (col < 0) col = 0; else if (col >= n_rows) col = n_rows - 1;
            long row = ((long)std::lround(khat) % n + n) % n;
            e->P[(size_t)col * n + row] += E;
        }
    }

    for (int r = 0; r < n_rows; ++r) {
        double* pr = e->P.data() + (size_t)r * n;
        float* orow = out_db + (size_t)r * n;
        for (int c2 = 0; c2 < n; ++c2) {
            int kk = c2 + half; if (kk >= n) kk -= n;   // fftshift
            orow[c2] = (float)(10.0 * std::log10(pr[kk] + eps));
        }
    }
}

} // extern "C"
