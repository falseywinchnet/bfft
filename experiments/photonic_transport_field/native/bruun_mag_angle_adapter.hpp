#pragma once

// The photonic renderer only needs Bruun's magnitude/phase kernel, not the
// FFT backend which normally supplies its small compile environment.  Keep
// this adapter deliberately narrow so the generated, tested table family is
// shared verbatim with BFFT.

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
#define BRUUN_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define BRUUN_ALWAYS_INLINE inline
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#define PHOTONIC_MAG_DEFINED_M_PI 1
#endif

namespace photonic_mag {

static constexpr double bruun_tau = 2.0 * M_PI;
static constexpr double bruun_pio2 = 0.5 * M_PI;

#include "../../../src/detail/MAG_REPRESENT_KERNEL.hpp"

} // namespace photonic_mag

#ifdef PHOTONIC_MAG_DEFINED_M_PI
#undef PHOTONIC_MAG_DEFINED_M_PI
#undef M_PI
#endif
#undef BRUUN_ALWAYS_INLINE
