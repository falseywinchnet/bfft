PREFIX ?= /usr/local
DESTDIR ?=
BUILD_DIR ?= build
LIB_NAME := bfft
VERSION := 0.1.0

CXX ?= c++
CC ?= cc
AR ?= ar
INSTALL ?= install
SED ?= sed

CXX_STD ?= c++17
CXXOPTFLAGS ?= -O3
CXXWARNFLAGS ?= -Wall -Wextra -Wpedantic
COPTFLAGS ?= -O2
CWARNFLAGS ?= -Wall -Wextra -Wpedantic
CPPFLAGS ?=
CXXFLAGS ?= $(CXXOPTFLAGS) -std=$(CXX_STD) -fPIC $(CXXWARNFLAGS)
CFLAGS ?= $(COPTFLAGS) -std=c11 $(CWARNFLAGS)
LDFLAGS ?=
LDLIBS ?= -lm

UNAME_M := $(shell uname -m 2>/dev/null || echo unknown)
UNAME_S := $(shell uname -s 2>/dev/null || echo unknown)
ifeq ($(UNAME_S),Darwin)
  DL_LIBS ?=
  ACCELERATE_LIBS ?= -framework Accelerate
  APPLE_EXAMPLES = $(APPLE_BENCH)
else
  DL_LIBS ?= -ldl
  ACCELERATE_LIBS ?=
  APPLE_EXAMPLES :=
endif
AVX2_FLAGS := $(shell $(CXX) -x c++ -std=$(CXX_STD) -mavx2 -mfma -c /dev/null -o /tmp/bfft-avx2-test.o >/dev/null 2>&1 && rm -f /tmp/bfft-avx2-test.o && echo '-mavx2 -mfma')
SSE2_FLAGS := $(shell $(CXX) -x c++ -std=$(CXX_STD) -msse2 -mno-avx -c /dev/null -o /tmp/bfft-sse2-test.o >/dev/null 2>&1 && rm -f /tmp/bfft-sse2-test.o && echo '-msse2 -mno-avx')
ifeq ($(findstring x86_64,$(UNAME_M)),x86_64)
  AUTO_SIMD_FLAGS := $(AVX2_FLAGS)
else
  AUTO_SIMD_FLAGS :=
endif

INCLUDES := -Iinclude
LIB_CPPFLAGS := $(CPPFLAGS) $(INCLUDES)
LIB_CXXFLAGS := $(CXXFLAGS) $(AUTO_SIMD_FLAGS)

SRC := src/bfft.cpp
OBJ := $(BUILD_DIR)/src/bfft.o
STATIC_LIB := $(BUILD_DIR)/lib$(LIB_NAME).a
SHARED_LIB := $(BUILD_DIR)/lib$(LIB_NAME).so
PC_FILE := $(BUILD_DIR)/$(LIB_NAME).pc
BENCH := $(BUILD_DIR)/examples/benchmark
APPLE_BENCH := $(BUILD_DIR)/examples/apple_benchmark
LOCALITY_PROBE := $(BUILD_DIR)/examples/locality_probe
C_DEMO := $(BUILD_DIR)/examples/c_api_demo
CPP_DEMO := $(BUILD_DIR)/examples/cpp_api_demo
CORRECTNESS_TEST := $(BUILD_DIR)/tests/correctness
C_API_TEST := $(BUILD_DIR)/tests/api_c
DIT_TEST := $(BUILD_DIR)/tests/test_dit
RADIX4_TEST := $(BUILD_DIR)/experiments/test_bruun_radix4
RADIX4_BENCH := $(BUILD_DIR)/experiments/benchmark_radix2_vs_radix4
COMPOSED_R4_TEST := $(BUILD_DIR)/experiments/test_chebyshev_composed_radix4
DEFERRED_PROBE := $(BUILD_DIR)/experiments/cheb_deferred_probe
MULTIPLY_PROBE := $(BUILD_DIR)/experiments/cheb_multiply_probe
STABILITY_PROBE := $(BUILD_DIR)/experiments/cheb_stability_probe
DCT_TEST := $(BUILD_DIR)/experiments/test_chebyshev_dct
DCT_BENCH := $(BUILD_DIR)/experiments/cheb_dct_bench
DCT_RADIX_PROBE := $(BUILD_DIR)/experiments/cheb_dct_radix_probe
SUPPORT_PROBE := $(BUILD_DIR)/tests/bfft_fftw_support_probe
INVARIANT_PROBE := $(BUILD_DIR)/tests/bfft_invariant_support_probe
SFDR_PROBE := $(BUILD_DIR)/tests/bfft_fftw_sfdr_probe
BH7_PROBE := $(BUILD_DIR)/tests/bfft_fftw_sfdr_bh7_probe
LIBRARY_COMPARE_PROBE := $(BUILD_DIR)/tests/bfft_library_compare_probe
AVX2_ASM := $(BUILD_DIR)/src/bfft_avx2.s
SSE2_ASM := $(BUILD_DIR)/src/bfft_sse2.s
DIT_AVX2_ASM := $(BUILD_DIR)/src/test_dit_avx2.s
DIT_SSE2_ASM := $(BUILD_DIR)/src/test_dit_sse2.s
ASM_OUTPUTS :=
ifneq ($(AVX2_FLAGS),)
  ASM_OUTPUTS += $(AVX2_ASM)
  ASM_OUTPUTS += $(DIT_AVX2_ASM)
endif
ifneq ($(SSE2_FLAGS),)
  ASM_OUTPUTS += $(SSE2_ASM)
  ASM_OUTPUTS += $(DIT_SSE2_ASM)
endif

.PHONY: all clean test install uninstall examples experiments radix4-test dct-bench dct-radix-probe probes docs asm-check check-standards check-cxx17 check-cxx20 check-cxx23

all: $(STATIC_LIB) $(SHARED_LIB) examples

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/src $(BUILD_DIR)/examples $(BUILD_DIR)/tests $(BUILD_DIR)/experiments

$(OBJ): $(SRC) include/bfft/bfft.h src/detail/bruun_kernel.hpp | $(BUILD_DIR)
	$(CXX) $(LIB_CPPFLAGS) $(LIB_CXXFLAGS) -c $< -o $@

$(STATIC_LIB): $(OBJ)
	$(AR) rcs $@ $^

$(SHARED_LIB): $(OBJ)
	$(CXX) -shared $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(PC_FILE): pkgconfig/bfft.pc.in | $(BUILD_DIR)
	$(SED) \
		-e 's|@CMAKE_INSTALL_LIBDIR@|lib|g' \
		-e 's|@CMAKE_INSTALL_INCLUDEDIR@|include|g' \
		-e 's|@PROJECT_VERSION@|$(VERSION)|g' \
		$< > $@

examples: $(BENCH) $(APPLE_EXAMPLES) $(LOCALITY_PROBE) $(C_DEMO) $(CPP_DEMO)

asm-check: $(ASM_OUTPUTS)
	@if [ -z "$(ASM_OUTPUTS)" ]; then echo "No x86 assembly variants supported by $(CXX)."; fi

$(AVX2_ASM): $(SRC) include/bfft/bfft.h src/detail/bruun_kernel.hpp | $(BUILD_DIR)
	$(CXX) $(LIB_CPPFLAGS) $(CXXFLAGS) $(AVX2_FLAGS) -S -fverbose-asm $< -o $@

$(SSE2_ASM): $(SRC) include/bfft/bfft.h src/detail/bruun_kernel.hpp | $(BUILD_DIR)
	$(CXX) $(LIB_CPPFLAGS) $(CXXFLAGS) $(SSE2_FLAGS) -S -fverbose-asm $< -o $@

$(DIT_AVX2_ASM): tests/test_dit.cpp src/detail/bruun_DIT_kernel.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $(AVX2_FLAGS) -S -fverbose-asm $< -o $@

$(DIT_SSE2_ASM): tests/test_dit.cpp src/detail/bruun_DIT_kernel.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $(SSE2_FLAGS) -S -fverbose-asm $< -o $@

$(BENCH): examples/benchmark.cpp include/bfft/bfft.hpp src/detail/bruun_DIT_kernel.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $(AUTO_SIMD_FLAGS) $< $(STATIC_LIB) $(LDLIBS) $(DL_LIBS) -o $@

$(APPLE_BENCH): examples/apple_benchmark.cpp include/bfft/bfft.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $< $(STATIC_LIB) $(LDLIBS) $(DL_LIBS) $(ACCELERATE_LIBS) -o $@

$(LOCALITY_PROBE): examples/locality_probe.cpp include/bfft/bfft.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $< $(STATIC_LIB) $(LDLIBS) -o $@

$(C_DEMO): examples/c_api_demo.c include/bfft/bfft.h $(STATIC_LIB) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) $< $(STATIC_LIB) $(LDLIBS) -lstdc++ -o $@

$(CPP_DEMO): examples/cpp_api_demo.cpp include/bfft/bfft.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $< $(STATIC_LIB) $(LDLIBS) -o $@

$(CORRECTNESS_TEST): tests/correctness.cpp include/bfft/bfft.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $< $(STATIC_LIB) $(LDLIBS) -o $@

$(C_API_TEST): tests/api_c.c include/bfft/bfft.h $(STATIC_LIB) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) $< $(STATIC_LIB) $(LDLIBS) -lstdc++ -o $@

$(DIT_TEST): tests/test_dit.cpp src/detail/bruun_DIT_kernel.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $(AUTO_SIMD_FLAGS) $< $(LDLIBS) -o $@

$(RADIX4_TEST): experiments/test_bruun_radix4.cpp include/bfft/bfft.hpp experiments/bruun_radix4_kernel.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $(AUTO_SIMD_FLAGS) $< $(STATIC_LIB) $(LDLIBS) -o $@

$(RADIX4_BENCH): experiments/benchmark_radix2_vs_radix4.cpp include/bfft/bfft.hpp experiments/bruun_radix4_kernel.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $(AUTO_SIMD_FLAGS) $< $(STATIC_LIB) $(LDLIBS) -o $@

$(COMPOSED_R4_TEST): experiments/test_chebyshev_composed_radix4.cpp experiments/chebyshev_composed_radix4_kernel.hpp experiments/bruun_radix4_kernel.hpp include/bfft/bfft.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $(AUTO_SIMD_FLAGS) $< $(STATIC_LIB) $(LDLIBS) -o $@

$(DEFERRED_PROBE): experiments/cheb_deferred_probe.cpp experiments/chebyshev_composed_radix4_kernel.hpp include/bfft/bfft.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $(AUTO_SIMD_FLAGS) $< $(STATIC_LIB) $(LDLIBS) -o $@

$(MULTIPLY_PROBE): experiments/cheb_multiply_probe.cpp experiments/chebyshev_composed_radix4_kernel.hpp include/bfft/bfft.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $(AUTO_SIMD_FLAGS) $< $(STATIC_LIB) $(LDLIBS) -o $@

$(STABILITY_PROBE): experiments/cheb_stability_probe.cpp experiments/chebyshev_composed_radix4_kernel.hpp include/bfft/bfft.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $(AUTO_SIMD_FLAGS) $< $(STATIC_LIB) $(LDLIBS) -o $@

$(DCT_TEST): experiments/test_chebyshev_dct.cpp experiments/chebyshev_dct_kernel.hpp experiments/chebyshev_dct_radix4.hpp experiments/dct1_dst1_via_fft.hpp experiments/cheb_dct_assemble.hpp experiments/bruun_dct.hpp experiments/chebyshev_composed_radix4_kernel.hpp include/bfft/bfft.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $(AUTO_SIMD_FLAGS) $< $(STATIC_LIB) $(LDLIBS) -o $@

$(DCT_BENCH): experiments/cheb_dct_bench.cpp experiments/chebyshev_dct_radix4.hpp experiments/cheb_dct_assemble.hpp experiments/dct1_dst1_via_fft.hpp experiments/bruun_dct.hpp experiments/chebyshev_composed_radix4_kernel.hpp include/bfft/bfft.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $(AUTO_SIMD_FLAGS) $< $(STATIC_LIB) $(LDLIBS) -o $@

$(DCT_RADIX_PROBE): experiments/cheb_dct_radix_probe.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $(AUTO_SIMD_FLAGS) $< $(LDLIBS) -o $@

dct-bench: $(DCT_BENCH)
	$(DCT_BENCH)

dct-radix-probe: $(DCT_RADIX_PROBE)
	$(DCT_RADIX_PROBE)

experiments: $(RADIX4_TEST) $(RADIX4_BENCH) $(COMPOSED_R4_TEST) $(DEFERRED_PROBE) $(MULTIPLY_PROBE) $(STABILITY_PROBE) $(DCT_TEST) $(DCT_BENCH) $(DCT_RADIX_PROBE)

radix4-test: $(RADIX4_TEST) $(COMPOSED_R4_TEST) $(DEFERRED_PROBE) $(MULTIPLY_PROBE) $(STABILITY_PROBE) $(DCT_TEST)
	$(RADIX4_TEST)
	$(COMPOSED_R4_TEST)
	$(DEFERRED_PROBE)
	$(MULTIPLY_PROBE)
	$(STABILITY_PROBE)
	$(DCT_TEST)

$(SUPPORT_PROBE): tests/bfft_fftw_support_probe.cpp include/bfft/bfft.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $< $(STATIC_LIB) $(LDLIBS) $(DL_LIBS) -o $@

$(INVARIANT_PROBE): tests/bfft_invariant_support_probe.cpp include/bfft/bfft.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $< $(STATIC_LIB) $(LDLIBS) $(DL_LIBS) -o $@

$(SFDR_PROBE): tests/bfft_fftw_sfdr_probe.cpp include/bfft/bfft.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $< $(STATIC_LIB) $(LDLIBS) $(DL_LIBS) -o $@

$(BH7_PROBE): tests/bfft_fftw_sfdr_bh7_probe.cpp include/bfft/bfft.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $< $(STATIC_LIB) $(LDLIBS) $(DL_LIBS) -o $@

$(LIBRARY_COMPARE_PROBE): tests/bfft_library_compare_probe.cpp include/bfft/bfft.hpp $(STATIC_LIB) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDES) $(CXXFLAGS) $< $(STATIC_LIB) $(LDLIBS) $(DL_LIBS) -o $@

probes: $(SUPPORT_PROBE) $(INVARIANT_PROBE) $(SFDR_PROBE) $(BH7_PROBE) $(LIBRARY_COMPARE_PROBE)

test: $(CORRECTNESS_TEST) $(C_API_TEST) $(DIT_TEST)
	$(CORRECTNESS_TEST)
	$(C_API_TEST)
	$(DIT_TEST)

check-cxx17:
	$(MAKE) clean BUILD_DIR=build-cxx17
	$(MAKE) test BUILD_DIR=build-cxx17 CXX_STD=c++17 CXXWARNFLAGS="$(CXXWARNFLAGS) -Werror" CWARNFLAGS="$(CWARNFLAGS) -Werror"

check-cxx20:
	$(MAKE) clean BUILD_DIR=build-cxx20
	$(MAKE) test BUILD_DIR=build-cxx20 CXX_STD=c++20 CXXWARNFLAGS="$(CXXWARNFLAGS) -Werror" CWARNFLAGS="$(CWARNFLAGS) -Werror"

check-cxx23:
	$(MAKE) clean BUILD_DIR=build-cxx23
	$(MAKE) test BUILD_DIR=build-cxx23 CXX_STD=c++23 CXXWARNFLAGS="$(CXXWARNFLAGS) -Werror" CWARNFLAGS="$(CWARNFLAGS) -Werror"

check-standards: check-cxx17 check-cxx20 check-cxx23

install: all $(PC_FILE)
	$(INSTALL) -d $(DESTDIR)$(PREFIX)/include/bfft
	$(INSTALL) -d $(DESTDIR)$(PREFIX)/lib
	$(INSTALL) -d $(DESTDIR)$(PREFIX)/lib/pkgconfig
	$(INSTALL) -m 0644 include/bfft/bfft.h $(DESTDIR)$(PREFIX)/include/bfft/bfft.h
	$(INSTALL) -m 0644 include/bfft/bfft.hpp $(DESTDIR)$(PREFIX)/include/bfft/bfft.hpp
	$(INSTALL) -m 0644 $(STATIC_LIB) $(DESTDIR)$(PREFIX)/lib/lib$(LIB_NAME).a
	$(INSTALL) -m 0755 $(SHARED_LIB) $(DESTDIR)$(PREFIX)/lib/lib$(LIB_NAME).so
	$(INSTALL) -m 0644 $(PC_FILE) $(DESTDIR)$(PREFIX)/lib/pkgconfig/$(LIB_NAME).pc

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/include/bfft/bfft.h
	rm -f $(DESTDIR)$(PREFIX)/include/bfft/bfft.hpp
	rm -f $(DESTDIR)$(PREFIX)/lib/lib$(LIB_NAME).a
	rm -f $(DESTDIR)$(PREFIX)/lib/lib$(LIB_NAME).so
	rm -f $(DESTDIR)$(PREFIX)/lib/pkgconfig/$(LIB_NAME).pc

clean:
	rm -rf $(BUILD_DIR)
