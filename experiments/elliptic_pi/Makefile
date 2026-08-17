CXX ?= clang++
CPPFLAGS += -I/opt/homebrew/opt/gmp/include -I/opt/homebrew/opt/mpfr/include
CXXFLAGS ?= -O3 -DNDEBUG -std=c++17 -Wall -Wextra -pedantic -pthread
LDFLAGS += -L/opt/homebrew/opt/gmp/lib -L/opt/homebrew/opt/mpfr/lib
LDLIBS += -lmpfr -lgmp

.PHONY: all clean bench run paired_convolution

all: elliptic_pi_cpp

elliptic_pi_cpp: elliptic_pi.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

bench: elliptic_pi_cpp
	./elliptic_pi_cpp --bits 1000
	./elliptic_pi_cpp --bits 10000
	./elliptic_pi_cpp --bits 100000

BITS ?= 10000
REPEATS ?= 3
SUITE ?= all
CONV_N ?= 262144
STREAM_STEPS ?= 16
run: elliptic_pi_cpp
	./elliptic_pi_cpp --bits $(BITS) --repeats $(REPEATS) --suite $(SUITE)

paired_convolution: bfft_paired_convolution.cpp
	$(CXX) $(CXXFLAGS) $< -o bfft_paired_convolution
	./bfft_paired_convolution $(CONV_N) $(REPEATS) $(STREAM_STEPS)

clean:
	rm -f elliptic_pi_cpp bfft_paired_convolution
