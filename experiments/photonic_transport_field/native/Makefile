CXX ?= clang++
CC ?= clang
CXXFLAGS ?= -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic
CONV_SOURCE := ../../../standalone_conv_resize_demo/native/conv_native.c

.PHONY: all clean test proof proof-test run-proof legacy diagnostic regime-diagnostic

all: window_transport_native

proof: regime_scene_native

run-proof: regime_scene_native
	./regime_scene_native

proof-test: regime_scene_native
	./regime_scene_native --self-test

regime_scene_native: regime_scene_native.cpp conv_native.o
	$(CXX) $(CXXFLAGS) regime_scene_native.cpp conv_native.o -pthread -o $@

legacy: photonic_native lightfield_native

photonic_native: photonic_native.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

lightfield_native: lightfield_native.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

conv_native.o: $(CONV_SOURCE)
	$(CC) -O3 -DNDEBUG -std=c11 -Wall -Wextra -Wpedantic -pthread -mcpu=native -c $< -o $@

window_transport_native: source_region_native.cpp conv_native.o
	$(CXX) $(CXXFLAGS) source_region_native.cpp conv_native.o -pthread -o $@

test: window_transport_native
	./window_transport_native --self-test

diagnostic: diagnose_window_reconstruction.cpp window_transport_native.cpp
	$(CXX) $(CXXFLAGS) diagnose_window_reconstruction.cpp -o diagnose_window_reconstruction

regime-diagnostic: diagnose_regime_branches.cpp conv_native.o
	$(CXX) $(CXXFLAGS) diagnose_regime_branches.cpp conv_native.o -pthread -o diagnose_regime_branches

clean:
	rm -f photonic_native lightfield_native window_transport_native regime_scene_native diagnose_window_reconstruction diagnose_regime_branches conv_native.o
