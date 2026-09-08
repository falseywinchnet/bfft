CXX ?= clang++
CC ?= clang
CXXFLAGS ?= -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic
CONV_SOURCE := ../../../standalone_conv_resize_demo/native/conv_native.c

.PHONY: all clean test proof proof-test run-proof legacy diagnostic regime-diagnostic retained retained-test

all: window_transport_native

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
RETAINED_LIBRARY := libphotonic_retained.dylib
RETAINED_SHARED_FLAGS := -dynamiclib
else
RETAINED_LIBRARY := libphotonic_retained.so
RETAINED_SHARED_FLAGS := -shared
endif

retained: $(RETAINED_LIBRARY)

$(RETAINED_LIBRARY): retained_transport.cpp retained_transport.h
	$(CXX) $(CXXFLAGS) -mcpu=native -fPIC -fvisibility=hidden $(RETAINED_SHARED_FLAGS) retained_transport.cpp -o $@

retained_test: retained_transport_test.cpp retained_transport.cpp retained_transport.h
	$(CXX) $(CXXFLAGS) -mcpu=native retained_transport_test.cpp retained_transport.cpp -o $@

retained-test: retained_test
	./retained_test

proof: regime_scene_native

run-proof: regime_scene_native
	./regime_scene_native

proof-test: regime_scene_native
	./regime_scene_native --self-test

regime_scene_native: child_boundary_response.hpp return_extinction.hpp retained_source_checks.hpp retained_boundary_checks.hpp retained_camera_checks.hpp retained_scene_update_checks.hpp regime_scene_native.cpp retained_transport.cpp retained_transport.h bruun_mag_angle_adapter.hpp \
		../../../src/detail/MAG_REPRESENT_KERNEL.hpp \
		../../../src/detail/generated/bruun_phase_slope_tables.hpp \
		../../../src/detail/generated/bruun_sincos_table256.hpp conv_native.o
	$(CXX) $(CXXFLAGS) -mcpu=native regime_scene_native.cpp retained_transport.cpp conv_native.o -pthread -o $@

return_extinction_probe: child_boundary_response.hpp return_extinction_probe.cpp return_extinction.hpp regime_scene_native.cpp retained_source_checks.hpp retained_boundary_checks.hpp retained_camera_checks.hpp retained_scene_update_checks.hpp retained_transport.cpp retained_transport.h bruun_mag_angle_adapter.hpp conv_native.o
	$(CXX) $(CXXFLAGS) -mcpu=native return_extinction_probe.cpp retained_transport.cpp conv_native.o -pthread -o $@

.PHONY: return-test
return-test: return_extinction_probe
	./return_extinction_probe

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
	rm -f photonic_native lightfield_native window_transport_native regime_scene_native return_extinction_probe child_boundary_probe observer_registration_probe diagnose_window_reconstruction diagnose_regime_branches conv_native.o $(RETAINED_LIBRARY) retained_test

.PHONY: update-test
update-test: regime_scene_native
	./regime_scene_native --update-self-test

.PHONY: camera-test
camera-test: regime_scene_native
	./regime_scene_native --camera-self-test

.PHONY: boundary-test
boundary-test: regime_scene_native
	./regime_scene_native --boundary-self-test

child_boundary_probe: child_boundary_probe.cpp child_boundary_response.hpp regime_scene_native.cpp return_extinction.hpp retained_source_checks.hpp retained_boundary_checks.hpp retained_camera_checks.hpp retained_scene_update_checks.hpp retained_transport.cpp retained_transport.h bruun_mag_angle_adapter.hpp conv_native.o
	$(CXX) $(CXXFLAGS) -mcpu=native child_boundary_probe.cpp retained_transport.cpp conv_native.o -pthread -o $@

.PHONY: child-boundary-test
child-boundary-test: child_boundary_probe
	./child_boundary_probe

observer_registration_probe: observer_registration_probe.cpp child_boundary_response.hpp regime_scene_native.cpp return_extinction.hpp retained_source_checks.hpp retained_boundary_checks.hpp retained_camera_checks.hpp retained_scene_update_checks.hpp retained_transport.cpp retained_transport.h bruun_mag_angle_adapter.hpp conv_native.o
	$(CXX) $(CXXFLAGS) -mcpu=native observer_registration_probe.cpp retained_transport.cpp conv_native.o -pthread -o $@

.PHONY: observer-registration-test
observer-registration-test: observer_registration_probe
	./observer_registration_probe
