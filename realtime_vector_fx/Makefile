CXX ?= c++
CXXFLAGS ?= -O3 -ffast-math -std=c++17 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Iinclude
BUILD := build
CORE := $(BUILD)/engine.o

.PHONY: all test benchmark sanitize clean
all: $(BUILD)/rvfx_entropy_tests $(BUILD)/rvfx_entropy_chain_tests $(BUILD)/rvfx_tests $(BUILD)/rvfx_benchmark $(BUILD)/rvfx_demo

$(BUILD):
	mkdir -p $(BUILD)

$(CORE): src/engine.cpp include/rvfx/engine.hpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/rvfx_tests: tests/test_engine.cpp $(CORE)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD)/rvfx_benchmark: tools/benchmark.cpp $(CORE)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD)/rvfx_demo: tools/demo.cpp $(CORE)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD)/rvfx_entropy_tests: tests/test_entropy_stretch.cpp src/entropy_stretch.cpp include/rvfx/entropy_stretch.hpp | $(BUILD)
	$(CXX) $(CPPFLAGS) -std=c++17 -O3 -Wall -Wextra tests/test_entropy_stretch.cpp src/entropy_stretch.cpp -o $@

$(BUILD)/rvfx_entropy_benchmark: tools/entropy_benchmark.cpp src/entropy_stretch.cpp tests/reference/entropy_stretch_v1.cpp include/rvfx/entropy_stretch.hpp tests/reference/entropy_stretch_v1.hpp | $(BUILD)
	$(CXX) $(CPPFLAGS) -std=c++17 -O3 tools/entropy_benchmark.cpp src/entropy_stretch.cpp tests/reference/entropy_stretch_v1.cpp -o $@

$(BUILD)/rvfx_entropy_chain_tests: tests/test_entropy_chains.cpp src/entropy_stretch.cpp include/rvfx/entropy_stretch.hpp | $(BUILD)
	$(CXX) $(CPPFLAGS) -std=c++17 -O3 -Wall -Wextra tests/test_entropy_chains.cpp src/entropy_stretch.cpp -o $@

test: $(BUILD)/rvfx_tests $(BUILD)/rvfx_entropy_tests $(BUILD)/rvfx_entropy_chain_tests $(BUILD)/rvfx_entropy_benchmark
	$(BUILD)/rvfx_tests
	$(BUILD)/rvfx_entropy_tests
	$(BUILD)/rvfx_entropy_chain_tests
	$(BUILD)/rvfx_entropy_benchmark --verify-only

benchmark: $(BUILD)/rvfx_benchmark
	$(BUILD)/rvfx_benchmark

sanitize: | $(BUILD)
	$(CXX) $(CPPFLAGS) -std=c++17 -O1 -g -fno-omit-frame-pointer \
		-fsanitize=address,undefined src/engine.cpp tests/test_engine.cpp \
		-o $(BUILD)/rvfx_tests_sanitize
	$(BUILD)/rvfx_tests_sanitize
	$(CXX) $(CPPFLAGS) -std=c++17 -O1 -g -fno-omit-frame-pointer \
		-fsanitize=address,undefined src/entropy_stretch.cpp tests/test_entropy_stretch.cpp \
		-o $(BUILD)/rvfx_entropy_tests_sanitize
	$(BUILD)/rvfx_entropy_tests_sanitize
	$(CXX) $(CPPFLAGS) -std=c++17 -O1 -g -fno-omit-frame-pointer \
		-fsanitize=address,undefined src/entropy_stretch.cpp tests/test_entropy_chains.cpp \
		-o $(BUILD)/rvfx_entropy_chain_tests_sanitize
	$(BUILD)/rvfx_entropy_chain_tests_sanitize

clean:
	rm -f $(CORE) $(BUILD)/rvfx_tests $(BUILD)/rvfx_tests_sanitize \
		$(BUILD)/rvfx_entropy_tests $(BUILD)/rvfx_entropy_tests_sanitize $(BUILD)/rvfx_entropy_benchmark \
		$(BUILD)/rvfx_entropy_chain_tests $(BUILD)/rvfx_entropy_chain_tests_sanitize \
		$(BUILD)/rvfx_benchmark $(BUILD)/rvfx_demo
