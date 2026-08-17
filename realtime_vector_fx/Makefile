CXX ?= c++
CXXFLAGS ?= -O3 -ffast-math -std=c++17 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Iinclude
BUILD := build
CORE := $(BUILD)/engine.o

.PHONY: all test benchmark sanitize clean
all: $(BUILD)/rvfx_tests $(BUILD)/rvfx_benchmark $(BUILD)/rvfx_demo

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

test: $(BUILD)/rvfx_tests
	$(BUILD)/rvfx_tests

benchmark: $(BUILD)/rvfx_benchmark
	$(BUILD)/rvfx_benchmark

sanitize: | $(BUILD)
	$(CXX) $(CPPFLAGS) -std=c++17 -O1 -g -fno-omit-frame-pointer \
		-fsanitize=address,undefined src/engine.cpp tests/test_engine.cpp \
		-o $(BUILD)/rvfx_tests_sanitize
	$(BUILD)/rvfx_tests_sanitize

clean:
	rm -f $(CORE) $(BUILD)/rvfx_tests $(BUILD)/rvfx_tests_sanitize \
		$(BUILD)/rvfx_benchmark $(BUILD)/rvfx_demo
