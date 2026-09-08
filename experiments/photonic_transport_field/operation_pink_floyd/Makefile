CXX ?= clang++
CXXFLAGS ?= -O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic

.PHONY: all clean test render-fourier render-reference render-both

all: pink_floyd_native

pink_floyd_native: pink_floyd_native.cpp
	$(CXX) $(CXXFLAGS) -mcpu=native $< -o $@

test: pink_floyd_native
	./pink_floyd_native --self-test

render-fourier: pink_floyd_native
	./pink_floyd_native --backend fourier --out /tmp/pink_floyd_fourier.ppm

render-reference: pink_floyd_native
	./pink_floyd_native --backend reference --out /tmp/pink_floyd_reference.ppm

render-both: pink_floyd_native
	./pink_floyd_native --backend fourier --out /tmp/pink_floyd_fourier.ppm
	./pink_floyd_native --backend reference --out /tmp/pink_floyd_reference.ppm

clean:
	rm -f pink_floyd_native
