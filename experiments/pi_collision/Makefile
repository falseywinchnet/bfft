CXX ?= clang++
CPPFLAGS += -I/opt/homebrew/opt/gmp/include -I/opt/homebrew/opt/mpfr/include
CXXFLAGS ?= -O3 -DNDEBUG -std=c++17 -Wall -Wextra -pedantic
LDFLAGS += -L/opt/homebrew/opt/gmp/lib -L/opt/homebrew/opt/mpfr/lib
LDLIBS += -lmpfr -lgmp

.PHONY: all clean bench

all: pi_collision_cpp

pi_collision_cpp: pi_collision.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

bench: pi_collision_cpp
	./pi_collision_cpp --bits 1000
	./pi_collision_cpp --bits 10000
	./pi_collision_cpp --bits 100000

clean:
	rm -f pi_collision_cpp
