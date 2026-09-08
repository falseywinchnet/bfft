CXX ?= c++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -pedantic
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
SHARED = -dynamiclib
else
SHARED = -shared -fPIC
endif
libmarkov.so: markov.cpp
	$(CXX) $(CXXFLAGS) $(SHARED) $< -o $@
clean:
	rm -f libmarkov.so libcv_control.so
libcv_control.so: cv_control.cpp
	$(CXX) $(CXXFLAGS) $(SHARED) $< -o $@
