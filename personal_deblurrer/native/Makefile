CXX ?= clang++
CXXFLAGS ?= -O3 -DNDEBUG -std=c++17 -fPIC -fvisibility=hidden \
	-Wall -Wextra -Wpedantic
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
LIBRARY := libpersonal_deblurrer.dylib
LDFLAGS ?= -dynamiclib
else
LIBRARY := libpersonal_deblurrer.so
LDFLAGS ?= -shared
endif

.PHONY: all clean

all: $(LIBRARY)

$(LIBRARY): path_operator.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@

clean:
	rm -f libpersonal_deblurrer.dylib libpersonal_deblurrer.so
