# Installation

## Build requirements

- C++17-capable compiler.
- C compiler for C examples and C consumers.
- `make` or CMake 3.16 or newer.
- Python and Sphinx only when building this documentation site.

## Build with Make

```sh
make
make test
```

The default build writes libraries and examples to `build/`.

## Build with CMake

```sh
cmake -S . -B build-cmake
cmake --build build-cmake
ctest --test-dir build-cmake --output-on-failure
```

## Install

```sh
sudo make install PREFIX=/usr/local
```

For packaging, stage the install:

```sh
make install DESTDIR=/tmp/bfft-package PREFIX=/usr
```

## Package discovery

With `pkg-config`:

```sh
cc app.c $(pkg-config --cflags --libs bfft)
```

With CMake:

```cmake
find_package(BFFT CONFIG REQUIRED)
add_executable(app app.cpp)
target_link_libraries(app PRIVATE bfft::static)
```

Use `bfft::shared` when the shared target was built and installed by CMake.
