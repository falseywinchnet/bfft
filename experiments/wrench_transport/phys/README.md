# Wrench Physics 0.1

A standalone, dependency-free C++20 rigid-body engine.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
build/wrench_minimal
```

Embed this directory with `add_subdirectory` and link `Wrench::Physics`,
or install it and call `find_package(WrenchPhysics CONFIG REQUIRED)`.
Include `<physics.hpp>`; the namespace is `zc::phys`.

Convex and compound bodies, static geometry, friction, restitution, rolling
resistance, sleeping islands, and a crane hold are supported. No rendering
or platform windowing code is required. There is no CCD or general joint system.

Research, source hashes, repeated benchmarks, and the paper:
https://github.com/falseywinchnet/bfft/tree/codex/wrench-engine-study/experiments/wrench_transport
