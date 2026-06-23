# Examples and benchmarks

Build examples with the default Makefile target:

```sh
make
```

Run quick demos:

```sh
./build/examples/c_api_demo
./build/examples/cpp_api_demo
```

Run the general benchmark:

```sh
./build/examples/benchmark 4096 200
```

Run the BODFT benchmark:

```sh
./build/examples/bodft_benchmark 4096 200
```

Build optional probes:

```sh
make probes
```

Run comparison probes:

```sh
./build/tests/bfft_fftw_sfdr_bh7_probe 16 8 8 bh7 f32-native
./build/tests/bfft_library_compare_probe 12
```

The library comparison probe reports which external FFT references are available
in the current environment.
