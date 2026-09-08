#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/../.."
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
  set -eu
  mkdir -p /tmp/quartic_native_final
  python3 experiments/real_fourier_walk/test_factorization.py > /tmp/quartic_native_final/factorization.json
  clang++ -O1 -g -std=c++17 -fsanitize=address,undefined -fno-omit-frame-pointer experiments/real_fourier_walk/test_native.cpp -o /tmp/quartic_native_test
  /tmp/quartic_native_test > /tmp/quartic_native_final/sanitizer_test.json
  clang++ -O3 -DNDEBUG -std=c++17 -ffp-contract=fast experiments/real_fourier_walk/test_native.cpp -o /tmp/quartic_native_test_opt
  /tmp/quartic_native_test_opt > /tmp/quartic_native_final/optimized_test.json
  clang++ -O3 -DNDEBUG -std=c++17 -ffp-contract=fast -DBFFT_BENCH_QUARTIC_WALK -Iinclude examples/benchmark.cpp src/bfft.cpp -o /tmp/quartic_bfft_benchmark
  clang++ -O3 -DNDEBUG -std=c++17 -ffp-contract=fast -DBFFT_BENCH_QUARTIC_WALK -Iinclude -S examples/benchmark.cpp -o /tmp/quartic_native_final/benchmark.s
  python3 experiments/real_fourier_walk/run_native.py
'
quartic_host="$(/Users/ultimussecundai/.local/bin/m4host)"
mkdir -p experiments/real_fourier_walk/native_m4
scp -r "$quartic_host:/tmp/quartic_native_final/." experiments/real_fourier_walk/native_m4/
python3 experiments/real_fourier_walk/analyze_native.py
