#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/../.."
python3 experiments/dip_packet_gauge/generate.py
/Users/ultimussecundai/.local/bin/m4build -- sh -c '
  set -eu
  clang++ -O3 -DNDEBUG -std=c++17 -ffp-contract=fast experiments/dip_packet_gauge/benchmark.cpp -o /tmp/dip_packet_gauge_final
  clang++ -O3 -DNDEBUG -std=c++17 -ffp-contract=fast experiments/dip_packet_gauge/cell_benchmark.cpp -o /tmp/dip_packet_cell_bench
  clang++ -O1 -g -std=c++17 -fsanitize=address,undefined -fno-omit-frame-pointer experiments/dip_packet_gauge/test_cells.cpp -o /tmp/dip_packet_gauge_test
  /tmp/dip_packet_gauge_test > /tmp/dip_packet_gauge_test.txt
  /tmp/dip_packet_gauge_final 20 15 > /tmp/dip_packet_gauge_final.jsonl
  /tmp/dip_packet_cell_bench > /tmp/dip_packet_cell_bench.jsonl
  uname -a > /tmp/dip_packet_gauge_host.txt
  clang++ --version >> /tmp/dip_packet_gauge_host.txt
  uptime >> /tmp/dip_packet_gauge_host.txt
'
dip_host="$(/Users/ultimussecundai/.local/bin/m4host)"
scp "$dip_host:/tmp/dip_packet_gauge_final.jsonl" experiments/dip_packet_gauge/final_m4.jsonl
scp "$dip_host:/tmp/dip_packet_cell_bench.jsonl" experiments/dip_packet_gauge/cell_m4.jsonl
scp "$dip_host:/tmp/dip_packet_gauge_test.txt" experiments/dip_packet_gauge/test_m4.txt
scp "$dip_host:/tmp/dip_packet_gauge_host.txt" experiments/dip_packet_gauge/host_m4.txt
python3 experiments/dip_packet_gauge/analyze.py
