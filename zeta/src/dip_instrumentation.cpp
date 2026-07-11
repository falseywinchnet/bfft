// Topology-only packet instrumentation for the trusted production kernel.
// The kernel has no observer hook; this enumerates its mathematical (d,e)
// packet tree without changing or duplicating cell arithmetic. Operator-level
// input/output energies and residuals are recorded by dip_hankel.py.
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {
void emit(FILE* f, int transform_id, const char* direction, int n,
          int depth, int d, int e) {
    const int width = n / e;
    const int phase_index = d * n / (2 * e);
    const double theta = std::acos(-1.0) * static_cast<double>(d) / e;
    std::fprintf(f, "%d,%s,%d,%d,%d,%d,%d,%.17g\n", transform_id,
                 direction, depth, d, e, width, phase_index, theta);
    if (width > 1) {
        emit(f, transform_id, direction, n, depth + 1, d, 2 * e);
        emit(f, transform_id, direction, n, depth + 1, e - d, 2 * e);
    }
}
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s N transform_id direction output.csv\n", argv[0]);
        return 2;
    }
    const int n = std::atoi(argv[1]);
    FILE* f = std::fopen(argv[4], "w");
    if (!f) return 3;
    std::fprintf(f, "transform_id,direction,depth,d,e,span_width,phase_table_index,theta\n");
    // The non-ridge packet roots of the level-8 seed, matching the header.
    if (n >= 8) {
        emit(f, std::atoi(argv[2]), argv[3], n, 0, 2, 8);
        emit(f, std::atoi(argv[2]), argv[3], n, 0, 1, 8);
        emit(f, std::atoi(argv[2]), argv[3], n, 0, 3, 8);
    } else {
        emit(f, std::atoi(argv[2]), argv[3], n, 0, 1, 4);
    }
    std::fclose(f);
    return 0;
}
