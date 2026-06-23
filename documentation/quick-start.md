# Quick start

## C API

```c
#include <bfft/bfft.h>

#include <stdlib.h>

int main(void) {
    bfft_plan* plan = NULL;
    bfft_status status = bfft_plan_create(1024, &plan);
    if (status != BFFT_OK) {
        return 1;
    }

    double* input = calloc(bfft_plan_size(plan), sizeof(double));
    double* work = calloc(bfft_plan_work_size(plan), sizeof(double));
    bfft_complex* output = calloc(bfft_plan_bins(plan), sizeof(bfft_complex));
    bfft_complex* scratch = calloc(
        bfft_plan_native_scratch_size(plan),
        sizeof(bfft_complex));

    status = bfft_forward(plan, input, output, work, scratch);

    free(input);
    free(work);
    free(output);
    free(scratch);
    bfft_plan_destroy(plan);

    if (status != BFFT_OK) {
        return 1;
    }
    return 0;
}
```

## C++ API

```cpp
#include <bfft/bfft.hpp>

#include <vector>

int main() {
    bfft::plan plan(1024);
    std::vector<double> input(plan.size());
    std::vector<bfft::complex> output = plan.forward(input);
    return output.empty();
}
```

## Magnitude-only transform

```cpp
#include <bfft/bfft.hpp>

#include <vector>

int main() {
    bfft::plan plan(1024);
    std::vector<double> input(plan.size());
    std::vector<double> magnitudes = plan.forward_magnitude(input);
    return magnitudes.empty();
}
```
