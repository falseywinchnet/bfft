#include <bfft/bfft.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    bfft_plan* plan = NULL;
    bfft_status status = bfft_plan_create(64, &plan);
    if (status != BFFT_OK) {
        fprintf(stderr, "create failed: %s\n", bfft_status_string(status));
        return 1;
    }

    const size_t n = bfft_plan_size(plan);
    const size_t bins = bfft_plan_bins(plan);
    double* input = (double*)calloc(n, sizeof(double));
    double* work = (double*)calloc(bfft_plan_work_size(plan), sizeof(double));
    double* inverse = (double*)calloc(n, sizeof(double));
    bfft_complex* output = (bfft_complex*)calloc(bins, sizeof(bfft_complex));
    bfft_complex* scratch = (bfft_complex*)calloc(bfft_plan_native_scratch_size(plan), sizeof(bfft_complex));

    if (input == NULL || work == NULL || inverse == NULL || output == NULL || scratch == NULL) {
        return 1;
    }

    for (size_t i = 0; i < n; ++i) {
        input[i] = cos(2.0 * 3.14159265358979323846 * (double)i / (double)n);
    }

    status = bfft_forward(plan, input, output, work, scratch);
    if (status != BFFT_OK) {
        fprintf(stderr, "forward failed: %s\n", bfft_status_string(status));
        return 1;
    }

    status = bfft_inverse(plan, output, inverse);
    if (status != BFFT_OK) {
        fprintf(stderr, "inverse failed: %s\n", bfft_status_string(status));
        return 1;
    }

    double max_error = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double error = fabs(input[i] - inverse[i]);
        if (error > max_error) {
            max_error = error;
        }
    }

    if (max_error > 1e-9) {
        fprintf(stderr, "roundtrip error %g\n", max_error);
        return 1;
    }

    printf("c api ok backend=%s policy=%s\n", bfft_backend_name(), bfft_plan_standard_policy(plan));

    free(input);
    free(work);
    free(inverse);
    free(output);
    free(scratch);
    bfft_plan_destroy(plan);
    return 0;
}
