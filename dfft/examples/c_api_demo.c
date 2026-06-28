#include <bfft/bfft.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const size_t n = 1024;
    bfft_plan* plan = NULL;
    bfft_status status = bfft_plan_create(n, &plan);
    if (status != BFFT_OK) {
        fprintf(stderr, "plan creation failed: %s\n", bfft_status_string(status));
        return 1;
    }

    double* input = (double*)calloc(bfft_plan_size(plan), sizeof(double));
    double* work = (double*)calloc(bfft_plan_work_size(plan), sizeof(double));
    bfft_complex* output = (bfft_complex*)calloc(bfft_plan_bins(plan), sizeof(bfft_complex));
    bfft_complex* scratch = (bfft_complex*)calloc(bfft_plan_native_scratch_size(plan), sizeof(bfft_complex));

    if (input == NULL || work == NULL || output == NULL || scratch == NULL) {
        fprintf(stderr, "allocation failed\n");
        bfft_plan_destroy(plan);
        free(input);
        free(work);
        free(output);
        free(scratch);
        return 1;
    }

    for (size_t i = 0; i < n; ++i) {
        input[i] = sin(2.0 * 3.14159265358979323846 * (double)i / (double)n);
    }

    status = bfft_forward(plan, input, output, work, scratch);
    if (status != BFFT_OK) {
        fprintf(stderr, "forward failed: %s\n", bfft_status_string(status));
        bfft_plan_destroy(plan);
        free(input);
        free(work);
        free(output);
        free(scratch);
        return 1;
    }

    printf("backend=%s policy=%s bin1=(%.6f, %.6f)\n",
           bfft_backend_name(),
           bfft_plan_standard_policy(plan),
           output[1].re,
           output[1].im);

    bfft_plan_destroy(plan);
    free(input);
    free(work);
    free(output);
    free(scratch);
    return 0;
}
