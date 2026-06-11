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
    float* input_f32 = (float*)calloc(n, sizeof(float));
    float* work_f32 = (float*)calloc(bfft_plan_work_size_f32(plan), sizeof(float));
    float* inverse_f32 = (float*)calloc(n, sizeof(float));
    bfft_complex_f32* output_f32 = (bfft_complex_f32*)calloc(bins, sizeof(bfft_complex_f32));
    bfft_complex_f32* native_f32 = (bfft_complex_f32*)calloc(bins, sizeof(bfft_complex_f32));
    bfft_complex_f32* standard_f32 = (bfft_complex_f32*)calloc(bins, sizeof(bfft_complex_f32));

    if (input == NULL || work == NULL || inverse == NULL || output == NULL || scratch == NULL ||
        input_f32 == NULL || work_f32 == NULL || inverse_f32 == NULL || output_f32 == NULL ||
        native_f32 == NULL || standard_f32 == NULL) {
        return 1;
    }

    for (size_t i = 0; i < n; ++i) {
        input[i] = cos(2.0 * 3.14159265358979323846 * (double)i / (double)n);
        input_f32[i] = (float)input[i];
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

    status = bfft_forward_f32(plan, input_f32, output_f32, work_f32, NULL);
    if (status != BFFT_OK) {
        fprintf(stderr, "f32 forward failed: %s\n", bfft_status_string(status));
        return 1;
    }

    status = bfft_inverse_f32(plan, output_f32, inverse_f32);
    if (status != BFFT_OK) {
        fprintf(stderr, "f32 inverse failed: %s\n", bfft_status_string(status));
        return 1;
    }

    double max_error_f32 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double error = fabs((double)input_f32[i] - (double)inverse_f32[i]);
        if (error > max_error_f32) {
            max_error_f32 = error;
        }
    }

    if (max_error_f32 > 1e-5) {
        fprintf(stderr, "f32 roundtrip error %g\n", max_error_f32);
        return 1;
    }

    status = bfft_forward_native_f32(plan, input_f32, native_f32, work_f32);
    if (status != BFFT_OK) {
        fprintf(stderr, "f32 native forward failed: %s\n", bfft_status_string(status));
        return 1;
    }

    status = bfft_native_to_standard_f32(plan, native_f32, standard_f32);
    if (status != BFFT_OK) {
        fprintf(stderr, "f32 native conversion failed: %s\n", bfft_status_string(status));
        return 1;
    }

    double max_convert_error_f32 = 0.0;
    for (size_t i = 0; i < bins; ++i) {
        double error = fabs((double)output_f32[i].re - (double)standard_f32[i].re);
        if (error > max_convert_error_f32) {
            max_convert_error_f32 = error;
        }
        error = fabs((double)output_f32[i].im - (double)standard_f32[i].im);
        if (error > max_convert_error_f32) {
            max_convert_error_f32 = error;
        }
    }

    if (max_convert_error_f32 > 1e-5) {
        fprintf(stderr, "f32 native conversion error %g\n", max_convert_error_f32);
        return 1;
    }

    status = bfft_forward_f32(plan, NULL, output_f32, work_f32, NULL);
    if (status != BFFT_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "f32 invalid argument guard failed\n");
        return 1;
    }

    printf("c api ok backend=%s policy=%s\n", bfft_backend_name(), bfft_plan_standard_policy(plan));

    free(input);
    free(work);
    free(inverse);
    free(output);
    free(scratch);
    free(input_f32);
    free(work_f32);
    free(inverse_f32);
    free(output_f32);
    free(native_f32);
    free(standard_f32);
    bfft_plan_destroy(plan);
    return 0;
}
