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
    double* magnitudes = (double*)calloc(bins, sizeof(double));
    bfft_complex* output = (bfft_complex*)calloc(bins, sizeof(bfft_complex));
    bfft_complex* mag_phase = (bfft_complex*)calloc(bins, sizeof(bfft_complex));
    bfft_complex* scratch = (bfft_complex*)calloc(bfft_plan_native_scratch_size(plan), sizeof(bfft_complex));
    float* input_f32 = (float*)calloc(n, sizeof(float));
    float* work_f32 = (float*)calloc(bfft_plan_work_size_f32(plan), sizeof(float));
    float* inverse_f32 = (float*)calloc(n, sizeof(float));
    float* magnitudes_f32 = (float*)calloc(bins, sizeof(float));
    bfft_complex_f32* output_f32 = (bfft_complex_f32*)calloc(bins, sizeof(bfft_complex_f32));
    bfft_complex_f32* mag_phase_f32 = (bfft_complex_f32*)calloc(bins, sizeof(bfft_complex_f32));
    bfft_complex_f32* native_f32 = (bfft_complex_f32*)calloc(bins, sizeof(bfft_complex_f32));
    bfft_complex_f32* standard_f32 = (bfft_complex_f32*)calloc(bins, sizeof(bfft_complex_f32));
    bfft_complex* native_output = (bfft_complex*)calloc(bins, sizeof(bfft_complex));
    bfft_complex* workspace_output = (bfft_complex*)calloc(bins, sizeof(bfft_complex));
    bfft_workspace* workspace = NULL;

    if (input == NULL || work == NULL || inverse == NULL || magnitudes == NULL || output == NULL ||
        scratch == NULL || mag_phase == NULL || input_f32 == NULL || work_f32 == NULL || inverse_f32 == NULL ||
        magnitudes_f32 == NULL || output_f32 == NULL || mag_phase_f32 == NULL || native_f32 == NULL ||
        standard_f32 == NULL ||
        native_output == NULL || workspace_output == NULL) {
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

    status = bfft_workspace_create(plan, &workspace);
    if (status != BFFT_OK || workspace == NULL) {
        fprintf(stderr, "workspace create failed: %s\n", bfft_status_string(status));
        return 1;
    }

    status = bfft_forward_native(plan, input, native_output, work);
    if (status != BFFT_OK) {
        fprintf(stderr, "native forward failed: %s\n", bfft_status_string(status));
        return 1;
    }

    status = bfft_forward_native_workspace(plan, workspace, input, workspace_output);
    if (status != BFFT_OK) {
        fprintf(stderr, "workspace native forward failed: %s\n", bfft_status_string(status));
        return 1;
    }

    double max_workspace_error = 0.0;
    for (size_t i = 0; i < bins; ++i) {
        double error = fabs(native_output[i].re - workspace_output[i].re);
        if (error > max_workspace_error) {
            max_workspace_error = error;
        }
        error = fabs(native_output[i].im - workspace_output[i].im);
        if (error > max_workspace_error) {
            max_workspace_error = error;
        }
    }

    if (max_workspace_error > 1e-9) {
        fprintf(stderr, "workspace forward error %g\n", max_workspace_error);
        return 1;
    }

    status = bfft_forward_magnitude(plan, input, magnitudes, work);
    if (status != BFFT_OK) {
        fprintf(stderr, "magnitude failed: %s\n", bfft_status_string(status));
        return 1;
    }

    double max_magnitude_error = 0.0;
    for (size_t i = 0; i < bins; ++i) {
        const double expected = hypot(output[i].re, output[i].im);
        const double error = fabs(magnitudes[i] - expected);
        if (error > max_magnitude_error) {
            max_magnitude_error = error;
        }
    }

    if (max_magnitude_error > 1e-9) {
        fprintf(stderr, "magnitude error %g\n", max_magnitude_error);
        return 1;
    }

    status = bfft_forward_mag_phase(plan, input, mag_phase, work);
    if (status != BFFT_OK) {
        fprintf(stderr, "mag-phase failed: %s\n", bfft_status_string(status));
        return 1;
    }

    double max_mag_phase_error = 0.0;
    for (size_t i = 0; i < bins; ++i) {
        const double re = mag_phase[i].re * cos(mag_phase[i].im);
        const double im = mag_phase[i].re * sin(mag_phase[i].im);
        double error = fabs(re - output[i].re);
        if (error > max_mag_phase_error) {
            max_mag_phase_error = error;
        }
        error = fabs(im - output[i].im);
        if (error > max_mag_phase_error) {
            max_mag_phase_error = error;
        }
    }

    if (max_mag_phase_error > 6.309573444801929e-8) {
        fprintf(stderr, "mag-phase error %g\n", max_mag_phase_error);
        return 1;
    }

    status = bfft_inverse_mag_phase(plan, mag_phase, inverse);
    if (status != BFFT_OK) {
        fprintf(stderr, "mag-phase inverse failed: %s\n", bfft_status_string(status));
        return 1;
    }

    double max_mag_phase_inverse_error = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double error = fabs(input[i] - inverse[i]);
        if (error > max_mag_phase_inverse_error) {
            max_mag_phase_inverse_error = error;
        }
    }
    if (max_mag_phase_inverse_error > 1e-9) {
        fprintf(stderr, "mag-phase inverse error %g\n", max_mag_phase_inverse_error);
        return 1;
    }

    status = bfft_inverse_mag_phase(plan, NULL, inverse);
    if (status != BFFT_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "mag-phase inverse null input guard failed\n");
        return 1;
    }

    status = bfft_forward_mag_phase(NULL, input, mag_phase, work);
    if (status != BFFT_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "mag-phase null plan guard failed\n");
        return 1;
    }
    status = bfft_forward_mag_phase(plan, NULL, mag_phase, work);
    if (status != BFFT_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "mag-phase null input guard failed\n");
        return 1;
    }
    status = bfft_forward_mag_phase(plan, input, NULL, work);
    if (status != BFFT_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "mag-phase null output guard failed\n");
        return 1;
    }
    status = bfft_forward_mag_phase(plan, input, mag_phase, NULL);
    if (status != BFFT_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "mag-phase null work guard failed\n");
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

    status = bfft_forward_magnitude_f32(plan, input_f32, magnitudes_f32, work_f32);
    if (status != BFFT_OK) {
        fprintf(stderr, "f32 magnitude failed: %s\n", bfft_status_string(status));
        return 1;
    }

    double max_magnitude_error_f32 = 0.0;
    for (size_t i = 0; i < bins; ++i) {
        const double expected = hypot((double)output_f32[i].re, (double)output_f32[i].im);
        const double error = fabs((double)magnitudes_f32[i] - expected);
        if (error > max_magnitude_error_f32) {
            max_magnitude_error_f32 = error;
        }
    }

    if (max_magnitude_error_f32 > 1e-5) {
        fprintf(stderr, "f32 magnitude error %g\n", max_magnitude_error_f32);
        return 1;
    }

    status = bfft_forward_mag_phase_f32(plan, input_f32, mag_phase_f32, work_f32);
    if (status != BFFT_OK) {
        fprintf(stderr, "f32 mag-phase failed: %s\n", bfft_status_string(status));
        return 1;
    }

    double max_mag_phase_error_f32 = 0.0;
    for (size_t i = 0; i < bins; ++i) {
        const double mag = (double)mag_phase_f32[i].re;
        const double phase = (double)mag_phase_f32[i].im;
        const double re = mag * cos(phase);
        const double im = mag * sin(phase);
        double error = fabs(re - (double)output_f32[i].re);
        if (error > max_mag_phase_error_f32) {
            max_mag_phase_error_f32 = error;
        }
        error = fabs(im - (double)output_f32[i].im);
        if (error > max_mag_phase_error_f32) {
            max_mag_phase_error_f32 = error;
        }
    }

    if (max_mag_phase_error_f32 > 1e-5) {
        fprintf(stderr, "f32 mag-phase error %g\n", max_mag_phase_error_f32);
        return 1;
    }

    status = bfft_inverse_mag_phase_f32(plan, mag_phase_f32, inverse_f32);
    if (status != BFFT_OK) {
        fprintf(stderr, "f32 mag-phase inverse failed: %s\n", bfft_status_string(status));
        return 1;
    }

    double max_mag_phase_inverse_error_f32 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double error = fabs((double)input_f32[i] - (double)inverse_f32[i]);
        if (error > max_mag_phase_inverse_error_f32) {
            max_mag_phase_inverse_error_f32 = error;
        }
    }
    if (max_mag_phase_inverse_error_f32 > 1e-5) {
        fprintf(stderr, "f32 mag-phase inverse error %g\n", max_mag_phase_inverse_error_f32);
        return 1;
    }

    status = bfft_inverse_mag_phase_f32(plan, NULL, inverse_f32);
    if (status != BFFT_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "f32 mag-phase inverse null input guard failed\n");
        return 1;
    }

    status = bfft_forward_mag_phase_f32(NULL, input_f32, mag_phase_f32, work_f32);
    if (status != BFFT_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "f32 mag-phase null plan guard failed\n");
        return 1;
    }
    status = bfft_forward_mag_phase_f32(plan, NULL, mag_phase_f32, work_f32);
    if (status != BFFT_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "f32 mag-phase null input guard failed\n");
        return 1;
    }
    status = bfft_forward_mag_phase_f32(plan, input_f32, NULL, work_f32);
    if (status != BFFT_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "f32 mag-phase null output guard failed\n");
        return 1;
    }
    status = bfft_forward_mag_phase_f32(plan, input_f32, mag_phase_f32, NULL);
    if (status != BFFT_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "f32 mag-phase null work guard failed\n");
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
    free(magnitudes);
    free(output);
    free(scratch);
    free(mag_phase);
    free(input_f32);
    free(work_f32);
    free(inverse_f32);
    free(magnitudes_f32);
    free(output_f32);
    free(mag_phase_f32);
    free(native_f32);
    free(standard_f32);
    free(native_output);
    free(workspace_output);
    bfft_workspace_destroy(workspace);
    bfft_plan_destroy(plan);
    return 0;
}
