#pragma once

#include <cstdint>
#include <memory>
#include <vector>

struct DipCombField {
	uint32_t width = 0;
	uint32_t height = 0;
	std::vector<float> displacement_x;
	std::vector<float> displacement_y;
	std::vector<float> residual;
	std::vector<float> uncertainty;
	std::vector<float> phase;
	std::vector<float> conv_admission;
	std::vector<float> conv_confidence;
};

struct DipCombTimings {
	double resample_ms = 0.0;
	double forward_ms = 0.0;
	double comb_ms = 0.0;
	double inverse_ms = 0.0;
	double expand_ms = 0.0;
	double total_ms = 0.0;
};

class DipCombTransport2D {
public:
	DipCombTransport2D();
	~DipCombTransport2D();
	DipCombTransport2D(const DipCombTransport2D &) = delete;
	DipCombTransport2D &operator=(const DipCombTransport2D &) = delete;

	bool process(const std::vector<float> &oklch_lightness,
		     uint32_t width, uint32_t height, double transpose_strength,
		     double reverse_strength, int thread_count, DipCombField &field);

	uint32_t transform_width() const noexcept;
	uint32_t transform_height() const noexcept;
	const char *simd_backend() const noexcept;
	const DipCombTimings &last_timings() const noexcept;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};
