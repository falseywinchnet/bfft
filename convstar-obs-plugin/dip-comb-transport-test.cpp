#include "dip-comb-transport.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr double tau = 6.283185307179586476925286766559;

std::vector<float> harmonic_scene(uint32_t width, uint32_t height,
				  double low_amplitude, double high_amplitude)
{
	std::vector<float> image(static_cast<size_t>(width) * height);
	for (uint32_t y = 0; y < height; ++y) {
		for (uint32_t x = 0; x < width; ++x) {
			const double u = static_cast<double>(x) / width;
			const double v = static_cast<double>(y) / height;
			const double low = std::sin(tau * (3.0 * u + 2.0 * v));
			const double high = std::sin(tau * (29.0 * u - 23.0 * v +
				0.18 * std::sin(tau * 5.0 * v)));
			image[static_cast<size_t>(y) * width + x] = static_cast<float>(
				127.5 + 54.0 * low_amplitude * low +
				42.0 * high_amplitude * high);
		}
	}
	return image;
}

std::vector<float> oriented_scene(uint32_t width, uint32_t height, bool vertical)
{
	std::vector<float> image(static_cast<size_t>(width) * height);
	for (uint32_t y = 0; y < height; ++y)
		for (uint32_t x = 0; x < width; ++x) {
			const double coordinate = vertical ?
				static_cast<double>(y) / height : static_cast<double>(x) / width;
			image[static_cast<size_t>(y) * width + x] = static_cast<float>(
				127.5 + 55.0 * std::sin(tau * 7.0 * coordinate));
		}
	return image;
}

double rms(const std::vector<float> &values)
{
	double sum = 0.0;
	for (float value : values)
		sum += static_cast<double>(value) * value;
	return std::sqrt(sum / std::max<size_t>(1, values.size()));
}

double mean(const std::vector<float> &values)
{
	double sum = 0.0;
	for (float value : values)
		sum += value;
	return sum / std::max<size_t>(1, values.size());
}

bool bounded(const std::vector<float> &values, double low, double high)
{
	return std::all_of(values.begin(), values.end(), [&](float value) {
		return std::isfinite(value) && value >= low && value <= high;
	});
}

} // namespace

int main()
{
	constexpr uint32_t width = 128;
	constexpr uint32_t height = 128;
	DipCombTransport2D transport;
	DipCombField identity, low, high, mixture, no_reverse, full_reverse;
	DipCombField horizontal, vertical;
	const auto mixed_scene = harmonic_scene(width, height, 1.0, 1.0);
	if (!transport.process(mixed_scene, width, height, 0.0, 0.0, 4, identity) ||
	    !transport.process(harmonic_scene(width, height, 1.0, 0.0),
			       width, height, 1.25, 0.72, 4, low) ||
	    !transport.process(harmonic_scene(width, height, 0.0, 1.0),
			       width, height, 1.25, 0.72, 4, high) ||
	    !transport.process(mixed_scene, width, height, 1.25, 0.72, 4, mixture)) {
		std::fprintf(stderr, "DIP comb transport setup failed\n");
		return 1;
	}
	transport.process(harmonic_scene(width, height, 1.0, 0.0),
		width, height, 1.25, 0.0, 4, no_reverse);
	transport.process(harmonic_scene(width, height, 1.0, 0.0),
		width, height, 1.25, 1.0, 4, full_reverse);
	transport.process(oriented_scene(width, height, false),
		width, height, 1.0, 0.72, 4, horizontal);
	transport.process(oriented_scene(width, height, true),
		width, height, 1.0, 0.72, 4, vertical);
	const double identity_error = rms(identity.residual);
	const double low_response = rms(low.residual);
	const double high_response = rms(high.residual);
	const double unreversed_response = rms(no_reverse.residual);
	const double reversed_response = rms(full_reverse.residual);
	const double horizontal_admission = mean(horizontal.conv_admission);
	const double vertical_admission = mean(vertical.conv_admission);
	double uncertainty_mean = 0.0;
	for (float value : mixture.uncertainty)
		uncertainty_mean += value;
	uncertainty_mean /= mixture.uncertainty.size();
	const bool valid = identity_error < 1e-10 &&
		high_response > 1.05 * low_response &&
		reversed_response < unreversed_response &&
		horizontal_admission < 0.1 && vertical_admission > 0.9 &&
		bounded(mixture.displacement_x, -0.1, 0.1) &&
		bounded(mixture.displacement_y, -0.1, 0.1) &&
		bounded(mixture.residual, -0.5, 0.5) &&
		bounded(mixture.uncertainty, 0.0, 1.0) &&
		bounded(mixture.phase, -0.500001, 0.500001) &&
		bounded(mixture.conv_admission, 0.0, 1.0) &&
		bounded(mixture.conv_confidence, 0.0, 1.0) &&
		uncertainty_mean > 1e-4;
	std::printf(
		"DIP comb invariant: grid=%ux%u identity=%.3e low=%.6f high=%.6f "
		"ratio=%.3f reverse=%.6f->%.6f eta=%.3f/%.3f uncertainty=%.6f "
		"forward=%.3fms inverse2=%.3fms\n",
		transport.transform_width(), transport.transform_height(), identity_error,
		low_response, high_response, high_response / std::max(1e-12, low_response),
		unreversed_response, reversed_response, horizontal_admission,
		vertical_admission, uncertainty_mean, transport.last_timings().forward_ms,
		transport.last_timings().inverse_ms);

	const auto live_scene = harmonic_scene(720, 405, 1.0, 1.0);
	DipCombField live;
	if (!transport.process(live_scene, 720, 405, 1.25, 0.72, 8, live))
		return 3;
	const DipCombTimings &timing = transport.last_timings();
	std::printf(
		"DIP comb 720x405: grid=%ux%u total=%.3fms resample=%.3fms "
		"forward=%.3fms comb=%.3fms inverse2=%.3fms expand=%.3fms\n",
		transport.transform_width(), transport.transform_height(), timing.total_ms,
		timing.resample_ms, timing.forward_ms, timing.comb_ms,
		timing.inverse_ms, timing.expand_ms);
	const auto direct_scene = harmonic_scene(512, 256, 1.0, 1.0);
	if (!transport.process(direct_scene, 512, 256, 1.25, 0.72, 8, live))
		return 4;
	const DipCombTimings &direct = transport.last_timings();
	std::printf(
		"DIP comb direct 512x256: total=%.3fms resample=%.3fms forward=%.3fms "
		"comb=%.3fms inverse2=%.3fms field=%.3fms\n",
		direct.total_ms, direct.resample_ms, direct.forward_ms,
		direct.comb_ms, direct.inverse_ms, direct.expand_ms);
	const auto high_detail_scene = harmonic_scene(1024, 512, 1.0, 1.0);
	if (!transport.process(high_detail_scene, 1024, 512, 1.25, 0.72, 8, live) ||
	    transport.transform_width() != 1024 ||
	    transport.transform_height() != 512 ||
	    !bounded(live.uncertainty, 0.0, 1.0))
		return 5;
	// Repeat after plan/buffer growth so the printed number is steady-state.
	if (!transport.process(high_detail_scene, 1024, 512, 1.25, 0.72, 8, live))
		return 6;
	const DipCombTimings &high_detail = transport.last_timings();
	std::printf(
		"DIP comb direct 1024x512: total=%.3fms forward=%.3fms comb=%.3fms "
		"inverse2=%.3fms field=%.3fms\n",
		high_detail.total_ms, high_detail.forward_ms, high_detail.comb_ms,
		high_detail.inverse_ms, high_detail.expand_ms);
	return valid ? 0 : 2;
}
