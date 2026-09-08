#include <obs-module.h>

#include "dip-comb-transport.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

OBS_DECLARE_MODULE()

MODULE_EXPORT const char *obs_module_description(void)
{
	return "DIP-CONV Gaussian harmonic-comb transport";
}

namespace {

constexpr const char *kMode = "convstar_toy_mode";
constexpr const char *kAmount = "convstar_toy_amount";
constexpr const char *kDeformation = "convstar_support_deformation";
constexpr const char *kSpeed = "convstar_toy_speed";
constexpr const char *kResolution = "convstar_support_resolution";
constexpr const char *kThreads = "convstar_cpu_threads";
constexpr const char *kHuePerversity = "convstar_oklch_hue_perversity";
constexpr const char *kLightnessPerversity = "convstar_oklch_lightness_perversity";
constexpr const char *kDipReverse = "convstar_dip_reverse";
const char *kEffect = R"(
uniform float4x4 ViewProj;
uniform texture2d image;
uniform texture2d transport_image;
uniform texture2d oklch_action_image;
uniform float2 support_texel;
uniform float support_blur;
uniform float transport_span;
uniform float oklch_colour;
uniform float hue_perversity;
uniform float lightness_perversity;

sampler_state linear_sampler {
	Filter = Linear;
	AddressU = Clamp;
	AddressV = Clamp;
};

struct VertData {
	float4 pos : POSITION;
	float2 uv : TEXCOORD0;
};

VertData VSDefault(VertData v_in)
{
	VertData v_out;
	v_out.pos = mul(float4(v_in.pos.xyz, 1.0), ViewProj);
	v_out.uv = v_in.uv;
	return v_out;
}

float3 srgb_to_linear(float3 c)
{
	float3 low = c / 12.92;
	float3 high = pow((c + 0.055) / 1.055, float3(2.4, 2.4, 2.4));
	return lerp(low, high, step(0.04045, c));
}

float3 linear_to_srgb(float3 c)
{
	c = max(c, 0.0);
	float3 low = 12.92 * c;
	float3 high = 1.055 * pow(c, float3(1.0 / 2.4, 1.0 / 2.4,
					     1.0 / 2.4)) - 0.055;
	return saturate(lerp(low, high, step(0.0031308, c)));
}

float3 linear_to_oklab(float3 c)
{
	float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
	float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
	float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
	float3 root = pow(max(float3(l, m, s), 0.0),
			  float3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0));
	return float3(
		0.2104542553 * root.x + 0.7936177850 * root.y - 0.0040720468 * root.z,
		1.9779984951 * root.x - 2.4285922050 * root.y + 0.4505937099 * root.z,
		0.0259040371 * root.x + 0.7827717662 * root.y - 0.8086757660 * root.z);
}

float3 oklab_to_linear(float3 lab)
{
	float l = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
	float m = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
	float s = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
	l = l * l * l;
	m = m * m * m;
	s = s * s * s;
	return float3(
		 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
		-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
		-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
}

float4 PSConvStar(VertData v_in) : TARGET
{
	float2 d = support_texel * support_blur;
	float2 encoded = transport_image.Sample(linear_sampler, v_in.uv).rg;
	if (support_blur > 0.5) {
		encoded *= 4.0;
		encoded += 2.0 * transport_image.Sample(linear_sampler, v_in.uv + float2(d.x, 0.0)).rg;
		encoded += 2.0 * transport_image.Sample(linear_sampler, v_in.uv - float2(d.x, 0.0)).rg;
		encoded += 2.0 * transport_image.Sample(linear_sampler, v_in.uv + float2(0.0, d.y)).rg;
		encoded += 2.0 * transport_image.Sample(linear_sampler, v_in.uv - float2(0.0, d.y)).rg;
		encoded += transport_image.Sample(linear_sampler, v_in.uv + d).rg;
		encoded += transport_image.Sample(linear_sampler, v_in.uv - d).rg;
		encoded += transport_image.Sample(linear_sampler, v_in.uv + float2(d.x, -d.y)).rg;
		encoded += transport_image.Sample(linear_sampler, v_in.uv + float2(-d.x, d.y)).rg;
		encoded /= 16.0;
	}
	float4 result = image.Sample(
		linear_sampler, v_in.uv + (encoded - (128.0 / 255.0)) * transport_span);
	if (oklch_colour > 0.5) {
		float4 action = oklch_action_image.Sample(linear_sampler, v_in.uv);
		float hue_turn = (action.r - (128.0 / 255.0)) * hue_perversity;
		float chroma_log = (action.g - (128.0 / 255.0)) * hue_perversity;
		float lightness_shift =
			(action.b - (128.0 / 255.0)) * lightness_perversity;
		float lightness_fold =
			(action.a - (128.0 / 255.0)) * lightness_perversity;
		float angle = hue_turn * 6.28318530718;
		float chroma_scale = exp(chroma_log * 2.0);
		float c = cos(angle);
		float s = sin(angle);
		float3 lab = linear_to_oklab(srgb_to_linear(result.rgb));
		lab.yz = chroma_scale * float2(c * lab.y - s * lab.z,
						 s * lab.y + c * lab.z);

		// The C++ DIP/Conv* operator has already constructed the action.
		// The shader performs only its OKLCH exponential-coordinate update.
		lab.x = saturate(lab.x + lightness_perversity * lightness_shift);
		result.rgb = linear_to_srgb(oklab_to_linear(lab));
	}
	return result;
}

technique Draw {
	pass {
		vertex_shader = VSDefault(v_in);
		pixel_shader = PSConvStar(v_in);
	}
}
)";

int sign_of(double value)
{
	return (value > 0.0) - (value < 0.0);
}

class ConvStarEngine {
	using Fibre = std::array<double, 5>;
	struct Descriptor {
		float mean = 0.0f;
		float signed_mean = 0.0f;
		float spread = 0.0f;
		float cancellation = 0.0f;
		float confidence = 0.0f;
		std::array<float, 5> probability{0.0f, 0.0f, 1.0f, 0.0f,
						 0.0f};
	};
	struct TransportPixel {
		float x = 0.0f;
		float y = 0.0f;
		float hue = 0.0f;
		float chroma = 0.0f;
		float lightness_shift = 0.0f;
		float lightness_fold = 0.0f;
	};
	std::vector<Descriptor> horizontal_cache;
	std::vector<Descriptor> vertical_cache;

public:
	static bool self_test()
	{
		const std::vector<double> source = {0.0, 1.0, 4.0, 2.0,
						    -1.0, 3.0, 5.0};
		std::vector<Fibre> raw, ledger;
		std::vector<double> delta;
		raw_current_bank(source, raw, delta);
		ordered_ledger(raw, delta, ledger);
		const Fibre first = project_fibre(raw[0], ledger[0], delta[0]);
		const Fibre last = project_fibre(raw[5], ledger[5], delta[5]);
		const auto near = [](double a, double b) {
			return std::abs(a - b) < 2e-12;
		};
		return near(raw[1][2], 1.5041666666666664) &&
		       ledger[2] == Fibre{1.0, -1.0, -1.0, -1.0, -1.0} &&
		       near(first[0], 0.0) && near(first[1], 0.4) &&
		       near(first[2], 0.0) && near(first[3], 0.25) &&
		       near(first[4], 0.35) && near(last[0], 0.5875) &&
		       near(last[1], 0.4875) && near(last[2], 0.0) &&
		       near(last[3], 0.7375) && near(last[4], 0.1875);
	}

	void process(const std::vector<float> &lightness_source,
		     const std::vector<float> &chroma_x_source,
		     const std::vector<float> &chroma_y_source, uint32_t width,
		     uint32_t height, int mode, double amount, double phase,
		     double deformation, int thread_count,
		     const DipCombField *dip_field,
		     std::vector<float> &output)
	{
		if (lightness_source.size() != static_cast<size_t>(width) * height ||
		    chroma_x_source.size() != lightness_source.size() ||
		    chroma_y_source.size() != lightness_source.size())
			return;
		amount = std::clamp(amount, 0.0, 1.0);
		if (amount <= 1e-9 || width < 10 || height < 10) {
			output.assign(lightness_source.size() * 6, 0.0f);
			return;
		}
		const bool has_dip = dip_field &&
			dip_field->displacement_x.size() == lightness_source.size() &&
			dip_field->conv_admission.size() == lightness_source.size();
		if (mode == 13 && has_dip) {
			output.resize(lightness_source.size() * 6);
			const int workers = std::clamp(
				thread_count, 1,
				static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
			std::atomic<uint32_t> next_row{0};
			auto worker = [&]() {
				for (;;) {
					const uint32_t y = next_row.fetch_add(1);
					if (y >= height)
						break;
					for (uint32_t x = 0; x < width; ++x) {
						const size_t i = static_cast<size_t>(y) * width + x;
						const double eta = std::clamp(
							static_cast<double>(dip_field->conv_admission[i]), 0.0, 1.0);
						const double confidence = 0.35 + 0.65 * std::clamp(
							static_cast<double>(dip_field->conv_confidence[i]), 0.0, 1.0);
						const double dx = dip_field->displacement_x[i];
						const double dy = dip_field->displacement_y[i];
						// The two Cartesian readings are exchanged by the same
						// involution used to transpose the DIP comb coordinates.
						const double admitted_x = (1.0 - eta) * dx + eta * dy;
						const double admitted_y = (1.0 - eta) * dy + eta * dx;
						const double gain = amount * confidence;
						output[6 * i] = static_cast<float>(std::clamp(
							5.5 * gain * admitted_x, -0.48, 0.48));
						output[6 * i + 1] = static_cast<float>(std::clamp(
							5.5 * gain * admitted_y, -0.48, 0.48));
						output[6 * i + 2] = static_cast<float>(std::clamp(
							gain * dip_field->phase[i], -0.48, 0.48));
						output[6 * i + 3] = static_cast<float>(std::clamp(
							-0.50 * gain * dip_field->uncertainty[i], -0.48, 0.48));
						output[6 * i + 4] = static_cast<float>(std::clamp(
							2.4 * gain * dip_field->residual[i], -0.48, 0.48));
						output[6 * i + 5] = 0.0f;
					}
				}
			};
			std::vector<std::thread> pool;
			for (int i = 1; i < workers; ++i)
				pool.emplace_back(worker);
			worker();
			for (auto &thread : pool)
				thread.join();
			return;
		}
		horizontal_cache.resize(lightness_source.size());
		vertical_cache.resize(lightness_source.size());
		auto &horizontal = horizontal_cache;
		auto &vertical = vertical_cache;
		output.resize(lightness_source.size() * 6);
		const auto profile_start = std::chrono::steady_clock::now();
		const int workers = std::clamp(
			thread_count, 1,
			static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
		constexpr uint32_t line_stride = 4;
		std::atomic<uint32_t> next_row{0};
		auto horizontal_worker = [&]() {
			std::vector<double> input_line(width);
			std::vector<Descriptor> descriptors;
			for (;;) {
				const uint32_t y = next_row.fetch_add(line_stride);
				if (y >= height)
					break;
				for (uint32_t x = 0; x < width; ++x)
					input_line[x] =
						lightness_source[static_cast<size_t>(y) * width + x];
				analyze_line(input_line, descriptors);
				for (uint32_t x = 0; x < width; ++x)
					horizontal[static_cast<size_t>(y) * width + x] =
							descriptors[x];
			}
		};
		std::vector<std::thread> pool;
		for (int worker = 1; worker < workers; ++worker)
			pool.emplace_back(horizontal_worker);
		horizontal_worker();
		for (auto &thread : pool)
			thread.join();
		for (uint32_t y = 0; y < height; ++y) {
			if (y % line_stride == 0)
				continue;
			const uint32_t upper = (y / line_stride) * line_stride;
			const uint32_t lower = upper + line_stride < height ?
				upper + line_stride : upper;
			const double mix = lower == upper ? 0.0 :
				static_cast<double>(y - upper) / (lower - upper);
			for (uint32_t x = 0; x < width; ++x) {
				const size_t i = static_cast<size_t>(y) * width + x;
				horizontal[i] = interpolate(
					horizontal[static_cast<size_t>(upper) * width + x],
					horizontal[static_cast<size_t>(lower) * width + x], mix);
			}
		}
		const auto profile_horizontal = std::chrono::steady_clock::now();

		std::atomic<uint32_t> next_column{0};
		auto vertical_worker = [&]() {
			std::vector<double> input_line(height);
			std::vector<Descriptor> descriptors;
			for (;;) {
				const uint32_t x = next_column.fetch_add(line_stride);
				if (x >= width)
					break;
				for (uint32_t y = 0; y < height; ++y)
					input_line[y] = lightness_source[
						static_cast<size_t>(y) * width + x];
				analyze_line(input_line, descriptors);
				for (uint32_t y = 0; y < height; ++y)
					vertical[static_cast<size_t>(y) * width + x] =
							descriptors[y];
			}
		};
		pool.clear();
		for (int worker = 1; worker < workers; ++worker)
			pool.emplace_back(vertical_worker);
		vertical_worker();
		for (auto &thread : pool)
			thread.join();
		for (uint32_t x = 0; x < width; ++x) {
			if (x % line_stride == 0)
				continue;
			const uint32_t left = (x / line_stride) * line_stride;
			const uint32_t right = left + line_stride < width ?
				left + line_stride : left;
			const double mix = right == left ? 0.0 :
				static_cast<double>(x - left) / (right - left);
			for (uint32_t y = 0; y < height; ++y) {
				const size_t i = static_cast<size_t>(y) * width + x;
				vertical[i] = interpolate(
					vertical[static_cast<size_t>(y) * width + left],
					vertical[static_cast<size_t>(y) * width + right], mix);
			}
		}
		const auto profile_vertical = std::chrono::steady_clock::now();

		const uint32_t synthesis_stride = 1u;
		std::atomic<uint32_t> next_output_row{0};
		auto synthesis_worker = [&]() {
			for (;;) {
				const uint32_t y = next_output_row.fetch_add(synthesis_stride);
				if (y >= height)
					break;
				for (uint32_t x = 0; x < width; x += synthesis_stride) {
					const size_t i = static_cast<size_t>(y) * width + x;
					const size_t left = static_cast<size_t>(y) * width +
						(x == 0 ? 0 : x - 1);
					const size_t right = static_cast<size_t>(y) * width +
						std::min(width - 1, x + 1);
					const size_t up = static_cast<size_t>(y == 0 ? 0 : y - 1) *
						width + x;
					const size_t down = static_cast<size_t>(std::min(height - 1, y + 1)) *
						width + x;
					const size_t partner =
						static_cast<size_t>(height - 1 - y) * width +
						(width - 1 - x);
					const double chromatic_x = 1.8 *
						((chroma_x_source[right] - chroma_x_source[left]) +
						 0.35 * (chroma_y_source[right] - chroma_y_source[left]));
					const double chromatic_y = 1.8 *
						((chroma_y_source[down] - chroma_y_source[up]) +
						 0.35 * (chroma_x_source[down] - chroma_x_source[up]));
					const TransportPixel transport = synthesize(
						horizontal[i], vertical[i], horizontal[partner],
						vertical[partner], chromatic_x, chromatic_y,
						chroma_x_source[partner], chroma_y_source[partner],
						has_dip ? dip_field->displacement_x[i] : 0.0,
						has_dip ? dip_field->displacement_y[i] : 0.0,
						has_dip ? dip_field->residual[i] : 0.0,
						has_dip ? dip_field->uncertainty[i] : 0.0,
						has_dip ? dip_field->phase[i] : 0.0,
						x, y, width, height, mode,
						amount, deformation, phase);
					output[6 * i] = transport.x;
					output[6 * i + 1] = transport.y;
					output[6 * i + 2] = transport.hue;
					output[6 * i + 3] = transport.chroma;
					output[6 * i + 4] = transport.lightness_shift;
					output[6 * i + 5] = transport.lightness_fold;
				}
			}
		};
		pool.clear();
		for (int worker = 1; worker < workers; ++worker)
			pool.emplace_back(synthesis_worker);
		synthesis_worker();
		for (auto &thread : pool)
			thread.join();
		if (synthesis_stride == 2) {
			for (uint32_t y = 0; y < height; y += 2) {
				for (uint32_t x = 1; x < width; x += 2) {
					const uint32_t left = x - 1;
					const uint32_t right = x + 1 < width ? x + 1 : left;
					const size_t i = static_cast<size_t>(y) * width + x;
					const size_t a = static_cast<size_t>(y) * width + left;
					const size_t b = static_cast<size_t>(y) * width + right;
					for (size_t channel = 0; channel < 6; ++channel)
						output[6 * i + channel] =
							0.5f * (output[6 * a + channel] +
								output[6 * b + channel]);
				}
			}
			for (uint32_t y = 1; y < height; y += 2) {
				const uint32_t upper = y - 1;
				const uint32_t lower = y + 1 < height ? y + 1 : upper;
				for (uint32_t x = 0; x < width; ++x) {
					const size_t i = static_cast<size_t>(y) * width + x;
					const size_t a = static_cast<size_t>(upper) * width + x;
					const size_t b = static_cast<size_t>(lower) * width + x;
					for (size_t channel = 0; channel < 6; ++channel)
						output[6 * i + channel] =
							0.5f * (output[6 * a + channel] +
								output[6 * b + channel]);
				}
			}
		}
		static std::atomic<bool> profiled{false};
		if (!profiled.exchange(true)) {
			const auto profile_end = std::chrono::steady_clock::now();
			blog(LOG_INFO,
			     "[DIP-CONV legacy] engine split H %.2f ms, V %.2f ms, synth %.2f ms",
			     std::chrono::duration<double, std::milli>(profile_horizontal - profile_start).count(),
			     std::chrono::duration<double, std::milli>(profile_vertical - profile_horizontal).count(),
			     std::chrono::duration<double, std::milli>(profile_end - profile_vertical).count());
		}
	}

private:
	static void local_two_jet(const std::vector<double> &source,
				  std::vector<double> &first,
				  std::vector<double> &second)
	{
		const size_t n = source.size();
		first.resize(n);
		second.resize(n);
		first[0] = (-3.0 * source[0] + 4.0 * source[1] - source[2]) / 2.0;
		first[1] = (source[2] - source[0]) / 2.0;
		first[n - 2] = (source[n - 1] - source[n - 3]) / 2.0;
		first[n - 1] =
			(3.0 * source[n - 1] - 4.0 * source[n - 2] +
			 source[n - 3]) /
			2.0;
		second[0] = 2.0 * source[0] - 5.0 * source[1] +
			    4.0 * source[2] - source[3];
		second[1] = source[0] - 2.0 * source[1] + source[2];
		second[n - 2] =
			source[n - 3] - 2.0 * source[n - 2] + source[n - 1];
		second[n - 1] = 2.0 * source[n - 1] - 5.0 * source[n - 2] +
				4.0 * source[n - 3] - source[n - 4];
		for (size_t i = 2; i + 2 < n; ++i) {
			first[i] = (source[i - 2] - 8.0 * source[i - 1] +
				    8.0 * source[i + 1] - source[i + 2]) /
				   12.0;
			second[i] = (-source[i + 2] + 16.0 * source[i + 1] -
				     30.0 * source[i] + 16.0 * source[i - 1] -
				     source[i - 2]) /
				    12.0;
		}
	}

	static void raw_current_bank(const std::vector<double> &source,
				     std::vector<Fibre> &raw,
				     std::vector<double> &delta)
	{
		std::vector<double> first, second;
		local_two_jet(source, first, second);
		const size_t intervals = source.size() - 1;
		raw.resize(intervals);
		delta.resize(intervals);
		for (size_t i = 0; i < intervals; ++i) {
			delta[i] = source[i + 1] - source[i];
			raw[i][0] = first[i] / 5.0;
			raw[i][1] = first[i] / 5.0 + second[i] / 20.0;
			raw[i][2] = delta[i] -
				(2.0 / 5.0) * (first[i] + first[i + 1]) +
				(second[i + 1] - second[i]) / 20.0;
			raw[i][3] = first[i + 1] / 5.0 - second[i + 1] / 20.0;
			raw[i][4] = first[i + 1] / 5.0;
		}
	}

	static void ordered_ledger(const std::vector<Fibre> &raw,
				   const std::vector<double> &delta,
				   std::vector<Fibre> &ledger)
	{
		const size_t intervals = raw.size();
		const size_t total = intervals * 5;
		std::vector<int> coarse(intervals);
		for (size_t i = 0; i < intervals; ++i)
			coarse[i] = sign_of(delta[i]);
		for (size_t i = 1; i < intervals; ++i)
			if (coarse[i] == 0)
				coarse[i] = coarse[i - 1];
		for (size_t i = intervals - 1; i-- > 0;)
			if (coarse[i] == 0)
				coarse[i] = coarse[i + 1];
		ledger.assign(intervals, Fibre{});
		if (std::none_of(coarse.begin(), coarse.end(),
				 [](int value) { return value != 0; }))
			return;

		std::vector<double> flat_raw(total);
		for (size_t i = 0; i < intervals; ++i)
			for (size_t k = 0; k < 5; ++k)
				flat_raw[5 * i + k] = raw[i][k];
		std::vector<std::pair<size_t, int>> boundaries;
		size_t previous = 0;
		for (size_t knot = 1; knot < intervals; ++knot) {
			if (coarse[knot - 1] == coarse[knot])
				continue;
			const size_t centre = 5 * knot;
			const size_t begin = std::max(previous + 1,
						      centre > 4 ? centre - 4 : size_t{1});
			const size_t end = std::min(total, centre + 5);
			const size_t local_begin = centre > 5 ? centre - 5 : 0;
			const size_t local_end = std::min(total, centre + 5);
			double best_cost = std::numeric_limits<double>::infinity();
			size_t best = begin;
			for (size_t candidate = begin; candidate < end; ++candidate) {
				double cost = 0.0;
				for (size_t j = local_begin; j < local_end; ++j) {
					const int expected = j < candidate
						? coarse[knot - 1]
						: coarse[knot];
					if (expected * flat_raw[j] < 0.0)
						cost += flat_raw[j] * flat_raw[j];
				}
				if (cost < best_cost) {
					best_cost = cost;
					best = candidate;
				}
			}
			boundaries.emplace_back(best, coarse[knot]);
			previous = best;
		}

		int current_sign = coarse[0];
		size_t boundary = 0;
		for (size_t j = 0; j < total; ++j) {
			while (boundary < boundaries.size() &&
			       j >= boundaries[boundary].first) {
				current_sign = boundaries[boundary].second;
				++boundary;
			}
			ledger[j / 5][j % 5] = static_cast<double>(current_sign);
		}
	}

	static Fibre project_fibre(const Fibre &raw, const Fibre &sign,
				   double delta)
	{
		Fibre result{};
		bool valid_sign = false;
		for (double value : sign)
			valid_sign = valid_sign || value != 0.0;
		if (!valid_sign)
			return result;
		Fibre sorted = raw;
		std::stable_sort(sorted.begin(), sorted.end());
		const double tolerance = 1e-11 * std::max(1.0, std::abs(delta));
		for (size_t split = 0; split <= 5; ++split) {
			const double lower = split == 0
				? -std::numeric_limits<double>::infinity()
				: sorted[split - 1];
			const double upper = split == 5
				? std::numeric_limits<double>::infinity()
				: sorted[split];
			double sum = 0.0;
			int active = 0;
			for (size_t k = 0; k < 5; ++k) {
				const bool on = (sign[k] > 0.0 && raw[k] >= upper) ||
						(sign[k] < 0.0 && raw[k] <= lower);
				if (on) {
					sum += raw[k];
					++active;
				}
			}
			if (!active)
				continue;
			const double lambda = (sum - delta) / active;
			if (lambda < lower - tolerance || lambda > upper + tolerance)
				continue;
			double reconstructed = 0.0;
			for (size_t k = 0; k < 5; ++k) {
				const double shifted = raw[k] - lambda;
				if (sign[k] * shifted > 0.0)
					result[k] = shifted;
				reconstructed += result[k];
			}
			if (std::abs(reconstructed - delta) <=
			    1e-8 * std::max(1.0, std::abs(delta)))
				return result;
			result.fill(0.0);
		}
		// This is unreachable for the Conv* ledger, but retain exact endpoint
		// transport under floating-point degeneracy.
		result[2] = delta;
		return result;
	}

	static Descriptor describe_fibre(const Fibre &current, double delta)
	{
		Descriptor descriptor;
		double mass = 0.0;
		for (double value : current)
			mass += std::abs(value);
		if (mass <= 1e-10)
			return descriptor;
		constexpr std::array<double, 5> position = {-1.0, -0.5, 0.0,
							     0.5, 1.0};
		double mean = 0.0;
		double signed_mean = 0.0;
		for (size_t k = 0; k < 5; ++k) {
			descriptor.probability[k] =
				static_cast<float>(std::abs(current[k]) / mass);
			mean += position[k] * descriptor.probability[k];
			signed_mean += position[k] * current[k] / mass;
		}
		double variance = 0.0;
		for (size_t k = 0; k < 5; ++k) {
			const double centred = position[k] - mean;
			variance += centred * centred * descriptor.probability[k];
		}
		descriptor.mean = static_cast<float>(mean);
		descriptor.signed_mean = static_cast<float>(signed_mean);
		descriptor.spread = static_cast<float>(std::sqrt(variance));
		descriptor.cancellation = static_cast<float>(
			std::clamp(1.0 - std::abs(delta) / mass, 0.0, 1.0));
		descriptor.confidence = static_cast<float>(1.0 - std::exp(-mass / 14.0));
		return descriptor;
	}

	static Descriptor interpolate(const Descriptor &a, const Descriptor &b,
				      double mix)
	{
		const float t = static_cast<float>(std::clamp(mix, 0.0, 1.0));
		const float s = 1.0f - t;
		Descriptor result;
		result.mean = s * a.mean + t * b.mean;
		result.signed_mean = s * a.signed_mean + t * b.signed_mean;
		result.spread = s * a.spread + t * b.spread;
		result.cancellation = s * a.cancellation + t * b.cancellation;
		result.confidence = s * a.confidence + t * b.confidence;
		float sum = 0.0f;
		for (size_t k = 0; k < 5; ++k) {
			result.probability[k] = s * a.probability[k] + t * b.probability[k];
			sum += result.probability[k];
		}
		if (sum > 1e-8f)
			for (float &value : result.probability)
				value /= sum;
		return result;
	}

	static Descriptor average(const Descriptor &a, const Descriptor &b)
	{
		return interpolate(a, b, 0.5);
	}

	static void analyze_line(const std::vector<double> &source,
				 std::vector<Descriptor> &output)
	{
		const size_t n = source.size();
		output.assign(n, Descriptor{});
		if (n < 5)
			return;
		std::vector<Fibre> raw, ledger;
		std::vector<double> delta;
		raw_current_bank(source, raw, delta);
		ordered_ledger(raw, delta, ledger);
		std::vector<Descriptor> intervals(raw.size());
		for (size_t i = 0; i < raw.size(); ++i)
			intervals[i] = describe_fibre(
				project_fibre(raw[i], ledger[i], delta[i]), delta[i]);
		output.front() = intervals.front();
		output.back() = intervals.back();
		for (size_t i = 1; i + 1 < n; ++i)
			output[i] = average(intervals[i - 1], intervals[i]);
	}

	static double inverse_discrete_cdf(const std::array<float, 5> &p,
					   double quantile)
	{
		quantile = std::clamp(quantile, 0.0, 1.0 - 1e-9);
		double cumulative = 0.0;
		for (size_t k = 0; k < 5; ++k) {
			cumulative += p[k];
			if (quantile < cumulative)
				return -1.0 + 0.5 * static_cast<double>(k);
		}
		return 1.0;
	}

	static double quantile_transport(const Descriptor &source,
					 const std::array<float, 5> &target,
					 double quantile)
	{
		return inverse_discrete_cdf(target, quantile) -
		       inverse_discrete_cdf(source.probability, quantile);
	}

	static uint32_t mix_hash(uint32_t value)
	{
		value ^= value >> 16;
		value *= 0x7feb352dU;
		value ^= value >> 15;
		value *= 0x846ca68bU;
		return value ^ (value >> 16);
	}

	static double hash01(uint32_t x, uint32_t y, uint32_t salt)
	{
		const uint32_t value = mix_hash(x * 0x9e3779b9U ^
						y * 0x85ebca6bU ^ salt);
		return static_cast<double>(value) /
		       static_cast<double>(std::numeric_limits<uint32_t>::max());
	}

	static TransportPixel synthesize(const Descriptor &h, const Descriptor &v,
					 const Descriptor &partner_h,
					 const Descriptor &partner_v,
					 double chromatic_x, double chromatic_y,
					 double partner_chroma_x,
					 double partner_chroma_y,
					 double dip_x, double dip_y,
					 double dip_residual, double dip_uncertainty,
					 double dip_phase,
					 uint32_t x, uint32_t y, uint32_t width,
					 uint32_t height, int mode, double amount,
					 double deformation, double phase)
	{
		constexpr double tau = 6.2831853071795864769;
		const double u = (static_cast<double>(x) + 0.5) / width;
		const double w = (static_cast<double>(y) + 0.5) / height;
		const double radial_x = 2.0 * u - 1.0;
		const double radial_y = 2.0 * w - 1.0;
		const double radial_norm =
			std::max(0.18, std::sqrt(radial_x * radial_x + radial_y * radial_y));
		const double rx = radial_x / radial_norm;
		const double ry = radial_y / radial_norm;
		const double confidence = std::clamp(
			0.20 + 0.80 * std::max(h.confidence, v.confidence), 0.0, 1.0);
		if (mode == 13) {
			const double gain = amount;
			const double admitted = 0.45 + 0.55 * confidence;
			TransportPixel result;
			result.x = static_cast<float>(std::clamp(
				gain * admitted * (5.5 * dip_x +
					0.012 * dip_uncertainty * h.signed_mean),
				-0.48, 0.48));
			result.y = static_cast<float>(std::clamp(
				gain * admitted * (5.5 * dip_y +
					0.012 * dip_uncertainty * v.signed_mean),
				-0.48, 0.48));
			result.hue = static_cast<float>(std::clamp(
				gain * (0.48 * dip_phase + 0.18 * dip_uncertainty *
					(h.signed_mean - v.signed_mean)), -0.48, 0.48));
			result.chroma = static_cast<float>(std::clamp(
				gain * 0.42 * (dip_uncertainty - 0.30), -0.48, 0.48));
			result.lightness_shift = static_cast<float>(std::clamp(
				gain * 2.4 * dip_residual, -0.48, 0.48));
			return result;
		}
		const double spread = 0.5 * (h.spread + v.spread);
		const double cancel = 0.5 * (h.cancellation + v.cancellation);
		double bx = h.mean + chromatic_x + 0.34 * (h.spread - 0.42) *
					 std::sin(tau * w + 0.7 * phase);
		double by = v.mean + chromatic_y + 0.34 * (v.spread - 0.42) *
					 std::sin(tau * u - 0.9 * phase);
		double tx = -by;
		double ty = bx;
		double dx = bx;
		double dy = by;
		double hue_turns = 0.0;
		double chroma_log = 0.0;
		double lightness_shift = 0.0;
		double lightness_fold = 0.0;

		switch (mode) {
		case 0: { // Conv* support pushes outward from the image centre.
			const double puff = 0.35 + spread + 0.55 * cancel;
			dx = bx + rx * puff;
			dy = by + ry * puff;
			break;
		}
		case 1: { // Tangent field with moving wide ribbons.
			const double ribbon =
				std::sin(tau * (2.0 * u + 1.35 * w) + phase);
			dx = tx * (0.55 + 0.85 * ribbon) + 0.28 * bx;
			dy = ty * (0.55 + 0.85 * ribbon) + 0.28 * by;
			break;
		}
		case 2: { // Rotate the admitted support direction continuously.
			const double angle = phase + tau * (0.35 * u - 0.27 * w) +
					     1.7 * spread;
			const double c = std::cos(angle);
			const double s = std::sin(angle);
			dx = c * bx - s * by + 0.35 * tx;
			dy = s * bx + c * by + 0.35 * ty;
			break;
		}
		case 3: { // Two support directions cross like a woven braid.
			const double braid =
				std::sin(tau * (3.0 * u - 2.0 * w) + 1.4 * phase);
			dx = braid * bx + (1.0 - std::abs(braid)) * tx;
			dy = braid * by + (1.0 - std::abs(braid)) * ty;
			dx += 0.42 * rx * cancel;
			dy += 0.42 * ry * cancel;
			break;
		}
		case 4: { // Two fixed expectation/maximization refinements.
			const std::array<double, 3> cx = {bx, tx, -0.72 * bx - 0.28 * tx};
			const std::array<double, 3> cy = {by, ty, -0.72 * by - 0.28 * ty};
			std::array<double, 3> prior = {
				0.3 + h.spread, 0.3 + v.spread,
				0.3 + cancel + 0.2 * std::sin(phase + tau * u)};
			double mx = bx, my = by;
			for (int iteration = 0; iteration < 2; ++iteration) {
				std::array<double, 3> responsibility{};
				double normalizer = 0.0;
				for (size_t k = 0; k < 3; ++k) {
					const double ex = cx[k] - mx;
					const double ey = cy[k] - my;
					responsibility[k] = std::max(0.02, prior[k]) *
						std::exp(-(ex * ex + ey * ey) / 0.42);
					normalizer += responsibility[k];
				}
				mx = my = 0.0;
				for (size_t k = 0; k < 3; ++k) {
					const double r = responsibility[k] /
							 std::max(1e-12, normalizer);
					mx += r * cx[k];
					my += r * cy[k];
				}
			}
			dx = mx;
			dy = my;
			break;
		}
		case 5: { // Bounded generators form a moving zonotope around support.
			const double a = std::sin(phase + tau * (u + 0.37 * w));
			const double b = std::cos(0.73 * phase + tau * (w - 0.21 * u));
			dx = bx + a * (0.45 + cancel) * tx + b * spread * rx;
			dy = by + a * (0.45 + cancel) * ty + b * spread * ry;
			break;
		}
		case 6: { // Enclose a closest pair of categorical displacement modes.
			const std::array<double, 4> cx = {bx, tx, bx + spread * rx,
							  tx - cancel * rx};
			const std::array<double, 4> cy = {by, ty, by + spread * ry,
							  ty - cancel * ry};
			size_t a = 0, b = 1;
			double best = std::numeric_limits<double>::infinity();
			for (size_t i = 0; i < 4; ++i)
				for (size_t j = i + 1; j < 4; ++j) {
					const double ex = cx[i] - cx[j];
					const double ey = cy[i] - cy[j];
					const double cost = ex * ex + ey * ey;
					if (cost < best) {
						best = cost;
						a = i;
						b = j;
					}
				}
			const double coefficient =
				std::sin(phase + tau * (1.7 * u - 1.3 * w));
			dx = 0.5 * (cx[a] + cx[b]) +
			     0.5 * coefficient * (cx[a] - cx[b]);
			dy = 0.5 * (cy[a] + cy[b]) +
			     0.5 * coefficient * (cy[a] - cy[b]);
			break;
		}
		case 7:
		case 8:
		case 9:
		case 10: {
			std::array<float, 5> qx{};
			std::array<float, 5> qy{};
			if (mode == 7) {
				qx = qy = {0.03f, 0.15f, 0.64f, 0.15f, 0.03f};
			} else if (mode == 8) {
				qx = qy = {0.43f, 0.055f, 0.03f, 0.055f, 0.43f};
			} else if (mode == 9) {
				const double centre_x =
					1.55 * std::sin(phase + tau * (u + 0.31 * w));
				const double centre_y =
					1.55 * std::cos(0.83 * phase + tau * (w - 0.27 * u));
				double sx = 0.0, sy = 0.0;
				for (size_t k = 0; k < 5; ++k) {
					const double px = static_cast<double>(k) - 2.0 - centre_x;
					const double py = static_cast<double>(k) - 2.0 - centre_y;
					qx[k] = static_cast<float>(std::exp(-1.6 * px * px));
					qy[k] = static_cast<float>(std::exp(-1.6 * py * py));
					sx += qx[k];
					sy += qy[k];
				}
				for (size_t k = 0; k < 5; ++k) {
					qx[k] = static_cast<float>(qx[k] / sx);
					qy[k] = static_cast<float>(qy[k] / sy);
				}
			} else {
				for (size_t k = 0; k < 5; ++k) {
					qx[k] = h.probability[4 - k];
					qy[k] = v.probability[4 - k];
				}
			}
			const double quantile_x = std::clamp(
				0.5 + 0.47 * std::sin(tau * (1.4 * u + 0.23 * w) + phase),
				0.01, 0.99);
			const double quantile_y = std::clamp(
				0.5 + 0.47 * std::cos(tau * (1.2 * w - 0.19 * u) + 0.9 * phase),
				0.01, 0.99);
			dx = quantile_transport(h, qx, quantile_x) + 0.22 * bx;
			dy = quantile_transport(v, qy, quantile_y) + 0.22 * by;
			break;
		}
		case 11: { // A support lens that alternates expansion and contraction.
			const double lens = std::sin(phase + tau * (radial_norm * 1.8));
			dx = 0.55 * bx + rx * lens * (0.55 + spread + cancel);
			dy = 0.55 * by + ry * lens * (0.55 + spread + cancel);
			break;
		}
		case 12: { // Bounded tile-wise generators make little moving rooms.
			const double cell_x = std::floor(u * 8.0);
			const double cell_y = std::floor(w * 6.0);
			const double a = std::sin(phase + 1.7 * cell_x + 2.3 * cell_y);
			const double b = std::cos(0.71 * phase - 2.1 * cell_x + 1.3 * cell_y);
			dx = 0.35 * bx + a * (0.45 + h.spread) + b * tx;
			dy = 0.35 * by + b * (0.45 + v.spread) + a * ty;
			break;
		}
#if 0
		// Retired 0.3/0.4 metaphorical presets. They remain in this short-lived
		// migration block only so saved OBS scene numbers can be audited; no
		// registered mode can execute them.
		case 13: { // Signed current behaves as a Wigner-like counterflow.
			const double negativity = cancel;
			const double interference =
				std::sin(tau * (4.0 * u - 3.0 * w) + phase +
					  5.0 * (h.signed_mean - v.signed_mean));
			dx = (1.0 + 3.0 * negativity) * h.signed_mean -
			     2.2 * negativity * tx + 0.45 * interference * rx;
			dy = (1.0 + 3.0 * negativity) * v.signed_mean -
			     2.2 * negativity * ty + 0.45 * interference * ry;
			hue_turns = 0.23 * negativity * interference;
			chroma_log = 0.18 * negativity * std::cos(phase + tau * u);
			lightness_shift = -0.20 * negativity * interference;
			lightness_fold = 0.42 * negativity *
				std::sin(phase + tau * (u + w));
			break;
		}
		case 14: { // Three reversible Hadamard coin/shift steps on each fibre.
			dx = quantum_walk_mean(h, phase + tau * (u + 0.31 * w));
			dy = quantum_walk_mean(v, -0.83 * phase + tau * (w - 0.27 * u));
			dx += 0.32 * tx * cancel;
			dy += 0.32 * ty * cancel;
			hue_turns = 0.16 * (dx - dy);
			chroma_log = 0.12 * cancel * std::cos(phase + tau * w);
			lightness_shift = 0.22 * (h.mean - v.mean);
			lightness_fold = 0.44 * std::sin(phase + tau * (u - w));
			break;
		}
		case 15: { // Support tunnels through a moving periodic barrier bank.
			const double energy = 0.45 * spread + 0.55 * cancel;
			const double barrier_x =
				0.5 + 0.5 * std::cos(tau * (7.0 * u + 0.8 * w) + phase);
			const double barrier_y =
				0.5 + 0.5 * std::cos(tau * (6.0 * w - 0.7 * u) - phase);
			const double tunnel_x =
				std::exp(-5.0 * barrier_x / (0.08 + energy));
			const double tunnel_y =
				std::exp(-5.0 * barrier_y / (0.08 + energy));
			dx = bx + (0.18 + 2.3 * tunnel_x) *
					std::tanh(7.0 * std::sin(tau * 3.0 * w + phase));
			dy = by + (0.18 + 2.3 * tunnel_y) *
					std::tanh(7.0 * std::cos(tau * 3.0 * u - phase));
			hue_turns = 0.20 * (tunnel_x - tunnel_y);
			chroma_log = 0.18 * (tunnel_x + tunnel_y - 0.5);
			lightness_shift = 0.24 * (tunnel_y - tunnel_x);
			lightness_fold = 0.42 *
				std::sin(tau * (barrier_x - barrier_y) + phase);
			break;
		}
		case 16: { // Closed support loops acquire a geometric holonomy.
			const double solid_angle = tau * (u * w + 0.35 * spread) +
						 3.0 * (h.signed_mean * v.signed_mean);
			const double angle = phase + solid_angle + std::atan2(by, bx);
			const double c = std::cos(angle);
			const double s = std::sin(angle);
			dx = c * bx - s * by + (0.45 + cancel) * s * rx;
			dy = s * bx + c * by - (0.45 + cancel) * s * ry;
			hue_turns = 0.20 * std::sin(solid_angle);
			chroma_log = 0.13 * std::cos(angle);
			lightness_shift = 0.19 * std::sin(angle);
			lightness_fold = 0.38 * std::cos(solid_angle);
			break;
		}
		case 17: { // Moving flux knots generate nonlocal-looking phase vortices.
			double ax = 0.0, ay = 0.0;
			for (int knot = 0; knot < 3; ++knot) {
				const double knot_phase = phase * (0.35 + 0.17 * knot) +
						  tau * knot / 3.0;
				const double cx = 0.5 + 0.28 * std::cos(knot_phase);
				const double cy = 0.5 + 0.24 * std::sin(1.31 * knot_phase);
				const double ex = u - cx;
				const double ey = w - cy;
				const double radius2 = ex * ex + ey * ey + 0.004;
				const double radius = std::sqrt(radius2);
				const double fringe = std::sin(36.0 * radius -
							 phase + std::atan2(ey, ex));
				ax += -ey * fringe / radius2;
				ay += ex * fringe / radius2;
			}
			dx = 0.2 * bx + 0.055 * ax * (0.35 + spread + cancel);
			dy = 0.2 * by + 0.055 * ay * (0.35 + spread + cancel);
			hue_turns = 0.22 * std::sin(0.18 * (ax - ay) + phase);
			chroma_log = 0.12 * std::cos(0.15 * (ax + ay));
			lightness_shift = 0.20 * std::sin(0.11 * (ax + ay) - phase);
			lightness_fold = 0.44 * std::cos(0.13 * (ax - ay));
			break;
		}
		case 18: { // A support spinor coherently flops normal to tangent.
			const double detuning = h.spread - v.spread;
			const double coupling = 0.18 + 1.7 * cancel;
			const double omega = std::sqrt(detuning * detuning +
						       coupling * coupling);
			const double population = std::sin(
				0.5 * omega * (phase + tau * (u + 0.63 * w)));
			const double spin = 2.0 * population * population - 1.0;
			dx = spin * bx + std::sqrt(std::max(0.0, 1.0 - spin * spin)) * tx;
			dy = spin * by + std::sqrt(std::max(0.0, 1.0 - spin * spin)) * ty;
			dx += 0.55 * detuning * rx;
			dy += 0.55 * detuning * ry;
			hue_turns = 0.24 * spin;
			chroma_log = 0.16 * population;
			lightness_shift = 0.22 * spin;
			lightness_fold = 0.44 * population;
			break;
		}
		case 19: { // Quenched disorder localizes flow except at resonances.
			const uint32_t cell_x = x / 9;
			const uint32_t cell_y = y / 9;
			const double disorder = 2.0 * hash01(cell_x, cell_y, 0xa511e9b3U) - 1.0;
			const double direction = tau * hash01(cell_x, cell_y, 0x63d83595U);
			const double energy = h.signed_mean - v.signed_mean + 0.5 * spread;
			const double resonance = std::exp(-28.0 *
							(disorder - energy) * (disorder - energy));
			const double escape = 0.04 + 2.7 * resonance;
			dx = 0.12 * bx + escape * std::cos(direction + 0.4 * phase);
			dy = 0.12 * by + escape * std::sin(direction - 0.4 * phase);
			hue_turns = 0.22 * disorder;
			chroma_log = 0.18 * (resonance - 0.5);
			lightness_shift = 0.25 * disorder * (1.0 - resonance);
			lightness_fold = 0.44 * (2.0 * resonance - 1.0);
			break;
		}
		case 20: { // Diametric sites share a correlated Bell-like displacement.
			const double partner_x = partner_h.mean + 0.7 * partner_h.signed_mean +
						 1.8 * partner_chroma_x;
			const double partner_y = partner_v.mean + 0.7 * partner_v.signed_mean +
						 1.8 * partner_chroma_y;
			const double correlation =
				std::cos(phase + tau * (0.5 * u + 0.5 * w));
			dx = correlation * (partner_x - bx) -
			     (0.75 + 0.25 * cancel) * radial_x;
			dy = correlation * (partner_y - by) -
			     (0.75 + 0.25 * cancel) * radial_y;
			hue_turns = 0.22 * correlation *
				(partner_h.signed_mean - partner_v.signed_mean);
			chroma_log = 0.16 * correlation;
			lightness_shift = 0.24 * correlation *
				(partner_h.mean - partner_v.mean);
			lightness_fold = 0.46 * correlation;
			break;
		}
		case 21: { // Coherent support collapses into one sampled fibre channel.
			const uint32_t cell_x = x / 7;
			const uint32_t cell_y = y / 7;
			const double sweep = std::fmod(
				hash01(cell_x, cell_y, 0xc2b2ae35U) + 0.035 * phase, 1.0);
			const double sweep_y = std::fmod(
				hash01(cell_y, cell_x, 0x27d4eb2fU) + 0.029 * phase, 1.0);
			const double outcome_x = inverse_discrete_cdf(h.probability, sweep);
			const double outcome_y = inverse_discrete_cdf(v.probability, sweep_y);
			dx = 1.65 * (outcome_x - h.mean) + 0.35 * tx;
			dy = 1.65 * (outcome_y - v.mean) + 0.35 * ty;
			hue_turns = 0.18 * (outcome_x - outcome_y);
			chroma_log = 0.20 * (sweep - 0.5);
			lightness_shift = 0.22 * (outcome_x + outcome_y);
			lightness_fold = 0.46 * (sweep_y - sweep);
			break;
		}
		case 22: { // Unstable periodic orbits retain concentrated transport scars.
			const double orbit = std::sin(tau * (2.0 * u + 0.08 * phase)) -
					     0.72 * std::cos(tau * (3.0 * w - 0.06 * phase));
			const double scar = std::exp(-22.0 * orbit * orbit);
			const double gx = 2.0 * std::cos(tau * (2.0 * u + 0.08 * phase));
			const double gy = 2.16 * std::sin(tau * (3.0 * w - 0.06 * phase));
			const double norm = std::max(0.1, std::sqrt(gx * gx + gy * gy));
			dx = (1.0 - scar) * 0.12 * bx - scar * gy / norm * (1.0 + cancel);
			dy = (1.0 - scar) * 0.12 * by + scar * gx / norm * (1.0 + cancel);
			hue_turns = 0.20 * scar * std::sin(phase + tau * (u + w));
			chroma_log = 0.18 * (scar - 0.5);
			lightness_shift = 0.22 * scar * std::cos(phase + tau * u);
			lightness_fold = 0.46 * (2.0 * scar - 1.0);
			break;
		}
		case 23: { // Support packets repel from occupied moving lattice cells.
			const double grid_x = 13.0 * u + 0.13 * phase;
			const double grid_y = 9.0 * w - 0.11 * phase;
			const double local_x = grid_x - std::floor(grid_x) - 0.5;
			const double local_y = grid_y - std::floor(grid_y) - 0.5;
			const double distance =
				std::max(0.04, std::sqrt(local_x * local_x + local_y * local_y));
			const double exclusion = 0.45 + spread + 1.4 * cancel;
			dx = exclusion * local_x / distance + 0.18 * tx;
			dy = exclusion * local_y / distance + 0.18 * ty;
			hue_turns = 0.24 * std::atan2(local_y, local_x) / tau;
			chroma_log = 0.22 * (0.35 - distance);
			lightness_shift = 0.20 * (distance - 0.25);
			lightness_fold = 0.46 * (0.5 - distance);
			break;
		}
		case 24: { // Dirac-like rapid interference between two support components.
			const double tremor = phase * 11.0 + tau * (12.0 * u - 9.0 * w) +
					      5.0 * (h.signed_mean + v.signed_mean);
			const double c = std::cos(tremor);
			const double s = std::sin(tremor);
			dx = c * bx + s * tx + 0.65 * s * rx * (0.3 + cancel);
			dy = c * by + s * ty - 0.65 * c * ry * (0.3 + cancel);
			hue_turns = 0.18 * s;
			chroma_log = 0.14 * c;
			lightness_shift = 0.22 * s;
			lightness_fold = 0.46 * c;
			break;
		}
#endif
		default:
			break;
		}

		double mode_gain = 1.0;
		if (mode == 1 || mode == 4 || mode == 6 || mode == 10)
			mode_gain = 3.0;
		else if (mode == 2 || mode == 3)
			mode_gain = 2.5;
		else if (mode == 5)
			mode_gain = 1.45;

		const double amplitude = 30.0 * mode_gain * amount *
			std::clamp(deformation, 0.0, 3.0) * confidence;
		const double limit = mode >= 13 ? 0.48 : 0.14;
		const double max_x = limit * width;
		const double max_y = limit * height;
		TransportPixel result;
		result.x = static_cast<float>(
			std::clamp(amplitude * dx, -max_x, max_x) / width);
		result.y = static_cast<float>(
			std::clamp(amplitude * dy, -max_y, max_y) / height);
		const double colour_gain = amount * std::min(2.0,
			std::clamp(deformation, 0.0, 3.0));
		result.hue = static_cast<float>(
			std::clamp(colour_gain * hue_turns, -0.48, 0.48));
		result.chroma = static_cast<float>(
			std::clamp(colour_gain * chroma_log, -0.48, 0.48));
		result.lightness_shift = static_cast<float>(
			std::clamp(colour_gain * lightness_shift, -0.48, 0.48));
		result.lightness_fold = static_cast<float>(
			std::clamp(colour_gain * lightness_fold, -0.48, 0.48));
		return result;
	}
};

struct Filter {
	obs_source_t *source = nullptr;
	std::mutex settings_mutex;
	int mode = 0;
	double amount = 1.0;
	double deformation = 1.0;
	double speed = 0.65;
	int resolution = 720;
	int threads = 6;
	double hue_perversity = 1.0;
	double lightness_perversity = 1.0;
	double dip_reverse = 0.72;
	double seconds = 0.0;

	gs_effect_t *effect = nullptr;
	gs_texrender_t *analysis = nullptr;
	std::array<gs_stagesurf_t *, 2> stage{};
	std::array<bool, 2> written{};
	size_t stage_index = 0;
	gs_texture_t *transport_texture = nullptr;
	gs_texture_t *oklch_action_texture = nullptr;
	uint32_t target_width = 0;
	uint32_t target_height = 0;
	uint32_t analysis_width = 0;
	uint32_t analysis_height = 0;
	bool ready = false;

	ConvStarEngine engine;
	DipCombTransport2D dip_transport;
	DipCombField dip_field;
	std::vector<float> input_lightness;
	std::vector<float> input_chroma_x;
	std::vector<float> input_chroma_y;
	std::vector<float> output;
	std::vector<uint8_t> transport_bytes;
	std::vector<uint8_t> oklch_action_bytes;
	float transport_span = 0.28f;
	uint64_t frames = 0;
	double total_ms = 0.0;
};

const char *filter_name(void *) { return "DIP-CONV Gaussian Comb Transport"; }

void filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, kMode, 13);
	obs_data_set_default_double(settings, kAmount, 1.0);
	obs_data_set_default_double(settings, kDeformation, 1.0);
	obs_data_set_default_double(settings, kSpeed, 0.65);
	obs_data_set_default_int(settings, kResolution, 720);
	obs_data_set_default_int(settings, kThreads, 6);
	obs_data_set_default_double(settings, kHuePerversity, 1.0);
	obs_data_set_default_double(settings, kLightnessPerversity, 1.0);
	obs_data_set_default_double(settings, kDipReverse, 0.72);
}

void filter_update(void *data, obs_data_t *settings)
{
	auto *filter = static_cast<Filter *>(data);
	std::lock_guard<std::mutex> lock(filter->settings_mutex);
	// Keep the persisted mode key only so pre-0.5 OBS scenes upgrade in place.
	filter->mode = 13;
	filter->amount = obs_data_get_double(settings, kAmount);
	filter->deformation = obs_data_get_double(settings, kDeformation);
	filter->speed = obs_data_get_double(settings, kSpeed);
	filter->resolution =
		static_cast<int>(obs_data_get_int(settings, kResolution));
	filter->threads = static_cast<int>(obs_data_get_int(settings, kThreads));
	filter->hue_perversity = obs_data_get_double(settings, kHuePerversity);
	filter->lightness_perversity =
		obs_data_get_double(settings, kLightnessPerversity);
	filter->dip_reverse = obs_data_get_double(settings, kDipReverse);
}

obs_properties_t *filter_properties(void *)
{
	obs_properties_t *props = obs_properties_create();
	obs_properties_add_float_slider(
		props, kAmount, "Transport amount", 0.0, 1.0, 0.01);
	obs_properties_add_float_slider(
		props, kDeformation, "Frequency trans-shift", 0.0, 3.0, 0.05);
	obs_properties_add_float_slider(
		props, kHuePerversity, "OKLCH phase/chroma gain", 0.0, 2.0, 0.05);
	obs_properties_add_float_slider(
		props, kLightnessPerversity, "OKLCH residual-lightness gain", 0.0, 2.0, 0.05);
	obs_properties_add_float_slider(
		props, kDipReverse, "DIP low-frequency reversal", 0.0, 1.0, 0.01);
	obs_property_t *resolution = obs_properties_add_list(
		props, kResolution, "Requested analysis long side",
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(resolution, "360 (fast)", 360);
	obs_property_list_add_int(resolution, "480", 480);
	obs_property_list_add_int(resolution, "540", 540);
	obs_property_list_add_int(resolution, "720 (default)", 720);
	obs_property_list_add_int(resolution, "900", 900);
	obs_property_list_add_int(resolution, "1080 (high detail)", 1080);
	obs_property_list_add_int(resolution, "1440 (very high detail)", 1440);
	obs_property_list_add_int(resolution, "Native", 0);
	obs_properties_add_int_slider(props, kThreads, "CPU threads", 1, 8, 1);
	return props;
}

void destroy_graphics(Filter *filter)
{
	if (filter->transport_texture) {
		gs_texture_destroy(filter->transport_texture);
		filter->transport_texture = nullptr;
	}
	if (filter->oklch_action_texture) {
		gs_texture_destroy(filter->oklch_action_texture);
		filter->oklch_action_texture = nullptr;
	}
	for (auto *&stage : filter->stage) {
		if (stage)
			gs_stagesurface_destroy(stage);
		stage = nullptr;
	}
	if (filter->analysis) {
		gs_texrender_destroy(filter->analysis);
		filter->analysis = nullptr;
	}
	if (filter->effect) {
		gs_effect_destroy(filter->effect);
		filter->effect = nullptr;
	}
}

void *filter_create(obs_data_t *settings, obs_source_t *source)
{
	auto *filter = new Filter;
	filter->source = source;
	filter_update(filter, settings);
	char *errors = nullptr;
	obs_enter_graphics();
	filter->effect =
		gs_effect_create(kEffect, "convstar-support.effect", &errors);
	obs_leave_graphics();
	if (errors) {
		blog(LOG_ERROR, "[DIP-CONV] shader: %s", errors);
		bfree(errors);
	}
	if (!filter->effect) {
		delete filter;
		return nullptr;
	}
	return filter;
}

void filter_destroy(void *data)
{
	auto *filter = static_cast<Filter *>(data);
	blog(LOG_INFO,
	     "[DIP-CONV] destroyed after %llu frames (%.2f ms/frame)",
	     static_cast<unsigned long long>(filter->frames),
	     filter->frames ? filter->total_ms / filter->frames : 0.0);
	obs_enter_graphics();
	destroy_graphics(filter);
	obs_leave_graphics();
	delete filter;
}

void filter_tick(void *data, float seconds)
{
	auto *filter = static_cast<Filter *>(data);
	std::lock_guard<std::mutex> lock(filter->settings_mutex);
	filter->seconds += std::max(0.0f, seconds);
}

void choose_analysis_shape(uint32_t width, uint32_t height, int resolution,
			   uint32_t &analysis_width,
			   uint32_t &analysis_height)
{
	const double scale = resolution <= 0 ? 1.0 : std::min(
		1.0, static_cast<double>(resolution) /
			static_cast<double>(std::max(width, height)));
	auto dip_extent = [](uint32_t requested, uint32_t ceiling) {
		requested = std::min(requested, ceiling);
		uint32_t result = 1;
		while (result <= requested / 2)
			result <<= 1;
		return std::max(16u, result);
	};
	analysis_width = dip_extent(
		static_cast<uint32_t>(std::lround(width * scale)), 1024);
	analysis_height = dip_extent(
		static_cast<uint32_t>(std::lround(height * scale)), 512);
}

bool ensure_graphics(Filter *filter, uint32_t width, uint32_t height)
{
	int resolution;
	{
		std::lock_guard<std::mutex> lock(filter->settings_mutex);
		resolution = filter->resolution;
	}
	uint32_t analysis_width, analysis_height;
	choose_analysis_shape(width, height, resolution, analysis_width,
			      analysis_height);
	if (filter->analysis && filter->target_width == width &&
	    filter->target_height == height &&
	    filter->analysis_width == analysis_width &&
	    filter->analysis_height == analysis_height)
		return true;

	if (filter->transport_texture) {
		gs_texture_destroy(filter->transport_texture);
		filter->transport_texture = nullptr;
	}
	if (filter->oklch_action_texture) {
		gs_texture_destroy(filter->oklch_action_texture);
		filter->oklch_action_texture = nullptr;
	}
	for (auto *&stage : filter->stage) {
		if (stage)
			gs_stagesurface_destroy(stage);
		stage = nullptr;
	}
	if (filter->analysis)
		gs_texrender_destroy(filter->analysis);
	filter->analysis = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	filter->stage[0] =
		gs_stagesurface_create(analysis_width, analysis_height, GS_RGBA);
	filter->stage[1] =
		gs_stagesurface_create(analysis_width, analysis_height, GS_RGBA);
	filter->transport_texture = gs_texture_create(
		analysis_width, analysis_height, GS_RGBA, 1, nullptr, GS_DYNAMIC);
	filter->oklch_action_texture = gs_texture_create(
		analysis_width, analysis_height, GS_RGBA, 1, nullptr, GS_DYNAMIC);
	const size_t count = static_cast<size_t>(analysis_width) * analysis_height;
	filter->input_lightness.resize(count);
	filter->input_chroma_x.resize(count);
	filter->input_chroma_y.resize(count);
	filter->output.resize(count * 6);
	filter->transport_bytes.resize(count * 4);
	filter->oklch_action_bytes.resize(count * 4);
	filter->target_width = width;
	filter->target_height = height;
	filter->analysis_width = analysis_width;
	filter->analysis_height = analysis_height;
	filter->written = {false, false};
	filter->stage_index = 0;
	filter->ready = false;
	blog(LOG_INFO, "[DIP-CONV] support lattice %ux%u for %ux%u",
	     analysis_width, analysis_height, width, height);
	return filter->analysis && filter->stage[0] && filter->stage[1] &&
	       filter->transport_texture && filter->oklch_action_texture;
}

void stage_target(Filter *filter)
{
	obs_source_t *target = obs_filter_get_target(filter->source);
	if (!target)
		return;
	obs_source_t *parent = obs_filter_get_parent(filter->source);
	gs_texrender_reset(filter->analysis);
	gs_viewport_push();
	gs_projection_push();
	gs_matrix_push();
	if (gs_texrender_begin(filter->analysis, filter->analysis_width,
			       filter->analysis_height)) {
		vec4 clear;
		vec4_zero(&clear);
		gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
		gs_matrix_identity();
		gs_ortho(0.0f, static_cast<float>(filter->target_width), 0.0f,
			 static_cast<float>(filter->target_height), -100.0f, 100.0f);
		const uint32_t flags = parent ? obs_source_get_output_flags(parent) : 0;
		const bool custom = (flags & OBS_SOURCE_CUSTOM_DRAW) != 0;
		const bool async = (flags & OBS_SOURCE_ASYNC) != 0;
		if (target == parent && !custom && !async)
			obs_source_default_render(target);
		else
			obs_source_video_render(target);
		gs_texrender_end(filter->analysis);
		gs_stage_texture(filter->stage[filter->stage_index],
				 gs_texrender_get_texture(filter->analysis));
		filter->written[filter->stage_index] = true;
	}
	gs_matrix_pop();
	gs_projection_pop();
	gs_viewport_pop();
}

double srgb_to_linear(double value)
{
	value /= 255.0;
	return value <= 0.04045 ? value / 12.92 :
		std::pow((value + 0.055) / 1.055, 2.4);
}

void srgb_to_oklch(const uint8_t *pixel, double &lightness, double &chroma,
		   double &hue)
{
	const double r = srgb_to_linear(pixel[0]);
	const double g = srgb_to_linear(pixel[1]);
	const double b = srgb_to_linear(pixel[2]);
	const double l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
	const double m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
	const double s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
	const double lp = std::cbrt(std::max(0.0, l));
	const double mp = std::cbrt(std::max(0.0, m));
	const double sp = std::cbrt(std::max(0.0, s));
	lightness = 0.2104542553 * lp + 0.7936177850 * mp - 0.0040720468 * sp;
	const double a = 1.9779984951 * lp - 2.4285922050 * mp + 0.4505937099 * sp;
	const double lab_b = 0.0259040371 * lp + 0.7827717662 * mp - 0.8086757660 * sp;
	chroma = std::sqrt(a * a + lab_b * lab_b);
	hue = std::atan2(lab_b, a);
}

struct OklchPotential {
	float lightness = 0.0f;
	float chroma_x = 0.0f;
	float chroma_y = 0.0f;
};

OklchPotential oklch_potential(const uint8_t *pixel)
{
	constexpr size_t side = 33;
	static const std::vector<OklchPotential> table = []() {
		std::vector<OklchPotential> values(side * side * side);
		for (size_t r = 0; r < side; ++r)
			for (size_t g = 0; g < side; ++g)
				for (size_t b = 0; b < side; ++b) {
					const uint8_t sample[3] = {
						static_cast<uint8_t>(std::lround(255.0 * r / (side - 1))),
						static_cast<uint8_t>(std::lround(255.0 * g / (side - 1))),
						static_cast<uint8_t>(std::lround(255.0 * b / (side - 1)))};
					double lightness, chroma, hue;
					srgb_to_oklch(sample, lightness, chroma, hue);
					OklchPotential &entry =
						values[(r * side + g) * side + b];
					entry.lightness = static_cast<float>(255.0 * lightness);
					entry.chroma_x = static_cast<float>(chroma * std::cos(hue));
					entry.chroma_y = static_cast<float>(chroma * std::sin(hue));
				}
		return values;
	}();
	const double scale = static_cast<double>(side - 1) / 255.0;
	const double fr = pixel[0] * scale;
	const double fg = pixel[1] * scale;
	const double fb = pixel[2] * scale;
	const size_t r0 = static_cast<size_t>(fr);
	const size_t g0 = static_cast<size_t>(fg);
	const size_t b0 = static_cast<size_t>(fb);
	const size_t r1 = std::min(side - 1, r0 + 1);
	const size_t g1 = std::min(side - 1, g0 + 1);
	const size_t b1 = std::min(side - 1, b0 + 1);
	const float tr = static_cast<float>(fr - r0);
	const float tg = static_cast<float>(fg - g0);
	const float tb = static_cast<float>(fb - b0);
	const auto sample = [&](size_t r, size_t g, size_t b) -> const OklchPotential & {
		return table[(r * side + g) * side + b];
	};
	OklchPotential result;
	for (size_t corner = 0; corner < 8; ++corner) {
		const bool high_r = (corner & 4) != 0;
		const bool high_g = (corner & 2) != 0;
		const bool high_b = (corner & 1) != 0;
		const float weight = (high_r ? tr : 1.0f - tr) *
				     (high_g ? tg : 1.0f - tg) *
				     (high_b ? tb : 1.0f - tb);
		const OklchPotential &entry = sample(high_r ? r1 : r0,
						     high_g ? g1 : g0,
						     high_b ? b1 : b0);
		result.lightness += weight * entry.lightness;
		result.chroma_x += weight * entry.chroma_x;
		result.chroma_y += weight * entry.chroma_y;
	}
	return result;
}

void process_surface(Filter *filter, size_t index)
{
	if (!filter->written[index])
		return;
	uint8_t *mapped = nullptr;
	uint32_t stride = 0;
	if (!gs_stagesurface_map(filter->stage[index], &mapped, &stride))
		return;
	const auto started = std::chrono::steady_clock::now();
	const uint32_t width = filter->analysis_width;
	const uint32_t height = filter->analysis_height;
	for (uint32_t y = 0; y < height; ++y) {
		const uint8_t *row = mapped + static_cast<size_t>(y) * stride;
		for (uint32_t x = 0; x < width; ++x) {
			const uint8_t *pixel = row + 4 * x;
			const OklchPotential potential = oklch_potential(pixel);
			const size_t i = static_cast<size_t>(y) * width + x;
			filter->input_lightness[i] = potential.lightness;
			filter->input_chroma_x[i] = potential.chroma_x;
			filter->input_chroma_y[i] = potential.chroma_y;
		}
	}
	gs_stagesurface_unmap(filter->stage[index]);
	const auto converted = std::chrono::steady_clock::now();

	int mode, threads;
	double amount, deformation, phase, dip_reverse;
	{
		std::lock_guard<std::mutex> lock(filter->settings_mutex);
		mode = std::clamp(filter->mode, 0, 13);
		amount = filter->amount;
		deformation = std::clamp(filter->deformation, 0.0, 3.0);
		phase = filter->seconds * std::clamp(filter->speed, 0.0, 3.0);
		threads = std::clamp(filter->threads, 1, 8);
		dip_reverse = std::clamp(filter->dip_reverse, 0.0, 1.0);
	}
	const DipCombField *dip_field = nullptr;
	if (mode == 13 && filter->dip_transport.process(
			filter->input_lightness, width, height, deformation,
			dip_reverse, threads, filter->dip_field))
		dip_field = &filter->dip_field;
	if (dip_field && filter->frames == 0) {
		const DipCombTimings &timing = filter->dip_transport.last_timings();
		blog(LOG_INFO,
		     "[DIP-CONV] DIP %ux%u %s: resample %.2f, forward %.2f, comb %.2f, inverse2 %.2f, field %.2f, total %.2f ms",
		     filter->dip_transport.transform_width(),
		     filter->dip_transport.transform_height(),
		     filter->dip_transport.simd_backend(), timing.resample_ms,
		     timing.forward_ms, timing.comb_ms, timing.inverse_ms,
		     timing.expand_ms, timing.total_ms);
	}
	filter->engine.process(filter->input_lightness, filter->input_chroma_x,
			       filter->input_chroma_y, width, height, mode, amount,
			       phase, deformation, threads, dip_field, filter->output);
	const auto transported = std::chrono::steady_clock::now();
	filter->transport_span = mode >= 13 ? 0.96f : 0.28f;
	for (size_t i = 0; i < filter->input_lightness.size(); ++i) {
		const double encoded_x = 128.0 +
			filter->output[6 * i] * (255.0 / filter->transport_span);
		const double encoded_y = 128.0 +
			filter->output[6 * i + 1] * (255.0 / filter->transport_span);
		const double encoded_hue = 128.0 + filter->output[6 * i + 2] * 255.0;
		const double encoded_chroma = 128.0 + filter->output[6 * i + 3] * 255.0;
		const double encoded_lightness_shift =
			128.0 + filter->output[6 * i + 4] * 255.0;
		const double encoded_lightness_fold =
			128.0 + filter->output[6 * i + 5] * 255.0;
		filter->transport_bytes[4 * i] = static_cast<uint8_t>(
			std::clamp(std::lround(encoded_x), 0L, 255L));
		filter->transport_bytes[4 * i + 1] = static_cast<uint8_t>(
			std::clamp(std::lround(encoded_y), 0L, 255L));
		filter->transport_bytes[4 * i + 2] = 128;
		filter->transport_bytes[4 * i + 3] = 128;
		filter->oklch_action_bytes[4 * i] = static_cast<uint8_t>(
			std::clamp(std::lround(encoded_hue), 0L, 255L));
		filter->oklch_action_bytes[4 * i + 1] = static_cast<uint8_t>(
			std::clamp(std::lround(encoded_chroma), 0L, 255L));
		filter->oklch_action_bytes[4 * i + 2] = static_cast<uint8_t>(
			std::clamp(std::lround(encoded_lightness_shift), 0L, 255L));
		filter->oklch_action_bytes[4 * i + 3] = static_cast<uint8_t>(
			std::clamp(std::lround(encoded_lightness_fold), 0L, 255L));
	}
	gs_texture_set_image(
		filter->transport_texture,
		filter->transport_bytes.data(), width * 4, false);
	gs_texture_set_image(
		filter->oklch_action_texture,
		filter->oklch_action_bytes.data(), width * 4, false);
	const auto packed = std::chrono::steady_clock::now();
	filter->ready = true;
	const double elapsed = std::chrono::duration<double, std::milli>(
				       std::chrono::steady_clock::now() - started)
				       .count();
	++filter->frames;
	filter->total_ms += elapsed;
	if (filter->frames == 1)
		blog(LOG_INFO,
		     "[DIP-CONV] first-frame stages OKLCH %.2f ms, operator %.2f ms, pack %.2f ms",
		     std::chrono::duration<double, std::milli>(converted - started).count(),
		     std::chrono::duration<double, std::milli>(transported - converted).count(),
		     std::chrono::duration<double, std::milli>(packed - transported).count());
	if (filter->frames % 300 == 0)
		blog(LOG_INFO, "[DIP-CONV] %.2f ms/frame",
		     filter->total_ms / filter->frames);
}

void filter_render(void *data, gs_effect_t *)
{
	auto *filter = static_cast<Filter *>(data);
	obs_source_t *target = obs_filter_get_target(filter->source);
	if (!target) {
		obs_source_skip_video_filter(filter->source);
		return;
	}
	const uint32_t width = obs_source_get_base_width(target);
	const uint32_t height = obs_source_get_base_height(target);
	if (!width || !height || !ensure_graphics(filter, width, height)) {
		obs_source_skip_video_filter(filter->source);
		return;
	}
	const size_t read = (filter->stage_index + 1) % 2;
	process_surface(filter, read);
	stage_target(filter);
	filter->stage_index = read;
	if (!filter->ready) {
		obs_source_skip_video_filter(filter->source);
		return;
	}
	if (!obs_source_process_filter_begin(filter->source, GS_RGBA,
					     OBS_NO_DIRECT_RENDERING))
		return;
	gs_effect_set_texture(
		gs_effect_get_param_by_name(filter->effect, "transport_image"),
		filter->transport_texture);
	gs_effect_set_texture(
		gs_effect_get_param_by_name(filter->effect, "oklch_action_image"),
		filter->oklch_action_texture);
	vec2 support_texel;
	vec2_set(&support_texel, 1.0f / filter->analysis_width,
		 1.0f / filter->analysis_height);
	gs_effect_set_vec2(
		gs_effect_get_param_by_name(filter->effect, "support_texel"),
		&support_texel);
	int mode;
	float hue_perversity;
	float lightness_perversity;
	{
		std::lock_guard<std::mutex> lock(filter->settings_mutex);
		mode = std::clamp(filter->mode, 0, 13);
		hue_perversity = static_cast<float>(
			std::clamp(filter->hue_perversity, 0.0, 2.0));
		lightness_perversity = static_cast<float>(
			std::clamp(filter->lightness_perversity, 0.0, 2.0));
	}
	float blur = mode == 12 ? 0.0f :
		(mode >= 7 && mode <= 10 ? 3.0f : 2.0f);
	if (mode == 13)
		blur = 1.0f;
	gs_effect_set_float(
		gs_effect_get_param_by_name(filter->effect, "support_blur"), blur);
	gs_effect_set_float(
		gs_effect_get_param_by_name(filter->effect, "transport_span"),
		filter->transport_span);
	const float colour = mode == 13 ? 1.0f : 0.0f;
	gs_effect_set_float(
		gs_effect_get_param_by_name(filter->effect, "oklch_colour"), colour);
	gs_effect_set_float(
		gs_effect_get_param_by_name(filter->effect, "hue_perversity"),
		hue_perversity);
	gs_effect_set_float(
		gs_effect_get_param_by_name(filter->effect, "lightness_perversity"),
		lightness_perversity);
	obs_source_process_filter_end(filter->source, filter->effect, width, height);
}

obs_source_info filter_info = {
	.id = "convstar_support_toys",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = filter_name,
	.create = filter_create,
	.destroy = filter_destroy,
	.get_width = nullptr,
	.get_height = nullptr,
	.get_defaults = filter_defaults,
	.get_properties = filter_properties,
	.update = filter_update,
	.video_tick = filter_tick,
	.video_render = filter_render,
};

} // namespace

bool obs_module_load(void)
{
	if (!ConvStarEngine::self_test()) {
		blog(LOG_ERROR,
		     "[DIP-CONV] legacy fused-bank/KKT conformance self-test failed");
		return false;
	}
	const uint8_t neutral_sample[3] = {128, 128, 128};
	(void)oklch_potential(neutral_sample);
	obs_register_source(&filter_info);
	blog(LOG_INFO, "[DIP-CONV] loaded");
	return true;
}
