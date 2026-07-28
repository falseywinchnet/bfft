#include <high_vision/high_vision.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

namespace {

void require(bool condition, const char *message)
{
	if (!condition) {
		std::cerr << "high_vision_tests: " << message << '\n';
		std::exit(1);
	}
}

std::vector<float> textured(std::size_t width, std::size_t height)
{
	std::vector<float> image(width * height);
	std::uint32_t state = 0x12345678u;
	for (float &value : image) {
		state = state * 1664525u + 1013904223u;
		value = 0.08f + 0.75f * static_cast<float>(state & 0xffffu) /
					   65535.0f;
	}
	return image;
}

std::vector<float> translate(const std::vector<float> &source,
			     std::size_t width, std::size_t height,
			     int dx, int dy)
{
	std::vector<float> result(width * height, 0.2f);
	for (std::size_t y = 0; y < height; ++y) {
		for (std::size_t x = 0; x < width; ++x) {
			const int sx = static_cast<int>(x) - dx;
			const int sy = static_cast<int>(y) - dy;
			if (sx >= 0 && sy >= 0 && sx < static_cast<int>(width) &&
			    sy < static_cast<int>(height))
				result[y * width + x] =
					source[static_cast<std::size_t>(sy) * width +
					       static_cast<std::size_t>(sx)];
		}
	}
	return result;
}

class AdditiveStage final : public high_vision::ExperimentalStage {
public:
	const char *name() const noexcept override { return "test-additive"; }
	void reset(std::size_t, std::size_t) override { ++resets; }
	void process(const float *, const float *, float *belief,
		     std::size_t width, std::size_t height,
		     const high_vision::FrameMetadata &,
		     const high_vision::Diagnostics &) override
	{
		++calls;
		for (std::size_t i = 0; i < width * height; ++i)
			belief[i] += 0.01f;
	}
	int calls = 0;
	int resets = 0;
};

} // namespace

int main()
{
	constexpr std::size_t width = 80;
	constexpr std::size_t height = 56;
	std::vector<float> output(width * height);

	{
		high_vision::Config config;
		config.mode = high_vision::Mode::passthrough;
		high_vision::Processor processor(config);
		auto input = textured(width, height);
		require(processor.process(input.data(), width, output.data(), width,
					  width, height),
			"passthrough frame was rejected");
		require(input == output, "passthrough changed the frame");
	}

	{
		high_vision::Config config;
		config.registration_radius = 6;
		config.local_search_radius = 0;
		config.tile_size = 24;
		config.scene_cut_threshold = 1.0f;
		config.tone_strength = 0.0f;
		high_vision::Processor processor(config);
		auto first = textured(width, height);
		auto second = translate(first, width, height, 3, -2);
		require(processor.process(first.data(), width, output.data(), width,
					  width, height),
			"first translated frame was rejected");
		require(processor.process(second.data(), width, output.data(), width,
					  width, height),
			"second translated frame was rejected");
		const auto &diagnostics = processor.diagnostics();
		require(std::abs(diagnostics.global_dx - 3.0f) < 0.1f,
			"global horizontal registration is wrong");
		require(std::abs(diagnostics.global_dy + 2.0f) < 0.1f,
			"global vertical registration is wrong");
	}

	{
		high_vision::Config config;
		config.registration_radius = 0;
		config.local_search_radius = 0;
		config.scene_cut_threshold = 1.0f;
		config.tone_strength = 0.0f;
		high_vision::Processor processor(config);
		auto first = textured(width, height);
		for (float &value : first)
			value = std::min(value, 0.5f);
		auto second = first;
		for (float &value : second)
			value = std::min(value * 2.0f, 1.0f);
		high_vision::FrameMetadata a;
		a.exposure_seconds = 0.01;
		a.analog_gain = 1.0;
		high_vision::FrameMetadata b = a;
		b.exposure_seconds = 0.02;
		require(processor.process(first.data(), width, output.data(), width,
					  width, height, a),
			"first exposure frame was rejected");
		const std::size_t bright =
			static_cast<std::size_t>(std::find(first.begin(), first.end(),
						      0.5f) -
						 first.begin());
		const float before = processor.belief()[bright];
		require(processor.process(second.data(), width, output.data(), width,
					  width, height, b),
			"second exposure frame was rejected");
		require(std::abs(processor.diagnostics().relative_exposure - 2.0f) <
				1e-4f,
			"camera telemetry did not anchor exposure");
		require(std::abs(processor.belief()[bright] - before) < 0.02f,
			"a clipped observation destroyed highlight belief");
	}

	{
		high_vision::Config config;
		config.registration_radius = 0;
		config.local_search_radius = 0;
		config.tone_strength = 0.0f;
		high_vision::Processor processor(config);
		std::vector<float> input(width * height, 0.5f);
		high_vision::FrameMetadata metadata;
		metadata.black_level = 0.1;
		metadata.white_level = 0.9;
		require(processor.process(input.data(), width, output.data(), width,
					  width, height, metadata),
			"black/white corrected frame was rejected");
		require(std::abs(processor.belief()[0] - 0.5f) < 1e-5f,
			"sensor black/white levels were not applied");
	}

	{
		high_vision::Config config;
		config.mode = high_vision::Mode::experimental;
		config.registration_radius = 0;
		config.local_search_radius = 0;
		config.scene_cut_threshold = 1.0f;
		config.tone_strength = 0.0f;
		high_vision::Processor processor(config);
		auto stage = std::make_unique<AdditiveStage>();
		AdditiveStage *stage_ptr = stage.get();
		processor.set_experimental_stage(std::move(stage));
		auto input = textured(width, height);
		require(processor.process(input.data(), width, output.data(), width,
					  width, height),
			"first experimental frame was rejected");
		const float before = processor.belief()[0];
		require(processor.process(input.data(), width, output.data(), width,
					  width, height),
			"second experimental frame was rejected");
		require(stage_ptr->calls == 2,
			"experimental stage was not invoked for every frame");
		require(processor.belief()[0] > before,
			"experimental stage could not modify belief");
	}

	{
		high_vision::Config hdr_config;
		hdr_config.mode = high_vision::Mode::synthetic_hdr;
		hdr_config.registration_radius = 4;
		hdr_config.local_search_radius = 2;
		hdr_config.scene_cut_threshold = 1.0f;
		hdr_config.support_limit = 120.0f;
		hdr_config.support_decay = 0.997f;
		hdr_config.tone_strength = 0.0f;
		hdr_config.local_contrast = 0.0f;
		high_vision::Config night_config = hdr_config;
		night_config.mode = high_vision::Mode::night_integrator;
		high_vision::Processor hdr(hdr_config);
		high_vision::Processor night(night_config);
		auto texture = textured(width, height);
		std::vector<float> dark(width * height);
		std::uint32_t noise = 0x42a1d39bu;
		for (int frame = 0; frame < 48; ++frame) {
			for (std::size_t i = 0; i < dark.size(); ++i) {
				noise = noise * 1664525u + 1013904223u;
				const float perturbation =
					0.012f *
					(static_cast<float>(noise & 0xffffu) /
						 65535.0f -
					 0.5f);
				dark[i] = std::clamp(
					0.012f + 0.004f * texture[i] +
						perturbation,
					0.0f, 1.0f);
			}
			require(hdr.process(dark.data(), width, output.data(), width,
					    width, height),
				"HDR rejected a noisy dark frame");
			require(night.process(
					dark.data(), width, output.data(), width,
					width, height),
				"Night rejected a noisy dark frame");
		}
		require(night.diagnostics().registration_confidence < 0.8f,
			"dark-frame control did not produce weak registration");
		require(night.diagnostics().mean_support > 20.0f,
			"Night failed to retain low-confidence evidence");
		require(night.diagnostics().mean_support >
				4.0f * hdr.diagnostics().mean_support,
			"Night still converts registration uncertainty into "
			"support loss");
	}

	{
		// A dark object entering an accumulated shadow is the adversarial
		// case that exposed the original absolute-residual gate.  The object
		// occupies the same camera coordinates for several frames, so its
		// signed innovation is coherent while read/shot noise is not.
		high_vision::Config config;
		config.mode = high_vision::Mode::night_integrator;
		config.registration_radius = 0;
		config.local_search_radius = 0;
		config.scene_cut_threshold = 1.0f;
		config.support_limit = 120.0f;
		config.support_decay = 0.997f;
		config.change_threshold = 0.15f;
		config.tone_strength = 0.0f;
		config.local_contrast = 0.0f;
		high_vision::Processor processor(config);
		std::vector<float> frame(width * height);
		std::uint32_t noise = 0x91e10da5u;
		auto render = [&](bool occluded) {
			for (std::size_t y = 0; y < height; ++y) {
				for (std::size_t x = 0; x < width; ++x) {
					noise = noise * 1664525u + 1013904223u;
					const float perturbation =
						0.008f *
						(static_cast<float>(noise & 0xffffu) /
							 65535.0f -
						 0.5f);
					const bool inside =
						x >= 28 && x < 52 && y >= 18 && y < 38;
					const float signal =
						occluded && inside ? 0.008f : 0.055f;
					frame[y * width + x] =
						std::clamp(signal + perturbation,
							   0.0f, 1.0f);
				}
			}
		};
		for (int index = 0; index < 120; ++index) {
			render(false);
			require(processor.process(
					frame.data(), width, output.data(), width,
					width, height),
				"Night rejected a shadow accumulation frame");
		}
		require(processor.diagnostics().mean_support > 80.0f,
			"shadow control did not build high support");
		require(std::abs(processor.diagnostics().relative_exposure - 1.0f) <
				0.02f,
			"fallback exposure gauge drifted on a stable shadow");
		for (int index = 0; index < 4; ++index) {
			render(true);
			require(processor.process(
					frame.data(), width, output.data(), width,
					width, height),
				"Night rejected a dark-object frame");
		}
		double object_mean = 0.0;
		std::size_t object_pixels = 0;
		for (std::size_t y = 18; y < 38; ++y)
			for (std::size_t x = 28; x < 52; ++x) {
				object_mean += processor.belief()[y * width + x];
				++object_pixels;
			}
		object_mean /= static_cast<double>(object_pixels);
		require(object_mean < 0.018,
			"high support integrated a persistent dark object away");
	}

	{
		// Night must honor the control surface. The original implementation
		// silently raised every persistence setting to 0.997, making a UI
		// value such as 0.943 look ineffective and allowing support to reach
		// the recursively warped regime anyway.
		high_vision::Config config;
		config.mode = high_vision::Mode::night_integrator;
		config.registration_radius = 0;
		config.local_search_radius = 0;
		config.scene_cut_threshold = 1.0f;
		config.support_limit = 120.0f;
		config.support_decay = 0.90f;
		config.tone_strength = 0.0f;
		high_vision::Processor processor(config);
		std::vector<float> frame(width * height, 0.04f);
		for (int index = 0; index < 100; ++index)
			require(processor.process(
					frame.data(), width, output.data(), width,
					width, height),
				"Night rejected a persistence-control frame");
		require(processor.diagnostics().mean_support < 11.0f,
			"Night ignored the configured evidence persistence");
	}

	{
		constexpr std::size_t frames = 8;
		constexpr std::size_t shifts = 17;
		std::vector<float> scores(frames * shifts);
		for (std::size_t frame = 0; frame < frames; ++frame)
			for (std::size_t shift = 0; shift < shifts; ++shift)
				scores[frame * shifts + shift] =
					-0.5f * std::abs(
						static_cast<float>(shift) -
						static_cast<float>((frame * 3) %
								   shifts));

		high_vision::EntropySupportController controller;
		std::vector<float> posterior;
		high_vision::EntropyProjectionDiagnostics projection;
		require(controller.project(scores.data(), frames, shifts, 6.0f,
					   posterior, &projection),
			"entropy projection rejected valid scores");
		require(std::abs(projection.effective_shifts - 6.0f) < 1e-3f,
			"entropy projection missed its support budget");
		for (std::size_t frame = 0; frame < frames; ++frame) {
			float sum = 0.0f;
			for (std::size_t shift = 0; shift < shifts; ++shift)
				sum += posterior[frame * shifts + shift];
			require(std::abs(sum - 1.0f) < 1e-5f,
				"entropy posterior row is not normalized");
		}

		require(controller.project(scores.data(), frames, shifts,
					   static_cast<float>(shifts),
					   posterior, &projection),
			"uniform entropy projection failed");
		require(projection.inverse_temperature == 0.0f,
			"full support did not produce a uniform posterior");
		for (float probability : posterior)
			require(std::abs(
					probability -
					1.0f / static_cast<float>(shifts)) <
					1e-6f,
				"full support posterior is not uniform");
	}

	{
		constexpr std::size_t frames = 6;
		constexpr std::size_t shifts = 8;
		high_vision::EntropyControllerConfig config;
		config.budgets = {2.0f, 4.0f, 8.0f};
		config.evidence_retention = 1.0f;
		high_vision::EntropySupportController controller(config);
		std::vector<float> first(frames * shifts);
		std::vector<float> second(frames * shifts);
		for (std::size_t frame = 0; frame < frames; ++frame) {
			const std::size_t peak = frame % shifts;
			for (std::size_t shift = 0; shift < shifts; ++shift) {
				const float score =
					-3.0f * std::abs(
						static_cast<float>(shift) -
						static_cast<float>(peak));
				first[frame * shifts + shift] = score;
				second[frame * shifts + shift] = score;
			}
		}
		require(controller.update(first.data(), second.data(), frames,
					  shifts, 4096),
			"entropy controller rejected complementary scores");
		require(controller.diagnostics().selected_budget == 4.0f,
			"support did not contract by exactly one adjacent tier");
		std::vector<float> posterior;
		require(controller.project_selected(
				first.data(), frames, shifts, posterior),
			"selected entropy projection failed");
		require(std::abs(
				controller.diagnostics().projection.effective_shifts -
				controller.diagnostics().selected_budget) <
				1e-3f,
			"selected posterior and controller budget disagree");
		controller.reset();
		require(controller.diagnostics().batches == 0,
			"entropy reset retained the batch count");
		require(controller.diagnostics().selected_budget == 8.0f,
			"entropy reset did not restore broad support");
		require(std::all_of(
				controller.cumulative_evidence().begin(),
				controller.cumulative_evidence().end(),
				[](double value) { return value == 0.0; }),
			"entropy reset retained cumulative evidence");
	}

	std::cout << "high_vision_tests: all checks passed\n";
	return 0;
}
