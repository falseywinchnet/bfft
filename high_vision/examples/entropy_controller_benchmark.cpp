#include <high_vision/high_vision.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

float random_signed(std::uint32_t &state)
{
	state = state * 1664525u + 1013904223u;
	return 2.0f * static_cast<float>(state & 0xffffu) / 65535.0f -
	       1.0f;
}

} // namespace

int main()
{
	constexpr std::size_t frames = 32;
	constexpr std::size_t shifts = 197;
	constexpr std::size_t pixels = 128 * 128;
	constexpr int batches = 31;

	high_vision::EntropySupportController controller;
	std::vector<float> first(frames * shifts);
	std::vector<float> second(frames * shifts);
	std::vector<float> full(frames * shifts);
	std::vector<float> posterior;
	std::uint32_t random = 0x91c742adu;

	const auto started = std::chrono::steady_clock::now();
	float previous_budget = controller.diagnostics().selected_budget;
	std::cout << "support: " << previous_budget;
	for (int batch = 0; batch < batches; ++batch) {
		// Complementary witnesses gradually become coherent around a latent
		// shift as photons accumulate. Noise stays independent between them.
		const float strength =
			0.0007f * static_cast<float>(std::max(batch - 4, 0));
		for (std::size_t frame = 0; frame < frames; ++frame) {
			const int center = static_cast<int>(
				(frame * 37 + static_cast<std::size_t>(batch) * 11) %
				shifts);
			for (std::size_t shift = 0; shift < shifts; ++shift) {
				const int raw = std::abs(
					static_cast<int>(shift) - center);
				const int distance = std::min(
					raw, static_cast<int>(shifts) - raw);
				const float signal =
					-strength * static_cast<float>(distance);
				const std::size_t index = frame * shifts + shift;
				first[index] = signal + 1.4f * random_signed(random);
				second[index] = signal + 1.4f * random_signed(random);
				full[index] = first[index] + second[index];
			}
		}
		if (!controller.update(
			    first.data(), second.data(), frames, shifts, pixels) ||
		    !controller.project_selected(
			    full.data(), frames, shifts, posterior))
			return 2;
		const float budget = controller.diagnostics().selected_budget;
		if (budget != previous_budget) {
			std::cout << " -> " << budget;
			previous_budget = budget;
		}
	}
	const double milliseconds =
		std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - started)
			.count();
	std::cout << "\n" << batches << " batches of " << frames << "x"
		  << shifts << " scores: " << std::fixed << std::setprecision(3)
		  << milliseconds / batches << " ms/batch, "
		  << controller.diagnostics().support_transitions
		  << " transitions, final effective support "
		  << controller.diagnostics().projection.effective_shifts << '\n';
	return 0;
}
