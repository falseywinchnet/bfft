#include <high_vision/high_vision.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
	const std::size_t width = argc > 1
					  ? static_cast<std::size_t>(std::strtoul(
						    argv[1], nullptr, 10))
					  : 512;
	const std::size_t height = argc > 2
					   ? static_cast<std::size_t>(std::strtoul(
						     argv[2], nullptr, 10))
					   : 256;
	const int frames = argc > 3 ? std::atoi(argv[3]) : 60;
	const bool night = argc > 4 && std::string(argv[4]) == "night";
	if (width < 8 || height < 8 || frames < 2)
		return 2;

	std::vector<float> source(width * height);
	std::vector<float> input(width * height);
	std::vector<float> output(width * height);
	std::uint32_t random = 0x724a91f3u;
	for (float &value : source) {
		random = random * 1664525u + 1013904223u;
		value = 0.02f + 0.9f * static_cast<float>(random & 0xffffu) /
					 65535.0f;
	}

	high_vision::Config config;
	config.mode = night ? high_vision::Mode::night_integrator
			    : high_vision::Mode::synthetic_hdr;
	high_vision::Processor processor(config);
	const auto started = std::chrono::steady_clock::now();
	for (int frame = 0; frame < frames; ++frame) {
		const int dx = frame % 5 - 2;
		const int dy = frame % 3 - 1;
		for (std::size_t y = 0; y < height; ++y) {
			for (std::size_t x = 0; x < width; ++x) {
				const int sx = static_cast<int>(x) - dx;
				const int sy = static_cast<int>(y) - dy;
				input[y * width + x] =
					sx >= 0 && sy >= 0 &&
							sx < static_cast<int>(width) &&
							sy < static_cast<int>(height)
						? source[static_cast<std::size_t>(sy) *
								 width +
							 static_cast<std::size_t>(sx)]
						: 0.02f;
			}
		}
		high_vision::FrameMetadata metadata;
		metadata.sequence = static_cast<std::uint64_t>(frame);
		metadata.timestamp_ns =
			static_cast<std::uint64_t>(frame) * 33'333'333u;
		if (!processor.process(input.data(), width, output.data(), width,
				       width, height, metadata))
			return 3;
	}
	const double milliseconds =
		std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - started)
			.count();
	const double per_frame = milliseconds / frames;
	const auto &diagnostics = processor.diagnostics();
	std::cout << width << 'x' << height << ", " << frames << ' '
		  << (night ? "night" : "hdr") << " frames: "
		  << std::fixed << std::setprecision(2) << per_frame
		  << " ms/frame (" << 1000.0 / per_frame << " fps), support "
		  << diagnostics.mean_support << ", meyer registration "
		  << (diagnostics.meyer_registration_applied ? "on" : "off")
		  << '\n';
	return 0;
}
