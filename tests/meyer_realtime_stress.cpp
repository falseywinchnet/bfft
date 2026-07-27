#include <bfft/meyer.h>

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

namespace {

std::atomic<size_t> allocation_count{0};
std::atomic<size_t> allocation_bytes{0};

void *tracked_allocate(std::size_t size)
{
	if (void *memory = std::malloc(size)) {
		allocation_count.fetch_add(1, std::memory_order_relaxed);
		allocation_bytes.fetch_add(size, std::memory_order_relaxed);
		return memory;
	}
	throw std::bad_alloc();
}

} // namespace

void *operator new(std::size_t size)
{
	return tracked_allocate(size);
}

void *operator new[](std::size_t size)
{
	return tracked_allocate(size);
}

void operator delete(void *memory) noexcept
{
	std::free(memory);
}

void operator delete[](void *memory) noexcept
{
	std::free(memory);
}

void operator delete(void *memory, std::size_t) noexcept
{
	std::free(memory);
}

void operator delete[](void *memory, std::size_t) noexcept
{
	std::free(memory);
}

int main()
{
	constexpr size_t height = 256;
	constexpr size_t width = 512;
	constexpr size_t count = height * width;
	bfft_meyer_plan *plan = nullptr;
	if (bfft_meyer_plan_create(height, width, 0.05, 40.0, 8, 32,
				   1e-4, 4, &plan) != BFFT_OK)
		return 1;

	std::vector<double> input(count);
	std::vector<double> cartoon(count);
	std::vector<double> texture(count);
	for (size_t i = 0; i < count; ++i)
		input[i] = 127.0 + 60.0 * std::sin(0.013 * double(i));

	// Warm every persistent worker and transform path before measuring.
	if (bfft_meyer_split(plan, input.data(), cartoon.data(),
			     texture.data()) != BFFT_OK)
		return 2;
	allocation_count.store(0, std::memory_order_relaxed);
	allocation_bytes.store(0, std::memory_order_relaxed);

	for (int frame = 0; frame < 12; ++frame) {
		const int passes = 4 + (frame * 7) % 21;
		if (bfft_meyer_plan_set_passes(plan, passes) != BFFT_OK)
			return 3;
		if (bfft_meyer_split(plan, input.data(), cartoon.data(),
				     texture.data()) != BFFT_OK)
			return 4;
	}

	const size_t allocations =
		allocation_count.load(std::memory_order_relaxed);
	const size_t bytes = allocation_bytes.load(std::memory_order_relaxed);
	std::printf("12 pass changes + frames: %zu allocations, %zu bytes\n",
		    allocations, bytes);

	// This is a realtime invariant: changing quality and processing another
	// frame must reuse the plan and every working buffer.
	if (allocations != 0 || bytes != 0)
		return 5;

	// An in-place pass change must remain bit-identical to a freshly created
	// plan with that pass count.
	constexpr int comparison_passes = 13;
	bfft_meyer_plan *reference_plan = nullptr;
	std::vector<double> reference_cartoon(count);
	std::vector<double> reference_texture(count);
	if (bfft_meyer_plan_set_passes(plan, comparison_passes) != BFFT_OK ||
	    bfft_meyer_split(plan, input.data(), cartoon.data(),
			     texture.data()) != BFFT_OK ||
	    bfft_meyer_plan_create(height, width, 0.05, 40.0,
				   comparison_passes, 32, 1e-4, 4,
				   &reference_plan) != BFFT_OK ||
	    bfft_meyer_split(reference_plan, input.data(),
			     reference_cartoon.data(),
			     reference_texture.data()) != BFFT_OK)
		return 6;
	double max_error = 0.0;
	for (size_t i = 0; i < count; ++i) {
		max_error = std::max(max_error,
				     std::abs(cartoon[i] - reference_cartoon[i]));
		max_error = std::max(max_error,
				     std::abs(texture[i] - reference_texture[i]));
	}
	bfft_meyer_plan_destroy(reference_plan);
	bfft_meyer_plan_destroy(plan);
	if (max_error != 0.0)
		return 7;
	return 0;
}
