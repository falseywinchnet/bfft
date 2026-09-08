#include "dip-comb-transport.hpp"

#include "../src/detail/bruun_dip_kernel.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>

namespace {

using Complex = std::complex<double>;
using Clock = std::chrono::steady_clock;
constexpr double tau = 6.283185307179586476925286766559;

uint32_t floor_power_two(uint32_t value, uint32_t ceiling)
{
	value = std::min(value, ceiling);
	uint32_t result = 1;
	while (result <= value / 2)
		result <<= 1;
	return result;
}

class ThreadTeam {
public:
	ThreadTeam()
	{
		for (int id = 1; id < 8; ++id)
			threads_.emplace_back([this, id] { worker_loop(id); });
	}

	~ThreadTeam()
	{
		{
			std::lock_guard<std::mutex> lock(mutex_);
			stopping_ = true;
			++generation_;
		}
		ready_.notify_all();
		for (std::thread &thread : threads_)
			thread.join();
	}

	ThreadTeam(const ThreadTeam &) = delete;
	ThreadTeam &operator=(const ThreadTeam &) = delete;

	template <class Function>
	void run(uint32_t count, int requested_threads, Function &&function)
	{
		const int hardware = static_cast<int>(
			std::max(1u, std::thread::hardware_concurrency()));
		{
			std::lock_guard<std::mutex> lock(mutex_);
			worker_count_ = std::clamp(requested_threads, 1,
				std::min(8, hardware));
			count_ = count;
			next_.store(0, std::memory_order_relaxed);
			function_ = std::forward<Function>(function);
			remaining_ = 7;
			++generation_;
		}
		ready_.notify_all();
		drain(0);
		std::unique_lock<std::mutex> lock(mutex_);
		done_.wait(lock, [this] { return remaining_ == 0; });
		function_ = {};
	}

private:
	void drain(int id)
	{
		if (id >= worker_count_)
			return;
		for (;;) {
			const uint32_t index = next_.fetch_add(1, std::memory_order_relaxed);
			if (index >= count_)
				break;
			function_(index, id);
		}
	}

	void worker_loop(int id)
	{
		uint64_t observed = 0;
		for (;;) {
			{
				std::unique_lock<std::mutex> lock(mutex_);
				ready_.wait(lock, [&] {
					return stopping_ || generation_ != observed;
				});
				if (stopping_)
					return;
				observed = generation_;
			}
			drain(id);
			{
				std::lock_guard<std::mutex> lock(mutex_);
				if (--remaining_ == 0)
					done_.notify_one();
			}
		}
	}

	std::vector<std::thread> threads_;
	std::mutex mutex_;
	std::condition_variable ready_;
	std::condition_variable done_;
	std::function<void(uint32_t, int)> function_;
	std::atomic<uint32_t> next_{0};
	uint32_t count_ = 0;
	int worker_count_ = 1;
	int remaining_ = 0;
	uint64_t generation_ = 0;
	bool stopping_ = false;
};

int signed_frequency(uint32_t k, uint32_t n)
{
	return k <= n / 2 ? static_cast<int>(k) :
		static_cast<int>(k) - static_cast<int>(n);
}

uint32_t folded_residue(int frequency, uint32_t denominator)
{
	int residue = frequency % static_cast<int>(denominator);
	if (residue < 0)
		residue += static_cast<int>(denominator);
	return static_cast<uint32_t>(std::min(
		residue, static_cast<int>(denominator) - residue));
}

Complex from_bruun(const bruun::complex_t &value)
{
	return {value.re, value.im};
}

bruun::complex_t to_bruun(const Complex &value)
{
	return {value.real(), value.imag()};
}

struct CombStatistics {
	double energy = 0.0;
	Complex correlation_x{};
	Complex correlation_y{};
	double correlation_energy_x0 = 0.0;
	double correlation_energy_x1 = 0.0;
	double correlation_energy_y0 = 0.0;
	double correlation_energy_y1 = 0.0;
	double mean_x = 0.0;
	double mean_y = 0.0;
	double sigma_x = 1.0;
	double sigma_y = 1.0;
	double coherence = 0.0;
};

} // namespace

struct DipCombTransport2D::Impl {
	struct WorkerScratch {
		std::vector<double> row_work;
		std::vector<bruun::complex_t> row_bins;
		std::vector<double> column_real;
		std::vector<double> column_imag;
		std::vector<double> column_work;
		std::vector<double> column_real_output;
		std::vector<double> column_imag_output;
		std::vector<bruun::complex_t> column_real_bins;
		std::vector<bruun::complex_t> column_imag_bins;
	};

	uint32_t nx = 0;
	uint32_t ny = 0;
	uint32_t bins_x = 0;
	bruun::DIP_RFFT_kernel row_plan;
	bruun::DIP_RFFT_kernel column_plan;
	std::vector<double> spatial;
	std::vector<double> transformed;
	std::vector<double> uncertainty_spatial;
	std::vector<double> gradient_x;
	std::vector<double> gradient_y;
	std::vector<float> coarse_displacement_x;
	std::vector<float> coarse_displacement_y;
	std::vector<float> coarse_residual;
	std::vector<float> coarse_uncertainty;
	std::vector<float> coarse_phase;
	std::vector<float> coarse_conv_admission;
	std::vector<float> coarse_conv_confidence;
	std::vector<Complex> row_spectrum;
	std::vector<Complex> spectrum;
	std::vector<Complex> uncertainty_spectrum;
	std::vector<double> radial_frequency;
	std::vector<double> reverse_profile;
	std::vector<Complex> transfer_x;
	std::vector<Complex> transfer_y;
	std::vector<double> attenuation_x;
	std::vector<double> attenuation_y;
	std::array<WorkerScratch, 8> scratch;
	ThreadTeam team;
	DipCombTimings timings;

	bool ensure(uint32_t width, uint32_t height)
	{
		const uint32_t wanted_x = floor_power_two(width, 1024);
		const uint32_t wanted_y = floor_power_two(height, 512);
		if (wanted_x < 16 || wanted_y < 16)
			return false;
		if (wanted_x == nx && wanted_y == ny)
			return true;
		nx = wanted_x;
		ny = wanted_y;
		bins_x = nx / 2 + 1;
		if (!row_plan.reset(static_cast<int>(nx)) ||
		    !column_plan.reset(static_cast<int>(ny)))
			return false;
		spatial.resize(static_cast<size_t>(nx) * ny);
		transformed.resize(spatial.size());
		uncertainty_spatial.resize(spatial.size());
		gradient_x.resize(spatial.size());
		gradient_y.resize(spatial.size());
		coarse_displacement_x.resize(spatial.size());
		coarse_displacement_y.resize(spatial.size());
		coarse_residual.resize(spatial.size());
		coarse_uncertainty.resize(spatial.size());
		coarse_phase.resize(spatial.size());
		coarse_conv_admission.resize(spatial.size());
		coarse_conv_confidence.resize(spatial.size());
		row_spectrum.resize(static_cast<size_t>(ny) * bins_x);
		spectrum.resize(static_cast<size_t>(bins_x) * ny);
		uncertainty_spectrum.resize(spectrum.size());
		radial_frequency.resize(spectrum.size());
		reverse_profile.resize(spectrum.size());
		for (uint32_t kx = 0; kx < bins_x; ++kx) {
			for (uint32_t ky = 0; ky < ny; ++ky) {
				const int sy = signed_frequency(ky, ny);
				const double fx = static_cast<double>(kx) / (nx / 2.0);
				const double fy = static_cast<double>(sy) / (ny / 2.0);
				const double rho = std::clamp(
					std::sqrt(0.5 * (fx * fx + fy * fy)), 0.0, 1.0);
				const size_t index = static_cast<size_t>(kx) * ny + ky;
				radial_frequency[index] = rho;
				reverse_profile[index] = std::pow(1.0 - rho, 1.7);
			}
		}
		for (WorkerScratch &worker : scratch) {
			worker.row_work.resize(nx);
			worker.row_bins.resize(bins_x);
			worker.column_real.resize(ny);
			worker.column_imag.resize(ny);
			worker.column_work.resize(ny);
			worker.column_real_output.resize(ny);
			worker.column_imag_output.resize(ny);
			worker.column_real_bins.resize(ny / 2 + 1);
			worker.column_imag_bins.resize(ny / 2 + 1);
		}
		return true;
	}

	void resample_input(const std::vector<float> &source, uint32_t width,
			    uint32_t height)
	{
		if (width == nx && height == ny) {
			for (size_t i = 0; i < source.size(); ++i)
				spatial[i] = source[i] / 255.0;
			return;
		}
		for (uint32_t y = 0; y < ny; ++y) {
			const double sy = (static_cast<double>(y) + 0.5) * height / ny - 0.5;
			const uint32_t y0 = static_cast<uint32_t>(
				std::clamp(std::floor(sy), 0.0, static_cast<double>(height - 1)));
			const uint32_t y1 = std::min(height - 1, y0 + 1);
			const double fy = std::clamp(sy - std::floor(sy), 0.0, 1.0);
			for (uint32_t x = 0; x < nx; ++x) {
				const double sx = (static_cast<double>(x) + 0.5) * width / nx - 0.5;
				const uint32_t x0 = static_cast<uint32_t>(
					std::clamp(std::floor(sx), 0.0, static_cast<double>(width - 1)));
				const uint32_t x1 = std::min(width - 1, x0 + 1);
				const double fx = std::clamp(sx - std::floor(sx), 0.0, 1.0);
				const double a = source[static_cast<size_t>(y0) * width + x0];
				const double b = source[static_cast<size_t>(y0) * width + x1];
				const double c = source[static_cast<size_t>(y1) * width + x0];
				const double d = source[static_cast<size_t>(y1) * width + x1];
				spatial[static_cast<size_t>(y) * nx + x] =
					((1.0 - fy) * ((1.0 - fx) * a + fx * b) +
					 fy * ((1.0 - fx) * c + fx * d)) / 255.0;
			}
		}
	}

	void forward_2d(int thread_count)
	{
		const auto started = Clock::now();
		team.run(ny, thread_count, [&](uint32_t y, int worker_index) {
			WorkerScratch &worker = scratch[static_cast<size_t>(worker_index)];
			row_plan.forward_standard(spatial.data() + static_cast<size_t>(y) * nx,
				worker.row_bins.data(), worker.row_work.data());
			for (uint32_t kx = 0; kx < bins_x; ++kx)
				row_spectrum[static_cast<size_t>(y) * bins_x + kx] =
					from_bruun(worker.row_bins[kx]);
		});
		team.run(bins_x, thread_count, [&](uint32_t kx, int worker_index) {
			WorkerScratch &worker = scratch[static_cast<size_t>(worker_index)];
			for (uint32_t y = 0; y < ny; ++y) {
				const Complex value = row_spectrum[static_cast<size_t>(y) * bins_x + kx];
				worker.column_real[y] = value.real();
				worker.column_imag[y] = value.imag();
			}
			column_plan.forward_standard(worker.column_real.data(),
				worker.column_real_bins.data(), worker.column_work.data());
			column_plan.forward_standard(worker.column_imag.data(),
				worker.column_imag_bins.data(), worker.column_work.data());
			for (uint32_t ky = 0; ky < ny; ++ky) {
				const uint32_t folded = ky <= ny / 2 ? ky : ny - ky;
				Complex r = from_bruun(worker.column_real_bins[folded]);
				Complex i = from_bruun(worker.column_imag_bins[folded]);
				if (ky > ny / 2) {
					r = std::conj(r);
					i = std::conj(i);
				}
				spectrum[static_cast<size_t>(kx) * ny + ky] =
					r + Complex(0.0, 1.0) * i;
			}
		});
		timings.forward_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - started).count();
	}

	void inverse_2d(const std::vector<Complex> &input,
			std::vector<double> &output, int thread_count)
	{
		const auto started = Clock::now();
		team.run(bins_x, thread_count, [&](uint32_t kx, int worker_index) {
			WorkerScratch &worker = scratch[static_cast<size_t>(worker_index)];
			for (uint32_t ky = 0; ky <= ny / 2; ++ky) {
				const Complex a = input[static_cast<size_t>(kx) * ny + ky];
				const Complex b = std::conj(input[static_cast<size_t>(kx) * ny +
					((ny - ky) % ny)]);
				const Complex real_part = 0.5 * (a + b);
				const Complex difference = a - b;
				const Complex imag_part(0.5 * difference.imag(),
						       -0.5 * difference.real());
				worker.column_real_bins[ky] = to_bruun(real_part);
				worker.column_imag_bins[ky] = to_bruun(imag_part);
			}
			column_plan.inverse_standard(worker.column_real_bins.data(),
				worker.column_real_output.data(), worker.column_work.data());
			column_plan.inverse_standard(worker.column_imag_bins.data(),
				worker.column_imag_output.data(), worker.column_work.data());
			for (uint32_t y = 0; y < ny; ++y)
				row_spectrum[static_cast<size_t>(y) * bins_x + kx] =
					{worker.column_real_output[y], worker.column_imag_output[y]};
		});
		output.resize(static_cast<size_t>(nx) * ny);
		team.run(ny, thread_count, [&](uint32_t y, int worker_index) {
			WorkerScratch &worker = scratch[static_cast<size_t>(worker_index)];
			for (uint32_t kx = 0; kx < bins_x; ++kx)
				worker.row_bins[kx] = to_bruun(
					row_spectrum[static_cast<size_t>(y) * bins_x + kx]);
			// The self-conjugate horizontal bins must be real for each row.
			worker.row_bins[0].im = 0.0;
			worker.row_bins[nx / 2].im = 0.0;
			row_plan.inverse_standard(worker.row_bins.data(),
				output.data() + static_cast<size_t>(y) * nx,
				worker.row_work.data());
		});
		timings.inverse_ms += std::chrono::duration<double, std::milli>(
			Clock::now() - started).count();
	}

	void transpose_combs(double strength, double reverse_strength)
	{
		const auto started = Clock::now();
		constexpr uint32_t denominator = 8;
		constexpr uint32_t folded_count = denominator / 2 + 1;
		std::array<CombStatistics, folded_count * folded_count> combs{};
		auto comb_index = [&](int kx, int ky) {
			return folded_residue(kx, denominator) * folded_count +
			       folded_residue(ky, denominator);
		};
		for (uint32_t kx = 0; kx < bins_x; ++kx) {
			for (uint32_t ky = 0; ky < ny; ++ky) {
				const int sy = signed_frequency(ky, ny);
				const Complex value = spectrum[static_cast<size_t>(kx) * ny + ky];
				CombStatistics &comb = combs[comb_index(static_cast<int>(kx), sy)];
				comb.energy += std::norm(value);
				if (kx >= denominator) {
					const Complex previous = spectrum[
						static_cast<size_t>(kx - denominator) * ny + ky];
					comb.correlation_x += value * std::conj(previous);
					comb.correlation_energy_x0 += std::norm(value);
					comb.correlation_energy_x1 += std::norm(previous);
				}
				const uint32_t previous_y = (ky + ny - denominator) % ny;
				const Complex previous = spectrum[
					static_cast<size_t>(kx) * ny + previous_y];
				comb.correlation_y += value * std::conj(previous);
				comb.correlation_energy_y0 += std::norm(value);
				comb.correlation_energy_y1 += std::norm(previous);
			}
		}
		for (CombStatistics &comb : combs) {
			const double coherence_x = std::clamp(
				std::abs(comb.correlation_x) /
				std::max(1e-12, std::sqrt(comb.correlation_energy_x0 *
					comb.correlation_energy_x1)), 0.0, 1.0);
			const double coherence_y = std::clamp(
				std::abs(comb.correlation_y) /
				std::max(1e-12, std::sqrt(comb.correlation_energy_y0 *
					comb.correlation_energy_y1)), 0.0, 1.0);
			comb.mean_x = -std::arg(comb.correlation_x) * nx /
				(tau * denominator);
			comb.mean_y = -std::arg(comb.correlation_y) * ny /
				(tau * denominator);
			comb.mean_x = std::clamp(comb.mean_x,
				-static_cast<double>(nx) / 8.0, static_cast<double>(nx) / 8.0);
			comb.mean_y = std::clamp(comb.mean_y,
				-static_cast<double>(ny) / 8.0, static_cast<double>(ny) / 8.0);
			// Adjacent teeth differ by angular frequency 2*pi*e/N. Under
			// Gaussian displacement their normalized correlation magnitude
			// is exp(-0.5*(2*pi*e*sigma/N)^2), so sigma follows by inversion.
			comb.sigma_x = std::clamp(
				nx * std::sqrt(-2.0 * std::log(std::max(1e-8, coherence_x))) /
					(tau * denominator),
				0.25, static_cast<double>(nx) / (2.0 * denominator));
			comb.sigma_y = std::clamp(
				ny * std::sqrt(-2.0 * std::log(std::max(1e-8, coherence_y))) /
					(tau * denominator),
				0.25, static_cast<double>(ny) / (2.0 * denominator));
			comb.coherence = std::sqrt(coherence_x * coherence_y);
		}
		transfer_x.resize(combs.size() * bins_x);
		attenuation_x.resize(combs.size() * bins_x);
		transfer_y.resize(combs.size() * ny);
		attenuation_y.resize(combs.size() * ny);
		for (size_t c = 0; c < combs.size(); ++c) {
			const CombStatistics &comb = combs[c];
			const double mean_x = comb.mean_y * nx / ny;
			const double mean_y = comb.mean_x * ny / nx;
			const double sigma_x = comb.sigma_y * nx / ny;
			const double sigma_y = comb.sigma_x * ny / nx;
			for (uint32_t kx = 0; kx < bins_x; ++kx) {
				const double normalized = static_cast<double>(kx) / (nx / 2.0);
				const double shift = strength *
					(0.08 + 1.42 * std::pow(normalized, 1.35));
				const double angular = tau * shift * kx / nx;
				const double attenuation = std::exp(
					-0.5 * angular * angular * sigma_x * sigma_x);
				const double theta = angular * mean_x;
				transfer_x[c * bins_x + kx] = {
					attenuation * std::cos(theta),
					-attenuation * std::sin(theta)};
				attenuation_x[c * bins_x + kx] = attenuation;
			}
			for (uint32_t ky = 0; ky < ny; ++ky) {
				const int sy = signed_frequency(ky, ny);
				const double normalized = std::abs(static_cast<double>(sy)) /
					(ny / 2.0);
				const double shift = strength *
					(0.08 + 1.42 * std::pow(normalized, 1.35));
				const double angular = tau * shift * sy / ny;
				const double attenuation = std::exp(
					-0.5 * angular * angular * sigma_y * sigma_y);
				const double theta = angular * mean_y;
				transfer_y[c * ny + ky] = {
					attenuation * std::cos(theta),
					-attenuation * std::sin(theta)};
				attenuation_y[c * ny + ky] = attenuation;
			}
		}

		for (uint32_t kx = 0; kx < bins_x; ++kx) {
			for (uint32_t ky = 0; ky < ny; ++ky) {
				const int sy = signed_frequency(ky, ny);
				const size_t c = comb_index(static_cast<int>(kx), sy);
				const CombStatistics &comb = combs[c];
				// Transpose the inferred displacement and covariance axes in
				// normalized comb coordinates, then restore pixel units.
				const double sigma_x = comb.sigma_y * nx / ny;
				const double sigma_y = comb.sigma_x * ny / nx;
				// If the transposed displacement V is Gaussian, its coherent
				// superposition of translation characters is exact:
				//   E exp(-i omega.V) = exp(-i omega.mu - omega^T Sigma omega/2).
				// Diagonal covariance makes it separable, so phase factors are
				// evaluated once per comb/axis and multiplied tooth-wise here.
				Complex transfer = transfer_x[c * bins_x + kx] *
					transfer_y[c * ny + ky];
				const double attenuation = attenuation_x[c * bins_x + kx] *
					attenuation_y[c * ny + ky];
				if (kx == 0 && sy == 0)
					transfer = {1.0, 0.0};
				const size_t index = static_cast<size_t>(kx) * ny + ky;
				const double rho = radial_frequency[index];
				const double reverse = std::clamp(reverse_strength, 0.0, 1.0) *
					reverse_profile[index];
				// The inverse is exact in the unitary/low-uncertainty limit.
				// Conditioning is introduced only where Gaussian cancellation
				// has already destroyed phase information; a fixed floor would
				// itself create a low-frequency residual.
				const double lost_amplitude = 1.0 - attenuation;
				const double regularizer = 1e-8 +
					0.08 * (1.0 - comb.coherence) *
					lost_amplitude * lost_amplitude * (0.2 + 0.8 * rho * rho);
				const Complex inverse = std::conj(transfer) /
					(std::norm(transfer) + regularizer);
				const Complex partial_inverse =
					(1.0 - reverse) * Complex(1.0, 0.0) + reverse * inverse;
				const Complex original = spectrum[index];
				const Complex moved = transfer * original;
				spectrum[index] = partial_inverse * moved;
				const double variance = sigma_x * sigma_x + sigma_y * sigma_y;
				const double uncertain = std::clamp(
					(1.0 - attenuation) +
					0.02 * std::sqrt(variance) * rho, 0.0, 1.0);
				// The uncertainty image is the spectrally lost component,
				// referred to the input coefficient. Multiplying by `moved`
				// would make maximal Gaussian cancellation look certain.
				uncertainty_spectrum[index] = uncertain * original;
			}
		}
		// Preserve the Hermitian constraint on the self-conjugate x columns.
		for (uint32_t kx : {0u, nx / 2}) {
			for (uint32_t ky = 1; ky < ny / 2; ++ky) {
				const size_t positive = static_cast<size_t>(kx) * ny + ky;
				const size_t negative = static_cast<size_t>(kx) * ny + (ny - ky);
				const Complex mean = 0.5 *
					(spectrum[positive] + std::conj(spectrum[negative]));
				spectrum[positive] = mean;
				spectrum[negative] = std::conj(mean);
				const Complex uncertainty_mean = 0.5 *
					(uncertainty_spectrum[positive] +
					 std::conj(uncertainty_spectrum[negative]));
				uncertainty_spectrum[positive] = uncertainty_mean;
				uncertainty_spectrum[negative] = std::conj(uncertainty_mean);
			}
			spectrum[static_cast<size_t>(kx) * ny].imag(0.0);
			spectrum[static_cast<size_t>(kx) * ny + ny / 2].imag(0.0);
			uncertainty_spectrum[static_cast<size_t>(kx) * ny].imag(0.0);
			uncertainty_spectrum[static_cast<size_t>(kx) * ny + ny / 2].imag(0.0);
		}
		timings.comb_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - started).count();
	}

	void expand_field(uint32_t width, uint32_t height, int thread_count,
			  DipCombField &field)
	{
		const auto started = Clock::now();
		team.run(ny, thread_count, [&](uint32_t y, int) {
			const uint32_t ym = y == 0 ? 0 : y - 1;
			const uint32_t yp = std::min(ny - 1, y + 1);
			for (uint32_t x = 0; x < nx; ++x) {
				const uint32_t xm = x == 0 ? 0 : x - 1;
				const uint32_t xp = std::min(nx - 1, x + 1);
				const size_t index = static_cast<size_t>(y) * nx + x;
				gradient_x[index] = 0.5 *
					(transformed[static_cast<size_t>(y) * nx + xp] -
					 transformed[static_cast<size_t>(y) * nx + xm]);
				gradient_y[index] = 0.5 *
					(transformed[static_cast<size_t>(yp) * nx + x] -
					 transformed[static_cast<size_t>(ym) * nx + x]);
				const double original = spatial[index];
				const double target = transformed[index];
				const double uncertain = std::abs(uncertainty_spatial[index]);
				const double source_dx = 0.5 *
					(spatial[static_cast<size_t>(y) * nx + xp] -
					 spatial[static_cast<size_t>(y) * nx + xm]);
				const double source_dy = 0.5 *
					(spatial[static_cast<size_t>(yp) * nx + x] -
					 spatial[static_cast<size_t>(ym) * nx + x]);
				const double current_norm2 =
					source_dx * source_dx + source_dy * source_dy;
				const double delta = target - original;
				const double denominator = gradient_x[index] * gradient_x[index] +
					gradient_y[index] * gradient_y[index] + 0.0015 +
					0.02 * uncertain * uncertain;
				const double flow_x = std::clamp(
					-delta * gradient_x[index] / denominator, -12.0, 12.0);
				const double flow_y = std::clamp(
					-delta * gradient_y[index] / denominator, -12.0, 12.0);
				coarse_displacement_x[index] = static_cast<float>(flow_x / nx);
				coarse_displacement_y[index] = static_cast<float>(flow_y / ny);
				coarse_residual[index] = static_cast<float>(
					std::clamp(delta, -0.5, 0.5));
				coarse_uncertainty[index] = static_cast<float>(
					std::clamp(2.5 * uncertain, 0.0, 1.0));
				coarse_phase[index] = static_cast<float>(
					std::atan2(flow_y, flow_x) / tau);
				// compiled operator): eta = Dy^2 / (Dx^2 + Dy^2). Q1 expansion
				// below is its unique separately affine, cardinal continuation.
				// CONV*'s exact nodal factor-order coordinate is
				// eta = Dy^2/(Dx^2+Dy^2). Q1 expansion below is its unique
				// separately affine, cardinal continuation.
				// compiled operator): eta = Dy^2 / (Dx^2 + Dy^2). Q1 expansion
				// below is its unique separately affine, cardinal continuation.
				coarse_conv_admission[index] = static_cast<float>(
					current_norm2 > 1e-16 ? source_dy * source_dy / current_norm2 : 0.5);
				coarse_conv_confidence[index] = static_cast<float>(
					1.0 - std::exp(-18.0 * std::sqrt(current_norm2)));
			}
		});
		const size_t count = static_cast<size_t>(width) * height;
		field.width = width;
		field.height = height;
		field.displacement_x.resize(count);
		field.displacement_y.resize(count);
		field.residual.resize(count);
		field.uncertainty.resize(count);
		field.phase.resize(count);
		field.conv_admission.resize(count);
		field.conv_confidence.resize(count);
		if (width == nx && height == ny) {
			field.displacement_x = coarse_displacement_x;
			field.displacement_y = coarse_displacement_y;
			field.residual = coarse_residual;
			field.uncertainty = coarse_uncertainty;
			field.phase = coarse_phase;
			field.conv_admission = coarse_conv_admission;
			field.conv_confidence = coarse_conv_confidence;
			timings.expand_ms = std::chrono::duration<double, std::milli>(
				Clock::now() - started).count();
			return;
		}
		team.run(height, thread_count, [&](uint32_t y, int) {
			const double gy = (static_cast<double>(y) + 0.5) * ny / height - 0.5;
			const double cy = std::clamp(gy, 0.0, static_cast<double>(ny - 1));
			const uint32_t y0 = static_cast<uint32_t>(cy);
			const uint32_t y1 = std::min(ny - 1, y0 + 1);
			const double fy = cy - y0;
			for (uint32_t x = 0; x < width; ++x) {
				const double gx = (static_cast<double>(x) + 0.5) * nx / width - 0.5;
				const double cx = std::clamp(gx, 0.0, static_cast<double>(nx - 1));
				const uint32_t x0 = static_cast<uint32_t>(cx);
				const uint32_t x1 = std::min(nx - 1, x0 + 1);
				const double fx = cx - x0;
				const size_t a = static_cast<size_t>(y0) * nx + x0;
				const size_t b = static_cast<size_t>(y0) * nx + x1;
				const size_t c = static_cast<size_t>(y1) * nx + x0;
				const size_t d = static_cast<size_t>(y1) * nx + x1;
				auto sample = [&](const std::vector<float> &source) {
					return (1.0 - fy) * ((1.0 - fx) * source[a] + fx * source[b]) +
					       fy * ((1.0 - fx) * source[c] + fx * source[d]);
				};
				const size_t index = static_cast<size_t>(y) * width + x;
				field.displacement_x[index] = static_cast<float>(sample(coarse_displacement_x));
				field.displacement_y[index] = static_cast<float>(sample(coarse_displacement_y));
				field.residual[index] = static_cast<float>(sample(coarse_residual));
				field.uncertainty[index] = static_cast<float>(sample(coarse_uncertainty));
				field.conv_admission[index] = static_cast<float>(sample(coarse_conv_admission));
				field.conv_confidence[index] = static_cast<float>(sample(coarse_conv_confidence));
				// Unwrap all four turns against the first corner, interpolate on
				// the unique local short arc, then wrap back to [-1/2,1/2).
				const double pa = coarse_phase[a];
				auto unwrap = [&](double p) {
					return pa + (p - pa) - std::floor((p - pa) + 0.5);
				};
				const double pb = unwrap(coarse_phase[b]);
				const double pc = unwrap(coarse_phase[c]);
				const double pd = unwrap(coarse_phase[d]);
				double turn = (1.0 - fy) * ((1.0 - fx) * pa + fx * pb) +
					fy * ((1.0 - fx) * pc + fx * pd);
				turn -= std::floor(turn + 0.5);
				field.phase[index] = static_cast<float>(turn);
			}
		});
		timings.expand_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - started).count();
	}
};

DipCombTransport2D::DipCombTransport2D() : impl_(std::make_unique<Impl>()) {}
DipCombTransport2D::~DipCombTransport2D() = default;

bool DipCombTransport2D::process(const std::vector<float> &oklch_lightness,
				 uint32_t width, uint32_t height,
				 double transpose_strength, double reverse_strength,
				 int thread_count, DipCombField &field)
{
	const auto started = Clock::now();
	if (oklch_lightness.size() != static_cast<size_t>(width) * height ||
	    !impl_->ensure(width, height))
		return false;
	const auto resample_started = Clock::now();
	impl_->resample_input(oklch_lightness, width, height);
	impl_->timings.resample_ms = std::chrono::duration<double, std::milli>(
		Clock::now() - resample_started).count();
	impl_->forward_2d(thread_count);
	impl_->transpose_combs(std::clamp(transpose_strength, 0.0, 3.0),
			       reverse_strength);
	impl_->timings.inverse_ms = 0.0;
	impl_->inverse_2d(impl_->spectrum, impl_->transformed, thread_count);
	impl_->inverse_2d(impl_->uncertainty_spectrum,
			  impl_->uncertainty_spatial, thread_count);
	impl_->expand_field(width, height, thread_count, field);
	impl_->timings.total_ms = std::chrono::duration<double, std::milli>(
		Clock::now() - started).count();
	return true;
}

uint32_t DipCombTransport2D::transform_width() const noexcept { return impl_->nx; }
uint32_t DipCombTransport2D::transform_height() const noexcept { return impl_->ny; }
const char *DipCombTransport2D::simd_backend() const noexcept
{
	return bruun::simd_backend_name();
}
const DipCombTimings &DipCombTransport2D::last_timings() const noexcept
{
	return impl_->timings;
}
