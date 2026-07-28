#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace high_vision {

// The engine works in scene-linear, normalized luminance. Camera adapters are
// responsible for decoding transfer functions before process() and encoding
// them afterwards.
enum class Mode {
	passthrough = 0,
	synthetic_hdr = 1,
	night_integrator = 2,
	night_likelihood = 3,
	experimental = 4,
};

struct FrameMetadata {
	std::uint64_t timestamp_ns = 0;
	std::uint64_t sequence = 0;

	// Optional camera telemetry. Zero means unknown. When both exposure and
	// gain are known they anchor radiance and prevent long-term AGC drift.
	double exposure_seconds = 0.0;
	double analog_gain = 0.0;
	double black_level = 0.0;
	double white_level = 1.0;
	double sensor_temperature_c = 0.0;
};

struct Config {
	Mode mode = Mode::synthetic_hdr;

	// Registration is a global translation plus a smoothly interpolated field
	// of independently matched tiles. All values are on the processing grid.
	int registration_radius = 6;
	int tile_size = 24;
	int local_search_radius = 2;
	bool meyer_registration = true;
	int meyer_registration_passes = 4;

	// Evidence is a bounded precision, not a frame counter. Old support decays
	// while reliable, registered observations refill it.
	float support_limit = 24.0f;
	float support_decay = 0.985f;
	float change_threshold = 0.08f;
	float scene_cut_threshold = 0.24f;

	// Experimental Night likelihood path. Rather than treating a residual
	// magnitude as change, it accumulates a sequential log-likelihood ratio
	// between the transported belief and meaningful darker/brighter
	// alternatives. Negative observations therefore count as evidence: a run
	// of unexpectedly dark samples can quickly bankrupt stale bright support.
	float likelihood_release_low = 3.0f;
	float likelihood_release_high = 8.0f;
	float likelihood_evidence_limit = 16.0f;

	// Approximate linear-camera reliability model. For scene-linear x,
	// Var[y | x] = read_noise^2 + shot_noise^2 * x. In a calibrated RAW
	// adapter, shot_noise^2 is the system-gain coefficient alpha.
	float shadow_floor = 0.008f;
	float highlight_knee = 0.94f;
	float read_noise = 0.006f;
	float shot_noise = 0.035f;

	// A sensor-fixed nuisance field lives in detector coordinates rather
	// than in the transported scene gauge. It is identifiable only while
	// registered scene content moves across the detector, so updates are
	// motion gated and constrained to have zero spatial mean.
	bool sensor_pattern_correction = true;
	float sensor_pattern_learning_rate = 0.02f;
	float sensor_pattern_limit = 0.04f;
	float sensor_pattern_min_motion = 0.75f;

	// Display transform. The belief remains scene-linear; only output is
	// percentile-normalized and contrast shaped.
	float black_percentile = 0.005f;
	float white_percentile = 0.995f;
	float tone_strength = 1.0f;
	float local_contrast = 0.15f;
};

struct Diagnostics {
	std::uint64_t frame_index = 0;
	float global_dx = 0.0f;
	float global_dy = 0.0f;
	float registration_error = 0.0f;
	float registration_confidence = 0.0f;
	float relative_exposure = 1.0f;
	float mean_support = 0.0f;
	float mean_change_probability = 0.0f;
	float clipped_fraction = 0.0f;
	float sensor_pattern_rms = 0.0f;
	std::uint64_t sensor_pattern_updates = 0;
	bool meyer_registration_applied = false;
	bool reset = false;
};

// A score backend supplies one row per frame and one column per admissible
// registration. Scores may omit terms that are constant across a row. This
// controller turns them into a bounded posterior while retaining only
// cumulative cross-photon evidence between calls.
struct EntropyControllerConfig {
	std::vector<float> budgets = {
		4.0f, 8.0f, 12.0f, 16.0f, 24.0f,
		32.0f, 48.0f, 64.0f, 96.0f,
	};
	int bisection_steps = 32;
	int max_budget_steps_per_batch = 1;

	// One is the validated no-forgetting operator. Values below one are
	// available for explicit scene-drift experiments; scene cuts should reset
	// the controller instead of silently erasing evidence.
	float evidence_retention = 1.0f;
};

struct EntropyProjectionDiagnostics {
	float target_effective_shifts = 1.0f;
	float effective_shifts = 1.0f;
	float mean_frame_effective_shifts = 1.0f;
	float mean_peak_probability = 1.0f;
	float inverse_temperature = 0.0f;
	float entropy_residual_nats = 0.0f;
};

struct EntropyControllerDiagnostics {
	std::uint64_t batches = 0;
	std::uint64_t support_transitions = 0;
	float selected_budget = 1.0f;
	float batch_best_budget = 1.0f;
	float cumulative_evidence_margin_per_pixel = 0.0f;
	EntropyProjectionDiagnostics projection;
};

class EntropySupportController {
public:
	explicit EntropySupportController(EntropyControllerConfig config = {});
	~EntropySupportController();
	EntropySupportController(EntropySupportController &&) noexcept;
	EntropySupportController &operator=(EntropySupportController &&) noexcept;
	EntropySupportController(const EntropySupportController &) = delete;
	EntropySupportController &operator=(const EntropySupportController &) =
		delete;

	void configure(const EntropyControllerConfig &config);
	const EntropyControllerConfig &config() const noexcept;
	void reset();

	// Complementary score fields must have frames * shifts row-major values.
	// Each half predicts the other; the resulting evidence is accumulated and
	// selects the support budget for the next unsplit projection.
	bool update(const float *first_scores,
		    const float *second_scores,
		    std::size_t frames,
		    std::size_t shifts,
		    std::size_t pixels_per_frame);

	// Project unsplit scores using the currently selected budget. The output
	// is resized to frames * shifts and every row sums to one.
	bool project_selected(const float *scores,
			      std::size_t frames,
			      std::size_t shifts,
			      std::vector<float> &posterior);

	// Stateless projection primitive, useful to score candidate budgets and
	// to test a backend before enabling cumulative selection.
	bool project(const float *scores,
		     std::size_t frames,
		     std::size_t shifts,
		     float target_effective_shifts,
		     std::vector<float> &posterior,
		     EntropyProjectionDiagnostics *diagnostics = nullptr) const;

	const EntropyControllerDiagnostics &diagnostics() const noexcept;
	const std::vector<double> &cumulative_evidence() const noexcept;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

// This is the landing point for inverse-diffusion and other future estimators.
// It receives the registered temporal belief and may replace it in-place. A
// stage never owns capture, registration, support transport, or tone mapping.
class ExperimentalStage {
public:
	virtual ~ExperimentalStage() = default;
	virtual const char *name() const noexcept = 0;
	virtual void reset(std::size_t width, std::size_t height) = 0;
	virtual void process(const float *observation,
			     const float *transported_support,
			     float *belief,
			     std::size_t width,
			     std::size_t height,
			     const FrameMetadata &metadata,
			     const Diagnostics &diagnostics) = 0;
};

class Processor {
public:
	explicit Processor(Config config = {});
	~Processor();
	Processor(Processor &&) noexcept;
	Processor &operator=(Processor &&) noexcept;
	Processor(const Processor &) = delete;
	Processor &operator=(const Processor &) = delete;

	void configure(const Config &config);
	const Config &config() const noexcept;
	void set_experimental_stage(std::unique_ptr<ExperimentalStage> stage);
	const ExperimentalStage *experimental_stage() const noexcept;

	void reset();

	// input and output may alias. Strides are measured in float elements.
	// Returns false for invalid arguments; a valid frame always produces output.
	bool process(const float *input,
		     std::size_t input_stride,
		     float *output,
		     std::size_t output_stride,
		     std::size_t width,
		     std::size_t height,
		     const FrameMetadata &metadata = {});

	const Diagnostics &diagnostics() const noexcept;
	const std::vector<float> &belief() const noexcept;
	const std::vector<float> &support() const noexcept;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

} // namespace high_vision
