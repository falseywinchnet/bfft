# Blur paper set and synthetic-generator consequences

The PDFs in this directory are public author, institutional, ECVA, or CVF
copies. The list is organized by what must be represented in synthetic blur,
not by model family.

Primary publication pages: [Levin 2009](https://mlanthology.org/cvpr/2009/levin2009cvpr-understanding/),
[Kohler 2012](https://is.mpg.de/ei/publications/kohlerhmshc2012),
[Kim and Lee 2015](https://openaccess.thecvf.com/content_cvpr_2015/html/Kim_Generalized_Video_Deblurring_2015_CVPR_paper.html),
[Portz et al. 2012](https://pages.cs.wisc.edu/~lizhang/projects/blurflow/),
[Wulff and Black 2014](https://ps.is.mpg.de/publications/wulff-eccv-2014),
[Kim et al. 2016](https://arxiv.org/abs/1603.04265),
[Su and Heidrich 2015](https://openaccess.thecvf.com/content_cvpr_2015/html/Su_Rolling_Shutter_Motion_2015_CVPR_paper.html),
[Lai et al. 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Lai_A_Comparative_Study_CVPR_2016_paper.html),
[Nah et al. 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.html),
[Brooks and Barron 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Brooks_Learning_to_Synthesize_Motion_Blur_CVPR_2019_paper.html),
[Lee et al. 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Deep_Defocus_Map_Estimation_Using_Domain_Adaptation_CVPR_2019_paper.html),
[Abuolaim and Brown 2020](https://arxiv.org/abs/2005.00305),
[RealBlur 2020](https://graphics.postech.ac.kr/researches/RealBlur/),
[RSBlur 2022](https://cg.postech.ac.kr/researches/RSBlur/),
[Jaiswal et al. 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Jaiswal_Physics-Driven_Turbulence_Image_Restoration_with_Stochastic_Refinement_ICCV_2023_paper.html),
[Shu et al. 2024](https://openaccess.thecvf.com/content/WACV2024/html/Shu_Deep_Plug-and-Play_Nighttime_Non-Blind_Deblurring_With_Saturated_Pixel_Handling_Schemes_WACV_2024_paper.html),
and [Yang et al. 2025](https://openaccess.thecvf.com/content/ICCV2025W/ECLR/html/Yang_Efficient_Depth-_and_Spatially-Varying_Image_Simulation_for_Defocus_Deblur_ICCVW_2025_paper.html).

The uncertainty extension additionally uses
[Pan et al. 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Pan_Robust_Kernel_Estimation_CVPR_2016_paper.html),
[Jin et al. 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Jin_Noise-Blind_Image_Deblurring_CVPR_2017_paper.html),
[Vasu et al. 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Vasu_Non-Blind_Deblurring_Handling_CVPR_2018_paper.html),
[Nan and Ji 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Nan_Deep_Learning_for_Handling_Kernelmodel_Uncertainty_in_Image_Deconvolution_CVPR_2020_paper.html),
[Nan et al. 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Nan_Variational-EM-Based_Deep_Learning_for_Noise-Blind_Image_Deblurring_CVPR_2020_paper.html),
[Tang et al. 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Tang_Uncertainty-Aware_Unsupervised_Image_Deblurring_With_Deep_Residual_Prior_CVPR_2023_paper.html),
[Sanghvi et al. 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Sanghvi_Structured_Kernel_Estimation_for_Photon-Limited_Deconvolution_CVPR_2023_paper.html),
[Sanghvi et al. 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7611_ECCV_2024_paper.php),
and [Senn et al. 2026](https://arxiv.org/abs/2601.09677).

The shift/mix factorization additionally uses
[Gupta et al. 2010](https://grail.cs.washington.edu/projects/mdf_deblurring/),
[Hirsch et al. 2011](https://webdav.tuebingen.mpg.de/pixel/fast_removal_of_camera_shake/),
[Whyte et al. 2012](https://www.robots.ox.ac.uk/~vgg/publications/2012/Whyte12/),
[Welk et al. 2012](https://arxiv.org/abs/1212.2245),
[Zheng et al. 2013](https://openaccess.thecvf.com/content_iccv_2013/html/Zheng_Forward_Motion_Deblurring_2013_ICCV_paper.html),
[Pan et al. 2019](https://arxiv.org/abs/1811.10185),
[Zhang et al. 2020](https://arxiv.org/abs/2010.02484),
[Dansereau et al. 2016](https://arxiv.org/abs/1606.04308),
[Son et al. 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Son_Single_Image_Defocus_Deblurring_Using_Kernel-Sharing_Parallel_Atrous_Convolutions_ICCV_2021_paper.html),
and [Liu et al. 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Motion-adaptive_Separable_Collaborative_Filters_for_Blind_Motion_Deblurring_CVPR_2024_paper.html).

## Blind blur and benchmark controls

- `levin_2009_understanding_blind_deconvolution.pdf` — naive joint MAP can
  prefer the no-blur explanation; kernel-only inference is better constrained;
  shift-invariant blur is often violated. Consequence: include identity traps,
  kernel-size errors, and non-uniform holdouts.
- `kohler_2012_camera_shake_benchmark.pdf` — recorded six-degree camera motion
  replayed on a robot. Consequence: include real 6-DoF trajectories and compare
  against sampled sharp images, not only a 2-D convolution.
- `lai_2016_comparative_blind_deblurring.pdf` — documents the synthetic/real
  performance gap and missing factors such as depth, saturation, camera
  response, noise, and compression. Consequence: each factor becomes a
  separate generator axis and audit.

## Motion formation

- `nah_2017_gopro_deblurring.pdf` — realistic dynamic blur pairs from averages
  of high-frame-rate video. Consequence: average in linear exposure time and
  include camera plus object motion.
- `kim_2015_dynamic_video_deblurring.pdf` — pixel-wise blur approximated by
  bidirectional optical flow for camera motion, object motion, and depth
  variation. Consequence: the non-uniform generator must transport along a
  field, not convolve every pixel with one PSF.
- `portz_2012_spatially_varying_blur_flow.pdf` — image appearance and blur are
  both functions of motion, so ordinary brightness-constancy flow is biased on
  blurred frames. Consequence: estimate motion through a blur-aware continuous
  formation objective and retain inverse-consistency evidence.
- `wulff_2014_modeling_blurred_video_layers.pdf` — layered flow handles
  distinct object/background motion but segmentation uncertainty at boundaries
  can create restoration artifacts. Consequence: layer ownership must become
  an explicit transported uncertainty state; it must not be a preselected
  reconstruction branch.
- `kim_2016_locally_adaptive_linear_blur.pdf` — bidirectional flow and local
  defocus are estimated with latent frames in one energy because flow between
  blurry observations and independent preprocessing are unreliable.
  Consequence: neighboring-camera motion supplies continuous exposure atoms,
  all pairwise flows enter one cycle-consistent consensus, and the latent is a
  shared state rather than a per-frame winner.
- `brooks_2019_learning_blur_synthesis.pdf` — frame interpolation used to make
  temporally denser motion-blur training data. Consequence: sparse high-speed
  frames must be interpolated before exposure integration when motion between
  frames is appreciable.
- `su_2015_rolling_shutter_motion_deblur.pdf` — each scanline integrates a
  different segment of the camera trajectory. Consequence: rolling shutter is
  a row-time exposure field, not a post-blur warp.
- `rim_2022_realistic_blur_synthesis.pdf` — analyzes the gap between frame
  averaging and real blur, including camera pipeline factors. Consequence:
  radiometric linearization, noise, saturation, ISP, and compression are part
  of the synthesis contract.
- `gupta_2010_motion_density_functions.pdf` and
  `hirsch_2011_fast_nonuniform_camera_shake.pdf` — camera shake is an exposure
  density over poses, hence a weighted sum of transformed sharp images rather
  than an arbitrary per-pixel blur label. Consequence: estimate coordinate
  transport first, then describe residual mixing in transported coordinates.
- `pan_2019_phase_only_kernel_estimation.pdf` — autocorrelation of the absolute
  phase-only image exposes directional motion extent. Consequence: use phase
  geometry as a provisional single-image estimator, never as the inverse
  itself and never as evidence of absolute translation.
- `liu_2024_motion_adaptive_filters.pdf` — motion neighborhoods are aligned to
  their motion middle before collaborative filtering. Consequence: centering
  is an operator factorization, not a cosmetic recentering step.
- `welk_2012_fast_robust_linear_motion.pdf` — a few robust, regularized
  Richardson--Lucy passes can be both fast and stable for known linear motion.
  Consequence: positive inversion is the physical basin; the differentiated
  line equation should remain a bounded complementary constraint.
- `zhang_2020_exposure_trajectory_recovery.pdf` — blur contains a dense law of
  exposed positions, but temporal ordering is partially destroyed by
  integration. Consequence: distinguish recoverable path support/tangents from
  unidentifiable time ordering and handedness.
- `zheng_2013_forward_motion_deblurring.pdf` — forward camera motion produces
  depth-dependent projective scaling, represented by weighted transformed
  latent images rather than one translation PSF. Consequence: the next chart
  must carry a spatial Jacobian and depth/plane discrepancy instead of
  extending the global rotated chart beyond its contract.
- `dansereau_2016_moving_light_field_rl.pdf` — replaces convolution inside
  Richardson--Lucy with the actual motion-blur rendering operator and retains
  a matched backprojection. Consequence: a curved or projective exposure must
  be represented by its forward transport and adjoint, not forced through a
  line-kernel inverse.

## Defocus and optics

- `lee_2019_defocus_map_estimation.pdf` — spatially varying synthetic depth of
  field with ground-truth depth. Consequence: include a per-depth circle of
  confusion and domain-gap controls.
- `abuolaim_2020_dual_pixel_defocus.pdf` — real spatially varying defocus, two
  sub-aperture views, and all-in-focus ground truth. Consequence: dual-pixel
  halves are complementary optical transport families and an important real
  benchmark.
- `yang_2025_spatially_varying_defocus_simulation.pdf` — depth-dependent
  defocus plus field-dependent optical aberration. Consequence: disk/Gaussian
  PSFs are only analytic controls; later generators need location-, depth-,
  wavelength-, and lens-dependent PSFs with occlusion-aware compositing.
- `son_2021_single_image_defocus_deblurring.pdf` — defocus is a spatially
  varying circle-of-confusion field. Consequence: it belongs to centered
  optical mixing; it does not supply a deterministic coordinate path to undo.

## Real data and adverse imaging

- `rim_2020_realblur.pdf` — simultaneously captured, aligned real blurred and
  sharp pairs. Consequence: synthetic success is not a real-image result;
  RealBlur is a required external promotion benchmark.
- `jaiswal_2023_turbulence_restoration.pdf` — turbulence is a stochastic
  combination of geometric distortion and blur. Consequence: turbulence must
  vary through time and space and cannot be reduced to a wider static PSF.
- `shu_2024_nighttime_saturated_deblurring.pdf` — saturation violates the
  linear blur model. Consequence: clipped pixels need a separate observation
  mask/likelihood and must not participate as ordinary transport evidence.

## Blur and noise uncertainty

- `zheng_xue_2024_smoothed_robust_phase_retrieval.pdf` — a smooth robust
  absolute loss is quadratic near credible residuals, linear for arbitrary
  corruptions, and approaches the exact robust objective as bandwidth shrinks.
  Consequence: use it to score blur evidence, but do not transfer the paper's
  random-sensing landscape theorem to convolutional images.
- `gonzalez_2026_zonotopic_mixture_filter.pdf` — finite probabilistic modes
  containing unknown-but-bounded realizations, data-consistency screening,
  probability-preserving enclosure merges, and explicit retained coverage.
  Consequence: keep a law over blur modes and never call an empirical image
  interval a guaranteed zonotope enclosure.
- `pan_2016_robust_kernel_outliers.pdf` — saturation and non-Gaussian outliers
  can produce a false delta kernel. Consequence: structured outliers cannot be
  absorbed into one Gaussian noise scalar.
- `jin_2017_noise_blind_deblurring.pdf` and
  `nan_2020_variational_em_noise_blind.pdf` — deconvolution must adapt to an
  unknown noise level and image-prior uncertainty. Consequence: estimate and
  report sensor-noise scale rather than hiding it in regularization.
- `vasu_2018_kernel_uncertainty.pdf`,
  `nan_2020_kernel_model_uncertainty.pdf`, and
  `tang_2023_uncertainty_aware_deblurring.pdf` — inaccurate kernels induce
  structured image residuals and ordinary non-blind inversion is brittle.
  Consequence: transport multiple operator hypotheses through reconstruction
  and measure their disagreement.
- `sanghvi_2023_photon_limited_deconvolution.pdf` — path keypoints give a
  low-dimensional physical kernel family under strong shot noise. Consequence:
  keep path coordinates as the uncertainty state rather than free PSF pixels.
- `sanghvi_2024_kernel_diffusion.pdf` — sample a conditional kernel law and use
  a non-blind solver instead of alternating one image/kernel mode. Consequence:
  conditional path sampling is a later continuous replacement for the finite
  catalog.
- `senn_2026_bayesian_semiblind_scale.pdf` — Fourier-domain marginal blur
  updates make large semi-blind posterior exploration tractable. Consequence:
  compare the finite transport mixture with marginal kernel sampling rather
  than only deterministic blind baselines.

## Generator ladder

The first executable level contains identity, Gaussian, disk, straight path,
curved path, additive read noise, and Poisson shot noise. The planned levels
are:

1. analytic global PSFs and exact negative controls;
2. continuous 2-D camera trajectories and complementary capture sets;
3. dense optical-flow exposure with moving-object alpha and occlusion order;
4. depth-layer defocus with finite aperture and field aberration;
5. rolling/global-reset shutter timing;
6. Poisson/read noise, saturation, CFA/demosaic, response curve, sharpening,
   and JPEG/codec perturbations; and
7. turbulence warp plus exposure-time-dependent blur.

Every level retains the sharp radiance, exposure samples, path/flow state,
kernel or local covariance, coverage map, noise realization, saturation mask,
and final encoded observation. This prevents a generator from producing a
picture without preserving the state needed to falsify its inverse.
