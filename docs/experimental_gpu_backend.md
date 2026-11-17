# Experimental GPU likelihood pilot

This branch starts a GPU-backend development effort; **it is not a GPU port of
ELISa's stellar model and not a production fitting option**. Existing CPU fitting
interfaces and defaults are unchanged. The optional code is in
`elisa.experimental.gpu_likelihood`.

## What works in this milestone

- ELISa constructs a real circular, synchronous, unspotted detached binary on
  the CPU: Roche geometry, surface temperatures, atmosphere/limb darkening and
  partial eclipse coverage are retained, not replaced by a spherical transit toy.
- A single band's curve is prepared on unique input phases and remapped to the
  original observation order. The ratio of synthesis-support mean to observation
  mean is retained so duplicate observations do not change the l3 semantics.
- Only `system@additional_light` and `nuisance@ln_f` can vary. All other physical
  parameters are frozen. Predictions follow the direct-phase, mean-normalized
  light-curve fitting path. Interpolation, time coordinates and multiple bands
  are not exposed by this pilot.
- An optional JAX FP64 batch likelihood executes on an explicitly requested CPU
  or GPU. GPU unavailability raises an error; there is no silent CPU fallback.
- `make_sampler` uses existing `emcee` ensemble moves with `vectorize=True`.
  The sampler/proposals remain on CPU and each batch transfers parameter vectors
  and returns densities. This is **not** device-resident MCMC.
- Bounds define a uniform prior; Gaussian priors/constraints are not supported.
  Observations and their errors must be supplied in mean-normalized flux units.

## Why this limited first step

It isolates device placement, precision, batched likelihoods, rejection and
sampler plumbing from a new physical engine. It avoids treating a brightness
scale as a substitute for temperature: changing temperature changes atmosphere,
limb-darkening and reflection physics. The likelihood uses ELISa's existing
variance `sigma**2 * (1 + exp(2*ln_f))`, not a model-dependent jitter law.

A cached two-parameter problem this small may run faster on CPU. **No GPU speedup
is claimed.** Comparing it only with full CPU model recomputation would mostly
measure caching, not acceleration.

## Running an experiment (Python)

Install ELISa normally and install JAX according to its official instructions
for the chosen CUDA environment: <https://docs.jax.dev/en/latest/installation.html>.
Do not assume `pip install jax` installs a GPU runtime. Enable FP64 before Python:
`JAX_ENABLE_X64=True` (tcsh: `setenv JAX_ENABLE_X64 True`).

```python
from elisa.experimental.gpu_likelihood import FixedBinaryLikelihood

# binary: an existing ELISa BinarySystem with fixed physical parameters.
# phases, normalized_flux, normalized_sigma: matching single-band arrays.
pilot = FixedBinaryLikelihood.from_binary(
    binary, phases, "Generic.Bessell.V", normalized_flux, normalized_sigma,
    device="gpu", bounds=((0.0, 0.95), (-10.0, 2.0)),
)
# Columns: additional_light, ln_f. Start walkers independently inside bounds.
sampler = pilot.make_sampler(32)
sampler.run_mcmc(initial_positions, 1000)
```

`device="cpu"` explicitly exercises the same JAX code on CPU. The constructor
accepting arrays is a lower-level test API: `support_mean` must be the mean of
the zero-l3 synthesis curve in the *same units* as the mapped baseline. Do not
feed arbitrary max-normalized per-band observation templates with unrelated
normalizations.

## Validation and benchmark requirements

Run `pytest tests/tests_experimental/test_gpu_likelihood.py`. CPU-JAX tests are
not GPU validation. The GPU check skips when hardware is absent and verifies
explicit unavailability behavior. Blackbody/custom-limb-darkening integration
tests use ELISa's real Roche geometry and actual fitter prediction layer without
external table downloads; they do not validate all atmosphere-table models.

For an A100/Colab run, verify `pilot.device`, dtype, CPU-vs-GPU densities and
posterior agreement before timing. Synchronize device results with
`block_until_ready()`. Report separately initial synthesis, compilation, warm
batch evaluation including host transfers, and complete sampling time. Compare
identical cached CPU and GPU likelihoods at multiple batch sizes, plus the
original full CPU workflow as a separately labelled baseline. Report independent
samples/sec and convergence, not only model calls/sec. Colab checkpointing and
production resume are not implemented in this pilot; use existing emcee storage
facilities explicitly for longer experiments.

## Next substantive milestone

Build a prepared-surface interface from ELISa's existing geometry and atmosphere
code and port flux integration in FP64, retaining per-face arrays and explicit
visibility/partial coverage. First validate *changing* physical systems against
CPU at eclipse contacts and ingress/egress. Then port Roche surface generation,
normal/gravity/temperature calculations and occultation to fixed-shape batched
GPU operations. Do not freeze those quantities when physical MCMC parameters
change. Compare mesh-resolution convergence separately from CPU/GPU parity.
Only after this should a general GPU model backend be offered through
`LCBinaryAnalyticsTask`; use explicit supported-model checks and no silent
physical approximations. Moving sampler state and transitions onto device is a
later, independently validated milestone.

## Initial validation record (2026-11-16)

Base: upstream `dev` at `8491de5fd18798340be109a1bf591ed7aef94aaa`.
Python 3.12, JAX/jaxlib 0.12.0.1, NumPy 2.5.5, emcee 3.1.6;
only a CPU device was available. Focused tests compare pointwise predictions
and Gaussian log likelihood with the real ELISa Observer through the fitter's
mapping/normalization layer, including repeated/cross-cycle phases, nonzero
phase shift and l3 near its boundary. The test injects real Observer synthesis
rather than exercising the fitter's full parameter serializer.

A 128-phase blackbody Roche smoke run (32 walkers, 100 steps) had maximum
absolute log-likelihood difference about 2.55e-11 from the existing CPU cost.
This is a correctness smoke test, not a converged scientific fit or GPU speed
measurement. The benchmark prints setup, compilation and timing separately.
Run it with `python scripts/benchmark_gpu_likelihood.py --device gpu` on the
actual GPU. The existing prediction-consistency regression module was collected
but skipped because this environment lacks its external scientific tables;
these skips are not passes.

## Second milestone: actual surface integration kernel

`elisa.experimental.gpu_surface.SurfaceFluxIntegrator` now evaluates the actual
ELISa per-face flux expression on an explicitly selected JAX device. It supports
linear (including cosine alias), logarithmic and square-root limb darkening,
heterogeneous padded batches, and empty selections and fully occulted zero-coverage faces. Prepared
containers retain CPU-generated Roche geometry, eclipse coverage, radiance and
LD coefficients. Coverage already includes surface area and semi-major-axis
scaling; neither it nor optical-depth normalization is applied a second time.
The kernel is standalone and is not yet a production Observer replacement.
See `docs/ada_gpu_handoff.md` for isolated, non-disruptive hardware validation.

## Next physics milestones and gates

1. Measure per-face throughput with `scripts/benchmark_gpu_surface.py`, including
   CPU preparation and transfers as separately reported costs. Establish whether
   any meaningful batch size amortizes launch/transfer; report negative results.
2. Port observer-frame normal projection for the static, circular synchronous
   mesh. Use ELISa's `rotate_item`/`move_sys_onpos` conventions and face-indexed
   comparison, not an assumed phase rotation. Preserve CPU selected indices at
   first to avoid near-limb mask changes; test both stars, inclination extremes,
   phases near eclipses, wrap/shift, and cancellation-sensitive near-zero cosines.
   Changing inclination also changes occultation: frozen coverage cannot be
   treated as a valid inclination proposal.
3. Port eclipse clipping/covered-area calculation, using the actual triangulated
   Roche faces and silhouette intersections. Preserve face identity and check
   contacts, partial/full occultation, and mesh-resolution sensitivity. GPU
   projection with CPU coverage alone is not a GPU physical forward model.
4. Only then introduce batched varying physical parameters, including geometry,
   temperature/atmosphere/LD/reflection updates as required. Reject unsupported
   dependencies. Recompute affected physics rather than silently reuse a fixed
   template. Validate full normalized model and accumulated log-posterior error
   against CPU before sampler integration.
5. Device-resident sampler and performance acceptance come after model parity;
   compare wall time and effective samples/second for identical physical models,
   priors and data, with compilation/preparation included in end-to-end timing.

The prepared surface batch API holds immutable device arrays and allows repeated
kernel evaluation without transfers. This is a throughput primitive, not a claim
that a physically varying MCMC proposal can reuse all those arrays unchanged.

## Changing-geometry milestone: bounded Roche ray solver

`elisa.experimental.gpu_roche.RocheRaySolver` now solves ELISa's actual radial
modified Kopal potential on device for batches of changing **mass ratio,
synchronicity and surface potential**, for either component. This is no longer
just arithmetic on a frozen flux template. It remains a primitive, not a full
GPU forward model or MCMC backend.

- Circular separation is fixed at 1 SMA; q=M2/M1 in [0.5,2], spin in [0.5,1.5].
- Supplied phi is companion-facing longitude, theta is north-pole colatitude.
  Each proposal uses the same supplied directions; these are NOT ELISa's full
  candidate-dependent angular grid/topology/mesh-correction algorithm.
- Each inner ray is bracketed in [1e-6,0.25] SMA. A conservative bound on the
  radial derivative is strictly negative throughout this domain (worst case
  secondary derivative upper bound is below -4.37), certifying uniqueness only
  for accepted supplied rays. This does not certify morphology/closed mesh.
- Eighty device bisections produce radii, local Cartesian points, potential
  residuals, final interval widths, per-ray status and proposal_valid. Finite
  precision, not the nominal bisection count, limits attained accuracy.
- Status 1 means root outside this primitive's domain; 2 means endpoint bracket
  ambiguity within 32*machine-epsilon*max(1,target); 3 means residual failure.
  These are unsupported evaluations, not physically forbidden binaries. A
  future sampler must fall back to CPU or fail explicitly rather than silently
  truncate its posterior. No HMC/autodiff gradients are claimed.
- Both stars' pointwise potential/radii are compared with ELISa's CPU equations,
  `radius.calculate_radius` first-inner-root solver and actual
  `surface.mesh.get_surface_points` Newton solver/Cartesian mapping. CPU Brent
  is an additional independent root method. Tests cover q/spin extremes,
  poles/axes and endpoint ambiguity, and are parameterized for real GPU runs.

Remaining before physical fitting: candidate-dependent angular placement and
triangulation/mesh correction, full surface normals/gravity/temperature/LD,
observer projection and changing eclipse coverage, then full model/likelihood
parity. Do not pair changed radii with a frozen brightness or coverage template.

## Triangle geometry and observer projection

`elisa.experimental.gpu_mesh.TriangleGeometry` computes face centres, triangular
areas, outward normals and observer-frame normals/cosines for batched changing
vertices with supplied triangle connectivity. It follows ELISa's co-rotating
binary-frame conventions and `rotate_item` z then inverse-y rotation, with LOS
(-1,0,0). Both components are checked against an actual built ELISa container and
`move_sys_onpos`, across eclipse phases, quadratures and repeated phases.
A chained test feeds changing Roche-ray vertices through fixed connectivity and
compares against ELISa's CPU mesh helper plus face/rotation routines.

Connectivity is provided, not re-triangulated. This controlled fixed-direction
experiment does not replace candidate-dependent mesh generation/correction.
Degenerate or nonfinite face geometry is explicitly unsupported. Facing uses
strict cosine>0; a 64-epsilon near-limb ambiguity flag identifies cases where
floating-point ordering can change selection. Do not apply old CPU eclipse
coverage to newly moved normals/vertices. In particular, varying inclination
changes occultation as well as projection. No occulted areas, brightness,
light curves, likelihoods or sampler results are produced by this layer.

Inputs are host-validated snapshots and transferred to the selected device;
the current ray-to-triangle test includes a host bridge. A fused resident
geometry/coverage/thermal path remains future work. Geometric parity does not
establish mesh-resolution convergence or physical-model accuracy.

Important CPU build-order distinction: ELISa can cache `star.face_centres`
before mesh correction, then scale its points while retaining those centres.
This primitive's centres describe the supplied vertices, so they are not
necessarily identical to that cached field. Actual-container parity tests cover
normals/areas/projection and explicitly expose this centroid distinction. Before
downstream integration, preserve CPU's pre-correction centres/normals, corrected
points/areas and build order; do not silently substitute recomputed centres.

Triangle geometry also rejects uncertain outward orientation, triangles with
sin(edge angle)<=0.05, and coordinates exceeding 1e4 times the shortest defining
edge. These conservative conditioning gates preserve the tested Roche meshes
but are not a universal error certificate. All flagged geometry requires CPU
fallback or explicit failure, not an automatic physical-prior rejection.

## Effective gravity on changing Roche vertices

`elisa.experimental.gpu_gravity.RocheSurfaceGravity` adds the next thermal-model
dependency: ELISa's outward potential gradients and effective acceleration in
SI, evaluated on supplied changing binary-frame vertices for either component.
It uses the same circular q/spin domain and inner-radius interval as the ray
primitive. It does not certify that supplied points form an equipotential or
closed mesh. Unsupported/nonfinite vertices and numerical scaling failures
have explicit status and validity arrays; invalid outputs are NaN, not zero
flux or an implicit physical-prior rejection.

Inputs are primary mass in kg and semi-major axis in m. Both stars use
`G * primary_mass / semi_major_axis**2` as the potential-gradient scale.
The secondary's mass ratio is already present in its dimensionless gradient;
adding another q factor would be incorrect. Face gravity averages the three
corner acceleration magnitudes, and internal `log_g` is log10 of SI gravity.

Build order matters: CPU ELISa evaluates gravity before `correct_mesh`, so
matching cached `star.potential_gradient_magnitudes` requires those same
pre-correction vertices. The container test builds and checks that exact stage.
Other controls compare both stars across changing Roche q/spin/potential,
different masses/length scales, an independent complex-step derivative of the
scalar potential, polar forces in physical units, and unsupported cases.

This primitive does not update temperatures, reflection, atmosphere/LD,
topology or changing eclipse coverage. Do not combine its changed gravity with
old thermal or coverage state and label the result a physical light curve.
Host snapshots and device transfers are explicit; no fused resident pipeline,
sampler integration, autodiff guarantee or performance gain is claimed.

## Device-resident Roche geometry/gravity pipeline

`elisa.experimental.gpu_pipeline.RocheGeometryPipeline` now composes the
existing ray, triangle/projection and gravity device kernels under one outer
JAX compilation. Directions/connectivity are copied once; a prepared batch
holds immutable q/spin/potential, primary-mass/SMA and orientation arrays on
the selected device. `PreparedRocheBatch.evaluate()` performs no intermediate
NumPy conversion or host round trip. Both stars use the existing binary-frame
placement and SI gravity conventions, with the same unsupported-state masks.
The combined `proposal_valid` is the conjunction of all three physics stages.

```python
from elisa.experimental.gpu_pipeline import RocheGeometryPipeline

pipeline = RocheGeometryPipeline("primary", phi, theta, faces, device="gpu")
batch = pipeline.prepare(q, spin, potential, primary_mass_kg, sma_m, azimuth, inclination)
result = batch.evaluate()  # nested device arrays: rays, geometry, gravity, validity
```

Topology remains supplied and fixed. The base call produces geometry/gravity;
optional fresh temperatures and passband emission are described below.
No mesh correction, eclipse clipping or reflection is applied. The pipeline intentionally
does not attach stale brightness or coverage to changing geometry. Areas are
in SMA squared; gravity is in m/s squared. These ray-mesh quantities are not the
final corrected/cached state of a CPU ELISa container. Invalid/unsupported
flags require explicit failure or CPU fallback, not a silent physical prior.

`scripts/benchmark_gpu_pipeline.py` compares original ELISa Newton/NumPy helpers,
the separate JAX stages with a vertex host bridge, and the resident composition.
Run on JAX CPU as well as GPU to distinguish batching from GPU acceleration.
All full paths return the same radii, areas, normals, line-of-sight cosines,
face gravity and validity arrays. Cold setup/compilation, initial parameter
upload, resident evaluation and fresh-result downloads are reported separately.
Download-only samples use fresh results to avoid timing cached host arrays.
The strict `disallow_explicit` transfer guard checks repeated resident work.
Performance on these fixed-direction changing proposals is not whole-model,
MCMC or effective-sample performance. The l3/ln_f-only pilot should continue to
cache its fixed physical curve rather than repeatedly synthesize it.

## Intrinsic gravity-darkened temperatures

`RocheGeometryPipeline.prepare_thermal` accepts the same seven physical arrays,
then intrinsic `t_eff` and gravity-darkening exponent `beta` (in `[0,1]`). Its
device result adds `thermal.temperatures` (K), `thermal.surface_flux` (emitted
bolometric W/m²), a luminosity closure ratio and unsupported-state masks.
`gpu_thermal.GravityDarkeningTemperature` also exposes this primitive separately.

The unspotted CPU law is

```
T_face = T_eff * (sum(A)/sum(A*g**beta))**0.25 * g**(beta/4)
```

The CPU polar-reference gravity cancels algebraically. The implementation scales
area/gravity by row maxima for numerical stability; it preserves the formula.
All supplied faces must have finite positive gravity and area. Invalid faces
invalidate stellar normalization explicitly. No implicit zero emission or
physical-prior rejection is substituted. Tests cover real CPU containers with
reflection disabled, the original CPU temperature function, scaling invariance,
beta zero, and independent energy closure using the returned temperatures.

Pipeline temperatures refer to the supplied **uncorrected ray mesh**. Production
ELISa's corrected areas/build-stage state still require a separate mesh-correction
implementation and validation. There is no irradiation, reflection, spot model,
atmosphere-table interpolation or changing limb-darkening state in this law.
Use `--thermal` on the bounded pipeline benchmark for its timings and CPU oracle.

## Blackbody passband surface emission

`gpu_blackbody.BlackbodyPassbandFlux` computes continuous-temperature Planck
quadrature on ELISa's actual Bessell/bolometric (or supplied) passband grids.
Construction snapshots each fixed response, reproduces the CPU standard/uniform
wavelength union and Simpson endpoint convention, and uploads fixed quadrature
weights once. Negative response boundary overshoot is clipped as in the CPU.
There is no whole-kelvin temperature rounding or interpolation table.

The device streams wavelength chunks (default 32), avoiding a full
`proposals*faces*all_wavelengths` temporary. Bound proposal/face counts explicitly;
this algorithm does not impose an allocator peak bound. Up to eight passbands,
4096 samples per band and 16384 combined samples are supported. Unsupported
configurations fail explicitly. Invalid temperatures/numeric fluxes produce NaN
and masks, never a valid zero-brightness proposal.

```python
from elisa import Observer
from elisa.experimental.gpu_blackbody import RocheBlackbodyPipeline

bands = Observer(passband=["Generic.Bessell.B", "Generic.Bessell.V", "bolometric"]).passband
emission = RocheBlackbodyPipeline(pipeline, bands)
batch = emission.prepare(q, spin, potential, primary_mass_kg, sma_m,
                         azimuth, inclination, t_eff, beta)
result = batch.evaluate()  # all changing physics remains on the selected device
band_flux = result["blackbody"]["surface_flux"]  # proposals, bands, faces; W/m²
```

These are hemispheric **emitted surface fluxes**, including the CPU's pi and
Angstrom-to-meter conversion. They precede optical-depth/limb correction,
projected-area integration, occultation and observer-distance scaling. The finite
bolometric grid retains the existing CPU quadrature/truncation error; it is not
replaced by sigma*T⁴. This does not support atmosphere tables or full light curves.
Tests compare with `NaiveInterpolatedAtm.black_body_radiance`, retain tiny
temperature changes, check the independent Planck energy integral, and use strict
GPU transfer guards over the full resident Roche-to-emission chain.

Run `scripts/benchmark_gpu_pipeline.py --device cpu --blackbody --directions 64
--rows 1 8 32 128 --repeats 5` and the corresponding GPU command in the bounded
authorized window. Larger spectral grids/reference batches can consume substantial
CPU memory; defaults for the geometry benchmark need not suit spectral workloads.
See [Ada progress](ada_progress.md) for measured results and tested fingerprints.

## Corrected state and functional hybrid observed curves

`gpu_state.CorrectedSurfaceState` now reconstructs ELISa's actual cached build
state from each candidate's CPU-generated, uncorrected mesh and topology. It
retains pre-correction centres/normals/gravity, corrects only points and areas,
and calculates fresh intrinsic temperatures with quarter-face symmetry and
energy normalization. The correction factor comes from the current CPU detached
resource, with the same interpolation and endpoint convention. Prepared state is
immutable and evaluated under strict device transfer guards. It is a separate
adapter; `RocheGeometryPipeline` retains its original uncorrected contract.

`gpu_binary.HybridBinaryLightCurve` connects that state to complete observed
curves with fresh CPU reflection/custom LD and device blackbody/optical-depth
radiance. Its current default uses device
projection/clipping/coverage/integration with a fresh CPU hull/visibility bridge;
`coverage="cpu"` retains the preceding CPU eclipse path. Each candidate's angular
placement, radial roots and connectivity are recomputed on CPU; the ray
experiment's directions are not frozen.
Each call rebuilds all dependent physical state. This is a functional **hybrid**
path, not an all-GPU or resident forward model and not a registered fitting backend.

```python
from elisa.experimental.gpu_binary import HybridBinaryLightCurve

engine = HybridBinaryLightCurve(
    ["Generic.Bessell.B", "Generic.Bessell.V", "bolometric"],
    device="gpu", fallback="cpu",
)
result = engine(binary, phases, normalize=True)
print(result.backend, result.fallback_reason, result.timings)
curves = result.fluxes
```

Output reproduces Observer's maximum normalization, phase shift, periodic
reduction/symmetry, original ordering/repeats and third-light mean. With
`normalize=False`, flux is W/m² using the original distance helper. Component
fluxes precede third light and distance/normalization. This explicit-phase API
does not expose times, magnitudes, interpolation or fitter mean normalization.
`diagnostics=True` retains intermediate containers in memory for validation;
it should be off for timing/inference. Fixed integration padding keeps the
compiled shape stable as visible-face counts change.

Hybrid support is circular detached, q in [0.5,2], spin in [0.5,1.5], raw radii
[1e-6,.25] SMA, beta in [0,1], unspotted/nonpulsating, blackbody and supplied
per-band/bolometric LD. CPU reflection follows current settings. Existing geometry
conditioning masks remain active. Outside this domain, default fallback runs
the **entire unmodified CPU Observer** and reports the reason; `fallback="raise"`
instead raises `UnsupportedHybridModelError`. Device unavailability/disabled FP64
never silently selects CPU. Unsupported states are not physical-prior penalties.
Atmosphere tables/tabulated LD, spots and eccentric/contact GPU states remain
unvalidated; CPU fallback preserves upstream data errors and exceptions.

Run `tests/tests_experimental/test_gpu_binary.py` and
`scripts/validate_gpu_binary.py --device cpu --repeats 3 --output validation.json`,
then the hardware variant in an authorized bounded restored window. Validation
compares fully built CPU containers, per-face/component/total flux, both eclipses,
contacts, three LD laws, reflection, changing parameters/topology and original
phase mapping. The first Ada milestone passed 163 tests with no skips plus seven
full-curve comparisons. Full hybrid GPU calls were slower than CPU Observer at
the tested 832/1280-face workloads. See [Ada progress](ada_progress.md) for actual
errors, transfer-inclusive samples, safety evidence and remaining work. No MCMC,
ESS/sec, resolution convergence or whole-model acceleration claim follows.

## Device projection and partial-face eclipse clipping

`gpu_occultation.DeviceEclipseIntegrator` projects each new corrected mesh and
its cached pre-correction normals on device. Projected arrays return to CPU for
the original Qhull/Path and visibility selectors. Fresh hulls/selection masks
then upload for `gpu_clipping.ConvexTriangleClipper`, uncovered-area calculation,
the original surface-noise threshold, SMA² scaling and batched band integration.
CPU candidate geometry, reflection/custom LD and hull construction remain
explicit costs. This is still a hybrid model, with no stale candidate state.

The device clipper follows the original Sutherland–Hodgman algorithm: three
triangle half-planes, triangle-local coordinates, ordered crossing/end vertices,
and translated shoelace area. It preserves exact-zero contacts and positive
slivers, without quantization or widened boundaries. Immutable prepared packets
have no intermediate evaluation transfers; host preparation and final downloads
remain caller costs. Degenerate/empty intersection semantics match the CPU;
numeric/buffer failures are explicit unsupported results, never prior penalties.

The default hybrid supports the configured Sutherland–Hodgman or reference
intersection backend. PYPEX uses full CPU fallback, preserving its distinct
numeric contract. Device storage bounds are 128 synthesis phases after Observer
reduction/symmetry, 4096 vertices/8192 faces per star, 128 hull vertices and 256
partial faces per component/phase; clipping packets contain eight phases. Larger
requests use the entire CPU curve or raise explicitly, without truncation.

The second background milestone passed **184 tests with no hardware skips** and
eight detailed CPU/GPU full-curve cases. Direct analytic geometry and 60 saved
Roche eclipse batches supplement original-CPU parity. Matched full GPU curves
took 66–102 ms versus 112–155 ms for the preceding hybrid and 66–103 ms for CPU
Observer. These small CPU/GPU differences do not establish a robust whole-curve
gain; the CPU hull bridge remains. See the latest [report](ada_progress.md) for
all errors, cold/full costs, unchanged thresholds, restoration and remaining
physical fitting/MCMC assessment. No sampler/convergence/ESS claim is made.

## Original physical fitting and likelihood integration

`gpu_fit.HybridBinarySynthesis` implements the existing synthesis-hook signature
using the original `lc.prepare_binary` serializer and a fresh hybrid curve for
every physical proposal. It preserves the supplied Observer's actual response
tables/bandwidths and resulting raw phase/flux state. The unchanged fitter
prediction layer resolves JD/phase coordinates and support, maps/interpolates
onto each band's observation order, then **mean** normalizes. Observer maximum
normalization on synthesis support is an earlier, distinct operation. Repeated
observations and the synthesis-support third-light convention are retained.

`HybridDetachedLeastSquaresFit` and `HybridDetachedMCMCFit` are opt-in subclasses
that replace only that hook. The existing constraint resolver, normalized
parameter coordinates, residuals, `sigma**2*(1+exp(2*ln_f))` noise model, priors,
failure handling, optimizer/emcee lifecycle and result serialization remain CPU
implementations. Unsupported device model/backend/resource states use the
**whole original CPU curve**, preserving posterior support; missing FP64/device
is a configuration error. CPU default fitters and task/backend selection are
unchanged. Experimental fitting is serial only; sampler proposals remain on CPU.

```python
from elisa import settings
from elisa.experimental.gpu_fit import HybridDetachedLeastSquaresFit

# data and x0 use the existing LCData/BinaryInitialParameters APIs.
# GPU code must run inside the independently restored, bounded allocation.
with settings.override(NUMBER_OF_PROCESSES=1, NUMBER_OF_MCMC_PROCESSES=1):
    fit = HybridDetachedLeastSquaresFit(device="gpu")
    fit.atmosphere_model = {"primary": "bb", "secondary": "bb"}
    fit.limb_darkening_coefficients = supplied_per_component_band_and_bolometric_ld
    result = fit.fit(data, x0, discretization=12, interp_treshold=1000,
                     diff_step=1e-4, max_nfev=16)
```

`scripts/validate_gpu_fit.py --device cpu --output fit-cpu.json` and the GPU
variant exercise actual serializer/mapping/costs, public LSQ fits and seeded
short public MCMC runs. The Ada milestone passed 203 tests without hardware
skips. GPU mean-normalized prediction error was at most 4.44e-16 and accumulated
log-posterior error 1.22e-8 for the tested 1e-4 uncertainties. Short chain coordinates
and acceptance fractions matched exactly; **three steps do not establish
convergence, posterior precision or ESS**. Deliberately coarse interpolation
fixtures have large original CPU discretization errors, reported separately.

Matched warm physical likelihood callbacks took 55.20 ms for hybrid GPU versus
50.45 ms for original CPU and 133.45 ms for hybrid JAX CPU. New-fitter public LSQ and
short sampler runs were slower on GPU after including setup/compilation and all
calls. There is no production fitting-speedup claim. Read [Ada progress](ada_progress.md)
for the exact domain, cold/full costs, thresholds, raw-synthesis versus mean
normalization distinction, restoration and remaining GPU physics work. The
l3/ln_f-only pilot should keep caching its fixed physical curve.

## Fresh device radiance in complete curves and physical fitting

`gpu_radiance.DeviceBlackbodyRadiance` combines the existing continuous Planck
quadrature with base-patch face mirroring and the same passband's integrated
limb-darkening factor. Hemispheric surface flux divided by that factor (including
pi) gives normal intensity. Directional LD, projected coverage, third light,
distance and curve/fitter normalization remain downstream, each applied once.
Independent angular quadrature checks this energy convention for all LD laws.

`HybridBinaryLightCurve` and the opt-in physical fitters now default to
`radiance="device"`; `radiance="cpu"` selects the preceding CPU spectral path
for comparison. Corrected device temperatures feed emission without re-upload
when reflection is off. When CPU reflection changes temperatures, its final
arrays upload afresh. Device radiance/LD arrays then pass directly to eclipse
packets; no normal-radiance host round trip occurs unless diagnostics or CPU
coverage is requested. CPU state downloads, reflection, mesh/topology/kinematics
and the fresh CPU hull/visibility bridge remain explicit hybrid costs.

Only immutable quadrature/code is reused. Every curve re-samples actual Observer
response functions on the CPU grids and checks a content revision, including
changes made without renaming a band or changing its table object. Fresh
temperature/LD/brightness/coverage is never cached across physical proposals.
Prepared emission snapshots support strict no-transfer repeated evaluation.

Device spectral bounds are 1–8 bands, 3–4096 samples per band, at most 16384
combined samples and 8192 faces per star. Outside the validated model or resource
domain, and for unsupported numeric output, the whole original CPU curve runs;
these are not prior exclusions. Missing device/FP64 remains a runtime error.
Real atmosphere/LD tables, spots and eccentric/contact GPU states remain outside
this declared domain.

The fourth background milestone passed 215 tests with zero hardware skips,
eight detailed full-curve cases on matched JAX CPU/GPU profiles, and the original
physical fitting assessment. Normal intensity agrees within 3.41e-15 relative;
mean-normalized fitter predictions within 6.67e-16 absolute. Actual bounded
least-squares fits agree and short sampler coordinates/acceptances replay exactly.
No previous tolerance was relaxed.

Matched warm full curves took 20–37 ms versus 65–96 ms for the preceding hybrid
and 67–99 ms for original CPU. Changed-parameter likelihood/prior callbacks took
23.20 ms versus 52.43 ms for the preceding hybrid, 47.69 ms for original CPU and
103.92 ms for hybrid JAX CPU. All CPU bridges, response checks and transfers are
included; compilation/setup are separate. Full bounded new-fitter LSQ and
three-step sampler runs remain slower than original CPU after compilation. These
small workloads and short runs do not establish general throughput, convergence
or ESS/sec. See [Ada progress](ada_progress.md) and its curated evidence for cold
costs, actual samples, domain, independent restoration and remaining work.
