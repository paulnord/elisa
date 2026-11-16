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
