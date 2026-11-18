# Experimental GPU likelihood pilot

## m037 — Projected primary heating at proposal 0

The calculation compares three primary spectra on the same eclipse quadrature points. Both reflection sweeps run to completion before the code constructs the intrinsic, face-mean and fully reflected spectra. Each uses the same gravity-darkened primary and the same fully reflected secondary. The supported case is the circular detached blackbody binary with linear limb darkening in the fixture. The physical likelihood uses the fully reflected spectrum and the existing resolution check. The other spectra measure sensitivity to deposited heat.

Reproduce the serial CPU comparison in the FP64 validation environment from the repository root. Each block covers nine input phase slots. The join checks their order and applies the original observation mapping, third light, normalization and fitted noise.

```bash
fixture=docs/ada_validation/background-m033-proposal-0.json
out=$(mktemp -d)
for rule in six duffy; do
  options=()
  if [[ "$rule" == six ]]; then options+=(--six-point); fi
  for depth in 3 4; do
    for part in 0 1; do
      python -m scripts.measure_primary_heating "$fixture" --backend reference \
        --depth "$depth" --part "$part" "${options[@]}" --output "$out/$rule-$depth-$part.json"
    done
    python -m scripts.compare_primary_heating "$fixture" "$out/$rule-$depth-0.json" \
      "$out/$rule-$depth-1.json" "${options[@]}" --output "$out/$rule-$depth.json"
  done
done
python -m scripts.check_interior_resolution --six "$out/six-3.json" "$out/six-4.json" \
  --alternative "$out/duffy-3.json" "$out/duffy-4.json" --alternative-rule duffy \
  --curved-heating --output "$out/resolution.json"
python -m scripts.compare_common_parent_heat "$fixture" --output "$out/common-parents.json"
python -m pytest -q tests/tests_experimental/test_projected_heating.py
```

For JAX CPU use `--backend cpu`. On Ada, each GPU block runs inside its own registered, checkpoint-verified unit with `--backend gpu --verified-bounded-gpu-window`. The block join and common-parent integration run on CPU. The latter uses the same coarse parents for both thermal meshes and weights the projected heat by visible area. This gives a heat difference in units of the primary effective temperature to the fourth power. The optical comparison above uses each mesh’s own faces and gives changes in observed passband flux and nominal log likelihood. [Results](ada_validation/background-m037.json)

## m035 — Quadratic deposited heat on the primary

Select `CurvedHeatingLightCurve` for the circular detached blackbody and linear limb-darkening configuration. It uses the existing irradiation interaction cap, positive optical controls and curved-area energy check. The new field requires a nonsingular angular chart and three paired neighbors on every full-surface triangle. Ill-conditioned stencils raise the experimental unsupported-geometry error. Each of the primary’s ten spectral nodes receives its own reconstructed deposited heat. The secondary retains the sampled six-node calculation.

Use the m034 complete-curve commands below with `--curved-heating` in place of `--primary-cubic` in the depth-three runs, the depth-four blocks and joins, and the sixteen-model comparison. Run `python scripts/check_receiver_curvature.py docs/ada_validation/background-m033-proposal-0.json --output contact.json` for the direct CPU contact rays and receiving-face mean checks. The pinned m031 sampler uses its own checkpoint and source fingerprint through draw five. [Results](ada_validation/background-m035.json) · [Tested source fingerprints](ada_validation/background-m035-source-hashes.json)

## m034 — Cubic primary temperature and optical moments

`PrimaryCubicLightCurve` uses ten nodes on each primary face and the established six-node secondary. It supports the same detached reflected blackbody and linear limb-darkening scope as the sampled-heating comparison. It enforces the primary symmetry, root, cutoff, positivity, curved-area and per-face T⁴ conservation checks. Checks on the cubic Bernstein coefficients bound temperature and intensity between interpolation nodes. Optical batches use at most three nodes per face per star. The original CPU observer and the default experimental quadratic field retain their existing entry points.

Run the following on the pinned fixture in the validated FP64 environment. On Ada, each GPU forward evaluation requires its own registered, checkpoint-verified unit. `reference` selects the serial CPU and `cpu` selects JAX CPU.

```bash
fixture=docs/ada_validation/background-m033-proposal-0.json
out=$(mktemp -d)
for rule in six duffy; do
  options=(--primary-cubic)
  if [[ "$rule" == six ]]; then options+=(--symmetric-interior 4); fi
  python scripts/validate_receiver_heating.py evaluate "$fixture" --backend reference --depth 3 --selective-shadow "${options[@]}" --output "$out/$rule-3.json"
  for part in 0 1; do
    python scripts/receiver_phase_parts.py evaluate "$fixture" --backend reference --depth 4 --part "$part" "${options[@]}" --output "$out/$rule-part-$part.json"
  done
  python scripts/receiver_phase_parts.py join "$fixture" "$out/$rule-part-0.json" "$out/$rule-part-1.json" "${options[@]}" --output "$out/$rule-4.json"
done
python scripts/check_interior_resolution.py --primary-cubic --six "$out/six-3.json" "$out/six-4.json" --alternative "$out/duffy-3.json" "$out/duffy-4.json" --alternative-rule duffy --output "$out/combined.json"
for level in 2 3; do
  python scripts/check_primary_spectrum.py "$fixture" --level "$level" --output "$out/spectrum-$level.json"
done
python scripts/check_primary_spectrum.py "$fixture" --shared-levels --output "$out/shared.json"
```

To reproduce a same-packet comparison, run `python scripts/check_primary_curve_pair.py "$fixture" --backend reference --depth 3 --interior duffy --output "$out/pair.json"`. Choose `six` for the other interior rule. At depth four, run both phase blocks with `--part 0` and `--part 1`.

The sixteen-model check withholds the usable density on either accuracy limit. The chain remains bound to its own target and verifies its checkpoint and source identity. [Source hashes](ada_validation/background-m034-source-hashes.json) · [Results](ada_validation/background-m034.json)

## m033 — Positive symmetric triangle cubature in reflected eclipse curves

`SymmetricInteriorRule` retains the curved limb/shadow slices, topology checks, singular fallback and original per-phase node cap. The six- and seven-point formulas integrate planar barycentric polynomials through degrees four and five. The symmetric rules subdivide planar panels with a unit-direction chord longer than 1/8 and charge every child node against the cap. The stellar integrand also contains the solid-angle Jacobian, Roche geometry and radiance. Its finite resolution is measured from complete curves. The default curved-edge rule still uses the Duffy nodes. Rule metadata distinguishes the optical integration from the three quadrature nodes on each irradiated face.

Run from the tested checkout with the private ELISa configuration or the matching test data, FP64 enabled and single numerical-library threads. Specify `--backend reference` for serial CPU or `--backend cpu` for JAX CPU. GPU evaluation on Ada uses `--backend gpu --verified-bounded-gpu-window` inside registered held-checkpoint units. For each proposal the following shell commands write three-band results in a fresh directory. Use separate bounded GPU units for each forward command.

```bash
out=$(mktemp -d)
fixture=docs/ada_validation/background-m033-proposal-38.json
for degree in 4 5; do
  python scripts/validate_receiver_heating.py evaluate "$fixture" --backend reference --depth 3 --selective-shadow --symmetric-interior "$degree" --output "$out/d3-$degree.json"
  for part in 0 1; do
    python scripts/receiver_phase_parts.py evaluate "$fixture" --backend reference --depth 4 --part "$part" --symmetric-interior "$degree" --output "$out/part-$degree-$part.json"
  done
  python scripts/receiver_phase_parts.py join "$fixture" "$out/part-$degree-0.json" "$out/part-$degree-1.json" --symmetric-interior "$degree" --output "$out/d4-$degree.json"
done
python scripts/check_interior_resolution.py --six "$out/d3-4.json" "$out/d4-4.json" --alternative "$out/d3-5.json" "$out/d4-5.json" --alternative-rule seven --output "$out/combined.json"
python scripts/compare_irradiating_sources.py docs/ada_validation/background-m033-proposal-0.json --output "$out/illumination.json"
```

For proposal 0 use its fixture with the six-point calculation. Reproduce the independent Duffy calculation by omitting `--symmetric-interior` on its evaluate and join commands, then choose `--alternative-rule duffy`. `usable_log_posterior` is null on an accuracy stop. Operational caps and unsupported physics continue to require an explicit stop or complete original CPU model. [Proposal 0](ada_validation/background-m033-proposal-0.json) · [Proposal 38](ada_validation/background-m033-proposal-38.json) · [Validation source hashes](ada_validation/background-m033-source-hashes.json)

## m032 — Original-facet ensemble with a shared GPU

`SharedNativeDetachedMCMCFit` accepts a service created by `shared_surface_owner`. Original CPU walker processes send their fresh selected face arrays to the owner for ordered FP64 reductions. The benchmark covers the reflected, circular detached binary with linear limb darkening. The opt-in fitter requires candidate exceptions to propagate under the original policy. Unsupported integration can be rerun with the complete native CPU fit. [Measured scope, scientific limits and reproduction](ada_mcmc_scaling.md)

## m031 — Sampled receiving-face irradiation

`ReceiverHeatingLightCurve` resolves the last deposited power at three receiver rays with an area-conserving, positivity-limited linear field. The optical source uses the six-node quadratic reconstruction. The `selective_shadow` option refines parents crossing the foreground shadow. `receiver_phase_parts.py` can evaluate depth-4 phase blocks in bounded units and restores the complete grid before applying third light and normalization. [Scientific scope and reproduction](ada_progress.md)

## m030 — Quadratic source fields and shadow-crossing checks

`QuadraticSourceLightCurve` evaluates six physical blackbody and linear-LD samples per original face. It preserves the area mean of its irradiated T⁴ field and rejects a spectral polynomial that fails the positivity check. `DeeperShadowLightCurve` adds one bounded subdivision on sampled shadow crossings. `validate_quadratic_source.py assess` returns a usable density only for a passing eight-model grid. The independent physical chains use the original m029 sampled model. [Scope, measured errors and reproduction](ada_progress.md)

## m029 — Shared-packet irradiation and checkpointed physical trials

`PairedEmittingAreaLightCurve` rebuilds both emitting resolutions and both thermal fields and integrates all four on each eclipse packet. `sample_reflected_binary.py` runs one serialized, checksummed Metropolis chain on the reflected synthetic fixture. Use its `init`, `evaluate` and `advance` commands. A resolution exception retains the current point and RNG state for the pending trial. `PairedQuadraticLightCurve` is an independent experiment with a different emitting-position expansion and its own finite-error assessment. [Validated scope, measured costs and reproduction](ada_progress.md)

## m028 — Reflection from curved emitting areas

Select `EmittingAreaLightCurve` in `gpu_nodal_source` or `--engine emitting-area` in the validation driver. Three local Roche points share the original emitting face’s area, temperature and bolometric limb-darkening coefficient. Linear limb darkening uses grouped directional and spatial moments away from both horizons. A group touching either horizon uses its individual points. Nonlinear laws evaluate individual points within the same interaction cap. Select `--receiver-points 3` for the validated three-point receiving rule. The eight-model check at nominal uncertainty 10⁻⁴ passes at the synthetic generating parameters and stops two other physical proposals. [Accuracy, timings and reproduction](ada_progress.md)

## m027 — Emitter displacements and receiving horizons

Select `SpatialReflectedLightCurve` or the validation engine `spatial-reflection` for the first-order emitting-position rule. The `receiver_points` and `donor_level` options follow `AreaReflectedLightCurve`. The host constructs a proposal-specific index for groups crossing a receiving tangent plane. The solver uses individual native terms there and includes the sparse expansion in the interaction cap. Directional moments and the native correction run on the selected backend. The reflected detached tests still exceed the finite likelihood-error limit at nominal flux uncertainty 10⁻⁴. [Curves, scope and reproduction](ada_progress.md)

## m026 — Directional donor emission

Select `AngularReflectedLightCurve` or the validation engine `angular-reflection`. The receiver count and donor cap have the same meaning as in `AreaReflectedLightCurve`. Signs that can change across the receiver box trigger native positions and normals for all children of that group. For linear and cosine limb darkening, sign-stable groups use the luminosity-weighted first and second moments of child normals on a shared centroid ray. Logarithmic and square-root laws evaluate their native faces under the same pair limit. The calculation retains the documented symmetric detached scope and strict resource, phase and likelihood checks.

The serial CPU and GPU use separate scalar and tiled moment evaluation. Both return fresh component and normalized curves. Host geometry selection, limb-darkening preparation and the remaining CPU observer stages are included in measured costs. The original observer and existing fallback entry points retain their established behavior. [Scientific controls and reproducible commands](ada_progress.md)

## m025 — Area-integrated reflection

Select `AreaReflectedLightCurve` in `gpu_nodal_source`, or `--engine area-reflection` in the complete validation driver. Its defaults are three curved-area samples per receiving face and donor level cap 2. Set `receiver_points=1` to evaluate incident flux at the original face centres. The donor cap accepts levels 0 through 3. A cap at or above the field level retains the original visible emitting faces. Each complete curve records both settings.

This opt-in model supports detached, circular, symmetric stars within the existing Roche, table and quadrature bounds. It preserves the cooler-first update order and separately rebuilds donor luminosities at every update. CPU and GPU use the same tested finite rule. Original observation mapping, host LD preparation and the CPU boundary work are included in complete costs. The explicit resource and accuracy errors propagate outside prior penalties. The default observer and existing whole-CPU fallback retain their established entry points. [Validation and reproduction](ada_progress.md)

## m024 — Receiver irradiation

Select `ReceiverSourceLightCurve` in `gpu_nodal_source`, or `--engine receiver-source` with the complete comparison driver. It requires the supported detached circular symmetric field path and preserves per-face post-reflection T⁴. The reconstructed slope uses a finite, positivity-limited gradient from adjacent original deposits. Depth 4 partitions the temporary curved-edge frontier and retains the total packet-node guard. The combined field/depth likelihood error on the irradiated fixture exceeds the nominal limit for fitting. [Validation and reproduction commands](ada_progress.md)

## m023 — Face-local spectral source

Use `NodalSourceLightCurve` in `gpu_nodal_source` with the usual passbands, curved-edge rule, levels, common integration level and device. It selects curved-area thermal integration and evaluates passband and LD moments separately at three physical corners of each parent. Its rule records `face_local_spectral_nodes=1` and `thermal_area_order=4`. Device mode keeps gravity and blackbody/CK04 spectral computations on the selected device with explicit host LD/table preparation and requested host outputs. The original CPU reference and `radiance="cpu"` bridge use original surface spectral routines. The model checks symmetry, source area, per-cell T⁴ closure, conditioning, spectral-table boundaries and allocation limits. Existing unsupported-case and full-CPU routing semantics continue to apply.

The validation driver accepts `--engine nodal-source`. The original observer and the previous constant and continuous field engines remain available. `scripts/validate_nodal_emission.py` measures unirradiated interior directional intensities independently of the whole-curve finite-resolution gate. The current reflected field/depth comparison exceeds the 0.01 nominal log-likelihood limit. [Validation, costs and reproduction commands](ada_progress.md)

## Area-integrated intrinsic temperature

Pass `area_thermal=True` to `FusedObserverLightCurve` to evaluate the gravity-darkening bolometric source over 16 positive curved-area nodes per original quarter cell. The engine stores the fourth root of the cell-average intrinsic emission after stellar energy normalization, then uses the existing irradiation, passband and observer calculations. The finite curve rule records `thermal_area_order=4`. Omitting the option selects the established centre-sampled source. The reference uses original CPU gravity and polar-temperature helpers. Supported detached geometry and existing explicit errors, full-CPU routing and accuracy safeguards continue to apply.

The validation driver selects this model with `--engine area-thermal`. Its `--without-irradiation` option disables model heating while retaining the same irradiated observational fixture. Use `scripts/compare_gpu_irradiation.py` to compare matched heated and intrinsic field pairs on identical eclipse packets. The script gives the mapped per-band perturbations and the signed likelihood cross term. Use `scripts/compare_gpu_rectangle.py` for the separate field/depth qualification test. The measured heated area-integrated case exceeds the nominal 0.01 likelihood budget at 10⁻⁴. [Protocol, results and replay commands](ada_progress.md)

## Conservative emitted-intensity fields

`ReconstructedObserverLightCurve` in `gpu_reconstructed_observer` accepts the same explicit device, field levels, common integration level and `CurvedEdgeRule` as `FusedObserverLightCurve`. It reconstructs band-specific intensity and limb-darkening moments at shared vertices of the two physical components. It evaluates those moments on the curved observer packets and returns the usual component, raw and normalized curves. The curve rule records `conservative_moment_reconstruction=1`. Original cell-constant engines and the application observer retain their previous behavior.

`RectangleInferenceBudget.compare(fine_deep, coarse_deep, fine_shallow, coarse_shallow, sigma, observed=..., reducers=..., mean_normalize=True)` applies direct and mixed field/depth controls to matched physical inputs. The default call raises `ResolutionNotQualifiedError` when the nominal bound exceeds either threshold. Use `require_budget=False` to inspect the diagnostic, including the mixed difference and component envelopes. Accuracy and resource limits remain outside physical-prior penalties.

On the reflected 10⁻⁴ fixture, continuous and cell-constant fields both exceed the accumulated likelihood tolerance. The new representation is experimental. [Validation and reproduction](ada_progress.md)


Current Ada allocation policy (2026-11-17): Fifth42 is intentionally checkpointed
and held inactive until genuine ELISa completion. Each GPU study remains a
registered <=300-second unit whose entry and cleanup verify the exact stopped
checkpoint, receipt/witness endpoints and original source/config/plan/service.
Do not resume production between windows. Historical per-window restoration
records below describe the earlier policy. The supervisor restores the original
service with durable advancement after genuine completion.

Current complete-curve/fitting APIs and measured limits are summarized in
[Ada progress](ada_progress.md). The sections below retain the development
history. Current irradiation is described in
[device mutual irradiation](#device-mutual-irradiation-in-complete-curves).
The broader physical-routing option is described in
[original physical routing with device integration](#broader-original-physical-routing-with-device-integration).
Opt-in cadence and finite-mesh qualification are described in
[observation-phase and mesh-qualified fitting](#observation-phase-and-mesh-qualified-fitting).
Finer reflected surfaces are described in
[tiled irradiation and paired mesh refinement](#tiled-irradiation-and-paired-mesh-refinement).

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
[1e-6,.25] SMA, beta in [0,1], unspotted/nonpulsating, two blackbody or two CK04
components and custom or original CPU-interpolated table LD. CPU reflection follows current settings. Existing geometry
conditioning masks remain active. Outside this domain, default fallback runs
the **entire unmodified CPU Observer** and reports the reason; `fallback="raise"`
instead raises `UnsupportedHybridModelError`. Device unavailability/disabled FP64
never silently selects CPU. Unsupported states are not physical-prior penalties.
With the default `general="cpu"`, mixed/other atmospheres, spots and
eccentric/contact GPU states use CPU fallback,
preserving upstream data errors and exceptions. See the table-emission section
below for the actual tested CK04/LD source coverage and conditioning guards.

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
This section records the fourth milestone's blackbody domain. CK04 and
CPU-interpolated table LD are extended below; spots/eccentric/contact GPU states
remain unsupported.


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

## Actual CK04 spectra and interpolated table LD

`gpu_atmosphere.DeviceAtmosphereRadiance` extends the complete hybrid to two
CK04 components, with custom or original CPU-interpolated LD. Two blackbody stars
can also use actual interpolated LD. Source selection/read/revision checks,
wavelength alignment/stripping, passband response and LD interpolation remain
original CPU operations. Device work integrates each unique prepared spectrum,
gathers face brackets, interpolates in T⁴, mirrors faces, divides by the same-band
LD integral and feeds fresh radiance directly into device eclipse packets.

This preserves the original **nearest** gravity/metallicity selection and
T⁴ atmosphere interpolation. LD remains piecewise linear in T/log(g), with
metallicity selecting the original ordered source tables. It does not add a
continuous gravity/metallicity atmosphere interpolation or change CPU physics.
Only fixed quadrature and source-dependent LD hulls are reused; actual spectra,
physical interpolation weights, LD and brightness are fresh per proposal.

Near atlas selection boundaries or an LD source's convex-hull boundary, small
state roundoff can change file/domain decisions. These cases run the whole
original CPU curve. The dispatch margin is four times the existing 3e-13 state
relative gate, not an increased acceptance tolerance or a prior exclusion.
Resource failures also use complete CPU fallback. Original missing/malformed
source errors keep their classes and fitter handling; missing device/FP64 raises.

The atlas path supports at most 8 bands, 8192 faces per star, 128 unique spectra
per band/component, 2–4096 wavelengths and 8388608 prepared spectral samples.
The sample cap excludes runtime/compiler workspace. With default `general="cpu"`,
mixed/other atlases, spots and eccentric/contact GPU states retain whole CPU
fallback. This table-emission milestone's
scientific fixtures use existing unmodified repository CK04/Van Hamme data,
V/bolometric bands, solar metallicity and three T/g regions; this is not general
table/model accuracy or resolution-convergence evidence.

Run `scripts/validate_gpu_atmosphere.py --device cpu --output tables-cpu.json`
with the private CPU wrapper. The GPU variant requires an independently restored
bounded allocation. It includes complete curves with both eclipses/contacts,
fresh table/response revisions and LD, physical fitting/likelihood, public fits,
short sampler replay and matching host-output transfer-inclusive costs. Read
[Ada progress](ada_progress.md) for current measurements and remaining work.

## Broader original physical routing with device integration

`general="device_integration"` explicitly opts into a second hybrid path when
the circular detached fast path cannot handle a model/state. It uses the
original CPU physical routers for synchronous/asynchronous spots, eccentric
curves with/without spots, and near-contact/overcontact shapes. CPU mesh and spot
evolution, changing separation, reflection, atmosphere/LD preparation, orientation,
coverage and configured approximations remain original operations. A fresh
all-band/component packet moves only per-face flux arithmetic and integration
to the requested JAX device. There is no resident GPU physical state on this
broader branch and no physical cache across candidates.

```python
from elisa.experimental.gpu_binary import HybridBinaryLightCurve

engine = HybridBinaryLightCurve(
    ["Generic.Bessell.V", "bolometric"],
    device="gpu", general="device_integration", fallback="cpu",
)
result = engine(binary, phases, normalize=True)
# "hybrid", "hybrid_native", or "cpu_fallback"; reason/timings disclose dispatch.
print(result.backend, result.fallback_reason, result.timings)
```

`general="cpu"` is the unchanged default: unsupported fast-path cases run the
whole original CPU curve. With the broader opt-in, `fallback="raise"` raises
only after both hybrid paths are ineligible. Partial native resource failures
discard partial results and rerun complete CPU synthesis. Original data errors
still propagate; resource limits are not physical prior penalties. Missing GPU
or disabled FP64 raises, as does nonserial opt-in configuration. In fitting,
these infrastructure checks remain outside emergency candidate penalties.

The broader bridge requires `NUMBER_OF_PROCESSES=1` and an original nonpulsating
`BinarySystem`. Bounds are 1–8 bands, 8192 faces per component and 256 physical
callbacks per curve, with power-of-two selected-face padding. Custom subclasses,
pulsations, unvalidated routes and resource failures use full CPU fallback.
Successful native output preserves original phase ordering/repeats, shift,
third light, distance and normalization. `component_fluxes` is empty because
native approximations can interpolate total curves; optional diagnostics retain
actual per-callback CPU containers, immutable packets and component/face fluxes.
The private working Observer delegate is restored on every exit; no global CPU
function or caller's model is patched.

The `gpu_ordered.OrderedSurfaceFluxIntegrator` retains FP64 expression stages
and the selected-length contiguous NumPy pairwise sum order. This resolved an
overcontact finite-difference optimizer sensitivity without changing the CPU
oracle, fit settings or gates. Independent signed/cancellation/boundary controls
compare actual NumPy reductions exactly on both devices. The tested release is
NumPy2.5.5; universal bitwise equality across releases/transcendentals is not
promised. The earlier fast-path integrator retains its existing implementation.

Physical fitting subclasses accept `general="device_integration"`:
`HybridDetachedLeastSquaresFit`, `HybridDetachedMCMCFit`,
`HybridOvercontactLeastSquaresFit` and `HybridOvercontactMCMCFit`. The new
overcontact classes inherit the original morphology validation and shared
potential constraints. Use original `LCData`/`BinaryInitialParameters` and public
`fit()`/`run_mcmc()` contracts as in the detached example above. Original phase
mapping/mean normalization, noise/third-light/prior/statistic conventions remain
in the original fitter. No `LCBinaryAnalyticsTask` backend/default is registered.

Run `tests/tests_experimental/test_gpu_native.py` and
`scripts/validate_gpu_native.py --device cpu --output native-cpu.json`; run the
GPU variant only inside the independently restored bounded allocation. Eight
V/bolometric fixtures include real spotted CK04/LD tables, changing spots and
separation, reflection and near/overcontact. Additional tests cover nonlinear
LD and fallback/error semantics. Public spot/overcontact fits and short sampler
replay use the original optimizer options; eccentric coverage is prediction/
posterior callbacks. Configured approximation error against rigorous CPU is
reported separately, including an inherited sparse-input error on both paths.

The hardware suite passed 269 tests with zero skips. Detailed linear-LD curve,
likelihood, fit and short-chain differences were zero, but the complete broader
GPU workloads were **slower than original CPU**. Three sampler steps do not
establish convergence or ESS/sec. See [curated measurements](ada_validation/background-m006.json)
and [Ada progress](ada_progress.md) for full costs, domain, source fingerprints,
production restoration evidence and remaining device-physics work.

## Device mutual irradiation in complete curves

The circular detached hybrid and its physical fitting subclasses now default
to `reflection="device"`. `reflection="cpu"` selects the preceding CPU-reflection
hybrid for comparison. Original CPU Observer/fitters and global settings retain
their defaults. The broader `hybrid_native` spot/eccentric/contact branch still
uses the original CPU physical routers, including their reflection routine.

`gpu_reflection.DeviceReflection` consumes each candidate's cached pre-correction
centres/normals and corrected areas/intrinsic temperatures. Original CPU helpers
select visibility/quarter-face indices and obtain bolometric LD from the fresh
intrinsic temperatures/log(g). Device work computes pairwise distance/join vectors,
directional LD, coupling matrices and reflection factors, updates the cooler star
first using the latest counterpart values, repeats the requested iteration count,
then mirrors final temperatures. Equal-temperature ties select primary first.
Intrinsic bolometric LD stays fixed during iteration, as in the original CPU;
the symmetry helper's LD coefficient **prefix ordering** is preserved exactly.
No change to that CPU scientific convention is bundled with this port.

Final device-reflected temperatures feed device blackbody or CK04 emission and
eclipse integration directly. A host copy still supports original LD/atlas file
selection and interpolation. CPU angular mesh/roots/topology/kinematics, visibility,
LD lookup, source preparation and observer hull bridges remain measured costs;
this is a hybrid model. Only code/compiled shapes are reused across proposals.
`diagnostics=True` exposes the immutable reflection packet and actual device
matrices/temperature outputs; diagnostics are off for complete timing/inference.

```python
engine = HybridBinaryLightCurve(
    ["Generic.Bessell.V", "bolometric"], device="gpu",
    reflection="device", fallback="cpu",
)
result = engine(binary, phases, normalize=True)
```

Reflection is bounded to the earlier circular detached unspotted/nonpulsating
state domain, separation1 SMA, 8192 faces per star, 2097152 combined prepared face
pairs and 32 iterations. The matrix cap excludes compiler/runtime workspace.
Disabled reflection or nonpositive iterations follow the original no-op behavior.
Unsupported resources/numeric states use complete CPU synthesis, or the explicit
broader bridge when requested; `fallback="raise"` raises after eligible paths
are exhausted. Original missing LD-source errors retain their classes and fitter
handling. Missing device/disabled FP64 remains an infrastructure error outside
candidate penalties.

Run `tests/tests_experimental/test_gpu_reflection.py` and
`scripts/validate_gpu_reflection.py --device cpu --output reflection-cpu.json`.
Run the GPU variant only inside the independently restored bounded allocation.
Tests include all four laws, both update orders/ties, real table LD, unilluminated
faces, zero albedo, iteration progression through32, changed albedo/physical
proposals, immutable snapshots/no-transfer evaluation, complete curves and public
blackbody/table fits with short sampler replay. The full 287-test hardware suite
ran in two independent restored windows with no skipped cases. Read
[Ada progress](ada_progress.md) for matched complete costs and scientific limits;
parity and short sampler replay do not establish mesh/spectral convergence or ESS.

## Resolution diagnostics for scientific accuracy

`gpu_accuracy.resolution_error(prediction, reference, sigma, observed=None,
ln_f=-inf, max_distance_sigma=0.1)` compares finite model predictions **after
original observation mapping and normalization**. It reports noise-scaled maximum,
RMS and joint distance, expected log-likelihood loss, and the data-dependent
chi²/log-likelihood difference using the original noise-inflation law. The default
joint-distance0.1 diagnostic budget corresponds to expected loss0.005. It does
not alter synthesis, CPU defaults, implementation gates, sampler priors or fallback.
Finite-reference agreement is not a continuum accuracy certificate.

`BlackbodySpectralProbe` is an independent **CPU diagnostic**: Gauss-Legendre
quadrature of the actual clipped response in log wavelength feeds an owned
Observer callback while retaining original fresh physics/coverage/normalization.
Original native-grid control and independent adaptive integrals verify it.
Responses are fixed snapshots; changed sampled responses require a fresh probe.
It does not replace atmosphere data, certify missing spectral information or
provide a new GPU performance path.

Run `tests/tests_experimental/test_gpu_accuracy.py` and
`scripts/validate_gpu_accuracy.py --device cpu --part curves --output curves-cpu.json`,
then `--part fitting` for the physical-fit resolution assessment. GPU variants
require independent bounded checkpoint/restore allocations. The original engines
are unchanged: at finer meshes/dense supports, existing bounds cause whole CPU
fallback, never a partial curve or physical-prior rejection. New targeted tests
and matched CPU/GPU profiles pass the unchanged implementation gates.

The current report finds mesh changes larger than1e-4 noise even for the last
finite refinement, substantial fit-parameter shifts, and a nonzero-third-light
support convention that grid density alone cannot reconcile with direct cadence
synthesis. The zero-l3 interpolation control and tested blackbody quadrature meet
their sampled diagnostic budgets. Read [Ada progress](ada_progress.md) and
[curated evidence](ada_validation/background-m008.json) for actual errors,
complete costs, source hashes, figure reproduction and unresolved accuracy scope.
No convergence, continuum posterior or ESS/sec claim follows.

## Observation-phase and mesh-qualified fitting

All four `Hybrid*LeastSquaresFit`/`Hybrid*MCMCFit` classes accept
`phase_mode="observations"`. Default `"native"` preserves original support
selection and interpolation. Observation mode synthesizes freshly at the original
observation union each call; it bypasses unused uniform/manual/adaptive support
preparation. Original dynamic JD/phase conversion, phase preservation/shift,
reducers, repeated per-band observations, maximum-then-mean normalization,
third-light and noise laws remain in the original predictor. This explicit
cadence-dependent third-light convention can differ from native support synthesis
even with arbitrarily dense interpolation grids. It is instantaneous synthesis,
without exposure integration.

```python
from elisa.experimental.gpu_fit import HybridDetachedLeastSquaresFit
from elisa.experimental.gpu_protocol import MeshResolutionCheck

fit = HybridDetachedLeastSquaresFit(
    device="gpu", phase_mode="observations",
    resolution_check=MeshResolutionCheck(
        refinement_degrees=(8.0, 6.0), max_distance_sigma=0.1,
    ),
)
# Use the original fit()/set_up() inputs with discretization=12.0.
```

The CPU-only counterparts in `elisa.experimental.gpu_protocol` are
`ProtocolDetachedLeastSquaresFit`, `ProtocolOvercontactLeastSquaresFit`,
`ProtocolDetachedMCMCFit`, and `ProtocolOvercontactMCMCFit`. They use the same
optional protocol and original CPU synthesis. No global CPU fitter/default or
analytics-task backend registration changes. Both protocols require serial
synthesis/optimizer/sampler use; instances are single-caller objects.

`resolution_check=None` disables mesh qualification. If supplied, it requires
observation mode and exactly two strictly finer decreasing references, each at
least 2 degrees, within the exercised CPU grid range. References below 3° require
explicit paired mesh mode as described below. At **each candidate**,
the original CPU model runs on the target and both finer meshes, on copied
Observers retaining passband/atmosphere/LD settings. Both target-to-finest and
middle-to-finest concatenated-band distances must satisfy the configured budget.
This guards against simple endpoint cancellation without assuming monotone
convergence. The noise scale is the original **normalized observational y_err**,
independent of fitted ln_f; increasing nuisance noise cannot evade qualification.
The original likelihood still uses its unchanged noise-inflation law.

Qualification executes **before** the original LSQ/MCMC candidate-penalty
handlers. `ResolutionNotQualifiedError.report` and `last_resolution_check` expose
the failed budget or failed refinement. The exception aborts the fit/run even
with `PENALIZE_ALL_EXCEPTIONS=True`; it is never an infinite residual, -inf prior
cut, diagnostic invalid-model count, or CPU fallback. A failed original target
retains its exact original physical-candidate rejection policy. Out-of-prior
MCMC proposals retain their original rejection without synthesis/check work.
Unsupported device physics/resources still use the complete original CPU target;
that fallback cannot cure shared discretization error.

Only successful empirical checks proceed to the requested CPU/hybrid target
evaluation. The three CPU checks and that final target are all included in
complete costs; the CPU path currently repeats the target deliberately. No
physical predictions or accuracy certificates are cached. A finite-grid pass is
**not a continuum bound**, model-systematics assessment, or guarantee of a
data-dependent likelihood change. D<=0.1 corresponds to expected likelihood loss
0.005 if the finite reference generates the data. Budget/reference/phase settings
are explicit construction-time choices; do not adapt them during a run to make
failed proposals pass. A failure requires a scientifically revised numerical
protocol and a fresh fit/run, rather than continuing a selectively truncated chain.

Run `tests/tests_experimental/test_gpu_protocol.py` and
`scripts/validate_gpu_protocol.py --device cpu --output protocol-cpu.json`;
GPU variants require an independently restored bounded allocation. The driver
tests original direct mapping and posterior parity, explicit 1e-4 mesh failure,
a separate 5% noise positive control with public physical fits/short sampler
replay, and complete transfer-inclusive qualification costs. See
[Ada progress](ada_progress.md) for actual results and remaining surface accuracy.

## Tiled irradiation and paired mesh refinement

`reflection="tiled"` opts into `TiledDeviceReflection`: at most 32 receiver rows
are processed against **all** visible donor faces at once. Each sequential
cooler-first irradiation iteration recomputes the tiles. It preserves the original
bolometric LD prefix order, fixed intrinsic LD, all donor contributions, update
order/ties and final symmetry mirroring. No distance cutoff, fitted heating law
or temperature cache is introduced. It trades repeated geometry work for memory;
total arithmetic remains quadratic. Full distance/gamma/coupling diagnostics are
omitted because retaining them would defeat the memory bound.

```python
engine = HybridBinaryLightCurve(
    ["Generic.Bessell.V", "bolometric"], device="gpu",
    reflection="tiled", coverage="cpu", fallback="raise",
)
result = engine(binary_with_explicit_finer_component_grids, phases, normalize=True)
```

The explicit mode supports up to 40000 faces per component and 268435456 prepared
reflection pairs, with the existing 0–32 iteration bound and circular detached,
unspotted/nonpulsating physics domain. Tile sizes 8–64 are available on the direct
reflection API; complete curves use 32. It retains immutable device input snapshots.
Missing FP64/device remains an infrastructure error; unsupported physics/resources
retain whole CPU fallback or explicit raise. Dense reflection and all original
defaults keep their preceding capacities.

For this mode, blackbody emission can accept 40000 symmetric faces through the
same bounded spectral chunks, consuming fresh reflected device temperatures.
The direct `DeviceBlackbodyRadiance(..., fine_mesh=True)` exposes that explicit
capacity. Original default 8192-face capacity is unchanged. Finer CK04 examples
select `radiance="cpu"` and retain original atmosphere/LD processing. Current
device occultation retains its preceding capacity, so finer examples explicitly
use original CPU orientation/hulls/eclipse coverage with device face integration.
These are **hybrid** complete curves. No new all-GPU forward model is claimed.

Physical fitting additionally accepts `mesh_mode="paired"` together with
`phase_mode="observations"`. It explicitly sets the secondary factor equal to
the primary's requested factor on every target/reference evaluation. Default
`mesh_mode="native"` retains the original temperature/radius-dependent secondary
adjustment and configured clamps (3–12° in these validation profiles). A finer primary alone may leave the
secondary substantially coarser: the tested 2.5° fitter yields an inferred 9.16°
secondary. Paired mode is an explicit change of numerical protocol, independent
of physical priors, with no global setting or original serializer change.

`MeshResolutionCheck` now admits reference degrees down to 2°, independently
validated against unmodified CPU synthesis; below 3° requires paired mode. Finer
finite 1.5° curves use an independent CPU diagnostic in
`scripts/fine_mesh_reference.py`, composed from original CPU mesh/gravity/
thermal, NumPy/Numba distance/gamma/LD and unchanged observer routers. Coarse
temperature controls and dense complete-curve checks through 2° establish parity.
It is labelled separately at 1.5°, rather than claiming an unperformed dense
CPU run or continuum truth. Qualification still aborts outside candidate
penalties; same-grid CPU fallback cannot establish surface convergence.

Run `test_gpu_reflection_tiled.py` and the full experimental suite. The finer
driver is `scripts/validate_gpu_fine.py --device cpu --part curves --output curves.json`;
`--part fitting --fit-level 3` or `--fit-level 2` exercises public physical LSQ
and changed posterior callbacks on a shared finite reference dataset. GPU variants
require independently restored bounded allocations. The GPU driver explicitly
selects JAX CPU and GPU backends for matched complete costs. Read
[Ada progress](ada_progress.md) for errors, memory accounting, timings, restored
production and remaining accuracy limits; these comparisons do not certify
continuum accuracy, finite exposure, physical model systematics or MCMC convergence.

## Nested surface/contact change estimator

`NestedMesh` and `NestedBinaryLightCurve` provide an opt-in, standalone numerical
experiment. They are not registered fitting backends. Every evaluation builds
the original candidate seed mesh, subdivides each triangle into four children
per level, projects new edge midpoints onto the original Roche equipotential
with CPU Brent roots, and rebuilds the complete corrected gravity/thermal,
irradiation, LD/radiance and eclipse state. No field or prediction is interpolated
from a different candidate. Original topological seam identities keep the four
quarters conforming; level zero preserves the original raw mesh exactly.

Levels 0–3 and at most 40000 faces per component are supported, within the
existing circular detached, symmetric, unspotted/nonpulsating q/spin/radius
domain. A paired 12° seed has 540/2160/8640/34560 faces per star. The original
scalar mesh-correction calibration is evaluated at seed degree / 2**level;
applying that calibration to this topology is a finite numerical convention,
not a newly established truncation-error law. This is uniform nested refinement
with local indicators, not selective adaptive refinement or extrapolation.

```python
from elisa.experimental.gpu_nested_mesh import NestedMesh
from elisa.experimental.gpu_surface_error import NestedBinaryLightCurve, compare_surfaces

# binary has explicit component seed factors; phases are the actual observations.
coarse = NestedBinaryLightCurve(bands, NestedMesh(2), device="gpu").snapshot(
    binary, phases, observer=observer, source_revision=fixed_scientific_assets_sha256,
)
fine = NestedBinaryLightCurve(bands, NestedMesh(3), device="gpu").snapshot(
    binary, phases, observer=observer, source_revision=fixed_scientific_assets_sha256,
)
report = compare_surfaces(
    coarse, fine, y_err, reducers=observation_reducers, mean_normalize=True,
    observed=y_data, ln_f=ln_f, require_budget=True,
)
```

Snapshots contain immutable host curves and root-face groups, with a signature
covering the physical candidate, actual seed topology, phases, passband contents,
relevant settings and caller-supplied scientific asset revision. Comparisons
preserve per-band observation reducers, order/repeats, third light and the
original final observation mean. The signed root-group changes, third-light
offset change and summation/normalization residual reconstruct the actual finite
prediction change. A host arithmetic guard is included. The sum of absolute
component changes encloses that measured change; a separate, more conservative
root-face L1 envelope and noise-weighted local scores locate cancellation and
limb/partial-eclipse contributions. Boundary classifications are diagnostics,
never acceptance masks. These scores describe changes in the fully coupled
models; selectively refining a patch can change irradiation and normalization
elsewhere, so its score is not an independent local truncation-error bound.

`within_finite_budget` and `require_budget` use the **nominal observational
errors**, independently of ln_f. Inflating nuisance noise cannot evade this
numerical gate. Likelihood diagnostics separately retain the original
sigma²*(1+exp(2*ln_f)) law. With data supplied, the finite absolute log-likelihood
change is bounded by sum(|r| E + E²/2), where r is the coarse residual and E the
component envelope in effective-sigma units. This includes the data-dependent
cross term. A passing finite pair is **not a bound on an uncomputed grid or the
continuum**; no order, monotonicity or Richardson extrapolation is assumed.

Call this preflight outside candidate-penalty handlers. A failed finite budget
raises `ResolutionNotQualifiedError`; unsupported refined numerics raise
`UnsupportedNestedMeshError`. Neither becomes a prior rejection or a silently
coarser CPU prediction. Existing CPU/default and hybrid fallback semantics are
unchanged. The existing public resolution-qualified fitter remains separate.
Do not adapt the numerical protocol to accept selectively failed proposals.

These snapshots use CPU midpoint roots, orientation/hulls/eclipse coverage and
host parent grouping, plus device corrected state/reflection/integration and
fresh blackbody emission. CK04 uses explicit `radiance="cpu"`. Full diagnostic
transfers and grouping are charged; snapshot timing has a larger output contract
than ordinary curves. The independent original-CPU reference now resides in
`elisa.experimental.gpu_reference`; `scripts/fine_mesh_reference.py` retains its
compatibility exports. Neither reference patches global CPU functions/settings.

Run `test_gpu_surface_error.py`, then the full experimental suite. Reproduce the
complete-curve and original physical-observation likelihood studies with
`scripts/validate_gpu_surface_error.py --device cpu --part curves --output curves.json`
and `--part observations`. GPU variants require independently restored bounded
allocations and explicitly compare matched JAX CPU/GPU snapshot contracts.
The observation study uses the original serializer, phase/JD mapping, prior,
third light, final mean and likelihood. It is not an optimized physical fit or a
sampler-convergence study. See [current results](ada_progress.md).

## Fixed selective surface refinement and correction audit

`SelectiveMesh` is an opt-in `NestedMesh` protocol for a fixed set of seed-quarter
patches. It starts at `levels-1`, projects only marked edge midpoints onto the
original Roche equipotential, splits selected triangles into four, and fills
neighboring triangles with conforming two/three/four-child patterns. Quarter
symmetry and root ancestry are preserved. Empty/full selections reproduce the
corresponding original uniform levels exactly. Supported maximum levels are
1–3, with the existing 40000-face bound and circular detached q/spin/radius domain.
CPU roots, orientation/hulls/coverage and host grouping remain explicit bridges.
All gravity/thermal, irradiation, LD/radiance and integration fields are rebuilt.

```python
from elisa.experimental.gpu_selective_mesh import choose_plan

# coarse/fine are a consecutive uniform pilot pair for this seed, using the
# preceding snapshot API. Pilot construction is part of numerical setup cost.
fixed = choose_plan(binary, coarse, fine, nominal_sigma, fraction=0.5)
engine = NestedBinaryLightCurve(bands, fixed, device="gpu")
result = engine.snapshot(binary, observation_phases, source_revision=assets_sha256)
```

The pilot ranks nominal-noise-weighted root-group flux changes across all supplied
bands/phases, then folds the four symmetry quarters. It uses original maximum
normalization, with no ln_f dilution. These are coupled finite-model indicators,
not independent local truncation errors. The phase grid and weights used to
choose the plan are explicit numerical design choices. Freeze the immutable plan
before subsequent evaluations; do not change it per proposal or after an accuracy
failure. A seed topology hash prevents index reassignment. Physical movement of
vertices is allowed when topology matches; incompatible topology aborts the
numerical protocol outside candidate penalties. No physical prediction is cached.

The default `correction="density"` evaluates the **original scalar mesh
correction** at base_angle*sqrt(base_faces/new_faces). This is an explicit
finite convention for a nonuniform grid, not a validated new correction law.
`"coarse"` and `"fine"` hold the scalar's angle at the base or fully refined
level for diagnostic ablations with identical raw topology. They are not fitted
corrections or physical priors. Existing CPU/default correction is unchanged.

`gpu_mesh_audit` independently integrates a star-shaped Roche surface with
Gauss-Legendre mu and periodic phi quadrature, CPU Brent roots and original
analytic potential gradients. Its area Jacobian is r²|grad Phi|/(grad Phi dot
radial_direction); volume uses r³/3. Exact-sphere controls and separate angular
orders check this geometry calculation. `mesh_measures` compares raw and scalar-
corrected faceted area/volume without changing cached state. These diagnostics
apply before mesh correction/thermal construction. Converged geometric integrals
are not complete-curve, eclipse or continuum likelihood certificates.

`compare_surfaces(..., allow_same_level=True)` explicitly admits alternate
maximum-level topologies sharing the same seed ancestry. Its enclosure remains
only the computed finite change, with nominal-sigma accuracy stops. Default
ordered-level behavior is unchanged. Original priors, unsupported-case CPU
routing and public fitting classes retain their preceding behavior. The new
selective experiment raises explicitly rather than silently coarsening the
requested numerical model; it is not a registered fitting/sampler backend.

Run `test_gpu_selective_mesh.py` and the full experimental suite. Reproduce with
`scripts/validate_gpu_selective.py --device cpu --part curves --case partial_bb --output partial.json`,
`--case others`, and `--part observations`. Use the private CPU wrapper while
production runs. GPU variants require independent bounded restoration. The driver
compares original CPU raw curves and original maximum normalization, full fresh
JAX CPU/GPU snapshots with matched outputs, correction and randomized-root-count
controls, independent area/volume quadrature, separate finite triangulations,
and original phase/JD likelihoods under a fixed plan. It includes pilot/setup,
constructor, per-call transfers and grouping in documented complete costs.

The current evidence **does not qualify selective refinement at 1e-4**. More
triangles or CPU/GPU parity alone cannot establish improvement when the scalar
correction and local indicators do not transfer to the mixed grid. See
[current numerical results](ada_progress.md) before using this experiment for
inference. Accuracy failures abort; no fit, chain mixing or ESS gain is claimed.

## True Roche patch quadrature and implicit eclipses

`CurvedBinaryLightCurve` is a standalone, opt-in numerical model. It integrates
fresh original finite thermal/reflection/radiance fields over the actual Roche
equipotential. It leaves the original Observer, fitters and all preceding
backends unchanged. This deliberately changes the integration rule; parity with
an unchanged original whole light curve is **not** asserted. Its independent CPU
reference uses original field/reflection/radiance and per-element LD/flux
primitives with the same new quadrature. Comparisons with original faceted
curves are separately labelled numerical-method differences.

```python
from elisa.experimental.curved_geometry import CurvedRule
from elisa.experimental.gpu_curved import CurvedBinaryLightCurve, curved_difference

engine = CurvedBinaryLightCurve(
    bands, CurvedRule(order=3, depth=3), field_levels=1, device="gpu",
)
result = engine(binary, phases, observer=observer, source_revision=assets_sha256)
# Compare independently designed boundary/order/field refinements BEFORE a fit.
report = curved_difference(
    coarse, fine, nominal_y_err, reducers=reducers, mean_normalize=True,
    observed=y_data, ln_f=ln_f, require_budget=True,
)
```

Each angular parent triangle has a linear barycentric vector v and unit ray
v/|v|. Duffy Gauss nodes use solid-angle Jacobian |det(a,b,c)|/|v|³. Original CPU
potential roots give true points; original analytic gradients give outward
normals and dA/dOmega=r²|gradPhi|/(gradPhi dot radial). No scalar mesh correction
is applied to quadrature geometry. The thermal, irradiation and radiance fields
remain **piecewise constant original-ELISa parent-face fields**, including their
preceding original correction/build order. They are rebuilt for every physical
candidate. Their own spatial convergence remains separate and unqualified.

Visibility uses the true normal. Occultation tests rays against the foreground
star's **same implicit Roche surface**, rather than a differently corrected
faceted silhouette. A positive-term inscribed-radius bound handles central
rays; elsewhere a CPU stationary-potential solve uses the original gradient and
an FP64 residual guard. Independent scalar Brent tangencies control that solve.
The radius<.25 bound supplies a conservative projected-overlap screen.

Boundary integration uses a fixed finite adaptive rule: seven probes and a
safety band select further barycentric subdivisions, then actual node visibility
and implicit eclipse masks select the integrand. **This indicator is empirical,
not an interval enclosure or exact boundary integral.** Node-cap exhaustion or
unsupported geometry aborts; the rule never silently coarsens. Analytic sphere
area/LD/eclipse controls and independent boundary-depth/order comparisons are
required. Changing a parent field grid also changes patch geometry and boundary
cell size; equal integer depth is not a pure field-error comparison. Matched
dyadic leaf-scale controls help diagnose that coupling and remain finite tests.

`device="reference"` uses original CPU finite fields and original element-flux
arithmetic. `"cpu"`/`"gpu"` use the preceding JAX field/integration kernels.
CPU radial roots, adaptive partition and eclipse masks are common fresh bridges;
state downloads and quadrature uploads are included in complete costs. This is
**hybrid**, with no GPU geometry or overall acceleration claim. CK04 examples
explicitly select `radiance="cpu"`; blackbodies use fresh device emission.
Only fixed code/response information is reused across candidates.

The new mode supports circular detached unspotted/nonpulsating systems in the
preceding q/spin/inner-radius domain, field levels 0–2 with at most 20276 parents
per component, 1–8 bands and 1–128 phases. Orders 2–4, depths 0–7 and at most
2097152 quadrature nodes per component/phase are explicit construction choices.
The node cap is not a total-memory bound: seven probes per candidate leaf,
CPU geometry arrays, device outputs, compiler and allocator add storage.
Numerical/domain/resource failures raise outside candidate penalties. Original
CPU/full-fallback routes are unchanged; no curved fitting backend is registered.

`curved_difference` preserves original observation reducers, order/repeats,
third light and final mean. Its component enclosure bounds only the computed
finite normalized difference, including rounding and normalization residuals.
The data-dependent likelihood bound retains the residual cross term. Numerical
qualification uses nominal sigma independently of ln_f; failures raise
`ResolutionNotQualifiedError`, never a prior cut or infinite-residual substitute.
A passing finite pair is not a continuum or continuous-field certificate.

Run `test_curved_geometry.py` and `test_gpu_curved.py`, then
`scripts/validate_gpu_curved.py --device cpu --part curves --case partial_bb --field-levels 1 --order 3 --depth 3 --output curves.json`
and `--part observations` in the private CPU wrapper. GPU variants require
independently restored bounded allocations. `--comparison` loads a preceding
profile with identical physical/source/phase provenance to assess a finite
protocol change. See [current evidence and remaining accuracy limits](ada_progress.md).

A CPU-only transit-like probe with potentials [8,100] fails the new shadow
stationarity residual guard; original CPU synthesis completes. This is an
explicit unsupported numerical case, not a prior rejection or a substitute
coarse prediction. Local-coordinate conditioning remains work for the curved
method; no GPU or accuracy qualification is claimed for that probe.

## Boundary-fitted Roche integration

`BoundaryBinaryLightCurve` adds a separate opt-in numerical protocol. The
preceding `CurvedBinaryLightCurve` keeps its original node-mask rule. Neither
changes the original Observer, public fitters, physical priors or CPU routes.

```python
from elisa.experimental.boundary_geometry import BoundaryRule
from elisa.experimental.gpu_boundary import BoundaryBinaryLightCurve
from elisa.experimental.gpu_curved import curved_difference

engine = BoundaryBinaryLightCurve(
    bands, BoundaryRule(order=3, depth=4), field_levels=1, device="gpu",
)
result = engine(binary, observation_phases, observer=observer,
                source_revision=assets_sha256)
# This preflight is outside physical candidate-penalty handlers.
report = curved_difference(coarse, fine, nominal_y_err, reducers=reducers,
                           mean_normalize=True, observed=y_data, ln_f=ln_f,
                           require_budget=True)
```

The shadow stationary solve now stays in star-local coordinates, using the
original potential and its analytic derivative. It preserves the preceding
positive-derivative, displacement and **64eps relative residual guard**. This
avoids loss of the secondary's small displacement through repeated unit-centre
addition/subtraction. Independent original-equation Brent tangencies and
extended-precision gradient/potential controls cover small-star conditioning.
The shadow margin has its zero at the stationary-potential silhouette; a
projected radius minus the full tangency radius would not have the same zero.
The previous [8,100] potential fixture is tested through complete new curves;
higher-potential local-equation tests do not qualify complete tiny-star curves.

Seven probes and the preceding two-spread/diameter safety band guide finite
barycentric subdivision. At the requested depth, sampled contours that cannot
be represented by endpoint signs receive bounded extra topology subdivisions.
Each accepted leaf is clipped first to the implicit visible limb and then to
the clear side of the implicit foreground shadow. Crossing edges receive
bracketed bisections. Triangulated leaf polygons retain their original parent
field identity and true Roche quadrature normals/Jacobians. Integration uses
the fitted shadow polygon without reapplying discontinuous node shadow masks;
true nonpositive limb cosines contribute zero.

**Straight leaf chords approximate a curved boundary.** Edge root brackets do
not certify the chord error, and seven probes do not exclude every sub-probe
contour loop. Topology exhaustion and node-cap failures abort explicitly.
Analytic sphere overlap, limb-darkening integrals, independent depth/order
controls and nominal-noise likelihood preflights remain necessary. A passing
finite pair is not an interval, continuum or continuous-field certificate.

`BoundaryRule` inherits orders 2–4, depths 0–7 and the 2097152-node cap per
component/phase. It adds fixed `edge_bisections` 32–48 (default 36), and
`topology_depth` 0–4 (default 4) extra subdivisions only for unresolved sampled
contours. The cap is checked before refinement and after clipping; it is not a
peak-memory limit. The preceding physical, phase, band and field-grid bounds
remain. Unsupported numerics never become prior penalties or silently coarser
predictions. CK04 explicitly uses `radiance="cpu"`.

Fresh original corrected thermal/reflection/radiance/LD fields remain piecewise
constant on angular parent faces. Root solving, adaptation and clipping remain
common CPU bridges; the original CPU reference and device kernels independently
integrate the same new rule. This is hybrid, with all bridges and transfers
charged in complete timings. Changing the field grid also changes the patch
basis; a matched dyadic leaf scale remains a coupled numerical comparison.

Run `test_boundary_geometry.py`, `test_gpu_boundary.py` and the preceding two
curved test modules. Reproduce complete curves with
`scripts/validate_gpu_boundary.py --device cpu --part curves --case partial_bb --field-levels 1 --order 3 --depth 4 --output curves.json`,
then repeat with a new depth/order/field protocol and `--comparison curves.json`.
Cases also include `total_bb`, `partial_ck04` and `small_occultor`.
`--part observations` evaluates fresh physical proposals through the original
serializer, constraints, priors, phase/JD mapping, reducers, third light, final
mean and noise/density callbacks. Both reference and selected callback timers
include physical serialization, constructor, prior, the same original
prediction layer and density. Numerical preflight and independent extra
controls are additionally charged in the complete study wall. Compilation
incurred is included; process caches can warm. These studies are not optimized
fits, converged chains or ESS measurements. GPU runs require independently
restored bounded allocations. See [current results](ada_progress.md).

## Shared quadrature for field and correction comparisons

`SharedFieldLightCurve` evaluates a fixed batch of fresh field models on the
**same clipped quadrature packets**. It is an opt-in diagnostic protocol; the
preceding curve engines, original Observer, public fitters and fallback routes
remain unchanged. There is no physical prediction cache across candidates.

```python
from elisa.experimental.boundary_geometry import BoundaryRule
from elisa.experimental.gpu_shared_fields import FieldVariant, SharedFieldLightCurve
from elisa.experimental.gpu_curved import curved_difference

variants = (FieldVariant(1, 1), FieldVariant(1, 2), FieldVariant(2, 2))
engine = SharedFieldLightCurve(
    bands, BoundaryRule(order=3, depth=3), variants,
    integration_level=2, device="gpu",
)
batch = engine(binary, observation_phases, observer=observer,
               source_revision=assets_sha256)
a, b, c = (batch.curves[v.key] for v in variants)
# Call each numerical preflight outside physical candidate-penalty handlers.
correction = curved_difference(a, b, nominal_y_err, require_budget=True)
field_grid = curved_difference(b, c, nominal_y_err, require_budget=True)
```

`FieldVariant(level, correction_level)` fixes the original finite field grid
and the argument of the original scalar mesh-correction formula independently.
Both levels are 0–2. The correction angle is seed_degrees/2**correction_level;
all raw topology, points and cached pre-correction gravity/centres/normals stay
at `level`. Every variant rebuilds subsequent original areas, thermal state,
irradiation, LD and radiance. Off-diagonal choices are correction ablations,
not calibrated new physical models or alternative priors. `FieldVariant(l,l)`
uses the preceding ordinary nested field convention.

The common angular grid is a fresh `NestedMesh(integration_level)`. Original
Roche projection places each edge midpoint on the same parent great-circle
edge, even when it is not the angular midpoint. Descendants partition their
parent cone in exact arithmetic. A direct 3x3 solve checks every child's corner
against its indexed ancestor with conditioning-scaled normwise FP64 guards and
a reconstruction check; condition above 1e5 aborts. This accounts for seam-axis
roundoff and is not an interval geometry certificate. An escaped parent is an
explicit error; no nearest-face reassignment or interpolated parent value is
substituted. Independent great-circle membership and exact spherical-triangle
solid-angle controls check the ancestry.

Each phase/component builds one true-Roche clipped packet. All variants use its
identical cosine, area and common-cell parent arrays, then map parents through
checked ancestry. A SHA256 digest includes the rule, integration level, phase,
component, node count and actual packet bytes. Reference/device runs independently
rebuild geometry and must produce identical hashes and adaptive traces. A field
or correction comparison inside one batch therefore holds boundary integration
fixed. Changing the common grid/depth remains a separate integration control.

The three-model example separates correction-only `a→b`, field-grid refinement
at fixed correction `b→c`, and the total `a→c`. Normalized changes telescope,
but their norm/enclosure distances need not add because components and phase
changes can cancel. The original third-light/max/mean conventions are retained.
The existing component enclosure and data-residual likelihood cross term bound
only computed finite differences. Gates use nominal uncertainty independently
of ln_f; failures abort outside prior penalties. A passing finite pair still
does not qualify an uncomputed field, boundary or continuous posterior.

The geometry and ancestry bridge remains CPU-side. Reference fields and element
flux use original CPU primitives; CPU/GPU backends use the preceding JAX state,
reflection, emission and integration kernels. CK04 explicitly selects
`radiance="cpu"`. Batch timings include constructors, every fresh field model,
common geometry/clipping/hash, all transfers and all raw/normalized/component
host curves. This is a three-model comparison output contract, not one ordinary
proposal or a claim of all-GPU synthesis. No fitting/sampler backend is registered.

`integration_level` is max(field levels) through 3, with the original 40000-face
common-grid cap. There are 1–4 unique variants, at most 20276 field faces per
component, 1–8 bands and 1–128 phases. The preceding boundary-rule/node caps and
circular detached q/spin/radius guards remain. Resource/unsupported failures
never silently coarsen a requested model or remove a physical prior region.
The node cap is not a peak-memory bound. `shared_area_moments` independently
reports original intrinsic T4 closure and the same temperatures on shared
order-4 curved areas; it applies no temperature renormalization and does not
certify reflected energy balance.

Run `test_shared_fields.py`. Reproduce complete reflected profiles with
`scripts/validate_gpu_shared_fields.py --device cpu --part curves --case partial_bb --integration-level 2 --depth 3 --output shared.json`.
Cases also include `total_bb`, `partial_ck04` and `small_occultor`.
Repeat at another common depth/grid using `--comparison shared.json`.
`--part observations --proposal base --integration-level 2 --depth 4` runs one
physical observation study; `changed` and `jd` select the other proposals so each
GPU study fits its independently restored bounded allocation. Both complete
comparison callbacks include original serialization/constraints, constructor,
prior, prediction mapping and density; both compute all three comparison curves
and return the fine curve through the original prediction layer. Extra coarse
boundary preflight and untimed same-candidate mapping oracles add to main wall.
This does not measure ordinary single-model MCMC throughput, mixing or ESS.
GPU execution requires the independently restored bounded pattern. See
[current finite errors, timings and remaining work](ada_progress.md).

## Curvature-consistent finite fields

`CurvatureFieldLightCurve` is a new opt-in numerical model. Every field cell's
collocation centre lies on the actual Roche equipotential, along the normalized
mean of its angular corner rays. Cell normals and SI gravity are evaluated at
that point. Order-4 Duffy quadrature integrates true curved area over the parent
patch, evaluating the original base quarter and mirroring it exactly. There is
**no scalar mesh correction**. Original CPU/default engines and fitters remain
unchanged; this new finite method is not an unchanged original faceted curve.

```python
from elisa.experimental.boundary_geometry import BoundaryRule
from elisa.experimental.gpu_curvature_fields import CurvatureFieldLightCurve

engine = CurvatureFieldLightCurve(
    bands, BoundaryRule(order=3, depth=3), levels=(1, 2),
    integration_level=2, device="gpu",
)
batch = engine(binary, observation_phases, observer=observer,
               source_revision=assets_sha256)
coarse = batch.curves["curvature_field1"]
fine = batch.curves["curvature_field2"]
```

Intrinsic temperatures use the original gravity-darkening law and original
unspotted renormalization, now on the curved cell areas. Point gravity uses
G*M_primary/a² for either component, matching the original force convention.
The device path composes the existing point-gravity and thermal kernels, retaining
fresh device arrays through tiled irradiation and blackbody emission. The force
kernel's repeated selector indices only select point forces; corner averages
are discarded and those indices never define geometric areas or normals.
Normals/areas/temperatures sent to irradiation all refer to this same new cell
geometry. This removes the preceding mixture of curved observer integration
with scalar-corrected faceted donor areas and cached planar field normals.

The independent reference evaluates original CPU gradient, thermal,
renormalization, reflection, LD, radiance and element-flux formulas on the new
geometry. Original full-donor sums, cooler-first iteration order, bolometric LD
prefix convention and visibility rules are retained. This is a finite
collocation approximation to irradiation, not a new continuum solution or an
independent proof of reflected energy balance. CK04 examples explicitly select
`radiance="cpu"` and preserve original table interpolation/unsupported semantics.

Fields remain constant within angular cells. Multiple requested field levels
use the same checked common quadrature described above. Original observational
phase/reducer/repeat, third-light, distance, normalization and noise semantics
remain. `curved_difference` measures finite field and boundary changes, with
nominal-noise gates outside candidate penalties. Even a passing distance gate
does not certify an uncomputed limit or a negligible accumulated log-likelihood
change. No curvature-field fitter/sampler backend is automatically registered.

Levels 0–3 and 1–4 unique field levels are supported, subject to the original
40000-face cap for each field/common component. `integration_level` must be at
least the maximum field level and at most 3. The preceding circular detached
unspotted/nonpulsating q/spin/radius, band/phase and boundary-node guards remain.
Resource or unsupported states abort explicitly, without coarsening or prior
cuts; broader original CPU/fallback routes stay available. Provenance records
`correction_level=-1` and `curvature_fields=1` to distinguish the method.

Actual roots, curved areas, ancestry, common clipping and original LD/table
preparation remain CPU bridges. Matched batch/callback costs include them,
construction, compilation incurred, every fresh field model, all transfers and
raw/normalized/component host outputs. They are comparison-batch costs, not
ordinary one-model MCMC proposal throughput. No physical prediction or packet
is cached across candidates; fixed code/response caches can warm.

Run `test_curvature_fields.py`. Reproduce complete contact curves with
`scripts/validate_gpu_curvature_fields.py --device cpu --part curves --case partial_bb --levels 1 2 --integration-level 2 --depth 3 --output curves.json`.
Other cases are `total_bb`, `partial_ck04` and `small_occultor`. Repeat field/common
levels or boundary depth using `--comparison curves.json`; only common field
levels are compared when batches differ. `--legacy-comparison` accepts a preceding
shared-field profile with identical physical/source phases and an identical
actual quadrature digest, reporting a **method change**, never parity.

For high-resolution original observation callbacks, use separately bounded
coarse/fine studies so the combined numerical preflight does not extend a GPU
window:

```text
python scripts/validate_gpu_curvature_fields.py --device cpu --part observations --proposal base --levels 2 3 --integration-level 3 --depth 2 --no-boundary-preflight --output coarse.json
python scripts/validate_gpu_curvature_fields.py --device cpu --part observations --proposal base --levels 2 3 --integration-level 3 --depth 3 --boundary-reference coarse.json --output fine.json
```

Use `changed`/`jd` for the other physical proposals. A coarse control that omits
its own boundary preflight is not a qualified fit. The paired fine study checks
source/proposal and original physical/phase signatures, then applies the original
finite component/likelihood enclosure using the saved complete coarse curve.
Count **both** study walls and both sets of callbacks; each GPU invocation needs
its own independently restored allocation. The same-candidate mapping oracle is
untimed as fresh synthesis. No optimized physical fit, chain convergence or
ESS/sec claim follows from these diagnostic studies. See [current results](ada_progress.md).

## Curved eclipse edges and accumulated likelihood controls

`CurvedEdgeLightCurve` keeps the preceding fresh curvature-consistent fields and
original reflection/radiance formulas, while changing the finite observer
integration rule. `SharedFieldLightCurve` now has a protected packet hook whose
default is exactly the preceding chord routine. Original CPU/default engines,
physical priors and broader CPU/fallback routes remain unchanged.

```python
from elisa.experimental.gpu_curved_edges import (
    CurvedEdgeLightCurve, CurvedEdgeRule, InferenceBudget,
)

engine = CurvedEdgeLightCurve(
    bands, CurvedEdgeRule(order=4, depth=2, slice_order=8,
                         junction_depth=3, singular_depth=3),
    levels=(2, 3), integration_level=3, device="gpu",
)
batch = engine(binary, phases, observer=observer, source_revision=assets_sha256)
coarse, fine = batch.curves["curvature_field2"], batch.curves["curvature_field3"]
InferenceBudget().compare(coarse, fine, errors, observed=data,
                          reducers=observation_reducers, mean_normalize=True)
```

That comparison is permitted to **abort**. Neither the class nor the example
automatically registers a fitter/sampler backend or certifies a physical model.
Every GPU invocation needs a bounded registered unit with the current allocation-policy checks.

For a clipped triangle with boundary endpoints a,b and visible vertex c, let
E(s)=(1-s)a+s b. Solve the original implicit limb/shadow for
C(s)=E(s)+lambda(s)(c-E(s)) at each Gaussian slice. The interior mapping is
v=(1-t)C+t c. Its solid-angle Jacobian is
abs(det(a,b,c))*(1-t)*(1-lambda)**2/abs(v)**3: the lambda-prime term cancels.
The rule needs no numerical contour derivative or signed strip correction.
True Roche points, normals and surface Jacobians are evaluated at these nodes.
Positive finite weights and source-cone membership are checked. Both the base
contour points and interior nodes must stay in the original angular leaf.

Limb/shadow junctions receive a prescribed additional subdivision before finite
chord integration. A fan without a supported bracketed boundary graph is also
explicitly counted: reintegrate its **whole original source leaf** with the
prescribed finer chords, removing all earlier fragments of that source to
prevent overlap or a lost crescent. This treatment is part of the declared
finite rule, not a hidden physical fallback. Exact zero endpoints are preserved;
only exact duplicate/zero-measure fragments are removed. No positive patch is
dropped by a tolerance. Junction/singular depth controls remain independent of
ordinary boundary depth and slice order; their residual error needs validation.

`order` retains 2–4, `slice_order` is 4–12 (default8), edge bisections remain
32–48 and curve bisections are 36–48 (default40). Junction/singular depths are
0–4 (default3). Existing phase/band, field/common level, face, node and domain
caps remain. Slice and singular allocations have explicit node guards; the
2,097,152-node cap is not a bound on compiler/allocator peak memory. Original
bounded Roche shadow conditioning, empirical topology probes and resource
errors remain. No interval topology or uncomputed-continuum certificate is given.

CPU reference and GPU execution use identical fresh packets/hashes and compare
all raw, component and normalized curves under unchanged implementation gates.
Independent projected sphere/circle-intersection controls test the boundary
method separately. Single-edge accuracy does not imply irradiation/field
convergence, reflected-energy closure or observational adequacy.

`InferenceBudget` adds a separate opt-in finite-pair diagnostic. It retains
nominal sigma and D<=.1, and also requires the data-dependent absolute Gaussian
log-likelihood enclosure sum(abs(r)*E+E**2/2)<=.01 by default. This bounds a
computed pair's **nominal diagonal-Gaussian** likelihood ratio by exp(±.01),
about1%; it is a predeclared project diagnostic, not a universal standard or
posterior-bias guarantee. Nuisance ln_f is never used to pass either gate.
Failure raises `ResolutionNotQualifiedError` outside candidate penalties,
including with PENALIZE_ALL_EXCEPTIONS enabled. Original fitters/defaults are
unchanged. Correlated noise, physical model adequacy and uncomputed limits
require separate analysis.

Reproduce with `test_curved_edges.py` and
`scripts/validate_gpu_curved_edges.py`. It retains the preceding complete curve
cases and original physical base/changed/JD callbacks. `--slice-order`,
`--junction-depth` and `--singular-depth` make independent controls explicit.
`--comparison` checks common field levels; `--chord-comparison` labels a preceding
curvature-field/chord result as a boundary-method comparison. Use separate
`--no-boundary-preflight` and `--boundary-reference coarse.json` invocations for
paired high-resolution likelihood studies. Count both complete callbacks and
both study/command walls, including construction, all CPU bridges, compilation
incurred, transfers and host outputs. They remain two-field comparison-batch
costs, not ordinary single-model MCMC throughput.

[Realistic analysis workloads](analysis_workloads.md) separately retain the
verified current TESS products, original/JAX CPU and four-process CPU baselines,
preliminary ellc fits and their failed finite refinement checks. Their new
standalone CPU benchmark does not initialize CUDA. No real-data GPU, physical
fit qualification, converged-chain or ESS/sec result is inferred from them.


## Batched resident field geometry (m018)

`BatchedRocheGeometry(component, device="cpu"|"gpu")` accepts `[q, spin, Phi]`
rows and shared or changed per-row rays/triangles. `sample(..., compact=True)`
returns points/normals/dA-dOmega/validity; the rich result retains residuals and
explicit status masks. `areas(..., order=2|3|4)` returns patch areas/reductions.
All remain resident until explicitly consumed/downloaded. `checked_host` raises
on an invalid row. A root/domain/resource failure is never a prior rejection.
FP64 and the original circular detached/unspotted/nonpulsating inner-branch
q[.5,2],spin[.5,1.5],radius[1e-6,.25] domain remain mandatory. The cap is total
padded quadrature nodes across proposals<=2097152, not per proposal or peak memory.

```python
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule, InferenceBudget
from elisa.experimental.gpu_resident_geometry import ResidentGeometryLightCurve

engine = ResidentGeometryLightCurve(
    ["Generic.Bessell.V", "bolometric"], CurvedEdgeRule(4, 1),
    levels=(1, 2), integration_level=2, device="gpu",
)
batch = engine.batch(binaries, phase_arrays, source_revision=asset_revision,
                     observer=observer)
# Each result contains complete host curves for both finite field levels.
# Before accepting a likelihood, apply the unchanged InferenceBudget to both
# curves using actual observation reducers, mean normalization, errors and data.
```

Only run GPU examples inside an authorized registered bounded allocation.
`max_resident_faces` (default524288) declares the full-field state wave budget,
1..2097152; an oversized single candidate raises before allocation. Shape groups
preserve actual fresh topology; all proposals run, without frozen mesh, cached
prediction or silent coarsening. Field levels0..3/max40000faces per component,
phase<=128,band<=8 and existing boundary/node guards stay active. The model is
finite curved-cell physics from m017; the independent `device="reference"`
path keeps the original CPU primitives, not the original faceted whole model.

One compiled graph consumes field geometry directly into SI gravity and thermal
state. Existing tiled irradiation and fresh device BB remain connected. Original
CPU angular topology/nested vertices, observer geometry/shadow/adaptation/clipping,
ancestry, hashing and LD/table bridges remain explicit, including necessary host
state copies. CK04 radiance stays CPU. This is a hybrid. The batch timer encloses
all planning, transfers, synchronization, complete synthesis and host outputs;
per-row timers exclude shared field preparation. Single `__call__` replaces its
stage total with the full enclosing singleton batch total.

Reproduce controls with `test_batched_geometry.py`, `test_resident_geometry.py`
and `ELISA_GEOMETRY_TEST_DEVICE=gpu` inside the allocation (missing GPU fails;
no skips). `benchmark_batched_geometry.py` retains original serial CPU,
vectorized NumPy and selected JAX CPU/GPU outputs, changing physical/ray inputs,
first calls, ten warm repetitions, nested timers and memory accounting.
`validate_gpu_resident_geometry.py --backend reference|cpu|gpu --batch 1|4`
retains every complete original-mapped prediction and density with strict accuracy
controls. `--jd` tests original time mapping. These are two-field comparison
batches, not qualified ordinary MCMC likelihoods. All CPU/default fitters and
explicit full-CPU fallback routes remain unchanged. See the current progress
report and hardware ledger for measured limitations and reproducible commands.


## Guarded observer device path (m019)

`ObserverGeometryBackend(device="cpu"|"gpu")` binds each actual binary/component,
with reusable compiled shape buckets and fresh physical parameters/LOS. Pure
shadow kernels support(B,N,3)points,(B,3)parameters and(B,3)LOS rows. The complete
adapter still runs candidate/phase CPU controllers. `ObserverDeviceLightCurve`
extends m018's full pipeline without changing its physical finite model:

```python
from elisa.experimental.gpu_observer_edges import ObserverDeviceLightCurve
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
engine = ObserverDeviceLightCurve(
    ["Generic.Bessell.V", "bolometric"], CurvedEdgeRule(3, 1),
    levels=(0, 1), integration_level=1, device="gpu",
)
batch = engine.batch(binaries, phase_arrays, source_revision=asset_revision,
                     observer=observer)
# Apply InferenceBudget to actual observation mapping before accepting inference.
```

GPU calls require an authorized registered bounded allocation on Ada. Reference
mode retains original CPU finite observer/field primitives. Device mode ports
80-step rays (unroll8),12-step conditioned stationary shadows and nested edge
prefixes. CPU topology/clipping/cone/singular guards, the last8root decisions,
final bracket checks and counted precision-seam replays remain.128eps is a
trigger to recompute in original arithmetic, not an integration tolerance or a
permission to drop measure.64eps stationary,3e-13implementation gates and all
physical/resource/FP64 caps remain unchanged. Unsupported cases abort outside
priors; original full-CPU fallback routes remain unchanged.

`last_observer_stats` records node/padding/work counts and enclosing sample/root
costs, including CPU tails/replays. The inherited CPU-named observer timer spans
mixed device/CPU work in this path. The curated report renames it to avoid CPU
attribution; do not reuse m018CPU-fraction estimates after this port. Packet
bytes can differ; strict adaptive traces and full flux/likelihood gates remain.

Controls: `test_observer_geometry.py`, plus affected batched/resident/curved/
shared-field regressions. Set`ELISA_GEOMETRY_TEST_DEVICE=gpu` in the protected
unit; missing GPU fails rather than skips. `benchmark_gpu_shadows.py` compares
original serial CPU, vectorized NumPy and JAX CPU/GPU with changing physical rows
and points, ten warm repeats and matched outputs. `validate_gpu_observer.py`
measures complete original-mapped likelihood batches; `--levels 2 3
--integration-level 3`keeps finer field diagnostics explicit, and `--jd`checks
original time mapping. Numerical fitting/MCMC qualification remains blocked by
the existing1e-4finite field/boundary checks, separately from implementation parity.

## Native nested observer roots and a joint finite inference budget

`elisa.experimental.gpu_fused_observer.FusedObserverLightCurve` is an additional
opt-in execution path for the current curved-intersection algorithm. Its public
complete-curve contract matches `ObserverDeviceLightCurve`; the earlier API and
all original CPU/default behavior are unchanged. See the current
[Ada report](ada_progress.md) for validation, actual transfer-inclusive costs and
remaining scientific limits.

The GPU edge prefix runs its outer bisections, all 80 inner Roche bisections and,
for a shadow, all 12 stationary Newton updates inside native per-tile loops.
Pallas/Triton grids also accept changing physical-row and line-of-sight batches.
The complete CPU controller still visits individual candidates and phases.
Explicit CUDA `add/sub/mul/div/sqrt.rn.f64` operations specify the scalar stage
rounding and prevent contraction across those stages. Cartesian scalar-axis
expressions and `r2*r2*sqrt(r2)` in the stationary derivative are mathematically
the original equations with a different FP64 evaluation tree. They are validated
against original trigonometric/NumPy arithmetic; bitwise equality of physical
arrays is not presumed. Existing FP64 barriers and defaults remain intact.

Before a value reaches the unchanged 128-eps ambiguity/branch guard, the native
lane saves its bracket and iteration **before making that decision**. Original
CPU arithmetic resumes the remaining decisions from there. Otherwise it runs
the predeclared last two decisions on CPU. Complete-curve validation uses the default two CPU decisions; zero/eight-tail
controls test primitive brackets. Original CPU checks both final bracket
ends and replays the full original solve if they disagree. Tail counts exclude
final bracket checks and full-replay decisions; replay rows are recorded
separately, and enclosing timings include all of this work. This retains the
original finite protocol without classifying a small region by a tolerance or
reusing a changed proposal's prediction. The original endpoint, branch,
stationarity, positive-weight, topology, cone, singular-leaf and resource guards
remain active. Unsupported states and numerical-accuracy failures remain outside
physical-prior penalties.

The native backend was tested with the pinned JAX/jaxlib/CUDA plugins 0.12.0.1
on Ada's NVIDIA GPU. It needs the CUDA Pallas/Triton backend; it neither installs
a compiler nor silently selects a CPU. That JAX backend emits a deprecation
warning, so future JAX upgrades need a separate compiler migration and validation.
`device="cpu"` uses rolled JAX observer graphs plus the same checkpointed CPU
continuation, providing a separate JAX CPU baseline. `device="reference"` retains
the original CPU equations. Original explicit full-CPU physical-routing options
remain available for cases outside the experimental circular, detached,
unspotted inner-branch domain. This new class does not broaden that domain.
CPU angular topology, clipping/adaptation, LD/table bridges, some observer
sampling and precision decisions remain; this is a hybrid forward model.

```python
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_fused_observer import FusedObserverLightCurve

engine = FusedObserverLightCurve(
    ["Generic.Bessell.V", "bolometric"], CurvedEdgeRule(3, 1),
    levels=(0, 1), integration_level=1, device="gpu",
)
# Actual binaries/phases must be fresh for changing physical proposals.
batch = engine.batch(binaries, phases_per_binary, source_revision=asset_revision)
```

`elisa.experimental.gpu_joint_budget.JointInferenceBudget` adds a common-anchor
control for multiple predeclared finite field/boundary comparisons. Every pair
uses the original observation reducers, normalization, third light and nominal
uncertainties. If pair i has pointwise component envelope E_i, the joint control
uses E=sum(E_i), D=norm(E/sigma), and
sum(abs(data-anchor)*E/sigma**2 + .5*(E/sigma)**2). Quadratic cross terms between
axes are included. Default gates remain D<=0.1 and absolute log-likelihood
bound<=0.01, independent of nuisance-noise inflation. The raised
`ResolutionNotQualifiedError` remains an accuracy stop.

This controls only the sum of the **computed finite perturbations** about the
common anchor. It does not certify an uncomputed combination, cross-axis
interaction, continuum model, posterior bias or observational adequacy. Combined
resolution changes still need explicit tests. An analytic Gaussian control shows
that individual axes can pass while their joint enclosure fails. Physical
fitting/MCMC must retain the other resolution controls; a joint passing value
alone is not qualification. The current intersection method is the numerical
path under development; accurate photometry can also use a suitably converged
fine triangle mesh. No rendering redesign is part of this milestone.
