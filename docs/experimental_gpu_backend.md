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
