# Ada GPU validation handoff (experimental, no production replacement)

## Safety and setup

Do not interrupt existing jobs, reset a GPU, kill processes, change drivers,
or install into an existing shared Python environment. Inspect `nvidia-smi`
first and use the cluster's normal scheduler/reservation mechanism. If all GPUs
are occupied, report that and wait; do not assume free memory means permission.
Use a new checkout/worktree and private virtual environment. Do not merge this
experimental branch into production. The user uses a non-bash shell; commands
below do not require `export`, activation, or bash syntax.

Required checkout: branch `experimental/gpu-backend` in `paulnord/elisa`. Install project dependencies in an isolated
environment. Use the official JAX installation instructions to select a CUDA
wheel compatible with Ada's driver; do not upgrade system CUDA or the driver.
The package's `experimental-jax` extra alone does not install CUDA support.
Set JAX FP64 with the shell-independent `env` command below. ELISa may need
`ELISA_CONFIG` pointing at an existing valid config; do not alter shared settings.
Blackbody/custom-LD fixtures do not require downloaded atmosphere tables.

## Checks to run

Replace `python` with the isolated environment's interpreter. Run from repository root:

```
env JAX_ENABLE_X64=True python -c 'import jax; print(jax.__version__); print(jax.devices("gpu")); print(jax.config.x64_enabled)'
env JAX_ENABLE_X64=True python -m pytest tests/tests_experimental -q -rs
env JAX_ENABLE_X64=True python scripts/benchmark_gpu_likelihood.py --device gpu --steps 100
env JAX_ENABLE_X64=True python scripts/benchmark_gpu_likelihood.py --device cpu --steps 100
```

Record GPU model, driver, Python/JAX/jaxlib versions, selected device, commit,
full test output, skipped-test reasons and benchmark JSON. All GPU surface
parity tests must run rather than skip. No speedup claim from these smoke runs.
100 MCMC steps are not convergence evidence. Do not submit long runs without a
clear time/resource budget and available scheduler allocation.

## What is implemented

- Fixed-physical-binary two-parameter batched likelihood (additional light and
  existing ln_f noise inflation), with CPU emcee and device likelihood batches.
- Actual ELISa prepared per-face flux integration on the device: radiance,
  line-of-sight projection, covered surface area, linear/cosine/logarithmic/
  square-root limb darkening, padded heterogeneous batches, row reduction.
- CPU and GPU tests capture real Roche binary surface containers at eclipse
  phases, compare every selected face and component total to CPU ELISa.

## What is NOT implemented

Roche mesh construction, changing geometry, eclipse coverage, atmosphere and
LD interpolation, temperature/reflection iteration, and sampler transitions
remain CPU-side. The surface kernel is a standalone experimental API, not yet
wired into the production Observer or general fitting task. No full GPU MCMC
backend, hardware result, convergence result or acceleration is claimed.

## Acceptance and next milestone

Investigate any per-face/totals discrepancy before performance work; don't
loosen tolerances simply to get green tests. Current FP64 tolerance is 3e-13
relative for real surface arrays. Compare normalized curves and accumulated
log-likelihood error to the target observations before scientific production.
GPU reduction order need not be bitwise equal. Next integrate the surface
batch API into a bounded synthesis path without per-face host/device round
trips, then port changing geometry and occultation with contact-focused parity.
