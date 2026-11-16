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

## Follow-on surface scaling test (only when production permits)

The first Ada report at `f661f004` was 31 tests passed on an RTX 3080 Ti
(JAX CUDA, FP64). Reported face/total relative errors were 2.08e-16/3.72e-16,
and likelihood difference 2.18e-11. The cached likelihood was slower on GPU:
0.170 ms versus 0.094 ms CPU. These are user-reported measurements, not evidence
of full-model acceleration. Production resumed; do not interrupt it.

The next benchmark uses real prepared Roche faces, repeating rows only to
measure throughput. It compares original per-star CPU integration, padded
NumPy, JAX transfer-inclusive integration and device-resident lower-bound
integration separately. Full-path timing includes original face indexing,
padding, host/device transfer, kernel and row-total download. CPU physical
synthesis/capture and cold JIT are reported separately. No MCMC/ESS speedup is
inferred: the existing l3/ln_f pilot should cache its integrated curve instead
of repeatedly integrating the same unchanged surfaces.

Only within an authorized idle allocation, run these shell-independent commands
in the private environment, first with the small default workload:

```
env JAX_PLATFORMS=cpu JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/benchmark_gpu_surface.py --device cpu --rows 8 32 128 512 --repeats 10
env JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/benchmark_gpu_surface.py --device gpu --rows 8 32 128 512 --repeats 10
```

Then, if within that allocation's resource/time budget, repeat with
`--discretization 5` and each `--law logarithmic` / `--law square_root`.
Preallocation must be disabled **before Python starts**. The 256 MiB input cap
is NOT a total GPU memory bound: outputs, runtime allocator and compiler use
additional memory. Follow scheduler limits; stop rather than contend with
production. Record all JSON, environment and allocation details. Do not tune
down scientific precision to obtain a speedup.

## Updated authorization: temporary fifth42 pause for this test

Paul explicitly authorizes Ada Codex to temporarily pause the current `fifth42`
production run, run this ELISa test, and restore production afterward. This
supersedes the earlier instruction to wait for idle time only for that named run
and this bounded test. No other job may be stopped, no GPU reset, driver change,
or reboot is authorized. The coordinating assistant does not act on Ada directly.

Before changing anything:
1. Identify the exact fifth42 job/process, owner, scheduler state, command,
   working directory, log/checkpoint path, GPU assignment, and current progress.
   Record these privately as the restoration checklist; do not expose credentials.
2. Establish its supported pause/checkpoint/resume mechanism. Prefer the existing
   production runner/scheduler's documented method. A generic SIGSTOP often
   retains CUDA allocations and is not proof that GPU memory was freed. Do not
   kill an uncheckpointed process or invent a restart command. If reliable
   restoration is not available, stop and report that specific blocker.
3. Pause only that confirmed job and verify its state. Check memory/device access
   and other running jobs. If pause retains too much memory, do not reset the GPU
   or terminate other work; use a proven checkpoint/restart path only if it
   preserves progress, otherwise report the blocker and restore production.

Run an initial bounded test in the isolated environment: the experimental tests,
then surface benchmark CPU/GPU `--rows 8 32 128 512 --repeats 10` at the default
10-degree mesh, linear law. Keep FP64 enabled, GPU preallocation disabled, and
CPU comparisons forced to `JAX_PLATFORMS=cpu`. Allow at most five minutes for
this initial suite/benchmark batch after environment setup. Cancel **only the
ELISa test process** if the budget is exceeded. Additional meshes/laws can wait
for results; do not leave production paused for open-ended investigation.

In a guaranteed cleanup/finally path, even if install, compilation, tests or
benchmark fail: stop any remaining ELISa test processes, restore fifth42 using
the verified original mechanism, and verify actual forward progress beyond the
recorded baseline (new log/checkpoint/step, not just a surviving PID). If pause
failed before taking effect, verify it remained running rather than launching a
second copy. Report the original and restored job identity, pause/resume times,
progress evidence, tested commit, hardware/versions and results. Do not report
success until fifth42 is confirmed making progress again. If restoration fails,
alert Paul immediately with the exact state and safe next action.
