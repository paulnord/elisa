# Next geometry-stage handoff: separate experiment

Branch: `experimental/gpu-roche-rays`. Keep the existing `gpu-backend` benchmark
checkout/environment/job untouched. This branch adds new geometry primitives,
not a production ELISa replacement. Do not run concurrently on a busy GPU or
extend a production pause silently. Wait for the current benchmark to finish
and production restoration to be verified; coordinate a separate approved test
window before another GPU run. Existing checkpoint/resume and allocation safety
rules in `ada_gpu_handoff.md` still apply, but this note is not new authorization
to pause production again.

## What changed

- `RocheRaySolver`: real batched primary/secondary radial potential roots with
  changing q, synchronicity and surface potential, bounded to small interior
  rays. Unsupported/ambiguous statuses are not physical prior rejections.
- `TriangleGeometry`: triangular centroids/areas/outward normals and observer
  projection using actual changed vertices and supplied connectivity. Near-limb
  mask ambiguity and degenerate/nonfinite geometry are explicit.
- CPU parity uses actual ELISa potential, first-inner-radius, mesh and face
  functions plus built binary containers and `move_sys_onpos`, not toy curves.

## Validation in the approved window

Use a separate checkout of this branch and record its exact commit. Reuse only
an appropriate private Python environment; don't alter the active production
or earlier benchmark environment. Run CPU checks without initializing the GPU:

```
env JAX_PLATFORMS=cpu JAX_ENABLE_X64=True python -m pytest tests/tests_experimental/test_gpu_roche.py tests/tests_experimental/test_gpu_mesh.py -q -rs
```

During the approved GPU allocation, preallocation disabled before startup:

```
env JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest tests/tests_experimental/test_gpu_roche.py tests/tests_experimental/test_gpu_mesh.py -q -rs
```

GPU cases must actually run, not skip. Report hardware/driver/JAX versions, exact
commit, full test output and failures; do not relax tolerances just to pass.
Allow five minutes for this focused validation, not an open-ended performance
study. Restore and verify production according to the explicitly approved
window's plan even if a test fails. No speedup is claimed by this stage.

## Not yet implemented

Complete candidate-specific angular grid/topology and mesh corrections, changing
eclipse coverage, gravity/temperature/reflection/atmosphere/limb-darkening state,
full normalized light-curve/posterior parity, and device-resident MCMC. The
ray-to-triangle test currently crosses the host; it is not a fused GPU forward
pipeline. Never combine changed geometry with stale coverage or brightness.

## Local follow-on: effective surface gravity

The development addition `gpu_gravity.py` evaluates ELISa's potential gradient,
SI corner gravity and corner-averaged face gravity. It preserves the CPU's
pre-mesh-correction gravity build stage and secondary scaling convention.
Include `tests/tests_experimental/test_gpu_gravity.py` in the focused commands
above when validating this addition. Its parameterized CPU/GPU tests include
real changing Roche vertices, actual CPU gravity containers, independent scalar
potential differentiation, polar forces, units and unsupported-state handling.
This addition grants no new production-pause authorization. Keep GPU validation
within a separately approved window and verify Fifth42 restoration afterward.

## Ada validation, 2026-11-16

The separate branch at `7d719ca09545d4c62a2e7e30f5e858e6c0d82eec`, plus the
local surface-gravity addition, passed all 84 experimental tests on Ada's
RTX 3080 Ti with JAX/jaxlib 0.12.0.1, FP64 and preallocation disabled. No GPU
case skipped. This includes the 30 original ray/mesh cases, 37 earlier
likelihood/surface cases and 17 gravity cases. Ruff and whitespace checks passed.

Maximum tested GPU radius error versus independent CPU Brent roots was
1.67e-16 SMA (1.11e-15 relative). The mesh/projection comparisons had maximum
absolute error 6.50e-15; near-zero quantities retain their original absolute
tolerances. New gravity comparisons had maximum relative error 5.39e-16.
These are measured test results, not universal numerical error certificates.
No performance or scientific-posterior conclusion follows from this validation.

Paul separately authorized the GPU pause. Fifth42 checkpointed cleanly and
resumed job j00858 at cursor 876310352; durable progress beyond that checkpoint
was verified before reporting success. All evidence and the reviewable local
patch are in `/home/pnord/elisa-roche-dev-20261116T201751Z/`.

## Resident pipeline performance experiment

The local follow-on `gpu_pipeline.py` joins the existing three physics kernels
without a vertex D2H/H2D bridge. Include `test_gpu_pipeline.py` in the parity
suite. Its GPU transfer-guard case must execute, and the existing physics
tolerances must remain unchanged. Do CPU development while production runs.
In an authorized bounded GPU window, run the full experimental suite followed
by the two benchmark commands below, with cleanup restoring Fifth42 on any exit:

```
env JAX_PLATFORMS=cpu JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/benchmark_gpu_pipeline.py --device cpu
env JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/benchmark_gpu_pipeline.py --device gpu
```

Defaults are 256 directions, batches 1/8/32/128/512, ten repeats, both stars.
The payload estimate excludes compiler/allocator workspace. Bound the complete
suite and benchmark batch to five minutes and cancel only ELISa processes if
needed. Report both the original CPU helpers and the batched JAX CPU baseline;
compare matching full result downloads, not just the resident GPU lower bound.
The current session explicitly authorizes checkpointed Fifth42 pauses as needed;
this document alone grants no production authority in another session.
