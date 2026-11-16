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
