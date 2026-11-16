# Ada GPU development progress

Updated **2026-11-16T22:00:46.876148+00:00**. Work in progress on
`codex/gpu-resident-roche-pipeline`, based on `experimental/gpu-roche-rays`
commit `7d719ca09545d4c62a2e7e30f5e858e6c0d82eec`. Nothing has been merged.

## Current work and validation

- Effective SI surface gravity on changing Roche vertices, preserving secondary
  coordinates and CPU pre-correction build order.
- Device-resident Roche rays → triangle geometry/projection → gravity. Static
  directions/topology upload once; proposal parameters upload per batch. Stage
  barriers preserve FP64 rounding without moving intermediates to the host.
- Intrinsic gravity-darkened temperatures and emitted bolometric surface flux,
  with independent energy closure. This is not observed light or an atmosphere model.
- **112 passed in 23.47 s, zero failures/skips** on the actual GPU, including CPU/GPU
  cases, real Roche/eclipses, temperatures, unsupported masks and strict transfer guards.
- CPU-only run: **71 passed, 41 GPU cases intentionally skipped**. Ruff and
  whitespace checks passed. All larger benchmark comparisons passed unchanged tolerances.

The first larger CPU benchmark found one near-limb projection difference of
about `1.57e-14` outside the existing tolerance. The run stopped and restored
production. Explicit FP64 stage boundaries fixed it; 128/512-proposal CPU/GPU
regression tests now cover both stars. No tolerance was relaxed.

## Measured geometry/gravity performance

RTX 3080 Ti 12 GiB, driver 595.99.02, official JAX/jaxlib/CUDA13 0.12.0.1,
FP64, GPU preallocation disabled. Ryzen 5950X CPU; BLAS/OMP/MKL thread counts 1.
256 fixed angular directions, 508 triangles per star, changing q/spin/potential,
10 warm repeats. All times below are median **milliseconds per batch**.

| Star | Proposals | Original ELISa CPU | Batched JAX CPU full | GPU staged full | GPU resident path full | GPU device work only | GPU download only |
|---|---:|---:|---:|---:|---:|---:|---:|
| primary | 1 | 0.504 | 0.586 | 2.640 | 1.628 | 0.646 | 0.553 |
| primary | 8 | 3.934 | 2.138 | 2.907 | 1.814 | 0.684 | 0.605 |
| primary | 32 | 15.289 | 5.411 | 3.056 | 1.853 | 0.743 | 0.691 |
| primary | 128 | 61.348 | 31.197 | 3.852 | 2.468 | 1.021 | 1.021 |
| primary | 512 | 242.976 | 96.290 | 7.647 | 5.690 | 2.469 | 6.239 |
| secondary | 1 | 0.528 | 0.575 | 2.464 | 1.525 | 0.614 | 0.521 |
| secondary | 8 | 4.085 | 1.836 | 2.647 | 1.578 | 0.672 | 0.556 |
| secondary | 32 | 15.794 | 5.214 | 3.142 | 1.810 | 0.754 | 0.668 |
| secondary | 128 | 62.429 | 26.595 | 4.055 | 2.553 | 1.097 | 1.029 |
| secondary | 512 | 256.698 | 104.459 | 8.402 | 5.774 | 2.700 | 6.189 |

“Full” includes per-batch host validation, parameter H2D transfer, physics and
identical final D2H outputs (radii, areas, normals, LOS cosines, SI gravity and
validity). Static topology/runtime initialization and compilation are separate.
The staged path includes the old intermediate vertex host bridge. The device-only
column is a lower bound, not an application timing. Download-only samples use
fresh device results to avoid JAX host caching; independently sampled medians
must not be added/subtracted as exact stage decomposition. These output-rich
measurements transfer about 13.5 MB for 512 proposals per component.

At 512 proposals the full resident GPU path is 1.34–1.46× faster than the staged
GPU path and about 43–44× faster than original CPU helpers in this narrow workload.
Single proposals remain faster on CPU. This is not a whole-light-curve, fit,
MCMC convergence or effective-sample-size speedup claim.

GPU resident versus staged outputs were identical in all measured fields.
Against original CPU helpers, maximum radius relative error was `9.91e-16`,
area relative error `1.98e-14`, SI gravity relative error `5.02e-15`, normal
absolute error `2.44e-14` and LOS absolute error `1.99e-14` (relative errors
near zero LOS cosines are not useful). All masks and tolerances passed.
Current intrinsic-temperature correctness passed on GPU; its bounded performance
benchmark is next and has no timing claim yet.

## Production restoration

The successful window ran physics/tests from 21:56:11 to 21:57:24 UTC.
Fifth42 stopped through its supported checkpoint handler and resumed the exact
source/config/plan-bound production service. Job `j00875` advanced
from saved cursor `949153360` to
`949196368`; durable cells advanced from
`1090897207` to `1090940215`.
Restoration was verified at 2026-11-16T21:58:00.863961+00:00. No GPU resets, driver
changes or changes to other machines. Every window remains bounded to five
minutes with restoration on success or failure.

## Review priorities for DOT

Review state/build order, unsupported masks, secondary gravity units,
temperature reference-gravity cancellation, energy conservation, FP64 boundaries
and final-transfer accounting. Candidate-specific angular placement/topology,
mesh correction, irradiation/reflection, atmosphere/LD updates, changing eclipse
coverage and full normalized-light-curve/posterior validation remain unfinished.
Changed physical parameters must not reuse stale coverage or brightness.

Evidence: [machine-readable progress](ada_validation/progress.json),
[GPU test output](ada_validation/window-02-tests.txt),
[numerical comparisons](ada_validation/window-02-parity.json),
[GPU benchmark samples](ada_validation/window-02-benchmark-gpu.json),
[CPU benchmark samples](ada_validation/window-02-benchmark-cpu.json),
[current CPU output](ada_validation/cpu-current-tests.txt),
[prior GPU output](ada_validation/window-01-tests.txt), and
[tested source fingerprints](ada_validation/source-hashes.json).

I will update this report after each validation cycle. The existing
`experimental/gpu-backend` benchmark branch remains untouched.
