# Ada GPU development progress

Updated **2026-11-16T22:17:16.273566+00:00**. Review branch: `codex/gpu-resident-roche-pipeline`.
Based on `experimental/gpu-roche-rays` commit
`7d719ca09545d4c62a2e7e30f5e858e6c0d82eec`. **Nothing merged.**

## Current implementation

- Actual Roche-ray roots, changing q/spin/potential, triangle geometry/projection
  and effective SI gravity stay on device. Directions/topology upload once;
  immutable proposal snapshots upload only their physical parameter arrays.
- Fresh intrinsic gravity-darkened temperatures preserve the CPU formula,
  secondary units and unsupported masks; independent energy closure is tested.
- Fresh continuous-temperature **B, V and bolometric emitted surface flux** uses
  the actual CPU passband grids, Akima response and Simpson endpoint convention.
  Wavelength streaming bounds the spectrum temporary; no stale brightness template.
- Core, thermal and spectral benchmarks compare original ELISa CPU helpers,
  batched JAX CPU, staged GPU with host bridges and the device-resident path,
  downloading matching final results in every full path.

## Tests and numerical agreement

**130 passed in 27.41 s on Ada, zero failures/errors/skips.** This includes
CPU/GPU cases, real Roche surfaces, original eclipse fixtures, effective gravity,
temperature/energy closure, actual CPU Planck/passband parity, tiny temperature
changes, immutable snapshots, invalid-state propagation and strict GPU transfer
guards. The latest isolated blackbody CPU checks passed **10**, with **8** GPU
cases intentionally skipped. The preceding full CPU-only checkpoint passed
**79**, with **47** hardware cases intentionally skipped; two subsequent test
pairs (energy limit and response snapshot/clipping) passed in the targeted CPU
run and the full hardware run. Ruff and whitespace checks passed.

The larger initial benchmark found one near-limb projection difference of
`1.57e-14` just outside an existing tolerance. That window stopped and restored
production. Explicit FP64 stage barriers fixed it while retaining all intermediates
on device. CPU/GPU regression tests cover 128/512 proposals for both stars.
**No existing tolerance was relaxed.**

At the measured geometry batches, GPU/staged results were identical. Original
CPU differences were at most `9.91e-16` relative in radii, `1.98e-14` in areas,
`5.02e-15` in gravity, `2.44e-14` absolute in normals and `1.99e-14` absolute in
LOS cosines. Near-zero projection relative errors are not informative.
At the thermal benchmark, temperature relative error was below `8.60e-16` and
emitted bolometric flux below `3.28e-15`. At the spectral benchmark, passband
flux relative error was below `4.43e-15`; resident/staged outputs were identical.
These are measured examples, not universal error certificates.

An independent infinite-band Planck integral also checks the finite bolometric
grid to `1e-6` relative across the tested 3000–40000 K range. This separate physical
grid-accuracy test retains existing CPU truncation/quadrature error; CPU/GPU
spectral parity remains `3e-13`. The legacy Stefan constant used in the temperature
law has not been changed to force the two different prescriptions to coincide.

## Hardware and timing contract

RTX 3080 Ti 12 GiB; driver 595.99.02; official PyPI JAX/jaxlib/CUDA13 0.12.0.1;
FP64; **GPU preallocation disabled**. Ryzen 5950X CPU, BLAS/OMP/MKL threads 1.
CPU/GPU comparison windows ran with Fifth42 checkpointed and no competing GPU job.
All numbers below are median **milliseconds per batch**, per component.

“Full” includes host validation, parameter H2D, physics and identical D2H results.
Cold runtime/topology/quadrature setup and compilation are separate in JSON.
Device-only values are lower bounds. Download-only samples use fresh evaluated
device arrays, avoiding JAX's host cache. Independently sampled medians must not
be added/subtracted as an exact stage decomposition.

### Changing geometry and gravity

256 directions, 508 faces, 10 warm repeats. Results contain radii, areas, normals,
LOS cosines, face SI gravity and validity (about 13.5 MB at 512 proposals).

| Star | Proposals | Original CPU | Batched JAX CPU full | GPU staged full | GPU resident path full | GPU device work only | GPU download only |
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

At 512 proposals, full resident GPU is 1.34–1.46× faster than staged GPU and
about 43–44× faster than original CPU helpers. Single proposals remain faster
on CPU. Batched JAX CPU is included to separate batching from GPU acceleration.

### Adding fresh intrinsic temperatures

Same 256-direction/508-face workload, 10 repeats. Additional matched outputs
are temperatures, emitted bolometric surface flux and energy closure.

| Star | Proposals | Original CPU | Batched JAX CPU full | GPU staged full | GPU resident path full | GPU device work only | GPU download only |
|---|---:|---:|---:|---:|---:|---:|---:|
| primary | 1 | 0.587 | 0.766 | 3.345 | 2.080 | 0.730 | 0.814 |
| primary | 32 | 17.672 | 6.094 | 3.298 | 2.139 | 0.746 | 0.873 |
| primary | 512 | 284.750 | 93.795 | 8.800 | 7.032 | 2.737 | 8.407 |
| secondary | 1 | 0.614 | 0.701 | 3.730 | 1.939 | 0.671 | 0.813 |
| secondary | 32 | 18.056 | 6.001 | 3.579 | 2.129 | 0.740 | 0.857 |
| secondary | 512 | 296.714 | 93.575 | 16.008 | 6.976 | 2.991 | 7.964 |

### Adding fresh B, V and bolometric passband emission

This deliberately smaller spectral workload uses **64 directions, 124 faces**,
three passbands, 32-wavelength chunks, batches 1/8/32/128 and **five** repeats.
All core/thermal outputs plus emitted band fluxes are downloaded (about 1.46 MB
at 128 proposals). It must not be compared with the larger geometry table as
if the mesh size were the same.

| Star | Proposals | Original CPU | Batched JAX CPU full | GPU staged full | GPU resident path full | GPU device work only | GPU download only |
|---|---:|---:|---:|---:|---:|---:|---:|
| primary | 1 | 14.203 | 1.903 | 4.474 | 3.072 | 1.546 | 0.887 |
| primary | 8 | 122.000 | 10.504 | 5.207 | 3.974 | 2.217 | 0.974 |
| primary | 32 | 472.924 | 18.111 | 7.815 | 6.567 | 4.795 | 1.255 |
| primary | 128 | 1863.572 | 65.115 | 14.605 | 14.204 | 12.052 | 1.355 |
| secondary | 1 | 12.563 | 1.919 | 4.546 | 3.272 | 1.480 | 0.936 |
| secondary | 8 | 104.379 | 11.888 | 5.210 | 3.997 | 2.217 | 1.027 |
| secondary | 32 | 437.146 | 18.296 | 7.695 | 6.506 | 4.845 | 1.096 |
| secondary | 128 | 1895.987 | 67.389 | 14.335 | 13.954 | 12.037 | 1.435 |

At 128 proposals the full spectral GPU path is about 4.6–4.8× faster than batched
JAX CPU and 131–136× faster than original helpers in this bounded workload.
The native CPU code loops over temperatures/passbands; the JAX CPU comparison
shows much of the native-helper gap also comes from batching and quadrature reuse.
This is emitted surface flux, not complete observed light curves or a fitting
speedup. No MCMC convergence/effective-sample claim is made.

### Wavelength-chunk experiment

A bounded sweep used chunks 32/8/16/64/128/32, with the 32-sample baseline
repeated at the end to expose order/clock effects. Same 128 proposals, 64
vertices, three passbands and complete outputs, 10 warm repeats. Each setting
passed the original CPU oracle and strict device transfer guard. These timings
compare chunk choices, not additional whole-model speedups.

| Star | Order | Wavelength chunk | Device work only (ms) | Full matching output (ms) |
|---|---:|---:|---:|---:|
| primary | 0 | 32 | 12.165 | 13.847 |
| primary | 1 | 8 | 16.173 | 17.898 |
| primary | 2 | 16 | 12.308 | 14.122 |
| primary | 3 | 64 | 10.380 | 12.389 |
| primary | 4 | 128 | 10.394 | 12.337 |
| primary | 5 | 32 | 10.868 | 12.608 |
| secondary | 0 | 32 | 12.057 | 14.017 |
| secondary | 1 | 8 | 15.674 | 17.627 |
| secondary | 2 | 16 | 12.135 | 14.027 |
| secondary | 3 | 64 | 10.295 | 12.269 |
| secondary | 4 | 128 | 10.397 | 12.238 |
| secondary | 5 | 32 | 10.837 | 12.786 |

Chunks 64/128 measured 12.2–12.4 ms full versus 12.6–12.8 ms for the later
32-sample baseline. Much of the apparent improvement over the initial 32 result
was warm-up/order dependent; the remaining chunk gain is modest (about 3–5%).
The conservative default remains 32; callers can explicitly select chunk_size.

## Production restoration

Every GPU window used the supported SIGTERM checkpoint handler, verified
journal/witness endpoints, the exact pinned source/config/plan, and a bounded
systemd test unit whose cleanup restores production even after failure. Each
restoration below verified a matching resume record and **durable advancement**.

| Window | Restoration | Job | Saved cursor | Verified cursor | Durable cells advanced |
|---|---|---|---:|---:|---:|
| 01 | restored_and_advancing | j00872 | 935771216 | 935814224 | 43008 |
| 02 | restored_and_advancing | j00875 | 949153360 | 949196368 | 43008 |
| 03 | restored_and_advancing | j00878 | 958551888 | 958582608 | 30720 |
| 04 | restored_and_advancing | j00878 | 960642896 | 960687952 | 45056 |
| 05 | restored_and_advancing | j00878 | 961144656 | 961189712 | 45056 |

Window 01's benchmark gate failed at the original rounding tolerance; its
97-test suite passed and production restoration was verified. Subsequent windows
passed their requested checks. No GPU reset, driver/global-setting changes or
merge. The other machines were not paused by this work.

## Review priorities and next physics work

DOT can review state/build order, secondary gravity units, unsupported masks,
reference-gravity cancellation, energy conservation, FP64 barriers, passband
pi/wavelength units, Simpson weights, streaming memory and transfer accounting.

The supplied directions/connectivity remain fixed and the ray mesh remains
**uncorrected**. Pipeline temperatures use those supplied ray-mesh areas; they
are not yet the final corrected areas/state of a fully built CPU ELISa container.
Candidate-specific angular placement/topology and CPU-compatible mesh correction
come next, then irradiation/reflection, atmosphere-table/LD updates, changing
mutual eclipse coverage and complete normalized light-curve/posterior validation.
Changed physical parameters must never reuse stale coverage or brightness.

## Evidence

- [Machine-readable progress](ada_validation/progress.json) and [tested source fingerprints](ada_validation/source-hashes.json).
- [Latest 130-test output](ada_validation/window-04-tests.txt) and [all numerical comparisons](ada_validation/window-04-parity.json).
- [Core GPU samples](ada_validation/window-02-benchmark-gpu.json) / [core JAX CPU samples](ada_validation/window-02-benchmark-cpu.json).
- [Thermal GPU samples](ada_validation/window-03-benchmark-gpu.json) / [thermal JAX CPU samples](ada_validation/window-03-benchmark-cpu.json).
- [Spectral GPU samples](ada_validation/window-04-benchmark-gpu.json) / [spectral JAX CPU samples](ada_validation/window-04-benchmark-cpu.json).
- [Isolated blackbody CPU output](ada_validation/blackbody-cpu-tests.txt) / [preceding CPU suite](ada_validation/cpu-before-energy-snapshot-tests.txt).
- [GPU wavelength-chunk sweep](ada_validation/window-05-chunk-tuning-gpu.json).

Raw machine-local supervision/checkpoint evidence is preserved in the isolated Ada workspace; public evidence omits process/environment snapshots. The existing `experimental/gpu-backend` benchmark branch remains untouched.
