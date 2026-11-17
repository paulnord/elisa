# Ada GPU development progress

Updated **2026-11-17T04:29:53Z**. Review branch: `codex/gpu-resident-roche-pipeline`.
**Nothing merged. The assignment continues.**

## Latest milestone: device projection and partial eclipse clipping

Fresh complete hybrid curves now use GPU observer projection, Sutherland–Hodgman
partial-face intersections, coverage assembly/SMA scaling and batched three-band
flux integration. **CPU Qhull/Path and the original visibility selectors remain
a fresh, timed bridge.** The candidate mesh/roots/topology, reflection and
radiance/LD are still CPU stages. This is a hybrid forward path, with no change
to existing CPU Observer or fitting defaults.

`gpu_clipping.ConvexTriangleClipper` ports the actual CPU algorithm: translate to
triangle-local coordinates, clip against the same three half-planes, retain
crossing/end-vertex order and calculate a translated shoelace area. There is no
coordinate quantization or half-plane widening. Buffers use the CPU's conservative
8*hull-capacity bound; nonfinite/overflow states remain explicit unsupported
outputs. Exact zero determinants and zero-area edge/vertex contacts are retained.

`gpu_occultation.DeviceEclipseIntegrator` snapshots each fresh corrected mesh,
cached normals, radiance and LD once, projects the real new geometry on device,
then sends projections to CPU for the exact existing hull/visibility rules.
Fresh hulls/masks/partial indices return to device. Clipping, uncovered-area
assembly, the original small-area threshold, physical area scaling and all bands
then execute in packets of eight phases. Prepared packets have immutable inputs
and pass strict no-transfer evaluation guards. Nothing is reused from a previous
physical candidate. The original fixed-direction ray pipeline remains separate.

### Parity and independent geometry gates

**184 tests passed in 72.06 s on Ada, zero failures/errors/skips.** CPU-only
checks passed **109**, with **75** explicit hardware skips. The eight detailed
whole-curve cases passed on JAX CPU and GPU in the same checkpointed window.
They retain all preceding changing-mass/potential/spin/temperature, three-band,
three-LD-law, reflection, ordering/repeat and both-eclipse cases, and add a tiny
occulter that exercises ELISa's separate small-object visibility branch.

The clipping tests also use 20 literal analytic/exact-rational area cases with
both windings (including positive slivers and exact edge/vertex contacts), plus
60 saved real Roche eclipse hull/triangle batches. Maximum absolute area error
was **1.12e-16** for the analytic cases and **2.17e-19 SMA²** for the saved eclipse
batches. This is independent geometry evidence in addition to CPU parity.

In the eight full GPU cases:

- Maximum normalized total-curve absolute difference: **4.45e-16**; maximum raw
  observed-flux relative difference: **1.09e-15**.
- Selected-face indices all matched. Maximum LOS-cosine absolute difference:
  **3.34e-16**; coverage absolute difference: **1.58e-17 SMA²**.
- Maximum per-face contribution difference, normalized by component flux:
  **1.02e-13**; maximum component-flux relative difference: **1.35e-13**.
  Small eclipsed components amplify relative roundoff while their total-curve
  effect remains tiny. Fully occulted zero coverage remains zero.
- Maximum accumulated three-band log-likelihood difference: **3.89e-11** at
  normalized errors `1e-4`, `ln_f=-2` and 45–69 observations per band.

The existing `3e-13` surface relative gate, `64*eps` geometry absolute gate,
`2e-12` normalized-flux gate and `1e-6` diagnostic likelihood gate are unchanged.
Analytic clipping areas pass relative `3e-13` with zero absolute tolerance;
positive slivers are explicitly required to remain positive. Geometry/numeric
unsupported results are not physical-prior rejections. These are implementation
parity checks, not mesh/time/spectral convergence or a validation of all possible
ill-conditioned polygons. No physical fitting, sampler or ESS claim is made.

### Matched full-curve timings

Median **milliseconds per fresh curve**, three warm repeats, all CPU work,
transfers and matching final host B/V/bolometric curves included. The following
paths ran in the same checkpointed window with affinity0–15 and thread counts1.
JAX CPU and GPU drivers ran sequentially. The original CPU and preceding hybrid
columns below come from the GPU driver; the JAX CPU column comes from its
separate matched-profile driver. CPU Observer samples from both drivers are
retained in JSON. Input `BinarySystem` creation is outside all these timers;
combined construction of both comparison engines and first-call/stage medians
are separate. First-call measurements include diagnostic captures; warm calls
use diagnostics=False. Hybrid extra
component/timing metadata is also included in its elapsed cost.

| Case | Original CPU Observer | Previous hybrid GPU (CPU coverage) | Batched hybrid JAX CPU | Batched hybrid GPU |
|---|---:|---:|---:|---:|
| Non-eclipsing | 66.48 | 111.87 | 63.37 | 66.40 |
| Grazing | 73.02 | 125.04 | 89.09 | 74.12 |
| Partial, reflection | 74.17 | 119.79 | 226.39 | 74.90 |
| Total, reflection | 80.19 | 135.27 | 275.69 | 78.99 |
| q=0.5, logarithmic LD | 74.98 | 127.62 | 276.81 | 72.33 |
| q=2, square-root LD, finer mesh | 102.86 | 154.71 | 252.44 | 101.21 |
| Equal components | 67.46 | 113.04 | 231.41 | 67.82 |
| Small occulter | 72.21 | 136.04 | 99.69 | 68.08 |

The batched GPU path reduces the preceding hybrid's latency, but is approximately
as expensive as original CPU Observer at these workloads. Only three repeats and
sequential run order do not establish a robust whole-curve gain from the small
CPU/GPU differences. Eclipse-heavy batched JAX CPU is substantially slower;
batching alone does not explain the GPU result. No fitting/MCMC speedup follows.

### Domain, fallback and next work

`HybridBinaryLightCurve(..., coverage="device")` is now the default experimental
path. `coverage="cpu"` retains the preceding complete hybrid path for comparison.
The existing circular-detached, unspotted, blackbody/custom-LD state domain is
unchanged. Device clipping preserves the configured Sutherland–Hodgman/reference
contract. A configured PYPEX backend uses the **whole original CPU Observer**,
with its original semantics and an explicit fallback reason.

Current device resource bounds are 128 synthesis-support phases after Observer
reduction/symmetry, 4096 vertices/8192 faces per star, 128 hull vertices and 256
partial faces per component/phase. These bound projection and clipping storage;
they are not physical-prior bounds. Exceeding them runs the full CPU curve or
raises explicitly when `fallback="raise"`; no phases are dropped. A 601-observation
regression verifies complete, exactly equal CPU fallback beyond the phase bound.
Other existing model/data fallbacks remain. This is not an all-GPU forward model.

Next assess the **original mean-normalized physical fitting/likelihood contract**
with fresh changed-parameter predictions, original phase mapping/normalization,
third light/noise and representative fit/sampler equivalence. Include candidate
construction, transfers and any shape recompilation in that assessment; short
chains are only smoke tests. CPU hull/visibility, resident mesh/state/emission
composition, real atmosphere/LD tables, spots and broader eccentric/contact GPU
support remain separate work. The l3/ln_f-only pilot still caches its fixed curve.

Reproduce with the existing private FP64 environment and CPU development wrapper:

```text
python -m pytest tests/tests_experimental -q -rs
python scripts/validate_gpu_binary.py --device cpu --repeats 3 --output validation-cpu.json
python scripts/validate_gpu_binary.py --device gpu --repeats 3 --output validation-gpu.json
```

GPU commands require the authorized independently restored systemd allocation.
This window's complete probe/test/CPU/GPU validation batch ran **120.79 s** within
its 270 s internal/300 s systemd limits. Cleanup restored the exact pinned
production service, verified a matching resume record and latest receipt/witness
endpoints, and observed **45,056 durable cells** of advancement. Unit cleanup
finished successfully. This is not a full historical production re-audit.

See [curated milestone measurements](ada_validation/background-m002.json),
[tested source/fixture hashes](ada_validation/background-m002-source-hashes.json),
and [current machine-readable status](ada_validation/progress.json). The 551-file
runtime/fixture manifest hashes to
`f880f63bf46945353b88aefcd7fc451720dafdd6cfc3ad99d7d5b21907bf9dc3`
using compact sorted-key JSON. Tested runtime files remained unchanged afterward.
All operational logs and process/environment/checkpoint snapshots remain private.

## Previous milestone: corrected state and functional hybrid observed curves

`CorrectedSurfaceState` and `HybridBinaryLightCurve` now produce complete fresh
**B, V and bolometric observed light curves**, including changing mutual eclipses,
Observer maximum normalization, third light, distance and original phase ordering.
The implementation is explicitly **hybrid**, with measured CPU stages and transfers.
Existing CPU Observer and fitting defaults are unchanged.

The inspected CPU call graph determines the sequence:

1. CPU `build_mesh`, `build_faces`, `build_velocities` recompute each candidate's
   angular placement, radial roots, topology and kinematics.
2. Device geometry/gravity use **uncorrected** vertices. Quarter-face symmetry
   and secondary units follow ELISa. Centres, normals and gravity remain cached
   from this stage.
3. Device point correction uses the actual CPU detached correction resource,
   scaling about the correct component centre. Corrected areas, cached gravity
   and fresh intrinsic temperatures then undergo the CPU-compatible energy
   normalization. Corrected points do not replace cached centres/normals/gravity.
4. The complete state returns to CPU for the original reflection, fresh
   blackbody radiance, optical-depth/LD correction, observer rotation and
   phase-specific silhouette/partial-face eclipse clipping.
5. Device integration returns both component totals. CPU remaps phases/repeats,
   adds third light using that mapped mean, and applies distance or normalization.
   Fixed padding avoids recompiling integration for each different face selection.

The original fixed-direction resident Roche-to-emission experiment below remains
separate and uncorrected. This functional bridge does not yet compose its ray or
blackbody kernels into a resident observed-light-curve model.

### Scientific evidence

**163 passed in 47.99 s on Ada; zero failures/errors/skips.** All original tests
retain their thresholds. CPU-only development passed **98**, with **65** explicit
hardware skips. The independent seven-system validation driver also passed on
JAX CPU and GPU, comparing unmodified ELISa Observer and fully built containers.

Cases cover q=0.5/0.7/1/2; spin=0.5/1/1.5; unequal/equal temperatures and
potentials; inclinations 60/79/85/87/88/90 degrees; no eclipse, grazing, partial
and total eclipses; both conjunctions, ingress/egress, contact-focused samples,
quadratures and unsorted/repeated/cross-cycle observations with nonzero phase
shift and third light. Actual candidate meshes have 832 or 1280 faces per star.
Linear, logarithmic and square-root LD and CPU reflection on/off are exercised.
Contact samples use radius estimates to locate phases; the actual oracle and
clipping use triangulated Roche silhouettes, not spherical occultation.

- GPU maximum normalized-curve absolute difference: **4.44e-16**;
  raw observed-flux relative difference: **1.09e-15**.
- GPU maximum SI face-gravity relative difference: **4.75e-16**;
  temperature relative difference: **4.20e-16**.
- Corrected points/areas and cached normals were identical in the measured GPU
  cases. Cached centres differed by at most **2.23e-16 SMA**.
- All selected-face indices matched. Coverage, per-band radiance/LD, per-face
  contributions, component totals and total curves passed. Zero-covered faces
  and a fully occulted secondary are explicitly exercised.
- Independent intrinsic `sum(A*T**4)/(sum(A)*T_eff**4)` closure passed.
  Reflection cases compare the original CPU prescription separately.
- Accumulated three-band log-likelihood difference was at most **2.94e-11** for
  normalized errors `1e-4`, `ln_f=-2` and 45–69 observations per band. This is a
  forward-model diagnostic, not a physical fit or sampler equivalence result.

State/surface parity retains relative `3e-13`, geometry absolute `64*eps`, and
the existing fitter's absolute normalized-flux gate `2e-12`. The independent
likelihood gate is `1e-6`, negligible relative to order-unity likelihood ratios.
Near-zero face terms use absolute normalized contributions; relative errors in
near-zero normal components are not interpreted as physical error. No prior
tolerance was relaxed. Implementation parity does not establish mesh, time or
spectral discretization convergence or accuracy of the CPU reflection model.

### Full-call timing: no hybrid speedup at these workloads

Median **milliseconds per fresh three-band curve**, three repeats. Both columns
below were measured in the same restored-production GPU window: CPU affinity
0–15, thread counts 1, no competing GPU work. Inputs are preconstructed
`BinarySystem` candidates for both paths. Each call rebuilds all dependent state.
The hybrid timer includes copying, CPU mesh/topology/reflection/radiance/coverage,
device uploads/evaluation/downloads, phase mapping and normalization. It also
includes extra component/timing metadata. Cold constructor/first-call compilation
and individual full-call samples are reported separately in JSON.

| Case | Observations/band | Original CPU Observer | Hybrid GPU |
|---|---:|---:|---:|
| Non-eclipsing | 45 | 66.98 | 106.65 |
| Grazing | 57 | 73.93 | 120.93 |
| Partial, reflection | 57 | 72.39 | 126.47 |
| Total, reflection | 69 | 85.52 | 141.16 |
| q=0.5, logarithmic LD | 69 | 82.02 | 142.69 |
| q=2, square-root LD, finer mesh | 57 | 104.27 | 146.54 |
| Equal components | 57 | 68.17 | 111.94 |

CPU-stage overhead and per-phase transfers/launches dominate this implementation.
The additional JAX CPU validation/timing run used cores 16–19/nice19 while
production ran; its measurements are disclosed separately and are not an
identical-profile CPU/GPU timing comparison. Prior resident primitive speedups
below do not imply whole-curve or fitting speedups.

### Supported domain, fallbacks and next milestones

The hybrid domain is circular detached, unspotted/nonpulsating, q in [0.5,2],
spin in [0.5,1.5], raw own-centre radii in [1e-6,0.25] SMA, beta in [0,1],
blackbody and explicit per-band/bolometric LD. Geometry conditioning gates remain
unchanged. Every candidate and phase gets fresh dependent state; no previous
candidate's mesh, coverage, brightness or temperature is reused.

Unsupported models use **full, unmodified CPU Observer**, with
`backend="cpu_fallback"` and a reason, or raise `UnsupportedHybridModelError`
when `fallback="raise"`. Unsupported states are not physical-prior rejections.
Dispatch tests cover eccentricity/separation, large-radius/near-contact meshes,
spots, atmosphere tables and tabulated LD; an actual e=0.1, 27-phase CPU fallback
matches exactly. Atmosphere/LD table directories here contain no scientific
tables, so dispatch tests are not table-physics validation. No spotted GPU model
or near-contact/eccentric GPU equivalence is claimed. The sparse-phase eccentric
CPU approximation router can itself raise a zero-size reduction error; fallback
preserves that CPU exception. It was not changed to obtain parity.

Next: reduce the measured CPU/transfer costs by connecting corrected state to
resident candidate geometry, observer projection and changing eclipse coverage;
validate contact masks before claiming a resident forward model. Reflection and
real atmosphere/LD tables need their own device validation, as do spots and
broader separation/contact domains. Then assess the original mean-normalized
fitter mapping/noise/third-light contract and representative physical fitting/MCMC.
The l3/ln_f-only pilot continues to cache its fixed physical curve. There is no
physical-posterior, MCMC convergence, ESS/sec or complete-assignment claim yet.

### Reproduction, fingerprints and restoration

Use the private environment, FP64 and process-only thread limits described in
the handoff. Force `JAX_PLATFORMS=cpu` during development. GPU commands require
the authorized independently restored systemd window; these commands alone do
not establish the allocation or restore production:

```text
python -m pytest tests/tests_experimental -q -rs
python scripts/validate_gpu_binary.py --device cpu --repeats 3 --output validation-cpu.json
python scripts/validate_gpu_binary.py --device gpu --repeats 3 --output validation-gpu.json
```

Qualification was read-only: clean published checkout, private FP64 CPU probe,
authenticated connected GitHub write capabilities, pinned production release,
last durable receipt/witness and actual advancement verified before any pause.
The registered background window ran about **64.88 s** of GPU probe/tests/driver
commands, within the five-minute systemd/internal limits. Guaranteed cleanup
restarted the exact service, matched its resume record, reverified source/config/
plan and the latest checkpoint evidence, and observed **24,576 durable cells**
of advancement. Test unit cleanup completed successfully. This validates the
latest receipt/witness/endpoints, not a full historical production re-audit.

Curated evidence: [full milestone measurements](ada_validation/background-m001.json),
[tested runtime source fingerprints](ada_validation/background-m001-source-hashes.json),
and [machine-readable status](ada_validation/progress.json). The new manifest
covers 546 Python/project files, including the unchanged CPU oracle, and hashes
to `daaa6fa16ee1e7ec817a28b627cd2186e9c4005a6e13522f9f26bb9f8325e04f`
using compact sorted-key JSON. Runtime files were checked unchanged after the
hardware window. Only documentation/evidence were added afterward. Operational
logs and process/environment/checkpoint snapshots remain private.

## Prior resident-primitive baseline

The following results were last updated **2026-11-16T22:17:16.273566+00:00**.
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

## Review priorities for the resident primitive baseline

DOT can review state/build order, secondary gravity units, unsupported masks,
reference-gravity cancellation, energy conservation, FP64 barriers, passband
pi/wavelength units, Simpson weights, streaming memory and transfer accounting.

The supplied directions/connectivity remain fixed and the ray mesh remains
**uncorrected**. Pipeline temperatures use those supplied ray-mesh areas; they
are not yet the final corrected areas/state of a fully built CPU ELISa container.
The latest hybrid adapter above uses fresh CPU angular placement/topology and
reconstructs corrected cached state. These remain CPU/host bridges rather than
resident extensions of this older ray pipeline. Resident reflection, atmosphere/
LD updates, changing eclipse coverage and complete fitter/posterior validation
remain. Changed physical parameters must never reuse stale coverage or brightness.

## Evidence

- [Machine-readable progress](ada_validation/progress.json) and [tested source fingerprints](ada_validation/source-hashes.json).
- [Latest 130-test output](ada_validation/window-04-tests.txt) and [all numerical comparisons](ada_validation/window-04-parity.json).
- [Core GPU samples](ada_validation/window-02-benchmark-gpu.json) / [core JAX CPU samples](ada_validation/window-02-benchmark-cpu.json).
- [Thermal GPU samples](ada_validation/window-03-benchmark-gpu.json) / [thermal JAX CPU samples](ada_validation/window-03-benchmark-cpu.json).
- [Spectral GPU samples](ada_validation/window-04-benchmark-gpu.json) / [spectral JAX CPU samples](ada_validation/window-04-benchmark-cpu.json).
- [Isolated blackbody CPU output](ada_validation/blackbody-cpu-tests.txt) / [preceding CPU suite](ada_validation/cpu-before-energy-snapshot-tests.txt).
- [GPU wavelength-chunk sweep](ada_validation/window-05-chunk-tuning-gpu.json).

Raw machine-local supervision/checkpoint evidence is preserved in the isolated Ada workspace; public evidence omits process/environment snapshots. The existing `experimental/gpu-backend` benchmark branch remains untouched.

## Continuing background assignment

Background handoff prepared **2026-11-17T03:41:23.890646+00:00**. Paul has assigned completion and scientific validation of the light-curve pipeline, followed by a fitting/MCMC assessment. See the [authorized assignment, milestones and safety rules](ada_background_handoff.md). The detached worker qualified successfully and completed its first tested milestone above. DOT can follow this report and the same development branch for each tested, pushed milestone. No merge is authorized.
