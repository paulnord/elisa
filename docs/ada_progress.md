# Ada GPU development progress

Updated **2026-11-17T19:26:12.598224+00:00**. Review branch: `codex/gpu-resident-roche-pipeline`.
**Nothing merged. The assignment continues. Fifth42 is intentionally held for ELISa.**

## Latest milestone: device observer geometry and guarded eclipse roots

`ObserverDeviceLightCurve` now composes fresh device observer rays, normals,
curved Jacobians and conditioned silhouette solves into complete reflected
curves and original mapped physical likelihoods. Nested limb/shadow edge
prefixes run on device; the original CPU topology, clipping and guarded final
bisection decisions remain explicit. **144 hardware controls pass without skips.**
CPU validation passes 124 with 20 explicit GPU skips. Another 472 unaffected
hardware controls are retained: 616 additive cases, not a fresh 616-case suite.
Original CPU/default synthesis and priors retain their behavior.

This is a substantive hybrid port, **not an all-GPU observer or qualified MCMC
backend**. Internal rays retain 80 bounded radial bisections, now optionally
unrolled 8 at a time; the default remains unroll 1. The shadow uses the original
12 Newton updates and 64 eps stationary gate in conditioned star-local coordinates.
Shape-bucketed node batches avoid recompilation at every adaptive leaf count.
The pure kernels also support changing physical-row/LOS batches; the complete
CPU controller still visits each candidate/phase. Proposal field geometry and
thermal state remain resident from m018.

The original finite protocol is especially sensitive at exact seams. CPU
arithmetic resolves near-zero limbs and inscribed/outer shadow branch seams.
The last 8 edge/slice decisions remain CPU, and any precision-ambiguous device
prefix or disagreeing final CPU bracket is replayed in original arithmetic.
These are counted work/transfer boundaries, never discarded patches, prior cuts
or cached predictions. The original positive-weight, cone, topology, singular
source, physical-domain and resource guards remain active. CPU development
caught a potential-sign transcription and two branch/zero-plateau cases; all
were resolved before hardware acceptance without changing an oracle or tolerance.

### Complete curves, likelihoods and costs

Six representative cases retain partial/total eclipse, unequal masses/spins,
three LD laws, CK04 tables and a tiny occultor. Both stars' fields, component,
raw and normalized curves pass. The full original adaptive traces match,
including the maximum slice shift. Final packet byte hashes may differ under
device arithmetic; the independent gates remain 3e-13 relative/zero raw/component
absolute allowance and 3e-13/2e-12 normalized. Maximum complete normalized curve or
prediction difference is **7.77156e-16**; maximum posterior difference is
**1.70985e-09**, under the unchanged 1e-6 absolute density gate.

Every complete batch changes q, potential, temperatures, inclination, l3 and ln_f and
uses original phase/JD mapping, uneven/repeated observations, normalization,
noise, prior and density. Both paths compute the same two finite field curves
per candidate and return all host comparison outputs. No fixed physical
prediction is reused. Enclosing timers include planning, CPU bridges, transfers,
synchronization, host outputs and accuracy checks. Constructor and fixed setup
are separate; first calls include encountered compilation and changed warm calls
may encounter new bucket shapes. One/two warm samples per complete shape are
preliminary ranges, not stable-tail or MCMC-throughput estimates.

| Complete comparison batch | Original CPU cold / warm | JAX CPU cold / warm | GPU cold / warm |
| --- | --- | --- | --- |
| 1 proposal | 10.731 / 10.520–10.541 s | 48.524 / 20.751–22.251 s | 35.967 / 7.500–9.045 s |
| 4 proposals | 42.357 / 42.452–42.452 s | 112.445 / 85.519–85.519 s | 60.224 / 32.259–32.259 s |
| 8 proposals | 84.045 / 84.077–84.077 s | not measured | 92.765 / 59.513–59.513 s |

Warm median original-CPU/GPU ratios are **1.273× / 1.316× /
1.413×** for B1/B4/B8 respectively. These are finite comparison-batch
costs, not a complete scientific-analysis or single-model sampler speedup. Cold
setup/compilation can erase a warm gain. JAX CPU is retained as its own measured
baseline, not assumed faster. Repeated independent CPU/GPU processes record
order/cache drift. Current baselines are serial; earlier V505 multiprocess CPU
evidence remains a separate model/contract.

### Finer controls, hardware and continued work

The additional field 2→3/common 3 comparison recomputes full original/GPU densities.
Its nominal accumulated log-likelihood enclosures span **0.955841–1.01537**.
The strict nominal D<=.1 and logL<=.01 guards remain independent of ln_f.
**1e-4 physical fitting/MCMC remains unqualified.** A numerical accuracy stop
remains outside prior penalties. Implementation parity is separate from finite
convergence, uncomputed continuum error and observational model adequacy.
No converged chain, posterior stability or ESS/sec result is claimed.

The [hardware ledger](hardware_characterization.md) adds active-load context,
original/vectorized NumPy/JAX CPU/GPU silhouette sweeps through B128 × 2048 nodes,
compiler/allocator versus sampled memory, and a focused complete-call CUDA/NVTX/
OSRT graph-node trace with an independent unprofiled run. The inherited timer
formerly called CPU geometry is **mixed CPU/GPU** here; m018's 97.8% CPU fraction
no longer describes this path. Copy duration, dispatch, synchronization and
unassigned gaps are not DRAM bandwidth or achieved FP64. Missing counter and
critical-path attribution are labeled; no new-processor speedup is predicted.

The [runnable Colab notebook](../notebooks/elisa_observer_batches.ipynb) pins
scientific code commit `a3a83b6` and 55 tested runtime dependencies. Its five pipeline,
configuration, runtime and archive cells passed on Ada in window137; a read-only
CPU-identification cell was added and exercised separately, with the earlier
cell hashes unchanged. Network installation was skipped using the pinned private
environment. [Execution evidence](ada_validation/colab-observer-ada.json) records
actual hardware, complete original/JAX CPU/GPU costs, source/dependency checks,
parity and the retained accuracy stop. **Real Colab execution is pending**;
no tenfold complete-analysis result is claimed.

All 25 bounded units, including notebook execution, preserved the exact active Fifth42 checkpoint hold. Nothing
was resumed or merged. Next: fuse the numerous device root/launch steps and
continue field/boundary convergence before qualified physical inference/MCMC.

[Curated samples/report](ada_validation/background-m019.json),
[source manifest](ada_validation/background-m019-source-hashes.json).

## Previous milestone: batched resident Roche geometry in complete curves

`BatchedRocheGeometry` now solves fresh changing-parameter Roche roots, gradients,
normals, curved Jacobians and Duffy areas on the GPU. `ResidentGeometryLightCurve`
composes these with gravity and intrinsic temperatures in one resident graph,
then runs the existing reflection, fresh radiance, observer and flux pipeline.
**36 new hardware controls and 36 CPU controls pass, with no skips.** All 626
previous source/data files are byte-identical; 556 earlier hardware controls are
retained, giving 592 cases of additive coverage, not a fresh 592-case suite.
Original CPU/default behavior and existing unsupported-case/fallback routes are
unchanged. These new APIs are opt-in and raise outside their declared domain.

The independent oracle retains the original trigonometric potential and 56
bisections, original gradients/thermal/reflection/radiance/flux formulas. The port
uses the same Roche equation in Cartesian arithmetic and 80 bounded bisections.
Existing 3e-13 relative gates, strict endpoint/inner-branch checks and FP64 remain.
Complete raw/component/normalized curves, both stars' fields/radiance, identical
CPU quadrature hashes/traces and original mapped likelihoods all pass. The six
fixtures include partial/total eclipse, unequal masses/spins, three LD laws,
CK04 and the tiny-occultor case. Additional tests cover changing actual angular
rays/topology, bounded waves, exact original Observer behavior before/after,
resident composition without callbacks, refined fields and 57 contact phases.

This is still a hybrid. Original CPU topology and nested vertex roots generate
fresh per-candidate angular meshes; observer roots, adaptation, eclipse clipping,
ancestry and hashing remain CPU work. LD/table bridges still download required
state. Reflection and radiance remain candidate-specific. No result is reused
across physical proposals; all proposals are evaluated, with shape groups and
bounded waves controlling allocations. No padding changes a physical prior.

### Matched performance and the next bottleneck

The following medians include fresh host input, synchronization and required
host output. Geometry-stage rows have ten changing batches each. Original CPU
and vectorized NumPy share 56 trigonometric bisections; JAX CPU/GPU share the
80-step Cartesian port. Algorithm/batching gains and hardware gains are distinct.
Private CPU profiles use cores16–19/nice19, while bounded device profiles and
original CPU baselines within them use cores0–15/nice10. BLAS/OMP/MKL/NumExpr
environment counts are one; actual JAX/XLA pool size is unmeasured. This shared-machine comparison is preliminary.

| Geometry stage / component | Proposals × nodes | Original CPU | Vectorized NumPy | JAX CPU | GPU |
| --- | --- | --- | --- | --- | --- |
| sample/primary | 128 × 2048 | 242.514 ms | 216.928 ms | 122.759 ms | 10.407 ms |
| sample/secondary | 128 × 2048 | 245.512 ms | 211.194 ms | 125.882 ms | 10.494 ms |
| area/primary | 128 × 2048 | 271.289 ms | 214.895 ms | 130.017 ms | 5.553 ms |
| area/secondary | 128 × 2048 | 269.406 ms | 222.019 ms | 130.348 ms | 5.600 ms |

Area rows mean 128 angular faces ×16 nodes per proposal. These are geometry
stages with supplied angular topology, not whole-curve speedups. Sweeps retain
batch1/32/128, sample512/2048 and area8/128 faces, first shape calls, constructor
costs, individual samples, nested transfer/dispatch/synchronization intervals,
interleaved backend ordering and compiler/allocator memory accounting.

Complete costs below compute **two finite field curves per candidate**, original
uneven/repeated observation mapping, phase/third-light/mean/noise semantics,
prior and density, all CPU bridges/transfers/host outputs and accuracy checks.
q, potential, temperatures, inclination, l3 and ln_f change in every batch.
Fixed observation setup and engine construction are recorded separately; cold
first calls include encountered compilation. Each warm range has only two
samples: it does not establish a stable tail or MCMC throughput.

| Complete comparison batch | Original CPU cold / warm range | JAX CPU cold / warm range | GPU cold / warm range |
| --- | --- | --- | --- |
| 1 proposals | 10.939 / 10.701–10.721 s | 15.344 / 10.942–10.967 s | 16.156 / 10.839–10.901 s |
| 4 proposals | 42.724 / 42.508–42.643 s | 47.506 / 43.490–43.629 s | 48.247 / 43.528–43.670 s |

Maximum complete normalized curve/prediction difference is **6.66134e-16**;
maximum posterior difference is **1.70257e-09** (unchanged absolute gate1e-6).
Original phase and JD callbacks both pass. Repeated original CPU baselines after
the device runs retain process/order drift evidence in the report.

The GPU does **not** speed up these complete likelihood batches. In the same warm
executions, CPU observer geometry/clipping/hash occupies **97.74–97.86%**
of enclosing batch wall time. Removing every other cost while keeping that CPU
work fixed gives only **1.023×** as a conditional upper ceiling.
This is not a limit after porting the observer stage or a future GPU prediction.
It is why the geometry-stage gains do not translate into current full-model gains.

### Hardware evidence and remaining scientific gates

Active-load samples observe **PCIe generation4, width8**, consistent with the
upstream port's advertised maximum width8. Idle generation1 is not active link
capability. Pageable fresh-buffer transfer calibration, active clock/power/thermal
samples, an independent stage process and a focused CUDA/NVTX/OSRT trace are
retained in the [hardware characterization](hardware_characterization.md).
Dispatch/synchronization API intervals are not pure link bandwidth. DRAM bytes,
L2, occupancy and achieved FP64 counters remain unavailable; no memory-bandwidth
or computation bottleneck is inferred from utilization or allocation bytes.

**Physical fitting/MCMC remains unqualified at1e-4.** The unchanged strict gate
requires nominal D<=.1 and accumulated Gaussian log-likelihood enclosure<=.01,
independently of ln_f. The complete coarse field comparisons have nominal log-likelihood enclosures
9.54–16.96 and still trigger accuracy stops. They are numerical diagnostic failures, not prior penalties or
CPU/GPU implementation discrepancies. Earlier finer field/boundary failures
remain in the preceding reports. No continuum solution, observational adequacy,
optimized fit, converged chain or ESS/sec claim follows from the port.

All successful bounded units preserve the active Fifth42 hold and exact checkpoint;
the initial shape-error-message test failure also left the hold intact. Nothing
was resumed or merged. Next: port/batch the measured CPU observer root/shadow and
boundary work, and continue field/boundary refinement against the accumulated
likelihood budget before any qualified physical sampler assessment.

[Curated results and samples](ada_validation/background-m018.json),
[source manifest](ada_validation/background-m018-source-hashes.json).

## Preceding milestone: curved eclipse edges and accumulated likelihood safeguards

`CurvedEdgeLightCurve` synthesizes complete reflected curves using implicit
curved-edge quadrature and the preceding curvature-consistent thermal,
irradiation and radiance fields. **All 84 hardware tests passed without skips**:
27 new cases and 57 affected legacy cases. CPU validation passed 64 with 20
explicit GPU skips. Another 472 unaffected controls remain under byte-identical
source/data: **556 cases of additive coverage**, not a fresh 556-case suite.
Only a protected packet hook changed an existing experimental file; its default
remains the preceding routine. Original CPU/default physics, kernels and fitters
retain their behavior.

Implementation parity passes, and finite edge changes are much smaller than
the preceding chord studies. **Strict accumulated-likelihood controls still
block 1e-4 fitting/MCMC qualification.** This milestone adds an explicit nominal
likelihood-error stop rather than treating a distance pass as sufficient.
No optimized curvature-field fit, converged chain or ESS/sec claim follows.

### Curved boundaries with positive weights

For boundary endpoints a,b and visible vertex c, E(s)=(1-s)a+s b. At each Gaussian
slice solve the original implicit limb/shadow for C(s)=E+lambda(s)(c-E), then
integrate v=(1-t)C+t c. The solid-angle Jacobian is
abs(det(a,b,c))*(1-t)*(1-lambda)**2/abs(v)**3: the lambda-prime term cancels.
No contour finite difference or signed strip correction is needed. True Roche
normals/area Jacobians are evaluated at the nodes; positive finite weights and
membership of contour/interior points in the original source cone are checked.

Profiles use order4 interior quadrature and order8 along curved slices. Junctions
receive a prescribed extra subdivision. A fan without a supported radial graph
uses explicitly counted finer chords on its **whole original source leaf**;
all earlier fragments of that source are removed to prevent overlap or a lost
crescent. Junction and singular depths are separately controlled. Exact zero
endpoints are preserved, and only exact duplicate/zero-measure fragments are
removed. No positive patch is discarded by a tolerance.

This remains a finite rule with empirical topology probes. Chords persist at
junctions/singular fans, and no interval topology or uncomputed-limit certificate
is given. CPU roots/adaptation/clipping, ancestry and original LD/table bridges
remain. Existing device point-gravity/thermal/tiled-irradiation/fresh BB/flux
kernels remain active; CK04 explicitly uses original CPU radiance/LD. This is
a hybrid. Both implementations rebuild the same actual packet independently;
packet hashes and full adaptive traces must match.

The original bounded circular detached, unspotted/nonpulsating q/spin/radius
domain and explicit resource/error semantics remain. Field/common levels0–3,
40000 faces per component, phase/band and node caps stay active. Slice order4–12
and junction/singular depths0–4 are declared before evaluation. Explicit slice
and singular-allocation guards never silently coarsen or cut physical priors.
The node cap does not bound compiler/allocator peak memory.

### Independent geometry and complete-curve controls

Analytic projected uniform-sphere tests use an independent circle-intersection
formula: annular, partial, grazing, outside contact, total and tiny-occultor
cases. The fixed target is 1e-7 of the unobscured disk, or .001 sigma at 1e-4 for
one point. Maximum refined absolute error is **7.05369e-09**. The report retains
all 18 order/depth controls, including **4 coarse target failures**.
CPU development corrected skipped small-shadow brackets and exact-zero seam
slivers; near-tangent graphs use the declared finer rule. Failure evidence stays
private; no oracle or tolerance was loosened.

Per-face original-equation controls and all raw/component/normalized host
curves pass existing gates. Maximum complete-curve normalized/raw/component-
relative differences are **4.44089e-16 /
1.24077e-24 W/m² /
1.46184e-15**. Existing gates remain 3e-13
relative with zero absolute allowance for raw/component flux and 3e-13/2e-12 for
normalized curves. Maximum intrinsic luminosity-ratio error is
5.55112e-16; this is not reflected-energy closure.

The partial fixture retains 57 contact-focused phases and V/bolometric bands.
Finite field comparisons use the same packet at each field level. Actual D uses
sigma1e-4 with original ln_f=-2; qualification uses nominal sigma independently
of nuisance noise and the unchanged conservative component enclosure D<=.1:

| Case | Field levels | Common level/depth | Actual D | Nominal component enclosure D |
| --- | --- | --- | --- | --- |
| partial_bb | 1→2 | 2/1 | 0.0307391 | 0.0448765 |
| partial_bb | 1→2 | 2/2 | 0.0307391 | 0.0448765 |
| partial_bb | 2→3 | 3/1 | 0.00809803 | 0.0113974 |
| total_bb | 1→2 | 2/1 | 0.0351378 | 0.0483437 |
| partial_ck04 | 1→2 | 2/1 | 0.00555747 | 0.00574603 |
| small_occultor | 1→2 | 2/2 | 0.0653255 | 0.0664032 |

Fixed-field integration controls separate ordinary depth from common partition:

| Fixed field | Integration control | Actual D | Nominal enclosure D |
| --- | --- | --- | --- |
| curvature_field1 | depth1→2 | 0.00152391 | 0.00154714 |
| curvature_field2 | depth1→2 | 0.00152378 | 0.00154701 |
| curvature_field2 | common2/depth2→common3/depth1 | 6.14713e-05 | 6.24525e-05 |

The curated report also compares the new and preceding chord integration at
fixed curvature fields. That is a numerical-method control, not an unchanged
original faceted whole-curve equivalence or an implementation-parity test.

### Original observation callbacks and stricter inference controls

Base/changed/JD proposals retain original physical serializers, constraints,
priors, phase/JD mapping, reducers/repeats, third light, mean normalization and
noise/density. Maximum mapped prediction/posterior differences are
**4.44089e-16 /
2.2701e-09**, inside the original 1e-6 density
gate. Out-of-prior vectors retain original -inf before synthesis. No physical
prediction is cached across candidates.

`InferenceBudget` retains nominal D<=.1 and requires the data-dependent absolute
Gaussian log-likelihood enclosure sum(abs(r)*E+E²/2)<=.01 by default. This
predeclared finite-pair diagnostic limits the computed nominal likelihood ratio
to exp(±.01), about1%; it is not a universal astronomy criterion or posterior-bias
guarantee. Both gates ignore nuisance ln_f. Failure raises
`ResolutionNotQualifiedError` outside candidate penalties, even with
PENALIZE_ALL_EXCEPTIONS. Original fitters/defaults are unchanged.

The following nominal controls use common level3, fields2/3 and edge depths1/2,
with separate bounded complete coarse/fine studies. The coarse control does not
qualify its own boundary. All costs count; the original effective-noise controls
are also retained separately in the machine report.

| Proposal | Finite control | Nominal enclosure D | Nominal delta logL | Absolute logL enclosure | Strict preflight |
| --- | --- | --- | --- | --- | --- |
| base | field2→3 | 0.00478513 | 1.35148 | 2.72661 | abort |
| base | edge depth1→2 | 0.00139386 | 0.459784 | 0.544025 | abort |
| changed | field2→3 | 0.00357479 | 0.192131 | 2.6194 | abort |
| changed | edge depth1→2 | 0.00115091 | 0.123396 | 0.593811 | abort |
| jd | field2→3 | 0.00478513 | 1.35148 | 2.72661 | abort |
| jd | edge depth1→2 | 0.00139386 | 0.459784 | 0.544025 | abort |

At fixed ordinary depth2/order4/slice8, a separate prescribed-chord refinement
changes only junction and singular depths3→4:

| Control at fixed common3/depth2/slice8 | Nominal enclosure D | Nominal delta logL | Absolute logL enclosure | Strict preflight |
| --- | --- | --- | --- | --- |
| junction/singular depth3→4 | 3.5097e-09 | -6.1757e-07 | 2.27516e-06 | pass finite pair |

Passing a finite distance gate still does not ensure negligible accumulated
likelihood error. Uncomputed finer boundaries/fields and observational/model
adequacy remain open; no fitter/sampler backend is automatically registered.

### Matched transfer-inclusive costs

These are **two-field comparison batches**, including constructors, all fresh
fields, CPU geometry/area/ancestry/clipping/hash, compilation incurred, every
transfer and raw/normalized/component host outputs. Both paths run in the same
unit, affinity 0–15, Nice 10, one thread, FP64. Code/response/shape caches can warm.
Single measurements are not distributions or ordinary one-model MCMC throughput.
Different integration/resolution contracts cannot isolate a hardware speed ratio.

| Case | Field levels; common level/depth | CPU reference batch s | GPU hybrid batch s |
| --- | --- | --- | --- |
| partial_bb | 1,2; 2/1 | 28.3603 | 32.9063 |
| partial_bb | 1,2; 2/2 | 46.9311 | 52.1705 |
| partial_bb | 2,3; 3/1 | 74.2358 | 73.6229 |
| total_bb | 1,2; 2/1 | 34.9193 | 39.7373 |
| partial_ck04 | 1,2; 2/1 | 28.801 | 33.1247 |
| small_occultor | 1,2; 2/2 | 52.4405 | 58.6584 |

Each callback includes physical serialization/constraints/construction, prior,
original prediction and density, and computes both fresh comparison models.
Fine curves pass through original mapping. Same-candidate mapping-only oracles
are outside those timers and never count as fresh synthesis. Paired totals are:

| Proposal | Coarse CPU/GPU s | Fine CPU/GPU s | Summed CPU/GPU s | Summed study wall s |
| --- | --- | --- | --- | --- |
| base | 49.4992 / 47.4673 | 76.5364 / 75.0327 | 126.036 / 122.5 | 249.025 |
| changed | 48.6126 / 46.56 | 75.6721 / 75.2489 | 124.285 / 121.809 | 246.585 |
| jd | 49.6718 / 47.2285 | 76.3805 / 74.8654 | 126.052 / 122.094 | 248.637 |

The extra base junction/singular study also counts: all three base studies total
**202.367s CPU /
197.762s GPU** callbacks
and **400.869s** combined study
wall. Study walls start after imports; command walls include startup and
serialization. Allocation/probe/checkpoint operations are separate. Private CPU
development timings are excluded from matched GPU evidence. No qualified
full-fit/MCMC speedup is inferred.

### Realistic workloads and current allocation policy

[The new analysis report](analysis_workloads.md) curates the completed independent
foreground work: three pinned current TESS products, original/JAX CPU and a
four-process CPU baseline, six preliminary ellc fits and failed refinement
controls. On all 3,150 V505 eclipse-window observations, retained median changed
calls were 2.30s original serial, 3.27s JAX CPU and 1.26s original four-process.
These are separate models/cost boundaries from the Roche quadrature experiment.
All four ellc finite refinement pairs fail D<=.1, including the finest at
D=.161 for sigma1e-4. No real-data GPU, final posterior or ESS/sec result is claimed.

The published pinned-input CPU driver reproduces two full observation proposals:
original arrays match retained and four-process outputs bit for bit; JAX CPU
passes existing curve/density gates. Those smoke checks do not replace the
retained eight-proposal timing study. Raw FITS/NPZ and operational logs remain
private; curated scientific figures and reports are linked from the analysis page.

Fifteen <=300s systemd units used 270s internal deadlines and unconditional
boundary verification. Before the user changed allocation policy, windows076–082
restored the exact original production service with durable advances of
**40,960, 43,008, 42,240, 43,008, 43,008, 43,008, 43,008 cells**. Windows083–090 instead preserve the **active user-requested
Fifth42 hold**, verifying the exact stopped checkpoint and strict latest
receipt/witness/endpoints at every entry/cleanup. Source/config/plan/service pins
remain unchanged. This is latest-evidence verification, not a historical re-audit.
The controlled supervisor restart preserved completed results and science source.
Current production state is **paused_for_elisa**. The supervisor restores the
original service with durable advancement only after genuine ELISa completion.
Test command sums range from 64.025 to 181.820s;
no unit exceeded its bounded allocation.

Frozen source/data: **626 files**, aggregate SHA256
`d585fbade92fec032e354719399c8f70ee2df0857cf221de3fef37ae2e53dde7`. Four files added; one experimental packet
dispatch changed; all other 621 baseline files remain byte-identical. See
[curated science and boundary evidence](ada_validation/background-m017.json),
[source manifest](ada_validation/background-m017-source-hashes.json), and
[API/reproduction/limits](experimental_gpu_backend.md#curved-eclipse-edges-and-accumulated-likelihood-controls).

Next prioritize the ready **batched Roche geometry GPU port**: review/integrate
it, run actual GPU controls and matched original/vectorized NumPy/JAX CPU/GPU
physical-proposal batches, then compose resident geometry into complete fresh
fields, curves and likelihoods. Keep the remaining field/boundary likelihood
stops active. No cached-curve nuisance-only substitute, physical-fit/MCMC
qualification or whole-assignment completion is claimed.

## Previous milestone: curvature-consistent fields and reflected comparison curves

`CurvatureFieldLightCurve` now uses actual Roche cell centres, point normals and
SI gravity, curved cell areas and area-consistent thermal normalization through
full mutual irradiation and fresh radiance. Complete reflected curves and original
physical likelihood callbacks pass CPU/GPU implementation checks. All **29 new
hardware tests passed without skips**; CPU validation passed 21 with eight GPU
skips. The preceding 500 hardware controls remain under byte-identical 619
baseline files: **529 cases of additive coverage**, not a fresh 529-case suite.
Original CPU/default engines, kernels, fitters and fallback routes are unchanged.

**Finite field-refinement distances now pass the 1e-4 diagnostic for these
fixtures. Accumulated likelihood errors still prevent physical-fit/MCMC
qualification.** Field2→3 changes the original observation log likelihood by
0.18–1.33; finer eclipse boundaries change it by about 17–25. A passing
finite distance gate does not establish a converged posterior or negligible bias.

### One geometry for finite fields and donor coupling

Each cell's collocation ray is the normalized mean of its angular corner rays.
Original-potential CPU radial roots put its centre on the actual Roche surface;
point normals and gravity use the original analytic gradient there. Order-4
Duffy quadrature integrates actual curved area over each parent patch. Evaluate
the original base quarter, then mirror it exactly. No scalar mesh correction is
applied. G*M_primary/a² scales both components, matching the original force
convention, including q in the secondary's dimensionless gradient.

The original unspotted gravity-darkening and renormalization formulas now use
these curved areas. Device point-gravity and thermal kernels retain fresh arrays
into the existing tiled full-donor reflection and blackbody emission. Original
cooler-first iteration, bolometric LD prefix and visibility conventions remain.
CK04 explicitly uses original CPU radiance/LD interpolation and unsupported-case
semantics. This is finite collocation irradiation, not a continuous illumination
solution or proof of reflected energy balance.

All requested field levels share the **same actual clipped quadrature packets**
within each fresh candidate batch, with checked angular ancestry. Independent
reference and GPU packet digests and adaptive traces match exactly. Fields remain
constant in their angular cells; roots, areas, ancestry, clipping and original
LD/table preparation remain CPU bridges. No cross-candidate physical cache is
used. Levels 0–3, 1–4 unique field levels and common level max(field levels)..3
are subject to the existing 40000-face/component and boundary/node/resource
guards. Unsupported/resource errors abort outside prior penalties; they never
silently coarsen or change physical priors. Existing broader CPU routes remain.

The independent reference applies **unchanged original CPU equations to this new
geometry**. It is a different finite numerical method from original faceted
whole-curve synthesis. A same-packet comparison with the preceding original
field2/correction2 model gives actual D=0.104319 and
nominal enclosure D=0.12261; its accuracy
preflight aborts. This is a recorded method difference, not implementation parity.

### Independent controls and complete-curve accuracy

The new tests check original per-face gravity/thermal/reflection/radiance,
independent angular-32 Brent-root areas, actual equipotential residuals, SI force
scaling, intrinsic curved-measure luminosity closure, uniform β=0/no-reflection
cell-level independence, and exact intrinsic preservation at zero albedo. They
cover partial/total/non-eclipsing, unequal q/spin, logarithmic/square-root LD,
CK04 and a small occultor. Changed physics/response/source, disabled FP64,
capacity/unsupported semantics and unchanged original CPU before/after controls pass.

Across complete-curve profiles, maximum normalized/raw/component-relative
implementation differences are **6.66134e-16 /
1.24077e-24 W/m² /
1.3411e-15**. Existing gates remain
3e-13 relative with zero absolute tolerance for raw/component flux and
3e-13/2e-12 for normalized curves. Maximum intrinsic luminosity-ratio error is
5.55112e-16; maximum cell-centre relative
potential residual is 1.55431e-14.
Intrinsic closure is not reflected-energy closure.

The partial fixture uses 57 contact-focused phases and V/bolometric bands.
Other complete curves retain their actual contact samples in the report.
Actual D uses sigma=1e-4 with original ln_f=-2 effective noise. Qualification
retains **nominal sigma independently of ln_f** and the unchanged conservative
component enclosure D<=0.1. This is a finite diagnostic, not a universal
astronomical accuracy standard.

| Case | Field levels | Common level/depth | Actual D | Nominal component enclosure D |
| --- | --- | --- | --- | --- |
| partial_bb | 1→2 | 2/3 | 0.030739 | 0.0448764 |
| partial_bb | 2→3 | 3/2 | 0.00809795 | 0.0113973 |
| partial_bb | 1→2 | 2/4 | 0.0307391 | 0.0448764 |
| total_bb | 1→2 | 2/3 | 0.0351376 | 0.0483436 |
| partial_ck04 | 1→2 | 2/3 | 0.00555745 | 0.00574601 |
| small_occultor | 1→2 | 2/3 | 0.0653274 | 0.0664051 |

All listed field controls pass. Partial field1→2 is stable under boundary
depth3→4. Independent fixed-field integration controls also pass:

| Fixed field | Integration change | Actual D | Nominal enclosure D |
| --- | --- | --- | --- |
| curvature_field2 | level2/depth3→level3/depth2 | 1.458e-05 | 1.55872e-05 |
| curvature_field1 | depth3→4 | 0.0314724 | 0.0321253 |
| curvature_field2 | depth3→4 | 0.0314724 | 0.0321253 |

These comparisons do not bound uncomputed finer field/boundary limits. Straight
leaf chords and empirical topology probes remain. Neither total curved area nor
finite adjacent-level agreement certifies every local field or hidden contour.

### Original observation likelihoods and matched costs

Fresh base/changed/JD proposals retain original physical serializers,
constraints/priors, phase/JD conversion, reducers/repeats, third light, mean
normalization and noise/density. Maximum mapped prediction/posterior differences
are **6.66134e-16 /
1.30967e-09**, within the unchanged 1e-6 density
gate. Out-of-prior vectors still return original -inf before synthesis. Numerical
accuracy failures stay outside candidate penalties, including nuisance-noise
inflation.

Each proposal's level2/3 fields use common level3 at depths2 and3, in **separate
independently restored windows**. The fine preflight checks source/proposal and
physical/phase signatures against the saved complete coarse curve. The coarse
control explicitly omits its own boundary qualification. Both sets of callback
costs and study walls count. Fine-field and boundary likelihood controls retain
the data-dependent residual cross term in |delta logL|<=sum(|r|E+E²/2):

| Proposal | Finite control | Nominal enclosure D | Actual delta logL | Absolute logL enclosure | Distance preflight |
| --- | --- | --- | --- | --- | --- |
| base | field2→3 | 0.00478513 | 1.32718 | 2.6776 | pass finite pair |
| base | boundary depth2→3 | 0.0283042 | 17.037 | 18.2145 | pass finite pair |
| changed | field2→3 | 0.00357479 | 0.181135 | 2.46928 | pass finite pair |
| changed | boundary depth2→3 | 0.0314852 | 25.2411 | 27.0756 | pass finite pair |
| jd | field2→3 | 0.00478513 | 1.32718 | 2.6776 | pass finite pair |
| jd | boundary depth2→3 | 0.0283042 | 17.037 | 18.2145 | pass finite pair |

The remaining likelihood changes are material even though these finite distance
preflights pass. No curvature-field fitting/sampler backend is automatically
registered; no optimized fit, parameter-bias recovery, chain convergence or
ESS/sec claim follows. Synthetic controls establish neither observational nor
physical-model adequacy. Real-data and JAX CPU whole-analysis assessments remain
separate work; JAX CPU already executes the new path in the CPU tests.

The matched costs below are **two-field comparison batches**, including
constructors, all fresh fields, common CPU geometry/clipping/hash, all transfers
and every raw/normalized/component host curve. Same unit, affinity 0–15, Nice 10,
one thread, FP64; compilation incurred is included, code/response/shape caches can
warm. These are single measurements, not distributions or ordinary one-model
MCMC proposal throughput. Output contracts differ from preceding three-model
studies, so their timings cannot isolate a hardware gain.

| Case | Field levels; common level/depth | CPU reference batch s | GPU hybrid batch s |
| --- | --- | --- | --- |
| partial_bb | 1,2; 2/3 | 43.6038 | 47.6761 |
| partial_bb | 2,3; 3/2 | 66.2321 | 64.1288 |
| partial_bb | 1,2; 2/4 | 86.5214 | 95.1956 |
| total_bb | 1,2; 2/3 | 55.6987 | 61.2063 |
| partial_ck04 | 1,2; 2/3 | 44.6793 | 48.4713 |
| small_occultor | 1,2; 2/3 | 50.9719 | 54.8988 |

Both comparison callback timers start before physical serialization/constraints
and construction, then include original prior, prediction and density. Each
returns the fine curve through original mapping while computing both field
models. Same-candidate mapping-only oracles are outside these timers and never
count as fresh synthesis. The paired control totals are:

| Proposal | Coarse CPU/GPU s | Fine CPU/GPU s | Summed CPU/GPU s | Summed study wall s |
| --- | --- | --- | --- | --- |
| base | 47.1415 / 44.2456 | 79.4989 / 77.483 | 126.64 / 121.729 | 248.86 |
| changed | 46.8144 / 45.7936 | 80.8776 / 77.6324 | 127.692 / 123.426 | 251.606 |
| jd | 46.347 / 44.031 | 79.3427 / 76.5529 | 125.69 / 120.584 | 246.755 |

Some high-field hybrid measurements are slightly faster than the reference;
most lower-field batches are slower. Common CPU geometry remains substantial.
These measurements establish no qualified fitting/MCMC acceleration. Study walls
start after imports; command walls include Python/import startup and serialization.
Allocation probes/checkpoint/restoration are separate, and CPU development
timings are excluded from matched evidence.

### Evidence and continuation

Thirteen independently restored systemd allocations used RuntimeMaxSec=300,
internal 270s deadlines and unconditional exact-original-service restoration.
All owned units ended inactive/success. Test command sums ranged from
87.274 to 184.601s. Pinned source/config/plan, exact original
command/service, matching resume records and latest receipt/witness/endpoints
were checked. Durable advances after restoration were **43,008, 40,960, 43,008, 40,960, 43,008, 41,728, 43,008, 40,704, 43,008, 41,984, 45,056, 42,240, 43,008 cells**.
This is latest-evidence verification, not a historical production re-audit.
Raw operational records remain private.

Frozen source/data: **622 files**, aggregate SHA256
`e85c1bd2b5075b7376d9159fea41f7dfb95da01e019926cbe47bd96e6a9febfd`. Three opt-in module/test/driver files
were added; all 619 baseline source/data files remain byte-identical. See
[curated report](ada_validation/background-m016.json),
[source manifest](ada_validation/background-m016-source-hashes.json), and
[API/reproduction/limits](experimental_gpu_backend.md#curvature-consistent-finite-fields).

Next improve eclipse-boundary integration and control accumulated likelihood
errors with independent curved-field convergence checks, before qualifying
1e-4 physical fits/MCMC. All accuracy stops remain active; the assignment continues.

## Previous milestone: shared quadrature isolates field and correction errors

`SharedFieldLightCurve` now synthesizes complete reflected comparison batches
with **identical clipped quadrature for every field variant**. This removes
boundary changes from within-batch field/correction comparisons. All **28 new
hardware tests passed without skips**; CPU validation passed 22 with six explicit
GPU skips. The preceding 472 hardware controls remain under byte-identical
616 baseline files. This is additive coverage of **500 cases**, not a fresh
500-case hardware suite. No original CPU/default, kernel or fitter changed.

Implementation parity passes. **Holding quadrature fixed still leaves failed
1e-4 field/correction accuracy controls.** Increasing boundary depth barely
changes those field distances. Finer observation boundaries pass the finite
distance gate but still cause meaningful accumulated likelihood changes.
Complete 1e-4 physical-fit/MCMC qualification remains open; no optimized fit,
converged chain or ESS/sec claim follows.

### A common physical integration domain

The original nested edge projection changes angular midpoint placement but
keeps each midpoint on its parent's great-circle edge. Descendants partition
the parent cone in exact arithmetic. Direct 3x3 solves check **every common-cell
corner** against its indexed field ancestor using conditioning-scaled normwise
FP64 coefficient/reconstruction guards. Condition above 1e5, escaped ancestry
or exhausted resources abort. No nearest-face reassignment or interpolated
parent field is substituted. Independent great-circle membership searches and
closed-form spherical-triangle solid angles control this mapping.

An early componentwise inverse-product check spuriously rejected exact seam
axes. A direct solve and normwise conditioning/reconstruction checks resolved
that CPU development issue; failed evidence remains private. Original parity
and numerical-accuracy gates remain unchanged. These checks account for FP64
roundoff; they are not interval geometry certificates.

At each phase/component, one fresh true-Roche clipped packet serves every
variant: the same cosine, area and common-cell ancestry arrays. SHA256 digests
include rule/grid/phase/component/node-count metadata and actual packet bytes.
Independent CPU-reference and GPU runs must produce identical hashes and
adaptive traces. All geometry is rebuilt for each physical candidate.

The fixed three-model batch uses:

| Model | Original finite field level | Original correction-angle level |
|---|---:|---:|
| A | 1 | 1 |
| B | 1 | 2 |
| C | 2 | 2 |

Correction angle is seed_degrees/2**level. A→B changes only the argument of the
original scalar correction while retaining raw topology and cached
pre-correction gravity/centres/normals. All later original areas, thermal state,
irradiation, LD and radiance are rebuilt. B→C refines those finite fields at
fixed correction angle. A→C is the total field change. Off-diagonal correction
is a diagnostic ablation, not a new calibrated model, prior or accuracy repair.

### Complete curves and independent accuracy controls

Reference fields/reflection/radiance and element-flux arithmetic use original
CPU primitives. Device kernels retain the preceding state/reflection/fresh BB
emission/weighted LD path; real CK04 explicitly uses CPU radiance/LD. Geometry,
ancestry and clipping remain common CPU bridges. The standalone experiment
preserves original phase mapping, repeats/order, third light, distance and
normalization. It differs from the original faceted whole-curve method.

Maximum normalized/raw/component-relative implementation differences are
**6.66134e-16 /
1.65436e-24 W/m² /
1.38583e-15**. Existing gates remain
3e-13 relative with zero absolute tolerance for raw/component fluxes and
3e-13/2e-12 for normalized curves. Per-face original field controls, changed
physics/response/source identity, exact singleton equivalence with the preceding
boundary engine, and unchanged original CPU before/after controls pass.
Tests cover reflected partial/total/non-eclipsing, unequal q/spin,
logarithmic/square-root LD, CK04 and the small-occultor fixture.

For the partial fixture's original 57 contact-focused phases and V/bolometric
bands, common grid level 2 holds all quadrature fixed within each batch.
D uses sigma=1e-4 and original ln_f=-2 effective noise; qualification uses
**nominal sigma independently of ln_f** with the unchanged enclosure D<=0.1 gate.

| Finite field control | Actual D at depth 3 | Nominal component enclosure D at depth 3 | Enclosure D at depth 4 |
|---|---:|---:|---:|
| correction only | 0.088094 | 0.215472 | 0.215472 |
| field grid fixed correction | 0.315132 | 0.531741 | 0.531741 |
| total field change | 0.349761 | 0.38528 | 0.38528 |

The component enclosure is conservative: primary/secondary, phase and
third-light normalization changes can cancel. Norms/enclosures do not add even
though signed normalized changes telescope. All three listed component
preflights retain their accuracy stops. The nearly unchanged field distances
under finer boundaries isolate an upstream finite-field/correction limitation
for this fixture, rather than the coupled field/boundary comparison used before.
They do not quantify an uncomputed continuous field.

The following comparison changes only common boundary depth 3→4 at fixed
fields and common base partition:

| Model | Actual D | Nominal component enclosure D | Preflight |
|---|---:|---:|---|
| field1_correction1 | 0.0314731 | 0.032126 | pass finite pair |
| field1_correction2 | 0.0314731 | 0.032126 | pass finite pair |
| field2_correction2 | 0.0314726 | 0.0321255 | pass finite pair |

A separate common-partition control changes level2/depth3→level3/depth2,
matching dyadic leaf scale while preserving **each variant's field model**:

| Model | Actual D | Nominal component enclosure D | Preflight |
|---|---:|---:|---|
| field1_correction1 | 1.45798e-05 | 1.5587e-05 | pass finite pair |
| field1_correction2 | 1.45798e-05 | 1.55869e-05 | pass finite pair |
| field2_correction2 | 1.45802e-05 | 1.55873e-05 | pass finite pair |

These are finite integration controls. A pass does not certify an uncomputed
boundary or posterior. Straight leaf chords, seven-point topology probes,
resource guards and the preceding physical domain remain; no continuum or
sub-probe contour certificate is claimed. Total, CK04 and small-occultor full
contact curves and every raw/component/normalized array are retained in the
curated report.

Independent CPU intrinsic T4 moments use unchanged original temperatures and
**the same order-4 curved area quadrature** for every variant. No temperature is
renormalized. The ratio is sum(T⁴A)/(T_eff⁴ sum(A))−1; this is intrinsic,
not reflected energy balance.

| Model | Component | Original corrected-area closure | Same fields, shared curved-area closure |
|---|---|---:|---:|
| field1_correction1 | primary | 0 | -2.72032e-07 |
| field1_correction1 | secondary | 2.22045e-16 | -4.5194e-08 |
| field1_correction2 | primary | 0 | -2.72032e-07 |
| field1_correction2 | secondary | 2.22045e-16 | -4.5194e-08 |
| field2_correction2 | primary | -1.11022e-16 | -6.83961e-08 |
| field2_correction2 | secondary | 0 | -1.13624e-08 |

Uniform correction rescaling cancels in original intrinsic normalization;
refinement still changes the finite temperature representation. These small
intrinsic biases alone do not explain or repair the full reflected-field curve
differences. Curvature-consistent fields and irradiation/radiance convergence
need independent validation.

### Original likelihood semantics and complete comparison costs

Fresh base/changed/JD proposals use original serializers, constraints, priors,
phase/JD mapping, reducers/repeats, third light, final mean and noise/density.
Maximum mapped prediction/posterior differences are
**4.44089e-16 /
9.02219e-10**, within the unchanged 1e-6 density
gate. All variants independently match original prediction-layer mapping;
out-of-prior vectors retain original -inf before synthesis. No physical model
is cached across candidates. Numerical preflights remain outside candidate
penalties, including nuisance-noise inflation.

The table retains the data-dependent residual cross term in
|delta logL| <= sum(|r|E+E²/2). Common grid level 2 and depth 4 serve all field
variants; the boundary preflight compares depth 3→4 for fine fields C.

| Proposal | Finite control | Nominal enclosure D | Actual delta logL | Absolute logL enclosure | Preflight |
|---|---|---:|---:|---:|---|
| base | correction only | 0.0675657 | 1.43253 | 27.9804 | pass finite pair |
| base | field grid fixed correction | 0.152132 | 48.8533 | 83.6336 | abort |
| base | total field change | 0.127437 | 50.2858 | 75.0193 | abort |
| base | boundary depth3→4 | 0.0283027 | 17.0359 | 18.2137 | pass finite pair |
| changed | correction only | 0.0504445 | 0.884455 | 29.1158 | pass finite pair |
| changed | field grid fixed correction | 0.118172 | 34.6396 | 78.3019 | abort |
| changed | total field change | 0.0974696 | 35.5241 | 69.6007 | pass finite pair |
| changed | boundary depth3→4 | 0.0314966 | 25.2504 | 27.0858 | pass finite pair |
| jd | correction only | 0.0675657 | 1.43253 | 27.9804 | pass finite pair |
| jd | field grid fixed correction | 0.152132 | 48.8533 | 83.6336 | abort |
| jd | total field change | 0.127437 | 50.2858 | 75.0193 | abort |
| jd | boundary depth3→4 | 0.0283027 | 17.0359 | 18.2137 | pass finite pair |

Passing a finite distance gate does not imply a negligible accumulated
log-likelihood change. These observations' residual cross terms remain
material, and upstream fields remain unqualified. No shared-field fitter or
sampler backend is automatically registered.

The costs below are **three-model comparison batches**. Both backends include
constructors, all fresh fields, one common CPU geometry/clipping/hash pass,
all transfers and all raw/normalized/component host outputs. Same unit,
affinity 0–15, Nice 10, threads 1, FP64. Compilation incurred is included;
sources/shapes may warm caches. Single measurements, not throughput distributions.

| Case | Common level/depth | CPU reference batch s | GPU hybrid batch s |
|---|---|---:|---:|
| partial_bb | 2/3 | 43.5916 | 49.8321 |
| partial_bb | 2/4 | 86.7479 | 93.2231 |
| partial_bb | 3/2 | 58.7673 | 66.4245 |
| total_bb | 2/3 | 55.1453 | 62.1155 |
| partial_ck04 | 2/3 | 44.3688 | 50.0738 |
| small_occultor | 2/3 | 50.5406 | 57.1737 |

Both matched comparison callback timers start before physical
serialization/constraints and constructors, then include original prior,
prediction callback and density. Both compute three comparison curves and
return C through original mapping. Extra coarse preflight and same-candidate
mapping-only oracles add to main wall and are not timed as fresh synthesis.

| Proposal | CPU reference comparison callback s | GPU hybrid comparison callback s |
|---|---:|---:|
| base | 62.6784 | 70.0633 |
| changed | 63.2169 | 70.0278 |
| jd | 62.5707 | 69.3125 |

This output contract is not an ordinary single-model MCMC proposal. No qualified
fitting/MCMC acceleration, convergence or ESS inference is made. The common CPU
geometry bridge still dominates. Main walls start after imports; separately
reported command walls include Python/import startup. Operational restoration
and probes are separate; CPU development timings are excluded from matched evidence.

### Evidence and continuation

Ten independent systemd allocations used RuntimeMaxSec=300, internal 270 s
deadlines and unconditional exact-original-service restoration hooks. All owned
units ended inactive/success. Pinned source/config/plan, exact command/service,
matching resume records and latest receipt/witness/endpoints were verified.
Durable advances after restoration were **20,480, 20,480, 20,480, 18,432, 16,384, 16,384, 14,336, 14,336, 14,336, 14,336 cells**. This checks latest
evidence, not a historical production re-audit. Raw operational records stay private.

Frozen source/data: **619 files**, aggregate SHA256
`22d8bd91785d4a6a98ab2ff4fbd79997260d96e7a1ae458a6c9393d1e47c74e3`. Three files added; all 616 baseline files
remain byte-identical. See [curated report](ada_validation/background-m015.json),
[source manifest](ada_validation/background-m015-source-hashes.json), and
[API/reproduction/limits](experimental_gpu_backend.md#shared-quadrature-for-field-and-correction-comparisons).

Next develop curvature-consistent or demonstrably converged finite thermal,
irradiation and radiance fields, validate them on shared quadrature, and control
accumulated likelihood errors before qualifying 1e-4 physical fits/MCMC.
All accuracy stops remain active; the assignment continues.

## Previous milestone: fitted eclipse boundaries and conditioned small-occultor shadows

`BoundaryBinaryLightCurve` now synthesizes complete reflected curves with
boundary-fitted leaf integration. The formerly rejected small-occultor fixture
also completes, under the **unchanged FP64 stationary-residual guard**.
**All 73 hardware tests passed without skips**: 40 new boundary cases and all
33 preceding curved cases. CPU validation passed 61 with 12 explicit GPU skips.
The 399 preceding hardware controls remain under byte-identical 606 baseline
files. This is additive coverage of **472 cases**, not a fresh 472-case suite.

CPU/GPU implementation parity passes. Boundary depth, quadrature order and
coupled field controls are reported below with the original nominal-noise
accuracy stops. **Complete 1e-4 physical fitting and continuous-field accuracy
remain unqualified.** No original CPU function, default Observer, public fitter
or physical prior changed. This remains a standalone hybrid experiment.

### Conditioned shadow solves and fitted leaf boundaries

The implicit foreground tangency solve now stays in star-local coordinates.
It evaluates the original potential/analytic gradient without repeatedly adding
and subtracting the secondary's unit centre. Positive derivative, displacement
and 64eps*|gradient| residual gates remain. Extended-precision comparisons with
the unchanged original equations and independent scalar Brent tangencies
control primary/secondary, q/spin boundaries and small-star conditioning.
The signed shadow margin now has its zero at the stationary-potential
silhouette, including a nonzero line-of-sight tangency offset.

The previous potentials [8,100] fixture completes through fresh reflected
CPU/GPU full curves. Original CPU synthesis remains available with its preceding
semantics. Much higher-potential local-equation tests are predicate controls;
they do not qualify complete curves in that extended regime.

Seven probes and a safety band guide subdivision of true Roche patches.
Crossing leaf edges are bracketed at the implicit limb and foreground shadow,
then clipped polygons are triangulated while retaining parent field identity.
Actual Roche normals/Jacobians and original LD/element-flux arithmetic integrate
the resulting domain. Shadow masks are not reapplied at quadrature nodes;
nonpositive true limb cosines contribute zero.

**Leaf chords approximate the curved boundary.** Edge root brackets do not
bound chord error; probes do not certify every sub-probe contour loop. A finer
CPU development run caught unresolved sampled topology. Fixed bounded extra
subdivision now resolves that case; the failed evidence remains private.
Topology exhaustion and node-cap failures still abort explicitly, without
coarsening or converting numerical errors into prior penalties.

Analytic uniform opaque-disk overlap controls cover partial, grazing and total
eclipses. At tested depth 5 the maximum unobscured-disk-normalized error is
**3.79149e-06**, within the independently
fixed 5e-6 target; the preceding published point-mask control used depth 7.
Analytic linear-LD disk integrals differ by at most
**8.88818e-10**. These are truncation controls,
separate from unchanged CPU/GPU implementation tolerances.

### Complete curves and finite numerical accuracy

Fresh original corrected gravity-darkened/reflected thermal, LD and radiance
fields remain piecewise constant on angular parent faces. Original build order,
irradiation iteration and atmosphere/LD semantics are preserved; each physical
candidate rebuilds its fields. They do not yet establish continuous-field
accuracy or bolometric closure under the new curved integration measure.
True roots, adaptation, conditioned shadow solves and clipping are common CPU
bridges. Device corrected state/reflection/fresh BB emission and weighted LD
integration are connected; CK04 explicitly uses original CPU radiance/LD.

The reference composes original CPU finite fields/reflection/radiance and
per-element flux arithmetic with the **same new quadrature**. Differences from
original faceted whole curves are numerical-method differences, separately
retained in the report. Maximum normalized/raw/component-relative parity errors
are **5.55112e-16 /
1.24077e-24 W/m² /
1.33443e-15**. Raw/component gates remain
3e-13 relative with zero absolute tolerance; normalized gates remain 3e-13/2e-12.
Tests cover reflected partial/total/non-eclipsing, small-occultor, real CK04,
unequal q/spin and logarithmic/square-root LD cases. Original CPU before/after
controls and fresh changed-physics/response identity controls pass.

Complete profiles retain original contact-focused phases, both eclipses,
ordering/repeats, phase shift and third light. D uses sigma=1e-4 and the original
ln_f=-2 effective noise; numerical qualification uses **nominal sigma** and the
unchanged D<=0.1 budget independently of ln_f.

| Partial-fixture finite comparison | Actual D | Nominal component enclosure D | Numerical preflight |
|---|---:|---:|---|
| Boundary depth 3→4 | 0.127937 | 0.1305 | abort |
| Boundary depth 4→5 | 0.0314168 | 0.032069 | pass finite pair |
| Order 3→4 at depth 4 | 0.000275316 | 0.000300679 | pass finite pair |
| Field1/depth4→field2/depth3 | 0.349771 | 0.385315 | abort |

These enclosures concern only the computed finite pairs. A passing boundary
pair does not certify an uncomputed boundary, continuous fields or a posterior.
Changing the field grid also changes patch basis; the last row matches dyadic
leaf scale and still couples field and integration changes. It is not a pure
isolation of continuous thermal/irradiation/radiance error. Its maximum V-band
normalized change is 4.94247e-06, while mean raw V
flux changes by -3.77283e-06 relative.
Retaining both shows that normalization does not remove the shape change.
Independent angular area controls and all raw/component/normalized arrays remain in the report.

A separate CPU-only intrinsic normalization diagnostic evaluates
sum(T⁴ A)/(T_eff⁴ sum(A))−1 with unchanged original temperatures. Original areas
are independently checked against original triangle areas in SMA²; curved
areas use order-4 true Roche quadrature on each parent. No temperature is
renormalized. This finite integral does not test reflected energy balance.

| Field level | Component | Original corrected-area closure | Same fields, curved-area closure |
|---|---|---:|---:|
| 0 | primary | 0 | -1.06339e-06 |
| 0 | secondary | -1.11022e-16 | -1.76703e-07 |
| 1 | primary | 0 | -2.72032e-07 |
| 1 | secondary | 2.22045e-16 | -4.5194e-08 |
| 2 | primary | -1.11022e-16 | -6.83961e-08 |
| 2 | secondary | 0 | -1.13624e-08 |

Changing the integration measure produces a small, resolution-dependent
intrinsic closure bias. Its size alone does not explain or repair the coupled
field/curve difference; continuous thermal/irradiation convergence still needs
independent work. The report retains the exact fixture and calculation method.

### Original likelihood semantics and matched costs

Fresh physical proposals use the original serializer, constraints, priors,
phase/JD mapping, reducers/repeats, third light, final mean, noise law and density
callback. Maximum mapped prediction/posterior differences are
**3.33067e-16 /
1.25146e-09**, within the unchanged 1e-6 density
gate. Out-of-prior vectors retain -inf rejection before synthesis.

The observation preflight compares boundary depths 3 and 4 at fixed fields.
Its absolute likelihood enclosure includes the data residual cross term,
sum(|r|E+E²/2), for this finite pair. Numerical accuracy failures remain outside
candidate penalties, including when nuisance noise is inflated.

| Proposal | Actual D | Nominal component enclosure D | Actual delta logL | Absolute logL enclosure | Preflight |
|---|---:|---:|---:|---:|---|
| phase base | 0.107561 | 0.114969 | 68.2752 | 72.543 | abort |
| phase changed | 0.100709 | 0.112886 | 89.7342 | 97.7686 | abort |
| JD base | 0.107561 | 0.114969 | 68.2752 | 72.543 | abort |

The residual cross term produces finite log-likelihood changes of 68–90 in
these observations. Small curve differences alone therefore cannot qualify
the posterior. All three observation preflights retain their accuracy stops.

Seconds below include constructor, fresh fields, CPU geometry/adaptation/
clipping, integration, transfers and raw/normalized/component host outputs.
Both backends run in the same unit with affinity 0–15, Nice 10, threads 1 and
FP64. Compilation incurred is included; process caches can warm. These are
single observations, not throughput distributions.

| Case | Field level | Order/depth | CPU reference s | GPU hybrid s |
|---|---:|---|---:|---:|
| partial_bb | 1 | 3/3 | 20.9307 | 23.1216 |
| partial_bb | 1 | 3/4 | 40.0783 | 42.2096 |
| partial_bb | 1 | 3/5 | 82.7403 | 84.6861 |
| partial_bb | 1 | 4/4 | 46.3026 | 48.8831 |
| partial_bb | 2 | 3/3 | 42.5978 | 44.8998 |
| total_bb | 1 | 3/4 | 51.2638 | 53.8309 |
| partial_ck04 | 1 | 3/4 | 41.4432 | 43.8341 |
| small_occultor | 1 | 3/4 | 46.3488 | 48.9043 |

The following **matched complete callback** timers both start before physical
serialization/constraints and constructor, then include original prior, the
same original prediction layer with prepared-binary construction/phase mapping,
and density. They replace the preceding unmatched callback sub-timers.
Extra coarse numerical preflight and independent controls add to the complete
study wall; Python/import startup is additionally in command walls.

| Proposal | CPU reference callback s | GPU hybrid callback s |
|---|---:|---:|
| phase base | 28.2441 | 30.8933 |
| phase changed | 28.5808 | 30.0027 |
| JD base | 27.9913 | 29.4547 |

CPU geometry remains the principal bridge. No qualified fitting/MCMC speedup,
optimized fit, chain convergence or ESS/sec claim follows from these results.
No boundary fitting/sampler backend is automatically registered.

### Evidence and continuation

Ten independently restored systemd allocations used RuntimeMaxSec=300 and
internal 270 s deadlines. Every owned unit ended inactive/success. Exact pinned
service/command/source/config/plan, matching resume records and latest
receipt/witness/endpoints were verified. Durable restoration advances were
**43,008, 47,616, 46,848, 47,104, 47,104, 47,104, 43,008, 43,008, 43,776, 42,496 cells**. This verifies latest evidence, not a historical production
re-audit. Raw operational records remain private.

Frozen source/data: **616 files**, aggregate SHA256
`bfaa1836ea53ad53475f1d374a208ae3ca2506553d64811260dfde1e828c0dbd`. Five files were added; only the preceding
experimental `gpu_curved.py` gained explicit geometry hooks with identical
defaults. Its complete 33-test hardware suite was rerun. All 606 earlier files
remain byte-identical. See the [curated report](ada_validation/background-m014.json),
[source manifest](ada_validation/background-m014-source-hashes.json), and
[API/reproduction/limits](experimental_gpu_backend.md#boundary-fitted-roche-integration).

Next isolate upstream field/correction error from boundary error on a shared
integration protocol, assess curved-measure thermal normalization, and extend
the original observation preflight to finer boundaries before qualifying
physical fits at 1e-4. Accuracy stops remain active; the assignment continues.

## Previous milestone: true curved Roche integration and implicit eclipses

`CurvedBinaryLightCurve` now synthesizes complete reflected light curves using
true Roche quadrature points/normals/Jacobians and foreground shadows defined by
the same implicit surface. It integrates fresh finite original thermal,
reflection, LD and radiance fields. **All 33 new hardware tests passed with no
skips**; CPU development passed 27 with six explicit GPU skips. The preceding
**399 hardware controls are retained under byte-identical 606 baseline files**.
This is additive coverage of 432 cases, **not a fresh 432-case hardware run**.
No original CPU function, default or fitter was changed.

Curved area integration improves substantially and implementation parity passes.
**The current point-masked boundary rule still fails the 1e-4 observational
budget.** Continuous-field accuracy remains unqualified. The new method is a
standalone numerical experiment; no physical fit, continuum posterior or MCMC
throughput qualification is claimed.

### What changed, and what remains finite

Each angular parent triangle maps a barycentric vector v to ray v/|v|.
Duffy Gauss weights use |det(a,b,c)|/|v|³. Original CPU potential roots give true
points; analytic gradients give outward normals and
r²|gradPhi|/(gradPhi dot radial) for dA/dOmega. Scalar mesh correction is absent
from **quadrature geometry**. Original corrected finite gravity/thermal,
full-donor irradiation, atmosphere/LD and radiance fields remain piecewise
constant on parent faces, retaining their preceding build order. They are
rebuilt for each candidate and need their own spatial convergence study.

Visibility uses true normals. Occultation tests the original foreground Roche
potential along the observer ray, using an inscribed-radius bound and a guarded
stationary-potential solve. Independent Brent tangencies control the predicate.
Both emission and occultation now refer to the same implicit geometry.
Seven probes and a safety band select further boundary subdivisions; actual
quadrature nodes receive true visibility/shadow masks. **This is a finite
empirical boundary indicator, not an interval bound or exact boundary integral.**
Node exhaustion and unsupported numerical states abort rather than coarsen.

CPU roots, normals/Jacobians, adaptation and masks are explicit bridges. The
preceding device field/reflection/blackbody and weighted LD integration kernels
are connected; CK04 examples select original CPU radiance/LD. These are
**hybrid** complete curves. Original Observer/fitters and whole-CPU routes remain
unchanged; the curved mode has no automatically registered fitting backend.

CPU parity uses original finite fields/reflection/radiance and original element
flux arithmetic with the **same new quadrature**. Differences from unchanged
original faceted whole curves are separately reported numerical-method changes.
Maximum normalized/raw/component-relative implementation differences are
**6.66e-16 / 8.27e-25 W/m² /
1.3e-15**. Existing normalized 3e-13/2e-12 gates and raw/component
3e-13 relative, zero-absolute gates remain. Real tests cover reflected partial,
total, non-eclipsing, CK04 and q/spin/non-linear-LD cases.

### Independent geometry and boundary controls

Original radius Newton roots independently check vector bisection; angular
finite differences check Jacobians; scalar Brent line solves check shadows.
Analytic sphere area/LD and uniform opaque-disk overlap controls include partial,
grazing and total eclipses. Tested depth 7 sphere overlap error is <=5e-6 of the
unobscured disk, a separate numerical truncation target. It is not a bound on
Roche contacts, continuous fields or every physical parameter.

Area ratios below compare curved triangle quadrature with independent
full-sphere angular order32 integration. Geometry uses SMA² units. For context,
the preceding original-corrected 3° faceted primary area differed by about
-1.34e-4 in the same partial fixture. Accurate area alone does not qualify
projected eclipse integrals.

| Field level | Gauss order | Primary relative area difference | Secondary relative area difference |
|---|---:|---:|---:|
| 1 | 3 | 9.2296e-10 | 8.96532e-10 |
| 1 | 4 | -7.16871e-13 | -6.56808e-13 |
| 2 | 3 | 1.45999e-11 | 1.40477e-11 |

Partial/total BB and partial CK04 profiles retain the original 57/69/57
contact-focused phases, both eclipses, ordering/repeats, phase shift and third
light. Validation uses process-only MIN=3°/MAX=12°; library defaults are unchanged.
D is ||delta/sigma_eff|| for sigma=1e-4 and original ln_f=-2. Numerical gates use
**nominal sigma independently of ln_f**, with budget D<=0.1.

| Partial-fixture finite comparison | Actual D | Nominal component enclosure D |
|---|---:|---:|
| Boundary depth 2→3 | 2.18266 | 2.22258 |
| Boundary depth 3→4 | 1.16752 | 1.18612 |
| Order 3→4 at depth 4 | 0.590024 | 0.600822 |
| Field1→2, same integer depth 4 | 0.414484 | 0.455288 |
| Field1/depth 4→field2/depth3 | 0.367162 | 0.401853 |

All listed enclosures fail. Changing the field grid also changes angular patch
basis and absolute boundary leaf size. Equal integer depth is therefore coupled
field/boundary refinement. The last row matches dyadic leaf scale, but its
angular bases still differ; it does not isolate continuous-field error. The
failed boundary/order controls preclude a 1e-4 inference claim regardless.

A CPU-only small-occultor probe, potentials [8,100], is explicitly rejected by
the unchanged FP64 stationary-residual guard. Original CPU synthesis completes
for that fixture. No inaccurate new curve or prior penalty is substituted.
Conditioned local-coordinate shadow solves need investigation; no curved GPU
parity or accuracy claim is made for that probe. Earlier broad physical routes
and CPU fallbacks remain available with their preceding semantics.

### Original observation and likelihood semantics

Fresh temperature/inclination/third-light/noise proposals retain original
serialization, constraints, priors, phase/JD mapping, per-band reducers/repeats,
mean normalization and noise law. Maximum mapped prediction/posterior errors
are **4.44e-16 / 1.6e-09**, within the unchanged
1e-6 density gate. Boundary-depth preflight retains the data residual cross term:
|delta logL| <= sum(|r|E+E²/2), for the **measured finite pair only**.

| Proposal | Actual D | Nominal component enclosure D | Actual delta logL | Absolute logL enclosure |
|---|---:|---:|---:|---:|
| phase base | 2.25351 | 2.30192 | 182.603 | 1100.39 |
| phase changed | 0.784328 | 0.890478 | -153.675 | 680.758 |
| JD base | 2.25351 | 2.30192 | 182.603 | 1100.39 |

All these numerical preflights abort outside candidate penalties. Out-of-prior
proposals retain original -inf rejection without synthesis. A finite pair passing
would not certify an uncomputed boundary or continuous thermal/irradiation/
radiance field. No new optimized fit, chain mixing, convergence or ESS/sec result
is asserted.

### Matched complete-curve costs

Seconds include constructor, fresh finite fields, CPU true geometry/adaptation/
shadow masks, integration, transfers and raw/normalized/component host outputs.
Both backends run in the same unit with affinity 0–15, Nice 10, threads 1 and FP64.
Compilation incurred is included; sources/shapes can warm process caches. These
are single observations, not throughput distributions.

| Case | Field level | Order/depth | Curved CPU reference s | Curved GPU hybrid s |
|---|---:|---|---:|---:|
| partial_bb | 1 | 3/2 | 8.17432 | 10.4086 |
| partial_bb | 1 | 3/3 | 17.0342 | 19.2064 |
| partial_bb | 1 | 3/4 | 35.6134 | 37.9711 |
| partial_bb | 1 | 4/4 | 49.6624 | 52.2414 |
| partial_bb | 2 | 3/4 | 80.8448 | 83.442 |
| total_bb | 1 | 3/4 | 48.2952 | 50.8076 |
| partial_ck04 | 1 | 3/4 | 36.3646 | 39.1261 |
| partial_bb | 2 | 3/3 | 38.5222 | 40.5752 |

The CPU geometry bridge dominates; these measurements show no overall GPU
acceleration. Curve main walls also include independent area checks, original
faceted references, finite controls and report conversion. Their Python/import
startup is included in separately reported command walls, all below 245 s.

Likelihood sub-timers have different setup boundaries: the CPU reference includes
the original prediction-layer serializer/time mapping; the selected sub-timer
starts with a prepared physical binary. Their ratio is **not** a full callback
speedup. All serialization, constructors, priors, coarse preflight and extra
references are charged in the complete likelihood-study wall. Accuracy failures
prevent interpreting any cost as qualified fitting or ESS throughput.

### Evidence and continuation

Ten separately registered systemd allocations used RuntimeMaxSec=300, internal
270 s deadlines and unconditional original-service restoration hooks. All owned
units ended inactive/success. Exact pinned service/command/source/config/plan,
matching resume records and latest receipt/witness/endpoints were verified.
Durable advances after restoration were **40,960, 10,240, 8,192, 10,240, 8,192, 8,192, 8,192, 8,192, 4,608, 47,104 cells**. This validates latest
evidence, not a full historical production audit. Raw operational records stay
private.

Frozen source/data: **611 files**, aggregate SHA256
`a30a2fab21c2e9102f3a9147b9d939b221e8518dbf01d0529b6d03088dfefe19`. All 606 previous files remain byte-identical;
five additive files implement and validate the new method. See
[curated report](ada_validation/background-m013.json),
[source manifest](ada_validation/background-m013-source-hashes.json), and
[API/reproduction/limits](experimental_gpu_backend.md#true-roche-patch-quadrature-and-implicit-eclipses).

Next implement boundary-fitted quadrature and investigate conditioned shadow
solves, then independently qualify boundary and continuous-field error before
physical fitting at 1e-4. All accuracy stops remain active; the assignment continues.

## Previous milestone: selective complete curves and correction/normalization assessment

A fixed selective surface protocol now runs complete reflected curves and
original physical-observation likelihoods with fresh CPU/device state. Independent
angular Roche area/volume integration separates geometric surface error from
CPU/GPU parity and flux normalization. **All 399 hardware tests passed, zero
failures/errors/skips**, across independent 89/75/235-test windows. The final CPU
suite passed **247 with 152 explicit GPU skips**; there are 29 new tests.

The scientific result is a negative control: **selective root indicators plus
the current global correction do not qualify 1e-4 accuracy and can worsen curve
differences despite adding triangles**. Correction ablations, random controls,
raw-flux changes and all failed numerical budgets are retained. No selective
fit, continuum posterior or MCMC throughput claim follows from passing parity.

### Fixed selective geometry and fresh full physics

`SelectiveMesh` refines chosen seed-quarter patches from a uniform base level.
Only marked edge midpoints are projected onto the original Roche equipotential.
Selected triangles split into four; neighboring triangles receive conforming
2/3/4-child patterns. Symmetry seams remain closed. Tests check edge incidences,
Euler characteristic 2, positive areas, ancestry and all three refinement levels;
empty/full plans exactly reproduce the corresponding uniform meshes.

The pilot folds nominal-noise-weighted root-group flux changes over the four
quarters, using all supplied bands/phases and original maximum normalization.
Freeze the plan before later evaluations. It is a finite coupled-model indicator,
not a local truncation-error estimate. Seed topology hashes prevent index
reassignment; physical vertex movement is allowed only with compatible topology.
Every changed candidate rebuilds gravity/thermal state, full-donor irradiation,
LD/radiance, observer projection, eclipse coverage and integration. No physical
prediction is cached and no proposal triggers automatic replanning.

CPU seed construction, selective roots/triangulation, orientation/hulls/coverage
and host grouping remain bridges. Device corrected state, tiled reflection,
fresh blackbody emission and integration are connected; CK04 uses explicit
original CPU radiance/LD. These are **hybrid** curves. Original CPU/default
correction, public fitting classes and broader full-CPU fallback remain unchanged.
The new standalone numerical protocol raises on unsupported topology/numerics;
it never silently substitutes a coarser prediction or rejects a physical prior.

New tests compare per-face points, centres, normals, areas, gravity/log(g) and
reflected temperatures with independent original CPU primitives at the unchanged
3e-13 relative / 64eps absolute gates. Complete raw observed flux uses 3e-13
relative with zero absolute tolerance; normalized curves retain 3e-13/2e-12.
Maximum raw/normalized errors in the new profile are
**1.65e-24 W/m² / 5.55e-16**.
Original raw-to-direct-normalization controls agree within
**2.22e-16**.

### Selective refinement is not an accuracy repair

Both eclipses, contact-focused phases, repeats/order, phase shift and third light
are retained in reflected V/bolometric fixtures. Uniform levels 2/3 have
8640/34560 faces per component. Validation uses process-only MIN=3°/MAX=12°;
library defaults are unchanged. The table uses sigma=1e-4 and original ln_f=-2
for actual D=||delta/sigma_eff||. The component enclosure uses **nominal sigma**,
independently of nuisance noise, and the numerical budget remains D<=0.1.

| Case | 50% root plan faces [primary,secondary] | Uniform2→3 actual D | Selective50→uniform3 actual D | Nominal component enclosure D |
|---|---:|---:|---:|---:|
| partial_bb | [22656, 22656] | 2.55969 | 6.19561 | 6.4486 |
| total_bb | [22784, 22720] | 1.85497 | 5.19884 | 5.45303 |
| partial_ck04 | [22672, 22736] | 2.76963 | 6.18688 | 6.39701 |

A 50% root plan includes green neighbors, so its face fraction exceeds 50%.
The added triangles do not make its finite changes smaller. The partial fixture
also retains 25/75% selections, a seeded random plan with the same selected-root
count (different final face counts), and identical-raw-topology correction
ablations. No result was selected to make a failed budget pass.

| Partial fixture model | Faces [primary,secondary] | Primary corrected area relative error | Raw V mean change vs uniform3 | Actual D to uniform3 |
|---|---:|---:|---:|---:|
| selective50 | [22656, 22656] | -0.000240352 | -0.000196811 | 6.19561 |
| selective25 | [16016, 16016] | -0.000299191 | -0.000200034 | 15.0688 |
| selective75 | [29104, 29152] | -0.000151269 | -9.42873e-05 | 5.14372 |
| random50_root_count_control | [23264, 23296] | -0.000231868 | -0.000170474 | 7.27245 |
| selective50_coarse_correction | [22656, 22656] | 0.000179183 | 0.000217087 | 5.63332 |
| selective50_fine_correction | [22656, 22656] | -0.00030036 | -0.000256012 | 6.29289 |

`correction="density"` applies the original scalar formula at
base_angle*sqrt(base_faces/new_faces); it is an explicit unvalidated mixed-grid
convention. `"coarse"`/`"fine"` hold that angle at the base/fully refined values.
All coupled physics is rebuilt for each ablation. Changing this global scalar
alone does not establish a repair. Finite group scores and the original scalar
calibration do not automatically transfer to a nonuniform grid. Different native
1.5° triangulations remain separate finite controls, not exact truth or
implementation oracles; their per-band comparisons are in the report.

### Independent geometric audit and normalization controls

The CPU audit integrates the implicit Roche surface over dOmega=dmu dphi with
Gauss-Legendre mu, periodic phi, original analytic gradients and independent
Brent radial roots. The radial Jacobian is r²|gradPhi|/(gradPhi dot radial); volume
uses r³/3. Sphere controls recover exact area/volume; separate angular orders
16/32 change tested area/volume by at most **2.22e-16 relative**.
These geometric integrals do not certify eclipse curves or thermal/spectral
physics. Raw and corrected triangulated geometry are reported separately,
using SMA²/SMA³ units and dimensionless relative errors.

| Case | Mesh | Primary corrected area relative error | Primary corrected volume relative error |
|---|---|---:|---:|
| partial_bb | uniform2 | -0.000134109 | -0.000462336 |
| partial_bb | uniform3 | -6.77787e-05 | -0.000166968 |
| partial_bb | selective50 | -0.000240352 | -0.000512802 |
| total_bb | uniform2 | -0.000134109 | -0.000462336 |
| total_bb | uniform3 | -6.77787e-05 | -0.000166968 |
| total_bb | selective50 | -0.000234296 | -0.000503083 |
| partial_ck04 | uniform2 | -0.000133808 | -0.000461715 |
| partial_ck04 | uniform3 | -6.77035e-05 | -0.000166813 |
| partial_ck04 | selective50 | -0.00024195 | -0.000515335 |

Scaling vertices by one factor changes area and volume as factor²/factor³.
Agreement of one integral cannot establish shape or eclipse accuracy. Raw observed
flux and original maximum normalization are both retained: removing a common
scale through normalization does not remove numerical shape or likelihood error.
Mesh correction, mixed quadrature and eclipse boundaries need independent work;
same-grid CPU/GPU fallback cannot cure their shared error.

### Fixed-plan original observation likelihoods

The same plan is retained across changed temperature/inclination/third-light/noise
proposals. Original constraints, physical serialization, phase/JD mapping,
per-band reducers/order/repeats, final mean, priors and noise law are preserved.
Original CPU synthesis and density callbacks independently control both the
selective and uniform numerical models. Maximum prediction/posterior differences
are **4.44e-16 / 4.31e-09**;
the unchanged density gate is 1e-6. Likelihood changes agree to
**2.18e-09**.

| Proposal | Actual D (sigma_eff) | Nominal component enclosure D | Actual delta logL | Absolute logL enclosure |
|---|---:|---:|---:|---:|
| phase base | 2.48669 | 2.58198 | 1102.44 | 1436.15 |
| phase changed | 1.61113 | 1.75311 | 929.055 | 1264.22 |
| JD base | 2.48669 | 2.58198 | 1102.44 | 1436.15 |

The enclosure retains the residual cross term, sum(|r|E+E²/2), for the measured
finite pair. `allow_same_level=True` explicitly compares alternative topologies
sharing a physical/seed/source snapshot. It never bounds an uncomputed grid or
the continuum. Numerical preflight failures abort outside candidate penalties,
including nuisance-noise inflation; out-of-prior proposals retain original -inf
rejection without numerical work. No selective public fitting/sampler backend
is registered. Existing public fit/sampler parity was revalidated by the full
suite; no mixing, convergence, bias recovery or ESS/sec result is claimed.

### Matched complete costs

Within each profile, JAX CPU/GPU return identical host snapshot plus raw-curve
outputs under the same affinity 0–15, Nice 10, threads 1 and FP64. Seconds include
fresh CPU roots/triangulation/state/coverage, reflection/radiance/integration,
all per-call transfers and parent grouping. Constructors, including any device
initialization, are outside model timers and inside driver main wall. Compilation
incurred is included; later shapes/source tables may reuse process caches. These
are single observations, not throughput distributions.

| Case | Mesh | Complete JAX CPU snapshot+raw s | Complete GPU snapshot+raw s |
|---|---|---:|---:|
| partial_bb | uniform3 | 6.62064 | 2.68262 |
| partial_bb | selective50 | 3.73421 | 2.27899 |
| total_bb | uniform3 | 6.63783 | 2.73839 |
| total_bb | selective50 | 3.96558 | 2.54338 |
| partial_ck04 | uniform3 | 6.18911 | 2.11148 |
| partial_ck04 | selective50 | 3.30487 | 1.95305 |

Pilot uniform evaluations and plan freezing, independent references/angular
controls, geometry/correction/error comparisons, constructors and report
conversions remain part of full main walls:
**90.99, 75.18, 88.30 s**
(partial curves, other curves, observations). Full command walls including
Python/import startup are **92.76, 76.94, 90.01 s**.
Mapped prediction/density model timings are in the JSON. Independent original CPU
curves have a smaller output contract than snapshots. Accuracy failures prevent
interpreting these ratios as qualified physical-fitting throughput or ESS gains.

### Evidence, restoration and continuation

Six independently registered systemd GPU allocations used RuntimeMaxSec=300,
internal 270 s deadlines, verified graceful checkpoint pause and unconditional
original-service restoration hooks. All owned units ended inactive/success.
Matching resume records, exact service/command, pinned source/config/plan and
latest checkpoint receipt/witness/endpoints were verified. Durable advancement
after each restoration was **32,768, 22,528, 14,336, 4,096, 47,104, 47,104 cells**. This validates the latest evidence,
not the entire historical production journal. Raw operational records stay private.

Exact repository source/data fingerprint: **606 files**, aggregate SHA256
`0c536362d5fa1480cccb066b9cd716257a404252c8ab72be135b8359721ab0a3`.
See [curated results](ada_validation/background-m012.json),
[source manifest](ada_validation/background-m012-source-hashes.json), and
[API/reproduction and limits](experimental_gpu_backend.md#fixed-selective-surface-refinement-and-correction-audit).
Reproduce with the full experimental suite and
`scripts/validate_gpu_selective.py --device cpu --part curves --case partial_bb --output partial.json`,
`--case others`, and `--part observations`; GPU variants need independent bounded
restoration. No tolerance, numerical budget or physical prior was relaxed.

Next replace the global mixed-grid correction/indicator prescription with
independently validated curved-surface quadrature and consistent eclipse boundary
integration, then qualify a fixed 1e-4 numerical protocol before physical fits
or chains. The assignment remains incomplete; all accuracy stops remain active.

## Previous milestone: nested surfaces and a finite contact/likelihood error estimator

The new standalone `NestedBinaryLightCurve` builds complete reflected curves on
conforming, nested Roche meshes and returns immutable root-face flux groups.
A new estimator reconstructs measured changes between two computed grids,
encloses their finite prediction/likelihood changes, and identifies limb/contact
contributions. **All 370 experimental hardware tests passed, zero failures,
errors or skips**, across independent 113/257-test windows. The final frozen CPU
suite passed **221 with 149 explicit GPU skips**. There are 23 new tests.

The numerical accuracy gate still fails at sigma=1e-4. These are useful finite
error indicators, **not continuum bounds or scientifically qualified fine fits**.
CPU/default behavior and the existing fitting protocol remain unchanged.

### Fresh nested geometry and coupled physics

A paired 12° seed produces **540, 2160, 8640 and 34560 faces per component**
through levels 0–3. Each split projects new edge midpoints onto the original
Roche equipotential using CPU Brent roots. ELISa's original seam identities keep
the four quarters closed; tests check edge incidences, Euler characteristic 2,
positive areas, symmetry/ancestry and roots against independent original CPU
radius calculations, including q/spin boundaries. Level 0 preserves the original
raw mesh exactly. The original scalar mesh correction is used at seed degree /
2**level on the new topology; this convention does not establish a new error law.

Every evaluation rebuilds corrected geometry/gravity, intrinsic temperatures,
tiled irradiation with all donor faces, fresh radiance/LD and mutual eclipse coverage.
CPU midpoint roots, orientation/hulls/coverage and extra parent grouping remain
explicit bridges; device state/reflection/integration and blackbody emission are
connected. CK04 uses explicit original CPU radiance/LD. These are **hybrid** curves.
The independent CPU reference composes original primitives and unchanged observer
routers; its compatibility script remains available after moving the reusable
implementation into `elisa.experimental.gpu_reference`.

Maximum normalized complete-curve implementation difference across the new
study is **4.44e-16**. Level 0 agrees exactly with unmodified dense CPU
in these controls.
The existing 3e-13 relative/2e-12 absolute curve gates remain unchanged.
Separate native 1.5° meshes are numerical cross-checks, not implementation oracles
or exact truth. Existing dense controls through 2° and broad physics/fallback,
public fitting and sampler-equivalence tests are included in the full suite.

### What the estimator bounds

Signed root-face-group changes plus third-light offset and
summation/normalization residual changes reconstruct the actual computed curve
difference. A host arithmetic guard is included. Summing absolute component
changes gives a finite enclosure; the more conservative root-face L1 envelope
and noise-weighted local scores expose cancellation and boundary contributions.
A source/candidate/seed/passband/settings signature rejects stale comparisons.

Observation reducers preserve unequal band support, order/repeats and original
final mean normalization. The accuracy gate uses **nominal y_err, independently
of ln_f**; a regression proves noise inflation cannot evade the stop.
Likelihood diagnostics retain the original sigma²*(1+exp(2ln_f)) law. With data,
**|delta logL| <= sum(|r| E + E²/2)**, where r is the coarse residual and E the
component enclosure in effective-sigma units. This includes the residual cross
term and bounds only the measured finite pair.

`require_budget=True` raises `ResolutionNotQualifiedError`. Call this standalone
preflight outside candidate penalties; neither accuracy errors nor unsupported
refined numerics become physical-prior cuts or silent coarse CPU substitutes.
The existing public resolution-qualified fitter is separate and unchanged.
A finite pair passing cannot certify an uncomputed grid, continuum posterior,
finite exposure or physical systematics. No convergence order, monotonicity or
extrapolation is assumed. Local scores describe globally coupled changes;
selective refinement must be rebuilt and independently validated.

### Surface accuracy remains unqualified

These reflected V/bolometric fixtures retain both eclipses, contact-focused
phases, phase shift, third light and ordering/repeats. D is concatenated-band
||delta/sigma_eff|| with sigma=1e-4, ln_f=-2; the numerical enclosure budget uses
nominal sigma and requires D<=0.1. The last column compares the finest nested
mesh to a separately triangulated native 1.5° mesh. Neither is continuum truth.

| Case | D, L0→L1 | D, L1→L2 | D, L2→L3 | Nominal component enclosure, L2→L3 | Different 1.5° grid D |
|---|---:|---:|---:|---:|---:|
| partial_bb | 23.9128 | 10.4334 | 2.55969 | 2.65844 | 0.812315 |
| total_bb | 48.8287 | 9.4912 | 1.85497 | 1.93113 | 2.10432 |
| partial_ck04 | 31.9652 | 8.1244 | 2.76963 | 2.80678 | 1.04029 |

All finest pairs and different-grid comparisons exceed the budget. Root-face L1
enclosures are much larger than component enclosures because of cancellation;
limb/partial-eclipse groups contribute roughly 8–11% of last-pair L1 in these
cases. This diagnostic does not justify refining only eclipse contacts or bound
uncomputed surface error. No adaptive accuracy gain is claimed.

### Original physical-observation likelihood checks

Two phase proposals and one JD proposal change physical shape/temperature and
nuisance parameters, using original serialization, time mapping, prior, third
light, mean normalization and likelihood. Independent CPU predictions/densities
are compared at both levels 2/3. Maximum observation prediction difference is
**4.44e-16**; maximum original posterior difference is
**4.66e-09**, within the unchanged 1e-6 density gate.
The measured likelihood changes agree with independent original callbacks to
**3.18e-09**. Out-of-prior proposals retain original
-inf rejection before nested evaluation. Every tested numerical preflight stops.

| Proposal | Actual D (sigma_eff) | Nominal component enclosure D | Actual delta logL | Absolute logL enclosure |
|---|---:|---:|---:|---:|
| phase base | 1.02954 | 1.12007 | 261.194 | 639.097 |
| phase changed | 0.685956 | 0.817223 | 376.345 | 987.899 |
| JD base | 1.02954 | 1.12007 | 261.194 | 639.097 |

These data come from the existing finite-grid fixture; the table measures
resolution changes against its residuals. It is not an optimized fit or a bias
estimate. No new physical-fit/MCMC backend, mixing, convergence or ESS/sec claim
is made. The existing public fit/sampler parity controls were revalidated.

### Matched transfer-inclusive costs

Both backends ran inside the same bounded profile allocation with identical
host snapshot outputs, affinity 0–15, Nice 10, threads 1 and FP64. Seconds include
fresh CPU projection/roots/state/bridges, reflection/radiance/eclipse calculation,
all transfers and extra CPU parent grouping. Constructors are outside row timers
and inside driver main wall, including any constructor device initialization.
Compilation incurred is included; later matching shapes may reuse process kernels. These are single observations, not throughput
distributions. Independent original-CPU curve timings have a smaller output
contract and are not used as snapshot speedup baselines.

| Case | Finest JAX CPU snapshot s | Finest GPU snapshot s | L2+L3 JAX CPU snapshots s | L2+L3 GPU snapshots s |
|---|---:|---:|---:|---:|
| partial_bb | 6.50975 | 2.71428 | 8.99906 | 4.69848 |
| total_bb | 6.39777 | 2.39009 | 8.60943 | 4.0045 |
| partial_ck04 | 6.01605 | 2.13586 | 8.11529 | 3.54088 |

A full numerical preflight also pays the comparison cost, included in full
driver main walls: **103.22 s curves /
60.78 s observations**, including independent
references, auxiliary comparisons, setup and report conversions, with imports
excluded. Full command walls including Python/import startup are
**104.98 s /
62.50 s**, respectively. Matched
observation snapshot/mapping/density timings are in the JSON report. Accuracy
failures prevent treating any of these costs as qualified fitting throughput.

### Scope, evidence and continuation

The standalone experiment supports levels 0–3, at most 40000 faces per component
and 128 snapshot phases in the existing circular detached symmetric unspotted,
nonpulsating q/spin/inner-radius/thermal domain. Unsupported refined numerics
raise explicitly. Existing CPU/default and hybrid complete fallback semantics
remain unchanged. This is uniform nested refinement with an error indicator,
not selective adaptive surface/contact quadrature.

Four independently restored systemd GPU allocations used the run's registered
unit prefix, RuntimeMaxSec=300, internal 270 s deadline, checkpoint pause and
unconditional original-service resume hooks. All owned units ended
inactive/success. Matching resume records, pinned source/config/plan, exact
service/command and latest receipt/witness/endpoints were verified. Durable
advancements after restoration were **43,008, 45,056, 45,056, 45,056 cells**. This is latest evidence
validation, not a full historical production audit. Raw operational logs stay
private.

Exact runtime/data fingerprint: **602 files**, aggregate SHA256
`49bb6af553e3cf60e567f24dd71b32fb88b9034ce3d418435c9f96627dbb8cde`.
See [curated machine-readable report](ada_validation/background-m011.json),
[source manifest](ada_validation/background-m011-source-hashes.json) and
[API, contracts and reproduction](experimental_gpu_backend.md#nested-surfacecontact-change-estimator).
Run the full experimental suite and
`scripts/validate_gpu_surface_error.py --device cpu --part curves --output curves.json`
or `--part observations`; GPU variants require independent bounded restoration.

Next use the validated coupled-model indicators to implement selective surface
quadrature and check mesh-correction/normalization errors independently, then
qualify a fixed 1e-4 numerical protocol over physical parameters before finer
fits or chains. Accuracy stops and separate CPU/GPU gates remain required.

## Previous milestone: bounded irradiation and finer complete curves/fitting

The new opt-in `reflection="tiled"` processes bounded receiver-row tiles against
**all donor faces**, enabling complete reflected curves with **36,180 faces per
star at 1.5°**. Fresh device temperatures feed the expanded blackbody emission
path; original CPU orientation/hulls/eclipse coverage remain explicit bridges.
An independent CPU reference uses original distance/gamma/LD and observer
primitives without allocating the original full irradiation matrices. Paired
component grids now make finer fitting/refinement choices explicit.

CPU/default dense GPU behavior remains unchanged. **All 347 experimental hardware
tests passed, zero failures/errors/skips** across independent 90/257-test windows
(125.06/217.38 s).
The full private CPU suite passed 201 with 146 explicit GPU skips; 19 new hardware
targets cover the tiled path. Matched complete-curve, physical-fit and changed
posterior profiles passed the existing gates. The finer surfaces still fail the
noise-scaled accuracy budget in the three eclipse fixtures: this milestone
improves the usable numerical resolution, **not continuum certification at 1e-4**.

### Original irradiation with bounded memory

`TiledDeviceReflection` preserves the original cached geometry/temperature build
order, CPU visibility and bolometric LD selection, LD coefficient prefix order,
cooler-first sequential iteration, equal-temperature tie choice and final
quarter-face mirroring. It recomputes coupling tiles each iteration, with no
pair cutoff, fitted heating approximation or cached physical prediction.
Receiver rows are padded only for the last tile; all donor faces contribute to
each original sum. Complete curves use 32-row tiles; direct API permits sizes 8–64; sizes 8/17/32 were
validated, including uneven tails and zero iterations/albedo.

At 1.5° in the partial blackbody fixture, the two updates include
**162,786,832 donor pairs per iteration**. At most
**620,288 pairs** occupy a tile. Compiled reflection temporary
workspace is **43.83 MiB GPU /
45.74 MiB JAX CPU**,
plus 6.14 MiB inputs and 0.62 MiB outputs.
These are compiled-program accounting, **not total allocator/compiler/runtime or
whole-curve memory peaks**. The original full visible distance matrix alone has
325,573,664 elements (2.43 GiB FP64),
plus a three-coordinate join matrix and other arrays. No dense 1.5° run is claimed.
Time remains quadratic despite bounded memory.

The explicit tiled mode admits 40000 faces per component, 268435456 prepared
reflection pairs and the existing 0–32 iteration bound. All prior circular
unspotted detached q/spin/radius/thermal restrictions remain. Blackbody emission
gets an explicit 40000-face capacity with its existing bounded spectral chunks,
consuming fresh reflected device temperatures; its default 8192 cap is unchanged.
Fine CK04 examples explicitly use original CPU radiance/LD. Fine eclipse coverage
uses CPU while face integration runs on device. These are **hybrid** curves.
Unsupported states retain complete CPU fallback/raise, never a physical-prior cut;
missing device/FP64 remains an infrastructure failure outside candidate penalties.

### Independent CPU controls and explicit component grids

`scripts/fine_mesh_reference.py` composes original CPU mesh, gravity and intrinsic
thermal normalization with original NumPy/Numba distance/gamma/LD functions in
independent tiles, then the unchanged Observer/circular curve router. It owns its
Observer/model and changes no global CPU function or setting. Coarse original CPU controls validate all four laws, both cooler-first orders,
equal-temperature ties, real table LD and temperatures. Unmodified dense complete-curve controls
extend through 2°. Maximum independent/dense curve
difference is **2.22e-16**. At 1.5°, the separately
labelled independent finite reference is compared with both JAX CPU and GPU;
maximum normalized implementation difference across the fine study is
**5.55e-16**. The existing 3e-13 relative/2e-12
absolute curve gates remain; no finest grid is treated as exact truth.

`mesh_mode="paired"`, with `phase_mode="observations"`, explicitly gives both
components the requested factor at every target and reference evaluation. Native
fitting retains the original temperature/radius-dependent secondary adjustment
and configured clamps (3–12° in this validation environment). A nominal 2.5° primary in the tested fitter produces an
inferred **9.158° secondary**; a finer primary alone therefore does not imply
comparable refinement of both stars. Paired mode preserves original physical
priors, serialization, observation mapping/repeats, third light and noise, while
explicitly choosing a different numerical grid. No global minimum is lowered.
`MeshResolutionCheck` now permits original CPU references down to 2°; references
below 3° require paired mode. Accuracy failures still abort outside candidate
penalties, and same-grid CPU fallback cannot repair shared surface error.

### Finer meshes help, but the 1e-4 accuracy budget still fails

The three reflected cases retain actual contact-focused phases, both eclipses,
ordering/repeats, phase shift and third light, with V/bolometric bands and
**8916/20276/36180 faces per star at 3/2/1.5°**. The table compares successive
finite original-primitives CPU curves using nominal sigma 1e-4 and original
ln_f=-2. Joint budget D<=0.1 corresponds to expected likelihood loss 0.005 only
if the finite reference generates the data; it does not bound a residual cross
term or continuum error. Instantaneous flux is used; no exposure integration.

| Case | D,3°→2° | D,2°→1.5° | Maximum 2°→1.5° change (sigma_eff) |
|---|---:|---:|---:|
| partial_bb | 2.11437 | 0.48843 | 0.10778 |
| total_bb | 2.89375 | 1.48231 | 0.22040 |
| partial_ck04 | 3.24404 | 0.57080 | 0.16438 |

All three last pairs exceed the joint budget. The smaller sampled changes do not
establish monotone convergence or certify 1.5°. Contact/surface quadrature still
needs adaptive refinement or a justified error estimator. CK04/LD physical model
systematics and missing spectral information remain separate limits.

### Physical fitting and likelihood at finer paired meshes

Both fit grids use the **same independent 1.5° finite truth/data**: T_primary 6500 K,
T_secondary 5000 K, inclination 86.4°, seed 1010 Gaussian noise at nominal 1e-4.
The original public LSQ fits one temperature, constraining secondary=primary−1500
and fixing inclination. It retains max_nfev 16, two-point/diff_step 1e-4 and original
convergence/serialization settings. This isolates a physical thermal fit at two
costly surface resolutions; it is not a general posterior-recovery certificate.

| Both component grids | CPU fitted T_primary (K) | Chi² | D vs 1.5° at fitted parameters |
|---|---:|---:|---:|
| 3° | 6498.056361 | 32.959631 | 0.68212 |
| 2° | 6498.379925 | 31.522794 | 0.24935 |

These parameter shifts combine noise and finite mesh effects, not a measured
continuum bias. Refinement comparisons use fresh 1.5° predictions at **each fitted
parameter set**, including original normalization/errors and the actual data
residual cross-term in JSON. CPU/GPU fit differences are at most
**2.71e-06 K / 2.31e-11 chi²**,
within unchanged 0.01 K/1e-5 gates. Changed physical posterior callbacks differ by
at most **3.13e-10**, within the unchanged 1e-6 gate.
The current full suite also reruns original public sampler replay; the finer
experiment adds density callbacks, **not new convergence/mixing/ESS evidence**.

### Matched transfer-inclusive costs

Seconds per partial BB V/bolometric curve. Each first complete call includes
compilation incurred; previously compiled shapes/code may already be warm.
Binary/shared original Observer and engine constructors are outside row timers,
inside complete profile wall. Original CPU starts at a prepared Observer;
independent CPU includes an owned Observer copy; hybrid includes its per-call
Observer setup/copy, fresh state/LD bridges, CPU coverage and all transfers.
These are matching host curve outputs, not device-resident lower bounds.

| Both grids | Original CPU first | Independent CPU first | Hybrid JAX CPU first | Hybrid GPU first |
|---|---:|---:|---:|---:|
| 3° | 1.132 | 1.028 | 2.303 | 1.897 |
| 2° | 4.317 | 2.674 | 2.969 | 2.042 |
| 1.5° | unperformed | 8.089 | 6.745 | 2.582 |

One repeated complete call per partial BB mesh, with the same hybrid API:

| Both grids | Hybrid JAX CPU repeat (s) | Hybrid GPU repeat (s) |
|---|---:|---:|
| 3° | 0.269 | 0.166 |
| 2° | 1.129 | 0.358 |
| 1.5° | 5.641 | 0.756 |

Other cases retain first-call costs only in JSON. The small sample count does
not establish general throughput or performance at certified 1e-4 accuracy.

| Public LSQ, constructor/setup/compile/callbacks/serialization (s) | Original CPU | Hybrid JAX CPU | Hybrid GPU |
|---|---:|---:|---:|
| 3° | 15.677 | 6.013 | 3.662 |
| 2° | 55.090 | 16.592 | 6.179 |

All matched paths use affinity0–15/Nice 10, threads 1, FP64. The GPU driver
explicitly selects CPU and GPU JAX backends in each restored allocation.
Synthetic references, auxiliary post-fit/refinement/parity/list conversion and
compiled-memory diagnostics lie outside fit/curve subtimers and inside full
profile wall times **99.87/52.64/121.59 s**
(curves/3° fits/2° fits). Development cores16–19/Nice19 costs are excluded.

### Reproduction, restoration and next work

```text
python -m pytest tests/tests_experimental -q -rs
python scripts/validate_gpu_fine.py --device gpu --part curves --output fine-curves.json
python scripts/validate_gpu_fine.py --device gpu --part fitting --fit-level 3 --output fine-fitting-3.json
python scripts/validate_gpu_fine.py --device gpu --part fitting --fit-level 2 --output fine-fitting-2.json
```

The validation process configuration uses MIN_DISCRETIZATION_FACTOR=3 and
MAX_DISCRETIZATION_FACTOR=12; library defaults are unchanged.
Use the private CPU wrapper while production runs; GPU variants require their
own bounded checkpoint/restore systemd allocation. Registered windows 018–022 used
**127.69/220.37/102.53/55.21/124.16 s** of test commands,
each under 270 s internal/300 s systemd limits. Each independently restored the
exact pinned service/source/config/plan, verified matching checkpoint resumes
and latest receipt/witness/endpoints, and observed
**44,288/43,008/43,008/47,104/84,992** durable cells of resumed progress.
All five units finished inactive/success. This is not a full historical production
re-audit; raw operational/process/environment logs remain private.

See [curated measurements](ada_validation/background-m010.json),
[597-file frozen runtime/data manifest](ada_validation/background-m010-source-hashes.json)
and [current status](ada_validation/progress.json). Compact sorted-key aggregate:
`7e60dcdd457638146307fbf48c3eb3e9755864cb7da4a2e21f27cd48b91c4a40`. Only explicit experimental adapters/validation
files changed; original CPU implementation/defaults remain unchanged.

Next implement and test adaptive surface/eclipse quadrature or an error estimator
at actual observation noise, concentrating on contacts and checking all affected
thermal/radiance/coverage state afresh. Retain independent finite references,
paired grid semantics, accuracy stops and separate CPU/GPU parity gates. No
continuum posterior or MCMC convergence/ESS claim is supported yet.

## Previous milestone: observation-phase synthesis and mesh-qualified fitting

Physical CPU/hybrid fitting now has opt-in `phase_mode="observations"` and
`MeshResolutionCheck`. The first requests fresh synthesis at actual observation
phases; the second runs fresh original CPU target/two-refinement comparisons
**before** LSQ/MCMC candidate penalties. Accuracy failures abort the fit/run,
including when emergency penalties are enabled. Original CPU engines, defaults,
constraints, priors and native interpolation behavior remain unchanged.

This implements an explicit numerical fitting protocol, **not a claim that the
tested surfaces have reached continuum accuracy at 1e-4**. That stricter fixture
fails the configured finite-mesh check as expected. A separate 5% noise control
exercises successful qualified public fitting and sampler replay.

### Actual observation support and original scientific conventions

Observation mode uses the existing direct predictor with no interpolation support,
bypassing unused uniform/manual/adaptive phase preparation. Original dynamic
JD/phase conversion, `PRESERVE_INPUT_PHASES`, phase shift, cross-cycle inputs,
unequal per-band grids, ordering/repeats and reducers remain. So do support-mean
third light, maximum-then-observation-mean normalization and the original
sigma²*(1+exp(2ln_f)) likelihood. No finite-exposure integration is added.

The driver compares explicit original direct synthesis, the CPU protocol wrapper
and hybrid synthesis across phase/JD/preserved-phase scenarios and three changed
physical vectors each, including temperature constraints, shape, third light and
noise. Maximum normalized CPU-wrapper/direct difference is
**0**; hybrid/direct difference is **4.44e-16**
and posterior difference **1.16e-09**.
The unchanged gates are 3e-13 relative/2e-12 absolute for predictions and 1e-6
absolute for accumulated posterior values.

The native coarse uniform-support fixture differs by up to
**0.11976** normalized flux. This includes sparse
interpolation and the original support-dependent third-light convention; it is
not a CPU/GPU error or a pure interpolation convergence estimate. Explicit
observation mode consistently selects the original cadence convention.

### Fresh finite-mesh checks abort outside prior penalties

`MeshResolutionCheck(refinement_degrees=(8.0,6.0), max_distance_sigma=0.1)` with
target 12° evaluates original CPU curves on 12/8/6° meshes using copied Observers,
retaining the actual passbands, atmospheres and LD. Both target-to-finest and
middle-to-finest concatenated-band distances must pass D<=0.1. The second check
guards against simple endpoint cancellation; monotone convergence is not assumed.
References must be strictly finer, decreasing and at least 3° within the already
exercised CPU range. These are empirical finite references, **not exact truth**.

The check uses original normalized observational `y_err`, independent of fitted
ln_f. Inflating nuisance noise cannot make the mesh pass. Actual posterior noise
semantics remain unchanged. The 0.1 distance budget corresponds to expected
likelihood loss 0.005 for data generated by the finite reference; it does not
bound a realized residual cross-term, omitted physics or continuum error.

At nominal 1e-4, the real reflected 28-observation B/V/bolometric fitting fixture
has D=**17.53242** (12-to-6°) and
**10.74171** (8-to-6°), failing both checks.
CPU and hybrid LSQ/MCMC raise `ResolutionNotQualifiedError` before requested
hybrid synthesis even with `PENALIZE_ALL_EXCEPTIONS=True`. No failed check becomes
a prior exclusion, large residual, invalid-model count or same-grid CPU fallback.
A failed finer reference also aborts; an actual invalid original target retains
its original physical-candidate penalty. Out-of-prior proposals retain their
original rejection without synthesis. Transient state clears on all exits.

Only successful checks proceed to the requested CPU/hybrid target. Every
candidate is checked afresh; no curve or accuracy certificate is cached. A run
that fails must be restarted under a scientifically revised numerical protocol,
rather than continuing with selective numerical prior cuts. Unsupported device
cases retain complete original CPU fallback; that does not cure shared mesh error.
CPU-only protocol wrappers and both morphologies are documented in the
[API guide](experimental_gpu_backend.md#observation-phase-and-mesh-qualified-fitting).
The native broader physical routers/device integration remain available.

### Successful control, public fits and sampler equivalence

A separate dataset assigns sigma 0.05 and matching independent Gaussian synthetic
noise, seed 909, using direct original 12° truth. This **5% control is not a relaxed
1e-4 criterion**. Its sampled candidate comparisons give joint D from
**0.01228 to 0.03460**.
Qualified CPU/hybrid predictions differ by at most
**6.66e-16** and posteriors by
**2.84e-14**, within unchanged gates.

The original public two-parameter LSQ uses 16 evaluations, two-point differences,
diff_step 1e-4 and original convergence settings. The CPU control returns
T_primary=**6000.000000 K**,
inclination=**84.789071295°**,
chi²=**14.25910955**. The weak noisy fit hits
the 6000 K lower bound, so this is computational replay evidence, not parameter
recovery or calibrated inference. CPU/GPU fitted differences are
**0 K /
4.35e-11°**, chi² difference
**2.13e-14**.

Public 8-walker/3-step runs retain identical coordinates/acceptance fractions;
stored log-density differences are at most **2.13e-14**.
These runs test callback/move/serialization equivalence. They establish no
convergence, ESS/sec, posterior precision or continuum posterior.

**All 328 experimental hardware tests passed, zero failures/errors/skips** in
independently restored partitions 126/202 (132.08/142.19 s). New protocol targets account for 29;
CPU development targets passed 19 with 10 explicit GPU skips. Full CPU experimental
suite passed 191 with 137 explicit GPU skips; the final added serial-contract
assertion separately passed before the complete hardware run. Existing Planck
exponential-overflow warnings are retained. Tests include direct phase/JD and
repeat handling, original invalid-target rejection, failed refinement, ln_f
independence, nonmonotone endpoint cancellation, explicit runtime/configuration
errors, complete unsupported-case fallback and overcontact constraint handling.

### Matched complete costs

Median **ms per complete physical posterior callback**, three changed-vector warm
samples. Original CPU columns below come from the GPU profile; the JAX CPU
profile ran sequentially in the same restored allocation. Both use identical
physics/data/output contracts, affinity0–15/Nice 10, threads 1, FP64. Each qualified
callback includes all three fresh CPU mesh checks plus requested target synthesis;
the CPU path deliberately repeats its target. Parameters, priors, CPU bridges,
dispatch and all transfers are charged. Setup and first per-instance callback/
compilation incurred are separate in JSON; process code caches can already be warm.

| Callback | Original CPU | Hybrid JAX CPU | Hybrid GPU |
|---|---:|---:|---:|
| Direct observation phases | 45.06 | 103.61 | 26.91 |
| With fresh mesh qualification | 258.88 | 320.73 | 245.18 |

| Public qualified work, constructor + fit (s) | Original CPU | Hybrid JAX CPU | Hybrid GPU |
|---|---:|---:|---:|
| LSQ | 5.037 | 6.725 | 6.327 |
| 8 walkers /3 steps | 9.143 | 11.712 | 10.126 |

Public fit subtimers include original setup, compilation incurred, qualification/
callbacks and result serialization. Ancillary dataset generation, option copies,
post-fit diagnostic prediction/parity and result-list conversions are outside
those subtimers and included in complete CPU/GPU driver wall times
**52.92/52.39 s**. CPU reference work dominates
qualified callbacks; small finite-grid timing differences do not establish
performance at 1e-4 accuracy. No ESS claim follows from these short replay runs.
Development checks on cores16–19/Nice19 are excluded from matched timings.

### Reproduction, restoration and next work

```text
python -m pytest tests/tests_experimental -q -rs
python scripts/validate_gpu_protocol.py --device cpu --output protocol-cpu.json
python scripts/validate_gpu_protocol.py --device gpu --output protocol-gpu.json
```

Use the private CPU wrapper while production runs. GPU variants require an
independent bounded checkpoint/restore systemd allocation. Registered windows 015/
016/017 used **134.85/145.09/109.47 s**
of test commands under 270 s internal/300 s systemd bounds. Each restored the exact
pinned service/source/config/plan, matched the checkpoint resume, verified latest
receipt/witness/endpoints and observed **6,144/4,096/2,048** durable cells of resumed progress.
All test units finished inactive/success. This is not a full historical production
re-audit; raw operational/process/environment evidence stays private.

See [curated evidence](ada_validation/background-m009.json),
[593-file source/data manifest](ada_validation/background-m009-source-hashes.json)
and [current status](ada_validation/progress.json). Only the existing hybrid fitter
adapter and validation helper changed among the previous 589 files; the new
protocol/driver/tests and unchanged prior plot helper are added. Compact
sorted-key aggregate: `803542842a961f781d36b63b27b99fffc6560d935e7a0880944b74ed27aaa7a5`.

Next address finer/adaptive surface convergence at 1e-4 for reflected eclipse
curves and physical fits, using this explicit protocol and separate CPU/GPU parity
gates. The current 3° finite reference was never certified; finer references need
bounded new validation. Missing atlas/model information, finite exposure and
long-chain convergence remain separate scientific limits. **The assignment continues.**

## Previous milestone: resolution error and physical fitting at 1e-4 uncertainty

The new opt-in `gpu_accuracy` diagnostics distinguish **implementation parity**
from finite mesh, phase-support and spectral errors. All CPU synthesis/defaults
and previous GPU runtime files remain unchanged. `resolution_error` applies the
original `sigma²*(1+exp(2 ln_f))` noise law to already mapped/normalized predictions
and reports maximum/RMS errors, joint distance `D=||delta/sigma_eff||`, and the
data-dependent chi²/log-likelihood change including its residual cross term.
It does not change a prior, fallback or parity tolerance.

The illustrative diagnostic budget **D<=0.1** bounds the expected likelihood loss
to0.005 when the finite reference generates the data. Passing it is a sampled
comparison, **not a continuum error certificate**. The target is nominal
normalized sigma`1e-4`, ln_f`-2`; physical-fit comparisons use the original
normalized observations/errors. Shared CPU/GPU resolution limits are reported
even when implementation parity passes.

![Finite mesh and phase-support errors](ada_validation/background-m008-resolution.svg)

### Mesh and phase choices dominate these accuracy tests

Three reflected cases use V/bolometric, q0.7 and partial/total blackbody or partial
CK04/table-LD eclipses, with45–69 requested phases including nearby contact
estimates, ordering/repeats, cross-cycle inputs, phase shift and third light.
Mesh levels12/8/6/4/3 degrees have **540/1280/2156/5076/8916 faces per component**.
The3° original CPU model is a finite reference. Original reflection iteration2,
atmosphere/LD and mesh correction conventions are retained.

| Case | Maximum 12° vs3° change (sigma_eff) | Joint D,4° vs3° |
|---|---:|---:|
| Partial blackbody | 4.08 | 6.03 |
| Total blackbody | 11.51 | 5.11 |
| Partial CK04/table LD | 7.47 | 3.31 |

Even the last mesh pair exceeds the joint0.1 budget. This assessment **does not
qualify3° as accurate at1e-4**, assume monotone convergence or extrapolate a
continuum limit. The partial case is visibly nonmonotone in the figure. Finer
geometry, quadrature or a justified mesh error estimate still needs work.

The phase audit uses the actual original physical fitter with28 B/V/bolometric
observations, repeated/unequal per-band grids and8° surfaces. Manual cubic
support spans[-0.1,1.1], with32/64/128/256/1024/8192 points; direct synthesis at
the observation union is the reference. At l3=0, joint D falls from736.75 at32
points to **0.02665 at8192**; this final sampled control meets the diagnostic
budget. At l3=0.2, the8192-point D remains **72.12**, with maximum19.34 sigma_eff.

That nonzero-l3 difference includes the **original synthesis-support mean used
for third light**, which changes between cadence-based direct synthesis and a
uniform support grid. It is not pure cubic interpolation error and need not
approach zero with grid refinement. Maximum-then-observation-mean normalization
and this original convention are preserved. A fitting protocol must specify its
support/l3 convention; grid density alone cannot make those two models identical.
Finite exposure integration is not included in these instantaneous fixtures.

### Spectral quadrature is small here; atlas/model accuracy remains separate

`BlackbodySpectralProbe` integrates the actual clipped passband response with
Gauss-Legendre quadrature in log wavelength, split at native throughput knots,
logarithmic panels and available Akima zero crossings. Orders8/16/32 test stability.
An owned CPU Observer callback changes only spectral emission, retaining fresh
CPU reflection, same-band LD division, actual eclipse coverage and normalization.
Native-grid control reproduces original complete curves. This CPU diagnostic
still incurs original radiance preparation and is **not GPU computation or a new
production performance path**. Changed sampled responses invalidate its snapshot.

Independent adaptive integrals validate the quadrature at3000/6500/30000 K to
1e-10 relative. Order16-to32 surface-flux changes are **1.56e-15 V /
2.23e-16 bolometric**. Whole reflected-curve/fitter comparison against native
blackbody quadrature gives maximum normalized prediction change **3.77e-8**,
joint D **0.000943**, and expected likelihood loss **4.45e-7**. That passes the
sampled diagnostic budget comfortably for this fixture.

The finite50–2,000,000 Å bolometric interval remains: its flux differs from the
infinite Planck integral by about7.01e-7 at3000 K down to7.07e-10 at30000 K.
The Planck-derived Stefan constant and legacy temperature-law constant remain
distinct, as in CPU; neither was adjusted. Oversampling these blackbody/response
functions cannot establish missing CK04 spectral information, atmosphere/LD
model accuracy or an atlas-wide accuracy claim. Real CK04 mesh/implementation
comparisons above retain the existing versioned scientific tables.

### Fitting impact and unchanged CPU/GPU gates

Both fitting engines use **the same native3° blackbody synthetic truth**, fixed seed908
noise at1e-4, and original phase mapping, normalization, constraints, priors,
third-light/noise semantics and serialization. Injected T_primary=6500 K and
inclination=86.4°. Public two-parameter LSQ uses the original16-evaluation,
two-point/diff_step1e-4 settings.

| Fit mesh | Original CPU T_primary (K) | Inclination (deg) | Minimum chi² |
|---|---:|---:|---:|
| 12° | 6508.956470 | 86.43822725 | 96.47574 |
| 6° | 6500.106505 | 86.40583154 | 21.84991 |

Relative to the injected parameters, the temperature shift drops from **+8.9565 K
to+0.1065 K** and inclination from+0.03823° to+0.00583°. This one realization
combines noise and finite mesh effects; it is not an exact bias estimator or a
continuum posterior. Improved fit quality does not certify surface convergence.

CPU/GPU differences remain within unchanged gates: at most **4.45e-16** absolute
for normalized mesh curves, **6.67e-16** for mapped phase-support predictions,
**5.06e-7 K /6.21e-10 deg** in fitted parameters and **2.43e-11 chi²**. Public
8-walker/3-step runs replay identical coordinates/acceptance fractions; stored
log-density differences are at most **3.79e-9**. These short runs assess callback/
move equivalence, **not convergence, posterior precision or ESS/sec**.

**12 targeted hardware tests passed in8.59 s, zero failures/errors/skips.** CPU
targets passed11 with1 explicit GPU skip in5.75 s. Tests independently compare
the likelihood error algebra with original cost functions, adaptive spectral
integrals, original-grid whole curves, response-revision/caller-state checks and
separate mesh-resolution failure from parity. The preceding **287-test full
hardware suite** is retained as unchanged-physics evidence: all its583 source/
data fingerprints remain identical. It was not rerun or represented as a new
full-suite pass. All new complete-curve and public-fitting profiles passed.

### Matched transfer-inclusive costs and resource fallback

Median **ms per host V/bolometric curve**. Original CPU and GPU columns below
come from the same GPU driver; the JAX CPU profile ran sequentially in the same
checkpointed window. Affinity0–15/Nice 10, threads 1, FP64. All per-call Observer,
source/CPU bridges, dispatch, evaluation and transfers are included. Binary and
one-time hybrid engine constructors are outside row timers and included in full
driver wall times. First calls/compilation are separate in JSON. There are three
warm samples at12/6° and one at8/4/3°; the4° rows are single-sample costs.

| Case / mesh | Original CPU | Hybrid JAX CPU | Hybrid GPU |
|---|---:|---:|---:|
| Partial BB /12° | 51.10 | 61.99 | 25.83 |
| Partial BB /6° | 149.58 | 480.11 | 44.94 |
| Partial BB /4° | 459.32 | 510.22 | 481.29 (CPU fallback) |
| Total BB /12° | 47.30 | 61.48 | 29.69 |
| Total BB /6° | 162.07 | 593.47 | 52.29 |
| Total BB /4° | 497.53 | 522.63 | 519.11 (CPU fallback) |
| CK04 /12° | 26.04 | 84.05 | 29.82 |
| CK04 /6° | 71.40 | 1031.59 | 57.95 |
| CK04 /4° | 217.38 | 243.07 | 238.26 (CPU fallback) |

Existing reflection pair/face and phase-count caps cause **entire original CPU
fallback at4°/3° and the denser support grids**. Those results are not counted
as GPU synthesis, partial curves, prior exclusions or a remedy for discretization
error. Finite-grid timing gains do not establish performance at the target
scientific accuracy, since surface convergence has not been established.

| Public work, all costs (s) | Original CPU | Hybrid JAX CPU | Hybrid GPU |
|---|---:|---:|---:|
| 12° LSQ | 1.095 | 4.630 | 3.597 |
| 6° LSQ | 2.281 | 6.708 | 3.767 |
| 12° 8-walker/3-step run | 1.771 | 4.717 | 2.817 |
| 6° 8-walker/3-step run | 3.811 | 10.020 | 3.106 |

These totals include new fitter construction, original setup/constraints,
compilation incurred, all callbacks and serialization; runtime caches can be
warm. Synthetic reference generation is included in complete profile wall time,
outside optimizer sub-timers. Single short runs support no general throughput or
ESS claim. Development checks on cores16–19/Nice19 while production ran are
correctness evidence and excluded from matched timings.

### Reproduction, restoration and next work

```text
python -m pytest tests/tests_experimental/test_gpu_accuracy.py -q -rs
python scripts/validate_gpu_accuracy.py --device cpu --part curves --output curves-cpu.json
python scripts/validate_gpu_accuracy.py --device cpu --part fitting --output fitting-cpu.json
python docs/ada_validation/plot_m008.py
```

Use the private CPU wrapper for development. GPU variants require an independent
checkpoint/restore systemd allocation. Registered windows013/014 used
**169.63/63.50 s** of test commands within270 s internal/300 s systemd bounds.
Both independently restored the exact pinned service/source/config/plan,
verified matching checkpoint resumes and latest receipt/witness/endpoints, and
observed **47,104/45,824 durable cells** of resumed advancement. Both test units
finished inactive/success. This is not a full historical production re-audit;
raw operational logs/process/environment snapshots remain private.

See [curated measurements](ada_validation/background-m008.json),
[589-file runtime/data manifest](ada_validation/background-m008-source-hashes.json)
and [current status](ada_validation/progress.json). The manifest adds the three
diagnostic Python files and B-response/two mesh-correction resources to the
unchanged preceding583 files. Compact sorted-key aggregate:
`0a5d555a1ef79cb5c36261ebee0ae5b68c25a2629a344392f2124e03b46be50f`.

Next implement an opt-in direct-observation synthesis and resolution-qualified
fitting protocol, with unresolved accuracy failures outside prior penalties,
then address finer/adaptive surface convergence. CPU defaults and scientific
conventions remain preserved. The target-noise surface accuracy is still
unqualified, so the assignment continues.

## Previous milestone: device mutual irradiation in complete curves and fitting

The circular detached hybrid now defaults to **device mutual irradiation**.
`gpu_reflection.DeviceReflection` builds fresh pair distances/join vectors,
directional LD and coupling matrices, executes the original cooler-first
sequential reflection iteration and mirrors final temperatures. Those device
temperatures feed fresh blackbody/CK04 emission and eclipse integration directly.
`reflection="cpu"` retains the preceding CPU-reflection hybrid for comparison.
The original CPU Observer, fitters, physics and global defaults are unchanged.

The CPU build sequence is preserved: cached centres/normals/gravity precede mesh
correction; areas and intrinsic temperatures follow it; reflection follows
intrinsic energy normalization. Original CPU helpers select reflection visibility
and quarter-face indices and obtain bolometric LD at **intrinsic** temperatures.
LD stays fixed during reflection iteration, the cooler component updates first,
the other component immediately uses that update, and equal-temperature ties
select primary first. The original symmetry helper's **LD coefficient prefix
ordering**, including varying table LD, is retained. This port does not repair
or reinterpret that CPU scientific convention.

The pipeline remains **hybrid**. CPU angular mesh/roots/topology/kinematics,
visibility/LD lookup, file/spectral preparation and observer hulls remain. Final
temperature host copies support original table/source decisions; reflection
matrices/factors stay on device during iteration. No candidate's temperatures,
geometry, LD or coupling matrices are cached. The broader `hybrid_native`
spot/eccentric/contact branch still uses its original CPU physics/reflection.

### Validation and scientific limits

**287 hardware tests passed, zero skips/failures/errors**, across two independent
windows: **85 in 100.87 s** and **202 in 142.11 s**, with no duplicate tests.
The full CPU suite passed **161**, with **126** explicit GPU skips, in 171.50 s.
Two/one warnings respectively are inherited original CPU Planck UV-tail overflow
in a finite cold-star fallback fixture. No existing acceptance gate changed.

Six detailed V/bolometric cases each have540 faces per component and45–69 requested
phases: primary cooler/q0.5/spins0.5,1.5/non-eclipsing; secondary cooler/q2/
spins1.5,0.5/total eclipse; equal temperatures/cosine LD; logarithmic LD/partial
eclipse; square-root LD/grazing; and CK04 with actual interpolated table LD.
Iteration counts are1,3,2,2,5,2. Both eclipses, nearby contact estimates, phase
ordering/repeats, cross-cycle observations, shift, third light and distance are
compared with unmodified CPU. Contact estimates locate useful phases; original
Roche surfaces/clipping remain the oracle. Existing contact-mask tests are in the
full suite. The real CK04/LD files are the preceding versioned fixtures, with
source hashes retained; these cases do not validate an entire stellar atlas.

Maximum measured GPU differences:

- Reflection coupling matrices: **1.12e-15 absolute**; gamma: **3.34e-16 absolute**.
  Isolated and complete reflected temperatures: **3.63e-16 relative**.
- Normal radiance: **3.20e-15 relative**; LD: **5.49e-16 relative**. Actual face
  flux divided by component flux: **7.98e-16 absolute**; component sums:
  **1.04e-15 relative**; coverage/SMA²: **1.00e-17 absolute**.
- Raw observed curves: **9.77e-16 relative**; normalized curves:
  **5.56e-16 absolute**. Original fitter mean-normalized predictions:
  **4.45e-16 absolute**; accumulated posterior: **3.50e-9 absolute**, at
  normalized uncertainty `1e-4` (28 B/V/bolometric blackbody or17 V/bolometric table observations).

Independent controls check that zero receiving albedo leaves that star unheated,
unilluminated faces stay unchanged, heating is nonnegative, and iterations
0/1/2/4/32 track original CPU and heating progression. Both temperature update
orders and the equality tie are tested. Receiving-face heating maxima in the
detailed fixtures range about1.59–45.43 K. Prepared packets are immutable and
pass strict no-transfer evaluation. Changed albedo, temperature, q/potential,
inclination and nuisance proposals recompute all dependent state. Resource
failure produces the exact original finite CPU posterior even in emergency
penalty mode; missing LD sources retain their original error class and disabled
FP64 raises outside candidate penalties.

Public fits retain the original 16-evaluation/two-point/diff_step1e-4 settings:

- Blackbody: `T_primary=6504.131860320 K`, `inclination=86.4018911275 deg`, chi²
  **21.9508086857**. CPU/GPU differences are **1.98e-10 K**, **3.70e-13 deg** and
  **1.08e-11 chi²**.
- CK04/table LD: `T_primary=10078.571433897 K`, `inclination=86.3998042190 deg`,
  chi² **9.4225894123**. Differences are **1.73e-10 K**, **2.85e-14 deg** and
  **2.62e-13 chi²**.
- Public 8-walker/3-step runs replay identical coordinates and acceptance
  fractions; stored density differences are at most **6.52e-9**. These are
  callback/move smoke checks, **not convergence, posterior precision or ESS/sec**.

### Matched transfer-inclusive costs

Median **ms per fresh host V/bolometric curve**, three warm samples. CPU and GPU
drivers ran sequentially in the same checkpointed window, affinity0–15/Nice10,
threads1, FP64. Every path includes fresh CPU bridges, uploads, device work and
final host outputs; Binary construction is outside every curve timer. Original
CPU and preceding-hybrid columns below come from the GPU driver; the separate
CPU profile's baselines, all samples and stage/cold costs remain in curated JSON.

| Case | Original CPU | CPU-reflection hybrid GPU | New hybrid JAX CPU | New hybrid GPU |
|---|---:|---:|---:|---:|
| Primary cooler, non-eclipsing | 44.55 | 21.80 | 18.62 | 20.87 |
| Secondary cooler, total | 56.35 | 30.12 | 108.26 | 29.35 |
| Equal temperatures, cosine | 48.42 | 28.46 | 192.01 | 27.99 |
| Logarithmic, partial | 52.07 | 26.80 | 101.28 | 25.12 |
| Square-root, grazing | 50.92 | 25.17 | 94.52 | 24.95 |
| CK04, actual table LD | 27.63 | 30.44 | 90.87 | 30.78 |

Combined engine construction costs **0.632 s**; first diagnostic full GPU calls,
including newly incurred compilation/CPU preparation, cost **0.698–2.519 s**.
Small differences against the preceding hybrid, fixed order and few samples
do **not establish a reflection speedup**. Blackbody gains over original CPU
mostly retain earlier device emission work; the CK04 curve is slower.

An isolated matched reflection comparison copies the same initial container and
returns both final host temperature arrays. Original CPU costs **0.88–1.27 ms**,
while device preparation/geometry/iteration/download costs **2.05–3.02 ms**.
Matrices download only in separate diagnostics. Device reflection alone is
**slower at these540-face workloads**. A resident kernel lower bound is not
substituted for this complete comparison.

| Physical fitting work | Original CPU | CPU-reflection hybrid GPU | New hybrid JAX CPU | New hybrid GPU |
|---|---:|---:|---:|---:|
| Blackbody likelihood/prior median (ms) | 55.55 | 24.64 | 105.22 | 25.31 |
| Table likelihood/prior median (ms) | 27.20 | 23.97 | 61.17 | 24.64 |
| Blackbody new-fitter public LSQ (s) | 1.170 | — | 3.949 | 2.320 |
| Table new-fitter public LSQ (s) | 0.344 | — | 2.348 | 2.050 |
| Blackbody new-fitter 8-walker/3-step run (s) | 1.883 | — | 4.775 | 2.940 |
| Table new-fitter 8-walker/3-step run (s) | 0.912 | — | 3.529 | 2.476 |

New hybrid setup costs **9.5–9.9 ms**, first likelihood callbacks
**1.496–1.561 s**, separately from warm costs. Likelihood timers include physical
transforms/model construction/serialization, original union-phase mapping,
mean normalization, priors, third-light/noise semantics and all transfers. Public
fit/run totals also include new object/original setup/constraints, compilation
incurred, callbacks and result serialization; runtime/source caches may be warm
from preceding gates. Full fits/runs remain slower than original CPU. These
measurements support no new reflection, table-fitting or ESS claim. CPU development
on cores16–19/Nice19 while production ran is excluded from matched timings.

### Domain, reproduction, restoration and next work

The earlier q/spin/radius/thermal/atmosphere/occultation gates remain. Reflection
adds bounds of8192 faces per component,2097152 combined prepared face pairs and32
iterations, in the symmetric circular detached separation1 domain. These are
resource/contract limits, not physical prior cuts. Unsupported states run the
entire original CPU curve, or the explicit broader bridge if requested;
`fallback="raise"` raises after eligible hybrid paths are exhausted. Disabled
reflection/nonpositive iterations retain the original no-op behavior. No global
CPU function, production default or caller's physical model is patched.

```text
python -m pytest tests/tests_experimental -q -rs
python scripts/validate_gpu_reflection.py --device cpu --output reflection-cpu.json
python scripts/validate_gpu_reflection.py --device gpu --output reflection-gpu.json
```

Use the private CPU wrapper during development. GPU commands require the
independent checkpoint/restore systemd allocation. Registered windows010/011/012
used **103.59/145.06/92.32 s** of test commands within270 s internal/300 s systemd
bounds. Each independently restored the exact pinned service/source/config/plan,
verified matching checkpoint resumes and latest receipt/witness/endpoints, and
observed **43,008/46,592/45,056 durable cells** of resumed advancement. All three
test units finished inactive/success. This is not a full historical production
re-audit; raw operational logs/process/environment snapshots remain private.

See [curated evidence](ada_validation/background-m007.json),
[583-file runtime/data manifest](ada_validation/background-m007-source-hashes.json)
and [current status](ada_validation/progress.json). Compact sorted-key manifest
aggregate: `2a0e3b807974cc2f34521bc189d8cb722d6e3c2993a8d23bc9709a318f135c04`.
All tested runtime/data files were unchanged after the hardware windows.

Next assess mesh/phase/spectral discretization error separately from implementation
parity for complete reflected curves and physical fits at `1e-4` uncertainty.
CPU mesh, LD and hull residency work remains; broader device physics and any
posterior/convergence claims require their own evidence. The assignment continues.

## Previous milestone: broader physical curves and morphology-specific fitting

Explicit `general="device_integration"` now extends the experimental hybrid to
fresh **spotted, eccentric, near-contact and overcontact** curves. The original
CPU routers build/evolve the mesh and spots, change separation, apply reflection,
prepare radiance/LD, orient surfaces, clip eclipses and apply configured curve
approximations. Each router callback uploads fresh selected faces for **one
all-band/component device flux integration**. Original Observer phase mapping,
third light, distance and normalization remain intact. The result reports
`backend="hybrid_native"`; this broader branch moves integration only. The earlier
circular detached device-state/radiance/coverage path remains available.

Default `general="cpu"` preserves complete CPU fallback outside the earlier fast
path. No CPU physical implementation, default, global function or shared model
is patched. A private per-call Observer delegate uses existing router callbacks
and is restored even on errors. Resource failures after partial callbacks discard
all partial results and run the **entire original CPU curve**. Missing GPU/FP64
and nonserial opt-in configuration raise outside emergency candidate penalties.

`HybridOvercontactLeastSquaresFit` and `HybridOvercontactMCMCFit` inherit the
original overcontact fitter lifecycle, morphology checks and equal-potential
constraints. Existing detached classes accept the same opt-in. Physical
temperature, inclination, spot longitude or eccentricity/potential and l3/noise
proposals rebuild dependent physics; no stale physical packet is reused.

### Coverage, parity and numerical resolution

**269 hardware tests passed in 221.05 s, zero failures/errors/skips.** Full CPU
checks passed **152**, with **117** explicit GPU skips, in 155.14 s. Two/one
warnings respectively are inherited original CPU Planck UV-tail overflow in a
finite cold-star fallback fixture. Existing acceptance gates are unchanged.

The eight detailed V/bolometric cases cover synchronous spots with reflection,
asynchronous spots (spin1.1), eccentricity0.2 with reflection, its configured
similar-neighbour approximation, eccentricity0.15 with spots, near-contact
q1/potentials3.9, overcontact q1/potentials3.5, and spotted CK04 with real table LD
and reflection. Meshes have 456–556 faces per star; actual callback separations
range 0.8025–1.1917 SMA. Eighteen requested phases include both original CPU
conjunctions with nearby samples, quadratures, ordering/repeats, cross-cycle phases
and nonzero shift/third light. These new samples are **not an independent exact
contact calculation**; earlier fast-path contact/clipping tests remain in the
full suite. Added tests change actual spot location/temperature and exercise
logarithmic/square-root LD, immutable packets and strict device transfer guards.
The spotted CK04 fixture uses the existing unmodified spectra and LD data from
the preceding milestone; there are no downloaded or synthetic replacements.

For these eight **linear-LD** detailed fixtures, actual device face fluxes,
component sums, raw/normalized observed curves and full-CPU fallback comparisons
all have **zero measured difference**. Three physical/nuisance proposal vectors
per spotted/eccentric/overcontact family give zero prediction and accumulated
posterior difference at normalized observational uncertainty `1e-4` (11 data).
This is finite tested equivalence, not a universal bitwise guarantee.

The overcontact numerical fit exposed a sensitive rounding issue during CPU
development: an ordinary device reduction gave forward differences around
`2e-16`, but changed minimum chi² by **0.00016697**, failing the existing `1e-5`
fit gate. Larger iteration budgets and central differences still left a
**0.00005516** difference. The new `gpu_ordered` kernel retains the original FP64
expression boundaries and **actual selected-length NumPy pairwise sum order**,
rather than reducing padded faces in a different order. Independent controls
compare actual installed NumPy sums at empty/8/128/split/tail/8192 boundaries,
including signed cancellation, with exact equality on CPU and GPU. See the
[primary NumPy implementation](https://github.com/numpy/numpy/blob/v2.5.5/numpy/_core/src/umath/loops_utils.h.src).
The CPU oracle, earlier fast kernel and tolerances were unchanged; failed
development evidence remains private. Current NumPy2.5.5 is the tested layout;
other versions/transcendental implementations need fresh validation.

Using the original **16-evaluation, two-point, diff_step1e-4** fit options:

- Spotted LSQ: `T_primary=6499.062955947 K`, `inclination=86.3973696824 deg`,
  chi² **7.0727437603**; all CPU/GPU fit differences are zero.
- Overcontact LSQ: `T_primary=6499.741941269 K`, `inclination=86.3991914604 deg`,
  shared potential **3.499978425604**, chi² **8.5173040391**; differences are zero.
- Public 8-walker/3-step runs for both families replay identical coordinates,
  densities and acceptance fractions. These are callback/move smoke tests,
  **not convergence, posterior accuracy or ESS/sec**. Eccentric fitting coverage
  here is changed-parameter prediction/posterior callbacks, not a new full public
  optimizer/sampler run.

Native approximation accuracy is separate. With `MAX_D_FLUX=0.01`, configured
similar-neighbour curves differ from rigorous original CPU by **0.00137973 V /
0.00129814 bolometric** in maximum normalized flux, equally on both paths. This
configuration is unsuitable for the `1e-4` fitting noise without an accuracy
assessment; fitting uses rigorous flags. An inherited original configured-
approximation error on a sparse six-phase input raises the same zero-size
reduction `ValueError` on both paths and is retained as an error-equivalence
test. A sufficiently sampled grid validates the actual approximation path.
No CPU error, discretization error or gate was hidden by fallback or relaxation.

### Matched complete costs: integration alone does not accelerate these cases

Median **ms per fresh host V/bolometric curve**, three warm samples. All original
CPU physical stages, dispatch/failed fast-path attempts, uploads, device work and
final host curves are included. Binary construction is outside every curve
timer. CPU/JAX-CPU and GPU drivers ran sequentially in the same checkpointed
window, affinity0–15/Nice 10, threads 1, FP64. Fixed order and few samples limit
performance inference. Original CPU and fallback columns below are from the GPU
driver; the separate CPU-driver baselines and all raw samples are in curated JSON.

| Case | Original CPU | Default CPU fallback | Hybrid JAX CPU | Hybrid GPU |
|---|---:|---:|---:|---:|
| Synchronous spot, reflection | 67.94 | 67.63 | 75.71 | 111.36 |
| Asynchronous spot | 435.69 | 437.92 | 460.05 | 492.88 |
| Eccentric exact, reflection | 464.13 | 464.50 | 479.23 | 511.07 |
| Eccentric configured neighbours | 106.09 | 106.92 | 115.29 | 152.43 |
| Eccentric spot | 818.42 | 827.98 | 851.03 | 885.16 |
| Near contact, reflection | 33.86 | 40.32 | 42.02 | 62.52 |
| Overcontact, reflection | 31.83 | 32.22 | 36.38 | 47.39 |
| Spotted CK04, table LD/reflection | 27.75 | 28.02 | 34.49 | 70.64 |

Combined fast/native engine construction costs **0.635 s**. First diagnostic GPU
calls, including newly incurred compilation/preparation and CPU work, cost
**0.093–0.893 s**, separately reported from warm calls. Diagnostic downloads are
off for timing/inference; only component totals return to host. Ordered reduction
and per-position transfers cost more than the tiny original CPU sum. Default
near-contact fallback also includes a failed bounded fast-path attempt.

| Physical fitting work | Original CPU | Hybrid JAX CPU | Hybrid GPU |
|---|---:|---:|---:|
| Spotted likelihood/prior median (ms) | 300.47 | 317.24 | 327.05 |
| Eccentric likelihood/prior median (ms) | 165.84 | 171.06 | 182.78 |
| Overcontact likelihood/prior median (ms) | 30.24 | 33.08 | 39.36 |
| Spotted new-fitter public LSQ, all costs (s) | 6.479 | 7.438 | 7.521 |
| Overcontact new-fitter public LSQ, all costs (s) | 1.320 | 1.660 | 1.986 |
| Spotted new-fitter 8-walker/3-step run (s) | 9.973 | 11.204 | 11.650 |
| Overcontact new-fitter 8-walker/3-step run (s) | 1.202 | 1.512 | 1.777 |

Likelihood costs include physical transforms, model construction/serialization,
original mapping/normalization/statistics and all transfers. Fit/run totals also
include new object construction, original setup/constraints, compilation incurred,
all callbacks and result serialization; runtime caches can be warm from earlier
gates. **No speedup is demonstrated for this broader bridge.** CPU development
checks on cores16–19/Nice19 while production ran are correctness evidence only
and excluded from these matched measurements.

### Domain, reproduction, restoration and next work

The broader opt-in requires an original nonpulsating `BinarySystem`, serial
physics (`NUMBER_OF_PROCESSES=1`), 1–8 passbands, at most8192 faces per component
and256 router callbacks per curve. Selected faces use power-of-two padding.
Pulsations/custom subclasses/unvalidated routes/resource failures retain full
CPU fallback (or explicit `fallback="raise"` after both hybrid paths are
ineligible). Available original CPU data/physics limits still apply; these eight
fixtures do not validate every CPU model. Final `component_fluxes` is empty on
the native bridge because original approximations can interpolate total curves;
`diagnostics=True` exposes actual per-callback component fluxes instead.

```text
python -m pytest tests/tests_experimental -q -rs
python scripts/validate_gpu_native.py --device cpu --output native-cpu.json
python scripts/validate_gpu_native.py --device gpu --output native-gpu.json
```

Use the private CPU wrapper for development; GPU commands require the independent
checkpoint/restore systemd allocation. Registered windows008/009 used
**224.14/175.07 s** of test commands within270 s internal/300 s systemd bounds.
Both independently restored the exact pinned service/source/config/plan,
matched checkpoint resume records, verified latest receipt/witness/endpoints and
observed **45,056/44,288 durable cells** of resumed advancement. Both test units
finished inactive/success. Subsequent read-only checks verified further durable
advancement. This is not a full historical production re-audit. Raw operational
logs and process/environment snapshots remain private.

See [curated evidence](ada_validation/background-m006.json),
[580-file runtime/data manifest](ada_validation/background-m006-source-hashes.json)
and [current status](ada_validation/progress.json). Compact sorted-key manifest
aggregate: `f3ec3634baa678172785b2f37e7070976692031503d09123af2fcaed68da6638`.

Next port mutual irradiation/reflection temperature iteration onto device for
the validated circular detached branch, preserving original build order,
bolometric LD and iteration semantics, then revalidate complete curves/fitter
costs. Candidate mesh/topology, LD/hull and broader spot/eccentric/contact physics
remain CPU bridges. Parity does not establish mesh/time/spectral convergence or
posterior accuracy, and the assignment remains active.

## Previous milestone: actual CK04 spectra and interpolated table limb darkening

The complete hybrid now supports **two CK04 components** and actual CPU-
interpolated limb darkening, alongside two blackbody components with custom or
table LD. `gpu_atmosphere.DeviceAtmosphereRadiance` uses unmodified ELISa for
source selection/revision checks, wavelength alignment/stripping and actual
passband response. Device work integrates unique prepared spectra with the CPU's
Simpson functional, gathers each face's brackets, interpolates in **T⁴**, mirrors
faces, divides by the same-band LD integral and feeds eclipse packets directly.
Fresh corrected device temperatures remain inputs without reflection; final CPU
post-reflection temperatures upload otherwise. No physical brightness/LD/coverage
is retained from an earlier proposal. Existing CPU defaults remain unchanged.

This preserves original **nearest gravity and metallicity** atmosphere selection,
not a new continuous interpolation in those coordinates. Table LD retains the
original piecewise-linear T/log(g) interpolation and ordered source selection.
Source/response/file changes rebuild fresh prepared spectra and coefficients;
only fixed wavelength quadrature and revision-keyed LD hulls are reused. CPU mesh,
kinematics, reflection, source/wavelength preparation, LD interpolation and hull
visibility remain measured bridges. This is still a **hybrid** model.

### Real fixture coverage and numerical evidence

**239 hardware tests passed in 169.90 s; zero failures/errors/skips.** Full CPU
checks passed **137**, with **102** explicit hardware skips in 127.27 s. Two/one
warnings respectively come from original CPU Planck UV-tail exponential overflow
in a cold-star fallback test; final curves remain finite. No tolerance changed.

The data are existing, unmodified [CK04 spectra](../tests/data/ck04) and
[Van Hamme coefficient tables](../tests/data/light_curves/limbdarkening), with
17 data/response/grid hashes in the report. No synthetic atmosphere or downloaded
replacement was substituted. The six detailed cases use V and bolometric bands,
540 faces per star, solar metallicity and three sparse available CK04 regions:
5250/5500 K at log(g)=3, 10000/10250 K at 4, 25000/26000 K at 4.5 (cgs). They cover
changed mass ratio, temperature, potential and inclination; non-eclipsing,
partial/total and both eclipses/contacts; repeats/phase shift/third light;
reflection off/on; and blackbody with actual interpolated LD. Additional real-
table tests exercise logarithmic V and square-root bolometric coefficients.
These are model-parity fixtures, not an evolutionary sample or a validation of
all atmosphere data, mesh/spectral resolution or astrophysical inference.

- Normal intensity maximum relative difference: **2.19e-15**; table LD: **5.49e-16**.
- Raw total-curve relative difference: **8.63e-16**; normalized absolute difference:
  **3.34e-16**. Actual device face contributions, scaled by component flux, differ
  by at most **3.87e-14**; component-flux relative difference is **3.69e-15**.
- Actual original fitter mean-normalized predictions differ by at most
  **4.45e-16**; accumulated posterior difference is **1.63e-9** across changed
  physical/nuisance proposals at normalized uncertainty `1e-4` (17 observations).
- Public bounded LSQ fits recover `T_primary=10078.571434 K` and
  `inclination=86.3998042190 deg`, with CPU/GPU differences **4.70e-10 K**,
  **4.27e-14 deg**, and **4.17e-12** in minimum chi² (about 9.4225894123).
- Public 8-walker/3-step MCMC calls replay identical coordinates/acceptance
  fractions; stored density difference is at most **3.96e-9**. This is a callback/
  move smoke test, **not convergence, posterior precision or ESS/sec**.

Tests replace actual copied spectra, LD tables and response functions without
manually clearing source caches, then compare with original CPU again. Prepared
packets remain immutable and pass strict no-transfer evaluation. Resource caps
retain a finite, exactly matching original CPU posterior, including emergency
penalty mode. Missing atmosphere/LD sources retain their original error classes.

Near atlas temperature/nearest-gravity selection boundaries or an LD table's
convex-hull boundary, state roundoff could change a source/domain decision. These
cases run the **whole original CPU curve**, as verified at legal boundary inputs.
The dispatch margin is four times the existing 3e-13 state relative gate; it is
neither a relaxed acceptance tolerance nor a physical-prior restriction. Original
errors still propagate on genuinely invalid sources/candidates.

### Matched full costs: no general table speedup

Median **ms per fresh host V/bolometric curve**, three warm repeats. Original CPU,
CPU-radiance hybrid and device-radiance GPU paths run in one driver; JAX CPU runs
separately in the same checkpointed window. Affinity0–15/Nice10, thread counts1,
FP64, warm source/interpolator/quadrature caches; fixed order, few samples. All
CPU bridges, response/source checks, transfers and matching final host outputs
are included. Binary construction is outside all curve timers. Combined engine
construction was 0.648 s; first diagnostic GPU calls, including new shape
compilation, ranged 0.044–1.844 s. Raw/stage/cold samples remain in curated JSON.

| Case | Original CPU | CPU-radiance hybrid GPU | Hybrid JAX CPU | Hybrid GPU |
|---|---:|---:|---:|---:|
| CK04 partial | 24.89 | 29.32 | 189.95 | 30.16 |
| CK04 total, reflection | 32.69 | 32.82 | 67.59 | 34.20 |
| CK04 non-eclipsing, changed q | 18.42 | 20.12 | 16.23 | 21.59 |
| CK04 cool giant, reflection | 27.51 | 28.81 | 72.69 | 30.74 |
| CK04 hot compact | 24.82 | 24.71 | 81.15 | 26.28 |
| Blackbody, table LD/reflection | 47.24 | 48.01 | 64.19 | 27.35 |

Only **two unique CK04 spectra per band/component** are selected in these
fixtures (152 V or 1221 bolometric wavelength samples). Original CPU already
integrates unique spectra before interpolation, so little spectral work remains
to amortize GPU preparation/launch/transfers. Device table preparation/emission
costs about 5.7–6.2 ms warm. The CPU-radiance column is an ablation of the newly
supported table hybrid, not a preceding GPU table implementation. The blackbody
case benefits from device continuous Planck emission; it must not be presented
as evidence of atlas acceleration.

| Physical fitting work | Original CPU | CPU-radiance hybrid GPU | Hybrid JAX CPU | Hybrid GPU |
|---|---:|---:|---:|---:|
| Median changed-parameter likelihood/prior (ms) | 26.22 | 23.67 | 63.11 | 24.97 |
| First callback, compilation included (ms) | 17.72 | 1101.44 | 1032.36 | 1318.98 |
| Object/observation setup (ms) | 2.00 | 1.68 | 8.30 | 8.15 |
| Bounded new-fitter public LSQ, all costs (s) | 0.333 | — | — | 1.687 |
| New-fitter 8-walker/3-step MCMC, all costs (s) | 0.706 | — | — | 2.110 |

The original CPU callback samples span 16.69–27.25 ms; these few measurements
support **no robust table likelihood speedup**. Fit/run costs include candidate
construction, original setup/constraints/mapping/statistics, compilation incurred,
all callbacks and result serialization; runtime/source caches may be warm from
correctness gates. Short full fits/runs are substantially slower on GPU. No
production default or ESS claim follows. Supplementary CPU development checks
used cores16–19/Nice19 while production ran and are excluded from this comparison.

### Domain, reproduction, restoration and next work

Existing circular detached q/spin/radius/beta and geometry/occultation gates
remain. Device atlas emission admits at most 8 bands, 8192 faces per star, 128 unique
spectra per band/component, 2–4096 wavelengths and 8388608 prepared spectral samples
(64 MiB of spectra, excluding runtime/compiler workspace). Unsupported resource/
numeric states, mixed/other atlases, spots and eccentric/contact GPU states use
**entire CPU curves**, with explicit reasons and original error semantics.

```text
python -m pytest tests/tests_experimental -q -rs
python scripts/validate_gpu_atmosphere.py --device cpu --output tables-cpu.json
python scripts/validate_gpu_atmosphere.py --device gpu --output tables-gpu.json
```

CPU development requires the private CPU wrapper. GPU commands require the
independent checkpoint/restore systemd allocation. Registered windows 006/007 used
**172.84/44.02 s** of test commands within 270 s internal/300 s systemd bounds.
Both restored the exact pinned service/source/config/plan, matched resume records,
verified latest receipt/witness endpoints, and observed **8,192/45,056 durable
cells** of advancement. Both test units finished inactive/success. This is not a
full historical production re-audit; raw operational logs remain private.

Next validate actual spot, eccentric/separation and near-contact curves and
physical fitting through the explicit full-CPU fallback, then extend supported
hybrid physics where justified. CPU mesh/reflection/LD/hull residency work remains;
this milestone does not silently drop that scope or complete the assignment.
See [curated evidence](ada_validation/background-m005.json),
[576-file runtime/data manifest](ada_validation/background-m005-source-hashes.json)
and [current status](ada_validation/progress.json). Manifest aggregate using
compact sorted-key JSON:
`7a9fbab7e03a5e82eaf87e562cbfeb332c8341da90f3c2eb2f9ed2157d6ca312`.

## Previous milestone: fresh device emission in complete curves and fitting

`gpu_radiance.DeviceBlackbodyRadiance` now composes the validated continuous
Planck/passband kernel with CPU-compatible symmetry and the **same band's**
hemispheric limb-darkening integral. Its normal intensity feeds the device
eclipse packets directly. The default experimental hybrid no longer downloads
normal radiance and uploads it again; diagnostics and `coverage="cpu"` explicitly
materialize it when requested. `radiance="cpu"` retains the preceding spectral
path for matched comparisons. Existing CPU Observer/fitter defaults are unchanged.

The CPU-compatible corrected state/build order is retained. Without reflection,
fresh corrected device temperatures feed emission directly. CPU reflection still
uses downloaded state; its **final changed temperatures** upload for emission.
The earlier intrinsic device temperatures cannot be reused after reflection.
CPU mesh/roots/topology/kinematics, reflection, custom-LD preparation and fresh
Qhull/Path visibility remain measured bridges. This is a complete **hybrid**
light-curve/fitting path, not an all-GPU or resident forward model.

Only fixed quadrature/code is reused. Every call re-samples the actual Observer
response on the exact CPU wavelength grids and checks its revision; this CPU
cost is included. A changed response callable invalidates reuse even with the
same labels/table. A single immutable response revision is retained per engine;
physical temperatures, LD, brightness and eclipse coverage are fresh. Planck
samples use the existing Simpson/end-interval, Akima clipping, pi and Angstrom
conventions; no temperature rounding or new spectral approximation was introduced.

### Parity and independent physical checks

**215 tests passed in 133.04 s on Ada, zero failures/errors/skips.** Full CPU
checks passed **125**, with **90** explicit hardware skips. The eight detailed
whole-curve cases passed on both JAX CPU and GPU under the same allocated runtime
profile. They cover non-eclipsing/grazing/partial/total eclipses, both eclipses,
contact-focused phases, repeats/reordering/phase shift/third light, unequal and
equal components, q/spin limits, changed potentials/temperatures/topology, a tiny
occulter, three LD laws, and reflection off/on. The original physical fitting
assessment also passed with the new emission path.

- Same-band normal intensity maximum relative difference: **3.41e-15**.
- Raw total-curve relative difference: **1.27e-15**; Observer-normalized absolute
  difference: **4.45e-16**; component-flux relative difference: **1.35e-13**.
  Tiny eclipsed components amplify relative geometric roundoff, as before.
- Full-curve diagnostic accumulated log-likelihood difference: **2.17e-11** at
  normalized errors `1e-4`. Original fitter mean-normalized predictions differ
  by at most **6.67e-16**; likelihood/posterior differences remain **1.22e-8**
  across direct/JD/manual/uniform/mass-mode/changed-parameter and prior cases.
- Independent 256-node angular quadrature recovers emitted hemispheric flux
  from normal intensity for linear/cosine, logarithmic and square-root LD;
  the largest emission/energy/snapshot comparison relative error is **3.38e-14**.
  This checks that optical-depth division and directional LD/pi are applied once.
- Prepared temperature/LD inputs remain immutable, strict no-transfer emission
  and eclipse packet evaluation passes, and eclipse packets retain the actual
  device radiance arrays. Changed real passband responses/coefficients/temperatures
  and reflection dispatch are tested. Invalid optical-depth rows are explicitly
  masked; a legal two-wavelength CPU quadrature case uses the exact full CPU curve.

The actual bounded public LSQ fits recover `T_primary=6504.130004 K` and
`inclination=86.4018902013 deg`; CPU/GPU differences are **9.94e-10 K**,
**6.12e-13 deg** and **1.74e-11 in minimum chi²**. Eight walkers/three public MCMC
steps replay identical coordinates/acceptance fractions, with stored density
difference **2.65e-9**. **This is not convergence, posterior precision or ESS.**
All preceding surface/geometry/curve/likelihood/optimizer tolerances remain
unchanged. The earlier deliberately coarse manual/uniform support errors remain
original CPU interpolation error, separate from implementation parity. Mesh,
spectral/time resolution and physical inference accuracy are not certified here.

### Matched transfer-inclusive costs

Median **ms per fresh curve**, three warm repeats. All final host B/V/bolometric
curves, fresh CPU/device physics, response revision checks and transfers are
included. Original CPU, preceding CPU-radiance hybrid GPU and new hybrid GPU run
in the same driver; JAX CPU runs separately in the same checkpointed window.
Affinity0–15, Nice10, thread counts1, FP64; no concurrent production GPU work.
Binary construction is outside all these curve timers. Combined construction of
both comparison engines was 0.637 s; first diagnostic curve calls including shape
compilation ranged 0.187–2.096 s. JSON retains each cold/stage/warm sample.

| Case | Original CPU | Previous hybrid GPU | New hybrid JAX CPU | New hybrid GPU |
|---|---:|---:|---:|---:|
| Non-eclipsing | 66.55 | 65.16 | 21.14 | 20.14 |
| Grazing | 73.07 | 71.03 | 88.37 | 25.99 |
| Partial, reflection | 75.48 | 74.81 | 196.22 | 34.32 |
| Total, reflection | 79.20 | 77.04 | 238.43 | 36.68 |
| q=0.5, logarithmic | 76.94 | 71.63 | 241.76 | 32.48 |
| q=2, square root | 99.20 | 96.49 | 202.40 | 31.19 |
| Equal components | 66.65 | 67.46 | 193.24 | 28.40 |
| Small occulter | 71.49 | 67.33 | 44.47 | 28.85 |

These fixtures have 832 or 1280 faces per star. Device spectral preparation,
revision checks, LD and emission take about **3.2–3.6 ms** warm. In the changed-
parameter callback, the preceding CPU spectral stage takes 31.8 ms and the new
stage 3.27 ms (individual last-call samples, not a subtracted median decomposition).
The faster non-eclipsing JAX CPU result also shows why batching/quadrature reuse
and device acceleration must be distinguished. These small workloads and three
samples do not establish universal throughput or production suitability.

| Physical fitting work | Original CPU | Previous hybrid GPU | New hybrid JAX CPU | New hybrid GPU |
|---|---:|---:|---:|---:|
| Median changed-parameter likelihood/prior (ms) | 47.69 | 52.43 | 103.92 | 23.20 |
| First callback, compilation included (ms) | 47.80 | 1086.61 | 1207.92 | 1207.57 |
| Object/observation setup (ms) | 2.82 | 2.80 | 9.17 | 8.89 |
| Bounded new-fitter public LSQ, all costs (s) | 1.192 | — | — | 1.767 |
| New-fitter 8-walker/3-step MCMC, all costs (s) | 1.956 | — | — | 2.535 |

Fitting includes parameter transforms/constraints, actual candidate construction,
all transfers/physics, original observation mapping/mean normalization and
statistical costs. Public fit/run costs include construction, original setup,
compilation incurred, all callbacks and result calculations/serialization. The
runtime may already be warm from correctness gates. Despite faster warm callbacks,
these short full fits/runs remain slower than original CPU. No MCMC convergence,
ESS/sec or production default change follows. Supplementary CPU development
checks ran on cores16–19/Nice19 while production ran; overlapping CPU checks are
correctness evidence and are excluded from these matched timing comparisons.

### Domain, reproduction and restoration

The existing circular detached unspotted/nonpulsating blackbody/custom-LD domain
and geometry/occultation bounds remain. Device emission accepts 1–8 bands,
3–4096 wavelengths per band, at most 16384 combined samples and 8192 faces per star.
Unsupported model/backend/resource/numeric states run the **entire original CPU
curve**, preserving posterior support and original CPU errors. Missing device or
FP64 remains an infrastructure error. Real atmosphere/LD tables, spots and
broader eccentric/contact GPU physics still require further scientific work.

```text
python -m pytest tests/tests_experimental -q -rs
python scripts/validate_gpu_binary.py --device cpu --repeats 3 --output curves-cpu.json
python scripts/validate_gpu_binary.py --device gpu --repeats 3 --output curves-gpu.json
python scripts/validate_gpu_fit.py --device gpu --output fit-gpu.json
```

Use the private CPU wrapper for development and independently restored allocation
for every GPU command. Registered windows 004/005 used **135.97/78.12 s** of
probe/test/assessment commands, each within 270 s internal/300 s systemd limits.
Both restored the exact pinned service/source/config/plan, matched resume records
and verified latest receipt/witness endpoints, with **45,056/47,104 durable cells**
of advancement. Both units finished inactive/success. This is not a full
historical production re-audit; operational logs remain private.

Next validate real atmosphere/LD-table scientific fixtures and extend the fresh
radiance bridge while preserving original interpolation and error semantics.
Broader irradiation/spot/eccentric/contact coverage and resident mesh/hull work
remain explicit scope; this milestone does not mark the assignment complete.
Curated [measurements](ada_validation/background-m004.json),
[556-file tested manifest](ada_validation/background-m004-source-hashes.json) and
[current status](ada_validation/progress.json) include unchanged CPU oracle hashes.
Manifest aggregate (compact sorted-key JSON):
`7988bbd7b89b30eda385eba2558f26bb3174ccdfc9b1782f8c140bb8ddaba70b`.

## Previous milestone: original physical fitter and sampler assessment

`gpu_fit.HybridBinarySynthesis`, `HybridDetachedLeastSquaresFit` and
`HybridDetachedMCMCFit` now provide an **opt-in serial hybrid synthesis hook** for
ELISa's original fitting classes. No CPU fitter/default/backend registration was
changed. Actual `prepare_binary` serialization and constraints build a fresh
physical binary for every proposal. The original prediction layer still selects
support, resolves JD/phase coordinates, maps/interpolates each band's observations
and mean normalizes **after** mapping/repeats. The original residuals, Gaussian
noise inflation, priors, candidate failure policy and public `fit()` lifecycle
remain in use. The callback also retains the prepared Observer's response tables,
bandwidths and resulting raw phase/flux state.

Unsupported GPU model/backend/resource states run the whole original CPU curve,
with an explicit reason. A q=0.4 regression verifies a finite, exactly matching
CPU posterior even with emergency exception penalties enabled; device bounds do
not silently truncate the posterior. Disabled FP64/device unavailability remains
an infrastructure error. Experimental GPU fitters permit serial operation only;
CPU multiprocessing defaults remain untouched. Compiled code/shapes are reused,
but no physical curve is cached across changed-parameter proposals. The existing
l3/ln_f-only pilot still caches its fixed curve.

### Scientific and fitting evidence

**203 tests passed in 119.53 s on Ada; zero failures/errors/skips.** Full CPU
checks passed **119**, with **84** hardware skips. The separate GPU assessment
passed all five direct/JD/manual/uniform/standard-mass scenarios against original
ELISa. It varies inclination, temperature, mass ratio or mass, surface potential,
third light and noise, retaining the secondary-temperature constraint. Tests also
cover reflection-enabled callbacks, Gaussian priors, prior exclusions without
synthesis, real changed Observer responses and runtime/configuration errors.

The three bands have different, unsorted/repeated/cross-cycle observation grids
(10 B, 13 V, 5 bolometric samples), with nonzero phase shift. Errors are `1e-4` in
normalized flux units. Truth is synthesized through the **original fitter's
union-phase prediction path**, so third light uses the correct synthesis-support
mean before per-band mapping. Initial models use 12-degree meshes (540 faces per
star at the baseline); these are model-parity fixtures, not resolution-converged
astrophysical inference.

- Maximum mean-normalized prediction absolute difference: **4.44e-16**.
- Maximum accumulated log-likelihood/posterior absolute difference: **1.22e-8**,
  including the deliberately coarse interpolation and far-proposal cases.
- The actual public two-parameter least-squares `fit()` calls recover
  `T_primary=6504.130004 K`, `inclination=86.4018902013 deg`; CPU/GPU differences
  are **7.49e-10 K** and **5.55e-13 deg**, respectively. Minimum chi² differs by
  **1.29e-11** (both about 21.9555423754).
- The actual public MCMC `fit()` calls, with the same starting ensemble/random
  stream, produce **identical chain coordinates and acceptance fractions** over
  8 walkers/3 steps. Stored log-density difference is at most **2.39e-9**.
  **This is callback/move replay, not convergence, posterior precision or ESS.**

Prediction gates remain `3e-13` relative/`2e-12` absolute, and accumulated-density
error remains below the existing `1e-6` gate. New standardized-residual tolerance
`2e-8` follows from the prediction gate divided by the nominal `1e-4` error.
Predeclared optimizer-agreement gates are 0.01 K/1e-4 deg (small fractions of the
parameter intervals), and `1e-5` in minimum chi² (negligible against unit changes).
They are not posterior uncertainty estimates. No previous tolerance was relaxed.
The bounded optimizer uses SciPy `max_nfev=16` (finite difference calls may be
additional) and relative finite difference step `1e-4`; this comparison does not
certify arbitrary optimizer termination or physical-model accuracy.

### Numerical parity versus interpolation accuracy

The manual support has the same number of coordinates as the observed union,
but **different** coordinates; the original code correctly interpolates it.
The intentionally coarse threshold-12 uniform/manual supports differ from direct
**CPU** synthesis by up to **0.0826/0.1006 normalized flux**, respectively. That
is original time/interpolation discretization error, much larger than the tested
observational uncertainty, and is not hidden by GPU parity. Those grids exercise
mapping branches; they are not recommended scientific fitting support. The
representative optimizer and sampler runs use direct synthesis. Mesh, spectral,
interpolation and convergence errors remain distinct from implementation error.

### Matched fitting costs: keep this experimental

All numbers below include parameter transforms/constraints, real candidate
construction, fresh physical state, transfers, final observation mapping and
existing statistical costs. Same checkpointed GPU window, affinity0–15, thread
counts1, serial MCMC. Model/input setup and the first callback are shown
separately; warm samples vary physical/nuisance parameters. Fit/run times include
new fitter construction, original public setup, compilation incurred, all model
calls, result calculations and serialization. The GPU runtime may already be
warm from preceding correctness gates. There are only three warm callback
samples and one bounded fit/run per path, so no broad throughput claim follows.

| Work | Original CPU | Hybrid JAX CPU | Hybrid GPU |
|---|---:|---:|---:|
| Median changed-parameter likelihood/prior callback (ms) | 50.45 | 133.45 | 55.20 |
| First callback, including shape compilation (ms) | 47.98 | 998.15 | 1048.26 |
| Object/observation setup (ms) | 2.73 | 9.51 | 3.03 |
| Bounded public LSQ fit, including construction (s) | 1.190 | — | 2.316 |
| Public 8-walker/3-step MCMC smoke, including construction (s) | 1.977 | — | 3.324 |

The hybrid GPU is slightly slower for warm callbacks here; compilation makes
these short full fits/runs slower still. The current CPU geometry, reflection,
radiance/LD and hull bridges remain substantial. **No production default, MCMC
speedup or ESS/sec claim is justified.** CPU-only development timings used
cores16–19/nice19 while production ran and are supplementary correctness/cost
evidence, not the matched timing comparison above.

### Reproduction, remaining scope and restoration

Use the isolated FP64 environment and the existing CPU wrapper for development:

```text
python -m pytest tests/tests_experimental -q -rs
python scripts/validate_gpu_fit.py --device cpu --output physical-fit-cpu.json
python scripts/validate_gpu_fit.py --device gpu --output physical-fit-gpu.json
```

GPU commands require the authorized independently restored systemd allocation;
the fitting classes themselves do not manage production. The full window's
probe/suite/assessment commands took **149.98 s**, within the 270 s internal/
300 s systemd limits. Cleanup restarted the exact pinned service, verified a
matching resume record and latest receipt/witness endpoints, and observed
**43,776 durable cells** of advancement. Test unit cleanup finished successfully.
This is not a full historical production re-audit. Operational logs remain private.

The complete light-curve path and this assessment are still **hybrid**, within
the previously declared circular detached blackbody/custom-LD device domain;
other states retain explicit full CPU fallback. Next compose the already tested
fresh blackbody/optical-depth emission with corrected temperatures and device
eclipse integration to reduce the measured CPU spectral/transfer cost. Real
atmosphere/LD-table scientific fixtures, irradiation device physics, spots,
broader eccentric/contact coverage and resident mesh/hull work remain; this
milestone does not silently reduce that assignment scope or mark it complete.

Curated evidence: [fitting assessment](ada_validation/background-m003.json),
[tested runtime/fixture manifest](ada_validation/background-m003-source-hashes.json),
and [current status](ada_validation/progress.json). The 554-file manifest hashes
to `a92edd45c17a248dee371b7331ea4c80a99ece9b8d16b540fa6943b4e1000875`
using compact sorted-key JSON. Runtime files were verified unchanged after testing.

## Previous milestone: device projection and partial eclipse clipping

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

Background handoff prepared **2026-11-17T03:41:23.890646+00:00**. Paul has assigned completion and scientific validation of the light-curve pipeline, followed by a fitting/MCMC assessment. See the [authorized assignment, milestones and safety rules](ada_background_handoff.md). The detached worker qualified successfully and has published the tested milestones above. DOT can follow this report and the same development branch for each tested, pushed milestone. No merge is authorized.
