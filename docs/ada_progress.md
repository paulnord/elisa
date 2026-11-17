# Ada GPU development progress

Updated **2026-11-17T11:15:29.929254+00:00**. Review branch: `codex/gpu-resident-roche-pipeline`.
**Nothing merged. The assignment continues.**

## Latest milestone: nested surfaces and a finite contact/likelihood error estimator

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

All matched paths use affinity0–15/Nice10, threads1, FP64. The GPU driver
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
physics/data/output contracts, affinity0–15/Nice10, threads1, FP64. Each qualified
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
checkpointed window. Affinity0–15/Nice10, threads1, FP64. All per-call Observer,
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
window, affinity0–15/Nice10, threads1, FP64. Fixed order and few samples limit
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
