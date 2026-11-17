# Ada workload and hardware characterization

## Current snapshot: m019 observer port

The [m019 ledger](ada_validation/background-m019.json) extends the preceding
measurements below. 144 hardware controls pass without skips; complete changing-
physics comparisons include B1/B4 (original CPU/JAX CPU/GPU) and B8 (original CPU/GPU),
original JD mapping, independent repeat processes, finer field 2→3 controls and a
complete-call CUDA/NVTX/OSRT graph-node trace. Ten changing warm silhouette
samples per shape retain original serial CPU, vectorized NumPy and JAX baselines.

**The inherited observer interval is now mixed CPU/GPU work.** Its old CPU-named
key is renamed in the curated report. Do not reuse m018's 97.8% CPU fraction or its
conditional ceiling after this port. CPU clipping/topology, last 8 edge decisions,
precision seams and full original replays remain counted. Device rays/shadows
and root prefixes transfer real fresh inputs and return required host state.

### Complete-call timeline

The single profiled changed warm B1 call spans **9.6144 s**, with
**260556 kernel events**, **12681 copy events**,
**1.01386 s kernel union**, **0.0697093 s copy union**, and
**1.08455 s union of recorded device activity**. CPU work/dispatch/waits/gaps
in the remaining span are not separately attributed. Per-range unions account
for overlap; API waits overlap device work and must not be added. Cold and warm
ranges and operation classes are retained. This experiment has profiler overhead;
the independent unprofiled repeat is the timing control.

Removing every recorded copy nanosecond, optimistically treating it all as
critical-path time, gives only **1.0073×**
for this captured range. That is not a 2×/4× interconnect bandwidth prediction:
copy latency/staging/dispatch and critical overlap are not isolated. Numerous
nested root/launch operations are directly observed; their hardware bandwidth,
FP64 execution and launch-latency fractions remain unknown without further
attribution/counters. Kernel count alone is not proof of a launch bottleneck.

The same warm range also marks original CPU boundary-field evaluations:
**5.38756 s**, with
**0 s** overlap with recorded device
activity. Their serialized fraction is **56.036%**.
Keeping all other work fixed, 2×/4× acceleration of this measured CPU subset gives
**1.3892× /
1.7249×**; eliminating it entirely
has a **2.2746×** ceiling.
These are conditional stage scenarios, not predictions for a particular CPU.
Other host work and launch/gap fractions remain unassigned, and the profiled run
is not the official throughput timing.

### Conditioned silhouette stages

Each row below is B128 × 2048 points, with ten changed warm physical batches and
matching opacity/margin outputs. Input placement, synchronization, required
host output and validation are counted. Reported compiler outputs also include
resident status/residual fields. The scientific host arrays are 9 bytes/node;
GPU validity validation adds 1 byte/node of required host transfer. Allocator/
compiler accounting is distinct from actual DRAM traffic and sampled device
high-water memory.

| Component | Original serial CPU | Vectorized NumPy | JAX CPU | GPU |
| --- | --- | --- | --- | --- |
| primary | 484.1924 ms | 585.1443 ms | 104.8459 ms | 8.4857 ms |
| secondary | 458.7723 ms | 560.1420 ms | 112.2512 ms | 8.3287 ms |

Original and vectorized NumPy keep the original trig potential and12 Newton
updates; JAX CPU/GPU keep those stationary equations and12 updates with Cartesian
potential. Baseline ordering alternates per repeat; source/input hashes and
individual samples retain the distinction between vectorization/algorithm and
hardware. These are silhouette stages, not whole-analysis gains.

Active-load PCIe/clock/power/temperature/utilization and sampled memory ranges
are retained for every new bounded unit. CPU affinity/nice settings and one-
thread BLAS/OMP/MKL/NumExpr environment match the preceding policy; actual
JAX/XLA pool size is unmeasured. This shared machine is not an exclusive or
thermally controlled benchmark. All original CPU baselines inside units share
their unit's affinity. B8 JAX CPU and a tuned multiprocessing baseline were not
measured for this model; prior V505 multiprocess results stay separate.

### Conditional next technologies

A faster host link has no demonstrated tenfold complete-path opportunity in
these measured calls. More VRAM did not remove a measured capacity bottleneck.
DRAM bytes/read-write bandwidth, L2, occupancy/stalls and achieved FP64 counters remain
unavailable; no quantitative DRAM/FP64hardware scaling fraction is assigned.
Coherent CPU-GPU memory still carries access/dispatch/synchronization costs;
GPU-GPU NVLink is not automatically a CPU-GPU link. Stronger FP64, bandwidth,
cache, CPU throughput and memory capacity are distinct hypotheses. The immediate
software experiment is to fuse nested root/launch work and reduce remaining CPU
bridges, then repeat the whole-path measurements with unchanged scientific gates.

Reproduce in separately registered bounded Ada units:

```sh
python scripts/benchmark_gpu_shadows.py --device gpu --component primary --verified-bounded-gpu-window --output shadow.json
python scripts/validate_gpu_observer.py --backend gpu --batch 4 --repeats 1 --verified-bounded-gpu-window --output complete.json
python scripts/validate_gpu_observer.py --backend gpu --batch 1 --levels 2 3 --integration-level 3 --repeats 1 --verified-bounded-gpu-window --output refined.json
```

Replace GPU with reference/cpu as appropriate and preserve both input/output
contracts. For full NVTX ranges, wrap `validate_gpu_observer.evaluate` with the
same push/pop instrumentation shown below and use `--cuda-graph-trace=node`.
Compare traces and full raw/component/normalized curves and original posteriors
before interpreting timings. The strict1e-4likelihood budget still prevents
physical fit/MCMC qualification; no ESS or Colab tenfold-analysis claim follows.

## Preceding snapshot: m018 field geometry

This ledger separates tested implementation parity, numerical convergence and
observational/model adequacy. [m018 samples and provenance](ada_validation/background-m018.json)
retain source/input identities, individual timings, first calls, stage timers,
actual proposals, output contracts, memory accounts and active-load samples.
Raw operational logs and profiler traces remain private; curated scientific
aggregates and trace hashes are published. Nothing changes clocks, links,
profiling permissions, drivers, production settings or numerical precision.

## Machine and measurement conditions

Ada has a Ryzen9 5950X (16cores/32threads), RTX3080Ti (12GiB), driver595.99.02,
private Python3.12.4/JAX0.12.0.1, FP64 and disabled preallocation. BLAS/OMP/MKL/NumExpr environment
counts are one; actual JAX/XLA runtime pool size is unmeasured. Private CPU stages use affinity16–19,
nice19; bounded device units and their own original CPU/NumPy baselines use
cores0–15,nice10. These are shared-machine results, not an exclusive machine or
optimal parallel CPU survey. Earlier V505 four-process evidence is retained in
[analysis workloads](analysis_workloads.md), with a different model/contract;
no tuned multiprocess baseline for the current curvature comparison batch is
claimed. CPU parallelism remains a separate candidate improvement.

The GPU sysfs route is
`/sys/devices/pci0000:00/0000:00:03.1/0000:0a:00.0` through upstream
`0000:00:03.1`. GPU maximum is generation4/width16; the upstream port advertises
width8. Inventory reported idle generation1/width8. **Worker-owned active-load
samples report generation4/width8**, rather than treating idle generation as a
bandwidth limit. Per-window clock, power, temperature, utilization and sampled
memory ranges remain in the report. Sampling requests every0.4s plus query
latency miss short peaks; memory utilization is not measured DRAM traffic.

## Workload matrix

| Workload | Proposals / work | Evidence | Accuracy and output contract |
| --- | --- | --- | --- |
| geometry samples | B1/32/128 ×512/2048 changed rays, both stars | 10 warm changed batches/shape, original CPU/vectorized NumPy/JAX CPU/GPU | Original primitives3e-13 relative; points/normals64eps absolute; host points/normals/Jacobians/validity |
| curved area | B1/32/128 ×8/128 changed faces ×16 nodes | same baselines,10 warm samples; independent primary B32 process20 samples | Same original3e-13 area gate, host patch and total areas |
| complete reflected comparison | B1/4 × original uneven/repeated multi-band observations, two fresh field levels | 1 cold+2 warm batches/backend; repeated original baseline processes | Full raw/component/normalized curves, original mapping/prior/density and strict finite accuracy controls; fits unqualified |
| JD callback | B1, original time mapping | 1 cold+1 warm original CPU/GPU | Same synthesis/mapping/posterior parity gates |
| transfer calibration | 4KiB,64KiB,1MiB,8MiB FP64 buffers | fresh pageable source/results,20 warm repeats/size | Blocking per-stage timer; no cached host reads, pinned-memory or pure wire-bandwidth claim |
| CUDA timeline | B32 ×128faces ×16nodes, primary | cold+10 changed warm NVTX ranges; CUDA/NVTX/OSRT, CPU sampling/context-switch tracing disabled | Instrumented run only; unprofiled stage timings remain official |

CPU and vectorized NumPy retain original trigonometric potential/gradient and
56bisections. JAX CPU and GPU share Cartesian arithmetic/80bisections. Input and
output gates match, but operation counts differ. Stage algorithm/batching gains
are distinct from hardware gains. Both matched stage baselines execute in
alternating order per repeat. The second independent stage process and repeated
complete original baselines expose order/process drift. Whole batch samples are
few; ranges and medians are preliminary, with no supported tail/ESS estimate.

Each complete timer starts before physical serialization/constructors/prior and
ends after original mapping/density, host comparison curves and accuracy checks.
The report separates fixed input setup, reusable-engine construction, first
encountered compilation and changed warm calls. No cached physical prediction is
timed as a fresh synthesis. Reported row stage timers exclude shared field batch
preparation; enclosing totals include it. All source/data identities are frozen.

## Timeline and transfers

The node-level trace has ten warm marked calls. The first retained trace grouped
some CUDA graph work; the second uses explicit graph-node tracing to resolve it. Median **range 2.5375 ms**, recorded
**kernel union 1.1167 ms**, **copy union 0.0326 ms**, and
**remaining range time 1.3885 ms**. Each contains178kernels and8copies (including2device-to-device copies).
These are medians of separate nested quantities, not an additive accounting;
the per-call unions/overlap are retained to avoid double-counting. Remaining
time includes host dispatch, scheduling, synchronization and gaps, whose separate
fractions are unknown. CUDA API waits overlap device activity and must not be
added to it. The profiler has overhead and these are not throughput timings.

Recorded copies occupy 1.17–1.43% of individual warm
ranges. Even treating every recorded copy nanosecond as removable critical-path
time gives only 1.014× as an optimistic elimination
ceiling for this isolated trace. Copy latency, host staging and dispatch are
not all proportional to link bandwidth; this is **not** a2×/4×PCIe prediction.

| Pageable buffer, each direction | Median H2D API+sync | Median first D2H host read |
| --- | --- | --- |
| 4096 bytes | 0.1375 ms | 0.1168 ms |
| 65536 bytes | 0.1624 ms | 0.1258 ms |
| 1048576 bytes | 0.3804 ms | 0.2426 ms |
| 8388608 bytes | 1.7394 ms | 1.8627 ms |

Calibration excludes fresh host-buffer generation and correctness checks from
per-call intervals; it includes device placement/dispatch, blocking computation
and first host read. JAX may internally stage pageable inputs. The fresh addition
kernel prevents a cached host read; its separate time is retained. Compiler
argument/output/temp bytes, JAX allocator peak/pool bytes and sampled device
memory are separate fields. None is GPU DRAM traffic or exact combined
compiler/runtime/process peak. No OOM or capacity limit was reached in these
measured batches; more VRAM has no demonstrated current benefit.

## Conditional technology scenarios

The same warm complete GPU runs spend most time in CPU observer geometry,
adaptation, clipping and hashing. The ledger computes conditional2×/4×acceleration
of every other operation from each enclosing timer, without subtracting separate
medians. The progress report states the resulting ceiling with that CPU work
fixed. It changes after that stage is ported or parallelized.

| Technology/change | Evidence and conditional interpretation |
| --- | --- |
| Faster CPU or observer algorithm | CPU clipping fraction is directly timed in the same complete execution. Accelerating this stage is the present full-path opportunity; frequency/product speedups are not inferred. |
| Faster host-device link | Active4×8 verified; copy-only elimination ceiling above is small for the traced stage. No isolated bandwidth-only critical fraction;2×/4×bandwidth sensitivity remains unknown. |
| Faster GPU DRAM/cache | No DRAM byte/L2/stall counters. Kernel names/utilization/compiler bytes do not establish a bandwidth bottleneck. No quantitative prediction. |
| Stronger FP64 compute | FP64 roots/division/square roots execute, but achieved compute/occupancy/stalls are unmeasured. No Tensor Core or reduced-precision gain is assumed. |
| More VRAM | Allocator/pool and sampled memory are recorded, with no observed pressure in this sweep. A larger field/resident workload would need a new capped test and unchanged accuracy requirements. |
| Coherent CPU-GPU memory | Could change explicit copy/dispatch patterns after a port; coherence still has access and synchronization costs. No measured speedup. GPU-GPU NVLink is not automatically a faster CPU-GPU path. |

Nsight Compute is unavailable, RmProfilingAdminOnly=1 and perf_event_paranoid=4
were observed; no permissions were changed. CUDA API/kernel/copy tracing succeeded
with Nsight Systems2026.3.2. DRAM/cache/occupancy/achieved-FP64 attribution and
critical-path bandwidth/compute fractions remain missing. The trace is stronger
evidence than an idle snapshot, but does not justify a specific new-processor
speedup. Recompute this ledger after observer work becomes resident.

## Reproduction

Use the private pinned environment/config and CPU affinity policy. GPU commands
below require a registered hold-aware systemd unit, independent checkpoint
integrity hooks, FP64/preallocation settings and<=300s deadline. The scripts do
not allocate or restore production. Run each complete backend/shape separately.

```sh
ELISA_GEOMETRY_TEST_DEVICE=gpu python -m pytest tests/tests_experimental/test_batched_geometry.py -q
ELISA_GEOMETRY_TEST_DEVICE=gpu python -m pytest tests/tests_experimental/test_resident_geometry.py -q
python scripts/benchmark_batched_geometry.py --device gpu --part area --component primary --batch 1 32 128 --nodes 8 128 --repeats 10 --verified-bounded-gpu-window --output stage.json
python scripts/validate_gpu_resident_geometry.py --backend gpu --batch 4 --repeats 2 --verified-bounded-gpu-window --output complete.json
python scripts/profile_gpu_data_path.py --verified-bounded-gpu-window --output transfers.json
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-graph-trace=node --stats=true -o timeline python nvtx_geometry.py --device gpu --part area --component primary --batch 32 --nodes 128 --repeats 10 --verified-bounded-gpu-window --output profiled.json
```

For the focused trace, create this portable `nvtx_geometry.py` wrapper in the
repository root. It marks the unmodified driver's complete `device_call`; raw
trace and actual local instrumentation hashes are retained in the report.

```python
import ctypes
from scripts import benchmark_batched_geometry as benchmark
nvtx = ctypes.CDLL("libnvToolsExt.so.1")
nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
nvtx.nvtxRangePop.argtypes = []
original, calls = benchmark.device_call, 0

def marked(*args, **kwargs):
    global calls
    label = "geometry_cold" if calls == 0 else f"geometry_warm_{calls}"
    calls += 1
    nvtx.nvtxRangePushA(label.encode())
    try:
        return original(*args, **kwargs)
    finally:
        nvtx.nvtxRangePop()

benchmark.device_call = marked
benchmark.main()
```

Compare matched per-row physical identities, quadrature hashes, phase traces,
raw/component/normalized curves and original posteriors before interpreting
costs. A standalone GPU primitive timing never qualifies a physical likelihood.
Current finite1e-4 field/boundary stops still prevent a qualified fit or MCMC;
no chain mixing, posterior convergence or ESS/sec claim is made.
