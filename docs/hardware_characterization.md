# Ada workload and hardware characterization

## m045 — Matched physical proposals at eight and sixteen workers

| Walker processes | Fields | Restore starts, s | GPU owner, s | Host pool through checkpoints, s | Complete, s | Qualified likelihoods/min during sampling |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | CPU | 2.8 | 0.0 | 2794.6 | 2797.4 | 0.344 |
| 8 | GPU | 2.8 | 576.9 | 2538.5 | 3118.2 | 0.308 |
| 16 | CPU | 2.8 | 0.0 | 2440.8 | 2443.6 | 0.393 |
| 16 | GPU | 2.8 | 561.8 | 2184.7 | 2749.3 | 0.350 |

Each of the four sequential runs processed sixteen distinct, changing physical proposals from a common set of saved RNG states. The worker count controls independent walkers on physical cores 0–7 or 0–15, with one numerical-library thread per process. The CPU arm constructed each walker's four reflected fields within its worker. The GPU field owner computed fresh fields for two successive proposals per bounded unit. Both arms serialized and read the same optical-field archive contract, used the same FP64 C++ stationary solve on the host, and evaluated eight phase blocks per in-prior draw. The sixteen complete curves include both eclipse depths, both thermal and donor resolutions, and both interior rules. All finite checks and durable decisions are included in the sampler wall clock. The cross-backend audit ran afterwards. [Per-run and per-stage data](ada_validation/background-m045.json)

In the eight-worker CPU run, fresh fields accounted for 10.9% of summed walker time. Eclipse geometry, clipping and quadrature hashing took 64.8% of summed observer time. These stage shares use time inside the workers. The elapsed times in the table include scheduling and overlap.

At sixteen workers, peak host cgroup memory was 26.8 GiB with CPU fields and 12.0 GiB with GPU fields prepared before the pool. The corresponding mean numbers of busy cores over the sampled host intervals were 15.8 and 15.9.

The field archives totalled 579.8 MB in the sixteen-worker GPU-field run. During the bounded field calculations, sampled device memory peaked at 1348 MiB and median active power was 114.5 W. Active field-owner samples reported PCIe generation 4 at width ×8. The owner interval also includes field construction, compilation and file I/O. Each physical fit configuration ran once. The complete rates use the previously qualified injection and selected checkpoints. Computing the injection density from fresh fields and data is a separate cost measured in m044. [Injection cost](#m044--complete-injection-and-physical-draw-costs)

Before this worker comparison the two streams each took one step using the native C++ CPU or CUDA stationary solves. For seed 20261117 those matched steps took 1119.0 s on the fused host and 1029.5 s on the GPU path. For seed 20261118 they took 1119.7 and 947.8 s. These costs sum field export, eight phase calls, numerical assessment and durable advancement. GPU times include the registered launches and verified cleanup. Separate low-priority CPU comparisons shared the host at this stage. The single JAX CPU depth-four control took 1371.1 s for four complete curves at the first seed's second proposal. It covers one quarter of the resolution grid.

| Workers | Original 12° facets, s | Original 6° facets, s |
| ---: | ---: | ---: |
| 8 | 2.65 | 3.05 |
| 16 | 3.12 | 3.49 |

The original ELISa timings include process startup, installation of the matched fitter in each worker, and two physical likelihoods for each of the sixteen start/proposal pairs. These conditional comparisons use a single stellar facet resolution per fit and the original native interpolation settings. They measure the model that generated the synthetic data at 12° and its 6° refinement. The curved fits recompute sixteen resolutions per in-prior proposal. Compiling the stationary library was measured separately in m043. All 36 registered GPU units passed their test and verified the original Fifth42 checkpoint at both boundaries. Registration through verified exit took at most 153.7 s. [Matched starts](ada_validation/background-m045-starts.json)

## m044 — Complete injection and physical draw costs

All arms use the same revised eclipse quadrature, original data, priors, noise, third light and output contract. Each complete time sums serial calls on one host thread per physical position. The two GPU draws ran sequentially. Independent CPU experiments shared the host. The timing begins with field construction and ends after the numerical assessment or durable sampler decision. Per-draw times start from an established checkpoint. The initial density was shared by both starts within each backend. Each start also verified its own density and wrote its independent random-number state. Those per-start costs are in the validation data.

| Physical position | Implementation | Fields and archive, s | Eight phase blocks, s | Assessment and transition, s | Complete, s |
| --- | --- | ---: | ---: | ---: | ---: |
| injection | NumPy | 106.7 | 1388.4 | 2.5 | 1497.6 |
| injection | Fused host | 107.8 | 1012.4 | 2.6 | 1122.8 |
| injection | GPU fields and CUDA tangencies | 103.2 | 923.0 | 3.3 | 1029.5 |
| control | Fused host | 108.6 | 1044.2 | 2.7 | 1155.4 |
| draw0 | Fused host | 106.5 | 919.8 | 4.9 | 1031.2 |
| draw0 | GPU fields and CUDA tangencies | 103.2 | 889.1 | 4.8 | 997.1 |
| draw1 | Fused host | 106.1 | 922.0 | 4.7 | 1032.8 |
| draw1 | GPU fields and CUDA tangencies | 96.0 | 829.0 | 4.5 | 929.5 |

GPU times include registered unit launch, the FP64 device probe, transfers, synchronization and checkpoint verification. Every proposal rebuilds fields, and every observer call computes fresh eclipse geometry. All implementations retain host clipping and passband flux integration. The compiled stationary library from m043 was reused. The independent cross-backend audit and the reference calculations sit outside the table. Each configuration was timed once. The comparison uses individual physical draws. The earlier eight- and sixteen-worker measurements used a different observer protocol.

Inside the eight injection phase blocks the geometry, eclipse clipping and partition checks took 631.1 s, or 71.0% of the timed observer calls. This includes the CUDA stationary solves and their host-side preparation. Those solves transferred 6.66 GB to the device and 2.22 GB back to the host. The counts cover this solver only.

Peak sampled GPU memory was 1342 MiB, peak temperature was 54 °C, and median active power was 111.6 W. All 28 registered units verified the held Fifth42 checkpoint at both boundaries. The longest interval from unit registration through verification of its exit was 152.8 s. The hardware regression run passed all 54 cases. [Block timings, field archive sizes and decisions](ada_validation/background-m044.json)

## m043 — Stationary-point solves on the injection and a qualified draw

Each row measures one serial evaluation of changing two-star physics on Ada, using one host hardware thread at a time. The injection row ends at a resource stop. The qualified row starts from the same saved physical proposal and ends at a durable sampler decision. All three implementations write and read a fresh field archive, use the same two observation blocks per orbit, and apply the same sixteen-model check. The GPU totals include every registered launch, device probe, synchronization, transfer and verified unit exit. Whole times sum the serial calls and exclude time spent on independent tests between them. Separate configurations ran concurrently on different CPU threads. These are single trials on a shared machine.

| Case | Implementation | Fields and archive, s | Eight block calls inside driver, s | Complete wall, s |
| --- | --- | ---: | ---: | ---: |
| Injection | NumPy stationary solve, CPU fields | 111.7 | 1110.6 | 1242.8 |
| Injection | Fused C++ stationary solve, CPU fields | 109.7 | 788.1 | 919.0 |
| Injection | CUDA stationary solve, GPU fields | 197.1 | 705.0 | 940.1 |
| Qualified | NumPy stationary solve, CPU fields | 121.5 | 1457.5 | 1600.1 |
| Qualified | Fused C++ stationary solve, CPU fields | 122.5 | 1051.1 | 1195.4 |
| Qualified | CUDA stationary solve, GPU fields | 95.8 | 876.1 | 1007.3 |

On the qualified draw, the native CUDA path sent 7.194 GB of projected ray arrays and returned 2.398 GB of stationary offsets, in decimal units. These counts cover array payloads in the stationary solver. The timed field archive was 36.24 MB. The 299,745,732 selected rays included 34,928,154 host rechecks near the contour or outside a fused validity check. Synchronous native calls, including array transfers, took 8.13 s in total. Allocation, clipping, surface integration and likelihood remain in the complete wall measurement. Building the shared library offline took 0.87 s in a separate rebuild. The executable device instructions matched the tested build.

Hardware telemetry from the 23 bounded units sampled 1342 MiB peak GPU memory and 53 °C peak temperature. Median power in samples with nonzero GPU utilization was 111.8 W. All 46 regression cases passed in the final bounded unit, including the hardware-only limb-darkened sphere check. Kernel execution on this test uses the RTX 3080 Ti's FP64 units. The timings above apply to a single physical proposal at a time. The earlier eight- and sixteen-worker comparisons used host eclipse geometry. [Per-block timings, transfers and resolution checks](ada_validation/background-m043.json)

## m040 — Transfer-inclusive cost per checked physical transition

The timed inputs were two single physical draws from the qualified m039 start. Each backend rebuilt four reflected field combinations and evaluated each at two eclipse depths and two interior rules. Both runs used logical CPUs 16–19, nice 19 and single-thread BLAS/OpenMP. The CPU reference scheduled up to four of the eight phase blocks concurrently. The GPU evaluated one block per registered unit and could use its four host cores. The serial units kept one device context active at a time. Seed 20261117 ran CPU first and seed 20261118 ran GPU first. Each measured backend ran separately.

| Seed | Backend | Checkpoint through assessment (s) | Recheck and durable decision or accuracy stop (s) | Transition total (s) | Sum of eight synthesis process times (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 20261117 | Four CPU phase-block processes | 796.85 | 2.16 | 799.01 | 2657.44 |
| 20261117 | One GPU process at a time | 1326.96 | 2.18 | 1329.14 | 1309.77 |
| 20261118 | Four CPU phase-block processes | 788.48 | 2.19 | 790.67 | 2629.04 |
| 20261118 | One GPU process at a time | 1324.86 | 2.15 | 1327.01 | 1305.64 |

For seed 20261117 on CPU, the first host assessment encountered an array comparison error. Replaying the saved blocks through the serialized profile interface produced the reported assessment. That CPU time combines the makespan of the eight successful synthesis commands with the successful assessment and decision. The failed assessment and repair interval are excluded.

The transition total includes restoring the initialized state, regenerating the proposal, fresh field and observer setup, compilation encountered, host/device transfers, synchronization, output, phase joins, density evaluation, resolution checks and the final decision. The GPU wall clock also includes all eight registration, checkpoint verification, device probe and cleanup cycles. The process sum isolates the eight synthesis commands and overlaps on the four-process CPU run. Both backends used existing filesystem and compiler caches. The independent parity audit and deliberate restart tests ran outside these clocks. Initializing the starting position was measured separately in m039, which includes matched timing for its depth-three model.

These measurements concern one walker draw at a time. The CPU processes in the table parallelize numerical blocks of that draw. Matched 8- and 16-walker-worker fits and reliable effective sample rates are still pending. All 16 bounded GPU units verified the original held Fifth42 checkpoint at entry and cleanup. [Validation data](ada_validation/background-m040.json)

Across the eight GPU blocks, the summed host eclipse geometry, clipping and hashing times were 881.4 s and 876.1 s. The corresponding fresh field preparation times were 326.6 s and 327.7 s. Each block rebuilt the fields for its current physical draw. The stage breakdown and sampled active PCIe links, clocks, power and memory are in the validation data.

Outside the eight GPU synthesis commands, registration, probes, hold verification, orchestration and the host assessment together accounted for 17.19 s and 19.23 s.

As an independent check of the split phase calculation, JAX CPU synthesized all 18 phases at depth three, with the six-point interior rule and both thermal and donor resolutions, in 574.66 s including construction and two returned densities. The separate process took 578.16 s including initialization and comparison with the saved phase blocks. This measurement covers four of the sixteen models used for a sampler transition.

## m039 — Secondary re-emission in complete curves

Each fresh process evaluated the identical changing-physics proposal, two thermal meshes, two donor groupings and 18 input phases, followed by two diagnostic fit densities. All runs used logical CPUs 16–19, nice 19 and one BLAS/OpenMP thread. JAX could use its own worker threads. Process times include startup, compilation encountered, transfers, synchronization and output. The validation grid also evaluated eclipse depth four and the independent Duffy interior rule.

| Backend | Process s | Shared setup s | Fresh fields and common partition s | Eclipse geometry and clipping s | Optical integration s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serial CPU | 306.55 | 0.00 | 96.10 | 184.11 | 24.77 |
| JAX CPU | 569.35 | 3.14 | 56.24 | 487.05 | 19.49 |
| RTX 3080 Ti | 172.18 | 4.30 | 41.13 | 118.30 | 4.93 |
| RTX 3080 Ti repeat | 172.87 | 4.19 | 40.94 | 119.20 | 5.00 |

At this boundary the GPU completed the diagnostic workload 1.78× as fast as the serial reference. Peak sampled device memory was 1378 MiB and median active power was 113 W. Posterior throughput requires proposals that pass the combined accuracy check. [Scientific results and timings](ada_validation/background-m039.json)

## m038 — Complete curves with refined primary receiver means

Each fresh process built four reflected stellar fields at proposal 0, crossed two thermal meshes with two donor groupings, integrated the 18 input phases at eclipse depth three, and returned two diagnostic fitted densities. The serial reference uses the same curved model with NumPy and ELISa’s CPU routines. The process clock includes startup, compilation encountered, CPU geometry and clipping, host/device transfers, synchronization and output. All four runs used logical CPUs 16–19, nice 19, and BLAS and OpenMP thread limits of one. JAX could use its own worker threads. Stage times below are internal measurements.

| Backend | Process s | Shared setup s | Fresh fields and common partition s | Eclipse geometry and clipping s | Optical integration s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serial CPU | 303.87 | 0.00 | 95.09 | 182.37 | 24.83 |
| JAX CPU | 570.80 | 3.18 | 55.93 | 488.27 | 20.04 |
| RTX 3080 Ti | 171.14 | 4.40 | 40.28 | 118.00 | 4.92 |
| RTX 3080 Ti repeat | 171.14 | 4.40 | 40.08 | 118.21 | 4.90 |

The first matched GPU run completed this workload 1.78× as fast as serial CPU. Its peak sampled device memory was 1378 MiB and median active power was 113 W. The full sixteen-model accuracy check also requires the deeper eclipse and the second observer rule. The native 8- and 16-worker fits use a different finite model. [Scientific results and hardware samples](ada_validation/background-m038.json)

## m037 — Complete disk heating comparisons at two thermal and donor resolutions

Each measurement rebuilt four reflected fields and evaluated three primary spectra over the two phase blocks of proposal 0. Each block started a fresh process. Both blocks rebuilt the fields and evaluated repeated orbital phases independently. The last CPU process joined the full curves, calculated the fit predictions and attributed the observed likelihood changes. The table sums these process clocks, including compilation encountered, host/device transfers, synchronization, file I/O and process startup. Stage clocks exclude the final CPU join. The table uses the isolated serial CPU repeat. The first CPU block overlapped the last qualification join for 1.61 s and is recorded separately in the ledger. Each backend used logical CPUs 16–19, nice 19 and one thread for BLAS and OpenMP. JAX could use its own CPU worker threads.

| Backend | Total process s | Blocks 1 / 2, s | Join and fit, s | Field and common geometry preparation, s | Observer geometry and clipping, s | Optical integration, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Serial CPU | 645.58 | 322.48 / 321.58 | 1.52 | 334.76 | 234.35 | 71.86 |
| JAX CPU | 839.80 | 418.79 / 419.49 | 1.53 | 121.18 | 653.73 | 56.84 |
| RTX 3080 Ti | 291.03 | 140.59 / 148.90 | 1.55 | 90.62 | 181.80 | 10.32 |
| RTX 3080 Ti repeat | 291.00 | 140.59 / 148.86 | 1.55 | 90.75 | 181.62 | 10.32 |

During the two GPU blocks of the first comparison, peak sampled device memory was 864 MiB. The GPU completed this diagnostic workload 2.22× as fast as the serial reference. The full sixteen-model accuracy check also requires depth four and the second interior rule. The eight- and sixteen-worker native ensemble uses a different finite model. [Timings and device samples](ada_validation/background-m037.json) · [Native sampler baselines](ada_cublas_native_spectra.md)

## m036 — Native blackbody integration at 1.5° primary spacing

Each pair uses the same 32-walker physical ensemble and two red–blue steps. Times below include interpreter startup, the entire fit, interprocess transfer and output. The unit monitor polled the child once per second. The application and sampler clocks give finer timing boundaries in the [cuBLAS report](ada_cublas_native_spectra.md). The denominator for likelihood throughput is 96 sampler evaluations. Each implementation also computed 32 preflight curves and one final summary curve.

| Walker workers | CPU matrix product, process s | cuBLAS, process s | CPU / cuBLAS | Repetitions, CPU / GPU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 147.84 | 131.26 | 1.13 | 1 / 1 |
| 4 | 89.44 | 80.83 | 1.11 | 1 / 1 |
| 8 | 88.23 | 78.58 | 1.12 | 2 / 2 |
| 12 | 85.71 | 76.47 | 1.12 | 1 / 1 |
| 16 | 87.10 | 78.10 | 1.12 | 2 / 2 |

The CPU implementation uses one numerical thread per walker. The GPU method uses the same CPU allocation and adds a single CUDA owner on logical CPU 16, which shares a physical core with logical CPU 0. Both use full-precision temperatures and the same quadrature. The three reverse-order groups of four independent likelihoods took 21.74 s on CPU and 18.66 s with cuBLAS over the outer process boundary. CUDA compilation preceded all timings. [Timing records and tests](ada_validation/background-m036.json)

## m035 — Primary irradiation curvature and complete depth-three likelihoods

On proposal 0, each process built four fresh reflected fields, integrated the same six-point-interior light curves at 18 supplied phases and returned two diagnostic fitted densities. Each uses FP64, CPU affinity 16–19, nice 19 and single-thread numerical libraries. Process times include encountered compilation, transfers, synchronization, host geometry and output. Stage clocks can nest. The sixteen-model accuracy assessment additionally requires depth four and the Duffy interior rule.

| Backend | Process (s) | Shared field setup (s) | Fresh fields and common partition (s) | Observer geometry and clipping (s) | Optical integration (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serial CPU | 297.49 | 0.00 | 88.53 | 182.75 | 24.64 |
| RTX 3080 Ti | 169.04 | 4.27 | 38.16 | 117.84 | 5.37 |
| JAX CPU | 564.13 | 3.18 | 49.53 | 488.44 | 19.63 |
| RTX 3080 Ti repeat | 169.83 | 4.34 | 38.60 | 118.31 | 5.13 |

During the first GPU process, active samples showed PCIe Gen 4 ×8 and peak observed memory of 864 MiB. Geometry for the neighboring-face fit, including its small matrix inverses, runs on the host. The device applies the fit to the fresh deposits, normalizes the curved-area mean and limits the heating before spectral integration. These timings measure one changing-physics workload. The original eight- and sixteen-worker sampler uses its own facet model. [Full timings](ada_validation/background-m035.json) · [Native CPU fit and mesh costs](ada_native_mesh_refinement.md)

## m034 — Matched cubic-primary eclipse curves and fitted densities

Each fresh process evaluated the same physical proposal 0 at depth three. Each built the four combinations of thermal-field and donor resolution, integrated 18 supplied phases (14 distinct orbital phases) and returned the original fitted density for the finer field at both donor resolutions. All ran with affinity 16–19, nice 19, FP64 and single-thread numerical libraries. The GPU process clock includes every host bridge, transfer and synchronization. The table gives seconds for one run per backend. The full sixteen-model check also needs the alternate interior rule and the deeper eclipse calculation.

| Backend | Process | Shared field preparation | Fresh fields and partition | Observer geometry and clipping | Optical integration |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serial CPU | 296.93 | 0.00 | 87.03 | 183.55 | 24.81 |
| JAX CPU | 564.36 | 3.25 | 49.21 | 488.90 | 19.64 |
| RTX 3080 Ti | 166.98 | 4.34 | 37.04 | 117.38 | 4.83 |

Serial CPU took 1.78 times the GPU process time. Device samples during utilization showed PCIe Gen 4 ×8, a median 112.5 W and a sampled peak of 864 MiB. The result covers this single proposal and output contract. [Validation data](ada_validation/background-m034.json)

## m033 — Positive interior quadrature, complete reflected curves

Proposal 38 uses the selective shadow boundary at depth three and the original fitted observations. The table gives one matched fresh-process run per cell, including imports, field and curve construction, original likelihood mapping and output serialization. All cases used affinity 16–19, nice 19, one numerical-library thread and FP64. The GPU rows include every transfer and synchronization. Processes share the private filesystem and compiler-cache policy. All three backends place and weight the new interior rays on the host. The hybrid evaluates the existing device geometry and optical kernels on those rays. A slash in the curve-stage column separates fresh fields and partition, observer geometry/clipping, and integration of the four field variants. Stage timers use the instrumentation in the curve implementation.

| Interior rule | Backend | Process (s) | Shared field preparation (s) | Curve stages (s) |
| --- | --- | ---: | ---: | --- |
| 6 nodes | Serial CPU | 280.69 | 0.00 | 78.17 / 179.56 / 21.33 |
| 6 nodes | RTX 3080 Ti | 158.49 | 4.31 | 33.57 / 112.82 / 4.38 |
| 6 nodes | JAX CPU | 548.46 | 3.20 | 48.30 / 477.99 / 15.72 |
| 9 nodes | Serial CPU | 292.62 | 0.00 | 77.68 / 184.26 / 29.15 |
| 9 nodes | RTX 3080 Ti | 161.61 | 4.45 | 33.42 / 114.53 / 5.86 |

The serial CPU/GPU process-time ratio with six nodes was 1.77×. Changing from nine to six nodes changed the serial CPU time from 292.62 to 280.69 s and the GPU time from 161.61 to 158.49 s. The fine-field curve and fitted density were recomputed for each donor resolution. Timing variability, memory bandwidth and ESS/s require separate measurements. The six-point GPU run supplied 194 samples with nonzero utilization, with PCIe Gen 4 ×8, median device power 112.3 W and peak sampled used device memory 862 MiB. [Curves, numerical checks and timings](ada_validation/background-m033.json)

## m032 — Native ensemble worker scaling

For 416 fresh densities, the best three-run median was 5.17 s with 16 original CPU workers, and 24.44 s with 12 worker processes sharing one FP64 GPU owner. The owner used the SMT sibling of CPU 0. The measurements include surface-array IPC, device transfers, synchronized totals and pool startup and shutdown. The experiment did not measure a per-call timing decomposition or memory bandwidth. [Worker counts, hardware, run order and individual results](ada_mcmc_scaling.md)

## m031 — Sampled irradiation at ordinary eclipse depths

| Backend | Depth 3 (s) | Depth 4 (s) | Likelihood check (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: |
| Serial CPU | 234.78 | 378.77 | 1.56 | 615.10 |
| JAX CPU | 445.43 | 813.38 | 1.57 | 1260.39 |
| RTX 3080 Ti | 101.38 | 170.07 | 1.59 | 273.04 |

| Backend | Stellar fields and common partition (s) | Eclipse packets and hashing (s) | Variant integration (s) |
| --- | ---: | ---: | ---: |
| Serial CPU | 156.73 | 377.93 | 75.71 |
| JAX CPU | 96.27 | 1100.01 | 50.24 |
| RTX 3080 Ti | 67.55 | 178.02 | 10.85 |

Stage times sum the curve stages from both synthesis processes. The three process-wall measurements additionally include construction, transfers, synchronization, outputs and likelihood assessment. All used cores 16–19, nice 19, FP64, single-thread BLAS and existing compiler and filesystem caches. CPU timings include their private environment wrapper. GPU timings begin after the unit’s device and checkpoint checks. The negotiated PCIe generation was 4 at width 8 during nonzero GPU-utilization samples. Sampled device-wide memory reached 858 MiB. Memory traffic and stalls would be needed to quantify a bandwidth upgrade. [Proposal and accuracy boundary](ada_progress.md)

## m030 — Eight-model reflected trial with six spectra per face

| Backend | Depth 3 (s) | Depth 4 (s) | Original fitter and gate (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: |
| Serial CPU | 236.95 | 381.02 | 1.53 | 619.50 |
| JAX CPU | 496.60 | 821.93 | 1.54 | 1320.08 |
| RTX 3080 Ti | 104.85 | 170.46 | 1.55 | 276.86 |

| Backend | Fresh fields and common partition (s) | Observer packets and hashes (s) | Variant flux integration (s) |
| --- | ---: | ---: | ---: |
| Serial CPU | 155.74 | 383.25 | 75.91 |
| JAX CPU | 95.77 | 1156.65 | 54.03 |
| RTX 3080 Ti | 68.33 | 180.48 | 11.67 |

Each stage time sums the measurements from the two depth processes. The observer column covers the line-of-sight transform, geometry evaluation, adaptive eclipse clipping and packet hashing. On GPU this stage accounted for 65.2% of the complete measured trial. CPU work and transfers remain inside these timings. The stage table omits setup outside the curve call and the final replay of the likelihood.

Fresh processes for each depth and the final original-density assessment used the same physical parameters, model and output. The compiler and filesystem caches were available to all three backends. The CPU wall times include its environment wrapper. GPU timings start with the science process after the unit’s checkpoint and hardware checks and end when it exits after writing the returned curves. Each total covers one complete evaluation. The chain runs recorded separately in the ledger used concurrent depth processes on serial CPU and sequential bounded units on GPU. The device's sampled active PCIe generation was [4] at width [8]. Sampled device-wide memory reached 860 MiB. The finite rule keeps host topology and clipping and performs both host-to-device and device-to-host transfers. The ledger includes the separately timed observer stages and integrated-field stages. Link speed alone cannot attribute a bottleneck to bandwidth. [Matched physics and resolution](ada_progress.md)


## m029 — Guarded physical trial

| Backend | Depth 3 (s) | Depth 4 (s) | Guard and original fitter (s) | Sum of process times (s) |
| --- | ---: | ---: | ---: | ---: |
| Serial CPU | 183.44 | 304.23 | 1.60 | 489.27 |
| JAX CPU | 321.51 | 643.86 | 1.59 | 966.95 |
| RTX 3080 Ti | 83.91 | 146.33 | 1.61 | 231.85 |

Each backend synthesized the identical first physical proposal with two depths in separate fresh processes. Four field/donor variants share each depth’s eclipse packets. Each process wall includes imports, fixed fixture setup, fresh stellar states and any compilation, host/device transfers and synchronization, and output. The third process reads the files, checks all eight corners and replays the original fitted density. All three backends used the same cores 16–19 and nice 19 with the single-thread BLAS/OMP settings. Each table entry is one measurement. Pre-existing filesystem and compiler caches were available. The wrapper used to enforce CPU settings is included in the CPU process measurements. Sampled active PCIe generations were [4] at widths [8], with device-wide sampled memory up to 858 MiB. Sampling establishes active link state. Bandwidth attribution requires measured traffic and stalls. The operational CPU chain evaluated its two depths concurrently while the GPU used sequential bounded units. Those separate wall times include contention and scheduling. [Scientific comparison](ada_progress.md)

## m028 — Complete curves with curved emitting-face irradiation

| Changing proposals | Serial CPU first / warm (s) | JAX CPU first / warm (s) | GPU first / warm (s) | Warm serial CPU/GPU | Whole-process serial CPU/GPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 41.68 / 41.66 | 95.40 / 66.19 | 58.30 / 25.20 | 1.65× | 0.98× |
| 8 | 82.13 / 81.74 | 163.96 / 128.34 | 85.27 / 46.89 | 1.74× | 1.22× |

One first and one changed warm batch per backend on the Ryzen 9 5950X and RTX 3080 Ti. The first and warm times include all CPU bridges, transfers, compilation encountered on that call, synchronization, host predictions, priors and accuracy checks for two complete models at each proposal. The whole-process ratios also include setup and output. Both timed serial CPU/GPU pairs and JAX CPU B4 used cores 0–15 at nice 10 with single-thread BLAS. JAX CPU B8 used the same cores at nice 19 with CUDA hidden. Brief auxiliary accuracy audits on sibling hardware threads overlapped parts of the JAX CPU comparisons. On B8 the GPU had 182 active samples, negotiated PCIe generation 4 with width ×8, and reached 608 MiB sampled device-wide.

| Backend and proposal | Curved surfaces (s) | Grouping and horizon index (s) | Four irradiation updates (s) | Remaining stage work (s) | Total reflection (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serial CPU, 0 | 0.106 | 0.789 | 10.133 | 0.019 | 11.047 |
| Serial CPU, 38 | 0.106 | 0.748 | 10.143 | 0.011 | 11.008 |
| GPU, 0 | 1.432 | 0.758 | 1.638 | 0.381 | 4.208 |
| GPU, 38 | 0.708 | 0.773 | 1.497 | 0.149 | 3.127 |

The stage timer starts with a fresh intrinsic surface. It includes both stars, building the sparse index on the host, transfers, two reflection sweeps and returning temperatures. The four device updates synchronize individually in this diagnostic. The remaining time includes device placement, other setup, checking flux and returning the temperatures. Both proposals used field level 3 and donor level 2. The CPU and GPU runs processed the same proposals in the same order with a persistent operator. A changed proposal can still compile new shapes. These host wall times locate work for profiling. Attributing it to PCIe or device memory bandwidth requires measurements of traffic and stalls. [Model comparisons and finite limits](ada_progress.md)

## m027 — Spatially resolved donor irradiation

| Changing proposals | Serial CPU first / warm (s) | JAX CPU first / warm (s) | GPU first / warm (s) | Warm serial CPU/GPU | Whole process serial CPU/GPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 40.88 / 41.05 | 94.63 / 65.72 | 52.63 / 21.18 | 1.94× | 1.09× |
| 8 | 81.29 / 81.16 | 158.93 / 124.11 | 74.70 / 39.48 | 2.06× | 1.40× |

One first and one changed warm batch per process on the Ryzen 9 5950X and RTX 3080 Ti. The finite reference, JAX CPU and GPU return the same two complete field resolutions and their mapped likelihood. The serial CPU and GPU timing pairs used cores 0–15 at nice 10 with single-thread BLAS. JAX CPU B4 used those settings. Its B8 timing used nice 19 with CUDA hidden. On B8 the GPU had 184 active samples. PCIe generation 4 and width ×8 were observed during activity, with a device-wide sampled peak of 602 MiB. Sampling measures GPU utilization and clocks over time. The effect of a faster interconnect requires stage-level traffic and stall measurements. [Source, model and accuracy limits](ada_progress.md)

## m026 — Directional donor moments and resolved horizons

The Ada Ryzen 9 5950X and RTX 3080 Ti synthesized two fresh, complete reflected curves per changing candidate at field levels 0 and 1 with donor cap 0. This cap exercises grouping at level 1. Timings include host selection and field setup, JIT compilation when encountered, CUDA transfers, synchronization and the returned predictions, log density and nominal accuracy checks.

| Changing proposals | Warm calls per process | Serial CPU first / warm (s) | JAX CPU first / warm (s) | GPU first / warm (s) | Warm serial CPU/GPU | Whole process serial CPU/GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 10.28 / 10.17 | 44.48 / 16.14 | 35.35 / 4.45 | 2.29× | 0.82× |
| 4 | 1 | 40.97 / 40.83 | 90.74 / 61.71 | 54.13 / 18.71 | 2.18× | 1.09× |
| 8 | 1 | 81.84 / 81.58 | 154.62 / 118.90 | 67.20 / 30.48 | 2.68× | 1.64× |

B1 warm entries are medians of three changed batches. B4 and B8 have one changed warm sample per backend. The matched serial CPU/GPU pairs ran on cores 0–15 at nice 10 with single-thread BLAS and OpenMP. B4 JAX CPU used the same settings. Supplementary JAX CPU B1 and B8 ran on the same cores at nice 19 with CUDA hidden. Some fine CPU accuracy controls used cores 16–19, the SMT siblings of cores 0–3, during part of the measurement series. The B8 serial CPU/GPU measurements followed those controls. The timing ledger retains per-process samples and affinity. The matched B4 GPU comparison took 16.89 s warm using one projected normal per group and 18.71 s using the individual directional moments and horizon splitting. Serial CPU times were 40.88 s and 40.83 s, respectively.

The profiled B1 warm range spanned 5.55 s. CUDA kernels occupied 0.972 s in union and device copies occupied 0.051 s. The trace captured 1,733 copies (199.2 MiB) from host to device and 3,249 copies (374.1 MiB) in the reverse direction. The CPU boundary-reference segment serialized with respect to device activity occupied 32.1% of the range. Doubling just that segment’s speed would yield 1.19× under the measured overlap. The optimistic ceiling for eliminating all recorded copy duration is 1.009×. Copy duration combines bandwidth and latency. A GPU memory-bandwidth projection still needs device traffic and stall counters.

During the B8 GPU run, 182 samples had nonzero device utilization. The observed active PCIe generation was 4, width ×8, and sampled device-wide memory peaked at 596 MiB. The Nsight trace ran separately from the throughput pairs. Colab’s earlier A100 measurement used m022. [Full timing and accuracy evidence](ada_validation/background-m026.json)

## m025 — Area-integrated reflection on changing physical proposals

Ada’s Ryzen 9 5950X and RTX 3080 Ti evaluated the same changing binaries, two complete multi-band curves per binary, the original observation mapping and the finite-error checks. The throughput cases use three samples per receiving face and field levels 0 and 1, where cap 2 retains every visible emitter. The fine level-3 timing in the progress report includes grouped emitters. All timings include host state assembly, LD-table work, eclipse integration, device transfers, synchronization and returned predictions. The GPU evaluates receiver and emitter geometry and the tiled irradiation sum. Host visibility selection, group construction and parts of the curved observer remain in the timed path.

| Changing proposals | Warm calls per process | Serial CPU first / warm (s) | JAX CPU first / warm (s) | GPU first / warm (s) | Warm serial CPU/GPU | Whole process serial CPU/GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 10.27 / 10.17 | 44.40 / 16.14 | 35.58 / 4.51 | 2.25× | 0.81× |
| 4 | 1 | 40.63 / 40.79 | 90.91 / 61.77 | 48.41 / 16.92 | 2.41× | 1.22× |
| 8 | 1 | 80.93 / 80.56 | 154.43 / 118.28 | 67.22 / 30.85 | 2.61× | 1.61× |

Warm entries are medians. B1 has three physically changed calls per process. B4 and B8 have one each, so their sampling uncertainty is undetermined. The serial CPU/GPU pairs ran on cores 0–15, nice 10, with single-thread BLAS and OpenMP. The B4 JAX CPU run used the same settings. The additional CPU-only B1 and B8 JAX processes used cores 0–15 at nice 19 with CUDA hidden. The ledger includes each enclosing call, process totals, source and input fingerprints and host peak RSS.

The profiled B1 warm call spanned 5.62 s. CUDA kernel intervals occupied 0.975 s in union and device copy intervals occupied 0.051 s. The trace recorded 1,731 host-to-device copies (199.3 MiB) and 3,249 device-to-host copies (374.1 MiB). Making just the explicitly timed, serialized CPU boundary reference segment (31.7% of the range) twice as fast would yield 1.19× for this call under the measured overlap. Eliminating the duration of all recorded copies would yield at most 1.009× at fixed other work. Copy duration includes latency. Estimating the effect of higher GPU DRAM bandwidth still requires memory traffic and kernel stall measurements.

During the B8 GPU run, 188 samples recorded nonzero device utilization. Negotiated PCIe generation was 4 and width was ×8. The largest sampled device-wide memory use was 596 MiB. The profiler runs separately from the unprofiled timing pairs. The earlier Colab A100 measurement below applies to m022. The m025 timing results cover Ada and the tested synthetic proposals. [Machine-readable profiles and timing distributions](ada_validation/background-m025.json)

## m024 — Transfer-inclusive receiver irradiation

The timing pairs compare fresh physical proposals, complete reflected curves at two finite resolutions and the same observational likelihood and accuracy checks. The host computes the stencil geometry, manages the adaptive eclipse frontier, prepares LD data and completes the original density. CUDA performs the thermal reconstruction, resident geometry and other existing device stages. The fine fitting accuracy gate remains active.

| Changing proposals | Warm calls per process | Serial CPU first / warm (s) | JAX CPU first / warm (s) | GPU first / warm (s) | Warm CPU/GPU | Complete process CPU/GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 10.25 / 10.20 | Unmeasured | 32.96 / 3.97 | 2.57× | 1.45× |
| 4 | 1 | 40.77 / 40.83 | 91.28 / 61.59 | 46.11 / 16.84 | 2.43× | 1.26× |

The B1 Nsight Systems warm range spanned 5.64 s. CUDA kernels occupied 0.970 s in union and device copy intervals occupied 0.051 s in union. Original CPU boundary reference work serialized with respect to recorded device activity occupied 32.0% of the range. Doubling that measured CPU segment alone would give a conditional 1.19× whole-range speedup. The trace includes profiling overhead. The throughput table uses independent unprofiled processes. A GPU DRAM bandwidth speedup remains unquantified. The ledger retains individual changed-call timings, sampled link states and memory, transfer counts/bytes and interval overlap. [Machine-readable results](ada_validation/background-m024.json)

## Colab A100 comparison for the m022 four-proposal batch

Paul ran the m022 curved-area thermal model on a Colab NVIDIA A100-SXM4-40GB. Each process evaluated the same four changing proposals twice, including both finite fields, eclipse curves, host outputs and likelihood checks. All eight finite-field checks stopped at the original accuracy threshold. All 24 reported adaptive traces matched their archived counterparts. The console transcription supplies one warm sample per backend. The raw Colab archive was unavailable at transcription.

| Host and backend | First complete batch (s) | Changed warm batch (s) | Process through both outputs (s) |
| --- | ---: | ---: | ---: |
| Colab serial CPU reference | 84.06 | 83.88 | 170.56 |
| Colab JAX CPU | 137.46 | 83.22 | 224.88 |
| Colab A100 | 80.14 | 30.75 | 123.32 |
| Ada serial CPU reference | 37.08 | 37.10 | 75.70 |
| Ada JAX CPU | 87.82 | 61.40 | 151.76 |
| Ada RTX 3080 Ti | 41.34 | 16.88 | 61.00 |

On Colab, the warm batch ran 2.73× faster on GPU than on the serial CPU reference. The ratio including startup and both outputs was 1.38×. Dependency installation took another 80.25 s outside these process times. The RTX host completed the warm GPU batch in 55% of the A100 host's time. Host CPU, runtime and accelerator all differed. Ada's geometry, clipping and hashing occupied 16.00 s of its 16.88 s warm batch, including nested GPU work. The supplied Colab output has no equivalent stage timeline. This experiment leaves the bandwidth contribution and posterior sampling rate unmeasured. [Matched m022 Ada evidence](ada_validation/background-m022.json)

## m023 — Face-local spectral fields

Ada's Ryzen 9 5950X and RTX 3080 Ti ran the same changing binaries, two field levels, multi-band curves and original mapped density and accuracy checks. The reference uses original CPU physics on the curved finite geometry. JAX CPU and CUDA use the compiled local thermal and sampling kernels. The CPU LD/atlas preparation, transfers of corner spectra and host packets, synchronization, and returned full curves are inside the timings. The implementation also computes the parent face spectra while preparing the irradiated star. Source moments remain on the selected device across a candidate's phases.

| Proposals | Serial CPU reference (s) | JAX CPU (s) | GPU (s) | Warm CPU/GPU | Complete process CPU/GPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10.27 / 10.07–10.17 | 45.60 / 14.91–16.38 | 33.36 / 3.92–5.24 | 2.27× | 0.85× |
| 4 | 40.71 / 40.87 | 91.03 / 60.96 | 45.89 / 16.71 | 2.45× | 1.27× |
| 8 | 81.20 / 80.93 | Unmeasured | 64.84 / 30.67 | 2.64× | 1.66× |

Times are cold / warm seconds. The primary comparison has 3/1/1 warm samples for batches of 1/4/8 proposals. An independent B1 series measured 2.20× warm and 0.70× for the whole process. A separate Julian-date pair measured 1.91× warm. Matched jobs used affinity 0–15, nice 10 and single-thread BLAS/OMP. Lower-priority CPU checks overlapped some early hardware units on cores 16–19. FP64 used JAX 0.12.0.1 and driver 595.99.02. The larger scientific configuration's cold reference/GPU ratio was 2.06×. Sample counts limit extrapolation.

On the first changed B1 call, common observer geometry, clipping and packet hashing took 8.871 s on reference CPU and 5.054 s on the hybrid GPU path. The separate warm CUDA trace spanned 5.509 s, with 33,481 kernels and 6,286 copies. Kernel intervals occupied 0.970 s, including 0.779 s in native nested roots. Copies occupied 0.051 s and moved 208.4 MB host-to-device, 392.1 MB device-to-host and 37.4 MB device-to-device. Eliminating all recorded copy duration would bound acceleration at 1.009×. The marked CPU boundary precision region occupied 1.771 s with 0.0000 s of device overlap. Conditional 2× and 4× reductions of that serialized portion give 1.192× and 1.318× overall. These intervals include profiler overhead. The throughput table uses unprofiled runs.

Nonzero GPU-utilization samples in the B1 run recorded PCIe generation 4 and width 8. Sampled shared-device memory reached 598 MiB. The trace leaves CPU preparation, launch and synchronization gaps partly unassigned. DRAM traffic, achieved FP64 throughput, exact allocator peaks and per-kernel stalls remain unmeasured. [Scientific profiles, telemetry and trace](ada_validation/background-m023.json)

## m022 — Curved-area thermal integration

The measured job constructs a changing binary for every proposal, produces two complete finite light curves on field grids 0/1, returns the curves and evaluates the original mapped density and accuracy checks. CPU and GPU jobs ran with affinity 0–15, nice 10 and single-thread BLAS/OMP on Ada's Ryzen 9 5950X and RTX 3080 Ti (12 GiB, FP64, JAX 0.12.0.1, NVIDIA driver 595.99.02). The CPU reference uses original physics routines on the declared curved finite model. Initial private CPU validation used cores 16–19 at nice 19 and overlapped some bounded units.

| Proposals | Serial CPU reference (s) | JAX CPU (s) | GPU (s) | Warm CPU/GPU | Complete process CPU/GPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.31 / 9.14–9.27 | 42.15 / 14.78–16.20 | 29.21 / 3.96–5.47 | 2.04× | 0.83× |
| 4 | 37.08 / 37.10 | 87.82 / 61.40 | 41.34 / 16.88 | 2.20× | 1.24× |
| 8 | 73.90 / 73.83 | Unmeasured | 60.10 / 30.79 | 2.40× | 1.59× |

Cells give cold / changed warm elapsed seconds. An independent B1 run gave a warm CPU/GPU ratio of 1.92×. B1/B4/B8 have 3/1/1 changed warm calls in the primary comparison. The independent Julian-date B1 comparison gave 1.66× on one changed call. The first changed call spent 8.852 s on the reference and 5.230 s on the hybrid observer's common geometry, clipping and packet hashing. The GPU time for batched field preparation was 0.035 s. The ledger records nested stage scopes, per-process RSS, cold costs, repeat distributions and conditional amortization.

The separately profiled warm B1 span was 5.520 s, containing 31,771 kernels and 6,112 copies. Kernels occupied 0.960 s, with 0.779 s in native nested roots. Copy intervals occupied 0.058 s, moving 306.4 MB host-to-device, 391.9 MB device-to-host and 37.1 MB device-to-device. Eliminating the entire recorded copy duration would bound acceleration at 1.011×. The original CPU boundary precision region occupied 1.780 s, with 0.0000 s overlapping recorded device activity. Conditional 2× and 4× reductions in its serialized time give 1.192× and 1.319× overall. Unassigned time also includes preparation, launch, synchronization and gaps. The throughput table uses unprofiled processes.

The B1 unit sampled negotiated PCIe generation 4 and width 8 at nonzero GPU utilization. Sampled shared-device memory reached 594 MiB. DRAM traffic, achieved FP64 throughput, hardware occupancy/stalls and critical bandwidth fraction remain unmeasured. These timings support conditional stage comparisons. Predicting a hardware upgrade or a Colab runtime requires measurements on that target. [Complete timings, telemetry and trace summary](ada_validation/background-m022.json)

## m021 — Continuous intensity reconstruction

Resident vertex moments are prepared per fresh candidate and field level. The compiled observer stage gathers by parent, interpolates the moments, evaluates limb darkening and sums across accepted rays for every passband. Required host packet data and returned CPU quantities are inside the elapsed measurements. The finite model differs from the m020 cell-constant protocol. Comparisons below pair the same physical proposals and output contract across backends.

| Proposals | Serial CPU reference | JAX CPU | GPU |
| ---: | ---: | ---: | ---: |
| 1 | 9.84 / 9.80–9.88 | 48.13 / 15.80–17.45 | 32.44 / 3.87–5.20 |
| 4 | 38.88 / 38.99 | 91.48 / 62.22 | 45.79 / 16.80 |
| 8 | 77.57 / 77.24 | Unmeasured | 63.72 / 30.29 |

Numbers are cold / warm seconds for two full finite curves per candidate and mapped density and accuracy checks. CPU and GPU processes used the same cores 0–15 and single-thread BLAS/OMP settings. The independent NumPy reference includes original CPU physics and a direct gnomonic solve. Reusing per-face conditioning avoids repeated SVDs at its sample points. JAX CPU provides a separate compiled comparison. The initial series overlapped some lower-priority CPU accuracy runs on cores 16–19. Final B1/B8 baseline and GPU runs were collected separately from those accuracy runs. B1 independent repeats, exact sample counts, host peak RSS and per-candidate stage values are in the ledger.

For the first changed B1 proposal moment preparation across both stars and both field levels took 0.068 s on the optimized CPU reference and 0.015 s on GPU. The 56 packet interpolation and integration calls took 0.458 s on CPU and 0.054 s on GPU, including required transfers and results. These are nested stage timings from the corresponding complete calls. The common geometry, clipping and packet hashing span was 9.025 s on CPU and 5.023 s on the hybrid GPU path. The latter includes CPU and device work.

The warm B1 CUDA trace spans **5.562 s** and contains **32,813 kernels** and **6,080 copies**. Kernel intervals cover 0.967 s, including 0.779 s in native nested root kernels. Copy intervals cover 0.050 s across 208.7 MB host-to-device, 391.9 MB device-to-host and 37.1 MB device-to-device (decimal MB). Removing all recorded copy duration on the critical path would bound acceleration by 1.009×. Staging, launch and synchronization gaps require separate attribution. The independently marked CPU boundary-reference work occupied 1.800 s, with 0.0000 s of device overlap. Conditional 2× and 4× reductions in its serialized duration yield 1.193× and 1.320× overall. Profiler overhead is included in the trace. The table uses separate unprofiled processes.

The B1 GPU run sampled PCIe Gen 4 ×8 under nonzero utilization. Shared-device memory reached 598 MiB in those unit samples. Ada runs a Ryzen 9 5950X and RTX 3080 Ti with 12 GiB, JAX 0.12.0.1, driver 595.99.02 and FP64. Kernel DRAM traffic, achieved FP64, cache behavior, stalls, exact allocator peaks and the fraction of transfer duration attributable to bandwidth remain unmeasured. The timeline cannot predict a device upgrade's effect. [Per-window telemetry, stage timings and trace](ada_validation/background-m021.json)

## m020 native nested observer roots

The [m020 ledger](ada_validation/background-m020.json) retains matched original
serial CPU/JAX CPU/native GPU B1/B4 comparisons, CPU/GPU B8, the previous GPU
observer path, independent repeats, full scientific arrays and fine joint error
controls. 31 new hardware controls pass, including 57 contact-focused phases;
strict finite error controls still prevent 1e-4 physical fitting/MCMC qualification.
The current port changes both native loop fusion and CPU precision continuation.
The complete observer still serializes candidate/phase topology controllers.

| Proposals | Original CPU cold / warm | JAX CPU cold / warm | Native GPU cold / warm |
| --- | --- | --- | --- |
| 1 | 9.265 / 9.061–9.201 s | 42.112 / 14.725–16.266 s | 28.457 / 3.914–5.219 s |
| 4 | 37.138 / 37.303–37.561 s | 87.390 / 60.561–60.561 s | 41.653 / 15.726–16.894 s |
| 8 | 74.953 / 74.675–74.675 s | not measured | 59.775 / 30.453–30.453 s |

Warm ratios and cold/process amortization are in [Ada progress](ada_progress.md).
Cold and repeated complete-process costs can give smaller gains than warm calls.
Each sample computes two finite field curves and original mapped densities with
all required host outputs, bridges, transfers and synchronization. Sample counts
remain small; these are preliminary distributions. Actual JAX/XLA pool sizes and
a tuned multiprocess baseline for this exact model are unmeasured. Current CPU
baselines share unit affinity 0–15/nice 10; BLAS/OMP/MKL/NumExpr environment counts 1.
Private CPU correctness checks ended before these timing runs. Shared-machine
thermal/CPU exclusivity is not assumed. Active-load link, clocks, temperatures,
power/utilization and sampled memory ranges are retained per unit.

### New complete-call timeline and conditional sensitivities

One changed warm B1 trace spans **5.47454 s**, containing
**31,799 kernels**, **6,112 copies**, and
**121 native nested root kernels** with
**0.779053 s** of interval union.
Kernel union is **0.960386 s**, copy union
**0.0577499 s**, and total recorded device activity union
**1.01885 s**. Interval unions avoid overlap counting;
CUDA API waits overlap device work and must not be added. The remaining
**4.45569 s** includes
host physics/topology/clipping, dispatch, synchronization and gaps without full
attribution. These are profiler experiments; independent unprofiled repeats
provide the official timing control. Kernel count alone does not establish a
critical launch-latency fraction. The prior m019 260,556 kernel count and 56.0% CPU
reference fraction describe the old path; do not apply them here.

Within that remaining host span, original CPU boundary-field references occupy **1.76842 s**,
with **0 s** recorded device overlap.
Their measured serialized fraction is **32.3%**.
With all other work fixed, 2×/4× acceleration of this measured subset gives
**1.1926× / 1.3197×**;
elimination gives a **1.4772×** ceiling.
Other CPU work remains unassigned. The prior kernel union was 1.01386 s and
CPU boundary-reference time 5.38756 s. GPU active time changed much less than
launch count; these separate profiler runs do not isolate causal contributions
of fusion versus CPU continuation. This is conditional stage scaling, not a
forecast for a named CPU or a ceiling after further porting.

Optimistically eliminating every recorded copy nanosecond gives only
**1.0107×** for this range.
That does not isolate bandwidth from latency/staging or account for all host
critical paths. No 2×/4× host-link or GPU-DRAM bandwidth prediction is justified.
DRAM traffic, cache, occupancy/stalls and achieved FP64 counters remain unavailable.
More VRAM/coherence/FP64 or bandwidth are distinct hypotheses, not measured new
processor benefits. No global clock/link/driver/profiling changes were made.

Reproduce each full driver in a separately registered <=300 s GPU unit:

```sh
python scripts/validate_gpu_fused_observer.py --backend gpu --batch 4 --repeats 2 --verified-bounded-gpu-window --output complete.json
python scripts/validate_gpu_fused_observer.py --backend gpu --batch 1 --levels 2 3 --integration-level 3 --depth 2 --slice-order 8 --repeats 1 --verified-bounded-gpu-window --output anchor.json
python scripts/compare_gpu_joint_budget.py --anchor anchor.json --slice-alternative slice.json --depth-alternative depth.json --output joint.json
```

Use reference/cpu for separate original/JAX CPU baselines; use `--engine observer`
for the old GPU execution path. For `slice.json` retain depth 2 but use slice 4;
for `depth.json` retain slice 8 but use depth 1, with the same levels/common grid.
All physical proposals, observations and output contracts must match. The joint
postprocessor is CPU-only. Curated report profiles contain its required inputs.
NVTX wraps the enclosing evaluate and original `_Field.reference`; trace with
cuda,nvtx,osrt and cuda-graph-trace=node, sample/CPU-context-switch collection off.
All raw operational/profiler logs remain private; trace/source hashes and
scientific aggregates are public. The pinned m019 Colab notebook is unchanged;
actual Colab/native GPU execution there remains pending. No 10× analysis/ESS claim.

## Previous snapshot: m019 observer port

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

## m041 — Physical walker scaling

The same sixteen changing proposals ran with eight and sixteen walker processes, one logical CPU per physical core. Each walker used one BLAS/OMP thread. Host and device jobs used nice 19. Each GPU unit could use the same host-CPU mask as the corresponding CPU run. The single owner prepared per-walker fields in a separate stage, with at most three proposals per bounded allocation. The four complete resumed fits ran sequentially, with the backend order reversed at sixteen workers.

| Workers | Field backend | Field allocation and handoff before pool, s | CPU pool and decision unit, s | Eclipse geometry and clipping, summed worker s | Complete resumed fit, s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | CPU | 0.00 | 3258.38 | 17669.21 | 3260.38 |
| 8 | GPU | 553.25 | 3001.28 | 17625.11 | 3556.54 |
| 16 | CPU | 0.00 | 2776.81 | 25247.40 | 2778.81 |
| 16 | GPU | 537.05 | 2520.64 | 25020.43 | 3059.70 |

In the isolated host Roche sampler, the original-to-tiled median time ratio ranged from 1.91× to 5.82× for batches of 1, 16, 256, 2,048 and 32,768 rays on each star. Each median uses three warmed calls. Sampled points, normals and area Jacobians matched bit for bit. The complete fits also include the stationary-shadow solve, adaptive clipping, radiation and inference.

Doubling the worker count changed elapsed time by a factor of 1.173 for CPU fields and 1.162 for GPU fields (eight-worker time divided by sixteen-worker time). GPU/CPU elapsed-time ratios were 1.091 and 1.101. These are single end-to-end runs per configuration. In the sixteen-worker GPU-field run, eclipse geometry and clipping accounted for 62.9% of the summed observer time. Summed stage seconds include concurrent work. In the CPU arm the worker pool built the stellar fields. In the GPU arm the clock includes archive writes, host reads, initialization/probes and unit cleanup.

At fixed worker and initialization costs, eliminating the entire separate GPU field stage would improve this schedule by at most 1.18× at eight workers and 1.21× at sixteen. That stage includes field calculation and bounded-unit overhead as well as data transfer. Faster PCIe alone can address only the transfer portion. The GPU owner prepared one changing system at a time and handed its optical fields to the host. Cross-proposal device batching and overlap with CPU eclipse work remain available for comparison.

During the host pools, median sampled CPU scaling frequencies were 4.49 and 4.34 GHz with CPU fields, and 4.49 and 4.34 GHz with GPU fields, in eight- and sixteen-worker order. Peak observed host-pool memory ranged from 7.5 to 28.6 GiB. Sampling began during the first CPU run. Each GPU run transferred 552.9 MiB of field archives between the field owner and the host walkers. This file size includes host geometry and metadata.

The highest sampled active device memory use was 2370 MiB. Active samples recorded a PCIe 4.0 ×8 link. These runs used a single link configuration. The maximum observed workload across the 13 bounded GPU units was 101.29 s under the 300 s limit. Each verified the original held checkpoint at entry and cleanup.

The independent JAX CPU control took 850.14 s inside a single call on one host core during the correctness checks, for four curves of one proposal. The matched walker times above cover sixteen resolutions for each of sixteen proposals. The original facet fitter exercised the same parameters, uncertainties, nuisance model and original prior with its own finite surface. Detailed numerical and per-density timing results are in the validation file. [Measurements](ada_validation/background-m041.json)

## m042 — Where the host spends time in eclipse

Two new physical draws from the first round were replayed on one CPU core using their own saved GPU-field archives. Each replay first ran an uninstrumented complete orbit, then the same orbit with one deterministic call profile per star and phase. Both passes checked the four field/emitter curves, adaptive phase records and quadrature hashes against the original eight-worker result. These diagnostics ran after the timed samplers. Their clocks start after archive loading and field attachment.

| Seed | Interior / depth | Uninstrumented full orbit, s | Profiled full orbit, s | Profiled packet time, s | Surface sampling | Shadow predicate | Other packet work |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261117 | six / 3 | 170.79 | 173.64 | 147.26 | 34.1% | 60.4% | 5.5% |
| 20261118 | duffy / 4 | 346.59 | 349.53 | 279.10 | 34.8% | 59.6% | 5.6% |

Surface sampling computes the Roche radius, normal and area Jacobian. The shadow predicate tests the projected radius and solves for the stationary potential along candidate lines of sight. The following partition groups the same packet time by the operation that requested those values. Percentages in each row share one denominator.

| Seed | Extra shadow-parent selection | Polygon clipping | Curved-edge slices | Planar quadrature | Remaining packet work | Rays sent to stationary solve / shadow rays |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261117 | 3.6% | 16.3% | 47.6% | 0.6% | 31.9% | 39,531,164 / 46,395,147 (85.2%) |
| 20261118 | 1.9% | 16.1% | 47.5% | 0.7% | 33.8% | 76,816,839 / 88,985,463 (86.3%) |

At fixed downstream cost, eliminating the initial selection of shadow-crossing faces would speed packet construction by at most 1.037× for the first draw and 1.019× for the second. A more substantial target is the repeated stationary-point calculation within adaptive refinement, polygon clipping and curved slices. A fused FP64 implementation can be tested against the existing signed margin, stationary residual, exact adaptive decisions and grazing phases.

Polygon clipping and curved slices also call the FP64 shadow predicate. A change to candidate selection would have to reproduce its signs and adaptive margin tests during both eclipses. The packet profiles retain the foreground component, node counts, surface and stationary-ray counts and elapsed times at each sampled phase. The largest relative component-flux difference over all four replay passes was 0.

| Round | Field backend | Field allocation and handoff before pool, s | Host pool through decision, s | Geometry and clipping, summed worker s | Integration, summed worker s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | CPU | 0.00 | 1643.69 | 8886.97 | 2864.88 |
| 1 | GPU | 275.11 | 1526.64 | 8797.54 | 3140.85 |
| 2 | CPU | 0.00 | 1653.70 | 8871.40 | 3009.95 |
| 2 | GPU | 276.13 | 1524.63 | 8800.61 | 3142.99 |

The CPU workers built fresh fields in memory. A single GPU prepared up to three proposals per unit and handed the synchronized optical archives to the host workers. The first round used CPU fields first and the second reversed the order. The independent JAX CPU control took 754.48 s for four complete curves of one changing physical proposal. Its one-core time covers field preparation, eclipse integration and density evaluation inside the call. The full sampler timed all sixteen resolutions for each in-prior draw. Across 6 registered GPU units the maximum measured workload was 101.25 s. The held Fifth42 checkpoint was verified at every boundary. [Per-phase profiles and timing data](ada_validation/background-m042.json)
