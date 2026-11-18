# Direct cuBLAS test in the native ELISa likelihood on Ada

The [native mesh study](ada_native_mesh_refinement.md) found that constructing blackbody spectra consumed 55.5% of a profiled 1.5° prediction. Here a CUDA double-precision kernel computes the same Planck spectrum and NVIDIA cuBLAS integrates it over the B, V and bolometric passbands. The surrounding code still constructs both Roche surfaces, performs both reflection iterations and eclipses, maps the same observations, and runs the original prior, likelihood and sampler.

The fastest observed complete two-step ensemble run took **74.78 s** using **CUDA math + cuBLAS with 12 walker workers**. Each run used the same 32 walkers, two red-blue steps, six sampled parameters, FP64 arithmetic, initial ensemble, seed 814 and 28 synthetic observations. Two steps yielded 64 proposed moves. The initial 32 likelihoods are also counted in the throughput denominator. The final summary curve is timed. The [native mesh](ada_native_mesh_refinement.md) missed its joint light-curve accuracy limit at all 48 examined positions. This comparison measures the cost of this finite model. A converged scientific chain and effective samples per second remain unmeasured.

## Four changing forward likelihoods per fresh process

Each implementation saw the same twelve distinct proposals, in three groups of four. The three methods rotated through all three run-order positions. The first batched Simpson group took 9.42 s. Its second group took 5.35 s. To measure sensitivity to startup and shared caches, we repeated the entire matrix with fresh processes in the reverse rotation after all methods had run. Spectral setup, device initialization, transfers, synchronization, fitter setup and likelihoods are inside each application clock. Complete subprocess wall times, including interpreter startup, are in the data.

| Implementation | First sweep for 12, s | Repeat for 12, s | Repeat likelihoods/s | Largest repeat flux difference | Largest repeat log-posterior difference |
|---|---:|---:|---:|---:|---:|
| Batched CPU Simpson | 20.127 | 16.048 | 0.748 | 0 | 0 |
| CPU matrix product | 14.924 | 14.954 | 0.802 | 4.44e-16 | 6.24e-11 |
| CUDA math + cuBLAS | 14.629 | 14.079 | 0.852 | 4.44e-16 | 5.85e-11 |

The CPU sampler used the faster CPU implementation that passed the checks, **CPU matrix product**. Both matrix-product paths use the same unrounded temperatures, union of native wavelengths, passband responses and Simpson quadrature weights. Each reevaluates the response signature before reusing a fixed quadrature. Both compute FP64 Planck values in blocks of at most 256 temperatures. CUDA builds a dense spectrum and calls the actual double-precision DGEMM in pedantic math mode. It synchronously returns every integrated passband value before the native observer continues. [CUDA implementation](../scripts/native_cublas_spectra.cu) · [NVIDIA cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/) · [NVIDIA CUDA math documentation](https://docs.nvidia.com/cuda/cuda-math-api/)

The independent face check ran on two proposals per implementation against the original CPU radiances. Tolerances were 3e-13 relative plus 1e-10 absolute per face. The mapped flux limit was 2e-12 absolute plus 3e-13 relative, and the absolute log-posterior limit was 1e-6.

| Implementation | Face values checked | Maximum relative error | Passed |
|---|---:|---:|:---:|
| Batched CPU Simpson | 58,386 | 0 | yes |
| CPU matrix product | 58,386 | 2.61e-15 | yes |
| CUDA math + cuBLAS | 58,386 | 2.33e-15 | yes |

## Complete MCMC fits

The original fitter parallelized walker likelihoods with `NUMBER_OF_MCMC_PROCESSES=N`. Each worker used one ELISa and numerical-library thread, and `NUMBER_OF_PROCESSES=1`. Every configuration ran separately. The hybrid sends exact temperatures from each worker to one independent CUDA owner and receives its face radiances. The CPU workers use logical CPUs 0 through N−1, one per physical core. The GPU owner uses logical CPU 16, the SMT sibling of core 0. Reported complete time includes owner startup and shutdown, IPC, device transfers and synchronization, native sampler work, final fit summaries, and trace collection. The process-level wall time additionally includes Python startup.

| Walker workers | Improved CPU, s | CPU + cuBLAS, s | Improved CPU, likelihoods/s | CPU + cuBLAS, likelihoods/s | CPU time / hybrid time | Repetitions CPU / GPU |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 145.81 | 129.30 | 0.66 | 0.74 | 1.13 | 1 / 1 |
| 4 | 87.26 | 78.49 | 1.10 | 1.22 | 1.11 | 1 / 1 |
| 8 | 86.68 | 77.01 | 1.11 | 1.25 | 1.13 | 2 / 2 |
| 12 | 83.67 | 74.78 | 1.15 | 1.28 | 1.12 | 1 / 1 |
| 16 | 85.52 | 75.81 | 1.12 | 1.27 | 1.13 | 2 / 2 |

The native fitter serially validated 32 supplied starting walkers before opening the walker pool. This startup is paid once per fit and takes a larger share in a two-step run. The clock below covers the sampler’s 96 likelihood evaluations after that validation, including emcee’s initial evaluation of all 32 walkers. Repeated measurements are medians.

| Workers | Improved CPU sampler, s | Hybrid sampler, s | Improved CPU sampler likelihoods/s | Hybrid sampler likelihoods/s |
|---:|---:|---:|---:|---:|
| 1 | 109.99 | 96.58 | 0.87 | 0.99 |
| 4 | 51.18 | 45.68 | 1.88 | 2.10 |
| 8 | 51.03 | 44.45 | 1.88 | 2.16 |
| 12 | 47.83 | 42.03 | 2.01 | 2.28 |
| 16 | 49.59 | 43.03 | 1.94 | 2.23 |

Timings are medians where repetitions were collected. Individual measurements, wall-clock boundaries and worker allocations are in the results. The unchanged CPU spectral calculation also ran on this exact finer sampler workload, with 8 workers in 136.27 s (71.51 s inside the sampler), 16 workers in 133.01 s (70.48 s inside the sampler). All timings here use the same two-step fit. The [published 12° MCMC matrix](ada_mcmc_scaling.md) used a different mesh and chain length.

The original reference accepted 36 of 64 moves. Its first 32 walker and 16 proposal fluxes and log posteriors reproduced the separate native mesh study. **16 of 16** complete fits met the trajectory and density checks. These checks compare all attempted coordinates, the two-step chain, each walker’s acceptance count, every flux, retained densities and every posterior evaluation. The peak absolute deviations are recorded per run. Verified GPU fits executed 5,031–5,031 cuBLAS calls each. Each GPU fit reports actual owner affinity, temperature and radiance bytes, device and library version. [Raw timings, hardware and numerical checks](ada_validation/cublas-native-spectra.json)

## Hardware and reproduction

Ada has one Ryzen 9 5950X with 16 physical cores and 32 logical threads. The measured GPU was NVIDIA GeForce RTX 3080 Ti, 12288 MiB, 595.99.02. Its runtime reported CUDA driver API 13020, runtime 13040 and cuBLAS 130800. The build used Build cuda_13.4.r13.4/compiler.38855100_0. All 37 registered cases recorded a Fifth42 checkpoint verification on exit. The original MCMC configuration and ELISa data setup are documented in the [scaling report](ada_mcmc_scaling.md).

Run these commands in the published checkout, with the same Python environment and configured ELISa data tables. The CUDA toolkit path and physical CPU IDs below are for Ada. Configure `nvcc` and the target architecture for another device. Each row uses a new process and a new output stem.

```bash
fixture=docs/ada_validation/mcmc-scaling-fixture.json
out=$(mktemp -d)
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
toolkit=/usr/local/cuda-13.4
"$toolkit/bin/nvcc" -ccbin /usr/bin/g++-15 -std=c++17 -O3 -arch=sm_86 -shared --cudart shared -Xcompiler=-fPIC "-Xlinker=-rpath=$toolkit/lib64" scripts/native_cublas_spectra.cu -lcublas -o "$out/native.so"
for i in 0 1 2; do
  case "$i" in
    0) modes="blocked cpu-dgemm cuda-dgemm" ;;
    1) modes="cpu-dgemm cuda-dgemm blocked" ;;
    2) modes="cuda-dgemm blocked cpu-dgemm" ;;
  esac
  for b in $modes; do
    python -m scripts.benchmark_cublas_native_spectra curves "$b" "$fixture" "$out/curves-$i-$b-a1.json" --library "$out/native.so" --cpus 8 --begin "$((33+4*i))" --end "$((37+4*i))"
  done
done
for b in blocked cpu-dgemm cuda-dgemm; do
  python -m scripts.benchmark_cublas_native_spectra curves "$b" "$fixture" "$out/face-$b-a1.json" --library "$out/native.so" --cpus 8 --begin 33 --end 35 --face-check
done
python -m scripts.benchmark_cublas_native_spectra chain native "$fixture" "$out/chain-reference-native-8-a1.json" --library "$out/native.so" --cpus 0,1,2,3,4,5,6,7 --owner-cpu 16 --steps 2
python -m scripts.benchmark_cublas_native_spectra chain native "$fixture" "$out/chain-reference-native-16-a1.json" --library "$out/native.so" --cpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --owner-cpu 16 --steps 2
python -m scripts.benchmark_cublas_native_spectra chain cpu-dgemm "$fixture" "$out/chain-r1-cpu-dgemm-1-a1.json" --library "$out/native.so" --cpus 0 --owner-cpu 16 --steps 2
python -m scripts.benchmark_cublas_native_spectra chain cuda-dgemm "$fixture" "$out/chain-r1-cuda-dgemm-1-a1.json" --library "$out/native.so" --cpus 0 --owner-cpu 16 --steps 2
python -m scripts.benchmark_cublas_native_spectra chain cuda-dgemm "$fixture" "$out/chain-r1-cuda-dgemm-4-a1.json" --library "$out/native.so" --cpus 0,1,2,3 --owner-cpu 16 --steps 2
python -m scripts.benchmark_cublas_native_spectra chain cpu-dgemm "$fixture" "$out/chain-r1-cpu-dgemm-4-a1.json" --library "$out/native.so" --cpus 0,1,2,3 --owner-cpu 16 --steps 2
python -m scripts.benchmark_cublas_native_spectra chain cpu-dgemm "$fixture" "$out/chain-r1-cpu-dgemm-8-a1.json" --library "$out/native.so" --cpus 0,1,2,3,4,5,6,7 --owner-cpu 16 --steps 2
python -m scripts.benchmark_cublas_native_spectra chain cuda-dgemm "$fixture" "$out/chain-r1-cuda-dgemm-8-a1.json" --library "$out/native.so" --cpus 0,1,2,3,4,5,6,7 --owner-cpu 16 --steps 2
python -m scripts.benchmark_cublas_native_spectra chain cuda-dgemm "$fixture" "$out/chain-r1-cuda-dgemm-12-a1.json" --library "$out/native.so" --cpus 0,1,2,3,4,5,6,7,8,9,10,11 --owner-cpu 16 --steps 2
python -m scripts.benchmark_cublas_native_spectra chain cpu-dgemm "$fixture" "$out/chain-r1-cpu-dgemm-12-a1.json" --library "$out/native.so" --cpus 0,1,2,3,4,5,6,7,8,9,10,11 --owner-cpu 16 --steps 2
python -m scripts.benchmark_cublas_native_spectra chain cpu-dgemm "$fixture" "$out/chain-r1-cpu-dgemm-16-a1.json" --library "$out/native.so" --cpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --owner-cpu 16 --steps 2
python -m scripts.benchmark_cublas_native_spectra chain cuda-dgemm "$fixture" "$out/chain-r1-cuda-dgemm-16-a1.json" --library "$out/native.so" --cpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --owner-cpu 16 --steps 2
python -m scripts.benchmark_cublas_native_spectra chain cuda-dgemm "$fixture" "$out/chain-r2-cuda-dgemm-16-a1.json" --library "$out/native.so" --cpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --owner-cpu 16 --steps 2
python -m scripts.benchmark_cublas_native_spectra chain cpu-dgemm "$fixture" "$out/chain-r2-cpu-dgemm-16-a1.json" --library "$out/native.so" --cpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --owner-cpu 16 --steps 2
python -m scripts.benchmark_cublas_native_spectra chain cuda-dgemm "$fixture" "$out/chain-r2-cuda-dgemm-8-a1.json" --library "$out/native.so" --cpus 0,1,2,3,4,5,6,7 --owner-cpu 16 --steps 2
python -m scripts.benchmark_cublas_native_spectra chain cpu-dgemm "$fixture" "$out/chain-r2-cpu-dgemm-8-a1.json" --library "$out/native.so" --cpus 0,1,2,3,4,5,6,7 --owner-cpu 16 --steps 2
for i in 0 1 2; do
  case "$i" in
    0) modes="cuda-dgemm cpu-dgemm blocked" ;;
    1) modes="cpu-dgemm blocked cuda-dgemm" ;;
    2) modes="blocked cuda-dgemm cpu-dgemm" ;;
  esac
  for b in $modes; do
    python -m scripts.benchmark_cublas_native_spectra curves "$b" "$fixture" "$out/warm-$i-$b-a1.json" --library "$out/native.so" --cpus 8 --begin "$((33+4*i))" --end "$((37+4*i))"
  done
done
python -m scripts.summarize_cublas_native_spectra "$out" docs/ada_validation/native-mesh-refinement.json "$out/summary.json"
```

[Benchmark driver](../scripts/benchmark_cublas_native_spectra.py) · [Analysis and parity checks](../scripts/summarize_cublas_native_spectra.py)
