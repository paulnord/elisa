# Experimental fused GPU intersection snapshot

Captured 2026-11-17T20:29:20.391649+00:00 from the continuing ELISa GPU work. This is a
preserved source snapshot of m020 in progress, based on `7ad2f9f83ad6621fa1a507520125e03f3fce9490`. Nothing is
merged. The running worker continues its planned validation and publication on
`codex/gpu-resident-roche-pipeline`.

The new opt-in implementation fuses nested Roche/observer root operations using
Pallas/Triton FP64 device loops. Precision-ambiguous roots resume original CPU
decisions from saved brackets. The current curved-intersection approach,
physical equations and scientific guards remain. A separate common-anchor
joint finite field/boundary budget is included as an experimental diagnostic.

The seven new source/test files match the frozen tested source manifest. All
639 earlier runtime source/data files remain byte-identical to the base commit.
The 30 fused/joint hardware controls and a separate contact-focused full-curve
control passed, without skips. This is 31 executed controls, not a rerun of every
historical test. See the [source manifest](ada_validation/fusion-snapshot-20261117-source-hashes.json)
and [curated test/timing evidence](ada_validation/fusion-snapshot-20261117.json).

## Preliminary complete comparison costs

Each four-proposal batch varies physical parameters, computes two finite field
curves per candidate, and includes CPU bridges, transfers, synchronization,
required host outputs, original mapping/density and accuracy diagnostics.

| Path | First call, including encountered compilation | Two changed warm calls |
| --- | --- | --- |
| Original serial CPU | 37.138 s | 37.303, 37.561 s |
| Fused GPU | 41.653 s | 16.894, 15.726 s |

Warm median ratio: **2.295x**, for this finite comparison workload. These are
few shared-machine samples against a serial baseline, not a complete scientific
analysis, fit or MCMC speedup. Cold compilation can erase the warm benefit.
Broader batch, accuracy and profiling studies are still in progress.

## Reproduction and scope

Use the pinned JAX/CUDA environment described in
[the backend documentation](experimental_gpu_backend.md). The native optional
path needs the CUDA Pallas/Triton backend. On Ada, GPU invocations belong inside
the existing registered bounded allocation procedure; these scripts do not
manage production or allocate GPU windows themselves.

```sh
python scripts/validate_gpu_fused_observer.py --help
python scripts/compare_gpu_joint_budget.py --help
```

The [existing notebook](../notebooks/elisa_observer_batches.ipynb) remains pinned
to its previously validated observer implementation. Its Ada execution does not
validate this new fusion snapshot or establish actual Colab provisioning.

CPU defaults and priors remain intact. CPU topology/clipping, precision checks,
residual root work and LD/table bridges remain. Existing finite-resolution
accuracy failures are retained: physical fits, converged posteriors, MCMC and
effective samples per second remain unqualified. No tenfold gain is claimed.
