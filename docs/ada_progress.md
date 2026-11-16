# Ada GPU development progress

Updated **2026-11-16T21:53:39.725211+00:00**. Work in progress on
`codex/gpu-resident-roche-pipeline`, based on `experimental/gpu-roche-rays`
commit `7d719ca09545d4c62a2e7e30f5e858e6c0d82eec`. No merge into the benchmark or
production branches is intended.

## Current work

- Effective surface gravity on changing Roche vertices, preserving secondary
  coordinates, primary-mass SI scaling and CPU pre-correction build order.
- A device-resident chain of Roche rays, triangle geometry/projection and
  gravity. Directions/topology upload once; proposal parameters upload per batch.
  Stage barriers preserve FP64 rounding while intermediate arrays stay on device.
- Fresh intrinsic gravity-darkened temperatures with bolometric energy closure.
  This includes emitted surface flux, not observed light or an atmosphere model.
- A bounded benchmark comparing the resident path with the vertex host bridge,
  original ELISa CPU helpers, and batched JAX CPU. Matched final outputs are
  downloaded; cold compilation and resident lower bounds are separate.

## Test state

| Check | Result |
|---|---|
| Current full CPU-only suite | **71 passed; 41 GPU cases intentionally skipped** |
| Current thermal + pipeline CPU checks | **15 passed; 13 GPU cases intentionally skipped** |
| Last hardware run, earlier source state | **97 passed; no skips** |
| Current revised pipeline + thermal hardware retest | **Pending** |
| Ruff and whitespace checks | Passed for the implementation |

The larger CPU benchmark found one projection difference just outside the
unchanged near-limb tolerance (about `1.57e-14` absolute). The batch stopped
and Fifth42 was restored. Preserving stage rounding boundaries passed subsequent
CPU checks at 128 and 512 proposals for both stars. A regression test now covers
that case. The intrinsic-temperature CPU benchmark passed all five batch sizes.
No new GPU speedup is claimed before the revised hardware run.

## Production and resources

RTX 3080 Ti 12 GiB, driver 595.99.02, JAX/jaxlib 0.12.0.1, FP64. GPU preallocation
is disabled. Each GPU test window is bounded to five minutes, with checkpointed
Fifth42 stop and guaranteed restoration on success or failure. CPU development
runs while production continues.

The last restoration verified job `j00872` advancing
from cursor `935771216` to
`935822416`. The other machines were not paused.

## Review priorities for DOT

Review the scientific state/build-order contracts, preservation of unsupported
masks, secondary gravity units, the temperature reference-gravity cancellation,
energy conservation, and the final-transfer benchmark accounting. Candidate-specific
angular placement/topology, mesh correction, irradiation/reflection, atmosphere/LD
updates, changing eclipse coverage and full normalized-light-curve/posterior
validation remain unfinished. These experiments must not reuse stale coverage
or brightness for changed physical parameters.

Evidence: [machine-readable progress](ada_validation/progress.json),
[current CPU output](ada_validation/cpu-current-tests.txt),
[prior GPU output](ada_validation/window-01-tests.txt), and
[current source fingerprints](ada_validation/source-hashes.json).

I will update this report after each validation cycle. The existing
`experimental/gpu-backend` benchmark branch remains untouched.
