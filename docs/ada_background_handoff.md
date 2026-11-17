# Background ELISa GPU assignment

Paul authorized this continuing background assignment on 2026-11-17 UTC:

> Continue the ELISa GPU work through a complete, scientifically validated light-curve pipeline, then assess integration with fitting/MCMC. Preserve CPU behavior and compare each stage against ELISa on representative systems, including eclipses. Document numerical differences, unsupported cases, transfer-inclusive timings, and remaining work. Commit and push progress at each milestone. You may use Ada for bounded GPU test windows: checkpoint and pause Fifth42 before each window, restore it afterward, and verify durable progress resumes. Stop and report if restoration fails or a scientific discrepancy cannot be resolved. Don’t merge into the main branch without asking.

Development continues on `codex/gpu-resident-roche-pipeline`; no merge into main,
production or the existing benchmark branches is authorized. DOT's entry point
remains [Ada progress](ada_progress.md), with committed source, reproducible
commands and curated numerical/timing evidence at each milestone.

## Execution and reporting

A detached local systemd user service supervises a persistent Codex session and
resumes it between milestones. User-service lingering is already enabled on Ada,
so this does not depend on an open terminal. The supervisor stops on completion,
a publication/runtime blocker, restoration failure or unresolved scientific
 discrepancy; it does not restart blindly after one of those outcomes. Detailed
operational logs and process/checkpoint snapshots stay private on Ada. Public
reports record milestone, tested commit/fingerprint, results, failures, limitations,
production restoration and next scientific step. No elapsed-time promise implies
scientific completion.

The new worker must first verify its checkout, connected GitHub write capability,
current production state and existing evidence, then publish and follow a staged
plan. Prior baseline: commit `dd047a7`, **130 passing CPU/GPU tests with no skips**.
That validates the existing primitives, not a complete changing light curve.

## Scientific scope and next work

The current resident path joins actual bounded Roche rays, supplied triangle
geometry, SI gravity, intrinsic gravity-darkened temperatures and continuous
B/V/bolometric Planck quadrature. The mesh currently has fixed supplied directions
and topology and is **uncorrected**. CPU container build order matters: gravity,
centres and normals may be cached before mesh correction, while later areas and
vertices use corrected state. Band emission is hemispheric surface flux before
optical-depth/LD correction, projected coverage and distance scaling.

1. Reconstruct CPU-compatible mesh correction, topology and dependent-state
   sequencing. Compare the actual CPU stage and full container state; preserve
   unsupported masks and CPU fallback for unsupported domains.
2. Connect fresh geometry, temperature, passband/atmosphere, LD and occultation
   state into complete observed light curves. Validate representative systems,
   unequal components, supported spins/potentials/inclinations, both eclipses,
   grazing/partial/total cases and ingress/egress/contact phases. Declare actual
   coverage of eccentricity, separation, irradiation, spots, atmosphere tables
   and near-contact regimes; do not silently turn unsupported cases into bad fits.
3. Compare component and normalized curves with unmodified ELISa, including
   phase ordering/repeats and physical conservation checks. Preserve CPU behavior,
   units and original tolerances. Separate implementation error from mesh,
   temporal and spectral discretization. Never pair changed geometry with stale
   brightness, coverage or LD state. Any hybrid CPU stage must be identified.
4. Assess fitting/MCMC with fresh changed-parameter predictions, observational
   phase mapping/normalization/noise/third light, numerical log-likelihood error
   relative to observational uncertainty, and representative fitting/sampler
   checks. Cache the fixed physical model for l3/ln_f-only inference. Short chains
   are smoke checks; convergence/ESS claims require sufficient evidence.
5. Measure cold setup/compilation, matched transfer-inclusive forward/likelihood
   work, resident lower bounds and JAX CPU/original CPU comparisons separately.
   Publish unsupported cases and remaining work alongside gains or regressions.

## GPU windows and Fifth42 restoration

The current session authorizes routine **checkpointed** pauses of Ada's Fifth42
for bounded windows; the older handoff's requests for another approval do not
apply to this assignment. No other job may be interrupted. No uncheckpointed
kill, GPU reset, driver/global-setting change, shared-environment modification
or reboot is authorized.

Each allocation re-inspects the exact current Fifth42 service/job/PID, owner,
command, GPU use and durable checkpoint. Reuse the verified supported SIGTERM
checkpoint handler, pinned release/config/plan checks, journal/witness validation
and exact service restart procedure. Do not substitute SIGSTOP or a forced stop.
If a clean safe pause cannot be established, do no GPU work.

Use a private FP64 JAX CUDA environment with GPU memory preallocation disabled
before process startup. CPU development forces JAX CPU while production runs.
Each GPU test/benchmark batch has a five-minute maximum and its own systemd
unit with guaranteed restoration in `ExecStopPost`, including on failed tests,
compilation or timeout. Cancel only the ELISa test processes. A new window may
start only after the previous restoration hook has completed successfully.

Restore the original production service/config/plan and verify a matching resume
record plus **durable advancement beyond the saved baseline**, not just a live
PID. Preserve naturally completed production state. If restoration fails, stop
further development/testing, report exact state/evidence and a safe next action,
and wait for intervention. If a scientific discrepancy cannot be resolved,
retain the failing case and stop/report; do not alter the oracle or loosen
existing tolerances to make it pass.
