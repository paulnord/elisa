# Pinned ELISa observer handoff

[Open the notebook](elisa_observer_batches.ipynb), or [run it in Colab](https://colab.research.google.com/github/paulnord/elisa/blob/codex/gpu-resident-roche-pipeline/notebooks/elisa_observer_batches.ipynb).

It pins scientific code commit `a3a83b6b4ee69a8244f772a03e19a8e1c669d408` and
[55 runtime dependency versions](observer-runtime-pins.txt). It identifies actual
CPU/GPU/runtime and FP64 settings, compares complete changing-physics original
CPU/JAX CPU/GPU batches, retains cold/warm/setup costs and the original parity
and finite-accuracy stops, and downloads an evidence archive.

[Ada execution evidence](../docs/ada_validation/colab-observer-ada.json) records
where its cells were tested. Real Colab execution and network provisioning remain
pending. No 10× complete-analysis result, qualified 1e-4 physical fit, MCMC or ESS gain
is claimed. The notebook reports actual results; a finite accuracy failure stops
scientific acceptance. See its opening scope statement before using the model.

On Ada, every GPU execution must stay inside the registered hold-aware bounded
unit policy. The notebook does not manage production. Its Colab path uses an
isolated virtual environment and versioned repository assets, with no private
inputs or changes to shared Colab packages.
