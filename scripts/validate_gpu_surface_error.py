"""Nested complete surface/contact error and physical likelihood assessment.

GPU commands require an independent bounded checkpoint/restore allocation.
The reported envelopes bound measured finite-grid changes, never continuum
error. No fitting backend, prior exclusion or CPU fallback is introduced.
"""

# ruff: noqa: INP001, S101, T201, SLF001 - standalone scientific diagnostic
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import Observer, settings
from elisa.analytics.models import cost_fns, lc
from elisa.analytics.params import parameters
from elisa.analytics.tools.utils import time_layer_resolver
from elisa.experimental.gpu_nested_mesh import NestedMesh
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.gpu_reference import observe
from elisa.experimental.gpu_surface_error import NestedBinaryLightCurve, compare_surfaces
from scripts.validate_gpu_atmosphere import BANDS, table_settings
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_binary import contact_phases, fixture
from scripts.validate_gpu_fit import compare, make_problem, normalized_vector, physical, prepare_fit


def asset_revision() -> str:
    """Hash the fixed versioned atmosphere/LD/mesh/passband resources."""
    root = Path(__file__).resolve().parents[1]
    paths = list((root/"tests/data/ck04").rglob("*.csv"))
    paths += list((root/"tests/data/light_curves/limbdarkening").glob("*.csv"))
    paths += list((root/"src/elisa/data/mesh_corrections").glob("*.npy"))
    paths += [root/"src/elisa/passband"/f"Generic.Bessell.{b}.csv" for b in ["B", "V"]]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def assert_envelope(report: dict[str, Any]) -> None:
    for row in report["bands"].values():
        assert np.all(np.abs(row["actual_change"]) <= row["component_envelope"])
        assert np.all(np.abs(row["actual_change"]) <= row["finite_envelope"])
        assert row["reconstruction_max_abs"] <= row["roundoff_guard_max"]
    assert not report["continuum_certified"]
    if report["finite_log_likelihood_absolute_bound"] is not None:
        assert abs(report["joint"]["delta_log_likelihood"]) <= report["finite_log_likelihood_absolute_bound"]


def curve_assessment(device: str, revision: str) -> list[dict[str, Any]]:
    """Nested grid controls, full coupled physics and boundary attribution."""
    rows = []
    selected = ["cpu", "gpu"] if device == "gpu" else ["cpu"]
    for name, inc, atlas in [("partial_bb", 85, "bb"), ("total_bb", 90, "bb"), ("partial_ck04", 88, "ck04")]:
        binary = fixture({"inclination": inc, "discretization": 12}) if atlas == "bb" else table_fixture(
            {"inclination": inc},
        )
        phases = contact_phases(binary)
        observer = Observer(passband=list(BANDS), system=binary)
        snapshots = {d: [] for d in selected}
        levels = []
        for level in range(4):
            mesh = NestedMesh(level)
            t = perf_counter()
            expected = observe(observer, phases, mesh=mesh)[1]
            reference_seconds = perf_counter()-t
            row = {"levels": level, "reference_cpu_seconds": reference_seconds, "evaluations": {}}
            if not level:
                original = observer.lc(phases=phases, normalize=True)[1]
                row["level_zero_vs_unmodified_cpu"] = {b: compare(expected[b], original[b]) for b in BANDS}
            for d in selected:
                engine = NestedBinaryLightCurve(list(BANDS), mesh, device=d,
                                               radiance="device" if atlas == "bb" else "cpu")
                t = perf_counter()
                snapshot = engine.snapshot(binary, phases, observer=observer, source_revision=revision)
                total = perf_counter()-t
                snapshots[d].append(snapshot)
                row["evaluations"][d] = {
                    "face_counts": snapshot.face_counts, "complete_snapshot_seconds": total,
                    "curve_seconds": snapshot.curve_seconds,
                    "group_payload_bytes": sum(v.nbytes for g in snapshot.groups.values() for v in g),
                    "parity": {b: compare(snapshot.fluxes[b], expected[b]) for b in BANDS},
                    "fluxes": {b: v.tolist() for b, v in snapshot.fluxes.items()},
                }
            levels.append(row)
        differences = {}
        for d in selected:
            differences[d] = [compare_surfaces(snapshots[d][i-1], snapshots[d][i], 1e-4, ln_f=-2.)
                              for i in range(1, 4)]
            for report in differences[d]:
                assert_envelope(report)
        # Withheld native1.5deg triangulation is a different finite grid, not a
        # parity oracle or an exact continuum model. Keep its discrepancy separate.
        binary.primary.discretization_factor = binary.secondary.discretization_factor = np.radians(1.5)
        heldout = observe(Observer(passband=list(BANDS), system=binary), phases)[1]
        from scripts.validate_gpu_accuracy import audit  # noqa: PLC0415

        cross_grid = audit(dict(snapshots[selected[-1]][-1].fluxes), heldout)
        rows.append({"case": name, "phases": phases.tolist(), "levels": levels, "finite_changes": differences,
                     "withheld_native_1p5deg_comparison": cross_grid,
                     "source_revision": revision,
                     "limit": "All levels use fresh full physics; envelopes enclose measured nested differences only. "
                     "Withheld different triangulation is a numerical comparison, not continuum truth."})
    return rows


def mapped(snapshot: Any, fit: Any) -> dict[str, np.ndarray]:
    result = {b: snapshot.fluxes[b][fit.x_data_reducer[b]] for b in fit.y_data}
    return {b: v/v.mean() for b, v in result.items()}


def observation_assessment(device: str, revision: str) -> list[dict[str, Any]]:
    """Original physical serialization, JD/phase mapping, means, noise and priors."""
    rows = []
    selected = ["cpu", "gpu"] if device == "gpu" else ["cpu"]
    for jd in [False, True]:
        problem = make_problem(jd=jd, free_nuisance=True, shape_parameters=True)
        fit = prepare_fit(problem, phase_mode="observations", mesh_mode="paired")
        base = normalized_vector(fit)
        engines = {d: {i: NestedBinaryLightCurve(list(fit.y_data), NestedMesh(i), device=d) for i in [2, 3]}
                   for d in selected}
        candidates = []
        # One JD proposal and two phase proposals; all are fresh physical models.
        proposals = [base] if jd else [base, np.linspace(.25, .7, len(base))]
        for vector in proposals:
            props = physical(fit, vector)
            phases, values = time_layer_resolver(fit.x_data_reduced, **props)
            extended = parameters.extend_json_with_atm_params(values, atmosphere_model=fit.atmosphere_model,
                                                             limb_darkening_coefficients=fit.limb_darkening_coefficients)
            extended["secondary@discretization_factor"] = fit.discretization
            binary = lc.prepare_binary(discretization=fit.discretization, **extended)
            ln_f = props["nuisance@ln_f"]
            prior = fit.ln_prior(vector, fit.norm_init_vector, fit.sigmas)
            row = {"physical": props, "prior": float(prior), "evaluations": {}, "reference": {}}
            expected = {}
            for level in [2, 3]:
                def synthesis(phases: Any, discretization: float, observer: Any, *, _level: int = level,
                              **kwargs: Any) -> dict[str, np.ndarray]:
                    observer._system = lc.prepare_binary(discretization=discretization, **kwargs)
                    return observe(observer, phases, mesh=NestedMesh(_level))[1]

                t = perf_counter()
                expected[level] = fit.predict_light_curves(props, synthesis=synthesis)
                density = cost_fns.likelihood_fn(fit.y_data, fit.y_err, expected[level], ln_f)+prior
                row["reference"][str(level)] = {"complete_prediction_and_density_seconds": perf_counter()-t,
                                                "log_posterior": float(density)}
            for d in selected:
                snapshots = []
                evaluations = []
                for level in [2, 3]:
                    t = perf_counter()
                    snapshot = engines[d][level].snapshot(binary, phases, observer=fit.observer,
                                                          source_revision=revision)
                    prediction = mapped(snapshot, fit)
                    density = cost_fns.likelihood_fn(fit.y_data, fit.y_err, prediction, ln_f)+prior
                    elapsed = perf_counter()-t
                    evaluations.append({"level": level, "complete_snapshot_mapping_density_seconds": elapsed,
                                        "prediction_parity": {b: compare(prediction[b], expected[level][b])
                                                              for b in prediction},
                                        "posterior_parity": compare(
                                            density, row["reference"][str(level)]["log_posterior"], atol=1e-6, rtol=0,
                                        )})
                    snapshots.append(snapshot)
                report = compare_surfaces(*snapshots, fit.y_err, reducers=fit.x_data_reducer, mean_normalize=True,
                                          ln_f=ln_f, observed=fit.y_data)
                assert_envelope(report)
                exact = row["reference"]["3"]["log_posterior"]-row["reference"]["2"]["log_posterior"]
                parity = compare(report["joint"]["delta_log_likelihood"], exact, atol=1e-6, rtol=0)
                stopped = False
                try:
                    compare_surfaces(*snapshots, fit.y_err, reducers=fit.x_data_reducer, mean_normalize=True,
                                     ln_f=ln_f, observed=fit.y_data, require_budget=True)
                except ResolutionNotQualifiedError:
                    stopped = True
                row["evaluations"][d] = {"models": evaluations, "error_estimate": report,
                                          "likelihood_change_parity": parity, "finite_budget_stop": stopped}
            candidates.append(row)
        excluded = fit.ln_prior(np.full(len(base), -.1), fit.norm_init_vector, fit.sigmas)
        assert np.isneginf(excluded)
        rows.append({"jd": jd, "candidates": candidates, "original_out_of_prior_control": "-inf, no nested evaluation",
                     "limit": "Standalone numerical preflight/likelihood-error study outside legacy penalty handlers. "
                     "No new public fitting backend or sampler convergence claim."})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--part", choices=["curves", "observations"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    revision = asset_revision()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        result = curve_assessment(args.device, revision) if args.part == "curves" else observation_assessment(
            args.device, revision,
        )
    report = {"part": args.part, "device": args.device, "source_revision": revision, "results": result,
              "all_implementation_parity_gates_passed": True, "wall_seconds": perf_counter()-started,
              "contract": "Complete curves/snapshots and observation likelihoods, including CPU projection, fresh "
              "state/irradiation/LD/radiance/coverage, transfers and parent grouping. Snapshot output is larger than "
              "ordinary curve output; compare equal contracts. Finite changes/envelopes are not continuum bounds."}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    keys = ["part", "device", "all_implementation_parity_gates_passed", "wall_seconds"]
    print(json.dumps({k: report[k] for k in keys}))


if __name__ == "__main__":
    main()
