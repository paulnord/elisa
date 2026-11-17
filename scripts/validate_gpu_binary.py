"""Validate and time complete changing HYBRID light curves against ELISa.

Run GPU only in an authorized bounded allocation, preallocation disabled.
This script neither pauses nor restores production; the independent supervisor
must establish that window. All reference physics uses unmodified ELISa.
"""

# ruff: noqa: INP001, T201, PLR2004, S101 - standalone scientific validation driver
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.testing import assert_allclose

from elisa import BinarySystem, Observer, Star, settings
from elisa.analytics.models.cost_fns import prepare_observations, prepared_likelihood
from elisa.base.curves.utils import calculate_surface_element_fluxes
from elisa.binary_system.curves.c_router import prep_initial_system
from elisa.binary_system.curves.utils import prep_surface_params
from elisa.binary_system.surface.coverage import compute_surface_coverage
from elisa.binary_system.utils import move_sys_onpos
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from elisa.experimental.gpu_surface import SurfaceFluxIntegrator

BANDS = ["Generic.Bessell.B", "Generic.Bessell.V", "bolometric"]
CASES = [
    {"name": "non_eclipsing", "inclination": 60, "reflection": False},
    {"name": "grazing", "inclination": 79, "reflection": False},
    {"name": "partial_reflection", "inclination": 85, "reflection": True},
    {"name": "total_reflection", "inclination": 90, "reflection": True},
    {"name": "q_half_logarithmic", "inclination": 88, "q": 0.5,
     "spin": [0.5, 1.5], "law": "logarithmic", "reflection": False},
    {"name": "q_two_square_root", "inclination": 87, "q": 2,
     "spin": [1.5, 0.5], "potentials": [12, 16], "temperatures": [7500, 4800],
     "discretization": 8, "law": "square_root", "reflection": False},
    {"name": "equal_components", "inclination": 88, "q": 1,
     "potentials": [12, 12], "temperatures": [6500, 6500], "reflection": False},
    {"name": "small_occulter", "inclination": 90, "potentials": [8, 100], "reflection": False},
]


def fixture(case: dict[str, Any]) -> BinarySystem:
    q, spin = case.get("q", 0.7), case.get("spin", [1, 1])
    potentials, temperatures = case.get("potentials", [8, 12]), case.get("temperatures", [6500, 5000])
    coeff = [0.5] if case.get("law", "linear") == "linear" else [0.3, 0.2]
    common = {"albedo": 0.5, "gravity_darkening": 0.32, "atmosphere": "bb",
              "limb_darkening_coefficients": dict.fromkeys(BANDS, coeff),
              "discretization_factor": case.get("discretization", 10)}
    return BinarySystem(
        Star(mass=1.2, t_eff=temperatures[0], surface_potential=potentials[0], synchronicity=spin[0], **common),
        Star(mass=1.2 * q, t_eff=temperatures[1], surface_potential=potentials[1], synchronicity=spin[1], **common),
        inclination=case["inclination"], eccentricity=0, period=3, argument_of_periastron=90,
        phase_shift=0.017, additional_light=0.2, distance=25,
    )


def contact_phases(binary: BinarySystem) -> np.ndarray:
    """Contact-focused estimates plus both half-orbits, repeats and reordering."""
    radii = [binary.primary.side_radius + binary.secondary.side_radius,
             abs(binary.primary.side_radius - binary.secondary.side_radius)]
    cosine, sine = np.cos(binary.inclination), np.sin(binary.inclination)
    contacts = [np.arcsin(np.sqrt((r**2 - cosine**2) / sine**2)) / (2 * np.pi)
                for r in radii if r > abs(cosine)]
    phases = list(np.linspace(0, 1, 41))
    for contact in contacts:
        for centre in (0, 0.5):
            for sign in (-1, 1):
                phases.extend(centre + sign * contact + np.array([-1e-5, 0, 1e-5]))
    return np.concatenate((np.array(phases)[::-1], [1.0, -0.5, 0.01, 1.005])) + binary.phase_shift


def measurement(actual: Any, reference: Any, *, rtol: float = 3e-13, atol: float = 0) -> dict[str, Any]:
    a, b = np.broadcast_arrays(np.asarray(actual), np.asarray(reference))
    assert_allclose(a, b, rtol=rtol, atol=atol)
    nonzero = b != 0
    return {"elements": a.size, "max_abs": float(np.max(np.abs(a - b), initial=0)),
            "max_relative_nonzero": float(np.max(np.abs((a[nonzero] - b[nonzero]) / b[nonzero]), initial=0)),
            "rtol": rtol, "atol": atol}


def details(result: Any, binary: BinarySystem, engine: HybridBinaryLightCurve) -> dict[str, Any]:
    """Compare complete CPU state and every new phase's coverage/face flux."""
    cpu = prep_initial_system(binary)
    observer = Observer(passband=BANDS, system=binary)
    prep_surface_params(cpu, return_values=False, write_to_containers=True, passband=observer.passband,
                        left_bandwidth=observer.left_bandwidth, right_bandwidth=observer.right_bandwidth)
    output: dict[str, Any] = {"state": {}, "phases": [], "intrinsic_energy_ratio": {}}
    for name in ("primary", "secondary"):
        a, b = getattr(result.diagnostics["initial_system"], name), getattr(cpu, name)
        output["state"][name] = {
            key: measurement(getattr(a, key), getattr(b, key), atol=64 * np.finfo(float).eps)
            for key in ("points", "face_centres", "normals", "areas", "velocities",
                        "potential_gradient_magnitudes", "log_g", "temperatures")
        }
        output["state"][name]["faces_identical"] = bool(np.array_equal(a.faces, b.faces))
        if not settings.REFLECTION_EFFECT:
            ratio = np.sum(a.areas * a.temperatures**4) / (np.sum(a.areas) * a.t_eff**4)
            measurement(ratio, 1.0)
            output["intrinsic_energy_ratio"][name] = float(ratio)
    integrator = SurfaceFluxIntegrator(settings.LIMB_DARKENING_LAW, device=engine.device.platform)
    for index, on_pos in enumerate(result.diagnostics["positions"]):
        reference = move_sys_onpos(cpu, on_pos.position, on_copy=True)
        compute_surface_coverage(reference, binary.semi_major_axis,
                                 in_eclipse=bool(result.diagnostics["in_eclipse"][index]),
                                 return_values=False, write_to_containers=True)
        row: dict[str, Any] = {"phase": float(on_pos.position.phase), "components": {}}
        for name in ("primary", "secondary"):
            star, expected = getattr(on_pos, name), getattr(reference, name)
            eps = 64 * np.finfo(float).eps
            differing = np.setxor1d(star.indices, expected.indices)
            if len(differing):
                assert np.all(np.abs(star.los_cosines[differing]) <= eps)
                assert np.all(np.abs(expected.los_cosines[differing]) <= eps)
            entry = {
                "selected_indices_identical": bool(np.array_equal(star.indices, expected.indices)),
                "near_limb_selection_differences": len(differing),
                "los_cosines": measurement(star.los_cosines, expected.los_cosines, atol=eps),
                "coverage_sma_squared": measurement(star.coverage / binary.semi_major_axis**2,
                                                    expected.coverage / binary.semi_major_axis**2, atol=eps),
                "zero_coverage_selected_faces": int(np.sum(star.coverage[star.indices] == 0)),
                "selected_faces": len(star.indices), "bands": {},
            }
            for band in BANDS:
                faces, total = (np.asarray(x) for x in integrator.from_containers(band, [star]))
                cpu_faces = calculate_surface_element_fluxes(band, expected)
                full_actual, full_expected = np.zeros(len(star.faces)), np.zeros(len(expected.faces))
                full_actual[star.indices] = faces[0]
                full_expected[expected.indices] = cpu_faces
                scale = float(cpu_faces.sum())
                # Tiny near-limb terms are controlled by the same absolute
                # normalized-flux gate as full curves, not a meaningless relative ratio.
                entry["bands"][band] = {
                    "radiance": measurement(star.normal_radiance[band], expected.normal_radiance[band]),
                    "ld": measurement(star.ld_cfs[band], expected.ld_cfs[band]),
                    "face_flux_over_component": measurement(full_actual / scale if scale else full_actual,
                                                           full_expected / scale if scale else full_expected,
                                                           atol=2e-12),
                    "component_flux": measurement(total[0], scale, atol=0),
                }
            row["components"][name] = entry
        output["phases"].append(row)
    return output


def run(device: str, repeats: int) -> dict[str, Any]:
    started = perf_counter()
    engine = HybridBinaryLightCurve(BANDS, device=device, fallback="raise")
    preceding = HybridBinaryLightCurve(BANDS, device=device, fallback="raise", radiance="cpu")
    construction_seconds = perf_counter() - started
    output: dict[str, Any] = {
        "device": engine.device.platform, "device_kind": engine.device.device_kind,
        "contract": ("Host B/V/bolometric observed curves; fresh candidate mesh, "
                     "all host bridges and final downloads included"),
        "model": ("HYBRID: CPU mesh/roots/topology/kinematics, reflection, custom LD and fresh hull/visibility; "
                  "device corrected state, fresh blackbody/optical-depth radiance, projection, "
                  "partial-face clipping, coverage and batched integration"),
        "normalization": ("Observer maximum; l3 uses original observation order/repeats; "
                          "CPU distance helper for raw flux"),
        "constructor_seconds": construction_seconds, "repeats": repeats, "cases": [],
        "limitations": ["No all-GPU or resident forward-model claim", "No atmosphere-table/LD-table validation",
                        "Parity is not mesh or spectral convergence", "No fitting/MCMC or ESS claim"],
    }
    rng = np.random.default_rng(712)
    for case in CASES:
        with settings.override(REFLECTION_EFFECT=case["reflection"], NUMBER_OF_PROCESSES=1,
                               LIMB_DARKENING_LAW=case.get("law", "linear")):
            binary = fixture(case)
            phases = contact_phases(binary)
            result = engine(binary, phases, normalize=True, diagnostics=True)
            expected = Observer(passband=BANDS, system=binary).lc(phases=phases, normalize=True)[1]
            raw_result = engine(binary, phases)
            raw_expected = Observer(passband=BANDS, system=binary).lc(phases=phases)[1]
            errors = dict.fromkeys(BANDS, np.full(len(phases), 1e-4))
            observed = {band: curve + rng.normal(0, 1e-4, len(phases)) for band, curve in expected.items()}
            prepared = prepare_observations(observed, errors)
            l_cpu = prepared_likelihood(prepared, expected, -2.0)
            l_hybrid = prepared_likelihood(prepared, result.fluxes, -2.0)
            likelihood_error = measurement(l_hybrid, l_cpu, rtol=0, atol=1e-6)
            earlier = preceding(binary, phases, normalize=True)
            preceding_parity = {band: measurement(result.fluxes[band], earlier.fluxes[band], atol=2e-12)
                                for band in BANDS}
            occultation = result.diagnostics["occultation"]
            row = {
                "fixture": case, "phases": phases.tolist(),
                "face_counts": [len(getattr(result.diagnostics["initial_system"], name).faces)
                                for name in ("primary", "secondary")],
                "normalized": {band: measurement(result.fluxes[band], expected[band], atol=2e-12) for band in BANDS},
                "raw_w_m_squared": {band: measurement(raw_result.fluxes[band], raw_expected[band]) for band in BANDS},
                "log_likelihood_error_sigma_1e_4": likelihood_error,
                "preceding_cpu_radiance_hybrid_parity": preceding_parity,
                "device_radiance": {k: result.diagnostics["radiance"][k]
                                    for k in ("response_revision", "temperature_source")},
                "device_coverage": {key: occultation[key] for key in ("hull_bridge", "phase_chunk", "hull_capacity",
                                                                     "partial_capacity", "partial_faces")},
                "first_call_seconds_including_compilation": result.timings,
                "intermediates": details(result, binary, engine),
                "warm_hybrid_seconds": [], "warm_original_cpu_seconds": [], "warm_stages_seconds": [],
                "warm_preceding_cpu_radiance_hybrid_seconds": [],
            }
            for _ in range(repeats):
                before = perf_counter()
                warm = engine(binary, phases, normalize=True)
                row["warm_hybrid_seconds"].append(perf_counter() - before)
                row["warm_stages_seconds"].append(warm.timings)
                before = perf_counter()
                Observer(passband=BANDS, system=binary).lc(phases=phases, normalize=True)
                row["warm_original_cpu_seconds"].append(perf_counter() - before)
                before = perf_counter()
                preceding(binary, phases, normalize=True)
                row["warm_preceding_cpu_radiance_hybrid_seconds"].append(perf_counter() - before)
            row["median_hybrid_seconds"] = float(np.median(row["warm_hybrid_seconds"]))
            row["median_original_cpu_seconds"] = float(np.median(row["warm_original_cpu_seconds"]))
            row["median_preceding_cpu_radiance_hybrid_seconds"] = float(
                np.median(row["warm_preceding_cpu_radiance_hybrid_seconds"]),
            )
            output["cases"].append(row)
    output["all_correctness_gates_passed"] = True
    output["wall_seconds"] = perf_counter() - started
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("repeats must lie in [1,10]")
    report = run(args.device, args.repeats)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("device", "all_correctness_gates_passed", "wall_seconds")}))


if __name__ == "__main__":
    main()
