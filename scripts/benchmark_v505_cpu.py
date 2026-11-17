"""Fresh V505 observation-window original/JAX CPU workloads, with pinned input.

Explicit blackbody/linear-LD Roche seed; no detrending, optimized fit, posterior,
numerical certification or GPU/ESS result. Run CPU-only in a private environment.
"""

# ruff: noqa: E402, INP001, S101, T201, PLR2004 - standalone measured analysis driver
from time import perf_counter

PROCESS_STARTED = perf_counter()

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import brentq

from elisa import BinarySystem, Observer, Star, settings
from elisa.analytics.models.cost_fns import prepare_observations, prepared_likelihood

BAND = "TESS"
INPUT_SHA256 = "4351efe790cf3a4c718c9fdb1a9cf1ecd21d078374e90e6440b437c034618f1d"
SEED = {"period": 4.2220216, "inclination": 87.9166, "masses": [1.2745, 1.2577],
        "temperatures": [6512., 6460.], "fractional_radii": [.086370, .084536],
        "epoch_bjd_tdb": 2458798.516720}


def binary(potentials: list, *, degrees: float = 8., inclination: float = 87.9166,
           temperature: float = 6512.) -> BinarySystem:
    common = {"synchronicity": 1., "atmosphere": "bb", "metallicity": 0., "albedo": .5,
              "gravity_darkening": .32, "discretization_factor": degrees,
              "limb_darkening_coefficients": {BAND: [.4], "bolometric": [.4]}}
    return BinarySystem(
        Star(mass=SEED["masses"][0], t_eff=temperature, surface_potential=potentials[0], **common),
        Star(mass=SEED["masses"][1], t_eff=temperature-52., surface_potential=potentials[1], **common),
        inclination=inclination, eccentricity=0., argument_of_periastron=90., period=SEED["period"],
        phase_shift=0., additional_light=0., distance=25.,
    )


def seed_potentials() -> list:
    potentials = [12.6, 12.7]
    for index, name in enumerate(("primary", "secondary")):
        def residual(p: float, *, i: int = index, component: str = name) -> float:
            candidate = potentials.copy()
            candidate[i] = p
            return getattr(binary(candidate), component).equivalent_radius-SEED["fractional_radii"][i]
        potentials[index] = brentq(residual, 8., 25., xtol=1e-11)
    return potentials


def observations(path: Path, max_points: int) -> tuple:
    if hashlib.sha256(path.read_bytes()).hexdigest() != INPUT_SHA256:
        msg = "Input differs from pinned Sector18 V505 product; inspect provenance before analysis"
        raise ValueError(msg)
    with fits.open(path) as h:
        table, header = h[1].data, h[1].header
        assert header["TIMESYS"] == "TDB"
        assert header["BJDREFI"] == 2457000
        assert header["BJDREFF"] == 0
        good = (table["QUALITY"] == 0) & np.isfinite(table["TIME"])
        good &= np.isfinite(table["SAP_FLUX"]) & np.isfinite(table["SAP_FLUX_ERR"])
        good &= (table["SAP_FLUX"] > 0) & (table["SAP_FLUX_ERR"] > 0)
        time = np.array(table["TIME"][good], float)+header["BJDREFI"]
        flux, error = [np.array(table[c][good], float) for c in ("SAP_FLUX", "SAP_FLUX_ERR")]
    phases = (time-SEED["epoch_bjd_tdb"])/SEED["period"]
    cos, sin = np.cos(np.radians(SEED["inclination"])), np.sin(np.radians(SEED["inclination"]))
    half = np.arcsin(np.sqrt((sum(SEED["fractional_radii"])**2-cos**2)/sin**2))/(2*np.pi)
    distance = np.abs((phases+.25) % .5-.25)
    selected = np.flatnonzero(distance <= 2*half)
    full = len(selected)
    if max_points and len(selected) > max_points:
        selected = selected[np.linspace(0, len(selected)-1, max_points, dtype=int)]
    scale = np.median(flux[distance > 1.3*half])
    return phases[selected], flux[selected]/scale, error[selected]/scale, {
        "full_valid_rows": len(time), "full_eclipse_window_rows": full, "selected_rows": len(selected),
        "original_row_indices_after_valid_filter": selected.tolist(), "input_sha256": INPUT_SHA256,
        "selection": "QUALITY0/finite positive SAP/error; twice spherical contact half-width; no detrending",
    }


def evaluate(path: str, engine: object, phases: np.ndarray, prepared: object, potentials: list, *,
             exposure_samples: int = 1, **kwargs: object) -> tuple:
    start = perf_counter()
    model = binary(potentials, **kwargs)
    offsets = ((np.arange(exposure_samples)+.5)/exposure_samples-.5)*120/86400/SEED["period"]
    expanded = (phases[:, None]+offsets).ravel()
    if path == "original_cpu":
        raw = Observer(passband=[BAND], system=model).lc(phases=expanded, normalize=False)[1][BAND]
        stages = {}
    else:
        result = engine(model, expanded, normalize=False)
        if result.backend != "hybrid":
            msg = "Unexpected fallback in explicitly selected JAX CPU workload"
            raise RuntimeError(msg)
        raw, stages = result.fluxes[BAND], result.timings
    curve = raw.reshape(len(phases), exposure_samples).mean(axis=1)
    curve /= curve.max()
    density = float(prepared_likelihood(prepared, {BAND: curve}, -2.))
    return curve, {"complete_constructor_forward_mapping_density_seconds": perf_counter()-start,
                   "stages": stages, "log_likelihood": density}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fits", type=Path, required=True)
    p.add_argument("--path", choices=["original_cpu", "jax_cpu_hybrid"], required=True)
    p.add_argument("--processes", type=int, choices=[1, 4], default=1)
    p.add_argument("--points", type=int, default=0)
    p.add_argument("--proposals", type=int, default=8)
    p.add_argument("--features", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.points < 0 or not 0 <= a.proposals <= 256 or (a.path != "original_cpu" and a.processes != 1):
        msg = "Use nonnegative points,0-256proposals and original CPU for multiprocessing"
        raise ValueError(msg)
    # Never initialize CUDA here; a GPU port requires its own bounded allocation.
    if a.path == "jax_cpu_hybrid":
        if os.environ.get("JAX_PLATFORMS") != "cpu":
            msg = "Set JAX_PLATFORMS=cpu before starting this CPU analysis"
            raise RuntimeError(msg)
        import jax  # noqa: PLC0415 - original CPU does not initialize JAX

        from elisa.experimental.gpu_binary import HybridBinaryLightCurve  # noqa: PLC0415
        assert jax.config.x64_enabled
        assert all(d.platform == "cpu" for d in jax.devices())
    phases, flux, errors, selection = observations(a.fits, a.points)
    prepared = prepare_observations({BAND: flux}, {BAND: errors})
    report = {"selection": selection, "seed": SEED, "path": a.path, "elisa_processes": a.processes,
              "affinity": sorted(os.sched_getaffinity(0)), "results": [], "features": {}}
    with settings.override(NUMBER_OF_PROCESSES=a.processes, NUMBER_OF_THREADS=1, LIMB_DARKENING_LAW="linear",
                           REFLECTION_EFFECT=False, REFLECTION_EFFECT_ITERATIONS=2, SUPPRESS_WARNINGS=True):
        setup = perf_counter()
        potentials = seed_potentials()
        engine = HybridBinaryLightCurve([BAND], device="cpu", fallback="raise", coverage="cpu",
                                        radiance="device", reflection="tiled") if a.path == "jax_cpu_hybrid" else None
        report["setup_seconds"] = perf_counter()-setup
        rng, curves = np.random.default_rng(20261117), {}
        proposals = [{"potentials": potentials, "inclination": 87.9166, "temperature": 6512.}]
        proposals.extend({"potentials": (np.array(potentials)+rng.normal(0, .004, 2)).tolist(),
                              "inclination": float(87.9166+rng.normal(0, .01)),
                              "temperature": float(6512+rng.normal(0, 15.))} for _ in range(a.proposals))
        report["physical_proposals"] = proposals
        for i, proposal in enumerate(proposals):
            curve, row = evaluate(a.path, engine, phases, prepared, **proposal)
            curves[f"{a.path}_proposal_{i}"] = curve
            report["results"].append(row)
        if a.features:
            for label, options in [("mesh_4deg", {"degrees": 4.}),
                                   ("exposure_3point_120s", {"exposure_samples": 3}),
                                   ("reflection_2iterations", {})]:
                with settings.override(REFLECTION_EFFECT=label.startswith("reflection")):
                    curve, row = evaluate(a.path, engine, phases, prepared, potentials, **options)
                    curves[f"{a.path}_feature_{label}"] = curve
                    row["difference_from_seed_max_abs"] = float(np.max(np.abs(curve-curves[f"{a.path}_proposal_0"])))
                    report["features"][label] = row
    report["process_wall_seconds"] = perf_counter()-PROCESS_STARTED
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["limits"] = ["Blackbody/linearLD Roche seed differs from published quadraticLD JKTEBOP model.",
                        "Original Gaussian density on undetrended SAP; no adequacy or posterior claim.",
                        "120s nominal box averages raw flux before normalization; frame/deadtime kernel not derived.",
                        "Fresh changing proposals, not a vectorized walker/MCMC/ESS benchmark or convergence proof.",
                        "CPU single measurements; setup/imports/process wall separate, caches can warm."]
    np.savez_compressed(a.output.with_suffix(".npz"), **curves)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"path": a.path, "observations": len(phases),
                      "process_wall_seconds": report["process_wall_seconds"]}))


if __name__ == "__main__":
    main()
