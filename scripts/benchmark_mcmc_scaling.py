"""Measure complete original and shared-GPU ELISa ensemble fits on a frozen fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

# Apply before loading any numerical library, including in spawned device owners.
THREAD_ENV = {
    key: "1" for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                         "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")
}
THREAD_ENV.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE", JAX_ENABLE_X64="True",
                  XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false")
os.environ.update(THREAD_ENV)

import numpy as np

from elisa import settings, units
from elisa.analytics import LCData
from elisa.analytics.binary_fit.chain_tools import ensemble_autocorrelation
from elisa.analytics.binary_fit.mcmc import DetachedLightCurveFit
from elisa.analytics.params import parameters
from elisa.experimental.gpu_fit import HybridDetachedMCMCFit
from elisa.experimental.gpu_protocol import MeshResolutionCheck, ResolutionNotQualifiedError
from elisa.experimental.gpu_walker_pool import SharedNativeDetachedMCMCFit, shared_surface_owner
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import FitProblem, make_problem, prepare_fit

WALKERS, STEPS, MOVE_SEED = 32, 12, 814
FLUX_ATOL, FLUX_RTOL, LOGP_ATOL = 2e-12, 3e-13, 1e-6
PHYSICS = dict(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2,
               LIMB_DARKENING_LAW="linear", NUMBER_OF_PROCESSES=1, NUMBER_OF_THREADS=1,
               PENALIZE_ALL_EXCEPTIONS=False)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(a):
    array = np.asarray(a, dtype="<f8", order="C")
    return hashlib.sha256(array.tobytes()).hexdigest()


def make_fixture(path):
    problem = make_problem(jd=False, sampling="direct", free_nuisance=True, shape_parameters=True)
    fit = prepare_fit(problem)
    truth = dict(problem.truth, **{"nuisance@ln_f": -2.0})
    centre = np.array(parameters.vector_normalizer([truth[k] for k in fit.fitable],
                                                   fit.fitable, fit.normalization))
    initial = centre + np.random.default_rng(20261118).normal(0, 0.001, (WALKERS, len(centre)))
    data = {band: {"x": v.x_data.tolist(), "y": v.y_data.tolist(), "error": v.y_err.tolist()}
            for band, v in problem.data.items()}
    write_json(path, dict(config=problem.config, setup=problem.setup, atmosphere=problem.atmosphere,
                          limb_darkening=problem.ld, truth=truth, data=data,
                          normalized_parameters=list(fit.fitable), centre=centre.tolist(),
                          initial=initial.tolist(), ensemble_seed=20261118, data_seed=717,
                          walkers=WALKERS, steps=STEPS, burn_in=0, move_seed=MOVE_SEED, physics=PHYSICS))


def read_problem(f):
    data = {band: LCData(x_data=np.array(v["x"]), y_data=np.array(v["y"]), y_err=np.array(v["error"]),
                         x_unit=units.dimensionless_unscaled, y_unit=units.dimensionless_unscaled)
            for band, v in f["data"].items()}
    return FitProblem(f["config"], data, f["setup"], f["atmosphere"], f["limb_darkening"], f["truth"])


class _TraceMixin:
    """Observe the original public fit and every proposed density, including rejected moves."""

    def predict_light_curves(self, model_parameters, **kwargs):
        values = super().predict_light_curves(model_parameters, **kwargs)
        self._last_flux = np.concatenate([values[b] for b in self.y_data]).tolist()
        return values

    def ln_probability(self, xn):
        self._last_flux = None
        density = super().ln_probability(xn)
        if np.isnan(density) or np.isposinf(density):
            raise ValueError("Invalid posterior")
        record = (xn.tolist(), float(density) if np.isfinite(density) else None, self._last_flux)
        path = self.trace_dir / f"worker-{os.getpid()}.jsonl"
        if not path.exists():
            meta = dict(affinity=sorted(os.sched_getaffinity(0)), walker_processes=settings.NUMBER_OF_MCMC_PROCESSES,
                        curve_processes=settings.NUMBER_OF_PROCESSES, elisa_threads=settings.NUMBER_OF_THREADS,
                        os_threads=len(list(Path("/proc/self/task").iterdir())))
            write_json(path.with_suffix(".json"), meta)
        with path.open("a") as out:
            out.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        return density

    def worker(self, *args, **kwargs):
        # Optional replay for device libraries that consume the legacy NumPy RNG
        # before emcee constructs its private proposal generator.
        if getattr(self, "proposal_seed", None) is not None:
            sampler = args[0]
            state = sampler.random_state
            self.proposal_rng_before_replay = array_hash(np.r_[state[1].astype(float), state[2:]])
            sampler.random_state = np.random.RandomState(self.proposal_seed).get_state()
        start = perf_counter()
        result = super().worker(*args, **kwargs)
        self.sampling_seconds = perf_counter() - start
        return result


class _OriginalTrace(_TraceMixin, DetachedLightCurveFit):
    pass


class _HybridTrace(_TraceMixin, SharedNativeDetachedMCMCFit):
    pass


class _ExistingSerialHybrid(_TraceMixin, HybridDetachedMCMCFit):
    pass


def read_trace(folder, nobs):
    records, work, settings_seen = [], [], []
    for path in sorted(folder.glob("worker-*.jsonl")):
        batch = [json.loads(line) for line in path.read_text().splitlines()]
        records.extend(batch)
        work.append(len(batch))
        settings_seen.append(json.loads(path.with_suffix(".json").read_text()))
    records.sort(key=lambda row: row[0])
    positions = np.asarray([r[0] for r in records])
    logp = np.asarray([-np.inf if r[1] is None else r[1] for r in records])
    flux = np.asarray([np.full(nobs, np.nan) if r[2] is None else r[2] for r in records])
    return dict(positions=positions, posterior=logp, flux=flux), sorted(work), settings_seen


def run(args):
    fixture = json.loads(args.fixture.read_text())
    problem = read_problem(fixture)
    initial = np.array(fixture["initial"], dtype=np.float64)
    cpus = {int(c) for c in args.cpus.split(",")}
    if args.backend == "serial-hybrid" and args.workers != 1:
        raise ValueError("The existing physical hybrid supports one walker worker")
    if len(cpus) != args.workers:
        raise ValueError("Choose one distinct walker CPU for each worker")
    if args.backend == "hybrid" and args.owner_cpu in cpus:
        raise ValueError("Reserve a separate logical CPU for the GPU owner")
    os.sched_setaffinity(0, cpus)
    args.output.mkdir(parents=True, exist_ok=False)
    trace_dir = args.output / "trace"
    trace_dir.mkdir()
    load = os.getloadavg()
    start_utc = datetime.now(timezone.utc).isoformat()
    np.random.seed(fixture["move_seed"])
    nobs = sum(len(v.y_data) for v in problem.data.values())
    accuracy = dict(flux_atol=FLUX_ATOL, flux_rtol=FLUX_RTOL, posterior_atol=LOGP_ATOL)

    with settings.override(NUMBER_OF_MCMC_PROCESSES=args.workers):
        start = perf_counter()
        owner = (shared_surface_owner(settings.LIMB_DARKENING_LAW, cpu=args.owner_cpu,
                                      visible_devices=args.gpu) if args.backend == "hybrid" else nullcontext())
        with owner as service:
            if args.backend == "serial-hybrid":
                fit = _ExistingSerialHybrid(device="gpu")
            elif service is not None:
                fit = _HybridTrace(service, settings.LIMB_DARKENING_LAW)
            else:
                fit = _OriginalTrace()
            if args.pin_proposal_seed:
                fit.proposal_seed = fixture["move_seed"]
            fit.atmosphere_model = deepcopy(problem.atmosphere)
            fit.limb_darkening_coefficients = deepcopy(problem.ld)
            fit.trace_dir = trace_dir
            fit.fit(problem.data, parameters.BinaryInitialParameters(**deepcopy(problem.config)),
                    **problem.setup, nwalkers=fixture["walkers"], nsteps=fixture["steps"],
                    burn_in=fixture["burn_in"], initial_state=initial.copy(), save=False, progress=False)
            device = None if service is None else service.statistics()
            if args.backend == "serial-hybrid":
                synthesis = fit.hybrid_synthesis
                device = dict(calls=synthesis.hybrid_calls, fallback_calls=synthesis.cpu_fallback_calls,
                              platform=synthesis.device.platform, device=synthesis.device.device_kind,
                              x64=bool(synthesis._jax.config.x64_enabled), affinity=sorted(cpus))
        elapsed = perf_counter() - start

    if list(fit.fitable) != fixture["normalized_parameters"]:
        raise RuntimeError("Parameter ordering differs from the frozen ensemble")
    sampler = fit.last_sampler
    chain, logp = sampler.get_chain(), sampler.get_log_prob()
    if chain.dtype != np.float64 or logp.dtype != np.float64:
        raise RuntimeError("Sampler did not retain FP64")
    trace, work, configs = read_trace(trace_dir, nobs)
    expected_config = dict(affinity=sorted(cpus), walker_processes=args.workers, curve_processes=1, elisa_threads=1)
    if len(work) != args.workers or any(any(c[k] != v for k, v in expected_config.items()) for c in configs):
        raise RuntimeError(f"Unexpected worker allocation: {work}, {configs}")
    if device and (device["platform"] != "gpu" or not device["x64"] or device["calls"] < len(trace["posterior"])
                   or device.get("fallback_calls", 0) or
                   (args.backend == "hybrid" and device["affinity"] != [args.owner_cpu])):
        raise RuntimeError(f"Incorrect device execution: {device}")
    expected = fixture["walkers"] * (1 + fixture["steps"])  # no burn-in, default stretch move
    if len(trace["positions"]) != expected:
        raise RuntimeError(f"Observed {len(trace['positions'])} posterior calls, expected {expected}")
    physical = int(np.isfinite(trace["flux"]).all(axis=1).sum())
    accepted = np.rint(sampler.acceptance_fraction * fixture["steps"]).astype(int)
    _, tau = ensemble_autocorrelation(chain)
    tau_report = [float(x) if np.isfinite(x) else None for x in tau]
    np.savez_compressed(args.output / "trace.npz", **trace, chain=chain, chain_logp=logp, accepted=accepted)
    result = dict(backend=args.backend, workers=args.workers, started_utc=start_utc,
                  finished_utc=datetime.now(timezone.utc).isoformat(), load_before=load, load_after=os.getloadavg(),
                  walker_logical_cpus=sorted(cpus), worker_posterior_calls=work, observed_worker_settings=expected_config,
                  os_threads_per_walker_process=sorted(c["os_threads"] for c in configs),
                  thread_environment=THREAD_ENV, fixture_sha256=sha256(args.fixture), device=device,
                  seconds=elapsed, sampling_seconds=fit.sampling_seconds, posterior_calls=expected,
                  physical_likelihoods=physical, nonfinite_posteriors=int((~np.isfinite(trace["posterior"])).sum()),
                  likelihoods_per_second=physical/elapsed, sampling_likelihoods_per_second=physical/fit.sampling_seconds,
                  initial_preflight=fit._initialization_diagnostics, accepted=int(accepted.sum()),
                  proposed_moves=fixture["steps"] * fixture["walkers"], chain_sha256=array_hash(chain),
                  evaluated_coordinates_sha256=array_hash(trace["positions"]), trace_sha256=sha256(args.output / "trace.npz"),
                  autocorrelation_steps=tau_report, tolerances=accuracy)
    if args.pin_proposal_seed:
        result["emcee_proposal_seed"] = fit.proposal_seed
        result["proposal_rng_before_replay_sha256"] = fit.proposal_rng_before_replay
    write_json(args.output / "result.json", result)
    print(f"{args.backend} {args.workers} workers, {physical} evaluations, {elapsed:.3f} seconds", flush=True)


def compare(reference, candidate):
    with np.load(reference / "trace.npz") as a, np.load(candidate / "trace.npz") as b:
        deltas = {}
        for k in ("positions", "chain", "accepted"):
            if not np.array_equal(a[k], b[k]):
                raise ValueError(f"CPU and candidate {k} differ")
        for k, atol, rtol in (("flux", FLUX_ATOL, FLUX_RTOL), ("posterior", LOGP_ATOL, 0),
                              ("chain_logp", LOGP_ATOL, 0)):
            if not np.array_equal(np.isfinite(a[k]), np.isfinite(b[k])):
                raise ValueError(f"CPU and candidate {k} finite status differ")
            np.testing.assert_allclose(b[k], a[k], rtol=rtol, atol=atol, equal_nan=True)
            valid = np.isfinite(a[k])
            deltas[k] = float(np.max(np.abs(b[k][valid] - a[k][valid]), initial=0))
        return dict(max_abs=deltas, attempted_positions_equal=True, chain_equal=True, accepted_equal=True)


def check_mesh(fixture, output):
    f = json.loads(fixture.read_text())
    fit = prepare_fit(read_problem(f), phase_mode="observations", resolution_check=MeshResolutionCheck())
    checks = {}
    for label, position in (("generating_point", f["centre"]), ("initial_walker_0", f["initial"][0]),
                            ("initial_walker_31", f["initial"][-1])):
        try:
            value = fit.ln_probability(np.array(position))
            checks[label] = dict(density=float(value), report=fit.last_resolution_check)
        except ResolutionNotQualifiedError as exc:
            checks[label] = dict(density=None, report=exc.report)
    write_json(output, checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="cmd", required=True)
    gen = commands.add_parser("fixture")
    gen.add_argument("output", type=Path)
    bench = commands.add_parser("run")
    bench.add_argument("--backend", choices=("cpu", "hybrid", "serial-hybrid"), required=True)
    bench.add_argument("--workers", type=int, required=True)
    bench.add_argument("--cpus", required=True, help="comma-separated walker logical CPU IDs")
    bench.add_argument("--owner-cpu", type=int, default=16)
    bench.add_argument("--gpu", default="0")
    bench.add_argument("--pin-proposal-seed", action="store_true",
                       help="Set the fixture move seed directly on emcee after solver initialization")
    bench.add_argument("--fixture", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("compare")
    audit.add_argument("reference", type=Path)
    audit.add_argument("candidate", type=Path)
    mesh = commands.add_parser("mesh")
    mesh.add_argument("fixture", type=Path)
    mesh.add_argument("output", type=Path)
    args = parser.parse_args()
    with table_settings(), settings.override(**PHYSICS):
        if args.cmd == "fixture":
            make_fixture(args.output)
        elif args.cmd == "run":
            run(args)
        elif args.cmd == "compare":
            print(json.dumps(compare(args.reference, args.candidate), indent=2))
        else:
            check_mesh(args.fixture, args.output)


if __name__ == "__main__":
    main()
