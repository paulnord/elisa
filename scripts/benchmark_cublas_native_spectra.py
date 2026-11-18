"""Opt-in, transfer-inclusive cuBLAS and CPU comparison in ELISa's original likelihood."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import multiprocessing
import os
import threading
from contextlib import ExitStack
from copy import deepcopy
from multiprocessing.managers import BaseManager
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from scripts import benchmark_mcmc_scaling as scaling
from scripts.benchmark_native_blackbody_blocks import blocked_blackbody
from scripts.study_native_mesh_refinement import Native, read, settings, store, study_positions, table_settings
import numpy as np

from elisa import const
from elisa.atm import atm
from elisa.experimental.gpu_blackbody import _prepare_quadrature, _sample_passbands
from elisa.experimental.gpu_walker_pool import _configure_owner


def band_signature(bands):
    digest = hashlib.sha256()
    for name, (wave, response) in zip(bands, _sample_passbands(bands), strict=True):
        digest.update(name.encode() + b"\x00")
        digest.update(len(wave).to_bytes(8, "little"))
        digest.update(wave.tobytes())
        digest.update(response.tobytes())
    return digest.hexdigest()


def fixed_arrays(bands):
    wave, weights, _ = _prepare_quadrature(bands, chunk_size=1)
    prefactor = (2 * const.PLANCK_CONST * const.C**2) / wave**5
    exponent = (const.PLANCK_CONST * const.C / const.BOLTZMAN_CONST) / wave
    return tuple(np.ascontiguousarray(a, dtype=np.float64) for a in (prefactor, exponent, weights))


class CpuQuadrature:
    def __init__(self, prefactor, exponent, weights):
        self.prefactor, self.exponent, self.weights = prefactor, exponent, weights

    def evaluate(self, temperatures):
        result = np.empty((len(temperatures), len(self.weights)))
        for first in range(0, len(result), 256):
            rows = temperatures[first:first + 256, None]
            with np.errstate(over="ignore"):
                spectrum = self.prefactor / np.expm1(self.exponent / rows)
            result[first:first + len(rows)] = spectrum @ self.weights.T
        return result


class CudaQuadrature:
    def __init__(self, library, prefactor, exponent, weights):
        self.lib = lib = ctypes.CDLL(str(library))
        array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ctx, integer = ctypes.c_void_p, ctypes.c_int
        p_int = ctypes.POINTER(integer)
        lib.spectra_create.argtypes = [integer, integer, array, array, array]
        lib.spectra_create.restype = ctx
        lib.spectra_device.argtypes = [ctx, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t, p_int, p_int, p_int]
        lib.spectra_device.restype = integer
        lib.spectra_integrate.argtypes = [ctx, array, integer, array]
        lib.spectra_integrate.restype = integer
        lib.spectra_error.argtypes = []
        lib.spectra_error.restype = ctypes.c_char_p
        lib.spectra_close.argtypes = [ctx]
        lib.spectra_close.restype = None
        self.bands = len(weights)
        self.ctx = lib.spectra_create(len(exponent), self.bands, prefactor, exponent, weights)
        if not self.ctx:
            raise RuntimeError(lib.spectra_error().decode())
        name = ctypes.create_string_buffer(256)
        versions = [integer() for _ in range(3)]
        if lib.spectra_device(self.ctx, name, len(name), *(ctypes.byref(v) for v in versions)) < 0:
            self.close()
            raise RuntimeError(lib.spectra_error().decode())
        self.device = dict(name=name.value.decode(), runtime=versions[0].value, driver=versions[1].value,
                           cublas=versions[2].value, precision="FP64", math_mode="CUBLAS_PEDANTIC_MATH")
        self.gemm_calls = 0

    def evaluate(self, temperatures):
        out = np.empty((len(temperatures), self.bands), dtype=np.float64)
        tiles = self.lib.spectra_integrate(self.ctx, temperatures, len(temperatures), out)
        if tiles < 1:
            raise RuntimeError(self.lib.spectra_error().decode())
        self.gemm_calls += tiles
        return out

    def close(self):
        if self.ctx:
            self.lib.spectra_close(self.ctx)
            self.ctx = None


class SpectralOwner:
    """One locked context accepts full-precision temperature arrays from CPU workers."""
    def __init__(self, library):
        self.library = library
        self.kernels = {}
        self.lock = threading.Lock()
        self.calls = 0
        self.temperature_bytes = 0
        self.radiance_bytes = 0

    def install(self, key, arrays):
        with self.lock:
            if key not in self.kernels:
                self.kernels[key] = CudaQuadrature(self.library, *arrays)

    def evaluate(self, key, temperatures):
        with self.lock:
            values = self.kernels[key].evaluate(temperatures)
            self.calls += 1
            self.temperature_bytes += temperatures.nbytes
            self.radiance_bytes += values.nbytes
            return values

    def statistics(self):
        with self.lock:
            return dict(calls=self.calls, gemm_calls=sum(k.gemm_calls for k in self.kernels.values()),
                        devices=[k.device for k in self.kernels.values()], owner_affinity=sorted(os.sched_getaffinity(0)),
                        temperature_bytes=self.temperature_bytes, radiance_bytes=self.radiance_bytes)

    def close(self):
        with self.lock:
            for kernel in self.kernels.values():
                kernel.close()


class SpectralManager(BaseManager):
    pass


SpectralManager.register("spectral", SpectralOwner, exposed=("install", "evaluate", "statistics", "close"))


class Callback:
    def __init__(self, backend, owner, reference, check_faces, receipts):
        self.backend, self.owner, self.reference = backend, owner, reference
        self.check_faces, self.receipts = check_faces, receipts
        self.known = {}
        self.face = dict(values=0, max_relative=0.0, passed=True)

    def __call__(self, temperature, **kwargs):
        original = self.reference
        if self.backend == "native":
            result = original(temperature, **kwargs)
        elif self.backend == "blocked":
            result = blocked_blackbody(temperature, **kwargs)
        else:
            bands = kwargs["passband"]
            key = band_signature(bands)
            if key not in self.known:
                arrays = fixed_arrays(bands)
                if self.owner is None:
                    self.known[key] = CpuQuadrature(*arrays)
                else:
                    self.owner.install(key, arrays)
                    self.known[key] = True
            unique, reverse = np.unique(temperature, return_inverse=True)
            if not len(unique) or not np.isfinite(unique).all() or np.min(unique) <= 0:
                raise ValueError("Native stellar surface has a nonphysical temperature")
            values = self.known[key].evaluate(unique) if self.owner is None else self.owner.evaluate(key, unique)
            if values.dtype != np.float64 or not np.isfinite(values).all() or np.min(values) <= 0:
                raise ValueError("Invalid integrated surface radiance")
            result = {name:values[reverse, i] for i, name in enumerate(bands)}

        if self.check_faces:
            expected = original(temperature, **kwargs)
            if result.keys() != expected.keys():
                raise ValueError("Mismatch in observed passbands")
            for name, flux in expected.items():
                if flux.shape != result[name].shape or not np.all(flux > 0):
                    raise ValueError("Invalid reference shape or radiance")
                rel = np.max(np.abs(result[name]-flux)/flux)
                self.face["max_relative"] = max(self.face["max_relative"], float(rel))
                self.face["values"] += flux.size
                self.face["passed"] &= bool(np.allclose(result[name], flux, atol=1e-10, rtol=3e-13))
        with (self.receipts / str(os.getpid())).open("ab") as trace:
            trace.write(b"1")
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("curves", "chain"))
    p.add_argument("backend", choices=("native", "blocked", "cpu-dgemm", "cuda-dgemm"))
    p.add_argument("fixture", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--library", type=Path)
    p.add_argument("--degrees", type=float, default=1.5)
    p.add_argument("--begin", type=int, default=33)
    p.add_argument("--end", type=int, default=37)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--cpus", default="8")
    p.add_argument("--owner-cpu", type=int, default=16)
    p.add_argument("--face-check", action="store_true")
    args = p.parse_args()
    cpus = {int(v) for v in args.cpus.split(",")}
    if not cpus or (args.backend == "cuda-dgemm" and not args.library):
        p.error("Set explicit walker CPUs and a built library for cuda-dgemm")
    if args.action == "chain" and len(cpus)>1 and multiprocessing.get_context("fork").get_start_method()!="fork":
        p.error("Multicore benchmark uses ELISa's Linux fork context")
    os.sched_setaffinity(0, cpus)
    fixture = read(args.fixture)
    receipts = args.output.with_name(args.output.stem + "-calls")
    receipts.mkdir(parents=True, exist_ok=False)
    rows, chain, gpu = [], None, None
    start = perf_counter()
    with ExitStack() as stack:
        stack.enter_context(table_settings())
        stack.enter_context(settings.override(**scaling.PHYSICS, NUMBER_OF_MCMC_PROCESSES=len(cpus),
                                              MAX_DISCRETIZATION_FACTOR=12, MESH_GENERATOR="trapezoidal"))
        owner = None
        if args.backend == "cuda-dgemm":
            if args.action == "chain":
                manager = SpectralManager(ctx=multiprocessing.get_context("spawn"))
                manager.start(initializer=_configure_owner, initargs=(args.owner_cpu, "gpu", "0"))
                stack.callback(manager.shutdown)
                owner = manager.spectral(str(args.library.resolve()))
            else:
                owner = SpectralOwner(str(args.library.resolve()))
            stack.callback(owner.close)
        callback = Callback(args.backend, owner, atm.NaiveInterpolatedAtm.black_body_radiance, args.face_check, receipts)
        stack.enter_context(patch.object(atm.NaiveInterpolatedAtm, "black_body_radiance", staticmethod(callback)))
        if args.action == "curves":
            positions = study_positions(fixture)[0]
            if not 0 <= args.begin < args.end <= len(positions):
                p.error("Coordinates fall outside the frozen proposal list")
            model = Native(fixture, args.degrees)
            for label, position in positions[args.begin:args.end]:
                rows.append(dict(label=label, position=position, **model.measure(position)))
        else:
            spec = deepcopy(fixture)
            spec["steps"] = args.steps
            spec["setup"]["discretization"] = args.degrees
            chain_fixture = args.output.with_name(args.output.stem + "-fixture.json")
            store(chain_fixture, spec)
            folder = args.output.with_name(args.output.stem + "-chain")
            # The native fitter owns the emcee move, acceptance, prior, normalized
            # likelihood and final fit summary. The scoped callback changes only spectra.
            scaling.run(argparse.Namespace(fixture=chain_fixture, cpus=args.cpus, workers=len(cpus), backend="cpu",
                                           output=folder, owner_cpu=args.owner_cpu, pin_proposal_seed=True))
            chain = read(folder/"result.json")
            chain["spectral_backend"] = args.backend
        if owner is not None:
            gpu = owner.statistics()
    total = perf_counter()-start
    counts = sorted(f.stat().st_size for f in receipts.iterdir())
    if not counts or (gpu and (gpu["calls"] != sum(counts) or gpu["gemm_calls"] < gpu["calls"])):
        raise RuntimeError("The measured radiance calls and hardware execution counters disagree")
    store(args.output, dict(action=args.action,backend=args.backend,primary_degrees=args.degrees,
                            fixture_sha256=scaling.sha256(args.fixture), complete_wall_s=total, rows=rows,chain=chain,
                            blackbody_calls_by_process=counts, face_check=callback.face if args.face_check else None,
                            gpu=gpu,cpu_affinity=sorted(cpus),regular_cpu_threads=1))


if __name__ == "__main__":
    main()
