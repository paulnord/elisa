"""Opt-in original facet physics with parallel CPU walkers and one FP64 device owner."""

from __future__ import annotations

import multiprocessing
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.managers import BaseManager
from pathlib import Path

import numpy as np

from elisa import settings
from elisa.analytics.binary_fit.mcmc import DetachedLightCurveFit
from elisa.analytics.models.lc import prepare_binary
from elisa.experimental.gpu_native import NativePhysicsIntegrator
from elisa.experimental.gpu_ordered import OrderedSurfaceFluxIntegrator


class _SurfaceOwner:
    """Serialize the bounded surface reductions from independent walker processes."""

    def __init__(self, law, device):
        self.kernel = OrderedSurfaceFluxIntegrator(law, device=device)
        self.lock = threading.Lock()
        self.calls = 0

    def totals(self, arrays):
        with self.lock:
            packet = self.kernel.prepare(*arrays)
            _, totals = packet.evaluate()
            values = np.asarray(totals)
            if values.dtype != np.float64 or not np.isfinite(values).all():
                raise RuntimeError("Device surface reduction returned invalid FP64 totals")
            self.calls += 1
            return values

    def statistics(self):
        device = self.kernel.device
        return dict(calls=self.calls, platform=device.platform, device=device.device_kind,
                    x64=bool(self.kernel._jax.config.x64_enabled),
                    affinity=sorted(os.sched_getaffinity(0)), os_threads=len(list(Path("/proc/self/task").iterdir())))


class _SurfaceManager(BaseManager):
    pass


_SurfaceManager.register("surface", _SurfaceOwner, exposed=("totals", "statistics"))


def _configure_owner(cpu, device, visible_devices):
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    os.environ["JAX_PLATFORMS"] = "cuda" if device == "gpu" else "cpu"
    if device == "gpu" and visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices


@contextmanager
def shared_surface_owner(law, *, device="gpu", cpu=None, visible_devices=None):
    """Start a fresh, local Unix-socket service. Only its process initializes JAX."""
    if device not in ("gpu", "cpu"):
        raise ValueError("Select an explicit device")
    manager = _SurfaceManager(ctx=multiprocessing.get_context("spawn"))
    manager.start(initializer=_configure_owner, initargs=(cpu, device, visible_devices))
    try:
        yield manager.surface(law, device)
    finally:
        manager.shutdown()


@dataclass
class _RemotePacket:
    service: object
    arrays: tuple

    def evaluate(self):
        # Full host face diagnostics are not requested by this synthesis path.
        return None, self.service.totals(self.arrays)


@dataclass
class SharedNativeSynthesis:
    """Picklable synthesis callback for the original detached ensemble fitter."""

    service: object
    law: str

    def _prepare(self, *arrays):
        return _RemotePacket(self.service, arrays)

    def __call__(self, phases, discretization, observer, **physical):
        if settings.LIMB_DARKENING_LAW != self.law:
            raise ValueError("The device and the current model must use the same limb-darkening law")
        observer._system = prepare_binary(discretization=discretization, **physical)
        result = NativePhysicsIntegrator.integrate_with(
            observer, phases, normalize=True, ncoeff=1 if self.law in ("linear", "cosine") else 2,
            prepare=self._prepare,
        )
        return result["fluxes"]


class SharedNativeDetachedMCMCFit(DetachedLightCurveFit):
    """Use native ELISa walker multiprocessing with one shared surface integrator."""

    def __init__(self, service, law):
        super().__init__()
        self.synthesis = SharedNativeSynthesis(service, law)

    def fit(self, *args, **kwargs):
        if settings.PENALIZE_ALL_EXCEPTIONS:
            raise ValueError("Shared GPU device errors must propagate outside candidate penalties")
        return super().fit(*args, **kwargs)

    def predict_light_curves(self, model_parameters, **kwargs):
        return super().predict_light_curves(model_parameters, synthesis=self.synthesis, **kwargs)
