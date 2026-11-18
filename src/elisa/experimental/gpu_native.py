"""Broader HYBRID curves: original physical routers, fresh device band integration.

No CPU oracle/global function is patched. A private per-call Observer proxy
delegates the binary state and substitutes only the router's existing flux
callback. CPU spot evolution, eccentric shape/reflection/coverage and configured
approximations remain original operations. This is not resident GPU geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import settings
from elisa.binary_system import dynamic
from elisa.binary_system.curves import c_router
from elisa.binary_system.system import BinarySystem
from elisa.conf.enums import CurveType
from elisa.experimental.gpu_ordered import OrderedSurfaceFluxIntegrator

if TYPE_CHECKING:
    from collections.abc import Callable

    from elisa.experimental.gpu_surface import PreparedSurfaceBatch

_MAX_BANDS, _MAX_FACES, _MAX_CALLBACKS = 8, 8192, 256


class UnsupportedNativeIntegrationError(ValueError):
    """A bridge resource/contract limit requires whole original CPU synthesis."""


@dataclass(frozen=True)
class NativeSurfaceRecord:
    """Diagnostic copy of actual fresh CPU physics and its immutable device packet."""

    system: Any
    packet: PreparedSurfaceBatch
    face_flux: np.ndarray
    component_flux: np.ndarray
    selected_counts: tuple[int, ...]


class _RoutedBinary:
    """Single-call state delegate; Observer retains its original BinarySystem tag."""

    def __init__(self, binary: Any, callback: Any) -> None:
        self.binary, self.callback = binary, callback
        self.route: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.binary, name)

    def compute_lightcurve(self, **kwargs: Any) -> Any:
        selected = c_router.resolve_curve_method(self.binary, curve=CurveType.LC)
        self.route = selected.__name__
        labels = list(kwargs["passband"])
        if self.route == "_compute_circular_synchronous_lightcurve":
            initial = c_router.prep_initial_system(self.binary)
            phases, reverse = dynamic.phase_crv_symmetry(initial, kwargs.pop("phases"))
            curves = c_router.produce_circular_sync_curves(
                self.binary, initial, phases, self.callback, labels, **kwargs,
            )
            return {band: values[reverse] for band, values in curves.items()}
        routes = {
            "_compute_circular_spotty_asynchronous_lightcurve": c_router.produce_circular_spotty_async_curves,
            "_compute_eccentric_spotty_lightcurve": c_router.produce_ecc_curves_with_spots,
            "_compute_eccentric_lightcurve": c_router.produce_ecc_curves_no_spots,
        }
        if self.route not in routes:
            msg = "Original router selected an unvalidated device-integration route; use complete CPU"
            raise UnsupportedNativeIntegrationError(msg)
        return routes[self.route](self.binary, self.callback, labels, **kwargs)


class NativePhysicsIntegrator:
    """Serial original binary physics with one fresh all-band GPU batch per callback.

    The validated surface expression integrates actual original CPU-selected
    faces, radiance/LD, cosines and coverage, including spot faces. All current
    bands/components share a power-of-two padded batch. Nothing physical is
    reused across candidates; only kernels/shapes are reused. CPU native curve
    approximations, when enabled, retain exactly their upstream meaning.
    Pulsations, multiprocessing/custom binary subclasses and bounded resource
    failures need complete CPU fallback. No fitting or CPU defaults change.
    """

    def __init__(self, law: str, *, device: str = "gpu") -> None:
        self.kernel = OrderedSurfaceFluxIntegrator(law, device=device)

    def __call__(
        self, observer: Any, phases: np.ndarray, *, normalize: bool, diagnostics: bool = False,
    ) -> dict[str, Any]:
        if not self.kernel._jax.config.x64_enabled:  # noqa: SLF001
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        return self.integrate_with(
            observer, phases, normalize=normalize, diagnostics=diagnostics,
            ncoeff=self.kernel._ncoeff, prepare=self.kernel.prepare,
        )

    @staticmethod
    def integrate_with(
        observer: Any, phases: np.ndarray, *, normalize: bool, ncoeff: int,
        prepare: Callable, diagnostics: bool = False,
    ) -> dict[str, Any]:
        """Route original CPU physics through an explicit surface packet evaluator.

        The evaluator must perform the ordered FP64 reduction and validate the
        supplied faces. Keeping it outside the router lets ensemble workers send
        their fresh surfaces to a single device owner without inheriting CUDA.
        """
        binary = observer._system  # noqa: SLF001 - per-call Observer owned by hybrid adapter
        if type(binary) is not BinarySystem or binary.has_pulsations():
            msg = "Native integration bridge requires original nonpulsating BinarySystem physics; use complete CPU"
            raise UnsupportedNativeIntegrationError(msg)
        if settings.NUMBER_OF_PROCESSES != 1:
            msg = "Native device integration bridge requires NUMBER_OF_PROCESSES=1"
            raise RuntimeError(msg)
        bands = tuple(observer.passband)
        if not 1 <= len(bands) <= _MAX_BANDS:
            msg = "Native device integration supports 1 to 8 passbands; use complete CPU"
            raise UnsupportedNativeIntegrationError(msg)
        preparation = evaluation = 0.0
        calls, max_width = 0, 0
        records = []

        def collect(curves: Any, index: int, labels: list[str], system: Any) -> Any:
            nonlocal preparation, evaluation, calls, max_width
            start = perf_counter()
            if tuple(labels) != bands or calls >= _MAX_CALLBACKS:
                msg = "Native integration band/callback capacity exceeded; use whole CPU curve"
                raise UnsupportedNativeIntegrationError(msg)
            stars = [system.primary, system.secondary]
            counts = [len(star.indices) for star in stars]
            width = max([1, *counts])
            if max(len(s.faces) for s in stars) > _MAX_FACES:
                msg = "Native integration mesh exceeds 8192 faces per star; use whole CPU curve"
                raise UnsupportedNativeIntegrationError(msg)
            width = 2**(width-1).bit_length()
            radiance, mu, coverage = (np.zeros((2*len(bands), width)) for _ in range(3))
            coefficients = np.zeros((2*len(bands), width, ncoeff))
            for b, band in enumerate(bands):
                for comp, star in enumerate(stars):
                    ix, count = star.indices, counts[comp]
                    row = 2*b+comp
                    radiance[row, :count] = star.normal_radiance[band][ix]
                    mu[row, :count], coverage[row, :count] = star.los_cosines[ix], star.coverage[ix]
                    coefficients[row, :count] = star.ld_cfs[band][ix]
            try:
                packet = prepare(radiance, mu, coverage, coefficients, counts*len(bands))
            except ValueError as exc:
                raise UnsupportedNativeIntegrationError(str(exc)) from exc
            preparation += perf_counter()-start
            start = perf_counter()
            face_device, totals_device = packet.evaluate()
            totals = np.asarray(totals_device)
            # Every selected face participates in an IEEE addition tree without
            # clipping/nan replacement: nonfinite used faces cannot yield a
            # finite total. Padding is identically zero. Download full faces
            # only for explicit diagnostics; complete host curves need totals.
            if not np.isfinite(totals).all():
                msg = "Nonfinite device surface integration; use complete original CPU curve"
                raise UnsupportedNativeIntegrationError(msg)
            totals = totals.reshape((len(bands), 2))
            for b, band in enumerate(bands):
                curves[band][index] = totals[b, 0] + totals[b, 1]
            evaluation += perf_counter()-start
            calls += 1
            max_width = max(max_width, width)
            if diagnostics:
                records.append(NativeSurfaceRecord(system.copy(), packet, np.asarray(face_device).copy(),
                                                    totals.copy(), tuple(counts)))
            return curves

        routed = _RoutedBinary(binary, collect)
        started = perf_counter()
        observer._system = routed  # noqa: SLF001
        try:
            returned_phases, fluxes = observer.lc(phases=phases, normalize=normalize)
        finally:
            observer._system = binary  # noqa: SLF001 - restore on errors and resource fallback too
        total = perf_counter()-started
        return {"phases": returned_phases, "fluxes": fluxes,
                "timings": {"cpu_native_physics_routing": total-preparation-evaluation,
                            "surface_snapshot_upload": preparation, "device_flux_download": evaluation,
                            "native_observer_total": total},
                "diagnostics": {"route": routed.route, "callbacks": calls, "max_padded_faces": max_width,
                                "records": records} if diagnostics else None}
