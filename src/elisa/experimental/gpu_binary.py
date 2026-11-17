"""Functional HYBRID observed binary light curves, with explicit CPU fallbacks.

CPU candidate mesh/topology -> device corrected state -> CPU reflection/custom
LD -> device blackbody, optical-depth radiance and projection/clipping/integration
with a CPU hull bridge.
Host bridges are intentional and timed. This is not a resident forward model
and does not alter Observer or any existing CPU fitting/default interface.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from elisa import const, settings
from elisa.binary_system import dynamic
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.curves.utils import get_limbdarkening_cfs, prep_surface_params
from elisa.binary_system.surface.coverage import compute_surface_coverage
from elisa.binary_system.surface.gravity import calculate_polar_potential_gradient_magnitude
from elisa.binary_system.surface.temperature import apply_reflection_effect
from elisa.binary_system.utils import move_sys_onpos
from elisa.experimental.gpu_occultation import DeviceEclipseIntegrator, UnsupportedOccultationError
from elisa.experimental.gpu_radiance import DeviceBlackbodyRadiance, UnsupportedRadianceError, passband_revision
from elisa.experimental.gpu_state import CorrectedSurfaceState
from elisa.experimental.gpu_surface import SurfaceFluxIntegrator
from elisa.observer import utils as observer_utils
from elisa.observer.observer import Observer

_COMPONENTS = ("primary", "secondary")
_Q_MIN, _Q_MAX = 0.5, 2.0
_SPIN_MIN, _SPIN_MAX = 0.5, 1.5


class UnsupportedHybridModelError(ValueError):
    """A device evaluation is unsupported, not a physical prior rejection."""


@dataclass(frozen=True)
class HybridCurveResult:
    """Host curves in original phase order, plus disclosed execution costs.

    component_fluxes are intrinsic observer-direction luminosities before third
    light/distance/normalization, mapped to the original observation order.
    They are absent on full CPU fallback. diagnostics are opt-in private-in-
    memory intermediate containers for scientific validation, not a cache.
    """

    phases: np.ndarray
    fluxes: dict[str, np.ndarray]
    component_fluxes: dict[str, np.ndarray]
    backend: str
    fallback_reason: str | None
    timings: dict[str, float]
    diagnostics: dict[str, Any] | None = None


class HybridBinaryLightCurve:
    """Rebuild all dependent physical state for every supplied BinarySystem.

    Supported hybrid domain: circular detached, unspotted/nonpulsating binaries,
    q in [0.5,2], spin in [0.5,1.5], raw own-centre radii [1e-6,.25] SMA,
    beta in [0,1], blackbody and explicit per-band AND bolometric LD. Reflection
    uses the original CPU implementation and current settings. Real atmosphere
    tables, tabulated LD, eccentric/contact systems and spots use unmodified
    Observer when fallback='cpu' (default), or raise UnsupportedHybridModelError.
    Missing GPU/FP64 never silently changes the requested device.

    Only explicit phases and W/m2 or Observer maximum-normalized output are
    exposed here; gpu_fit adapts the original physical fitting synthesis hook.
    No physical curve is reused across calls; compilation reuse depends on
    shapes and fixed spectral quadrature is checked against fresh responses.
    coverage="device" uses GPU projection/clipping with a fresh CPU hull/visibility
    bridge. coverage="cpu" retains the preceding per-phase CPU coverage path.
    radiance="device" streams fresh Planck/optical-depth emission directly into
    eclipse integration. radiance="cpu" retains the preceding CPU spectral path.
    """

    def __init__(
        self, passbands: list[str], *, device: str = "gpu", fallback: str = "cpu", coverage: str = "device",
        radiance: str = "device",
    ) -> None:
        if fallback not in ("cpu", "raise"):
            msg = "fallback must be 'cpu' or 'raise'"
            raise ValueError(msg)
        if not passbands or len(set(passbands)) != len(passbands):
            msg = "passbands must be a nonempty list of unique names"
            raise ValueError(msg)
        if coverage not in ("cpu", "device"):
            msg = "coverage must be 'cpu' or 'device'"
            raise ValueError(msg)
        if radiance not in ("cpu", "device"):
            msg = "radiance must be 'cpu' or 'device'"
            raise ValueError(msg)
        self.coverage = coverage
        self.radiance = radiance
        self.passbands = tuple(passbands)
        self.fallback = fallback
        self._states = {name: CorrectedSurfaceState(name, device=device) for name in _COMPONENTS}
        self.device = self._states["primary"].device
        self._integrators: dict[str, SurfaceFluxIntegrator] = {}
        self._occultations: dict[str, DeviceEclipseIntegrator] = {}
        self._radiance_engine: tuple[tuple[str, str], DeviceBlackbodyRadiance] | None = None

    def _unsupported_reason(self, binary: Any) -> str | None:  # noqa: PLR0911 - explicit fallback reasons
        if self.coverage == "device" and settings.INTERSECTION_BACKEND not in ("sutherland_hodgman", "reference"):
            return ("Device clipping preserves Sutherland-Hodgman/reference semantics; "
                    "other intersection backends use CPU")
        if binary.eccentricity != 0 or binary.morphology != "detached":
            return "Hybrid state requires a circular detached binary; eccentric/contact physics uses CPU"
        if not _Q_MIN <= binary.mass_ratio <= _Q_MAX:
            return "Mass ratio outside the validated device domain [0.5,2]"
        for name in _COMPONENTS:
            star = getattr(binary, name)
            if star.has_spots() or star.has_pulsations():
                return "Spots/pulsations require the full CPU model"
            if not _SPIN_MIN <= star.synchronicity <= _SPIN_MAX or not 0 <= star.gravity_darkening <= 1:
                return "Spin/beta outside the validated device domain"
            if star.atmosphere != "bb":
                return "Atmosphere-table state is not validated in the hybrid path; use CPU"
            if star.limb_darkening_coefficients is None or any(
                band not in star.limb_darkening_coefficients for band in (*self.passbands, "bolometric")
            ):
                return "Tabulated/missing limb darkening requires the full CPU model"
        return None

    def build_intrinsic_system(self, binary: Any) -> tuple[Any, dict[str, float]]:
        """Fresh CPU mesh/kinematics, device state, then CPU reflection.

        Returned container owns its arrays and has not yet been rotated. This
        method never performs a fallback: unsupported states raise explicitly.
        The public curve call owns the full-model fallback decision.
        """
        system, timings, _ = self._build_intrinsic_system(binary)
        return system, timings

    def _build_intrinsic_system(self, binary: Any) -> tuple[Any, dict[str, float], Any]:
        reason = self._unsupported_reason(binary)
        if reason:
            raise UnsupportedHybridModelError(reason)
        started = perf_counter()
        system = OrbitalPositionContainer.from_binary_system(binary, const.Position(0, 1.0, 0.0, 0.0, 0.0))
        system.build_mesh(components_distance=1.0)
        system.build_faces(components_distance=1.0)
        system.build_velocities(components_distance=1.0)
        mesh_done = perf_counter()
        device_temperatures = {}
        for name, kernel in self._states.items():
            star = getattr(system, name)
            prepared = kernel.prepare(star, binary.mass_ratio, system.primary.mass, binary.semi_major_axis)
            # Deliberate device -> CPU bridge for reflection/atmosphere/clipping.
            on_device = prepared.evaluate()
            result = kernel._jax.tree.map(np.asarray, on_device)  # noqa: SLF001
            if not bool(result["proposal_valid"]):
                msg = f"Unsupported corrected {name} state (geometry/gravity/thermal masks); use full CPU"
                raise UnsupportedHybridModelError(msg)
            device_temperatures[name] = on_device["temperatures"]
            for key in ("points", "face_centres", "normals", "areas",
                        "potential_gradient_magnitudes", "log_g", "temperatures"):
                setattr(star, key, np.array(result[key], copy=True))
            star.polar_potential_gradient_magnitude = calculate_polar_potential_gradient_magnitude(
                1.0, binary.mass_ratio, star.polar_radius, name, star.synchronicity,
            )
        state_done = perf_counter()
        apply_reflection_effect(system, 1.0, settings.REFLECTION_EFFECT_ITERATIONS)
        system.flat_it()
        if settings.REFLECTION_EFFECT and settings.REFLECTION_EFFECT_ITERATIONS > 0:
            # CPU reflection changed the intrinsic arrays. Re-upload its final
            # temperatures; retaining old device temperatures would be stale.
            device_temperatures = None
        return system, {
            "cpu_mesh_topology_kinematics": mesh_done - started,
            "device_state_with_transfers_compile": state_done - mesh_done,
            "cpu_reflection": perf_counter() - state_done,
        }, device_temperatures

    def _prepare_radiance(
        self, system: Any, observer: Any, device_temperatures: Any, *, diagnostics: bool,
    ) -> tuple[Any, Any]:
        """Fresh LD/temperature snapshot, revision-checked fixed quadrature reuse."""
        ld_cfs = get_limbdarkening_cfs(system, passband=observer.passband)
        key = (settings.LIMB_DARKENING_LAW, passband_revision(observer.passband))
        if self._radiance_engine is None or self._radiance_engine[0] != key:
            engine = DeviceBlackbodyRadiance(observer.passband, key[0], device=self.device.platform)
            self._radiance_engine = key, engine  # one fixed response revision; no physical cache
        engine = self._radiance_engine[1]
        packet = engine.prepare(system, ld_cfs, device_temperatures=device_temperatures)
        radiance = packet.evaluate()
        if not np.asarray(radiance["proposal_valid"]).all():
            msg = "Unsupported fresh device spectral/optical-depth state; use full CPU"
            raise UnsupportedRadianceError(msg)
        for name in _COMPONENTS:
            getattr(system, name).ld_cfs = ld_cfs[name]
        if diagnostics or self.coverage == "cpu":
            values = np.asarray(radiance["normal_radiance"])
            for comp, name in enumerate(_COMPONENTS):
                star = getattr(system, name)
                star.normal_radiance = {band: values[comp, b, :len(star.faces)].copy()
                                        for b, band in enumerate(self.passbands)}
        detail = {"prepared_batch": packet, "result": radiance, "response_revision": key[1],
                  "temperature_source": "CPU post-reflection upload" if device_temperatures is None
                  else "fresh corrected device temperatures"} if diagnostics else None
        return radiance, detail

    def _integrate_curve(
        self, system: Any, model: Any, motion: Any, in_eclipse: Any, law: str, *, diagnostics: bool,
        device_radiance: Any = None,
    ) -> tuple[Any, Any, Any, Any]:
        if self.coverage == "device":
            if law not in self._occultations:
                self._occultations[law] = DeviceEclipseIntegrator(law, device=self.device.platform)
            result = self._occultations[law](system, motion, in_eclipse, self.passbands, diagnostics=diagnostics,
                                            device_radiance=device_radiance)
            return result["components"], result["timings"], result["positions"], result["details"]
        if law not in self._integrators:
            self._integrators[law] = SurfaceFluxIntegrator(law, device=self.device.platform)
        integrator = self._integrators[law]
        # Bound each host/device batch to ONE phase (two component rows).
        # Candidate/phase batching and fewer host bridges are later optimizations.
        components = {band: np.empty((len(motion), 2)) for band in self.passbands}
        captures = [] if diagnostics else None
        coverage_seconds = integration_seconds = 0.0
        for index, position in enumerate(motion):
            step = perf_counter()
            on_pos = move_sys_onpos(system, position, on_copy=True)
            compute_surface_coverage(on_pos, model.semi_major_axis, in_eclipse=bool(in_eclipse[index]),
                                     return_values=False, write_to_containers=True)
            coverage_seconds += perf_counter() - step
            step = perf_counter()
            for band in self.passbands:
                _, total = integrator.from_containers(
                    band, [on_pos.primary, on_pos.secondary],
                    pad_to=max(len(on_pos.primary.faces), len(on_pos.secondary.faces)),
                )
                components[band][index] = np.asarray(total)
            integration_seconds += perf_counter() - step
            if captures is not None:
                captures.append(on_pos)
        return components, {"cpu_orientation_coverage": coverage_seconds,
                            "device_integration_with_transfers": integration_seconds}, captures, None

    @staticmethod
    def _observed_fluxes(components: dict[str, np.ndarray], model: Any, *, normalize: bool) -> dict[str, np.ndarray]:
        curves = {band: np.sum(values, axis=1) for band, values in components.items()}
        if not all(np.isfinite(value).all() for value in curves.values()):
            msg = "Nonfinite hybrid integrated flux; no likelihood/prior value may be inferred"
            raise FloatingPointError(msg)
        # Match Observer: remap repeats first, THEN additional light on that mean.
        curves = {band: values + values.mean() * model.additional_light / (1 - model.additional_light)
                  for band, values in curves.items()}
        if normalize:
            curves, _ = observer_utils.normalize_light_curve(curves, kind="maximum", top_fraction_to_average=0.0)
            return curves
        return observer_utils.adjust_flux_for_distance(curves, model.distance)

    def __call__(
        self, binary: Any, phases: Any, *, normalize: bool = False, diagnostics: bool = False,
        observer: Any | None = None,
    ) -> HybridCurveResult:
        """Return curves, optionally retaining a prepared Observer's passband state.

        The supplied Observer is copied before assigning this candidate. Its
        passbands/order must match the engine; response tables and bandwidths
        are preserved. This is the fitting synthesis hook's Observer contract.
        Full elapsed time includes the copy and all host/device bridges.
        """
        if not self._states["primary"]._jax.config.x64_enabled:  # noqa: SLF001
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        started = perf_counter()
        phases = np.array(phases, dtype=np.float64, copy=True)
        if phases.ndim != 1 or not phases.size or not np.isfinite(phases).all():
            msg = "phases must be a nonempty finite vector"
            raise ValueError(msg)
        # CPU builders are not allowed to mutate the caller's candidate.
        model = deepcopy(binary)
        if observer is None:
            observer = Observer(passband=list(self.passbands), system=model)
        else:
            if tuple(observer.passband) != self.passbands:
                msg = "Supplied Observer passbands/order must match the hybrid engine"
                raise ValueError(msg)
            observer = deepcopy(observer)
            observer._system = model  # noqa: SLF001 - prepared fitting Observer contract
        try:
            system, timings, device_temperatures = self._build_intrinsic_system(model)
            prepared_at = perf_counter()
            radiance, radiance_detail = None, None
            if self.radiance == "device":
                radiance, radiance_detail = self._prepare_radiance(
                    system, observer, device_temperatures, diagnostics=diagnostics,
                )
                timings["device_radiance_ld_with_transfers"] = perf_counter() - prepared_at
            else:
                prep_surface_params(
                    system, return_values=False, write_to_containers=True, passband=observer.passband,
                    left_bandwidth=observer.left_bandwidth, right_bandwidth=observer.right_bandwidth,
                )
                timings["cpu_radiance_ld"] = perf_counter() - prepared_at
        except (UnsupportedHybridModelError, UnsupportedRadianceError) as exc:
            if self.fallback == "raise":
                raise UnsupportedHybridModelError(str(exc)) from exc
            out_phases, fluxes = observer.lc(phases=phases, normalize=normalize)
            return HybridCurveResult(out_phases, fluxes, {}, "cpu_fallback", str(exc),
                                     {"total": perf_counter() - started})
        resolved = observer.manage_time_series(phases=phases)
        base, origin = observer.phase_interval_reduce(resolved)
        support, reverse = dynamic.phase_crv_symmetry(system, base)
        motion = model.get_positions_method()(input_argument=support, return_nparray=False, calculate_from="phase")
        in_eclipse = dynamic.in_eclipse_test(
            np.array([p.azimuth for p in motion]), dynamic.get_eclipse_boundaries(model, 1.0),
        )
        try:
            components, integration_times, captures, occultation = self._integrate_curve(
                system, model, motion, in_eclipse, settings.LIMB_DARKENING_LAW, diagnostics=diagnostics,
                device_radiance=radiance,
            )
        except UnsupportedOccultationError as exc:
            if self.fallback == "raise":
                raise UnsupportedHybridModelError(str(exc)) from exc
            out_phases, fluxes = observer.lc(phases=phases, normalize=normalize)
            return HybridCurveResult(out_phases, fluxes, {}, "cpu_fallback", str(exc),
                                     {"total": perf_counter() - started})
        timings.update(integration_times)
        mapped_components = {band: values[reverse][origin] for band, values in components.items()}
        curves = self._observed_fluxes(mapped_components, model, normalize=normalize)
        timings["total"] = perf_counter() - started
        detail = {"initial_system": system, "positions": captures, "support_phases": support,
                  "support_to_original": reverse[origin], "in_eclipse": in_eclipse,
                  "occultation": occultation, "coverage_mode": self.coverage, "radiance_mode": self.radiance,
                  "radiance": radiance_detail} if diagnostics else None
        return HybridCurveResult(resolved + model.phase_shift, curves, mapped_components,
                                 "hybrid", None, timings, detail)
