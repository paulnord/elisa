"""Functional HYBRID observed binary light curves, with explicit CPU fallbacks.

CPU candidate mesh/topology -> device corrected state -> device reflection/CPU LD
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
from elisa.experimental.gpu_atmosphere import DeviceAtmosphereRadiance, check_ld_boundaries, check_table_transitions
from elisa.experimental.gpu_native import NativePhysicsIntegrator, UnsupportedNativeIntegrationError
from elisa.experimental.gpu_occultation import DeviceEclipseIntegrator, UnsupportedOccultationError
from elisa.experimental.gpu_radiance import DeviceBlackbodyRadiance, UnsupportedRadianceError, passband_revision
from elisa.experimental.gpu_reflection import DeviceReflection, UnsupportedReflectionError
from elisa.experimental.gpu_reflection_tiled import TiledDeviceReflection
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
    They are absent on full CPU fallback and the native-router bridge (which
    retains original approximation/interpolation of total curves). diagnostics are opt-in private-in-
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
    beta in [0,1], two blackbody OR two CK04 components, custom or CPU-interpolated
    table LD. Source selection/wavelength preparation/LD interpolation use
    original CPU implementations. Mixed/other atlases, eccentric/contact
    systems and spots use unmodified Observer when fallback='cpu' (default), or
    raise UnsupportedHybridModelError.
    Missing GPU/FP64 never silently changes the requested device.

    Only explicit phases and W/m2 or Observer maximum-normalized output are
    exposed here; gpu_fit adapts the original physical fitting synthesis hook.
    No physical curve is reused across calls; compilation reuse depends on
    shapes and fixed spectral quadrature is checked against fresh responses.
    coverage="device" uses GPU projection/clipping with a fresh CPU hull/visibility
    bridge. coverage="cpu" retains the preceding per-phase CPU coverage path.
    radiance="device" streams fresh Planck/optical-depth emission directly into
    eclipse integration. radiance="cpu" retains the preceding CPU spectral path.
    reflection="device" uses the original cooler-first sequential iteration;
    reflection="cpu" retains the preceding CPU-reflection path.
    general="device_integration" opts into broader original CPU physical routers
    with fresh device integration, including their original CPU reflection.
    """

    def __init__(
        self, passbands: list[str], *, device: str = "gpu", fallback: str = "cpu", coverage: str = "device",
        radiance: str = "device",
        general: str = "cpu",
        reflection: str = "device",
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
        if general not in ("cpu", "device_integration"):
            msg = "general must be 'cpu' or 'device_integration'"
            raise ValueError(msg)
        self.general = general
        if reflection not in ("cpu", "device", "tiled"):
            msg = "reflection must be 'cpu', 'device' or 'tiled'"
            raise ValueError(msg)
        self.reflection = reflection
        self.passbands = tuple(passbands)
        self.fallback = fallback
        self._states = {name: CorrectedSurfaceState(name, device=device) for name in _COMPONENTS}
        self.device = self._states["primary"].device
        self._integrators: dict[str, SurfaceFluxIntegrator] = {}
        self._occultations: dict[str, DeviceEclipseIntegrator] = {}
        self._radiance_engine: tuple[tuple[str, str], DeviceBlackbodyRadiance] | None = None
        self._atmosphere_engines: dict[str, DeviceAtmosphereRadiance] = {}
        self._native_integrators: dict[str, NativePhysicsIntegrator] = {}
        self._reflection_engines: dict[str, DeviceReflection] = {}

    def _unsupported_curve(
        self, observer: Any, phases: np.ndarray, *, normalize: bool, diagnostics: bool,
        reason: str, started: float,
    ) -> HybridCurveResult:
        """Explicit opt-in original-physics bridge, otherwise full CPU fallback."""
        if self.general == "device_integration":
            law = settings.LIMB_DARKENING_LAW
            if law not in self._native_integrators:
                self._native_integrators[law] = NativePhysicsIntegrator(law, device=self.device.platform)
            try:
                result = self._native_integrators[law](observer, phases, normalize=normalize,
                                                       diagnostics=diagnostics)
            except UnsupportedNativeIntegrationError as exc:
                reason += "; " + str(exc)
            else:
                timings = dict(result["timings"], total=perf_counter()-started)
                return HybridCurveResult(result["phases"], result["fluxes"], {}, "hybrid_native",
                                         "Original CPU physics bridge: " + reason, timings, result["diagnostics"])
        if self.fallback == "raise":
            raise UnsupportedHybridModelError(reason)
        out_phases, fluxes = observer.lc(phases=phases, normalize=normalize)
        return HybridCurveResult(out_phases, fluxes, {}, "cpu_fallback", reason,
                                 {"total": perf_counter()-started})

    def _unsupported_reason(self, binary: Any) -> str | None:  # noqa: PLR0911 - explicit fallback reasons
        if self.coverage == "device" and settings.INTERSECTION_BACKEND not in ("sutherland_hodgman", "reference"):
            return ("Device clipping preserves Sutherland-Hodgman/reference semantics; "
                    "other intersection backends use CPU")
        if binary.eccentricity != 0 or binary.morphology != "detached":
            return "Hybrid state requires a circular detached binary; eccentric/contact physics uses CPU"
        if not _Q_MIN <= binary.mass_ratio <= _Q_MAX:
            return "Mass ratio outside the validated device domain [0.5,2]"
        atlases = {getattr(binary, name).atmosphere for name in _COMPONENTS}
        if atlases not in ({"bb"}, {"ck04"}):
            return "Hybrid emission supports two blackbody or two CK04 stars; mixed/other atlases use CPU"
        for name in _COMPONENTS:
            star = getattr(binary, name)
            if star.has_spots() or star.has_pulsations():
                return "Spots/pulsations require the full CPU model"
            if not _SPIN_MIN <= star.synchronicity <= _SPIN_MAX or not 0 <= star.gravity_darkening <= 1:
                return "Spin/beta outside the validated device domain"
            if star.limb_darkening_coefficients is not None and any(
                band not in star.limb_darkening_coefficients for band in (*self.passbands, "bolometric")
            ):
                return "Missing custom limb darkening requires the full CPU model"
        return None

    def build_intrinsic_system(self, binary: Any) -> tuple[Any, dict[str, float]]:
        """Fresh CPU mesh/kinematics, device state, then configured reflection.

        Returned container owns its arrays and has not yet been rotated. This
        method never performs a fallback: unsupported states raise explicitly.
        The public curve call owns the full-model fallback decision.
        """
        system, timings, _, _ = self._build_intrinsic_system(binary)
        return system, timings

    def _initial_mesh(self, system: Any) -> None:
        """Build the original fresh raw mesh; explicit experiments may refine it."""
        system.build_mesh(components_distance=1.0)
        system.build_faces(components_distance=1.0)

    def _build_intrinsic_system(  # noqa: C901, PLR0912 - explicit fresh state and reflection dispatch
        self, binary: Any, *, diagnostics: bool = False,
    ) -> tuple[Any, dict[str, float], Any, Any]:
        reason = self._unsupported_reason(binary)
        if reason:
            raise UnsupportedHybridModelError(reason)
        started = perf_counter()
        system = OrbitalPositionContainer.from_binary_system(binary, const.Position(0, 1.0, 0.0, 0.0, 0.0))
        self._initial_mesh(system)
        system.build_velocities(components_distance=1.0)
        mesh_done = perf_counter()
        device_temperatures = {}
        device_states = {}
        for name, kernel in self._states.items():
            star = getattr(system, name)
            prepared = kernel.prepare(star, binary.mass_ratio, system.primary.mass, binary.semi_major_axis)
            # CPU bridge for original visibility/LD/atlas preparation and hulls;
            # retain device arrays for irradiation and fresh spectral emission.
            on_device = prepared.evaluate()
            result = kernel._jax.tree.map(np.asarray, on_device)  # noqa: SLF001
            if not bool(result["proposal_valid"]):
                msg = f"Unsupported corrected {name} state (geometry/gravity/thermal masks); use full CPU"
                raise UnsupportedHybridModelError(msg)
            device_temperatures[name] = on_device["temperatures"]
            device_states[name] = on_device
            for key in ("points", "face_centres", "normals", "areas",
                        "potential_gradient_magnitudes", "log_g", "temperatures"):
                setattr(star, key, np.array(result[key], copy=True))
            star.polar_potential_gradient_magnitude = calculate_polar_potential_gradient_magnitude(
                1.0, binary.mass_ratio, star.polar_radius, name, star.synchronicity,
            )
        state_done = perf_counter()
        active_reflection = settings.REFLECTION_EFFECT and settings.REFLECTION_EFFECT_ITERATIONS > 0
        reflection_detail = None
        if active_reflection:
            check_ld_boundaries(system, ("bolometric",))
        if active_reflection and self.reflection in ("device", "tiled"):
            law = settings.LIMB_DARKENING_LAW
            if law not in self._reflection_engines:
                cls = TiledDeviceReflection if self.reflection == "tiled" else DeviceReflection
                self._reflection_engines[law] = cls(law, device=self.device.platform)
            packet = self._reflection_engines[law].prepare(
                system, settings.REFLECTION_EFFECT_ITERATIONS, device_states=device_states,
            )
            reflected = packet.evaluate()
            if not bool(reflected["proposal_valid"]):
                msg = "Unsupported nonfinite device reflection state; use entire CPU curve"
                raise UnsupportedReflectionError(msg)
            for name, values in zip(_COMPONENTS, reflected["temperatures"], strict=True):
                getattr(system, name).temperatures = np.asarray(values).copy()
                device_temperatures[name] = values
            if diagnostics:
                reflection_detail = {"prepared": packet, "result": reflected}
        else:
            apply_reflection_effect(system, 1.0, settings.REFLECTION_EFFECT_ITERATIONS)
            if active_reflection:
                device_temperatures = None
        system.flat_it()
        if system.primary.atmosphere == "ck04":
            check_table_transitions(system)
        return system, {
            "cpu_mesh_topology_kinematics": mesh_done - started,
            "device_state_with_transfers_compile": state_done - mesh_done,
            ("device_reflection_with_transfers_compile" if active_reflection and self.reflection in ("device", "tiled")
             else "cpu_reflection"): perf_counter() - state_done,
        }, device_temperatures, reflection_detail

    def _prepare_radiance(
        self, system: Any, observer: Any, device_temperatures: Any, *, diagnostics: bool,
    ) -> tuple[Any, Any]:
        """Fresh LD/temperature snapshot, revision-checked fixed quadrature reuse."""
        table = system.primary.atmosphere == "ck04"
        check_ld_boundaries(system, observer.passband)
        ld_cfs = get_limbdarkening_cfs(system, passband=observer.passband)
        revision = None
        if table:
            law = settings.LIMB_DARKENING_LAW
            if law not in self._atmosphere_engines:
                self._atmosphere_engines[law] = DeviceAtmosphereRadiance(self.passbands, law,
                                                                       device=self.device.platform)
            packet = self._atmosphere_engines[law].prepare(
                system, ld_cfs, passband=observer.passband, left_bandwidth=observer.left_bandwidth,
                right_bandwidth=observer.right_bandwidth, device_temperatures=device_temperatures,
            )
        else:
            key = (settings.LIMB_DARKENING_LAW, passband_revision(observer.passband))
            revision = key[1]
            if self._radiance_engine is None or self._radiance_engine[0] != key:
                engine = DeviceBlackbodyRadiance(observer.passband, key[0], device=self.device.platform,
                                                fine_mesh=self.reflection == "tiled")
                self._radiance_engine = key, engine  # one fixed response revision; no physical cache
            packet = self._radiance_engine[1].prepare(system, ld_cfs, device_temperatures=device_temperatures)
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
        detail = {"prepared_batch": packet, "result": radiance, "response_revision": revision,
                  "emission": "CPU-prepared CK04/device quadrature and T4 interpolation" if table else "device Planck",
                  "source_preparation": packet.metadata if table else None,
                  "temperature_source": "CPU post-reflection upload" if device_temperatures is None
                  else ("fresh post-reflection device temperatures"
                        if self.reflection in ("device", "tiled") and settings.REFLECTION_EFFECT
                        and settings.REFLECTION_EFFECT_ITERATIONS > 0
                        else "fresh corrected device temperatures")} if diagnostics else None
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
        if self.general == "device_integration" and settings.NUMBER_OF_PROCESSES != 1:
            msg = "Native device integration requires NUMBER_OF_PROCESSES=1"
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
            system, timings, device_temperatures, reflection_detail = self._build_intrinsic_system(
                model, diagnostics=diagnostics,
            )
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
        except (UnsupportedHybridModelError, UnsupportedRadianceError, UnsupportedReflectionError) as exc:
            return self._unsupported_curve(observer, phases, normalize=normalize, diagnostics=diagnostics,
                                           reason=str(exc), started=started)
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
            return self._unsupported_curve(observer, phases, normalize=normalize, diagnostics=diagnostics,
                                           reason=str(exc), started=started)
        timings.update(integration_times)
        mapped_components = {band: values[reverse][origin] for band, values in components.items()}
        curves = self._observed_fluxes(mapped_components, model, normalize=normalize)
        timings["total"] = perf_counter() - started
        detail = {"initial_system": system, "positions": captures, "support_phases": support,
                  "support_to_original": reverse[origin], "in_eclipse": in_eclipse,
                  "occultation": occultation, "coverage_mode": self.coverage, "radiance_mode": self.radiance,
                  "radiance": radiance_detail, "reflection_mode": self.reflection,
                  "reflection": reflection_detail} if diagnostics else None
        return HybridCurveResult(resolved + model.phase_shift, curves, mapped_components,
                                 "hybrid", None, timings, detail)
