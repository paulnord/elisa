"""Independent CPU diagnostic with tiled original reflection primitives.

No global function/settings patch. This composes original mesh/gravity/thermal,
NumPy/Numba distance/gamma/LD and observer curve routers. It is validated against
unmodified dense ELISa through2 degrees before use as a finer finite reference.
"""

# ruff: noqa: SLF001 - owned diagnostic Observer delegate
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from elisa import const, ld, settings, utils
from elisa.binary_system import dynamic
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.curves import c_router
from elisa.binary_system.curves.lc_point import compute_lc_on_pos
from elisa.binary_system.surface import faces
from elisa.numba_functions import reflection_effect as re_numba

NAMES = ("primary", "secondary")
_MAX_FACES = 40000


def cpu_reflect(system: Any, iterations: int, *, block_size: int = 32) -> Any:
    """Apply the original full-donor sum using independent CPU primitive tiles."""
    if not settings.REFLECTION_EFFECT or iterations <= 0:
        return system
    stars = [getattr(system, n) for n in NAMES]
    if max(len(star.faces) for star in stars) > _MAX_FACES:
        msg = "Fine CPU diagnostic exceeds40000 faces per component"
        raise ValueError(msg)
    xlim = faces.faces_visibility_x_limits(stars[0].polar_radius, stars[1].polar_radius, 1.)
    visible, symmetric, ldc, factor, integral = [], [], [], [], []
    for name, star in zip(NAMES, stars, strict=True):
        v, s = faces.get_visibility_tests(star.face_centres, xlim[name], name, system.morphology, q_test=True)
        visible.append(np.flatnonzero(v))
        symmetric.append(np.flatnonzero(s))
        coefficients = ld.get_bolometric_ld_coefficients(star.temperatures, star.log_g, star.metallicity,
                                                        star.limb_darkening_coefficients)
        ldc.append(coefficients)
        integral.append(ld.calculate_integrated_limb_darkening_factor(settings.LIMB_DARKENING_LAW, coefficients))
        factor.append(np.ones(len(star.faces)))
    order = [0, 1] if stars[0].t_eff <= stars[1].t_eff else [1, 0]
    for _ in range(iterations):
        for receiver in order:
            donor = 1-receiver
            a, b = stars[receiver], stars[donor]
            vis, sym = visible[donor], symmetric[receiver]
            vector = factor[donor][vis]*b.temperatures[vis]**4*b.areas[vis]
            for start in range(0, len(sym), block_size):
                rows = sym[start:start+block_size]
                distance, join = utils.calculate_distance_matrix(a.face_centres[rows], b.face_centres[vis],
                                                                 return_join_vector_matrix=True)
                ga = np.maximum(re_numba.gamma_primary(a.normals[rows], join), 0)
                gb = np.maximum(re_numba.gamma_secondary(b.normals[vis], join), 0)
                directional = ld.limb_darkening_factor(coefficients=ldc[donor][:, :len(vis)].T,
                                                       cos_theta=gb.T, limb_darkening_law=settings.LIMB_DARKENING_LAW).T
                coupling = ga*gb/distance**2*directional
                factor[receiver][rows] = 1+(a.albedo/integral[receiver][rows]/a.temperatures[rows]**4)*(coupling@vector)
            factor[receiver] = a.mirror_face_values(factor[receiver])
    for i, star in enumerate(stars):
        rows = symmetric[i]
        star.temperatures[rows] *= factor[i][rows]**.25
        star.temperatures = star.mirror_face_values(star.temperatures)
    return system


def intrinsic(binary: Any, mesh: Any = None) -> Any:
    """Build original CPU geometry/thermal state without invoking dense reflection."""
    system = OrbitalPositionContainer.from_binary_system(binary, const.Position(0, 1., 0., 0., 0.))
    system.build_mesh(components_distance=1.)
    if mesh is None:
        system.build_faces_and_kinematic_quantities(components_distance=1.)
    else:
        system.build_faces(components_distance=1.)
        mesh.apply(system)
        system.build_velocities(components_distance=1.)
        system.build_surface_gravity(components_distance=1.)
        system.build_faces_orientation(components_distance=1.)
        system.correct_mesh(components_distance=1.)
        system.build_surface_areas()
    # Calling each component separately invokes original intrinsic normalization
    # and avoids only the all-components reflection dispatch, with no setting patch.
    for name in NAMES:
        system.build_temperature_distribution(components_distance=1., component=name)
    return system


class _ReferenceBinary:
    def __init__(self, binary: Any, mesh: Any = None) -> None:
        self.binary, self.mesh = binary, mesh

    def __getattr__(self, name: str) -> Any:
        return getattr(self.binary, name)

    def compute_lightcurve(self, **kwargs: Any) -> Any:
        system = cpu_reflect(intrinsic(self.binary, self.mesh), settings.REFLECTION_EFFECT_ITERATIONS)
        system.flat_it()
        phases, reverse = dynamic.phase_crv_symmetry(system, kwargs.pop("phases"))
        curves = c_router.produce_circular_sync_curves(self.binary, system, phases, compute_lc_on_pos,
                                                      list(kwargs["passband"]), **kwargs)
        return {b: values[reverse] for b, values in curves.items()}


def observe(observer: Any, phases: Any, *, normalize: bool = True, mesh: Any = None) -> Any:
    """Observe a fresh owned circular detached unspotted finite CPU reference."""
    owned = deepcopy(observer)
    binary = owned._system
    if (binary.eccentricity != 0 or binary.morphology != "detached" or binary.has_spots()
            or binary.has_pulsations() or settings.NUMBER_OF_PROCESSES != 1):
        msg = "Fine CPU reference requires serial circular detached unspotted/nonpulsating binary"
        raise ValueError(msg)
    owned._system = _ReferenceBinary(binary, mesh)
    try:
        return owned.lc(phases=np.asarray(phases), normalize=normalize)
    finally:
        owned._system = binary
