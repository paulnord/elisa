"""Luminosity-preserving donor groups and interactions in the finite likelihood enclosure."""

import numpy as np
from numpy.testing import assert_allclose

from elisa import ld, settings
from elisa.binary_system.surface.faces import faces_visibility_x_limits, visibility_test
from elisa.experimental.area_reflection import AreaReflection, _emission
from scripts.compare_reflection_resolution import DEPTH, DONOR, FIELD, envelope
from scripts.validate_gpu_atmosphere import fixture, table_settings
from scripts.validate_reflection_area import intrinsic_state


def test_merged_donor_matches_sum_of_each_emitting_face_at_common_angle():
    with table_settings(), settings.override(LIMB_DARKENING_LAW="linear", USE_SINGLE_LD_COEFFICIENTS=False):
        binary = fixture({"inclination": 85})
        system, _ = intrinsic_state(binary, 1)
        s = system.secondary
        limits = faces_visibility_x_limits(system.primary.polar_radius, s.polar_radius, 1.0)
        solver = AreaReflection("linear", device="reference", donor_level=0)
        surface = solver._surface(binary, system, "secondary", 1, limits["secondary"])
        f = np.flatnonzero(visibility_test(s.face_centres, limits["secondary"], "secondary"))
        baseline = s.temperatures[: s.base_symmetry_faces_number] ** 4
        heat = baseline * (0.07 + 0.06 * np.sin(np.arange(len(baseline))))
        energy = (s.temperatures[f] ** 4 + heat[s.face_symmetry_vector[f]]) * s.areas[f]
        coeff = ld.get_bolometric_ld_coefficients(s.temperatures, s.log_g, s.metallicity, s.limb_darkening_coefficients)
        grouped, avg = _emission(np, surface, heat)
        assert_allclose(grouped.sum(), energy.sum(), rtol=3e-14)
        for mu in [0.01, 0.5, 0.97]:
            directional = ld.limb_darkening_factor(
                coefficients=coeff[:, : len(f)].T, cos_theta=np.full(len(f), mu), limb_darkening_law="linear"
            )
            by_group = ld.limb_darkening_factor(
                coefficients=avg.T, cos_theta=np.full(len(grouped), mu), limb_darkening_law="linear"
            )
            assert_allclose(grouped * by_group, np.bincount(surface.group, weights=directional * energy), rtol=3e-13)


def test_eight_corners_include_pure_three_way_term_and_cancellation():
    singles = {FIELD: np.array([0.1, -0.4]), DONOR: np.array([0.2, 0.1]), DEPTH: np.array([-0.3, 0.2])}
    interaction = np.array([0.8, -0.5])
    changes = {m: sum(v for bit, v in singles.items() if bit & m) for m in range(1, 8)}
    changes[7] += interaction
    limits, terms = envelope(changes, {k: abs(v) for k, v in singles.items()})
    assert_allclose(terms[7], interaction, atol=1e-14)
    assert_allclose(limits, sum(abs(v) for v in singles.values()) + abs(interaction), atol=1e-13)
