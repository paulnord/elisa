"""Independent expansion of the curved-emitter quadrature and both reflection updates."""

import os

import numpy as np
import pytest

from elisa import settings
from elisa.binary_system.surface import faces
from elisa.experimental import area_reflection
from elisa.experimental.emitting_area import EmittingAreaReflection
from elisa.experimental.gpu_reflection import UnsupportedReflectionError
from scripts.validate_emitter_groups import incident
from scripts.validate_emitting_cell_area import projected_rule
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture
from scripts.validate_reflection_area import NAMES, intrinsic_state


def coupling(r, d, grouped):
    """Expand each source separately, including its individual spatial derivative."""
    plan = grouped.spatial
    group = grouped.group
    lit = np.unpackbits(plan.lit, axis=1, bitorder="little")[:, : len(grouped.emitter)]
    matrix = np.zeros((len(r.intrinsic), len(d.intrinsic)))
    a = d.prefix_ld[0]
    centre = grouped.emitter[group]
    offset = d.emitter - centre
    for i, (recv, h) in enumerate(zip(r.receiver, r.normal, strict=True)):
        mask = np.zeros(len(offset), dtype=bool)
        mask[plan.horizon_face[plan.horizon_row == i]] = True
        use = lit[i, group].astype(bool)
        assert not np.any(mask & use)

        def rays(origin):
            delta = recv - origin
            distance = np.linalg.norm(delta, axis=1)
            u = delta / distance[:, None]
            return u, distance, -(u @ h), np.sum(d.outward * u, axis=1)

        u, dist, mu, b = rays(centre)
        _, r0, mu0, b0 = rays(d.emitter)
        assert np.all(mu[use] > 0) and np.all(b[use] > 0) and np.all(mu0[use] > 0) and np.all(b0[use] > 0)
        exact = np.maximum(mu0, 0) * np.maximum(b0, 0) * (1 - a + a * np.maximum(b0, 0)) / r0**2
        assert np.all(exact[~use & ~mask] == 0)
        k = b * (1 - a + a * b)
        prime = 1 - a + 2 * a * b
        slope = (
            mu * (3 * k + b * prime) * np.sum(u * offset, axis=1)
            + k * (offset @ h)
            - mu * prime * np.sum(d.outward * offset, axis=1)
        ) / dist**3
        kernel = np.where(mask, exact, np.where(use, mu * k / dist**2 + slope, 0))
        parent = r.parent[i]
        matrix[parent] += np.bincount(
            d.visible_base,
            weights=kernel * d.visible_area * r.fraction[i] * r.absorption[parent],
            minlength=len(d.intrinsic),
        )
    return matrix


@pytest.mark.parametrize("law", ["linear", "logarithmic"])
def test_iterated_patch_quadrature(law, monkeypatch):
    device = os.getenv("ELISA_GEOMETRY_TEST_DEVICE", "cpu")
    with table_settings(), settings.override(REFLECTION_EFFECT=True, LIMB_DARKENING_LAW=law):
        binary = fixture(dict(discretization=14, temperatures=[4900, 6400], inclination=86, law=law))
        system, tri = intrinsic_state(binary, 1)
        limits = faces.faces_visibility_x_limits(system.primary.polar_radius, system.secondary.polar_radius, 1.0)
        op = EmittingAreaReflection(law, device=device, donor_level=0)
        src = [op._surface(binary, system, n, 1, limits[n]) for n in NAMES]
        grouped, controls = op._prepare_pair(src, system, limits, 1)
        matrix = []
        for i, name in enumerate(NAMES):
            r, d, g = src[1 - i], src[i], grouped[i]
            star = getattr(system, name)
            keep = faces.visibility_test(star.face_centres, limits[name], name)
            pos, normal, fraction = projected_rule(binary, name, tri[name][keep], 3)
            np.testing.assert_allclose(d.emitter, pos, rtol=3e-13, atol=2e-14)
            np.testing.assert_allclose(d.outward, normal, rtol=3e-13, atol=2e-14)
            np.testing.assert_allclose(d.visible_area.reshape(-1, 3), star.areas[keep, None] * fraction, rtol=3e-13)
            if law == "linear":
                assert controls[i]["extra_native_pairs"] > 0
                matrix.append(coupling(r, d, g))
            else:
                assert g.spatial is None and len(g.emitter) == 3 * np.count_nonzero(keep)
        heat = [np.zeros_like(d.intrinsic) for d in src]
        for _ in range(2):
            for i in [0, 1]:
                r, d = src[i], src[1 - i]
                light = d.intrinsic + heat[1 - i]
                if law == "linear":
                    heat[i] = matrix[1 - i] @ light
                else:
                    exact = incident(
                        r,
                        np.arange(len(r.parent)),
                        d.emitter,
                        d.outward,
                        d.visible_area * light[d.visible_base],
                        d.prefix_ld,
                    )
                    heat[i] = np.bincount(r.parent, weights=exact * r.fraction, minlength=len(r.intrinsic))
        temps, stats = op.evaluate(binary, system, 1, 2)
        for i, name in enumerate(NAMES):
            star = getattr(system, name)
            expected = (src[i].intrinsic + heat[i]) ** 0.25
            np.testing.assert_allclose(temps[name], expected[star.face_symmetry_vector], rtol=3e-13)
        monkeypatch.setattr(area_reflection, "_MAX_PAIRS", stats["primary"]["pairs_per_sweep_both_stars"] - 1)
        with pytest.raises(UnsupportedReflectionError, match="pair cap"):
            op.evaluate(binary, system, 1, 2)
