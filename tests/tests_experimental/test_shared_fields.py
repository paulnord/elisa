"""Independent angular ancestry, original field physics and shared-curve controls."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import Observer, settings
from elisa.experimental.boundary_geometry import BoundaryRule
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from elisa.experimental.gpu_boundary import BoundaryBinaryLightCurve
from elisa.experimental.gpu_curved import curved_difference
from elisa.experimental.gpu_nested_mesh import NestedMesh
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.gpu_selective_mesh import seed_system
from elisa.experimental.gpu_shared_fields import (
    FieldVariant,
    SharedFieldLightCurve,
    shared_area_moments,
    shared_parent_map,
)
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture

from .test_gpu_binary import require_device
from .test_gpu_boundary import CASES
from .test_gpu_likelihood import fp64  # noqa: F401


def angular(binary: object, level: int, name: str) -> np.ndarray:
    s = seed_system(binary)
    NestedMesh(level).apply(s)
    star = getattr(s, name)
    rays = star.points-np.array([0. if name == "primary" else 1., 0, 0])
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    return rays[star.faces]


def solid_angle(t: np.ndarray) -> np.ndarray:
    a, b, c = t.transpose(1, 0, 2)
    determinant = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
    denominator = 1+np.einsum("ij,ij->i", a, b)+np.einsum("ij,ij->i", b, c)+np.einsum("ij,ij->i", c, a)
    return 2*np.arctan2(determinant, denominator)


@pytest.mark.parametrize("name", ["primary", "secondary"])
@pytest.mark.parametrize(("coarse_level", "fine_level"), [(0, 1), (1, 2), (1, 3), (2, 2)])
def test_nested_parent_map_matches_independent_great_circle_membership_and_solid_angle(
    name: str, coarse_level: int, fine_level: int,
) -> None:
    b = fixture({"inclination": 85, "discretization": 12, "q": .5, "spin": [.5, 1.5]})
    coarse, fine = angular(b, coarse_level, name), angular(b, fine_level, name)
    parent, diagnostic = shared_parent_map(fine, coarse, fine_level-coarse_level)
    assert not parent.flags.writeable
    assert diagnostic["maximum_condition"] < 1e5
    # Independent half-space oracle uses great-circle edge normals, not inverse
    # coefficients, and searches all coarse cells for random interior rays.
    edges = np.cross(coarse, np.roll(coarse, -1, axis=1))
    orientation = np.sign(np.einsum("nij,nij->ni", edges, np.roll(coarse, 1, axis=1)))
    chosen = np.random.default_rng(1501).choice(len(fine), 15, replace=False)
    for i in chosen:
        point = fine[i].sum(axis=0)
        member = np.flatnonzero(np.all((edges@point)*orientation > 0, axis=1))
        assert np.array_equal(member, [parent[i]])
    grouped = np.bincount(parent, weights=solid_angle(fine), minlength=len(coarse))
    assert_allclose(grouped, solid_angle(coarse), rtol=2e-12, atol=3e-16)
    assert_allclose(grouped.sum(), 4*np.pi, rtol=3e-15, atol=0)


def test_parent_mismatch_and_nonfinite_arrays_abort_instead_of_nearest_interpolation() -> None:
    b = fixture({"inclination": 85, "discretization": 12})
    a, f = angular(b, 0, "primary"), angular(b, 1, "primary")
    with pytest.raises(UnsupportedCurvedGeometryError, match="cone"):
        shared_parent_map(f, a[::-1], 1)
    f[0] = np.nan
    with pytest.raises(UnsupportedCurvedGeometryError, match="finite"):
        shared_parent_map(f, a, 1)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("case", CASES)
def test_complete_shared_fields_and_correction_variants_match_original_cpu(device: str, case: dict) -> None:
    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2,
                                           LIMB_DARKENING_LAW=case.get("law", "linear")):
        b = table_fixture(case) if case.get("atlas") == "ck04" else fixture({"discretization": 12, **case})
        variants = (FieldVariant(0, 0), FieldVariant(0, 1), FieldVariant(1, 1))
        bands, phases = ["Generic.Bessell.V", "bolometric"], [.017, .027, .25, .517, 1.027]
        kwargs = {"integration_level": 1, "radiance": "cpu" if case.get("atlas") == "ck04" else "device"}
        a, c = [SharedFieldLightCurve(bands, BoundaryRule(3, 1), variants, device=d, **kwargs)(
            b, phases, source_revision="assets", diagnostics=True,
        ) for d in ["reference", device]]
        assert a.quadrature_sha256 == c.quadrature_sha256
        assert a.diagnostics["phase_records"] == c.diagnostics["phase_records"]
        for v in variants:
            for name in ["primary", "secondary"]:
                original = getattr(a.diagnostics["initial_systems"][v.key], name)
                actual = getattr(c.diagnostics["initial_systems"][v.key], name)
                for k in ["points", "normals", "face_centres", "areas", "temperatures", "log_g"]:
                    assert_allclose(getattr(actual, k), getattr(original, k), rtol=3e-13, atol=64*np.finfo(float).eps)
                for band in bands:
                    assert_allclose(actual.normal_radiance[band], original.normal_radiance[band], rtol=3e-13, atol=0)
            for band in bands:
                for key in ["component_fluxes", "raw_fluxes", "fluxes"]:
                    assert_allclose(getattr(c.curves[v.key], key)[band], getattr(a.curves[v.key], key)[band],
                                    rtol=3e-13, atol=2e-12 if key == "fluxes" else 0)
            assert c.curves[v.key].diagnostics["quadrature_sha256"] == c.quadrature_sha256
        # Correction ablation keeps the original cached pre-correction state;
        # both original and device paths recompute all later field dependencies.
        one, two = (a.diagnostics["initial_systems"][v.key].primary for v in variants[:2])
        assert_allclose(one.face_centres, two.face_centres, rtol=0, atol=0)
        assert_allclose(one.log_g, two.log_g, rtol=0, atol=0)
        assert not np.array_equal(one.areas, two.areas)


def test_single_variant_matches_preceding_complete_rule_and_preserves_cpu_default() -> None:
    with table_settings(), settings.override(REFLECTION_EFFECT=True):
        b = fixture({"inclination": 85, "discretization": 12})
        bands, phases, rule = ["bolometric"], [.017, .027, .25, .517], BoundaryRule(3, 2)
        observer = Observer(passband=bands, system=b)
        before = observer.lc(phases=phases, normalize=True)[1]
        a = SharedFieldLightCurve(bands, rule, (FieldVariant(1, 1),), integration_level=1, device="reference")(
            b, phases, observer=observer, source_revision="assets",
        ).curves["field1_correction1"]
        old = BoundaryBinaryLightCurve(bands, rule, field_levels=1, device="reference")(
            b, phases, observer=observer, source_revision="assets",
        )
        for key in ["raw_fluxes", "fluxes", "component_fluxes"]:
            assert_allclose(getattr(a, key)["bolometric"], getattr(old, key)["bolometric"], rtol=0, atol=0)
        assert_allclose(observer.lc(phases=phases, normalize=True)[1]["bolometric"], before["bolometric"],
                        rtol=0, atol=0)


def test_changed_physics_response_provenance_and_finite_accuracy_stops() -> None:
    with table_settings(), settings.override(REFLECTION_EFFECT=True):
        bands, variants = ["Generic.Bessell.V"], (FieldVariant(0, 0), FieldVariant(1, 1))
        engine = SharedFieldLightCurve(bands, BoundaryRule(3, 1), variants, integration_level=1, device="cpu")
        b = fixture({"inclination": 85, "discretization": 12})
        phases = [.017, .027, .25, .517]
        a = engine(b, phases, source_revision="assets")
        c = engine(fixture({"inclination": 87, "temperatures": [6600, 5100], "discretization": 12}),
                   phases, source_revision="assets")
        assert a.quadrature_sha256 != c.quadrature_sha256
        x, y = a.curves.values()
        assert x.signature != c.curves[variants[0].key].signature
        reducer = {bands[0]: np.array([3, 0, 1, 1])}
        # Four sparse observations can pass1e-4; this guard test requests1e-6
        # nominal uncertainty, while nuisance inflation must not evade its gate.
        with settings.override(PENALIZE_ALL_EXCEPTIONS=True), pytest.raises(ResolutionNotQualifiedError):
            curved_difference(x, y, 1e-6, reducers=reducer, mean_normalize=True, ln_f=20., require_budget=True)
        with pytest.raises(ValueError, match="matching physical"):
            curved_difference(x, replace(y, signature="stale"), 1e-4)
        observer = Observer(passband=bands, system=b)
        tab = observer.passband[bands[0]].table.copy(deep=True)
        tab[settings.PASSBAND_DATAFRAME_THROUGHPUT] *= 1.3
        observer.passband[bands[0]].table = tab
        d = engine(b, phases, source_revision="assets", observer=observer)
        assert d.quadrature_sha256 == a.quadrature_sha256
        for v in variants:
            assert d.curves[v.key].signature != a.curves[v.key].signature
            assert_allclose(d.curves[v.key].raw_fluxes[bands[0]], 1.3*a.curves[v.key].raw_fluxes[bands[0]],
                            rtol=3e-13, atol=0)


def test_intrinsic_shared_area_moments_preserve_original_closure_without_renormalizing() -> None:
    with table_settings():
        b = fixture({"inclination": 85, "discretization": 12})
        variants = (FieldVariant(0, 0), FieldVariant(0, 1), FieldVariant(1, 1))
        r = shared_area_moments(b, 1, variants)
        for v in variants:
            for name in ["primary", "secondary"]:
                assert_allclose(r[v.key][name]["original_intrinsic_closure_relative"], 0, rtol=0, atol=3e-13)
        for name in ["primary", "secondary"]:
            assert_allclose(r[variants[0].key][name]["curved_area_SMA2"],
                            r[variants[2].key][name]["curved_area_SMA2"], rtol=3e-15, atol=0)


@pytest.mark.parametrize("args", [(3, 2), (True, 1), (1, -1)])
def test_field_variant_invalid_levels_are_explicit(args: tuple) -> None:
    with pytest.raises(ValueError, match="levels"):
        FieldVariant(*args)


def test_shared_capacity_and_unsupported_cases_never_silently_coarsen() -> None:
    with table_settings():
        with pytest.raises(ValueError, match="unique"):
            SharedFieldLightCurve(["bolometric"], BoundaryRule(), (FieldVariant(1, 1),), integration_level=0)
        engine = SharedFieldLightCurve(["bolometric"], BoundaryRule(3, 3, 10), (FieldVariant(0, 0),),
                                      integration_level=0, device="cpu")
        with pytest.raises(UnsupportedCurvedGeometryError, match="node cap"):
            engine(fixture({"inclination": 85, "discretization": 12}), [0], source_revision="assets")
        with pytest.raises(UnsupportedCurvedGeometryError, match="q in"):
            engine(fixture({"inclination": 85, "q": .4}), [0], source_revision="assets")
