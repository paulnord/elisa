"""Selective topology, full reflected state/curves and independent geometry audit."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import Observer, settings, utils
from elisa.experimental.gpu_mesh_audit import angular_grid, mesh_measures, radial_measures, roche_measures
from elisa.experimental.gpu_nested_mesh import NestedMesh, UnsupportedNestedMeshError
from elisa.experimental.gpu_reference import cpu_reflect, intrinsic, observe
from elisa.experimental.gpu_selective_mesh import SelectiveMesh, choose_plan, seed_system, topology_id
from elisa.experimental.gpu_surface_error import NestedBinaryLightCurve, compare_surfaces
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture

from .test_gpu_binary import require_device
from .test_gpu_likelihood import fp64  # noqa: F401


def plan(binary: Any, fraction: float = .5, *, levels: int = 3, correction: str = "density") -> SelectiveMesh:
    system = seed_system(binary)
    roots, tokens = [], []
    for name in ("primary", "secondary"):
        star = getattr(system, name)
        # Deterministic distributed patches exercise green conformity at seams.
        count = star.base_symmetry_faces_number
        rng = np.random.default_rng(1200)
        roots.append(tuple(sorted(int(i) for i in rng.permutation(count)[:int(np.ceil(count*fraction))])))
        tokens.append(topology_id(star))
    return SelectiveMesh(levels, roots=tuple(roots), topology=tuple(tokens), correction=correction)


@pytest.mark.parametrize("fraction", [0., .15, .5, .85, 1.])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_selective_mesh_conformity_symmetry_roots_and_uniform_endpoints(fraction: float, level: int) -> None:
    binary = fixture({"inclination": 85, "discretization": 12})
    numerical = plan(binary, fraction, levels=level)
    system = seed_system(binary)
    info = numerical.apply(system)
    for name in ("primary", "secondary"):
        star = getattr(system, name)
        edges = np.sort(star.faces[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2)), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        assert np.all(counts == 2)
        assert len(star.points)-len(unique)+len(star.faces) == 2
        assert np.all(utils.triangle_areas(star.faces, star.points) > 0)
        assert 540*4**(level-1) <= len(star.faces) <= 540*4**level
        assert np.array_equal(np.unique(info[name]["parents"]), np.arange(info[name]["root_faces"]))
        assert not info[name]["parents"].flags.writeable
        assert_allclose(star.mirror_face_values(np.arange(star.base_symmetry_faces_number)),
                        star.face_symmetry_vector, rtol=0, atol=0)
        if fraction in (0., 1.):
            original = seed_system(binary)
            NestedMesh(level-1 if fraction == 0 else level).apply(original)
            reference = getattr(original, name)
            assert_allclose(star.points, reference.points, rtol=0, atol=0)
            assert np.array_equal(star.faces, reference.faces)
            assert_allclose(star.discretization_factor, reference.discretization_factor, rtol=0, atol=0)


@pytest.mark.parametrize("case", [{"q": .5, "spin": [.5, 1.5]},
                                  {"q": 2., "spin": [1.5, .5], "potentials": [12, 16]}])
def test_selective_mesh_q_spin_boundary_is_closed(case: dict) -> None:
    binary = fixture({"inclination": 86, "discretization": 12, **case})
    numerical = plan(binary, levels=2)
    system = seed_system(binary)
    numerical.apply(system)
    for name in ("primary", "secondary"):
        star = getattr(system, name)
        edges = np.sort(star.faces[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2)), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        assert np.all(counts == 2)
        audit = mesh_measures(star, name)
        assert audit["raw_volume"] > 0


@pytest.mark.parametrize("order", [4, 12, 32])
def test_independent_angular_jacobian_has_exact_sphere_area_volume(order: int) -> None:
    directions, weights = angular_grid(order)
    r = .12
    result = radial_measures(directions, np.full(len(directions), r), directions*7, weights)
    assert_allclose(result["area"], 4*np.pi*r*r, rtol=3e-15, atol=0)
    assert_allclose(result["volume"], 4*np.pi*r**3/3, rtol=3e-15, atol=0)
    with pytest.raises(UnsupportedNestedMeshError, match="star-shaped"):
        radial_measures(directions, np.full(len(directions), r), -directions, weights)


@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_angular_roche_order_control_and_mesh_correction_are_separate(component: str) -> None:
    binary = fixture({"inclination": 85, "discretization": 12})
    a, b = (roche_measures(binary, component, n) for n in [16, 32])
    assert_allclose([a["area"], a["volume"]], [b["area"], b["volume"]], rtol=2e-11, atol=0)
    system = seed_system(binary)
    plan(binary).apply(system)
    measured = mesh_measures(getattr(system, component), component)
    assert measured["raw_area"] < b["area"]
    assert measured["raw_volume"] < b["volume"]
    assert_allclose(measured["corrected_area"]/measured["raw_area"], measured["scalar_correction"]**2,
                    rtol=3e-15, atol=0)
    with pytest.raises(UnsupportedNestedMeshError, match="fresh raw"):
        mesh_measures(getattr(intrinsic(binary), component), component)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize(("inc", "atlas"), [(85, "bb"), (90, "bb"), (88, "ck04")])
def test_selective_complete_state_and_reflected_curve_match_original_cpu(device: str, inc: float, atlas: str) -> None:
    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        binary = fixture({"inclination": inc, "discretization": 12}) if atlas == "bb" else table_fixture(
            {"inclination": inc},
        )
        numerical = plan(binary, levels=2)
        bands = ["Generic.Bessell.V", "bolometric"]
        observer = Observer(passband=bands, system=binary)
        phases = np.array([.5, .49, 0, .01, .025, .25, 1.01, -.5])+.017
        engine = NestedBinaryLightCurve(bands, numerical, device=device,
                                       radiance="device" if atlas == "bb" else "cpu")
        actual = engine(binary, phases, normalize=True, diagnostics=True, observer=observer)
        expected = observe(observer, phases, mesh=numerical)[1]
        cpu = cpu_reflect(intrinsic(binary, numerical), 2)
        for name in ("primary", "secondary"):
            star = getattr(actual.diagnostics["initial_system"], name)
            reference = getattr(cpu, name)
            assert np.array_equal(star.faces, reference.faces)
            for key in ("points", "face_centres", "normals", "areas", "potential_gradient_magnitudes",
                        "log_g", "temperatures"):
                assert_allclose(getattr(star, key), getattr(reference, key), rtol=3e-13, atol=64*np.finfo(float).eps)
        for band in bands:
            assert_allclose(actual.fluxes[band], expected[band], rtol=3e-13, atol=2e-12)


def test_pilot_plan_is_fixed_and_explicit_alternate_topology_comparison() -> None:
    with table_settings(), settings.override(REFLECTION_EFFECT=True):
        binary = fixture({"inclination": 85, "discretization": 12})
        phases, bands = [0, .01, .5, .25], ["Generic.Bessell.V"]
        coarse, fine = [NestedBinaryLightCurve(bands, NestedMesh(i), device="cpu").snapshot(
            binary, phases, source_revision="fixed-assets",
        ) for i in [0, 1]]
        a = choose_plan(binary, coarse, fine, 1e-4, fraction=.3)
        assert a == choose_plan(binary, coarse, fine, 1e-4, fraction=.3)
        selective = NestedBinaryLightCurve(bands, a, device="cpu").snapshot(
            binary, phases, source_revision="fixed-assets",
        )
        result = compare_surfaces(selective, fine, 1e-4, allow_same_level=True)
        assert not result["continuum_certified"]
        with pytest.raises(ValueError, match="ordered nested"):
            compare_surfaces(selective, fine, 1e-4)
        with pytest.raises(ValueError, match="matching consecutive"):
            choose_plan(binary, coarse, replace(fine, signature="stale"), 1e-4)
        with pytest.raises(ValueError, match="frozen sorted"):
            replace(a, roots=((1, 1), ()))
        broken = seed_system(binary)
        broken.primary.faces[[0, 1]] = broken.primary.faces[[1, 0]]
        with pytest.raises(UnsupportedNestedMeshError, match="topology changed"):
            a.apply(broken)
