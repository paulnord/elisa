"""Nested physical meshes, complete curves and finite error/likelihood envelopes."""

from dataclasses import replace
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import Observer, const, settings, utils
from elisa.analytics.models import cost_fns
from elisa.binary_system import model
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.radius import calculate_radius
from elisa.experimental.gpu_nested_mesh import NestedMesh, UnsupportedNestedMeshError
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.gpu_reference import intrinsic, observe
from elisa.experimental.gpu_surface_error import NestedBinaryLightCurve, SurfaceSnapshot, compare_surfaces
from elisa.observer.utils import normalize_light_curve
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture

from .test_gpu_binary import require_device
from .test_gpu_likelihood import fp64  # noqa: F401


def raw_system(binary: Any) -> Any:
    system = OrbitalPositionContainer.from_binary_system(binary, const.Position(0, 1., 0., 0., 0.))
    system.build_mesh(components_distance=1.)
    system.build_faces(components_distance=1.)
    return system


@pytest.mark.parametrize(("level", "case"), [(0, {}), (1, {}), (2, {}), (3, {}),
                                            (2, {"q": .5, "spin": [.5, 1.5]}),
                                            (2, {"q": 2., "spin": [1.5, .5], "potentials": [12, 16]})])
def test_nested_mesh_is_closed_conforming_symmetric_and_on_original_potential(level: int, case: dict) -> None:
    binary = fixture({"inclination": 85, "discretization": 12, **case})
    system = raw_system(binary)
    seeds = {n: (getattr(system, n).points.copy(), getattr(system, n).faces.copy(),
                 getattr(system, n).base_symmetry_points_number) for n in ("primary", "secondary")}
    info = NestedMesh(level).apply(system)
    for name in ("primary", "secondary"):
        star = getattr(system, name)
        points, faces = star.points, star.faces
        edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2)), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        assert np.all(counts == 2)
        assert len(points)-len(unique)+len(faces) == 2
        assert np.all(utils.triangle_areas(faces, points) > 0)
        assert len(faces) == len(seeds[name][1])*4**level
        assert np.array_equal(np.bincount(info[name]["parents"]), np.full(len(seeds[name][1]), 4**level))
        assert not info[name]["parents"].flags.writeable
        if not level:
            assert_allclose(points, seeds[name][0], rtol=0, atol=0)
            assert np.array_equal(faces, seeds[name][1])
            continue
        p = star.symmetry_points()[seeds[name][2]:].copy()
        if name == "secondary":
            p[:, 0] = 1-p[:, 0]
        radius = np.linalg.norm(p, axis=1)
        phi, theta = np.arctan2(p[:, 1], p[:, 0]), np.arccos(p[:, 2]/radius)
        coeff = getattr(model, f"pre_calculate_for_potential_value_{name}")(
            star.synchronicity, system.mass_ratio, 1., phi, theta, return_as_tuple=True,
        )
        actual = getattr(model, f"potential_value_{name}")(radius, system.mass_ratio, *coeff)
        assert_allclose(actual, star.surface_potential, rtol=2e-13, atol=0)
        sample = np.linspace(0, len(phi)-1, 5).astype(int)
        roots = [calculate_radius(star.synchronicity, system.mass_ratio, star.surface_potential, name,
                                  1., float(phi[i]), float(theta[i])) for i in sample]
        assert_allclose(radius[sample], roots, rtol=2e-11, atol=1e-13)


def synthetic_snapshot(groups: list[np.ndarray], level: int, *, third_light: float = .2) -> SurfaceSnapshot:
    raw = sum(v.sum(axis=1) for v in groups)
    flux, _ = normalize_light_curve({"v": raw+raw.mean()*third_light/(1-third_light)},
                                    kind="maximum", top_fraction_to_average=0.)
    n = len(raw)
    masks = tuple(np.zeros_like(v, dtype=bool) for v in groups)
    mu = tuple(np.ones_like(v) for v in groups)
    return SurfaceSnapshot(level, "same-candidate-seed-sources", third_light, np.arange(n),
                           MappingProxyType(flux), MappingProxyType({"v": raw}),
                           MappingProxyType({"v": tuple(groups)}), masks, mu, mu,
                           tuple(v.shape[1]*4**level for v in groups), 0., 0.)


@pytest.mark.parametrize("mean", [False, True])
@pytest.mark.parametrize("third_light", [0., .2])
def test_finite_envelope_reconstruction_and_original_likelihood_with_observation_reducers(
    mean: bool, third_light: float,
) -> None:
    rng = np.random.default_rng(913)
    groups = [rng.uniform(.05, .2, (8, n)) for n in [7, 9]]
    fine_groups = [v+rng.normal(0, 1e-4, v.shape) for v in groups]
    coarse, fine = synthetic_snapshot(groups, 0, third_light=third_light), synthetic_snapshot(fine_groups, 1,
                                                                                          third_light=third_light)
    reducer = np.array([4, 0, 2, 2, 7, 0])
    a, b = coarse.fluxes["v"][reducer], fine.fluxes["v"][reducer]
    if mean:
        a, b = a/a.mean(), b/b.mean()
    sigma = np.linspace(1e-4, 2e-4, len(reducer))
    observed = a+rng.normal(0, sigma)
    report = compare_surfaces(coarse, fine, {"v": sigma}, reducers={"v": reducer}, mean_normalize=mean,
                              ln_f=-2., observed={"v": observed})
    row = report["bands"]["v"]
    assert np.all(np.abs(b-a) <= row["finite_envelope"])
    assert np.all(np.abs(b-a) <= row["component_envelope"])
    assert row["reconstruction_max_abs"] <= row["roundoff_guard_max"]
    expected = cost_fns.likelihood_fn({"v": observed}, {"v": sigma}, {"v": b}, -2.)
    expected -= cost_fns.likelihood_fn({"v": observed}, {"v": sigma}, {"v": a}, -2.)
    assert_allclose(report["joint"]["delta_log_likelihood"], expected, rtol=1e-12, atol=1e-12)
    assert abs(expected) <= report["finite_log_likelihood_absolute_bound"]
    assert not report["continuum_certified"]


def test_component_cancellation_is_separate_from_localization_bound_and_accuracy_abort() -> None:
    groups = [np.ones((5, 4)), np.ones((5, 4))*2]
    changed = [v.copy() for v in groups]
    changed[0][:, 0] += .1
    changed[0][:, 1] -= .1
    coarse, fine = synthetic_snapshot(groups, 0), synthetic_snapshot(changed, 1)
    report = compare_surfaces(coarse, fine, 1e-4, require_budget=True)
    assert report["within_finite_budget"]
    assert report["finite_envelope_joint"]["distance_sigma"] > 1
    changed[0][:, 0] += np.arange(5)*.1
    fine = synthetic_snapshot(changed, 1)
    with settings.override(PENALIZE_ALL_EXCEPTIONS=True), pytest.raises(ResolutionNotQualifiedError):
        compare_surfaces(coarse, fine, 1e-4, require_budget=True)
    # Nuisance inflation may reduce likelihood error but cannot evade the
    # independent numerical budget or turn an accuracy failure into a prior cut.
    with pytest.raises(ResolutionNotQualifiedError):
        compare_surfaces(coarse, fine, 1e-4, require_budget=True, ln_f=20.)
    with pytest.raises(ValueError, match="same candidate"):
        compare_surfaces(coarse, replace(fine, signature="different"), 1e-4)
    with pytest.raises(ValueError, match="reducer"):
        compare_surfaces(coarse, fine, 1e-4, reducers={"v": [-1]})


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize(("inclination", "atlas", "level"), [(85, "bb", 2), (90, "bb", 3), (88, "ck04", 1)])
def test_nested_complete_physics_matches_independent_original_cpu_state(
    device: str, inclination: float, atlas: str, level: int,
) -> None:
    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        binary = fixture({"inclination": inclination, "discretization": 12}) if atlas == "bb" else table_fixture(
            {"inclination": inclination},
        )
        bands = ["Generic.Bessell.V", "bolometric"]
        phases = np.array([.5, .49, 0, .01, .025, .25, 1.01, -.5])+.017
        engine = NestedBinaryLightCurve(bands, NestedMesh(level), device=device,
                                       radiance="device" if atlas == "bb" else "cpu")
        actual = engine.snapshot(binary, phases, source_revision="versioned-test-assets")
        expected = observe(Observer(passband=bands, system=binary), phases, mesh=NestedMesh(level))[1]
        for band in bands:
            assert_allclose(actual.fluxes[band], expected[band], rtol=3e-13, atol=2e-12)
            reconstructed = sum(v.sum(axis=1) for v in actual.groups[band])
            assert_allclose(reconstructed/actual.totals[band].max(), actual.totals[band]/actual.totals[band].max(),
                            rtol=3e-13, atol=2e-12)
            with pytest.raises(ValueError, match="read-only"):
                actual.fluxes[band][0] = 0
        with pytest.raises(TypeError):
            actual.fluxes[bands[0]] = np.zeros(len(phases))


@pytest.mark.parametrize("level", [-1, 4, True, 1.5])
def test_invalid_refinement_level_and_contracts(level: Any) -> None:
    with pytest.raises(ValueError, match="levels"):
        NestedMesh(level)


def test_refinement_rejects_stale_states_capacity_and_unsupported_physics() -> None:
    binary = fixture({"inclination": 85, "discretization": 8})
    with pytest.raises(UnsupportedNestedMeshError, match="40000"):
        NestedMesh(3).apply(raw_system(binary))
    with pytest.raises(UnsupportedNestedMeshError, match="fresh raw mesh"):
        NestedMesh(1).apply(intrinsic(binary))
    unsupported = fixture({"inclination": 85, "q": .4})
    with pytest.raises(UnsupportedNestedMeshError, match="q in"):
        NestedMesh(1).apply(raw_system(unsupported))


def test_physical_and_passband_snapshots_cannot_be_reused_across_changes() -> None:
    with table_settings(), settings.override(REFLECTION_EFFECT=True):
        band = "Generic.Bessell.V"
        binary = fixture({"inclination": 85, "discretization": 12})
        original = NestedBinaryLightCurve([band], NestedMesh(0), device="cpu").snapshot(
            binary, [0, .01, .5], source_revision="fixed-assets",
        )
        engine = NestedBinaryLightCurve([band], NestedMesh(1), device="cpu")
        binary.primary.t_eff += 100
        changed = engine.snapshot(binary, [0, .01, .5], source_revision="fixed-assets")
        with pytest.raises(ValueError, match="same candidate"):
            compare_surfaces(original, changed, 1e-4)
        binary.primary.t_eff -= 100
        observer = Observer(passband=[band], system=binary)
        table = observer.passband[band].table.copy(deep=True)
        table[settings.PASSBAND_DATAFRAME_THROUGHPUT] *= np.linspace(.3, 1.7, len(table))
        observer.passband[band].table = table
        changed = engine.snapshot(binary, [0, .01, .5], observer=observer, source_revision="fixed-assets")
        with pytest.raises(ValueError, match="same candidate"):
            compare_surfaces(original, changed, 1e-4)
