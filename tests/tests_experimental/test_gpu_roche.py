"""Compare changing physical Roche geometries with ELISa's real mesh solver."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import brentq

jax = pytest.importorskip("jax")
from elisa.experimental.gpu_roche import RocheRaySolver  # noqa: E402

from .test_gpu_likelihood import fp64  # noqa: E402, F401


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_changing_roche_geometry(component: str, device: str) -> None:
    from elisa.binary_system import model
    from elisa.binary_system.radius import calculate_radius
    from elisa.binary_system.surface.mesh import get_surface_points

    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")
    q = np.array([0.5, 1, 2, 0.7, 1.7])
    sync = np.array([0.5, 1, 1.5, 1.2, 0.8])
    potentials = np.array([8.0, 10.0, 12.0, 9.0, 11.0])
    ph, th = np.meshgrid(np.linspace(0, 2 * np.pi, 17), np.linspace(0, np.pi, 13))
    ph, th = ph.ravel(), th.ravel()
    rng = np.random.default_rng(904)
    ph = np.concatenate([ph, rng.uniform(0, 2*np.pi, 20)])
    th = np.concatenate([th, rng.uniform(0, np.pi, 20)])
    solver = RocheRaySolver(component, device=device)
    result = {k: np.asarray(v) for k, v in solver(q, sync, potentials, ph, th).items()}
    assert result["proposal_valid"].all()
    precalc = getattr(model, f"pre_calculate_for_potential_value_{component}")
    value = getattr(model, f"potential_value_{component}")
    implicit = getattr(model, f"potential_{component}_fn")
    derivative = getattr(model, f"radial_{component}_potential_derivative")
    for i, (qq, ss, pp) in enumerate(zip(q, sync, potentials, strict=True)):
        coeff = precalc(ss, qq, 1.0, ph, th, return_as_tuple=True)
        cpu_value = value(result["radii"][i], qq, *coeff)
        assert_allclose(cpu_value, pp, rtol=2e-13, atol=0)
        # Independent bracketing of the CPU equations (not copied GPU formulas).
        roots = np.array(
            [
                brentq(
                    lambda r, j=j, qq=qq, coeff=coeff, pp=pp: value(r, qq, *(a[j] for a in coeff)) - pp,
                    1e-6,
                    0.25,
                    xtol=5e-16,
                )
                for j in range(ph.size)
            ],
        )
        assert_allclose(result["radii"][i], roots, rtol=1e-13, atol=5e-16)
        first_inner = np.array(
            [calculate_radius(ss, qq, pp, component, 1.0, float(ph[j]), float(th[j])) for j in range(0, ph.size, 17)],
        )
        assert_allclose(result["radii"][i, ::17], first_inner, rtol=2e-11, atol=1e-13)
        # Actual ELISa mesh helper using its Newton solver and coordinate mapping.
        points = get_surface_points(ph, th, float(np.mean(roots)), 1.0, precalc, implicit, derivative, pp, qq, ss)
        assert_allclose(result["points_local"][i], points, rtol=2e-10, atol=1e-13)
    assert not np.allclose(result["radii"][0], result["radii"][-1])


def test_roche_invalid_roots_and_domain() -> None:
    solver = RocheRaySolver("primary", device="cpu")
    # Both too-large surface radius and below-inner-bound radius are rejected.
    result = solver([1, 1, 1], [1, 1, 1], [1, 1e9, 10], [0, np.pi], [np.pi / 2, np.pi / 2])
    assert np.array_equal(result["proposal_valid"], [False, False, True])
    assert np.isnan(np.asarray(result["radii"])[:2]).all()
    with pytest.raises(ValueError, match="Unsupported domain"):
        solver([0.1], [1], [10], [0], [0])
    with pytest.raises(ValueError, match="finite"):
        solver([np.nan], [1], [10], [0], [0])
    with pytest.raises(ValueError, match="Angles"):
        solver([1], [1], [10], [-0.1], [0])
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
        solver([1], [1], [10], [0], [0])


def test_monotonic_domain_bound() -> None:
    # Conservative analytic derivative bound over all admitted directions/radii.
    # -M/r² + m/(1-r)² + m + sync²(1+q)r, at upper radius .25.
    q = np.linspace(0.5, 2, 101)
    upper = 0.25
    primary = -1 / upper**2 + q / (1 - upper) ** 2 + q + 1.5**2 * (1 + q) * upper
    secondary = -q / upper**2 + 1 / (1 - upper) ** 2 + 1 + 1.5**2 * (1 + q) * upper
    assert np.max(primary) < 0
    assert np.max(secondary) < 0


@pytest.mark.parametrize("component", ["primary", "secondary"])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_near_bracket_endpoints(component: str, device: str) -> None:
    from elisa.binary_system import model

    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")
    q, sync = np.array([0.5, 2.0]), np.array([1.5, 0.5])
    phi, theta = np.array([0.731]), np.array([1.213])
    desired = np.array([1e-6 * (1 + 1e-7), 0.25 * (1 - 1e-7)])
    targets = []
    for qq, ss, rr in zip(q, sync, desired, strict=True):
        coeff = getattr(model, f"pre_calculate_for_potential_value_{component}")(
            ss,
            qq,
            1.0,
            float(phi[0]),
            float(theta[0]),
        )
        targets.append(getattr(model, f"potential_value_{component}")(rr, qq, *coeff))
    result = RocheRaySolver(component, device=device)(q, sync, targets, phi, theta)
    assert np.asarray(result["proposal_valid"]).all()
    assert_allclose(np.asarray(result["radii"])[:, 0], desired, rtol=3e-13, atol=0)


@pytest.mark.parametrize("component", ["primary", "secondary"])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_ambiguous_endpoint_is_not_physical_rejection(component: str, device: str) -> None:
    from elisa.binary_system import model

    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")
    phi, theta = .731, 1.213
    coeff = getattr(model, f"pre_calculate_for_potential_value_{component}")(1., 1., 1., phi, theta)
    solver = RocheRaySolver(component, device=device)
    for endpoint in (1e-6, .25):
        target = getattr(model, f"potential_value_{component}")(endpoint, 1., *coeff)
        targets = [np.nextafter(target, -np.inf), target, np.nextafter(target, np.inf)]
        result = solver([1.]*3, [1.]*3, targets, [phi], [theta])
        assert not np.asarray(result["proposal_valid"]).any()
        assert (np.asarray(result["status"]) == 2).all()
