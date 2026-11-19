from types import SimpleNamespace
import os

import numpy as np
import pytest

from elisa.experimental.area_reflection import _BARY
from elisa.experimental.gpu_reflection import UnsupportedReflectionError
from elisa.experimental.horizon_receiver import HorizonReceiverLightCurve, HorizonReceiverReflection, _SIX


def test_refined_mean_and_native_shape_samples():
    # Cancel the solid-angle Jacobian to integrate a quartic on the reference simplex.
    tri = np.eye(3)
    nodes, solid = _SIX.interior_nodes(tri[None])
    area = solid * np.linalg.norm(nodes, axis=1) ** 3
    full = np.vstack((_BARY, nodes))
    star = SimpleNamespace(points=tri, faces=[[0, 1, 2]], base_symmetry_faces_number=1, t_eff=2.0)
    receiver = SimpleNamespace(parent=np.zeros(9, int), receiver=full, fraction=np.r_[np.zeros(3), 2 * area])
    power = star.t_eff**4 * full[:, 0]**4
    got = HorizonReceiverReflection._receiver_record(star, "primary", receiver, power, [star.t_eff**4 / 15])
    samples, mean = got["receiver_deposits"]
    np.testing.assert_array_equal(samples[0], _BARY[:, 0]**4)
    np.testing.assert_allclose(mean, 1/15)
    assert got["six_point_receiver_rays"] == 6
    with pytest.raises(UnsupportedReflectionError, match="reduction"):
        HorizonReceiverReflection._receiver_record(star, "primary", receiver, power, [np.mean(power[:3])])
    receiver.receiver = full.copy()
    receiver.receiver[-1] = full[-2]
    with pytest.raises(UnsupportedReflectionError, match="slot"):
        HorizonReceiverReflection._receiver_record(star, "primary", receiver, power, [star.t_eff**4 / 15])


def test_two_sweeps_eclipse_parity_and_cap():
    from elisa import settings
    from elisa.experimental.gpu_curved_edges import SymmetricInteriorRule
    from elisa.experimental.gpu_reflection_tiled import _MAX_PAIRS
    from scripts.validate_gpu_atmosphere import table_settings
    from scripts.validate_gpu_binary import fixture

    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        model = fixture(dict(discretization=14, inclination=86))
        def solve(device):
            solver = HorizonReceiverLightCurve(["bolometric"], SymmetricInteriorRule(3, 1), device=device,
                                               levels=(0, 1), integration_level=1, selective_shadow=True)
            result = solver(model, [0.024, 0.25], source_revision="horizon-control")
            for stars in result.diagnostics["ancestry_controls"].values():
                p, s = [stars[k]["reconstructed_field"] for k in ("primary", "secondary")]
                assert 0 < p["faces_with_six_point_rays"] <= p["six_point_receiver_rays"]
                assert p["pairs_per_sweep_both_stars"] == s["pairs_per_sweep_both_stars"] <= _MAX_PAIRS
                assert max(p["cell_T4_relative_error"], s["quadratic_T4_relative_error"]) <= 3e-13
            return result
        cpu = solve("reference")
        xp = solve(os.environ.get("ELISA_GEOMETRY_TEST_DEVICE", "cpu"))
        for k, c in cpu.curves.items():
            np.testing.assert_allclose(c.component_fluxes["bolometric"], xp.curves[k].component_fluxes["bolometric"],
                                       rtol=3e-13, atol=0)
