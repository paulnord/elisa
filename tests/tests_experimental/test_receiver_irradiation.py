"""Receiver gradients, bounded eclipse frontiers and complete finite-model parity."""

import os

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import settings
from elisa.experimental import gpu_reconstructed_observer as packets
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from elisa.experimental.gpu_curvature_fields import curvature_geometry
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_nodal_source import ReceiverSourceLightCurve
from elisa.experimental.receiver_irradiation import receiver_slopes
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture

from .test_gpu_boundary import CASES

DEVICE = os.getenv("ELISA_GEOMETRY_TEST_DEVICE", "cpu")


def test_conservative_nonnegative_receiver_slope() -> None:
    heat = np.array([0.0, 0.25, 3.0])
    neighbours = np.array([[1, 2, 1], [0, 2, 0], [1, 1, 0]])
    mass = np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2], [0.25, 0.25, 0.5]])
    slope, limiter = receiver_slopes(np, heat, neighbours, np.broadcast_to(np.eye(3), (3, 3, 3)), mass)
    reconstructed = heat[:, None] + slope
    assert np.min(reconstructed) >= 0
    assert np.ptp(reconstructed[1]) > 0
    assert limiter[0] == 0 and 0 < limiter[1] < 1
    assert_allclose(np.sum(reconstructed * mass, axis=1), heat, atol=1e-15)
    np.testing.assert_array_equal(receiver_slopes(np, np.ones(3), neighbours, np.eye(3)[None], mass)[0], 0)


def test_partition_preserves_ray_integrals_and_global_cap(monkeypatch) -> None:
    with table_settings():
        binary = fixture({"inclination": 85, "discretization": 12})
        _, cells, _ = curvature_geometry(binary, 0)
        tri = cells["primary"]["triangles"]
        g = ConditionedRocheGeometry(binary, "primary")
        line = np.array([0.8, 0.0, 0.6])
        rule = CurvedEdgeRule(3, 4)
        one = packets.recorded_packet(tri, g, line, rule)
        monkeypatch.setattr(packets, "_PARTITION_FACES", len(tri) // 4)
        split = packets.partitioned_packet(tri, g, line, rule)
        for key in ("nodes", "selected_nodes", "counts", "limb_edge_roots", "curved_edge_triangles"):
            assert one[key] == split[key]

        def integrated(packet):
            return np.bincount(
                packet["parents"], weights=packet["area"] * packet["mu"] * packet.rays[:, 0], minlength=len(tri)
            )

        assert_allclose(integrated(one), integrated(split), rtol=3e-13, atol=1e-17)
        with pytest.raises(UnsupportedCurvedGeometryError, match="Combined"):
            packets.partitioned_packet(tri, g, line, CurvedEdgeRule(3, 4, max_nodes=one["nodes"] - 1))


@pytest.mark.parametrize("case", CASES)
def test_reflected_receiver_curve(case: dict) -> None:
    with (
        table_settings(),
        settings.override(
            REFLECTION_EFFECT=True,
            REFLECTION_EFFECT_ITERATIONS=2,
            LIMB_DARKENING_LAW=case.get("law", "linear"),
            USE_SINGLE_LD_COEFFICIENTS=False,
        ),
    ):
        binary = (table_fixture if case.get("atlas") else fixture)({"discretization": 12, **case})
        bands = ["Generic.Bessell.V", "bolometric"]
        outputs = [
            ReceiverSourceLightCurve(bands, CurvedEdgeRule(3, 1), device=device, levels=(0, 1), integration_level=1)(
                binary, [0.017, 0.027, 0.25, 0.517, 0.533, 1.027], source_revision="fixture"
            )
            for device in (DEVICE, "reference")
        ]
        a, r = outputs
        assert a.diagnostics["phase_records"] == r.diagnostics["phase_records"]
        for key in r.curves:
            for name in ("primary", "secondary"):
                x, y = [out.diagnostics["ancestry_controls"][key][name]["reconstructed_field"] for out in outputs]
                assert x["cell_T4_relative_error"] < 3e-13
                assert y["maximum_deposit_T4_span"] > 0
                assert_allclose(x["maximum_deposit_T4_span"], y["maximum_deposit_T4_span"], atol=3e-14)
            for attr in ("component_fluxes", "raw_fluxes", "fluxes"):
                for band in bands:
                    assert_allclose(
                        getattr(a.curves[key], attr)[band],
                        getattr(r.curves[key], attr)[band],
                        rtol=3e-13,
                        atol=2e-12 if attr == "fluxes" else 0,
                    )
