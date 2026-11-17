"""Fresh resident geometry through reflection, radiance and complete curves."""

import os
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import Observer, settings
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule, InferenceBudget
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.gpu_resident_geometry import ResidentGeometryLightCurve, _angular_inputs, _ResidentField
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import contact_phases, fixture

from .test_gpu_boundary import CASES

DEVICE = os.getenv("ELISA_GEOMETRY_TEST_DEVICE", "cpu")
BANDS = ["Generic.Bessell.V", "bolometric"]
PHASES = np.array([0., .017, .027, .25, .517, 1.027])


def engine(device: str, **kwargs: object) -> ResidentGeometryLightCurve:
    return ResidentGeometryLightCurve(BANDS, CurvedEdgeRule(3, 1), levels=(0, 1),
                                     integration_level=1, device=device, **kwargs)


def parity(actual: object, reference: object) -> None:
    assert actual.quadrature_sha256 == reference.quadrature_sha256
    assert actual.diagnostics["phase_records"] == reference.diagnostics["phase_records"]
    for key in reference.curves:
        for attr in ["fluxes", "raw_fluxes", "component_fluxes"]:
            for band in BANDS:
                assert_allclose(getattr(actual.curves[key], attr)[band], getattr(reference.curves[key], attr)[band],
                                rtol=3e-13, atol=2e-12 if attr == "fluxes" else 0)
        for name in ["primary", "secondary"]:
            a, b = [getattr(r.diagnostics["initial_systems"][key], name) for r in [actual, reference]]
            for attr in ["face_centres", "normals", "areas", "potential_gradient_magnitudes", "log_g", "temperatures"]:
                assert_allclose(getattr(a, attr), getattr(b, attr), rtol=3e-13,
                                atol=64*np.finfo(float).eps if attr in ["face_centres", "normals"] else 0)
            for band in BANDS:
                assert_allclose(a.normal_radiance[band], b.normal_radiance[band], rtol=3e-13, atol=0)


@pytest.mark.parametrize("case", CASES)
def test_complete_original_cpu_fields_reflection_and_curves(case: dict) -> None:
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2,
                                           LIMB_DARKENING_LAW=case.get("law", "linear")):
        b = table_fixture(case) if case.get("atlas") else fixture({"discretization": 12, **case})
        kwargs = {"radiance": "cpu" if case.get("atlas") else "device"}
        reference = engine("reference", **kwargs)(b, PHASES, source_revision="assets", diagnostics=True)
        actual = engine(DEVICE, **kwargs)(b, PHASES, source_revision="assets", diagnostics=True)
        parity(actual, reference)


def test_changed_physics_shapes_waves_original_observer_and_no_stale_transaction() -> None:
    with table_settings(), settings.override(REFLECTION_EFFECT=True):
        binaries = [fixture({"inclination": 85+i, "discretization": 12, "q": .7+.04*i,
                             "spin": [.8+.03*i, 1.1-.02*i], "potentials": [8+.02*i, 12+.03*i],
                             "temperatures": [6400+50*i, 5000+40*i]}) for i in range(3)]
        observer = Observer(passband=BANDS, system=binaries[0])
        saved = deepcopy(binaries[0].to_json())
        before = observer.lc(phases=PHASES, normalize=True)[1]
        instance = engine(DEVICE)
        batch = instance.batch(binaries, [PHASES]*3, observer=observer, source_revision="assets", diagnostics=True)
        assert binaries[0].to_json() == saved
        after = observer.lc(phases=PHASES, normalize=True)[1]
        for band in BANDS:
            assert_allclose(after[band], before[band], rtol=0, atol=0)
        assert len({r.curves["curvature_field1"].signature for r in batch.results}) == 3
        for b, r in zip(binaries, batch.results, strict=True):
            parity(r, engine("reference")(b, PHASES, source_revision="assets", diagnostics=True))
        faces = max(sum(len(getattr(_angular_inputs(b, v)[0], n).faces)
                        for v in [0, 1] for n in ["primary", "secondary"]) for b in binaries)
        waves = engine(DEVICE, max_resident_faces=faces).batch(binaries, [PHASES]*3,
                         source_revision="assets", diagnostics=True)
        for a, b in zip(waves.results, batch.results, strict=True):
            parity(a, b)
        assert all(e.prepared is None for e in instance.engines.values())
        if DEVICE != "reference":
            with pytest.raises(RuntimeError, match="fresh batch transaction"):
                instance.engines["curvature_field0"]._fields(binaries[0], observer)


def test_resident_composition_no_host_callback_and_accuracy_resource_stops() -> None:
    with table_settings(), settings.override(REFLECTION_EFFECT=True, PENALIZE_ALL_EXCEPTIONS=True):
        b = fixture({"inclination": 85, "discretization": 12})
        field = _ResidentField("secondary", device=DEVICE)
        _, row = _angular_inputs(b, 0)
        result = field.evaluate([row["secondary"]])
        assert result["temperatures"].device.platform == DEVICE
        assert abs(float(result["intrinsic_luminosity_ratio"][0])-1) < 3e-13
        arrays = [row["secondary"][k][None] for k in ["parameters", "rays", "triangles", "physics"]]
        arrays[2] = arrays[2][:, :len(row["secondary"]["rays"])]
        graph = str(field.jax.make_jaxpr(field.kernel)(*arrays))
        assert "callback" not in graph
        with pytest.raises(UnsupportedCurvedGeometryError, match="face budget"):
            engine(DEVICE, max_resident_faces=1)(b, PHASES, source_revision="assets")
        with pytest.raises(UnsupportedCurvedGeometryError, match="q in"):
            engine(DEVICE)(fixture({"inclination": 85, "q": .4}), PHASES, source_revision="assets")
        r = engine(DEVICE)(b, PHASES, source_revision="assets")
        with pytest.raises(ResolutionNotQualifiedError):
            InferenceBudget(max_distance_sigma=1e-12, max_log_likelihood=1e-12).compare(
                r.curves["curvature_field0"], r.curves["curvature_field1"], 1e-4,
                                      observed=r.curves["curvature_field1"].fluxes)


def test_refined_complete_contact_curves() -> None:
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        b = fixture({"inclination": 85, "discretization": 12})
        phases = contact_phases(b)
        results = [ResidentGeometryLightCurve(BANDS, CurvedEdgeRule(4, 1), levels=(1, 2),
                      integration_level=2, device=d)(b, phases, source_revision="assets", diagnostics=True)
                   for d in ["reference", DEVICE]]
        parity(results[1], results[0])
