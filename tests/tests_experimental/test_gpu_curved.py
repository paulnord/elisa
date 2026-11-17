"""Complete curved Roche model parity, original finite fields and accuracy stops."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import Observer, settings
from elisa.analytics.models import cost_fns
from elisa.experimental.curved_geometry import CurvedRule, UnsupportedCurvedGeometryError
from elisa.experimental.gpu_curved import CurvedBinaryLightCurve, curved_difference
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture

from .test_gpu_binary import require_device
from .test_gpu_likelihood import fp64  # noqa: F401

CASES = [{"inclination": 85}, {"inclination": 90}, {"inclination": 88, "atlas": "ck04"},
         {"inclination": 60, "q": .5, "spin": [.5, 1.5], "law": "logarithmic"},
         {"inclination": 87, "q": 2., "spin": [1.5, .5], "potentials": [12, 16], "law": "square_root"}]


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("case", CASES)
def test_complete_curved_reflected_fields_and_curves_match_original_cpu_primitives(device: str, case: dict) -> None:
    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2,
                                           LIMB_DARKENING_LAW=case.get("law", "linear")):
        binary = table_fixture(case) if case.get("atlas") == "ck04" else fixture({"discretization": 12, **case})
        bands = ["Generic.Bessell.V", "bolometric"]
        phases = np.array([.5, .49, 0, .01, .025, .25, 1.01, -.5])+.017
        kwargs = {"rule": CurvedRule(3, 2), "field_levels": 0,
                  "radiance": "cpu" if case.get("atlas") == "ck04" else "device"}
        reference = CurvedBinaryLightCurve(bands, device="reference", **kwargs)(
            binary, phases, source_revision="fixed-assets", diagnostics=True,
        )
        actual = CurvedBinaryLightCurve(bands, device=device, **kwargs)(
            binary, phases, source_revision="fixed-assets", diagnostics=True,
        )
        assert actual.signature == reference.signature
        assert actual.diagnostics["phase_records"] == reference.diagnostics["phase_records"]
        for name in ["primary", "secondary"]:
            a, b = (getattr(r.diagnostics["initial_system"], name) for r in [actual, reference])
            for key in ["points", "face_centres", "normals", "areas", "potential_gradient_magnitudes", "log_g",
                        "temperatures"]:
                assert_allclose(getattr(a, key), getattr(b, key), rtol=3e-13, atol=64*np.finfo(float).eps)
            for band in bands:
                assert_allclose(a.normal_radiance[band], b.normal_radiance[band], rtol=3e-13, atol=0)
        for band in bands:
            assert_allclose(actual.fluxes[band], reference.fluxes[band], rtol=3e-13, atol=2e-12)
            assert_allclose(actual.raw_fluxes[band], reference.raw_fluxes[band], rtol=3e-13, atol=0)
            assert_allclose(actual.component_fluxes[band], reference.component_fluxes[band], rtol=3e-13, atol=0)


def test_curved_engine_rebuilds_changed_physics_response_and_preserves_original_cpu() -> None:
    with table_settings(), settings.override(REFLECTION_EFFECT=True):
        band = "Generic.Bessell.V"
        binary = fixture({"inclination": 85, "discretization": 12})
        phases = [0, .01, .25, .5]
        observer = Observer(passband=[band], system=binary)
        original = deepcopy(binary.to_json())
        before = observer.lc(phases=phases, normalize=True)[1]
        engine = CurvedBinaryLightCurve([band], CurvedRule(3, 1), field_levels=0, device="cpu")
        a = engine(binary, phases, observer=observer, source_revision="assets")
        assert binary.to_json() == original
        after = observer.lc(phases=phases, normalize=True)[1]
        assert_allclose(before[band], after[band], rtol=0, atol=0)
        changed = fixture({"inclination": 86, "discretization": 12, "temperatures": [6600, 5100]})
        b = engine(changed, phases, observer=observer, source_revision="assets")
        cpu = CurvedBinaryLightCurve([band], CurvedRule(3, 1), field_levels=0, device="reference")(
            changed, phases, observer=observer, source_revision="assets",
        )
        assert_allclose(b.fluxes[band], cpu.fluxes[band], rtol=3e-13, atol=2e-12)
        assert a.signature != b.signature
        table = observer.passband[band].table.copy(deep=True)
        table[settings.PASSBAND_DATAFRAME_THROUGHPUT] *= 1.3
        observer.passband[band].table = table
        c = engine(binary, phases, observer=observer, source_revision="assets")
        assert c.signature != a.signature
        assert_allclose(c.raw_fluxes[band], 1.3*a.raw_fluxes[band], rtol=3e-13, atol=0)
        with pytest.raises(ValueError, match="matching physical"):
            curved_difference(a, c, 1e-4)


def test_curved_finite_enclosure_uses_original_mapping_noise_and_aborts_outside_priors() -> None:
    with table_settings(), settings.override(REFLECTION_EFFECT=True):
        band = "Generic.Bessell.V"
        b = fixture({"inclination": 85, "discretization": 12})
        phases = [0, .007, .01, .025, .25, .48, .5]
        a, c = [CurvedBinaryLightCurve([band], CurvedRule(3, depth), field_levels=0, device="reference")(
            b, phases, source_revision="assets",
        ) for depth in [0, 2]]
        reducer = np.array([4, 0, 2, 2, 6, 0])
        y1, y2 = a.fluxes[band][reducer], c.fluxes[band][reducer]
        y1, y2 = y1/y1.mean(), y2/y2.mean()
        sigma = np.linspace(1e-4, 2e-4, len(reducer))
        data = y1+np.random.default_rng(1301).normal(0, sigma)
        kwargs = {"reducers": {band: reducer}, "mean_normalize": True, "observed": {band: data}}
        r = curved_difference(a, c, {band: sigma}, ln_f=-2., **kwargs)
        actual = cost_fns.likelihood_fn({band: data}, {band: sigma}, {band: y2}, -2.)
        actual -= cost_fns.likelihood_fn({band: data}, {band: sigma}, {band: y1}, -2.)
        assert_allclose(r["actual"]["delta_log_likelihood"], actual, rtol=1e-11, atol=1e-10)
        assert abs(actual) <= r["finite_log_likelihood_absolute_bound"]
        assert np.all(np.abs(r["bands"][band]["actual_change"]) <= r["bands"][band]["component_envelope"])
        assert not r["continuum_certified"]
        with settings.override(PENALIZE_ALL_EXCEPTIONS=True), pytest.raises(ResolutionNotQualifiedError):
            curved_difference(a, c, {band: sigma}, ln_f=20., require_budget=True, **kwargs)
        with pytest.raises(ValueError, match="matching physical"):
            curved_difference(a, replace(c, signature="stale"), 1e-4)


def test_curved_resource_and_unsupported_protocol_never_fallback_or_change_prior() -> None:
    with table_settings():
        b = fixture({"inclination": 85, "discretization": 12})
        engine = CurvedBinaryLightCurve(["bolometric"], CurvedRule(3, 3, 10), device="cpu", field_levels=0)
        with pytest.raises(UnsupportedCurvedGeometryError, match="node cap"):
            engine(b, [0, .25], source_revision="assets")
        unsupported = fixture({"inclination": 85, "q": .4})
        with pytest.raises(UnsupportedCurvedGeometryError, match="q in"):
            engine(unsupported, [0], source_revision="assets")
        with pytest.raises(ValueError, match="levels"):
            CurvedBinaryLightCurve(["bolometric"], CurvedRule(), device="cpu", field_levels=3)
