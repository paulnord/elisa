"""Actual versioned CK04/Van Hamme spectra/LD through fresh curves and fits."""

from copy import deepcopy
from pathlib import Path
from shutil import copytree

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")
from elisa.experimental.gpu_binary import HybridBinaryLightCurve  # noqa: E402
from scripts.validate_gpu_atmosphere import BANDS, CASES, DATA, fit_problem, fixture, table_settings  # noqa: E402
from scripts.validate_gpu_fit import fit_assessment, normalized_vector, prepare_fit  # noqa: E402

from .test_gpu_binary import require_device  # noqa: E402
from .test_gpu_likelihood import fp64  # noqa: E402, F401


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_real_atmosphere_ld_full_curves(device: str, case: dict) -> None:
    from elisa import Observer, settings
    from elisa.binary_system.curves.c_router import prep_initial_system
    from elisa.binary_system.curves.utils import prep_surface_params

    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=case["reflection"]):
        binary = fixture(case)
        phases = np.array([.5, .49, .25, .01, 0, 1, -.5, .75]) + binary.phase_shift
        engine = HybridBinaryLightCurve(list(BANDS), device=device, fallback="raise")
        actual = engine(binary, phases, normalize=True, diagnostics=True)
        observer = Observer(passband=list(BANDS), system=binary)
        expected = observer.lc(phases=phases, normalize=True)[1]
        assert actual.backend == "hybrid"
        for band in BANDS:
            assert_allclose(actual.fluxes[band], expected[band], rtol=3e-13, atol=2e-12)
        cpu = prep_initial_system(binary)
        prep_surface_params(cpu, return_values=False, write_to_containers=True, passband=observer.passband,
                            left_bandwidth=observer.left_bandwidth, right_bandwidth=observer.right_bandwidth)
        for name in ("primary", "secondary"):
            a, b = getattr(actual.diagnostics["initial_system"], name), getattr(cpu, name)
            for band in BANDS:
                assert_allclose(a.normal_radiance[band], b.normal_radiance[band], rtol=3e-13, atol=0)
                assert_allclose(a.ld_cfs[band], b.ld_cfs[band], rtol=3e-13, atol=0)
                assert np.ptp(a.ld_cfs[band]) > 0, "fresh T/g-dependent table LD must vary across faces"


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize(("law", "bands"), [("logarithmic", [BANDS[0]]), ("square_root", [BANDS[1]])])
def test_real_table_ld_other_laws(device: str, law: str, bands: list[str]) -> None:
    from elisa import Observer, settings

    require_device(device)
    with table_settings(), settings.override(LIMB_DARKENING_LAW=law, REFLECTION_EFFECT=(law == "square_root")):
        binary = fixture(CASES[0])
        actual = HybridBinaryLightCurve(bands, device=device, fallback="raise")(binary, [.5, .25, .01, 0],
                                                                             normalize=True)
        expected = Observer(passband=bands, system=binary).lc(phases=[.5, .25, .01, 0], normalize=True)[1]
        for band in bands:
            assert_allclose(actual.fluxes[band], expected[band], rtol=3e-13, atol=2e-12)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_real_source_revision_response_and_prepared_snapshot(device: str, tmp_path: Path) -> None:
    from elisa import Observer, settings

    require_device(device)
    copytree(DATA / "ck04", tmp_path / "ck04")
    copytree(DATA / "light_curves/limbdarkening", tmp_path / "ld")
    with table_settings(), settings.override(CK04_ATM_TABLES=str(tmp_path / "ck04"), LD_TABLES=str(tmp_path / "ld"),
                                            REFLECTION_EFFECT=False):
        binary = fixture(CASES[0])
        observer = Observer(passband=list(BANDS), system=binary)
        engine = HybridBinaryLightCurve(list(BANDS), device=device, fallback="raise")
        original = engine(binary, [0, .25, .5], diagnostics=True, observer=observer)
        packet = original.diagnostics["radiance"]["prepared_batch"].block_until_ready()
        before = jax.tree.map(np.asarray, packet.evaluate())
        path = tmp_path / "ck04/ckp00/ckp00_10250_g40.csv"
        table = pd.read_csv(path)
        table["flux"] *= 1.01
        table.to_csv(path, index=False)
        path = tmp_path / "ld/lin.Generic.Bessell.V.p00.csv"
        table = pd.read_csv(path)
        table["xlin"] += .02
        table.to_csv(path, index=False)
        response = observer.passband[BANDS[0]].akima
        observer.passband[BANDS[0]].akima = lambda w: response(w) * (w / 5500)**2
        changed = engine(binary, [0, .25, .5], observer=observer)
        expected = deepcopy(observer).lc(phases=[0, .25, .5])[1]
        for band in BANDS:
            assert_allclose(changed.fluxes[band], expected[band], rtol=3e-13, atol=0)
        assert not np.allclose(original.fluxes[BANDS[0]], changed.fluxes[BANDS[0]], rtol=1e-7, atol=0)
        with jax.transfer_guard("disallow_explicit"):
            result = jax.block_until_ready(packet.evaluate())
        assert result["normal_radiance"].device.platform == device
        assert_allclose(result["normal_radiance"], before["normal_radiance"], rtol=0, atol=0)
        for eclipse in original.diagnostics["occultation"]["prepared_batches"]:
            assert eclipse._arrays[4] is original.diagnostics["radiance"]["result"]["normal_radiance"]


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_resource_and_exact_grid_fallback_keep_cpu_posterior(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from elisa import Observer, settings

    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=False, PENALIZE_ALL_EXCEPTIONS=True):
        problem = fit_problem(shape_parameters=True)
        original, hybrid = prepare_fit(problem), prepare_fit(problem, device=device)
        vector = normalized_vector(original)
        monkeypatch.setattr("elisa.experimental.gpu_atmosphere._MAX_MODELS", 1)
        actual, expected = hybrid.ln_probability(vector), original.ln_probability(vector)
        assert np.isfinite(actual)
        assert_allclose(actual, expected, rtol=0, atol=0)
        assert hybrid.hybrid_synthesis.last_evaluation["backend"] == "cpu_fallback"
        assert "capacity" in hybrid.hybrid_synthesis.last_evaluation["fallback_reason"]
        binary = fixture({**CASES[0], "temperatures": [10000, 10160]})
        binary.primary.gravity_darkening = 0
        actual = HybridBinaryLightCurve(list(BANDS), device=device)(binary, [0, .25, .5], normalize=True)
        expected = Observer(passband=list(BANDS), system=binary).lc(phases=[0, .25, .5], normalize=True)[1]
        assert actual.backend == "cpu_fallback"
        assert "selection boundary" in actual.fallback_reason
        for band in BANDS:
            assert_allclose(actual.fluxes[band], expected[band], rtol=0, atol=0)
        binary = fixture({**CASES[0], "atlas": "bb", "temperatures": [3500+1e-9, 10160]})
        binary.primary.gravity_darkening = 0
        actual = HybridBinaryLightCurve(list(BANDS), device=device)(binary, [0, .25, .5], normalize=True)
        expected = Observer(passband=list(BANDS), system=binary).lc(phases=[0, .25, .5], normalize=True)[1]
        assert actual.backend == "cpu_fallback"
        assert "LD convex-hull" in actual.fallback_reason
        for band in BANDS:
            assert_allclose(actual.fluxes[band], expected[band], rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("kind", ["atmosphere", "ld"])
def test_original_source_errors_survive(device: str, kind: str, tmp_path: Path) -> None:
    from elisa import Observer, settings
    from elisa.base.error import ElisaAtmosphereError, ElisaLimbDarkeningError

    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=False, **{
        "CK04_ATM_TABLES" if kind == "atmosphere" else "LD_TABLES": str(tmp_path),
    }):
        binary = fixture(CASES[0])
        error = ElisaAtmosphereError if kind == "atmosphere" else ElisaLimbDarkeningError
        with pytest.raises(error) as reference:
            Observer(passband=list(BANDS), system=binary).lc(phases=[0, .25, .5])
        with pytest.raises(type(reference.value)):
            HybridBinaryLightCurve(list(BANDS), device=device)(binary, [0, .25, .5])


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_real_table_public_fit_sampler_smoke(device: str) -> None:
    from elisa import settings

    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=True):
        result = fit_assessment(device, steps=2, problem=fit_problem())
    assert result["chain_parity"]["max_abs"] == 0
    assert result["chain_density_parity"]["max_abs"] <= 1e-6
