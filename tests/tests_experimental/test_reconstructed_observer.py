"""Complete reconstructed curves, original state and strict inference stops."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import Observer, settings
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule, InferenceBudget
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.gpu_reconstructed_observer import ReconstructedObserverLightCurve
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture

from .test_gpu_boundary import CASES

DEVICE = os.environ.get("ELISA_GEOMETRY_TEST_DEVICE", "cpu")
BANDS = ["Generic.Bessell.V", "bolometric"]


@pytest.mark.parametrize("case", CASES)
def test_complete_observer_curves_fields_and_adaptive_counts(case: dict) -> None:
    with (
        table_settings(),
        settings.override(
            REFLECTION_EFFECT=True,
            REFLECTION_EFFECT_ITERATIONS=2,
            LIMB_DARKENING_LAW=case.get("law", "linear"),
        ),
    ):
        b = table_fixture(case) if case.get("atlas") else fixture({"discretization": 12, **case})
        phases = [0.0, 0.017, 0.027, 0.25, 0.517, 1.027]
        results = [
            ReconstructedObserverLightCurve(
                BANDS,
                CurvedEdgeRule(3, 1),
                levels=(0, 1),
                integration_level=1,
                device=d,
                radiance="cpu" if case.get("atlas") else "device",
            )(b, phases, source_revision="assets", diagnostics=True)
            for d in ["reference", DEVICE]
        ]
        a, r = results[1], results[0]
        assert a.diagnostics["phase_records"] == r.diagnostics["phase_records"]
        for key in r.curves:
            for attr in ["fluxes", "raw_fluxes", "component_fluxes"]:
                for band in BANDS:
                    assert_allclose(
                        getattr(a.curves[key], attr)[band],
                        getattr(r.curves[key], attr)[band],
                        rtol=3e-13,
                        atol=2e-12 if attr == "fluxes" else 0,
                    )
            for name in ["primary", "secondary"]:
                x, y = [getattr(v.diagnostics["initial_systems"][key], name) for v in [a, r]]
                for attr in ["face_centres", "normals", "areas", "temperatures"]:
                    assert_allclose(getattr(x, attr), getattr(y, attr), rtol=3e-13, atol=64 * np.finfo(float).eps)


def test_changes_rebuild_observer_physics_and_original_cpu_is_exact() -> None:
    with table_settings(), settings.override(REFLECTION_EFFECT=True):
        b = fixture({"inclination": 85, "discretization": 12})
        phases = [0, 0.017, 0.25, 0.5]
        observer = Observer(passband=BANDS, system=b)
        before = observer.lc(phases=phases, normalize=True)[1]
        e = ReconstructedObserverLightCurve(
            BANDS,
            CurvedEdgeRule(3, 1),
            levels=(0, 1),
            integration_level=1,
            device=DEVICE,
        )
        changed = fixture(
            {
                "inclination": 86,
                "discretization": 12,
                "q": 0.8,
                "potentials": [8.1, 12.2],
                "temperatures": [6600, 5100],
                "spin": [1.1, 0.9],
            },
        )
        batch = e.batch([b, changed], [phases, phases], observer=observer, source_revision="assets")
        assert len({r.curves["curvature_field1"].signature for r in batch.results}) == 2
        after = observer.lc(phases=phases, normalize=True)[1]
        for band in BANDS:
            assert_allclose(before[band], after[band], rtol=0, atol=0)
        reference = ReconstructedObserverLightCurve(
            BANDS,
            CurvedEdgeRule(3, 1),
            levels=(0, 1),
            integration_level=1,
            device="reference",
        )(changed, phases, observer=observer, source_revision="assets")
        for band in BANDS:
            assert_allclose(
                batch.results[1].curves["curvature_field1"].fluxes[band],
                reference.curves["curvature_field1"].fluxes[band],
                rtol=3e-13,
                atol=2e-12,
            )
        assert e.last_observer_stats["edge_solve_calls"] > 0


def test_accuracy_stops_remain_outside_priors() -> None:
    with table_settings(), settings.override(PENALIZE_ALL_EXCEPTIONS=True):
        b = fixture({"inclination": 85, "discretization": 12})
        r = ReconstructedObserverLightCurve(
            BANDS,
            CurvedEdgeRule(3, 1),
            levels=(0, 1),
            integration_level=1,
            device=DEVICE,
        )(b, [0, 0.017, 0.25, 0.5], source_revision="assets")
        with pytest.raises(ResolutionNotQualifiedError):
            InferenceBudget(max_distance_sigma=1e-12, max_log_likelihood=1e-12).compare(
                r.curves["curvature_field0"],
                r.curves["curvature_field1"],
                1e-4,
                observed=r.curves["curvature_field1"].fluxes,
            )


def test_refined_contact_grid_preserves_complete_adaptive_traces() -> None:
    from scripts.validate_gpu_binary import contact_phases

    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        binary = fixture({"inclination": 85, "discretization": 12})
        phases = contact_phases(binary)
        assert len(phases) == 57
        reference, actual = [
            ReconstructedObserverLightCurve(BANDS, CurvedEdgeRule(4, 1), levels=(1, 2), integration_level=2, device=d)(
                binary,
                phases,
                source_revision="assets",
            )
            for d in ["reference", DEVICE]
        ]
        assert actual.diagnostics["phase_records"] == reference.diagnostics["phase_records"]
        for key in reference.curves:
            for attr in ["fluxes", "raw_fluxes", "component_fluxes"]:
                for band in BANDS:
                    assert_allclose(
                        getattr(actual.curves[key], attr)[band],
                        getattr(reference.curves[key], attr)[band],
                        rtol=3e-13,
                        atol=2e-12 if attr == "fluxes" else 0,
                    )


def test_original_constant_cpu_hook_is_bitwise_identical_to_published_m020(tmp_path: Path) -> None:
    import hashlib
    import json
    import subprocess
    import sys

    # tests/__init__.py selects older Astropy physical constants before imports.
    # The published CLI uses the environment's default constants. Match the
    # complete physical-input/runtime protocol in a CPU-forced child process.
    root = Path(__file__).resolve().parents[2]
    profile = json.loads((root / "docs/ada_validation/background-m020.json").read_text())["complete_profiles"]["158"]
    output = tmp_path / "constant.json"
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "True", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    subprocess.run(  # noqa: S603 - fixed repository driver, current interpreter, forced CPU
        [
            sys.executable,
            str(root / "scripts/validate_gpu_fused_observer.py"),
            "--backend",
            "reference",
            "--batch",
            "1",
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    actual = json.loads(output.read_text())
    assert actual["source_revision"] == profile["source_revision"]
    for x, y in zip(actual["samples"], profile["samples"][:2], strict=True):
        row, golden = x["rows"][0], y["rows"][0]
        assert row["physical"] == golden["physical"]
        assert row["finite_metadata"] == golden["finite_metadata"]
        for key in golden["curves"]:
            for attr in golden["curves"][key]:
                for band in golden["curves"][key][attr]:
                    np.testing.assert_array_equal(row["curves"][key][attr][band], golden["curves"][key][attr][band])
        assert row["posterior"] == golden["posterior"]
        trace = hashlib.sha256(
            json.dumps(row["phase_records"], sort_keys=True, separators=(",", ":")).encode(),
        ).hexdigest()
        assert trace == golden["full_adaptive_trace_sha256"]


def test_reentry_during_flux_preserves_active_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    with table_settings(), settings.override(REFLECTION_EFFECT=True):
        binary = fixture({"inclination": 85, "discretization": 12})
        engine = ReconstructedObserverLightCurve(
            BANDS, CurvedEdgeRule(3, 1), levels=(0,), integration_level=0, device=DEVICE,
        )
        integrate = engine._integrate_packets
        calls = []

        def check(*args: Any) -> dict:
            with pytest.raises(ValueError, match="non-reentrant"):
                engine.batch([binary], [[0.25]], source_revision="assets")
            values = integrate(*args)
            calls.append(1)
            return values

        monkeypatch.setattr(engine, "_integrate_packets", check)
        result = engine(binary, [0.25], source_revision="assets")
        assert calls
        assert np.isfinite(result.curves["curvature_field0"].raw_fluxes[BANDS[0]]).all()
