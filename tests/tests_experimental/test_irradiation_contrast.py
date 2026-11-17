"""Signed Gaussian decomposition with a measured irradiation response."""

from copy import deepcopy

import numpy as np
import pytest

from scripts.compare_gpu_irradiation import compare


def profile(c: list, f: list, *, heated: bool) -> dict:
    sigma = [0.1, 0.2, 0.1]
    reducer = [1, 0, 1]
    prediction = np.array(f)[reducer]
    row = {
        "physical": {"q": 1},
        "quadrature_sha256": "same",
        "phase_records": [],
        "curves": {key: {"fluxes": {"B": value}} for key, value in [("c", c), ("f", f)]},
        "finite_metadata": {
            key: {"field_levels": level, "rule": {"depth": 2}, "phases": [0, 0.5], "additional_light": 0.2}
            for level, key in enumerate(("c", "f"))
        },
        "prediction": {"B": (prediction / prediction.mean()).tolist()},
    }
    return {
        "irradiation": heated,
        "backend": "reference",
        "engine": "area-thermal",
        "batch": 1,
        "jd": False,
        "levels": [0, 1],
        "integration_level": 1,
        "depth": 2,
        "slice_order": 8,
        "order": 3,
        "source_revision": "assets",
        "physical_constants": {},
        "observations": {"B": [1.02, 0.9, 1.01]},
        "uncertainties": {"B": sigma},
        "reducers": {"B": reducer},
        "iteration_start": 0,
        "samples": [{"rows": [row]}],
    }


def test_observation_mapping_and_cross_term_match_direct_density() -> None:
    on = profile([0.8, 1.03], [0.9, 1.04], heated=True)
    off = profile([0.72, 1.01], [0.79, 1.02], heated=False)
    r = compare(on, off)[0]["bands"]["B"]
    total = (
        r["delta_logL_intrinsic_at_heated_anchor"]
        + r["delta_logL_response_at_heated_anchor"]
        + r["gaussian_cross_term"]
    )
    assert abs(r["gaussian_cross_term"]) > 0
    np.testing.assert_allclose(total, r["delta_logL_heated"], atol=5e-14)
    for bad in ["quadrature_sha256", "physical"]:
        wrong = deepcopy(off)
        wrong["samples"][0]["rows"][0][bad] = "different"
        with pytest.raises(ValueError, match="match"):
            compare(on, wrong)
    with pytest.raises(ValueError, match="intrinsic-only"):
        compare(on, on)
