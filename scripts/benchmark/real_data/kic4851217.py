# Copyright 2026 ELISa contributors
"""Regression benchmark for the KIC 4851217 light curve from tutorial 12.

This intentionally evaluates the saved tutorial least-squares parameters once. It
does not optimize parameters and does not read the saved MCMC result or chain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from astropy import physical_constants
from scipy.interpolate import interp1d

# Match the constants convention used throughout the ELISa regression tests.
physical_constants.set("astropyconst20")

from elisa.analytics import LCData  # noqa: E402
from elisa.analytics.binary_fit.least_squares import DetachedLightCurveFit  # noqa: E402
from elisa.analytics.models import cost_fns  # noqa: E402
from elisa.analytics.models import lc as lc_model  # noqa: E402
from elisa.analytics.params import parameters  # noqa: E402
from elisa.analytics.tools.utils import time_layer_resolver  # noqa: E402
from elisa.binary_system.system import BinarySystem  # noqa: E402
from elisa.observer.utils import normalize_light_curve  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OBSERVATIONS_PATH = (
    REPOSITORY_ROOT / "jupyter_tutorials/demo_data/lc_data/KIC04851217.lc.pulse_removed.dat"
)
PARAMETERS_PATH = REPOSITORY_ROOT / "jupyter_tutorials/demo_data/assets/lc_least_squares.json"

PERIOD_DAYS = 2.47028
PRIMARY_MINIMUM_BJD = 54953.900507000006
DISCRETIZATION = 10.0
INTERPOLATION_POINTS = 100
CHECKPOINT_INDICES = np.array([0, 499, 999, 1999, 2999, 3999, 4997])
EXPECTED_OBSERVATION_COUNT = 4998
EXPECTED_LN_F = -20.0

# Observational values are file/phase-folding contracts, so their absolute
# tolerances are close to double-precision rounding. Synthetic values use a
# 20-ppm relative allowance plus 2 ppm absolute allowance. The aggregate
# residual statistics allow 0.05% because uncertainty weighting amplifies tiny
# surface-integration differences over 4,998 samples.
OBSERVATION_ATOL = 5e-12
SYNTHETIC_RTOL = 2e-5
SYNTHETIC_ATOL = 2e-6
AGGREGATE_RTOL = 5e-4

EXPECTED_PHASE_CHECKPOINTS = np.array(
    [
        -0.14640579205942533,
        -0.4560816992426773,
        -0.295183015693242,
        0.06793541622397825,
        0.22688913806915778,
        -0.4019624900844292,
        -0.46020576615273967,
    ],
)
EXPECTED_UNCERTAINTY_CHECKPOINTS = np.array([3.8e-5, 3.5e-5, 3.7e-5, 3.5e-5, 3.8e-5, 3.8e-5, 3.6e-5])
EXPECTED_SYNTHETIC_CHECKPOINTS = np.array(
    [
        1.0240012383036228,
        1.0165643102480129,
        1.034586982141161,
        1.0149108844611214,
        1.037803592240562,
        1.0226555340275676,
        1.0130793969586882,
    ],
)
EXPECTED_RESIDUAL_CHECKPOINTS = np.array(
    [
        0.00972401156064495,
        -0.00420377349632917,
        0.00150592349624334,
        -0.00670844736289866,
        0.00171993830808548,
        -0.0023719197368226,
        -0.00372842359172898,
    ],
)
EXPECTED_RESIDUAL_RMS = 0.002090201923588669
EXPECTED_WSSR = 15058894.587033533
EXPECTED_LOG_LIKELIHOOD = -7483192.728093892


def _assert_close(
    name: str,
    actual: Any,
    expected: Any,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as exc:
        msg = f"KIC 4851217 regression mismatch in {name}: {exc}"
        raise AssertionError(msg) from exc


def run_benchmark() -> dict[str, Any]:
    """Evaluate the fixed tutorial model and validate its numerical fingerprint."""
    observations = LCData.load_from_file(OBSERVATIONS_PATH, x_unit=None, y_unit=None)
    if len(observations.x_data) != EXPECTED_OBSERVATION_COUNT:
        msg = f"Expected 4,998 observations, found {len(observations.x_data):,}."
        raise AssertionError(msg)
    if np.any(observations.y_err <= 0):
        msg = "All bundled flux uncertainties must be positive."
        raise AssertionError(msg)

    raw_uncertainties = observations.y_err.copy()
    observations.convert_to_phases(PERIOD_DAYS, PRIMARY_MINIMUM_BJD, centre=0.0)
    observed_phases = observations.x_data.copy()

    _assert_close(
        "observed phases",
        observed_phases[CHECKPOINT_INDICES],
        EXPECTED_PHASE_CHECKPOINTS,
        atol=OBSERVATION_ATOL,
    )
    _assert_close(
        "observational uncertainties",
        raw_uncertainties[CHECKPOINT_INDICES],
        EXPECTED_UNCERTAINTY_CHECKPOINTS,
        atol=np.finfo(float).eps,
    )

    tutorial_result = json.loads(PARAMETERS_PATH.read_text(encoding="utf-8"))
    ln_f = float(tutorial_result["nuisance"]["ln_f"]["value"])
    if ln_f != EXPECTED_LN_F:
        msg = f"Expected the least-squares fixture's fixed ln_f=-20, found {ln_f}."
        raise AssertionError(msg)

    fit = DetachedLightCurveFit()
    fit.atmosphere_model = None
    fit.limb_darkening_coefficients = None

    setup_started = perf_counter()
    fit.set_up(
        parameters.BinaryInitialParameters(**tutorial_result),
        data={"Kepler": observations},
        passband=["Kepler"],
        discretization=DISCRETIZATION,
        interp_treshold=INTERPOLATION_POINTS,
        samples="adaptive",
        observer_system_cls=BinarySystem,
    )
    setup_seconds = perf_counter() - setup_started

    model_kwargs = parameters.prepare_properties_set(
        fit.initial_vector,
        fit.fitable.keys(),
        fit.constrained,
        fit.fixed,
    )
    model_phases, model_kwargs = time_layer_resolver(fit.x_data_reduced, pop=False, **model_kwargs)
    model_kwargs = parameters.extend_json_with_atm_params(
        model_kwargs,
        atmosphere_model=None,
        limb_darkening_coefficients=None,
    )

    model_started = perf_counter()
    sampled_curve = lc_model.synthetic_binary(
        fit.fit_xs,
        fit.discretization,
        fit.observer,
        **model_kwargs,
    )
    synthetic = {
        band: interp1d(fit.fit_xs, curve, kind="cubic")(
            model_phases[fit.x_data_reducer[band]],
        )
        for band, curve in sampled_curve.items()
    }
    synthetic, _ = normalize_light_curve(synthetic, kind="average")
    model_seconds = perf_counter() - model_started

    observed = fit.y_data["Kepler"]
    uncertainties = fit.y_err["Kepler"]
    synthetic_kepler = synthetic["Kepler"]
    residuals = observed - synthetic_kepler
    residual_rms = float(np.sqrt(np.mean(residuals**2)))
    wssr = float(cost_fns.wssr({"Kepler": observed}, {"Kepler": uncertainties}, synthetic))
    log_likelihood = float(
        cost_fns.likelihood_fn(
            {"Kepler": observed},
            {"Kepler": uncertainties},
            synthetic,
            ln_f,
        ),
    )

    _assert_close(
        "synthetic light curve",
        synthetic_kepler[CHECKPOINT_INDICES],
        EXPECTED_SYNTHETIC_CHECKPOINTS,
        rtol=SYNTHETIC_RTOL,
        atol=SYNTHETIC_ATOL,
    )
    _assert_close(
        "residuals",
        residuals[CHECKPOINT_INDICES],
        EXPECTED_RESIDUAL_CHECKPOINTS,
        rtol=SYNTHETIC_RTOL,
        atol=SYNTHETIC_ATOL,
    )
    _assert_close(
        "residual RMS",
        residual_rms,
        EXPECTED_RESIDUAL_RMS,
        rtol=AGGREGATE_RTOL,
    )
    _assert_close("WSSR", wssr, EXPECTED_WSSR, rtol=AGGREGATE_RTOL)
    _assert_close(
        "log likelihood",
        log_likelihood,
        EXPECTED_LOG_LIKELIHOOD,
        rtol=AGGREGATE_RTOL,
    )

    return {
        "dataset": str(OBSERVATIONS_PATH.relative_to(REPOSITORY_ROOT)),
        "parameter_fixture": str(PARAMETERS_PATH.relative_to(REPOSITORY_ROOT)),
        "observation_count": len(observed_phases),
        "phase_range": [float(observed_phases.min()), float(observed_phases.max())],
        "uncertainty_range": [float(raw_uncertainties.min()), float(raw_uncertainties.max())],
        "synthetic_range": [float(synthetic_kepler.min()), float(synthetic_kepler.max())],
        "residual_range": [float(residuals.min()), float(residuals.max())],
        "residual_rms": residual_rms,
        "weighted_sum_squared_residuals": wssr,
        "log_likelihood": log_likelihood,
        "ln_f": ln_f,
        "adaptive_setup_seconds": setup_seconds,
        "synthetic_and_interpolation_seconds": model_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optionally write the JSON report to this path.")
    args = parser.parse_args()

    report = run_benchmark()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    sys.stdout.write(f"{rendered}\n")
    if args.output is not None:
        args.output.write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
