"""Worker transport must not carry prior chains or change posterior results."""
from __future__ import annotations

import multiprocessing as mp
import os
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from elisa import settings
from elisa.analytics.binary_fit import mcmc, mcmc_diagnostics
from elisa.analytics.params.parameters import BinaryInitialParameters
from tests.tests_analytics.fitting_helpers import rv_problem
from tests.tests_analytics.test_mcmc_diagnostics_integration import AnalyticFit


class ParentOnlyHistory:
    def __getstate__(self):
        raise AssertionError("previous sampler must remain in the parent")


class LimitedPickles:
    """Fail if a fitter is serialized repeatedly as ordinary sampling tasks."""

    def __init__(self):
        self.count = 0

    def __getstate__(self):
        self.count += 1
        if self.count > 2:
            raise AssertionError("fitter was serialized after worker initialization")
        return self.__dict__.copy()


class WorkerFailure(AnalyticFit):
    def __init__(self):
        super().__init__()
        self.parent_pid = os.getpid()

    def likelihood(self, xn):
        if os.getpid() != self.parent_pid:
            raise RuntimeError("injected worker likelihood failure")
        return super().likelihood(xn)


@pytest.fixture(params=[method for method in ("fork", "spawn") if method in mp.get_all_start_methods()])
def process_context(request, monkeypatch):
    context = mp.get_context(request.param)
    monkeypatch.setattr(mcmc, "get_mp_context", lambda: context)
    monkeypatch.setattr(mcmc_diagnostics, "get_mp_context", lambda: context)
    return request.param


def sample(fit, workers):
    with (
        settings.override(NUMBER_OF_PROCESSES=1, NUMBER_OF_MCMC_PROCESSES=workers),
        patch("numpy.random.get_state", return_value=np.random.RandomState(2026).get_state()),
    ):
        return fit._fit(8, 1, 6, 2, np.linspace(0.2, 0.6, 8)[:, None], save=False)


def test_previous_sampler_is_not_transferred(process_context):
    reference = sample(AnalyticFit(), 1)
    fit = AnalyticFit()
    fit.last_sampler = ParentOnlyHistory()
    actual = sample(fit, 2)
    np.testing.assert_array_equal(actual.get_chain(), reference.get_chain())
    np.testing.assert_array_equal(actual.get_log_prob(), reference.get_log_prob())
    # Returned emcee samplers still have a usable likelihood in the parent.
    assert actual.log_prob_fn(np.array([0.5])) == fit.ln_probability(np.array([0.5]))


def test_fitter_is_serialized_only_at_worker_start(process_context):
    fit = AnalyticFit()
    fit.payload = LimitedPickles()
    sample(fit, 2)
    assert fit.payload.count == (2 if process_context == "spawn" else 0)


def test_reused_physical_rv_fit_matches_serial(process_context):
    cfg, data = rv_problem()
    cfg["nuisance"] = {"ln_f": {"value": -7.0, "fixed": True}}
    initial = np.random.default_rng(97).uniform(0.3, 0.6, (8, 3))
    serial = mcmc.CentralRadialVelocity()
    parallel = mcmc.CentralRadialVelocity()
    previous = None
    previous_chain = None
    # Change the observations on reuse: each new pool must get the new fit state.
    for shift in (0.0, 350.0):
        shifted = deepcopy(data)
        for dataset in shifted.values():
            dataset.y_data = dataset.y_data + shift
        for fit, workers in ((serial, 1), (parallel, 2)):
            with (
                settings.override(NUMBER_OF_PROCESSES=1, NUMBER_OF_MCMC_PROCESSES=workers),
                patch("numpy.random.get_state", return_value=np.random.RandomState(2026).get_state()),
            ):
                fit.fit(
                    shifted, BinaryInitialParameters(**deepcopy(cfg)), nwalkers=8,
                    nsteps=4, burn_in=1, initial_state=initial.copy(), save=False,
                )
        np.testing.assert_array_equal(parallel.last_sampler.get_chain(), serial.last_sampler.get_chain())
        np.testing.assert_array_equal(parallel.last_sampler.get_log_prob(), serial.last_sampler.get_log_prob())
        if previous is not None:
            np.testing.assert_array_equal(previous.get_chain(), previous_chain)
        previous = parallel.last_sampler
        previous_chain = previous.get_chain().copy()


def test_worker_error_preserves_previous_results_and_cleans_up(process_context):
    fit = WorkerFailure()
    previous = ParentOnlyHistory()
    fit.last_sampler = previous
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(RuntimeError, match="injected worker likelihood failure"):
        sample(fit, 2)
    assert fit.last_sampler is previous
    assert {child.pid for child in mp.active_children()} <= before
