# Copyright 2026 ELISa contributors
# keep it first
# due to astropy units/constants implementation
from unittests import set_astropy_units

import numpy as np

from scripts.benchmark.real_data.kic4851217_mcmc import (
    EXPECTED_FREE_PARAMETERS,
    PROFILES,
    chain_diagnostics,
    normalized_parameter_space,
    seeded_initial_state,
    tutorial_mcmc_parameters,
)

set_astropy_units()


class _ShortSampler:
    def __init__(self) -> None:
        self._chain = np.full((2, 20, 10), 0.5)
        self._chain[1, 0, 0] = 0.51
        self._log_probability = np.full((2, 20), -100.0)
        self.acceptance_fraction = np.array([0.5] + [0.0] * 19)

    def get_chain(self) -> np.ndarray:
        return self._chain

    def get_log_prob(self) -> np.ndarray:
        return self._log_probability


def test_tutorial_mcmc_parameterization() -> None:
    initial = tutorial_mcmc_parameters()
    labels, center, normalization = normalized_parameter_space(initial)

    assert labels == EXPECTED_FREE_PARAMETERS
    assert len(labels) == 10
    assert np.all((center >= 0.0) & (center <= 1.0))
    assert normalization["system@mass_ratio"] == (0.001, 2.0)
    assert normalization["primary@t_eff"] == (3500, 10000)
    assert normalization["nuisance@ln_f"] == (-10, 2)
    assert initial["system"]["phase_shift"]["fixed"] is True
    assert initial["system"]["mass_ratio"]["sigma"] == 0.04
    assert initial["primary"]["t_eff"]["sigma"] == 380


def test_seeded_smoke_initial_state_is_reproducible_and_bounded() -> None:
    _, center, _ = normalized_parameter_space(tutorial_mcmc_parameters())
    first = seeded_initial_state(center, PROFILES["smoke"], seed=4851217)
    second = seeded_initial_state(center, PROFILES["smoke"], seed=4851217)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], center)
    assert first.shape == (20, 10)
    assert np.all((first >= 0.0) & (first <= 1.0))
    assert np.linalg.matrix_rank(first - first.mean(axis=0)) == center.size


def test_seeded_tutorial_initial_state_spans_prior() -> None:
    _, center, _ = normalized_parameter_space(tutorial_mcmc_parameters())
    state = seeded_initial_state(center, PROFILES["tutorial"], seed=4851217)

    assert state.shape == (100, 10)
    np.testing.assert_array_equal(state[0], center)
    assert np.all((state >= 0.0) & (state <= 1.0))
    assert np.ptp(state[1:], axis=0).min() > 0.8


def test_short_chain_is_execution_valid_but_not_converged() -> None:
    diagnostics = chain_diagnostics(_ShortSampler(), EXPECTED_FREE_PARAMETERS)

    assert diagnostics["chain_shape"] == [2, 20, 10]
    assert diagnostics["all_log_probabilities_finite"] is True
    assert diagnostics["usable_log_probability_count"] == 40
    assert diagnostics["moved_walker_count"] == 1
    assert diagnostics["accepted_walker_count"] == 1
    assert diagnostics["converged"] is False
    assert "autocorrelation unavailable" in diagnostics["convergence_reasons"][0]
