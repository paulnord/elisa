"""The standard ELISa ensemble pool can call one independent surface owner."""

from copy import deepcopy
from importlib.util import find_spec

import numpy as np
import pytest

from elisa import settings
from elisa.analytics.binary_fit.mcmc import DetachedLightCurveFit
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.experimental.gpu_walker_pool import SharedNativeDetachedMCMCFit, shared_surface_owner
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem


@pytest.mark.skipif(find_spec("jax") is None, reason="optional JAX dependency")
def test_two_walker_processes_share_one_fp64_owner(monkeypatch):
    monkeypatch.setenv("JAX_ENABLE_X64", "True")
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2,
                                             NUMBER_OF_PROCESSES=1, NUMBER_OF_MCMC_PROCESSES=2):
        problem = make_problem()
        initial = np.random.default_rng(17).uniform(0.35, 0.65, (6, 2))
        with shared_surface_owner("linear", device="cpu") as service:
            original = DetachedLightCurveFit()
            shared = SharedNativeDetachedMCMCFit(service, "linear")
            rng = np.random.get_state()
            try:
                for fit in (original, shared):
                    fit.atmosphere_model = deepcopy(problem.atmosphere)
                    fit.limb_darkening_coefficients = deepcopy(problem.ld)
                    np.random.seed(19)
                    fit.fit(deepcopy(problem.data), BinaryInitialParameters(**deepcopy(problem.config)),
                            **problem.setup, nwalkers=6, nsteps=2, burn_in=0,
                            initial_state=initial.copy(), save=False, progress=False)
            finally:
                np.random.set_state(rng)
            actual, expected = shared.last_sampler, original.last_sampler
            np.testing.assert_array_equal(actual.get_chain(), expected.get_chain())
            np.testing.assert_allclose(actual.get_log_prob(), expected.get_log_prob(), atol=1e-6, rtol=0)
            stats = service.statistics()
            assert stats["platform"] == "cpu" and stats["x64"] and stats["calls"] >= 18
