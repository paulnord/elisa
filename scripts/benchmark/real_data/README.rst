KIC 4851217 real-data regression
=================================

``kic4851217.py`` evaluates standard ELISa once at the fixed parameters and
assumptions used by tutorial 12. It folds the bundled pulsation-removed Kepler
observations, generates the tutorial's adaptive 100-point light curve at
discretization 10, interpolates it to every observed phase, and compares the
normalized synthetic curve, residuals, weighted residual sum, and Gaussian log
likelihood with a checked-in numerical fingerprint. Runtime fields are reported
for context only and are never pass/fail criteria.

Provenance
----------

* Observations: ``jupyter_tutorials/demo_data/lc_data/KIC04851217.lc.pulse_removed.dat``.
  The repository history introduced this file in commit
  ``aeb153d0703c63d017b76cee70ec956c54ab85d9``. It contains one header line and
  exactly 4,998 Kepler flux/error rows.
* Ephemeris and numerical settings: ``jupyter_tutorials/demo12-lc_fitting.ipynb``;
  period 2.47028 d, primary minimum BJD 54953.900507000006, discretization 10,
  adaptive interpolation threshold 100.
* Fixed model parameters: the tutorial's saved least-squares demonstration
  result, ``jupyter_tutorials/demo_data/assets/lc_least_squares.json`` (introduced
  in commit ``eb262bb63382d702febd0304f946de48faa122e5``).
* The reference fingerprint was generated with the default CK04 atmospheres and
  Van Hamme 2019 limb-darkening tables from ``mikecokina/elisa-assets`` commit
  ``5a61582fb954b1f98ceb598653fdc392f38a942a``. The reference archive SHA-256
  values are ``92b8b4ab94b9d8b95ae5552b98cd8a53297ee39d46a3b7aedff409aaf39123af``
  (``ck04.tar.gz``) and
  ``9aece679efe64991154e3b16978330bd67e221487aa0c1fc37226125d439cb7e``
  (``ld_vh19.tar.gz``).

Scope and interpretation
------------------------

This is a regression of standard ELISa behavior on a real observational cadence,
not independent validation of the astrophysical model and not a converged fit.
The pulsation-removed input remains real data, so structured residuals can expose
changes that synthetic-only tests miss. Conversely, agreement only shows that the
current implementation reproduces the recorded tutorial calculation within the
documented numerical tolerances. The saved MCMC parameters and chain are never
loaded: tutorial 12 explicitly describes that chain as unconverged, so it is not
treated as scientific truth.

The observed phase/error checkpoints use a 5e-12 absolute tolerance, close to
floating-point phase-folding noise. Synthetic checkpoints allow 2e-5 relative
plus 2e-6 absolute error (about 20 ppm at unit flux). Aggregate residual and
likelihood metrics allow 5e-4 relative error because uncertainty weighting
amplifies small cross-platform surface-integration changes over 4,998 points.

Running
-------

Configure ELISa with the standard passband, CK04 atmosphere, and Van Hamme 2019
limb-darkening assets, then run from the repository root::

    python -m scripts.benchmark.real_data.kic4851217

Use ``--output report.json`` to save the report. A missing support asset is an
environment/setup failure, not a skipped regression. The benchmark performs no
optimization and does not enforce runtime thresholds. The separate MCMC benchmark
below does sample parameters.

Actual MCMC baseline
--------------------

``kic4851217_mcmc.py`` executes the public ``LCBinaryAnalyticsTask`` MCMC API on
the same 4,998 observations. It reconstructs the tutorial 12 MCMC input from the
least-squares demonstration result exactly as the notebook does:

* phase shift becomes fixed;
* mass ratio becomes variable with value 1.08, Gaussian sigma 0.04, and bounds
  0.001--2.0;
* primary temperature becomes variable with value 7022 K, Gaussian sigma 380 K,
  and bounds 3500--10000 K;
* ``ln_f`` becomes variable with value -5 and bounds -10--2;
* the other least-squares free parameters, fixed parameters, constraints, and
  bounds are retained.

This produces ten free dimensions. The explicit seeded state is supplied in
normalized coordinates because ELISa's MCMC API accepts states on [0, 1]. This
makes backend comparisons reproducible without comparing exact stochastic
posterior samples.

Three profiles are provided:

``smoke``
  The bounded execution regression: 20 walkers, one burn-in step, two production
  steps, discretization 10, and 24 uniform interpolation points. Walkers start in
  a seeded local cloud (scale 0.002 in normalized coordinates) around the
  least-squares result. The reduced interpolation count and local initialization
  are explicit runtime concessions; all 4,998 observations and all ten tutorial
  parameters remain in the likelihood.

``tutorial``
  Reproduces the commented notebook call's 100 walkers, 200 production steps,
  zero burn-in, discretization 10, 100 interpolation points, and adaptive sample
  setup. The notebook left initial states unspecified, which ELISa implements as
  draws across the normalized prior. This profile supplies the equivalent draws
  explicitly from seed 4851217 so the run is repeatable.

``long``
  A diagnostic mode with 100 walkers, 200 burn-in steps, and 1,000 production
  steps at the tutorial's discretization/interpolation settings. It uses the
  local seeded initialization. This is a request for a convergence assessment,
  not a promise that convergence will be achieved.

Run the bounded profile with::

    python -m scripts.benchmark.real_data.kic4851217_mcmc \
        --profile smoke --output smoke.json

``--nwalkers``, ``--nsteps``, ``--burn-in``, and
``--interpolation-points`` permit controlled scaling experiments. The JSON
records versions, parameters, physical bounds, initial-state hash, elapsed time,
chain/log-probability shapes, normalized bounds, usable versus ELISa sentinel log
probabilities, acceptance, movement, and convergence diagnostics.

Validation and interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The smoke regression requires a correctly shaped chain, finite production log
probabilities, normalized samples within the declared bounds, at least one usable
non-sentinel likelihood, and accepted movement when there is more than one
production step. It does not assert exact samples, posterior percentiles, timing,
or acceptance fractions.

Convergence is reported only when all of these conservative criteria hold:

* finite positive integrated autocorrelation estimates for every dimension;
* at least 50 autocorrelation times in the production chain;
* at least 1,000 estimated effective samples in every dimension; and
* mean acceptance fraction between 0.1 and 0.7.

The 2026-11-18 reference smoke run used seed 4851217, Python 3.12.14, NumPy
2.5.5, SciPy 1.18.1, emcee 3.1.6, and ELISa 1.0.0.dev0. It completed in
194.98 s with chain shape ``(2, 20, 10)``. All 40 production log probabilities
were finite and usable, no sentinel likelihoods were present, 15 walkers had
nonzero acceptance, nine moved between stored production states, and mean
acceptance was 0.525. The initial-state SHA-256 was
``a5e6e2335ea64144eee424374bd912e2b3bd5515c6a72274e4562491b1691bb3``.
The run is explicitly **not converged** because two production steps cannot
estimate autocorrelation time.

Passing this benchmark establishes execution correctness through standard CPU
ELISa's tutorial fitting path and provides a reproducible baseline for later
backend comparisons. It does not prove equality to every historical ELISa build,
validate the astrophysical assumptions, or establish a scientifically adequate
posterior. The old bundled MCMC result and chain remain excluded because the
notebook identifies them as unconverged; they are neither loaded nor used as
reference truth. The ``tutorial`` and ``long`` profiles were not run for this
change because their expected cost is many hours; that omission must remain
visible in run reports and review notes.
