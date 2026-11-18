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
optimization and does not enforce runtime thresholds.
