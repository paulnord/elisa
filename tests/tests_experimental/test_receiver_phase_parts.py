"""Checks for independently reduced stellar-field diagnostics in reflected phase blocks."""

from copy import deepcopy

import numpy as np
import pytest

from scripts.receiver_phase_parts import compare_field_controls


def test_repeated_field_diagnostics():
    heat = 0.0001803366710172654
    first = {
        "fine": {
            "primary": {
                "deposited_T4_area_mean_over_Teff4": heat,
                "receiver_samples": 120,
                "quadratic_T4_relative_error": 1e-15,
            }
        }
    }
    second = deepcopy(first)
    star = second["fine"]["primary"]
    star["deposited_T4_area_mean_over_Teff4"] = np.nextafter(heat, np.inf)
    star["quadratic_T4_relative_error"] = 2e-15
    assert compare_field_controls(first, second) == np.spacing(heat)
    for key, value in (
        ("receiver_samples", 119),
        ("deposited_T4_area_mean_over_Teff4", heat * 1.00001),
        ("quadratic_T4_relative_error", 3.01e-13),
    ):
        bad = deepcopy(second)
        bad["fine"]["primary"][key] = value
        with pytest.raises(ValueError):
            compare_field_controls(first, bad)
