"""Time the original light-curve likelihood with two FP64 blackbody implementations."""

from __future__ import annotations

import argparse
import os
import resource
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from scripts.study_native_mesh_refinement import (
    PHYSICS, Native, read, settings, sha256, store, study_positions, table_settings,
)
import numpy as np

from elisa import const
from elisa.atm import atm
from elisa.experimental.gpu_blackbody import _sample_passbands


def blocked_blackbody(temperature, **kwargs):
    """Integrate the native spectrum once per exact temperature in bounded CPU arrays."""
    bands = kwargs["passband"]
    unique, reverse = np.unique(temperature, return_inverse=True)
    result = {}
    for (name, band), (wave, response) in zip(bands.items(), _sample_passbands(bands), strict=True):
        wave_si = wave * band.wave_to_si_mult
        radiance = np.empty_like(unique, dtype=np.float64)
        rows = max(1, 262144 // len(wave))
        for start in range(0, len(unique), rows):
            stop = min(len(unique), start + rows)
            flux = atm.planck_function(wave_si[None, :], unique[start:stop, None]) * response
            np.nan_to_num(flux, copy=False)
            np.maximum(flux, 0, out=flux)
            radiance[start:stop] = atm.compute_normal_intensity(
                flux, wave, flux_mult=const.PI, wave_mult=1e-10,
            )
        result[name] = radiance[reverse]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("original", "blocked", "radiance-check"))
    parser.add_argument("fixture", type=Path)
    parser.add_argument("degrees", type=float)
    parser.add_argument("output", type=Path)
    parser.add_argument("--begin", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()
    fixture = read(args.fixture)
    points, *_ = study_positions(fixture)
    if not 0 <= args.begin < args.end <= len(points):
        parser.error("Range must fall inside the frozen proposal list")

    original = atm.NaiveInterpolatedAtm.black_body_radiance
    direct = dict(values=0, max_relative_difference=0.0, passed=True)
    calls = 0

    def instrumented(temperature, **kwargs):
        nonlocal calls
        calls += 1
        if args.mode == "original":
            return original(temperature, **kwargs)
        values = blocked_blackbody(temperature, **kwargs)
        if args.mode == "radiance-check":
            reference = original(temperature, **kwargs)
            if reference.keys() != values.keys():
                raise ValueError("Passbands differ")
            for name, expected in reference.items():
                actual = values[name]
                if actual.shape != expected.shape or np.any(expected <= 0):
                    raise ValueError("Unexpected radiance shape or reference value")
                relative = np.abs(actual - expected) / expected
                direct["values"] += actual.size
                direct["max_relative_difference"] = max(direct["max_relative_difference"], float(np.max(relative)))
                direct["passed"] &= bool(np.allclose(actual, expected, atol=1e-10, rtol=3e-13))
        return values

    with table_settings(), settings.override(**PHYSICS, NUMBER_OF_MCMC_PROCESSES=1,
                                             MAX_DISCRETIZATION_FACTOR=12, MESH_GENERATOR="trapezoidal"), (
        patch.object(atm.NaiveInterpolatedAtm, "black_body_radiance", staticmethod(instrumented))
    ):
        start = perf_counter()
        model = Native(fixture, args.degrees)
        rows = []
        for label, position in points[args.begin:args.end]:
            rows.append(dict(label=label, position=position, **model.measure(position)))
        elapsed = perf_counter() - start
    if not calls:
        raise RuntimeError("The native model did not call the selected blackbody implementation")
    store(args.output, dict(mode=args.mode, primary_degrees=args.degrees, rows=rows,
                            fit_and_likelihoods_wall_s=elapsed, blackbody_calls=calls,
                            radiance_check=direct if args.mode == "radiance-check" else None,
                            linux_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                            cpu_affinity=sorted(os.sched_getaffinity(0)),
                            fixture_sha256=sha256(args.fixture), driver_sha256=sha256(Path(__file__))))


if __name__ == "__main__":
    main()
