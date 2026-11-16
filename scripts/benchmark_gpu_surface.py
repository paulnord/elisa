"""Actual Roche-face kernel throughput, NOT a whole-model speedup benchmark.

Repeat prepared phase/component rows to measure scaling. Repeated rows are not
independent changing-geometry walkers. Run only in an approved idle allocation.
"""

# ruff: noqa: INP001 - standalone command-line script
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from elisa import BinarySystem, Observer, Star, ld, settings
from elisa.base.curves import utils as curve_utils
from elisa.experimental.gpu_surface import SurfaceFluxIntegrator


def capture_surfaces(discretization: float, law: str) -> tuple[list[Any], float, float]:
    """Capture real eclipse surface records; interception confined to this process."""
    settings.configure(LIMB_DARKENING_LAW=law, NUMBER_OF_PROCESSES=1)
    coefficients = [0.5] if law == "linear" else [0.3, 0.2]
    common = {
        "mass": 1,
        "surface_potential": 5,
        "synchronicity": 1,
        "albedo": 0.5,
        "gravity_darkening": 0.32,
        "atmosphere": "bb",
        "limb_darkening_coefficients": {"bolometric": coefficients},
        "discretization_factor": discretization,
    }
    binary = BinarySystem(
        Star(t_eff=6500, **common),
        Star(t_eff=5500, **common),
        inclination=88,
        period=3,
        eccentricity=0,
        argument_of_periastron=90,
    )
    captured = []
    original = curve_utils.flux_from_star_container

    def capture(band: str, star: Any) -> float:
        captured.append(copy.deepcopy(star))
        return original(band, star)

    started = time.perf_counter()
    with patch.object(curve_utils, "flux_from_star_container", capture):
        Observer(passband=["bolometric"], system=binary).lc(
            phases=np.array([0, 0.01, 0.025, 0.04, 0.06, 0.1, 0.25, 0.45, 0.48, 0.5, 0.51, 0.54, 0.75]),
            normalize=False,
        )
    return captured, time.perf_counter() - started, binary.semi_major_axis


def pack(stars: list[Any], rows: int) -> tuple[list[Any], tuple[np.ndarray, ...]]:
    """Repeat physical snapshots, padding to a fixed maximum face count."""
    selected = [stars[i % len(stars)] for i in range(rows)]
    width = max(len(star.indices) for star in selected)
    ncoeff = selected[0].ld_cfs["bolometric"].shape[1]
    r, mu, cov = (np.zeros((rows, width)) for _ in range(3))
    coeff = np.zeros((rows, width, ncoeff))
    for i, star in enumerate(selected):
        n = len(star.indices)
        r[i, :n], mu[i, :n], cov[i, :n], coeff[i, :n] = (
            star.normal_radiance["bolometric"][star.indices],
            star.los_cosines[star.indices],
            star.coverage[star.indices],
            star.ld_cfs["bolometric"][star.indices],
        )
    return selected, (r, mu, cov, coeff)


def time_call(fn: Any, repeats: int) -> dict[str, float]:
    """Median synchronized wall time; caller supplies synchronization."""
    fn()  # warm each distinct timed path
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {"median": float(np.median(samples)), "min": min(samples), "max": max(samples)}


def main() -> None:  # noqa: PLR0915, C901 - sequential benchmark stages and safety gates
    """Measure CPU oracle and selected device under explicitly bounded workloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 32, 128, 512])
    parser.add_argument("--discretization", type=float, default=10)
    parser.add_argument("--law", choices=("linear", "logarithmic", "square_root"), default="linear")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-input-mib", type=float, default=256)
    args = parser.parse_args()
    if not (np.isfinite(args.max_input_mib) and np.isfinite(args.discretization)):
        parser.error("Memory budget and discretization must be finite")
    if min(args.rows) < 1 or args.repeats < 1 or args.max_input_mib <= 0 or args.discretization < 3:  # noqa: PLR2004 - workload safety minimum
        parser.error("Positive rows/repeats/memory required; discretization must be at least 3 degrees")
    import jax  # noqa: PLC0415 - optional dependency

    kernel = SurfaceFluxIntegrator(args.law, device=args.device)
    stars, setup, semi_major_axis = capture_surfaces(args.discretization, args.law)
    records = []
    for rows in args.rows:
        width = max(len(s.indices) for s in stars)
        estimate = rows * width * (3 + stars[0].ld_cfs["bolometric"].shape[1]) * 8
        if estimate > args.max_input_mib * 1024**2:
            records.append({"rows": rows, "skipped": "input-memory budget", "estimated_input_bytes": estimate})
            continue
        started = time.perf_counter()
        selected, arrays = pack(stars, rows)
        packing = time.perf_counter() - started
        started = time.perf_counter()
        prepared = kernel.prepare(*arrays)
        # Synchronize transfers without compiling/executing the surface kernel.
        prepared.block_until_ready()
        transfer = time.perf_counter() - started
        started = time.perf_counter()
        actual = jax.block_until_ready(prepared.evaluate())
        compile_first = time.perf_counter() - started
        covered = np.concatenate([s.coverage[s.indices] for s in selected])
        if not (np.any(covered == 0) and np.any(covered > 0)):
            msg = "Workload must include both eclipsed and uncovered faces"
            raise ValueError(msg)
        fractions = np.concatenate([
            s.coverage[s.indices] / (s.areas[s.indices] * semi_major_axis**2) for s in selected
        ])
        categories = {
            "zero": int(np.count_nonzero(fractions == 0)),
            "partial": int(np.count_nonzero((fractions > .01) & (fractions < .99))),  # noqa: PLR2004
            "full": int(np.count_nonzero(fractions > .999)),  # noqa: PLR2004
        }
        if not all(categories.values()):
            msg = "Workload must include zero, partial and fully uncovered faces"
            raise ValueError(msg)
        expected = [curve_utils.calculate_surface_element_fluxes("bolometric", s) for s in selected]
        face, totals = map(np.asarray, actual)
        max_scaled = 0.0
        for i, ref in enumerate(expected):
            np.testing.assert_allclose(face[i, : len(ref)], ref, rtol=3e-13, atol=0)
            scale = max(float(np.max(np.abs(ref), initial=0)), np.finfo(float).tiny)
            max_scaled = max(max_scaled, float(np.max(np.abs(face[i, : len(ref)] - ref), initial=0)) / scale)
        reference_totals = np.array([x.sum() for x in expected])
        np.testing.assert_allclose(totals, reference_totals, rtol=3e-13, atol=0)

        def cpu_oracle(selected: list[Any] = selected) -> list[float]:
            return [curve_utils.flux_from_star_container("bolometric", s) for s in selected]

        resident = time_call(lambda prepared=prepared: jax.block_until_ready(prepared.evaluate()), args.repeats)
        inclusive = time_call(lambda arrays=arrays: jax.block_until_ready(kernel(*arrays)), args.repeats)
        cpu = time_call(cpu_oracle, args.repeats)

        def observer_path(rows: int = rows) -> np.ndarray:
            fresh = [stars[i % len(stars)] for i in range(rows)]
            # Includes original indexing, host packing, validation, H2D, kernel and D2H totals.
            return np.asarray(kernel.from_containers("bolometric", fresh)[1])

        def numpy_batch(arrays: tuple[np.ndarray, ...] = arrays) -> np.ndarray:
            r, mu, cov, c = arrays
            correction = ld.limb_darkening_factor(
                cos_theta=mu.reshape(-1),
                coefficients=c.reshape(-1, c.shape[-1]),
                limb_darkening_law=args.law,
            ).reshape(mu.shape)
            return (r * ((mu * cov) * correction)).sum(axis=1)

        np.testing.assert_allclose(numpy_batch(), reference_totals, rtol=3e-13, atol=0)
        np.testing.assert_allclose(observer_path(), reference_totals, rtol=3e-13, atol=0)
        full = time_call(observer_path, args.repeats)
        numpy_seconds = time_call(numpy_batch, args.repeats)
        records.append(
            {
                "rows": rows,
                "coverage_face_counts": categories,
                "padded_faces_per_row": prepared.shape[1],
                "selected_faces": sum(len(s.indices) for s in selected),
                "input_bytes": prepared.nbytes,
                "packing_seconds": packing,
                "validation_transfer_seconds": transfer,
                "compile_first_seconds": compile_first,
                "resident_kernel_seconds_lower_bound": resident,
                "validation_transfer_kernel_seconds_no_download": inclusive,
                "original_cpu_oracle_seconds": cpu,
                "gather_transfer_kernel_totals_download_seconds": full,
                "numpy_padded_batch_seconds": numpy_seconds,
                "max_face_error_divided_by_row_max_flux": max_scaled,
                "max_total_relative_error": float(
                    np.max(
                        np.abs(totals - reference_totals) / np.maximum(np.abs(reference_totals), np.finfo(float).tiny),
                    ),
                ),
            },
        )
    print(  # noqa: T201 - JSON command-line report
        json.dumps(
            {
                "purpose": "surface kernel throughput only; repeated fixed physical snapshots, not full-model walkers",
                "device": str(kernel.device),
                "device_kind": kernel.device.device_kind,
                "jax": jax.__version__,
                "cpu_count": os.cpu_count(),
                "thread_environment": {
                    k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
                },
                "python": platform.python_version(),
                "dtype": "float64",
                "law": args.law,
                "discretization_degrees": args.discretization,
                "cpu_physical_capture_seconds": setup,
                "captured_rows": len(stars),
                "repeats": args.repeats,
                "results": records,
                "caveat": ("No full-model speedup/ESS claim. Resident kernel is a lower bound. "
                "Only gather_transfer_kernel_totals_download includes output download. "
                "Input budget excludes compiler/output workspace."),
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
