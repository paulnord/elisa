"""Bounded Roche-to-passband chunk experiment with matching complete downloads.

Run only in an authorized idle-device window. This measures chunk choices for
fixed supplied topology, not observed light, full synthesis or MCMC performance.
"""

# ruff: noqa: INP001, T201 - standalone benchmark with JSON output
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from benchmark_gpu_pipeline import compare, cpu_reference, directions, outputs, parameters, time_call

from elisa import Observer
from elisa.experimental.gpu_blackbody import RocheBlackbodyPipeline
from elisa.experimental.gpu_pipeline import RocheGeometryPipeline

_MAX_ROWS, _MAX_DIRECTIONS, _MAX_REPEATS, _MAX_CHUNK = 512, 1024, 50, 128
_MIN_DIRECTIONS = 16


def main() -> None:
    """Keep chunks bounded and preserve correctness gates during timing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu", "gpu"))
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--directions", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--chunks", nargs="+", type=int, default=[32, 8, 16, 64, 128, 32])
    parser.add_argument("--max-payload-mib", type=float, default=256)
    args = parser.parse_args()
    if not 1 <= args.rows <= _MAX_ROWS or not _MIN_DIRECTIONS <= args.directions <= _MAX_DIRECTIONS:
        parser.error("Rows/directions exceed bounded experiment")
    if not 1 <= args.repeats <= _MAX_REPEATS or any(not 1 <= c <= _MAX_CHUNK for c in args.chunks):
        parser.error("Repeats/chunks exceed bounded experiment")
    if not np.isfinite(args.max_payload_mib) or args.max_payload_mib <= 0:
        parser.error("Payload budget must be positive and finite")
    import jax  # noqa: PLC0415 - optional explicit device experiment

    phi, theta, faces = directions(args.directions)
    params = parameters(args.rows, thermal=True)
    bands = Observer(passband=["Generic.Bessell.B", "Generic.Bessell.V", "bolometric"]).passband
    records = []
    for component in ["primary", "secondary"]:
        geometry = RocheGeometryPipeline(component, phi, theta, faces, device=args.device).block_until_ready()
        reference = cpu_reference(component, phi, theta, faces, params, bands)
        for ordinal, chunk in enumerate(args.chunks):
            estimate = args.rows * (args.directions * 12 + len(faces) * (41 + 3 * chunk)) * 8
            if estimate > args.max_payload_mib * 1024**2:
                msg = "Estimated payload exceeds limit (compiler/allocator peak not included)"
                raise ValueError(msg)
            started = time.perf_counter()
            pipeline = RocheBlackbodyPipeline(geometry, bands, chunk_size=chunk).block_until_ready()
            setup = time.perf_counter() - started
            batch = pipeline.prepare(*params).block_until_ready()
            started = time.perf_counter()
            result = jax.block_until_ready(batch.evaluate())
            cold = time.perf_counter() - started
            errors = compare(outputs(result), reference, cpu=True)
            with jax.transfer_guard("disallow_explicit"):
                jax.block_until_ready(batch.evaluate())
            resident = time_call(lambda batch=batch: jax.block_until_ready(batch.evaluate()), args.repeats)
            inclusive = time_call(
                lambda pipeline=pipeline: outputs(pipeline.prepare(*params).evaluate()),
                args.repeats,
            )
            records.append(
                {
                    "component": component,
                    "ordinal": ordinal,
                    "chunk": chunk,
                    "estimated_payload_bytes": estimate,
                    "fixed_quadrature_bytes": pipeline.quadrature_nbytes,
                    "quadrature_setup_upload_seconds": setup,
                    "compile_first_seconds": cold,
                    "resident_physics_seconds_lower_bound": resident,
                    "parameters_to_matching_download_seconds": inclusive,
                    "vs_original_cpu_errors": errors,
                    "strict_transfer_guard_passed": True,
                },
            )
    print(
        json.dumps(
            {
                "scope": "Chunk tuning of changing Roche geometry/gravity/temperature and emitted B/V/bolometric flux",
                "device": str(geometry.device),
                "device_kind": geometry.device.device_kind,
                "jax": jax.__version__,
                "precision": "float64",
                "rows": args.rows,
                "directions": args.directions,
                "faces": len(faces),
                "repeats": args.repeats,
                "baseline_repeated_at_end": args.chunks[0] == args.chunks[-1],
        "caveat": (
            "Fixed topology, no correction/irradiation/LD/eclipse/observed light/full fit. "
            "CPU oracle once per component; full timings include identical outputs "
            "and parameter upload; cold setup separate."
        ),
                "results": records,
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
