"""Matched geometry-stage batch costs; not whole light curves or MCMC/ESS.

GPU execution is only for an hold-aware, registered bounded Ada window.
This driver does not allocate, pause or restore production. CPU tests run under
run_cpu.py; GPU runs must use the background worker's existing safety driver.
"""

# ruff: noqa: C901, PLR0912, E402, INP001, PLR2004, PLR0915, SLF001, T201 - standalone staged benchmark
from time import perf_counter

PROCESS_START = perf_counter()
import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import elisa.experimental.gpu_batched_geometry as geometry_module
from elisa.binary_system import model
from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import duffy_nodes
from elisa.experimental.gpu_batched_geometry import BatchedRocheGeometry
from scripts.batched_geometry_cases import triangles


def original_geometry(component: Any, parameters: Any) -> Any:
    q, spin, potential = parameters
    star = SimpleNamespace(synchronicity=spin, surface_potential=potential)
    binary = SimpleNamespace(
        mass_ratio=q,
        morphology="detached",
        eccentricity=0.0,
        has_spots=lambda: False,
        has_pulsations=lambda: False,
        primary=star,
        secondary=star,
    )
    # Metadata only: both backends receive identical already-declared geometry
    # parameters. This deliberately excludes full BinarySystem construction.
    return ConditionedRocheGeometry(binary, component)


def numpy_batch(component: Any, parameters: Any, vectors: Any) -> Any:
    """Vectorized original potential/gradient, original 56 bisections.

    This is a separate CPU algorithm baseline alongside original per-proposal
    RocheGeometry.sample. The compiled port uses 80 iterations and Cartesian
    potential arithmetic; its JAX CPU and GPU paths share that implementation.
    """
    proposals, nodes = vectors.shape[:2]
    d = vectors.reshape((-1, 3)).copy()
    d /= np.linalg.norm(d, axis=1)[:, None]
    local = d.copy()
    if component == "secondary":
        local[:, 0] *= -1
    q, spin, target = [np.repeat(parameters[:, i], nodes) for i in range(3)]
    coeff = getattr(model, f"pre_calculate_for_potential_value_{component}")(
        spin, q, 1.0, np.arctan2(local[:, 1], local[:, 0]),
        np.arccos(np.clip(local[:, 2], -1, 1)), return_as_tuple=True,
    )
    potential = getattr(model, f"potential_value_{component}")
    lo, hi = np.full(len(d), 1e-6), np.full(len(d), 0.25)
    if np.any(potential(lo, q, *coeff) <= target) or np.any(potential(hi, q, *coeff) >= target):
        msg = "Original reference is outside its declared radial bracket"
        raise RuntimeError(msg)
    for _ in range(56):
        mid = (lo + hi) / 2
        above = potential(mid, q, *coeff) > target
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    radius = (lo + hi) / 2
    centre = np.array([0.0 if component == "primary" else 1.0, 0.0, 0.0])
    points = centre + radius[:, None] * d
    gradient = calculate_potential_gradient(1.0, component, points, spin, q)
    magnitude = np.linalg.norm(gradient, axis=1)
    radial = np.einsum("ij,ij->i", gradient, d)
    residual = potential(radius, q, *coeff) - target
    valid = (
        np.isfinite(gradient).all(axis=1)
        & (radial > 0)
        & (magnitude > 0)
        & (np.abs(residual) <= 3e-13 * np.maximum(1.0, target))
    )
    return {
        "points": points.reshape((proposals, nodes, 3)),
        "normals": (gradient / magnitude[:, None]).reshape((proposals, nodes, 3)),
        "area_per_solid_angle": (radius**2 * magnitude / radial).reshape((proposals, nodes)),
        "proposal_valid": valid.reshape((proposals, nodes)).all(axis=1),
    }


def original_loop(component: Any, parameters: Any, vectors: Any) -> Any:
    rows = [original_geometry(component, p).sample(v) for p, v in zip(parameters, vectors, strict=True)]
    return {
        "points": np.stack([row[0] for row in rows]),
        "normals": np.stack([row[1] for row in rows]),
        "area_per_solid_angle": np.stack([row[2] for row in rows]),
        "proposal_valid": np.ones(len(parameters), dtype=bool),
    }


def cpu_areas(component: Any, parameters: Any, faces: Any, *, batched: Any) -> Any:
    proposals, nfaces = faces.shape[:2]
    if batched:
        nodes, weights = duffy_nodes(faces.reshape((-1, 3, 3)), 4)
        jacobian = numpy_batch(component, parameters, nodes.reshape((proposals, -1, 3)))["area_per_solid_angle"]
        areas = (weights.reshape((proposals, nfaces, 16)) * jacobian.reshape((proposals, nfaces, 16))).sum(axis=-1)
    else:
        output = []
        for p, f in zip(parameters, faces, strict=True):
            nodes, weights = duffy_nodes(f, 4)
            jacobian = original_geometry(component, p).sample(nodes)[2]
            output.append((weights * jacobian).reshape((nfaces, 16)).sum(axis=1))
        areas = np.stack(output)
    return {"areas": areas, "total_area": areas.sum(axis=1), "proposal_valid": np.ones(proposals, dtype=bool)}


def assert_physics(actual: Any, expected: Any) -> Any:
    errors = {}
    if not actual["proposal_valid"].all() or not expected["proposal_valid"].all():
        msg = "Unsupported geometry must stop the benchmark"
        raise AssertionError(msg)
    for key in expected:
        if key == "proposal_valid":
            continue
        # Existing per-geometry floating-point control, not an inferred-fit gate.
        absolute = 64 * np.finfo(float).eps if key in ("points", "normals") else 0.0
        np.testing.assert_allclose(actual[key], expected[key], rtol=3e-13, atol=absolute)
        denominator = np.abs(expected[key])
        nonzero = denominator > 0
        errors[key] = {
            "max_absolute": float(np.max(np.abs(actual[key] - expected[key]))),
            "max_relative_nonzero": float(
                np.max(np.abs((actual[key][nonzero] - expected[key][nonzero]) / expected[key][nonzero])),
            ),
        }
    return errors


def device_call(engine: Any, part: Any, parameters: Any, geometry: Any) -> Any:
    start = perf_counter()
    pp = engine.jax.device_put(parameters, engine.device)
    gg = engine.jax.device_put(geometry, engine.device)
    active = engine.jax.device_put(np.ones(geometry.shape[:2], dtype=bool), engine.device)
    engine.jax.tree.map(lambda x: x.block_until_ready(), (pp, gg, active))
    transferred = perf_counter()
    if part == "sample":
        result = engine.sample(pp, gg, active=active, compact=True)
        keys = ("points", "normals", "area_per_solid_angle", "proposal_valid")
    else:
        result = engine.areas(pp, gg, active=active, order=4)
        keys = ("areas", "total_area", "proposal_valid")
    engine.jax.tree.map(lambda x: x.block_until_ready(), result)
    computed = perf_counter()
    host = {key: np.asarray(result[key]) for key in keys}
    downloaded = perf_counter()
    return (
        host,
        {
            "host_input_to_host_output_seconds": downloaded - start,
            "input_placement_and_synchronization_seconds": transferred - start,
            "device_call_and_synchronization_seconds": computed - transferred,
            "host_download_seconds": downloaded - computed,
        },
        (pp, gg, active),
    )


def main() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--part", choices=["sample", "area"], default="sample")
    parser.add_argument("--component", choices=["primary", "secondary"], default="primary")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--nodes", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--verified-bounded-gpu-window", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "gpu" and not args.verified_bounded_gpu_window:
        msg = "GPU driver requires the worker's registered hold-aware bounded allocation"
        raise RuntimeError(msg)
    if args.repeats < 2 or any(x <= 0 for x in args.batch + args.nodes):
        msg = "Use positive batch/node counts and at least two changed repeats"
        raise ValueError(msg)
    t = perf_counter()
    engine = BatchedRocheGeometry(args.component, device=args.device)
    construction = perf_counter() - t
    rows = []
    for proposals in args.batch:
        for requested in args.nodes:
            rng = np.random.default_rng(6111700 + proposals + requested)
            parameters = np.column_stack(
                [rng.uniform(0.6, 1.9, proposals), rng.uniform(0.6, 1.4, proposals),
                 rng.uniform(12.0, 20.0, proposals)],
            )
            if args.part == "sample":
                geometry = rng.normal(size=(proposals, requested, 3))
                geometry /= np.linalg.norm(geometry, axis=-1)[..., None]
                count = requested
                cpu0 = partial(original_loop, args.component)
                cpu1 = partial(numpy_batch, args.component)
            else:
                # Octahedral angular patches at a fixed declared refinement,
                # replicated/rotated per proposal, not a replacement stellar mesh.
                if requested not in (8, 32, 128):
                    msg = "Area --nodes specifies face counts 8,32 or128"
                    raise ValueError(msg)
                level = {8: 0, 32: 1, 128: 2}[requested]
                base = triangles(level)
                geometry = np.broadcast_to(base, (proposals, *base.shape)).copy()
                for i in range(proposals):
                    angle = 0.013 * i
                    rotation = np.array(
                        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]],
                    )
                    geometry[i] = geometry[i] @ rotation.T
                count = requested * 16
                cpu0 = partial(cpu_areas, args.component, batched=False)
                cpu1 = partial(cpu_areas, args.component, batched=True)
            # The whole preparation is shared. Cold cases are first calls for
            # this process/shape; filesystem/compiler caches may already exist.
            original = cpu0(parameters, geometry)
            batched = cpu1(parameters, geometry)
            batch_parity = assert_physics(batched, original)
            result, first, placed = device_call(engine, args.part, parameters, geometry)
            parity = assert_physics(result, original)
            costs = {"original_cpu_loop": [], "batched_numpy_cpu": [], "compiled": []}
            orders = []
            for iteration in range(args.repeats):
                changed = parameters.copy()
                changed[:, 0] += (iteration + 1) * 0.0001
                changed[:, 1] -= (iteration + 1) * 0.0002
                changed[:, 2] += (iteration + 1) * 0.001
                changed_geometry = geometry.copy()
                # Same shape, actual changing ray/patch directions; no cached
                # physical result is reused across changed proposal batches.
                angle = 0.0003 * (iteration + 1)
                rotation = np.array(
                    [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]],
                )
                changed_geometry = changed_geometry @ rotation.T
                order = ["original_cpu_loop", "batched_numpy_cpu", "compiled"]
                if iteration % 2:
                    order.reverse()
                values, durations = {}, {}
                for backend in order:
                    start = perf_counter()
                    if backend == "compiled":
                        values[backend], cost, placed = device_call(engine, args.part, changed, changed_geometry)
                        costs[backend].append(cost)
                    else:
                        values[backend] = (cpu0 if backend == "original_cpu_loop" else cpu1)(changed, changed_geometry)
                        durations[backend] = perf_counter()-start
                        costs[backend].append(durations[backend])
                assert_physics(values["batched_numpy_cpu"], values["original_cpu_loop"])
                assert_physics(values["compiled"], values["original_cpu_loop"])
                orders.append(order)
            compiled = (engine._sample_compact if args.part == "sample" else engine._areas[4]).lower(*placed).compile()
            memory = compiled.memory_analysis()
            row = {
                "proposals": proposals,
                "nodes_per_proposal": count,
                "input_shape": list(geometry.shape),
                "first_complete_compiled_call": first,
                "warm_changed_batches": costs,
                "paired_execution_order": orders,
                "device_allocator_stats_after_shape": engine.device.memory_stats(),
                "input_bytes": parameters.nbytes+geometry.nbytes+proposals*geometry.shape[1],
                "output_bytes": sum(a.nbytes for a in result.values()),
                "medians_seconds": {
                    "original_cpu_loop": float(np.median(costs["original_cpu_loop"])),
                    "batched_numpy_cpu": float(np.median(costs["batched_numpy_cpu"])),
                    "compiled_host_to_host": float(
                        np.median([v["host_input_to_host_output_seconds"] for v in costs["compiled"]]),
                    ),
                },
                "compiled_memory_accounting_bytes": {
                    key: getattr(memory, key, None)
                    for key in (
                        "temp_size_in_bytes",
                        "argument_size_in_bytes",
                        "output_size_in_bytes",
                        "alias_size_in_bytes",
                    )
                },
                "parity_compiled_vs_original": parity,
                "parity_numpy_batch_vs_original": batch_parity,
            }
            rows.append(row)
            print(json.dumps({"proposals": proposals, "nodes": count, **row["medians_seconds"]}), flush=True)
            report = {
                "state": "in_progress",
                "device": args.device,
                "component": args.component,
                "part": args.part,
                "rows": rows,
            }
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report.update(
        state="complete",
        constructor_seconds=construction,
        tested_checkout_head=subprocess.check_output(["/usr/bin/git", "rev-parse", "HEAD"], text=True).strip(),
        process_wall_seconds=perf_counter() - PROCESS_START,
        peak_process_host_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        affinity=sorted(os.sched_getaffinity(0)),
        thread_environment={k: os.getenv(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(geometry_module.__file__)]
        },
        limitations=[
            "Geometry stage only; no full-light-curve/likelihood/MCMC/ESS claim.",
            "Circular detached bounded inner branches; actual angular topology supplied by caller.",
            ("Original and vectorized NumPy controls use original potential/gradient and56bisections; "
             "compiled JAX CPU/GPU share Cartesian arithmetic and80bisections."),
            ("Host input/output and synchronization included; input generation/reference/parity/memory "
             "diagnostics are in command wall, outside per-path call timers."),
            "Compiled memory is not total allocator/compiler/runtime GPU or process memory.",
            "Shared-machine preliminary stage timings; cold-process shape calls may see caches.",
        ],
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
