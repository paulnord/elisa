"""Matched original/NumPy/JAX CPU/GPU conditioned-shadow batches, not full curves."""

# ruff: noqa: INP001, T201, PLR2004, PLR0915 - bounded scientific driver
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elisa.binary_system import model
from elisa.experimental.gpu_observer_geometry import ObserverGeometryBackend
from scripts.benchmark_batched_geometry import original_geometry


def original(component: str, parameters: Any, points: Any, lines: Any) -> dict:
    rows = [original_geometry(component, p).shadow(x, los) for p, x, los in zip(parameters, points, lines, strict=True)]
    return {"blocked": np.stack([r[0] for r in rows]), "distance": np.stack([r[1] for r in rows])}


def vectorized(component: str, parameters: Any, points: Any, lines: Any) -> dict:
    """Original12Newton updates, vectorized proposals and original trig potential."""
    b, n = points.shape[:2]
    q, spin, target = [np.repeat(parameters[:, i], n) for i in range(3)]
    primary = component == "primary"
    mass = np.ones_like(q) if primary else q
    other = q if primary else np.ones_like(q)
    line = np.repeat(lines, n, axis=0)
    delta = points.reshape((-1, 3)) - np.array([0.0 if primary else 1.0, 0, 0])
    base = delta - np.sum(delta * line, axis=1)[:, None] * line
    rho = np.linalg.norm(base, axis=1)
    inner = mass / target
    selected = (rho >= inner) & (rho < 0.25)
    blocked = rho < inner
    distance = rho - inner
    x = base[selected]
    los = line[selected]
    rhos = rho[selected]
    q, spin, target, mass, other = [a[selected] for a in [q, spin, target, mass, other]]
    shift = np.array([-1.0 if primary else 1.0, 0, 0])
    t = np.zeros(len(x))

    def gradient(v: Any) -> Any:
        c = v + shift
        g = mass[:, None] * v / np.linalg.norm(v, axis=1)[:, None] ** 3
        g += other[:, None] * c / np.linalg.norm(c, axis=1)[:, None] ** 3
        g[:, :2] -= (spin**2 * (1 + q))[:, None] * v[:, :2]
        g[:, 0] += q if primary else -1.0
        return g

    for _ in range(12):
        v = x + t[:, None] * los
        g = gradient(v)
        derivative = np.zeros_like(t)
        for a, m in [(v, mass), (v + shift, other)]:
            r2 = np.sum(a * a, axis=1)
            derivative += m * (r2 - 3 * np.sum(a * los, axis=1) ** 2) / r2**2.5
        derivative -= spin**2 * (1 + q) * np.sum(los[:, :2] ** 2, axis=1)
        if np.any(derivative <= 0):
            msg = "Original stationary branch failed"
            raise RuntimeError(msg)
        t -= np.clip(np.sum(g * los, axis=1) / derivative, -rhos / 4, rhos / 4)
    local = x + t[:, None] * los
    g = gradient(local)
    if np.any(abs(t) > rhos / 2) or np.any(
        abs(np.sum(g * los, axis=1)) > 64 * np.finfo(float).eps * np.linalg.norm(g, axis=1),
    ):
        msg = "Original stationary residual gate failed"
        raise RuntimeError(msg)
    p = local.copy()
    if not primary:
        p[:, 0] *= -1
    radius = np.linalg.norm(p, axis=1)
    coeff = getattr(model, "pre_calculate_for_potential_value_" + component)(
        spin, q, 1.0, np.arctan2(p[:, 1], p[:, 0]), np.arccos(np.clip(p[:, 2] / radius, -1, 1)), return_as_tuple=True,
    )
    value = getattr(model, "potential_value_" + component)(radius, q, *coeff)
    blocked[selected] = value > target
    distance[selected] = (target - value) * rhos * rhos / mass
    return {"blocked": blocked.reshape((b, n)), "distance": distance.reshape((b, n))}


def check(a: dict, b: dict) -> float:
    np.testing.assert_array_equal(a["blocked"], b["blocked"])
    np.testing.assert_allclose(a["distance"], b["distance"], rtol=3e-13, atol=64 * np.finfo(float).eps)
    return float(np.max(abs(a["distance"] - b["distance"])))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "gpu"], required=True)
    p.add_argument("--component", choices=["primary", "secondary"], required=True)
    p.add_argument("--batch", nargs="+", type=int, default=[1, 32, 128])
    p.add_argument("--nodes", nargs="+", type=int, default=[512, 2048])
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--verified-bounded-gpu-window", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.device == "gpu" and not args.verified_bounded_gpu_window:
        msg = "Registered bounded GPU unit required"
        raise RuntimeError(msg)
    if args.repeats < 10 or any(b * n > 2097152 or b < 1 or n < 1 for b in args.batch for n in args.nodes):
        msg = "Declare10+repeats and positive totalnodes<=2097152"
        raise ValueError(msg)
    started = perf_counter()
    backend = ObserverGeometryBackend(device=args.device)
    construct = perf_counter() - started
    kernel = backend.shadow_calls[args.component]
    rows = []
    for b in args.batch:
        for n in args.nodes:
            rng = np.random.default_rng(1900 + b + n)
            parameters = np.column_stack([rng.uniform(0.6, 1.9, b), rng.uniform(0.6, 1.4, b), rng.uniform(12, 20, b)])
            angles = rng.uniform(0, 2 * np.pi, (b, n))
            rho = rng.uniform(0, 0.26, (b, n))
            lines = np.tile(np.array([0.98, 0.12, 0.15]) / np.linalg.norm([0.98, 0.12, 0.15]), (b, 1))
            u = np.cross(lines, [0.0, 0.0, 1.0])
            u /= np.linalg.norm(u, axis=1)[:, None]
            v = np.cross(lines, u)
            points = np.array([0.0 if args.component == "primary" else 1.0, 0, 0]) + rho[..., None] * (
                np.cos(angles)[..., None] * u[:, None] + np.sin(angles)[..., None] * v[:, None]
            )
            samples = []
            max_error = 0.0
            for iteration in range(args.repeats + 1):
                params = parameters + np.array([0.0001, -0.0001, 0.001]) * iteration
                coords = points + iteration * 1e-7
                los = lines.copy()
                order = ["original", "numpy", "compiled"] if iteration % 2 == 0 else ["compiled", "numpy", "original"]
                values, costs = {}, {}
                for mode in order:
                    t = perf_counter()
                    if mode == "original":
                        values[mode] = original(args.component, params, coords, los)
                    elif mode == "numpy":
                        values[mode] = vectorized(args.component, params, coords, los)
                    else:
                        placed = [
                            backend.jax.device_put(a, backend.device, may_alias=False) for a in [params, coords, los]
                        ]
                        backend.jax.tree.map(lambda x: x.block_until_ready(), placed)
                        transferred = perf_counter()
                        result = kernel(*placed)
                        backend.jax.tree.map(lambda x: x.block_until_ready(), result)
                        computed = perf_counter()
                        if not np.asarray(result["valid"]).all():
                            msg = "GPU stationary gate failed"
                            raise RuntimeError(msg)
                        values[mode] = {k: np.asarray(result[k]) for k in ["blocked", "distance"]}
                        done = perf_counter()
                        costs["placement_dispatch_sync"] = transferred - t
                        costs["device_call_sync"] = computed - transferred
                        costs["download_and_validation"] = done - computed
                    costs[mode] = perf_counter() - t
                max_error = max(
                    max_error, check(values["numpy"], values["original"]),
                    check(values["compiled"], values["original"]),
                )
                samples.append(
                    {
                        "order": order,
                        "seconds": costs,
                        "input_sha256": hashlib.sha256(params.tobytes() + coords.tobytes() + los.tobytes()).hexdigest(),
                    },
                )
            compiled = kernel.lower(*placed).compile().memory_analysis()
            rows.append(
                {
                    "batch": b,
                    "nodes": n,
                    "samples": samples,
                    "first_call_is_cold": True,
                    "max_absolute_margin_error": max_error,
                    "input_bytes": sum(a.nbytes for a in [params, coords, los]),
                    "output_bytes": sum(a.nbytes for a in values["compiled"].values()),
                    "compiler_bytes": {
                        k: getattr(compiled, k)
                        for k in ["temp_size_in_bytes", "argument_size_in_bytes", "output_size_in_bytes"]
                    },
                    "allocator_stats": backend.device.memory_stats(),
                },
            )
            report = {
                "component": args.component,
                "device": args.device,
                "rows": rows,
                "constructor_seconds": construct,
                "main_wall_seconds": perf_counter() - started,
                "host_peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "affinity": sorted(os.sched_getaffinity(0)),
                "contract": (
                    "Conditioned silhouette stage only. Original and vectorized CPU use original potential/12Newton "
                    "steps; JAX uses same stationary equations/12updates with Cartesian potential. Fresh changed "
                    "q/spin/Phi/points, matched masks/margins. Host dispatch, transfers and synchronization "
                    "included. CPU branch/seam corrections in complete path separately measured. No fullcurve, fit "
                    "or MCMC/ESS gain claimed."
                ),
            }
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"batch": b, "nodes": n, "max_error": max_error}), flush=True)


if __name__ == "__main__":
    main()
