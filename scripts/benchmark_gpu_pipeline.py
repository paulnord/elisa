"""Bounded changing Roche geometry/gravity benchmark, including result downloads.

Fixed directions/connectivity, changing physical proposal rows. This does not
benchmark full ELISa synthesis, changing eclipses, flux, MCMC or ESS. Cold setup,
compilation, warm resident work and host-inclusive calls are reported separately.
"""

# ruff: noqa: INP001, T201 - standalone benchmark command with JSON output
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

from elisa import const, utils
from elisa.base.surface.faces import calculate_normals, calculate_surface_centres
from elisa.base.surface.temperature import calculate_effective_temperatures
from elisa.binary_system import model
from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.binary_system.surface.mesh import get_surface_points
from elisa.const import Position
from elisa.experimental.gpu_gravity import RocheSurfaceGravity
from elisa.experimental.gpu_mesh import TriangleGeometry
from elisa.experimental.gpu_pipeline import RocheGeometryPipeline
from elisa.experimental.gpu_roche import RocheRaySolver
from elisa.experimental.gpu_thermal import GravityDarkeningTemperature

_MIN_DIRECTIONS, _MAX_DIRECTIONS = 16, 4096
_MAX_ROWS, _MAX_REPEATS = 4096, 100
_THERMAL_PARAMETER_COUNT = 9


def directions(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed spherical directions and supplied convex-hull connectivity."""
    i = np.arange(count)
    theta = np.arccos(1 - 2 * (i + 0.5) / count)
    phi = (i * np.pi * (3 - np.sqrt(5))) % (2 * np.pi)
    unit = utils.spherical_to_cartesian(np.column_stack([np.ones(count), phi, theta]))
    return phi, theta, ConvexHull(unit).simplices


def parameters(rows: int, *, thermal: bool = False) -> tuple[np.ndarray, ...]:
    """Vary q, rotation, potential, SI scales and observer orientation per row."""
    t = np.linspace(0, 1, rows) if rows > 1 else np.array([0.5])
    base = (
        0.5 + 1.5 * t,
        0.5 + t,
        8 + 6 * t,
        (1 + t) * 1e30,
        (3 + 3 * t) * 1e9,
        -np.pi + 2 * np.pi * t,
        0.2 + (np.pi - 0.4) * t,
    )
    return (*base, 4500 + 4500 * t, 0.1 + 0.9 * t) if thermal else base


def outputs(result: dict[str, Any]) -> dict[str, np.ndarray]:
    """Download the SAME physics outputs for both JAX paths, synchronizing them."""
    values = {
        "radii": result["rays"]["radii"],
        "areas": result["geometry"]["areas"],
        "normals": result["geometry"]["normals"],
        "los_cosines": result["geometry"]["los_cosines"],
        "face_gravity": result["gravity"]["face_gravity"],
        "proposal_valid": result["proposal_valid"],
    }
    if "thermal" in result:
        values.update({key: result["thermal"][key] for key in ["temperatures", "surface_flux", "luminosity_ratio"]})
    return {key: np.asarray(value) for key, value in values.items()}


def cpu_reference(
    component: str,
    phi: np.ndarray,
    theta: np.ndarray,
    faces: np.ndarray,
    params: tuple[np.ndarray, ...],
) -> dict[str, np.ndarray]:
    """Run original ELISa Newton mesh and CPU geometry/gravity helpers per row."""
    q, spin, potential, mass, axis, az, inc = params[:7]
    rows = []
    com = 0.0 if component == "primary" else 1.0
    for row in range(len(q)):
        points = get_surface_points(
            phi,
            theta,
            0.1,
            1.0,
            getattr(model, f"pre_calculate_for_potential_value_{component}"),
            getattr(model, f"potential_{component}_fn"),
            getattr(model, f"radial_{component}_potential_derivative"),
            potential[row],
            q[row],
            spin[row],
        )
        radii = np.linalg.norm(points, axis=1)
        if component == "secondary":
            points[:, 0] = 1 - points[:, 0]
        normals = calculate_normals(points, faces, calculate_surface_centres(points, faces), com)
        rotated = utils.rotate_item(normals, Position(row, 1.0, az[row], 0.0, 0.0), inc[row])
        gradient = calculate_potential_gradient(1.0, component, points, spin[row], q[row])
        acceleration = (const.G * mass[row] / axis[row] ** 2) * gradient
        rows.append(
            {
                "radii": radii,
                "areas": utils.triangle_areas(faces, points),
                "normals": normals,
                "los_cosines": -rotated[:, 0],
                "face_gravity": np.mean(np.linalg.norm(acceleration, axis=1)[faces], axis=1),
            },
        )
        if len(params) == _THERMAL_PARAMETER_COUNT:
            t_eff, beta = params[7:]
            entry = rows[-1]
            view = SimpleNamespace(
                areas=entry["areas"],
                potential_gradient_magnitudes=entry["face_gravity"],
                polar_potential_gradient_magnitude=float(np.median(entry["face_gravity"])),
                t_eff=t_eff[row],
                gravity_darkening=beta[row],
                has_spots=lambda: False,
                symmetry_test=lambda: False,
            )
            temperatures = calculate_effective_temperatures(view, entry["face_gravity"])
            entry.update(
                temperatures=temperatures,
                surface_flux=const.STEFAN_BOLTZMAN_CONST * temperatures**4,
                luminosity_ratio=np.sum(entry["areas"] * temperatures**4) / (np.sum(entry["areas"]) * t_eff[row] ** 4),
            )
    result = {key: np.stack([r[key] for r in rows]) for key in rows[0]}
    result["proposal_valid"] = np.all(
        np.column_stack([np.isfinite(value).reshape(len(q), -1).all(axis=1) for value in result.values()]),
        axis=1,
    )
    return result


def time_call(fn: Any, repeats: int) -> dict[str, Any]:
    """Warm a path, then report synchronized wall times for complete calls."""
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {"median": float(np.median(samples)), "min": min(samples), "max": max(samples), "samples": samples}


def time_download(prepared: Any, repeats: int, jax: Any) -> dict[str, Any]:
    """Time downloads of FRESH results, excluding their prior synchronized work.

    Re-reading one JAX result can reuse its cached host value; it would not
    measure another device transfer. Each sample here receives new arrays.
    """
    samples = []
    for _ in range(repeats):
        fresh = jax.block_until_ready(prepared.evaluate())
        started = time.perf_counter()
        outputs(fresh)
        samples.append(time.perf_counter() - started)
    return {"median": float(np.median(samples)), "min": min(samples), "max": max(samples), "samples": samples}


def compare(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray], *, cpu: bool) -> dict[str, Any]:
    """Apply existing independent-Newton tolerances and retain numerical errors."""
    summary = {}
    atol = {
        "radii": 1e-13,
        "areas": 1e-15,
        "normals": 1e-12,
        "los_cosines": 1e-12,
        "face_gravity": 0.0,
        "temperatures": 0.0,
        "surface_flux": 0.0,
        "luminosity_ratio": 0.0,
    }
    for key, reference in expected.items():
        if key == "proposal_valid":
            if not np.array_equal(actual[key], reference) or not actual[key].all():
                msg = "Unsupported or inconsistent proposal in benchmark"
                raise ValueError(msg)
            continue
        thermal = key in ["temperatures", "surface_flux", "luminosity_ratio"]
        np.testing.assert_allclose(
            actual[key],
            reference,
            rtol=3e-10 if cpu and not thermal else 3e-13,
            atol=atol[key] if cpu else 64 * np.finfo(float).eps,
            err_msg=key,
        )
        nonzero = reference != 0
        summary[key] = {
            "max_abs_error": float(np.max(np.abs(actual[key] - reference))),
            "max_relative_error_nonzero": float(
                np.max(np.abs((actual[key][nonzero] - reference[nonzero]) / reference[nonzero]), initial=0),
            ),
        }
    return summary


def main() -> None:  # noqa: C901, PLR0915 - explicit sequential measurements
    """Benchmark admitted physical batches within explicit size limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--components", choices=("primary", "secondary"), nargs="+", default=["primary", "secondary"])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--directions", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-payload-mib", type=float, default=256)
    parser.add_argument("--thermal", action="store_true", help="Include fresh intrinsic gravity-darkened temperatures")
    args = parser.parse_args()
    if not np.isfinite(args.max_payload_mib) or args.max_payload_mib <= 0:
        parser.error("Payload budget must be positive and finite")
    if not _MIN_DIRECTIONS <= args.directions <= _MAX_DIRECTIONS:
        parser.error("Direction count must be between 16 and 4096")
    if min(args.rows) < 1 or max(args.rows) > _MAX_ROWS or not 1 <= args.repeats <= _MAX_REPEATS:
        parser.error("Rows must be in [1,4096], repeats in [1,100]")
    import jax  # noqa: PLC0415 - optional dependency

    phi, theta, faces = directions(args.directions)
    records = []
    for component in args.components:
        started = time.perf_counter()
        pipeline = RocheGeometryPipeline(component, phi, theta, faces, device=args.device).block_until_ready()
        topology_seconds = time.perf_counter() - started
        ray = RocheRaySolver(component, device=args.device)
        geometry, gravity = TriangleGeometry(device=args.device), RocheSurfaceGravity(component, device=args.device)
        temperature_kernel = GravityDarkeningTemperature(device=args.device)
        prepare = pipeline.prepare_thermal if args.thermal else pipeline.prepare
        com = 0.0 if component == "primary" else 1.0
        for rows in args.rows:
            # This is a payload estimate, NOT an allocator/compiler peak bound.
            estimate = rows * (args.directions * 12 + len(faces) * (32 if args.thermal else 24)) * 8
            if estimate > args.max_payload_mib * 1024**2:
                records.append(
                    {"component": component, "rows": rows, "skipped": "payload budget", "estimated_bytes": estimate},
                )
                continue
            params = parameters(rows, thermal=args.thermal)
            started = time.perf_counter()
            prepared = prepare(*params).block_until_ready()
            upload_seconds = time.perf_counter() - started
            started = time.perf_counter()
            result = jax.block_until_ready(prepared.evaluate())
            cold_fused = time.perf_counter() - started
            fused_output = outputs(result)
            if not fused_output["proposal_valid"].all():
                msg = "Benchmark proposals are outside the supported kernel domain"
                raise ValueError(msg)

            def staged(
                params: tuple[np.ndarray, ...] = params,
                *,
                ray: Any = ray,
                geometry: Any = geometry,
                gravity: Any = gravity,
                component: str = component,
                rows: int = rows,
                com: float = com,
                temperature_kernel: Any = temperature_kernel,
            ) -> dict[str, np.ndarray]:
                q, spin, potential, mass, axis, az, inc = params[:7]
                rays = ray(q, spin, potential, phi, theta)
                vertices = np.array(rays["points_local"], copy=True)  # intermediate D2H bridge
                if component == "secondary":
                    vertices[..., 0] = 1 - vertices[..., 0]
                mesh = geometry(vertices, faces, np.full(rows, com), az, inc)  # first vertex H2D
                field = gravity(vertices, faces, q, spin, mass, axis)  # second vertex H2D
                valid = rays["proposal_valid"] & mesh["proposal_valid"] & field["proposal_valid"]
                result = {"rays": rays, "geometry": mesh, "gravity": field, "proposal_valid": valid}
                if len(params) == _THERMAL_PARAMETER_COUNT:
                    temperature = temperature_kernel(field["face_gravity"], mesh["areas"], *params[7:])
                    result.update(thermal=temperature, proposal_valid=valid & temperature["proposal_valid"])
                return outputs(result)

            started = time.perf_counter()
            staged_output = staged()
            cold_staged = time.perf_counter() - started
            staged_error = compare(fused_output, staged_output, cpu=False)
            started = time.perf_counter()
            reference = cpu_reference(component, phi, theta, faces, params)
            cold_cpu = time.perf_counter() - started
            cpu_error = compare(fused_output, reference, cpu=True)

            # Prove that repeated evaluation needs no H2D, D2H or D2D transfer.
            with jax.transfer_guard("disallow_explicit"):
                jax.block_until_ready(prepared.evaluate())
            resident = time_call(lambda prepared=prepared: jax.block_until_ready(prepared.evaluate()), args.repeats)
            download = time_download(prepared, args.repeats, jax)
            inclusive = time_call(
                lambda params=params, prepare=prepare: outputs(prepare(*params).evaluate()), args.repeats,
            )
            bridge = time_call(staged, args.repeats)
            original = time_call(
                lambda params=params, component=component: cpu_reference(component, phi, theta, faces, params),
                args.repeats,
            )
            record = {
                "component": component,
                "rows": rows,
                "thermal": args.thermal,
                "vertices": args.directions,
                "faces": len(faces),
                "shared_topology_bytes": pipeline.topology_nbytes,
                "proposal_input_bytes": prepared.nbytes,
                "output_bytes": sum(x.nbytes for x in fused_output.values()),
                "staged_vertex_bridge_bytes_lower_bound": 3 * rows * args.directions * 3 * 8,
                "estimated_payload_bytes": estimate,
                "topology_setup_upload_seconds": topology_seconds,
                "initial_parameter_validation_upload_seconds": upload_seconds,
                "fused_compile_first_seconds": cold_fused,
                "staged_compile_first_including_download_seconds": cold_staged,
                "original_cpu_first_seconds": cold_cpu,
                "resident_physics_seconds_lower_bound": resident,
                "repeated_download_seconds": download,
                "fused_parameters_to_download_seconds": inclusive,
                "staged_host_bridge_to_download_seconds": bridge,
                "original_elisa_cpu_seconds": original,
                "fused_vs_staged_errors": staged_error,
                "fused_vs_original_cpu_errors": cpu_error,
                "strict_transfer_guard_passed": True,
            }
            records.append(record)
    print(
        json.dumps(
            {
                "scope": (
                    "Changing q/spin/potential fixed-direction Roche geometry/gravity; optional intrinsic temperature. "
                    "No observed light, coverage or MCMC."
                ),
                "device": str(pipeline.device),
                "device_kind": pipeline.device.device_kind,
                "dtype": "float64",
                "jax": jax.__version__,
                "python": platform.python_version(),
                "directions": args.directions,
                "repeats": args.repeats,
                "thermal": args.thermal,
                "thread_environment": {
                    k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
                },
                "result_contract": (
                    "radii, face areas/normals/LOS cosines, face SI gravity, proposal validity; "
                    "plus temperatures/emitted surface flux/energy closure when thermal=true"
                ),
                "caveat": (
                    "Resident timing is a lower bound. Full paths include identical output downloads; "
                    "compilation is separate. CPU helpers use original Newton and NumPy physics; "
                    "device retains additional unsupported-state guards. No candidate-dependent topology/mesh "
                    "correction, irradiation/atmosphere update, eclipse or whole-model speedup claim."
                ),
                "results": records,
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
