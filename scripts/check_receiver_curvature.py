"""Compare two conservative primary heat reconstructions against direct individual donor rays."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.experimental.area_reflection import _BARY
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import duffy_nodes
from elisa.experimental.gpu_reconstructed_field import shape_rule
from elisa.experimental.gpu_shared_fields import shared_parent_map
from elisa.experimental.receiver_curvature import curvature_stencil, heat_basis, heat_controls
from scripts.check_primary_spectrum import eclipse_rays, native_heat, physical_power, prepared
from scripts.compare_irradiating_sources import final_sweep
from scripts.validate_gpu_atmosphere import table_settings


def field(base, level):
    _, model, curves, system, _, diagnostics = prepared(base, level)
    star = system.primary
    n = star.base_symmetry_faces_number
    tri = curves[0].moment_backends["primary"].thermal_triangles(star, 26)
    geom = ConditionedRocheGeometry(model, "primary")
    rays, weights = duffy_nodes(tri[:n], 4)
    weights = (weights * geom.sample(rays)[2]).reshape(n, 16)
    mass = (weights / weights.sum(axis=1, keepdims=True)) @ heat_basis(np, shape_rule()[0])
    diag = diagnostics["primary"]
    samples, means = diag["reflection_quadrature"]["receiver_deposits"]
    stencil = curvature_stencil(star.faces, tri, n)
    controls, limited = heat_controls(np, samples, means, stencil, mass)
    np.testing.assert_allclose((mass * controls).sum(axis=1), means, atol=1e-17, rtol=3e-13)
    update = final_sweep(model, level)["primary"]

    def sample(parents, bary, rays, xyz):
        _, linear, _ = physical_power(model, system, tri, diag, parents, bary, xyz)
        gids = star.face_symmetry_vector[parents]
        curved = (heat_basis(np, bary) * controls[gids]).sum(axis=1)
        native = native_heat(update, system, parents, xyz, geom.sample(rays)[1])
        return dict(linear=linear, quadratic=curved, direct=native)

    def mean_check(parents):
        # Retain physical parents on both sides of the symmetry boundaries. At each
        # sample apply the native emitter horizon and receiver visibility tests.
        gids = star.face_symmetry_vector[parents]
        estimates = []
        for nodes, reference_weights in ((_BARY, np.full(3, 1 / 3)), shape_rule()):
            directions = nodes @ tri[parents]
            xyz, normals, jac = geom.sample(directions.reshape(-1, 3))
            jac = jac.reshape(len(parents), -1)
            # The common angular determinant cancels from the face mean.
            w = jac * reference_weights / np.linalg.norm(directions, axis=-1) ** 3
            values = native_heat(update, system, np.repeat(parents, len(nodes)), xyz, normals)
            estimates.append((w * values.reshape(w.shape)).sum(axis=1) / w.sum(axis=1))
        three, sixteen = estimates
        return {
            name: summarize(delta)
            for name, delta in (
                ("stored_minus_direct_three", means[gids] - three),
                ("direct_three_minus_sixteen", three - sixteen),
                ("stored_minus_direct_sixteen", means[gids] - sixteen),
            )
        }

    return (
        model,
        tri,
        sample,
        mean_check,
        dict(quarter_faces=n, limited=int(limited.sum()), zero_mean=int(np.sum(means == 0))),
    )


def summarize(delta):
    return dict(rms=float(np.sqrt(np.mean(delta**2))), mean=float(delta.mean()), peak=float(abs(delta).max()))


def compare(base):
    model, ct, coarse, coarse_mean, cc = field(base, 2)
    _, ft, fine, fine_mean, fc = field(base, 3)
    parent, _ = shared_parent_map(ft, ct, 1)
    phases = {}
    for phase, ids, bary, rays, xyz, _ in eclipse_rays(model, ft):
        ci = parent[ids]
        cb = np.linalg.solve(ct[ci].transpose(0, 2, 1), rays[..., None])[..., 0]
        cb /= cb.sum(axis=1, keepdims=True)
        if not np.all(cb >= 0):
            raise ArithmeticError("Held-out ray escaped its coarse thermal parent")
        c, f = coarse(ci, cb, rays, xyz), fine(ids, bary, rays, xyz)
        stats = {}
        for name, sl in (("all", slice(None)), ("rim", slice(96)), ("disk", slice(96, None))):
            subset = {"direct_level_change": summarize((f["direct"] - c["direct"])[sl])}
            for method in ("linear", "quadratic"):
                low, high = (v[method] - v["direct"] for v in (c, f))
                subset[method] = {
                    k: summarize(v[sl])
                    for k, v in (
                        ("coarse_minus_direct", low),
                        ("fine_minus_direct", high),
                        ("residual_change", high - low),
                        ("field_change", f[method] - c[method]),
                    )
                }
            stats[name] = subset
        ix = np.arange(0, len(ids), 8)
        phases[phase] = dict(
            rays=len(ids),
            rim_rays=96,
            errors=stats,
            parent_mean_check=dict(sampled_rays=len(ix), coarse=coarse_mean(ci[ix]), fine=fine_mean(ids[ix])),
        )
    return dict(units="deposited power / primary Teff^4", coarse=cc, fine=fc, phases=phases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        result = compare(json.loads(args.baseline.read_bytes()))
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
