"""Check eclipse topology, both stellar fluxes and the physical likelihood at each available resolution."""

import numpy as np

from scripts import sample_reemission_chain as chain
from scripts import sample_reflected_binary as sampler
from scripts.benchmark_stationary_eclipse import inputs
from scripts.compare_gpu_joint_budget import curve
from scripts.compare_reemission_draws import gaussian_from_components
from scripts.compare_reflected_posteriors import mapped_density
from scripts.receiver_phase_parts import compare_field_controls, join


def read_grid(root, checkpoint, injection, rule, depth, *, compact=False):
    state, fit, vector, key, prior, base = inputs(checkpoint, injection, compact=compact)
    parts, hashes, stops = [], [], []
    for part in (0, 1):
        binding = chain.block_binding(key, rule, depth, part)
        path = chain.block_path(root / "checkpoint.json", state["step"] + 1, rule, depth, part)
        if not path.exists():
            stop = sampler.read_record(root / f"{rule}-{depth}-{part}.json")
            if not (injection or compact) or stop.get("status") != "node_cap" or stop.get("binding") != binding:
                raise ValueError("Missing or incorrect phase-block stop receipt")
            stops.append(part)
            continue
        block = chain.read_block(path, binding)
        if block["backend"] != state["backend"]:
            raise ValueError("Unexpected field backend")
        parts.append(block)
        hashes.append(block["quadrature_sha256"])
    if stops:
        return dict(stops=stops, curves={})
    full = join(base, key, parts, symmetric_interior=4 if rule == "six" else None, node_reemission=True)
    sampler.validate_physics(fit, vector, full["profiles"], state["backend"])
    curves = {}
    max_error = 0.0
    for profile in full["profiles"]:
        row = profile["samples"][0]["rows"][0]
        for level in (2, 3):
            name = f"curvature_field{level}"
            c = curve(row, name)
            props = row["physical"]
            pred, density = mapped_density(fit, props, prior, c)
            independent, observed = gaussian_from_components(
                c.component_fluxes, dict(additional_light=c.additional_light, ln_f=props["nuisance@ln_f"]), fit, prior
            )
            for band, y in pred.items():
                np.testing.assert_allclose(y, observed[band], rtol=3e-13, atol=2e-12)
            np.testing.assert_allclose(independent, density, rtol=0, atol=1e-6)
            max_error = max(max_error, abs(independent - density))
            curves[profile["donor_level"], level] = (row["finite_metadata"][name], c, pred, density)
    return dict(
        curves=curves,
        fields=parts[0]["fields"],
        records=row["phase_records"],
        hashes=hashes,
        stops=[],
        physics=props,
        max_independent_error=max_error,
    )


def compare(first, second, *, same_partition=True):
    if first["stops"] != second["stops"]:
        raise ArithmeticError("The resource stops differ")
    if first["stops"]:
        return dict(stopped_parts=first["stops"], curves=0)
    if (
        first["physics"] != second["physics"]
        or first["records"] != second["records"]
        or first["curves"].keys() != second["curves"].keys()
        or len(first["curves"]) != 4
    ):
        raise ArithmeticError("The physical system or eclipse refinement differs")
    if same_partition and first["hashes"] != second["hashes"]:
        raise ArithmeticError("The physical quadrature arrays differ on the same phase blocks")
    compare_field_controls(first["fields"], second["fields"])
    errors = dict(component_relative=0.0, raw_relative=0.0, normalized=0.0, density=0.0)
    for key, (meta, x, xp, xd) in first["curves"].items():
        other, y, yp, yd = second["curves"][key]
        if sampler.digest(meta) != sampler.digest(other):
            raise ArithmeticError("The physical curve metadata differs")
        for a, b, name, relative in [
            (x.component_fluxes, y.component_fluxes, "component_relative", True),
            (x.raw_fluxes, y.raw_fluxes, "raw_relative", True),
            (x.fluxes, y.fluxes, "normalized", False),
            (xp, yp, "normalized", False),
        ]:
            for band, arr in a.items():
                delta = abs(arr - b[band])
                np.testing.assert_allclose(arr, b[band], rtol=3e-13, atol=0 if relative else 2e-12)
                if relative:
                    delta = delta[arr != 0] / np.abs(arr[arr != 0])
                errors[name] = max(errors[name], float(np.max(delta, initial=0)))
        np.testing.assert_allclose(xd, yd, rtol=0, atol=1e-6)
        errors["density"] = max(errors["density"], abs(xd - yd))
    return dict(
        curves=4,
        phases=len(first["records"]),
        exact_phase_topology=True,
        same_partition_hashes=first["hashes"] == second["hashes"],
        errors=errors,
        max_independent_density_error=max(first["max_independent_error"], second["max_independent_error"]),
    )
