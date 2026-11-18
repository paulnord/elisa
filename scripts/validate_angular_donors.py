"""Compare grouped donor directions and native apparent-limb crossings on the same rays."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.binary_system.surface.faces import faces_visibility_x_limits
from elisa.experimental.angular_reflection import AngularReflection, _cpu_incident, _moments
from elisa.experimental.area_reflection import AreaReflection, _emission
from scripts.validate_emitter_groups import incident
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem, prepare_fit
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_gpu_surface_error import asset_revision
from scripts.validate_reflection_area import intrinsic_state


def compare(profile):
    if profile["source_revision"] != asset_revision() or max(profile["levels"]) != 3:
        raise ValueError("Use a field-3 profile with matching atmosphere and passband fixtures")
    fit = prepare_fit(
        make_problem(jd=profile["jd"], free_nuisance=True, shape_parameters=True),
        phase_mode="observations",
        mesh_mode="paired",
    )
    physical = profile["samples"][0]["rows"][0]["physical"]
    _, binary = prepare_binary(fit, physical)
    system, _ = intrinsic_state(binary, 3)
    names = ("primary", "secondary")
    limits = faces_visibility_x_limits(system.primary.polar_radius, system.secondary.polar_radius, 1.0)
    result = []
    for cap in (1, 2):
        angular = AngularReflection(settings.LIMB_DARKENING_LAW, device="reference", donor_level=cap)
        legacy = AreaReflection(settings.LIMB_DARKENING_LAW, device="reference", donor_level=cap)
        native = [angular._surface(binary, system, name, 3, limits[name]) for name in names]
        projected = [legacy._surface(binary, system, name, 3, limits[name]) for name in names]
        groups, details = angular._prepare_pair(native, system, limits, 3)
        for i, name in enumerate(names):
            receiver, d, g, old = native[i], native[1 - i], groups[1 - i], projected[1 - i]
            rows = np.linspace(0, len(receiver.parent) - 1, 128, dtype=int)
            for warmed in (False, True):
                heat = (
                    d.intrinsic * (0.07 + 0.06 * np.sin(np.arange(len(d.intrinsic))))
                    if warmed
                    else np.zeros_like(d.intrinsic)
                )
                light = (d.intrinsic + heat)[d.visible_base] * d.visible_area
                truth = incident(receiver, rows, d.emitter, d.outward, light, d.prefix_ld)
                expanded = incident(receiver, rows, g.emitter[g.group], d.outward, light, d.prefix_ld)
                measured = (
                    _cpu_incident(
                        receiver.receiver[rows], receiver.normal[rows], g.emitter, g.outward, _moments(np, g, heat)
                    )
                    * receiver.absorption[receiver.parent[rows]]
                )
                weighted, coefficients = _emission(np, old, heat)
                before = incident(receiver, rows, old.emitter, old.outward, weighted, coefficients)
                scale = getattr(system, name).t_eff ** 4
                # Independent ELISa geometry/LD evaluation at each native orientation.
                np.testing.assert_allclose(measured, expanded, rtol=3e-13, atol=1e-15 * scale)
                errors = {k: (v - truth) / scale for k, v in (("projected", before), ("angular", measured))}
                result.append(
                    dict(
                        receiver=name,
                        cap=cap,
                        varied_luminosity=warmed,
                        samples=len(rows),
                        explicit_child_direction_max_difference=float(abs(measured - expanded).max() / scale),
                        rms={k: float(np.linalg.norm(v) / np.sqrt(len(rows))) for k, v in errors.items()},
                        maximum={k: float(abs(v).max()) for k, v in errors.items()},
                        donor=details[1 - i],
                    )
                )
    return dict(physical=physical, rows=result, unit="deposited T^4 / Teff^4", native_reference=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = args.profile.read_bytes()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        result = compare(json.loads(data))
    result["input_sha256"] = hashlib.sha256(data).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
