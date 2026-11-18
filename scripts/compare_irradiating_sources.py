"""Separate emitter geometry, intrinsic power and re-emitted heat on common receiving rays."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.experimental.area_reflection import _MAX_PAIRS
from elisa.experimental.gpu_reflection import UnsupportedReflectionError
from elisa.experimental.receiver_heating import SampledReceiverReflection
from scripts.check_receiver_samples import _CaptureSweep
from scripts.receiver_phase_parts import context
from scripts.validate_emitter_groups import incident
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_reflection_area import intrinsic_state


def final_sweep(binary, level):
    system, _ = intrinsic_state(binary, level)
    solver = _CaptureSweep("linear", device="reference", receiver_points=3, donor_level=2)
    solver.updates = []
    captured, _ = solver.evaluate(binary, system, level, 2)
    sampled = SampledReceiverReflection("linear", device="reference", receiver_points=3, donor_level=2)
    expected, _ = sampled.evaluate(binary, system, level, 2)
    for name in ("primary", "secondary"):
        np.testing.assert_allclose(captured[name], expected[name], rtol=3e-13, atol=0)
    order = sorted(("primary", "secondary"), key=lambda name: getattr(system, name).t_eff)
    return dict(zip(order, solver.updates[-2:], strict=True))


def incoming(receiver, rows, donor, power):
    """Use individual emitter positions and horizon predicates, including for grouped donors."""
    points = donor.emitter if donor.spatial is None else donor.spatial.base.native_positions
    normals = donor.outward if donor.spatial is None else donor.native_normals
    if len(points) != len(power) or len(rows) * len(power) > _MAX_PAIRS:
        raise UnsupportedReflectionError("Illumination diagnostic exceeds the native pair budget")
    return incident(receiver, rows, points, normals, donor.visible_area * power, donor.prefix_ld)


def source_changes(coarse_update, fine_update, teff):
    _, coarse, old_heat, _ = coarse_update
    receiver, fine, new_heat, _ = fine_update
    rows = np.linspace(0, len(receiver.parent) - 1, min(96, len(receiver.parent)), dtype=int)
    if len(fine.intrinsic) != 4 * len(coarse.intrinsic):
        raise ValueError("Expected successive nested emitting fields")
    native = fine.visible_base
    parent = native // 4
    # The incident integral is linear in the emitting T^4. Hold the fine geometry and LD
    # fixed while changing the two powers separately at the same physical receiving rays.
    old = incoming(receiver, rows, coarse, (coarse.intrinsic + old_heat)[coarse.visible_base])
    transported = incoming(receiver, rows, fine, (coarse.intrinsic + old_heat)[parent])
    intrinsic = incoming(receiver, rows, fine, fine.intrinsic[native] - coarse.intrinsic[parent])
    reemitted = incoming(receiver, rows, fine, new_heat[native] - old_heat[parent])
    new = incoming(receiver, rows, fine, (fine.intrinsic + new_heat)[native])
    terms = dict(geometry_area_LD=transported - old, intrinsic_T4=intrinsic, reemitted_T4=reemitted, total=new - old)
    if any(not np.isfinite(v).all() for v in terms.values()):
        raise ArithmeticError("Nonfinite irradiating-source difference")
    np.testing.assert_allclose(transported + intrinsic + reemitted, new, rtol=3e-13, atol=0)
    result = {}
    for key, change in terms.items():
        fraction = change / teff**4
        result[key] = dict(rms=float(np.sqrt(np.mean(fraction**2))), maximum=float(np.max(np.abs(fraction))))
    return dict(
        receiver_rays=len(rows),
        coarse_emitting_nodes=len(coarse.visible_area),
        fine_emitting_nodes=len(native),
        changes=result,
    )


def compare(baseline):
    _, _, _, _, binary = context(baseline)
    coarse, fine = (final_sweep(binary, level) for level in (2, 3))
    return {name: source_changes(coarse[name], fine[name], getattr(binary, name).t_eff) for name in fine}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with (
        table_settings(),
        settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2, LIMB_DARKENING_LAW="linear"),
    ):
        result = compare(json.loads(args.baseline.read_bytes()))
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
