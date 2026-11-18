"""Compare recovered irradiating flux against independently reflected, finer CPU cells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import Observer, settings
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_nodal_source import ReceiverSourceLightCurve
from elisa.experimental.gpu_shared_fields import shared_parent_map
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture


def compare(atlas: str, level: int) -> list[dict]:
    """Evaluate heat at the four child centres using the actual parent thermal sampler."""
    case = {"discretization": 12, "inclination": 85, "spin": [0.85, 1.15]}
    binary = (table_fixture if atlas == "ck04" else fixture)(case)
    bands = ["Generic.Bessell.V", "bolometric"]
    observer = Observer(passband=bands, system=binary)
    renderer = ReceiverSourceLightCurve(
        bands, CurvedEdgeRule(3, 1), device="reference", levels=(level, level + 1), integration_level=level + 1
    )
    systems = []
    for variant in renderer.variants:
        engine = renderer.engines[variant.key]
        engine.keep_states = True
        star_system, triangles = engine._fields(binary, observer)
        systems.append((star_system, triangles, engine.last_diagnostics))
    (low, cones, initial), (high, child_cones, child_initial) = systems
    rows = []
    for name, backend in renderer.moment_backends.items():
        star, child = getattr(low, name), getattr(high, name)
        count, nchild = star.base_symmetry_faces_number, child.base_symmetry_faces_number
        parent, _ = shared_parent_map(child_cones[name][:nchild], cones[name][:count], 1)
        rays = child_cones[name][:nchild].mean(axis=1)
        bary = np.linalg.solve(cones[name][parent].transpose(0, 2, 1), rays[:, :, None])[:, :, 0]
        bary /= bary.sum(axis=1)[:, None]
        baseline = initial[name]["intrinsic_temperatures"]
        # Subtract two production thermal evaluations at identical corner gravity and area.
        previous = backend.thermal_nodes(binary, star)[1]
        _, reconstructed, _, control = backend.thermal_nodes(binary, star, baseline)
        slope = ((reconstructed / star.t_eff) ** 4 - (previous / star.t_eff) ** 4).reshape(-1, 3)
        mean = (star.temperatures[:count] / star.t_eff) ** 4 - (baseline[:count] / star.t_eff) ** 4
        new = mean[parent] + np.sum(bary * slope[parent], axis=1)
        expected = (child.temperatures[:nchild] / child.t_eff) ** 4 - (
            child_initial[name]["intrinsic_temperatures"][:nchild] / child.t_eff
        ) ** 4
        weights = child.areas[:nchild]
        rows.append(
            {
                "atlas": atlas,
                "component": name,
                "level": level,
                "samples": nchild,
                "rms_constant": float(np.sqrt(np.average((mean[parent] - expected) ** 2, weights=weights))),
                "rms_reconstructed": float(np.sqrt(np.average((new - expected) ** 2, weights=weights))),
                "maximum_absolute_constant": float(np.max(np.abs(mean[parent] - expected))),
                "maximum_absolute_reconstructed": float(np.max(np.abs(new - expected))),
                "control": control,
            }
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with (
        table_settings(),
        settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2, USE_SINGLE_LD_COEFFICIENTS=False),
    ):
        rows = [r for atlas in ("bb", "ck04") for level in (0, 1) for r in compare(atlas, level)]
    args.output.write_text(json.dumps({"unit": "deposited T^4 / Teff^4", "rows": rows}, indent=2, allow_nan=False))
