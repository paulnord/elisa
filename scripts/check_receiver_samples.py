"""Check the reconstructed heating slope on native receiver rays during the last reflection sweep."""

import argparse
from copy import copy
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.experimental.area_reflection import _BARY, _MAX_PAIRS
from elisa.experimental.gpu_nodal_source import _NodalBackend
from elisa.experimental.quadratic_reflection import QuadraticReflection
from scripts.reflected_trial import fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_reflection_area import intrinsic_state


class _CaptureSweep(QuadraticReflection):
    def _reference_update(self, receiver, donor, deposited):
        avg = super()._reference_update(receiver, donor, deposited)
        self.updates.append((receiver, donor, deposited, avg))
        return avg


def measure(binary, system, name, temps, update):
    receiver, donor, donor_heat, mean = update
    star = copy(getattr(system, name))
    intrinsic = star.temperatures
    star.temperatures = temps
    backend = _NodalBackend(name, "linear", device="reference")
    _, varying, _, _ = backend.thermal_nodes(binary, star, intrinsic)
    _, flat, _, _ = backend.thermal_nodes(binary, star)
    teff4 = star.t_eff**4
    slope = ((varying / star.t_eff) ** 4 - (flat / star.t_eff) ** 4).reshape(-1, 3)
    all_faces = np.flatnonzero(np.bincount(receiver.parent, minlength=len(mean)) == 3)
    pick = all_faces[np.linspace(0, len(all_faces) - 1, min(128, len(all_faces)), dtype=int)]
    rows = np.searchsorted(receiver.parent, pick)[:, None] + np.arange(3)
    np.testing.assert_array_equal(receiver.parent[rows], np.tile(pick[:, None], (1, 3)))
    positions = donor.emitter if donor.spatial is None else donor.spatial.base.native_positions
    normals = donor.outward if donor.spatial is None else donor.native_normals
    light = (donor.intrinsic + donor_heat)[donor.visible_base] * donor.visible_area
    pairs = rows.size * len(light)
    if pairs > _MAX_PAIRS:
        raise ValueError("Diagnostic exceeds the receiver/emitter interaction cap")
    native = np.empty(rows.shape)
    for index in np.ndindex(rows.shape):
        row = rows[index]
        ray = receiver.receiver[row] - positions
        distance = np.linalg.norm(ray, axis=1)
        ray /= distance[:, None]
        incidence = np.maximum(0.0, -(ray @ receiver.normal[row]))
        emission = np.maximum(0.0, np.sum(ray * normals, axis=1))
        angular = 1 + donor.prefix_ld[0] * (emission - 1)
        native[index] = (
            receiver.absorption[receiver.parent[row]] * (light @ (incidence * emission * angular / distance**2)) / teff4
        )
    w = receiver.fraction[rows]
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=2e-14, rtol=0)
    predicted = mean[pick, None] / teff4 + slope[pick] @ _BARY.T
    direct_mean = np.sum(w * native, axis=1)
    shape_error = predicted - np.sum(w * predicted, axis=1)[:, None] - (native - direct_mean[:, None])

    def stats(diff):
        return dict(rms=float(np.sqrt(np.sum(w * diff**2) / len(pick))), maximum=float(np.max(np.abs(diff))))

    return dict(
        total_fully_sampled_faces=len(all_faces),
        tested_faces=len(pick),
        native_pairs=pairs,
        maximum_direct_mean_difference=float(np.max(np.abs(direct_mean - mean[pick] / teff4))),
        constant_face_error=stats(mean[pick, None] / teff4 - native),
        recovered_slope_error=stats(predicted - native),
        within_face_shape_error=stats(shape_error),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    props = json.loads(args.baseline.read_bytes())["samples"][0]["rows"][0]["physical"]
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        fit, _ = fixture()
        _, binary = prepare_binary(fit, props)
        results = {}
        for level in (2, 3):
            system, _ = intrinsic_state(binary, level)
            solver = _CaptureSweep("linear", device="reference", receiver_points=3, donor_level=2)
            solver.updates = []
            ts, _ = solver.evaluate(binary, system, level, 2)
            order = (
                ("primary", "secondary") if system.primary.t_eff <= system.secondary.t_eff else ("secondary", "primary")
            )
            assert len(solver.updates) == 4
            results[level] = {
                name: measure(binary, system, name, ts[name], u)
                for name, u in zip(order, solver.updates[-2:], strict=True)
            }
    args.output.write_text(json.dumps(dict(physical=props, levels=results), allow_nan=False, indent=2) + "\n")
