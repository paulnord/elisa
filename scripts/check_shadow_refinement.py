"""Held-out component fluxes at eclipse phases with an extra level on shadow-crossed parents."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.shadow_refinement import DeeperShadowLightCurve
from scripts.reflected_trial import fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_gpu_surface_error import asset_revision


def evaluate(base, backend, depth):
    if base["source_revision"] != asset_revision() or base["engine"] != "emitting-area":
        raise ValueError("Use the matched original fixture")
    fit, _ = fixture()
    row = base["samples"][0]["rows"][0]
    phases, model = prepare_binary(fit, row["physical"])
    returned = np.array(row["finite_metadata"]["curvature_field3"]["phases"])
    ix = []
    for p in (0.024, 0.037, 0.027, 0.533, 0.267):
        hit = np.flatnonzero(np.isclose(returned, p, rtol=0, atol=1e-12))
        if len(hit) != 1:
            raise ValueError("Expected one synthesis row at each selected observation phase")
        ix.append(int(hit[0]))
    rule = CurvedEdgeRule(3, depth, slice_order=12, junction_depth=4)
    calc = DeeperShadowLightCurve(list(fit.y_data), rule, device=backend)
    result = calc.batch([model], [phases[ix]], source_revision=base["source_revision"], observer=fit.observer).results[
        0
    ]
    components = {}
    for key, c in result.curves.items():
        np.testing.assert_allclose(c.phases, returned[ix], atol=1e-14, rtol=0)
        components[key] = c.component_fluxes
    return dict(
        backend=backend,
        depth=depth,
        physical=row["physical"],
        indices=ix,
        phases=returned[ix],
        quadrature_sha256=result.quadrature_sha256,
        phase_records=result.diagnostics["phase_records"],
        components=components,
        rule=dict(slice_order=12, junction_depth=4, shadow_crossing_extra_depth=1),
        timings=result.timings,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("--backend", choices=("reference", "cpu", "gpu"), required=True)
    p.add_argument("--depth", type=int, choices=(3, 4), default=4)
    p.add_argument("--verified-bounded-gpu-window", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.backend == "gpu" and not a.verified_bounded_gpu_window:
        p.error("GPU synthesis requires a verified bounded unit")
    start = perf_counter()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        out = evaluate(json.loads(a.baseline.read_bytes()), a.backend, a.depth)
    out["evaluation_seconds"] = perf_counter() - start
    a.output.write_text(json.dumps(out, allow_nan=False, default=lambda v: v.tolist()) + "\n")
