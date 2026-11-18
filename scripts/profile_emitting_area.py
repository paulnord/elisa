"""Measure the complete reflection stage on two fresh binary systems."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.experimental.emitting_area import EmittingAreaReflection
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem, prepare_fit
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_gpu_surface_error import asset_revision
from scripts.validate_reflection_area import NAMES, intrinsic_state


def measure(op, binary, system):
    times = {}

    def measured(name, function):
        def call(*args):
            start = perf_counter()
            value = function(*args)
            if hasattr(value, "block_until_ready"):
                value.block_until_ready()
            times.setdefault(name, []).append(perf_counter() - start)
            return value

        return call

    update = "_reference_update" if op.mode == "reference" else "_update"
    with (
        patch.object(op, "_surface", measured("surface", op._surface)),
        patch.object(op, "_prepare_pair", measured("group_and_horizon_layout", op._prepare_pair)),
        patch.object(op, update, measured("irradiation_update", getattr(op, update))),
    ):
        start = perf_counter()
        temperatures, stats = op.evaluate(binary, system, 3, 2)
        total = perf_counter() - start
    return dict(
        timings=times,
        reflection_seconds=total,
        other_seconds=total - sum(sum(parts) for parts in times.values()),
        mean_deposited={k: v["deposited_T4_area_mean_over_Teff4"] for k, v in stats.items()},
        temperatures={k: temperatures[k][: getattr(system, k).base_symmetry_faces_number].tolist() for k in NAMES},
        pairs=stats["primary"]["pairs_per_sweep_both_stars"],
    )


def run(args):
    if args.backend == "gpu" and not args.verified_bounded_gpu_window:
        raise ValueError("GPU profiling requires a verified bounded unit")
    fit = prepare_fit(
        make_problem(free_nuisance=True, shape_parameters=True), phase_mode="observations", mesh_mode="paired"
    )
    start = perf_counter()
    op = EmittingAreaReflection(settings.LIMB_DARKENING_LAW, device=args.backend, donor_level=2)
    constructor = perf_counter() - start
    rows = []
    for path in args.profiles:
        p = json.loads(path.read_text())
        if (
            p["source_revision"] != asset_revision()
            or p["levels"] != [2, 3]
            or p["donor_level"] != 2
            or p["engine"] != "emitting-area"
            or p["receiver_points"] != 3
            or not p["irradiation"]
        ):
            raise ValueError("Use matched field-3 and donor-cap-2 irradiation profiles")
        ref = p["samples"][0]["rows"][0]
        _, binary = prepare_binary(fit, ref["physical"])
        system, _ = intrinsic_state(binary, 3)
        row = measure(op, binary, system)
        row["physical"] = ref["physical"]
        for name in NAMES:
            expected = ref["field_controls"]["curvature_field3"][name]["reconstructed_field"]
            np.testing.assert_allclose(
                row["mean_deposited"][name], expected["deposited_T4_area_mean_over_Teff4"], rtol=3e-13
            )
            if row["pairs"] != expected["pairs_per_sweep_both_stars"]:
                raise ValueError("Irradiation resource contract changed")
        rows.append(row)
    args.output.write_text(
        json.dumps(
            dict(
                backend=args.backend,
                constructor_seconds=constructor,
                rows=rows,
                contract="Start with each fresh intrinsic surface. Time the normal complete reflection entry point including host grouping, all four updates, transfers and temperature return. Each device update synchronizes. Field preparation before reflection and subsequent spectra and observer integration are outside this stage timer.",
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profiles", nargs="+", type=Path)
    p.add_argument("--backend", choices=("reference", "gpu"), required=True)
    p.add_argument("--verified-bounded-gpu-window", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        run(args)
