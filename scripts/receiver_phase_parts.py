"""Evaluate reflected phase blocks and reconstruct the original full-observation likelihood."""

import argparse
from copy import copy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.binary_system import dynamic
from elisa.analytics.params import parameters
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from elisa.experimental.gpu_curved import CurvedCurveResult
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule, SymmetricInteriorRule
from scripts.validate_receiver_heating import source_class
from scripts.compare_reflected_posteriors import mapped_density
from scripts.reflected_trial import fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import physical
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_gpu_surface_error import asset_revision


def context(base):
    fit, _ = fixture()
    props = base["samples"][0]["rows"][0]["physical"]
    vector = np.asarray(parameters.vector_normalizer([props[k] for k in fit.fitable], fit.fitable, fit.normalization))
    prior = float(fit.ln_prior(vector, fit.norm_init_vector, fit.sigmas))
    if not np.isfinite(prior) or physical(fit, vector) != props or base["source_revision"] != asset_revision():
        raise ValueError("Expected the matching in-prior physical proposal and table revision")
    for band in fit.y_data:
        for actual, expected in (
            (base["observations"], fit.y_data),
            (base["uncertainties"], fit.y_err),
            (base["reducers"], fit.x_data_reducer),
        ):
            np.testing.assert_array_equal(actual[band], expected[band])
    phases, model = prepare_binary(fit, props)
    return fit, props, prior, phases, model


def evaluate_block(base, identity, backend, depth, part, *, symmetric_interior=None, primary_cubic=False, curved_heating=False, horizon_mean=False, node_reemission=False):
    fit, props, _, phases, model = context(base)
    indices = np.array_split(np.arange(len(phases)), 2)[part]
    rule_type = SymmetricInteriorRule if symmetric_interior else CurvedEdgeRule
    options = {"interior_degree": symmetric_interior} if symmetric_interior else {}
    rule = rule_type(3, depth, slice_order=12, junction_depth=4, **options)
    solver_type = source_class(primary_cubic, curved_heating, horizon_mean, node_reemission)
    solver = solver_type(list(fit.y_data), rule, device=backend, selective_shadow=True)
    batch = solver.batch([model], [phases[indices]], observer=fit.observer, source_revision=base["source_revision"])
    result = batch.results[0]
    return pack_block(identity, backend, depth, part, indices, phases, props, model.phase_shift, result, batch)


def pack_block(identity, backend, depth, part, indices, phases, props, phase_shift, result, batch):
    arrays = {}
    for key, c in result.curves.items():
        np.testing.assert_array_equal(c.phases, (phases[indices] - phase_shift) + phase_shift)
        arrays[key] = dict(
            components=c.component_fluxes,
            rule=c.rule,
            signature=c.signature,
            field_levels=c.field_levels,
            additional_light=c.additional_light,
        )
    controls = {
        k: {n: v["reconstructed_field"] for n, v in stars.items()}
        for k, stars in result.diagnostics["ancestry_controls"].items()
    }
    return dict(
        baseline_sha256=identity,
        physical=props,
        backend=backend,
        depth=depth,
        part=part,
        indices=indices,
        input_phases=phases[indices],
        curves=arrays,
        phase_records=result.diagnostics["phase_records"],
        quadrature_sha256=result.quadrature_sha256,
        stages=result.timings,
        batch=batch.timings,
        fields=controls,
    )


def compare_field_controls(a, b):
    """Require identical topology and verify floating reductions at the field solver's FP64 limits."""
    if a.keys() != b.keys():
        raise ValueError("Different stellar fields between phase blocks")
    maximum_heat_difference = 0.0
    for variant, stars in a.items():
        if stars.keys() != b[variant].keys():
            raise ValueError("Different stars in phase blocks")
        for name, first in stars.items():
            second = b[variant][name]
            if first.keys() != second.keys():
                raise ValueError("Different field diagnostics in phase blocks")
            for key, x in first.items():
                y = second[key]
                if key == "deposited_T4_area_mean_over_Teff4":
                    if min(x, y) < 0 or not np.isfinite([x, y]).all() or not np.isclose(x, y, rtol=3e-13, atol=0):
                        raise ValueError("Inconsistent deposited heat between phase blocks")
                    maximum_heat_difference = max(maximum_heat_difference, abs(x - y))
                elif key in ("area_relative_error", "cell_T4_relative_error", "quadratic_T4_relative_error"):
                    if not all(0 <= err <= 3e-13 for err in (x, y)):
                        raise ValueError("Field area or thermal closure exceeds the synthesis limit")
                elif x != y:
                    raise ValueError(f"Changed field control {key} between phase blocks")
    return maximum_heat_difference


def join(base, identity, parts, *, symmetric_interior=None, primary_cubic=False, curved_heating=False, horizon_mean=False, node_reemission=False):
    fit, props, prior, phases, model = context(base)
    ordered = sorted(parts, key=lambda p: p["part"])
    if [p["part"] for p in ordered] != [0, 1]:
        raise ValueError("Both distinct phase blocks are required")
    a, b = ordered
    for p, ix in zip(ordered, np.array_split(np.arange(len(phases)), 2), strict=True):
        if (
            p["baseline_sha256"] != identity
            or p["physical"] != props
            or p["backend"] != a["backend"]
            or p["depth"] != a["depth"]
        ):
            raise ValueError("Phase blocks differ in physical proposal, baseline, backend or depth")
        np.testing.assert_array_equal(p["indices"], ix)
        np.testing.assert_array_equal(p["input_phases"], phases[ix])
    reduction_difference = compare_field_controls(a["fields"], b["fields"])
    if a["curves"].keys() != b["curves"].keys():
        raise ValueError("Inconsistent fresh stellar fields between phase blocks")
    # Symmetry-reduced orbital phases may appear in both blocks. Keep a single topology record per phase.
    records = {}
    for p in ordered:
        for r in p["phase_records"]:
            if records.setdefault(r["phase"], r) != r:
                raise ValueError("Repeating an orbital phase changed its quadrature")
    digest = hashlib.sha256(json.dumps([p["quadrature_sha256"] for p in ordered]).encode()).hexdigest()
    observer = copy(fit.observer)
    observer._system = model
    resolved = observer.manage_time_series(phases=phases)
    folded, origin = observer.phase_interval_reduce(resolved)
    support, inverse = dynamic.phase_crv_symmetry(model, folded)
    np.testing.assert_array_equal(support, sorted(records))
    groups = inverse[origin]
    repeated = [np.flatnonzero(groups == i) for i, n in enumerate(np.bincount(groups)) if n > 1]
    profiles = []
    returned = resolved + model.phase_shift
    engine = ("symmetric-" if symmetric_interior else "") + "shadow-"
    engine += source_class(primary_cubic, curved_heating, horizon_mean, node_reemission).rule_prefix.replace("_", "-")
    rule_prefix = engine.replace("-", "_")
    for cap in (2, 1):
        curves, metadata = {}, {}
        for level in (2, 3):
            key = f"curvature_field{level}"
            rows = [p["curves"][f"{key}_donor{cap}"] for p in ordered]
            meta = {k: v for k, v in rows[0].items() if k != "components"}
            if (
                meta["rule"].get("interior_degree") != symmetric_interior
                or meta["rule"].get(f"{rule_prefix}_receiver3_donor_cap{cap}") != 1
                or any(rows[1][k] != v for k, v in meta.items())
                or meta["field_levels"] != level
                or meta["additional_light"] != model.additional_light
            ):
                raise ValueError("Inconsistent phase-block model or numerical rule")
            if any(r["components"].keys() != fit.y_data.keys() for r in rows):
                raise ValueError("Incomplete phase-block passbands")
            components = {}
            for band in fit.y_data:
                vals = [np.asarray(r["components"][band]) for r in rows]
                if any(
                    v.shape != (len(p["indices"]), 2) or not np.isfinite(v).all() or np.any(v < 0)
                    for v, p in zip(vals, ordered, strict=True)
                ):
                    raise ValueError("Invalid physical component fluxes in phase block")
                components[band] = np.concatenate(vals)
                for ix in repeated:
                    expected = np.broadcast_to(components[band][ix[0]], (len(ix), 2))
                    np.testing.assert_allclose(components[band][ix], expected, rtol=3e-13, atol=0)
            metadata[key] = dict(phases=returned, **meta)
            curves[key] = dict(
                component_fluxes=components,
                fluxes=HybridBinaryLightCurve._observed_fluxes(components, model, normalize=True),
                raw_fluxes=HybridBinaryLightCurve._observed_fluxes(components, model, normalize=False),
            )
        c = CurvedCurveResult(**metadata["curvature_field3"], **curves["curvature_field3"], timings={}, diagnostics={})
        pred, density = mapped_density(fit, props, prior, c)
        row = dict(
            physical=props,
            phase_records=[records[k] for k in sorted(records)],
            quadrature_sha256=digest,
            finite_metadata=metadata,
            curves=curves,
            prediction=pred,
            posterior=density,
        )
        profiles.append(
            dict(
                engine=engine,
                source_revision=base["source_revision"],
                backend=a["backend"],
                batch=1,
                jd=False,
                iteration_start="guarded_physical",
                physical_constants=base["physical_constants"],
                levels=[2, 3],
                donor_level=cap,
                integration_level=3,
                depth=a["depth"],
                order=3,
                slice_order=12,
                junction_depth=4,
                receiver_points=3,
                irradiation=True,
                observations_irradiated=True,
                reducers=fit.x_data_reducer,
                observations=fit.y_data,
                uncertainties=fit.y_err,
                samples=[dict(rows=[row])],
            )
        )
    return dict(
        profiles=profiles,
        measured=dict(
            fields=a["fields"],
            second_block_fields=b["fields"],
            max_repeat_mean_T4_difference=reduction_difference,
            repeated_phase_groups=len(repeated),
            phase_block_stages=[p["stages"] for p in ordered],
        ),
        timings=dict(synthesis_process_inner_seconds=sum(p["inner_seconds"] for p in ordered)),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("evaluate", "join"))
    ap.add_argument("baseline", type=Path)
    ap.add_argument("parts", type=Path, nargs="*")
    ap.add_argument("--backend", choices=("reference", "cpu", "gpu"), default="reference")
    ap.add_argument("--depth", choices=(3, 4), type=int)
    ap.add_argument("--part", choices=(0, 1), type=int)
    ap.add_argument(
        "--symmetric-interior", nargs="?", const=4, type=int, choices=(4, 5), metavar="DEGREE",
        help="Use six (degree 4) or seven (degree 5) positive interior nodes per triangle. Default is Duffy quadrature.",
    )
    ap.add_argument("--node-reemission", action="store_true", help="Re-emit secondary heating at each curved donor node.")
    ap.add_argument("--horizon-mean", action="store_true", help="Refine the primary receiver mean near the irradiation horizon.")
    ap.add_argument("--curved-heating", action="store_true")
    ap.add_argument("--primary-cubic", action="store_true")
    ap.add_argument("--verified-bounded-gpu-window", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    raw = args.baseline.read_bytes()
    base = json.loads(raw)
    identity = hashlib.sha256(raw).hexdigest()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        start = perf_counter()
        if args.command == "evaluate":
            if (
                args.depth is None
                or args.part is None
                or (args.backend == "gpu" and not args.verified_bounded_gpu_window)
            ):
                ap.error("Select depth and phase block. GPU synthesis requires a verified bounded unit.")
            out = evaluate_block(base, identity, args.backend, args.depth, args.part, symmetric_interior=args.symmetric_interior, primary_cubic=args.primary_cubic, curved_heating=args.curved_heating, horizon_mean=args.horizon_mean, node_reemission=args.node_reemission)
        else:
            out = join(base, identity, [json.loads(p.read_bytes()) for p in args.parts], symmetric_interior=args.symmetric_interior, primary_cubic=args.primary_cubic, curved_heating=args.curved_heating, horizon_mean=args.horizon_mean, node_reemission=args.node_reemission)
        out["inner_seconds"] = perf_counter() - start
    args.output.write_text(json.dumps(out, allow_nan=False, default=lambda a: a.tolist()) + "\n")
