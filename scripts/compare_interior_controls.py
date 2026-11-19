"""Compare a revised interior integration with the same physical draw on the original panels."""

import numpy as np

from elisa.experimental.gpu_curved_edges import InferenceBudget
from scripts import sample_reflected_binary as sampler
from scripts.receiver_phase_parts import compare_field_controls


def compare_controls(original, compact, fit, *, nominal):
    if original["stops"] or compact["stops"] or original["physics"] != compact["physics"]:
        raise ValueError("Expected two complete physical curves on the same binary")
    compare_field_controls(original["fields"], compact["fields"])
    phase_nodes = []
    for first, second in zip(original["records"], compact["records"], strict=True):
        if first["phase"] != second["phase"] or first["components"].keys() != second["components"].keys():
            raise ValueError("The projected phase or stellar components differ")
        for star, before in first["components"].items():
            after = second["components"][star]
            if before.keys() != after.keys():
                raise ValueError("Changed eclipse diagnostics")
            for name, value in before.items():
                if name in ("nodes", "selected_nodes", "largest_partition_nodes"):
                    if after[name] > value:
                        raise ArithmeticError("Interior merge increased the quadrature size")
                elif value != after[name]:
                    raise ArithmeticError(f"The eclipse boundary changed at {name}")
            phase_nodes.append(dict(phase=first["phase"], star=star, before=before["nodes"], after=after["nodes"]))
    if nominal and any(row["before"] != row["after"] for row in phase_nodes):
        raise ArithmeticError("The nominal six-point panels changed")
    rows = []
    for key, (meta, a, pred, density) in original["curves"].items():
        other, b, new_pred, new_density = compact["curves"][key]
        new_rule = dict(b.rule)
        if new_rule.pop("duffy_sibling_levels", None) != 1 or new_rule != a.rule:
            raise ValueError("A numerical setting besides the Duffy merge changed")
        if sampler.digest(meta) != sampler.digest(dict(other, rule=new_rule)):
            raise ValueError("Different physical metadata in the interior comparison")
        # Both curves have their own quadrature metadata. The finite-pair test uses
        # the physical arrays, original observation mapping and nominal noise.
        report = InferenceBudget().compare(
            a, b, fit.y_err, observed=fit.y_data, reducers=fit.x_data_reducer, mean_normalize=True
        )["nominal_report"]
        error = max(float(np.max(abs(new_pred[k] - x))) for k, x in pred.items())
        if nominal:
            for k in a.component_fluxes:
                np.testing.assert_array_equal(a.component_fluxes[k], b.component_fluxes[k])
        rows.append(
            dict(
                donor=key[0],
                field=key[1],
                max_prediction_error=error,
                fitted_density_difference=new_density - density,
                distance_sigma=report["actual"]["distance_sigma"],
                nominal_likelihood_bound=report["finite_log_likelihood_absolute_bound"],
            )
        )
    return dict(phase_nodes=phase_nodes, corners=rows)


def original_duffy_phases(template, field_file, omitted_phases):
    """Physical fluxes on the original depth-four panels at the selected supported phases."""
    from copy import copy

    from elisa.binary_system import dynamic
    from scripts.benchmark_stationary_eclipse import inputs
    from scripts.receiver_phase_parts import context
    from scripts.reemission_walker_fields import WalkerObserver, binding, load_fields, rule_for

    state, fit, _, key, _, base = inputs(template, True)
    _, _, _, phases, model = context(base)
    identity = binding(model, fit.observer, state, key)
    meta, arrays = load_fields(field_file, identity)
    solver = WalkerObserver(list(fit.y_data), rule_for("duffy", 4), device="reference", selective_shadow=True)
    solver.attach(model, meta, arrays, identity)
    observer = copy(fit.observer)
    observer._system = model
    folded, origin = observer.phase_interval_reduce(observer.manage_time_series(phases=phases))
    support, reverse = dynamic.phase_crv_symmetry(model, folded)
    if not set(omitted_phases) <= set(support):
        raise ValueError("Omitted phase is absent from the physical fixture")
    ix = np.flatnonzero(~np.isin(support[reverse[origin]], omitted_phases))
    result = solver.batch(
        [model], [phases[ix]], observer=fit.observer, source_revision=base["source_revision"]
    ).results[0]
    return dict(
        indices=ix,
        omitted_folded_phases=omitted_phases,
        records=result.diagnostics["phase_records"],
        components={key: c.component_fluxes for key, c in result.curves.items()},
        timings=result.timings,
    )
