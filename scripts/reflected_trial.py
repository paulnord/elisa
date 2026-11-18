"""Evaluate four field/donor combinations on common eclipse packets for one fit proposal."""

from time import perf_counter

import numpy as np

from elisa import const, units
from elisa.analytics.params import parameters
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.paired_reflection import PairedEmittingAreaLightCurve
from scripts.compare_reflected_posteriors import mapped_density
from scripts.validate_gpu_fit import make_problem, physical, prepare_fit
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_gpu_surface_error import asset_revision


def fixture():
    problem = make_problem(free_nuisance=True, shape_parameters=True)
    fit = prepare_fit(problem, phase_mode="observations", mesh_mode="paired")
    truth = {**problem.truth, "nuisance@ln_f": -2.0}
    start = parameters.vector_normalizer([truth[k] for k in fit.fitable], fit.fitable, fit.normalization)
    return fit, np.array(start)


def evaluate_depth(fit, vector, depth, backend, *, observer_type=PairedEmittingAreaLightCurve):
    start = perf_counter()
    if depth not in (3, 4) or vector.shape != (len(fit.fitable),) or not np.isfinite(vector).all():
        raise ValueError("Expected one finite physical proposal and eclipse depth 3 or 4")
    prior = float(fit.ln_prior(vector, fit.norm_init_vector, fit.sigmas))
    if not np.isfinite(prior):
        raise ValueError("Out-of-prior proposal at synthesis entry")
    props = physical(fit, vector)
    phases, binary = prepare_binary(fit, props)
    observer = observer_type(list(fit.y_data), CurvedEdgeRule(3, depth, slice_order=8), device=backend)
    revision = asset_revision()
    built = perf_counter()
    result = observer.batch([binary], [phases], observer=fit.observer, source_revision=revision).results[0]
    finished = perf_counter()
    profiles = []
    for cap in (2, 1):
        curves, metadata = {}, {}
        for variant in observer.variants:
            if variant.donor_level != cap:
                continue
            c = result.curves[variant.key]
            key = f"curvature_field{variant.level}"
            curves[key] = {k: getattr(c, k) for k in ("fluxes", "component_fluxes", "raw_fluxes")}
            metadata[key] = {k: getattr(c, k) for k in ("phases", "signature", "rule", "field_levels", "additional_light")}
            if variant.level == 3:
                prediction, density = mapped_density(fit, props, prior, c)
        row = dict(physical=props, phase_records=result.diagnostics["phase_records"],
                   quadrature_sha256=result.quadrature_sha256, finite_metadata=metadata, curves=curves,
                   prediction=prediction, posterior=density)
        profiles.append(dict(engine=observer.rule_prefix.replace("_", "-"), source_revision=revision,
                             backend=backend, batch=1, jd=False, iteration_start="guarded_physical",
                             physical_constants=dict(G_SI=const.G, solar_mass_kg=float(units.solMass.to(units.kg))),
                             levels=[2, 3], donor_level=cap, integration_level=3, depth=depth,
                             order=3, slice_order=8, receiver_points=3, irradiation=True, observations_irradiated=True,
                             reducers=fit.x_data_reducer, observations=fit.y_data, uncertainties=fit.y_err,
                             samples=[{"rows": [row]}]))
    return dict(profiles=profiles, timings=dict(construct=built-start, batch=finished-built, total=perf_counter()-start))
