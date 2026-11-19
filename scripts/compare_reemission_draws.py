"""Audit a paired transition against the original observations and all sixteen finite resolutions."""

import argparse
import json
from math import exp, fsum, log, pi
from pathlib import Path

import numpy as np

from elisa import settings
from scripts import sample_reflected_binary as sampler
from scripts import sample_reemission_chain as chain
from scripts.compare_reflection_resolution import DEPTH, DONOR, FIELD
from scripts.receiver_phase_parts import compare_field_controls, join
from scripts.reflected_trial import fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_selective import prepare_binary


def gaussian_from_components(row, band_meta, fit, prior):
    """Map component flux to repeated observations and evaluate the diagonal Gaussian independently."""
    likelihood = []
    predictions = {}
    for band, components in row.items():
        flux = [fsum(pair) for pair in components]
        l3 = band_meta["additional_light"]
        extra = fsum(flux) / len(flux) * l3 / (1 - l3)
        sampled = [flux[i] + extra for i in fit.x_data_reducer[band]]
        mean = fsum(sampled) / len(sampled)
        pred = np.asarray(sampled) / mean
        variance = np.square(fit.y_err[band]) * (1 + exp(2 * band_meta["ln_f"]))
        likelihood.extend(
            -0.5 * ((obs - m) ** 2 / v + log(2 * pi * v))
            for obs, m, v in zip(fit.y_data[band], pred, variance, strict=True)
        )
        predictions[band] = pred
    return prior + fsum(likelihood), predictions


def reconstruct(start, checkpoint, step, fit):
    state = sampler.read_record(start)
    chain.check_target(state, fit)
    if state["step"] != step - 1:
        raise ValueError("Expected the preceding chain generation")
    vector, log_u, rng, key, prior, base = chain.trial_inputs(state, fit)
    if not np.isfinite(prior):
        raise ValueError("This physical-curve comparison needs an in-prior draw")
    assessment = sampler.read_record(checkpoint.parent / f"assessment-{step:06d}.json")
    if assessment["key"] != key or assessment["outside_prior"]:
        raise ValueError("Assessment is for a different trial")
    scores = assessment["evidence"]["corner_log_posteriors"]
    curves = {}
    hashes = []
    max_error = 0.0
    for rule, degree in (("six", 4), ("duffy", None)):
        for depth in (3, 4):
            blocks = [
                chain.read_block(
                    chain.block_path(checkpoint, step, rule, depth, part), chain.block_binding(key, rule, depth, part)
                )
                for part in (0, 1)
            ]
            hashes.extend(sampler.digest(v) for v in blocks)
            full = join(base, key, blocks, symmetric_interior=degree, node_reemission=True)
            for profile in full["profiles"]:
                if profile["backend"] != state["backend"] or profile["depth"] != depth:
                    raise ValueError("Backend or eclipse depth changed")
                row = profile["samples"][0]["rows"][0]
                for level in (2, 3):
                    name = f"curvature_field{level}"
                    meta = row["finite_metadata"][name]
                    obs = row["curves"][name]
                    likelihood, pred = gaussian_from_components(
                        obs["component_fluxes"],
                        dict(additional_light=meta["additional_light"], ln_f=row["physical"]["nuisance@ln_f"]),
                        fit,
                        prior,
                    )
                    cap = profile["donor_level"]
                    corner = (FIELD if level == 2 else 0) | (DEPTH if depth == 3 else 0) | (DONOR if cap == 1 else 0)
                    err = abs(likelihood - scores[rule][str(corner)])
                    if err > 1e-6:
                        raise ArithmeticError("Independent component-flux density differs from the fitted posterior")
                    max_error = max(max_error, err)
                    if level == 3:
                        for b, v in pred.items():
                            np.testing.assert_allclose(v, row["prediction"][b], atol=2e-12, rtol=3e-13)
                    curves[rule, depth, cap, level] = (
                        meta,
                        obs,
                        pred,
                        row["phase_records"],
                        full["measured"]["fields"],
                    )
    if len(curves) != 16 or assessment["evidence"]["input_hashes"] != hashes:
        raise ValueError("Incomplete or changed resolution grid")
    props = base["samples"][0]["rows"][0]["physical"]
    _, model = prepare_binary(fit, props)
    return (
        dict(
            proposed=vector,
            log_uniform=log_u,
            rng_after=rng,
            prior=prior,
            physical=props,
            volume_equivalent_radii_over_separation={
                name: getattr(model, name).equivalent_radius for name in ("primary", "secondary")
            },
            max_independent_density_difference=max_error,
        ),
        curves,
        assessment["evidence"],
    )


def audit(reference_start, reference, gpu_start, gpu, step):
    starts = [sampler.read_record(path) for path in (reference_start, gpu_start)]
    cpu_state, gpu_state = starts
    if [s["backend"] for s in starts] != ["reference", "gpu"]:
        raise ValueError("Expected distinct reference and GPU checkpoints in that order")
    for name in ("step", "seed", "target", "scales", "position", "rng"):
        if sampler.digest(cpu_state[name]) != sampler.digest(gpu_state[name]):
            raise ValueError(f"Starting checkpoints differ in {name}")
    np.testing.assert_allclose(cpu_state["density"], gpu_state["density"], atol=1e-6, rtol=0)
    fit, _ = fixture()
    a, cp, c = reconstruct(reference_start, reference, step, fit)
    b, gp, g = reconstruct(gpu_start, gpu, step, fit)
    for field in (
        "proposed",
        "log_uniform",
        "rng_after",
        "prior",
        "physical",
        "volume_equivalent_radii_over_separation",
    ):
        if sampler.digest(a[field]) != sampler.digest(b[field]):
            raise ValueError(f"Draws differ in {field}")
    mx = dict(normalized=0.0, component_relative=0.0, raw_relative=0.0, density=0.0)
    for key in cp:
        cm, co, pc, cr, cf = cp[key]
        gm, go, pg, gr, gf = gp[key]
        if sampler.digest(cm) != sampler.digest(gm) or cr != gr:
            raise ValueError("Eclipse quadrature, curve identity or phase map differs")
        compare_field_controls(cf, gf)
        for attr, metric in (
            ("fluxes", "normalized"),
            ("component_fluxes", "component_relative"),
            ("raw_fluxes", "raw_relative"),
        ):
            for band, values in co[attr].items():
                x, y = np.asarray(values), np.asarray(go[attr][band])
                np.testing.assert_allclose(x, y, atol=2e-12 if attr == "fluxes" else 0, rtol=3e-13)
                delta = abs(x - y) if attr == "fluxes" else abs((x - y)[x != 0] / x[x != 0])
                mx[metric] = max(mx[metric], float(np.max(delta, initial=0)))
        for band, pred in pc.items():
            np.testing.assert_allclose(pred, pg[band], atol=2e-12, rtol=3e-13)
            mx["normalized"] = max(mx["normalized"], float(abs(pred - pg[band]).max()))
    for rule, scores in c["corner_log_posteriors"].items():
        for i, val in scores.items():
            mx["density"] = max(mx["density"], abs(val - g["corner_log_posteriors"][rule][i]))
    if mx["density"] > 1e-6 or c["report"]["pass_finite_check"] != g["report"]["pass_finite_check"]:
        raise ValueError("Backends disagree on the physical posterior or qualification")
    for field in ("distance_sigma", "log_likelihood_bound", "max_measured_nominal_log_likelihood_change"):
        np.testing.assert_allclose(c["report"][field], g["report"][field], atol=1e-6, rtol=0)
    return dict(
        draw={k: v for k, v in a.items() if k not in ("rng_after", "max_independent_density_difference")},
        max_independent_density_difference=max(x["max_independent_density_difference"] for x in (a, b)),
        parity=mx,
        reference=c,
        gpu=g,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("reference-start", "reference", "gpu-start", "gpu", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        out = audit(args.reference_start, args.reference, args.gpu_start, args.gpu, args.step)
    args.output.write_text(json.dumps(out, default=lambda x: x.tolist(), allow_nan=False) + "\n")
