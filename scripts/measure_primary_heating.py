"""Measure primary heating on the actual visible disk, with both reflection iterations held fixed."""

import argparse
from copy import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from elisa import settings
from elisa.binary_system import dynamic
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule, SymmetricInteriorRule
from elisa.experimental.gpu_reflection import UnsupportedReflectionError
from elisa.experimental.primary_cubic import CurvedHeatingLightCurve
from scripts.receiver_phase_parts import context, pack_block
from scripts.validate_gpu_atmosphere import table_settings


class ProjectedHeating(CurvedHeatingLightCurve):
    """Evaluate two diagnostic primary spectra beside the reflected field on each clipped packet."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.primary_fields = {}
        self.disk_flux = {}

    def _prepare_field(self, model, system, observer, variant):
        key = variant.key
        engine = self.engines[key]
        data, reflection = {}, {}
        for name, backend in self.moment_backends.items():
            star = getattr(system, name)
            if star.atmosphere != "bb":
                raise UnsupportedReflectionError("The projected heating comparison requires blackbody spectra")
            data[name], reflection[name] = self._source_data(backend, model, star, engine.last_diagnostics[name])
        physical = self._optical_fields(system, observer, engine, data, reflection)
        diagnostics = engine.last_diagnostics["primary"]
        deposits = diagnostics["reflection_quadrature"].get("receiver_deposits")
        if deposits is None:
            raise UnsupportedReflectionError("Expected primary heat from the final reflection sweep")
        # Reconstruct the identical intrinsic field on the same angular face and gravity samples.
        star = copy(system.primary)
        star.temperatures = diagnostics["intrinsic_temperatures"]
        backend = self.moment_backends["primary"]
        tri, intrinsic, log_g, control = backend.emission_nodes(model, star, star.temperatures)
        np.testing.assert_array_equal(tri, data["primary"][0])
        np.testing.assert_allclose(log_g, data["primary"][2], rtol=0, atol=3e-13)
        mean = np.asarray(deposits[1])
        if mean.shape != (len(intrinsic),) or not np.isfinite(mean).all() or np.any(mean < 0):
            raise UnsupportedReflectionError("Expected a finite nonnegative deposit for each primary face")
        mean_temperature = star.t_eff * ((intrinsic / star.t_eff) ** 4 + mean[:, None]) ** 0.25
        self.primary_fields[key] = {}
        self.disk_flux[key] = []
        for mode, temperature in (("intrinsic", intrinsic), ("mean", mean_temperature)):
            data["primary"] = (tri, temperature, log_g, control)
            fields = self._optical_fields(system, observer, engine, data, reflection)
            self.primary_fields[key][mode] = fields["primary"]
        return physical

    def _integrate_packets(self, model, fields, maps, packets, variant):
        physical = super()._integrate_packets(model, fields, maps, packets, variant)
        packet = packets["primary"]
        parents = maps[variant.key]["primary"][packet["parents"]]
        area = packet["area"] * model.semi_major_axis**2
        diagnostic = {
            mode: field.flux(parents, packet.rays, packet["mu"], area)
            for mode, field in self.primary_fields[variant.key].items()
        }
        diagnostic["full"] = [physical[b][0] for b in self.passbands]
        self.disk_flux[variant.key].append(diagnostic)
        return physical


def evaluate(base, identity, backend, depth, part, degree):
    fit, props, _, phases, model = context(base)
    ix = np.array_split(np.arange(len(phases)), 2)[part]
    options = dict(slice_order=12, junction_depth=4)
    rule = (
        SymmetricInteriorRule(3, depth, interior_degree=degree, **options)
        if degree
        else CurvedEdgeRule(3, depth, **options)
    )
    solver = ProjectedHeating(list(fit.y_data), rule, device=backend, selective_shadow=True)
    batch = solver.batch([model], [phases[ix]], observer=fit.observer, source_revision=base["source_revision"])
    (result,) = batch.results
    block = pack_block(identity, backend, depth, part, ix, phases, props, model.phase_shift, result, batch)
    observer = copy(fit.observer)
    observer._system = model
    folded, origin = observer.phase_interval_reduce(observer.manage_time_series(phases=phases[ix]))
    support, inverse = dynamic.phase_crv_symmetry(model, folded)
    np.testing.assert_array_equal(support, [r["phase"] for r in block["phase_records"]])
    block["primary_flux"] = {}
    for key, records in solver.disk_flux.items():
        if len(records) != len(support):
            raise ValueError("Incomplete projected-disk comparison")
        block["primary_flux"][key] = {}
        for mode in ("intrinsic", "mean", "full"):
            values = np.asarray([r[mode] for r in records])[inverse][origin]
            if values.shape != (len(ix), len(solver.passbands)) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Invalid integrated primary flux")
            if mode == "full":
                for j, band in enumerate(solver.passbands):
                    np.testing.assert_array_equal(values[:, j], block["curves"][key]["components"][band][:, 0])
            else:
                block["primary_flux"][key][mode] = dict(zip(solver.passbands, values.T, strict=True))
    return block


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("--backend", choices=("reference", "cpu", "gpu"), required=True)
    p.add_argument("--depth", choices=(3, 4), type=int, required=True)
    p.add_argument("--part", choices=(0, 1), type=int, required=True)
    p.add_argument("--six-point", action="store_true")
    p.add_argument("--verified-bounded-gpu-window", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.backend == "gpu" and not args.verified_bounded_gpu_window:
        p.error("GPU execution requires a verified bounded unit")
    raw = args.baseline.read_bytes()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        start = perf_counter()
        out = evaluate(
            json.loads(raw),
            hashlib.sha256(raw).hexdigest(),
            args.backend,
            args.depth,
            args.part,
            4 if args.six_point else None,
        )
        out["inner_seconds"] = perf_counter() - start
    args.output.write_text(json.dumps(out, allow_nan=False, default=lambda a: a.tolist()) + "\n")


if __name__ == "__main__":
    main()
