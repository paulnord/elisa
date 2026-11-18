"""Integrate both primary source fields on the very same eclipse packets and thermal state."""

import argparse
from copy import copy
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.binary_system import dynamic
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule, SymmetricInteriorRule
from elisa.experimental.primary_cubic import PrimaryCubicLightCurve
from elisa.experimental.quadratic_source import QuadraticSourceBackend
from scripts.receiver_phase_parts import context
from scripts.validate_gpu_atmosphere import table_settings


class PairedPrimary(PrimaryCubicLightCurve):
    """One validation call. Both optical integrations receive the same maps, rays, cosines and areas."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comparison = copy(self)
        self.comparison.moment_backends = {
            **self.moment_backends,
            "primary": QuadraticSourceBackend("primary", self.law, device=self.device),
        }
        self.comparison._reconstruction = {}
        self.quadratic_fluxes = {}

    def _prepare_field(self, model, system, observer, variant):
        original = super()._prepare_field(model, system, observer, variant)
        other = super(PairedPrimary, self.comparison)._prepare_field(model, system, observer, variant)
        for name, field in other.items():
            self.comparison._reconstruction[variant.key, name] = field
        self.quadratic_fluxes[variant.key] = []
        return original

    def _integrate_packets(self, model, fields, maps, packets, variant):
        args = model, fields, maps, packets, variant
        alt = super(PairedPrimary, self.comparison)._integrate_packets(*args)
        values = super()._integrate_packets(*args)
        self.quadratic_fluxes[variant.key].append(alt)
        for band in self.passbands:
            np.testing.assert_allclose(alt[band][1], values[band][1], rtol=3e-13, atol=0)
        return values


def evaluate(base, backend, interior, depth, part):
    fit, props, _, phases, model = context(base)
    indices = np.arange(len(phases))
    if part is not None:
        indices = np.array_split(indices, 2)[part]
    phases = phases[indices]
    rule_type = SymmetricInteriorRule if interior == "six" else CurvedEdgeRule
    observer = PairedPrimary(
        list(fit.y_data), rule_type(3, depth, slice_order=12, junction_depth=4), device=backend, selective_shadow=True
    )
    result = observer(model, phases, observer=fit.observer, source_revision=base["source_revision"])
    local = copy(fit.observer)
    local._system = model
    folded, origin = local.phase_interval_reduce(local.manage_time_series(phases=phases))
    support, inverse = dynamic.phase_crv_symmetry(model, folded)
    phase_records = result.diagnostics["phase_records"]
    np.testing.assert_array_equal(support, [p["phase"] for p in phase_records])
    fluxes = {}
    for v in observer.variants:
        alt = observer.quadratic_fluxes[v.key]
        if len(alt) != len(support):
            raise ValueError("Incomplete shared-packet integration")
        c = result.curves[v.key]
        fluxes[v.key] = dict(
            quadratic={b: np.array([row[b] for row in alt])[inverse][origin] for b in fit.y_data},
            cubic=c.component_fluxes,
            phases=c.phases,
            signature=c.signature,
            additional_light=c.additional_light,
        )
    return dict(
        backend=backend,
        physical=props,
        indices=indices,
        interior=interior,
        depth=depth,
        shared_quadrature_sha256=result.quadrature_sha256,
        phase_records=phase_records,
        curves=fluxes,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("--backend", choices=("reference", "cpu", "gpu"), default="reference")
    p.add_argument("--depth", type=int, choices=(3, 4), required=True)
    p.add_argument("--part", type=int, choices=(0, 1))
    p.add_argument("--interior", choices=("six", "duffy"), required=True)
    p.add_argument("--verified-bounded-gpu-window", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.backend == "gpu" and not args.verified_bounded_gpu_window:
        p.error("GPU synthesis requires a registered bounded window")
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        out = evaluate(json.loads(args.baseline.read_bytes()), args.backend, args.interior, args.depth, args.part)
    args.output.write_text(json.dumps(out, allow_nan=False, default=lambda a: a.tolist()) + "\n")
