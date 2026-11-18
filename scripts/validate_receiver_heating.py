"""Rebuild full reflected curves from local receiver samples and qualify the eight finite resolutions."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.experimental.gpu_curved_edges import SymmetricInteriorRule
from elisa.experimental.primary_cubic import CurvedHeatingLightCurve, PrimaryCubicLightCurve
from elisa.experimental.receiver_heating import ReceiverHeatingLightCurve
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_quadratic_source import assess, evaluate


def source_class(primary_cubic, curved_heating):
    if curved_heating:
        return CurvedHeatingLightCurve
    return PrimaryCubicLightCurve if primary_cubic else ReceiverHeatingLightCurve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "assess"))
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--depth", choices=(3, 4), type=int)
    parser.add_argument("--backend", choices=("reference", "cpu", "gpu"), default="reference")
    parser.add_argument(
        "--symmetric-interior", nargs="?", const=4, type=int, choices=(4, 5), metavar="DEGREE",
        help="Use six (degree 4) or seven (degree 5) positive interior nodes per triangle. Default is Duffy quadrature.",
    )
    parser.add_argument("--curved-heating", action="store_true", help="Use quadratic primary heating with cubic emission.")
    parser.add_argument("--primary-cubic", action="store_true")
    parser.add_argument("--selective-shadow", action="store_true")
    parser.add_argument("--verified-bounded-gpu-window", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.boundary = "refined"
    inputs = [json.loads(p.read_bytes()) for p in args.inputs]
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        if args.command == "evaluate":
            if len(inputs) != 1 or args.depth is None:
                parser.error("evaluate needs one physical baseline and an eclipse depth")
            result = evaluate(
                args,
                inputs[0],
                source_type=source_class(args.primary_cubic, args.curved_heating),
                rule_type=partial(SymmetricInteriorRule, interior_degree=args.symmetric_interior) if args.symmetric_interior else None,
                source_options=dict(selective_shadow=args.selective_shadow),
            )
        else:
            prefix = "shadow-" if args.selective_shadow else ""
            if args.symmetric_interior:
                prefix = "symmetric-" + prefix
            engine = prefix + source_class(args.primary_cubic, args.curved_heating).rule_prefix.replace("_", "-")
            result = assess(inputs, engine=engine, interior_degree=args.symmetric_interior)
    args.output.write_text(json.dumps(result, allow_nan=False, default=lambda a: a.tolist()) + "\n")


if __name__ == "__main__":
    main()
