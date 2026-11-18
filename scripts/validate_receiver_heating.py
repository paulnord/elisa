"""Rebuild full reflected curves from local receiver samples and qualify the eight finite resolutions."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.experimental.receiver_heating import ReceiverHeatingLightCurve
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_quadratic_source import assess, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "assess"))
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--depth", choices=(3, 4), type=int)
    parser.add_argument("--backend", choices=("reference", "cpu", "gpu"), default="reference")
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
                source_type=ReceiverHeatingLightCurve,
                source_options=dict(selective_shadow=args.selective_shadow),
            )
        else:
            prefix = "shadow-" if args.selective_shadow else ""
            result = assess(inputs, engine=prefix + "sampled-receiver-heat")
    args.output.write_text(json.dumps(result, allow_nan=False, default=lambda a: a.tolist()) + "\n")


if __name__ == "__main__":
    main()
