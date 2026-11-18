"""Replay retained depth results across interrupted, corrupted and repeated checkpoint operations."""

import argparse
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from scripts.reflected_trial import fixture
from scripts.sample_reflected_binary import context, initialize, part_path, read_record, take_step
from scripts.validate_gpu_atmosphere import table_settings


def verify(path, initial):
    final = read_record(path)
    with TemporaryDirectory() as folder:
        checkpoint = Path(folder) / "state.json"
        args = SimpleNamespace(
            checkpoint=checkpoint,
            backend=final["backend"],
            seed=final["seed"],
            scale=final["scales"][0],
            initial=initial,
            command="advance",
        )
        fit, truth = fixture()
        initialize(args, fit, truth, context(fit))
        for event in final["events"]:
            args.step = event["step"]
            if event["report"] is not None:
                old = checkpoint.read_bytes()
                depth3, depth4 = [part_path(checkpoint.parent, args.step, d) for d in (3, 4)]
                shutil.copyfile(path.parent / depth3.name, depth3)
                try:
                    take_step(args, fit, read_record(checkpoint))
                except FileNotFoundError:
                    if checkpoint.read_bytes() != old:
                        raise RuntimeError("Partial depth result changed the chain")
                else:
                    raise RuntimeError("Advanced with a missing depth")
                source = path.parent / depth4.name
                depth4.write_bytes(source.read_bytes().replace(b'"sha256":"', b'"sha256":"bad-', 1))
                try:
                    take_step(args, fit, read_record(checkpoint))
                except ValueError as exc:
                    if "checksum" not in str(exc) or checkpoint.read_bytes() != old:
                        raise
                else:
                    raise RuntimeError("Advanced with a corrupt depth")
                shutil.copyfile(source, depth4)
            take_step(args, fit, read_record(checkpoint))
            saved = checkpoint.read_bytes()
            try:
                take_step(args, fit, read_record(checkpoint))
            except ValueError as exc:
                if "generation" not in str(exc) or checkpoint.read_bytes() != saved:
                    raise
            else:
                raise RuntimeError("Repeated an already committed transition")
        if checkpoint.read_bytes() != path.read_bytes():
            raise RuntimeError("Restart replay changed the final checkpoint or RNG state")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--initial", type=Path, nargs="+", required=True)
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        verify(args.checkpoint, args.initial)
    print("Every replayed checkpoint operation passed.", flush=True)
