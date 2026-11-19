"""Continue independently checked physical walkers from their exact selected states."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from elisa import settings
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from scripts import sample_reemission_chain as chain
from scripts import sample_reflected_binary as sampler
from scripts.benchmark_reemission_walkers import completed
from scripts.reflected_trial import fixture
from scripts.validate_gpu_atmosphere import table_settings


def snapshot(previous, count=None):
    """Verify the selected states and the phase blocks used by the preceding likelihood."""
    starts = sorted(previous.glob("walker-*/start.json"))
    if not starts or (count is not None and not 1 <= count <= len(starts)):
        raise ValueError("Expected a nonempty cohort from a preceding physical walker run")
    states = []
    for path in starts[:count]:
        start = sampler.read_record(path)
        row = completed(path, start)
        step, key = row["sampled_step"], row["trial"]
        assessment = sampler.read_record(path.parent / f"assessment-{step:06d}.json")
        if row["status"] == "accuracy_stop":
            raise ResolutionNotQualifiedError("Cannot advance a cohort past a resolution stop", assessment["evidence"]["report"])
        if not assessment["outside_prior"]:
            hashes = []
            for rule in ("six", "duffy"):
                for depth in (3, 4):
                    for part in (0, 1):
                        binding = chain.block_binding(key, rule, depth, part)
                        block = chain.read_block(chain.block_path(path, step, rule, depth, part), binding)
                        hashes.append(sampler.digest(block))
            if hashes != assessment["evidence"]["input_hashes"]:
                raise ValueError("Prior transition's physical curves differ from its resolution assessment")
        states.append(sampler.read_record(path.with_name("state.json")))
    validate(states)
    return states


def validate(states):
    if not states or len({s["seed"] for s in states}) != len(states):
        raise ValueError("Expected nonempty, independently seeded chains")
    targets = {(s["backend"], sampler.digest(s["target"])) for s in states}
    if len(targets) != 1 or states[0]["backend"] not in ("reference", "gpu"):
        raise ValueError("Cannot combine distinct targets or backends")
    fit, _ = fixture()
    chain.check_target(states[0], fit)
    for state in states:
        if [e["step"] for e in state["events"]] != list(range(1, state["step"] + 1)):
            raise ValueError("Missing generation in physical chain")
        if state["events"]:
            last = state["events"][-1]
            if last["selected"] != state["position"] or last["selected_density"] != state["density"]:
                raise ValueError("Selected physical state differs from its last decision")


def initialize(states, dest):
    """Expose the complete next cohort atomically after validating the saved target."""
    validate(states)
    if dest.exists():
        raise FileExistsError(dest)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        batch = Path(tmp) / "generation"
        batch.mkdir()
        for i, state in enumerate(states):
            walker = batch / f"walker-{i:02d}"
            walker.mkdir()
            sampler.write_record(walker / "start.json", state)
        batch.rename(dest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("export", "restore", "next"))
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--count", type=int, help="First walkers in directory order, selected before drawing")
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        if args.command == "restore":
            if args.count is not None:
                p.error("restore uses the complete saved cohort")
            initialize(sampler.read_record(args.source), args.destination)
        else:
            states = snapshot(args.source, args.count)
            if args.command == "next":
                initialize(states, args.destination)
            else:
                if args.destination.exists():
                    raise FileExistsError(args.destination)
                sampler.write_record(args.destination, states)


if __name__ == "__main__":
    main()
