"""Record and independently regenerate the same eclipse packets for a donor-only contrast."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_nodal_source import EmittingAreaLightCurve, SpatialReflectedLightCurve
from elisa.experimental.gpu_reconstructed_observer import _Packet
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem, prepare_fit
from scripts.validate_gpu_reconstructed import evaluate
from scripts.validate_gpu_surface_error import asset_revision


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


class CommonRays:
    """Only packet geometry is shared. Both donor fields are synthesized afresh."""

    def __init__(self, packet, folder, protocol, mode):
        self.packet, self.folder, self.mode = packet, folder, mode
        self.protocol, self.calls, self.rows = protocol, 0, []
        if mode == "record":
            folder.mkdir(exist_ok=False)
        else:
            archive = json.loads((folder / "manifest.json").read_text())
            if archive["protocol"] != protocol:
                raise ValueError("Cannot reuse packets from different physical geometry or settings")
            self.records = archive["packets"]

    def __call__(self, triangles, geometry, line, rule, foreground=None):
        current = self.packet(triangles, geometry, line, rule, foreground)
        arrays = dict(mu=current["mu"], area=current["area"], parents=current["parents"], rays=current.rays)
        header = dict(
            component=geometry.component,
            foreground=foreground.component if foreground is not None else None,
            line=digest(line),
            triangles=digest(triangles),
            rule=asdict(rule),
            phase_metadata={k: v for k, v in current.items() if k not in arrays},
        )
        # JSON normalizes NumPy scalar types consistently for regular and partitioned packets.
        header = json.loads(json.dumps(header, default=lambda a: a.item()))
        prefix = self.folder / f"packet-{self.calls:03d}"
        if self.mode == "record":
            hashes = {}
            for key, a in arrays.items():
                np.save(str(prefix) + f"-{key}.npy", a, allow_pickle=False)
                hashes[key] = digest(a)
            self.rows.append(dict(header=header, hashes=hashes))
        else:
            record = self.records[self.calls]
            if header != record["header"]:
                raise ValueError("New eclipse geometry changed the packet inputs or adaptive record")
            archived = _Packet(header["phase_metadata"])
            diff = {}
            for key, a in arrays.items():
                saved = np.load(str(prefix) + f"-{key}.npy", mmap_mode="r", allow_pickle=False)
                if saved.dtype != a.dtype or saved.shape != a.shape or digest(saved) != record["hashes"][key]:
                    raise ValueError("Packet shape, dtype or archive fingerprint changed")
                if key == "parents":
                    np.testing.assert_array_equal(a, saved)
                else:
                    # This compares fresh versus archived geometry. The actual
                    # donor contrast below requires byte-identical saved arrays.
                    np.testing.assert_allclose(a, saved, rtol=3e-13, atol=64 * np.finfo(float).eps)
                diff[key] = dict(
                    differing_elements=int(np.count_nonzero(a != saved)),
                    max_absolute=float(np.max(abs(a - saved), initial=0)),
                )
                if key == "rays":
                    archived.rays = saved
                else:
                    archived[key] = saved
            self.rows.append(diff)
            current = archived
        self.calls += 1
        return current

    def finish(self):
        if self.mode == "record":
            (self.folder / "manifest.json").write_text(
                json.dumps(dict(protocol=self.protocol, packets=self.rows)) + "\n"
            )
        elif self.calls != len(self.records):
            raise ValueError("Inconsistent number of component packets")
        return dict(
            mode=self.mode,
            packets=self.calls,
            archive_sha256=hashlib.sha256((self.folder / "manifest.json").read_bytes()).hexdigest(),
            regeneration_errors=self.rows if self.mode == "replay" else [],
        )


def run(args):
    previous = json.loads(args.profile.read_text())
    if (
        previous["engine"] not in ("spatial-reflection", "emitting-area")
        or not previous["irradiation"]
        or previous["batch"] != 1
        or len(previous["samples"]) != 1
        or previous["source_revision"] != asset_revision()
    ):
        raise ValueError("Use a single reflected spatial profile and identical table fixtures")
    if previous["backend"] == "gpu" and not args.verified_bounded_gpu_window:
        raise ValueError("GPU packet recording and replay require a verified bounded unit")
    started = perf_counter()
    fit = prepare_fit(
        make_problem(jd=previous["jd"], free_nuisance=True, shape_parameters=True),
        phase_mode="observations",
        mesh_mode="paired",
    )
    for key, actual in (("observations", fit.y_data), ("uncertainties", fit.y_err), ("reducers", fit.x_data_reducer)):
        for b, v in actual.items():
            np.testing.assert_array_equal(v, previous[key][b])
    setup = perf_counter()
    rule = CurvedEdgeRule(previous["order"], previous["depth"], slice_order=previous["slice_order"])
    engine = args.engine or previous["engine"]
    observer = EmittingAreaLightCurve if engine == "emitting-area" else SpatialReflectedLightCurve
    model = observer(
        list(fit.y_data),
        rule,
        levels=tuple(previous["levels"]),
        integration_level=previous["integration_level"],
        device=previous["backend"],
        receiver_points=previous["receiver_points"],
        donor_level=previous["donor_level"],
    )
    protocol = {
        k: previous[k]
        for k in (
            "iteration_start",
            "jd",
            "source_revision",
            "backend",
            "integration_level",
            "levels",
            "physical_constants",
            "irradiation",
            "order",
            "depth",
            "slice_order",
            "reducers",
            "observations",
            "uncertainties",
        )
    }
    original = previous["samples"][0]["rows"][0]
    protocol.update(
        physical=original["physical"],
        passbands=list(fit.y_data),
        phases=next(iter(original["finite_metadata"].values()))["phases"],
    )
    packets = CommonRays(model._packet, args.packets, protocol, args.mode)
    model._packet = packets
    built = perf_counter()
    sample = evaluate(model, fit, 1, previous["iteration_start"], previous["source_revision"])
    row = sample["rows"][0]
    if engine != previous["engine"]:
        suffix = f"_receiver{previous['receiver_points']}_donor_cap{previous['donor_level']}"
        for metadata in original["finite_metadata"].values():
            rule = metadata["rule"]
            old = previous["engine"].replace("-", "_") + suffix
            if rule.pop(old) != 1:
                raise ValueError("Unexpected reflection rule in input profile")
            rule[model.field_rule] = 1
    for k in ["physical", "finite_metadata", "phase_records"]:
        if json.loads(json.dumps(row[k], default=lambda a: a.tolist())) != original[k]:
            raise ValueError("Proposal, finite model or observation schedule changed")
    for k in [
        "backend",
        "irradiation",
        "observations_irradiated",
        "batch",
        "jd",
        "levels",
        "iteration_start",
        "integration_level",
        "engine",
        "depth",
        "receiver_points",
        "donor_level",
        "slice_order",
        "order",
        "observations",
        "uncertainties",
        "reducers",
        "source_revision",
        "physical_constants",
    ]:
        if k not in previous:
            raise KeyError(k)
    report = {k: previous[k] for k in protocol if k in previous}
    for k in ("engine", "observations_irradiated", "batch", "receiver_points", "donor_level"):
        report[k] = previous[k]
    report.update(
        engine=engine,
        samples=[sample],
        archived_geometry=packets.finish(),
        fixed_input_setup_seconds=setup - started,
        constructor_seconds=built - setup,
        main_wall_seconds=perf_counter() - started,
        contract="Single-proposal shared quadrature control including packet I/O and independent regeneration. Both emitting fields and full curves are rebuilt. Validation timing only.",
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False, default=lambda a: a.tolist()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--engine", choices=("spatial-reflection", "emitting-area"))
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--mode", choices=("record", "replay"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verified-bounded-gpu-window", action="store_true")
    args = parser.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        run(args)
