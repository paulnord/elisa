"""Transfer a single physical walker's fresh optical fields to a CPU eclipse integrator."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from elisa.experimental.gpu_curved_edges import CurvedEdgeRule, SymmetricInteriorRule
from elisa.experimental.gpu_resident_geometry import _angular_inputs, _signature
from elisa.experimental.node_reemission import NodeReemissionLightCurve
from elisa.experimental.primary_cubic import CurvedHeatingBackend
from elisa.experimental.quadratic_source import QuadraticSourceBackend
from scripts import sample_reflected_binary as sampler
from scripts.walker_roche_geometry import WalkerRocheGeometry

STARS = ("primary", "secondary")


def rule_for(interior, depth):
    if depth not in (3, 4) or interior not in ("six", "duffy"):
        raise ValueError("Expected a pinned eclipse depth and interior rule")
    options = dict(slice_order=12, junction_depth=4)
    if interior == "six":
        return SymmetricInteriorRule(3, depth, interior_degree=4, **options)
    return CurvedEdgeRule(3, depth, **options)


def binding(model, observer, state, key):
    return dict(
        target=sampler.digest(state["target"]),
        trial=key,
        backend=state["backend"],
        system=_signature(model),
        bands=list(observer.passband),
    )


class _ExportNodes:
    def optical_field(self, tri, values, bands, control):
        result = super().optical_field(tri, values, bands, control)
        result.collocation_samples = np.asarray(values)
        return result


class _PrimaryExport(_ExportNodes, CurvedHeatingBackend):
    pass


class _SecondaryExport(_ExportNodes, QuadraticSourceBackend):
    pass


class FieldExporter(NodeReemissionLightCurve):
    """Record the collocated optical moments before conversion to a device polynomial basis."""

    @staticmethod
    def field_backend(component, law, **kwargs):
        cls = {"primary": _PrimaryExport, "secondary": _SecondaryExport}[component]
        return cls(component, law, **kwargs)

    def collect(self, model, observer, identity):
        if self.device == "gpu":
            faces = 0
            plans = {}
            for variant in self.variants:
                system, inputs = _angular_inputs(model, variant.level)
                faces += sum(len(getattr(system, n).faces) for n in STARS)
                plans[variant.key] = system, inputs
            if faces > self.max_resident_faces:
                raise ValueError("Walker exceeds the resident geometry budget")
            for key, (system, inputs) in plans.items():
                states = {}
                for name in STARS:
                    group = self.samplers[name].evaluate([inputs[name]])
                    states[name] = {k: a[0] for k, a in group.items()}
                self.engines[key].prepared = (_signature(model), system, inputs, states)
        _, triangles, _, maps, controls = self._prepare(model, observer)
        if self.device == "gpu" and any(engine.prepared is not None for engine in self.engines.values()):
            raise RuntimeError("Incomplete device field evaluation")
        arrays = {n: triangles[n] for n in STARS}
        for (variant, name), field in self._reconstruction.items():
            cells, samples = field.triangles, field.collocation_samples
            prefix = variant + "_" + name
            arrays[prefix + "_cells"] = cells
            arrays[prefix + "_samples"] = samples
            arrays[prefix + "_parents"] = maps[variant][name]
        return dict(binding=identity, controls=controls), arrays


def save_fields(path, meta, arrays):
    """Commit the transport only after all device arrays and closure checks return."""
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            np.savez(tmp, metadata=np.frombuffer(sampler.encoded(meta), dtype=np.uint8), **arrays)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)


def load_fields(path, expected):
    with np.load(path, allow_pickle=False) as source:
        meta = json.loads(source["metadata"].tobytes())
        if meta["binding"] != expected:
            raise ValueError("Optical fields do not belong to this walker and target")
        arrays = {k: source[k] for k in source.files if k != "metadata"}
    return meta, arrays


class WalkerObserver(NodeReemissionLightCurve):
    """Use one walker's optical solution at each independent observer resolution."""

    _geometry = staticmethod(WalkerRocheGeometry)

    def attach(self, model, meta, arrays, expected):
        if meta["binding"] != expected or _signature(model) != expected["system"]:
            raise ValueError("Incompatible walker optical fields")
        self.prepared_fields = {}
        triangles = {n: arrays[n] for n in STARS}
        maps, geometry = {}, {}
        for variant in self.variants:
            key = variant.key
            maps[key] = {}
            sizes = {}
            for name in STARS:
                tag = key + "_" + name
                cells, samples = arrays[tag + "_cells"], arrays[tag + "_samples"]
                parents = arrays[tag + "_parents"]
                nodes = 10 if name == "primary" else 6
                if (
                    any(a.dtype != np.float64 for a in (triangles[name], cells, samples))
                    or cells.shape != (4 * len(samples), 3, 3)
                    or samples.shape[1:] != (nodes, len(self.passbands), 2)
                    or not np.issubdtype(parents.dtype, np.integer)
                    or parents.shape != (len(triangles[name]),)
                    or np.any((parents < 0) | (parents >= len(cells)))
                ):
                    raise ValueError("Invalid FP64 walker field or angular parent map")
                control = meta["controls"][key][name]["reconstructed_field"]
                self.prepared_fields[key, name] = self.moment_backends[name].optical_field(
                    cells, samples, self.passbands, control
                )
                maps[key][name] = parents
                sizes[name] = SimpleNamespace(faces=range(len(cells)))
            geometry[key] = SimpleNamespace(**sizes)
        self.preparation = (model, triangles, geometry, maps, meta["controls"])

    def _prepare(self, model, observer):
        if _signature(model) != _signature(self.preparation[0]) or tuple(observer.passband) != self.passbands:
            raise ValueError("Cannot reuse fields for a different star or observer")
        self._reconstruction = self.prepared_fields
        return self.preparation
