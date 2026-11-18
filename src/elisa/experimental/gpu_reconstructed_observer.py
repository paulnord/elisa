"""Opt-in continuous conservative fields in the current curved-intersection path."""

from __future__ import annotations

from dataclasses import replace
from threading import Lock
from typing import Any

import numpy as np

from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from elisa.experimental.gpu_curved_edges import curved_edge_packet
from elisa.experimental.gpu_fused_observer import FusedObserverLightCurve
from elisa.experimental.gpu_observer_edges import observer_packet
from elisa.experimental.gpu_reconstructed_field import MomentFieldBackend


class _RecordingGeometry:
    def __init__(self, geometry: Any) -> None:
        self.geometry = geometry
        self.last = None

    def __getattr__(self, key: str) -> Any:
        return getattr(self.geometry, key)

    @property
    def line(self) -> Any:
        return getattr(self.geometry, "line", None)

    @line.setter
    def line(self, value: Any) -> None:
        self.geometry.line = value

    def sample(self, rays: Any) -> tuple:
        result = self.geometry.sample(rays)
        self.last = (rays, result[1])
        return result


class _Packet(dict):
    """Ray snapshot is an attribute, not a phase-record/hash metadata field."""


def recorded_packet(triangles: Any, geometry: Any, line: Any, rule: Any, foreground: Any = None) -> dict:
    g = _RecordingGeometry(geometry)
    method = observer_packet if hasattr(geometry, "field") else curved_edge_packet
    result = _Packet(method(triangles, g, line, rule, foreground))
    if result["nodes"]:
        rays, normal = g.last
        mu = normal @ line
        valid = mu > 0
        if not np.array_equal(mu[valid], result["mu"]):
            msg = "Reconstruction ray snapshot does not match the actual final packet"
            raise UnsupportedCurvedGeometryError(msg)
        result.rays = np.asarray(rays)[valid]
    else:
        result.rays = np.empty((0, 3))
    if len(result.rays) != len(result["parents"]):
        msg = "Reconstruction packet node identity mismatch"
        raise UnsupportedCurvedGeometryError(msg)
    return result


_PARTITION_FACES = 8192


def partitioned_packet(triangles: Any, geometry: Any, line: Any, rule: Any, foreground: Any = None) -> dict:
    """Bound temporary depth-four frontiers and enforce the cap on the combined final packet."""
    block = _PARTITION_FACES
    if rule.depth < 4 or len(triangles) <= block:
        return recorded_packet(triangles, geometry, line, rule, foreground)
    parts = []
    total = 0
    for start in range(0, len(triangles), block):
        part = recorded_packet(triangles[start : start + block], geometry, line, rule, foreground)
        total += part["nodes"]
        if total > rule.max_nodes:
            msg = "Combined curved-edge quadrature exceeds explicit node cap"
            raise UnsupportedCurvedGeometryError(msg)
        part["parents"] += start
        parts.append(part)
    result = _Packet()
    for key in ("mu", "area", "parents"):
        result[key] = np.concatenate([p[key] for p in parts])
    result.rays = np.concatenate([p.rays for p in parts])
    counts = []
    for depth in range(max(len(p["counts"]) for p in parts)):
        rows = [p["counts"][depth] for p in parts if len(p["counts"]) > depth]
        counts.append({k: depth if k == "depth" else sum(r[k] for r in rows) for k in rows[0]})
    result["counts"] = counts
    for key in parts[0].keys() - result.keys():
        values = [p[key] for p in parts]
        result[key] = max(values) if key == "max_absolute_slice_shift" else sum(values)
    result["partition_faces"] = block
    result["largest_partition_nodes"] = max(p["nodes"] for p in parts)
    return result


class ReconstructedObserverLightCurve(FusedObserverLightCurve):
    """Fresh positive C0 moment fields; unchanged thermal/reflection equations.

    Continuous reconstruction is a declared finite approximation. Original
    constant fields/default CPU synthesis remain available unchanged. Both
    representations need independent field/depth/combined-resolution gates.
    """

    field_backend = MomentFieldBackend
    field_rule = "conservative_moment_reconstruction"
    field_contract = (
        "Positive conservative C0 emitted-intensity moments; finite field approximation, no continuum/fit certificate"
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.moment_backends = {
            name: self.field_backend(name, self.law, device=self.device) for name in ["primary", "secondary"]
        }
        self._packet = recorded_packet
        self._reconstruction = {}
        self.last_field_stats = {}
        self._batch_lock = Lock()

    def _prepare(self, model: Any, observer: Any) -> tuple:
        self._reconstruction = {}
        common, triangles, fields, maps, controls = super()._prepare(model, observer)
        for variant in self.variants:
            prepared = self._prepare_field(model, fields[variant.key], observer, variant)
            for name, item in prepared.items():
                self._reconstruction[variant.key, name] = item
                controls[variant.key][name]["reconstructed_field"] = item.control
        return common, triangles, fields, maps, controls

    def _prepare_field(self, model: Any, system: Any, observer: Any, variant: Any) -> dict:  # noqa: ARG002
        return {
            name: backend.prepare(model, getattr(system, name), self.passbands)
            for name, backend in self.moment_backends.items()
        }

    def _integrate_packets(self, model: Any, fields: dict, maps: dict, packets: dict, variant: Any) -> dict:  # noqa: ARG002
        values = []
        for name in ["primary", "secondary"]:
            packet = packets[name]
            index = maps[variant.key][name][packet["parents"]]
            values.append(
                self._reconstruction[variant.key, name].flux(
                    index,
                    packet.rays,
                    packet["mu"],
                    packet["area"] * model.semi_major_axis**2,
                ),
            )
        return {band: np.array([v[i] for v in values]) for i, band in enumerate(self.passbands)}

    def batch(self, *args: Any, **kwargs: Any) -> Any:
        # A rejected overlapping call must leave the active candidate's arrays intact.
        if not self._batch_lock.acquire(blocking=False):
            msg = "Reconstructed observer batches are non-reentrant"
            raise ValueError(msg)
        try:
            for backend in self.moment_backends.values():
                backend.stats = {}
            result = super().batch(*args, **kwargs)
            self.last_field_stats = {n: dict(b.stats) for n, b in self.moment_backends.items()}
            return replace(
                result,
                results=tuple(
                    replace(
                        r,
                        curves={k: replace(v, rule={**v.rule, self.field_rule: 1}) for k, v in r.curves.items()},
                        diagnostics={
                            **r.diagnostics,
                            "field_contract": self.field_contract,
                        },
                    )
                    for r in result.results
                ),
            )
        finally:
            self._reconstruction = {}
            self._batch_lock.release()
