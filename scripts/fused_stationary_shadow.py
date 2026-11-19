"""Opt-in fused FP64 trial for the walker's conditioned stationary shadow."""

import ctypes
from time import perf_counter

import numpy as np

from scripts.walker_roche_geometry import WalkerRocheGeometry


class StationarySolve:
    """Own a synchronous CPU or CUDA solver for one observer evaluation."""

    def __init__(self, library, *, gpu, audit=False):
        self.lib = ctypes.CDLL(str(library))
        dll = self.lib
        dll.eclipse_create.argtypes = []
        dll.eclipse_create.restype = ctypes.c_void_p
        dll.eclipse_destroy.argtypes = [ctypes.c_void_p]
        dll.eclipse_destroy.restype = None
        ptr = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        dll.eclipse_solve.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ptr, ptr,
                                      *[ctypes.c_double]*5, ptr]
        dll.eclipse_solve.restype = ctypes.c_int
        self.handle = dll.eclipse_create()
        if not self.handle:
            raise MemoryError("Could not allocate stationary solver")
        self.gpu, self.audit = gpu, audit
        self.stats = dict(calls=0, selected_rays=0, reference_rays=0, audited_rays=0, max_margin_difference=0.,
                          h2d_bytes=0, d2h_bytes=0, solve_seconds=0.)

    def geometry(self, binary, component):
        geom = FusedShadow(binary, component)
        geom.solver = self
        return geom

    def offsets(self, base, line, geometry):
        if not self.handle:
            raise RuntimeError("Closed stationary solver")
        if base.ndim != 2 or base.shape[1] != 3 or line.shape != (3,):
            raise ValueError("Expected three Cartesian coordinates per ray and line of sight")
        n = len(base)
        primary = geometry.component == "primary"
        q = geometry.binary.mass_ratio
        out = np.empty(n, dtype=np.float64)
        start = perf_counter()
        error = self.lib.eclipse_solve(self.handle, self.gpu, n, np.ascontiguousarray(base),
                                       np.ascontiguousarray(line), geometry.mass, q if primary else 1.,
                                       -1. if primary else 1., geometry.star.synchronicity**2*(1+q),
                                       q if primary else -1., out)
        self.stats["solve_seconds"] += perf_counter()-start
        if error:
            raise RuntimeError(f"CUDA stationary solve or transfer failed with runtime code {error}")
        self.stats["selected_rays"] += n
        if self.gpu:
            self.stats["h2d_bytes"] += base.nbytes
            self.stats["d2h_bytes"] += out.nbytes
        return out

    def close(self):
        if self.handle:
            self.lib.eclipse_destroy(self.handle)
            self.handle = None


class FusedShadow(WalkerRocheGeometry):
    def shadow(self, points, line):
        solver = self.solver
        solver.stats["calls"] += 1
        delta = points - self.centre
        base = delta - np.outer(delta @ line, line)
        rho = np.linalg.norm(base, axis=1)
        inside = rho < self.inner
        distance = rho - self.inner
        selected = np.flatnonzero((rho >= self.inner) & (rho < .25))
        if not selected.size:
            return inside, distance
        radii = rho[selected]
        t = solver.offsets(base[selected], line, self)
        # The reference checks the final gradient and evaluates the original stellar potential on the host.
        valid_t = np.isfinite(t) & (np.abs(t) <= radii/2)
        local = base[selected] + np.outer(np.where(valid_t, t, 0.), line)
        g = self.gradient_local(local)
        value = self.potential_local(local)
        gap = self.star.surface_potential - value
        eps = np.finfo(np.float64).eps
        valid = (valid_t & np.isfinite(g).all(axis=1) & np.isfinite(value)
                 & (np.abs(g @ line) <= 64*eps*np.linalg.norm(g, axis=1)))
        # Near the zero set, run the original solve so that last-bit differences cannot pick a clipping branch.
        retry = ~valid | (np.abs(gap) <= 256*eps*np.maximum(np.abs(value), abs(self.star.surface_potential)))
        inside[selected] = gap < 0
        distance[selected] = gap*radii*radii/self.mass
        if np.any(retry):
            ids = selected[retry]
            inside[ids], distance[ids] = super().shadow(points[ids], line)
            solver.stats["reference_rays"] += len(ids)
        if solver.audit:
            # Sample across the ray array and near the physical silhouette in every selected packet.
            ids = np.unique(np.r_[selected[np.linspace(0, len(selected)-1, min(len(selected), 9), dtype=int)],
                                  selected[np.argmin(abs(distance[selected]))]])
            opaque, margin = super().shadow(points[ids], line)
            np.testing.assert_array_equal(inside[ids], opaque)
            np.testing.assert_allclose(distance[ids], margin, rtol=3e-13, atol=2e-12)
            solver.stats["audited_rays"] += len(ids)
            solver.stats["max_margin_difference"] = max(solver.stats["max_margin_difference"],
                                                         float(np.max(np.abs(distance[ids]-margin))))
        return inside, distance
