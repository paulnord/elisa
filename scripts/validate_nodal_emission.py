"""Check cell-local intensity at interior points against direct CPU source evaluation."""

# ruff: noqa: INP001 - bounded independent CPU validation
from __future__ import annotations

import argparse
import json
import sys
from copy import copy
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import Observer, const, ld, settings
from elisa.atm import atm
from elisa.base.curves.utils import get_component_limbdarkening_cfs
from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import duffy_nodes
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_nodal_source import NodalSourceLightCurve
from scripts.validate_gpu_atmosphere import fixture as atmosphere_fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture


def assess(atlas: str, level: int) -> list[dict]:
    bands = ["Generic.Bessell.V", "bolometric"]
    case = {"inclination": 85, "discretization": 12, "spin": [0.85, 1.15]}
    binary = atmosphere_fixture(case) if atlas == "ck04" else fixture(case)
    obs = Observer(passband=bands, system=binary)
    engine = NodalSourceLightCurve(
        bands,
        CurvedEdgeRule(3, 1),
        device="reference",
        levels=(level,),
        integration_level=level,
    )
    variant = engine.variants[0]
    system, triangles = engine.engines[variant.key]._fields(binary, obs)
    fields = engine._prepare_field(binary, system, obs, variant)  # noqa: SLF001 - exercise the production field builder
    rows = []
    for name, field in fields.items():
        star = getattr(system, name)
        n = star.base_symmetry_faces_number
        rays, weights = duffy_nodes(triangles[name][:n], 8)
        points, _, jac = ConditionedRocheGeometry(binary, name).sample(rays)
        area = weights * jac
        g = np.linalg.norm(
            calculate_potential_gradient(1.0, name, points, star.synchronicity, binary.mass_ratio),
            axis=1,
        )
        power = g**star.gravity_darkening
        direct = copy(star)
        direct.temperatures = star.t_eff * (power / np.average(power, weights=area)) ** 0.25
        direct.log_g = np.log10(g * (const.G * binary.primary.mass / binary.semi_major_axis**2))
        coeff = get_component_limbdarkening_cfs(direct, bands, symmetry_test=False)
        # Keep spectral allocations bounded while preserving the original LD query for the complete star.
        chunks = {band: [] for band in bands}
        for start in range(0, len(rays), 512):
            sel = slice(start, start + 512)
            flux = atm.NaiveInterpolatedAtm.radiance(
                temperature=direct.temperatures[sel],
                log_g=direct.log_g[sel],
                metallicity=star.metallicity,
                atlas=star.atmosphere,
                passband=obs.passband,
                left_bandwidth=obs.left_bandwidth,
                right_bandwidth=obs.right_bandwidth,
            )
            for band in bands:
                chunks[band].append(flux[band])
        intensity = atm.correct_normal_radiance_to_optical_depth(
            {name: {band: np.concatenate(parts) for band, parts in chunks.items()}},
            {name: coeff},
        )[name]
        parents = np.repeat(np.arange(n), 64)
        moments = field.sample(parents, rays)
        for i, band in enumerate(bands):
            rad = moments[:, i, 0]
            nodal_c = moments[:, i, 1:] / rad[:, None]
            for cosine in [0.2, 0.7, 1.0]:

                def directional(normal: np.ndarray, coefficients: np.ndarray, mu: float = cosine) -> np.ndarray:
                    return normal * ld.limb_darkening_factor(
                        coefficients=coefficients,
                        limb_darkening_law=settings.LIMB_DARKENING_LAW,
                        cos_theta=np.full(len(normal), mu),
                    )

                truth = directional(intensity[band], coeff[band])
                values = {
                    "face_constant": directional(star.normal_radiance[band][parents], star.ld_cfs[band][parents]),
                    "face_local": directional(rad, nodal_c),
                }
                metrics = {}
                for mode, value in values.items():
                    relative = value / truth - 1.0
                    metrics[mode] = {
                        "rms_relative": float(np.sqrt(np.average(relative**2, weights=area))),
                        "max_abs_relative": float(np.max(np.abs(relative))),
                    }
                rows.append(
                    {
                        "atlas": atlas,
                        "component": name,
                        "level": level,
                        "band": band,
                        "mu": cosine,
                        "quarter_faces": n,
                        "interior_samples": len(rays),
                        "metrics": metrics,
                    },
                )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=False, USE_SINGLE_LD_COEFFICIENTS=False):
        rows = [row for atlas in ["bb", "ck04"] for level in (0, 1) for row in assess(atlas, level)]
    args.output.write_text(json.dumps({"rows": rows, "order": 8, "irradiation": False}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
