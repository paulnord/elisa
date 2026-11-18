"""Attribute resolution changes on the projected primary disk to intrinsic light, mean heat and heat variation."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from elisa import settings
from scripts.check_interior_resolution import mapped, stellar_changes
from scripts.compare_gpu_joint_budget import curve
from scripts.receiver_phase_parts import join
from scripts.validate_gpu_atmosphere import table_settings


def field_contrast(coarse, fine, controls, reducers, sigma, observed):
    """Apply the original observation mapping to every mixed curve before taking differences.

    Average the primary replacement with either secondary mesh held fixed. For
    each secondary, the difference between full and mean heating measures the
    effect of spatial variation on the coarse-to-fine contrast, including its
    interaction with the nonlinear spectrum and observational normalization.
    """
    labels = ("intrinsic", "mean", "full")
    primary = {}
    for level, c in ((2, coarse), (3, fine)):
        for mode in labels:
            primary[level, mode] = (
                {b: a[:, 0] for b, a in c.component_fluxes.items()} if mode == "full" else controls[level][mode]
            )

    def prediction(p, s):
        components = {b: np.column_stack((p[b], a[:, 1])) for b, a in s.component_fluxes.items()}
        sample = mapped(replace(fine, component_fluxes=components), reducers)
        return np.concatenate([sample[b] for b in sigma])

    errors = np.concatenate([sigma[b] for b in sigma])
    data = np.concatenate([observed[b] for b in sigma])

    def delta(lo, hi):
        change = (hi - lo) / errors
        return np.stack((change, change * ((data - lo) / errors - 0.5 * change)))

    contrasts = {}
    for mode in labels:
        shifts = [delta(prediction(primary[2, mode], s), prediction(primary[3, mode], s)) for s in (coarse, fine)]
        contrasts[mode] = 0.5 * (shifts[0] + shifts[1])

    secondary = 0.5 * sum(
        delta(prediction(primary[level, "full"], coarse), prediction(primary[level, "full"], fine)) for level in (2, 3)
    )
    total = delta(prediction(primary[2, "full"], coarse), prediction(primary[3, "full"], fine))
    components = dict(
        intrinsic=contrasts["intrinsic"],
        face_mean=contrasts["mean"] - contrasts["intrinsic"],
        within_face=contrasts["full"] - contrasts["mean"],
        secondary=secondary,
    )
    np.testing.assert_allclose(sum(components.values()), total, atol=2e-12, rtol=3e-13)
    components.update(primary=contrasts["full"], total=total)
    independent = stellar_changes(coarse, fine, reducers, sigma, observed)
    for name in ("primary", "secondary", "total"):
        np.testing.assert_allclose(
            components[name][1].sum(),
            independent["nominal_log_likelihood_changes"][name]["signed_change"],
            atol=2e-10,
            rtol=3e-13,
        )
    return {
        "observations": [
            {"band": b, "index": i, "phase": float(fine.phases[j])} for b in sigma for i, j in enumerate(reducers[b])
        ],
        "changes": {
            name: dict(
                joint_distance_sigma=float(np.linalg.norm(z[0])),
                nominal_delta_log_likelihood=float(z[1].sum()),
                flux_change_sigma=z[0],
                log_likelihood_by_observation=z[1],
            )
            for name, z in components.items()
        },
    }


def assemble(base, identity, blocks, degree):
    ordered = sorted(blocks, key=lambda x: x["part"])
    # join verifies the physical inputs, both full fields, all phase indices, symmetry and finite controls.
    out = join(base, identity, ordered, symmetric_interior=degree, curved_heating=True)
    for profile in out["profiles"]:
        cap = profile["donor_level"]
        (row,) = profile["samples"][0]["rows"]
        controls = {}
        for level in (2, 3):
            key = f"curvature_field{level}_donor{cap}"
            c = curve(row, f"curvature_field{level}")
            controls[level] = {mode: {} for mode in ("intrinsic", "mean")}
            for mode, bands in controls[level].items():
                for band in c.component_fluxes:
                    chunks = [np.asarray(p["primary_flux"][key][mode][band]) for p in ordered]
                    if any(chunk.shape != (len(p["indices"]),) for chunk, p in zip(chunks, ordered, strict=True)):
                        raise ValueError("Primary control does not cover exactly the same phases")
                    values = np.concatenate(chunks)
                    if np.any(values < 0) or not np.isfinite(values).all():
                        raise ValueError("Invalid primary control spectrum")
                    bands[band] = values
        row["projected_heating"] = field_contrast(
            curve(row, "curvature_field2"),
            curve(row, "curvature_field3"),
            controls,
            profile["reducers"],
            profile["uncertainties"],
            profile["observations"],
        )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("blocks", type=Path, nargs=2)
    p.add_argument("--six-point", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    raw = args.baseline.read_bytes()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        out = assemble(
            json.loads(raw),
            hashlib.sha256(raw).hexdigest(),
            [json.loads(path.read_bytes()) for path in args.blocks],
            4 if args.six_point else None,
        )
    args.output.write_text(json.dumps(out, default=lambda a: a.tolist(), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
