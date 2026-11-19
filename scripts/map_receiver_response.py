"""Map the projected primary spectral response to the original normalized eclipse observations."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from scripts.check_interior_resolution import mapped, stellar_changes
from scripts.compare_gpu_joint_budget import curve
from scripts.validate_receiver_heating import source_class


def compare(observed_run, optical):
    cap = optical["donor_level"]
    profile, = (p for p in observed_run["profiles"] if p["donor_level"] == cap)
    row = profile["samples"][0]["rows"][0]
    cls = source_class(False, True, optical["horizon_mean"])
    if (profile["engine"] != "symmetric-shadow-" + cls.rule_prefix.replace("_", "-")
            or row["physical"] != optical["physical"] or profile["depth"] != optical["depth"]
            or profile["source_revision"] != optical["source_revision"]):
        raise ValueError("Projected heat and complete curve describe different models")
    disks = optical["disks"]
    bands = optical["bands"]
    sigma, observed, reducer = (profile[k] for k in ("uncertainties", "observations", "reducers"))
    coarse, fine = (curve(row, f"curvature_field{i}") for i in (2, 3))
    order = np.asarray(optical["indices_to_visible_disks"])
    phases = np.asarray(optical["input_phases"])
    shift = row["physical"]["system@phase_shift"]
    np.testing.assert_array_equal(fine.phases, (phases - shift) + shift)
    for disk, record in zip(disks, row["phase_records"], strict=True):
        if disk["phase"] != record["phase"] or disk["nodes"] != record["components"]["primary"]["selected_nodes"]:
            raise ValueError("Changed primary observer packets")
    for i, c in enumerate((coarse, fine)):
        for k, b in enumerate(bands):
            np.testing.assert_allclose([disks[j]["primary_flux"][i][k] for j in order], c.component_fluxes[b][:,0],
                                       rtol=3e-13, atol=0)

    def mix(primary, secondary):
        both = {b: np.column_stack((primary[b], secondary.component_fluxes[b][:,1])) for b in bands}
        return mapped(replace(fine, component_fluxes=both), reducer)

    low, high = ({b: c.component_fluxes[b][:,0] for b in bands} for c in (coarse, fine))
    def delta(term, k):
        return np.array([sum(disks[j]["changes_by_band_and_moment"][term][k]) for j in order])
    transfer = {b: high[b] - delta("transferred_heat", k) for k, b in enumerate(bands)}
    with_mean = {b: transfer[b] + delta("face_mean", k) for k, b in enumerate(bands)}
    with_shape = {b: with_mean[b] + delta("within_face", k) for k, b in enumerate(bands)}
    names = ("other_field_changes", "face_mean", "within_face", "linearization_error")
    out = {k: [] for k in names}
    for s in (coarse, fine):
        states = [mix(p, s) for p in (low, transfer, with_mean, with_shape, high)]
        for key, a, z in zip(names, states[:-1], states[1:], strict=True):
            terms = []
            for b in bands:
                step = (z[b] - a[b]) / sigma[b]
                terms.extend(step * ((observed[b] - a[b]) / sigma[b] - 0.5 * step))
            out[key].append(terms)
    by_observation = {key: np.mean(val, axis=0) for key, val in out.items()}
    sums = {key: float(val.sum()) for key, val in by_observation.items()}
    total = stellar_changes(coarse, fine, reducer, sigma, observed)["nominal_log_likelihood_changes"]["primary"]
    np.testing.assert_allclose(sum(sums.values()), total["signed_change"], atol=2e-10, rtol=3e-13)
    return dict(nominal_log_likelihood=sums, primary_total=total, by_observation=by_observation,
                observations=[dict(band=b, index=i, phase=fine.phases[j]) for b in bands for i,j in enumerate(reducer[b])])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("curve", type=Path)
    p.add_argument("response", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    out = compare(json.loads(args.curve.read_bytes()), json.loads(args.response.read_bytes()))
    args.output.write_text(json.dumps(out, default=lambda a: a.tolist(), allow_nan=False) + "\n")
