"""Separate the measured irradiation response from intrinsic field refinement."""

# ruff: noqa: INP001 - standalone analysis of complete scientific profiles
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _prediction(record: dict, level: str, band: str, indices: list) -> np.ndarray:
    normalized = np.asarray(record["curves"][level]["fluxes"][band])[indices]
    return normalized / normalized.mean()


def _delta_log_likelihood(change: np.ndarray, residual: np.ndarray, errors: np.ndarray) -> float:
    return float(np.sum((residual * change - 0.5 * change**2) / errors**2))


def _validate_protocols(h: dict, i: dict) -> tuple[str, str]:
    if (
        h["physical"] != i["physical"]
        or h["quadrature_sha256"] != i["quadrature_sha256"]
        or h["phase_records"] != i["phase_records"]
    ):
        msg = "Physical proposals and eclipse quadrature must match exactly"
        raise ValueError(msg)
    levels = sorted(h["curves"], key=lambda k: h["finite_metadata"][k]["field_levels"])
    for key in levels:
        for attr in ("rule", "field_levels", "phases", "additional_light"):
            if h["finite_metadata"][key][attr] != i["finite_metadata"][key][attr]:
                msg = "Finite thermal/observer protocols must match"
                raise ValueError(msg)
    coarse, fine = levels
    return coarse, fine


def compare(heated: dict, intrinsic: dict) -> list[dict]:
    """Compare complete, paired fields after each original observation mapping."""
    keys = (
        "backend",
        "engine",
        "batch",
        "jd",
        "levels",
        "integration_level",
        "depth",
        "slice_order",
        "order",
        "source_revision",
        "physical_constants",
        "observations",
        "uncertainties",
        "reducers",
        "iteration_start",
    )
    if (
        heated["irradiation"] is not True
        or intrinsic["irradiation"] is not False
        or any(heated[k] != intrinsic[k] for k in keys)
    ):
        msg = "Expected matched reflection-on and intrinsic-only profiles"
        raise ValueError(msg)
    rows = []
    for on, off in zip(heated["samples"], intrinsic["samples"], strict=True):
        for h, i in zip(on["rows"], off["rows"], strict=True):
            coarse, fine = _validate_protocols(h, i)
            bands = {}
            for band, reducer in heated["reducers"].items():
                hf, hc = _prediction(h, fine, band, reducer), _prediction(h, coarse, band, reducer)
                uf, uc = _prediction(i, fine, band, reducer), _prediction(i, coarse, band, reducer)
                # Check this is the mapping actually consumed by the timed likelihood.
                for row, pred in [(h, hf), (i, uf)]:
                    np.testing.assert_allclose(pred, row["prediction"][band], rtol=3e-13, atol=2e-12)
                heated_change, intrinsic_change = hc - hf, uc - uf
                differential_response = (hc - uc) - (hf - uf)
                np.testing.assert_allclose(
                    intrinsic_change + differential_response,
                    heated_change,
                    atol=8 * np.finfo(float).eps,
                    rtol=0,
                )
                sigma = np.asarray(heated["uncertainties"][band])
                residual = np.asarray(heated["observations"][band]) - hf

                cross = float(-np.sum(intrinsic_change * differential_response / sigma**2))
                bands[band] = {
                    "fine_irradiation_response": (hf - uf).tolist(),
                    "intrinsic_refinement": intrinsic_change.tolist(),
                    "irradiation_response_refinement": differential_response.tolist(),
                    "heated_refinement": heated_change.tolist(),
                    "delta_logL_intrinsic_at_heated_anchor": _delta_log_likelihood(intrinsic_change, residual, sigma),
                    "delta_logL_response_at_heated_anchor": _delta_log_likelihood(
                        differential_response, residual, sigma,
                    ),
                    "gaussian_cross_term": cross,
                    "delta_logL_heated": _delta_log_likelihood(heated_change, residual, sigma),
                }
            rows.append({"physical": h["physical"], "quadrature_sha256": h["quadrature_sha256"], "bands": bands})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("heated", "intrinsic", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    inputs = [p.read_bytes() for p in (args.heated, args.intrinsic)]
    report = {
        "rows": compare(*(json.loads(data) for data in inputs)),
        "input_sha256": [hashlib.sha256(data).hexdigest() for data in inputs],
        "contract": "Finite differences after original normalization and third light. "
        "Irradiation response includes spectral, LD and normalization changes. "
        "Signed decomposition is diagnostic. The separate full-curve accuracy gate applies.",
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
