"""Reproduce the finite-resolution figure from curated milestone measurements."""

# ruff: noqa: INP001 - standalone scientific figure, not runtime synthesis
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("background-m008.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("background-m008-resolution.svg"))
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()
    curves = json.loads(args.input.read_text())["profiles"]["gpu"]["curves"]
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "elisa-m008"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for case, label in zip(curves["mesh"], ["Partial BB", "Total BB", "Partial CK04"], strict=True):
        rows = case["levels"][:-1]
        axes[0].plot([r["discretization_deg"] for r in rows],
                     [r["vs_finest_3deg"]["joint"]["distance_sigma"] for r in rows], "o-", label=label)
    axes[0].set(title="Mesh change: finite 3° CPU reference", xlabel="Discretization (degrees)",
                xticks=[12, 8, 6, 4], xlim=(12.5, 3.5))
    for key, label in [("0.0", "l3=0: interpolation control"), ("0.2", "l3=0.2: also changes support mean")]:
        rows = curves["phase"][key]["levels"]
        axes[1].plot([r["support_points"] for r in rows],
                     [r["native_vs_direct"]["joint"]["distance_sigma"] for r in rows], "o-", label=label)
    axes[1].set(title="Phase support vs direct observations", xlabel="Synthesis support points", xscale="log")
    for axis in axes:
        axis.axhline(.1, color="0.35", ls="--", label="D=0.1 diagnostic budget")
        axis.set(yscale="log", ylabel=r"Noise-scaled distance $D=\sqrt{\sum(\Delta F/\sigma_{eff})^2}$")
        axis.grid(alpha=.2, which="both")
        axis.legend(fontsize=8, loc="best")
    fig.text(.5, .025, r"$\sigma=10^{-4}$; original $\ln f=-2$. Finite comparisons are not continuum error bounds. "
             "CPU/GPU parity is checked separately.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(args.output, metadata={"Date": None})
    if args.output.suffix.lower() == ".svg":
        args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n")
    if args.png:
        fig.savefig(args.png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
