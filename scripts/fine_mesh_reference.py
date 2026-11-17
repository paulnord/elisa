"""Compatibility imports for the independent CPU original-primitives reference."""

# ruff: noqa: INP001 - standalone validation compatibility module

from elisa.experimental.gpu_reference import NAMES, cpu_reflect, intrinsic, observe

__all__ = ["NAMES", "cpu_reflect", "intrinsic", "observe"]
