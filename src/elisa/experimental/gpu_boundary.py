"""Opt-in complete curves with finite boundary-fitted Roche leaf quadrature."""

from typing import Any

from elisa.experimental.boundary_geometry import BoundaryRule, ConditionedRocheGeometry, boundary_packet
from elisa.experimental.gpu_curved import CurvedBinaryLightCurve


class BoundaryBinaryLightCurve(CurvedBinaryLightCurve):
    """Fresh finite original fields on curved patches with fitted leaf chords.

    The same CPU geometry/adaptation/clipping bridge serves reference/cpu/gpu.
    Fields, original phases/normalization and integration contracts are inherited.
    This standalone experiment is not a registered fitting or sampler backend.
    """

    _geometry_type = ConditionedRocheGeometry
    _packet = staticmethod(boundary_packet)
    _method_description = ("True Roche integration of piecewise-constant fresh original finite fields; "
                           "CPU conditioned shadows and finite fitted leaf chords; no continuum certificate")

    def __init__(self, passbands: list[str], rule: BoundaryRule, **kwargs: Any) -> None:
        if not isinstance(rule, BoundaryRule):
            msg = "Boundary curves require an explicit BoundaryRule"
            raise TypeError(msg)
        super().__init__(passbands, rule, **kwargs)
