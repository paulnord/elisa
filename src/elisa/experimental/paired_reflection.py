"""Cross two thermal fields with two donor resolutions on each fresh observer packet."""

from dataclasses import dataclass
from functools import partial

from elisa.experimental.gpu_curvature_fields import CurvatureFieldSpec, _CurvatureEngine
from elisa.experimental.gpu_nodal_source import EmittingAreaLightCurve
from elisa.experimental.gpu_resident_geometry import _ResidentEngine


@dataclass(frozen=True)
class _DonorField(CurvatureFieldSpec):
    donor_level: int

    @property
    def key(self):
        return f"{super().key}_donor{self.donor_level}"


class PairedEmittingAreaLightCurve(EmittingAreaLightCurve):
    """One observer traversal per depth, with four independently irradiated fields."""

    def __init__(self, passbands, rule, *, device, levels=(2, 3), integration_level=3):
        if len(levels) != 2:
            raise ValueError("Select two distinct surface field levels for the donor comparison")
        super().__init__(passbands, rule, device=device, levels=levels, integration_level=integration_level,
                         receiver_points=3, donor_level=2)
        variants = tuple(_DonorField(v.level, cap) for cap in (2, 1) for v in self.variants)
        engines = {}
        coarse = self.solver_type(self.law, device=device, receiver_points=3, donor_level=1)
        for v in variants:
            if v.donor_level == 2:
                engine = self.engines[CurvatureFieldSpec(v.level).key]
            else:
                if device == "reference":
                    engine = _CurvatureEngine(passbands, v, device=device, radiance=self.radiance)
                    engine.area_thermal = self.area_thermal
                else:
                    engine = _ResidentEngine(passbands, device=device, radiance=self.radiance)
                engine.area_reflection = partial(coarse.evaluate, level=v.level)
            engines[v.key] = engine
        self.variants, self.engines = variants, engines
        self._rules = {v.key: f"{self.rule_prefix}_receiver3_donor_cap{v.donor_level}" for v in variants}

    def _rule_for_field(self, key):
        return self._rules[key]
