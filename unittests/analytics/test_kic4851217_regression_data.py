# Copyright 2026 ELISa contributors
# keep it first
# due to astropy units/constants implementation
from unittests import set_astropy_units

from pathlib import Path

import numpy as np

from elisa.analytics import LCData

set_astropy_units()


DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "jupyter_tutorials/demo_data/lc_data/KIC04851217.lc.pulse_removed.dat"
)
CHECKPOINT_INDICES = np.array([0, 499, 999, 1999, 2999, 3999, 4997])
EXPECTED_OBSERVATION_COUNT = 4998


def test_kic4851217_observation_count_phases_and_uncertainties() -> None:
    observations = LCData.load_from_file(DATA_PATH, x_unit=None, y_unit=None)

    assert len(observations.x_data) == EXPECTED_OBSERVATION_COUNT
    np.testing.assert_allclose(
        observations.y_err[CHECKPOINT_INDICES],
        [3.8e-5, 3.5e-5, 3.7e-5, 3.5e-5, 3.8e-5, 3.8e-5, 3.6e-5],
        rtol=0.0,
        atol=np.finfo(float).eps,
    )
    assert np.all(observations.y_err > 0)

    observations.convert_to_phases(2.47028, 54953.900507000006, centre=0.0)
    np.testing.assert_allclose(
        observations.x_data[CHECKPOINT_INDICES],
        [
            -0.14640579205942533,
            -0.4560816992426773,
            -0.295183015693242,
            0.06793541622397825,
            0.22688913806915778,
            -0.4019624900844292,
            -0.46020576615273967,
        ],
        rtol=0.0,
        atol=5e-12,
    )
