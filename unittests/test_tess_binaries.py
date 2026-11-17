"""Offline tests of TESS photometry through the public dataset and synthesis APIs."""
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from elisa import BinarySystem, LCData, Observer, Star, settings, units as u


DATA = Path(__file__).parent / "data" / "tess_binaries"
BTJD_ZERO = 2457000.0


class BinaryCase(NamedTuple):
    product: str
    epoch_btjd: float
    period: float
    eccentricity: float
    omega: float
    inclination: float
    masses: tuple
    temperatures: tuple
    potentials: tuple
    morphology: str = "detached"


# Detached star sizes use rounded Roche potentials. Reference observations and
# model assumptions, including the NT Aps morphology, are described with the data.
CASES = [
    BinaryCase("v505_per_s18_lc", 1798.516720, 4.2220216, 0.0, 90.0, 87.9166,
               (1.2745, 1.2577), (6512, 6460), (12.559, 12.674)),
    BinaryCase("ai_phe_s02_lc", 1362.82850, 24.5924, 0.188425, 110.2315, 88.356,
               (1.1938, 1.2438), (6310, 5010), (28.0, 18.0)),
    BinaryCase("nt_aps_s66_lc", 3120.67442, 0.29475540, 0.011, 206.0, 80.7,
               (1.493, 0.463), (5520, 5313), (2.54, 2.54)),
]


# Explicit synthetic control at the NT Aps cadences. A common potential of 2.44
# is inside the contact interval at this mass ratio (circular L1 ~2.4885).
CONTACT_CONTROL = CASES[-1]._replace(eccentricity=0.0, omega=90.0, potentials=(2.44, 2.44), morphology="over-contact")


def read_rows(case):
    rows = np.genfromtxt(DATA / (case.product + ".csv"), delimiter=",", names=True, dtype=None, encoding="utf-8")
    assert len(rows) == len(np.unique(rows["cadence"])) == 10
    assert np.all(rows["quality"] == 0)
    return rows


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.product)
def test_fold_measured_flux_and_uncertainties(case):
    rows = read_rows(case)
    flux, err = rows["flux_e_s"], rows["error_e_s"]
    reference = np.median(flux[np.char.endswith(rows["position"], "reference")])
    # The FITS columns are absolute count rates, so use one common scale for
    # both columns before passing dimensionless relative flux to ELISa.
    data = LCData(
        x_data=BTJD_ZERO + rows["time_btjd"], x_unit=u.d,
        y_data=flux / reference, y_err=err / reference, y_unit=u.dimensionless_unscaled,
    )
    data.convert_to_phases(case.period, BTJD_ZERO + case.epoch_btjd, centre=0.5)
    expected_phase = ((rows["time_btjd"] - case.epoch_btjd) / case.period) % 1.0
    # Adding the full JD can round each of the two operands by half a time ULP.
    phase_resolution = np.spacing(BTJD_ZERO + case.epoch_btjd) / case.period
    np.testing.assert_allclose(data.x_data, expected_phase, atol=phase_resolution, rtol=0)
    np.testing.assert_allclose(data.y_data, flux / reference, atol=0, rtol=1e-13)
    np.testing.assert_allclose(data.y_err, err / reference, atol=0, rtol=1e-13)


@pytest.mark.parametrize("case", [*CASES, pytest.param(CONTACT_CONTROL, id="nt_aps_circular_contact_control")],
                         ids=lambda case: case.product)
def test_cpu_observer_on_measured_eclipses(case, monkeypatch):
    rows = read_rows(case)
    # Uniform test assumptions with no external atmosphere or limb-darkening tables.
    for name, value in (("LIMB_DARKENING_LAW", "linear"), ("MAX_DISCRETIZATION_FACTOR", 12),
                        ("NUMBER_OF_PROCESSES", 1), ("MESH_GENERATOR", "trapezoidal")):
        monkeypatch.setattr(type(settings), name, value)
    stars = [
        Star(mass=mass, t_eff=teff, surface_potential=potential, atmosphere="bb",
             synchronicity=1.0, albedo=0.5, gravity_darkening=0.32, metallicity=0.0,
             discretization_factor=8.0, limb_darkening_coefficients={"TESS": [0.4], "bolometric": [0.4]})
        for mass, teff, potential in zip(case.masses, case.temperatures, case.potentials)
    ]
    epoch = BTJD_ZERO + case.epoch_btjd
    binary = BinarySystem(*stars, period=case.period, primary_minimum_time=epoch, inclination=case.inclination,
                          argument_of_periastron=case.omega, eccentricity=case.eccentricity)
    # ELISa classifies the printed NT Aps 2023 median inputs as detached.
    assert binary.morphology == case.morphology
    observer = Observer(system=binary, passband="TESS")
    times = BTJD_ZERO + rows["time_btjd"]
    phases = ((times - epoch) / case.period) % 1.0
    _, by_time = observer.lc(times=times, normalize=True)
    # A reversed schedule and a second observation at the same time exercise the
    # phase deduplication and order restoration needed by multi-night data.
    order = np.r_[np.arange(len(rows) - 1, -1, -1), 2]
    actual_phases, by_phase = observer.lc(phases=phases[order], normalize=True)
    np.testing.assert_allclose(actual_phases, phases[order], rtol=0, atol=1e-13)
    np.testing.assert_allclose(by_phase["TESS"], by_time["TESS"][order], rtol=1e-12, atol=1e-12)

    # Test the qualitative shape on both sides of each observed eclipse. A
    # depth- or residual-level fit would depend on radiative and reduction choices.
    positions = ["left_reference", "left_shoulder", "center", "right_shoulder", "right_reference"]
    for eclipse in ("primary", "secondary"):
        selected = rows["eclipse"] == eclipse
        assert rows["position"][selected].tolist() == positions
        for values in (rows["flux_e_s"][selected], by_time["TESS"][selected]):
            assert values[0] > values[1] > values[2] < values[3] < values[4]


@pytest.mark.xfail(
    raises=AssertionError, strict=True,
    reason="dev LCData returns fractional flux errors for magnitude input, independent of pointwise flux",
)
def test_magnitude_uncertainty_scales_with_local_flux():
    """Equal magnitude errors give absolute flux errors proportional to local brightness."""
    rows = np.genfromtxt(DATA / "v505_per_s18_lc.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    sample = rows[(rows["eclipse"] == "primary") & np.isin(rows["position"], ("left_reference", "center"))]
    relative_flux = sample["flux_e_s"] / sample["flux_e_s"][0]
    # Give both real brightness levels the same magnitude uncertainty, using the
    # out-of-eclipse measurement for its size. Both finite and differential
    # conversion obey this ratio because flux is exponential in magnitude.
    sigma_mag = (2.5 / np.log(10.0)) * sample["error_e_s"][0] / sample["flux_e_s"][0]
    data = LCData(x_data=sample["time_btjd"], x_unit=u.d, y_data=-2.5 * np.log10(relative_flux),
                  y_err=np.full(2, sigma_mag), y_unit=u.mag, reference_magnitude=0.0)
    np.testing.assert_allclose(data.y_err[1] / data.y_err[0], relative_flux[1], rtol=1e-12, atol=0)
