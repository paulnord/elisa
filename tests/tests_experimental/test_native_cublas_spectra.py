"""Validate the opt-in spectral callback when temperatures or passband responses change."""

from contextlib import ExitStack
import os
from pathlib import Path

import numpy as np
import pytest

from elisa import Observer
from elisa.atm.atm import NaiveInterpolatedAtm
from scripts.benchmark_cublas_native_spectra import Callback, SpectralOwner


@pytest.fixture
def spectral_callback(tmp_path):
    # A supplied library exercises the same direct CUDA entry point used in the fit.
    library = os.environ.get("ELISA_TEST_CUBLAS_LIBRARY")
    with ExitStack() as stack:
        owner = SpectralOwner(Path(library)) if library else None
        if owner is not None:
            stack.callback(owner.close)
        yield Callback(
            "cuda-dgemm" if owner else "cpu-dgemm", owner, NaiveInterpolatedAtm.black_body_radiance, False, tmp_path
        )


def test_changing_temperatures_and_response(spectral_callback):
    bands = Observer(passband=["Generic.Bessell.B", "Generic.Bessell.V", "bolometric"]).passband
    blue = "Generic.Bessell.B"
    # 257 unique temperatures cross the 256-row tile boundary. Repeats are
    # noncontiguous and the sub-kelvin pair must remain distinguishable.
    temperatures = np.linspace(4100.0, 7100.0, 257)
    temperatures[1] = temperatures[0] + 1e-6
    temperatures = temperatures[np.r_[256, np.arange(257), 0]]

    def compare(values, passbands):
        result = spectral_callback(values, passband=passbands)
        with np.errstate(over="ignore"):
            native = NaiveInterpolatedAtm.black_body_radiance(values, passband=passbands)
        assert tuple(result) == tuple(passbands)
        for key in passbands:
            np.testing.assert_allclose(result[key], native[key], atol=1e-10, rtol=3e-13)
        return result

    first = compare(temperatures, bands)
    assert first[blue][2] > first[blue][1]
    assert first[blue][0] == first[blue][-2] and first[blue][1] == first[blue][-1]
    warmed = compare(temperatures + 1.75, bands)
    assert np.all(warmed[blue] > first[blue])
    # Mutate the response on the very same container, keeping its wavelength limits and label.
    response = bands[blue].akima
    bands[blue].akima = lambda wavelength: 0.75 * response(wavelength)
    changed = compare(temperatures, bands)
    np.testing.assert_allclose(changed[blue], 0.75 * first[blue], atol=1e-10, rtol=3e-13)
    compare(temperatures, dict(reversed(bands.items())))
    if spectral_callback.owner is not None:
        stats = spectral_callback.owner.statistics()
        assert stats["calls"] == 4 and stats["gemm_calls"] == 8


def test_unsupported_temperature_and_wavelength_units(spectral_callback):
    bands = Observer(passband=["Generic.Bessell.V"]).passband
    for invalid in (0.0, -300.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="temperature"):
            spectral_callback(np.array([5200.0, invalid]), passband=bands)
    bands["Generic.Bessell.V"].wave_to_si_mult = 1.0
    with pytest.raises(ValueError, match="Angstrom"):
        spectral_callback(np.array([5200.0]), passband=bands)
