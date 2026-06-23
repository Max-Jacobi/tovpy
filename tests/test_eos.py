"""
EOS sanity/regression tests. tovpy had no test suite at all before this --
these are deliberately simple: round-trip and physical-sanity checks plus a
few regression-pinned values, meant as a safety net before any
performance refactoring, not exhaustive physics validation.
"""
import numpy as np
import pytest

from tovpy.eos import EOSPiecewisePolytropic, EOSTabular


@pytest.fixture
def poly():
    return EOSPiecewisePolytropic('piecewise_poly_1', gamma=2.0, K=100.0)


@pytest.fixture
def tabular():
    """
    A small synthetic tabulated EOS (n = K * p, monotonic, in CGS units
    expected by EOSTabular: energy density g/cm^3, pressure dyn/cm^2,
    enthalpy cm^2/s^2, baryon density cm^-3) -- self-contained, no
    dependency on any real nuclear EOS data file.
    """
    n = np.geomspace(1e36, 1e39, 60)
    p = 1e-30 * n**(5/3)
    e = p / 0.6 + n * 1e-24
    h = np.zeros_like(n)
    data = np.stack((e, p, h, n)).T
    return EOSTabular('from_ndarray', data=data)


class TestEOSPiecewisePolytropic:

    def test_pressure_of_density_matches_definition(self, poly):
        rho = 1e-3
        K, gamma = poly.kTab[0], poly.gammaTab[0]
        assert poly.Pressure_Of_RestMassDensity(rho) == pytest.approx(K * rho**gamma)

    @pytest.mark.parametrize("p", [1e-6, 1e-4, 1e-3, 1e-2])
    def test_pressure_pseudoenthalpy_roundtrip(self, poly, p):
        h = poly.PseudoEnthalpy_Of_Pressure(p)
        p_back = poly.Pressure_Of_PseudoEnthalpy(h)
        assert p_back == pytest.approx(p, rel=1e-10)

    def test_energy_density_deriv_matches_finite_difference(self, poly):
        p = 1e-3
        dp = p * 1e-6
        dedp_fd = (poly.EnergyDensity_Of_Pressure(p + dp) - poly.EnergyDensity_Of_Pressure(p - dp)) / (2 * dp)
        assert poly.EnergyDensityDeriv_Of_Pressure(p) == pytest.approx(dedp_fd, rel=1e-4)

    def test_regression_known_values(self, poly):
        """Pins the current, manually-verified output for a Gamma=2 polytrope."""
        p = 1e-3
        assert poly.EnergyDensity_Of_Pressure(p) == pytest.approx(0.004162277660168379, rel=1e-9)


class TestEOSTabular:

    def test_pressure_pseudoenthalpy_roundtrip_inside_table(self, tabular):
        p = np.sqrt(tabular.min_pTab * tabular.max_pTab)  # well inside the table
        h = tabular.PseudoEnthalpy_Of_Pressure(p)
        p_back = tabular.Pressure_Of_PseudoEnthalpy(h)
        assert p_back == pytest.approx(p, rel=1e-6)

    def test_energy_density_pressure_roundtrip_inside_table(self, tabular):
        e = np.sqrt(tabular.min_eTab * tabular.max_eTab)
        p = tabular.Pressure_Of_EnergyDensity(e)
        e_back = tabular.EnergyDensity_Of_Pressure(p)
        assert e_back == pytest.approx(e, rel=1e-6)

    def test_energy_density_is_monotonic_in_pressure(self, tabular):
        ps = np.geomspace(tabular.min_pTab * 1.01, tabular.max_pTab * 0.99, 20)
        es = [tabular.EnergyDensity_Of_Pressure(p) for p in ps]
        assert np.all(np.diff(es) > 0)

    def test_below_table_uses_nonrelativistic_extrapolation(self, tabular):
        p = tabular.min_pTab * 0.5
        e = tabular.EnergyDensity_Of_Pressure(p)
        K = tabular.min_eTab / tabular.min_pTab**(3/5)
        assert e == pytest.approx(K * p**(3/5), rel=1e-10)

    def test_above_table_uses_ultrarelativistic_extrapolation(self, tabular):
        p = tabular.max_pTab * 2
        e = tabular.EnergyDensity_Of_Pressure(p)
        K = tabular.max_eTab / tabular.max_pTab**(3/4)
        assert e == pytest.approx(K * p**(3/4), rel=1e-10)

    def test_energy_density_deriv_matches_finite_difference(self, tabular):
        p = np.sqrt(tabular.min_pTab * tabular.max_pTab)
        dp = p * 1e-6
        dedp_fd = (tabular.EnergyDensity_Of_Pressure(p + dp) - tabular.EnergyDensity_Of_Pressure(p - dp)) / (2 * dp)
        assert tabular.EnergyDensityDeriv_Of_Pressure(p) == pytest.approx(dedp_fd, rel=1e-4)
