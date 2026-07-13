"""
TOV solver sanity/regression tests, using a simple Gamma=2 polytrope (no
external EOS table needed). These pin down current behavior -- including
Love numbers and the maximum-mass search -- so that performance work on
tov.py/eos.py/utils.py can be checked against a known-good baseline.
"""
import numpy as np
import pytest

from tovpy.eos import EOSPiecewisePolytropic
from tovpy.tov import TOV
from tovpy.utils import Target


@pytest.fixture
def poly_eos():
    return EOSPiecewisePolytropic('piecewise_poly_1', gamma=2.0, K=100.0)


class TestTOVSolve:

    def test_solve_returns_positive_mass_and_radius(self, poly_eos):
        tov = TOV(eos=poly_eos)
        M, R, C = tov.solve(5e-4)
        assert M > 0
        assert R > 0
        assert 0 < C < 0.5

    def test_compactness_equals_mass_over_radius(self, poly_eos):
        tov = TOV(eos=poly_eos)
        M, R, C = tov.solve(5e-4)
        assert C == pytest.approx(M / R, rel=1e-10)

    def test_higher_central_pressure_changes_mass(self, poly_eos):
        """Sanity check that solve() actually responds to pc (not pinned
        to a fixed point), on the low-pc/stable branch."""
        tov = TOV(eos=poly_eos)
        M_low, *_ = tov.solve(1e-4)
        M_high, *_ = tov.solve(3e-4)
        assert M_high != pytest.approx(M_low)

    def test_regression_known_mass_radius(self, poly_eos):
        """Pins the current, manually-verified M/R/C for this EOS at pc=5e-4
        (M, R in raw geometric units -- meters -- as returned by solve())."""
        tov = TOV(eos=poly_eos)
        M, R, C = tov.solve(5e-4)
        assert M == pytest.approx(1.6006921847469748, rel=1e-6)
        assert R == pytest.approx(8.41393461228989, rel=1e-6)
        assert C == pytest.approx(0.19024300265049712, rel=1e-6)

    def test_love_number_k2_is_physical(self, poly_eos):
        """Standard NS-like Love numbers k2 fall well within [0, 1]."""
        tov = TOV(eos=poly_eos, leven=[2])
        M, R, C, k, h = tov.solve(5e-4)
        assert 0 < k[2] < 1

    def test_love_and_shape_numbers_are_finite_for_several_ell(self, poly_eos):
        tov = TOV(eos=poly_eos, leven=[2, 3, 4], lodd=[2, 3])
        M, R, C, k, h, j = tov.solve(5e-4)
        for l in (2, 3, 4):
            assert np.isfinite(k[l])
            assert np.isfinite(h[l])
        for l in (2, 3):
            assert np.isfinite(j[l])

    def test_non_converged_solve_does_not_warn_and_returns_nan_love_numbers(self, poly_eos):
        """
        Regression test for the divide-by-zero fix in solve(): a
        deliberately tiny central pressure (deep in the non-converged,
        near-zero-mass regime) must not raise RuntimeWarning, and must
        report NaN (not a garbage finite number) for the Love numbers.
        """
        tov = TOV(eos=poly_eos, leven=[2, 3, 4], lodd=[2, 3])
        with np.errstate(all="raise"):
            M, R, C, k, h, j = tov.solve(1e-10)
        if M <= 0 or R <= 0:
            assert all(np.isnan(v) for v in k.values())
            assert all(np.isnan(v) for v in h.values())
            assert all(np.isnan(v) for v in j.values())


class TestTargetMaximumMass:

    def test_maximum_mass_is_positive_and_larger_than_low_pc_mass(self, poly_eos):
        """
        Target.M_max is in geometric units (meters), consistent with
        TOV.solve()'s raw M -- so they're directly comparable. p0 is
        seeded close to the EOS's actual max-mass pc (~1e-3, found by
        scanning) since the class default (5e-10) undershoots badly for
        this polytrope and the derivative-free search won't recover
        from a wildly wrong guess.
        """
        target = Target(eos=poly_eos, warn=False)
        target.maximum_mass(p0=1e-3)
        assert target.M_max > 0
        assert target.pc_max > 0

        tov = TOV(eos=poly_eos)
        M_low, *_ = tov.solve(target.pc_max * 1e-2)
        assert target.M_max > M_low

    def test_regression_known_max_mass(self, poly_eos):
        """Pins the current, manually-verified M_max (geometric meters)
        /pc_max for this EOS."""
        target = Target(eos=poly_eos, warn=False)
        target.maximum_mass(p0=1e-3)
        assert target.M_max == pytest.approx(1.6372646658643164, rel=1e-3)
        assert target.pc_max == pytest.approx(0.0010000000000000002, rel=1e-3)
